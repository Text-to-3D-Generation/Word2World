import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, Any
from Renderer import Renderer
from DynamicCamera import DynamicCamera, rotate_camera,safe_normalize
from StaticCamera import StaticCamera
from guidance.mvdream_utils import MVDream
from TrainerIO import TrainerIO
from primitives import generate_random_point_cloud
class GaussianTrainer:
    """Core training functionality without GUI dependencies"""
    
    def __init__(self, prompt):
        self.prompt = prompt
        self.device = torch.device("cuda")
        point_cloud = generate_random_point_cloud(500)
        self.renderer = Renderer(pcd = point_cloud)
        self.cam = DynamicCamera(800, 800, radius=3.8, fovy=49.1)
        
        # Training state
        self.training = False
        self.optimizer = None

    def prepare_train(self):
        """Setup training environment"""
        self.renderer.gaussians_handler.optimizer_setup()
        self.optimizer = self.renderer.gaussians_handler.optimizer
        self.step = 0
        if not hasattr(self, 'mvdream'):
            self.mvdream = MVDream(self.device)
            print(f"[LOGS] MVDream Loaded!")
        with torch.no_grad():
            self.mvdream.get_text_embeds([self.prompt])

    def _get_resolution_for_step(self, step_ratio: float) -> int:
        """Determine render resolution based on training progress"""
        if step_ratio < 0.3:
            return 128
        elif step_ratio < 0.6:
            return 256
        return 512
    
    def _render_training_views(self, resolution: int) :
        """Render training views with random camera positions"""
        vertical_view = np.random.randint(-30, 30)
        horizontal_view = np.random.randint(-180, 180)
        images, views = [], []
        for view_idx in range(4):
            #azimuth offset (0°, 90°, 180°, 270°)
            azimuth = horizontal_view + 90 * view_idx
            
            # Generate pose and camera
            elevation = np.deg2rad(vertical_view)
            azimuth = np.deg2rad(azimuth)
    
            r = 3.8
            x = r* np.cos(elevation) * np.sin(azimuth)  # x = r*cos(θ)*sin(φ)
            y = -r * np.sin(elevation)                   # y = -r*sin(θ)
            z = r * np.cos(elevation) * np.cos(azimuth)  # z = r*cos(θ)*cos(φ)
    
            target = np.zeros(3, dtype=np.float32)
    
            campos = np.array([x, y, z]) + target  # offset by target position
    
            pose = np.eye(4, dtype=np.float32)  # initialize 4x4 identity
            forward = safe_normalize(campos - target)
            up = np.array([0, 1, 0], dtype=np.float32)
            right = safe_normalize(np.cross(up, forward))  # right-handed
            up = np.cross(forward, right)  # recompute up
            up = safe_normalize(up) #re-normalize
            pose[:3, :3] = np.stack([right, up, forward], axis=1)  # set rotation part
            pose[:3, 3] = campos
            view = pose
            views.append(view)
            
            cam = StaticCamera(
                view, resolution, resolution,
                self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far
            )
            
            # Render and store image
            render_out = self.renderer.render(cam, bg_color=torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device))
            images.append(render_out["image"].unsqueeze(0))
            
            # Store first view's full output
            if view_idx == 0:
                out = render_out

        # Stack and process outputs
        stacked_images = torch.cat(images, dim=0)        
        resized_images = torch.nn.functional.interpolate(
            stacked_images,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        stacked_views = torch.from_numpy(np.stack(views)).to(self.device)
        return out, stacked_images, stacked_views, resized_images


    def _compute_guidance_loss(self, images: torch.Tensor, poses: torch.Tensor, step_ratio: float) -> torch.Tensor:
        """Compute loss using guidance model"""
        return self.mvdream.train_step(images, poses, step_ratio=step_ratio)

    def _should_densify(self) -> bool:
        """Check if we should perform densification at current step"""
        return (self.step > 0 and 
                self.step <= 1000)

    def train_step(self) -> float:
        """Execute one training step, return elapsed time in ms"""
        self.step += 1
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        self.renderer.gaussians_handler.update_mean_lr(self.step)
        render_resolution = self._get_resolution_for_step(min(1, self.step / 500))
        out, images, poses, _ = self._render_training_views(render_resolution)
        loss = self._compute_guidance_loss(images, poses, min(1, self.step / 500))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if self._should_densify():
            self._handle_densification(out)

        ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender),loss

    def _handle_densification(self, render_output):
        """Handle densification and pruning"""
        visibility_filter = render_output["visibility_filter"]
        viewspace_points = render_output["viewspace_points"]
        self.renderer.gaussians_handler.collect_densification_info(viewspace_points, visibility_filter)

        #if self.step and self.step % self.opt.aggressive_split_interval == 0:
            #self.renderer.gaussians_handler.agressive_splitting(self.opt.th1,self.opt.th2,self.opt.num_tiles,self.opt.split_factor)
        
        if self.step % 100 == 0:
            self.renderer.gaussians_handler.densification_cycle(max_grad=0.01,min_opacity=0.01)

        if self.step % 250 == 0:
            self.renderer.gaussians_handler.opacity_decay()

    def train(self, iters=500):
        """Main training loop without GUI"""
        #HANY I dont think we'll need this after marwan completes the gui
        if iters > 0:
            self.prepare_train()
            progress = tqdm(range(iters), desc="Training")
            
            for _ in progress:
                train_time,_ = self.train_step()
                progress.set_postfix({"time": f"{train_time:.2f}ms"})
        self.save_model(mode=0)
        self.save_model(mode=2)


    def save_model(self, mode=1, texture_size=1024, user_save=False, model_name="", save_dir=""):
        """Save model"""
        t_name = "mesh"
        extension = "ply"
        if mode == 0:
            t_name = "model"
        if mode == 2:
            extension = "obj"
        return TrainerIO.save_model(renderer=self.renderer,path=os.path.join("outputs", os.path.join(save_dir, f"{model_name}_{t_name}.{extension}")),mode=mode,density_thresh=1,texture_size=texture_size,device=self.device,compress=False,camera = self.cam)