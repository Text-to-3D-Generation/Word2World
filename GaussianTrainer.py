import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, Any
from Renderer import Renderer
from DynamicCamera import DynamicCamera, rotate_camera,safe_normalize
from guidance.mvdream_utils import MVDream
from TrainerIO import TrainerIO
from misc_utils import get_projection_matrix
from primitives import generate_random_point_cloud
class GaussianTrainer:
    """Core training functionality without GUI dependencies"""
    
    def __init__(self, prompt):
        self.prompt = prompt
        self.device = torch.device("cuda")
        point_cloud = generate_random_point_cloud(500)
        self.renderer = Renderer(pcd = point_cloud)
        self.cam = DynamicCamera(800, 800, radius=3.8)
        
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
            
            if not isinstance(view, torch.Tensor):
                c2w = torch.tensor(view, dtype=torch.float32)

            c2w = c2w.to(torch.float32).cuda()

            IW = resolution
            IH = resolution
            FY = self.cam.fovy
            FX = self.cam.fovx
            ZN = self.cam.near
            ZF = self.cam.far

            # Compute inverse: world-to-camera
            W_to_C = torch.linalg.inv(c2w)

            # Apply NeRF coordinate system adjustment
            W_to_C[1:3, :3] *= -1  # Flip Y and Z rotation axes
            W_to_C[:3, 3] *= -1    # Flip translation

            # Store transform in PyTorch (transpose for matmul order)
            W_V_transform = W_to_C.transpose(0, 1).contiguous()

            # Projection matrix from external utility (assumed to be torch-compatible)
            Proj_Matrix = (
                get_projection_matrix(
                    z_near=ZN,
                    z_far=ZF,
                    fov_x=FX,
                    fov_y=FY
                )
                .transpose(0, 1)
                .contiguous()
                .cuda()
            )

            # Full projection transform: MVP matrix
            FULL_PROJ = W_V_transform @ Proj_Matrix

            # Camera center in world space
            CC = -c2w[:3, 3]
            
            # Render and store image
            render_out = self.renderer.render(
                    FX, FY, 
                    W_V_transform, FULL_PROJ,
                    CC, 
                    IW, IH,
                bg_color=torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device))
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
        gl = self.mvdream.train_step(images, poses, step_ratio=step_ratio)
        gl.backward()
        return gl


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
        gl = self._compute_guidance_loss(images, poses, min(1, self.step / 500))
        self.optimizer.step()
        if self._should_densify():
            self._handle_densification(out)
        ender.record()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender),gl

    def _handle_densification(self, render_output):
        """Handle densification and pruning"""
        avaialble_pts = render_output["avaialble_pts"]
        accum_grads = render_output["accum_grads"]
        self.renderer.gaussians_handler.collect_densification_info(accum_grads, avaialble_pts)

        #if self.step and self.step % self.opt.aggressive_split_interval == 0:
            #self.renderer.gaussians_handler.agressive_splitting(self.opt.th1,self.opt.th2,self.opt.num_tiles,self.opt.split_factor)
        
        if self.step % 100 == 0:
            self.renderer.gaussians_handler.densification_cycle(max_grad=0.01,min_opacity=0.01)

        # if self.step % 250 == 0:
        #     self.renderer.gaussians_handler.opacity_decay()


    def save_model(self, mode=1, texture_size=1024, user_save=False, model_name="", save_dir=""):
        """Save model"""
        t_name = "mesh"
        extension = "ply"
        if mode == 0:
            t_name = "model"
        if mode == 2:
            extension = "obj"
        return TrainerIO.save_model(renderer=self.renderer,path=os.path.join("outputs", os.path.join(save_dir, f"{model_name}_{t_name}.{extension}")),mode=mode,density_thresh=1,texture_size=texture_size,device=self.device,compress=False,camera = self.cam)