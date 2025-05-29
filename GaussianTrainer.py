import os
import numpy as np
import torch
from Renderer import Renderer
from DynamicCamera import DynamicCamera,safe_normalize
from guidance.mvdream_interface import MVDream
from TrainerIO import TrainerIO
from misc_utils import get_projection_matrix
from primitives import generate_random_point_cloud
class GaussianTrainer:

    def __init__(self, prompt):
        self.prompt = prompt
        self.device = torch.device("cuda")
        point_cloud = generate_random_point_cloud(500)
        self.renderer = Renderer(pcd = point_cloud)
        self.cam = DynamicCamera(800, 800, radius=3.8)
        self.training = False
        self.optimizer = None
        
    def pre_traininig(self):
        self.renderer.gaussians_handler.optimizer_setup()
        self.optimizer = self.renderer.gaussians_handler.optimizer
        self.step = 0
        if not hasattr(self, 'mvdream'):
            self.mvdream = MVDream(self.device)
            print(f"[LOGS] MVDream Loaded!")
        with torch.no_grad():
            self.mvdream.get_text_embeds([self.prompt])

    def get_resolution_for_step(self, step_ratio: float) -> int:
        if step_ratio < 0.3:
            return 128
        elif step_ratio < 0.6:
            return 256
        return 512
    
    def render_training_views(self, resolution: int) :
        vertical_view = np.random.randint(-30, 30)
        horizontal_view = np.random.randint(-180, 180)
        azmiuths = [horizontal_view + 90 * view_idx for view_idx in range(4)]
        images, views = [], []
        for azimuth in azmiuths:
            #azimuth offset (0°, 90°, 180°, 270°)
            # Generate pose and camera
            elevation_rad = np.deg2rad(vertical_view)
            azimuth_rad = np.deg2rad(azimuth)
            r = 3.8
            x = r* np.cos(elevation_rad) * np.sin(azimuth_rad)  # x = r*cos(θ)*sin(φ)
            y = -r * np.sin(elevation_rad)                   # y = -r*sin(θ)
            z = r * np.cos(elevation_rad) * np.cos(azimuth_rad)  # z = r*cos(θ)*cos(φ)
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
            W_to_C = torch.linalg.inv(c2w)
            W_to_C[1:3, :3] *= -1  # Flip Y and Z rotation axes
            W_to_C[:3, 3] *= -1    # Flip translation
            W_V_transform = W_to_C.transpose(0, 1).contiguous()
            Proj_Matrix = (get_projection_matrix(z_near=ZN,z_far=ZF,fov_x=FX,fov_y=FY).transpose(0, 1).contiguous()
                .cuda())
            FULL_PROJ = W_V_transform @ Proj_Matrix
            CC = -c2w[:3, 3]

            render_out = self.renderer.render(FX, FY, W_V_transform, FULL_PROJ,CC, IW, IH,bg_color=torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device))
            images.append(render_out["image"].unsqueeze(0))

            if azimuth == azmiuths[0]: 
                out = render_out
        stacked_images = torch.cat(images, dim=0)        
        resized_images = torch.nn.functional.interpolate(stacked_images,size=(256, 256),mode='bilinear',
            align_corners=False)
        stacked_views = torch.from_numpy(np.stack(views)).to(self.device)
        return out, stacked_images, stacked_views, resized_images

    def compute_guidance_loss(self, images: torch.Tensor, poses: torch.Tensor, step_ratio: float) -> torch.Tensor:
        gl = self.mvdream.train_step(images, poses, step_ratio=step_ratio)
        gl.backward()
        return gl

    def should_densify(self) -> bool:
        return (self.step > 0 and self.step <= 1000)
    
    def optimizaiton_iteration(self) -> float:
        self.step += 1
        render_resolution = self.get_resolution_for_step(min(1, self.step / 500))
        out, images, poses, _ = self.render_training_views(render_resolution)
        self.renderer.gaussians_handler.update_mean_lr(self.step)
        gl = self.compute_guidance_loss(images, poses, min(1, self.step / 500))
        self.optimizer.step()
        if self.should_densify(): self.handle_densification(out)
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        return gl

    def handle_densification(self, render_output):
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
        t_name = "mesh"
        extension = "ply"
        if mode == 0:
            t_name = "model"
        if mode == 2:
            extension = "obj"
        return TrainerIO.save_model(renderer=self.renderer,path=os.path.join("outputs", os.path.join(save_dir, f"{model_name}_{t_name}.{extension}")),mode=mode,density_thresh=1,texture_size=texture_size,device=self.device,compress=False,camera = self.cam)