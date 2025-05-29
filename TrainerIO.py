import os
import json
import numpy as np
import torch
from typing import Dict, Any
from misc_utils import get_projection_matrix
from plyfile import PlyData, PlyElement
import zlib
import torch.nn.functional as F
import base64
import nvdiffrast.torch as dr
from insert_in_grid import mipmap_bilinear_insert_2d
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__ndarray__': True,
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        return super().default(obj)


def numpy_decoder(dct):
    if '__ndarray__' in dct:
        return np.array(dct['data'], dtype=dct['dtype']).reshape(dct['shape'])
    return dct
def norm(x):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=1e-20))

class TrainerIO:
    """Handles all model input/output operations with versioning and compression support"""
    
    CURRENT_VERSION = "1.2"
    HEADER_FIELD = "gaussian_splatting_data"
    
    @staticmethod
    def save_model(
        renderer,
        path: str,
        mode: int = 0,
        density_thresh: float = 0.01,
        texture_size: int = 1024,
        device: str = "cuda",
        compress: bool = False,
        camera = None
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if mode == 0:
            TrainerIO._save_gaussian_model(renderer.gaussians_handler, path, compress)
        elif mode == 1:
            TrainerIO._save_mesh(renderer, path, density_thresh, False, device)
        elif mode == 2:
            TrainerIO._save_mesh(renderer, path, density_thresh, True, device, texture_size, camera)     
        print(f"[LOGS] Successfully saved model to {path}")
      

    @staticmethod
    def _save_gaussian_model(gaussians, path: str, compress: bool = False):
        """Save the raw Gaussian model data"""
        if compress:
            data = {
                "version": TrainerIO.CURRENT_VERSION,
                "mean": gaussians._mean.detach().cpu().numpy(),
                "sh_coefficients_dc": gaussians._sh_coefficients_dc.detach().cpu().numpy(),
                "sh_coefficients_ac": gaussians._sh_coefficients_ac.detach().cpu().numpy(),
                "opacities": gaussians._opacity.detach().cpu().numpy(),
                "svec": gaussians._svec.detach().cpu().numpy(),
                "quaternions": gaussians._quaternion.detach().cpu().numpy(),
                "sh_degree": gaussians.max_sh_order
            }
            json_str = json.dumps(data, cls=NumpyEncoder)
            compressed = zlib.compress(json_str.encode('utf-8'))
            encoded = base64.b64encode(compressed).decode('ascii')
            vertex = np.zeros(1, dtype=[(TrainerIO.HEADER_FIELD, 'O')])
            vertex[TrainerIO.HEADER_FIELD] = encoded
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el], text=True).write(path)
        else:
            gaussians.save_ply(path)

    @staticmethod
    def _save_mesh(renderer, path: str, density_thresh: float, with_texture: bool, 
                  device: str, texture_size: int = 1024, camera=None):
        """Save extracted mesh with optional texture"""
        print(f"[INFO] Extracting mesh...")
        mesh = renderer.gaussians_handler.extract_mesh_tetra(path, density_thresh)
        
        if with_texture:
            print(f"[INFO] Generating texture...")
            TrainerIO._save_textured_mesh(mesh,renderer, path, density_thresh, texture_size, device,camera)
        else:
            mesh.write_ply(path)

    @staticmethod
    def _save_textured_mesh(mesh,renderer, path: str, density_thresh: float, 
                           texture_size: int, device: str,camera) -> None:
        """Save mesh with generated texture atlas"""
        
        
    
        h = w = texture_size
        
        # Prepare mesh UVs and normals
        print("[LOGS] Unwrapping UVs...")
        mesh.auto_uv()
        mesh.auto_normal()

        # Initialize texture buffers
        albedo = torch.zeros((h, w, 3), device=device, dtype=torch.float32)
        cnt = torch.zeros((h, w, 1), device=device, dtype=torch.float32)

        # Define camera angles for texture projection
        vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
        hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]
        render_resolution = 512

        # Initialize render context
        glctx = TrainerIO._init_render_context()

        # Project each view to texture
        for ver, hor in zip(vers, hors):
            TrainerIO._project_view_to_texture(
                renderer, mesh, ver, hor, 
                albedo, cnt, glctx, h, w, render_resolution, device, camera
            )

        # Post-process and save final texture
        albedo = TrainerIO._postprocess_texture(albedo, cnt, h, w)
        mesh.albedo = albedo
        mesh.write(path)

    @staticmethod
    def _init_render_context() -> Any:
        """Initialize nvdiffrast render context based on platform"""
        
        return dr.RasterizeGLContext()
       
    @staticmethod
    def _project_view_to_texture(renderer, mesh, ver: float, hor: float,
                                albedo: torch.Tensor, cnt: torch.Tensor,
                                glctx: Any, h: int, w: int,
                                render_resolution: int, device: str, cam) -> None:
        """
        Project a camera view onto the texture map, guaranteeing that the
        RGB pass and the depth/mask pass have identical resolution so the
        boolean-mask IndexError can never occur.
        """

        # ---------- 1)  Render RGB at the requested resolution ----------
        if hasattr(renderer, "set_resolution"):
            renderer.set_resolution(render_resolution, render_resolution)
        r = cam.radius
        x = r* np.cos(ver) * np.sin(hor)  # x = r*cos(θ)*sin(φ)
        y = -r * np.sin(ver)                   # y = -r*sin(θ)
        z = r * np.cos(ver) * np.cos(hor)  # z = r*cos(θ)*cos(φ)
        target = np.zeros(3, dtype=np.float32)
        campos = np.array([x, y, z]) + target  # offset by target position
        pose = np.eye(4, dtype=np.float32)  # initialize 4x4 identity
        forward = norm(campos - target)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = norm(np.cross(up, forward))  # right-handed
        up = np.cross(forward, right)  # recompute up
        up = norm(up) #re-normalize
        pose[:3, :3] = np.stack([right, up, forward], axis=1)  # set rotation part
        pose[:3, 3] = campos
        # cur_cam = StaticCamera(
        #     pose,
        #     render_resolution,           # width
        #     render_resolution,           # height
        #     cam.fovy,
        #     cam.fovx,
        #     cam.near,
        #     cam.far,
        # )

        c2w = torch.tensor(pose, dtype=torch.float32)
        c2w = c2w.to(torch.float32).cuda()
        IW = render_resolution
        IH = render_resolution
        FY = cam.fovy
        FX = cam.fovx
        ZN = cam.near
        ZF = cam.far

        # Compute inverse: world-to-camera
        W_to_C = torch.linalg.inv(c2w)

        # Apply NeRF coordinate system adjustment
        W_to_C[1:3, :3] *= -1  # Flip Y and Z rotation axes
        W_to_C[:3, 3] *= -1    # Flip translation

        # Store transform in PyTorch (transpose for matmul order)
        W_V_transform = W_to_C.transpose(0, 1).contiguous()

        # Projection matrix from external utility (assumed to be torch-compatible)
        Proj_Matrix = (get_projection_matrix(z_near=ZN,z_far=ZF,
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

        out = renderer.render(
            FX, FY, 
            W_V_transform, FULL_PROJ,
            CC, 
            IW, IH)
        
        rgb_img  = out["image"]                 # (3, H, W)
        H, W     = rgb_img.shape[-2:]           # read true resolution
        rgb_img  = rgb_img.unsqueeze(0)         # (1, 3, H, W)

        # ---------- 2)  Transform vertices into clip space ----------
        pose_t = torch.from_numpy(pose.astype(np.float32)).to(device)
        proj   = torch.from_numpy(cam.get_perspective().astype(np.float32)).to(device)

        v_cam = torch.matmul(
            F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0),
            torch.inverse(pose_t).T
        ).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        # ---------- 3)  Rasterise at exactly (H, W) ----------
        rast, _ = dr.rasterize(glctx, v_clip, mesh.f, (H, W))

        # ---------- 4)  Interpolate attrs & build visibility mask ----------
        uvs, _    = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)
        normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
        normal    = norm(normal[0])

        rot_normal = normal @ pose_t[:3, :3]
        viewcos    = rot_normal[..., [2]]

        mask = (rast[0, ..., 3:] > 0) & (viewcos > 0.5)  # (H, W, 1) → bool
        mask = mask.view(-1)                             # (H·W,)

        # ---------- 5)  Gather valid UVs and RGBs (sizes now match) ----------
        valid_uvs  = uvs.view(-1, 2).clamp(0, 1)[mask]          # (N, 2)
        valid_rgbs = (
            rgb_img.squeeze(0)           # (3, H, W)
                .permute(1, 2, 0)     # (H, W, 3)
                .reshape(-1, 3)[mask] # (N, 3)
                .contiguous()
        )

        # ---------- 6)  Blend into the global texture ----------
        cur_albedo, cur_cnt = mipmap_bilinear_insert_2d(
            h, w,
            valid_uvs[..., [1, 0]] * 2 - 1,   # UV → NDC
            valid_rgbs,
            min_res=256,
            return_count=True,
        )

        update_mask = cnt.squeeze(-1) < 0.1
        albedo[update_mask] += cur_albedo[update_mask]
        cnt[update_mask]    += cur_cnt[update_mask]


    @staticmethod
    def _postprocess_texture(albedo: torch.Tensor, cnt: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Fill holes in texture using nearest neighbor inpainting"""
        from sklearn.neighbors import NearestNeighbors
        from scipy.ndimage import binary_dilation, binary_erosion
        
        # Normalize
        mask = cnt.squeeze(-1) > 0
        albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)
        mask = mask.view(h, w)

        mask = mask.detach().cpu().numpy()
        
        # Find holes to inpaint
        inpaint_region = binary_dilation(mask, iterations=32)
        inpaint_region[mask] = 0

        # Find valid source regions
        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        # Perform inpainting
        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        albedo_np = albedo.detach().cpu().numpy()
        albedo_np[tuple(inpaint_coords.T)] = albedo_np[tuple(search_coords[indices[:, 0]].T)]
        
        return torch.from_numpy(albedo_np).to(albedo.device)
