import torch
import numpy as np
import ctypes
import os
from typing import Tuple
import platform

from misc_utils import compute_3d_gaussian_coefficient, extract_symmetric, build_scaling_rotation_matrix


def load_cuda_library():
    """Load the compiled CUDA library"""
    system = platform.system()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if system == "Windows":
        lib_name = "compute_voxel_field.dll"
        possible_paths = [
            os.path.join(script_dir, lib_name),  
            os.path.join(os.getcwd(), lib_name),
            lib_name 
        ]
        
        lib_path = None
        for path in possible_paths:
            if os.path.exists(path):
                lib_path = os.path.abspath(path)
                break
        
        if lib_path is None:
            raise FileNotFoundError(
                f"CUDA library {lib_name} not found. Searched in:\n" + 
                "\n".join(f"  - {path}" for path in possible_paths) +
                "\n\nPlease compile it using:\n" +
                "nvcc -O3 -shared -o compute_voxel_field.dll compute_voxel_field.cu"
            )
        
        try:
            lib = ctypes.WinDLL(lib_path)
        except Exception as e:
            lib = ctypes.CDLL(lib_path, winmode=0)
    else:
        lib_name = "compute_voxel_field.so"
        possible_paths = [
            os.path.join(script_dir, lib_name),
            os.path.join(os.getcwd(), lib_name),
            f"./{lib_name}"
        ]
        
        lib_path = None
        for path in possible_paths:
            if os.path.exists(path):
                lib_path = path
                break
        
        if lib_path is None:
            raise FileNotFoundError(
                f"CUDA library {lib_name} not found. Please compile it using:\n" +
                "nvcc -O3 -shared -Xcompiler -fPIC -o compute_voxel_field.so compute_voxel_field.cu"
            )
        
        lib = ctypes.CDLL(lib_path)
    
    lib.compute_voxel_field_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # h_occ
        ctypes.POINTER(ctypes.c_float),  # h_means
        ctypes.POINTER(ctypes.c_float),  # h_covs
        ctypes.POINTER(ctypes.c_float),  # h_opacities
        ctypes.c_int,                     # n_gaussians
        ctypes.c_int,                     # resolution
        ctypes.c_int,                     # num_blocks
        ctypes.c_float                    # relax_ratio
    ]
    
    print(f"Successfully loaded CUDA library from: {lib_path if 'lib_path' in locals() else lib_name}")
    
    return lib


def extract_fields_cuda(
    means: torch.Tensor,
    covs: torch.Tensor, 
    opacities: torch.Tensor,
    resolution: int = 128,
    num_blocks: int = 16,
    relax_ratio: float = 1.5
) -> torch.Tensor:
    """
    CUDA implementation of extract_fields function.
    
    Args:
        means: Gaussian means tensor (N, 3) - already normalized to [-1, 1]
        covs: Gaussian covariances tensor (N, 6)
        opacities: Gaussian opacities tensor (N,)
        resolution: Voxel grid resolution
        num_blocks: Number of blocks per dimension
        relax_ratio: Relaxation ratio for block boundaries
        
    Returns:
        torch.Tensor: Occupancy field (resolution, resolution, resolution)
    """
    lib = load_cuda_library()
    
    means_np = means.cpu().numpy().astype(np.float32).flatten()
    covs_np = covs.cpu().numpy().astype(np.float32).flatten()
    opacities_np = opacities.cpu().numpy().astype(np.float32).flatten()
    
    occ_np = np.zeros((resolution * resolution * resolution,), dtype=np.float32)
    
    n_gaussians = means.shape[0]
    
    lib.compute_voxel_field_cuda(
        occ_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        means_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        covs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        opacities_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n_gaussians),
        ctypes.c_int(resolution),
        ctypes.c_int(num_blocks),
        ctypes.c_float(relax_ratio)
    )
    
    occ = torch.from_numpy(occ_np).reshape(resolution, resolution, resolution)
    
    occ = occ.to(means.device)
    
    return occ


def extract_fields_cpu(
    means: torch.Tensor,
    covs: torch.Tensor,
    opacities: torch.Tensor,
    resolution: int = 128,
    num_blocks: int = 16,
    relax_ratio: float = 1.5
) -> torch.Tensor:
    """
    CPU implementation of extract_fields function (original version).
    
    Args:
        means: Gaussian means tensor (N, 3) - already normalized to [-1, 1]
        covs: Gaussian covariances tensor (N, 6)
        opacities: Gaussian opacities tensor (N,)
        resolution: Voxel grid resolution
        num_blocks: Number of blocks per dimension
        relax_ratio: Relaxation ratio for block boundaries
        
    Returns:
        torch.Tensor: Occupancy field (resolution, resolution, resolution)
    """
    device = means.device
    block_size = 2 / num_blocks
    assert resolution % num_blocks == 0
    split_size = resolution // num_blocks
    
    occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)
    
    X = torch.linspace(-1, 1, resolution).split(split_size)
    Y = torch.linspace(-1, 1, resolution).split(split_size)
    Z = torch.linspace(-1, 1, resolution).split(split_size)
    
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                
                vmin, vmax = pts.amin(0), pts.amax(0)
                vmin -= block_size * relax_ratio
                vmax += block_size * relax_ratio
                mask = (means < vmax).all(-1) & (means > vmin).all(-1)
                
                if not mask.any():
                    continue
                    
                mask_means = means[mask]
                mask_covs = covs[mask]
                mask_opas = opacities[mask].view(1, -1)
                
                g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_means.unsqueeze(0)
                g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1)
                
                batch_g = 1024
                val = 0
                for start in range(0, g_covs.shape[1], batch_g):
                    end = min(start + batch_g, g_covs.shape[1])
                    w = compute_3d_gaussian_coefficient(
                        g_pts[:, start:end].reshape(-1, 3), 
                        g_covs[:, start:end].reshape(-1, 6)
                    ).reshape(pts.shape[0], -1)
                    val += (mask_opas[:, start:end] * w).sum(-1)
                
                occ[xi * split_size: xi * split_size + len(xs), 
                    yi * split_size: yi * split_size + len(ys), 
                    zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs))
    
    return occ


def prepare_gaussians_for_extraction(
    means: torch.Tensor,
    stds: torch.Tensor,
    quaternions: torch.Tensor,
    opacities: torch.Tensor,
    opacity_threshold: float = 0.005
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Prepare Gaussian parameters for field extraction.
    
    Args:
        means: Gaussian means (N, 3)
        stds: Gaussian standard deviations (N, 3)
        quaternions: Gaussian rotations (N, 4)
        opacities: Gaussian opacities (N,)
        opacity_threshold: Minimum opacity threshold
        
    Returns:
        Tuple of (normalized_means, covariances, filtered_opacities, scale, center)
    """
    mask = (opacities > opacity_threshold).squeeze(-1)
    
    means = means[mask]
    stds = stds[mask]
    quaternions = quaternions[mask]
    opacities = opacities[mask]
    
    mn, mx = means.amin(0), means.amax(0)
    center = (mn + mx) / 2
    scale = 1.8 / (mx - mn).amax().item()
    
    normalized_means = (means - center) * scale
    normalized_stds = stds * scale
    
    L = build_scaling_rotation_matrix(normalized_stds, quaternions)
    actual_covariance = L @ L.transpose(1, 2)
    covs = extract_symmetric(actual_covariance)
    
    return normalized_means, covs, opacities, scale, center