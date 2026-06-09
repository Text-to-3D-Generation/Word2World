import torch
import numpy as np
import ctypes
import os
from typing import Tuple
import platform

from misc_utils import compute_3d_gaussian_coefficient, extract_symmetric, build_scaling_rotation_matrix


def load_cuda_library():

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
                "\n".join(f"  - {path}" for path in possible_paths)
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
                f"CUDA library {lib_name} not found.\n"
            )
        
        lib = ctypes.CDLL(lib_path)
    
    lib.compute_voxel_field_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float),  
        ctypes.POINTER(ctypes.c_float),  
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),  
        ctypes.c_int,                     
        ctypes.c_int,                     
        ctypes.c_int,                     
        ctypes.c_float                    
    ]
    
    print(f"Successfully loaded the CUDA library from: {lib_path if 'lib_path' in locals() else lib_name}")
    
    return lib


def computeVoxelFeildsCuda(means, covs, opacities, resolution, num_blocks, relax_ratio):

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




def prepare_gaussians_for_extraction(means,stds,quaternions,opacities,opacity_threshold):

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
