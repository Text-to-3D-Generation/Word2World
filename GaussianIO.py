import os
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
from misc_utils import flatten_sh_coefficients
class GaussianIO:
    """Handles all input/output operations for GaussianModel"""
    
    @staticmethod
    def save_ply(
        path: str,
        mean: torch.Tensor,
        sh_coefficients_dc: torch.Tensor,
        sh_coefficients_ac: torch.Tensor,
        opacities: torch.Tensor,
        svec: torch.Tensor,
        quaternions: torch.Tensor,
    ):
        """Save Gaussian model to PLY file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert tensors to numpy
        mean_np = mean.detach().cpu().numpy()
        normals = np.zeros_like(mean_np)
        sh_coefficients_dc_np = flatten_sh_coefficients(sh_coefficients_dc)
        sh_coefficients_ac_np = flatten_sh_coefficients(sh_coefficients_ac)
        opacities_np = opacities.detach().cpu().numpy()
        svec_np = svec.detach().cpu().numpy()
        quaternions_np = quaternions.detach().cpu().numpy()

        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attrs += [f'sh_coefficients_dc_{i}' for i in range(sh_coefficients_dc.shape[1] * sh_coefficients_dc.shape[2])]
        attrs += [f'sh_coefficients_ac_{i}' for i in range(sh_coefficients_ac.shape[1] * sh_coefficients_ac.shape[2])]
        attrs += ['opacity']
        attrs += [f'svec_{i}' for i in range(svec.shape[1])]
        attrs += [f'quaternion_{i}' for i in range(quaternions.shape[1])]
        
        # Create structured array
        dtype_full = [(attr, 'f4') for attr in attrs]
        elements = np.empty(mean_np.shape[0], dtype=dtype_full)
        data = np.concatenate((mean_np, normals, sh_coefficients_dc_np, sh_coefficients_ac_np, opacities_np, svec_np, quaternions_np), axis=1)
        elements[:] = list(map(tuple, data))
        
        # Write to file
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    @staticmethod
    def load_ply(path: str, max_sh_order: int) -> dict:
        """Load Gaussian model from PLY file"""
        plydata = PlyData.read(path)

        # Extract basic properties
        mean = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # Extract features
        sh_coefficients_dc = np.zeros((mean.shape[0], 3, 1))
        sh_coefficients_dc[:, 0, 0] = np.asarray(plydata.elements[0]["sh_coefficients_dc_0"])
        sh_coefficients_dc[:, 1, 0] = np.asarray(plydata.elements[0]["sh_coefficients_dc_1"])
        sh_coefficients_dc[:, 2, 0] = np.asarray(plydata.elements[0]["sh_coefficients_dc_2"])

        # Extract rest features
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sh_coefficients_ac_")]
        assert len(extra_f_names) == 3 * (max_sh_order + 1) ** 2 - 3
        features_extra = np.zeros((mean.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sh_coefficients_ac = features_extra.reshape((mean.shape[0], 3, (max_sh_order + 1) ** 2 - 1))

        # Extract svec and quaternions
        svec_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("svec_")]
        svec = np.zeros((mean.shape[0], len(svec_names)))
        for idx, attr_name in enumerate(svec_names):
            svec[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("quaternion")]
        rots = np.zeros((mean.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return {
            "mean": mean,
            "sh_coefficients_dc": sh_coefficients_dc,
            "sh_coefficients_ac": sh_coefficients_ac,
            "opacities": opacities,
            "svec": svec,
            "quaternions": rots
        }
    

