import os
import numpy as np
from plyfile import PlyData, PlyElement
from misc_utils import flatten_sh_coefficients


class GaussianIO:
    
    @staticmethod
    def save_as_ply(path,mean,sh_coefficients_dc,sh_coefficients_ac,opacities,svec,quaternions):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        position_data = mean.detach().cpu().numpy()
        dc_flat = flatten_sh_coefficients(sh_coefficients_dc)
        ac_flat = flatten_sh_coefficients(sh_coefficients_ac)
        opacity_data = opacities.detach().cpu().numpy()
        scale_data = svec.detach().cpu().numpy()
        rotation_data = quaternions.detach().cpu().numpy()
        prop_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        for i in range(dc_flat.shape[1]):
            prop_list.append((f'sh_coefficients_dc_{i}', 'f4'))
        for i in range(ac_flat.shape[1]):
            prop_list.append((f'sh_coefficients_ac_{i}', 'f4'))
        prop_list.append(('opacity', 'f4'))
        for i in range(scale_data.shape[1]):
            prop_list.append((f'svec_{i}', 'f4'))
        for i in range(rotation_data.shape[1]):
            prop_list.append((f'quaternion_{i}', 'f4'))

        dtype = np.dtype(prop_list)
        vertex_count = position_data.shape[0]
        array_data = np.empty(vertex_count, dtype=dtype)
        array_data['x'] = position_data[:, 0]
        array_data['y'] = position_data[:, 1]
        array_data['z'] = position_data[:, 2]
        for i in range(dc_flat.shape[1]):
            array_data[f'sh_coefficients_dc_{i}'] = dc_flat[:, i]
        for i in range(ac_flat.shape[1]):
            array_data[f'sh_coefficients_ac_{i}'] = ac_flat[:, i]
        array_data['opacity'] = opacity_data[:, 0]
        for i in range(scale_data.shape[1]):
            array_data[f'svec_{i}'] = scale_data[:, i]
        for i in range(rotation_data.shape[1]):
            array_data[f'quaternion_{i}'] = rotation_data[:, i]
        vertex_element = PlyElement.describe(array_data, 'vertex')
        PlyData([vertex_element]).write(path)

    @staticmethod
    def load_from_ply(path):
        ply_data = PlyData.read(path)
        vertex_data = ply_data['vertex']
        positions = np.column_stack((vertex_data['x'],vertex_data['y'],vertex_data['z']))
        dc_channels = 3
        dc_coeffs = np.zeros((positions.shape[0], dc_channels, 1))
        dc_coeffs[:, 0, 0] = vertex_data['sh_coefficients_dc_0']
        dc_coeffs[:, 1, 0] = vertex_data['sh_coefficients_dc_1']
        dc_coeffs[:, 2, 0] = vertex_data['sh_coefficients_dc_2']
        ac_props = [p.name for p in vertex_data.properties if p.name.startswith('sh_coefficients_ac_')]
        ac_count = len(ac_props)
        ac_coeffs = np.zeros((positions.shape[0], dc_channels, ac_count // dc_channels))
        for i in range(ac_count):
            chan = i % dc_channels
            idx = i // dc_channels
            ac_coeffs[:, chan, idx] = vertex_data[f'sh_coefficients_ac_{i}']
        opacities = vertex_data['opacity'].reshape(-1, 1)
        
        scale_props = [p.name for p in vertex_data.properties if p.name.startswith('svec_')]
        scales = np.column_stack([vertex_data[p] for p in scale_props])
        rot_props = [p.name for p in vertex_data.properties 
                    if p.name.startswith('quaternion_')]
        rotations = np.column_stack([vertex_data[p] for p in rot_props])
        return {
            "mean": positions,"sh_coefficients_dc": dc_coeffs,"sh_coefficients_ac": ac_coeffs,"opacities": opacities,"svec": scales,"quaternions": rotations}