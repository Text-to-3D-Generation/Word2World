import torch
#from kornia.geometry.conversions import quaternion_to_rotation_matrix
from OptimizerManager import OptimizerManager
#from utils.transforms import qsvec2rotmat_batched, qvec2rotmat_batched
from typing import Optional
import numpy as np
import ctypes,os
#from pytorch3d.ops import knn_points
import torch.nn.functional as F
from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    quaternion_to_rotation_matrix,
)

def qvec2rotmat_batched(qvec):
    return quaternion_to_rotation_matrix(qvec, QuaternionCoeffOrder.WXYZ)



class Densification:
    def __init__(self, optimizer, device,inv_ac=torch.log):
        # Device and optimizer manager
        self.device = device
        self.optimizer = OptimizerManager(optimizer)
        self.svec_inv_act = inv_ac
        #self.densify = ctypes.CDLL(os.path.abspath("densify.dll")).densify_point_flags
        # self.densify.restype  = None
        # self.densify.argtypes = [ctypes.POINTER(ctypes.c_float),
        #             ctypes.c_int, ctypes.c_int,
        #             ctypes.c_float, ctypes.c_float,
        #             ctypes.POINTER(ctypes.c_ubyte)]
        

    def get_params_by_mask(self, mask):
        """
        Retrieves the Shape parameters for the points selected by the mask.

        Args:
            mask (Tensor): Boolean mask indicating selected points.

        Returns:
            dict: Dictionary of selected parameters.
        """
        mask = mask.to(self.device)
        param_dict = {group["name"]: group["params"][0] for group in self.optimizer.optimizer.param_groups
        if group.get("params")}
        mean_tensor = param_dict["mean"]
        sh_coefficients_dc_tensor = param_dict["sh_coefficients_dc"]
        sh_coefficients_ac_tensor = param_dict["sh_coefficients_ac"]
        opacity_tensor = param_dict["opacity"]
        svec_tensor = param_dict["svec"]
        quaternion_tensor = param_dict["quaternion"]
        params = {"mean": mean_tensor[mask],"sh_coefficients_dc": sh_coefficients_dc_tensor[mask],"sh_coefficients_ac": sh_coefficients_ac_tensor[mask],"opacity": opacity_tensor[mask],"svec": svec_tensor[mask],
            "quaternion": quaternion_tensor[mask]}
        return params

    # ===============================
    # CLONING FUNCTIONS
    # ===============================

    def _clone_detection(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec):
        """
        Detects points eligible for cloning based on accumulated gradient norms and scaling factors.

        Args:
            accum_grads (Tensor): Accumulated gradients for each Shape.
            grads_cnt (Tensor): Count of gradient updates for each Shape.
            grad_thresh (float): Threshold for gradient-based densification.
            split_thresh (float): Maximum allowed scale value for cloning.
            svec (Tensor): Scaling factors of Shapes after activation.

        Returns:
            Tensor: Boolean mask indicating points to be cloned.
        """
        # comput normalized gradient magnitude and check if above threshold
        grads_norm = (accum_grads / (grads_cnt + 1e-5))
        grad_mask = torch.where(torch.norm(grads_norm, dim=-1) >= grad_thresh, True, False)
        svec_mask = (svec <= split_thresh).any(dim=-1)
        #print(grad_mask.shape)
        #print(svec_mask.shape)
        selected_pts_mask = torch.logical_and(grad_mask, svec_mask)
        print(selected_pts_mask.shape)
        return selected_pts_mask

    def _clone_treatment(self, mask):
        """
        Performs cloning operation by creating new Shapes from selected points.

        Args:
            mask (Tensor): Boolean mask of points to be cloned.

        Returns:
            dict: Newly created parameters.
        """
        new_params = self.get_params_by_mask(mask)
        all_params = self.optimizer.densify_on_optimizer(new_params)
        return all_params

    def clone_shapes(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec):
        """
        Clones Shape shapes based on gradient magnitude and size constraints.

        Args:
            accum_grads (Tensor): Accumulated gradient norms.
            grads_cnt (Tensor): Count of gradient updates.
            grad_thresh (float): Threshold for gradient-based densification.
            split_thresh (float): Maximum allowed scale value for cloning.
            svec (Tensor): Scaling factors of Shapes.

        Returns:
            tuple: (New parameters dictionary, number of newly cloned points)
        """
        mask = self._clone_detection(accum_grads, grads_cnt, grad_thresh, split_thresh, svec)
        all_params = self._clone_treatment(mask)
        return all_params, torch.count_nonzero(mask).item()

    #######################################################################################################

    # ===============================
    # SPLITTING FUNCTIONS
    # ===============================

    def _split_detection(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec):
        """
        Detects points eligible for splitting based on accumulated gradient norms and scaling factors.

        Args:
            accum_grads (Tensor): Accumulated gradients for each Shape.
            grads_cnt (Tensor): Count of gradient updates for each Shape.
            grad_thresh (float): Threshold for gradient-based densification.
            split_thresh (float): Maximum allowed scale value for splitting.
            svec (Tensor): Scaling factors of Shapes after activation.

        Returns:
            Tensor: Boolean mask indicating points to be splitted.
        """
        grads_norm = (accum_grads / (grads_cnt + 1e-5))
        grad_mask = torch.where(torch.norm(grads_norm, dim=-1) >= grad_thresh, True, False)
        svec_mask = (svec.data > split_thresh).any(dim=-1)
        selected_pts_mask = torch.logical_and(grad_mask, svec_mask)
        return selected_pts_mask


    def _split_treatment(self, mask,svec,split_factor,split_shrink=1):
        """
        Performs the splitting operation by creating new Shapes from selected points.

        Args:
            mask (Tensor): Boolean mask of points to be split.

        Returns:
            dict: Newly created parameters.
        """
        num_split = mask.sum().item()
        new_params = self.get_params_by_mask(mask)
        split_mean = new_params["mean"].data.repeat(split_factor, 1)
        split_quaternion = new_params["quaternion"].data.repeat(split_factor, 1)
        split_sh_coefficients_dc = new_params["sh_coefficients_dc"].repeat(split_factor,1,1)
        split_sh_coefficients_ac = new_params["sh_coefficients_ac"].repeat(split_factor,1,1)
        split_opacity = new_params["opacity"].data.repeat(split_factor,1)        
        split_svec = svec.data[mask].repeat(split_factor, 1)
        split_rotmat = qvec2rotmat_batched(split_quaternion).transpose(-1, -2)
        split_gn = (torch.randn(num_split * split_factor, 3, device=self.device) * split_svec )
        split_sampled_mean = split_mean + torch.einsum("bij, bj -> bi", split_rotmat, split_gn)
        split_raw_svec = self.svec_inv_act((split_svec / (split_factor*0.8))) #revisit this for the tile based
        new_params = {
            "mean":split_sampled_mean ,
            "sh_coefficients_dc": split_sh_coefficients_dc,
            "sh_coefficients_ac": split_sh_coefficients_ac,
            "opacity": split_opacity,
            "svec": split_raw_svec,
            "quaternion":split_quaternion
        }
        #print(new_params)
        # Add new points to the optimizer
        self.optimizer.densify_on_optimizer(new_params)
        prune_mask = torch.cat(
            (
                mask,
                torch.zeros(
                    split_factor * mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        all_params = self._prune_treatment(prune_mask)

        return all_params,mask


    def split_shapes(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec,split_factor,split_shrink=1):
        """
        Detects and performs the splitting operation for shapes.

        Args:
            accum_grads (Tensor): Accumulated gradients for each shape.(N,1)
            grads_cnt (Tensor): Count of gradient updates for each shape.(N,1)
            grad_thresh (float): Threshold for gradient-based splitting.
            split_thresh (float): Maximum allowed scale value for splitting.
            svec (Tensor): Scaling factors of shapes before activation.

        Returns:

            int: Number of new shapes created.
        """
        # Step 1: Detection phase - Find shapes that qualify for splitting
        mask = self._split_detection(accum_grads, grads_cnt, grad_thresh, split_thresh, svec)

        # Step 2: Treatment phase - Apply the splitting operation
        all_params,prune_mask = self._split_treatment(mask,svec,split_factor,split_shrink)

        return all_params, torch.count_nonzero(mask).item(),prune_mask

    # def tile_based_densification(self,threshold1,threshold2,num_tiles, svec,split_factor,split_shrink=1):
    #     param_dict = {
    #         group["name"]: group["params"][0]
    #         for group in self.optimizer.optimizer.param_groups
    #         if group.get("params")  # only include if non-empty
    #     }

    #     mean_tensor = param_dict["mean"]  # Shape (N, 3)
    #     num_gaussians = mean_tensor.shape[0]
    #     mask = np.empty(num_gaussians, dtype=np.uint8)  # Empty array of uint8
    #     flattened_mean = mean_tensor.T.contiguous().view(-1)  # Shape (3*N,)
        
    #     '''
    #     At this point we have all the required args
    #     Now safely call our new detection function
    #     '''
    #     self.densify(flattened_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    #     ctypes.c_int(num_gaussians), ctypes.c_int(num_tiles),
    #     ctypes.c_float(threshold1), ctypes.c_float(threshold2),
    #     mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))

    #     all_params,prune_mask = self._split_treatment(mask,svec,split_factor,split_shrink)

    #     return all_params, torch.count_nonzero(mask).item(),prune_mask


        




    # ===============================
    # PRUNING FUNCTIONS
    # ===============================


    def _prune_treatment(self, mask):
        """
        Prune Shapes based on a boolean mask.

        Args:
            mask (Tensor): Boolean mask indicating which components to PRUNE (True = prune, False = keep).(N,1)
        """
        # Invert the mask to get the valid components to KEEP
        shapes_to_keep_mask = ~mask

        # Prune the optimizer's parameters and stored state
        all_params = self.optimizer.prune_optimizer(shapes_to_keep_mask)


        # Prune additional attributes
        '''
        IMPORTANT NOTE:
        Better do this in the Shapes class to avoid issues related to deep and shallow copies

        self.max_radii2d = self.max_radii2d[shapes_to_keep_mask]
        self.mean_2d_grad_accum = self.mean_2d_grad_accum[shapes_to_keep_mask]
        self.cnt = self.cnt[shapes_to_keep_mask]

        '''
        return all_params


    def _prune_detection_by_radii(self,max_radii,radii_thresh):
        """
        Detects points eligible for pruning based on its max_radii detected.

        Args:
            max_radii (Tensor): Maximum radius detected so far for each shape(N,1).

            IMPORTANT NOTE: A function is provided at the end of this file to provide how max_radii will be computed in the main Shapes class
            radii_thresh (float): Threshold for max radii.


        Returns:
            Tensor: Boolean mask indicating points to be pruned.
        """
        mask = max_radii > radii_thresh
        return mask

    def _prune_detection_by_opacity(self,opacity,opacity_thresh):
        """
        Detects points eligible for pruning based on its opacity.

        Args:
            opacity_thresh (float): Threshold for opacity.


        Returns:
            Tensor: Boolean mask indicating points to be pruned.
        """
        
        
        mask = opacity < opacity_thresh
        return mask.squeeze(-1)

    def _prune_detection_by_3D_scale(self,scaling,svec_thresh):
        """
        Detects points eligible for pruning based on its opacity.

        Args:
            opacity_thresh (float): Threshold for max 3D scale.


        Returns:
            Tensor: Boolean mask indicating points to be pruned.
        """
        
        
        mask = (scaling.max(dim=1).values > 0.1*svec_thresh)
        return mask


    def prune_shapes(self,opacity,scaling,radii_thresh,opacity_thresh,svec_thresh):
        """
        Detects and performs the pruning operation for shapes.

        Args:
            max_radii (Tensor): max 2D radius for each shape. (N,1)
            radii_thresh (float): Threshold for gradient-based splitting.
            opacity_thresh (float): Maximum allowed scale value for splitting.
            svec_thresh(float):


        Returns:
            dict: all new parameters.
            int: Number of new shapes created.
        """
        # Step 1: Detection phase - Find shapes that qualify for pruning
        #NOTE: Not decided yet which detection strategy will be used
        #mask1 = self._prune_detection_by_radii(max_radii,radii_thresh)
        #print(torch.count_nonzero(mask1).item())
        mask2 = self._prune_detection_by_opacity(opacity,opacity_thresh)
        print(torch.count_nonzero(mask2).item())
        mask3 = self._prune_detection_by_3D_scale(scaling,svec_thresh)
        print(torch.count_nonzero(mask3).item())
        total_mask = torch.logical_or(mask2, mask3)
        #total_mask = mask2
        #print(total_mask.shape)

        # Step 2: Treatment phase - Apply the splitting operation
        all_params = self._prune_treatment(total_mask)

        return all_params, torch.count_nonzero(total_mask).item(), total_mask

    

    