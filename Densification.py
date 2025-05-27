import torch
from OptimizerManager import OptimizerManager

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


    def _clone_detection(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec):
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
        new_params = self.get_params_by_mask(mask)
        all_params = self.optimizer.densify_on_optimizer(new_params)
        return all_params

    def clone_shapes(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec):
        mask = self._clone_detection(accum_grads, grads_cnt, grad_thresh, split_thresh, svec)
        all_params = self._clone_treatment(mask)
        return all_params, torch.count_nonzero(mask).item()

    #######################################################################################################

    def _split_detection(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec):
        grads_norm = (accum_grads / (grads_cnt + 1e-5))
        grad_mask = torch.where(torch.norm(grads_norm, dim=-1) >= grad_thresh, True, False)
        svec_mask = (svec.data > split_thresh).any(dim=-1)
        selected_pts_mask = torch.logical_and(grad_mask, svec_mask)
        return selected_pts_mask


    def _split_treatment(self, mask,svec,split_factor,split_shrink=1):
        num_split = mask.sum().item()
        new_params = self.get_params_by_mask(mask)
        split_mean = new_params["mean"].data.repeat(split_factor, 1)
        split_quaternion = new_params["quaternion"].data.repeat(split_factor, 1)
        split_sh_coefficients_dc = new_params["sh_coefficients_dc"].repeat(split_factor,1,1)
        split_sh_coefficients_ac = new_params["sh_coefficients_ac"].repeat(split_factor,1,1)
        split_opacity = new_params["opacity"].data.repeat(split_factor,1)        
        split_svec = svec.data[mask].repeat(split_factor, 1)
        quaternion_norm = torch.sqrt(split_quaternion[:,0]*split_quaternion[:,0] + split_quaternion[:,1]*split_quaternion[:,1] + split_quaternion[:,2]*split_quaternion[:,2] + split_quaternion[:,3]*split_quaternion[:,3])
        normalzided_split_quaternion = split_quaternion / quaternion_norm[:, None]
        split_rotmat = torch.zeros((normalzided_split_quaternion.size(0), 3, 3), device='cuda')
        q_w, q_x, q_y, q_z = normalzided_split_quaternion[:, 0], normalzided_split_quaternion[:, 1], normalzided_split_quaternion[:, 2], normalzided_split_quaternion[:, 3]
        split_rotmat[:, 0, 0] = 1 - 2 * (q_y**2 + q_z**2)
        split_rotmat[:, 1, 0] = 2 * (q_x*q_y + q_w*q_z)
        split_rotmat[:, 2, 0] = 2 * (q_x*q_z - q_w*q_y)

        split_rotmat[:, 0, 1] = 2 * (q_x*q_y - q_w*q_z)
        split_rotmat[:, 1, 1] = 1 - 2 * (q_x**2 + q_z**2)
        split_rotmat[:, 2, 1] = 2 * (q_y*q_z + q_w*q_x)

        split_rotmat[:, 0, 2] = 2 * (q_x*q_z + q_w*q_y)
        split_rotmat[:, 1, 2] = 2 * (q_y*q_z - q_w*q_x)
        split_rotmat[:, 2, 2] = 1 - 2 * (q_x**2 + q_y**2)

        stds = split_svec
        means = torch.zeros((stds.size(0), 3),device="cuda")
        sampled_means = torch.normal(mean=means, std=stds)
        split_sampled_mean = torch.bmm(split_rotmat, sampled_means.unsqueeze(-1)).squeeze(-1) + split_mean
        split_raw_svec = self.svec_inv_act((split_svec / (split_factor*0.8))) #revisit this for the tile based
        new_params = {"mean":split_sampled_mean ,"sh_coefficients_dc": split_sh_coefficients_dc,"sh_coefficients_ac": split_sh_coefficients_ac,"opacity": split_opacity,
            "svec": split_raw_svec,"quaternion":split_quaternion}
        #print(new_params)
        self.optimizer.densify_on_optimizer(new_params)
        prune_mask = torch.cat((mask,torch.zeros(split_factor * mask.sum(), device=self.device, dtype=bool),))
        all_params = self._prune_treatment(prune_mask)
        return all_params,mask


    def split_shapes(self, accum_grads, grads_cnt, grad_thresh, split_thresh, svec,split_factor,split_shrink=1):
        mask = self._split_detection(accum_grads, grads_cnt, grad_thresh, split_thresh, svec)
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



    #######################################################################################################
    
    def _prune_treatment(self, mask):
        # Invert the mask to get the valid components to KEEP
        shapes_to_keep_mask = ~mask

        # Prune the optimizer's parameters and stored state
        all_params = self.optimizer.prune_optimizer(shapes_to_keep_mask)


        # Prune additional attributes
        '''
        reminder fo for me:
        Better do this in the Shapes class to avoid issues related to deep and shallow copies

        self.max_radii2d = self.max_radii2d[shapes_to_keep_mask]
        self.mean_2d_grad_accum = self.mean_2d_grad_accum[shapes_to_keep_mask]
        self.cnt = self.cnt[shapes_to_keep_mask]

        '''
        return all_params


    def _prune_detection_by_radii(self,max_radii,radii_thresh):
        """
            reminder for me: A function is provided at the end of this file to provide how max_radii will be computed in the main Shapes class
            radii_thresh (float): Threshold for max radii.
        """
        mask = max_radii > radii_thresh
        return mask

    def _prune_detection_by_opacity(self,opacity,opacity_thresh):
        mask = opacity < opacity_thresh
        return mask.squeeze(-1)

    def _prune_detection_by_3D_scale(self,scaling,svec_thresh):
        mask = (scaling.max(dim=1).values > 0.1*svec_thresh)
        return mask

    def prune_shapes(self,opacity,scaling,radii_thresh,opacity_thresh,svec_thresh):
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
        all_params = self._prune_treatment(total_mask)
        return all_params, torch.count_nonzero(total_mask).item(), total_mask

    

    