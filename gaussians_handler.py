import os
import numpy as np
from GaussianIO import GaussianIO
import torch
from torch import nn
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter
from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.util.numpy_support import numpy_to_vtk
from pyvista import wrap
from Densification import Densification
from mesh import Mesh
from mesh_utils import decimate_mesh, clean_mesh
from misc_utils import inverse_sigmoid, get_expon_lr_func, extract_symmetric, compute_3d_gaussian_coefficient, build_scaling_rotation_matrix
import kiui
import json

from primitives import convert_pcd_to_gaussians, GaussianPrimitive



class GaussiansHandler:
    """3D Gaussian Splatting model composed of multiple GaussianPrimitive instances."""
    
    def __init__(self,pcd):
        
        self.optimizer = None
        self.gaussians = convert_pcd_to_gaussians(pcd)
        #used for densification
        self.mean_gradient_accum = torch.zeros((len(self.gaussians), 1), device="cuda")
        self.counter = torch.zeros((len(self.gaussians), 1), device="cuda")
    
    def covariance_activation(self, scaling, scaling_modifier, rotation):
        """Construct covariance matrix from scaling and rotation for all Gaussians"""
        L = build_scaling_rotation_matrix(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return extract_symmetric(actual_covariance)
    
    
    
    # ============= Training Methods =============
    def optimizer_setup(self):
        """Configure training parameters and optimizer"""
        
        with open("optimizer_config.json", "r") as f:
            config = json.load(f)

        param_groups = []
        for entry in config:
            name = entry["param"]
            lr = entry["lr"]
            params = [getattr(g, name) for g in self.gaussians]
            param_groups.append({
                "params": params,
                "lr": lr,
                "name": name
            })

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.mean_scheduler_args = get_expon_lr_func(
            lr_init=0.001 * 10,
            lr_final=0.00002 * 10,
            lr_delay_mult=0.02,
            max_steps=300
        )
        self.densifier = Densification(self.optimizer, device="cuda")
    
    def update_mean_lr(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mean":
                param_group['lr'] = self.mean_scheduler_args(iteration)
    
    # ============= I/O Methods =============
    def save_ply(self, path):
     
        total_mean = []
        total_sh_coefficients_dc = []
        total_sh_coefficients_ac = []
        total_opacity = []
        total_svec = []
        total_quaternion = []

        for gaussian in self.gaussians:
            total_mean.append(gaussian.mean)
            total_sh_coefficients_dc.append(gaussian.sh_coefficients_dc)
            total_sh_coefficients_ac.append(gaussian.sh_coefficients_ac)
            total_opacity.append(gaussian.opacity)
            total_svec.append(gaussian.svec)
            total_quaternion.append(gaussian.quaternion)

        total_mean = torch.stack(total_mean)
        total_sh_coefficients_dc = torch.stack(total_sh_coefficients_dc)
        total_sh_coefficients_ac = torch.stack(total_sh_coefficients_ac)
        total_opacity = torch.stack(total_opacity)
        total_svec = torch.stack(total_svec)
        total_quaternion = torch.stack(total_quaternion)

        GaussianIO.save_ply(
            path=path,
            mean=total_mean,
            sh_coefficients_dc=total_sh_coefficients_dc,
            sh_coefficients_ac=total_sh_coefficients_ac,
            opacities=total_opacity,
            svec=total_svec,
            quaternions=total_quaternion,
        )

    def load_ply(self, path):
        data = GaussianIO.load_ply(path, self.max_sh_order)
        
        # Clear existing gaussians
        self.gaussians = []
        
        # Create new GaussianPrimitive objects from loaded data
        for i in range(data["mean"].shape[0]):
            g = GaussianPrimitive(
                mean=torch.tensor(data["mean"][i], dtype=torch.float, device="cuda"),
                opacity=torch.tensor(data["opacities"][i], dtype=torch.float, device="cuda").unsqueeze(0),
                sh_coefficients_dc=torch.tensor(data["sh_coefficients_dc"][i], dtype=torch.float, device="cuda").unsqueeze(0),
                sh_coefficients_ac=torch.tensor(data["sh_coefficients_ac"][i], dtype=torch.float, device="cuda").unsqueeze(0),
                svec=torch.tensor(data["svec"][i], dtype=torch.float, device="cuda"),
                quaternion=torch.tensor(data["quaternions"][i], dtype=torch.float, device="cuda")
                
            )
            self.gaussians.append(g)
        
        self.current_sh_order = self.max_sh_order
        self.mean_gradient_accum = torch.zeros((len(self.gaussians), 1), device="cuda")
        self.counter = torch.zeros((len(self.gaussians), 1), device="cuda")
    
    # ============= Mesh Extraction Methods =============
    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        """Extract occupancy field for mesh extraction"""
        block_size = 2 / num_blocks
        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = torch.sigmoid(self._opacity)
        mask = (opacities > 0.005).squeeze(1)
        
        means = self._mean[mask]
        stds = torch.exp(self._svec)[mask]
        
        # Normalize to ~ [-1, 1]
        mn, mx = means.amin(0), means.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        means = (means - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._quaternion[mask])

        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
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
        
        kiui.lo(occ, verbose=1)
        return occ
    
    def extract_mesh_tetra(self, path, density_thresh=1, resolution=128, decimate_target=1e5):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        occ = self.extract_fields(resolution).detach().cpu().numpy()
        flat = occ.flatten(order="F")

        img = vtkImageData()
        img.SetDimensions(resolution, resolution, resolution)
        spacing = 2.0 / (resolution - 1)
        img.SetSpacing(spacing, spacing, spacing)
        img.SetOrigin(-1, -1, -1)

        vtk_arr = numpy_to_vtk(flat, deep=True, array_type=VTK_FLOAT)
        vtk_arr.SetName("densities")
        img.GetPointData().SetScalars(vtk_arr)

        tri = vtkDataSetTriangleFilter()
        tri.SetInputData(img)
        tri.Update()
        tetGrid = tri.GetOutput()

        contour = vtkContour3DLinearGrid()
        contour.SetInputData(tetGrid)
        contour.SetValue(0, density_thresh)
        contour.SetMergePoints(True)
        contour.Update()

        cont = wrap(contour.GetOutput())
        verts = cont.points
        faces = cont.faces.reshape(-1, 4)[:, 1:]

        verts = verts / self.scale + self.center.detach().cpu().numpy()
        verts, faces = clean_mesh(verts, faces, remesh=True, remesh_size=0.015)
        
        if decimate_target > 0 and faces.shape[0] > decimate_target:
            verts, faces = decimate_mesh(verts, faces, decimate_target)

        v = torch.from_numpy(verts.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(faces.astype(np.int32)).contiguous().cuda()
        return Mesh(v=v, f=f, device="cuda")
    
    # ============= Densification Methods =============
    def opacity_decay(self):
        current_opacity = torch.sigmoid(self._opacity)
        decreased_opacity = current_opacity * 0.4
        decreased_opacity = torch.clamp(decreased_opacity, min=0.0, max=1.0)
        opacities_new = inverse_sigmoid(decreased_opacity)
        for i, g in enumerate(self.gaussians):
            g.opacity = nn.Parameter(opacities_new[i].unsqueeze(0).requires_grad_(True))
    
    def collect_densification_info(self, viewspace_point_tensor, update_filter):
        self.mean_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.counter[update_filter] += 1

    def reset_densification_info(self):
        self.mean_gradient_accum = torch.zeros((len(self.gaussians), 1), device="cuda")
        self.counter = torch.zeros((len(self.gaussians), 1), device="cuda")
    
    def update_parameters(self, new_params):
        """Update all Gaussians with new parameters"""

        self.gaussians = []
        for i in range(new_params["mean"].shape[0]):
            g = GaussianPrimitive(mean=new_params["mean"][i],opacity=new_params["opacity"][i].unsqueeze(0),
                sh_coefficients_dc=new_params["sh_coefficients_dc"][i].unsqueeze(0),
                sh_coefficients_ac=new_params["sh_coefficients_ac"][i].unsqueeze(0),svec=new_params["svec"][i],
                quaternion=new_params["quaternion"][i]
            )
            self.gaussians.append(g)
    
    def split_cycle(self, mean2d_thresh, scene_extent):

        total_svec = []
        for gaussian in self.gaussians:
            total_svec.append(gaussian.svec)
        total_svec = torch.stack(total_svec)
        new_params, num_split, mask = self.densifier.split_shapes(self.mean_gradient_accum, self.counter,mean2d_thresh,0.01*scene_extent,torch.exp(total_svec),2,0.8)
        self.update_parameters(new_params)
        
        if num_split > 0:
            new_accum_grads = torch.zeros(2*num_split, device="cuda").unsqueeze(1)
            new_grads_cnt = torch.zeros(2*num_split, device="cuda").unsqueeze(1)
            self.mean_gradient_accum = self.mean_gradient_accum[~mask]
            self.counter = self.counter[~mask]  
            self.mean_gradient_accum = torch.cat([self.mean_gradient_accum, new_accum_grads], dim=0)
            self.counter = torch.cat([self.counter, new_grads_cnt], dim=0)
            
        
        return num_split
    
    def clone_cycle(self, mean2d_thresh, scene_extent):

        total_svec = []
        for gaussian in self.gaussians:
            total_svec.append(gaussian.svec)
        total_svec = torch.stack(total_svec)
        new_params, num_cloned = self.densifier.clone_shapes(self.mean_gradient_accum,self.counter,mean2d_thresh,0.01*scene_extent,torch.exp(total_svec)
        )
        
        self.update_parameters(new_params)
        if num_cloned > 0:
            new_accum_grads = torch.zeros(num_cloned, device="cuda").unsqueeze(1)
            new_grads_cnt = torch.zeros(num_cloned, device="cuda").unsqueeze(1)
            self.mean_gradient_accum = torch.cat([self.mean_gradient_accum, new_accum_grads], dim=0)
            self.counter = torch.cat([self.counter, new_grads_cnt], dim=0)
            
        return num_cloned
    
    def prune_cycle(self, alpha_thresh, radii2d_thresh=1, extent=4):
        total_opacity = []
        for gaussian in self.gaussians:
            total_opacity.append(gaussian.opacity)
            total_svec.append(gaussian.svec)

        total_opacity = torch.stack(total_opacity)
        total_svec = torch.stack(total_svec)
        new_params, num_prunes, mask = self.densifier.prune_shapes(torch.sigmoid(total_opacity),torch.exp(total_svec),radii2d_thresh,alpha_thresh,extent
        )
        self.update_parameters(new_params)
        if num_prunes > 0:
            self.mean_gradient_accum = self.mean_gradient_accum[~mask]
            self.counter = self.counter[~mask]
        return num_prunes
    
    def densification_cycle(self, max_grad, min_opacity, extent=4, max_screen_size=1):  
        num_clone = self.clone_cycle(max_grad, extent)
        print(f"Number of clones: {num_clone}")
        num_split = self.split_cycle(max_grad, extent)
        print(f"Number of splits: {num_split}")
        self.reset_densify_info()
        num_prune = self.prune_cycle(min_opacity, max_screen_size, extent)
        print(f"Number of prunes: {num_prune}")
        torch.cuda.empty_cache()


    '''
    @torch.no_grad()
    def _detect_rogues(self, k: int = 8, z: float = 2.0) -> torch.BoolTensor:
        """Identify isolated Gaussians that should be removed."""
        means = self._mean.detach()          # <- fixed name
        N     = means.shape[0]

        if N < k + 1:
            return torch.zeros(N, dtype=torch.bool, device=means.device)

        dists      = torch.cdist(means, means)            # (N,N)
        knn_d, _   = torch.topk(dists, k+1, largest=False)
        local_mean = knn_d[:, 1:].mean(dim=1)

        mu, std = local_mean.mean(), local_mean.std()
        return local_mean - mu > 0.000005 or mu - local_mean >0.000005   
    @torch.no_grad()
    def _detect_rogues_topK(
            self,
            k: int = 8,          # number of neighbours for the isolation score
            top_n: int = 1000   # how many gaussians to prune
        ) -> torch.BoolTensor:
        """
        Mark the `top_n` Gaussians that are furthest from their local neighbourhood.

        For every Gaussian i we compute

            iso_score[i] = mean(‖x_i - x_j‖   for j in k nearest neighbours of i)

        and prune the `top_n` highest-scoring points.

        Returns
        -------
        BoolTensor [N]  –  True ⇒ prune   |   False ⇒ keep
        """
        means = self._mean.detach()                 # (N,3)   stays on the same device
        N     = means.shape[0]

        # nothing to do?
        if N == 0 or N <= k or top_n <= 0:
            return torch.zeros(N, dtype=torch.bool, device=means.device)

        # 1) pair-wise distances
        dists = torch.cdist(means, means)             # (N,N)  Euclidean

        # 2) k-NN distances (skip self-distance at column 0)
        knn_d, _ = torch.topk(dists, k + 1, largest=False)   # (N,k+1)
        iso_score = knn_d[:, 1:].mean(dim=1)          # (N,)

        # 3) pick the `top_n` highest scores
        prune_count = min(top_n, N)                   # cap at N to stay safe
        top_scores, top_idx = torch.topk(iso_score, prune_count, largest=True)

        # 4) build the mask
        mask = torch.zeros(N, dtype=torch.bool, device=means.device)
        mask[top_idx] = True                          # True ⇒ prune

        return mask
 
    
    @torch.no_grad()
    def _detect_rogues_cluster(
            self,
            eps: float = 0.02,          # radius (in the same units as your xyz)
            min_samples: int = 4,       # DBSCAN core-point neighbourhood size
            min_cluster_size: int = 20  # keep clusters with ≥ this many points
        ) -> torch.BoolTensor:
        """
        Density-based rogue detection with DBSCAN.

        A Gaussian is marked *rogue* when
        • DBSCAN classifies it as noise  (label == –1),  OR
        • it belongs to a tiny cluster (size < `min_cluster_size`).

        Returns
        -------
        BoolTensor [N]  –  True ⇒ prune   |   False ⇒ keep
        """
        means = self._mean.detach().cpu().numpy()    # (N,3) on CPU for sklearn
        N     = means.shape[0]

        if N == 0:                          # empty model -> nothing to prune
            return torch.zeros(0, dtype=torch.bool, device=self._mean.device)

        # --- DBSCAN ---------------------------------------------------------------
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(means)
        #   labels  ∈ {0 … C-1,  –1(noise)}   length N

        labels_t = torch.from_numpy(labels).to(self._mean.device)

        # --- cluster population ---------------------------------------------------
        # Count how many Gaussians in each label (ignore noise, i.e. label == -1)
        valid_labels = labels_t[labels_t >= 0]
        if len(valid_labels):
            counts = torch.bincount(valid_labels)
            # map label → cluster size
            cluster_size = torch.zeros_like(labels_t)
            cluster_size[labels_t >= 0] = counts[valid_labels]
        else:
            cluster_size = torch.zeros_like(labels_t)

        # --- rogue mask -----------------------------------------------------------
        rogues = (labels_t == -1) | (cluster_size < min_cluster_size)

        return rogues
    
    
    
    '''
    