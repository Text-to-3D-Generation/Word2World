from GaussianIO import GaussianIO
import torch
from torch import nn
from Densification import Densification
from misc_utils import inverse_sigmoid, create_exponential_learning_rate_schedule
import json
from primitives import convert_pcd_to_gaussians, GaussianPrimitive



class GaussiansHandler:

    def __init__(self,pcd):
        
        self.optimizer = None
        self.gaussians = convert_pcd_to_gaussians(pcd)
        #used for densification
        self.mean_gradient_accum = torch.zeros((len(self.gaussians), 1), device="cuda")
        self.counter = torch.zeros((len(self.gaussians), 1), device="cuda")
    
    def get_param_group_by_name(self, name):
        for group in self.optimizer.param_groups:
            if group.get("name") == name:
                return group["params"][0]
    
    # ============= Training Methods ===============================================++
    def optimizer_setup(self):
        with open("optimizer_config.json", "r") as f:
            config = json.load(f)

        param_stacks = {"mean": torch.stack([g.mean for g in self.gaussians]),"opacity": torch.stack([g.opacity for g in self.gaussians]),"sh_coefficients_dc": torch.stack([g.sh_coefficients_dc for g in self.gaussians]),
            "sh_coefficients_ac": torch.stack([g.sh_coefficients_ac for g in self.gaussians]),"svec": torch.stack([g.svec for g in self.gaussians]),
            "quaternion": torch.stack([g.quaternion for g in self.gaussians])}
        param_groups = []
        for entry in config:
            param_name = entry["param"]
            if param_name in param_stacks:
                stacked_param = nn.Parameter(param_stacks[param_name].requires_grad_(True))
                param_groups.append({"params": [stacked_param],"lr": entry["lr"],"name": param_name})
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        collected_params = {}
        for group in self.optimizer.param_groups:
            collected_params[group.get("name")] = group["params"][0]
        self.update_parameters(collected_params)
        self.densifier = Densification(self.optimizer, device="cuda")
        self.mean_scheduler_args = create_exponential_learning_rate_schedule(0.01,0.0002,300)
        print("-------------Optimizer Params----------")
        for group in self.optimizer.param_groups:
            print(f"{group['name']} shape: {group['params'][0].shape}")
        print("-------------------------------------------")
    
    def update_mean_lr(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mean":
                param_group['lr'] = self.mean_scheduler_args(iteration)
                
    def save_as_ply(self, path):
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

        # total_mean = self.get_param_group_by_name("mean")
        # total_sh_coefficients_dc = self.get_param_group_by_name("sh_coefficients_dc")
        # total_sh_coefficients_ac = self.get_param_group_by_name("sh_coefficients_ac")
        # total_opacity = self.get_param_group_by_name("opacity")
        # total_svec = self.get_param_group_by_name("svec")
        # total_quaternion = self.get_param_group_by_name("quaternion")

        GaussianIO.save_as_ply(path=path,mean=total_mean,sh_coefficients_dc=total_sh_coefficients_dc,sh_coefficients_ac=total_sh_coefficients_ac,opacities=total_opacity,
            svec=total_svec,quaternions=total_quaternion)

    def load_from_ply(self, path):
        data = GaussianIO.load_from_ply(path, self.max_sh_order)
        self.gaussians = []
        for i in range(data["mean"].shape[0]):
            g = GaussianPrimitive(mean=torch.tensor(data["mean"][i], dtype=torch.float, device="cuda"),
                opacity=torch.tensor(data["opacities"][i], dtype=torch.float, device="cuda").unsqueeze(0),
                sh_coefficients_dc=torch.tensor(data["sh_coefficients_dc"][i], dtype=torch.float, device="cuda").unsqueeze(0),
                sh_coefficients_ac=torch.tensor(data["sh_coefficients_ac"][i], dtype=torch.float, device="cuda").unsqueeze(0),
                svec=torch.tensor(data["svec"][i], dtype=torch.float, device="cuda"),
                quaternion=torch.tensor(data["quaternions"][i], dtype=torch.float, device="cuda"))
            self.gaussians.append(g)
        self.mean_gradient_accum = torch.zeros((len(self.gaussians), 1), device="cuda")
        self.counter = torch.zeros((len(self.gaussians), 1), device="cuda")
    
    def opacity_decay(self):
        total_opacity = []
        for gaussian in self.gaussians:
            total_opacity.append(gaussian.opacity)
        total_opacity = torch.sigmoid(torch.stack(total_opacity))
        decreased_opacity = total_opacity * 0.4
        decreased_opacity = torch.clamp(decreased_opacity, min=0.0, max=1.0)
        opacities_new = inverse_sigmoid(decreased_opacity)
        for i, g in enumerate(self.gaussians):
            g.opacity = nn.Parameter(opacities_new[i].requires_grad_(True))
    
    def collect_densification_info(self, viewspace_point_tensor, update_filter):
        self.counter[update_filter] += 1
        self.mean_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        

    def update_parameters(self, new_params):
        self.gaussians = []
        for i in range(new_params["mean"].shape[0]):
            g = GaussianPrimitive(mean=new_params["mean"][i],opacity=new_params["opacity"][i],
                sh_coefficients_dc=new_params["sh_coefficients_dc"][i],
                sh_coefficients_ac=new_params["sh_coefficients_ac"][i],svec=new_params["svec"][i],
                quaternion=new_params["quaternion"][i]
            )
            self.gaussians.append(g)
    
    def split_cycle(self, mean2d_thresh):
        total_svec = []
        for gaussian in self.gaussians:
            total_svec.append(gaussian.svec)
        total_svec = torch.stack(total_svec)
        # total_svec = self.get_param_group_by_name("svec")
        new_params, num_split, mask = self.densifier.split_shapes(self.mean_gradient_accum, self.counter,mean2d_thresh,4,torch.exp(total_svec),2,0.8)
        self.update_parameters(new_params)
        if num_split > 0:
            new_accum_grads = torch.zeros(2*num_split, device="cuda").unsqueeze(1)
            new_grads_cnt = torch.zeros(2*num_split, device="cuda").unsqueeze(1)
            self.mean_gradient_accum = self.mean_gradient_accum[~mask]
            self.counter = self.counter[~mask]  
            self.mean_gradient_accum = torch.cat([self.mean_gradient_accum, new_accum_grads], dim=0)
            self.counter = torch.cat([self.counter, new_grads_cnt], dim=0)
        return num_split
    
    def clone_cycle(self, mean2d_thresh):

        total_svec = []
        for gaussian in self.gaussians:
            total_svec.append(gaussian.svec)
        total_svec = torch.stack(total_svec)
        # total_svec = self.get_param_group_by_name("svec")
        new_params, num_cloned = self.densifier.clone_shapes(self.mean_gradient_accum,self.counter,mean2d_thresh,0.04,torch.exp(total_svec)
        )
        
        self.update_parameters(new_params)
        if num_cloned > 0:
            new_accum_grads = torch.zeros(num_cloned, device="cuda").unsqueeze(1)
            new_grads_cnt = torch.zeros(num_cloned, device="cuda").unsqueeze(1)
            self.mean_gradient_accum = torch.cat([self.mean_gradient_accum, new_accum_grads], dim=0)
            self.counter = torch.cat([self.counter, new_grads_cnt], dim=0)
            
        return num_cloned
    
    def prune_cycle(self, alpha_thresh):
        total_opacity = []
        total_svec = []
        for gaussian in self.gaussians:
            total_opacity.append(gaussian.opacity)
            total_svec.append(gaussian.svec)

        total_opacity = torch.stack(total_opacity)
        total_svec = torch.stack(total_svec)

        new_params, num_prunes, mask = self.densifier.prune_shapes(torch.sigmoid(total_opacity),torch.exp(total_svec),1,alpha_thresh,4
        )
        self.update_parameters(new_params)
        if num_prunes > 0:
            self.mean_gradient_accum = self.mean_gradient_accum[~mask]
            self.counter = self.counter[~mask]
        return num_prunes
    
    def adc_cycle(self, max_grad, min_opacity):  
        num_clone = self.clone_cycle(max_grad)
        print(f"Number of clones: {num_clone}")
        num_split = self.split_cycle(max_grad)
        print(f"Number of splits: {num_split}")
        self.mean_gradient_accum = torch.zeros((len(self.gaussians), 1), device="cuda")
        self.counter = torch.zeros((len(self.gaussians), 1), device="cuda")
        num_prune = self.prune_cycle(min_opacity)
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
    