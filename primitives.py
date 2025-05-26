import torch
from torch import nn
import numpy as np
from misc_utils import inverse_sigmoid

class PointCloud:
    """A struct for 3D point cloud"""
    points: torch.Tensor  #(N, 3)
    sh_coefficients: torch.Tensor  #(N, 3)

    def __init__(self, points, sh_coefficients):
        self.points = points
        self.sh_coefficients = sh_coefficients


class GaussianPrimitive:
    """It represent one gaussian in space"""
    def __init__(self, mean,opacity,sh_coefficients_dc,sh_coefficients_ac,svec,quaternion):

        self.mean = mean
        self.opacity = opacity
        self.sh_coefficients_dc = sh_coefficients_dc
        self.sh_coefficients_ac = sh_coefficients_ac
        self.svec = svec
        self.quaternion = quaternion

    

def generate_random_point_cloud(num_points):
    """Initialize random points on a sphere of radius 0.5."""

    cos_thetas = np.random.uniform(-1, 1, num_points)
    thetas = np.arccos(cos_thetas) #should from 0 to 180
    phis = (2*np.pi) * np.random.random(num_points) #rotations around vertical(z) axis, +ve z  
    
    r = 0.5 * np.cbrt(np.random.random(num_points)) #radius sampling in 3D a volume
    #projection on coordinates
    x = r*np.sin(thetas) * np.cos(phis)
    y = r*np.sin(thetas) * np.sin(phis)
    z = r*np.cos(thetas)

    mean = np.stack([x,y,z], axis=1)
    sh_coefficients = np.random.random((num_points, 3)) / 255.0

    pcd = PointCloud(
        points=torch.tensor(np.asarray(mean)).float().cuda(),
        sh_coefficients=torch.tensor(np.asarray(sh_coefficients)).float().cuda(),
    )
    
    return pcd



def convert_pcd_to_gaussians(pcd):
    """Convert a PointCloud to a list of GaussianPrimitive objects."""
    points = pcd.points  #  on gpu
    colors = pcd.sh_coefficients

    N = points.shape[0]
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # [N, N, 3]
    dist2 = (diff * diff).sum(-1)  # [N, N]
    k = 5
    knn_dists = torch.topk(dist2, k=k, largest=False, dim=1).values  # [N, k]
    knn_dists = knn_dists[:, 1:k]  # skip self-distance
    avg_dist2 = knn_dists.mean(dim=1)
    avg_dist3 = torch.clamp_min(avg_dist2,0.0000001)
    svec = torch.log(torch.sqrt(avg_dist3))[..., None].repeat(1, 3)  # [N, 3]

    # Initialize quaternions (identity)
    quaternions = torch.zeros((N, 4), device="cuda")
    quaternions[:, 0] = 1.0

    # Initial opacity (inverse sigmoid of small value)
    opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), device="cuda"))

    # Initialize SH features
    sh_order = 3
    feature_dim = (sh_order + 1) ** 2
    features = torch.zeros((N, 3, feature_dim), device="cuda")
    features[:, :, 0] = colors  # fill DC term (band 0)
    features[:, :, 1:] = 0.0  # AC components zero

    # Construct Gaussians
    gaussians = []
    for i in range(N):
        g = GaussianPrimitive(
            mean=points[i],
            sh_coefficients_dc=features[i, :, 0:1].transpose(0, 1).contiguous(),
            sh_coefficients_ac=features[i, :, 1:].transpose(0, 1).contiguous(),
            svec=svec[i],
            quaternion=quaternions[i],
            opacity=opacities[i],
        )
        gaussians.append(g)

    return gaussians





