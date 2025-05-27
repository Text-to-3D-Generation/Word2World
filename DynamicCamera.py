import torch
from scipy.spatial.transform import Rotation as R
from utils_for_camera import *
class DynamicCamera:
    def __init__(self, W, H, radius=2, fovy=49.1, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = radius
        self.fovy = torch.deg2rad(torch.tensor(fovy, dtype=torch.float32))
        self.near = near
        self.far = far
        self.center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.rot = R.from_matrix(torch.eye(3).numpy())
        self.up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    @property
    def fovx(self):
        return compute_fovx(self.fovy, self.W, self.H)

    @property
    def camera_position(self):
        return self.pose[:3, 3]

    def get_pose_matrix(self):
        return build_pose_matrix(self.rot, self.radius, self.center)

    @property
    def pose(self):
        return self.get_pose_matrix().cpu().numpy()

    def get_perspective(self):
        return compute_perspective(self.fovy, self.W, self.H, self.near, self.far)
