import torch
from scipy.spatial.transform import Rotation as R
from typing import Optional

def vector_ops(x, eps: float = 1e-20, normalize: bool = False):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    length = torch.sqrt(torch.clamp(torch.dot(x, x), min=eps))
    return x / length if normalize else length

def compute_fovx(fovy, width, height):
    return (2 * torch.atan(torch.tan(fovy / 2) * width / height)).cpu().numpy()

def dot(x, y):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    return torch.sum(x*y, -1)

def build_pose_matrix(rot, radius, center):
    T = torch.eye(4)
    T[2, 3] = radius
    R4 = torch.eye(4)
    R4[:3, :3] = torch.tensor(rot.as_matrix(), dtype=torch.float32)
    pose = torch.matmul(R4, T)
    pose[:3, 3] -= center
    return pose

def compute_perspective(fovy, width, height, near, far):
    y = torch.tan(fovy / 2)
    aspect = width / height
    persp = torch.tensor([
        [1 / (y * aspect), 0, 0, 0],
        [0, -1 / y, 0, 0],
        [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
        [0, 0, -1, 0]
    ], dtype=torch.float32)
    return persp.cpu().numpy()

