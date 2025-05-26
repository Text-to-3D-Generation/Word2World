import torch
from scipy.spatial.transform import Rotation as R
from typing import Optional


def dot(x, y):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    return torch.sum(x * y, -1)


def length(x, eps: float = 1e-20):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps: float = 1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl: bool = True):
    batch = len(campos.shape) == 2
    campos = campos.unsqueeze(0) if not batch else campos
    target = target.unsqueeze(0) if not batch else target

    forward = safe_normalize(campos - target if opengl else target - campos)
    up = torch.tensor([0, 1, 0], dtype=campos.dtype, device=campos.device).expand_as(forward)

    if opengl:
        right = safe_normalize(torch.cross(up, forward, dim=-1))
        up = safe_normalize(torch.cross(forward, right, dim=-1))
    else:
        right = safe_normalize(torch.cross(forward, up, dim=-1))
        up = safe_normalize(torch.cross(right, forward, dim=-1))

    rot = torch.stack([right, up, forward], dim=-2)
    return rot.squeeze(0).cpu().numpy() if not batch else rot.cpu().numpy()


def rotate_camera(
    elevation: float,
    azimuth: float,
    radius: float = 1,
    is_degree: bool = True,
    target: Optional[torch.Tensor] = None,
    opengl: bool = True
):
    if is_degree:
        elevation = torch.deg2rad(torch.tensor(elevation))
        azimuth = torch.deg2rad(torch.tensor(azimuth))

    x = radius * torch.cos(elevation) * torch.sin(azimuth)
    y = -radius * torch.sin(elevation)
    z = radius * torch.cos(elevation) * torch.cos(azimuth)

    campos = torch.tensor([x, y, z], dtype=torch.float32)
    if target is None:
        target = torch.zeros(3, dtype=torch.float32)

    campos += target

    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = torch.tensor(look_at(campos, target, opengl))
    pose[:3, 3] = campos
    return pose.cpu().numpy()


def compute_fovx(fovy, width, height):
    return (2 * torch.atan(torch.tan(fovy / 2) * width / height)).cpu().numpy()


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


def compute_intrinsics(fovy, width, height):
    focal = height / (2 * torch.tan(fovy / 2))
    return torch.tensor([focal, focal, width // 2, height // 2], dtype=torch.float32).cpu().numpy()

def compute_mvp(pose, perspective):
    persp = torch.tensor(perspective)
    pose_inv = torch.linalg.inv(torch.tensor(pose))
    return (persp @ pose_inv).cpu().numpy()
