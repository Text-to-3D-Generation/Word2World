import torch
import numpy as np
from typing import Callable
import math

def inverse_sigmoid(y):
    torch.clamp(y, 1e-6, 1 - 1e-6)
    return torch.log(y/(1 - y))


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000
) -> Callable[[int], float]:
    def learning_rate_schedule(step: int) -> float:
        """Calculate learning rate for current step."""
        # Constant learning rate case
        if lr_init == lr_final:
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return learning_rate_schedule


def extract_symmetric(L: torch.Tensor) -> torch.Tensor:
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]  # xx
    uncertainty[:, 1] = L[:, 0, 1]  # xy
    uncertainty[:, 2] = L[:, 0, 2]  # xz
    uncertainty[:, 3] = L[:, 1, 1]  # yy
    uncertainty[:, 4] = L[:, 1, 2]  # yz
    uncertainty[:, 5] = L[:, 2, 2]  # zz
    
    return uncertainty

def flatten_sh_coefficients(sh_coefficients: torch.Tensor) -> np.ndarray:
        """Helper function to flatten the spherical harmonics coefficients"""
        return sh_coefficients.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()


def compute_3d_gaussian_coefficient(
    means: torch.Tensor,
    covs: torch.Tensor
) -> torch.Tensor:
    """Compute coefficients for 3D Gaussian distribution.
    
    Args:
        means: Points in 3D space (N, 3)
        covs: Covariance matrix elements (N, 6) in order:
              [xx, xy, xz, yy, yz, zz]
    
    Returns:
        torch.Tensor: Gaussian coefficients for each point
    """
    x, y, z = means[:, 0], means[:, 1], means[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # Calculate inverse covariance matrix elements
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    # Compute Gaussian exponent
    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) \
            - x * y * inv_b - x * z * inv_c - y * z * inv_e

    # Handle numerical instability (replace invalid values with very small weights)
    power[power > 0] = -1e10
    
    return torch.exp(power)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.
    
    Args:
        q: Quaternions in (w, x, y, z) format (N, 4)
    
    Returns:
        torch.Tensor: Rotation matrices (N, 3, 3)
    """
    # Normalize quaternions
    norm = torch.sqrt(q[:,0]*q[:,0] + q[:,1]*q[:,1] + q[:,2]*q[:,2] + q[:,3]*q[:,3])
    q = q / norm[:, None]

    # Initialize rotation matrices
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    
    # Extract quaternion components
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Compute rotation matrix elements
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    return R


def build_scaling_rotation_matrix(
    scales: torch.Tensor,
    rotations: torch.Tensor
) -> torch.Tensor:
    """Combine scaling and rotation into a transformation matrix.
    
    Args:
        scales: Scaling factors (N, 3)
        rotations: Rotation quaternions (N, 4)
    
    Returns:
        torch.Tensor: Combined transformation matrices (N, 3, 3)
    """
    # Initialize scaling matrices
    L = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float, device="cuda")
    
    # Create rotation matrices
    R = quaternion_to_rotation_matrix(rotations)

    # Apply scaling
    L[:, 0, 0] = scales[:, 0]  # x scaling
    L[:, 1, 1] = scales[:, 1]  # y scaling
    L[:, 2, 2] = scales[:, 2]  # z scaling

    # Combine rotation and scaling
    return R @ L


def get_projection_matrix(z_near: float, z_far: float, fov_x: float, fov_y: float) -> torch.Tensor:
    """
    Constructs a perspective projection matrix for 3D rendering.

    Args:
        z_near (float): Distance to the near clipping plane.
        z_far (float): Distance to the far clipping plane.
        fov_x (float): Field of view in the X direction (in radians).
        fov_y (float): Field of view in the Y direction (in radians).

    Returns:
        torch.Tensor: A 4x4 projection matrix.
    """
    # Calculate tangents of half the field of view angles
    tan_half_fov_x = math.tan(fov_x / 2.0)
    tan_half_fov_y = math.tan(fov_y / 2.0)

    # Initialize projection matrix with zeros
    projection = torch.zeros(4, 4)

    # Assume a right-handed coordinate system with positive Z pointing forward
    z_sign = 1.0

    # Set matrix values according to perspective projection formula
    projection[0, 0] = 1.0 / tan_half_fov_x
    projection[1, 1] = 1.0 / tan_half_fov_y
    projection[2, 2] = z_sign * z_far / (z_far - z_near)
    projection[2, 3] = -(z_far * z_near) / (z_far - z_near)
    projection[3, 2] = z_sign

    return projection