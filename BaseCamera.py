import numpy as np
import torch
from typing import Optional

class BaseCamera:
    """Base class for cameras, providing shared properties and methods."""
    
    def __init__(self, width: int, height: int, fovy: float, znear: float, zfar: float):
        """
        Initialize the base camera with common parameters.

        Args:
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            fovy (float): Vertical field of view in radians.
            znear (float): Near clipping plane distance.
            zfar (float): Far clipping plane distance.
        """
        self.image_width = width
        self.image_height = height
        self.fovy = fovy
        self.znear = znear
        self.zfar = zfar

    @property
    def fovx(self) -> float:
        """Calculate horizontal FOV from vertical FOV and aspect ratio."""
        aspect_ratio = self.image_width / self.image_height
        return 2 * np.arctan(np.tan(self.fovy / 2) * aspect_ratio)

    @property
    def intrinsics(self) -> np.ndarray:
        """Calculate camera intrinsics as [fx, fy, cx, cy]."""
        fx = self.image_width / (2 * np.tan(self.fovx / 2))
        fy = self.image_height / (2 * np.tan(self.fovy / 2))
        cx = self.image_width / 2
        cy = self.image_height / 2
        return np.array([fx, fy, cx, cy], dtype=np.float32)

    def get_projection_matrix(self) -> np.ndarray:
        """Generate a perspective projection matrix."""
        y = np.tan(self.fovy / 2)
        aspect = self.image_width / self.image_height
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],  # Flip Y-axis for OpenGL convention
                [0, 0, -(self.zfar + self.znear) / (self.zfar - self.znear), 
                 -(2 * self.zfar * self.znear) / (self.zfar - self.znear)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    def get_view_matrix(self) -> Optional[np.ndarray]:
        """Placeholder for view matrix, to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the `get_view_matrix` method.")

    def get_pose_matrix(self) -> Optional[np.ndarray]:
        """Placeholder for pose matrix, to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the `get_pose_matrix` method.")