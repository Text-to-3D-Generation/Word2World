import torch
import numpy as np
import BaseCamera
from misc_utils import get_projection_matrix  # External projection matrix generator

class StaticCamera (BaseCamera):
    """A fixed-position camera for rendering, following NeRF's coordinate conventions.
    
    This camera cannot be moved interactively - it represents a pre-defined viewpoint
    from datasets like NeRF or COLMAP. All transforms are pre-computed during initialization
    and stored as PyTorch tensors on GPU for efficient rendering.
    """

    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        """
        Initialize a static camera with fixed parameters.

        Args:
            c2w (np.ndarray[float32]): 4x4 camera-to-world matrix in NeRF convention.
                                      This transforms points from camera to world space.
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            fovy (float): Vertical field of view in radians.
            fovx (float): Horizontal field of view in radians.
            znear (float): Near clipping plane distance.
            zfar (float): Far clipping plane distance.
        """
        # Store basic camera properties
        self.image_width = width    # Viewport width (pixels)
        self.image_height = height  # Viewport height (pixels)
        self.FoVy = fovy           # Vertical FOV (radians)
        self.FoVx = fovx           # Horizontal FOV (radians)
        self.znear = znear         # Near clipping distance
        self.zfar = zfar           # Far clipping distance

        # Convert camera-to-world (c2w) to world-to-camera (w2c) matrix
        # This inverts the transformation (world -> camera space)
        w2c = np.linalg.inv(c2w)

        # NeRF Convention Adjustment:
        # NeRF uses a different coordinate system than standard OpenGL
        # Flip Y and Z axes to convert from OpenGL-style to NeRF-style coordinates
        w2c[1:3, :3] *= -1  # Flip Y and Z rotation axes
        w2c[:3, 3] *= -1    # Flip translation components

        # Convert to PyTorch tensors and send to GPU
        # Transpose to match PyTorch's matrix multiplication conventions
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        
        # Generate the perspective projection matrix
        # This matrix projects 3D points to 2D image coordinates
        self.projection_matrix = (
            get_projection_matrix(
                z_near=self.znear, 
                z_far=self.zfar, 
                fov_x=self.FoVx, 
                fov_y=self.FoVy
            )
            .transpose(0, 1)  # Transpose for PyTorch compatibility
            .cuda()           # Move to GPU
        )

        # Combined full projection transform (MVP matrix)
        # Computed as: Projection @ View
        # Transforms world space -> clip space
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        # Camera center in world coordinates
        # Extracted from c2w translation and negated (NeRF convention)
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    # Note: No transform update methods - this is a static camera!
    # All transforms are computed once during initialization


    def get_pose_matrix(self):
        """Returns the camera-to-world matrix (c2w) in NeRF convention."""
        w2c = self.world_view_transform.cpu().numpy().T
        w2c[1:3, :3] *= -1  # Reverse NeRF adjustment
        w2c[:3, 3] *= -1
        return np.linalg.inv(w2c)

    def get_view_matrix(self):
        """Returns the world-to-camera matrix (w2c)."""
        return self.world_view_transform.cpu().numpy().T

    @property
    def intrinsics(self):
        """Returns camera intrinsics as [fx, fy, cx, cy]."""
        fx = self.image_width / (2 * np.tan(self.FoVx / 2))
        fy = self.image_height / (2 * np.tan(self.FoVy / 2))
        return np.array([
            fx, fy,
            self.image_width / 2,  # Principal point X (cx)
            self.image_height / 2  # Principal point Y (cy)
        ], dtype=np.float32)