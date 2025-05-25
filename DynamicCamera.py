import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from typing import Union, Optional, Tuple

# Type alias for functions that can accept either numpy arrays or torch tensors
ArrayOrTensor = Union[np.ndarray, torch.Tensor]

def dot(x: ArrayOrTensor, y: ArrayOrTensor) -> ArrayOrTensor:
    """Compute dot product along last dimension for 3D vectors.
    
    Handles both numpy arrays and torch tensors with appropriate method calls.
    The keepdims/keepdim preserves the input dimensionality for broadcasting.
    
    Args:
        x: Input array/tensor of shape [..., 3] (batch of 3D vectors)
        y: Input array/tensor of shape [..., 3] (must match x's shape)
    
    Returns:
        Dot product result of shape [..., 1] (scalar per vector)
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)  # numpy version
    else:
        return torch.sum(x * y, -1, keepdim=True)  # torch version

def length(x: ArrayOrTensor, eps: float = 1e-20) -> ArrayOrTensor:
    """Compute Euclidean length of 3D vectors with numerical stability.
    
    Adds epsilon to prevent division by zero in case of zero-length vectors.
    Uses different implementations for numpy vs torch while maintaining same behavior.
    
    Args:
        x: Input array/tensor of shape [..., 3]
        eps: Small value added for numerical stability
    
    Returns:
        Length values of shape [..., 1]
    """
    if isinstance(x, np.ndarray):
        # numpy uses maximum to ensure we never take sqrt of negative numbers
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        # torch uses clamp for same purpose
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x: ArrayOrTensor, eps: float = 1e-20) -> ArrayOrTensor:
    """Normalize vectors to unit length safely.
    
    Divides each vector by its length, protected against division by zero.
    Maintains the input type (numpy array or torch tensor).
    
    Args:
        x: Input array/tensor of shape [..., 3]
        eps: Small value to prevent division by zero
    
    Returns:
        Normalized vectors of same shape as input
    """
    return x / length(x, eps)

def look_at(
    campos: np.ndarray, 
    target: np.ndarray, 
    opengl: bool = True
) -> np.ndarray:
    """Generate camera rotation matrix looking from campos to target.
    
    Constructs a coordinate system where:
    - Forward axis points toward/away from target based on opengl flag
    - Up axis tries to maintain world-up (0,1,0) where possible
    - Right axis is orthogonal to forward and up
    
    Args:
        campos: Camera position(s) of shape [N, 3]
        target: Target point(s) to look at of shape [N, 3]
        opengl: If True, +Z is forward (OpenGL). If False, -Z is forward.
    
    Returns:
        Rotation matrices of shape [N, 3, 3] where each matrix is orthonormal
    """
    if not opengl:
        # Non-OpenGL convention: forward points from camera to target (-Z)
        forward = safe_normalize(target - campos)
        up = np.array([0, 1, 0], dtype=np.float32)  # world up
        right = safe_normalize(np.cross(forward, up))  # right-handed system
        up = safe_normalize(np.cross(right, forward))  # recompute up to ensure orthogonality
    else:
        # OpenGL convention: forward points from target to camera (+Z)
        forward = safe_normalize(campos - target)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = safe_normalize(np.cross(up, forward))  # right-handed system
        up = safe_normalize(np.cross(forward, right))  # recompute up
    
    # Stack columns to form rotation matrix: [right, up, forward]
    return np.stack([right, up, forward], axis=1)

def rotate_camera(
    elevation: float,
    azimuth: float,
    radius: float = 1,
    is_degree: bool = True,
    target: Optional[np.ndarray] = None,
    opengl: bool = True
) -> np.ndarray:
    """Generate camera pose matrix from spherical coordinates.
    
    Converts spherical coordinates (elevation, azimuth, radius) to:
    1. Cartesian camera position
    2. Rotation matrix looking at target
    3. Combined 4x4 pose matrix
    
    Spherical coordinates:
    - Elevation: angle above/below horizon (-90° to +90°)
    - Azimuth: angle around vertical axis (0° to 360°)
    - Radius: distance from target
    
    Args:
        elevation: Vertical angle in degrees (-90 to 90)
        azimuth: Horizontal angle in degrees (-180 to 180)
        radius: Distance from target
        is_degree: If True, angles are in degrees (converted to radians)
        target: Target point to look at (origin if None)
        opengl: Whether to use OpenGL coordinate convention
    
    Returns:
        4x4 camera pose matrix in world coordinates (camera-to-world)
    """
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    
    # Convert spherical to Cartesian coordinates (right-handed system)
    x = radius * np.cos(elevation) * np.sin(azimuth)  # x = r*cos(θ)*sin(φ)
    y = -radius * np.sin(elevation)                   # y = -r*sin(θ)
    z = radius * np.cos(elevation) * np.cos(azimuth)  # z = r*cos(θ)*cos(φ)
    
    if target is None:
        target = np.zeros(3, dtype=np.float32)
    
    campos = np.array([x, y, z]) + target  # offset by target position
    
    pose = np.eye(4, dtype=np.float32)  # initialize 4x4 identity
    pose[:3, :3] = look_at(campos, target, opengl)  # set rotation part
    pose[:3, 3] = campos  # set translation part
    
    return pose

class DynamicCamera:
    """Interactive camera controller for orbit navigation.
    
    Maintains camera state (position, orientation) and provides:
    - Properties for view/projection matrices
    - Methods for interactive control (orbit, zoom, pan)
    - Supports OpenGL coordinate convention (+Z forward)
    """
    
    def __init__(self, W, H, radius=2, fovy=60, near=0.01, far=100):
        """Initialize camera with viewport and projection parameters.
        
        Args:
            W: Viewport width in pixels
            H: Viewport height in pixels
            radius: Initial orbit distance from center
            fovy: Vertical field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
        """
        self.W = W
        self.H = H
        self.radius = radius  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # convert to radians
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look-at point
        self.rot = R.from_matrix(np.eye(3))  # initial rotation (identity)
        self.up = np.array([0, 1, 0], dtype=np.float32)  # world up vector

    @property
    def fovx(self):
        """Calculate horizontal FOV from vertical FOV and aspect ratio."""
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        """Camera position in world coordinates (extracted from pose)."""
        return self.pose[:3, 3]

    def get_pose_matrix(self):
        """Camera-to-world transformation matrix (4x4).
        
        Constructed by:
        1. Placing camera at radius distance along -Z (OpenGL looks toward +Z)
        2. Applying current rotation
        3. Translating to orbit center
        """
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # initial position (OpenGL convention)
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()  # apply current rotation
        res = rot @ res  # rotate camera position
        res[:3, 3] -= self.center  # translate relative to center
        return res

    def get_view_matrix(self):
        """World-to-camera view matrix (inverse of pose)."""
        return np.linalg.inv(self.pose)

    @property
    def perspective(self):
        """Perspective projection matrix (OpenGL convention).
        
        Projects 3D points to 2D viewport coordinates.
        Note: Y-axis is flipped (-1/y term) because screen coordinates increase downward.
        """
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],  # x scaling
                [0, -1 / y, 0, 0],             # y scaling (flipped)
                [0, 0, -(self.far + self.near) / (self.far - self.near),  # depth mapping
                 -(2 * self.far * self.near) / (self.far - self.near)],
                [0, 0, -1, 0],  # perspective divide
            ],
            dtype=np.float32,
        )

    @property
    def intrinsics(self):
        """Camera intrinsics as [fx, fy, cx, cy].
        
        fx,fy: focal lengths in pixels
        cx,cy: principal point (image center)
        """
        focal = self.H / (2 * np.tan(self.fovy / 2))  # fy (assume fx = fy)
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        """Model-View-Projection matrix for vertex transformation.
        
        Computed as: projection @ view (transforms world to clip space)
        """
        return self.perspective @ np.linalg.inv(self.pose)

    def orbit(self, dx, dy):
        """Rotate camera around target point.
        
        Args:
            dx: Horizontal mouse delta (pixels)
            dy: Vertical mouse delta (pixels)
        """
        side = self.rot.as_matrix()[:3, 0]  # camera's right vector
        rotvec_x = self.up * np.radians(-0.05 * dx)  # horizontal rotation around world up
        rotvec_y = side * np.radians(-0.05 * dy)  # vertical rotation around camera right
        # Apply rotations (order matters: y then x)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        """Zoom camera by adjusting orbit radius.
        
        Args:
            delta: Mouse wheel delta (positive zooms in)
        """
        self.radius *= 1.1 ** (-delta)  # exponential scaling

    def pan(self, dx, dy, dz=0):
        """Pan camera by moving the look-at center.
        
        Args:
            dx: Horizontal pan (pixels)
            dy: Vertical pan (pixels)
            dz: Depth pan (pixels)
        """
        # Transform screen deltas to camera space and apply
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])