import torch
import math
import numpy as np
from typing import List, Optional, Dict, Union
from PointE import PointEModel
from gaussians_handler import GaussiansHandler
from diff_gaussian_rasterization import _C 



# ----------------------------------------------------------------------
#  CUDA bridge – pulled out of diff_gaussian_rasterization/__init__.py
# ----------------------------------------------------------------------
import torch
from typing import Tuple

def fill_missing_attributes(means3D, shs, colors_precomp, scales, rotations, cov3D_precomp):
    empty_like = lambda x: torch.empty(0, device=means3D.device, dtype=x.dtype if isinstance(x, torch.Tensor) else torch.float32)

    values = {
        'shs': shs,
        'colors_precomp': colors_precomp,
        'scales': scales,
        'rotations': rotations,
        'cov3D_precomp': cov3D_precomp,
    }

    for key in values:
        if values[key] is None:
            values[key] = empty_like(means3D)

    return values['shs'], values['colors_precomp'], values['scales'], values['rotations'], values['cov3D_precomp']



# --- direct rasterization --------------------------------------------
def _render_gaussians_inline(
        means3D, means2D, shs, colors_precomp, opacity,
        scales, rotations, cov3D_precomp,
        bg,                           # -- formerly rs.bg
        scale_modifier,               # -- rs.scale_modifier
        viewmatrix, projmatrix,       # -- rs.viewmatrix / rs.projmatrix
        tanfovx, tanfovy,             # -- rs.tanfovx / rs.tanfovy
        sh_degree, campos,            # -- rs.sh_degree / rs.campos
        prefiltered=False, debug=False):
    """
    Inline Gaussian rasterizer that hides the autograd.Function
    instead of declaring it at module scope.  All raster-settings
    are passed explicitly (no SimpleNamespace needed).
    """
    scale_modifier = 1
    # ------------------------------------------------------------
    # 1.  Fill in any missing per-point attributes (unchanged)
    # ------------------------------------------------------------
    shs, colors_precomp, scales, rotations, cov3D_precomp = fill_missing_attributes(
        means3D, shs, colors_precomp, scales, rotations, cov3D_precomp
    )
    focal_x = 800/(2*tanfovx)
    focal_y = 800/(2*tanfovy)

    # ------------------------------------------------------------
    # 2.  Lazily create (and cache) the autograd.Function
    # ------------------------------------------------------------
    if not hasattr(_render_gaussians_inline, "_Func"):

        class _Func(torch.autograd.Function):

            # ---------- forward ----------
            @staticmethod
            def forward(ctx,
                        means3D, means2D, sh, colors_precomp, opacities,
                        scales, rotations, cov3Ds_precomp,
                        bg, scale_modifier, viewmatrix, projmatrix,
                        tanfovx, tanfovy, sh_degree, campos,
                        prefiltered, debug):

                num_rend, color, depth, alpha, \
                radii, geomBuf, binBuf, imgBuf = _C.rasterize_gaussians(
                    sh,            # 1
                    focal_y,       # 2
                    scales,        # 3
                    means3D,       # 4
                    projmatrix,    # 5
                    sh_degree,     # 6
                    opacities,     # 7
                    rotations,     # 8
                    campos,        # 9
                    viewmatrix,    # 10
                    focal_x        # 11
                )

                # ---- save context ----
                ctx.num_rend  = num_rend
                ctx.bg        = bg
                ctx.scale_mod = scale_modifier
                ctx.vmat      = viewmatrix
                ctx.pmat      = projmatrix
                ctx.fx        = tanfovx
                ctx.fy        = tanfovy
                ctx.sh_deg    = sh_degree
                ctx.campos    = campos
                ctx.debug     = debug

                ctx.save_for_backward(colors_precomp, means3D, scales, rotations,
                                      cov3Ds_precomp, radii, sh,
                                      geomBuf, binBuf, imgBuf, alpha)

                return color, radii, depth, alpha

            # ---------- backward ----------
            @staticmethod
            def backward(ctx, g_color, g_radii, g_depth, g_alpha):

                (colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
                 radii, sh, geomBuf, binBuf, imgBuf, alpha) = ctx.saved_tensors

                g_means2D, g_colors, g_opac, g_means3D, \
                g_cov3D,  g_sh,     g_scales, g_rots = _C.rasterize_gaussians_backward(
                    ctx.bg,
                    means3D,
                    radii,
                    colors_precomp,
                    scales,
                    rotations,
                    ctx.scale_mod,
                    cov3Ds_precomp,
                    ctx.vmat,
                    ctx.pmat,
                    ctx.fx,
                    ctx.fy,
                    g_color,
                    g_depth,
                    g_alpha,
                    sh,
                    ctx.sh_deg,
                    ctx.campos,
                    geomBuf,
                    ctx.num_rend,
                    binBuf,
                    imgBuf,
                    alpha,
                    ctx.debug,
                )

                # Return gradients for every forward input (None for scalars)
                return (
                    g_means3D,   # means3D
                    g_means2D,   # means2D
                    g_sh,        # sh
                    g_colors,    # colors_precomp
                    g_opac,      # opacities
                    g_scales,    # scales
                    g_rots,      # rotations
                    g_cov3D,     # cov3Ds_precomp
                    None, None, None, None, None, None, None, None, None, None
                )

        _render_gaussians_inline._Func = _Func

    # ------------------------------------------------------------
    # 3.  Invoke the cached Function
    # ------------------------------------------------------------
    return _render_gaussians_inline._Func.apply(
        means3D, means2D, shs, colors_precomp, opacity,
        scales, rotations, cov3D_precomp,
        bg, scale_modifier, viewmatrix, projmatrix,
        tanfovx, tanfovy, sh_degree, campos,
        prefiltered, debug
    )





# ----------------------------------------------------------------------


class Renderer:
    """Differentiable 3D Gaussian Splatting renderer
    """
    
    def __init__(self, pcd, white_background: bool = True):
        self.white_background = white_background
        self.gaussians_handler = GaussiansHandler(pcd)
        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda"
        )
    


    def render(
        self,
        viewpoint_camera,
        scaling_modifier: float = 1.0,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scaling_modifier = 1
        """
        Render the Gaussian model from a given viewpoint.

        Args:
            viewpoint_camera: Camera parameters.
            scaling_modifier: Scale adjustment factor (default=1.0).
            bg_color: Optional background-color override.
            override_color: Optional per-point color override.
            compute_cov3D_python: Unused here (kept for API parity).
            convert_SHs_python:  Unused here (kept for API parity).

        Returns:
            dict with 'image', 'depth', 'alpha', 'viewspace_points',
            'visibility_filter', and 'radii'.
        """
        # ------------------------------------------------------------
        # 1.  Differentiable screen-space placeholder (unchanged)
        # ------------------------------------------------------------
        total_mean = []
        for gaussian in self.gaussians_handler.gaussians:
            total_mean.append(gaussian.mean)
        total_mean = torch.stack(total_mean)
        # total_mean = self.gaussians_handler.get_param_group_by_name("mean")

        screenspace_points = torch.zeros_like(
            total_mean,
            dtype=total_mean.dtype,
            requires_grad=True,
            device="cuda",
        )
        # try:
        screenspace_points.retain_grad()
        # except RuntimeError:
        #     pass

        # ------------------------------------------------------------
        # 2.  Camera / render parameters (all scalars & matrices)
        # ------------------------------------------------------------
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        bg          = self.bg_color if bg_color is None else bg_color
        viewmatrix  = viewpoint_camera.world_view_transform
        projmatrix  = viewpoint_camera.full_proj_transform
        campos      = viewpoint_camera.camera_center
        prefiltered = False
        debug       = False

        # ------------------------------------------------------------
        # 3.  Point-cloud attributes
        # ------------------------------------------------------------

        total_mean = []
        total_sh_coefficients_dc = []
        total_sh_coefficients_ac = []
        total_opacity = []
        total_svec = []
        total_quaternion = []

        for gaussian in self.gaussians_handler.gaussians:
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

        # total_mean = self.gaussians_handler.get_param_group_by_name("mean")
        # total_sh_coefficients_dc = self.gaussians_handler.get_param_group_by_name("sh_coefficients_dc")
        # total_sh_coefficients_ac = self.gaussians_handler.get_param_group_by_name("sh_coefficients_ac")
        # total_opacity = self.gaussians_handler.get_param_group_by_name("opacity")
        # total_svec = self.gaussians_handler.get_param_group_by_name("svec")
        # total_quaternion = self.gaussians_handler.get_param_group_by_name("quaternion")


        means3D   = total_mean
        means2D   = screenspace_points
        opacity   = torch.sigmoid(total_opacity)
        scales    = torch.exp(total_svec)
        rotations = torch.nn.functional.normalize(total_quaternion)

        cov3D_precomp = None            # keeping the hook for future use
        shs             = torch.cat((total_sh_coefficients_dc, total_sh_coefficients_ac), dim=1)
        colors_precomp = None           # override_color path omitted for brevity

        # ------------------------------------------------------------
        # 4.  Rasterize
        # ------------------------------------------------------------
        try:
            rendered_image, radii, rendered_depth, rendered_alpha = _render_gaussians_inline(
                means3D, means2D, shs, colors_precomp, opacity,
                scales, rotations, cov3D_precomp,
                bg, scaling_modifier,
                viewmatrix, projmatrix,
                tanfovx, tanfovy,
                0, campos,
                prefiltered, debug
            )
            torch.cuda.synchronize() 

    # -------- handle only CUDA illegal-access errors --------------------
        except RuntimeError as e:
            if "illegal memory access" in str(e):
                print("[WARN] CUDA illegal access in _render_gaussians_inline – "
                    "skipping this frame")
                torch.cuda.empty_cache()      # free any leaked buffers
                torch.cuda.synchronize()      # flush the error state

                H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
                dtype = means3D.dtype
                device = means3D.device

                rendered_image = torch.zeros((H, W, 3), device=device, dtype=dtype)
                rendered_depth = torch.full((H, W), float("inf"), device=device, dtype=dtype)
                rendered_alpha = torch.zeros((H, W), device=device, dtype=dtype)
                radii          = torch.zeros(means3D.shape[0], device=device, dtype=dtype)

            else:
                # any *other* runtime error is still a real bug
                raise

        # ------------------------------------------------------------
        # 5.  Package outputs
        # ------------------------------------------------------------
        
        return {
            "image":  rendered_image.clamp(0, 1),
            "depth":  rendered_depth,
            "alpha":  rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii":  radii,
        }
