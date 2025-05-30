import numpy as np
import pymeshlab as pml
from typing import Tuple
import torch
import kiui
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter
from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.util.numpy_support import numpy_to_vtk
from pyvista import wrap
from TriangleMesh import TriangleMesh
import os
from misc_utils import  extract_symmetric, compute_3d_gaussian_coefficient, build_scaling_rotation_matrix

def covariance_activation(scaling, scaling_modifier, rotation):
        """Construct covariance matrix from scaling and rotation for all Gaussians"""
        L = build_scaling_rotation_matrix(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return extract_symmetric(actual_covariance)

def simplify_mesh_geometry(
    vertices: np.ndarray,
    triangles: np.ndarray,
    face_goal: int,
    engine: str = "pymeshlab",
    do_remesh: bool = False,
    use_optimal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce triangle count in a mesh using the specified simplification engine.
    """

    original_v_shape = vertices.shape
    original_f_shape = triangles.shape

    match engine:
        case "pyfqmr":
            import pyfqmr

            decimator = pyfqmr.Simplify()
            decimator.setMesh(vertices, triangles)
            decimator.simplify_mesh(
                target_count=face_goal, preserve_border=False, verbose=False
            )
            vertices, triangles, _ = decimator.getMesh()

        case "pymeshlab":
            mesh = pml.Mesh(vertices, triangles)
            mesh_set = pml.MeshSet()
            mesh_set.add_mesh(mesh, "input")

            mesh_set.meshing_decimation_quadric_edge_collapse(
                targetfacenum=face_goal,
                optimalplacement=use_optimal,
            )

            if do_remesh:
                mesh_set.meshing_isotropic_explicit_remeshing(
                    iterations=3,
                    targetlen=pml.PercentageValue(1),
                )

            output = mesh_set.current_mesh()
            vertices = output.vertex_matrix()
            triangles = output.face_matrix()

        case _:
            raise ValueError(f"Unsupported engine '{engine}' provided.")

    print(
        f"[LOG] Mesh simplification: {original_v_shape} → {vertices.shape}, {original_f_shape} → {triangles.shape}"
    )
    return vertices, triangles


def refine_mesh_topology(
    vertices: np.ndarray,
    triangles: np.ndarray,
    merge_threshold_pct: float = 1.0,
    min_faces: int = 64,
    min_diameter_pct: float = 20.0,
    auto_repair: bool = True,
    auto_remesh: bool = True,
    target_edge_length: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean and optionally remesh a mesh to remove redundant data and improve topology.
    """

    original_v_shape = vertices.shape
    original_f_shape = triangles.shape

    mesh = pml.Mesh(vertices, triangles)
    mesh_set = pml.MeshSet()
    mesh_set.add_mesh(mesh, "to_clean")

    # Remove unlinked vertices
    mesh_set.meshing_remove_unreferenced_vertices()

    # Merge close vertices if applicable
    if merge_threshold_pct > 0:
        mesh_set.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(merge_threshold_pct)
        )

    # Eliminate degenerate and duplicate faces
    mesh_set.meshing_remove_duplicate_faces()
    mesh_set.meshing_remove_null_faces()

    # Remove small disconnected components
    if min_diameter_pct > 0:
        mesh_set.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_diameter_pct)
        )

    if min_faces > 0:
        mesh_set.meshing_remove_connected_component_by_face_number(
            mincomponentsize=min_faces
        )

    # Repair non-manifold geometry
    if auto_repair:
        mesh_set.meshing_repair_non_manifold_edges(method=0)
        mesh_set.meshing_repair_non_manifold_vertices(vertdispratio=0)

    # Optional isotropic remeshing
    if auto_remesh:
        mesh_set.meshing_isotropic_explicit_remeshing(
            iterations=3,
            targetlen=pml.PureValue(target_edge_length),
        )

    final_mesh = mesh_set.current_mesh()
    vertices = final_mesh.vertex_matrix()
    triangles = final_mesh.face_matrix()

    print(
        f"[LOG] Mesh cleanup: {original_v_shape} → {vertices.shape}, {original_f_shape} → {triangles.shape}"
    )
    return vertices, triangles



@torch.no_grad()
def extract_fields(total_mean,total_opacity,total_svec,total_quaternion,resolution=128, num_blocks=16, relax_ratio=1.5):
        """Extract occupancy field for mesh extraction"""
        block_size = 2 / num_blocks
        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = torch.sigmoid(total_opacity)
        mask = (opacities > 0.005).squeeze(1)
        
        means = total_mean[mask]
        stds = torch.exp(total_svec)[mask]
        
        # Normalize to ~ [-1, 1]
        mn, mx = means.amin(0), means.amax(0)
        center = (mn + mx) / 2
        scale = 1.8 / (mx - mn).amax().item()

        means = (means - center)*scale
        stds = stds*scale

        covs = covariance_activation(stds, 1, total_quaternion[mask])

        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (means < vmax).all(-1) & (means > vmin).all(-1)
                    
                    if not mask.any():
                        continue
                        
                    mask_means = means[mask]
                    mask_covs = covs[mask]
                    mask_opas = opacities[mask].view(1, -1)

                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_means.unsqueeze(0)
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1)

                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = compute_3d_gaussian_coefficient(
                            g_pts[:, start:end].reshape(-1, 3), 
                            g_covs[:, start:end].reshape(-1, 6)
                        ).reshape(pts.shape[0], -1)
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs))
        
        kiui.lo(occ, verbose=1)
        return occ,center,scale

def extract_tetrahedral_mesh(gaussians,path, density_thresh=1, resolution=128, decimate_target=1e5):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        total_mean = []
        total_sh_coefficients_dc = []
        total_sh_coefficients_ac = []
        total_opacity = []
        total_svec = []
        total_quaternion = []

        for gaussian in gaussians:
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
        occ,center,scale = extract_fields(total_mean,total_opacity,total_svec,total_quaternion,resolution)
        occ = occ.detach().cpu().numpy()
        flat = occ.flatten(order="F")

        img = vtkImageData()
        img.SetDimensions(resolution, resolution, resolution)
        spacing = 2.0 / (resolution - 1)
        img.SetSpacing(spacing, spacing, spacing)
        img.SetOrigin(-1, -1, -1)

        vtk_arr = numpy_to_vtk(flat, deep=True, array_type=VTK_FLOAT)
        vtk_arr.SetName("densities")
        img.GetPointData().SetScalars(vtk_arr)

        tri = vtkDataSetTriangleFilter()
        tri.SetInputData(img)
        tri.Update()
        tetGrid = tri.GetOutput()

        contour = vtkContour3DLinearGrid()
        contour.SetInputData(tetGrid)
        contour.SetValue(0, density_thresh)
        contour.SetMergePoints(True)
        contour.Update()

        cont = wrap(contour.GetOutput())
        verts = cont.points
        faces = cont.faces.reshape(-1, 4)[:, 1:]

        verts = verts/scale+center.detach().cpu().numpy()
        verts, faces = refine_mesh_topology(verts, faces, remesh=True, remesh_size=0.015)
        
        if decimate_target > 0 and faces.shape[0] > decimate_target:
            verts, faces = simplify_mesh_geometry(verts, faces, decimate_target)

        v = torch.from_numpy(verts.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(faces.astype(np.int32)).contiguous().cuda()
        return TriangleMesh(vertices=v, faces=f, device="cuda")