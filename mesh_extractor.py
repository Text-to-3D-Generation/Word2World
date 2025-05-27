import torch
import tensorflow as tf
import pymeshlab as pml
from typing import Optional, Dict, Tuple
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid, vtkDataSetTriangleFilter


class MeshDecoderFromGaussians:
    def __init__(self, runtime: str = 'cuda'):
        self.runtime = runtime
        self.origin = None
        self.zoom = None

    def convert_gaussians_to_mesh(
        self,
        blob: object,
        threshold_val: float = 1.0,
        grid_resolution: int = 128,
        poly_limit: int = 100000,
        cleanup_opts: Optional[Dict] = None,
        simplification_opts: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:

        cleanup_opts = cleanup_opts or {
            'vmerge_ratio': 1,
            'minfaces': 64,
            'mindiam': 20,
            'fix_mesh': True,
            'enable_remesh': True,
            'resample_len': 0.01
        }

        simplification_opts = simplification_opts or {
            'method': "pymeshlab",
            'postremesh': False,
            'place_optimal': True
        }

        occupancy_grid = self.extract_fields(blob, grid_resolution)

        raw_verts, raw_tris = self._run_contouring(occupancy_grid, threshold_val)
        raw_verts = raw_verts / self.zoom + self.origin.detach().cpu().numpy()

        cleaned_verts, cleaned_faces = self.perform_cleanup(raw_verts, raw_tris, **cleanup_opts)

        if poly_limit > 0 and cleaned_faces.shape[0] > poly_limit:
            cleaned_verts, cleaned_faces = self.reduce_faces(
                cleaned_verts, cleaned_faces,
                target=poly_limit,
                **simplification_opts
            )

        return {
            'vertices': torch.tensor(cleaned_verts.astype("float32")).to(self.runtime),
            'faces': torch.tensor(cleaned_faces.astype("int32")).to(self.runtime)
        }

    def _run_contouring(self, grid: torch.Tensor, level: float) -> Tuple[tf.Tensor, tf.Tensor]:
        flat_array = grid.detach().cpu().numpy().flatten(order='F')
        spacing = 2.0 / (grid.shape[0] - 1)

        vol = vtkImageData()
        vol.SetDimensions(*grid.shape)
        vol.SetSpacing(spacing, spacing, spacing)
        vol.SetOrigin(-1, -1, -1)

        vtkdata = numpy_to_vtk(flat_array, deep=True, array_type=VTK_FLOAT)
        vtkdata.SetName("field")
        vol.GetPointData().SetScalars(vtkdata)

        tetra = vtkDataSetTriangleFilter()
        tetra.SetInputData(vol)
        tetra.Update()

        isoline = vtkContour3DLinearGrid()
        isoline.SetInputData(tetra.GetOutput())
        isoline.SetValue(0, level)
        isoline.SetMergePoints(True)
        isoline.Update()

        iso_result = isoline.GetOutput()
        coords = tf.convert_to_tensor(
            [iso_result.GetPoint(i) for i in range(iso_result.GetNumberOfPoints())],
            dtype=tf.float32
        )
        polys = tf.convert_to_tensor(
            [iso_result.GetCell(i).GetPointIds() for i in range(iso_result.GetNumberOfCells())],
            dtype=tf.int32
        )

        return coords.numpy(), polys.numpy()

    @staticmethod
    def reduce_faces(
        vbuf: tf.Tensor,
        fbuf: tf.Tensor,
        target: int,
        method: str = "pymeshlab",
        postremesh: bool = False,
        place_optimal: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        v_start, f_start = vbuf.shape, fbuf.shape

        if method == "pyfqmr":
            import pyfqmr
            sim = pyfqmr.Simplify()
            sim.setMesh(vbuf, fbuf)
            sim.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
            vbuf, fbuf, _ = sim.getMesh()
        else:
            mdl = pml.Mesh(vbuf, fbuf)
            ms = pml.MeshSet()
            ms.add_mesh(mdl, "input")
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=int(target),
                optimalplacement=place_optimal
            )
            if postremesh:
                ms.meshing_isotropic_explicit_remeshing(
                    iterations=3,
                    targetlen=pml.PercentageValue(1)
                )
            result = ms.current_mesh()
            vbuf = result.vertex_matrix()
            fbuf = result.face_matrix()

        print(
            f"[MeshReducer] verts: {v_start} → {vbuf.shape}, faces: {f_start} → {fbuf.shape}"
        )
        return vbuf, fbuf

    @staticmethod
    def perform_cleanup(
        vcloud: tf.Tensor,
        fcloud: tf.Tensor,
        vmerge_ratio: float = 1,
        minfaces: int = 64,
        mindiam: int = 20,
        fix_mesh: bool = True,
        enable_remesh: bool = True,
        resample_len: float = 0.01
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        original_v = vcloud.shape
        original_f = fcloud.shape

        obj = pml.Mesh(vcloud, fcloud)
        mset = pml.MeshSet()
        mset.add_mesh(obj, "raw")

        mset.meshing_remove_unreferenced_vertices()
        if vmerge_ratio > 0:
            mset.meshing_merge_close_vertices(threshold=pml.PercentageValue(vmerge_ratio))
        mset.meshing_remove_duplicate_faces()
        mset.meshing_remove_null_faces()

        if mindiam > 0:
            mset.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pml.PercentageValue(mindiam)
            )
        if minfaces > 0:
            mset.meshing_remove_connected_component_by_face_number(mincomponentsize=minfaces)

        if fix_mesh:
            mset.meshing_repair_non_manifold_edges(method=0)
            mset.meshing_repair_non_manifold_vertices(vertdispratio=0)

        if enable_remesh:
            mset.meshing_isotropic_explicit_remeshing(
                iterations=3,
                targetlen=pml.PureValue(resample_len)
            )

        updated = mset.current_mesh()
        vcloud = updated.vertex_matrix()
        fcloud = updated.face_matrix()

        print(
            f"[MeshCleanup] verts: {original_v} → {vcloud.shape}, faces: {original_f} → {fcloud.shape}"
        )

        return vcloud, fcloud

    def extract_fields(
        self,
        gaussians: object,
        resolution: int = 128,
        num_blocks: int = 16,
        relax_ratio: float = 1.5
    ) -> torch.Tensor:
        opacities = torch.sigmoid(gaussians.opacity)
        mask = (opacities > 0.005).squeeze(1)

        means = gaussians.mean[mask]
        stds = torch.exp(gaussians.scaling)[mask]

        mn, mx = means.amin(0), means.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        means = (means - self.center) * self.scale
        stds = stds * self.scale
        covs = gaussians.covariance_activation(stds, 1, gaussians.rotation[mask])

        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        block_size = 2 / num_blocks
        split_size = resolution // num_blocks

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)

                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (means < vmax).all(-1) & (means > vmin).all(-1)

                    if not mask.any():
                        continue

                    batch_g = 1024
                    val = torch.zeros(pts.shape[0], device=device)

                    for start in range(0, mask.sum(), batch_g):
                        end = min(start + batch_g, mask.sum())
                        g_pts = pts.unsqueeze(1) - means[mask][start:end].unsqueeze(0)
                        w = self._gaussian_density(
                            g_pts.reshape(-1, 3),
                            covs[mask][start:end].repeat(pts.shape[0], 1)
                        ).reshape(pts.shape[0], -1)
                        val += (opacities[mask][start:end] * w).sum(-1)

                    occ[
                        xi*split_size:(xi+1)*split_size,
                        yi*split_size:(yi+1)*split_size,
                        zi*split_size:(zi+1)*split_size
                    ] = val.reshape(len(xs), len(ys), len(zs))

        return occ

    def _gaussian_density(self, pts: torch.Tensor, covars: torch.Tensor) -> torch.Tensor:
        matrix = torch.zeros((covars.shape[0], 3, 3), device=covars.device)
        matrix[:, 0, 0] = covars[:, 0]
        matrix[:, 0, 1] = covars[:, 1]
        matrix[:, 0, 2] = covars[:, 2]
        matrix[:, 1, 1] = covars[:, 3]
        matrix[:, 1, 2] = covars[:, 4]
        matrix[:, 2, 2] = covars[:, 5]
        matrix[:, 1, 0] = covars[:, 1]
        matrix[:, 2, 0] = covars[:, 2]
        matrix[:, 2, 1] = covars[:, 4]

        diffs = pts.unsqueeze(1)
        inverse = torch.linalg.inv(matrix)
        expo = -0.5 * (diffs @ inverse @ diffs.transpose(-1, -2)).squeeze()
        det = torch.linalg.det(2 * torch.pi * matrix)
        output = torch.exp(expo) / torch.sqrt(det)

        return output
