import numpy as np
import torch
import pymeshlab as pml
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid, vtkDataSetTriangleFilter
from vtkmodules.vtkCommonCore import VTK_FLOAT
from typing import Optional, Dict, Tuple

class GaussianMeshExtractor:
    def __init__(self, device: str = 'cuda'):
        """
        Module for extracting meshes from Gaussian splatting representations.
        
        Args:
            device: The device to use for computations ('cuda' or 'cpu')
        """
        self.device = device
        self.center = None
        self.scale = None
    
    def extract_mesh(
        self,
        gaussians: object,
        density_thresh: float = 1.0,
        resolution: int = 128,
        decimate_target: int = 100000,
        clean_mesh_params: Optional[Dict] = None,
        decimate_mesh_params: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main interface to extract a mesh from Gaussian splats.
        
        Args:
            gaussians: Gaussian splatting model with required attributes:
                      - mean: (N,3) tensor of positions
                      - opacity: (N,1) tensor of opacities
                      - scaling: (N,3) tensor of scales
                      - rotation: (N,4) tensor of quaternions
                      - covariance_activation: Function to compute covariance matrices
            density_thresh: Density threshold for contouring
            resolution: Resolution of the occupancy grid
            decimate_target: Target number of faces for decimation (0 to disable)
            clean_mesh_params: Parameters for mesh cleaning (see clean_mesh())
            decimate_mesh_params: Parameters for mesh decimation (see decimate_mesh())
            
        Returns:
            Dictionary containing:
            - 'vertices': (V,3) tensor of mesh vertices
            - 'faces': (F,3) tensor of mesh faces
        """
        # Default parameters
        if clean_mesh_params is None:
            clean_mesh_params = {
                'v_pct': 1,
                'min_f': 64,
                'min_d': 20,
                'repair': True,
                'remesh': True,
                'remesh_size': 0.01
            }
        
        if decimate_mesh_params is None:
            decimate_mesh_params = {
                'backend': "pymeshlab",
                'remesh': False,
                'optimalplacement': True
            }
        
        # Step 1: Compute occupancy field
        occ = self._extract_fields(gaussians, resolution)
        
        # Step 2: Extract mesh via contouring
        verts, faces = self._contour_occupancy(occ, density_thresh)
        
        # Transform back to original space
        verts = verts / self.scale + self.center.detach().cpu().numpy()
        
        # Step 3: Clean and process mesh
        verts, faces = self.clean_mesh(verts, faces, **clean_mesh_params)
        
        if decimate_target > 0 and faces.shape[0] > decimate_target:
            verts, faces = self.decimate_mesh(
                verts, faces, 
                target=decimate_target,
                **decimate_mesh_params
            )
            
        return {
            'vertices': torch.from_numpy(verts.astype(np.float32)).to(self.device),
            'faces': torch.from_numpy(faces.astype(np.int32)).to(self.device)
        }

    def _extract_fields(
        self,
        gaussians: object,
        resolution: int = 128,
        num_blocks: int = 16,
        relax_ratio: float = 1.5
    ) -> torch.Tensor:
        """Compute 3D occupancy field from Gaussian splats"""
        opacities = torch.sigmoid(gaussians.opacity)
        mask = (opacities > 0.005).squeeze(1)
        
        means = gaussians.mean[mask]
        stds = torch.exp(gaussians.scaling)[mask]
        
        # Normalize to ~ [-1, 1]
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
                    pts = torch.cat([
                        xx.reshape(-1, 1), 
                        yy.reshape(-1, 1), 
                        zz.reshape(-1, 1)
                    ], dim=-1).to(device)
                    
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (means < vmax).all(-1) & (means > vmin).all(-1)
                    
                    if not mask.any():
                        continue
                        
                    batch_g = 1024  # Process in batches for memory efficiency
                    val = torch.zeros(pts.shape[0], device=device)
                    
                    for start in range(0, mask.sum(), batch_g):
                        end = min(start + batch_g, mask.sum())
                        g_pts = pts.unsqueeze(1) - means[mask][start:end].unsqueeze(0)
                        w = self._compute_gaussian_weights(
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

    def _compute_gaussian_weights(
        self, 
        pts: torch.Tensor, 
        covs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian weights for points given covariance matrices.
        
        Args:
            pts: (N,3) tensor of points
            covs: (N,6) tensor of flattened covariance matrices
            
        Returns:
            (N,) tensor of weights
        """
        # Convert flattened covariances to 3x3 matrices
        cov_matrices = torch.zeros((covs.shape[0], 3, 3), device=covs.device)
        cov_matrices[:, 0, 0] = covs[:, 0]
        cov_matrices[:, 0, 1] = covs[:, 1]
        cov_matrices[:, 0, 2] = covs[:, 2]
        cov_matrices[:, 1, 1] = covs[:, 3]
        cov_matrices[:, 1, 2] = covs[:, 4]
        cov_matrices[:, 2, 2] = covs[:, 5]
        cov_matrices[:, 1, 0] = covs[:, 1]
        cov_matrices[:, 2, 0] = covs[:, 2]
        cov_matrices[:, 2, 1] = covs[:, 4]
        
        # Compute Mahalanobis distance
        diff = pts.unsqueeze(1)  # (N,1,3)
        inv_cov = torch.linalg.inv(cov_matrices)  # (N,3,3)
        
        # Compute Gaussian PDF
        exponent = -0.5 * (diff @ inv_cov @ diff.transpose(-1, -2)).squeeze()
        det = torch.linalg.det(2 * np.pi * cov_matrices)
        weights = torch.exp(exponent) / torch.sqrt(det)
        
        return weights

    def _contour_occupancy(
        self, 
        occ: torch.Tensor, 
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mesh from occupancy field using VTK contouring"""
        occ_np = occ.detach().cpu().numpy().flatten(order='F')
        
        img = vtkImageData()
        img.SetDimensions(*occ.shape)
        spacing = 2.0 / (occ.shape[0] - 1)
        img.SetSpacing(spacing, spacing, spacing)
        img.SetOrigin(-1, -1, -1)
        
        vtk_arr = numpy_to_vtk(occ_np, deep=True, array_type=VTK_FLOAT)
        vtk_arr.SetName("density")
        img.GetPointData().SetScalars(vtk_arr)
        
        # Tetrahedralize
        tri = vtkDataSetTriangleFilter()
        tri.SetInputData(img)
        tri.Update()
        
        # Contour
        contour = vtkContour3DLinearGrid()
        contour.SetInputData(tri.GetOutput())
        contour.SetValue(0, threshold)
        contour.SetMergePoints(True)
        contour.Update()
        
        # Get results
        mesh = contour.GetOutput()
        verts = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
        faces = np.array([mesh.GetCell(i).GetPointIds() for i in range(mesh.GetNumberOfCells())])
        
        return verts, faces

    @staticmethod
    def decimate_mesh(
        verts: np.ndarray,
        faces: np.ndarray,
        target: int,
        backend: str = "pymeshlab",
        remesh: bool = False,
        optimalplacement: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce the number of faces in a mesh.
        
        Args:
            verts: (V,3) array of vertices
            faces: (F,3) array of faces
            target: Target number of faces
            backend: Backend to use ('pymeshlab' or 'pyfqmr')
            remesh: Whether to remesh after decimation
            optimalplacement: For pymeshlab, whether to optimize vertex placement
            
        Returns:
            Tuple of (verts, faces) after decimation
        """
        _ori_vert_shape = verts.shape
        _ori_face_shape = faces.shape

        if backend == "pyfqmr":
            import pyfqmr
            solver = pyfqmr.Simplify()
            solver.setMesh(verts, faces)
            solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
            verts, faces, _ = solver.getMesh()
        else:
            m = pml.Mesh(verts, faces)
            ms = pml.MeshSet()
            ms.add_mesh(m, "mesh")

            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=int(target), 
                optimalplacement=optimalplacement
            )

            if remesh:
                ms.meshing_isotropic_explicit_remeshing(
                    iterations=3, 
                    targetlen=pml.PercentageValue(1)
                )

            m = ms.current_mesh()
            verts = m.vertex_matrix()
            faces = m.face_matrix()

        print(
            f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, "
            f"{_ori_face_shape} --> {faces.shape}"
        )

        return verts, faces

    @staticmethod
    def clean_mesh(
        verts: np.ndarray,
        faces: np.ndarray,
        v_pct: float = 1,
        min_f: int = 64,
        min_d: int = 20,
        repair: bool = True,
        remesh: bool = True,
        remesh_size: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean and repair a mesh.
        
        Args:
            verts: (V,3) array of vertices
            faces: (F,3) array of faces
            v_pct: Percentage threshold for merging close vertices
            min_f: Minimum face count for connected components
            min_d: Minimum diameter for connected components
            repair: Whether to repair non-manifold geometry
            remesh: Whether to remesh
            remesh_size: Target edge length for remeshing
            
        Returns:
            Tuple of (verts, faces) after cleaning
        """
        _ori_vert_shape = verts.shape
        _ori_face_shape = faces.shape

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")

        # Basic cleaning
        ms.meshing_remove_unreferenced_vertices()
        if v_pct > 0:
            ms.meshing_merge_close_vertices(threshold=pml.PercentageValue(v_pct))
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()

        # Remove small components
        if min_d > 0:
            ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pml.PercentageValue(min_d)
            )
        if min_f > 0:
            ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

        # Repair
        if repair:
            ms.meshing_repair_non_manifold_edges(method=0)
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

        # Remesh
        if remesh:
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, 
                targetlen=pml.PureValue(remesh_size)
            )

        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

        print(
            f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, "
            f"{_ori_face_shape} --> {faces.shape}"
        )

        return verts, faces