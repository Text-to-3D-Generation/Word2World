import numpy as np
import pymeshlab as pml
from typing import Tuple


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
