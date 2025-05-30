import os
import cv2
import torch
import trimesh
import numpy as np


class TriangleMesh:
    def __init__(
        self,
        vertices=None,
        faces=None,
        vertex_normals=None,
        face_normals=None,
        vertex_textures=None,
        face_textures=None,
        texture_map=None,
        vertex_colors=None,
        device=None,
    ):
        self.device = device
        self.vertices = vertices
        self.vertex_normals = vertex_normals
        self.vertex_textures = vertex_textures
        self.faces = faces
        self.face_normals = face_normals
        self.face_textures = face_textures
        self.texture_map = texture_map
        self.vertex_colors = vertex_colors

        self.original_center = 0
        self.original_scale = 1

    @classmethod
    def from_file(cls, path=None, resize=True, recalculate_normals=True, regenerate_uv=False, front_direction='+z', **kwargs):
        if path is None:
            mesh = cls(**kwargs)
        elif path.endswith(".obj"):
            mesh = cls.load_obj_file(path, **kwargs)
        else:
            mesh = cls.load_with_trimesh(path, **kwargs)

        print(f"[Mesh loading] vertices: {mesh.vertices.shape}, faces: {mesh.faces.shape}")
        
        if resize:
            mesh.normalize_size()
        if recalculate_normals or mesh.vertex_normals is None:
            mesh.calculate_normals()
            print(f"[Mesh loading] vertex normals: {mesh.vertex_normals.shape}, face normals: {mesh.face_normals.shape}")
        if regenerate_uv or (mesh.texture_map is not None and mesh.vertex_textures is None):
            mesh.generate_uv_coordinates(cache_path=path)
            print(f"[Mesh loading] vertex textures: {mesh.vertex_textures.shape}, face textures: {mesh.face_textures.shape}")

        if front_direction != "+z":
            if "-z" in front_direction:
                transform = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=mesh.device, dtype=torch.float32)
            elif "+x" in front_direction:
                transform = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "-x" in front_direction:
                transform = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "+y" in front_direction:
                transform = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            elif "-y" in front_direction:
                transform = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            else:
                transform = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            
            if '1' in front_direction:
                transform @= torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            elif '2' in front_direction:
                transform @= torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            elif '3' in front_direction:
                transform @= torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            
            mesh.vertices @= transform
            mesh.vertex_normals @= transform

        return mesh

    @classmethod
    def load_obj_file(cls, path, texture_path=None, device=None):
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()
        mesh.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(path, "r") as file:
            lines = file.readlines()

        def parse_face_vertex(face_vertex):
            components = [int(x) - 1 if x else -1 for x in face_vertex.split("/")]
            components.extend([-1] * (3 - len(components)))
            return components[0], components[1], components[2]

        vertices, texture_coords, normals = [], [], []
        faces, texture_faces, normal_faces = [], [], []
        material_path = None

        for line in lines:
            parts = line.split()
            if not parts:
                continue
            
            prefix = parts[0].lower()
            if prefix == "mtllib":
                material_path = parts[1]
            elif prefix == "usemtl":
                continue
            elif prefix == "v":
                vertices.append([float(v) for v in parts[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in parts[1:]])
            elif prefix == "vt":
                coords = [float(v) for v in parts[1:]]
                texture_coords.append([coords[0], 1.0 - coords[1]])
            elif prefix == "f":
                face_vertices = parts[1:]
                v0, t0, n0 = parse_face_vertex(face_vertices[0])
                for i in range(len(face_vertices) - 2):
                    v1, t1, n1 = parse_face_vertex(face_vertices[i + 1])
                    v2, t2, n2 = parse_face_vertex(face_vertices[i + 2])
                    faces.append([v0, v1, v2])
                    texture_faces.append([t0, t1, t2])
                    normal_faces.append([n0, n1, n2])

        mesh.vertices = torch.tensor(vertices, dtype=torch.float32, device=mesh.device)
        mesh.vertex_textures = torch.tensor(texture_coords, dtype=torch.float32, device=mesh.device) if texture_coords else None
        mesh.vertex_normals = torch.tensor(normals, dtype=torch.float32, device=mesh.device) if normals else None

        mesh.faces = torch.tensor(faces, dtype=torch.int32, device=mesh.device)
        mesh.face_textures = torch.tensor(texture_faces, dtype=torch.int32, device=mesh.device) if texture_coords else None
        mesh.face_normals = torch.tensor(normal_faces, dtype=torch.int32, device=mesh.device) if normals else None

        has_vertex_color = False
        if mesh.vertices.shape[1] == 6:
            has_vertex_color = True
            mesh.vertex_colors = mesh.vertices[:, 3:]
            mesh.vertices = mesh.vertices[:, :3]
            print(f"[load_obj_file] using vertex colors: {mesh.vertex_colors.shape}")

        if not has_vertex_color:
            material_path_candidates = []
            if material_path:
                material_path_candidates.append(material_path)
                material_path_candidates.append(os.path.join(os.path.dirname(path), material_path))
            material_path_candidates.append(path.replace(".obj", ".mtl"))

            found_material_path = None
            for candidate in material_path_candidates:
                if os.path.exists(candidate):
                    found_material_path = candidate
                    break
            
            if found_material_path and not texture_path:
                with open(found_material_path, "r") as file:
                    for line in file:
                        if "map_Kd" in line.split():
                            texture_path = os.path.join(os.path.dirname(path), line.split()[1])
                            print(f"[load_obj_file] using texture from: {texture_path}")
                            break
            
            if not texture_path or not os.path.exists(texture_path):
                default_texture = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])
            else:
                texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
                texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                texture = texture.astype(np.float32) / 255
                print(f"[load_obj_file] loaded texture: {texture.shape}")

            mesh.texture_map = torch.tensor(default_texture if not texture_path or not os.path.exists(texture_path) else texture, 
                                          dtype=torch.float32, device=mesh.device)

        return mesh

    @classmethod
    def load_with_trimesh(cls, path, device=None):
        mesh = cls()
        mesh.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh_data = trimesh.load(path)
        if isinstance(mesh_data, trimesh.Scene):
            if len(mesh_data.geometry) == 1:
                mesh_part = list(mesh_data.geometry.values())[0]
            else:
                combined = []
                for geometry in mesh_data.geometry.values():
                    if isinstance(geometry, trimesh.Trimesh):
                        combined.append(geometry)
                mesh_part = trimesh.util.concatenate(combined)
        else:
            mesh_part = mesh_data
        
        if mesh_part.visual.kind == 'vertex':
            colors = mesh_part.visual.vertex_colors
            colors = np.array(colors[..., :3]).astype(np.float32) / 255
            mesh.vertex_colors = torch.tensor(colors, dtype=torch.float32, device=mesh.device)
            print(f"[load_with_trimesh] using vertex colors: {mesh.vertex_colors.shape}")
        elif mesh_part.visual.kind == 'texture':
            material = mesh_part.visual.material
            if isinstance(material, trimesh.visual.material.PBRMaterial):
                texture = np.array(material.baseColorTexture).astype(np.float32) / 255
            elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                texture = np.array(material.to_pbr().baseColorTexture).astype(np.float32) / 255
            else:
                raise NotImplementedError(f"Material type {type(material)} not supported!")
            mesh.texture_map = torch.tensor(texture, dtype=torch.float32, device=mesh.device)
            print(f"[load_with_trimesh] loaded texture: {texture.shape}")
        else:
            default_texture = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])
            mesh.texture_map = torch.tensor(default_texture, dtype=torch.float32, device=mesh.device)
            print(f"[load_with_trimesh] failed to load texture")

        vertices = mesh_part.vertices

        try:
            uv_coords = mesh_part.visual.uv
            uv_coords[:, 1] = 1 - uv_coords[:, 1]
        except Exception:
            uv_coords = None

        try:
            norms = mesh_part.vertex_normals
        except Exception:
            norms = None

        faces = uv_faces = normal_faces = mesh_part.faces

        mesh.vertices = torch.tensor(vertices, dtype=torch.float32, device=mesh.device)
        mesh.vertex_textures = torch.tensor(uv_coords, dtype=torch.float32, device=mesh.device) if uv_coords else None
        mesh.vertex_normals = torch.tensor(norms, dtype=torch.float32, device=mesh.device) if norms else None

        mesh.faces = torch.tensor(faces, dtype=torch.int32, device=mesh.device)
        mesh.face_textures = torch.tensor(uv_faces, dtype=torch.int32, device=mesh.device) if uv_coords else None
        mesh.face_normals = torch.tensor(normal_faces, dtype=torch.int32, device=mesh.device) if norms else None

        return mesh

    def get_bounding_box(self):
        return torch.min(self.vertices, dim=0).values, torch.max(self.vertices, dim=0).values

    @torch.no_grad()
    def normalize_size(self):
        min_point, max_point = self.get_bounding_box()
        self.original_center = (max_point + min_point) / 2
        self.original_scale = 1.2 / torch.max(max_point - min_point).item()
        self.vertices = (self.vertices - self.original_center) * self.original_scale

    def calculate_normals(self):
        idx0, idx1, idx2 = self.faces[:, 0].long(), self.faces[:, 1].long(), self.faces[:, 2].long()
        v0, v1, v2 = self.vertices[idx0, :], self.vertices[idx1, :], self.vertices[idx2, :]

        face_norms = torch.cross(v1 - v0, v2 - v0)

        vertex_norms = torch.zeros_like(self.vertices)
        vertex_norms.scatter_add_(0, idx0[:, None].repeat(1, 3), face_norms)
        vertex_norms.scatter_add_(0, idx1[:, None].repeat(1, 3), face_norms)
        vertex_norms.scatter_add_(0, idx2[:, None].repeat(1, 3), face_norms)

        vertex_norms = torch.where(
            torch.sum(vertex_norms * vertex_norms, -1, keepdim=True) > 1e-20,
            vertex_norms,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vertex_norms.device),
        )
        vertex_norms = vertex_norms / torch.sqrt(torch.clamp(torch.sum(vertex_norms * vertex_norms, -1, keepdim=True), min=1e-20))

        self.vertex_normals = vertex_norms
        self.face_normals = self.faces

    def generate_uv_coordinates(self, cache_path=None, remap_vertices=True):
        if cache_path:
            cache_path = os.path.splitext(cache_path)[0] + "_uv.npz"
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            uv_verts, uv_faces, v_map = data["vt"], data["ft"], data["vmapping"]
        else:
            import xatlas

            verts_np = self.vertices.detach().cpu().numpy()
            faces_np = self.faces.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(verts_np, faces_np)
            options = xatlas.ChartOptions()
            atlas.generate(chart_options=options)
            v_map, uv_faces, uv_verts = atlas[0]

            if cache_path:
                np.savez(cache_path, vt=uv_verts, ft=uv_faces, vmapping=v_map)
        
        uv_vertices = torch.from_numpy(uv_verts.astype(np.float32)).to(self.device)
        uv_face_indices = torch.from_numpy(uv_faces.astype(np.int32)).to(self.device)
        self.vertex_textures = uv_vertices
        self.face_textures = uv_face_indices

        if remap_vertices:
            vertex_mapping = torch.from_numpy(v_map.astype(np.int64)).long().to(self.device)
            self.remap_vertices_to_uv(vertex_mapping)
    
    def remap_vertices_to_uv(self, vertex_mapping=None):
        if vertex_mapping is None:
            uv_faces = self.face_textures.view(-1).long()
            faces = self.faces.view(-1).long()
            vertex_mapping = torch.zeros(self.vertex_textures.shape[0], dtype=torch.long, device=self.device)
            vertex_mapping[uv_faces] = faces

        self.vertices = self.vertices[vertex_mapping]
        self.faces = self.face_textures
        if self.vertex_normals:
            self.vertex_normals = self.vertex_normals[vertex_mapping]
            self.face_normals = self.face_textures

    def to_device(self, device):
        self.device = device
        for attr in ["vertices", "faces", "vertex_normals", "face_normals", "vertex_textures", "face_textures", "texture_map"]:
            tensor = getattr(self, attr)
            if tensor:
                setattr(self, attr, tensor.to(device))
        return self
    
    def save(self, path):
        if path.endswith(".ply"):
            self.save_as_ply(path)
        elif path.endswith(".obj"):
            self.save_obj(path)
        elif path.endswith(".glb") or path.endswith(".gltf"):
            self.save_gltf(path)
        else:
            raise NotImplementedError(f"Format {path} not supported!")
    
    def save_as_ply(self, path):
        vertices_np = self.vertices.detach().cpu().numpy()
        faces_np = self.faces.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        mesh.export(path)

    def save_gltf(self, path):
        assert self.vertex_normals and self.vertex_textures

        if self.vertices.shape[0] != self.vertex_textures.shape[0]:
            self.remap_vertices_to_uv()

        import pygltflib

        faces_np = self.faces.detach().cpu().numpy().astype(np.uint32)
        vertices_np = self.vertices.detach().cpu().numpy().astype(np.float32)
        uv_np = self.vertex_textures.detach().cpu().numpy().astype(np.float32)

        texture = self.texture_map.detach().cpu().numpy()
        texture = (texture * 255).astype(np.uint8)
        texture = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)

        faces_data = faces_np.flatten().tobytes()
        vertices_data = vertices_np.tobytes()
        uv_data = uv_np.tobytes()
        texture_data = cv2.imencode('.png', texture)[1].tobytes()

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[pygltflib.Mesh(primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=1, TEXCOORD_0=2, 
                    ),
                    indices=0, material=0,
                )
            ])],
            materials=[
                pygltflib.Material(
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                    alphaCutoff=0,
                    doubleSided=True,
                )
            ],
            textures=[
                pygltflib.Texture(sampler=0, source=0),
            ],
            samplers=[
                pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR, wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT),
            ],
            images=[
                pygltflib.Image(bufferView=3, mimeType="image/png"),
            ],
            buffers=[
                pygltflib.Buffer(byteLength=len(faces_data) + len(vertices_data) + len(uv_data) + len(texture_data))
            ],
            bufferViews=[
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(faces_data),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(faces_data),
                    byteLength=len(vertices_data),
                    byteStride=12,
                    target=pygltflib.ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(faces_data) + len(vertices_data),
                    byteLength=len(uv_data),
                    byteStride=8,
                    target=pygltflib.ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(faces_data) + len(vertices_data) + len(uv_data),
                    byteLength=len(texture_data),
                ),
            ],
            accessors=[
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT,
                    count=faces_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(faces_np.max())],
                    min=[int(faces_np.min())],
                ),
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    count=len(vertices_np),
                    type=pygltflib.VEC3,
                    max=vertices_np.max(axis=0).tolist(),
                    min=vertices_np.min(axis=0).tolist(),
                ),
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(uv_np),
                    type=pygltflib.VEC2,
                    max=uv_np.max(axis=0).tolist(),
                    min=uv_np.min(axis=0).tolist(),
                ),
            ],
        )

        gltf.set_binary_blob(faces_data + vertices_data + uv_data + texture_data)
        gltf.save(path)

    def save_obj(self, path):
        material_path = path.replace(".obj", ".mtl")
        texture_path = path.replace(".obj", "_albedo.png")

        vertices_np = self.vertices.detach().cpu().numpy()
        uv_np = self.vertex_textures.detach().cpu().numpy() if self.vertex_textures else None
        normals_np = self.vertex_normals.detach().cpu().numpy() if self.vertex_normals else None
        faces_np = self.faces.detach().cpu().numpy()
        uv_faces_np = self.face_textures.detach().cpu().numpy() if self.face_textures else None
        normal_faces_np = self.face_normals.detach().cpu().numpy() if self.face_normals else None

        with open(path, "w") as file:
            file.write(f"mtllib {os.path.basename(material_path)} \n")

            for v in vertices_np:
                file.write(f"v {v[0]} {v[1]} {v[2]} \n")

            if uv_np:
                for uv in uv_np:
                    file.write(f"vt {uv[0]} {1 - uv[1]} \n")

            if normals_np:
                for n in normals_np:
                    file.write(f"vn {n[0]} {n[1]} {n[2]} \n")

            file.write(f"usemtl defaultMat \n")
            for i in range(len(faces_np)):
                file.write(
                    f'f {faces_np[i, 0] + 1}/{uv_faces_np[i, 0] + 1 if uv_faces_np is not None else ""}/{normal_faces_np[i, 0] + 1 if normal_faces_np is not None else ""} \
                             {faces_np[i, 1] + 1}/{uv_faces_np[i, 1] + 1 if uv_faces_np is not None else ""}/{normal_faces_np[i, 1] + 1 if normal_faces_np is not None else ""} \
                             {faces_np[i, 2] + 1}/{uv_faces_np[i, 2] + 1 if uv_faces_np is not None else ""}/{normal_faces_np[i, 2] + 1 if normal_faces_np is not None else ""} \n'
                )

        with open(material_path, "w") as file:
            file.write(f"newmtl defaultMat \n")
            file.write(f"Ka 1 1 1 \n")
            file.write(f"Kd 1 1 1 \n")
            file.write(f"Ks 0 0 0 \n")
            file.write(f"Tr 1 \n")
            file.write(f"illum 1 \n")
            file.write(f"Ns 0 \n")
            file.write(f"map_Kd {os.path.basename(texture_path)} \n")

        texture = self.texture_map.detach().cpu().numpy()
        texture = (texture * 255).astype(np.uint8)
        cv2.imwrite(texture_path, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))