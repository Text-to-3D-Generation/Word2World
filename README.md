# Word2World: Text-to-3D Generation with 3D Gaussian Splatting

Word2World is a research system for generating 3D scenes from text prompts. It combines **3D Gaussian Splatting** with **Score Distillation Sampling (SDS)** guidance from a multi-view diffusion model (MVDream) to optimize a set of 3D Gaussians that represent a coherent object or scene described by a text prompt.

---

## How It Works

1. **Initialization** — A random point cloud is generated on a sphere and converted into a set of Gaussian primitives (position, opacity, scale, rotation, spherical-harmonic color coefficients).
2. **Rendering** — A custom CUDA forward/backward rasterizer projects the Gaussians into 2D images from multiple camera viewpoints simultaneously.
3. **Guidance** — MVDream scores the rendered views via SDS loss, backpropagating gradients into the Gaussian parameters.
4. **Densification** — Periodically, under-reconstructed regions are densified by cloning or splitting Gaussians, and low-opacity Gaussians are pruned.
5. **Export** — The optimized Gaussians can be exported as a raw `.ply` point cloud, a triangle mesh (`.ply`), or a textured mesh (`.obj`).

---

## Repository Structure

```
Word2World/
├── main.py                         # Entry point (GUI mode)
├── GaussianTrainer.py              # Core training loop
├── GUITrainer.py                   # Tkinter-based live training GUI
│
├── Renderer.py                     # Differentiable Gaussian renderer (PyTorch autograd)
├── gaussians_handler.py            # Gaussian parameter management & densification
├── primitives.py                   # GaussianPrimitive & PointCloud data classes
├── Densification.py                # Clone / split / prune logic
├── OptimizerManager.py             # Adam optimizer state management during densification
│
├── GaussianIO.py                   # PLY save/load for Gaussian models
├── TrainerIO.py                    # Full model export (raw, mesh, textured mesh)
├── TriangleMesh.py                 # Triangle mesh class with OBJ/PLY/GLTF export
├── utils_for_mesh.py               # Occupancy field extraction & tetrahedral meshing
├── utils_for_camera.py             # Camera math utilities
├── DynamicCamera.py                # Orbital camera
├── misc_utils.py                   # Projection, covariance, SH, LR schedule helpers
├── insert_in_grid.py               # Differentiable 2D/3D grid insertion (for UV baking)
│
├── compute_voxel_field.cu          # CUDA kernel for fast voxel field extraction
├── compute_voxel_field_wrapper.py  # Python ctypes wrapper for the above kernel
│
├── PointE.py                       # Optional Point-E initialization backend
├── optimizer_config.json           # Per-parameter learning rates
│
├── guidance/
│   └── mvdream_interface.py        # MVDream multi-view diffusion guidance
│
├── cudaForwardBackwardRenderer/    # Custom CUDA rasterizer (C++/CUDA extension)
│   ├── gaussianRasterizer.cu       # Top-level forward pass orchestration
│   ├── gaussianForwardPass.cu      # SH color evaluation
│   ├── projection.cu               # 3D → 2D Gaussian projection
│   ├── ADRculling.cu               # Adaptive-radius tile culling
│   ├── pixelGaussiansFull.cu       # Tile-based alpha compositing
│   ├── prepareSort.cu              # Custom 2-bit radix sort for tile keys
│   ├── backward.cu                 # Backward pass (gradients)
│   ├── gradientPass.cu             # Backward pass entry point
│   └── setup.py                    # Build script for the CUDA extension
│
├── prefix-sum-optimizations/       # Standalone CUDA prefix-sum experiments
├── sorting-optimizations/          # Standalone CUDA radix-sort experiments
└── tile-based-densification/       # Standalone tile-based densification experiments
```

---

## Requirements

### System
- Linux or Windows (CUDA toolkit required)
- NVIDIA GPU with CUDA compute capability ≥ 7.0
- CUDA 11.x or 12.x
- Python 3.10+

### Python packages

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include: `torch`, `numpy`, `scipy`, `plyfile`, `ttkbootstrap`, `diffusers`, `transformers`, `trimesh`, `PyMCubes`, `nvdiffrast`, `xatlas`, `pymeshlab`, `pyvista`, `kiui`.

### Build the CUDA rasterizer

```bash
cd cudaForwardBackwardRenderer
pip install -e .
```

### Build the voxel field CUDA kernel (optional, for faster mesh extraction)

**Linux:**
```bash
nvcc -O3 -shared -fPIC -o compute_voxel_field.so compute_voxel_field.cu
```

**Windows:**
```bash
nvcc -O3 -shared -o compute_voxel_field.dll compute_voxel_field.cu
```

---

## Usage

### GUI mode

```bash
python main.py --prompt "a red fire truck"
```

This opens an interactive window showing four live-rendered views of the object as it trains. Use the toolbar to start/stop training and export the result.

### Headless / scripted training

```python
from GaussianTrainer import GaussianTrainer

trainer = GaussianTrainer(prompt="a red fire truck")
trainer.pre_traininig()

for step in range(500):
    loss = trainer.optimizaiton_iteration()
    print(f"Step {step}: loss = {loss:.4f}")

# Export as textured mesh
trainer.save_model(model_type=2, texture_size=1024,
                   model_name="fire_truck", save_dir="outputs")
```

### Export options

| `model_type` | Output |
|---|---|
| `0` | Raw Gaussian `.ply` |
| `1` | Triangle mesh `.ply` (no texture) |
| `2` | Textured mesh `.obj` + albedo PNG |

---

## Architecture Details

### Custom CUDA Rasterizer

The renderer (`cudaForwardBackwardRenderer`) is a fully differentiable tile-based Gaussian rasterizer built from scratch:

- **Projection** — Each Gaussian's 3D covariance is projected to a 2D screen-space covariance via the Jacobian of the perspective projection.
- **Tile culling** — Adaptive-radius (ADR) culling assigns each Gaussian to the screen tiles it overlaps, using a minimum of the 3-sigma radius and an opacity-derived adaptive radius.
- **Sorting** — A custom 2-bit radix sort orders Gaussian instances by `(tile_id, depth)` for correct front-to-back compositing.
- **Compositing** — Each 16×16 pixel tile processes its assigned Gaussians in depth order using alpha compositing with early termination.
- **Backward pass** — Gradients flow back through the compositing, projection, spherical harmonics, and covariance to the raw Gaussian parameters.

### Densification

Adaptive Density Control (ADC) runs periodically during training:

- **Clone** — Small Gaussians with high 2D gradient magnitude are duplicated in place.
- **Split** — Large Gaussians with high gradient magnitude are split into two smaller Gaussians sampled along the principal axes.
- **Prune** — Gaussians with opacity below a threshold, or with excessively large scale, are removed.

### Mesh Extraction

Mesh export uses a VTK-based marching-tetrahedra pipeline:
1. A 3D occupancy field is computed by evaluating weighted Gaussian contributions on a voxel grid (128³ by default; CUDA-accelerated if the kernel is compiled).
2. A contour is extracted at a density threshold.
3. The mesh is cleaned with PyMeshLab (duplicate removal, remeshing).
4. Optionally decimated to a target face count.
5. For textured export, the albedo is baked by rendering from 22 viewpoints and projecting RGB values onto a UV atlas via mipmap bilinear splatting.

---

## Configuration

Learning rates for each parameter group are set in `optimizer_config.json`:

```json
[
  { "param": "mean",               "lr": 0.01   },
  { "param": "sh_coefficients_dc", "lr": 0.01   },
  { "param": "sh_coefficients_ac", "lr": 0.0005 },
  { "param": "opacity",            "lr": 0.05   },
  { "param": "svec",               "lr": 0.005  },
  { "param": "quaternion",         "lr": 0.005  }
]
```

The mean position learning rate additionally follows an exponential decay schedule (0.01 → 0.0002 over 300 steps).

---

## Evaluation

CLIP similarity scores between rendered views and the text prompt can be computed using `CLIP_similarity_measure.ipynb`, following the RichDreamer evaluation protocol (16 views, remove min/max outliers, average).

---

## Acknowledgements

This project builds on ideas from:
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., 2023)
- [MVDream](https://mv-dream.github.io/) multi-view diffusion guidance
- [Point-E](https://github.com/openai/point-e) (optional initialization)
- [DreamGaussian](https://dreamgaussian.github.io/) and related SDS-based 3DGS works
