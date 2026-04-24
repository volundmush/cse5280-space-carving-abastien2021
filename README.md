# Space Carving with PyTorch3D + Open3D

This project implements the full visual hull pipeline requested in `GOALS.md`:

1. Multi-view rendering with PyTorch3D
2. Silhouette extraction and camera export
3. Pinhole projection (`x ~ K[R|t]X`) for voxel projection
4. Space carving to estimate occupancy (visual hull)
5. Mesh reconstruction with Open3D

All implementation code is in `carving.py`.

## 1) Method

### 1.1 Multi-view rendering

We render a mesh from multiple viewpoints using PyTorch3D.

- Cameras are sampled around the object with azimuth sweep `[0, 360)`.
- Elevation changes linearly between `elev_min` and `elev_max`.
- For each view we save:
  - RGB image
  - Binary silhouette mask
  - Optional depth map
  - Camera parameters converted to OpenCV form (`R`, `t`, `K`)

### 1.2 Pinhole projection model (Math-tastic Mess)

For each voxel center `X` in world coordinates:

1. Transform to camera coordinates:

`X_c = R X + t`

2. Project with intrinsics:

`x_h = K X_c`

3. Normalize homogeneous coordinates:

`u = x_h[0] / x_h[2], v = x_h[1] / x_h[2]`

4. Keep only valid projections:

- `z = X_c[2] > 0`
- pixel inside image bounds

This is implemented in `project_points()` in `carving.py`.

### 1.3 Space carving

We start with a dense voxel grid in a cubic bounding volume.

For each camera view:

- project currently active voxels
- test whether each projected pixel lies inside the silhouette
- remove voxels that are invalid or outside the silhouette

After all views, remaining voxels approximate the visual hull.

This is approximately like when a sculptor takes pictures around a posing model and then starts chiseling a hunk of stone or molding clay to match what they see from that angle.

### 1.4 Reconstruction

Occupied voxel centers are exported as a point cloud and reconstructed into a mesh using Open3D:

- `poisson` (default)
- `ball_pivoting` (optional)

## 2) Code structure

- `carving.py`
  - `render_views()` - renders RGB/silhouette/depth
  - `pytorch3d_to_opencv()` - converts camera model
  - `project_points()` - pinhole projection math
  - `carve_space()` - voxel carving loop
  - `export_visual_hull()` - point cloud output
  - `reconstruct_mesh()` - Open3D reconstruction
  - `run_pipeline()` - end-to-end execution
- `run_parallel.sh`
  - runs multiple experiment configs in parallel
  - writes per-run `metrics.json`
  - writes consolidated `summary.json`

## 3) Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `pytorch3d` installation is environment-specific (PyTorch/CUDA compatibility matters).
- If needed, install a matching PyTorch3D wheel manually for your system.

I had a seriously hard time getting this working on my Bazzite desktop which is running an old 980ti and doesn't support the new CUDA stuff... had to render using CPU mode, ugh.

## 4) Running

### 4.1 Single run

Without mesh path, the script uses a normalized ico-sphere test mesh:

```bash
python carving.py \
  --out-dir outputs/default_run \
  --num-views 30 \
  --voxel-resolution 96 \
  --image-size 256
```

With an OBJ mesh:

```bash
python carving.py \
  --mesh-path path/to/model.obj \
  --out-dir outputs/model_run \
  --num-views 30 \
  --voxel-resolution 96
```

### 4.2 Parallel experiments

Run preset experiments (`views, voxel_resolution`) in parallel:

```bash
bash run_parallel.sh
```

Override defaults:

```bash
PARALLEL=4 DEVICE=cpu OUT_ROOT=outputs/experiments MESH_PATH=path/to/model.obj bash run_parallel.sh
```

Since I didn't have a compatible GPU (my 3090ti died a few weeks ago and now I had to dig out an old 980ti), CPU rendering mode saved the day at a full PARALLEL=32, because I have a beefy AMD Ryzen 5950X. 

## 5) Outputs

Each run produces:

- `render/rgb/*.png`
- `render/silhouette/*.png`
- `render/depth/*.npy` (if `--save-depth`)
- `render/cameras.npz` (`R`, `t`, `K`, image size)
- `carving/occupied_points.npy`
- `carving/visual_hull_points.ply`
- `carving/projection_overlay_view0.png` (debug overlay)
- `reconstruction/visual_hull_mesh.ply` (unless skipped)
- `metrics.json`

## 6) Experiments and analysis

This section addresses the requested analysis dimensions.

### 6.1 Effect of number of views

Compare 5, 10, 30, 50 views while keeping voxel resolution fixed.

Expected trend:

- More views produce a tighter visual hull.
- Runtime increases approximately linearly with views.
- Very low view counts overestimate volume (loose hull).

### 6.2 Effect of voxel resolution

Compare 64 vs 96 (and optionally 128) while keeping views fixed.

Expected trend:

- Higher resolution captures finer geometry.
- Memory and runtime increase quickly with grid size (`O(N^3)` voxels).

### 6.3 Reconstruction limitations

- **Missing concavities:** visual hull cannot recover concave regions not visible in silhouettes.
- **Silhouette sensitivity:** thresholding errors directly affect occupancy.
- **Camera mismatch risk:** wrong conventions (`R`, `t`, handedness, pixel indexing) can destroy carving quality.

## 7) Key pitfalls and mitigations

- Camera convention mismatch: use PyTorch3D-to-OpenCV conversion and consistent projection math.
- Intrinsic misuse (normalized vs pixels): projection in `carving.py` uses pixel-space `K`.
- Row/column confusion: silhouette indexing uses `mask[v, u]`.
- Invalid depth: enforce visibility check `z > 0`.
- Performance bottlenecks: carving is chunked with `--chunk-size`.

## 8) Result table template

After running experiments, fill this table with values from each `metrics.json`.

| Views | Vox Res | Occupied Voxels | Occupancy Ratio | Runtime (s) | Mesh Quality Notes |
|---:|---:|---:|---:|---:|---|
| 5 | 64 |  |  |  |  |
| 10 | 64 |  |  |  |  |
| 30 | 96 |  |  |  |  |
| 50 | 96 |  |  |  |  |

## 9) Reproducibility checklist

- Use same mesh and normalization across runs.
- Keep image size and camera distance fixed unless explicitly testing them.
- Change one variable at a time (views or voxel resolution).
- Record runtime, occupancy ratio, and output paths from `metrics.json`.
