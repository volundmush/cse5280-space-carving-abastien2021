#!/usr/bin/env python3

"""Space carving pipeline with PyTorch3D + Open3D.

Implements:
1) Multi-view rendering and silhouette extraction
2) Pinhole projection verification using K[R|t]
3) Space carving on a voxel grid
4) Visual hull export as point cloud
5) Mesh reconstruction with Open3D (Poisson or Ball Pivoting)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if not requested.startswith("cuda"):
        return torch.device(requested)

    if not torch.cuda.is_available():
        print("[warning] CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")

    index = 0
    if ":" in requested:
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError:
            index = 0

    capability = torch.cuda.get_device_capability(index)
    cap_code = f"{capability[0]}{capability[1]}"

    supported = set()
    for arch in torch.cuda.get_arch_list():
        if arch.startswith("sm_"):
            supported.add(arch.replace("sm_", ""))
        elif arch.startswith("compute_"):
            supported.add(arch.replace("compute_", ""))

    if supported and cap_code not in supported:
        device_name = torch.cuda.get_device_name(index)
        print(
            "[warning] Requested CUDA device is unsupported by this Torch build: "
            f"{device_name} (sm_{cap_code}). Supported arches: {sorted(supported)}. "
            "Falling back to CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")

    return torch.device(requested)


def _require_pytorch3d() -> None:
    try:
        import pytorch3d  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch3D is required. Install a compatible wheel for your Torch/CUDA setup."
        ) from exc


def _require_open3d() -> None:
    try:
        import open3d  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Open3D is required for mesh reconstruction.") from exc


@dataclass
class CameraData:
    R: np.ndarray  # (V, 3, 3)
    t: np.ndarray  # (V, 3)
    K: np.ndarray  # (V, 3, 3)
    image_size: Tuple[int, int]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def normalize_mesh(mesh) -> object:
    verts = mesh.verts_packed()
    center = verts.mean(0)
    verts = verts - center
    scale = verts.norm(dim=1).max().clamp(min=1e-8)
    verts = verts / scale
    return mesh.update_padded(verts[None])


def make_default_asymmetric_mesh(mesh):
    verts = mesh.verts_packed()
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    scale = torch.tensor([1.25, 0.78, 0.92], dtype=verts.dtype, device=verts.device)
    verts = verts * scale[None, :]

    ripple = 1.0 + 0.16 * torch.sin(4.0 * x) * torch.cos(3.0 * y) + 0.10 * torch.sin(5.0 * z)
    verts = verts * ripple[:, None]
    return mesh.update_padded(verts[None])


def add_default_vertex_textures(mesh):
    _require_pytorch3d()
    from pytorch3d.renderer import TexturesVertex

    if mesh.textures is not None:
        return mesh

    verts = mesh.verts_padded()
    vmin = verts.amin(dim=1, keepdim=True)
    vmax = verts.amax(dim=1, keepdim=True)
    verts_rgb = (verts - vmin) / (vmax - vmin + 1e-8)
    verts_rgb = verts_rgb.clamp(0.0, 1.0)
    mesh.textures = TexturesVertex(verts_features=verts_rgb)
    return mesh


def load_or_create_mesh(mesh_path: Optional[Path], device: torch.device):
    _require_pytorch3d()
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.utils import ico_sphere

    if mesh_path is not None:
        mesh = load_objs_as_meshes([str(mesh_path)], device=device)
    else:
        mesh = ico_sphere(level=4, device=device)
        mesh = make_default_asymmetric_mesh(mesh)
    mesh = normalize_mesh(mesh)
    mesh = add_default_vertex_textures(mesh)
    return mesh


def generate_cameras(
    num_views: int,
    image_size: int,
    fov_deg: float,
    distance: float,
    elev_min: float,
    elev_max: float,
    device: torch.device,
):
    _require_pytorch3d()
    from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform

    azim = torch.linspace(0.0, 360.0, num_views + 1, device=device)[:-1]
    elev = torch.linspace(elev_min, elev_max, num_views, device=device)
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)

    fov_rad = math.radians(fov_deg)
    focal_px = 0.5 * float(image_size) / math.tan(0.5 * fov_rad)
    focal = torch.full((num_views, 2), focal_px, dtype=torch.float32, device=device)
    principal = torch.full((num_views, 2), float(image_size) * 0.5, dtype=torch.float32, device=device)
    image_size_tensor = torch.full((num_views, 2), float(image_size), dtype=torch.float32, device=device)

    cameras = PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal,
        principal_point=principal,
        image_size=image_size_tensor,
        in_ndc=False,
        device=device,
    )
    return cameras


def pytorch3d_to_opencv(cameras, image_size: int) -> CameraData:
    n = cameras.R.shape[0]
    focal = to_numpy(cameras.focal_length)
    principal = to_numpy(cameras.principal_point)

    if focal.ndim == 1:
        focal = np.repeat(focal[None, :], n, axis=0)
    if focal.shape[1] == 1:
        focal = np.repeat(focal, 2, axis=1)

    K_cv = np.zeros((n, 3, 3), dtype=np.float32)
    K_cv[:, 0, 0] = focal[:, 0]
    K_cv[:, 1, 1] = focal[:, 1]
    K_cv[:, 0, 2] = principal[:, 0]
    K_cv[:, 1, 2] = principal[:, 1]
    K_cv[:, 2, 2] = 1.0

    return CameraData(
        R=to_numpy(cameras.R),
        t=to_numpy(cameras.T),
        K=K_cv,
        image_size=(image_size, image_size),
    )


def render_views(
    mesh,
    cameras,
    image_size: int,
    out_dir: Path,
    silhouette_threshold: float,
    save_depth: bool,
) -> Dict[str, object]:
    _require_pytorch3d()
    from pytorch3d.renderer import (
        BlendParams,
        HardPhongShader,
        MeshRasterizer,
        MeshRenderer,
        PointLights,
        RasterizationSettings,
        SoftSilhouetteShader,
    )

    render_dir = out_dir / "render"
    ensure_dir(render_dir)
    ensure_dir(render_dir / "rgb")
    ensure_dir(render_dir / "silhouette")
    if save_depth:
        ensure_dir(render_dir / "depth")

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
    )
    phong_blend = BlendParams(background_color=(0.02, 0.02, 0.03))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            cameras=cameras,
            lights=PointLights(location=[[2.5, 2.0, -2.0]], device=cameras.device),
            blend_params=phong_blend,
        ),
    )

    sil_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-5, gamma=1e-5)),
    )

    mesh_batch = mesh.extend(cameras.R.shape[0])
    rgb = phong_renderer(mesh_batch)
    sil = sil_renderer(mesh_batch)[..., 3]
    fragments = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)(mesh_batch)
    zbuf = fragments.zbuf[..., 0]

    rgb_np = (to_numpy(rgb[..., :3]).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    sil_np = to_numpy(sil)
    depth_np = to_numpy(zbuf)

    masks = (sil_np >= silhouette_threshold).astype(np.uint8)

    for i in range(rgb_np.shape[0]):
        imageio.imwrite(render_dir / "rgb" / f"rgb_{i:03d}.png", rgb_np[i])
        imageio.imwrite(render_dir / "silhouette" / f"sil_{i:03d}.png", (masks[i] * 255).astype(np.uint8))
        if save_depth:
            np.save(render_dir / "depth" / f"depth_{i:03d}.npy", depth_np[i])

    return {
        "num_views": int(rgb_np.shape[0]),
        "image_size": int(image_size),
        "silhouette_threshold": float(silhouette_threshold),
    }


def save_cameras(camera_data: CameraData, out_dir: Path) -> None:
    render_dir = out_dir / "render"
    ensure_dir(render_dir)
    np.savez_compressed(
        render_dir / "cameras.npz",
        R=camera_data.R,
        t=camera_data.t,
        K=camera_data.K,
        image_size=np.array(camera_data.image_size, dtype=np.int32),
    )


def load_silhouettes(render_dir: Path, num_views: int) -> np.ndarray:
    masks = []
    for i in range(num_views):
        mask = imageio.imread(render_dir / "silhouette" / f"sil_{i:03d}.png")
        masks.append((mask > 0).astype(np.uint8))
    return np.stack(masks, axis=0)


def make_voxel_grid(bounds: float, resolution: int) -> np.ndarray:
    axis = np.linspace(-bounds, bounds, resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


def project_points(points: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # OpenCV/pinhole model: x ~ K (R X + t)
    cam = points @ R.T + t[None, :]
    z = cam[:, 2]
    xy_h = cam @ K.T
    uv = xy_h[:, :2] / np.maximum(xy_h[:, 2:3], 1e-8)
    return uv, z


def carve_space(
    camera_data: CameraData,
    silhouettes: np.ndarray,
    points: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    h, w = camera_data.image_size
    occupied = np.ones(points.shape[0], dtype=bool)

    for view_idx in range(camera_data.R.shape[0]):
        alive_idx = np.flatnonzero(occupied)
        if alive_idx.size == 0:
            break

        R = camera_data.R[view_idx]
        t = camera_data.t[view_idx]
        K = camera_data.K[view_idx]
        mask = silhouettes[view_idx]

        keep_flags = np.zeros(alive_idx.shape[0], dtype=bool)
        for s in range(0, alive_idx.shape[0], chunk_size):
            e = min(s + chunk_size, alive_idx.shape[0])
            idx = alive_idx[s:e]
            uv, z = project_points(points[idx], K, R, t)
            u = np.rint(uv[:, 0]).astype(np.int32)
            v = np.rint(uv[:, 1]).astype(np.int32)
            valid = (z > 0.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            inside = np.zeros_like(valid)
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size > 0:
                inside[valid_idx] = mask[v[valid_idx], u[valid_idx]] > 0
            keep_flags[s:e] = valid & inside

        occupied[alive_idx] = keep_flags

    return occupied


def export_visual_hull(points: np.ndarray, occupied: np.ndarray, out_dir: Path) -> Path:
    _require_open3d()
    import open3d as o3d

    carve_dir = out_dir / "carving"
    ensure_dir(carve_dir)

    occ_points = points[occupied]
    np.save(carve_dir / "occupied_points.npy", occ_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occ_points.astype(np.float64))
    pcd_path = carve_dir / "visual_hull_points.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    return pcd_path


def reconstruct_mesh(
    pcd_path: Path,
    out_dir: Path,
    method: str,
    voxel_size: float,
    poisson_depth: int,
) -> Path:
    _require_open3d()
    import open3d as o3d

    recon_dir = out_dir / "reconstruction"
    ensure_dir(recon_dir)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if len(pcd.points) == 0:
        raise RuntimeError("No points in visual hull point cloud; cannot reconstruct mesh.")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(2.0 * voxel_size, 1e-3), max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    if method == "poisson":
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=poisson_depth,
        )
        density = np.asarray(density)
        keep = density > np.quantile(density, 0.02)
        mesh.remove_vertices_by_mask(~keep)
    elif method == "ball_pivoting":
        radii = [voxel_size, 2.0 * voxel_size, 4.0 * voxel_size]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii),
        )
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh_path = recon_dir / "visual_hull_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    return mesh_path


def save_projection_debug(
    camera_data: CameraData,
    points: np.ndarray,
    occupied: np.ndarray,
    out_dir: Path,
    max_points: int = 4000,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    img_path = out_dir / "render" / "rgb" / "rgb_000.png"
    if not img_path.exists():
        return None

    image = imageio.imread(img_path)
    occ_points = points[occupied]
    if occ_points.shape[0] == 0:
        return None

    if occ_points.shape[0] > max_points:
        idx = np.random.choice(occ_points.shape[0], max_points, replace=False)
        occ_points = occ_points[idx]

    uv, z = project_points(occ_points, camera_data.K[0], camera_data.R[0], camera_data.t[0])
    u = uv[:, 0]
    v = uv[:, 1]

    valid = (
        (z > 0.0)
        & (u >= 0)
        & (u < image.shape[1])
        & (v >= 0)
        & (v < image.shape[0])
    )
    if not np.any(valid):
        return None

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.scatter(u[valid], v[valid], s=1.0, c="cyan", alpha=0.55)
    ax.set_title("Projected carved voxels on view 0")
    ax.axis("off")

    out_path = out_dir / "carving" / "projection_overlay_view0.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    start = time.time()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    mesh_path = Path(args.mesh_path) if args.mesh_path else None
    mesh = load_or_create_mesh(mesh_path, device=device)
    cameras = generate_cameras(
        num_views=args.num_views,
        image_size=args.image_size,
        fov_deg=args.fov_deg,
        distance=args.camera_distance,
        elev_min=args.elev_min,
        elev_max=args.elev_max,
        device=device,
    )

    render_info = render_views(
        mesh=mesh,
        cameras=cameras,
        image_size=args.image_size,
        out_dir=out_dir,
        silhouette_threshold=args.silhouette_threshold,
        save_depth=args.save_depth,
    )

    camera_data = pytorch3d_to_opencv(cameras, image_size=args.image_size)
    save_cameras(camera_data, out_dir)

    silhouettes = load_silhouettes(out_dir / "render", num_views=args.num_views)
    points = make_voxel_grid(bounds=args.grid_bounds, resolution=args.voxel_resolution)
    occupied = carve_space(
        camera_data=camera_data,
        silhouettes=silhouettes,
        points=points,
        chunk_size=args.chunk_size,
    )

    pcd_path = export_visual_hull(points, occupied, out_dir)
    mesh_out = None
    if not args.skip_reconstruction:
        voxel_size = 2.0 * args.grid_bounds / max(args.voxel_resolution - 1, 1)
        mesh_out = reconstruct_mesh(
            pcd_path=pcd_path,
            out_dir=out_dir,
            method=args.recon_method,
            voxel_size=voxel_size,
            poisson_depth=args.poisson_depth,
        )

    overlay = save_projection_debug(camera_data, points, occupied, out_dir)

    occupied_count = int(occupied.sum())
    metrics = {
        "config": vars(args),
        "resolved_device": str(device),
        "render": render_info,
        "voxel_grid_total": int(points.shape[0]),
        "occupied_voxels": occupied_count,
        "occupancy_ratio": float(occupied_count / max(points.shape[0], 1)),
        "outputs": {
            "point_cloud": str(pcd_path),
            "mesh": str(mesh_out) if mesh_out is not None else None,
            "projection_overlay": str(overlay) if overlay is not None else None,
        },
        "runtime_sec": float(time.time() - start),
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Space carving with PyTorch3D + Open3D")
    parser.add_argument("--mesh-path", type=str, default=None, help="Path to OBJ mesh. If omitted, uses an ico-sphere.")
    parser.add_argument("--out-dir", type=str, default="outputs/default_run")
    parser.add_argument("--num-views", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--camera-distance", type=float, default=2.7)
    parser.add_argument("--elev-min", type=float, default=-20.0)
    parser.add_argument("--elev-max", type=float, default=40.0)
    parser.add_argument("--silhouette-threshold", type=float, default=0.5)
    parser.add_argument("--save-depth", action="store_true")

    parser.add_argument("--voxel-resolution", type=int, default=96)
    parser.add_argument("--grid-bounds", type=float, default=1.1)
    parser.add_argument("--chunk-size", type=int, default=300000)

    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--recon-method", type=str, choices=["poisson", "ball_pivoting"], default="poisson")
    parser.add_argument("--poisson-depth", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metrics = run_pipeline(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
