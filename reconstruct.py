"""
reconstruct.py
===============
Core 3D Reconstruction module (Aryan's part).

Pipeline: CameraFrame + DepthMap  -->  point cloud
          (+ optional LidarScan)  -->  fused point cloud
                                   -->  cleaned point cloud
                                   -->  mesh (Poisson)
                                   -->  exported files

Run this file directly to test against synthetic data (see test_synthetic.py).
"""

import os
import numpy as np
import open3d as o3d
from typing import List, Optional
from interfaces import CameraFrame, DepthMap, LidarScan, ReconstructionOutput


def depth_to_pointcloud(frame: CameraFrame, depth: DepthMap, rgb_image: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Step 2: Back-project a single depth map into a colored 3D point cloud,
    using the camera intrinsics and pose from CameraFrame.
    """
    depth_m = depth.depth.astype(np.float32)
    if not depth.is_metric:
        depth_m = depth_m * depth.scale_factor

    h, w = depth_m.shape
    fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
    cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]

    depth_o3d = o3d.geometry.Image(depth_m)
    color_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Transform from camera space into world space using the pose
    pcd.transform(frame.pose)
    return pcd


def fuse_frames(frames: List[CameraFrame], depths: List[DepthMap], rgb_images: List[np.ndarray]) -> o3d.geometry.PointCloud:
    """
    Step 2 (continued): merge per-frame point clouds into one global point cloud.
    """
    fused = o3d.geometry.PointCloud()
    for frame, depth, rgb in zip(frames, depths, rgb_images):
        pcd = depth_to_pointcloud(frame, depth, rgb)
        fused += pcd
    return fused


def fuse_with_lidar(depth_pcd: o3d.geometry.PointCloud, lidar: LidarScan,
                     icp_threshold: float = None) -> o3d.geometry.PointCloud:
    """
    Step 3: align and fuse LiDAR points (accurate, sparse) with
    depth-derived points (dense, less accurate scale) using ICP.

    icp_threshold=None (default) auto-scales to the LiDAR scan's own size,
    same reasoning as clean_pointcloud()'s voxel_size -- a fixed 0.5m
    threshold tuned for a small synthetic scene is not automatically
    right for a 700m real scene. Override explicitly if ICP fails to
    converge or over-converges.
    """
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar.points)

    if icp_threshold is None:
        diag = np.linalg.norm(lidar.points.max(axis=0) - lidar.points.min(axis=0))
        icp_threshold = max(diag / 1000.0, 1e-3)
        print(f"  Auto icp_threshold = {icp_threshold:.4f}m (LiDAR scan diagonal = {diag:.2f}m)")

    # ICP aligns depth_pcd onto lidar_pcd (lidar treated as more trustworthy)
    reg = o3d.pipelines.registration.registration_icp(
        depth_pcd, lidar_pcd, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    depth_pcd.transform(reg.transformation)

    fused = depth_pcd + lidar_pcd
    print(f"  ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
    return fused


def clean_pointcloud(pcd: o3d.geometry.PointCloud, voxel_size: float = None) -> o3d.geometry.PointCloud:
    """
    Step 4: remove noise and downsample so meshing doesn't choke on outliers.

    voxel_size=None (default) auto-scales to the point cloud's own size --
    critical because a fixed small voxel size (e.g. 0.02m, fine for a
    ~2m synthetic test sphere) becomes catastrophically wrong at real-world
    scale: a 700m-diagonal scene at 2cm resolution implies tens of millions
    of voxels, which is painfully slow or crashes outright. Auto-scaling
    targets roughly bbox_diagonal / 200 as a sane starting resolution --
    override explicitly once you know what detail level you actually need.
    """
    if voxel_size is None:
        pts = np.asarray(pcd.points)
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        voxel_size = max(diag / 200.0, 1e-4)
        print(f"  Auto voxel_size = {voxel_size:.4f}m (scene diagonal = {diag:.2f}m)")

    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def mesh_from_pointcloud(pcd: o3d.geometry.PointCloud, depth: int = 9) -> o3d.geometry.TriangleMesh:
    """
    Step 5: Poisson surface reconstruction -- turns the point cloud into a solid mesh.
    Requires point normals, so we estimate them first.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Trim low-density (unreliable) vertices -- common Poisson cleanup step
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def export_results(pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh,
                    out_dir: str = ".", mesh_format: str = "obj") -> ReconstructionOutput:
    """
    Step 7: export files in the format Gauri's UI needs.
    CONFIRM mesh_format with Gauri -- "obj", "gltf", or "ply".
    """
    os.makedirs(out_dir, exist_ok=True)

    cloud_path = f"{out_dir}/fused_cloud.ply"
    mesh_path = f"{out_dir}/model.{mesh_format}"

    o3d.io.write_point_cloud(cloud_path, pcd)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    return ReconstructionOutput(
        mesh_path=mesh_path,
        point_cloud_path=cloud_path,
        format=mesh_format,
        num_points=len(pcd.points),
        num_vertices=len(mesh.vertices),
        num_faces=len(mesh.triangles),
    )


def run_pipeline(frames: List[CameraFrame], depths: List[DepthMap], rgb_images: List[np.ndarray],
                  lidar: Optional[LidarScan] = None, out_dir: str = ".") -> ReconstructionOutput:
    """
    Full pipeline, callable end-to-end. This is the function your teammates'
    real data will eventually be plugged into.
    """
    print("[1/5] Fusing depth maps into point cloud...")
    pcd = fuse_frames(frames, depths, rgb_images)
    print(f"      -> {len(pcd.points)} raw points")

    if lidar is not None:
        print("[2/5] Fusing with LiDAR data...")
        pcd = fuse_with_lidar(pcd, lidar)
        print(f"      -> {len(pcd.points)} points after LiDAR fusion")
    else:
        print("[2/5] No LiDAR data provided, skipping fusion.")

    print("[3/5] Cleaning point cloud...")
    pcd = clean_pointcloud(pcd)
    print(f"      -> {len(pcd.points)} points after cleanup")

    print("[4/5] Meshing (Poisson reconstruction)...")
    mesh = mesh_from_pointcloud(pcd)
    print(f"      -> {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    print("[5/5] Exporting...")
    result = export_results(pcd, mesh, out_dir=out_dir)
    print(f"      -> saved {result.mesh_path} and {result.point_cloud_path}")

    return result