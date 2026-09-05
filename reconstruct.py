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