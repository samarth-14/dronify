"""
test_synthetic.py
==================
This file is responsible for generating FAKE FRAMES, poses, and depth maps so that
one dosent have to wait for the testing of the reconstruction pipeline

When real data is ready, replace the data-generation functions below with
real loaders -- run_pipeline() in reconstruct.py does not need to change.
"""

import numpy as np
import open3d as o3d
from interfaces import CameraFrame, DepthMap
from reconstruct import run_pipeline


def make_synthetic_scene(n_views: int = 16, img_size: int = 200):
    """
    Simulates a drone orbiting a sphere and 'filming' it: generates
    camera poses in a circle, and renders a fake depth map + RGB image
    for each viewpoint using Open3D's own raycasting scene. This
    guarantees the depth values match Open3D's own unprojection
    convention exactly, so there's no camera-convention mismatch bug.
    """
    sphere_radius = 1.0
    camera_distance = 3.0

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=40)
    sphere.translate([0, 0, sphere_radius])
    sphere.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sphere))

    fx = fy = 200.0
    cx = cy = img_size / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    frames, depths, rgb_images = [], [], []

    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        eye = [camera_distance * np.cos(angle), camera_distance * np.sin(angle), sphere_radius + 0.3]
        center = [0, 0, sphere_radius]
        up = [0, 0, 1]

        rays = scene.create_rays_pinhole(
            fov_deg=2 * np.degrees(np.arctan(cx / fx)),
            center=center, eye=eye, up=up,
            width_px=img_size, height_px=img_size
        )
        ans = scene.cast_rays(rays)
        depth_map = ans['t_hit'].numpy().astype(np.float32)
        depth_map[np.isinf(depth_map)] = 0.0

        forward = np.array(center) - np.array(eye)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        R = np.stack([right, -true_up, forward], axis=1)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = eye

        rgb = np.full((img_size, img_size, 3), 40, dtype=np.uint8)
        hit_mask = depth_map > 0
        rgb[hit_mask] = [180, 130, 210]

        frame = CameraFrame(frame_id=i, image_path=f"synthetic_{i}.png", intrinsics=K, pose=pose)
        depth = DepthMap(frame_id=i, depth=depth_map, is_metric=True, scale_factor=1.0)

        frames.append(frame)
        depths.append(depth)
        rgb_images.append(rgb)

    return frames, depths, rgb_images


if __name__ == "__main__":
    print("Generating synthetic drone-orbit data (fake sphere scene)...")
    frames, depths, rgb_images = make_synthetic_scene(n_views=12, img_size=200)
    print(f"Generated {len(frames)} synthetic views.\n")

    result = run_pipeline(frames, depths, rgb_images, lidar=None, out_dir="output")

    print("\n--- RESULT ---")
    print(result)