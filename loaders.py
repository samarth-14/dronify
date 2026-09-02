"""
loaders.py
===========
One loader function per teammate / data source. Each converts raw
outside data into the shared interfaces (CameraFrame, DepthMap, LidarScan)
defined in interfaces.py, so reconstruct.py's run_pipeline() never has
to change regardless of where the data actually came from.
"""

import os
import glob
import json
import re
import numpy as np
import cv2

from interfaces import CameraFrame, DepthMap, LidarScan


# ----------------------------------------------------------------------
# UNNATI -- Frame Selection (if she also provides poses)
# ----------------------------------------------------------------------
def load_frames_and_poses(folder: str):
    """
    ASSUMED layout (confirm with Unnati):
        folder/
          frames/
            0001.jpg
            0002.jpg
            ...
          poses.json   <- list of {frame_id, intrinsics (3x3), pose (4x4)}

    Returns: (List[CameraFrame], List[np.ndarray] rgb_images)
    """
    poses_path = os.path.join(folder, "poses.json")
    if not os.path.exists(poses_path):
        raise FileNotFoundError(
            f"Expected {poses_path} -- confirm with Unnati whether she outputs "
            f"poses at all, or only selects frames (in which case YOU need to "
            f"run SfM/COLMAP yourself to get poses -- see estimate_poses_with_colmap() below)."
        )

    with open(poses_path) as f:
        pose_entries = json.load(f)

    frame_paths = sorted(glob.glob(os.path.join(folder, "frames", "*.jpg")))

    frames, rgb_images = [], []
    for entry in pose_entries:
        fid = entry["frame_id"]
        img_path = frame_paths[fid] if fid < len(frame_paths) else None
        if img_path is None:
            continue
        rgb = cv2.imread(img_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        frame = CameraFrame(
            frame_id=fid,
            image_path=img_path,
            intrinsics=np.array(entry["intrinsics"], dtype=np.float64),
            pose=np.array(entry["pose"], dtype=np.float64),
        )
        frames.append(frame)
        rgb_images.append(rgb)

    return frames, rgb_images


# ----------------------------------------------------------------------
# COLMAP fallback -- only needed if a data source does NOT provide poses
# ----------------------------------------------------------------------
def run_colmap(image_folder: str, output_folder: str):
    """
    Runs COLMAP's standard sparse pipeline and writes a TEXT model
    (cameras.txt, images.txt, points3D.txt) which colmap_to_frames()
    below can parse without extra dependencies.
    Requires `colmap` installed: apt install colmap (or build from source).
    """
    os.makedirs(output_folder, exist_ok=True)
    db_path = os.path.join(output_folder, "database.db")
    sparse_path = os.path.join(output_folder, "sparse", "0")
    text_path = os.path.join(output_folder, "sparse_text")
    os.makedirs(os.path.dirname(sparse_path), exist_ok=True)
    os.makedirs(text_path, exist_ok=True)

    commands = [
        f"colmap feature_extractor --database_path {db_path} --image_path {image_folder}",
        f"colmap exhaustive_matcher --database_path {db_path}",
        f"colmap mapper --database_path {db_path} --image_path {image_folder} "
        f"--output_path {os.path.dirname(sparse_path)}",
        f"colmap model_converter --input_path {sparse_path} --output_path {text_path} "
        f"--output_type TXT",
    ]
    for cmd in commands:
        print(f"  $ {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"COLMAP step failed: {cmd}")

    return text_path


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,       2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,       1 - 2*qx**2 - 2*qy**2],
    ])


def colmap_to_frames(colmap_text_path: str, image_folder: str):
    """
    Parses COLMAP's TEXT model output (cameras.txt + images.txt) into
    CameraFrame objects.

    IMPORTANT (verified 2026-09-01): COLMAP's images.txt stores
    WORLD-TO-CAMERA extrinsics (X_cam = R * X_world + t). Your
    reconstruction pipeline needs CAMERA-TO-WORLD poses, so this
    function inverts it:
        R_c2w = R^T
        t_c2w = -R^T @ t
    Verified mathematically to round-trip correctly -- do not skip this.

    Returns: List[CameraFrame], mapping frame_id = COLMAP image_id
    """
    cameras_path = os.path.join(colmap_text_path, "cameras.txt")
    images_path = os.path.join(colmap_text_path, "images.txt")

    cameras = {}
    with open(cameras_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            params = list(map(float, parts[4:]))

            if model in ("PINHOLE",):
                fx, fy, cx, cy = params
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                f_, cx, cy = params[0], params[1], params[2]
                fx = fy = f_
            else:
                raise NotImplementedError(
                    f"COLMAP camera model '{model}' not handled -- add it here."
                )

            cameras[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    frames = []
    with open(images_path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cam_id = int(parts[8])
        name = parts[9]

        R_w2c = qvec_to_rotmat(qvec)
        t_w2c = tvec

        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        pose_c2w = np.eye(4)
        pose_c2w[:3, :3] = R_c2w
        pose_c2w[:3, 3] = t_c2w

        frames.append(CameraFrame(
            frame_id=image_id,
            image_path=os.path.join(image_folder, name),
            intrinsics=cameras[cam_id],
            pose=pose_c2w,
        ))

    frames.sort(key=lambda f: f.frame_id)
    return frames


def estimate_poses_with_colmap(image_folder: str, output_folder: str):
    """Convenience wrapper: runs COLMAP end-to-end and returns ready-to-use CameraFrame objects."""
    text_path = run_colmap(image_folder, output_folder)
    return colmap_to_frames(text_path, image_folder)


# ----------------------------------------------------------------------
# UAVScenes-specific: real ground-truth poses (no COLMAP needed)
# ----------------------------------------------------------------------
def load_frames_and_poses_uavscenes(json_path: str, images_folder: str):
    """
    CONFIRMED real format from UAVScenes' sampleinfos_interpolated.json.
    This dataset ships REAL GROUND-TRUTH POSES per frame -- no COLMAP needed.

    Verified (2026-09-01): T4x4 is a proper camera-to-world rigid transform
    (rotation determinant == 1.0, orthonormal) -- matches CameraFrame.pose
    directly, no inversion required (unlike the raw-COLMAP case above).

    Returns: (List[CameraFrame], List[np.ndarray] rgb_images)
    """
    with open(json_path) as f:
        data = json.load(f)

    data.sort(key=lambda e: e["SortedImageID"])

    frames, rgb_images = [], []
    missing = 0
    for entry in data:
        img_name = entry["OriginalImageName"]
        img_path = os.path.join(images_folder, img_name)
        if not os.path.exists(img_path):
            missing += 1
            continue

        rgb = cv2.imread(img_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        frame = CameraFrame(
            frame_id=entry["SortedImageID"],
            image_path=img_path,
            intrinsics=np.array(entry["P3x3"], dtype=np.float64),
            pose=np.array(entry["T4x4"], dtype=np.float64),
        )
        frames.append(frame)
        rgb_images.append(rgb)

    if missing:
        print(f"  Warning: {missing}/{len(data)} images listed in JSON were not "
              f"found in {images_folder} -- check the folder path.")

    return frames, rgb_images


# ----------------------------------------------------------------------
# CHEHAK -- Depth Estimation
# ----------------------------------------------------------------------
def load_depth_maps(folder: str, frame_ids: list, is_metric: bool = True, scale_factor: float = 1.0):
    """
    ASSUMED layout (confirm with Chehak):
        folder/
          0001_depth.npy   <- HxW float32 array
          0002_depth.npy
          ...

    CRITICAL: confirm with Chehak whether depth is metric or relative --
    this is the #1 source of fusion bugs.

    Returns: List[DepthMap]
    """
    depths = []
    for fid in frame_ids:
        path = os.path.join(folder, f"{fid:04d}_depth.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected {path} -- confirm exact filename pattern and array "
                f"format (npy vs png vs exr) with Chehak."
            )
        depth_arr = np.load(path).astype(np.float32)
        depths.append(DepthMap(
            frame_id=fid,
            depth=depth_arr,
            is_metric=is_metric,
            scale_factor=scale_factor,
        ))
    return depths


# ----------------------------------------------------------------------
# SAMARTH -- LiDAR Data
# ----------------------------------------------------------------------
def load_lidar(file_path: str, max_points: int = 2_000_000) -> LidarScan:
    """
    Handles both:
      - .laz/.las (e.g. Samarth's early OpenTopography test file)
      - .ply (e.g. Samarth's final HKairport03_merged_clean.ply -- USE THIS ONE)

    max_points subsamples randomly if the scan exceeds this, so ICP
    doesn't choke on millions of points.

    Returns: LidarScan, or None if no LiDAR is available for this run.
    """
    if file_path is None or not os.path.exists(file_path):
        print("  No LiDAR file found/provided -- continuing without LiDAR fusion.")
        return None

    if file_path.endswith((".laz", ".las")):
        import laspy
        las = laspy.read(file_path)
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

        intensities = None
        if hasattr(las, "intensity"):
            intensities = np.asarray(las.intensity, dtype=np.float64).reshape(-1, 1)

        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            if intensities is not None:
                intensities = intensities[idx]

        return LidarScan(points=points, intensities=intensities)

    if file_path.endswith(".ply"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
        return LidarScan(points=points)

    raise NotImplementedError(f"Unhandled LiDAR file format for {file_path}.")


# ----------------------------------------------------------------------
# UAVScenes-specific: fuse raw per-frame LiDAR scans into one map
# (fallback / cross-check only -- Samarth's HKairport03_merged_clean.ply
#  already does this + cleanup, prefer load_lidar() with his .ply directly)
# ----------------------------------------------------------------------
HK_GNSS_CALIBRATION = {
    "camera_ext_R": [0.00322743, -0.999736, -0.022768,
                      -0.0560389, 0.0227496, -0.999725,
                      0.999979, 0.00335414, -0.00552898],
    "camera_ext_t": [0.00242461, 0.0765454, -0.0313375],
}


def _lidar_to_cam_transform(calib: dict) -> np.ndarray:
    R = np.array(calib["camera_ext_R"], dtype=np.float64).reshape(3, 3)
    t = np.array(calib["camera_ext_t"], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _parse_lidar_filename_timestamp(filename: str) -> str:
    m = re.match(r"image(\d+)_(\d+)_lidar", os.path.basename(filename))
    if not m:
        raise ValueError(f"Unrecognized LiDAR filename pattern: {filename}")
    return f"{m.group(1)}.{m.group(2)}"


def fuse_uavscenes_lidar(json_path: str, lidar_folder: str,
                          calib: dict = HK_GNSS_CALIBRATION,
                          max_frames: int = None) -> LidarScan:
    """
    Fuses raw per-frame LiDAR .txt scans into one merged world-frame
    point cloud. Verified correct on real data (2026-09-01) -- Samarth's
    independent notebook implementation matches this exact approach.
    Prefer his cleaned output (HKairport03_merged_clean.ply via load_lidar())
    for the actual pipeline; use this only as a fallback/cross-check.
    """
    with open(json_path) as f:
        pose_data = json.load(f)
    pose_by_ts = {e["OriginalImageName"].replace(".jpg", ""): e for e in pose_data}

    T_lidar_to_cam = _lidar_to_cam_transform(calib)

    lidar_files = sorted(glob.glob(os.path.join(lidar_folder, "image*_lidar*.txt")))
    if max_frames:
        lidar_files = lidar_files[:max_frames]

    all_points = []
    matched, unmatched = 0, 0

    for lf in lidar_files:
        ts_key = _parse_lidar_filename_timestamp(lf)
        pose_entry = pose_by_ts.get(ts_key)
        if pose_entry is None:
            unmatched += 1
            continue

        T_cam_to_world = np.array(pose_entry["T4x4"], dtype=np.float64)
        T_lidar_to_world = T_cam_to_world @ T_lidar_to_cam

        pts_local = np.loadtxt(lf)
        pts_h = np.hstack([pts_local, np.ones((len(pts_local), 1))])
        pts_world = (T_lidar_to_world @ pts_h.T).T[:, :3]

        all_points.append(pts_world)
        matched += 1

    if unmatched:
        print(f"  Warning: {unmatched}/{len(lidar_files)} LiDAR frames had no matching pose.")
    print(f"  Fused {matched} LiDAR frames into world frame.")

    merged = np.vstack(all_points) if all_points else np.zeros((0, 3))
    return LidarScan(points=merged)