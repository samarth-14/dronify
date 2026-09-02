"""
interfaces.py
=============
This file acts as a CONTRACT between the team members so that everyone 
working agrees on a particular format before integration day.

Since you're all working separately right now, build your module against
these interfaces using fake/synthetic data. When real data arrives from
teammates, you just need to convert it into these shapes -- your core
reconstruction logic never has to change.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, List


@dataclass
class CameraFrame:
    """
    One selected frame + its camera pose.
    This is what you expect FROM UNNATI (frame selection) per frame,
    OR what you compute yourself in Step 1 if she doesn't provide poses.
    """
    frame_id: int
    image_path: str          # path to the RGB frame (jpg/png)
    intrinsics: np.ndarray   # 3x3 camera intrinsic matrix K
    pose: np.ndarray         # 4x4 camera-to-world extrinsic matrix (R|t)


@dataclass
class DepthMap:
    """
    What you expect FROM CHEHAK per frame.
    IMPORTANT: confirm with her whether values are:
      - metric depth (meters, real-world scale), or
      - relative/inverse depth (needs a scale factor to become metric)
    This single detail causes most fusion bugs -- lock it down explicitly.
    """
    frame_id: int
    depth: np.ndarray        # HxW float array, same resolution as the RGB frame
    is_metric: bool          # True = real-world meters, False = relative/needs scaling
    scale_factor: float = 1.0  # multiply by this to convert to meters, if not metric


@dataclass
class LidarScan:
    """
    What you expect FROM SAMARTH, if LiDAR is available/used.
    """
    points: np.ndarray       # Nx3 array of XYZ points, in meters, world frame
    intensities: Optional[np.ndarray] = None  # optional Nx1 reflectivity values


@dataclass
class ReconstructionOutput:
    """
    What YOU hand to Gauri (UI) and Hansika (evaluation).
    """
    mesh_path: str           # exported mesh file, e.g. "model.obj" or "model.glTF"
    point_cloud_path: str    # exported raw fused point cloud, e.g. "cloud.ply"
    format: str               # "obj" | "gltf" | "ply" -- CONFIRM WITH GAURI
    num_points: int
    num_vertices: int
    num_faces: int