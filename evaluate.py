"""
evaluate.py
============
Evaluation module (Hansika's part, drafted by Aryan so it's ready the
moment a real reconstruction exists).

Core metric: Chamfer distance -- the standard way to score "how close is
my reconstructed point cloud to the ground truth", used in essentially
every 3D reconstruction paper/benchmark (including Tanks & Temples, which
we referenced earlier). Lower is better; 0 = identical point sets.

Usage:
    python3 evaluate.py <reconstructed.ply> <ground_truth.ply>
"""

import sys
import numpy as np
import open3d as o3d


def chamfer_distance(pcd_a: o3d.geometry.PointCloud, pcd_b: o3d.geometry.PointCloud) -> dict:
    """
    Symmetric Chamfer distance between two point clouds:
      1. For every point in A, find its nearest neighbor in B, average those distances.
      2. For every point in B, find its nearest neighbor in A, average those distances.
      3. Chamfer distance = sum of both directions (or report separately -- both useful).

    Direction A->B measures "completeness" (does the reconstruction cover
    everything the ground truth has). Direction B->A measures "accuracy"
    (are the reconstructed points actually where they should be, or is
    there spurious/wrong geometry). Reporting both separately is more
    informative than just the single combined number.
    """
    dist_a_to_b = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    dist_b_to_a = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))

    return {
        "accuracy_mean_m": float(dist_a_to_b.mean()),      # reconstruction -> ground truth
        "accuracy_median_m": float(np.median(dist_a_to_b)),
        "completeness_mean_m": float(dist_b_to_a.mean()),  # ground truth -> reconstruction
        "completeness_median_m": float(np.median(dist_b_to_a)),
        "chamfer_mean_m": float(dist_a_to_b.mean() + dist_b_to_a.mean()),
        "chamfer_median_m": float(np.median(dist_a_to_b) + np.median(dist_b_to_a)),
    }


def f_score(pcd_a: o3d.geometry.PointCloud, pcd_b: o3d.geometry.PointCloud, threshold: float) -> dict:
    """
    F-score at a given distance threshold (meters) -- same metric Tanks &
    Temples uses for its leaderboard. A point "counts" as correct if its
    nearest neighbor in the other cloud is within `threshold`.

    threshold should be chosen relative to your scene scale and the
    accuracy you actually need -- e.g. 0.1m might be reasonable for a
    building-scale reconstruction, but pick based on what "close enough"
    means for your use case (NTRO reconnaissance use case -- ask Hansika/
    the team what tolerance actually matters).
    """
    dist_a_to_b = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    dist_b_to_a = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))

    precision = float((dist_a_to_b < threshold).mean())   # of reconstructed points, how many are correct
    recall = float((dist_b_to_a < threshold).mean())       # of ground truth points, how many were captured

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold_m": threshold,
        "precision": precision,
        "recall": recall,
        "f_score": f1,
    }


def align_before_comparison(reconstruction: o3d.geometry.PointCloud,
                             ground_truth: o3d.geometry.PointCloud,
                             icp_threshold: float = None) -> o3d.geometry.PointCloud:
    """
    IMPORTANT: your reconstruction and the ground truth must be in the same
    coordinate frame before comparing distances, or every number above is
    meaningless. If your reconstruction pipeline already fuses with LiDAR
    (fuse_with_lidar() in reconstruct.py), it should already be aligned --
    this is a safety-net re-alignment step in case it's evaluated standalone
    or drifted.
    """
    if icp_threshold is None:
        pts = np.asarray(ground_truth.points)
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        icp_threshold = max(diag / 200.0, 1e-3)

    reg = o3d.pipelines.registration.registration_icp(
        reconstruction, ground_truth, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    reconstruction.transform(reg.transformation)
    print(f"  Alignment ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
    return reconstruction


def evaluate(reconstruction_path: str, ground_truth_path: str,
             f_score_threshold: float = None, align: bool = True) -> dict:
    """
    Full evaluation entry point. Loads both point clouds, optionally
    re-aligns them, and reports Chamfer distance + F-score.
    """
    recon = o3d.io.read_point_cloud(reconstruction_path)
    gt = o3d.io.read_point_cloud(ground_truth_path)

    print(f"Reconstruction: {len(recon.points)} points")
    print(f"Ground truth:   {len(gt.points)} points")

    if align:
        print("Aligning reconstruction to ground truth (ICP)...")
        recon = align_before_comparison(recon, gt)

    if f_score_threshold is None:
        pts = np.asarray(gt.points)
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        f_score_threshold = diag / 1000.0  # 0.1% of scene scale, as a reasonable default

    metrics = chamfer_distance(recon, gt)
    metrics.update(f_score(recon, gt, f_score_threshold))

    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate.py <reconstructed.ply> <ground_truth.ply>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2])