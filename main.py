"""
main.py
========
Single entry point for your module. Run this today with USE_REAL_DATA=False
to keep testing against synthetic data. Flip it to True once real data
is in place -- nothing else changes.
This file basically tells which dataset we are running against. 
Is it the temporary dataset or the actual one.
"""

from reconstruct import run_pipeline
from loaders import load_frames_and_poses_uavscenes, load_depth_maps, load_lidar

USE_REAL_DATA = False   # <-- flip this once Chehak's depth maps are ready

UAVSCENES_JSON = "data/sampleinfos_interpolated.json"        # real ground-truth poses, no COLMAP needed
UAVSCENES_IMAGES = "data/interval5_CAM"                        # matching images folder
CHEHAK_FOLDER = "data/chehak_output"
SAMARTH_LIDAR_FILE = "data/terra_ground_truth/HKairport03_merged_clean.ply"  # verified real ground truth
OUT_DIR = "output"


def main():
    if USE_REAL_DATA:
        print("Loading REAL data (UAVScenes)...")

        frames, rgb_images = load_frames_and_poses_uavscenes(UAVSCENES_JSON, UAVSCENES_IMAGES)

        frame_ids = [f.frame_id for f in frames]
        depths = load_depth_maps(CHEHAK_FOLDER, frame_ids)
        lidar = load_lidar(SAMARTH_LIDAR_FILE)
    else:
        print("Loading SYNTHETIC data (no real depth maps yet)...")
        from test_synthetic import make_synthetic_scene
        frames, depths, rgb_images = make_synthetic_scene(n_views=12, img_size=200)
        lidar = None

    result = run_pipeline(frames, depths, rgb_images, lidar=lidar, out_dir=OUT_DIR)
    print("\nDone:", result)


if __name__ == "__main__":
    main()