"""
main.py
========
Single entry point for your module. Run this today with USE_REAL_DATA=False
to keep testing against synthetic data. Flip it to True once real data
is in place -- nothing else changes.
"""

from reconstruct import run_pipeline
from loaders import (
    load_frames_and_poses_uavscenes, load_depth_maps, load_lidar,
    to_selection_input, frames_from_selection,
)
from frame_selection.quality_sampler import select_frames

USE_REAL_DATA = False   # <-- flip this once Chehak's depth maps are ready
USE_FRAME_SELECTION = True  # <-- Unnati's adaptive selector; set False to use every frame

UAVSCENES_JSON = "data/sampleinfos_interpolated.json"        # real ground-truth poses, no COLMAP needed
UAVSCENES_IMAGES = "data/interval5_CAM"                        # matching images folder
CHEHAK_FOLDER = "data/chehak_output"
SAMARTH_LIDAR_FILE = "data/terra_ground_truth/HKairport03_merged_clean.ply"  # verified real ground truth
OUT_DIR = "output"


def main():
    if USE_REAL_DATA:
        print("Loading REAL data (UAVScenes)...")

        frames, rgb_images = load_frames_and_poses_uavscenes(UAVSCENES_JSON, UAVSCENES_IMAGES)

        if USE_FRAME_SELECTION:
            print("Running Unnati's adaptive frame selection...")
            selection_input = to_selection_input(frames, rgb_images)
            selected = select_frames(selection_input, f"{OUT_DIR}/selected_frames", f"{OUT_DIR}/selection.csv")
            frames, rgb_images = frames_from_selection(selected, frames, rgb_images)

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