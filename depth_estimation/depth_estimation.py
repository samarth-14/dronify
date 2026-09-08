import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEPTH_ANYTHING_ROOT = PROJECT_ROOT.parent / "Depth-Anything-V2"

DATA_ROOT = PROJECT_ROOT / "data"

CAMERA_DIR = DATA_ROOT / "interval5_CAM"
JSON_PATH = DATA_ROOT / "sampleinfos_interpolated.json"

OUTPUT_DIR = PROJECT_ROOT / "chehak_output"

CHECKPOINT_PATH = (
    DEPTH_ANYTHING_ROOT
    / "metric_depth"
    / "checkpoints"
    / "depth_anything_v2_metric_vkitti_vits.pth"
)


# ============================================================
# IMPORT METRIC DEPTH ANYTHING V2
# ============================================================

# IMPORTANT:
# Use metric_depth version, NOT the relative-depth version.
sys.path.insert(0, str(DEPTH_ANYTHING_ROOT / "metric_depth"))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# CHECK PATHS
# ============================================================

print("Checking paths...")

print("Camera folder:", CAMERA_DIR)
print("Metadata file:", JSON_PATH)
print("Checkpoint:", CHECKPOINT_PATH)

if not CAMERA_DIR.exists():
    raise FileNotFoundError(f"Camera folder not found: {CAMERA_DIR}")

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {JSON_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Metric Depth Anything V2...")

model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=80
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD FRAME METADATA
# ============================================================

print("\nLoading frame metadata...")

with open(JSON_PATH, "r") as f:
    metadata = json.load(f)

frame_map = {
    item["OriginalImageName"]: item["SortedImageID"]
    for item in metadata
}

print(f"Metadata entries: {len(frame_map)}")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Output directory:", OUTPUT_DIR)


# ============================================================
# FIND CAMERA IMAGES
# ============================================================

image_paths = sorted(CAMERA_DIR.glob("*.jpg"))

print(f"\nFound {len(image_paths)} camera images.")


# ============================================================
# PROCESS ALL FRAMES
# ============================================================

for i, image_path in enumerate(image_paths, start=1):

    filename = image_path.name

    # --------------------------------------------------------
    # Find frame ID
    # --------------------------------------------------------

    if filename not in frame_map:
        print(f"[SKIP] No frame ID found for {filename}")
        continue

    frame_id = frame_map[filename]

    output_path = OUTPUT_DIR / f"{frame_id:04d}_depth.npy"


    # --------------------------------------------------------
    # Skip existing output
    # --------------------------------------------------------

    if output_path.exists():

        print(
            f"[{i}/{len(image_paths)}] "
            f"Frame {frame_id:04d}: already exists"
        )

        continue


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        print(
            f"[ERROR] Could not read image: {filename}"
        )

        continue


    # --------------------------------------------------------
    # Estimate metric depth
    # --------------------------------------------------------

    depth = model.infer_image(image)

    depth = depth.astype(np.float32)


    # --------------------------------------------------------
    # Save depth
    # --------------------------------------------------------

    np.save(output_path, depth)


    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"[{i}/{len(image_paths)}] "
        f"Frame {frame_id:04d} → "
        f"{output_path.name} | "
        f"shape={depth.shape} | "
        f"min={depth.min():.2f}m | "
        f"max={depth.max():.2f}m | "
        f"mean={depth.mean():.2f}m"
    )


# ============================================================
# DONE
# ============================================================

print("\n======================================")
print("DEPTH ESTIMATION COMPLETE")
print("======================================")

print(f"Outputs saved to:")
print(OUTPUT_DIR)