# Dronify — SIH26158

Single-Pass Drone Video to Accurate 3D Model Generation System (NTRO, SIH 2026)

## Team
- Gauri — Object detection and UI
- Unnati — Frame selection + Integration
- Hansika — Evaluation and research
- Chehak — Depth estimation
- Samarth — LiDAR data
- Aryan — 3D reconstruction

## Module: 3D Reconstruction (`interfaces.py`, `reconstruct.py`, `loaders.py`, `main.py`)

`data_contract.txt` documents the exact data format this module expects from each
teammate — read this first before wiring in your own output.

### Setup
```bash
pip install -r requirements.txt
```

### Quick test (no real data needed)
```bash
python3 test_synthetic.py
```
Generates a fake drone-orbit scene and runs it through the full pipeline —
confirms the code works before any real data is ready.

### Full pipeline
```bash
python3 main.py
```
Toggle `USE_REAL_DATA = True/False` at the top of `main.py` to switch between
synthetic and real data.

## Dataset

We're using **UAVScenes** (built on MARS-LVIG) — real DJI M300 drone flights
with synchronized camera + LiDAR, real ground-truth poses, no COLMAP needed.
Scene: `HKairport03`.

- Dataset homepage: https://github.com/sijieaaa/UAVScenes
- **Full dataset download (Google Drive, HKairport03 only):** https://drive.google.com/file/d/1icqwi7kCu9bHAl0FTULwO_-mrQcgAoAv/view?usp=drive_link

The full dataset (`interval5_CAM` images, `interval5_LIDAR` raw scans) is **not
committed to this repo** — it's multiple GB and would make clones painfully
slow for everyone. Download it from the Drive link above and place it under
`data/` locally (already gitignored).

### Small real sample (committed, for quick testing)
`data/sample_HKairport03/` has 4 real frames (LiDAR scans + matching poses +
RTK positions) — enough to test the loaders and fusion code without
downloading the full dataset.

```python
from loaders import fuse_uavscenes_lidar
scan = fuse_uavscenes_lidar(
    json_path="data/sample_HKairport03/sampleinfos_interpolated.json",
    lidar_folder="data/sample_HKairport03"
)
```

### Real LiDAR ground truth (Samarth's merged + cleaned scan)
`data/terra_ground_truth/HKairport03_merged_clean.ply` — 605 frames fused,
statistical outlier removal + 5cm voxel downsample applied. ~15M points,
~586m x 435m extent. Coordinate frame: DJI Terra's local map frame (not
raw lat/lon/UTM). This is the ground truth to compare your reconstruction
against.

### Expected local folder layout (after downloading the full dataset)
```
data/
  interval5_CAM/                    # camera images
  interval5_LIDAR/                  # raw per-frame LiDAR scans
  sampleinfos_interpolated.json     # real ground-truth poses + intrinsics
  rtk_positions_raw.csv             # raw RTK GPS positions
  sample_HKairport03/               # small committed sample (already in repo)
  terra_ground_truth/               # Samarth's merged + cleaned LiDAR ground truth
```