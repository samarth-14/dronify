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
- Full dataset download (Google Drive):(https://drive.google.com/drive/folders/1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN)

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

### Expected local folder layout (after downloading the full dataset)
```
data/
  interval5_CAM/                    # camera images
  interval5_LIDAR/                  # raw per-frame LiDAR scans
  sampleinfos_interpolated.json     # real ground-truth poses + intrinsics
  rtk_positions_raw.csv             # raw RTK GPS positions
  sample_HKairport03/               # small committed sample (already in repo)
```