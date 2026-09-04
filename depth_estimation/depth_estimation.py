import sys
import numpy as np
import torch
import cv2

sys.path.append(r"C:\Users\HP\Documents\chehak\depth_estimation\Depth-Anything-V2"
)

from depth_anything_v2.dpt import DepthAnythingV2

model=DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48,96,192,384]
)

model.load_state_dict(
    torch.load(
        r"C:\Users\HP\Documents\chehak\depth_estimation\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth",
        map_location="cpu"
    )
)

model.eval()

image_path = r"C:\Users\HP\Downloads\interval5_HKairport03\interval5_HKairport03\interval5_CAM\1671607392.199796915.jpg"
image=cv2.imread(image_path)

depth = model.infer_image(image)

print("Depth shape:", depth.shape)
print("Depth dtype:", depth.dtype)

np.save("0000_depth.npy", depth)

print("Saved:", "0000_depth.npy")