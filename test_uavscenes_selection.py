from loaders import load_frames_and_poses_uavscenes
from frame_selection.quality_sampler import select_frames


DATASET = "/Users/unnatijindal/Downloads/interval5_HKairport03"

JSON_PATH = DATASET + "/sampleinfos_interpolated.json"
IMAGE_FOLDER = DATASET + "/interval5_CAM"

OUTPUT_FOLDER = "test_output/uavscenes_selected"
OUTPUT_CSV = "test_output/uavscenes_frame_selection.csv"


# Load UAVScenes using the existing repo loader
uav_frames, rgb_images = load_frames_and_poses_uavscenes(
    JSON_PATH,
    IMAGE_FOLDER
)

print(f"UAVScenes frames available: {len(uav_frames)}")


# Convert to the format expected by our generic selector
frames = []

for frame, image in zip(uav_frames, rgb_images):

    frames.append({
    "frame_id": frame.frame_id,
    "timestamp": float(
        frame.image_path.split("/")[-1].replace(".jpg", "")
    ),
    "image": image,
    "pose": frame.pose,
    "intrinsics": frame.intrinsics
    })


# Run the SAME selector used for video
selected_frames = select_frames(
    frames,
    OUTPUT_FOLDER,
    OUTPUT_CSV
)


print()
print("UAVScenes selection finished.")
print(f"Input frames: {len(frames)}")
print(f"Selected frames: {len(selected_frames)}")

print()
print("Selected UAVScenes frame IDs:")

for frame in selected_frames[:20]:
    print(
        frame["frame_id"],
        frame["timestamp"],
        frame["image_path"]
    )