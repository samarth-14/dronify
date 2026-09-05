import cv2


def read_video(video_path, max_frames=None):
    """
    Read frames from a drone video.

    Returns:
        List of dictionaries containing:
        - frame_id
        - timestamp
        - image
    """

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frames = []

    frame_id = 0

    while True:

        success, image = capture.read()

        if not success:
            break

        timestamp = frame_id / fps

        frames.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image": image
            }
        )

        frame_id += 1

        if max_frames is not None and frame_id >= max_frames:
            break

    capture.release()

    return frames, fps