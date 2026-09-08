import os
import csv
import cv2
import numpy as np

from frame_selection.redundancy import compare_features


def _calculate_quality(image, orb):
    small_image = cv2.resize(
    image,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
    small_image,
    cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    sharpness_score = min(
        blur_score / 2000.0,
        1.0
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    feature_count = len(keypoints) if keypoints else 0

    h, w = gray.shape
    occupied_cells = set()

    if keypoints:
        for kp in keypoints:
            x, y = kp.pt

            cell_x = min(
                int(x / w * 8),
                7
            )

            cell_y = min(
                int(y / h * 6),
                5
            )

            occupied_cells.add(
                (cell_x, cell_y)
            )

    coverage = len(occupied_cells) / 48.0

    quality_score = (
        0.6 * sharpness_score +
        0.4 * coverage
    )

    return (
        gray,
        keypoints,
        descriptors,
        blur_score,
        sharpness_score,
        feature_count,
        coverage,
        quality_score
    )


def _adaptive_threshold(values):
    """
    Calculate a robust threshold from the video's
    own observed distribution.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:
        return 0.0

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    robust_std = 1.4826 * mad

    if robust_std < 1e-6:
        return float(
            np.percentile(values, 75)
        )

    threshold = median + robust_std

    threshold = max(
        threshold,
        np.percentile(values, 50)
    )

    threshold = min(
        threshold,
        np.percentile(values, 90)
    )

    return float(threshold)


def select_frames(
    frames,
    output_folder,
    output_csv
):
    """
    Adaptive, reconstruction-aware frame selection.

    The algorithm:

    1. Extracts ORB features and quality information once.
    2. Uses the first-pass measurements to understand the
       video's own geometric-change distribution.
    3. Selects frames by comparing each candidate with the
       LAST SELECTED frame.
    4. Uses no fixed frame percentage.
    5. Uses no fixed frame gap.
    """

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    orb = cv2.ORB_create(
    nfeatures=1000
    )

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    analysis = []

    print("Analyzing frames...")

    # =========================================================
    # PASS 1 — QUALITY + FEATURES
    # =========================================================

    for index, frame in enumerate(frames):

        image = frame["image"]

        (
            gray,
            keypoints,
            descriptors,
            blur_score,
            sharpness_score,
            feature_count,
            coverage,
            quality_score
        ) = _calculate_quality(
            image,
            orb
        )

        analysis.append({
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "image": image,
            "pose": frame.get("pose"),
            "intrinsics": frame.get("intrinsics"),
            "gray": gray,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "blur_score": blur_score,
            "sharpness_score": sharpness_score,
            "feature_count": feature_count,
            "coverage": coverage,
            "quality_score": quality_score
        })

        if (index + 1) % 50 == 0:
            print(
                f"  Analyzed {index + 1}/{len(frames)} frames"
            )

    if not analysis:
        return []

    # =========================================================
    # PASS 2 — BUILD GEOMETRIC CHANGE DISTRIBUTION
    #
    # Compare consecutive frames only to understand the
    # natural amount of motion/change in this video.
    # Features are already computed, so ORB is NOT repeated.
    # =========================================================

    print("Learning video motion distribution...")

    novelty_values = []
    parallax_values = []

    for i in range(1, len(analysis)):

        previous = analysis[i - 1]
        current = analysis[i]

        (
            redundancy,
            inliers,
            parallax
        ) = compare_features(
            previous["keypoints"],
            previous["descriptors"],
            current["keypoints"],
            current["descriptors"],
            previous["gray"].shape,
            matcher
        )

        if inliers >= 8:
            novelty_values.append(
                1.0 - redundancy
            )

            parallax_values.append(
                parallax
            )

    novelty_threshold = _adaptive_threshold(
        novelty_values
    )

    parallax_threshold = _adaptive_threshold(
        parallax_values
    )

    quality_values = [
        x["quality_score"]
        for x in analysis
    ]

    quality_threshold = float(
        np.percentile(
            quality_values,
            25
        )
    )

    # =========================================================
    # PASS 3 — ACTUAL SELECTION
    #
    # Compare each frame with the LAST SELECTED frame.
    # =========================================================

    print("Selecting reconstruction-useful frames...")

    selected_frames = []

    last_selected = None

    for i, item in enumerate(analysis):

        if i == 0:

            decision = "KEEP"

            redundancy = 0.0
            inliers = 0
            parallax = 0.0
            novelty = 1.0

        else:

            (
                redundancy,
                inliers,
                parallax
            ) = compare_features(
                last_selected["keypoints"],
                last_selected["descriptors"],
                item["keypoints"],
                item["descriptors"],
                last_selected["gray"].shape,
                matcher
            )

            novelty = 1.0 - redundancy

            quality_ok = (
                item["quality_score"]
                >= quality_threshold
            )

            geometric_change = (
                novelty >= novelty_threshold
                or parallax >= parallax_threshold
            )

            if quality_ok and geometric_change:
                decision = "KEEP"
            else:
                decision = "SKIP"

        item["geometric_redundancy"] = redundancy
        item["geometric_inliers"] = inliers
        item["parallax_score"] = parallax
        item["novelty_score"] = novelty
        item["decision"] = decision

        if decision == "KEEP":

            filename = (
                f"frame_{item['frame_id']:06d}.jpg"
            )

            image_path = os.path.join(
                output_folder,
                filename
            )

            cv2.imwrite(
                image_path,
                item["image"]
            )

            selected_frame = {
                "frame_id": item["frame_id"],
                "timestamp": item["timestamp"],
                "image_path": os.path.relpath(image_path),
                "image": item["image"]
                }

            if item.get("pose") is not None:
                selected_frame["pose"] = item["pose"]

            if item.get("intrinsics") is not None:
                selected_frame["intrinsics"] = item["intrinsics"]

            selected_frames.append(
                selected_frame
            )

            last_selected = item

    # =========================================================
    # CSV
    # =========================================================

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True
    )

    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame_id",
            "timestamp",
            "blur_score",
            "sharpness_score",
            "feature_count",
            "coverage",
            "quality_score",
            "novelty_score",
            "geometric_redundancy",
            "geometric_inliers",
            "parallax_score",
            "decision"
        ])

        for item in analysis:

            writer.writerow([
                item["frame_id"],
                item["timestamp"],
                item["blur_score"],
                item["sharpness_score"],
                item["feature_count"],
                item["coverage"],
                item["quality_score"],
                item["novelty_score"],
                item["geometric_redundancy"],
                item["geometric_inliers"],
                item["parallax_score"],
                item["decision"]
            ])

    print()
    print("Adaptive frame selection complete.")
    print(f"Input frames: {len(frames)}")
    print(f"Selected frames: {len(selected_frames)}")

    if len(frames) > 0:
        print(
            f"Selected percentage: "
            f"{len(selected_frames) / len(frames) * 100:.2f}%"
        )

    print(
        f"Adaptive novelty threshold: "
        f"{novelty_threshold:.4f}"
    )

    print(
        f"Adaptive parallax threshold: "
        f"{parallax_threshold:.4f}"
    )

    print(
        f"Adaptive quality threshold: "
        f"{quality_threshold:.4f}"
    )

    return selected_frames