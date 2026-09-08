import cv2
import numpy as np


def calculate_geometric_redundancy(
    image1,
    image2,
    orb,
    matcher,
    ratio_threshold=0.75
):
    """
    Compare two frames using ORB features and geometric consistency.

    Returns:
        geometric_redundancy
        inlier_count
        parallax_score
    """

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    return _compare_features(
        keypoints1,
        descriptors1,
        keypoints2,
        descriptors2,
        gray1.shape,
        matcher,
        ratio_threshold
    )


def compare_features(
    keypoints1,
    descriptors1,
    keypoints2,
    descriptors2,
    image_shape,
    matcher,
    ratio_threshold=0.75
):
    """
    Compare already-computed ORB features.

    This avoids recalculating ORB features repeatedly.
    """

    return _compare_features(
        keypoints1,
        descriptors1,
        keypoints2,
        descriptors2,
        image_shape,
        matcher,
        ratio_threshold
    )


def _compare_features(
    keypoints1,
    descriptors1,
    keypoints2,
    descriptors2,
    image_shape,
    matcher,
    ratio_threshold
):
    if descriptors1 is None or descriptors2 is None:
        return 1.0, 0, 0.0

    if len(descriptors1) < 8 or len(descriptors2) < 8:
        return 1.0, 0, 0.0

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return 1.0, len(good_matches), 0.0

    points1 = np.float32(
        [keypoints1[m.queryIdx].pt for m in good_matches]
    )

    points2 = np.float32(
        [keypoints2[m.trainIdx].pt for m in good_matches]
    )

    _, mask = cv2.findFundamentalMat(
        points1,
        points2,
        cv2.FM_RANSAC,
        1.0,
        0.99
    )

    if mask is None:
        return 1.0, 0, 0.0

    mask = mask.ravel().astype(bool)

    inlier_count = int(np.sum(mask))

    if inlier_count == 0:
        return 1.0, 0, 0.0

    inlier_points1 = points1[mask]
    inlier_points2 = points2[mask]

    displacements = np.linalg.norm(
        inlier_points2 - inlier_points1,
        axis=1
    )

    median_displacement = float(
        np.median(displacements)
    )

    height, width = image_shape

    image_diagonal = np.sqrt(
        width ** 2 + height ** 2
    )

    parallax_score = (
        median_displacement / image_diagonal
    )

    geometric_redundancy = (
        inlier_count / len(good_matches)
    )

    return (
        float(geometric_redundancy),
        inlier_count,
        float(parallax_score)
    )