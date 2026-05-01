"""Unit tests for the pure math helpers in tmux_eyes.vision.

These tests do NOT instantiate FaceTracker — that requires MediaPipe and a
real model file.  Only ``compute_head_yaw`` and ``compute_iris_ratio`` are
exercised here; both are module-level pure functions with no external deps
beyond numpy and opencv-python-headless.
"""

from __future__ import annotations

import numpy as np

from tmux_eyes.vision import compute_head_yaw, compute_iris_ratio


# ===========================================================================
# compute_iris_ratio
# ===========================================================================


class TestComputeIrisRatio:
    """Iris ratio: 0 = full left, 0.5 = centre, 1 = full right."""

    def test_centered_both_eyes(self) -> None:
        """Iris at midpoint of each eye → ratio ≈ 0.5."""
        ratio = compute_iris_ratio(
            left_iris_x=50.0,
            left_inner_x=0.0,
            left_outer_x=100.0,
            right_iris_x=50.0,
            right_inner_x=0.0,
            right_outer_x=100.0,
        )
        assert abs(ratio - 0.5) < 0.01

    def test_iris_near_inner_corner_both_eyes(self) -> None:
        """Iris close to inner corner → ratio < 0.42 (looking left)."""
        ratio = compute_iris_ratio(
            left_iris_x=25.0,
            left_inner_x=0.0,
            left_outer_x=100.0,
            right_iris_x=25.0,
            right_inner_x=0.0,
            right_outer_x=100.0,
        )
        assert ratio < 0.42

    def test_iris_near_outer_corner_both_eyes(self) -> None:
        """Iris close to outer corner → ratio > 0.58 (looking right)."""
        ratio = compute_iris_ratio(
            left_iris_x=75.0,
            left_inner_x=0.0,
            left_outer_x=100.0,
            right_iris_x=75.0,
            right_inner_x=0.0,
            right_outer_x=100.0,
        )
        assert ratio > 0.58

    def test_asymmetric_left_looking_left_right_looking_center(self) -> None:
        """Left eye looks left (0.25), right eye looks centre (0.5) → avg ≈ 0.375."""
        ratio = compute_iris_ratio(
            left_iris_x=25.0,
            left_inner_x=0.0,
            left_outer_x=100.0,
            right_iris_x=50.0,
            right_inner_x=0.0,
            right_outer_x=100.0,
        )
        expected = (0.25 + 0.50) / 2.0  # = 0.375
        assert abs(ratio - expected) < 0.001

    def test_normalized_coordinates(self) -> None:
        """Function should give the same ratio for pixel and normalised coords."""
        ratio_pixels = compute_iris_ratio(
            left_iris_x=50.0,
            left_inner_x=0.0,
            left_outer_x=100.0,
            right_iris_x=50.0,
            right_inner_x=0.0,
            right_outer_x=100.0,
        )
        # Same proportions in [0, 1] normalised space
        ratio_norm = compute_iris_ratio(
            left_iris_x=0.5,
            left_inner_x=0.0,
            left_outer_x=1.0,
            right_iris_x=0.5,
            right_inner_x=0.0,
            right_outer_x=1.0,
        )
        assert abs(ratio_pixels - ratio_norm) < 1e-9

    def test_degenerate_span_returns_half(self) -> None:
        """When inner == outer (zero span), fall back to 0.5 for that eye."""
        ratio = compute_iris_ratio(
            left_iris_x=10.0,
            left_inner_x=10.0,
            left_outer_x=10.0,  # zero span
            right_iris_x=50.0,
            right_inner_x=0.0,
            right_outer_x=100.0,
        )
        # Left eye → 0.5 fallback; right eye → 0.5; average = 0.5
        assert abs(ratio - 0.5) < 0.001


# ===========================================================================
# compute_head_yaw
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper: build a synthetic 6-point landmark array from a known 3-D face pose.
#
# Strategy: project the canonical 3-D model points through a known camera
# matrix and rotation, then recover yaw with compute_head_yaw.  This means
# the test is self-consistent: we encode a rotation, project it, decode it,
# and check the sign + magnitude.
# ---------------------------------------------------------------------------

_IMAGE_W = 640
_IMAGE_H = 480
_IMAGE_SIZE = (_IMAGE_W, _IMAGE_H)

# Same 3-D model used in vision.py — duplicated here to keep tests self-contained.
_MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ],
    dtype=np.float64,
)


def _project_landmarks(yaw_deg: float) -> np.ndarray:
    """Project the 3-D model with a pure yaw rotation onto a 640×480 image.

    Returns a (6, 2) float64 array of pixel coordinates.
    """
    import cv2

    focal = float(_IMAGE_W)
    cx, cy = _IMAGE_W / 2.0, _IMAGE_H / 2.0
    camera_matrix = np.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Build rotation matrix for the requested yaw around the Y axis.
    yaw_rad = np.deg2rad(yaw_deg)
    Ry = np.array(
        [
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(Ry)

    # Translate the face 500 mm along Z so all points project in front of camera.
    tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)

    pts_2d, _ = cv2.projectPoints(
        _MODEL_POINTS_3D, rvec, tvec, camera_matrix, dist_coeffs
    )
    return pts_2d.reshape(6, 2)


class TestComputeHeadYaw:
    """Head-pose yaw: positive = user's right, negative = user's left."""

    def test_symmetric_face_yaw_near_zero(self) -> None:
        """Symmetric (0° yaw) projection → recovered |yaw| < 1°."""
        landmarks = _project_landmarks(0.0)
        yaw = compute_head_yaw(landmarks, _IMAGE_SIZE)
        assert abs(yaw) < 1.0, f"expected |yaw| < 1°, got {yaw:.2f}°"

    def test_face_rotated_right_positive_yaw(self) -> None:
        """15° rightward rotation → recovered yaw > +5°."""
        landmarks = _project_landmarks(15.0)
        yaw = compute_head_yaw(landmarks, _IMAGE_SIZE)
        assert yaw > 5.0, f"expected yaw > 5°, got {yaw:.2f}°"

    def test_face_rotated_left_negative_yaw(self) -> None:
        """−15° leftward rotation → recovered yaw < −5°."""
        landmarks = _project_landmarks(-15.0)
        yaw = compute_head_yaw(landmarks, _IMAGE_SIZE)
        assert yaw < -5.0, f"expected yaw < -5°, got {yaw:.2f}°"

    def test_right_larger_than_left_magnitude(self) -> None:
        """Symmetry: |yaw(+15°)| ≈ |yaw(-15°)| within 5°."""
        yaw_r = compute_head_yaw(_project_landmarks(15.0), _IMAGE_SIZE)
        yaw_l = compute_head_yaw(_project_landmarks(-15.0), _IMAGE_SIZE)
        assert abs(abs(yaw_r) - abs(yaw_l)) < 5.0, (
            f"asymmetric magnitudes: right={yaw_r:.2f}°, left={yaw_l:.2f}°"
        )

    def test_sign_convention_right_is_positive(self) -> None:
        """Positive yaw = user looking right; negative = looking left."""
        yaw_r = compute_head_yaw(_project_landmarks(20.0), _IMAGE_SIZE)
        yaw_l = compute_head_yaw(_project_landmarks(-20.0), _IMAGE_SIZE)
        assert yaw_r > 0, f"right gaze should give positive yaw, got {yaw_r:.2f}°"
        assert yaw_l < 0, f"left gaze should give negative yaw, got {yaw_l:.2f}°"
