"""Vision module: webcam frame → FaceSignal.

Uses the MediaPipe Tasks API (``FaceLandmarker``) in IMAGE mode (synchronous).
IMAGE mode is far simpler than LIVE_STREAM and adequate at 30 FPS for the POC.

The legacy ``mp.solutions.face_mesh`` API was removed in current MediaPipe
versions, so we always run the Tasks API. When ``model_path`` is empty, the
canonical ``face_landmarker.task`` model is downloaded from the MediaPipe CDN
to ``$XDG_CACHE_HOME/tmux-eyes/`` (default ``~/.cache/tmux-eyes/``) on first run.

The pure math helpers ``compute_head_yaw`` and ``compute_iris_ratio`` are
module-level functions so they can be unit-tested without loading MediaPipe.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from tmux_eyes.types import FaceSignal, Frame

logger = logging.getLogger(__name__)

# Canonical face landmarker model (478 landmarks incl. iris). Float16, ~3.7 MB.
# Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_FILENAME = "face_landmarker.task"


def _default_model_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tmux-eyes"


def _ensure_face_landmarker_model(target_dir: Optional[Path] = None) -> Path:
    """Return a path to face_landmarker.task, downloading on first use."""
    cache_dir = target_dir or _default_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _FACE_LANDMARKER_FILENAME
    if target.exists() and target.stat().st_size > 0:
        return target

    logger.info("downloading MediaPipe face_landmarker model → %s", target)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(_FACE_LANDMARKER_URL, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        tmp.rename(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target

# ---------------------------------------------------------------------------
# Generic 3-D face model (millimetres, origin at nose tip).
# Landmark order matches the 6-point selection used throughout this module:
#   0 – nose tip (1)
#   1 – chin     (199)
#   2 – left eye outer corner  (33)
#   3 – right eye outer corner (263)
#   4 – left mouth corner  (61)
#   5 – right mouth corner (291)
# Values from a canonical OpenCV head-pose tutorial / dlib 68-point model.
# ---------------------------------------------------------------------------
_MODEL_POINTS_3D: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],          # nose tip
        [0.0, -63.6, -12.5],      # chin
        [-43.3, 32.7, -26.0],     # left eye outer corner
        [43.3, 32.7, -26.0],      # right eye outer corner
        [-28.9, -28.9, -24.1],    # left mouth corner
        [28.9, -28.9, -24.1],     # right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe 478-landmark indices for the 6 head-pose points.
_POSE_LM_INDICES = [1, 199, 33, 263, 61, 291]

# Iris and eye-corner landmark indices.
_LEFT_IRIS_IDX = 468
_RIGHT_IRIS_IDX = 473
_LEFT_EYE_INNER = 133   # inner corner (medial)
_LEFT_EYE_OUTER = 33    # outer corner (lateral)
_RIGHT_EYE_INNER = 362  # inner corner (medial)
_RIGHT_EYE_OUTER = 263  # outer corner (lateral)


# ---------------------------------------------------------------------------
# Pure math helpers — no MediaPipe dependency, fully unit-testable.
# ---------------------------------------------------------------------------

def compute_head_yaw(
    landmarks_2d: np.ndarray,
    image_size: tuple[int, int],
) -> float:
    """Estimate head-pose yaw from 6 facial landmarks via solvePnP.

    Parameters
    ----------
    landmarks_2d:
        Shape (6, 2) array of pixel coordinates in the order:
        nose tip, chin, left-eye-outer, right-eye-outer,
        left-mouth, right-mouth.
    image_size:
        (width, height) of the source frame in pixels.

    Returns
    -------
    float
        Yaw in degrees.  Positive = user looking to *their* right (camera
        left); negative = user looking to their left (camera right).
    """
    w, h = image_size
    focal = w  # reasonable approximation: focal length ≈ image width
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _tvec = cv2.solvePnP(
        _MODEL_POINTS_3D,
        landmarks_2d.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    # RQDecomp3x3 returns Euler angles in degrees: (pitch, yaw, roll)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    yaw_deg = float(angles[1])
    return yaw_deg


def compute_iris_ratio(
    left_iris_x: float,
    left_inner_x: float,
    left_outer_x: float,
    right_iris_x: float,
    right_inner_x: float,
    right_outer_x: float,
) -> float:
    """Horizontal iris-to-eye-corner ratio, averaged across both eyes.

    For each eye:  ratio = (iris_x - inner_x) / (outer_x - inner_x)

    Works in any consistent unit (pixels or normalized [0,1]) because the
    result is a pure ratio.  Returns 0.5 if either eye span is degenerate.

    Lower values (~0.0) → gaze to user's left; higher (~1.0) → to the right.
    """
    def _ratio(iris: float, inner: float, outer: float) -> float:
        span = outer - inner
        if abs(span) < 1e-6:
            return 0.5
        return (iris - inner) / span

    left = _ratio(left_iris_x, left_inner_x, left_outer_x)
    right = _ratio(right_iris_x, right_inner_x, right_outer_x)
    return (left + right) / 2.0


# ---------------------------------------------------------------------------
# FaceTracker class
# ---------------------------------------------------------------------------

class FaceTracker:
    """Wraps MediaPipe FaceLandmarker (Tasks API) and produces a FaceSignal.

    If ``model_path`` is empty the canonical ``face_landmarker.task`` model is
    fetched from the MediaPipe CDN and cached under
    ``$XDG_CACHE_HOME/tmux-eyes/`` on first run. Iris landmarks
    (indices 468–477) are always available since the canonical model includes
    them.
    """

    def __init__(self, model_path: str = "") -> None:
        self._landmarker = None
        path = Path(model_path) if model_path else _ensure_face_landmarker_model()
        self._init_tasks_api(str(path))

    def _init_tasks_api(self, model_path: str) -> None:
        import mediapipe as mp  # type: ignore[import]

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        logger.debug("FaceTracker: Tasks API initialised from %s", model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: Frame) -> FaceSignal:
        """Run landmark detection on one frame and return a FaceSignal."""
        pixels: np.ndarray = frame.pixels  # type: ignore[assignment]
        h, w = pixels.shape[:2]

        landmarks = self._detect_tasks(pixels)

        if landmarks is None:
            return FaceSignal(
                timestamp_ms=frame.timestamp_ms,
                detected=False,
                yaw_deg=0.0,
                iris_ratio=None,
            )

        # landmarks is a list of (x_norm, y_norm) tuples, length 468 or 478.
        has_iris = len(landmarks) >= 478

        # --- Head-pose yaw ---
        pose_pts = np.array(
            [[landmarks[i][0] * w, landmarks[i][1] * h] for i in _POSE_LM_INDICES],
            dtype=np.float64,
        )
        yaw = compute_head_yaw(pose_pts, (w, h))

        # --- Iris ratio ---
        iris_ratio: Optional[float] = None
        if has_iris:
            try:
                iris_ratio = compute_iris_ratio(
                    left_iris_x=landmarks[_LEFT_IRIS_IDX][0] * w,
                    left_inner_x=landmarks[_LEFT_EYE_INNER][0] * w,
                    left_outer_x=landmarks[_LEFT_EYE_OUTER][0] * w,
                    right_iris_x=landmarks[_RIGHT_IRIS_IDX][0] * w,
                    right_inner_x=landmarks[_RIGHT_EYE_INNER][0] * w,
                    right_outer_x=landmarks[_RIGHT_EYE_OUTER][0] * w,
                )
            except Exception:  # pragma: no cover
                logger.debug("iris ratio computation failed", exc_info=True)

        return FaceSignal(
            timestamp_ms=frame.timestamp_ms,
            detected=True,
            yaw_deg=yaw,
            iris_ratio=iris_ratio,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _detect_tasks(
        self, pixels: np.ndarray
    ) -> Optional[list[tuple[float, float]]]:
        import mediapipe as mp  # type: ignore[import]

        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        return [(lm.x, lm.y) for lm in lms]
