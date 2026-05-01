"""Cross-platform webcam capture, yields Frames at the configured FPS.

Picks the lowest-latency cv2 backend per platform: V4L2 on Linux, AVFoundation
on macOS, DirectShow on Windows. Falls back to OpenCV's default backend if the
preferred one isn't available.

Use as an iterator:

    cam = Camera(device=0, width=640, height=480, fps=30)
    try:
        for frame in cam.frames():
            ...
    finally:
        cam.close()
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from typing import Optional

import cv2

from .types import Frame


class CameraError(RuntimeError):
    pass


def _preferred_backend() -> int:
    """Return the cv2 backend best suited to the current OS."""
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class Camera:
    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        backend = _preferred_backend()
        self._cap = cv2.VideoCapture(device, backend)
        if not self._cap.isOpened():
            # Fallback to OpenCV's default chooser.
            self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise CameraError(f"could not open camera device={device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # Small buffer reduces latency at the cost of dropping stale frames.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[Frame]:
        ok, pixels = self._cap.read()
        if not ok or pixels is None:
            return None
        return Frame(pixels=pixels, timestamp_ms=int(time.monotonic() * 1000))

    def frames(self) -> Iterator[Frame]:
        while True:
            f = self.read()
            if f is None:
                # Brief pause to avoid a hot loop if the camera hiccups.
                time.sleep(0.01)
                continue
            yield f

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
