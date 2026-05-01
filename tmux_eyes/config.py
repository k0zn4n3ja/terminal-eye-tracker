"""Tunable thresholds for the daemon.

Defaults come from the design doc (sol_arch.md §3 and §6). All values
overridable via environment variables prefixed with TMUX_EYES_.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"TMUX_EYES_{name}")
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"TMUX_EYES_{name}")
    return int(raw) if raw is not None else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"TMUX_EYES_{name}", default)


@dataclass(frozen=True)
class Config:
    # --- camera ---
    camera_device: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # --- vision ---
    # MediaPipe FaceLandmarker model. None → download default on first run.
    face_model_path: str = ""

    # --- classification ---
    yaw_left_deg: float = -12.0   # yaw < this → LEFT
    yaw_right_deg: float = 12.0   # yaw > this → RIGHT
    ema_alpha: float = 0.35       # smoothing factor (higher = more responsive)
    dwell_ms: int = 250           # must hold the same target this long to switch
    cooldown_ms: int = 250        # min gap between switches

    # --- iris confirmation (Phase 2) ---
    use_iris_confirmation: bool = False
    iris_left_ratio: float = 0.42
    iris_right_ratio: float = 0.58

    # --- multiplexer ---
    backend: str = "auto"  # "auto" | "tmux" | "wezterm"
    tmux_socket: str = ""  # empty = default (only used when backend == tmux)

    # --- runtime ---
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            camera_device=_env_int("CAMERA_DEVICE", 0),
            camera_width=_env_int("CAMERA_WIDTH", 640),
            camera_height=_env_int("CAMERA_HEIGHT", 480),
            camera_fps=_env_int("CAMERA_FPS", 30),
            face_model_path=_env_str("FACE_MODEL_PATH", ""),
            yaw_left_deg=_env_float("YAW_LEFT_DEG", -12.0),
            yaw_right_deg=_env_float("YAW_RIGHT_DEG", 12.0),
            ema_alpha=_env_float("EMA_ALPHA", 0.35),
            dwell_ms=_env_int("DWELL_MS", 250),
            cooldown_ms=_env_int("COOLDOWN_MS", 250),
            use_iris_confirmation=_env_str("USE_IRIS_CONFIRMATION", "0") == "1",
            iris_left_ratio=_env_float("IRIS_LEFT_RATIO", 0.42),
            iris_right_ratio=_env_float("IRIS_RIGHT_RATIO", 0.58),
            backend=_env_str("BACKEND", "auto"),
            tmux_socket=_env_str("TMUX_SOCKET", ""),
            log_level=_env_str("LOG_LEVEL", "INFO"),
        )
