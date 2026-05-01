"""Shared types and interface contracts for tmux-eyes.

Every other module in this package codes against these types.
Keep this file dependency-free (only stdlib + numpy typing) so
tests can import it without pulling in mediapipe/opencv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


class GazeClass(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Frame:
    """A single webcam frame, BGR uint8.

    `pixels` is a numpy ndarray of shape (H, W, 3); we keep it loosely typed
    so this module doesn't need to import numpy at the type level.
    """

    pixels: object  # np.ndarray
    timestamp_ms: int  # monotonic milliseconds


@dataclass(frozen=True)
class FaceSignal:
    """Per-frame output of the vision module.

    `yaw_deg` is the head-pose yaw in degrees: positive = looking right,
    negative = looking left, zero = facing camera. Range typically (-45, 45).

    `iris_ratio` is the average horizontal iris-to-eye-corner ratio across
    both eyes, in [0, 1]: lower = looking left, ~0.5 = center, higher = right.
    None when the face is not detected or iris landmarks aren't available.
    """

    timestamp_ms: int
    detected: bool
    yaw_deg: float = 0.0
    iris_ratio: Optional[float] = None


@dataclass(frozen=True)
class PaneInfo:
    """One tmux pane, geometry in terminal cells."""

    pane_id: str  # e.g. "%2"
    left: int
    top: int
    width: int
    height: int
    active: bool


@dataclass(frozen=True)
class SwitchDecision:
    """Emitted by the classifier when it wants to switch the focused pane."""

    target_pane_id: str
    reason: str  # human-readable, for logging


# --- Protocol interfaces (for static checking + dependency injection in tests) ---


class MultiplexerClientProto(Protocol):
    """Common interface for any terminal-multiplexer backend (tmux, wezterm, …)."""

    def get_panes(self) -> dict[str, PaneInfo]: ...
    def get_active_pane(self) -> str: ...
    def select_pane(self, pane_id: str) -> None: ...
    def close(self) -> None: ...


# Back-compat alias — older modules still import TmuxClientProto.
TmuxClientProto = MultiplexerClientProto
