"""Classifier module — fusion, EMA smoothing, and dwell state machine.

Pure state machine: no I/O, no time.time() calls.  All timestamps come
from the FaceSignal, making the classifier deterministically testable.
"""

from __future__ import annotations

from .config import Config
from .types import FaceSignal, GazeClass, PaneInfo, SwitchDecision

# Numeric encoding for EMA smoothing.
_GAZE_TO_NUM: dict[GazeClass, float] = {
    GazeClass.LEFT: -1.0,
    GazeClass.CENTER: 0.0,
    GazeClass.RIGHT: 1.0,
}

# Re-discretisation thresholds applied to the smoothed value.
_EMA_LEFT_THRESHOLD = -0.4
_EMA_RIGHT_THRESHOLD = 0.4


class Classifier:
    """Stateful per-session classifier.

    Call :meth:`update` once per incoming :class:`~tmux_eyes.types.FaceSignal`.
    Returns a :class:`~tmux_eyes.types.SwitchDecision` when the dwell timer
    expires, otherwise ``None``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config

        # EMA state.
        self._ema_value: float | None = None  # None until first valid sample

        # Dwell state.
        self._candidate_pane_id: str | None = None
        self._candidate_since_ms: int = 0
        self._last_switch_ms: int = -1_000_000  # far in the past

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        signal: FaceSignal,
        active_pane_id: str,
        panes: dict[str, PaneInfo],
    ) -> SwitchDecision | None:
        """Process one frame.

        Returns a :class:`SwitchDecision` when a switch should fire, else
        ``None``.  Called every frame.
        """
        # Need at least two panes to have anything to switch to.
        if len(panes) < 2:
            return None

        # Step 1: classify raw signal → GazeClass.
        raw_class = self._classify_raw(signal)

        # Step 2: EMA-smooth (UNKNOWN frames are skipped).
        smoothed_class = self._update_ema(raw_class)

        # Step 3 & 4: pane targeting + dwell state machine.
        return self._dwell_update(signal.timestamp_ms, smoothed_class, active_pane_id, panes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_raw(self, signal: FaceSignal) -> GazeClass:
        """Map a raw FaceSignal to a GazeClass using head-pose yaw and,
        optionally, iris-ratio confirmation."""
        if not signal.detected:
            return GazeClass.UNKNOWN

        cfg = self._cfg

        # Head-pose classification (primary signal).
        if signal.yaw_deg < cfg.yaw_left_deg:
            head_class = GazeClass.LEFT
        elif signal.yaw_deg > cfg.yaw_right_deg:
            head_class = GazeClass.RIGHT
        else:
            head_class = GazeClass.CENTER

        # Iris confirmation (Phase 2, opt-in).
        if cfg.use_iris_confirmation and signal.iris_ratio is not None:
            iris_class = self._classify_iris(signal.iris_ratio)
            # Suppress when head-pose and iris disagree *and* iris is not CENTER.
            if head_class != iris_class and iris_class != GazeClass.CENTER:
                return GazeClass.CENTER

        return head_class

    def _classify_iris(self, iris_ratio: float) -> GazeClass:
        cfg = self._cfg
        if iris_ratio < cfg.iris_left_ratio:
            return GazeClass.LEFT
        if iris_ratio > cfg.iris_right_ratio:
            return GazeClass.RIGHT
        return GazeClass.CENTER

    def _update_ema(self, gaze_class: GazeClass) -> GazeClass:
        """Apply EMA smoothing and return the re-discretised class.

        UNKNOWN frames do not update the EMA; they return UNKNOWN so the
        dwell machine can clear its candidate.
        """
        if gaze_class == GazeClass.UNKNOWN:
            return GazeClass.UNKNOWN

        x = _GAZE_TO_NUM[gaze_class]
        alpha = self._cfg.ema_alpha

        if self._ema_value is None:
            self._ema_value = x
        else:
            self._ema_value = alpha * x + (1.0 - alpha) * self._ema_value

        s = self._ema_value
        if s < _EMA_LEFT_THRESHOLD:
            return GazeClass.LEFT
        if s > _EMA_RIGHT_THRESHOLD:
            return GazeClass.RIGHT
        return GazeClass.CENTER

    def _identify_left_right_panes(
        self, panes: dict[str, PaneInfo]
    ) -> tuple[str, str]:
        """Return (left_pane_id, right_pane_id) by sorting on pane_left.

        Works for 2-pane layouts (Phase 1 MVP) but degrades gracefully for
        N-pane layouts by returning only the leftmost and rightmost panes.
        """
        sorted_panes = sorted(panes.values(), key=lambda p: p.left)
        return sorted_panes[0].pane_id, sorted_panes[-1].pane_id

    def _dwell_update(
        self,
        timestamp_ms: int,
        smoothed_class: GazeClass,
        active_pane_id: str,
        panes: dict[str, PaneInfo],
    ) -> SwitchDecision | None:
        cfg = self._cfg
        left_pane_id, right_pane_id = self._identify_left_right_panes(panes)

        # Map smoothed class → target pane id.
        if smoothed_class == GazeClass.LEFT:
            target_pane_id: str | None = left_pane_id
        elif smoothed_class == GazeClass.RIGHT:
            target_pane_id = right_pane_id
        else:
            # CENTER or UNKNOWN: no target.
            target_pane_id = None

        # Clear candidate when there is no directed gaze or gaze is at active pane.
        if target_pane_id is None or target_pane_id == active_pane_id:
            self._candidate_pane_id = None
            return None

        # New candidate: reset dwell timer.
        if self._candidate_pane_id != target_pane_id:
            self._candidate_pane_id = target_pane_id
            self._candidate_since_ms = timestamp_ms
            return None

        # Candidate is unchanged — check dwell + cooldown.
        dwell_elapsed = timestamp_ms - self._candidate_since_ms
        cooldown_elapsed = timestamp_ms - self._last_switch_ms

        if dwell_elapsed >= cfg.dwell_ms and cooldown_elapsed >= cfg.cooldown_ms:
            decision = SwitchDecision(
                target_pane_id=target_pane_id,
                reason=(
                    f"dwell={dwell_elapsed}ms >= {cfg.dwell_ms}ms, "
                    f"gaze={smoothed_class.value}"
                ),
            )
            self._last_switch_ms = timestamp_ms
            self._candidate_pane_id = None
            return decision

        return None
