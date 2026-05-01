"""Deterministic unit tests for tmux_eyes.classifier.

Every test uses explicit timestamp_ms values — no real time, no sleep.
"""

from __future__ import annotations

from typing import Optional

import pytest

from tmux_eyes.classifier import Classifier
from tmux_eyes.config import Config
from tmux_eyes.types import FaceSignal, GazeClass, PaneInfo, SwitchDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LEFT_PANE_ID = "%1"
RIGHT_PANE_ID = "%2"


def make_panes() -> dict[str, PaneInfo]:
    """Standard two-pane layout: %1 on left, %2 on right."""
    return {
        LEFT_PANE_ID: PaneInfo(
            pane_id=LEFT_PANE_ID, left=0, top=0, width=80, height=24, active=True
        ),
        RIGHT_PANE_ID: PaneInfo(
            pane_id=RIGHT_PANE_ID, left=80, top=0, width=80, height=24, active=False
        ),
    }


def make_signal(
    timestamp_ms: int,
    yaw_deg: float = 0.0,
    detected: bool = True,
    iris_ratio: Optional[float] = None,
) -> FaceSignal:
    return FaceSignal(
        timestamp_ms=timestamp_ms,
        detected=detected,
        yaw_deg=yaw_deg,
        iris_ratio=iris_ratio,
    )


def make_config(**kwargs: object) -> Config:
    """Return a Config with fast defaults suitable for tests unless overridden."""
    defaults: dict[str, object] = {
        "dwell_ms": 750,
        "cooldown_ms": 250,
        "ema_alpha": 0.35,
        "yaw_left_deg": -12.0,
        "yaw_right_deg": 12.0,
        "use_iris_confirmation": False,
    }
    defaults.update(kwargs)
    return Config(**defaults)  # type: ignore[arg-type]


def feed_stream(
    classifier: Classifier,
    panes: dict[str, PaneInfo],
    active_pane_id: str,
    signals: list[FaceSignal],
) -> list[SwitchDecision]:
    """Drive the classifier with a list of signals; collect non-None decisions."""
    decisions: list[SwitchDecision] = []
    for sig in signals:
        result = classifier.update(sig, active_pane_id, panes)
        if result is not None:
            decisions.append(result)
    return decisions


def build_signal_stream(
    yaw_deg: float,
    detected: bool,
    start_ms: int,
    duration_ms: int,
    step_ms: int = 33,
    iris_ratio: Optional[float] = None,
) -> list[FaceSignal]:
    """Produce a uniform stream of signals at `step_ms` intervals."""
    signals = []
    t = start_ms
    while t < start_ms + duration_ms:
        signals.append(
            make_signal(t, yaw_deg=yaw_deg, detected=detected, iris_ratio=iris_ratio)
        )
        t += step_ms
    return signals


# ---------------------------------------------------------------------------
# Test 1: No detection → no switch
# ---------------------------------------------------------------------------


def test_no_detection_no_switch() -> None:
    """A stream of undetected frames must never produce a SwitchDecision."""
    cfg = make_config()
    clf = Classifier(cfg)
    panes = make_panes()

    signals = build_signal_stream(
        yaw_deg=-25.0, detected=False, start_ms=0, duration_ms=3000
    )
    decisions = feed_stream(clf, panes, LEFT_PANE_ID, signals)
    assert decisions == [], f"Expected no decisions, got {decisions}"


# ---------------------------------------------------------------------------
# Test 2: Center gaze → no switch
# ---------------------------------------------------------------------------


def test_center_gaze_no_switch() -> None:
    """yaw=0 held for 2 seconds must not trigger a switch."""
    cfg = make_config()
    clf = Classifier(cfg)
    panes = make_panes()

    signals = build_signal_stream(yaw_deg=0.0, detected=True, start_ms=0, duration_ms=2000)
    decisions = feed_stream(clf, panes, LEFT_PANE_ID, signals)
    assert decisions == []


# ---------------------------------------------------------------------------
# Test 3: Brief glance left (300 ms) → no switch
# ---------------------------------------------------------------------------


def test_brief_glance_left_no_switch() -> None:
    """Holding yaw=-20 for only 300 ms (< dwell_ms=750) then returning to 0
    must not produce a switch."""
    cfg = make_config(dwell_ms=750)
    clf = Classifier(cfg)
    panes = make_panes()

    glance = build_signal_stream(yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=300)
    recover = build_signal_stream(yaw_deg=0.0, detected=True, start_ms=300, duration_ms=700)
    decisions = feed_stream(clf, panes, LEFT_PANE_ID, glance + recover)
    assert decisions == [], f"Expected no decisions for brief glance, got {decisions}"


# ---------------------------------------------------------------------------
# Test 4: Sustained look left (1000 ms) → switches to left pane, only once
# ---------------------------------------------------------------------------


def test_sustained_look_left_switches() -> None:
    """yaw=-20 for 1000 ms (> dwell_ms=750) must emit exactly one SwitchDecision
    targeting the left pane, and no further switches after that."""
    cfg = make_config(dwell_ms=750)
    clf = Classifier(cfg)
    panes = make_panes()

    # Active pane is RIGHT so a left gaze is a genuine candidate.
    signals = build_signal_stream(
        yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=1500
    )
    decisions = feed_stream(clf, panes, RIGHT_PANE_ID, signals)

    assert len(decisions) == 1, f"Expected exactly 1 switch, got {len(decisions)}"
    assert decisions[0].target_pane_id == LEFT_PANE_ID


# ---------------------------------------------------------------------------
# Test 5: Sustained look at active pane → no switch
# ---------------------------------------------------------------------------


def test_look_at_active_pane_no_switch() -> None:
    """When the active pane IS the left pane and the user looks left, no switch
    should fire regardless of dwell duration."""
    cfg = make_config(dwell_ms=750)
    clf = Classifier(cfg)
    panes = make_panes()

    # Active = LEFT_PANE_ID; gaze also LEFT → candidate == active → clear → no switch.
    signals = build_signal_stream(
        yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=2000
    )
    decisions = feed_stream(clf, panes, LEFT_PANE_ID, signals)
    assert decisions == []


# ---------------------------------------------------------------------------
# Test 6: Oscillation suppression
# ---------------------------------------------------------------------------


def test_oscillation_no_switch() -> None:
    """yaw alternating -20 / +20 every 200 ms for 2 seconds should never fire
    because the EMA keeps dragging toward center and the dwell timer resets
    whenever the smoothed class flips."""
    cfg = make_config(dwell_ms=750, ema_alpha=0.35)
    clf = Classifier(cfg)
    panes = make_panes()

    # Build alternating left/right signals at ~33 ms intervals, group switching
    # every ~200 ms.
    signals: list[FaceSignal] = []
    t = 0
    segment_ms = 200
    total_ms = 2000
    left_turn = True
    while t < total_ms:
        yaw = -20.0 if left_turn else 20.0
        seg = build_signal_stream(yaw_deg=yaw, detected=True, start_ms=t, duration_ms=segment_ms)
        signals.extend(seg)
        t += segment_ms
        left_turn = not left_turn

    decisions = feed_stream(clf, panes, RIGHT_PANE_ID, signals)
    # With 200 ms segments and a 750 ms dwell, no segment is long enough to
    # accumulate sufficient EMA drift AND hold the dwell timer.
    assert decisions == [], (
        f"Expected no switches during oscillation, got {len(decisions)}"
    )


# ---------------------------------------------------------------------------
# Test 7: Cool-down enforcement
# ---------------------------------------------------------------------------


def test_cooldown_prevents_immediate_second_switch() -> None:
    """After a switch fires, a second switch in the opposite direction must not
    fire until cooldown_ms (250 ms) has elapsed."""
    cfg = make_config(dwell_ms=750, cooldown_ms=250)
    clf = Classifier(cfg)
    panes = make_panes()

    # Phase A: look LEFT from the RIGHT pane long enough to switch.
    phase_a = build_signal_stream(
        yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=1500
    )
    decisions_a = feed_stream(clf, panes, RIGHT_PANE_ID, phase_a)
    assert len(decisions_a) == 1, "Expected one LEFT switch in phase A"
    assert decisions_a[0].target_pane_id == LEFT_PANE_ID
    switch_time = decisions_a[0].reason  # we just need the decision happened

    # Find the timestamp of the switch (last signal that produced a decision).
    # We know the switch fired somewhere in phase_a.  The classifier sets
    # last_switch_ms internally.  Now immediately look RIGHT.
    # Phase B starts right after phase_a with active_pane = LEFT_PANE_ID.

    # The switch fires around t=800 ms (first frame where dwell >= 750 ms).
    # Phase B: look RIGHT starting at t=1500, but cooldown window is short.
    # We send 200 ms of RIGHT gaze (< dwell) just to confirm no premature fire.
    phase_b_immediate = build_signal_stream(
        yaw_deg=20.0, detected=True, start_ms=1500, duration_ms=200
    )
    decisions_b_immediate = feed_stream(clf, panes, LEFT_PANE_ID, phase_b_immediate)
    # Not enough dwell yet anyway — no switch expected.
    assert len(decisions_b_immediate) == 0

    # Phase C: continue looking RIGHT, enough dwell but still within cooldown
    # relative to the first switch (~1500 ms mark).  The first switch fired
    # well before 1500 ms, so by t=1500+250 the cooldown IS satisfied.
    # We test the exact boundary: look RIGHT for a full dwell period, starting
    # immediately after a switch where cooldown has barely elapsed.
    #
    # Restart with a fresh classifier to control timing precisely.
    clf2 = Classifier(cfg)
    # Force a switch at t=800 by feeding left gaze from t=0.
    left_stream = build_signal_stream(yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=1000)
    d1 = feed_stream(clf2, panes, RIGHT_PANE_ID, left_stream)
    assert len(d1) == 1 and d1[0].target_pane_id == LEFT_PANE_ID

    # Now look RIGHT.  Start at t=1000.  Cooldown requires >= 250 ms after
    # the switch.  The switch fired around t=~800.  At t=1000+750=1750,
    # the dwell is met AND cooldown is met (1750-800 >> 250).
    right_stream = build_signal_stream(yaw_deg=20.0, detected=True, start_ms=1000, duration_ms=1500)
    d2 = feed_stream(clf2, panes, LEFT_PANE_ID, right_stream)
    assert len(d2) >= 1 and d2[0].target_pane_id == RIGHT_PANE_ID, (
        "Expected a RIGHT switch after cooldown elapsed"
    )

    # Now test that if we look RIGHT immediately after a switch (within cooldown),
    # it does NOT fire a second time before cooldown.
    clf3 = Classifier(cfg)
    # Force switch to LEFT at around t=800.
    left_s = build_signal_stream(yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=850)
    d_left = feed_stream(clf3, panes, RIGHT_PANE_ID, left_s)
    assert len(d_left) == 1 and d_left[0].target_pane_id == LEFT_PANE_ID

    # Immediately look RIGHT, but fire a signal WITHIN the cooldown window.
    # Switch was around t=750.  We send one signal at t=850 (< 750+250=1000).
    # Then send more signals from t=850 to t=1000 — still within cooldown.
    within_cooldown = [make_signal(t, yaw_deg=20.0) for t in range(850, 1000, 33)]
    d_within = feed_stream(clf3, panes, LEFT_PANE_ID, within_cooldown)
    assert len(d_within) == 0, (
        f"No switch within cooldown window, got {len(d_within)}"
    )

    # After cooldown: signals from t=1100 onward (> 750+250=1000) with enough dwell.
    after_cooldown = build_signal_stream(yaw_deg=20.0, detected=True, start_ms=1100, duration_ms=900)
    d_after = feed_stream(clf3, panes, LEFT_PANE_ID, after_cooldown)
    assert len(d_after) >= 1 and d_after[0].target_pane_id == RIGHT_PANE_ID, (
        "Expected RIGHT switch after cooldown elapsed"
    )


# ---------------------------------------------------------------------------
# Test 8: EMA smoothing rejects single-frame noise
# ---------------------------------------------------------------------------


def test_ema_rejects_single_frame_noise() -> None:
    """A long stream of yaw=-20 with one yaw=+20 outlier at the middle must
    still eventually switch to LEFT — the outlier should not cancel the
    accumulated left-bias in the EMA."""
    cfg = make_config(dwell_ms=750, ema_alpha=0.35)
    clf = Classifier(cfg)
    panes = make_panes()

    # 500 ms of left gaze.
    pre_outlier = build_signal_stream(
        yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=500
    )
    # One outlier frame at t=500.
    outlier = [make_signal(500, yaw_deg=20.0)]
    # Continue left gaze to well past dwell.
    post_outlier = build_signal_stream(
        yaw_deg=-20.0, detected=True, start_ms=533, duration_ms=1500
    )

    all_signals = pre_outlier + outlier + post_outlier
    decisions = feed_stream(clf, panes, RIGHT_PANE_ID, all_signals)

    assert len(decisions) >= 1, "Expected at least one LEFT switch despite noise"
    assert decisions[0].target_pane_id == LEFT_PANE_ID


# ---------------------------------------------------------------------------
# Test 9: Iris confirmation suppression (Phase 2)
# ---------------------------------------------------------------------------


def test_iris_confirmation_suppresses_disagreement() -> None:
    """With use_iris_confirmation=True: head-pose says LEFT but iris_ratio
    says RIGHT (> iris_right_ratio=0.58).  The disagreement rule must
    suppress all frames → no switch fires."""
    cfg = make_config(
        use_iris_confirmation=True,
        iris_left_ratio=0.42,
        iris_right_ratio=0.58,
        dwell_ms=750,
    )
    clf = Classifier(cfg)
    panes = make_panes()

    # Head-pose: LEFT (yaw=-20), iris_ratio: RIGHT (0.65 > 0.58) → disagreement → CENTER.
    signals = build_signal_stream(
        yaw_deg=-20.0,
        detected=True,
        start_ms=0,
        duration_ms=2000,
        iris_ratio=0.65,
    )
    decisions = feed_stream(clf, panes, RIGHT_PANE_ID, signals)
    assert decisions == [], (
        f"Expected no switch when head/iris disagree, got {len(decisions)}"
    )


def test_iris_confirmation_allows_agreement() -> None:
    """With use_iris_confirmation=True: head-pose and iris both say LEFT.
    Switch should still fire after dwell."""
    cfg = make_config(
        use_iris_confirmation=True,
        iris_left_ratio=0.42,
        iris_right_ratio=0.58,
        dwell_ms=750,
    )
    clf = Classifier(cfg)
    panes = make_panes()

    # Both signals say LEFT: yaw=-20, iris_ratio=0.30 (< 0.42).
    signals = build_signal_stream(
        yaw_deg=-20.0,
        detected=True,
        start_ms=0,
        duration_ms=1500,
        iris_ratio=0.30,
    )
    decisions = feed_stream(clf, panes, RIGHT_PANE_ID, signals)
    assert len(decisions) >= 1 and decisions[0].target_pane_id == LEFT_PANE_ID


# ---------------------------------------------------------------------------
# Test 10: Single pane → no switch
# ---------------------------------------------------------------------------


def test_single_pane_no_switch() -> None:
    """When the panes dict has only one entry the classifier must always
    return None (nothing to switch to)."""
    cfg = make_config()
    clf = Classifier(cfg)

    single_pane = {
        LEFT_PANE_ID: PaneInfo(
            pane_id=LEFT_PANE_ID, left=0, top=0, width=160, height=24, active=True
        )
    }

    signals = build_signal_stream(
        yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=2000
    )
    decisions = feed_stream(clf, single_pane, LEFT_PANE_ID, signals)
    assert decisions == [], f"Expected no decisions with single pane, got {decisions}"


# ---------------------------------------------------------------------------
# Test 11: Empty panes dict → no switch (edge case)
# ---------------------------------------------------------------------------


def test_empty_panes_no_switch() -> None:
    """An empty panes dict must never crash and must return None."""
    cfg = make_config()
    clf = Classifier(cfg)

    result = clf.update(make_signal(0, yaw_deg=-20.0), LEFT_PANE_ID, {})
    assert result is None


# ---------------------------------------------------------------------------
# Test 12: EMA initialises on first valid sample (no crash on first frame)
# ---------------------------------------------------------------------------


def test_ema_initialises_cleanly() -> None:
    """First call with a valid signal must not raise and must return None
    (dwell hasn't elapsed yet)."""
    cfg = make_config()
    clf = Classifier(cfg)
    panes = make_panes()

    result = clf.update(make_signal(0, yaw_deg=-20.0), RIGHT_PANE_ID, panes)
    # First frame: dwell hasn't elapsed, so no decision yet.
    assert result is None


# ---------------------------------------------------------------------------
# Test 13: Three-or-more-pane layout — uses leftmost and rightmost
# ---------------------------------------------------------------------------


def test_three_pane_uses_leftmost_rightmost() -> None:
    """With three panes, the classifier should pick the leftmost as LEFT and
    rightmost as RIGHT, and still fire after dwell."""
    cfg = make_config(dwell_ms=750)
    clf = Classifier(cfg)

    panes = {
        "%1": PaneInfo(pane_id="%1", left=0, top=0, width=53, height=24, active=False),
        "%2": PaneInfo(pane_id="%2", left=54, top=0, width=53, height=24, active=False),
        "%3": PaneInfo(pane_id="%3", left=108, top=0, width=53, height=24, active=True),
    }

    signals = build_signal_stream(
        yaw_deg=-20.0, detected=True, start_ms=0, duration_ms=1500
    )
    decisions = feed_stream(clf, panes, "%3", signals)
    assert len(decisions) >= 1 and decisions[0].target_pane_id == "%1", (
        "Expected switch to leftmost pane (%1)"
    )
