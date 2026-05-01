//! Deterministic unit tests for the Classifier state machine.
//!
//! Every test uses explicit timestamp_ms values — no real time, no sleep.

use std::collections::HashMap;

use tmux_eyes::classifier::Classifier;
use tmux_eyes::config::Config;
use tmux_eyes::types::{FaceSignal, PaneInfo, SwitchDecision};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const LEFT_PANE_ID: &str = "%1";
const RIGHT_PANE_ID: &str = "%2";

fn panes_two() -> HashMap<String, PaneInfo> {
    let mut m = HashMap::new();
    m.insert(
        LEFT_PANE_ID.to_owned(),
        PaneInfo {
            pane_id: LEFT_PANE_ID.to_owned(),
            left: 0,
            top: 0,
            width: 80,
            height: 24,
            active: true,
        },
    );
    m.insert(
        RIGHT_PANE_ID.to_owned(),
        PaneInfo {
            pane_id: RIGHT_PANE_ID.to_owned(),
            left: 80,
            top: 0,
            width: 80,
            height: 24,
            active: false,
        },
    );
    m
}

fn make_signal(timestamp_ms: u64, yaw_deg: f32, detected: bool, iris_ratio: Option<f32>) -> FaceSignal {
    FaceSignal { timestamp_ms, detected, yaw_deg, iris_ratio }
}

fn make_config_defaults() -> Config {
    Config {
        dwell_ms: 750,
        cooldown_ms: 250,
        ema_alpha: 0.35,
        yaw_left_deg: -12.0,
        yaw_right_deg: 12.0,
        use_iris_confirmation: false,
        ..Config::default()
    }
}

/// Build a uniform stream of signals at `step_ms` intervals.
fn build_signal_stream(
    yaw_deg: f32,
    detected: bool,
    start_ms: u64,
    duration_ms: u64,
    step_ms: u64,
    iris_ratio: Option<f32>,
) -> Vec<FaceSignal> {
    let mut signals = Vec::new();
    let mut t = start_ms;
    while t < start_ms + duration_ms {
        signals.push(make_signal(t, yaw_deg, detected, iris_ratio));
        t += step_ms;
    }
    signals
}

/// Drive the classifier with a slice of signals; collect non-None decisions.
fn feed_stream(
    clf: &mut Classifier,
    panes: &HashMap<String, PaneInfo>,
    active_pane_id: &str,
    signals: &[FaceSignal],
) -> Vec<SwitchDecision> {
    signals
        .iter()
        .filter_map(|sig| clf.update(*sig, active_pane_id, panes))
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: No detection → no switch
// ---------------------------------------------------------------------------

#[test]
fn test_no_detection_no_switch() {
    let mut clf = Classifier::new(make_config_defaults());
    let panes = panes_two();

    let signals = build_signal_stream(-25.0, false, 0, 3000, 33, None);
    let decisions = feed_stream(&mut clf, &panes, LEFT_PANE_ID, &signals);
    assert!(decisions.is_empty(), "Expected no decisions, got {:?}", decisions);
}

// ---------------------------------------------------------------------------
// Test 2: Center gaze (yaw=0) for 2 seconds → no switch
// ---------------------------------------------------------------------------

#[test]
fn test_center_gaze_no_switch() {
    let mut clf = Classifier::new(make_config_defaults());
    let panes = panes_two();

    let signals = build_signal_stream(0.0, true, 0, 2000, 33, None);
    let decisions = feed_stream(&mut clf, &panes, LEFT_PANE_ID, &signals);
    assert!(decisions.is_empty(), "Expected no decisions for center gaze, got {:?}", decisions);
}

// ---------------------------------------------------------------------------
// Test 3: Brief glance left (300 ms) → no switch
// ---------------------------------------------------------------------------

#[test]
fn test_brief_glance_left_no_switch() {
    let cfg = Config { dwell_ms: 750, ..make_config_defaults() };
    let mut clf = Classifier::new(cfg);
    let panes = panes_two();

    let mut signals = build_signal_stream(-20.0, true, 0, 300, 33, None);
    signals.extend(build_signal_stream(0.0, true, 300, 700, 33, None));

    // Active is %2 (right) so left is a genuine candidate.
    let decisions = feed_stream(&mut clf, &panes, RIGHT_PANE_ID, &signals);
    assert!(decisions.is_empty(), "Expected no decisions for brief glance, got {:?}", decisions);
}

// ---------------------------------------------------------------------------
// Test 4: Sustained look left (1000 ms) → exactly one switch to %1
// ---------------------------------------------------------------------------

#[test]
fn test_sustained_look_left_switches() {
    let cfg = Config { dwell_ms: 750, ..make_config_defaults() };
    let mut clf = Classifier::new(cfg);
    let panes = panes_two();

    let signals = build_signal_stream(-20.0, true, 0, 1500, 33, None);
    let decisions = feed_stream(&mut clf, &panes, RIGHT_PANE_ID, &signals);

    assert_eq!(decisions.len(), 1, "Expected exactly 1 switch, got {}", decisions.len());
    assert_eq!(decisions[0].target_pane_id, LEFT_PANE_ID);
}

// ---------------------------------------------------------------------------
// Test 5: Sustained look at active pane → no switch
// ---------------------------------------------------------------------------

#[test]
fn test_look_at_active_pane_no_switch() {
    let cfg = Config { dwell_ms: 750, ..make_config_defaults() };
    let mut clf = Classifier::new(cfg);
    let panes = panes_two();

    // Active = LEFT_PANE_ID; gaze also LEFT → candidate == active → clear → no switch.
    let signals = build_signal_stream(-20.0, true, 0, 2000, 33, None);
    let decisions = feed_stream(&mut clf, &panes, LEFT_PANE_ID, &signals);
    assert!(decisions.is_empty(), "Expected no switch when gazing at active pane, got {:?}", decisions);
}

// ---------------------------------------------------------------------------
// Test 6: Oscillation suppression
// ---------------------------------------------------------------------------

#[test]
fn test_oscillation_no_switch() {
    let cfg = Config { dwell_ms: 750, ema_alpha: 0.35, ..make_config_defaults() };
    let mut clf = Classifier::new(cfg);
    let panes = panes_two();

    // Alternating left/right every 200 ms for 2 seconds.
    let mut signals: Vec<FaceSignal> = Vec::new();
    let mut t = 0u64;
    let segment_ms = 200u64;
    let total_ms = 2000u64;
    let mut left_turn = true;
    while t < total_ms {
        let yaw = if left_turn { -20.0 } else { 20.0 };
        signals.extend(build_signal_stream(yaw, true, t, segment_ms, 33, None));
        t += segment_ms;
        left_turn = !left_turn;
    }

    let decisions = feed_stream(&mut clf, &panes, RIGHT_PANE_ID, &signals);
    assert!(
        decisions.is_empty(),
        "Expected no switches during oscillation, got {}",
        decisions.len()
    );
}

// ---------------------------------------------------------------------------
// Test 7: Cooldown enforcement
// ---------------------------------------------------------------------------

#[test]
fn test_cooldown_prevents_immediate_second_switch() {
    let cfg = Config { dwell_ms: 750, cooldown_ms: 250, ..make_config_defaults() };
    let panes = panes_two();

    // Phase A: look LEFT from the RIGHT pane long enough to switch.
    let mut clf = Classifier::new(cfg.clone());
    let phase_a = build_signal_stream(-20.0, true, 0, 1500, 33, None);
    let decisions_a = feed_stream(&mut clf, &panes, RIGHT_PANE_ID, &phase_a);
    assert_eq!(decisions_a.len(), 1, "Expected one LEFT switch in phase A");
    assert_eq!(decisions_a[0].target_pane_id, LEFT_PANE_ID);

    // Phase B: look RIGHT immediately, but not enough dwell yet.
    let phase_b = build_signal_stream(20.0, true, 1500, 200, 33, None);
    let decisions_b = feed_stream(&mut clf, &panes, LEFT_PANE_ID, &phase_b);
    assert_eq!(decisions_b.len(), 0, "Not enough dwell — no switch expected");

    // --- Fresh classifier for precise cooldown boundary test ---

    // Force a switch at ~t=800 by feeding left gaze from t=0.
    let mut clf2 = Classifier::new(cfg.clone());
    let left_stream = build_signal_stream(-20.0, true, 0, 1000, 33, None);
    let d1 = feed_stream(&mut clf2, &panes, RIGHT_PANE_ID, &left_stream);
    assert_eq!(d1.len(), 1);
    assert_eq!(d1[0].target_pane_id, LEFT_PANE_ID);

    // Now look RIGHT. The switch fired around t=~800. At t=1000+750=1750,
    // both dwell and cooldown are satisfied.
    let right_stream = build_signal_stream(20.0, true, 1000, 1500, 33, None);
    let d2 = feed_stream(&mut clf2, &panes, LEFT_PANE_ID, &right_stream);
    assert!(d2.len() >= 1, "Expected a RIGHT switch after cooldown elapsed");
    assert_eq!(d2[0].target_pane_id, RIGHT_PANE_ID);

    // --- Test that signals within cooldown do NOT fire ---

    let mut clf3 = Classifier::new(cfg.clone());
    // Force switch to LEFT at around t=750.
    let left_s = build_signal_stream(-20.0, true, 0, 850, 33, None);
    let d_left = feed_stream(&mut clf3, &panes, RIGHT_PANE_ID, &left_s);
    assert_eq!(d_left.len(), 1);
    assert_eq!(d_left[0].target_pane_id, LEFT_PANE_ID);

    // Signals within cooldown window (switch was ~t=750, cooldown ends at ~t=1000).
    let within_cooldown: Vec<FaceSignal> = (850u64..1000).step_by(33)
        .map(|t| make_signal(t, 20.0, true, None))
        .collect();
    let d_within = feed_stream(&mut clf3, &panes, LEFT_PANE_ID, &within_cooldown);
    assert_eq!(d_within.len(), 0, "No switch within cooldown window, got {}", d_within.len());

    // After cooldown: signals from t=1100 onward with enough dwell.
    let after_cooldown = build_signal_stream(20.0, true, 1100, 900, 33, None);
    let d_after = feed_stream(&mut clf3, &panes, LEFT_PANE_ID, &after_cooldown);
    assert!(d_after.len() >= 1, "Expected RIGHT switch after cooldown elapsed");
    assert_eq!(d_after[0].target_pane_id, RIGHT_PANE_ID);
}

// ---------------------------------------------------------------------------
// Test 8: EMA smoothing rejects single-frame noise
// ---------------------------------------------------------------------------

#[test]
fn test_ema_rejects_single_frame_noise() {
    let cfg = Config { dwell_ms: 750, ema_alpha: 0.35, ..make_config_defaults() };
    let mut clf = Classifier::new(cfg);
    let panes = panes_two();

    // 500 ms of left gaze.
    let mut signals = build_signal_stream(-20.0, true, 0, 500, 33, None);
    // One outlier frame at t=500.
    signals.push(make_signal(500, 20.0, true, None));
    // Continue left gaze well past dwell.
    signals.extend(build_signal_stream(-20.0, true, 533, 1500, 33, None));

    let decisions = feed_stream(&mut clf, &panes, RIGHT_PANE_ID, &signals);
    assert!(decisions.len() >= 1, "Expected at least one LEFT switch despite noise");
    assert_eq!(decisions[0].target_pane_id, LEFT_PANE_ID);
}

// ---------------------------------------------------------------------------
// Test 9a: Iris confirmation suppresses disagreement
// ---------------------------------------------------------------------------

#[test]
fn test_iris_confirmation_suppresses_disagreement() {
    let cfg = Config {
        use_iris_confirmation: true,
        iris_left_ratio: 0.42,
        iris_right_ratio: 0.58,
        dwell_ms: 750,
        ..make_config_defaults()
    };
    let mut clf = Classifier::new(cfg);
    let panes = panes_two();

    // Head-pose: LEFT (yaw=-20), iris_ratio: RIGHT (0.65 > 0.58) → disagreement → CENTER.
    let signals = build_signal_stream(-20.0, true, 0, 2000, 33, Some(0.65));
    let decisions = feed_stream(&mut clf, &panes, RIGHT_PANE_ID, &signals);
    assert!(
        decisions.is_empty(),
        "Expected no switch when head/iris disagree, got {}",
        decisions.len()
    );
}

// ---------------------------------------------------------------------------
// Test 9b: Iris confirmation allows agreement
// ---------------------------------------------------------------------------

#[test]
fn test_iris_confirmation_allows_agreement() {
    let cfg = Config {
        use_iris_confirmation: true,
        iris_left_ratio: 0.42,
        iris_right_ratio: 0.58,
        dwell_ms: 750,
        ..make_config_defaults()
    };
    let mut clf = Classifier::new(cfg);
    let panes = panes_two();

    // Both signals say LEFT: yaw=-20, iris_ratio=0.30 (< 0.42).
    let signals = build_signal_stream(-20.0, true, 0, 1500, 33, Some(0.30));
    let decisions = feed_stream(&mut clf, &panes, RIGHT_PANE_ID, &signals);
    assert!(decisions.len() >= 1, "Expected LEFT switch when head and iris agree");
    assert_eq!(decisions[0].target_pane_id, LEFT_PANE_ID);
}

// ---------------------------------------------------------------------------
// Test 10: Single pane → no switch
// ---------------------------------------------------------------------------

#[test]
fn test_single_pane_no_switch() {
    let mut clf = Classifier::new(make_config_defaults());
    let mut single_pane = HashMap::new();
    single_pane.insert(
        LEFT_PANE_ID.to_owned(),
        PaneInfo {
            pane_id: LEFT_PANE_ID.to_owned(),
            left: 0,
            top: 0,
            width: 160,
            height: 24,
            active: true,
        },
    );

    let signals = build_signal_stream(-20.0, true, 0, 2000, 33, None);
    let decisions = feed_stream(&mut clf, &single_pane, LEFT_PANE_ID, &signals);
    assert!(decisions.is_empty(), "Expected no decisions with single pane, got {:?}", decisions);
}

// ---------------------------------------------------------------------------
// Bonus: Empty panes dict → no switch (edge case)
// ---------------------------------------------------------------------------

#[test]
fn test_empty_panes_no_switch() {
    let mut clf = Classifier::new(make_config_defaults());
    let empty: HashMap<String, PaneInfo> = HashMap::new();
    let result = clf.update(make_signal(0, -20.0, true, None), LEFT_PANE_ID, &empty);
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// Bonus: EMA initialises cleanly on first frame
// ---------------------------------------------------------------------------

#[test]
fn test_ema_initialises_cleanly() {
    let mut clf = Classifier::new(make_config_defaults());
    let panes = panes_two();
    // First frame: dwell hasn't elapsed, so no decision yet.
    let result = clf.update(make_signal(0, -20.0, true, None), RIGHT_PANE_ID, &panes);
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// Bonus: Three-pane layout — uses leftmost and rightmost
// ---------------------------------------------------------------------------

#[test]
fn test_three_pane_uses_leftmost_rightmost() {
    let cfg = Config { dwell_ms: 750, ..make_config_defaults() };
    let mut clf = Classifier::new(cfg);

    let mut panes = HashMap::new();
    for (id, left) in [("%1", 0u32), ("%2", 54u32), ("%3", 108u32)] {
        panes.insert(
            id.to_owned(),
            PaneInfo {
                pane_id: id.to_owned(),
                left,
                top: 0,
                width: 53,
                height: 24,
                active: id == "%3",
            },
        );
    }

    let signals = build_signal_stream(-20.0, true, 0, 1500, 33, None);
    let decisions = feed_stream(&mut clf, &panes, "%3", &signals);
    assert!(decisions.len() >= 1, "Expected switch to leftmost pane (%1)");
    assert_eq!(decisions[0].target_pane_id, "%1");
}
