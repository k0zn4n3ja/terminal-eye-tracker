//! Classifier module — fusion, EMA smoothing, and dwell state machine.
//!
//! Pure state machine: no I/O, no clock calls. All timestamps come from
//! the `FaceSignal`, making the classifier deterministically testable.

use std::collections::HashMap;

use crate::config::Config;
use crate::types::{FaceSignal, GazeClass, PaneInfo, SwitchDecision};

// Numeric encoding for EMA smoothing.
const GAZE_LEFT_NUM: f32 = -1.0;
const GAZE_CENTER_NUM: f32 = 0.0;
const GAZE_RIGHT_NUM: f32 = 1.0;

// Re-discretisation thresholds applied to the smoothed value.
const EMA_LEFT_THRESHOLD: f32 = -0.4;
const EMA_RIGHT_THRESHOLD: f32 = 0.4;

/// Stateful per-session classifier.
///
/// Call [`Classifier::update`] once per incoming [`FaceSignal`].
/// Returns a [`SwitchDecision`] when the dwell timer expires, otherwise `None`.
pub struct Classifier {
    cfg: Config,
    ema_value: Option<f32>,
    candidate_pane_id: Option<String>,
    candidate_since_ms: u64,
    last_switch_ms: i64, // signed because we initialize to "far in the past"
}

impl Classifier {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            ema_value: None,
            candidate_pane_id: None,
            candidate_since_ms: 0,
            last_switch_ms: -1_000_000,
        }
    }

    /// Process one frame. Returns `Some(decision)` when a switch should fire,
    /// else `None`.
    pub fn update(
        &mut self,
        signal: FaceSignal,
        active_pane_id: &str,
        panes: &HashMap<String, PaneInfo>,
    ) -> Option<SwitchDecision> {
        // Need at least two panes to have anything to switch to.
        if panes.len() < 2 {
            return None;
        }

        // Step 1: classify raw signal → GazeClass.
        let raw_class = self.classify_raw(&signal);

        // Step 2: EMA-smooth (Unknown frames are skipped).
        let smoothed_class = self.update_ema(raw_class);

        // Step 3 & 4: pane targeting + dwell state machine.
        self.dwell_update(signal.timestamp_ms, smoothed_class, active_pane_id, panes)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn classify_raw(&self, signal: &FaceSignal) -> GazeClass {
        if !signal.detected {
            return GazeClass::Unknown;
        }

        // Head-pose classification (primary signal).
        let head_class = if signal.yaw_deg < self.cfg.yaw_left_deg {
            GazeClass::Left
        } else if signal.yaw_deg > self.cfg.yaw_right_deg {
            GazeClass::Right
        } else {
            GazeClass::Center
        };

        // Iris confirmation (opt-in).
        if self.cfg.use_iris_confirmation {
            if let Some(iris_ratio) = signal.iris_ratio {
                let iris_class = self.classify_iris(iris_ratio);
                // Suppress when head-pose and iris disagree AND iris is not Center.
                if head_class != iris_class && iris_class != GazeClass::Center {
                    return GazeClass::Center;
                }
            }
        }

        head_class
    }

    fn classify_iris(&self, iris_ratio: f32) -> GazeClass {
        if iris_ratio < self.cfg.iris_left_ratio {
            GazeClass::Left
        } else if iris_ratio > self.cfg.iris_right_ratio {
            GazeClass::Right
        } else {
            GazeClass::Center
        }
    }

    fn update_ema(&mut self, gaze_class: GazeClass) -> GazeClass {
        // Unknown frames do not update the EMA; return Unknown so the dwell
        // machine can clear its candidate.
        if gaze_class == GazeClass::Unknown {
            return GazeClass::Unknown;
        }

        let x = match gaze_class {
            GazeClass::Left => GAZE_LEFT_NUM,
            GazeClass::Center => GAZE_CENTER_NUM,
            GazeClass::Right => GAZE_RIGHT_NUM,
            GazeClass::Unknown => unreachable!(),
        };

        let alpha = self.cfg.ema_alpha;
        let ema = match self.ema_value {
            None => x,
            Some(prev) => alpha * x + (1.0 - alpha) * prev,
        };
        self.ema_value = Some(ema);

        if ema < EMA_LEFT_THRESHOLD {
            GazeClass::Left
        } else if ema > EMA_RIGHT_THRESHOLD {
            GazeClass::Right
        } else {
            GazeClass::Center
        }
    }

    fn identify_left_right_panes<'a>(&self, panes: &'a HashMap<String, PaneInfo>) -> (&'a str, &'a str) {
        // Sort pane values by `left` coordinate; return leftmost and rightmost ids.
        let mut sorted: Vec<&PaneInfo> = panes.values().collect();
        sorted.sort_by_key(|p| p.left);
        (sorted[0].pane_id.as_str(), sorted[sorted.len() - 1].pane_id.as_str())
    }

    fn dwell_update(
        &mut self,
        timestamp_ms: u64,
        smoothed_class: GazeClass,
        active_pane_id: &str,
        panes: &HashMap<String, PaneInfo>,
    ) -> Option<SwitchDecision> {
        let (left_pane_id, right_pane_id) = self.identify_left_right_panes(panes);

        // Map smoothed class → target pane id.
        let target_pane_id: Option<&str> = match smoothed_class {
            GazeClass::Left => Some(left_pane_id),
            GazeClass::Right => Some(right_pane_id),
            GazeClass::Center | GazeClass::Unknown => None,
        };

        // Clear candidate when there is no directed gaze or gaze is at active pane.
        let target = match target_pane_id {
            None => {
                self.candidate_pane_id = None;
                return None;
            }
            Some(t) if t == active_pane_id => {
                self.candidate_pane_id = None;
                return None;
            }
            Some(t) => t,
        };

        // New candidate: reset dwell timer.
        let same_candidate = self
            .candidate_pane_id
            .as_deref()
            .map_or(false, |c| c == target);

        if !same_candidate {
            self.candidate_pane_id = Some(target.to_owned());
            self.candidate_since_ms = timestamp_ms;
            return None;
        }

        // Candidate is unchanged — check dwell + cooldown.
        let dwell_elapsed = timestamp_ms.saturating_sub(self.candidate_since_ms);
        let cooldown_elapsed = timestamp_ms as i64 - self.last_switch_ms;

        if dwell_elapsed >= self.cfg.dwell_ms && cooldown_elapsed >= self.cfg.cooldown_ms as i64 {
            let decision = SwitchDecision {
                target_pane_id: target.to_owned(),
                reason: "dwell".into(),
            };
            self.last_switch_ms = timestamp_ms as i64;
            self.candidate_pane_id = None;
            return Some(decision);
        }

        None
    }
}
