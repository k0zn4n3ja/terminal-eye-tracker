//! Shared types and trait contracts for tmux-eyes.
//!
//! Every other module codes against these types. Keep this file dependency-light
//! so tests (and parallel workers) can import it without pulling heavy deps.

use std::collections::HashMap;

/// Discrete gaze classification (LEFT / CENTER / RIGHT / UNKNOWN).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GazeClass {
    Left,
    Center,
    Right,
    Unknown,
}

/// One captured webcam frame, BGR uint8.
#[derive(Debug, Clone)]
pub struct Frame {
    pub pixels: Vec<u8>,    // raw BGR bytes, length = width * height * 3
    pub width: u32,
    pub height: u32,
    pub timestamp_ms: u64,  // monotonic ms
}

/// Per-frame output of the vision module.
///
/// `yaw_deg` is positive for "looking right", negative for "looking left",
/// zero for facing camera. Range typically in (-45, +45).
///
/// `iris_ratio` is the average horizontal iris-to-eye-corner ratio across
/// both eyes, in [0, 1]: lower = looking left, ~0.5 = center, higher = right.
/// `None` when the face isn't detected or iris landmarks aren't present.
#[derive(Debug, Clone, Copy)]
pub struct FaceSignal {
    pub timestamp_ms: u64,
    pub detected: bool,
    pub yaw_deg: f32,
    pub iris_ratio: Option<f32>,
}

/// One terminal-multiplexer pane, geometry in cells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaneInfo {
    pub pane_id: String,  // e.g. "%2" (tmux) or "1" (wezterm)
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
    pub active: bool,
}

/// Emitted by the classifier when it wants to switch the focused pane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchDecision {
    pub target_pane_id: String,
    pub reason: String,
}

/// Common interface for any terminal-multiplexer backend (tmux, wezterm, …).
pub trait MultiplexerClient: Send {
    fn get_panes(&mut self) -> anyhow::Result<HashMap<String, PaneInfo>>;
    fn get_active_pane(&mut self) -> anyhow::Result<String>;
    fn select_pane(&mut self, pane_id: &str) -> anyhow::Result<()>;
    fn close(&mut self);
}
