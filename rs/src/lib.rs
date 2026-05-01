//! tmux-eyes: webcam-driven pane switcher for tmux and wezterm.
//!
//! Crate root that re-exports the public modules. Binary entry point is in
//! `main.rs`; everything below is library code so it can be tested.

pub mod backend;
pub mod camera;
pub mod classifier;
pub mod config;
pub mod multiplexer;
pub mod types;
pub mod vision;
