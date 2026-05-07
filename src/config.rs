//! User-facing configuration (M0 scope).
//!
//! Held as a single [`AiTermConfig`] struct. Later milestones will load from
//! `$XDG_CONFIG_HOME/aiterm/config.toml`; for now [`AiTermConfig::default`] is
//! the only constructor.

use std::env;

use serde::{Deserialize, Serialize};

/// Top-level configuration for the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiTermConfig {
    /// Primary font family (cosmic-text will fall back if missing).
    pub font_family: String,
    /// Font size in logical points.
    pub font_size: f32,
    /// Shell to spawn for new PTY sessions. Falls back to `/bin/sh`.
    pub default_shell: String,
}

impl Default for AiTermConfig {
    fn default() -> Self {
        let default_shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        Self {
            font_family: "monospace".to_string(),
            font_size: 14.0,
            default_shell,
        }
    }
}
