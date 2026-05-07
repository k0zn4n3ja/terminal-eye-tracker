//! User-facing configuration (M0 scope).
//!
//! Held as a single [`AiTermConfig`] struct. Later milestones will load from
//! `$XDG_CONFIG_HOME/aiterm/config.toml`; for now [`AiTermConfig::default`] is
//! the typical constructor and [`AiTermConfig::from_toml`] parses a serialised
//! form.
//!
//! Environment lookup is dependency-injected through
//! [`AiTermConfig::with_env_getter`] so tests can exercise the fallback logic
//! without touching the process-wide environment (which is `unsafe` in newer
//! Rust and races under parallel test execution).

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

impl AiTermConfig {
    /// Build a default config, looking up environment variables via the
    /// supplied `get_env` closure. This is the dependency-injection seam used
    /// by tests; production code calls [`Default::default`] which wraps
    /// [`std::env::var`].
    pub fn with_env_getter<F>(get_env: F) -> Self
    where
        F: Fn(&str) -> Option<String>,
    {
        Self {
            font_family: "monospace".to_string(),
            font_size: 14.0,
            default_shell: resolve_default_shell(&get_env),
        }
    }

    /// Parse a config from a TOML string.
    pub fn from_toml(s: &str) -> anyhow::Result<Self> {
        let cfg: Self = toml::from_str(s)?;
        Ok(cfg)
    }

    /// Serialise this config to TOML.
    pub fn to_toml(&self) -> anyhow::Result<String> {
        Ok(toml::to_string(self)?)
    }
}

impl Default for AiTermConfig {
    fn default() -> Self {
        Self::with_env_getter(|k| env::var(k).ok())
    }
}

/// Pure helper: resolve the default shell from an env-getter closure.
///
/// Returns the value of `$SHELL` if the getter yields one, otherwise falls
/// back to `/bin/sh`. The previous direct `env::var` call returned the
/// `/bin/sh` fallback only for `Err` (missing or non-Unicode); routing through
/// `Option` here preserves that fallback for both error cases identically
/// (callers that want fallback on empty must filter the value themselves).
pub(crate) fn resolve_default_shell<F>(get_env: F) -> String
where
    F: Fn(&str) -> Option<String>,
{
    get_env("SHELL").unwrap_or_else(|| "/bin/sh".to_string())
}
