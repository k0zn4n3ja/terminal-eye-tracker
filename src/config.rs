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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- resolve_default_shell ----------------------------------------------

    #[test]
    fn resolve_default_shell_uses_env_value_when_present() {
        let got = resolve_default_shell(|k| {
            assert_eq!(k, "SHELL");
            Some("/usr/bin/zsh".to_string())
        });
        assert_eq!(got, "/usr/bin/zsh");
    }

    #[test]
    fn resolve_default_shell_falls_back_when_unset() {
        let got = resolve_default_shell(|_| None);
        assert_eq!(got, "/bin/sh");
    }

    #[test]
    fn resolve_default_shell_passes_through_empty_string() {
        // Spec preserves prior behaviour: an explicitly empty $SHELL is not
        // treated as "missing"; only an absent env var triggers the fallback.
        let got = resolve_default_shell(|_| Some(String::new()));
        assert_eq!(got, "");
    }

    #[test]
    fn resolve_default_shell_handles_paths_with_whitespace() {
        let weird = "/opt/strange path/with tabs\tand spaces/sh";
        let got = resolve_default_shell(|_| Some(weird.to_string()));
        assert_eq!(got, weird);
    }

    #[test]
    fn resolve_default_shell_only_consults_shell_key() {
        // Make sure the helper does not look at unrelated env vars.
        let got = resolve_default_shell(|k| {
            if k == "PATH" {
                Some("/should/not/be/used".to_string())
            } else {
                None
            }
        });
        assert_eq!(got, "/bin/sh");
    }

    // ---- AiTermConfig::with_env_getter --------------------------------------

    #[test]
    fn with_env_getter_populates_defaults_and_shell() {
        let cfg = AiTermConfig::with_env_getter(|_| Some("/bin/fish".to_string()));
        assert_eq!(cfg.font_family, "monospace");
        assert!((cfg.font_size - 14.0).abs() < f32::EPSILON);
        assert_eq!(cfg.default_shell, "/bin/fish");
    }

    #[test]
    fn with_env_getter_falls_back_on_missing_shell() {
        let cfg = AiTermConfig::with_env_getter(|_| None);
        assert_eq!(cfg.default_shell, "/bin/sh");
    }

    // ---- Default ------------------------------------------------------------

    #[test]
    fn default_yields_sane_constants() {
        // We can't pin `default_shell` without inspecting the host env, but
        // font defaults are stable across environments.
        let cfg = AiTermConfig::default();
        assert_eq!(cfg.font_family, "monospace");
        assert!((cfg.font_size - 14.0).abs() < f32::EPSILON);
        // default_shell should be either an env value or the /bin/sh fallback;
        // either way it parses as a non-control-character string.
        assert!(cfg
            .default_shell
            .chars()
            .all(|c| !c.is_control() || c == '\t'));
    }

    // ---- Trait derivations --------------------------------------------------

    #[test]
    fn debug_impl_includes_field_names() {
        let cfg = AiTermConfig::with_env_getter(|_| Some("/bin/sh".to_string()));
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("font_family"));
        assert!(dbg.contains("font_size"));
        assert!(dbg.contains("default_shell"));
    }

    #[test]
    fn clone_produces_equal_struct() {
        let a = AiTermConfig::with_env_getter(|_| Some("/bin/dash".to_string()));
        let b = a.clone();
        assert_eq!(a.font_family, b.font_family);
        assert!((a.font_size - b.font_size).abs() < f32::EPSILON);
        assert_eq!(a.default_shell, b.default_shell);
    }

    // ---- TOML ---------------------------------------------------------------

    #[test]
    fn from_toml_parses_full_document() {
        let src = r#"
            font_family = "JetBrains Mono"
            font_size = 18.5
            default_shell = "/usr/bin/zsh"
        "#;
        let cfg = AiTermConfig::from_toml(src).expect("valid toml parses");
        assert_eq!(cfg.font_family, "JetBrains Mono");
        assert!((cfg.font_size - 18.5).abs() < f32::EPSILON);
        assert_eq!(cfg.default_shell, "/usr/bin/zsh");
    }

    #[test]
    fn from_toml_rejects_partial_document() {
        // No defaults inside `from_toml` — missing fields surface as errors.
        let src = r#"font_family = "Mono""#;
        let err = AiTermConfig::from_toml(src).expect_err("partial toml should error");
        let msg = err.to_string();
        assert!(
            msg.contains("font_size") || msg.contains("default_shell") || msg.contains("missing"),
            "error should mention the missing field, got: {msg}"
        );
    }

    #[test]
    fn from_toml_rejects_garbage() {
        let err = AiTermConfig::from_toml("this is not = = toml [[")
            .expect_err("garbage should not parse");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn to_toml_round_trips_through_from_toml() {
        let original = AiTermConfig::with_env_getter(|_| Some("/bin/bash".to_string()));
        let serialised = original.to_toml().expect("default config serialises");
        let parsed = AiTermConfig::from_toml(&serialised).expect("default-shaped toml parses");

        assert_eq!(parsed.font_family, original.font_family);
        assert!((parsed.font_size - original.font_size).abs() < f32::EPSILON);
        assert_eq!(parsed.default_shell, original.default_shell);
    }

    #[test]
    fn from_toml_accepts_extreme_font_sizes() {
        // The struct does not validate font_size today; cover both extremes
        // so any future validation is a deliberate, test-driven change.
        let huge = r#"
            font_family = "Mono"
            font_size = 1000000.0
            default_shell = "/bin/sh"
        "#;
        let cfg = AiTermConfig::from_toml(huge).expect("huge font parses");
        assert!(cfg.font_size > 999_999.0);

        let neg = r#"
            font_family = "Mono"
            font_size = -1.0
            default_shell = "/bin/sh"
        "#;
        let cfg = AiTermConfig::from_toml(neg).expect("negative font parses (no validation)");
        assert!(cfg.font_size < 0.0);
    }
}
