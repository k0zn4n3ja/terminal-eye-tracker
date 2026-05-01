//! Multiplexer backend selection.
//!
//! Factory that returns a `Box<dyn MultiplexerClient>` based on the resolved
//! backend name, so the rest of the daemon doesn't care whether the underlying
//! terminal multiplexer is tmux or wezterm.

use crate::config::Config;
use crate::multiplexer::tmux::TmuxClient;
use crate::multiplexer::wezterm::WeztermClient;
use crate::types::MultiplexerClient;

// ---------------------------------------------------------------------------
// PATH search helper (no `which` crate available)
// ---------------------------------------------------------------------------

fn binary_in_path(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            if dir.join(name).is_file() {
                return true;
            }
            #[cfg(windows)]
            if dir.join(format!("{}.exe", name)).is_file() {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Detect the best available multiplexer backend.
///
/// Order of precedence:
///   1. `$TMUX` env var set → tmux (we're inside a tmux session).
///   2. `$WEZTERM_PANE` set → wezterm (we're inside a wezterm pane).
///   3. `tmux` binary on PATH → tmux.
///   4. `wezterm` binary on PATH → wezterm.
///   5. Error.
pub fn detect_backend() -> anyhow::Result<&'static str> {
    if std::env::var("TMUX").is_ok() {
        return Ok("tmux");
    }
    if std::env::var("WEZTERM_PANE").is_ok() {
        return Ok("wezterm");
    }
    if binary_in_path("tmux") {
        return Ok("tmux");
    }
    if binary_in_path("wezterm") {
        return Ok("wezterm");
    }
    anyhow::bail!(
        "no multiplexer detected — set TMUX_EYES_BACKEND=tmux or wezterm"
    )
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build a `MultiplexerClient` based on `cfg.backend`.
///
/// `"auto"` triggers [`detect_backend`]. `"tmux"` and `"wezterm"` are explicit.
pub fn select_backend(cfg: &Config) -> anyhow::Result<Box<dyn MultiplexerClient>> {
    let name = if cfg.backend.is_empty() {
        "auto".to_string()
    } else {
        cfg.backend.to_lowercase()
    };

    let resolved = if name == "auto" {
        detect_backend()?.to_string()
    } else {
        name
    };

    match resolved.as_str() {
        "tmux" => {
            let client = TmuxClient::new(&cfg.tmux_socket)?;
            Ok(Box::new(client))
        }
        "wezterm" => Ok(Box::new(WeztermClient::new())),
        other => anyhow::bail!(
            "Unknown backend {:?}. Use 'tmux', 'wezterm', or 'auto'.",
            other
        ),
    }
}
