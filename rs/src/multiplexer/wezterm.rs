//! WezTerm CLI IPC client.
//!
//! Stateless — each call spawns a fresh `wezterm cli …` subprocess. There is
//! no persistent connection to manage, so `close()` is a no-op.

use std::collections::HashMap;
use std::process::Command;
use std::sync::OnceLock;

use regex::Regex;
use serde::Deserialize;

use crate::types::{MultiplexerClient, PaneInfo};

// ---------------------------------------------------------------------------
// Regex constant
// ---------------------------------------------------------------------------

/// Accepts only purely numeric pane ids (wezterm style: "0", "1", "42").
static PANE_ID_RE: OnceLock<Regex> = OnceLock::new();

fn pane_id_re() -> &'static Regex {
    PANE_ID_RE.get_or_init(|| Regex::new(r"^\d+$").expect("static regex is valid"))
}

// ---------------------------------------------------------------------------
// JSON deserialization shapes
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PaneSize {
    cols: u32,
    rows: u32,
}

#[derive(Deserialize)]
struct RawPane {
    pane_id: u64,
    left_col: u32,
    top_row: u32,
    size: PaneSize,
    is_active: bool,
}

// ---------------------------------------------------------------------------
// Pure parsing helper (pub so mod.rs can re-export it for tests)
// ---------------------------------------------------------------------------

/// Parse the stdout of `wezterm cli list --format json` into a `{pane_id: PaneInfo}` map.
///
/// Returns an empty map for an empty JSON array. Returns `Err` for invalid JSON
/// or entries missing required fields.
pub fn parse_wezterm_list(json_text: &str) -> anyhow::Result<HashMap<String, PaneInfo>> {
    let entries: Vec<RawPane> = serde_json::from_str(json_text)
        .map_err(|e| anyhow::anyhow!("Failed to parse wezterm JSON output: {}", e))?;

    let mut result = HashMap::new();
    for entry in entries {
        let pane_id = format!("{}", entry.pane_id);
        result.insert(
            pane_id.clone(),
            PaneInfo {
                pane_id,
                left: entry.left_col,
                top: entry.top_row,
                width: entry.size.cols,
                height: entry.size.rows,
                active: entry.is_active,
            },
        );
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Pane-id validation (pub so tests can call it directly)
// ---------------------------------------------------------------------------

/// Validates that `pane_id` is purely numeric. Returns `Err` otherwise.
pub fn validate_pane_id(pane_id: &str) -> anyhow::Result<()> {
    if pane_id_re().is_match(pane_id) {
        Ok(())
    } else {
        anyhow::bail!(
            "Invalid wezterm pane_id {:?}. Must be numeric digits only (e.g. '1', '42').",
            pane_id
        )
    }
}

// ---------------------------------------------------------------------------
// WeztermClient
// ---------------------------------------------------------------------------

/// Stateless WezTerm CLI client.
///
/// Each public method shells out to `wezterm cli …` fresh. No persistent
/// subprocess is maintained.
pub struct WeztermClient;

impl WeztermClient {
    pub fn new() -> Self {
        WeztermClient
    }

    fn run_list(&self) -> anyhow::Result<HashMap<String, PaneInfo>> {
        let output = Command::new("wezterm")
            .args(["cli", "list", "--format", "json"])
            .output()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    anyhow::anyhow!("wezterm CLI not found in PATH")
                } else {
                    anyhow::anyhow!("Failed to run wezterm cli list: {}", e)
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stderr = stderr.trim();
            anyhow::bail!(
                "wezterm cli list exited with code {}{}",
                output.status.code().unwrap_or(-1),
                if stderr.is_empty() { String::new() } else { format!(": {}", stderr) }
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            anyhow::bail!("wezterm cli list returned empty output");
        }

        parse_wezterm_list(&stdout)
    }
}

impl Default for WeztermClient {
    fn default() -> Self {
        WeztermClient::new()
    }
}

impl MultiplexerClient for WeztermClient {
    fn get_panes(&mut self) -> anyhow::Result<HashMap<String, PaneInfo>> {
        self.run_list()
    }

    fn get_active_pane(&mut self) -> anyhow::Result<String> {
        let panes = self.run_list()?;
        panes
            .into_values()
            .find(|p| p.active)
            .map(|p| p.pane_id)
            .ok_or_else(|| anyhow::anyhow!("No active pane found in wezterm cli list output."))
    }

    fn select_pane(&mut self, pane_id: &str) -> anyhow::Result<()> {
        validate_pane_id(pane_id)?;
        Command::new("wezterm")
            .args(["cli", "activate-pane", "--pane-id", pane_id])
            .status()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    anyhow::anyhow!("wezterm CLI not found in PATH")
                } else {
                    anyhow::anyhow!("Failed to run wezterm cli activate-pane: {}", e)
                }
            })?;
        Ok(())
    }

    fn close(&mut self) {
        // no-op: no persistent process to clean up
    }
}
