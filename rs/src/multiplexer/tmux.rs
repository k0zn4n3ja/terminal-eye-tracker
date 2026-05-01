//! Tmux control-mode IPC client.
//!
//! Maintains a single `tmux -C attach-session` subprocess. Commands are sent
//! over stdin; responses are `%begin … %end` blocks read from stdout.
//! Notification lines (e.g. `%window-pane-changed`) outside blocks are
//! silently discarded.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::OnceLock;

use regex::Regex;

use crate::types::{MultiplexerClient, PaneInfo};

// ---------------------------------------------------------------------------
// Regex constants
// ---------------------------------------------------------------------------

/// Matches one line of `list-panes -a -F '...'` output.
/// Groups: pane_id, left, top, width, height, active(0|1).
static PANE_LINE_RE: OnceLock<Regex> = OnceLock::new();

fn pane_line_re() -> &'static Regex {
    PANE_LINE_RE.get_or_init(|| {
        Regex::new(r"^(%\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([01])\s*$")
            .expect("static regex is valid")
    })
}

/// Accepts only `%<digits>` — anything else could be injected into stdin.
static PANE_ID_RE: OnceLock<Regex> = OnceLock::new();

fn pane_id_re() -> &'static Regex {
    PANE_ID_RE.get_or_init(|| {
        Regex::new(r"^%\d+$").expect("static regex is valid")
    })
}

// ---------------------------------------------------------------------------
// Pure parsing helper (pub so mod.rs can re-export it for tests)
// ---------------------------------------------------------------------------

/// Parse the stdout of `list-panes -a -F '...'` into a `{pane_id: PaneInfo}` map.
///
/// Blank/whitespace-only lines are skipped. Malformed lines return `Err`.
pub fn parse_list_panes(output: &str) -> anyhow::Result<HashMap<String, PaneInfo>> {
    let mut result = HashMap::new();
    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let re = pane_line_re();
        let caps = re
            .captures(line)
            .ok_or_else(|| anyhow::anyhow!("Malformed list-panes line: {:?}", raw_line))?;

        let pane_id = caps[1].to_string();
        let left: u32 = caps[2].parse()?;
        let top: u32 = caps[3].parse()?;
        let width: u32 = caps[4].parse()?;
        let height: u32 = caps[5].parse()?;
        let active = &caps[6] == "1";

        result.insert(
            pane_id.clone(),
            PaneInfo { pane_id, left, top, width, height, active },
        );
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Pane-id validation (pub so tests can call it directly)
// ---------------------------------------------------------------------------

/// Validates that `pane_id` is exactly `%<digits>`. Returns `Err` otherwise.
pub fn validate_pane_id(pane_id: &str) -> anyhow::Result<()> {
    if pane_id_re().is_match(pane_id) {
        Ok(())
    } else {
        anyhow::bail!(
            "Invalid tmux pane_id {:?}. Must match %<digits> (e.g. %0, %12).",
            pane_id
        )
    }
}

// ---------------------------------------------------------------------------
// TmuxClient
// ---------------------------------------------------------------------------

/// Persistent tmux control-mode client.
///
/// Spawns `tmux [-S <socket>] -C attach-session` once and keeps the process
/// alive for the application lifetime. Single-threaded — do not share across
/// threads.
pub struct TmuxClient {
    socket: String,
    child: Option<Child>,
    reader: Option<BufReader<ChildStdout>>,
}

impl TmuxClient {
    /// Create and connect a new client.
    ///
    /// `socket` is the path passed to `tmux -S`; empty means default socket.
    pub fn new(socket: &str) -> anyhow::Result<Self> {
        let mut client = TmuxClient {
            socket: socket.to_string(),
            child: None,
            reader: None,
        };
        client.start()?;
        Ok(client)
    }

    fn start(&mut self) -> anyhow::Result<()> {
        let mut cmd = Command::new("tmux");
        if !self.socket.is_empty() {
            cmd.args(["-S", &self.socket]);
        }
        cmd.args(["-C", "attach-session"]);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn tmux: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("tmux stdout not available"))?;
        let reader = BufReader::new(stdout);

        self.child = Some(child);
        self.reader = Some(reader);

        self.drain_initial_handshake()
    }

    /// Consume the `%begin/%end` block that tmux emits upon attach, before we
    /// send any commands. Without this drain, the first real command's response
    /// would be silently lost.
    fn drain_initial_handshake(&mut self) -> anyhow::Result<()> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("reader not initialised"))?;

        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                // stdout closed — tmux exited early
                let stderr = self.read_stderr();
                anyhow::bail!(
                    "tmux exited before completing control-mode handshake.{}",
                    if stderr.is_empty() { String::new() } else { format!(" stderr: {}", stderr) }
                );
            }
            let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
            if trimmed.starts_with("%end") {
                return Ok(());
            }
            if trimmed.starts_with("%error") {
                let stderr = self.read_stderr();
                anyhow::bail!(
                    "tmux -C attach-session failed.{}",
                    if stderr.is_empty() { String::new() } else { format!(" stderr: {}", stderr) }
                );
            }
            // %begin, notifications, etc. — keep reading
        }
    }

    fn read_stderr(&mut self) -> String {
        if let Some(child) = self.child.as_mut() {
            if let Some(mut stderr) = child.stderr.take() {
                use std::io::Read;
                let mut buf = String::new();
                let _ = stderr.read_to_string(&mut buf);
                return buf.trim().to_string();
            }
        }
        String::new()
    }

    fn assert_alive(&mut self) -> anyhow::Result<()> {
        let child = self
            .child
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("TmuxClient is closed."))?;
        if let Some(status) = child.try_wait()? {
            anyhow::bail!(
                "tmux control-mode subprocess exited with status {}.",
                status
            );
        }
        Ok(())
    }

    /// Send `cmd\n` to stdin, collect and return the `%begin…%end` payload.
    fn run_command(&mut self, cmd: &str) -> anyhow::Result<Vec<String>> {
        self.assert_alive()?;

        // Write command to stdin
        {
            let child = self.child.as_mut().unwrap();
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("tmux stdin not available"))?;
            writeln!(stdin, "{}", cmd)?;
            stdin.flush()?;
        }

        // Read response
        let reader = self.reader.as_mut().unwrap();
        let mut payload: Vec<String> = Vec::new();
        let mut in_block = false;
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                anyhow::bail!("tmux stdout closed unexpectedly while waiting for %end.");
            }
            let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');

            if trimmed.starts_with("%begin") {
                in_block = true;
                payload.clear();
                continue;
            }
            if trimmed.starts_with("%end") {
                if in_block {
                    return Ok(payload);
                }
                // stray %end — ignore
                continue;
            }
            if trimmed.starts_with("%error") {
                anyhow::bail!("tmux returned an error for command {:?}", cmd);
            }
            if in_block {
                payload.push(trimmed.to_string());
            }
            // else: notification line (e.g. %window-pane-changed) — discard
        }
    }
}

impl MultiplexerClient for TmuxClient {
    fn get_panes(&mut self) -> anyhow::Result<HashMap<String, PaneInfo>> {
        let lines = self.run_command(
            "list-panes -a -F '#{pane_id} #{pane_left} #{pane_top} #{pane_width} #{pane_height} #{pane_active}'"
        )?;
        parse_list_panes(&lines.join("\n"))
    }

    fn get_active_pane(&mut self) -> anyhow::Result<String> {
        let lines = self.run_command("display-message -p '#{pane_id}'")?;
        lines
            .into_iter()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("display-message returned no output."))
    }

    fn select_pane(&mut self, pane_id: &str) -> anyhow::Result<()> {
        validate_pane_id(pane_id)?;
        self.run_command(&format!("select-pane -t {}", pane_id))?;
        Ok(())
    }

    fn close(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Close stdin — signals EOF to tmux
            drop(child.stdin.take());
            // Wait up to 2 s, then kill
            use std::time::{Duration, Instant};
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    _ => {}
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        self.reader = None;
    }
}

impl Drop for TmuxClient {
    fn drop(&mut self) {
        self.close();
    }
}
