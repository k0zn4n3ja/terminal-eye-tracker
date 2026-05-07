# aiterm Design & Milestones

This document outlines the development roadmap for aiterm, from project scaffold to frontier features. Each milestone includes concrete acceptance criteria.

## M0 — Project Scaffold (Current)

**Goal:** Establish a buildable Rust workspace with module stubs and passing CI.

**Acceptance Criteria:**
- Cargo workspace compiles cleanly: `cargo check` passes.
- Module stubs exist: `src/main.rs`, `src/app.rs`, `src/render.rs`, `src/pty.rs`, `src/parser.rs`, `src/grid.rs`, `src/block.rs`, `src/llm.rs`, `src/config.rs`.
- CI (GitHub Actions) is green for Linux x86_64.
- No clippy warnings (or documented exceptions).
- Dependencies locked and documented in ARCHITECTURE.md.

## M1 — winit Hello Window

**Goal:** Open a native window and render text via GPU.

**Acceptance Criteria:**
- `cargo run` opens a window titled "aiterm".
- Text "Hello, AI-first terminal." renders via wgpu + cosmic-text at approximately 60 fps.
- Window resize events are handled; text remains visible and centered.
- Frame times logged; no stuttering observed during resize.

## M2 — Single-Pane Terminal

**Goal:** Run a shell, parse its output, and render it in the terminal grid.

**Acceptance Criteria:**
- `portable-pty` spawns a shell (bash/zsh/fish auto-detected).
- `vte` parser consumes PTY output; grid state mutates in real-time.
- Terminal applications work without visible rendering bugs: `vim`, `htop`, `less`.
- Scrollback (up/down arrow keys) shows historical output.
- Alt-screen mode (used by `vim`, `less`) switches between main buffer and alt buffer.
- Wide-character support: emoji and CJK characters display correctly.
- Mouse selection: drag to select text; selection is copy-able.

## M3 — Block Model

**Goal:** Detect and address command/output pairs as discrete blocks.

**Acceptance Criteria:**
- Shell integration scripts (bash, zsh, fish) are installed/available and emit OSC 133 sequences.
- Parser recognizes OSC 133 `CommandStart`, `CommandEnd`, `OutputStart`, `OutputEnd` markers.
- Each block is assigned a unique ID.
- Blocks are visually distinguished (e.g., light border or background shade).
- Clicking a block selects it; keyboard shortcuts (`[`/`]`) navigate between blocks.
- Block boundaries are preserved across scrolling.

## M4 — LLM Integration

**Goal:** Query Claude on block content via the Anthropic API.

**Acceptance Criteria:**
- UI: Cmd+I (macOS) or Ctrl+I (Linux) on a focused block opens an inline prompt.
- Prompt text + focused block context sent to Claude Sonnet (default model).
- API key sourced from `~/.config/aiterm/config.toml` or `ANTHROPIC_API_KEY` env var.
- Response streams character-by-character below the block (no full buffer wait).
- Escape key cancels ongoing stream.
- Errors (API timeout, invalid key) display inline; terminal remains usable.

## M5 — Tabs & Splits

**Goal:** Manage multiple panes and tabs within a single window.

**Acceptance Criteria:**
- Tab strip at the top; each tab is a separate terminal session.
- New tab: Cmd+T (macOS) or Ctrl+T (Linux).
- Close tab: Cmd+W or Ctrl+W.
- Binary-tree pane splits: Cmd+D (vertical split), Cmd+Shift+D (horizontal split).
- Mouse drag on splitter resizes panes.
- Each pane has independent focus, cursor, and scrollback.
- Killing a pane removes it; killing the last pane closes the tab.

## M6 — Voice Input

**Goal:** Transcribe speech to text and insert into the prompt.

**Acceptance Criteria:**
- `whisper-rs` (whisper.cpp bindings) integrated.
- Push-to-talk hotkey (configurable; default: Cmd+` on macOS, Ctrl+` on Linux).
- On first use, `tiny.en` Whisper model auto-downloads and caches.
- Hotkey press starts recording; release stops and transcribes.
- Transcribed text appears in the command prompt; user can edit before sending.
- Timeouts (no speech detected) or errors display a subtle notification.

## M7 — Gaze Tracking

**Goal:** Detect eye position and enable gaze-targeted block prompting.

**Acceptance Criteria:**
- Re-vendor face-mesh ONNX work from main branch (`rs/src/vision.rs` preserved).
- Upgrade output from yaw classification to continuous screen (x, y) coordinates via 5-point homography calibration.
- Calibration UI: point to 5 on-screen targets; 20 samples per target.
- Gaze point rendered as a subtle crosshair (togglable via config).
- Voice + Gaze: hold push-to-talk hotkey, look at a block, speak a prompt — the focused block is auto-included in the request to Claude.

## M8+ — Frontier Features

### Cache-Aware Prompt Assembly
- Claude API prompt caching: static context (shell history, system prompt) cached between requests.
- UI shows context budget: remaining tokens before cache miss, recommended query size.
- Automatic summarization of long blocks to fit within budget.

### Data Class Egress Filter (DLP)
- Regex patterns configured for sensitive data (PII, API keys, credentials).
- Redact or warn before sending block content to Claude.
- Configurable per-user and per-organization.

### Filesystem Snapshot Safety
- Integrate with btrfs/ZFS snapshots for undo: `Cmd+Z` rolls back the filesystem to the last named snapshot.
- Useful after destructive commands: `rm -rf /` can be reverted.

### Shared Sessions & Agent State
- Multi-user access to a named session (e.g., `aiterm --session=collab`).
- Synced cursor, scrollback, and block state across clients.
- Background agent can ingest block output and populate an Inbox pane with summaries/alerts.

### Background Watcher Inbox
- Background task monitors PTY output for patterns (errors, security events, long-running jobs).
- Inbox pane (right sidebar or tab) accumulates notifications.
- Click to jump to the relevant block and see full context.
