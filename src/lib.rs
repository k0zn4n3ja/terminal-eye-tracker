//! `aiterm` — AI-first terminal emulator (Linux first).
//!
//! Module map (milestone scope in parens):
//!   - [`app`]    (M1) winit/wgpu application skeleton
//!   - [`config`] (M0) user-facing configuration
//!   - [`pty`]    (M2) PTY spawn + I/O
//!   - [`parser`] (M2) VT escape parser
//!   - [`grid`]   (M2) terminal grid model
//!   - [`render`] (M1) GPU text renderer
//!   - [`block`]  (M3) "Warp-style" command blocks
//!   - [`llm`]    (M4) LLM streaming client + agent loop

pub mod app;
pub mod block;
pub mod config;
pub mod grid;
pub mod llm;
pub mod parser;
pub mod pty;
pub mod render;

/// Library entrypoint. The `aiterm` binary calls this after initialising
/// tracing. For now it just forwards to [`app::launch`].
pub fn run() -> anyhow::Result<()> {
    app::launch()
}
