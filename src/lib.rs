//! `aiterm` — AI-first terminal emulator (Linux first).
//!
//! Module map (milestone scope in parens):
//!   - [`app`]    (M1/M2a) winit/wgpu application skeleton + integration
//!   - [`config`] (M0) user-facing configuration
//!   - [`pty`]    (M2a) PTY spawn + I/O
//!   - [`parser`] (M2a) VT escape parser
//!   - [`grid`]   (M2a) terminal grid model
//!   - [`render`] (M1/M2a) GPU text renderer
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

use anyhow::Context as _;

/// Library entrypoint. The `aiterm` binary calls this after initialising
/// tracing.
///
/// ## Async runtime
///
/// The PTY layer relies on [`tokio::task::spawn_blocking`] to drain the
/// master-side reader (a blocking `std::io::Read`) without stalling the GUI.
/// winit's event loop is single-threaded and synchronous, so we do not call
/// `block_on` here — instead we build a multi-threaded tokio runtime and
/// install it on the current thread via [`tokio::runtime::Handle::enter`].
/// While the [`tokio::runtime::EnterGuard`] is alive, any
/// `spawn_blocking` call (such as the one inside [`crate::pty::Pty::spawn`])
/// finds the runtime through the thread-local handle and dispatches to its
/// blocking pool. The runtime is dropped when [`run`] returns, which joins
/// or aborts every still-running blocking task.
///
/// `new_multi_thread` is preferred over `new_current_thread` here because it
/// gives `spawn_blocking` an independent worker pool — we never advance the
/// async executor on the GUI thread, so there is nothing for a current-thread
/// runtime to do other than own the blocking workers anyway.
pub fn run() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;
    let _guard = runtime.enter();
    app::launch()
}
