//! winit application — wires PTY → parser → grid → renderer (M2a).
//!
//! Each frame:
//!   1. **between frames** ([`ApplicationHandler::about_to_wait`]):
//!      drain whatever bytes the PTY background reader pushed onto the mpsc
//!      channel since the last tick and feed each chunk through the VT
//!      [`Parser`] into the [`Grid`]. If the grid changed, request a redraw.
//!   2. **on `RedrawRequested`**: hand the grid to the [`Renderer`].
//!   3. **on `Resized`**: only the renderer's swapchain rescales; the grid
//!      stays at the M2a-fixed `24×80` so text letterboxes inside the
//!      window. PTY-side resize lands in M2c.
//!   4. **on `CloseRequested`**: drop [`AppState`] (drops the [`Pty`], which
//!      kills the child process via `Drop`) and exit the event loop.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc::{error::TryRecvError, UnboundedReceiver};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::config::AiTermConfig;
use crate::grid::Grid;
use crate::parser::Parser;
use crate::pty::Pty;
use crate::render::Renderer;

const WINDOW_TITLE: &str = "aiterm v0";
const WINDOW_WIDTH: f64 = 1024.0;
const WINDOW_HEIGHT: f64 = 640.0;

/// M2a fixes the grid at 24×80; the renderer letterboxes inside whatever
/// window the user actually opens. The PTY is told the same dimensions so
/// the shell sees a consistent geometry. Window resize handling is M2c.
const GRID_ROWS: u16 = 24;
const GRID_COLS: u16 = 80;

/// Run the winit event loop until the window is closed.
///
/// Builds an [`AiTermConfig`] from the ambient environment and hands it to
/// the [`App`]. The function returns when [`ActiveEventLoop::exit`] is
/// called from inside an event handler (typically on `CloseRequested`).
pub fn launch() -> Result<()> {
    let config = AiTermConfig::default();
    let event_loop = EventLoop::new().context("create winit event loop")?;
    let mut app = App::new(config);
    event_loop
        .run_app(&mut app)
        .context("winit event loop terminated abnormally")?;
    Ok(())
}

/// Application root. Holds configuration plus the lazily-initialised
/// [`AppState`] that comes alive on the first `resumed` event.
struct App {
    config: AiTermConfig,
    state: Option<AppState>,
}

impl App {
    fn new(config: AiTermConfig) -> Self {
        Self { config, state: None }
    }
}

/// Live application state — owned by the event loop after `resumed`.
struct AppState {
    window: Arc<Window>,
    renderer: Renderer,
    grid: Grid,
    parser: Parser,
    /// Held primarily for its [`Drop`] side-effect (kills the child shell).
    /// Future milestones will call `pty.write` from key handlers, hence the
    /// `dead_code` allow rather than `_pty`.
    #[allow(dead_code)]
    pty: Pty,
    rx: UnboundedReceiver<Vec<u8>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title(WINDOW_TITLE);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(err) => {
                tracing::error!(error = %err, "failed to create window");
                event_loop.exit();
                return;
            }
        };

        let renderer = match pollster::block_on(Renderer::new(window.clone(), event_loop)) {
            Ok(r) => r,
            Err(err) => {
                tracing::error!(error = ?err, "failed to initialise renderer");
                event_loop.exit();
                return;
            }
        };

        // Spawn the shell. `Pty::spawn` internally calls `tokio::task::
        // spawn_blocking`, which requires an active tokio runtime context —
        // `aiterm::run` installs one via `Handle::enter` for the duration
        // of the event loop, so this call finds it on the current thread.
        let (pty, rx) = match Pty::spawn(GRID_ROWS, GRID_COLS, &self.config.default_shell) {
            Ok(t) => t,
            Err(err) => {
                tracing::error!(
                    error = ?err,
                    shell = %self.config.default_shell,
                    "failed to spawn PTY",
                );
                event_loop.exit();
                return;
            }
        };

        self.state = Some(AppState {
            window,
            renderer,
            grid: Grid::new(GRID_ROWS, GRID_COLS),
            parser: Parser::new(),
            pty,
            rx,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // CloseRequested needs to mutate `self.state` directly, so handle it
        // before re-borrowing through `as_mut` below.
        if matches!(event, WindowEvent::CloseRequested) {
            tracing::info!("close requested — exiting");
            // Tear down state explicitly so the PTY kills its child before
            // we ask the event loop to exit, rather than relying on the
            // implicit Drop when `App` itself is dropped after `run_app`.
            self.state = None;
            event_loop.exit();
            return;
        }

        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::Resized(new_size) => {
                state.renderer.resize(new_size);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(err) = state.renderer.render(&state.grid) {
                    tracing::error!(error = ?err, "render failed");
                }
            }
            // CloseRequested handled above; everything else (keyboard, mouse,
            // focus, …) is M2b/M2c territory.
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Drain whatever the PTY reader produced since the last tick and
        // apply each chunk to the grid through the VT parser. We use
        // try_recv because we MUST NOT block the event loop on the channel —
        // about_to_wait is winit's "between frames" hook and is the right
        // place for this work. Once the channel drains, we issue a redraw
        // iff the grid actually changed.
        let mut updated = false;
        loop {
            match state.rx.try_recv() {
                Ok(chunk) => {
                    state.parser.feed(&mut state.grid, &chunk);
                    updated = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Reader task ended (child exited or read error).
                    // Whatever was in the grid at exit-time is the final
                    // frame; we keep redrawing it and let the user close
                    // the window manually.
                    break;
                }
            }
        }

        if updated {
            state.window.request_redraw();
        }
    }
}
