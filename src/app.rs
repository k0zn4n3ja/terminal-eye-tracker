//! winit application skeleton (M1).
//!
//! Owns the window and the [`Renderer`]. M1 just opens a 1024x640 window
//! titled "aiterm v0" and paints the hello string every frame; M2+ will wire
//! in the PTY/grid pipeline.

use std::sync::Arc;

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::render::Renderer;

const WINDOW_TITLE: &str = "aiterm v0";
const WINDOW_WIDTH: f64 = 1024.0;
const WINDOW_HEIGHT: f64 = 640.0;

/// Run the winit event loop until the window is closed.
pub fn launch() -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .context("winit event loop terminated abnormally")?;
    Ok(())
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    renderer: Renderer,
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

        self.state = Some(AppState { window, renderer });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("close requested — exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                state.renderer.resize(new_size);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(err) = state.renderer.render() {
                    tracing::error!(error = ?err, "render failed");
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}
