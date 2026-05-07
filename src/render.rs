//! GPU text renderer (M1 → M2a).
//!
//! A wgpu surface plus a `glyphon` text renderer fed by `cosmic-text`.
//! [`Renderer::render`] takes the current [`Grid`] and rebuilds the glyphon
//! [`Buffer`] from its rows each frame: every cell's `ch` is concatenated
//! row-by-row with `\n` separators, then handed to cosmic-text for shaping.
//!
//! ## Why rebuild the buffer every frame?
//!
//! The simplest correct path: cosmic-text owns the line-layout, fallback,
//! and glyph caching; we just push a fresh string each tick. Damage tracking
//! (only re-shape changed rows) and per-cell colour application (CSI SGR)
//! land in M2c.
//!
//! ## Font fallback
//!
//! We rely on system fonts via `cosmic-text`'s default [`FontSystem`], which
//! uses `fontdb` to scan `/usr/share/fonts`, `~/.local/share/fonts`, etc.
//! If no monospace family is present the text still renders, but in whatever
//! family `Family::Monospace` resolves to via fontdb's fallback chain.
//! Bundling a default mono font is M2 polish.
//!
//! ## Headless / CI note
//!
//! Adapter acquisition can fail without a GPU. The unit tests below cover
//! the constants only; the live render path is exercised via `cargo run`.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::grid::Grid;

const FONT_SIZE: f32 = 16.0;
const LINE_HEIGHT: f32 = 20.0;

/// Inset (in pixels) so the text doesn't hug the window edge. Applied to
/// both axes; the rest of the window is letterboxed in [`BACKGROUND`].
const TEXT_INSET: f32 = 8.0;

/// Foreground colour used for every cell in M2a — per-cell colours from
/// CSI SGR land in M2c.
const FG_COLOR: Color = Color::rgb(0xE6, 0xED, 0xF3);

/// Background colour (R, G, B, A) — a near-black tuned for low contrast.
const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.06,
    b: 0.08,
    a: 1.0,
};

/// Owns the wgpu pipeline and the cosmic-text/glyphon stack used to paint a
/// single frame.
///
/// Drop order is fine: `Surface<'static>` internally retains an `Arc<Window>`
/// via the `create_surface` handle-source mechanism, so the window cannot be
/// freed before the surface even though we also hold our own [`Arc<Window>`]
/// for `request_redraw` calls.
pub struct Renderer {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,

    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    text_buffer: Buffer,

    window: Arc<Window>,
}

impl Renderer {
    /// Build the renderer for `window`. Must be called on the event-loop
    /// thread (`ActiveEventLoop` is needed to wire wgpu's display handle).
    pub async fn new(window: Arc<Window>, event_loop: &ActiveEventLoop) -> Result<Self> {
        let physical_size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(event_loop.owned_display_handle()),
        ));

        let surface = instance
            .create_surface(window.clone())
            .context("create wgpu surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("acquire wgpu adapter")?;

        let info = adapter.get_info();
        tracing::info!(
            backend = ?info.backend,
            adapter = %info.name,
            "wgpu adapter ready"
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .context("acquire wgpu device")?;

        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: physical_size.width.max(1),
            height: physical_size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut atlas = TextAtlas::new(&device, &queue, &cache, format);
        let text_renderer =
            TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);

        // The buffer starts empty; `render(grid)` populates it on every
        // frame. We still need to size it so cosmic-text knows the wrap
        // width; `resize` keeps that in sync with the swapchain.
        let mut text_buffer = Buffer::new(&mut font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
        text_buffer.set_size(
            &mut font_system,
            Some(surface_config.width as f32),
            Some(surface_config.height as f32),
        );

        Ok(Self {
            instance,
            surface,
            surface_config,
            device,
            queue,
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            text_buffer,
            window,
        })
    }

    /// Reconfigure the swapchain and reshape the text buffer for `new_size`.
    ///
    /// The grid itself stays fixed at its M2a 24×80 dimensions — this just
    /// updates the renderer's idea of the surface and the wrap width
    /// cosmic-text uses for line layout.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);

        self.text_buffer.set_size(
            &mut self.font_system,
            Some(new_size.width as f32),
            Some(new_size.height as f32),
        );
    }

    /// Acquire the next surface frame and paint `grid`.
    ///
    /// Rebuilds the cosmic-text buffer from the grid's rows, shapes it,
    /// then prepares + submits a single draw call. Errors from glyphon /
    /// wgpu propagate; transient swapchain states (Outdated/Suboptimal/
    /// Lost/Timeout/Occluded) are recovered in-place via `request_redraw`.
    pub fn render(&mut self, grid: &Grid) -> Result<()> {
        // 1. Build the frame text from the grid. One char per cell, rows
        //    separated by '\n'. We include trailing spaces — colour-aware
        //    rendering arrives in M2c and will replace this with per-cell
        //    spans, so trimming now would only have to be reversed.
        let (rows, cols) = grid.dimensions();
        let mut text = String::with_capacity(rows as usize * (cols as usize + 1));
        for (i, row) in grid.iter_rows().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            for cell in row {
                text.push(cell.ch);
            }
        }

        self.text_buffer.set_text(
            &mut self.font_system,
            &text,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.text_buffer
            .shape_until_scroll(&mut self.font_system, false);

        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.surface_config.width,
                height: self.surface_config.height,
            },
        );

        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                [TextArea {
                    buffer: &self.text_buffer,
                    left: TEXT_INSET,
                    top: TEXT_INSET,
                    scale: 1.0,
                    bounds: TextBounds::default(),
                    default_color: FG_COLOR,
                    custom_glyphs: &[],
                }],
                &mut self.swash_cache,
            )
            .map_err(|e| anyhow!("glyphon prepare failed: {e}"))?;

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .instance
                    .create_surface(self.window.clone())
                    .context("recreate surface after Lost")?;
                self.surface.configure(&self.device, &self.surface_config);
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(anyhow!("wgpu surface validation error"));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aiterm-frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aiterm-text"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .map_err(|e| anyhow!("glyphon render failed: {e}"))?;
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        self.atlas.trim();
        Ok(())
    }
}

// COVERAGE: GPU-bound paths (Renderer::new / resize / render) require a
// real wgpu surface and can't run under cargo test without a display.
// Coverage here is limited to the published colour and metric constants;
// the GPU code is exercised manually via `cargo run`.
//
// The M1 `compute_hello_position` helper and its tests were intentionally
// removed in M2a: the renderer no longer paints a single centred string,
// so the helper has no callers and tests would have locked in dead
// behaviour.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn background_constant_is_opaque_dark() {
        // Sanity-check the published colour constant so later style changes
        // are intentional rather than accidental.
        assert!((BACKGROUND.a - 1.0).abs() < f64::EPSILON);
        assert!(BACKGROUND.r < 0.2);
        assert!(BACKGROUND.g < 0.2);
        assert!(BACKGROUND.b < 0.2);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn font_metrics_constants_are_consistent() {
        // Line height must accommodate the font size, otherwise rows would
        // collide vertically.
        assert!(FONT_SIZE > 0.0);
        assert!(LINE_HEIGHT >= FONT_SIZE);
        // Inset must be non-negative — a negative value would push the
        // first row out of the surface.
        assert!(TEXT_INSET >= 0.0);
    }
}
