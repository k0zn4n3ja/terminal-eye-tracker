//! GPU text renderer (M1).
//!
//! A wgpu surface plus a `glyphon` text renderer fed by `cosmic-text`. The
//! M1 deliverable just paints a single hello string; later milestones will
//! replace the static buffer with the live terminal grid.
//!
//! Font fallback: we rely on system fonts via `cosmic-text`'s default
//! `FontSystem`, which uses `fontdb` to scan `/usr/share/fonts`,
//! `~/.local/share/fonts`, etc. If no monospace family is present the text
//! still renders, but in whatever family `Family::Monospace` resolves to via
//! fontdb's fallback chain. Bundling a default mono font (e.g. JetBrains
//! Mono) is deferred to M2 polish.
//!
//! Headless / CI note: adapter acquisition can fail without a GPU. M1's
//! verification only requires `cargo check` and `cargo clippy` — interactive
//! smoke testing happens on a developer workstation.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

const HELLO_TEXT: &str = "Hello, AI-first terminal.";
const FONT_SIZE: f32 = 16.0;
const LINE_HEIGHT: f32 = 20.0;

/// Background colour (R, G, B, A) — a near-black tuned for low contrast.
const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.06,
    b: 0.08,
    a: 1.0,
};

/// Pure helper: compute the top-left pixel position of the hello string.
///
/// The renderer paints the string centred horizontally, with its top edge at
/// roughly one third of the surface height. Extracted so the math is unit-
/// testable without a GPU; the live `render` path forwards
/// `(line_w, LINE_HEIGHT)` from `cosmic-text`'s shaped layout.
///
/// `text_extent` is `(width, height)` in pixels; only `width` is used today
/// but `height` is accepted so M2's grid renderer can vertically centre
/// without changing the signature.
pub(crate) fn compute_hello_position(
    surface: PhysicalSize<u32>,
    text_extent: (f32, f32),
) -> (f32, f32) {
    let (line_w, _line_h) = text_extent;
    let left = ((surface.width as f32 - line_w) / 2.0).max(0.0);
    let top = surface.height as f32 / 3.0;
    (left, top)
}

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

        let mut text_buffer = Buffer::new(&mut font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
        text_buffer.set_size(
            &mut font_system,
            Some(surface_config.width as f32),
            Some(surface_config.height as f32),
        );
        text_buffer.set_text(
            &mut font_system,
            HELLO_TEXT,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        text_buffer.shape_until_scroll(&mut font_system, false);

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
        self.text_buffer
            .shape_until_scroll(&mut self.font_system, false);
    }

    /// Acquire the next surface frame and paint the hello string.
    pub fn render(&mut self) -> Result<()> {
        // Centre horizontally, place the baseline ~1/3 down.
        let line_w = self
            .text_buffer
            .layout_runs()
            .next()
            .map_or(0.0, |run| run.line_w);
        let surface_size = PhysicalSize::new(self.surface_config.width, self.surface_config.height);
        let (left, top) = compute_hello_position(surface_size, (line_w, LINE_HEIGHT));

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
                    left,
                    top,
                    scale: 1.0,
                    bounds: TextBounds::default(),
                    default_color: Color::rgb(0xE6, 0xED, 0xF3),
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
// Coverage here comes from compute_hello_position + constant assertions
// only; the GPU code is exercised manually via `cargo run`.
#[cfg(test)]
mod tests {
    use super::*;

    fn size(w: u32, h: u32) -> PhysicalSize<u32> {
        PhysicalSize::new(w, h)
    }

    #[test]
    fn centred_horizontally_when_text_fits() {
        let (left, top) = compute_hello_position(size(1024, 600), (200.0, 20.0));
        assert!((left - 412.0).abs() < f32::EPSILON, "got left={left}");
        assert!((top - 200.0).abs() < f32::EPSILON, "got top={top}");
    }

    #[test]
    fn left_clamped_to_zero_when_text_wider_than_surface() {
        // A 100px-wide window with 400px-wide text would otherwise yield
        // left = -150.0 → clamp to 0.0.
        let (left, top) = compute_hello_position(size(100, 300), (400.0, 20.0));
        assert!(left.abs() < f32::EPSILON, "left should clamp to 0, got {left}");
        assert!((top - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn handles_zero_text_width() {
        let (left, top) = compute_hello_position(size(800, 600), (0.0, 0.0));
        assert!((left - 400.0).abs() < f32::EPSILON);
        assert!((top - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn handles_one_by_one_surface() {
        // Pathological tiny surface — just confirms no panic and the clamp
        // still applies.
        let (left, top) = compute_hello_position(size(1, 1), (10.0, 20.0));
        assert!(left.abs() < f32::EPSILON);
        assert!((top - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn handles_tall_narrow_surface() {
        let (left, top) = compute_hello_position(size(40, 4_000), (20.0, 20.0));
        assert!((left - 10.0).abs() < f32::EPSILON);
        assert!((top - (4_000.0 / 3.0)).abs() < 1e-3);
    }

    #[test]
    fn handles_wide_short_surface() {
        let (left, top) = compute_hello_position(size(4_000, 40), (200.0, 20.0));
        assert!((left - 1_900.0).abs() < f32::EPSILON);
        assert!((top - (40.0 / 3.0)).abs() < 1e-3);
    }

    #[test]
    fn ignores_text_height_for_now() {
        // text_extent.1 is currently unused; confirm two heights yield the
        // same position so future changes show up as test failures.
        let a = compute_hello_position(size(800, 600), (300.0, 10.0));
        let b = compute_hello_position(size(800, 600), (300.0, 9_999.0));
        assert_eq!(a, b);
    }

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
    #[allow(clippy::assertions_on_constants, clippy::const_is_empty)]
    fn font_metrics_constants_are_consistent() {
        assert!(FONT_SIZE > 0.0);
        assert!(LINE_HEIGHT >= FONT_SIZE);
        assert!(!HELLO_TEXT.is_empty());
    }
}
