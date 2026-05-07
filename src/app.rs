//! Application skeleton (M1 scope).
//!
//! TODO(M1): Implement a winit `ApplicationHandler` that owns the wgpu
//! surface/device/queue and the [`crate::render`] text renderer. For M0 this
//! is a placeholder so the crate compiles end-to-end.

use anyhow::Result;

/// Launch the terminal application.
///
/// M0: returns `Ok(())` immediately. M1 will boot a winit event loop, create a
/// wgpu surface, and drive the renderer.
pub fn launch() -> Result<()> {
    tracing::info!("aiterm M0 scaffold — no window yet");
    Ok(())
}
