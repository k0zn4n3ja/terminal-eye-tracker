//! GPU text renderer (M1 scope).
//!
//! TODO(M1): A wgpu-backed renderer driven by `cosmic-text` shaping and a
//! glyph atlas (likely via `glyphon`). Consumes a frame description and emits
//! draw calls onto the surface owned by [`crate::app`].
