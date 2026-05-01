//! Webcam capture via `nokhwa` (v0.10).
//!
//! # nokhwa quirks (0.10)
//!
//! * `Camera::new` does NOT open the stream — you must call `open_stream()`
//!   afterwards before calling `frame()`, otherwise `frame()` returns an error.
//!
//! * `RequestedFormatType::Closest` is the safest way to ask for a specific
//!   width × height × fps: the driver picks the nearest supported mode rather
//!   than hard-failing when the exact combination is unavailable.
//!
//! * `frame().decode_image::<RgbFormat>()` returns an
//!   `image::ImageBuffer<Rgb<u8>, Vec<u8>>`.  We store the pixels as-is (RGB
//!   order).  The `Frame.pixels` doc-comment says "BGR", but the vision module
//!   only uses pixel intensities (grayscale math), so the byte order does not
//!   affect correctness.
//!
//! * On Linux the native backend is V4L2 (via the `input-native` feature).
//!   Devices that only expose YUYV need the decode step — `RgbFormat` handles
//!   the conversion transparently.

use std::time::Instant;

use anyhow::{Context, Result};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera as NkhCamera,
};

use crate::types::Frame;

pub struct Camera {
    cam: NkhCamera,
    start: Instant,
}

impl Camera {
    /// Open the camera at `device_index` and begin streaming.
    ///
    /// Uses `RequestedFormatType::Closest` so the driver gracefully selects
    /// the nearest supported mode when the exact `width × height × fps` trio
    /// is unavailable (common with USB webcams).
    pub fn new(device_index: u32, width: u32, height: u32, fps: u32) -> Result<Self> {
        let index = CameraIndex::Index(device_index);

        // Pin YUYV explicitly. Without this, nokhwa picks the highest-FPS mode
        // which on most USB webcams is MJPEG at high res — and nokhwa's MJPEG
        // decode path calls into a mozjpeg-derived helper that panics with
        // "Not available on WASM" on Linux builds. YUYV decodes via nokhwa's
        // pure-Rust path and always works.
        let target = CameraFormat::new(
            Resolution::new(width, height),
            FrameFormat::YUYV,
            fps,
        );
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(target));

        let mut cam =
            NkhCamera::new(index, requested).context("failed to open camera device")?;

        cam.open_stream().context("failed to start camera stream")?;

        tracing::info!(
            device = device_index,
            "camera opened: {}",
            cam.camera_format()
        );

        Ok(Self {
            cam,
            start: Instant::now(),
        })
    }

    /// Capture one frame, decoding it to RGB bytes.
    ///
    /// Returns `None` on a transient decode error so the caller can retry
    /// without tearing down the whole camera.  Persistent hardware faults will
    /// recur, but a retry loop with a brief sleep is sufficient for the daemon.
    pub fn read(&mut self) -> Option<Frame> {
        let raw = match self.cam.frame() {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("camera frame() error (transient?): {e}");
                return None;
            }
        };

        let img = match raw.decode_image::<RgbFormat>() {
            Ok(img) => img,
            Err(e) => {
                tracing::warn!("frame decode error (transient?): {e}");
                return None;
            }
        };

        let width = img.width();
        let height = img.height();
        let timestamp_ms = self
            .start
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);

        Some(Frame {
            pixels: img.into_raw(),
            width,
            height,
            timestamp_ms,
        })
    }

    /// Stop the camera stream.  Idempotent — errors from `stop_stream` are
    /// logged but not propagated since this is called during cleanup.
    pub fn close(&mut self) {
        if let Err(e) = self.cam.stop_stream() {
            tracing::warn!("camera stop_stream error: {e}");
        }
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.close();
    }
}
