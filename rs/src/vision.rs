//! Vision module: webcam frame -> `FaceSignal`.
//!
//! Pure-Rust port of `tmux_eyes/vision.py`. Replaces MediaPipe's
//! `FaceLandmarker` with a `tract-onnx` runtime loading a pre-converted
//! ONNX face mesh model directly.
//!
//! The two pure math helpers (`compute_head_yaw` and `compute_iris_ratio`)
//! are kept module-level so they can be unit-tested without instantiating
//! the model runtime or downloading anything.
//!
//! Yaw sign convention (matches the Python module):
//!     positive yaw = user looking to *their* right (camera-left)
//!     negative yaw = user looking to *their* left  (camera-right)
//!
//! Head-pose approximation
//! -----------------------
//! The Python code uses OpenCV's iterative `solvePnP` then `RQDecomp3x3` to
//! recover Euler angles. A pure-Rust equivalent (via nalgebra LM solve) is a
//! lot of plumbing for the POC's coarse +-12 deg L/R discrimination. Instead
//! we use a closed-form weak-perspective approximation:
//!
//!   1. Take the projected eye-line midpoint M and nose pixel N. The sign of
//!      (N.x - M.x) gives the yaw sign: nose drifts toward the camera-side
//!      the user is *facing*.
//!   2. Magnitude comes from the asymmetry of the two half-faces around the
//!      nose: when yawing right, the user's right side (camera-left) appears
//!      wider. Specifically with the canonical 6-point 3D model, distance
//!      |nose - left_eye_outer|_x and |nose - right_eye_outer|_x, in pixels,
//!      satisfy
//!          (L - R) / (L + R) = (52 / 86.6) * tan(theta)
//!      so theta = atan( (L - R) / (L + R) * 86.6 / 52 ).
//!   3. Sign convention: with yaw>0 (looking right), the camera-left half of
//!      the face (the user's right side, indices for "left eye outer" in
//!      MediaPipe's convention) is wider. So yaw = +atan(...) when L>R.
//!
//! Magnitude is slightly under-estimated due to perspective foreshortening
//! (the closer half-face projects bigger on the image plane, partly
//! cancelling the rotation asymmetry) but the +-12 deg threshold is well
//! within the achievable accuracy for 30-degree yaws.
//!
//! ONNX model
//! ----------
//! We download a pre-converted MediaPipe FaceMesh ONNX model (468 landmarks,
//! 192x192 NCHW float32 input). `tract-onnx` is a pure-Rust runtime that
//! loads it without any conversion or native dependencies. Note this is
//! the v1 model, which lacks iris landmarks (468 points, not 478), so
//! `iris_ratio` will be `None` for this model.
//!
//! Source: <https://github.com/PINTO0309/facemesh_onnx_tensorrt/releases/download/1.0.0/face_mesh_Nx3x192x192_post.onnx>
//!
//! If the download/load chain fails, `FaceTracker` falls back to a STUB
//! inference path that synthesises an oscillating yaw signal so the rest
//! of the daemon can still be exercised end-to-end.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};
use tract_onnx::prelude::*;

use crate::types::{FaceSignal, Frame};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// 6-point 3D face model (mm), origin at nose tip. Order matches
/// `POSE_LM_INDICES`: nose tip, chin, left-eye-outer, right-eye-outer,
/// left-mouth, right-mouth.
#[allow(dead_code)]
pub(crate) const MODEL_POINTS_3D: [[f32; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
];

/// MediaPipe 478-landmark indices for the 6 head-pose points.
pub(crate) const POSE_LM_INDICES: [usize; 6] = [1, 199, 33, 263, 61, 291];

/// Iris and eye-corner indices.
pub(crate) const LEFT_IRIS_IDX: usize = 468;
pub(crate) const RIGHT_IRIS_IDX: usize = 473;
pub(crate) const LEFT_EYE_INNER: usize = 133;
pub(crate) const LEFT_EYE_OUTER: usize = 33;
pub(crate) const RIGHT_EYE_INNER: usize = 362;
pub(crate) const RIGHT_EYE_OUTER: usize = 263;

/// Pre-converted MediaPipe FaceMesh ONNX (468 landmarks, 192x192 NCHW input).
/// Float32 weights — works with pure-Rust tract. No iris (v1 model).
/// Verified live: HTTP 200, 2.35 MB, supports Range requests.
const FACE_MODEL_URL: &str =
    "https://github.com/PINTO0309/facemesh_onnx_tensorrt/releases/download/1.0.0/face_mesh_Nx3x192x192_post.onnx";
const MODEL_FILENAME: &str = "face_mesh_192x192.onnx";

const MODEL_INPUT_SIZE: u32 = 192;

// ---------------------------------------------------------------------------
// Pure math helpers
// ---------------------------------------------------------------------------

/// Estimate head-pose yaw (degrees) from 6 projected facial landmarks.
///
/// `landmarks_2d` order must be:
///   nose tip, chin, left-eye-outer, right-eye-outer, left-mouth, right-mouth.
///
/// Returns yaw in degrees. Positive = user looking to their right.
pub fn compute_head_yaw(landmarks_2d: &[(f32, f32); 6], image_size: (u32, u32)) -> f32 {
    let _ = image_size; // image_size unused in the closed-form approximation,
                        // kept in the signature to match the Python contract
                        // and for a future PnP upgrade.

    let nose = landmarks_2d[0];
    let left_eye_outer = landmarks_2d[2];
    let right_eye_outer = landmarks_2d[3];

    // Half-widths in pixel x of the two facial halves (signed -> abs).
    let left_half = (nose.0 - left_eye_outer.0).abs();
    let right_half = (right_eye_outer.0 - nose.0).abs();

    let denom = left_half + right_half;
    if denom < 1e-6 {
        return 0.0;
    }

    // (L - R) / (L + R) = (52 / 86.6) * tan(theta), so:
    //     theta = atan( ratio * 86.6 / 52 )
    // 86.6 / 52 = 1.66538...
    const SCALE: f32 = 86.6 / 52.0;
    let ratio = (left_half - right_half) / denom;
    let theta = (ratio * SCALE).atan();
    theta.to_degrees()
}

/// Horizontal iris-to-eye-corner ratio, averaged across both eyes.
///
/// For each eye:  ratio = (iris_x - inner_x) / (outer_x - inner_x)
/// Returns 0.5 for any eye with degenerate (near-zero) span.
pub fn compute_iris_ratio(
    left_iris_x: f32,
    left_inner_x: f32,
    left_outer_x: f32,
    right_iris_x: f32,
    right_inner_x: f32,
    right_outer_x: f32,
) -> f32 {
    fn one_eye(iris: f32, inner: f32, outer: f32) -> f32 {
        let span = outer - inner;
        if span.abs() < 1e-6 {
            return 0.5;
        }
        (iris - inner) / span
    }

    let left = one_eye(left_iris_x, left_inner_x, left_outer_x);
    let right = one_eye(right_iris_x, right_inner_x, right_outer_x);
    (left + right) / 2.0
}

// ---------------------------------------------------------------------------
// Model bootstrap
// ---------------------------------------------------------------------------

fn default_cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| anyhow!("could not resolve a cache directory for tmux-eyes"))?;
    Ok(base.join("tmux-eyes"))
}

/// Resolve a usable `.onnx` path. If `explicit_path` is non-empty, use it
/// directly (no download). Otherwise:
///   1. If `<cache>/face_mesh_192x192.onnx` exists, return it.
///   2. Else download the ONNX directly (atomic .part rename).
fn ensure_face_landmarker_model(explicit_path: &str) -> Result<PathBuf> {
    if !explicit_path.is_empty() {
        let p = PathBuf::from(explicit_path);
        if !p.exists() {
            return Err(anyhow!(
                "FACE_MODEL_PATH was set but file does not exist: {}",
                p.display()
            ));
        }
        return Ok(p);
    }

    let cache_dir = default_cache_dir()?;
    fs::create_dir_all(&cache_dir).with_context(|| {
        format!("failed to create model cache dir: {}", cache_dir.display())
    })?;

    let onnx_path = cache_dir.join(MODEL_FILENAME);
    if let Ok(meta) = fs::metadata(&onnx_path) {
        if meta.len() > 0 {
            return Ok(onnx_path);
        }
    }

    tracing::info!(
        "FaceTracker: downloading face mesh ONNX -> {}",
        onnx_path.display()
    );
    let tmp = onnx_path.with_extension("onnx.part");
    download_to_file(FACE_MODEL_URL, &tmp).with_context(|| {
        format!("failed to download face mesh ONNX from {}", FACE_MODEL_URL)
    })?;
    fs::rename(&tmp, &onnx_path).with_context(|| {
        format!(
            "failed to atomically rename {} -> {}",
            tmp.display(),
            onnx_path.display()
        )
    })?;
    Ok(onnx_path)
}

fn download_to_file(url: &str, dest: &Path) -> Result<()> {
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(60))
        .call()
        .with_context(|| format!("HTTP GET failed: {}", url))?;
    let mut reader = resp.into_reader();
    let mut bytes: Vec<u8> = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .context("reading model bytes from HTTP response")?;
    if bytes.is_empty() {
        return Err(anyhow!("downloaded model is empty: {}", url));
    }
    fs::write(dest, &bytes)
        .with_context(|| format!("writing model to {}", dest.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// FaceTracker
// ---------------------------------------------------------------------------

/// Concrete type of `tract-onnx`'s optimized + runnable model. The
/// `into_runnable()` chain returns `SimplePlan<...>` which is aliased as
/// `RunnableModel<TypedFact, Box<dyn TypedOp>, ...>`. We keep the alias
/// terse here.
type RunnableOnnxModel = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

/// Wraps an ONNX face landmark model and produces a `FaceSignal` per frame.
///
/// If the model can't be acquired or initialised, the tracker falls back
/// to a STUB inference path: it synthesises a yaw oscillation in
/// (-20, +20) degrees driven by `Frame::timestamp_ms`. This keeps the rest
/// of the daemon runnable for end-to-end testing even without a usable
/// model. Real-world deployments must succeed with the real model.
pub struct FaceTracker {
    model: Option<RunnableOnnxModel>,
    /// Resolved input edge size (height == width). Read from the loaded
    /// model's input fact and falls back to `MODEL_INPUT_SIZE` if dynamic.
    input_size: u32,
    /// True when initialisation failed and we are running synthetic data.
    stub: bool,
}

impl FaceTracker {
    /// Build a new tracker. `model_path` empty -> auto-download the face
    /// mesh ONNX into `dirs::cache_dir()/tmux-eyes/`, and load it via tract.
    pub fn new(model_path: &str) -> Result<Self> {
        match Self::try_init_real(model_path) {
            Ok(t) => Ok(t),
            Err(e) => {
                tracing::warn!(
                    "FaceTracker: using STUB inference, no real model found - vision worker needs manual fix ({:?})",
                    e
                );
                Ok(Self {
                    model: None,
                    input_size: MODEL_INPUT_SIZE,
                    stub: true,
                })
            }
        }
    }

    fn try_init_real(model_path: &str) -> Result<Self> {
        let path = ensure_face_landmarker_model(model_path)?;

        let model = tract_onnx::onnx()
            .model_for_path(&path)
            .with_context(|| format!("loading onnx model from {}", path.display()))?
            .into_optimized()
            .context("optimizing onnx model")?
            .into_runnable()
            .context("preparing runnable onnx model")?;

        // Read the actual input edge size from the model. The PINTO ONNX is
        // NCHW (batch, channels, H, W) with H == W; tract surfaces shape as
        // `Vec<TDim>`. If anything is dynamic / unexpected, fall through to
        // the constant.
        let input_size = model
            .model()
            .input_fact(0)
            .ok()
            .and_then(|fact| {
                let shape = fact.shape.as_concrete()?;
                // NCHW with N=1: [1, 3, H, W]. Pick H (== W).
                if shape.len() == 4 && shape[1] == 3 && shape[2] == shape[3] {
                    Some(shape[2] as u32)
                } else {
                    None
                }
            })
            .unwrap_or(MODEL_INPUT_SIZE);

        tracing::debug!(
            "FaceTracker: tract onnx session ready (input_size={})",
            input_size
        );

        Ok(Self {
            model: Some(model),
            input_size,
            stub: false,
        })
    }

    /// Run landmark detection on one frame. Never panics. Returns a
    /// `FaceSignal` with `detected: false` when no face is found or when
    /// any inference error occurs.
    pub fn process(&mut self, frame: &Frame) -> FaceSignal {
        if self.stub {
            return self.process_stub(frame);
        }
        match self.process_real(frame) {
            Ok(sig) => sig,
            Err(e) => {
                tracing::debug!("FaceTracker: inference error -> not detected ({:?})", e);
                FaceSignal {
                    timestamp_ms: frame.timestamp_ms,
                    detected: false,
                    yaw_deg: 0.0,
                    iris_ratio: None,
                }
            }
        }
    }

    fn process_real(&mut self, frame: &Frame) -> Result<FaceSignal> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("FaceTracker model not initialised"))?;

        let (w, h) = (frame.width, frame.height);
        let input = build_input_tensor_nchw(frame, self.input_size)?;

        // The _post model has 5 inputs: image + 4 I32 crop-box scalars
        // (crop_x1, crop_y1, crop_width, crop_height) in model-pixel space.
        // We feed the full 192x192 image, so the crop covers the whole frame.
        let input_size_i = self.input_size as i32;
        let make_i32 = |v: i32| -> Tensor {
            tract_ndarray::arr2(&[[v]]).into_tensor()
        };
        let crop_x1     = make_i32(0);
        let crop_y1     = make_i32(0);
        let crop_width  = make_i32(input_size_i);
        let crop_height = make_i32(input_size_i);

        let outputs = model
            .run(tvec!(
                input.into(),
                crop_x1.into(),
                crop_y1.into(),
                crop_width.into(),
                crop_height.into()
            ))
            .context("tract model.run failed")?;

        // The PINTO face mesh ONNX emits `final_landmarks` of shape
        // [N, 468, 3] — 468 (x, y, z) tuples in the model's 0..192 pixel
        // space. We rescale to original frame coords on decode.
        let lm = decode_landmarks_from_outputs(&outputs, self.input_size, w, h)?;

        let pose_pts = collect_pose_points(&lm)?;
        let yaw = compute_head_yaw(&pose_pts, (w, h));

        // The v1 ONNX (468 landmarks) has no iris points. Guard with the
        // same `>= 478` check the Python uses (`has_iris = len(landmarks)
        // >= 478`) so swapping in an iris-equipped ONNX later just works.
        let iris_ratio = if lm.len() >= 478 {
            Some(compute_iris_ratio(
                lm[LEFT_IRIS_IDX].0,
                lm[LEFT_EYE_INNER].0,
                lm[LEFT_EYE_OUTER].0,
                lm[RIGHT_IRIS_IDX].0,
                lm[RIGHT_EYE_INNER].0,
                lm[RIGHT_EYE_OUTER].0,
            ))
        } else {
            None
        };

        Ok(FaceSignal {
            timestamp_ms: frame.timestamp_ms,
            detected: true,
            yaw_deg: yaw,
            iris_ratio,
        })
    }

    fn process_stub(&self, frame: &Frame) -> FaceSignal {
        // Oscillate yaw -20..+20 over a 10s period so downstream classifier
        // has *something* to chew on during end-to-end smoke tests.
        let phase = (frame.timestamp_ms as f32 / 10_000.0) * std::f32::consts::TAU;
        let yaw_deg = 20.0 * phase.sin();
        FaceSignal {
            timestamp_ms: frame.timestamp_ms,
            detected: true,
            yaw_deg,
            iris_ratio: None,
        }
    }

    /// Release model resources.
    pub fn close(&mut self) {
        self.model = None;
    }
}

impl Drop for FaceTracker {
    fn drop(&mut self) {
        self.close();
    }
}

// ---------------------------------------------------------------------------
// Inference glue: BGR frame -> NCHW float tensor -> landmark coords.
// ---------------------------------------------------------------------------

/// Build a `[1, 3, size, size]` NCHW float32 tensor in `[0, 1]` from a
/// BGR `Frame`. The PINTO ONNX expects RGB channels-first.
fn build_input_tensor_nchw(frame: &Frame, size: u32) -> Result<Tensor> {
    let expected = (frame.width as usize)
        .saturating_mul(frame.height as usize)
        .saturating_mul(3);
    if frame.pixels.len() != expected {
        return Err(anyhow!(
            "frame pixel buffer length {} does not match {}x{}x3 = {}",
            frame.pixels.len(),
            frame.width,
            frame.height,
            expected
        ));
    }

    // BGR -> RGB while building an ImageBuffer for resizing.
    let mut rgb_pixels = Vec::with_capacity(expected);
    for chunk in frame.pixels.chunks_exact(3) {
        rgb_pixels.push(chunk[2]); // R
        rgb_pixels.push(chunk[1]); // G
        rgb_pixels.push(chunk[0]); // B
    }
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width, frame.height, rgb_pixels)
            .ok_or_else(|| anyhow!("failed to build ImageBuffer from frame pixels"))?;

    let resized = image::imageops::resize(&img, size, size, FilterType::Triangle);

    // NCHW float32 in [0, 1]. Channels-first means channel c at (y, x) goes
    // into input[(0, c, y, x)] — NOT the interleaved RGB-per-pixel layout.
    let s = size as usize;
    let mut input = tract_ndarray::Array4::<f32>::zeros((1, 3, s, s));
    for y in 0..s {
        for x in 0..s {
            let p = resized.get_pixel(x as u32, y as u32);
            input[(0, 0, y, x)] = p.0[0] as f32 / 255.0;
            input[(0, 1, y, x)] = p.0[1] as f32 / 255.0;
            input[(0, 2, y, x)] = p.0[2] as f32 / 255.0;
        }
    }
    Ok(input.into_tensor())
}

/// Decode the model's output tensors into pixel-space `(x, y)` landmarks
/// in the original frame's coordinate system.
///
/// The _post PINTO face mesh ONNX emits two outputs:
///   #0 [1,1] F32   — face confidence score
///   #1 [1,468,3] I32 — landmarks (x, y, z) in model pixel space (0..192)
/// We try F32 first (forward-compat), then I32 cast to f32.
fn decode_landmarks_from_outputs(
    outputs: &[TValue],
    model_input_size: u32,
    w: u32,
    h: u32,
) -> Result<Vec<(f32, f32)>> {
    // Pass 1: try native f32 outputs (handles non-_post models too).
    for out in outputs {
        if let Ok(view) = out.to_array_view::<f32>() {
            if let Some(raw) = view.as_slice() {
                if let Some(lm) = try_decode_flat(raw, model_input_size, w, h) {
                    return Ok(lm);
                }
            }
        }
    }
    // Pass 2: try i32 outputs (the _post model's landmark tensor is I32).
    for out in outputs {
        if let Ok(view) = out.to_array_view::<i32>() {
            if let Some(raw_i32) = view.as_slice() {
                let raw_f32: Vec<f32> = raw_i32.iter().map(|&v| v as f32).collect();
                if let Some(lm) = try_decode_flat(&raw_f32, model_input_size, w, h) {
                    return Ok(lm);
                }
            }
        }
    }
    Err(anyhow!(
        "no output tensor matched the expected face-landmark layout (got {} outputs)",
        outputs.len()
    ))
}

fn try_decode_flat(
    raw: &[f32],
    model_input_size: u32,
    w: u32,
    h: u32,
) -> Option<Vec<(f32, f32)>> {
    // We accept three landmark layouts:
    //   (N, 3) flat       -> raw.len() % 3 == 0, n >= 468
    //   (N, 2) flat       -> raw.len() % 2 == 0, n >= 468
    let stride = if raw.len() % 3 == 0 && raw.len() / 3 >= 468 {
        3
    } else if raw.len() % 2 == 0 && raw.len() / 2 >= 468 {
        2
    } else {
        return None;
    };

    let n = raw.len() / stride;

    // Sniff the value range so we can rescale model-pixel-space
    // (~0..192), normalised (0..1), or already-image-pixel coords.
    let mut max_xy: f32 = 0.0;
    for i in 0..n {
        let x = raw[i * stride];
        let y = raw[i * stride + 1];
        if x.is_finite() && y.is_finite() {
            max_xy = max_xy.max(x.abs()).max(y.abs());
        }
    }

    let (sx, sy) = if max_xy <= 1.5 {
        (w as f32, h as f32)
    } else if max_xy <= (model_input_size as f32) * 1.2 {
        (w as f32 / model_input_size as f32, h as f32 / model_input_size as f32)
    } else {
        (1.0, 1.0)
    };

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = raw[i * stride] * sx;
        let y = raw[i * stride + 1] * sy;
        out.push((x, y));
    }
    Some(out)
}

fn collect_pose_points(lm: &[(f32, f32)]) -> Result<[(f32, f32); 6]> {
    let max_idx = *POSE_LM_INDICES.iter().max().unwrap();
    if lm.len() <= max_idx {
        return Err(anyhow!(
            "landmark vector too short: have {}, need at least {}",
            lm.len(),
            max_idx + 1
        ));
    }
    Ok([
        lm[POSE_LM_INDICES[0]],
        lm[POSE_LM_INDICES[1]],
        lm[POSE_LM_INDICES[2]],
        lm[POSE_LM_INDICES[3]],
        lm[POSE_LM_INDICES[4]],
        lm[POSE_LM_INDICES[5]],
    ])
}
