//! Tunable thresholds. Mirrors `tmux_eyes/config.py`.
//!
//! All values can be overridden via environment variables prefixed `TMUX_EYES_`.

use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    // --- camera ---
    pub camera_device: u32,
    pub camera_width: u32,
    pub camera_height: u32,
    pub camera_fps: u32,

    // --- vision ---
    /// Path to a `.task` / `.onnx` face landmarker model. Empty → auto-download.
    pub face_model_path: String,

    // --- classification ---
    pub yaw_left_deg: f32,
    pub yaw_right_deg: f32,
    pub ema_alpha: f32,
    pub dwell_ms: u64,
    pub cooldown_ms: u64,

    // --- iris confirmation (Phase 2) ---
    pub use_iris_confirmation: bool,
    pub iris_left_ratio: f32,
    pub iris_right_ratio: f32,

    // --- multiplexer ---
    /// "auto" | "tmux" | "wezterm".
    pub backend: String,
    pub tmux_socket: String,

    // --- runtime ---
    pub log_level: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_device: 0,
            camera_width: 640,
            camera_height: 480,
            camera_fps: 30,
            face_model_path: String::new(),
            yaw_left_deg: -12.0,
            yaw_right_deg: 12.0,
            ema_alpha: 0.35,
            dwell_ms: 250,
            cooldown_ms: 250,
            use_iris_confirmation: false,
            iris_left_ratio: 0.42,
            iris_right_ratio: 0.58,
            backend: "auto".into(),
            tmux_socket: String::new(),
            log_level: "INFO".into(),
        }
    }
}

fn env_str(name: &str, default: &str) -> String {
    env::var(format!("TMUX_EYES_{}", name)).unwrap_or_else(|_| default.into())
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    match env::var(format!("TMUX_EYES_{}", name)) {
        Ok(s) => s.parse::<T>().unwrap_or(default),
        Err(_) => default,
    }
}

impl Config {
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            camera_device: env_parse("CAMERA_DEVICE", d.camera_device),
            camera_width: env_parse("CAMERA_WIDTH", d.camera_width),
            camera_height: env_parse("CAMERA_HEIGHT", d.camera_height),
            camera_fps: env_parse("CAMERA_FPS", d.camera_fps),
            face_model_path: env_str("FACE_MODEL_PATH", &d.face_model_path),
            yaw_left_deg: env_parse("YAW_LEFT_DEG", d.yaw_left_deg),
            yaw_right_deg: env_parse("YAW_RIGHT_DEG", d.yaw_right_deg),
            ema_alpha: env_parse("EMA_ALPHA", d.ema_alpha),
            dwell_ms: env_parse("DWELL_MS", d.dwell_ms),
            cooldown_ms: env_parse("COOLDOWN_MS", d.cooldown_ms),
            use_iris_confirmation: env_str("USE_IRIS_CONFIRMATION", "0") == "1",
            iris_left_ratio: env_parse("IRIS_LEFT_RATIO", d.iris_left_ratio),
            iris_right_ratio: env_parse("IRIS_RIGHT_RATIO", d.iris_right_ratio),
            backend: env_str("BACKEND", &d.backend),
            tmux_socket: env_str("TMUX_SOCKET", &d.tmux_socket),
            log_level: env_str("LOG_LEVEL", &d.log_level),
        }
    }
}
