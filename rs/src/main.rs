//! tmux-eyes daemon entry point.
//!
//! Wires camera → vision → classifier → multiplexer backend into a single
//! loop.  See `--help` for CLI options.
//!
//! # Signal handling
//! TODO: Add graceful shutdown via SIGTERM/SIGINT using `signal-hook` (not
//! currently in deps).  For now the OS terminates the process on Ctrl-C, which
//! is acceptable for a POC — the camera `Drop` impl and OS-level cleanup handle
//! resource teardown.

use std::{
    collections::HashMap,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use tmux_eyes::{
    backend::select_backend,
    camera::Camera,
    classifier::Classifier,
    config::Config,
    types::{PaneInfo, SwitchDecision},
    vision::FaceTracker,
};

// How often to re-list all panes (geometry rarely changes).
const PANE_REFRESH_MS: u128 = 1_000;
// How often to query the active pane.
const ACTIVE_REFRESH_MS: u128 = 100;

#[derive(Parser)]
#[command(name = "tmux-eyes", about = "Eye-tracking pane switcher for tmux/wezterm.")]
struct Args {
    /// DEBUG / INFO / WARN / ERROR. Overrides TMUX_EYES_LOG_LEVEL.
    #[arg(long)]
    log_level: Option<String>,

    /// Log every frame's classification (verbose).
    #[arg(long)]
    show_classification: bool,

    /// Run vision + classifier but don't actually switch panes.
    #[arg(long)]
    dry_run: bool,
}

fn setup_logging(level: &str) {
    let filter = EnvFilter::try_new(level.to_lowercase())
        .unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).with_target(false).init();
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = Config::from_env();
    let log_level = args.log_level.clone().unwrap_or_else(|| cfg.log_level.clone());
    setup_logging(&log_level);
    run_daemon(cfg, args)
}

fn run_daemon(cfg: Config, args: Args) -> Result<()> {
    tracing::info!("starting tmux-eyes (dry_run={})", args.dry_run);

    // --- backend ---
    let mut mux = select_backend(&cfg).context("failed to initialise multiplexer backend")?;

    // --- camera ---
    let mut cam = Camera::new(
        cfg.camera_device,
        cfg.camera_width,
        cfg.camera_height,
        cfg.camera_fps,
    )
    .context("failed to open camera")?;

    // --- vision + classifier ---
    let mut tracker =
        FaceTracker::new(&cfg.face_model_path).context("failed to initialise face tracker")?;
    let mut classifier = Classifier::new(cfg.clone());

    // --- initial tmux state ---
    let mut panes: HashMap<String, PaneInfo> = mux
        .get_panes()
        .context("failed to get initial pane list")?;
    let mut active_pane: String = mux
        .get_active_pane()
        .context("failed to get initial active pane")?;

    if panes.is_empty() {
        tracing::error!("no panes found — is your multiplexer running?");
        std::process::exit(2);
    }

    tracing::info!(
        "ready: {} panes, active={}",
        panes.len(),
        active_pane
    );

    // Timestamps for refresh cadence.
    let mut last_pane_refresh = Instant::now();
    let mut last_active_refresh = Instant::now();

    // --- main loop ---
    loop {
        // Capture a frame; on transient failure sleep briefly and retry.
        let frame = match cam.read() {
            Some(f) => f,
            None => {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        // Refresh pane geometry every PANE_REFRESH_MS.
        if last_pane_refresh.elapsed().as_millis() >= PANE_REFRESH_MS {
            match mux.get_panes() {
                Ok(p) => panes = p,
                Err(e) => tracing::warn!("get_panes failed (will retry next cycle): {e}"),
            }
            last_pane_refresh = Instant::now();
        }

        // Refresh active pane every ACTIVE_REFRESH_MS.
        if last_active_refresh.elapsed().as_millis() >= ACTIVE_REFRESH_MS {
            match mux.get_active_pane() {
                Ok(p) => active_pane = p,
                Err(e) => tracing::warn!("get_active_pane failed (will retry next cycle): {e}"),
            }
            last_active_refresh = Instant::now();
        }

        // Vision inference.
        let signal = tracker.process(&frame);

        // Classification.
        let decision: Option<SwitchDecision> =
            classifier.update(signal, &active_pane, &panes);

        // Verbose per-frame logging.
        if args.show_classification {
            tracing::debug!(
                "yaw={:+6.1}° iris={} detected={} active={} decision={}",
                signal.yaw_deg,
                signal
                    .iris_ratio
                    .map(|r| format!("{:.2}", r))
                    .unwrap_or_else(|| "—".into()),
                signal.detected,
                active_pane,
                decision
                    .as_ref()
                    .map(|d| d.target_pane_id.as_str())
                    .unwrap_or("—"),
            );
        }

        // Act on a switch decision.
        if let Some(ref d) = decision {
            tracing::info!(
                "switch {} → {}  (reason: {})",
                active_pane,
                d.target_pane_id,
                d.reason
            );

            if !args.dry_run {
                match mux.select_pane(&d.target_pane_id) {
                    Ok(()) => {
                        // Optimistic local update; next refresh reconciles if needed.
                        active_pane = d.target_pane_id.clone();
                    }
                    Err(e) => tracing::error!("select_pane failed: {e}"),
                }
            }
        }
    }
}
