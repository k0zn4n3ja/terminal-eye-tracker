//! `aiterm` binary entrypoint.
//!
//! Initialises tracing from the `RUST_LOG` env var (defaulting to `info`) and
//! delegates to [`aiterm::run`]. The process exit code mirrors the result.

use std::process::ExitCode;

use tracing_subscriber::{fmt, EnvFilter};

fn main() -> ExitCode {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    match aiterm::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!(error = %err, "aiterm exited with error");
            for cause in err.chain().skip(1) {
                tracing::error!(cause = %cause, "caused by");
            }
            ExitCode::FAILURE
        }
    }
}
