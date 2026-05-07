//! PTY layer — spawn a shell via [`portable_pty`], expose async read/write
//! halves, and surface resize events so the grid can reflow.
//!
//! # Overview
//!
//! [`Pty::spawn`] opens a native PTY pair, starts the requested shell inside
//! it, and launches a background [`tokio::task::spawn_blocking`] worker that
//! continuously drains the master reader (blocking std I/O) and forwards
//! byte chunks over an unbounded [`tokio::sync::mpsc`] channel.  Callers
//! consume the receiver to feed raw VT bytes into the parser.
//!
//! # Example
//!
//! ```no_run
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let (mut pty, mut rx) = aiterm::pty::Pty::spawn(24, 80, "/bin/sh")?;
//! pty.write(b"echo hello\n")?;
//! while let Some(chunk) = rx.recv().await {
//!     // raw VT bytes — feed into the parser
//!     let _ = chunk;
//! }
//! # Ok(())
//! # }
//! ```

use std::io::Read;

use anyhow::Context as _;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use tokio::sync::mpsc;

/// A running PTY session wrapping a native pseudoterminal pair.
///
/// Dropping this struct sends SIGHUP to the child process via [`Drop`].
pub struct Pty {
    /// Master side of the PTY (used for resize).
    master: Box<dyn portable_pty::MasterPty + Send>,
    /// Handle to the spawned child process.
    child: Box<dyn portable_pty::Child + Send + Sync>,
    /// Write half of the master PTY (simulates keyboard input).
    writer: Box<dyn std::io::Write + Send>,
}

impl Pty {
    /// Spawn `shell` inside a new PTY with the given initial dimensions.
    ///
    /// Returns:
    /// - A [`Pty`] handle for writing and resizing.
    /// - An unbounded [`mpsc::UnboundedReceiver`] that yields raw byte chunks
    ///   read from the shell.  The receiver closes when the child exits or an
    ///   I/O error occurs.
    ///
    /// # Arguments
    ///
    /// * `rows` / `cols` — initial terminal size in characters.
    /// * `shell` — absolute path to the shell binary, e.g. `"/bin/sh"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the PTY cannot be opened, or if the shell fails to
    /// spawn.
    pub fn spawn(
        rows: u16,
        cols: u16,
        shell: &str,
    ) -> anyhow::Result<(Self, mpsc::UnboundedReceiver<Vec<u8>>)> {
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("openpty failed")?;

        // Destructure so we can move master and slave independently.
        let (slave, master) = (pair.slave, pair.master);

        let mut cmd = CommandBuilder::new(shell);
        cmd.env("TERM", "xterm-256color");
        // Pass through essential ambient variables so the shell has a sane
        // environment.  Failures are intentionally silenced — the variable
        // simply won't be forwarded if it is unset.
        for var in ["PATH", "HOME", "USER"] {
            if let Ok(val) = std::env::var(var) {
                cmd.env(var, val);
            }
        }

        let child = slave.spawn_command(cmd).context("spawn_command failed")?;
        // Drop the slave end; we no longer need it now that the child is
        // running.  The master is sufficient for all subsequent I/O.
        drop(slave);

        // Obtain a *blocking* reader for the master side.  portable-pty
        // readers are plain std::io::Read — they are NOT tokio AsyncRead, so
        // we must not poll them from an async context.  Use spawn_blocking.
        let mut reader = master
            .try_clone_reader()
            .context("try_clone_reader failed")?;

        let (tx, rx) = mpsc::unbounded_channel::<Vec<u8>>();

        tokio::task::spawn_blocking(move || {
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,           // EOF — child exited
                    Ok(n) => {
                        if tx.send(buf[..n].to_vec()).is_err() {
                            break; // receiver was dropped; nothing to forward
                        }
                    }
                    Err(_) => break, // I/O error (e.g. EIO after child exit)
                }
            }
        });

        let writer = master.take_writer().context("take_writer failed")?;

        Ok((Self { master, child, writer }, rx))
    }

    /// Write raw bytes to the PTY, simulating keyboard input.
    ///
    /// The write is flushed immediately so the shell sees it without delay.
    pub fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        self.writer.write_all(bytes)?;
        self.writer.flush()
    }

    /// Notify the kernel (and thus the child process) that the terminal has
    /// been resized to `rows` × `cols` characters.
    pub fn resize(&mut self, rows: u16, cols: u16) -> anyhow::Result<()> {
        self.master
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("resize failed")
    }
}

impl Drop for Pty {
    /// Kill the child process when the [`Pty`] is dropped.
    ///
    /// Errors are silently ignored — the process may have already exited.
    fn drop(&mut self) {
        self.child.kill().ok();
    }
}
