//! Integration tests for the PTY layer.
//!
//! These tests require a Unix environment (they exercise `/bin/sh` directly)
//! and are gated with `#[cfg(unix)]`.

#[cfg(unix)]
mod unix {
    use std::time::Duration;
    use tokio::time::timeout;

    /// Smoke-test: spawn `/bin/sh`, send `echo hello\n`, and verify that the
    /// string "hello" appears in the output within 5 seconds.
    #[tokio::test]
    async fn echo_hello() {
        let (mut pty, mut rx) =
            aiterm::pty::Pty::spawn(24, 80, "/bin/sh").expect("spawn failed");

        // Give the shell a moment to emit its prompt before we send a command,
        // then write the command.
        pty.write(b"echo hello\n").expect("write failed");

        let mut collected = Vec::<u8>::new();

        let result = timeout(Duration::from_secs(5), async {
            while let Some(chunk) = rx.recv().await {
                collected.extend_from_slice(&chunk);
                // Check for "hello" in the accumulated output.
                if collected.windows(5).any(|w| w == b"hello") {
                    return true;
                }
            }
            false
        })
        .await;

        assert!(
            result.unwrap_or(false),
            "did not see 'hello' in PTY output within 5 s; got: {:?}",
            String::from_utf8_lossy(&collected)
        );
    }
}
