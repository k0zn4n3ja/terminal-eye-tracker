//! Smoke test: verify the public crate entry points exist and have the
//! expected types.
//!
//! We do NOT actually call [`aiterm::run`] — that opens a window and needs
//! a display, which is unavailable in CI / `cargo test`. The function-pointer
//! coercion below fails at compile time if `aiterm::run`'s signature ever
//! drifts away from `fn() -> anyhow::Result<()>`, which is exactly the
//! breakage we want to catch early.

use aiterm::config::AiTermConfig;

#[test]
fn run_has_expected_signature() {
    // Coerce to a function pointer with the documented signature; the test
    // is "this compiles". The runtime cost is just storing a pointer.
    let _entry: fn() -> anyhow::Result<()> = aiterm::run;
}

#[test]
fn config_module_is_publicly_reachable() {
    // The library re-exports config so downstream binaries can build a
    // config without depending on the internal module path. If someone
    // demotes the visibility this test stops compiling.
    let cfg = AiTermConfig::with_env_getter(|_| Some("/bin/sh".to_string()));
    assert_eq!(cfg.default_shell, "/bin/sh");
}

#[test]
fn placeholder_modules_are_publicly_reachable() {
    // M2+ stub modules — confirmed reachable through the public crate root.
    // No items inside them yet, so this is purely a "the path resolves"
    // check that fails to compile if anyone accidentally hides one.
    #[allow(unused_imports)]
    use aiterm::{block, grid, llm, parser, pty};
}
