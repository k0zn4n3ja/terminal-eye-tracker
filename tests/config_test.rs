//! Integration-level tests for `aiterm::config`.
//!
//! These exercise the public surface as an external crate would see it:
//! TOML round-trips, missing-field error reporting, dependency-injected
//! environment lookup, and trait derivations. Inline `#[cfg(test)]` tests
//! in `src/config.rs` cover unit-level edges; this file covers the
//! end-to-end happy + sad paths through the public API.

use aiterm::config::AiTermConfig;

#[test]
fn default_via_public_api_has_stable_font_metadata() {
    let cfg = AiTermConfig::default();
    assert_eq!(cfg.font_family, "monospace");
    assert!((cfg.font_size - 14.0).abs() < f32::EPSILON);
    // default_shell is environment-dependent, but it must never be empty
    // when the env-getter falls back.
    assert!(!cfg.default_shell.is_empty() || cfg.default_shell == "/bin/sh");
}

#[test]
fn with_env_getter_is_pure_function_of_its_input() {
    let a = AiTermConfig::with_env_getter(|_| Some("/bin/zsh".to_string()));
    let b = AiTermConfig::with_env_getter(|_| Some("/bin/zsh".to_string()));
    assert_eq!(a.default_shell, b.default_shell);
    assert_eq!(a.font_family, b.font_family);

    let c = AiTermConfig::with_env_getter(|_| None);
    assert_eq!(c.default_shell, "/bin/sh");
}

#[test]
fn round_trip_default_to_toml_and_back() {
    // Pin the env so the round trip is deterministic across hosts.
    let original = AiTermConfig::with_env_getter(|_| Some("/bin/bash".to_string()));
    let toml = original.to_toml().expect("serialise");
    let reparsed = AiTermConfig::from_toml(&toml).expect("reparse");

    assert_eq!(reparsed.font_family, original.font_family);
    assert!((reparsed.font_size - original.font_size).abs() < f32::EPSILON);
    assert_eq!(reparsed.default_shell, original.default_shell);
}

#[test]
fn round_trip_through_explicit_toml_document() {
    let document = r#"
font_family = "Iosevka"
font_size = 12.0
default_shell = "/usr/local/bin/fish"
"#;
    let cfg = AiTermConfig::from_toml(document).expect("explicit doc parses");
    assert_eq!(cfg.font_family, "Iosevka");
    assert!((cfg.font_size - 12.0).abs() < f32::EPSILON);
    assert_eq!(cfg.default_shell, "/usr/local/bin/fish");

    let reserialised = cfg.to_toml().expect("re-serialise");
    let again = AiTermConfig::from_toml(&reserialised).expect("re-parse");
    assert_eq!(again.font_family, cfg.font_family);
    assert!((again.font_size - cfg.font_size).abs() < f32::EPSILON);
    assert_eq!(again.default_shell, cfg.default_shell);
}

#[test]
fn missing_field_errors_are_informative() {
    let cases: &[&str] = &[
        // missing font_size and default_shell
        r#"font_family = "Mono""#,
        // missing default_shell
        r#"font_family = "Mono"
font_size = 14.0"#,
        // missing font_family
        r#"font_size = 14.0
default_shell = "/bin/sh""#,
    ];
    for src in cases {
        let err = AiTermConfig::from_toml(src)
            .expect_err("partial toml should not parse")
            .to_string();
        assert!(!err.is_empty(), "error message must not be empty");
    }
}

#[test]
fn malformed_toml_returns_err_not_panic() {
    let cases: &[&str] = &[
        "[unterminated",
        "= = = =",
        "font_family = ",
        "font_family = 12 # wrong type for a string field",
    ];
    for src in cases {
        let result = AiTermConfig::from_toml(src);
        assert!(result.is_err(), "input {src:?} should not parse");
    }
}

#[test]
fn font_size_accepts_unusual_but_typed_values() {
    // We don't validate; document that fact with explicit assertions so any
    // future validation must update these tests deliberately.
    let zero = r#"
font_family = "Mono"
font_size = 0.0
default_shell = "/bin/sh"
"#;
    let cfg = AiTermConfig::from_toml(zero).expect("zero font parses");
    assert!(cfg.font_size.abs() < f32::EPSILON);
}

#[test]
fn debug_format_renders_all_fields() {
    let cfg = AiTermConfig::with_env_getter(|_| Some("/bin/dash".to_string()));
    let s = format!("{cfg:?}");
    for needle in ["font_family", "font_size", "default_shell", "monospace"] {
        assert!(s.contains(needle), "Debug output missing {needle:?}: {s}");
    }
}

#[test]
fn clone_is_independent_copy() {
    let a = AiTermConfig::with_env_getter(|_| Some("/bin/sh".to_string()));
    let b = a.clone();
    assert_eq!(a.font_family, b.font_family);
    assert_eq!(a.default_shell, b.default_shell);
    assert!((a.font_size - b.font_size).abs() < f32::EPSILON);
}
