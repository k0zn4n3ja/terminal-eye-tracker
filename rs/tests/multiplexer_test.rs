//! Unit tests for multiplexer parsing helpers and pane-id validators.
//!
//! These tests cover pure functions only — no real subprocesses are spawned.

use tmux_eyes::multiplexer::{parse_list_panes, parse_wezterm_list};
use tmux_eyes::multiplexer::tmux::validate_pane_id as tmux_validate;
use tmux_eyes::multiplexer::wezterm::validate_pane_id as wezterm_validate;

// ---------------------------------------------------------------------------
// parse_list_panes
// ---------------------------------------------------------------------------

#[test]
fn parse_list_panes_canonical_two_panes() {
    let input = "%0 0 0 80 24 1\n%1 80 0 40 24 0\n";
    let map = parse_list_panes(input).expect("should parse");
    assert_eq!(map.len(), 2);

    let p0 = map.get("%0").expect("%0 missing");
    assert_eq!(p0.pane_id, "%0");
    assert_eq!(p0.left, 0);
    assert_eq!(p0.top, 0);
    assert_eq!(p0.width, 80);
    assert_eq!(p0.height, 24);
    assert!(p0.active);

    let p1 = map.get("%1").expect("%1 missing");
    assert_eq!(p1.pane_id, "%1");
    assert_eq!(p1.left, 80);
    assert_eq!(p1.top, 0);
    assert_eq!(p1.width, 40);
    assert_eq!(p1.height, 24);
    assert!(!p1.active);
}

#[test]
fn parse_list_panes_whitespace_tolerance() {
    // Leading/trailing whitespace on lines and blank lines should be handled.
    let input = "  \n%0 0 0 80 24 1  \n\n  %1 80 0 40 24 0\n  \n";
    let map = parse_list_panes(input).expect("should parse");
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("%0"));
    assert!(map.contains_key("%1"));
}

#[test]
fn parse_list_panes_empty_input() {
    let map = parse_list_panes("").expect("empty should return empty map");
    assert!(map.is_empty());
}

#[test]
fn parse_list_panes_whitespace_only() {
    let map = parse_list_panes("   \n\n\t\n").expect("whitespace-only should return empty map");
    assert!(map.is_empty());
}

#[test]
fn parse_list_panes_malformed_returns_error() {
    // Missing the active flag field.
    let input = "%0 0 0 80 24\n";
    assert!(
        parse_list_panes(input).is_err(),
        "malformed line should produce Err"
    );
}

// ---------------------------------------------------------------------------
// parse_wezterm_list
// ---------------------------------------------------------------------------

#[test]
fn parse_wezterm_list_canonical_two_panes() {
    let json = r#"[
        {"pane_id":0,"left_col":0,"top_row":0,"size":{"cols":80,"rows":24},"is_active":true},
        {"pane_id":1,"left_col":80,"top_row":0,"size":{"cols":40,"rows":24},"is_active":false}
    ]"#;
    let map = parse_wezterm_list(json).expect("should parse");
    assert_eq!(map.len(), 2);

    let p0 = map.get("0").expect("pane 0 missing");
    assert_eq!(p0.pane_id, "0");
    assert_eq!(p0.left, 0);
    assert_eq!(p0.top, 0);
    assert_eq!(p0.width, 80);
    assert_eq!(p0.height, 24);
    assert!(p0.active);

    let p1 = map.get("1").expect("pane 1 missing");
    assert_eq!(p1.pane_id, "1");
    assert_eq!(p1.left, 80);
    assert_eq!(p1.top, 0);
    assert_eq!(p1.width, 40);
    assert_eq!(p1.height, 24);
    assert!(!p1.active);
}

#[test]
fn parse_wezterm_list_empty_array() {
    let map = parse_wezterm_list("[]").expect("empty array should return empty map");
    assert!(map.is_empty());
}

#[test]
fn parse_wezterm_list_malformed_json() {
    assert!(
        parse_wezterm_list("{not valid json}").is_err(),
        "malformed JSON should produce Err"
    );
}

// ---------------------------------------------------------------------------
// tmux::validate_pane_id
// ---------------------------------------------------------------------------

#[test]
fn tmux_validate_accepts_valid_ids() {
    assert!(tmux_validate("%0").is_ok());
    assert!(tmux_validate("%1").is_ok());
    assert!(tmux_validate("%999").is_ok());
}

#[test]
fn tmux_validate_rejects_bare_percent() {
    assert!(tmux_validate("%").is_err(), "bare % should be rejected");
}

#[test]
fn tmux_validate_rejects_numeric_only() {
    assert!(tmux_validate("1").is_err(), "numeric-only should be rejected");
}

#[test]
fn tmux_validate_rejects_injection_attempt() {
    assert!(
        tmux_validate("%1; rm -rf /").is_err(),
        "injection attempt should be rejected"
    );
}

#[test]
fn tmux_validate_rejects_empty_string() {
    assert!(tmux_validate("").is_err(), "empty string should be rejected");
}

// ---------------------------------------------------------------------------
// wezterm::validate_pane_id
// ---------------------------------------------------------------------------

#[test]
fn wezterm_validate_accepts_valid_ids() {
    assert!(wezterm_validate("0").is_ok());
    assert!(wezterm_validate("1").is_ok());
    assert!(wezterm_validate("999").is_ok());
}

#[test]
fn wezterm_validate_rejects_tmux_style() {
    assert!(
        wezterm_validate("%1").is_err(),
        "tmux-style %1 should be rejected"
    );
}

#[test]
fn wezterm_validate_rejects_injection_attempt() {
    assert!(
        wezterm_validate("1; rm -rf /").is_err(),
        "injection attempt should be rejected"
    );
}

#[test]
fn wezterm_validate_rejects_empty_string() {
    assert!(
        wezterm_validate("").is_err(),
        "empty string should be rejected"
    );
}
