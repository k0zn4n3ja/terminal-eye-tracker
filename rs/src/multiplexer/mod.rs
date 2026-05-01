//! Terminal-multiplexer IPC clients: tmux (control-mode) and wezterm (CLI).
//!
//! Re-exports the two concrete implementations and exposes the pure parsing
//! helpers so they can be unit-tested without spawning real subprocesses.

pub mod tmux;
pub mod wezterm;

pub use tmux::parse_list_panes;
pub use wezterm::parse_wezterm_list;
