//! VT escape parser (M2 scope).
//!
//! TODO(M2): Wrap `vte::Parser` with a `Performer` that mutates [`crate::grid`]
//! state. Targets a useful subset of xterm sequences first (SGR, CSI cursor
//! moves, ED/EL, OSC 0/2 title, OSC 8 hyperlinks).
