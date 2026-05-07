//! VT escape parser (M2a scope).
//!
//! Wraps [`vte::Parser`] with a minimal [`Performer`] that mutates a
//! [`Grid`] in response to printable characters, common control bytes,
//! and the small set of CSI sequences sufficient to render shell output.
//!
//! Out of scope for M2a (deferred):
//!   - SGR colour / attribute application (CSI `m` is a no-op for now)
//!   - OSC sequences (titles, hyperlinks)
//!   - DCS (`hook` / `put` / `unhook`)
//!   - ED/EL with parameter `1` (erase to start of display/line)
//!
//! The parser **never panics** on input: unrecognised sequences are
//! debug-logged and ignored.
//!
//! # Example
//! ```
//! use aiterm::grid::Grid;
//! use aiterm::parser::Parser;
//!
//! let mut g = Grid::new(4, 8);
//! let mut p = Parser::new();
//! p.feed(&mut g, b"hi\r\nok");
//! assert_eq!(g.cell_at(0, 0).unwrap().ch, 'h');
//! assert_eq!(g.cell_at(1, 0).unwrap().ch, 'o');
//! ```

use crate::grid::Grid;
use vte::{Params, Perform};

/// Stateful VT escape parser. Hold one of these per terminal session and
/// feed it byte slices as they arrive from the PTY via [`Self::feed`].
pub struct Parser {
    vte: vte::Parser,
}

impl Parser {
    /// Build a fresh parser with empty internal state.
    pub fn new() -> Self {
        Self {
            vte: vte::Parser::new(),
        }
    }

    /// Feed a chunk of bytes from the PTY. Each byte is dispatched through
    /// the underlying [`vte::Parser`] and applied to `grid`. Safe to call
    /// with empty slices and partial escape sequences — `vte` keeps state
    /// across calls.
    pub fn feed(&mut self, grid: &mut Grid, bytes: &[u8]) {
        let mut perf = Performer { grid };
        for &b in bytes {
            self.vte.advance(&mut perf, b);
        }
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge from `vte`'s parser callbacks to grid mutations. Constructed
/// fresh per [`Parser::feed`] call so we always hold a unique mutable
/// borrow of the grid for the duration of dispatch.
struct Performer<'a> {
    grid: &'a mut Grid,
}

impl Performer<'_> {
    /// First parameter from a CSI argument list, with VT-style defaulting:
    /// missing or zero collapses to `default`. Suitable for cursor-move
    /// counts where `0` is conventionally treated as `1`.
    fn first_param_or(params: &Params, default: u16) -> u16 {
        params
            .iter()
            .next()
            .and_then(|p| p.first().copied())
            .filter(|&v| v != 0)
            .unwrap_or(default)
    }

    /// First parameter as a literal selector — used for ED/EL where `0`
    /// is a meaningful mode, distinct from "absent". Missing → `default`.
    fn first_param_literal(params: &Params, default: u16) -> u16 {
        params
            .iter()
            .next()
            .and_then(|p| p.first().copied())
            .unwrap_or(default)
    }
}

impl Perform for Performer<'_> {
    fn print(&mut self, c: char) {
        self.grid.write_char(c);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            0x08 => self.grid.backspace(),
            0x09 => self.grid.tab(),
            0x0A => self.grid.line_feed(),
            0x0D => self.grid.carriage_return(),
            other => tracing::debug!(byte = other, "unhandled control byte"),
        }
    }

    fn csi_dispatch(
        &mut self,
        params: &Params,
        intermediates: &[u8],
        ignore: bool,
        action: char,
    ) {
        if ignore {
            tracing::debug!(?params, ?intermediates, ?action, "vte CSI marked ignore");
            return;
        }
        // CSI with intermediates (private markers like '?', '>') are mode
        // toggles or vendor extensions we do not yet handle. Log + bail
        // rather than risk misinterpreting the parameters.
        if !intermediates.is_empty() {
            tracing::debug!(
                ?intermediates,
                ?action,
                ?params,
                "unhandled CSI with intermediates"
            );
            return;
        }

        match action {
            'A' => self.grid.cursor_up(Self::first_param_or(params, 1)),
            'B' => self.grid.cursor_down(Self::first_param_or(params, 1)),
            'C' => self.grid.cursor_right(Self::first_param_or(params, 1)),
            'D' => self.grid.cursor_left(Self::first_param_or(params, 1)),
            'H' | 'f' => {
                // CSI cursor position is 1-based; missing fields default
                // to 1. We collapse `0` to `1` to match xterm behaviour.
                let mut iter = params.iter();
                let row = iter
                    .next()
                    .and_then(|p| p.first().copied())
                    .filter(|&v| v != 0)
                    .unwrap_or(1);
                let col = iter
                    .next()
                    .and_then(|p| p.first().copied())
                    .filter(|&v| v != 0)
                    .unwrap_or(1);
                self.grid
                    .cursor_move_to(row.saturating_sub(1), col.saturating_sub(1));
            }
            'J' => match Self::first_param_literal(params, 0) {
                0 => self.grid.erase_display_to_end(),
                1 => {
                    // TODO(M2c): erase from start of display to cursor.
                    tracing::debug!("ED 1 (erase to start) not implemented in M2a");
                }
                2 | 3 => self.grid.erase_display_all(),
                n => tracing::debug!(param = n, "unhandled ED parameter"),
            },
            'K' => match Self::first_param_literal(params, 0) {
                0 => self.grid.erase_line_to_end(),
                1 => {
                    // TODO(M2c): erase from start of line to cursor.
                    tracing::debug!("EL 1 (erase to start) not implemented in M2a");
                }
                2 => self.grid.erase_line_all(),
                n => tracing::debug!(param = n, "unhandled EL parameter"),
            },
            'm' => {
                // SGR — colours / attributes. M2a leaves cells at default
                // colours; M2c will plumb attributes through the grid.
                // Intentional no-op.
            }
            other => {
                tracing::debug!(action = ?other, ?params, "unhandled CSI");
            }
        }
    }

    fn osc_dispatch(&mut self, params: &[&[u8]], _bell_terminated: bool) {
        tracing::debug!(param_count = params.len(), "unhandled OSC");
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], _ignore: bool, byte: u8) {
        tracing::debug!(?intermediates, byte, "unhandled ESC");
    }

    fn hook(&mut self, params: &Params, intermediates: &[u8], _ignore: bool, action: char) {
        tracing::debug!(?params, ?intermediates, ?action, "unhandled DCS hook");
    }

    fn put(&mut self, byte: u8) {
        tracing::debug!(byte, "unhandled DCS put");
    }

    fn unhook(&mut self) {
        tracing::debug!("unhandled DCS unhook");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Cursor;

    fn parse(grid: &mut Grid, bytes: &[u8]) {
        let mut p = Parser::new();
        p.feed(grid, bytes);
    }

    // ---- printable + execute ---------------------------------------------

    #[test]
    fn print_advances_cursor_one_per_char() {
        let mut g = Grid::new(2, 8);
        parse(&mut g, b"abc");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'b');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
        assert_eq!(g.cursor(), Cursor { row: 0, col: 3 });
    }

    #[test]
    fn cr_lf_returns_to_col0_and_advances_row() {
        let mut g = Grid::new(3, 5);
        parse(&mut g, b"hi\r\nx");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'h');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'i');
        assert_eq!(g.cell_at(1, 0).unwrap().ch, 'x');
        assert_eq!(g.cursor(), Cursor { row: 1, col: 1 });
    }

    #[test]
    fn backspace_via_execute_decrements_column() {
        let mut g = Grid::new(1, 5);
        parse(&mut g, b"ab\x08");
        assert_eq!(g.cursor(), Cursor { row: 0, col: 1 });
    }

    #[test]
    fn tab_via_execute_jumps_to_next_tab_stop() {
        let mut g = Grid::new(1, 40);
        parse(&mut g, b"\t");
        assert_eq!(g.cursor().col, 8);
    }

    #[test]
    fn unknown_control_byte_is_ignored_no_panic() {
        let mut g = Grid::new(1, 5);
        // 0x07 BEL is not in our handled set.
        parse(&mut g, &[0x07]);
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    // ---- CSI cursor moves -------------------------------------------------

    #[test]
    fn csi_cuf_with_explicit_count_moves_right() {
        let mut g = Grid::new(1, 20);
        parse(&mut g, b"\x1b[5C");
        assert_eq!(g.cursor().col, 5);
    }

    #[test]
    fn csi_cub_default_count_moves_left_by_one() {
        let mut g = Grid::new(1, 10);
        parse(&mut g, b"\x1b[5C\x1b[D");
        assert_eq!(g.cursor().col, 4);
    }

    #[test]
    fn csi_cuu_with_default_param_moves_up_one() {
        let mut g = Grid::new(5, 5);
        parse(&mut g, b"\x1b[3B\x1b[A");
        assert_eq!(g.cursor().row, 2);
    }

    #[test]
    fn csi_cud_with_explicit_count_moves_down() {
        let mut g = Grid::new(5, 5);
        parse(&mut g, b"\x1b[3B");
        assert_eq!(g.cursor().row, 3);
    }

    #[test]
    fn csi_cup_h_no_params_moves_to_origin() {
        let mut g = Grid::new(5, 5);
        parse(&mut g, b"\x1b[3B\x1b[2C\x1b[H");
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    #[test]
    fn csi_cup_h_full_params_is_one_based() {
        let mut g = Grid::new(20, 30);
        parse(&mut g, b"\x1b[10;20H");
        assert_eq!(g.cursor(), Cursor { row: 9, col: 19 });
    }

    #[test]
    fn csi_cup_f_alias_behaves_like_h() {
        let mut g = Grid::new(20, 30);
        parse(&mut g, b"\x1b[5;7f");
        assert_eq!(g.cursor(), Cursor { row: 4, col: 6 });
    }

    #[test]
    fn csi_cup_with_zero_params_defaults_to_one() {
        // ESC[0;0H should behave as ESC[1;1H per xterm convention.
        let mut g = Grid::new(5, 5);
        parse(&mut g, b"\x1b[3;3H\x1b[0;0H");
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    #[test]
    fn csi_cup_with_only_row_param_uses_col_default() {
        let mut g = Grid::new(10, 10);
        parse(&mut g, b"\x1b[5H");
        assert_eq!(g.cursor(), Cursor { row: 4, col: 0 });
    }

    #[test]
    fn csi_cursor_move_with_zero_step_is_treated_as_one() {
        // ESC[0C should still move one column right (xterm-compat).
        let mut g = Grid::new(1, 10);
        parse(&mut g, b"\x1b[0C");
        assert_eq!(g.cursor().col, 1);
    }

    // ---- CSI ED / EL ------------------------------------------------------

    #[test]
    fn csi_ed_default_clears_to_end() {
        let mut g = Grid::new(2, 3);
        parse(&mut g, b"abcDEF");
        // Move to (0,1); ESC[J → erase from there to end.
        parse(&mut g, b"\x1b[1;2H\x1b[J");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, ' ');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, ' ');
        assert_eq!(g.cell_at(1, 0).unwrap().ch, ' ');
    }

    #[test]
    fn csi_ed_2_erases_entire_display() {
        let mut g = Grid::new(2, 3);
        parse(&mut g, b"abcDEF");
        parse(&mut g, b"\x1b[2J");
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(g.cell_at(r, c).unwrap().ch, ' ');
            }
        }
    }

    #[test]
    fn csi_ed_3_erases_entire_display_too() {
        // xterm extension; we treat 3 the same as 2 for M2a.
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"abc\x1b[3J");
        for c in 0..3 {
            assert_eq!(g.cell_at(0, c).unwrap().ch, ' ');
        }
    }

    #[test]
    fn csi_ed_1_is_logged_no_panic_no_state_change() {
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"abc\x1b[1;2H\x1b[1J");
        // ED 1 is a TODO; cells must remain unchanged.
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'b');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
    }

    #[test]
    fn csi_ed_unknown_param_is_logged_no_panic() {
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"abc\x1b[9J");
        // No state change.
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
    }

    #[test]
    fn csi_el_default_clears_to_eol() {
        let mut g = Grid::new(1, 5);
        parse(&mut g, b"abcde");
        // Move to col 2 (1-based: 3), then EL.
        parse(&mut g, b"\x1b[1;3H\x1b[K");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'b');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, ' ');
        assert_eq!(g.cell_at(0, 3).unwrap().ch, ' ');
        assert_eq!(g.cell_at(0, 4).unwrap().ch, ' ');
    }

    #[test]
    fn csi_el_2_clears_entire_line() {
        let mut g = Grid::new(2, 3);
        parse(&mut g, b"abcDEF");
        parse(&mut g, b"\x1b[1;1H\x1b[2K");
        for c in 0..3 {
            assert_eq!(g.cell_at(0, c).unwrap().ch, ' ');
        }
        // Row 1 untouched.
        assert_eq!(g.cell_at(1, 0).unwrap().ch, 'D');
    }

    #[test]
    fn csi_el_1_is_logged_no_panic_no_state_change() {
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"abc\x1b[1;2H\x1b[1K");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'b');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
    }

    #[test]
    fn csi_el_unknown_param_is_logged_no_panic() {
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"abc\x1b[9K");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
    }

    // ---- SGR is no-op (M2a) ----------------------------------------------

    #[test]
    fn csi_sgr_does_not_panic_and_leaves_grid_unchanged() {
        let mut g = Grid::new(1, 5);
        parse(&mut g, b"\x1b[1;31m");
        // No printable produced, no cursor move.
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
        // Default cells everywhere.
        for c in 0..5 {
            assert_eq!(g.cell_at(0, c).unwrap().ch, ' ');
        }
    }

    #[test]
    fn csi_sgr_does_not_apply_attributes_in_m2a() {
        // After SGR, written cells should still have default colours/attrs.
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"\x1b[31mX");
        let cell = g.cell_at(0, 0).unwrap();
        assert_eq!(cell.ch, 'X');
        // Crucially: M2a does NOT propagate SGR yet.
        assert_eq!(cell.fg, crate::grid::DEFAULT_FG);
        assert_eq!(cell.bg, crate::grid::DEFAULT_BG);
        assert_eq!(cell.attrs, 0);
    }

    #[test]
    fn csi_sgr_reset_alone_is_no_panic() {
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"\x1b[m");
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    // ---- private modes / unknowns ---------------------------------------

    #[test]
    fn private_csi_question_mark_is_ignored_no_panic() {
        // ESC[?47h — alt-screen toggle, not handled. Must not panic and
        // must leave the grid unchanged.
        let mut g = Grid::new(2, 3);
        parse(&mut g, b"hi\x1b[?47h");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'h');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'i');
        assert_eq!(g.cursor(), Cursor { row: 0, col: 2 });
    }

    #[test]
    fn unknown_csi_action_is_logged_no_panic() {
        // ESC[5X — ECH-ish, not handled in M2a.
        let mut g = Grid::new(1, 5);
        parse(&mut g, b"abc\x1b[5X");
        // State unchanged: characters remain.
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
    }

    #[test]
    fn osc_sequence_is_ignored_no_panic() {
        // OSC 0 ; "title" BEL — set window title.
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"\x1b]0;hi\x07X");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'X');
    }

    #[test]
    fn esc_sequence_is_ignored_no_panic() {
        // ESC c — RIS (full reset). We log + ignore.
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"a\x1bcb");
        // 'a' was printed; 'b' was printed after the ESC c was consumed.
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'b');
    }

    #[test]
    fn dcs_sequence_is_ignored_no_panic() {
        // ESC P ... ESC \  — DCS string. Hook + put + unhook all log+ignore.
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"\x1bPq#0\x1b\\X");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'X');
    }

    // ---- parser plumbing -------------------------------------------------

    #[test]
    fn default_constructs_a_usable_parser() {
        let mut p = Parser::default();
        let mut g = Grid::new(1, 3);
        p.feed(&mut g, b"ok");
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'o');
    }

    #[test]
    fn feed_is_safe_on_empty_input() {
        let mut g = Grid::new(1, 3);
        parse(&mut g, b"");
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    #[test]
    fn feed_can_be_called_repeatedly_with_split_sequences() {
        // Feeding a single CSI in two halves must still apply correctly:
        // vte holds parser state across calls.
        let mut g = Grid::new(5, 5);
        let mut p = Parser::new();
        p.feed(&mut g, b"\x1b[3;");
        p.feed(&mut g, b"4H");
        assert_eq!(g.cursor(), Cursor { row: 2, col: 3 });
    }

    // ---- helper coverage -------------------------------------------------

    #[test]
    fn first_param_or_uses_default_when_absent() {
        let params = Params::default();
        assert_eq!(Performer::first_param_or(&params, 1), 1);
        assert_eq!(Performer::first_param_or(&params, 7), 7);
    }

    #[test]
    fn first_param_literal_uses_default_when_absent() {
        let params = Params::default();
        assert_eq!(Performer::first_param_literal(&params, 0), 0);
        assert_eq!(Performer::first_param_literal(&params, 5), 5);
    }
}
