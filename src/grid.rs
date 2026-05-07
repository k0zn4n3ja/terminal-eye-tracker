//! Terminal grid model (M2a scope).
//!
//! Minimal in-memory representation of the terminal screen sufficient to
//! render shell output. Held as a flat row-major [`Vec<Cell>`] of fixed
//! dimensions plus a [`Cursor`].
//!
//! Out of scope for M2a (deferred to later milestones):
//!   - scrollback ring buffer
//!   - alt-screen / main-screen switching
//!   - mouse selection / regions
//!   - wide-character (CJK / emoji) cell pairs
//!   - reflow on resize (we clamp + pad instead)
//!
//! ## Invariants
//!
//! - `rows >= 1` and `cols >= 1` are enforced by [`Grid::new`] and
//!   [`Grid::resize`] (they clamp zero values to one). This keeps every
//!   public method panic-free regardless of input.
//! - `cells.len() == rows * cols` at all times.
//! - `cursor.row < rows` and `cursor.col <= cols`. The cursor is allowed to
//!   sit one column past the last printable column — this is the standard
//!   "pending wrap" position used by xterm-class terminals: the next
//!   [`Grid::write_char`] wraps to the next row before placing the glyph.
//!
//! All public mutators clamp out-of-bounds inputs; nothing in this module
//! panics on user-controlled input.

/// Bold attribute bit for [`Cell::attrs`].
pub const ATTR_BOLD: u8 = 1 << 0;
/// Underline attribute bit for [`Cell::attrs`].
pub const ATTR_UNDERLINE: u8 = 1 << 1;
/// Inverse-video attribute bit for [`Cell::attrs`].
pub const ATTR_INVERSE: u8 = 1 << 2;

/// Default foreground (white, fully opaque, RGBA 0xRRGGBBAA-packed).
pub const DEFAULT_FG: u32 = 0xFFFF_FFFF;
/// Default background (black, fully opaque, RGBA 0xRRGGBBAA-packed).
pub const DEFAULT_BG: u32 = 0x0000_00FF;

/// Tab stop interval in columns. Standard VT terminals use multiples of 8.
pub const TAB_WIDTH: u16 = 8;

/// One terminal cell: a single glyph plus its colours and attributes.
///
/// Colours are 32-bit RGBA values (`0xRRGGBBAA`). Attributes are an OR of
/// the `ATTR_*` constants in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    /// The character displayed in this cell. Default is `' '` (space).
    pub ch: char,
    /// Foreground colour as packed RGBA.
    pub fg: u32,
    /// Background colour as packed RGBA.
    pub bg: u32,
    /// Attribute bitfield: OR of [`ATTR_BOLD`], [`ATTR_UNDERLINE`],
    /// [`ATTR_INVERSE`].
    pub attrs: u8,
}

impl Cell {
    /// Construct a fresh default cell (space, default colours, no attrs).
    #[inline]
    pub const fn blank() -> Self {
        Self {
            ch: ' ',
            fg: DEFAULT_FG,
            bg: DEFAULT_BG,
            attrs: 0,
        }
    }
}

impl Default for Cell {
    #[inline]
    fn default() -> Self {
        Self::blank()
    }
}

/// Cursor position. `row` is 0-indexed from the top, `col` from the left.
///
/// Note `col` may legally equal `cols` (the "pending wrap" position) even
/// though no cell sits there; the next [`Grid::write_char`] will wrap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cursor {
    /// Row index, 0-based from the top.
    pub row: u16,
    /// Column index, 0-based from the left.
    pub col: u16,
}

/// Fixed-size terminal grid. Row-major flat storage.
///
/// The grid never panics on out-of-bounds inputs — every mutator clamps.
/// Sizes are clamped to a minimum of `1 x 1` so empty grids cannot exist.
#[derive(Debug, Clone)]
pub struct Grid {
    rows: u16,
    cols: u16,
    cells: Vec<Cell>,
    cursor: Cursor,
}

impl Grid {
    /// Build a fresh grid filled with default cells. Dimensions are clamped
    /// to a minimum of `1 x 1`; the cursor starts at `(0, 0)`.
    pub fn new(rows: u16, cols: u16) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        let len = rows as usize * cols as usize;
        Self {
            rows,
            cols,
            cells: vec![Cell::default(); len],
            cursor: Cursor::default(),
        }
    }

    /// Current `(rows, cols)` dimensions.
    #[inline]
    pub fn dimensions(&self) -> (u16, u16) {
        (self.rows, self.cols)
    }

    /// Read-only borrow of the cursor.
    #[inline]
    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    /// Borrow the cell at `(row, col)`. Returns `None` if either coordinate
    /// is past the last valid index.
    #[inline]
    pub fn cell_at(&self, row: u16, col: u16) -> Option<&Cell> {
        let idx = self.index(row, col)?;
        Some(&self.cells[idx])
    }

    /// Iterate over rows as `&[Cell]` slices, top to bottom.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[Cell]> {
        let cols = self.cols as usize;
        self.cells.chunks_exact(cols)
    }

    /// Resize the grid. Existing content that fits in the new dimensions is
    /// preserved at its original `(row, col)` position; freshly exposed cells
    /// are blanked. The cursor is clamped into the new bounds.
    ///
    /// Sizes are clamped to a minimum of `1 x 1`.
    pub fn resize(&mut self, rows: u16, cols: u16) {
        let new_rows = rows.max(1);
        let new_cols = cols.max(1);
        if new_rows == self.rows && new_cols == self.cols {
            return;
        }

        let mut next = vec![Cell::default(); new_rows as usize * new_cols as usize];
        let copy_rows = self.rows.min(new_rows);
        let copy_cols = self.cols.min(new_cols);
        for r in 0..copy_rows {
            for c in 0..copy_cols {
                let src = r as usize * self.cols as usize + c as usize;
                let dst = r as usize * new_cols as usize + c as usize;
                next[dst] = self.cells[src];
            }
        }

        self.rows = new_rows;
        self.cols = new_cols;
        self.cells = next;
        // Clamp cursor strictly inside the new printable area. Resize is
        // an external event that resets the pending-wrap state — it would
        // be confusing for a freshly-resized terminal to wrap on the next
        // glyph because the previous size left col == cols.
        if self.cursor.row >= self.rows {
            self.cursor.row = self.rows - 1;
        }
        if self.cursor.col >= self.cols {
            self.cursor.col = self.cols - 1;
        }
    }

    // -- writing -------------------------------------------------------------

    /// Write a single character at the cursor with current default
    /// attributes, then advance the cursor.
    ///
    /// Behaviour matches a minimal xterm:
    /// - If the cursor is at the pending-wrap position (`col == cols`),
    ///   wrap to column 0 of the next row first (scrolling if at bottom).
    /// - After writing, advance one column. If that lands at `cols`, the
    ///   cursor stays in the pending-wrap position (no immediate scroll).
    pub fn write_char(&mut self, ch: char) {
        // Pending-wrap: move to next line before placing the glyph.
        if self.cursor.col >= self.cols {
            self.cursor.col = 0;
            self.advance_row_or_scroll();
        }
        let row = self.cursor.row;
        let col = self.cursor.col;
        if let Some(idx) = self.index(row, col) {
            self.cells[idx] = Cell {
                ch,
                fg: DEFAULT_FG,
                bg: DEFAULT_BG,
                attrs: 0,
            };
        }
        // Advance. We deliberately allow col == cols (pending-wrap).
        self.cursor.col = self.cursor.col.saturating_add(1).min(self.cols);
    }

    /// Carriage return: move cursor to column 0 of the current row.
    #[inline]
    pub fn carriage_return(&mut self) {
        self.cursor.col = 0;
    }

    /// Line feed: move cursor down one row, scrolling if at the bottom.
    /// Column is unchanged.
    pub fn line_feed(&mut self) {
        self.advance_row_or_scroll();
    }

    /// Backspace: move cursor left one column, clamped at column 0. Does
    /// not erase the cell underneath; that is the caller's job.
    pub fn backspace(&mut self) {
        if self.cursor.col > 0 {
            // If we were in the pending-wrap position, drop back to the last
            // real column rather than going past it.
            if self.cursor.col > self.cols {
                self.cursor.col = self.cols;
            }
            self.cursor.col -= 1;
        }
    }

    /// Tab: advance cursor to the next multiple of [`TAB_WIDTH`], capped
    /// at `cols - 1` (so the cursor always remains on a printable column
    /// after a tab on a non-empty grid).
    pub fn tab(&mut self) {
        let next = next_tab_stop(self.cursor.col, self.cols, TAB_WIDTH);
        self.cursor.col = next;
    }

    // -- cursor moves --------------------------------------------------------

    /// Move cursor to `(row, col)`, clamped into bounds.
    pub fn cursor_move_to(&mut self, row: u16, col: u16) {
        self.cursor.row = row.min(self.rows.saturating_sub(1));
        self.cursor.col = col.min(self.cols.saturating_sub(1));
    }

    /// Move cursor up `n` rows, clamped at row 0.
    pub fn cursor_up(&mut self, n: u16) {
        self.cursor.row = self.cursor.row.saturating_sub(n);
    }

    /// Move cursor down `n` rows, clamped at the last row.
    pub fn cursor_down(&mut self, n: u16) {
        let target = self.cursor.row.saturating_add(n);
        self.cursor.row = target.min(self.rows - 1);
    }

    /// Move cursor left `n` cols, clamped at col 0.
    pub fn cursor_left(&mut self, n: u16) {
        // If cursor is in the pending-wrap position, normalise first so the
        // step count refers to printable columns.
        if self.cursor.col > self.cols {
            self.cursor.col = self.cols;
        }
        self.cursor.col = self.cursor.col.saturating_sub(n);
    }

    /// Move cursor right `n` cols, clamped at the last printable column.
    pub fn cursor_right(&mut self, n: u16) {
        let target = self.cursor.col.saturating_add(n);
        self.cursor.col = target.min(self.cols - 1);
    }

    // -- erase ---------------------------------------------------------------

    /// Clear from cursor (inclusive) to end of the current row.
    pub fn erase_line_to_end(&mut self) {
        let row = self.cursor.row;
        let from = self.cursor.col.min(self.cols);
        for c in from..self.cols {
            if let Some(idx) = self.index(row, c) {
                self.cells[idx] = Cell::default();
            }
        }
    }

    /// Clear the entire current row.
    pub fn erase_line_all(&mut self) {
        let row = self.cursor.row;
        for c in 0..self.cols {
            if let Some(idx) = self.index(row, c) {
                self.cells[idx] = Cell::default();
            }
        }
    }

    /// Clear from cursor (inclusive) to end of screen: the rest of the
    /// current row plus all rows below.
    pub fn erase_display_to_end(&mut self) {
        self.erase_line_to_end();
        for r in (self.cursor.row + 1)..self.rows {
            for c in 0..self.cols {
                let idx = r as usize * self.cols as usize + c as usize;
                self.cells[idx] = Cell::default();
            }
        }
    }

    /// Clear the entire display.
    pub fn erase_display_all(&mut self) {
        for cell in &mut self.cells {
            *cell = Cell::default();
        }
    }

    // -- internals -----------------------------------------------------------

    /// Convert `(row, col)` to a flat index, or `None` if out of bounds.
    #[inline]
    fn index(&self, row: u16, col: u16) -> Option<usize> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        Some(row as usize * self.cols as usize + col as usize)
    }

    /// Move cursor down one row; if that would leave the grid, scroll
    /// the contents up by one row instead. Column is unchanged.
    fn advance_row_or_scroll(&mut self) {
        if self.cursor.row + 1 >= self.rows {
            self.scroll_up();
        } else {
            self.cursor.row += 1;
        }
    }

    /// Discard row 0, shift every other row up by one, blank the bottom row.
    /// Cursor row is left unchanged (it stays on the now-blank bottom row
    /// when called from [`Self::advance_row_or_scroll`]).
    pub(crate) fn scroll_up(&mut self) {
        let cols = self.cols as usize;
        // Shift rows 1..rows up by one.
        if self.rows > 1 {
            self.cells.copy_within(cols.., 0);
        }
        // Blank the last row.
        let last_row_start = (self.rows as usize - 1) * cols;
        for cell in &mut self.cells[last_row_start..last_row_start + cols] {
            *cell = Cell::default();
        }
    }
}

/// Pure helper: compute the column position after a tab from `col`, given
/// the total `cols` and tab `width`. Cursor advances to the next multiple
/// of `width`, capped at `cols - 1` so the cursor always lands on a
/// printable column.
///
/// Extracted so it can be unit-tested without constructing a [`Grid`].
pub(crate) fn next_tab_stop(col: u16, cols: u16, width: u16) -> u16 {
    debug_assert!(cols >= 1);
    let last = cols - 1;
    if width == 0 {
        // Defensive: zero width would loop forever; treat as "no tab".
        return col.min(last);
    }
    // Snap to next multiple of width, strictly greater than `col`.
    let stop = (col / width + 1) * width;
    stop.min(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Cell --------------------------------------------------------------

    #[test]
    fn cell_default_is_space_with_default_colours() {
        let c = Cell::default();
        assert_eq!(c.ch, ' ');
        assert_eq!(c.fg, DEFAULT_FG);
        assert_eq!(c.bg, DEFAULT_BG);
        assert_eq!(c.attrs, 0);
    }

    #[test]
    fn cell_blank_matches_default() {
        assert_eq!(Cell::blank(), Cell::default());
    }

    #[test]
    fn attr_constants_are_distinct_bits() {
        assert_ne!(ATTR_BOLD, 0);
        assert_ne!(ATTR_UNDERLINE, 0);
        assert_ne!(ATTR_INVERSE, 0);
        assert_eq!(ATTR_BOLD & ATTR_UNDERLINE, 0);
        assert_eq!(ATTR_BOLD & ATTR_INVERSE, 0);
        assert_eq!(ATTR_UNDERLINE & ATTR_INVERSE, 0);
    }

    // ---- construction ------------------------------------------------------

    #[test]
    fn new_clamps_zero_dimensions_to_one() {
        let g = Grid::new(0, 0);
        assert_eq!(g.dimensions(), (1, 1));
        assert_eq!(g.cell_at(0, 0).copied(), Some(Cell::default()));
    }

    #[test]
    fn new_fills_with_default_cells() {
        let g = Grid::new(3, 4);
        assert_eq!(g.dimensions(), (3, 4));
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(g.cell_at(r, c).copied(), Some(Cell::default()));
            }
        }
    }

    #[test]
    fn new_initialises_cursor_at_origin() {
        let g = Grid::new(5, 5);
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    #[test]
    fn cell_at_returns_none_out_of_bounds() {
        let g = Grid::new(2, 2);
        assert!(g.cell_at(2, 0).is_none());
        assert!(g.cell_at(0, 2).is_none());
        assert!(g.cell_at(99, 99).is_none());
    }

    #[test]
    fn iter_rows_yields_one_slice_per_row() {
        let g = Grid::new(3, 4);
        let rows: Vec<&[Cell]> = g.iter_rows().collect();
        assert_eq!(rows.len(), 3);
        for r in &rows {
            assert_eq!(r.len(), 4);
        }
    }

    // ---- write_char --------------------------------------------------------

    #[test]
    fn write_char_places_glyph_and_advances_cursor() {
        let mut g = Grid::new(2, 4);
        g.write_char('A');
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'A');
        assert_eq!(g.cursor(), Cursor { row: 0, col: 1 });
    }

    #[test]
    fn write_char_wraps_at_pending_wrap_position() {
        let mut g = Grid::new(2, 3);
        for ch in ['a', 'b', 'c'] {
            g.write_char(ch);
        }
        // Cursor is in pending-wrap (col == cols), but still on row 0.
        assert_eq!(g.cursor(), Cursor { row: 0, col: 3 });
        // Next write wraps to row 1.
        g.write_char('d');
        assert_eq!(g.cell_at(1, 0).unwrap().ch, 'd');
        assert_eq!(g.cursor(), Cursor { row: 1, col: 1 });
    }

    #[test]
    fn write_char_scrolls_when_wrapping_at_bottom() {
        let mut g = Grid::new(2, 2);
        // Fill row 0, row 1, then trigger another wrap.
        g.write_char('a');
        g.write_char('b');
        g.write_char('c');
        g.write_char('d');
        // Pending wrap at (1, 2). Next write should scroll: row 0 becomes
        // old row 1 ("cd"), row 1 starts with the new char.
        g.write_char('e');
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'c');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'd');
        assert_eq!(g.cell_at(1, 0).unwrap().ch, 'e');
        assert_eq!(g.cell_at(1, 1).unwrap().ch, ' ');
        assert_eq!(g.cursor(), Cursor { row: 1, col: 1 });
    }

    #[test]
    fn write_char_uses_default_colours_and_no_attrs() {
        let mut g = Grid::new(1, 2);
        g.write_char('Z');
        let cell = g.cell_at(0, 0).unwrap();
        assert_eq!(cell.fg, DEFAULT_FG);
        assert_eq!(cell.bg, DEFAULT_BG);
        assert_eq!(cell.attrs, 0);
    }

    // ---- carriage_return / line_feed --------------------------------------

    #[test]
    fn carriage_return_resets_column_only() {
        let mut g = Grid::new(3, 4);
        g.cursor_move_to(2, 3);
        g.carriage_return();
        assert_eq!(g.cursor(), Cursor { row: 2, col: 0 });
    }

    #[test]
    fn line_feed_advances_row() {
        let mut g = Grid::new(3, 3);
        g.cursor_move_to(0, 1);
        g.line_feed();
        assert_eq!(g.cursor(), Cursor { row: 1, col: 1 });
    }

    #[test]
    fn line_feed_at_bottom_scrolls_up() {
        let mut g = Grid::new(2, 2);
        // Lay out row 0 = "ab", row 1 = "cd".
        for ch in ['a', 'b'] {
            g.write_char(ch);
        }
        g.carriage_return();
        g.line_feed();
        for ch in ['c', 'd'] {
            g.write_char(ch);
        }
        // Cursor now at (1, 2) pending-wrap. line_feed should scroll.
        g.line_feed();
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'c');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'd');
        assert_eq!(g.cell_at(1, 0).unwrap().ch, ' ');
        assert_eq!(g.cell_at(1, 1).unwrap().ch, ' ');
        assert_eq!(g.cursor().row, 1);
    }

    // ---- backspace ---------------------------------------------------------

    #[test]
    fn backspace_decrements_column() {
        let mut g = Grid::new(1, 5);
        g.cursor_move_to(0, 3);
        g.backspace();
        assert_eq!(g.cursor(), Cursor { row: 0, col: 2 });
    }

    #[test]
    fn backspace_at_col_zero_is_noop_no_panic() {
        let mut g = Grid::new(1, 3);
        g.backspace();
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
        // Repeated calls remain a no-op.
        for _ in 0..10 {
            g.backspace();
        }
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    #[test]
    fn backspace_from_pending_wrap_lands_on_last_column() {
        let mut g = Grid::new(1, 3);
        g.write_char('a');
        g.write_char('b');
        g.write_char('c');
        // Pending-wrap at (0, 3).
        assert_eq!(g.cursor().col, 3);
        g.backspace();
        // Should drop back to col 2, the last printable column.
        assert_eq!(g.cursor(), Cursor { row: 0, col: 2 });
    }

    // ---- tab --------------------------------------------------------------

    #[test]
    fn tab_advances_to_next_multiple_of_eight() {
        let mut g = Grid::new(1, 40);
        g.cursor_move_to(0, 0);
        g.tab();
        assert_eq!(g.cursor().col, 8);
        g.tab();
        assert_eq!(g.cursor().col, 16);
    }

    #[test]
    fn tab_from_a_tab_stop_jumps_to_next_one() {
        let mut g = Grid::new(1, 40);
        g.cursor_move_to(0, 8);
        g.tab();
        assert_eq!(g.cursor().col, 16);
    }

    #[test]
    fn tab_caps_at_last_column() {
        // cols=10 → last printable col = 9. Tabs snap to multiples of 8
        // and cap at 9.
        let mut g = Grid::new(1, 10);
        g.cursor_move_to(0, 5);
        g.tab();
        // Next 8-boundary is 8 — still inside the grid, so no cap needed yet.
        assert_eq!(g.cursor().col, 8);
        g.tab();
        // Next 8-boundary would be 16; cap to last printable col (9).
        assert_eq!(g.cursor().col, 9);
        // Already at the cap: another tab is a no-op.
        g.tab();
        assert_eq!(g.cursor().col, 9);
    }

    #[test]
    fn next_tab_stop_helper_handles_edges() {
        assert_eq!(next_tab_stop(0, 80, 8), 8);
        assert_eq!(next_tab_stop(7, 80, 8), 8);
        assert_eq!(next_tab_stop(8, 80, 8), 16);
        // Past last tab boundary: cap.
        assert_eq!(next_tab_stop(75, 80, 8), 79);
        // Zero width: defensive no-op (returns clamped col).
        assert_eq!(next_tab_stop(3, 10, 0), 3);
        assert_eq!(next_tab_stop(20, 10, 0), 9);
        // cols == 1: cap is 0 regardless of starting col.
        assert_eq!(next_tab_stop(0, 1, 8), 0);
    }

    // ---- erase ------------------------------------------------------------

    fn fill(g: &mut Grid, ch: char) {
        // Fill every cell with `ch` deterministically (no scrolling).
        let (rows, cols) = g.dimensions();
        for r in 0..rows {
            for c in 0..cols {
                g.cursor_move_to(r, c);
                g.write_char(ch);
            }
        }
    }

    #[test]
    fn erase_line_to_end_clears_only_from_cursor() {
        let mut g = Grid::new(2, 5);
        fill(&mut g, 'X');
        g.cursor_move_to(0, 2);
        g.erase_line_to_end();
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'X');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'X');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, ' ');
        assert_eq!(g.cell_at(0, 4).unwrap().ch, ' ');
        // Other rows untouched.
        for c in 0..5 {
            assert_eq!(g.cell_at(1, c).unwrap().ch, 'X');
        }
    }

    #[test]
    fn erase_line_all_clears_only_current_row() {
        let mut g = Grid::new(2, 3);
        fill(&mut g, 'X');
        g.cursor_move_to(1, 1);
        g.erase_line_all();
        for c in 0..3 {
            assert_eq!(g.cell_at(0, c).unwrap().ch, 'X');
            assert_eq!(g.cell_at(1, c).unwrap().ch, ' ');
        }
    }

    #[test]
    fn erase_display_to_end_clears_rest_of_row_and_below() {
        let mut g = Grid::new(3, 3);
        fill(&mut g, 'X');
        g.cursor_move_to(1, 1);
        g.erase_display_to_end();
        // Row 0 untouched.
        for c in 0..3 {
            assert_eq!(g.cell_at(0, c).unwrap().ch, 'X');
        }
        // Row 1: col 0 untouched, cols 1..3 cleared.
        assert_eq!(g.cell_at(1, 0).unwrap().ch, 'X');
        assert_eq!(g.cell_at(1, 1).unwrap().ch, ' ');
        assert_eq!(g.cell_at(1, 2).unwrap().ch, ' ');
        // Row 2 fully cleared.
        for c in 0..3 {
            assert_eq!(g.cell_at(2, c).unwrap().ch, ' ');
        }
    }

    #[test]
    fn erase_display_all_clears_everything() {
        let mut g = Grid::new(3, 3);
        fill(&mut g, 'X');
        g.cursor_move_to(2, 2);
        g.erase_display_all();
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(g.cell_at(r, c).unwrap().ch, ' ');
            }
        }
    }

    #[test]
    fn erase_line_to_end_at_pending_wrap_is_noop() {
        let mut g = Grid::new(1, 3);
        for ch in ['a', 'b', 'c'] {
            g.write_char(ch);
        }
        // Cursor at (0, 3) pending-wrap.
        g.erase_line_to_end();
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'b');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
    }

    // ---- cursor moves -----------------------------------------------------

    #[test]
    fn cursor_move_to_clamps_into_bounds() {
        let mut g = Grid::new(3, 4);
        g.cursor_move_to(99, 99);
        assert_eq!(g.cursor(), Cursor { row: 2, col: 3 });
        g.cursor_move_to(0, 0);
        assert_eq!(g.cursor(), Cursor { row: 0, col: 0 });
    }

    #[test]
    fn cursor_up_clamps_at_zero() {
        let mut g = Grid::new(3, 3);
        g.cursor_move_to(1, 0);
        g.cursor_up(99);
        assert_eq!(g.cursor().row, 0);
    }

    #[test]
    fn cursor_down_clamps_at_last_row() {
        let mut g = Grid::new(3, 3);
        g.cursor_down(99);
        assert_eq!(g.cursor().row, 2);
    }

    #[test]
    fn cursor_left_clamps_at_zero() {
        let mut g = Grid::new(3, 3);
        g.cursor_move_to(0, 1);
        g.cursor_left(99);
        assert_eq!(g.cursor().col, 0);
    }

    #[test]
    fn cursor_left_normalises_pending_wrap() {
        let mut g = Grid::new(1, 4);
        for ch in ['a', 'b', 'c', 'd'] {
            g.write_char(ch);
        }
        // Pending-wrap at col=4. cursor_left(1) should land at col=3.
        g.cursor_left(1);
        assert_eq!(g.cursor().col, 3);
    }

    #[test]
    fn cursor_right_clamps_at_last_col() {
        let mut g = Grid::new(3, 3);
        g.cursor_right(99);
        assert_eq!(g.cursor().col, 2);
    }

    #[test]
    fn cursor_moves_zero_step_is_noop() {
        let mut g = Grid::new(3, 3);
        g.cursor_move_to(1, 1);
        g.cursor_up(0);
        g.cursor_down(0);
        g.cursor_left(0);
        g.cursor_right(0);
        assert_eq!(g.cursor(), Cursor { row: 1, col: 1 });
    }

    // ---- resize -----------------------------------------------------------

    #[test]
    fn resize_preserves_content_that_fits() {
        let mut g = Grid::new(2, 3);
        for ch in ['a', 'b', 'c'] {
            g.write_char(ch);
        }
        g.carriage_return();
        g.line_feed();
        for ch in ['d', 'e', 'f'] {
            g.write_char(ch);
        }
        g.resize(3, 5);
        assert_eq!(g.dimensions(), (3, 5));
        // Content preserved.
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'a');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'b');
        assert_eq!(g.cell_at(0, 2).unwrap().ch, 'c');
        assert_eq!(g.cell_at(1, 0).unwrap().ch, 'd');
        assert_eq!(g.cell_at(1, 1).unwrap().ch, 'e');
        assert_eq!(g.cell_at(1, 2).unwrap().ch, 'f');
        // New area blanked.
        assert_eq!(g.cell_at(0, 3).unwrap().ch, ' ');
        assert_eq!(g.cell_at(2, 0).unwrap().ch, ' ');
    }

    #[test]
    fn resize_truncates_when_shrinking() {
        let mut g = Grid::new(3, 4);
        // Fill row 0 with 'X'.
        for _ in 0..4 {
            g.write_char('X');
        }
        g.resize(2, 2);
        assert_eq!(g.dimensions(), (2, 2));
        // Only the visible top-left 2x2 of the original content survives.
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'X');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'X');
    }

    #[test]
    fn resize_clamps_cursor_inside_new_bounds() {
        let mut g = Grid::new(5, 5);
        g.cursor_move_to(4, 4);
        g.resize(2, 2);
        let cur = g.cursor();
        assert!(cur.row < 2);
        assert!(cur.col < 2);
    }

    #[test]
    fn resize_to_same_size_is_noop() {
        let mut g = Grid::new(3, 3);
        g.write_char('Q');
        g.resize(3, 3);
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'Q');
        assert_eq!(g.cursor(), Cursor { row: 0, col: 1 });
    }

    #[test]
    fn resize_clamps_zero_dimensions() {
        let mut g = Grid::new(3, 3);
        g.resize(0, 0);
        assert_eq!(g.dimensions(), (1, 1));
    }

    // ---- scroll_up --------------------------------------------------------

    #[test]
    fn scroll_up_shifts_rows_and_blanks_bottom() {
        let mut g = Grid::new(3, 2);
        // Row 0 = "ab", row 1 = "cd", row 2 = "ef".
        g.write_char('a');
        g.write_char('b');
        g.cursor_move_to(1, 0);
        g.write_char('c');
        g.write_char('d');
        g.cursor_move_to(2, 0);
        g.write_char('e');
        g.write_char('f');
        g.scroll_up();
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'c');
        assert_eq!(g.cell_at(0, 1).unwrap().ch, 'd');
        assert_eq!(g.cell_at(1, 0).unwrap().ch, 'e');
        assert_eq!(g.cell_at(1, 1).unwrap().ch, 'f');
        assert_eq!(g.cell_at(2, 0).unwrap().ch, ' ');
        assert_eq!(g.cell_at(2, 1).unwrap().ch, ' ');
    }

    #[test]
    fn scroll_up_on_one_row_grid_blanks_the_row() {
        let mut g = Grid::new(1, 3);
        for ch in ['x', 'y', 'z'] {
            g.write_char(ch);
        }
        g.scroll_up();
        for c in 0..3 {
            assert_eq!(g.cell_at(0, c).unwrap().ch, ' ');
        }
    }

    // ---- robustness -------------------------------------------------------

    #[test]
    fn fuzzy_writes_never_panic() {
        // Sanity sweep: feed a bunch of mixed control + printable chars at
        // random-ish offsets and ensure nothing panics.
        let mut g = Grid::new(4, 6);
        let stream = "hello\nworld\t!\nbackspace\x08\x08fix\rOK";
        for ch in stream.chars() {
            match ch {
                '\n' => g.line_feed(),
                '\r' => g.carriage_return(),
                '\t' => g.tab(),
                '\x08' => g.backspace(),
                other => g.write_char(other),
            }
        }
        // Cursor must remain inside legal range.
        let cur = g.cursor();
        assert!(cur.row < 4);
        assert!(cur.col <= 6);
    }

    #[test]
    fn one_by_one_grid_is_usable() {
        let mut g = Grid::new(1, 1);
        g.write_char('X');
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'X');
        // Pending-wrap at (0, 1).
        assert_eq!(g.cursor(), Cursor { row: 0, col: 1 });
        // Next write scrolls (which on a 1-row grid blanks the only row)
        // before placing the char.
        g.write_char('Y');
        assert_eq!(g.cell_at(0, 0).unwrap().ch, 'Y');
    }
}
