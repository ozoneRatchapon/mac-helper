//! Path-aware Sudoku constraint pruner.
//!
//! Enforces row, column, and 3×3 box constraints during speculative decode.
//! The pruner is stateless — it derives all state from `parent_tokens` (the
//! prefix of cells filled so far). This matches katopz's design: hard
//! structural validity lives in the pruner, not the model.
//!
//! # Vocabulary
//!
//! - TokenId `1..=9` → digits 1–9
//! - TokenId `0`     → unused / blank (always invalid)
//!
//! # Givens
//!
//! Pre-filled cells from the puzzle. The pruner enforces:
//! 1. A given cell only accepts its given digit.
//! 2. A free cell cannot conflict with any future given (row/col/box),
//!    because that given WILL be placed when the decode loop reaches it.
//!
//! This forward-looking constraint propagation is what makes the pruner
//! powerful: dead branches are cut before they're explored.

use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

/// Grid dimensions for standard 9×9 Sudoku.
const GRID_SIZE: usize = 9;
const BOX_SIZE: usize = 3;
const NUM_CELLS: usize = GRID_SIZE * GRID_SIZE;

/// Stable arm identifier for [`SudokuPruner`].
///
/// Used for diagnostics and cross-pattern attribution inside a
/// [`BanditPruner`](crate::bandit::BanditPruner). The trial log keys arms
/// by Vec index, not by this ID, so the value just needs to be stable and
/// unique within a deployment.
pub const SUDOKU_ARM_ID: ArmId = 1;

/// Default arm label reported by [`SudokuPruner::arm_label`].
pub const SUDOKU_LABEL: &str = "sudoku";

/// Errors during Sudoku puzzle parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SudokuError {
    /// Encountered a character that is not a digit, dot, or whitespace.
    InvalidChar(char),
    /// Puzzle string has the wrong number of cells after stripping whitespace.
    WrongLength(usize),
}

/// Path-aware Sudoku constraint pruner.
///
/// Stores the puzzle givens as a fixed array. All validity checks derive
/// from the givens and the `parent_tokens` prefix passed to `is_valid`.
/// No mutable state — the pruner is a pure function of (givens, depth, token, prefix).
#[derive(Clone, Debug)]
pub struct SudokuPruner {
    /// `givens[cell_index] = Some(digit)` if pre-filled by the puzzle.
    /// `None` means the cell is free.
    givens: [Option<u8>; NUM_CELLS],
}

impl Default for SudokuPruner {
    fn default() -> Self {
        Self {
            givens: [None; NUM_CELLS],
        }
    }
}

impl SudokuPruner {
    /// Create a pruner for an empty grid (no givens).
    ///
    /// Useful for generating Sudoku solutions or testing constraint propagation
    /// without a specific puzzle.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a pruner from a puzzle string.
    ///
    /// # Format
    ///
    /// 81 characters, where:
    /// - digits `1`–`9` → given cells
    /// - `.` or `0`     → blank cells
    /// - whitespace     → ignored
    ///
    /// # Errors
    ///
    /// Returns [`SudokuError::InvalidChar`] for non-digit, non-dot characters.
    /// Returns [`SudokuError::WrongLength`] if the cell count ≠ 81.
    pub fn from_puzzle(input: &str) -> Result<Self, SudokuError> {
        let mut givens = [None; NUM_CELLS];
        let mut idx = 0usize;

        for ch in input.chars() {
            if ch.is_whitespace() {
                continue;
            }

            if idx >= NUM_CELLS {
                return Err(SudokuError::WrongLength(idx + 1));
            }

            match ch {
                '.' | '0' => {
                    // Blank cell — leave as None
                    idx += 1;
                }
                '1'..='9' => {
                    givens[idx] = Some(ch as u8 - b'0');
                    idx += 1;
                }
                _ => return Err(SudokuError::InvalidChar(ch)),
            }
        }

        match idx {
            NUM_CELLS => Ok(Self { givens }),
            n => Err(SudokuError::WrongLength(n)),
        }
    }

    /// Access the givens array (for verification and debugging).
    pub fn givens(&self) -> &[Option<u8>; NUM_CELLS] {
        &self.givens
    }

    /// Check if a specific cell has a given value.
    pub fn given_at(&self, cell: usize) -> Option<u8> {
        self.givens.get(cell).copied().flatten()
    }

    /// Convert a token to a digit (1–9).
    ///
    /// Returns `None` for token 0 (unused) or out-of-range tokens.
    fn token_to_digit(token: TokenId) -> Option<u8> {
        match token {
            1..=9 => Some(token as u8),
            _ => None,
        }
    }

    /// Check if two cells share a unit (row, column, or 3×3 box).
    ///
    /// Two cells in the same unit cannot have the same digit.
    fn same_unit(a: usize, b: usize) -> bool {
        let ar = a / GRID_SIZE;
        let ac = a % GRID_SIZE;
        let br = b / GRID_SIZE;
        let bc = b % GRID_SIZE;

        // Same row
        if ar == br {
            return true;
        }
        // Same column
        if ac == bc {
            return true;
        }
        // Same 3×3 box
        let abr = (ar / BOX_SIZE) * BOX_SIZE;
        let abc = (ac / BOX_SIZE) * BOX_SIZE;
        let bbr = (br / BOX_SIZE) * BOX_SIZE;
        let bbc = (bc / BOX_SIZE) * BOX_SIZE;
        abr == bbr && abc == bbc
    }

    /// Check if placing `digit` at `cell` conflicts with any already-placed
    /// token or any future given.
    ///
    /// # Arguments
    ///
    /// - `cell` — the cell index being filled (0..81)
    /// - `digit` — the digit being placed (1..9)
    /// - `parent_tokens` — tokens placed at cells 0..cell
    /// - `givens` — the puzzle's pre-filled cells
    fn has_conflict(
        cell: usize,
        digit: u8,
        parent_tokens: &[TokenId],
        givens: &[Option<u8>; NUM_CELLS],
    ) -> bool {
        // Check already-placed cells (includes past givens, which were forced)
        for (i, &tok) in parent_tokens.iter().enumerate() {
            let other = match Self::token_to_digit(tok) {
                Some(d) => d,
                None => continue,
            };
            if other == digit && Self::same_unit(i, cell) {
                return true;
            }
        }

        // Check future givens (cells beyond `cell` that are pre-filled)
        // These WILL be placed when the decode loop reaches them, so a conflict
        // now means a dead branch later. Cut it early.
        for (i, maybe_given) in givens.iter().enumerate().skip(cell + 1) {
            if let Some(other) = maybe_given {
                if *other == digit && Self::same_unit(i, cell) {
                    return true;
                }
            }
        }

        false
    }
}

impl ConstraintPruner for SudokuPruner {
    fn is_valid(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        // Depth must be within the grid
        if depth >= NUM_CELLS {
            return false;
        }

        // Token must be a valid digit (1..9)
        let digit = match Self::token_to_digit(token) {
            Some(d) => d,
            None => return false,
        };

        // Given constraint: if this cell is pre-filled, only that digit is valid
        if let Some(given) = self.givens[depth] {
            return given == digit;
        }

        // Structural constraints (row, col, box) against placed tokens + future givens
        !Self::has_conflict(depth, digit, parent_tokens, &self.givens)
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        // Fast path for out-of-range depth — all invalid
        if depth >= NUM_CELLS {
            let len = candidates.len().min(results.len());
            results[..len].fill(false);
            return;
        }

        // Pre-compute the given constraint once for this depth
        let given = self.givens[depth];

        let len = candidates.len().min(results.len());
        for i in 0..len {
            let token = candidates[i];

            // Token must be a valid digit
            let digit = match Self::token_to_digit(token) {
                Some(d) => d,
                None => {
                    results[i] = false;
                    continue;
                }
            };

            // Given constraint check
            if let Some(g) = given {
                results[i] = g == digit;
                continue;
            }

            // Structural constraint check
            results[i] = !Self::has_conflict(depth, digit, parent_tokens, &self.givens);
        }
    }

    fn propagate(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {
        // Stateless pruner — no internal state to update.
        // All validity derives from parent_tokens at check time.
    }
}

// ── ScreeningPruner impl ───────────────────────────────────────

impl ScreeningPruner for SudokuPruner {
    fn arm_id(&self) -> ArmId {
        SUDOKU_ARM_ID
    }

    fn arm_label(&self) -> &str {
        SUDOKU_LABEL
    }

    fn screen(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        // Binary: 1.0 for valid digits, 0.0 otherwise. The pruner's job is
        // hard structural correctness — there is no "partial" Sudoku move.
        match self.is_valid(depth, token, parent_tokens) {
            true => 1.0,
            false => 0.0,
        }
    }

    fn batch_screen(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [f32],
    ) {
        // Single batch_is_valid call amortizes the per-depth setup work
        // (givens lookup, conflict pre-computation) across all candidates.
        let len = candidates.len().min(results.len());
        let mut mask = vec![false; len];
        self.batch_is_valid(depth, candidates, parent_tokens, &mut mask);
        for i in 0..len {
            results[i] = match mask[i] {
                true => 1.0,
                false => 0.0,
            };
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard Sudoku puzzle (medium difficulty, unique solution).
    const TEST_PUZZLE: &str = "
        5 3 . . 7 . . . .
        6 . . 1 9 5 . . .
        . 9 8 . . . . 6 .
        8 . . . 6 . . . 3
        4 . . 8 . 3 . . 1
        7 . . . 2 . . . 6
        . 6 . . . . 2 8 .
        . . . 4 1 9 . . 5
        . . . . 8 . . 7 9
    ";

    #[test]
    fn test_empty_pruner_accepts_valid_digits() {
        let pruner = SudokuPruner::empty();
        // First cell (0,0) — no constraints yet, any digit 1-9 is valid
        for d in 1..=9u32 {
            assert!(pruner.is_valid(0, d, &[]), "digit {d} should be valid");
        }
    }

    #[test]
    fn test_empty_pruner_rejects_token_zero() {
        let pruner = SudokuPruner::empty();
        assert!(!pruner.is_valid(0, 0, &[]), "token 0 is not a digit");
    }

    #[test]
    fn test_empty_pruner_rejects_out_of_range_depth() {
        let pruner = SudokuPruner::empty();
        assert!(!pruner.is_valid(81, 5, &[]), "depth 81 is out of range");
        assert!(!pruner.is_valid(100, 5, &[]), "depth 100 is out of range");
    }

    #[test]
    fn test_row_constraint() {
        let pruner = SudokuPruner::empty();
        // Place 5 at cell 0 (row 0, col 0)
        let parent = vec![5];
        // Cell 1 is in row 0 — 5 should be invalid
        assert!(!pruner.is_valid(1, 5, &parent));
        // Cell 1 — 4 should be valid (different digit)
        assert!(pruner.is_valid(1, 4, &parent));
    }

    #[test]
    fn test_column_constraint() {
        let pruner = SudokuPruner::empty();
        // Place 3 at cell 0 (row 0, col 0)
        let parent = vec![3];
        // Cell 9 is at row 1, col 0 — same column
        assert!(!pruner.is_valid(9, 3, &parent));
        // Different digit is fine
        assert!(pruner.is_valid(9, 7, &parent));
    }

    #[test]
    fn test_box_constraint() {
        let pruner = SudokuPruner::empty();
        // Place 7 at cell 0 (row 0, col 0, box 0)
        let parent = vec![7];
        // Cell 10 is at row 1, col 1 — same box (0,0)-(2,2)
        assert!(!pruner.is_valid(10, 7, &parent));
        // Cell 30 is at row 3, col 3 — different box
        assert!(pruner.is_valid(30, 7, &parent));
    }

    #[test]
    fn test_different_unit_no_conflict() {
        let pruner = SudokuPruner::empty();
        // Place 5 at cell 0 (row 0, col 0, box 0)
        let parent = vec![5];
        // Cell 4 is at row 0, col 4 — same row, CONFLICT
        assert!(!pruner.is_valid(4, 5, &parent));
        // Cell 40 is at row 4, col 4 — different row/col/box, OK
        assert!(pruner.is_valid(40, 5, &parent));
    }

    #[test]
    fn test_from_puzzle_parses_valid() {
        let pruner = SudokuPruner::from_puzzle(TEST_PUZZLE);
        assert!(pruner.is_ok(), "valid puzzle should parse");

        let pruner = pruner.unwrap();
        // Cell 0 = 5 (given)
        assert_eq!(pruner.given_at(0), Some(5));
        // Cell 1 = 3 (given)
        assert_eq!(pruner.given_at(1), Some(3));
        // Cell 2 = None (blank)
        assert_eq!(pruner.given_at(2), None);
        // Cell 80 = 9 (given)
        assert_eq!(pruner.given_at(80), Some(9));
    }

    #[test]
    fn test_from_puzzle_enforces_givens() {
        let pruner = SudokuPruner::from_puzzle(TEST_PUZZLE).unwrap();
        // Cell 0 is given as 5 — only 5 is valid there
        assert!(pruner.is_valid(0, 5, &[]));
        assert!(!pruner.is_valid(0, 3, &[]));
        assert!(!pruner.is_valid(0, 1, &[]));
    }

    #[test]
    fn test_from_puzzle_rejects_wrong_length() {
        let result = SudokuPruner::from_puzzle("1 2 3 4 5");
        assert!(matches!(result, Err(SudokuError::WrongLength(5))));
    }

    #[test]
    fn test_from_puzzle_rejects_invalid_char() {
        let result = SudokuPruner::from_puzzle("X");
        assert!(matches!(result, Err(SudokuError::InvalidChar('X'))));
    }

    #[test]
    fn test_from_puzzle_accepts_dot_and_zero() {
        let input = ".0.".repeat(27); // 81 chars: mix of . and 0
        let pruner = SudokuPruner::from_puzzle(&input);
        assert!(pruner.is_ok());
        assert_eq!(pruner.unwrap().given_at(0), None);
    }

    #[test]
    fn test_future_given_conflict_detected() {
        // Build a puzzle where cell 1 (row 0, col 1) is given as 5.
        let mut input = String::new();
        input.push('.');
        input.push('5'); // Cell 1 = 5
        for _ in 2..NUM_CELLS {
            input.push('.');
        }
        let pruner = SudokuPruner::from_puzzle(&input).unwrap();

        // Cell 0 (row 0, col 0): placing 5 should conflict with future given at cell 1
        // Cell 1 is at row 0, col 1 — same row as cell 0
        assert!(
            !pruner.is_valid(0, 5, &[]),
            "digit 5 at cell 0 should conflict with future given at cell 1"
        );
        // Placing 3 at cell 0 is fine (no conflict)
        assert!(pruner.is_valid(0, 3, &[]));
    }

    #[test]
    fn test_batch_is_valid_matches_individual() {
        let pruner = SudokuPruner::empty();
        let parent = vec![5, 3, 4, 6, 7, 8, 9, 1, 2]; // First row complete
        let candidates: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0];

        let mut batch_results = vec![false; candidates.len()];
        pruner.batch_is_valid(9, &candidates, &parent, &mut batch_results);

        for (i, &tok) in candidates.iter().enumerate() {
            let individual = pruner.is_valid(9, tok, &parent);
            assert_eq!(
                batch_results[i], individual,
                "batch result mismatch for token {tok}"
            );
        }
    }

    #[test]
    fn test_batch_is_valid_out_of_range_depth() {
        let pruner = SudokuPruner::empty();
        let candidates = vec![1, 2, 3];
        let mut results = vec![true; 3];
        pruner.batch_is_valid(100, &candidates, &[], &mut results);
        assert_eq!(results, vec![false, false, false]);
    }

    #[test]
    fn test_batch_is_valid_handles_short_results() {
        let pruner = SudokuPruner::empty();
        let candidates = vec![1, 2, 3, 4, 5];
        let mut results = vec![false; 2]; // Shorter than candidates
        pruner.batch_is_valid(0, &candidates, &[], &mut results);
        // Only first 2 written, both valid (empty grid, no constraints)
        assert_eq!(results, vec![true, true]);
    }

    #[test]
    fn test_manifold_score_binary() {
        let pruner = SudokuPruner::empty();
        // Valid token → score 1.0
        let valid = pruner.manifold_score(0, 5, &[]);
        assert!((valid - 1.0).abs() < 1e-6);
        // Invalid token → score 0.0
        let invalid = pruner.manifold_score(0, 0, &[]);
        assert!((invalid - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_propagate_is_noop() {
        let mut pruner = SudokuPruner::empty();
        // Propagate should not change behavior (stateless)
        pruner.propagate(0, 5, &[]);
        assert!(pruner.is_valid(1, 3, &[5]));
        assert!(!pruner.is_valid(1, 5, &[5]));
    }

    #[test]
    fn test_same_unit_identity() {
        // Same cell — vacuously true (won't be called in practice, but safe)
        assert!(SudokuPruner::same_unit(0, 0));
    }

    #[test]
    fn test_same_unit_all_pairs() {
        // Row neighbors
        for c in 1..9 {
            assert!(SudokuPruner::same_unit(0, c), "cell 0 and {c} share a row");
        }
        // Column neighbors
        for r in 1..9 {
            assert!(
                SudokuPruner::same_unit(0, r * 9),
                "cell 0 and {} share a column",
                r * 9
            );
        }
        // Box neighbors (0,0)-(2,2)
        let box_cells = [1, 2, 9, 10, 11, 18, 19, 20];
        for &c in &box_cells {
            assert!(SudokuPruner::same_unit(0, c), "cell 0 and {c} share a box");
        }
        // NOT in same unit: (0,0) vs (3,3)
        assert!(!SudokuPruner::same_unit(0, 30));
        assert!(!SudokuPruner::same_unit(0, 40));
    }

    #[test]
    fn test_default_is_empty_grid() {
        let pruner = SudokuPruner::default();
        for cell in 0..NUM_CELLS {
            assert_eq!(pruner.given_at(cell), None);
        }
    }
}
