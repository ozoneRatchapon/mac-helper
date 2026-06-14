//! Phase 1 end-to-end proof: solve Sudoku via the draft → prune → verify loop.
//!
//! This test demonstrates the core "modelless" thesis from katopz:
//! a uniform (completely uninformative) draft model + a path-aware
//! ConstraintPruner solves a hard constraint problem via speculative
//! decode with backtracking. The pruner does ALL the work — the draft
//! model contributes zero domain knowledge.
//!
//! Run with:
//!   cargo test --test phase1_sudoku -- --nocapture
//!   cargo test --test phase1_sudoku --release -- --nocapture

use ns_engine::{
    draft::UniformDraftModel, pruners::SudokuPruner, speculative_decode, ConstraintPruner,
    DecodeConfig, DecodeResult, DraftModel,
};

// ── Test puzzles ───────────────────────────────────────────────

/// Standard medium-difficulty Sudoku puzzle with a unique solution.
/// Used across the test suite for consistency.
const MEDIUM_PUZZLE: &str = "
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

/// Arto Inkala's "World's Hardest Sudoku" (puzzle credits: Arto Inkala, 2012).
/// A stress test for the backtracking decode loop — many dead branches.
const INKALA_HARDEST: &str = "
    8 . . . . . . . .
    . . 3 6 . . . . .
    . 7 . . 9 . 2 . .
    . 5 . . . 7 . . .
    . . . . 4 5 7 . .
    . . . 1 . . . 3 .
    . . 1 . . . . 6 8
    . . 8 5 . . . 1 .
    . 9 . . . . 4 . .
";

// ── Draft model contribution tests ─────────────────────────────

#[test]
fn uniform_draft_model_is_truly_uninformative() {
    let model = UniformDraftModel::new(10);

    let logits_empty = model.log_probs(&[]);
    let logits_after_some_tokens = model.log_probs(&[1, 2, 3, 4, 5]);

    // Both must be identical — uniform model ignores context
    assert_eq!(logits_empty.len(), logits_after_some_tokens.len());
    for (a, b) in logits_empty.iter().zip(logits_after_some_tokens.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "uniform model should not depend on context"
        );
    }

    // All logits equal — no preference
    let first = logits_empty[0];
    for (i, &l) in logits_empty.iter().enumerate() {
        assert!((l - first).abs() < 1e-6, "logit[{i}] should equal logit[0]");
    }
}

#[test]
fn pruner_carries_all_domain_logic() {
    let pruner = SudokuPruner::empty();

    // Token 0 is invalid (not a digit)
    assert!(!pruner.is_valid(0, 0, &[]));

    // Digits 1-9 are valid for the first cell
    for d in 1..=9u32 {
        assert!(
            pruner.is_valid(0, d, &[]),
            "digit {d} should be valid in empty grid cell 0"
        );
    }

    // After placing 5, another 5 in the same row is invalid
    assert!(!pruner.is_valid(1, 5, &[5]));
    assert!(pruner.is_valid(1, 4, &[5]));
}

// ── Empty grid solution ────────────────────────────────────────

#[test]
fn solve_empty_grid_backtracking() {
    let draft = UniformDraftModel::new(10);
    let mut pruner = SudokuPruner::empty();
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 500_000,
        ..Default::default()
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    assert_eq!(result.tokens.len(), 81, "all 81 cells must be filled");
    assert!(result.verified, "solution must pass verification");
    assert_valid_sudoku_grid(&result.tokens);
    assert!(
        result.attempts < 500_000,
        "empty grid should solve well within budget (got {} attempts)",
        result.attempts
    );

    print_grid("Empty grid solution", &result.tokens);
}

// ── Known puzzle solution ──────────────────────────────────────

#[test]
fn solve_medium_puzzle_backtracking() {
    let draft = UniformDraftModel::new(10);
    let mut pruner = SudokuPruner::from_puzzle(MEDIUM_PUZZLE).expect("puzzle should parse");
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 1_000_000,
        ..Default::default()
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    assert_eq!(result.tokens.len(), 81, "puzzle must be fully solved");
    assert!(result.verified, "solution must pass verification");
    assert_valid_sudoku_grid(&result.tokens);
    assert_respects_givens(MEDIUM_PUZZLE, &result.tokens);

    print_grid("Medium puzzle solution", &result.tokens);
    println!("  attempts: {}", result.attempts);
    println!("  hash: {}", hex_hash(&result.hash));
}

// ── Inkala's hardest (stress test) ─────────────────────────────

#[test]
fn solve_inkala_hardest_puzzle() {
    let draft = UniformDraftModel::new(10);
    let mut pruner = SudokuPruner::from_puzzle(INKALA_HARDEST).expect("puzzle should parse");
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        // Inkala's puzzle requires significant backtracking — give generous budget
        max_attempts: 5_000_000,
        ..Default::default()
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    assert_eq!(
        result.tokens.len(),
        81,
        "Inkala puzzle must be fully solved"
    );
    assert!(result.verified, "solution must pass verification");
    assert_valid_sudoku_grid(&result.tokens);
    assert_respects_givens(INKALA_HARDEST, &result.tokens);

    println!("Inkala hardest solved in {} attempts", result.attempts);
}

// ── Greedy mode (no backtracking) ──────────────────────────────

#[test]
fn greedy_mode_stops_on_dead_end() {
    let draft = UniformDraftModel::new(10);
    let mut pruner = SudokuPruner::empty();
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: false,
        ..Default::default()
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    // Greedy mode without backtracking cannot solve a full Sudoku grid —
    // it inevitably hits a dead-end before filling all 81 cells.
    assert!(
        result.tokens.len() < 81,
        "greedy should stop early on dead-end"
    );
    assert!(!result.verified, "incomplete sequence should not verify");
}

#[test]
fn greedy_mode_with_no_pruner_fills_all_tokens() {
    let draft = UniformDraftModel::new(5);
    let mut pruner = ns_engine::pruners::NoPruner;
    let config = DecodeConfig {
        max_tokens: 20,
        top_k: 5,
        backtrack: false,
        ..Default::default()
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    // NoPruner allows everything → greedy always finds a candidate
    assert_eq!(result.tokens.len(), 20);
    assert!(result.verified);
}

// ── Determinism and reproducibility ────────────────────────────

#[test]
fn decode_is_deterministic_same_seed() {
    let draft = UniformDraftModel::new(10);
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 500_000,
        seed: 12345,
    };

    let mut pruner_a = SudokuPruner::empty();
    let mut pruner_b = SudokuPruner::empty();
    let result_a = speculative_decode(&draft, &mut pruner_a, &config);
    let result_b = speculative_decode(&draft, &mut pruner_b, &config);

    assert_eq!(result_a.tokens, result_b.tokens, "same seed → same output");
    assert_eq!(result_a.hash, result_b.hash, "same output → same hash");
    assert_eq!(result_a.attempts, result_b.attempts);
}

#[test]
fn hash_uniquely_identifies_solution() {
    let draft = UniformDraftModel::new(10);
    let mut pruner = SudokuPruner::empty();
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 500_000,
        seed: 1,
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    // The hash must match a manual computation of the same tokens
    let manual_hash = DecodeResult::hash_tokens(&result.tokens);
    assert_eq!(result.hash, manual_hash);

    // Changing any token must change the hash
    let mut modified = result.tokens.clone();
    let original_first = modified[0];
    modified[0] = if original_first == 1 { 2 } else { 1 };
    let modified_hash = DecodeResult::hash_tokens(&modified);
    assert_ne!(
        result.hash, modified_hash,
        "hash must change when tokens change"
    );
}

// ── Budget limits ──────────────────────────────────────────────

#[test]
fn attempt_budget_zero_yields_empty_result() {
    let draft = UniformDraftModel::new(10);
    let mut pruner = SudokuPruner::empty();
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 0,
        ..Default::default()
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    assert!(!result.verified);
    assert_eq!(result.tokens.len(), 0);
}

#[test]
fn attempt_budget_too_small_yields_unsolved() {
    let draft = UniformDraftModel::new(10);
    let mut pruner = SudokuPruner::empty();
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 50, // way too small for a full Sudoku
        ..Default::default()
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    assert!(
        result.tokens.len() < 81,
        "should not complete with tiny budget"
    );
    assert!(!result.verified);
}

// ── Puzzle parsing ─────────────────────────────────────────────

#[test]
fn from_puzzle_parses_givens_correctly() {
    let pruner = SudokuPruner::from_puzzle(MEDIUM_PUZZLE).expect("valid puzzle");

    // Spot-check known givens from the medium puzzle
    assert_eq!(pruner.given_at(0), Some(5)); // row 0, col 0
    assert_eq!(pruner.given_at(1), Some(3)); // row 0, col 1
    assert_eq!(pruner.given_at(4), Some(7)); // row 0, col 4
    assert_eq!(pruner.given_at(80), Some(9)); // row 8, col 8

    // Blank cells return None
    assert_eq!(pruner.given_at(2), None); // row 0, col 2
    assert_eq!(pruner.given_at(3), None);
}

#[test]
fn from_puzzle_rejects_wrong_length() {
    let too_short = "1 2 3 4 5";
    let result = SudokuPruner::from_puzzle(too_short);
    assert!(result.is_err());

    let too_long = "1".repeat(100);
    let result = SudokuPruner::from_puzzle(&too_long);
    assert!(result.is_err());
}

// ── Forward-looking constraint propagation ─────────────────────

#[test]
fn pruner_detects_future_given_conflicts() {
    // Build a puzzle where cell 1 (row 0, col 1) is given as 5.
    let mut input = String::new();
    input.push('.'); // cell 0: blank
    input.push('5'); // cell 1: given 5
    for _ in 2..81 {
        input.push('.');
    }

    let pruner = SudokuPruner::from_puzzle(&input).expect("valid puzzle");

    // Placing 5 at cell 0 should be detected as a conflict, because cell 1
    // (same row) is fixed at 5 and will be placed by the decode loop.
    assert!(
        !pruner.is_valid(0, 5, &[]),
        "5 at cell 0 conflicts with future given 5 at cell 1"
    );

    // A different digit is fine.
    assert!(pruner.is_valid(0, 3, &[]));
    assert!(pruner.is_valid(0, 9, &[]));
}

// ── Verification helpers ───────────────────────────────────────

/// Verify that a token sequence forms a valid complete Sudoku grid.
///
/// Checks: 81 cells, digits 1-9 only, no duplicates in any row, column, or box.
fn assert_valid_sudoku_grid(tokens: &[u32]) {
    assert_eq!(tokens.len(), 81, "Sudoku grid must have 81 cells");

    // Check all digits are in range 1-9
    for (i, &t) in tokens.iter().enumerate() {
        assert!(
            (1..=9).contains(&(t as usize)),
            "cell {i}: token {t} is not a valid digit (1-9)"
        );
    }

    // Check rows
    for row in 0..9 {
        let mut seen = [false; 10];
        for col in 0..9 {
            let digit = tokens[row * 9 + col] as usize;
            assert!(
                !seen[digit],
                "duplicate digit {digit} in row {row} (col {col})"
            );
            seen[digit] = true;
        }
    }

    // Check columns
    for col in 0..9 {
        let mut seen = [false; 10];
        for row in 0..9 {
            let digit = tokens[row * 9 + col] as usize;
            assert!(
                !seen[digit],
                "duplicate digit {digit} in col {col} (row {row})"
            );
            seen[digit] = true;
        }
    }

    // Check 3×3 boxes
    for box_row in 0..3 {
        for box_col in 0..3 {
            let mut seen = [false; 10];
            for r in 0..3 {
                for c in 0..3 {
                    let row = box_row * 3 + r;
                    let col = box_col * 3 + c;
                    let digit = tokens[row * 9 + col] as usize;
                    assert!(
                        !seen[digit],
                        "duplicate digit {digit} in box ({box_row},{box_col})"
                    );
                    seen[digit] = true;
                }
            }
        }
    }
}

/// Verify that the solution respects all pre-filled givens from the puzzle.
fn assert_respects_givens(puzzle: &str, solution: &[u32]) {
    let chars: Vec<char> = puzzle.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(chars.len(), 81, "puzzle must have 81 cells");

    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_digit() && c != '0' {
            let given = c as u32 - b'0' as u32;
            assert_eq!(
                solution[i], given,
                "cell {i}: given {given} was overwritten with {}",
                solution[i]
            );
        }
    }
}

/// Pretty-print a Sudoku grid to stdout (visible with --nocapture).
fn print_grid(title: &str, tokens: &[u32]) {
    println!("\n=== {title} ===");
    for row in 0..9 {
        if row % 3 == 0 && row > 0 {
            println!("   ------+-------+------");
        }
        print!("    ");
        for col in 0..9 {
            if col % 3 == 0 && col > 0 {
                print!("| ");
            }
            let digit = tokens[row * 9 + col];
            print!("{digit} ");
        }
        println!();
    }
}

/// Format a 32-byte hash as lowercase hex for display.
fn hex_hash(hash: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in hash {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
