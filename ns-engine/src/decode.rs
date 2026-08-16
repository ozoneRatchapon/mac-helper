//! Speculative decode loop: draft → prune → verify.
//!
//! ```text
//! draft.log_probs(context)
//!     │
//!     ▼
//! top_k_indices(logits)          — best candidates by logit
//!     │
//!     ▼
//! pruner.batch_is_valid(...)     — filter to valid-only branches
//!     │
//!     ▼
//! accept best valid candidate    — or backtrack on dead-end
//!     │
//!     ▼
//! verify_sequence(pruner, ...)   — full validity check on the final output
//! ```
//!
//! Two modes:
//! - **Greedy** — pick best valid candidate, stop on dead-end. Faithful to
//!   LLM speculative decode (no backtracking).
//! - **Backtracking** — DFS with candidate stack. Solves hard constraint
//!   problems (Sudoku) by exploring the valid-only subtree.

use crate::traits::{ConstraintPruner, DraftModel};
use crate::types::{DecodeConfig, DecodeResult, TokenId};
use fastrand::Rng;
use std::cmp::Ordering;

/// Run the speculative decode loop.
///
/// Pipeline per step:
///   1. Draft model produces log-prob distribution
///   2. Top-k candidates are extracted (sorted by logit descending)
///   3. ConstraintPruner filters invalid candidates
///   4. Best valid candidate is accepted (or backtrack on dead-end)
///
/// Final verification: the complete sequence is re-checked against the
/// pruner to catch any logic errors in the decode loop itself.
pub fn speculative_decode(
    draft: &dyn DraftModel,
    pruner: &mut dyn ConstraintPruner,
    config: &DecodeConfig,
) -> DecodeResult {
    let mut tokens: Vec<TokenId> = Vec::with_capacity(config.max_tokens);
    let mut attempts = 0u64;

    let solved = match config.backtrack {
        true => decode_with_backtrack(draft, pruner, config, &mut tokens, &mut attempts),
        false => decode_greedy(draft, pruner, config, &mut tokens),
    };

    let verified = solved && verify_sequence(pruner, &tokens);
    let hash = DecodeResult::hash_tokens(&tokens);

    DecodeResult {
        tokens,
        verified,
        attempts,
        hash,
    }
}

/// Greedy decode: pick best valid candidate at each step, stop on dead-end.
///
/// Faithful to LLM speculative decoding — no backtracking. If no valid
/// candidate exists at some depth, the loop terminates early.
fn decode_greedy(
    draft: &dyn DraftModel,
    pruner: &mut dyn ConstraintPruner,
    config: &DecodeConfig,
    tokens: &mut Vec<TokenId>,
) -> bool {
    while tokens.len() < config.max_tokens {
        let depth = tokens.len();
        let logits = draft.log_probs(tokens);
        let candidates = top_k_indices(&logits, config.top_k);
        let valid = valid_candidates(pruner, depth, tokens, &candidates);

        match valid.first() {
            Some(&token) => {
                tokens.push(token);
                pruner.propagate(depth, token, tokens);
            }
            None => return false,
        }
    }
    true
}

/// Backtracking decode: DFS over the valid-only subtree.
///
/// Maintains a per-depth candidate stack (`untried`). On first visit to a
/// depth, generates all valid candidates (sorted best-first). Pops the top;
/// on dead-end (empty stack), backtracks to the previous depth.
///
/// This is conceptually equivalent to katopz's DDTree: the pruner ensures
/// only valid branches are expanded, and the stack manages the search.
fn decode_with_backtrack(
    draft: &dyn DraftModel,
    pruner: &mut dyn ConstraintPruner,
    config: &DecodeConfig,
    tokens: &mut Vec<TokenId>,
    attempts: &mut u64,
) -> bool {
    // Seeded RNG for deterministic randomized DFS (tie-breaking in candidate
    // ordering). Same seed → same shuffle → same solution path.
    let mut rng = Rng::with_seed(config.seed);
    // untried[d] = remaining valid candidates at depth d, best on top.
    // Lazily populated on first visit to each depth.
    let mut untried: Vec<Vec<TokenId>> = Vec::with_capacity(config.max_tokens);

    loop {
        let depth = tokens.len();

        // Guard: target length reached
        if depth >= config.max_tokens {
            return true;
        }
        // Guard: attempt budget exhausted
        if *attempts >= config.max_attempts {
            return false;
        }

        // First visit to this depth: generate valid candidates
        if depth >= untried.len() {
            let logits = draft.log_probs(tokens);
            let candidates = top_k_indices(&logits, config.top_k);
            let mut valid = valid_candidates(pruner, depth, tokens, &candidates);
            // Shuffle within tied-logit groups for randomized DFS.
            // Escapes adversarial orderings (e.g. Inkala's hardest Sudoku)
            // while preserving priority ordering for non-uniform models.
            shuffle_tied_groups(&mut valid, &logits, &mut rng);
            // valid is best-first, but pop() consumes from the end — reverse
            // so the best candidate really is "on top" of the stack.
            valid.reverse();
            untried.push(valid);
        }

        *attempts += 1;

        match untried[depth].pop() {
            Some(token) => {
                tokens.push(token);
                pruner.propagate(depth, token, tokens);
            }
            None => {
                // Dead end at this depth — backtrack
                match tokens.pop() {
                    Some(token) => {
                        let popped_depth = tokens.len();
                        pruner.on_backtrack(popped_depth, token, tokens);
                        // Clear stale candidates beyond the new depth.
                        // They were generated for a different prefix and must be
                        // regenerated when we return to that depth.
                        untried.truncate(depth);
                    }
                    None => return false,
                }
            }
        }
    }
}

/// Extract indices of the top-k highest-logit tokens, sorted descending.
///
/// Stable sort preserves index order for tied logits, giving deterministic
/// candidate ordering when the draft model is uniform.
pub(crate) fn top_k_indices(logits: &[f32], k: usize) -> Vec<TokenId> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(TokenId, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, l)| (i as TokenId, l))
        .collect();

    // Sort by logit descending; stable for ties (preserves index order)
    indexed.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
        Some(ord) => ord,
        None => Ordering::Equal,
    });

    indexed[..k].iter().map(|(i, _)| *i).collect()
}

/// Filter candidates through the pruner, returning only valid ones.
///
/// Preserves the input order (descending logit), so the first element of
/// the result is the best valid candidate.
pub(crate) fn valid_candidates(
    pruner: &dyn ConstraintPruner,
    depth: usize,
    parent_tokens: &[TokenId],
    candidates: &[TokenId],
) -> Vec<TokenId> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut mask = vec![false; candidates.len()];
    pruner.batch_is_valid(depth, candidates, parent_tokens, &mut mask);

    candidates
        .iter()
        .zip(mask.iter())
        .filter(|(_, &valid)| valid)
        .map(|(&tok, _)| tok)
        .collect()
}

/// Shuffle candidates within groups of equal logit value.
///
/// Preserves inter-group ordering (high-logit groups stay first), randomizes
/// intra-group ordering. For uniform draft models (all logits equal), this
/// shuffles the entire list — giving the backtracking search diversity to
/// escape adversarial cell/digit orderings.
///
/// For non-uniform models, ties are rare, so this is effectively a no-op
/// while still providing the same deterministic seed-based guarantee.
pub(crate) fn shuffle_tied_groups(candidates: &mut [TokenId], logits: &[f32], rng: &mut Rng) {
    if candidates.len() <= 1 {
        return;
    }

    let mut start = 0;
    while start < candidates.len() {
        let ref_logit = logits[candidates[start] as usize];
        let mut end = start + 1;
        while end < candidates.len() {
            let other_logit = logits[candidates[end] as usize];
            if (other_logit - ref_logit).abs() > 1e-9 {
                break;
            }
            end += 1;
        }
        // Fisher-Yates shuffle within [start..end)
        let group_len = end - start;
        for i in (1..group_len).rev() {
            let j = rng.usize(0..=i);
            candidates.swap(start + j, start + i);
        }
        start = end;
    }
}

/// Verify the final token sequence against the pruner.
///
/// Re-checks every token at its original depth. This catches any logic
/// errors in the decode loop and provides a safety net for production use.
fn verify_sequence(pruner: &dyn ConstraintPruner, tokens: &[TokenId]) -> bool {
    for (depth, &token) in tokens.iter().enumerate() {
        if !pruner.is_valid(depth, token, &tokens[..depth]) {
            return false;
        }
    }
    true
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::UniformDraftModel;
    use crate::pruners::{NoPruner, SudokuPruner};

    // ── top_k_indices ──

    #[test]
    fn test_top_k_indices_sorted_descending() {
        let logits = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        let result = top_k_indices(&logits, 3);
        assert_eq!(result, vec![1, 4, 2]); // indices of 0.9, 0.7, 0.5
    }

    #[test]
    fn test_top_k_indices_handles_k_larger_than_input() {
        let logits = vec![0.3, 0.1, 0.2];
        let result = top_k_indices(&logits, 10);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0); // highest
    }

    #[test]
    fn test_top_k_indices_zero_k_returns_empty() {
        let logits = vec![0.5, 0.3];
        let result = top_k_indices(&logits, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_indices_empty_logits() {
        let result = top_k_indices(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_indices_ties_preserve_index_order() {
        // All equal logits — stable sort keeps index order
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let result = top_k_indices(&logits, 3);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_top_k_indices_handles_nan() {
        let logits = vec![0.5, f32::NAN, 0.3];
        let result = top_k_indices(&logits, 3);
        assert_eq!(result.len(), 3);
        // NaN comparison falls back to Equal — should not panic
    }

    // ── valid_candidates ──

    #[test]
    fn test_valid_candidates_filters_invalid() {
        let pruner = NoPruner;
        let candidates = vec![1, 2, 3, 4, 5];
        let result = valid_candidates(&pruner, 0, &[], &candidates);
        assert_eq!(result, candidates); // NoPruner accepts all
    }

    #[test]
    fn test_valid_candidates_empty_input() {
        let pruner = NoPruner;
        let result = valid_candidates(&pruner, 0, &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_valid_candidates_preserves_order() {
        // Sudoku: cell 0 accepts any digit, cell 1 rejects duplicates of cell 0
        let pruner = SudokuPruner::empty();
        let parent = vec![5];
        let candidates = vec![1, 5, 3, 5, 2]; // two 5s should be filtered
        let result = valid_candidates(&pruner, 1, &parent, &candidates);
        assert_eq!(result, vec![1, 3, 2]);
    }

    // ── verify_sequence ──

    #[test]
    fn test_verify_sequence_valid_sudoku_row() {
        let pruner = SudokuPruner::empty();
        // Valid first row: 1-9, no duplicates
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert!(verify_sequence(&pruner, &tokens));
    }

    #[test]
    fn test_verify_sequence_catches_duplicate() {
        let pruner = SudokuPruner::empty();
        // Invalid: two 5s in row 0
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 7, 5, 9];
        assert!(!verify_sequence(&pruner, &tokens));
    }

    #[test]
    fn test_verify_sequence_empty_passes() {
        let pruner = SudokuPruner::empty();
        assert!(verify_sequence(&pruner, &[]));
    }

    // ── speculative_decode: greedy mode ──

    #[test]
    fn test_greedy_no_pruner_fills_max_tokens() {
        let draft = UniformDraftModel::new(5);
        let mut pruner = NoPruner;
        let config = DecodeConfig {
            max_tokens: 10,
            top_k: 5,
            backtrack: false,
            ..Default::default()
        };

        let result = speculative_decode(&draft, &mut pruner, &config);

        assert_eq!(result.tokens.len(), 10);
        assert!(result.verified);
        // With uniform draft, all tokens tie → index order → all token 0
        assert!(result.tokens.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_greedy_sudoku_stops_on_dead_end() {
        let draft = UniformDraftModel::new(10);
        let mut pruner = SudokuPruner::empty();
        let config = DecodeConfig {
            max_tokens: 81,
            top_k: 10,
            backtrack: false,
            max_attempts: 1000,
            ..Default::default()
        };

        let result = speculative_decode(&draft, &mut pruner, &config);

        // Greedy (no backtrack) will hit a dead-end well before 81 cells
        // on an empty Sudoku grid. The loop stops early.
        assert!(result.tokens.len() < 81);
        assert!(!result.verified); // incomplete
    }

    // ── speculative_decode: backtracking mode ──

    #[test]
    fn test_backtrack_solves_empty_sudoku() {
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

        assert_eq!(result.tokens.len(), 81, "must fill all 81 cells");
        assert!(result.verified, "solution must pass verification");
        assert_valid_sudoku(&result.tokens);
    }

    #[test]
    fn test_backtrack_solves_known_puzzle() {
        let puzzle = "
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

        let draft = UniformDraftModel::new(10);
        let mut pruner = SudokuPruner::from_puzzle(puzzle).expect("valid puzzle");
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
        assert_valid_sudoku(&result.tokens);
        assert_respects_givens(puzzle, &result.tokens);
    }

    #[test]
    fn test_backtrack_respects_attempt_budget() {
        // max_attempts = 0 → immediate failure
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
    fn test_decode_hash_is_deterministic() {
        let draft = UniformDraftModel::new(10);
        let config = DecodeConfig {
            max_tokens: 5,
            top_k: 10,
            backtrack: false,
            ..Default::default()
        };

        let mut pruner_a = NoPruner;
        let mut pruner_b = NoPruner;
        let result_a = speculative_decode(&draft, &mut pruner_a, &config);
        let result_b = speculative_decode(&draft, &mut pruner_b, &config);

        assert_eq!(result_a.hash, result_b.hash);
        assert_eq!(result_a.tokens, result_b.tokens);
    }

    // ── Sudoku validity helpers ──

    fn assert_valid_sudoku(tokens: &[TokenId]) {
        assert_eq!(tokens.len(), 81, "must have 81 cells");

        // Check rows
        for row in 0..9 {
            let mut seen = [false; 10];
            for col in 0..9 {
                let digit = tokens[row * 9 + col] as usize;
                assert!(
                    (1..=9).contains(&digit),
                    "digit {digit} out of range at row {row} col {col}"
                );
                assert!(
                    !seen[digit],
                    "duplicate digit {digit} in row {row} at col {col}"
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
                    "duplicate digit {digit} in col {col} at row {row}"
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

    fn assert_respects_givens(puzzle: &str, solution: &[TokenId]) {
        let chars: Vec<char> = puzzle.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(chars.len(), 81, "puzzle must have 81 cells");

        for (i, &c) in chars.iter().enumerate() {
            if let '1'..='9' = c {
                let given = c as u32 - b'0' as u32;
                assert_eq!(
                    solution[i], given,
                    "cell {i}: given {given} overwritten with {}",
                    solution[i]
                );
            }
        }
    }
}
