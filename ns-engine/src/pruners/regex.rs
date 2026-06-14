//! Regex-constrained pruner: validates token sequences against a regex pattern.
//!
//! A token is valid if appending its character to the current prefix could
//! still lead to a string that matches the regex. This is the
//! "prefix-validity" semantics used by `outlines`, `guidance`, and similar
//! constrained-generation tools.
//!
//! # DFA-based prefix validity
//!
//! The pruner compiles the pattern into an anchored dense DFA via
//! [`regex_automata::dfa::dense::Builder`] with [`StartKind::Anchored`].
//! `is_valid` walks the DFA one byte at a time:
//!
//! ```text
//! candidate_prefix = parent_tokens ──► chars ──► string
//! candidate_prefix + token_char
//!     │
//!     ▼ anchored DFA walk
//! start_state ──► next_state(byte₁) ──► … ──► next_state(byteₙ)
//!     │
//!     ▼
//! non-dead? ──── yes ──► is_valid = true  (prefix is completable)
//!     │
//!    no
//!     │
//!     ▼
//! is_valid = false (dead state = no possible completion)
//! ```
//!
//! The DFA is compiled once at construction time via
//! [`dense::Builder`](regex_automata::dfa::dense::Builder) with
//! [`StartKind::Anchored`], ensuring matches must begin at position 0.
//!
//! # Vocabulary
//!
//! Token IDs map to characters via a configurable mapping (default: token `t`
//! in `0..=0x10FFFF` maps to `char::from_u32(t)`; higher tokens are invalid).
//! Use [`RegexPruner::with_token_map`] to override.
//!
//! # Use as a bandit arm
//!
//! `screen` returns `1.0` for valid prefixes and `0.0` otherwise — binary,
//! like [`SudokuPruner`](super::SudokuPruner). The bandit's reward signal
//! therefore tracks "did this arm accept a regex-conformant token" rather
//! than a graded confidence.
//!
//! # Stateless
//!
//! The pruner holds no mutable state. [`propagate`](crate::ConstraintPruner::propagate)
//! and [`on_backtrack`](crate::ConstraintPruner::on_backtrack) are no-ops.
//! All validity and scoring derives from the immutable DFA and the
//! `parent_tokens` prefix.

use regex_automata::dfa::dense;
use regex_automata::dfa::{Automaton, StartKind};
use regex_automata::util::primitives::StateID;
use regex_automata::{Anchored, Input};

use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

/// Stable arm identifier for [`RegexPruner`].
pub const REGEX_ARM_ID: ArmId = 3;

/// Default arm label reported by [`RegexPruner::arm_label`].
pub const REGEX_LABEL: &str = "regex";

/// Maximum token ID supported by the default token-to-char mapping.
///
/// Tokens in `0..=MAX_DEFAULT_TOKEN` map to the corresponding Unicode scalar
/// via `char::from_u32`. Higher tokens are invalid.
pub const MAX_DEFAULT_TOKEN: TokenId = 0x10FFFF;

/// Errors that can occur while constructing a [`RegexPruner`].
#[derive(Debug, Clone)]
pub struct RegexPrunerError {
    message: String,
}

impl std::fmt::Display for RegexPrunerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "regex DFA build error: {message}",
            message = self.message
        )
    }
}

impl std::error::Error for RegexPrunerError {}

/// Regex-constrained screening pruner.
///
/// See the [module docs](self) for DFA-based prefix-validity semantics.
pub struct RegexPruner {
    /// Compiled dense DFA (anchored mode). Immutable after construction.
    dfa: dense::DFA<Vec<u32>>,
    /// Original pattern string, kept for diagnostics.
    pattern: String,
    /// `token_to_char[t] = Some(ch)` if token `t` maps to character `ch`.
    token_to_char: Vec<Option<char>>,
    /// Diagnostics identifier (see [`REGEX_ARM_ID`]).
    arm_id: ArmId,
    /// Human-readable label for logs and arena reports.
    label: String,
}

impl RegexPruner {
    /// Build a pruner from a regex pattern, using the default
    /// token-to-char mapping (token `t` ↔ `char::from_u32(t)` for
    /// `t <= MAX_DEFAULT_TOKEN`).
    ///
    /// # Errors
    ///
    /// Returns [`RegexPrunerError`] if the pattern fails to compile or the
    /// DFA cannot be constructed.
    pub fn from_pattern(pattern: &str) -> Result<Self, RegexPrunerError> {
        Self::with_token_map(pattern, default_token_to_char)
    }

    /// Build a pruner using a custom token-to-char mapping.
    ///
    /// `map(token)` returns `Some(ch)` for valid tokens, `None` for tokens
    /// that should always be rejected. The mapping is materialized into a
    /// `Vec<Option<char>>` at construction time — the closure itself is not
    /// stored.
    ///
    /// # Errors
    ///
    /// Returns [`RegexPrunerError`] if the pattern fails to compile.
    pub fn with_token_map<F>(pattern: &str, map: F) -> Result<Self, RegexPrunerError>
    where
        F: Fn(TokenId) -> Option<char>,
    {
        // Anchor the pattern at the end with `$` so that prefix validity
        // requires the ENTIRE generated string to match the regex — not just
        // a match starting at position 0. Without `$`, a pattern like `a*`
        // would accept 'b' (the empty match at position 0 succeeds and the
        // DFA stays non-dead). With `$`, 'b' drives the DFA to a dead state
        // because "b..." can never match `^a*$`.
        //
        // The pattern is wrapped in a non-capturing group first to keep `$`
        // scoped to the whole alternation (e.g., "yes|no" → "(?:yes|no)$").
        let anchored_pattern = format!("(?:{pattern})$");

        let dfa = dense::Builder::new()
            .configure(dense::Config::new().start_kind(StartKind::Anchored))
            .build(&anchored_pattern)
            .map_err(|e| RegexPrunerError {
                message: format!("{e}"),
            })?;

        let token_to_char = build_token_to_char_vec(&map);

        Ok(Self {
            dfa,
            pattern: pattern.to_string(),
            token_to_char,
            arm_id: REGEX_ARM_ID,
            label: REGEX_LABEL.to_string(),
        })
    }

    /// Override the default [`arm_id`](ScreeningPruner::arm_id).
    pub fn with_arm_id(mut self, arm_id: ArmId) -> Self {
        self.arm_id = arm_id;
        self
    }

    /// Override the default arm label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Read-only access to the original pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Number of tokens in the configured mapping.
    pub fn vocab_size(&self) -> usize {
        self.token_to_char.len()
    }

    /// Look up the character a token maps to, if any.
    pub fn token_to_char(&self, token: TokenId) -> Option<char> {
        self.token_to_char.get(token as usize).copied().flatten()
    }

    /// Convert a token sequence to its string representation.
    ///
    /// Tokens without a character mapping are skipped (the resulting string
    /// is the concatenation of all mapped tokens, in order).
    fn tokens_to_string(&self, tokens: &[TokenId]) -> String {
        let mut out = String::with_capacity(tokens.len());
        for &token in tokens {
            if let Some(ch) = self.token_to_char(token) {
                out.push(ch);
            }
        }
        out
    }

    /// Walk the anchored DFA through `prefix` and return the final state ID.
    ///
    /// Returns `None` if the DFA enters a dead state at any point (including
    /// if the start state itself is dead — e.g., for an unsatisfiable pattern).
    fn walk_prefix(&self, prefix: &str) -> Option<StateID> {
        let input = Input::new(prefix).anchored(Anchored::Yes);

        let mut state = match self.dfa.start_state_forward(&input) {
            Ok(s) => s,
            Err(_) => return None,
        };

        if self.dfa.is_dead_state(state) {
            return None;
        }

        for &byte in prefix.as_bytes() {
            state = self.dfa.next_state(state, byte);
            if self.dfa.is_dead_state(state) {
                return None;
            }
        }

        Some(state)
    }

    /// Transition from `state` through the UTF-8 bytes of `ch`.
    ///
    /// Returns `false` if the DFA enters a dead state at any byte.
    fn transition_char(&self, state: StateID, ch: char) -> bool {
        let mut buf = [0u8; 4];
        let bytes = ch.encode_utf8(&mut buf);
        let mut s = state;
        for &byte in bytes.as_bytes() {
            s = self.dfa.next_state(s, byte);
            if self.dfa.is_dead_state(s) {
                return false;
            }
        }
        true
    }
}

impl ConstraintPruner for RegexPruner {
    fn is_valid(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        let prefix = self.tokens_to_string(parent_tokens);
        let Some(state) = self.walk_prefix(&prefix) else {
            return false;
        };
        let Some(ch) = self.token_to_char(token) else {
            return false;
        };
        self.transition_char(state, ch)
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        let prefix = self.tokens_to_string(parent_tokens);
        let prefix_state = match self.walk_prefix(&prefix) {
            Some(s) => s,
            None => {
                let len = candidates.len().min(results.len());
                results[..len].fill(false);
                return;
            }
        };

        let len = candidates.len().min(results.len());
        for i in 0..len {
            results[i] = match self.token_to_char(candidates[i]) {
                Some(ch) => self.transition_char(prefix_state, ch),
                None => false,
            };
        }
    }

    fn manifold_score(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        ScreeningPruner::screen(self, depth, token, parent_tokens)
    }

    fn propagate(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {
        // Stateless — no internal structure to update.
    }

    fn on_backtrack(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {
        // Stateless — nothing to undo.
    }
}

impl ScreeningPruner for RegexPruner {
    fn arm_id(&self) -> ArmId {
        self.arm_id
    }

    fn arm_label(&self) -> &str {
        &self.label
    }

    fn screen(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
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

// ── Helpers ────────────────────────────────────────────────────

/// Default token-to-char mapping: token `t` ↔ `char::from_u32(t)` for
/// `t <= MAX_DEFAULT_TOKEN`, else `None`.
fn default_token_to_char(token: TokenId) -> Option<char> {
    if token <= MAX_DEFAULT_TOKEN {
        char::from_u32(token)
    } else {
        None
    }
}

/// Materialize a token-to-char mapping into a dense `Vec<Option<char>>`.
///
/// Samples the mapping for tokens `0..` until a long run of `None` results
/// indicates the useful range is exhausted, then trims trailing `None`s.
fn build_token_to_char_vec<F>(map: &F) -> Vec<Option<char>>
where
    F: Fn(TokenId) -> Option<char>,
{
    const CAP: usize = 256;
    const NONE_RUN_LIMIT: usize = 64;

    let mut out: Vec<Option<char>> = Vec::with_capacity(CAP);
    let mut none_run = 0usize;

    let mut token = 0u32;
    while token < TokenId::MAX && none_run < NONE_RUN_LIMIT {
        match map(token) {
            Some(ch) => {
                out.push(Some(ch));
                none_run = 0;
            }
            None => {
                out.push(None);
                none_run += 1;
            }
        }
        token += 1;
    }

    while out.last().is_some_and(|last| last.is_none()) {
        out.pop();
    }

    out
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──

    #[test]
    fn test_from_pattern_lowercase_letters() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid pattern");
        assert_eq!(pruner.pattern(), "[a-z]+");
        assert!(pruner.vocab_size() > 0);
    }

    #[test]
    fn test_from_pattern_invalid_returns_error() {
        let result = RegexPruner::from_pattern("[a-z");
        assert!(result.is_err(), "unclosed character class must error");
    }

    #[test]
    fn test_with_arm_id_overrides_default() {
        let pruner = RegexPruner::from_pattern("[a-z]+")
            .expect("valid")
            .with_arm_id(42);
        assert_eq!(pruner.arm_id(), 42);
    }

    #[test]
    fn test_with_label_overrides_default() {
        let pruner = RegexPruner::from_pattern("[a-z]+")
            .expect("valid")
            .with_label("custom-regex");
        assert_eq!(pruner.arm_label(), "custom-regex");
    }

    #[test]
    fn test_with_token_map_custom_mapping() {
        let pruner = RegexPruner::with_token_map("[ab]+", |t| match t {
            0 => Some('a'),
            1 => Some('b'),
            _ => None,
        })
        .expect("valid");

        assert_eq!(pruner.token_to_char(0), Some('a'));
        assert_eq!(pruner.token_to_char(1), Some('b'));
        assert_eq!(pruner.token_to_char(2), None);
    }

    // ── Default token-to-char mapping ──

    #[test]
    fn test_default_token_to_char_lowercase_letters() {
        assert_eq!(default_token_to_char(b'a' as TokenId), Some('a'));
        assert_eq!(default_token_to_char(b'z' as TokenId), Some('z'));
    }

    #[test]
    fn test_default_token_to_char_digits() {
        assert_eq!(default_token_to_char(b'0' as TokenId), Some('0'));
        assert_eq!(default_token_to_char(b'9' as TokenId), Some('9'));
    }

    #[test]
    fn test_default_token_to_char_rejects_non_chars() {
        assert_eq!(default_token_to_char(0xD800), None);
    }

    #[test]
    fn test_default_token_to_char_rejects_out_of_range() {
        assert_eq!(default_token_to_char(0x110000), None);
    }

    // ── is_valid: lowercase patterns ──

    #[test]
    fn test_is_valid_lowercase_pattern_accepts_lowercase() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        assert!(
            pruner.is_valid(0, b'a' as TokenId, &[]),
            "token 'a' should be valid for [a-z]+"
        );
    }

    #[test]
    fn test_is_valid_lowercase_pattern_rejects_digit() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        assert!(
            !pruner.is_valid(0, b'0' as TokenId, &[]),
            "token '0' should be invalid for [a-z]+"
        );
    }

    #[test]
    fn test_is_valid_lowercase_pattern_rejects_uppercase() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        assert!(
            !pruner.is_valid(0, b'A' as TokenId, &[]),
            "token 'A' should be invalid for [a-z]+"
        );
    }

    // ── is_valid: digit patterns ──

    #[test]
    fn test_is_valid_digit_pattern_accepts_digit() {
        let pruner = RegexPruner::from_pattern("[0-9]+").expect("valid");
        assert!(
            pruner.is_valid(0, b'5' as TokenId, &[]),
            "token '5' should be valid for [0-9]+"
        );
    }

    #[test]
    fn test_is_valid_digit_pattern_rejects_letter() {
        let pruner = RegexPruner::from_pattern("[0-9]+").expect("valid");
        assert!(
            !pruner.is_valid(0, b'a' as TokenId, &[]),
            "token 'a' should be invalid for [0-9]+"
        );
    }

    // ── is_valid: anchored patterns ──

    #[test]
    fn test_is_valid_fixed_length_pattern_first_digit() {
        let pruner = RegexPruner::from_pattern("[0-9]{4}").expect("valid");
        assert!(
            pruner.is_valid(0, b'1' as TokenId, &[]),
            "first digit should be valid for [0-9]{{4}}"
        );
    }

    #[test]
    fn test_is_valid_fixed_length_pattern_completes() {
        let pruner = RegexPruner::from_pattern("[0-9]{4}").expect("valid");
        let prefix = [b'1' as TokenId, b'2' as TokenId, b'3' as TokenId];
        assert!(
            pruner.is_valid(3, b'4' as TokenId, &prefix),
            "fourth digit should complete [0-9]{{4}} match"
        );
    }

    #[test]
    fn test_is_valid_fixed_length_pattern_rejects_fifth_digit() {
        let pruner = RegexPruner::from_pattern("[0-9]{4}").expect("valid");
        let prefix = [
            b'1' as TokenId,
            b'2' as TokenId,
            b'3' as TokenId,
            b'4' as TokenId,
        ];
        assert!(
            !pruner.is_valid(4, b'5' as TokenId, &prefix),
            "fifth digit should be invalid for exact [0-9]{{4}}"
        );
    }

    #[test]
    fn test_is_valid_rejects_unmapped_token() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        assert!(!pruner.is_valid(0, 0x110000, &[]));
    }

    // ── is_valid: structured patterns (the key DFA test) ──

    #[test]
    fn test_is_valid_phone_number_first_digit() {
        let pruner = RegexPruner::from_pattern("[0-9]{3}-[0-9]{4}").expect("valid");
        assert!(
            pruner.is_valid(0, b'1' as TokenId, &[]),
            "first digit '1' should be valid for [0-9]{{3}}-[0-9]{{4}}"
        );
    }

    #[test]
    fn test_is_valid_phone_number_after_three_digits_needs_dash() {
        let pruner = RegexPruner::from_pattern("[0-9]{3}-[0-9]{4}").expect("valid");
        let prefix = [b'1' as TokenId, b'2' as TokenId, b'3' as TokenId];

        // Dash should be accepted after 3 digits.
        assert!(
            pruner.is_valid(3, b'-' as TokenId, &prefix),
            "'-' should be valid after 3 digits"
        );

        // A 4th digit should be rejected (pattern expects dash next).
        assert!(
            !pruner.is_valid(3, b'4' as TokenId, &prefix),
            "4th digit should be invalid — pattern expects dash"
        );
    }

    #[test]
    fn test_is_valid_phone_number_complete_sequence() {
        let pruner = RegexPruner::from_pattern("[0-9]{3}-[0-9]{4}").expect("valid");

        let mut prefix: Vec<TokenId> = Vec::new();

        for &digit in b"123" {
            assert!(
                pruner.is_valid(prefix.len(), digit as TokenId, &prefix),
                "digit '{digit_ch}' should be valid",
                digit_ch = digit as char
            );
            prefix.push(digit as TokenId);
        }

        assert!(pruner.is_valid(prefix.len(), b'-' as TokenId, &prefix));
        prefix.push(b'-' as TokenId);

        for &digit in b"4567" {
            assert!(
                pruner.is_valid(prefix.len(), digit as TokenId, &prefix),
                "digit '{digit_ch}' should be valid",
                digit_ch = digit as char
            );
            prefix.push(digit as TokenId);
        }

        // 8th character after dash (too many digits) should be invalid.
        assert!(
            !pruner.is_valid(prefix.len(), b'8' as TokenId, &prefix),
            "extra digit after complete match should be invalid"
        );
    }

    // ── is_valid: alternation ──

    #[test]
    fn test_is_valid_alternation_yes_no() {
        let pruner = RegexPruner::from_pattern("yes|no").expect("valid");

        // 'y' and 'n' are valid at depth 0.
        assert!(pruner.is_valid(0, b'y' as TokenId, &[]));
        assert!(pruner.is_valid(0, b'n' as TokenId, &[]));

        // 'a' is invalid at depth 0 (neither "yes" nor "no" starts with 'a').
        assert!(!pruner.is_valid(0, b'a' as TokenId, &[]));

        // After 'y', 'e' is valid but 'o' is invalid.
        assert!(pruner.is_valid(1, b'e' as TokenId, &[b'y' as TokenId]));
        assert!(!pruner.is_valid(1, b'o' as TokenId, &[b'y' as TokenId]));

        // After "no", any further character is invalid.
        let no_prefix = [b'n' as TokenId, b'o' as TokenId];
        assert!(!pruner.is_valid(2, b'!' as TokenId, &no_prefix));
    }

    // ── is_valid: sequence progression ──

    #[test]
    fn test_is_valid_lowercase_sequence() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");

        let mut prefix: Vec<TokenId> = Vec::new();
        for &ch in b"abc" {
            assert!(
                pruner.is_valid(prefix.len(), ch as TokenId, &prefix),
                "token '{ch_char}' at depth {depth} should be valid",
                ch_char = ch as char,
                depth = prefix.len()
            );
            prefix.push(ch as TokenId);
        }
    }

    // ── Batch operations ──

    #[test]
    fn test_batch_is_valid_matches_individual() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let candidates: Vec<TokenId> = vec![
            b'a' as TokenId,
            b'0' as TokenId,
            b'z' as TokenId,
            b'A' as TokenId,
        ];
        let parent: Vec<TokenId> = vec![b'h' as TokenId];

        let mut batch = vec![false; candidates.len()];
        pruner.batch_is_valid(1, &candidates, &parent, &mut batch);

        for (i, &tok) in candidates.iter().enumerate() {
            let individual = pruner.is_valid(1, tok, &parent);
            let batch_result = batch[i];
            assert_eq!(
                batch_result, individual,
                "batch[{i}]={batch_result} != individual {individual} for token {tok}"
            );
        }
    }

    #[test]
    fn test_batch_is_valid_handles_shorter_results() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let candidates: Vec<TokenId> = vec![
            b'a' as TokenId,
            b'b' as TokenId,
            b'c' as TokenId,
            b'd' as TokenId,
        ];
        let mut results = vec![false; 2];

        pruner.batch_is_valid(0, &candidates, &[], &mut results);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_batch_is_valid_empty_candidates() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let mut results: Vec<bool> = vec![];

        pruner.batch_is_valid(0, &[], &[], &mut results);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_is_valid_invalid_prefix_rejects_all() {
        let pruner = RegexPruner::from_pattern("[0-9]+").expect("valid");
        // Parent "abc" is already an invalid prefix for [0-9]+.
        let parent: Vec<TokenId> = vec![b'a' as TokenId];
        let candidates: Vec<TokenId> = vec![b'1' as TokenId, b'2' as TokenId];

        let mut results = vec![true; candidates.len()];
        pruner.batch_is_valid(1, &candidates, &parent, &mut results);
        assert_eq!(results, vec![false, false]);
    }

    // ── screen / batch_screen ──

    #[test]
    fn test_screen_returns_one_for_valid() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let score = pruner.screen(0, b'a' as TokenId, &[]);
        assert!((score - 1.0).abs() < 1e-6, "valid token should score 1.0");
    }

    #[test]
    fn test_screen_returns_zero_for_invalid() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let score = pruner.screen(0, b'0' as TokenId, &[]);
        assert!((score - 0.0).abs() < 1e-6, "invalid token should score 0.0");
    }

    #[test]
    fn test_batch_screen_returns_binary_scores() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let candidates: Vec<TokenId> = vec![b'a' as TokenId, b'0' as TokenId, b'z' as TokenId];

        let mut scores = vec![0.0f32; candidates.len()];
        pruner.batch_screen(0, &candidates, &[], &mut scores);

        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[1] - 0.0).abs() < 1e-6);
        assert!((scores[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_manifold_score_equals_screen() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let ms = pruner.manifold_score(0, b'a' as TokenId, &[]);
        let sc = ScreeningPruner::screen(&pruner, 0, b'a' as TokenId, &[]);
        assert!((ms - sc).abs() < 1e-6);
    }

    // ── Stateless: propagate and on_backtrack ──

    #[test]
    fn test_propagate_is_noop() {
        let mut pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let before = pruner.is_valid(0, b'a' as TokenId, &[]);
        pruner.propagate(0, b'a' as TokenId, &[b'a' as TokenId]);
        let after = pruner.is_valid(0, b'a' as TokenId, &[]);
        assert_eq!(before, after);
    }

    #[test]
    fn test_on_backtrack_is_noop() {
        let mut pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let before = pruner.is_valid(0, b'a' as TokenId, &[]);
        pruner.on_backtrack(0, b'a' as TokenId, &[]);
        let after = pruner.is_valid(0, b'a' as TokenId, &[]);
        assert_eq!(before, after);
    }

    // ── Trait object dispatch ──

    #[test]
    fn test_constraint_pruner_trait_object() {
        let pruner: Box<dyn ConstraintPruner> =
            Box::new(RegexPruner::from_pattern("[a-z]+").expect("valid"));
        assert!(pruner.is_valid(0, b'a' as TokenId, &[]));
        assert!(!pruner.is_valid(0, b'0' as TokenId, &[]));
    }

    #[test]
    fn test_screening_pruner_trait_object() {
        let pruner: Box<dyn ScreeningPruner> =
            Box::new(RegexPruner::from_pattern("[a-z]+").expect("valid"));
        assert_eq!(pruner.arm_id(), REGEX_ARM_ID);
        assert_eq!(pruner.arm_label(), REGEX_LABEL);

        let score = pruner.screen(0, b'a' as TokenId, &[]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    // ── build_token_to_char_vec ──

    #[test]
    fn test_build_token_to_char_vec_dense_mapping() {
        // Map ONLY tokens 0-25 to 'a'-'z'; everything else is None.
        // Without the explicit None branch, the mapping would cover the
        // entire Unicode range (millions of entries).
        let vec = build_token_to_char_vec(&|t: TokenId| match t {
            n if n < 26 => char::from_u32(t + b'a' as u32),
            _ => None,
        });
        assert_eq!(vec.len(), 26);
        assert_eq!(vec[0], Some('a'));
        assert_eq!(vec[25], Some('z'));
    }

    #[test]
    fn test_build_token_to_char_vec_sparse_mapping() {
        let vec = build_token_to_char_vec(&|t: TokenId| match t {
            0 => Some('x'),
            1 => Some('y'),
            _ => None,
        });
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], Some('x'));
        assert_eq!(vec[1], Some('y'));
    }

    // ── tokens_to_string ──

    #[test]
    fn test_tokens_to_string_simple() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        let tokens = vec![
            b'h' as TokenId,
            b'e' as TokenId,
            b'l' as TokenId,
            b'l' as TokenId,
            b'o' as TokenId,
        ];
        assert_eq!(pruner.tokens_to_string(&tokens), "hello");
    }

    #[test]
    fn test_tokens_to_string_skips_invalid() {
        let pruner = RegexPruner::with_token_map("[a-z]+", |t| match t {
            0 => Some('a'),
            _ => None,
        })
        .expect("valid");

        let tokens = vec![0u32, 99, 0];
        assert_eq!(pruner.tokens_to_string(&tokens), "aa");
    }

    #[test]
    fn test_tokens_to_string_empty() {
        let pruner = RegexPruner::from_pattern("[a-z]+").expect("valid");
        assert_eq!(pruner.tokens_to_string(&[]), "");
    }

    // ── Send + Sync ──

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RegexPruner>();
    }

    // ── Error display ──

    #[test]
    fn test_regex_pruner_error_display() {
        let err = RegexPruner::from_pattern("[a-z")
            .err()
            .expect("expected compile error");
        let display = format!("{err}");
        assert!(display.contains("regex DFA build error"));
    }

    // ── Edge cases ──

    #[test]
    fn test_is_valid_empty_string_pattern() {
        // Pattern that matches empty string: a* matches "", "a", "aa", etc.
        let pruner = RegexPruner::from_pattern("a*").expect("valid");
        assert!(pruner.is_valid(0, b'a' as TokenId, &[]));
        // 'b' is invalid for a*.
        assert!(!pruner.is_valid(0, b'b' as TokenId, &[]));
    }

    #[test]
    fn test_is_valid_word_boundary_pattern() {
        let pruner = RegexPruner::from_pattern("[a-zA-Z]+").expect("valid");
        assert!(pruner.is_valid(0, b'a' as TokenId, &[]));
        assert!(pruner.is_valid(0, b'A' as TokenId, &[]));
        assert!(!pruner.is_valid(0, b' ' as TokenId, &[]));
    }

    #[test]
    fn test_is_valid_optional_group() {
        // Pattern: "ab" optionally followed by "c".
        let pruner = RegexPruner::from_pattern("abc?").expect("valid");

        // 'a' at depth 0.
        assert!(pruner.is_valid(0, b'a' as TokenId, &[]));

        // 'b' at depth 1.
        assert!(pruner.is_valid(1, b'b' as TokenId, &[b'a' as TokenId]));

        // 'c' at depth 2 (optional but valid).
        assert!(pruner.is_valid(2, b'c' as TokenId, &[b'a' as TokenId, b'b' as TokenId]));

        // 'd' at depth 2 (invalid — neither completes "ab" nor "abc").
        assert!(!pruner.is_valid(2, b'd' as TokenId, &[b'a' as TokenId, b'b' as TokenId]));
    }
}
