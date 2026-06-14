//! Decoupled type definitions for the neuro-symbolic engine.
//!
//! All structs and impls live here, separate from traits (`traits.rs`) and
//! logic (`decode.rs`, `pruners/`, `draft/`). This keeps the dependency graph
//! flat: traits depend on types, logic depends on both.

/// Token identifier in the vocabulary.
pub type TokenId = u32;

/// Log-probability distribution over the vocabulary (len = vocab_size).
pub type Logits = Vec<f32>;

/// Stable identifier for a bandit arm (pruner) within a [`BanditPruner`](crate::bandit::BanditPruner).
///
/// Assigned by arm insertion order. Used by the [`TrialLog`](crate::bandit::TrialLog) to
/// track per-arm statistics and by policies to report selections.
pub type ArmId = u32;

/// Configuration for the speculative decode loop.
#[derive(Clone, Debug)]
pub struct DecodeConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Number of top candidates to consider per step (draft beam width).
    pub top_k: usize,
    /// RNG seed for reproducible runs.
    pub seed: u64,
    /// Enable backtracking search on dead-ends (tree search mode).
    pub backtrack: bool,
    /// Maximum exploration attempts before giving up (backtrack mode only).
    pub max_attempts: u64,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            max_tokens: 81,
            top_k: 9,
            seed: 42,
            backtrack: true,
            max_attempts: 100_000,
        }
    }
}

/// Result of a speculative decode run.
#[derive(Clone, Debug)]
pub struct DecodeResult {
    /// The generated token sequence.
    pub tokens: Vec<TokenId>,
    /// Whether the final sequence passed verification.
    pub verified: bool,
    /// Number of exploration attempts (forward steps + backtracks).
    pub attempts: u64,
    /// BLAKE3 hash of the token sequence for audit/reproducibility.
    pub hash: [u8; 32],
}

impl DecodeResult {
    /// Compute BLAKE3 hash from a token slice.
    ///
    /// Used for audit trails and verifying two decode runs produced
    /// identical output. Zero-allocation via streaming hasher.
    pub fn hash_tokens(tokens: &[TokenId]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for t in tokens {
            hasher.update(&t.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_config_default() {
        let config = DecodeConfig::default();
        assert_eq!(config.max_tokens, 81);
        assert_eq!(config.top_k, 9);
        assert!(config.backtrack);
    }

    #[test]
    fn test_hash_tokens_deterministic() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let hash_a = DecodeResult::hash_tokens(&tokens);
        let hash_b = DecodeResult::hash_tokens(&tokens);
        assert_eq!(hash_a, hash_b, "same tokens must produce same hash");
    }

    #[test]
    fn test_hash_tokens_differ_on_change() {
        let a = vec![1u32, 2, 3];
        let b = vec![1u32, 2, 4];
        let hash_a = DecodeResult::hash_tokens(&a);
        let hash_b = DecodeResult::hash_tokens(&b);
        assert_ne!(
            hash_a, hash_b,
            "different tokens must produce different hash"
        );
    }

    #[test]
    fn test_hash_tokens_empty() {
        let hash = DecodeResult::hash_tokens(&[]);
        // Empty input still produces a valid BLAKE3 hash (the zero-message hash)
        assert_ne!(hash, [0u8; 32], "BLAKE3 of empty input is not all zeros");
    }
}
