//! Shared deterministic BLAKE3 token→vector projection.
//!
//! Both [`InMemoryKgStore`](super::InMemoryKgStore) (triple embeddings via
//! role-scoped seed tags) and [`SchemaCentroid`](super::SchemaCentroid)
//! (entity embeddings) derive their continuous representations from this
//! single projection. Centralizing it guarantees the modelless thesis holds:
//! no learned weights anywhere — every embedding is a deterministic function
//! of `(token, seed_tag, dim)`.
//!
//! # Algorithm
//!
//! BLAKE3 counter-mode expansion. Each 32-byte hash block (seeded by
//! `seed_tag || token || counter`) yields 8 floats; blocks accumulate until
//! `dim` floats are available, then the byte stream is truncated to the exact
//! required length. Each 4-byte lane is read as a little-endian `u32` and
//! mapped to `[-1.0, 1.0]` via `(u / u32::MAX) * 2.0 - 1.0`.
//!
//! # Determinism
//!
//! Identical `(token, dim, seed_tag)` always produces the same vector, across
//! calls and across process restarts. There is no RNG state, no I/O, no
//! global mutable state. This is the bedrock property that makes KG
//! embeddings reproducible and auditable.

use blake3::Hasher;

use crate::types::TokenId;

/// Deterministic latent vector for `token`, seeded by `seed_tag`.
///
/// The byte signature `seed_tag` is mixed into the BLAKE3 seed so that
/// callers can namespace embeddings by role or category (e.g. `b"s"` for
/// subject, `b"entity"` for entity-init). Two callers using different seed
/// tags get statistically independent vectors for the same token, which is
/// how the role-positioned triple embedding in
/// [`InMemoryKgStore`](super::InMemoryKgStore) keeps subject/predicate/object
/// coordinates disjoint.
///
/// # Parameters
///
/// - `token`: vocabulary token ID to embed.
/// - `dim`: desired output dimension in floats. `0` returns an empty vector.
/// - `seed_tag`: byte namespace mixed into the hash seed.
///
/// # Returns
///
/// A `Vec<f32>` of length `dim` with every coordinate in `[-1.0, 1.0]`.
///
/// # Determinism
///
/// Same `(token, dim, seed_tag)` → identical bytes across calls, threads,
/// and runs. Safe to call from multiple threads concurrently (no shared
/// mutable state).
pub fn token_vector(token: TokenId, dim: usize, seed_tag: &[u8]) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let needed_bytes = dim.saturating_mul(4);
    let mut bytes = Vec::with_capacity(needed_bytes);
    let mut counter: u64 = 0;
    while bytes.len() < needed_bytes {
        let mut hasher = Hasher::new();
        hasher.update(seed_tag);
        hasher.update(&token.to_le_bytes());
        hasher.update(&counter.to_le_bytes());
        bytes.extend_from_slice(hasher.finalize().as_bytes());
        counter = match counter.checked_add(1) {
            Some(next) => next,
            // u64 overflow at ~1.8e19 blocks (≈ 5.9e20 floats). Infeasible
            // to reach in practice; return what we have rather than wrap
            // the counter and corrupt determinism.
            None => break,
        };
    }
    bytes.truncate(needed_bytes);
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            // `chunks_exact(4)` guarantees `chunk.len() == 4`.
            let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let u = u32::from_le_bytes(raw);
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shape ──

    #[test]
    fn test_zero_dim_returns_empty() {
        let v = token_vector(42, 0, b"entity");
        assert!(v.is_empty());
    }

    #[test]
    fn test_output_length_matches_dim() {
        for dim in [1usize, 2, 7, 8, 9, 31, 32, 33, 96, 100, 256] {
            let v = token_vector(7, dim, b"entity");
            assert_eq!(v.len(), dim, "dim {dim} must produce length {dim}");
        }
    }

    // ── Determinism ──

    #[test]
    fn test_same_inputs_produce_identical_vectors() {
        let a = token_vector(123, 32, b"entity");
        let b = token_vector(123, 32, b"entity");
        assert_eq!(a, b, "same inputs must produce identical vectors");
    }

    #[test]
    fn test_different_tokens_produce_different_vectors() {
        let a = token_vector(1, 32, b"entity");
        let b = token_vector(2, 32, b"entity");
        assert_ne!(a, b, "different tokens must produce different vectors");
    }

    #[test]
    fn test_different_seed_tags_produce_different_vectors() {
        // Same token, different seed tag → different vector (namespacing).
        let a = token_vector(5, 32, b"entity");
        let b = token_vector(5, 32, b"s");
        assert_ne!(a, b, "different seed tags must produce different vectors");
    }

    #[test]
    fn test_different_dims_produce_different_lengths() {
        let a = token_vector(9, 16, b"entity");
        let b = token_vector(9, 32, b"entity");
        assert_eq!(a.len(), 16);
        assert_eq!(b.len(), 32);
        // Prefix independence: the first 16 of the 32-vector need NOT equal
        // the 16-vector (truncation is on the byte stream, not the float
        // stream), so we only assert lengths here.
    }

    #[test]
    fn test_token_zero_is_not_special() {
        // Token 0 must produce a real, non-degenerate vector — not zeros.
        let v = token_vector(0, 32, b"entity");
        assert_eq!(v.len(), 32);
        assert!(
            v.iter().any(|&x| x != 0.0),
            "token 0 must not produce an all-zero vector"
        );
    }

    // ── Range ──

    #[test]
    fn test_coordinates_bounded_to_unit_range() {
        for token in [0u32, 1, 7, 42, 255, 1024, u32::MAX] {
            let v = token_vector(token, 64, b"entity");
            for (i, x) in v.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(x),
                    "token {token} coord {i} = {x} outside [-1, 1]"
                );
            }
        }
    }

    // ── Seed-tag isolation ──

    #[test]
    fn test_seed_tag_namespaces_are_independent() {
        // The four role/entity tags used across the crate must all produce
        // distinct vectors for the same token. This is the namespacing
        // guarantee that the triple embedding and SchemaCentroid rely on.
        let token = 42;
        let dim = 32;
        let entity = token_vector(token, dim, b"entity");
        let subject = token_vector(token, dim, b"s");
        let predicate = token_vector(token, dim, b"p");
        let object = token_vector(token, dim, b"o");

        let mut seen = vec![entity.clone()];
        for v in [&subject, &predicate, &object] {
            assert!(
                !seen.contains(v),
                "seed tag namespace collision for token {token}"
            );
            seen.push(v.clone());
        }
    }

    // ── Larger dims (stress the counter-mode expansion path) ──

    #[test]
    fn test_large_dim_uses_multiple_hash_blocks() {
        // dim = 256 needs 1024 bytes = 32 BLAKE3 blocks. Verifies the
        // counter loop accumulates across blocks correctly.
        let v = token_vector(99, 256, b"entity");
        assert_eq!(v.len(), 256);
        for x in &v {
            assert!((-1.0..=1.0).contains(x));
        }
        // At least some coordinates must be non-zero (sanity).
        assert!(
            v.iter().any(|&x| x.abs() > 1e-6),
            "large-dim vector must not be degenerate"
        );
    }

    #[test]
    fn test_dim_one_works() {
        // Boundary: smallest non-zero dim. Single float from first 4 bytes.
        let v = token_vector(7, 1, b"entity");
        assert_eq!(v.len(), 1);
        assert!((-1.0..=1.0).contains(&v[0]));
    }

    // ── Cross-call stability ──

    #[test]
    fn test_repeated_calls_are_stable() {
        let mut prev = token_vector(13, 48, b"entity");
        for _ in 0..100 {
            let cur = token_vector(13, 48, b"entity");
            assert_eq!(cur, prev, "repeated calls must remain stable");
            prev = cur;
        }
    }

    // ── Empty seed tag is legal (not recommended, but defined) ──

    #[test]
    fn test_empty_seed_tag_produces_valid_vector() {
        // An empty seed tag is legal BLAKE3 input; the token + counter
        // still produce a deterministic stream. Mostly a defensive test.
        let v = token_vector(5, 16, b"");
        assert_eq!(v.len(), 16);
        for x in &v {
            assert!((-1.0..=1.0).contains(x));
        }
    }
}
