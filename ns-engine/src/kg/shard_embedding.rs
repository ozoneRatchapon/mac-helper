//! JL (Johnson-Lindenstrauss) random orthogonal projection for KG embedding
//! dimensionality reduction.
//!
//! KG triple embeddings produced by [`InMemoryKgStore`](super::InMemoryKgStore)
//! are intentionally high-dimensional (default 96-dim: 32 per role × 3 roles)
//! so that subject, predicate, and object coordinates occupy disjoint
//! subspaces. For mid-layer K/V injection, that full width is wasteful —
//! the decode loop needs compact keys and values, not the full disentangled
//! representation.
//!
//! [`ShardEmbedding`] projects a high-dim f32 vector down to a smaller f32
//! vector via a random orthonormal matrix (rows mutually orthogonal, unit
//! norm). This is a JL-style guarantee: for a target dimension
//! `d = O(ε⁻² log n)`, pairwise distances between `n` projected vectors
//! approximate the original distances within factor `(1 ± ε)` with high
//! probability. The orthonormal construction additionally guarantees that
//! the projected norm never exceeds the input norm (it's an orthogonal
//! projection onto a random subspace).
//!
//! # Determinism
//!
//! The projection matrix is derived purely from `(input_dim, output_dim, seed)`
//! via BLAKE3-seeded Box-Muller Gaussian generation + Gram-Schmidt
//! orthonormalization. No learned weights, no global RNG state. Same
//! construction parameters → identical matrix across calls and runs. This
//! honors the modelless thesis: every reduction is a reproducible function
//! of the seed.
//!
//! # Construction cost
//!
//! [`ShardEmbedding::new`] runs Gram-Schmidt in `O(output_dim² × input_dim)`.
//! For typical KG sizes (96 → 32), that is ≈ 100k flops — negligible, paid
//! once at construction. [`ShardEmbedding::project`] is
//! `O(output_dim × input_dim)` per call.

use blake3::Hasher;

/// Seed tag namespacing ShardEmbedding's Gaussian stream from all other
/// BLAKE3-derived structures in the crate (projection.rs uses `b"entity"`,
/// `b"s"`, `b"p"`, `b"o"`; here we use `b"shard"`).
const SEED_TAG: &[u8] = b"shard";

/// Tolerance for Gram-Schmidt residual norm (below this, a candidate row
/// is treated as linearly dependent and discarded). Compared against the
/// *squared* residual norm to avoid a sqrt per candidate.
const ORTHO_EPSILON_SQ: f32 = 1e-12;

// ── Gaussian stream (BLAKE3-seeded Box-Muller) ─────────────────

/// Deterministic standard-normal sample stream from a BLAKE3 seed.
///
/// Yields `N(0, 1)` floats via Box-Muller: each pair of BLAKE3-derived u32
/// values produces two independent standard normals. The stream is purely a
/// function of `(seed, counter)` — no global state, no I/O. Thread-safe by
/// construction (owned, not shared).
struct GaussianStream {
    seed: u64,
    counter: u64,
    /// Second normal from the last Box-Muller call (buffered for the next
    /// [`next`](Self::next)). Box-Muller produces a pair; we return one and
    /// stash the other to avoid waste.
    pending: Option<f32>,
}

impl GaussianStream {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            counter: 0,
            pending: None,
        }
    }

    /// Draw the next standard-normal sample.
    fn next(&mut self) -> f32 {
        match self.pending.take() {
            Some(z) => z,
            None => {
                let u1 = self.next_u32();
                let u2 = self.next_u32();
                let (z0, z1) = box_muller(u1, u2);
                self.pending = Some(z1);
                z0
            }
        }
    }

    /// Next u32 from the BLAKE3 counter stream.
    ///
    /// Mirrors the counter-mode expansion in
    /// [`projection::token_vector`](super::projection::token_vector): each
    /// call hashes `(SEED_TAG, seed, counter)` and reads the first 4 bytes
    /// of the digest as a little-endian u32.
    fn next_u32(&mut self) -> u32 {
        let mut hasher = Hasher::new();
        hasher.update(SEED_TAG);
        hasher.update(&self.seed.to_le_bytes());
        hasher.update(&self.counter.to_le_bytes());
        let digest = *hasher.finalize().as_bytes();
        // Saturate on overflow — u64 overflow needs ~1.8e19 calls, infeasible.
        // Saturating keeps the stream well-defined rather than wrapping back
        // to counter=0 (which would replay bytes and break independence).
        self.counter = self.counter.saturating_add(1);
        u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]])
    }
}

/// Box-Muller transform: two uniform u32 bits → two independent `N(0,1)`.
///
/// `u1` is mapped to `(0, 1]` (open at 0 to keep `ln(u1)` finite);
/// `u2` is mapped to `[0, 1)` for the angle. Computation in f64 then cast
/// to f32 for precision.
fn box_muller(u1_bits: u32, u2_bits: u32) -> (f32, f32) {
    let denom = u32::MAX as f64 + 1.0;
    let u1 = (u1_bits as f64 + 1.0) / denom; // ∈ (0, 1]
    let u2 = u2_bits as f64 / denom; // ∈ [0, 1)
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = 2.0_f64 * std::f64::consts::PI * u2;
    ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
}

// ── ShardEmbedding ─────────────────────────────────────────────

/// JL random orthogonal projection: reduces a high-dim f32 embedding to a
/// smaller-dim f32 embedding.
///
/// Constructed with `(input_dim, output_dim, seed)`. The projection matrix
/// has `output_dim` rows (clamped to `input_dim` if larger), each
/// orthonormal. [`project`](Self::project) computes `y = R · x` where `R`
/// is the matrix and `x` is the input. Because `R`'s rows are orthonormal,
/// `‖y‖ ≤ ‖x‖` — the projection never amplifies, and the component of `x`
/// lying in the random subspace spanned by `R`'s rows is preserved exactly.
///
/// # Example
///
/// ```
/// use ns_engine::kg::ShardEmbedding;
///
/// // Reduce a 96-dim KG embedding to 32-dim for compact K/V injection.
/// let shard = ShardEmbedding::new(96, 32, 42);
/// let input = vec![0.5f32; 96];
/// let output = shard.project(&input);
/// assert_eq!(output.len(), 32);
/// assert_eq!(shard.input_dim(), 96);
/// assert_eq!(shard.output_dim(), 32);
/// ```
#[derive(Clone, Debug)]
pub struct ShardEmbedding {
    /// Row-major projection matrix: `output_dim` rows × `input_dim` cols.
    /// Rows are orthonormal (mutually orthogonal, unit norm), except
    /// possibly trailing zero rows in the degenerate case where Gram-Schmidt
    /// could not find enough independent vectors (only when
    /// `output_dim > input_dim`, which [`new`](Self::new) clamps).
    matrix: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    seed: u64,
}

impl ShardEmbedding {
    /// Construct a projection from `input_dim` → `output_dim` dimensions.
    ///
    /// The projection matrix is built by drawing `output_dim` random
    /// Gaussian vectors (BLAKE3-seeded Box-Muller) and orthonormalizing
    /// them via modified Gram-Schmidt.
    ///
    /// If `output_dim > input_dim`, it is clamped to `input_dim` (no more
    /// than `input_dim` mutually orthogonal vectors exist in
    /// `input_dim`-space). `input_dim = 0` yields an empty matrix that
    /// projects everything to empty.
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let effective_output = output_dim.min(input_dim);
        let matrix = build_orthonormal_matrix(input_dim, effective_output, seed);
        Self {
            matrix,
            input_dim,
            output_dim: effective_output,
            seed,
        }
    }

    /// Project `input` (length `input_dim`) down to `output_dim` floats.
    ///
    /// Computes the matrix-vector product `y = R · x`. If `input.len()` is
    /// less than `input_dim`, missing coordinates are treated as `0.0`; if
    /// longer, extra coordinates are ignored. Both cases are programmer
    /// errors caught by `debug_assert_eq!` in debug builds.
    pub fn project(&self, input: &[f32]) -> Vec<f32> {
        if self.output_dim == 0 {
            return Vec::new();
        }
        debug_assert_eq!(
            input.len(),
            self.input_dim,
            "input length must equal input_dim"
        );
        self.matrix
            .chunks_exact(self.input_dim)
            .map(|row| {
                row.iter()
                    .zip(input.iter())
                    .map(|(m, x)| m * x)
                    .sum::<f32>()
            })
            .collect()
    }

    /// Input dimensionality `D` of this projection.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Output dimensionality `d ≤ D` of this projection.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Seed used to construct the projection matrix (for audit).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// BLAKE3 audit hash of the projection matrix and its parameters.
    ///
    /// Two `ShardEmbedding` values with identical hash have identical
    /// matrices and thus produce identical projections. Useful for
    /// verifying two instances are the same projection without comparing
    /// float arrays directly.
    pub fn matrix_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed.to_le_bytes());
        hasher.update(&self.input_dim.to_le_bytes());
        hasher.update(&self.output_dim.to_le_bytes());
        for f in &self.matrix {
            hasher.update(&f.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

// ── Matrix construction ────────────────────────────────────────

/// Build an `output_dim × input_dim` row-major matrix with orthonormal rows.
///
/// Draws Gaussian vectors from a BLAKE3-seeded stream, orthonormalizes via
/// modified Gram-Schmidt. Candidate vectors whose Gram-Schmidt residual
/// falls below [`ORTHO_EPSILON_SQ`] are discarded (linearly dependent — only
/// happens in practice if `output_dim > input_dim`, which the caller
/// clamps). If fewer than `output_dim` rows are accepted, the matrix is
/// zero-padded so the shape remains `output_dim × input_dim`.
fn build_orthonormal_matrix(input_dim: usize, output_dim: usize, seed: u64) -> Vec<f32> {
    if input_dim == 0 || output_dim == 0 {
        return Vec::new();
    }
    let mut stream = GaussianStream::new(seed);
    let mut accepted: Vec<Vec<f32>> = Vec::with_capacity(output_dim);
    // Generous retry budget; for output_dim ≤ input_dim, random Gaussian
    // vectors are linearly dependent with probability zero, so we virtually
    // always succeed in exactly output_dim draws.
    let max_attempts = output_dim.saturating_mul(8).saturating_add(8);

    let mut attempts = 0usize;
    while accepted.len() < output_dim && attempts < max_attempts {
        attempts += 1;
        // Draw a fresh Gaussian vector.
        let mut v: Vec<f32> = std::iter::repeat_with(|| stream.next())
            .take(input_dim)
            .collect();

        // Modified Gram-Schmidt: subtract projections onto each accepted row.
        for u in &accepted {
            let dot: f32 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            for (vi, ui) in v.iter_mut().zip(u.iter()) {
                *vi -= dot * ui;
            }
        }

        // Accept only if residual is non-degenerate. Compare squared norm
        // against squared epsilon to avoid a sqrt per candidate.
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        if norm_sq > ORTHO_EPSILON_SQ {
            let norm = norm_sq.sqrt();
            for x in &mut v {
                *x /= norm;
            }
            accepted.push(v);
        }
        // Else: near-linearly-dependent, discard and try again (stream has
        // advanced, so the next candidate differs).
    }

    // Flatten to row-major. Zero-pad if degenerate (measure-zero for
    // output_dim ≤ input_dim).
    let mut matrix = Vec::with_capacity(output_dim * input_dim);
    for v in &accepted {
        matrix.extend_from_slice(v);
    }
    matrix.resize(output_dim * input_dim, 0.0);
    matrix
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shape ──

    #[test]
    fn test_output_dim_matches_constructor() {
        for (input_dim, output_dim) in [(96, 32), (64, 16), (32, 8), (128, 64)] {
            let shard = ShardEmbedding::new(input_dim, output_dim, 42);
            let input = vec![0.5f32; input_dim];
            let output = shard.project(&input);
            assert_eq!(output.len(), output_dim, "dim {input_dim}->{output_dim}");
        }
    }

    #[test]
    fn test_accessor_dims() {
        let shard = ShardEmbedding::new(96, 32, 7);
        assert_eq!(shard.input_dim(), 96);
        assert_eq!(shard.output_dim(), 32);
        assert_eq!(shard.seed(), 7);
    }

    // ── Edge cases: zero dims ──

    #[test]
    fn test_zero_output_dim_produces_empty() {
        let shard = ShardEmbedding::new(96, 0, 42);
        let output = shard.project(&[0.5f32; 96]);
        assert!(output.is_empty());
        assert_eq!(shard.output_dim(), 0);
    }

    #[test]
    fn test_zero_input_dim_produces_empty() {
        let shard = ShardEmbedding::new(0, 32, 42);
        // output_dim is clamped to min(0, 32) = 0
        assert_eq!(shard.output_dim(), 0);
        let output = shard.project(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_output_clamped_to_input_dim() {
        // Cannot have more orthogonal rows than input_dim.
        let shard = ShardEmbedding::new(8, 32, 42);
        assert_eq!(shard.output_dim(), 8);
        assert_eq!(shard.input_dim(), 8);
        let output = shard.project(&[0.5f32; 8]);
        assert_eq!(output.len(), 8);
    }

    // ── Determinism ──

    #[test]
    fn test_same_seed_produces_same_projection() {
        let a = ShardEmbedding::new(96, 32, 42);
        let b = ShardEmbedding::new(96, 32, 42);
        let input = vec![0.3f32; 96];
        assert_eq!(a.project(&input), b.project(&input));
        assert_eq!(a.matrix_hash(), b.matrix_hash());
    }

    #[test]
    fn test_different_seeds_produce_different_projections() {
        let a = ShardEmbedding::new(96, 32, 1);
        let b = ShardEmbedding::new(96, 32, 2);
        let input = vec![0.3f32; 96];
        assert_ne!(a.project(&input), b.project(&input));
        assert_ne!(a.matrix_hash(), b.matrix_hash());
    }

    #[test]
    fn test_repeated_project_calls_are_stable() {
        let shard = ShardEmbedding::new(64, 16, 99);
        let input = vec![0.7f32; 64];
        let mut prev = shard.project(&input);
        for _ in 0..50 {
            let cur = shard.project(&input);
            assert_eq!(cur, prev, "project must be deterministic across calls");
            prev = cur;
        }
    }

    // ── Matrix properties: orthonormal rows ──

    #[test]
    fn test_rows_are_unit_norm() {
        let shard = ShardEmbedding::new(96, 32, 42);
        for (i, row) in shard.matrix.chunks_exact(96).enumerate() {
            let norm_sq: f32 = row.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "row {i} norm² = {norm_sq}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_rows_are_mutually_orthogonal() {
        let shard = ShardEmbedding::new(96, 16, 42);
        let rows: Vec<&[f32]> = shard.matrix.chunks_exact(96).collect();
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                let dot: f32 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-4, "rows {i},{j} dot = {dot}, expected ≈ 0");
            }
        }
    }

    // ── Projection properties ──

    #[test]
    fn test_projection_never_amplifies_norm() {
        // Orthonormal rows ⇒ ‖Rx‖ ≤ ‖x‖.
        let shard = ShardEmbedding::new(96, 32, 42);
        let input = vec![0.5f32; 96];
        let output = shard.project(&input);
        let in_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let out_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            out_norm <= in_norm + 1e-4,
            "output norm {out_norm} must not exceed input norm {in_norm}"
        );
    }

    #[test]
    fn test_zero_input_produces_zero_output() {
        let shard = ShardEmbedding::new(96, 32, 42);
        let output = shard.project(&[0.0f32; 96]);
        for (i, &x) in output.iter().enumerate() {
            assert!(x.abs() < 1e-6, "zero input coord {i} = {x}, expected ≈ 0");
        }
    }

    #[test]
    fn test_projection_is_linear() {
        // R·(a + b) = R·a + R·b.
        let shard = ShardEmbedding::new(64, 16, 42);
        let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32) * -0.02).collect();
        let sum_input: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let ra = shard.project(&a);
        let rb = shard.project(&b);
        let rab = shard.project(&sum_input);
        for i in 0..16 {
            let combined = ra[i] + rb[i];
            let rab_i = rab[i];
            assert!(
                (rab_i - combined).abs() < 1e-3,
                "linearity violated at coord {i}: {rab_i} vs {combined}"
            );
        }
    }

    #[test]
    fn test_projection_preserves_canonical_basis_vector() {
        // If input is e_j (unit vector along axis j), output[i] = matrix[i][j].
        let shard = ShardEmbedding::new(16, 8, 42);
        let mut e3 = vec![0.0f32; 16];
        e3[3] = 1.0;
        let output = shard.project(&e3);
        for (i, &y) in output.iter().enumerate() {
            let expected = shard.matrix[i * 16 + 3];
            assert!(
                (y - expected).abs() < 1e-6,
                "e₃ projection coord {i} = {y}, expected matrix entry {expected}"
            );
        }
    }

    // ── Audit hash ──

    #[test]
    fn test_matrix_hash_deterministic() {
        let a = ShardEmbedding::new(96, 32, 42);
        let b = ShardEmbedding::new(96, 32, 42);
        assert_eq!(a.matrix_hash(), b.matrix_hash());
    }

    #[test]
    fn test_matrix_hash_differs_on_seed() {
        let a = ShardEmbedding::new(96, 32, 1);
        let b = ShardEmbedding::new(96, 32, 2);
        assert_ne!(a.matrix_hash(), b.matrix_hash());
    }

    #[test]
    fn test_matrix_hash_differs_on_dims() {
        let a = ShardEmbedding::new(96, 32, 42);
        let b = ShardEmbedding::new(64, 32, 42);
        assert_ne!(a.matrix_hash(), b.matrix_hash());
    }

    // ── Box-Muller unit ──

    #[test]
    fn test_box_muller_produces_finite_values() {
        for u1 in [0u32, 1, 42, u32::MAX / 2, u32::MAX] {
            for u2 in [0u32, 1, 42, u32::MAX / 2, u32::MAX] {
                let (z0, z1) = box_muller(u1, u2);
                assert!(z0.is_finite(), "z0 not finite for ({u1},{u2})");
                assert!(z1.is_finite(), "z1 not finite for ({u1},{u2})");
            }
        }
    }

    #[test]
    fn test_box_muller_distribution_is_spread() {
        // Over many samples, mean should be ≈ 0, variance ≈ 1.
        let mut stream = GaussianStream::new(12345);
        let n = 100_000usize;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for _ in 0..n {
            let z = stream.next() as f64;
            sum += z;
            sum_sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean = {mean}, expected ≈ 0");
        assert!((var - 1.0).abs() < 0.1, "variance = {var}, expected ≈ 1.0");
    }

    // ── Trait bounds ──

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ShardEmbedding>();
    }

    #[test]
    fn test_clone_is_equal() {
        let a = ShardEmbedding::new(96, 32, 42);
        let b = a.clone();
        let input = vec![0.3f32; 96];
        assert_eq!(a.project(&input), b.project(&input));
        assert_eq!(a.matrix_hash(), b.matrix_hash());
    }
}
