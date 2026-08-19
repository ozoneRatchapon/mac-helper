//! Phase 5 — Benchmark suite: per-domain TTFT, acceptance rate, correctness.
//!
//! Companion to [`phase5_arena`]. Where the Arena proves adaptive > static >
//! random, this suite measures three production-relevant metrics for each
//! Phase 5 domain driven by its real pruner plus a small directional draft
//! model (fluency only — the pruner owns correctness):
//!
//! - **TTFT** (time to first token): wall-clock latency of producing the first
//!   committed action/token. Measured as the duration of a single-step
//!   (`max_tokens == 1`) greedy run — the cost of one draft → prune → step
//!   iteration. A first-class latency metric for interactive generators.
//! - **Acceptance rate**: committed outputs / exploration attempts. From the
//!   result's `attempts` field (every candidate evaluation counts once). Greedy
//!   runs accept every step that finds a valid candidate; backtracking runs
//!   spend attempts on dead-ends too, so the rate reflects pruning efficiency.
//! - **Correctness**: the domain objective is met — `goal` for the game
//!   domains (Bomber, Quest), `is_complete` for JSON.
//!
//! The full table prints under `--nocapture`. The assertions pin CORRECTNESS
//! (the machine-independent invariant); TTFT and acceptance rate are reported,
//! not asserted to a hard threshold, since they are hardware-dependent.
//!
//! Run with:
//!   cargo test --test phase5_benchmark -- --nocapture

use std::time::Instant;

use ns_engine::pruners::{
    encode_quest_action, BomberActionPruner, BomberConfig, BomberState, JsonSchemaPruner,
    QuestActionPruner, QuestConfig, QuestState, ACTION_MOVE_E, ACTION_MOVE_N, ACTION_MOVE_S,
    ACTION_MOVE_W, QUEST_ACTION_ACCEPT, QUEST_ACTION_COMPLETE, QUEST_ACTION_FAIL,
};
use ns_engine::traits::{ConstraintPruner, DraftModel};
use ns_engine::{
    speculative_decode, speculative_generate, DecodeConfig, DecodeResult, GenerateResult, Logits,
    TokenId,
};

/// One measured domain row of the benchmark table.
struct Row {
    domain: &'static str,
    /// Time to first token, in microseconds.
    ttft_us: u128,
    /// Acceptance rate: committed outputs / attempts.
    acceptance: f64,
    /// Whether the domain objective was met.
    correct: bool,
    /// Committed output count (actions or tokens).
    output_len: usize,
}

impl Row {
    /// Render the row as a fixed-width table line.
    fn render(&self) -> String {
        format!(
            "{domain:<14} {ttft:>8} µs   {acc:>6.2}   {correct:<3}   {out:>4}",
            domain = self.domain,
            ttft = self.ttft_us,
            acc = self.acceptance,
            correct = if self.correct { "yes" } else { "no" },
            out = self.output_len,
        )
    }
}

/// Print the benchmark table header + rows under `--nocapture`.
fn print_table(rows: &[Row]) {
    eprintln!();
    eprintln!("Phase 5 benchmark — directional draft + real pruner");
    eprintln!();
    eprintln!(
        "{domain:<14} {ttft:>11}   {acc:>6}   {correct:<3}   {out:>4}",
        domain = "domain",
        ttft = "TTFT",
        acc = "accept",
        correct = "ok",
        out = "out",
    );
    eprintln!(
        "{:<14} {:>11}   {:>6}   {:<3}   {:>4}",
        "------------", "-----------", "------", "---", "----"
    );
    for row in rows {
        eprintln!("{}", row.render());
    }
    eprintln!();
}

/// Local top-k by logit (descending), ties broken by ascending token index.
///
/// Mirrors `decode::top_k_indices` semantics, which is `pub(crate)` and thus
/// not reachable from an integration test. Used only for the acceptance
/// measurement, not to drive the loop.
fn top_k(logits: &[f32], k: usize) -> Vec<TokenId> {
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
    // Stable sort by descending logit; equal logits keep ascending index order.
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Measure the pruner's ACCEPTANCE RATE along a committed solution trajectory:
/// at each depth, the fraction of the draft's top-k proposals the pruner accepts.
///
/// This is the speculative-decoding acceptance rate — of the candidates the
/// draft proposed, how many survived pruning. A fresh pruner instance is used
/// because the solve run consumed the mutable borrow.
///
/// Greedy decode does not populate `attempts` (only backtracking does), so this
/// direct measurement is the honest way to report acceptance for greedy runs.
fn acceptance_rate(
    solution: &[TokenId],
    draft: &dyn DraftModel,
    pruner: &mut dyn ConstraintPrunerLike,
    top_k_count: usize,
) -> f64 {
    if solution.is_empty() {
        return 0.0;
    }
    let mut accepted_total = 0usize;
    let mut proposed_total = 0usize;
    for depth in 0..solution.len() {
        let prefix = &solution[..depth];
        let logits = draft.log_probs(prefix);
        let candidates = top_k(&logits, top_k_count);
        if candidates.is_empty() {
            continue;
        }
        let mut mask = vec![false; candidates.len()];
        pruner.batch_is_valid(depth, &candidates, prefix, &mut mask);
        accepted_total += mask.iter().filter(|&&v| v).count();
        proposed_total += candidates.len();
    }
    match proposed_total {
        0 => 0.0,
        proposed => accepted_total as f64 / proposed as f64,
    }
}

/// Helper trait alias so `acceptance_rate` accepts either a `ConstraintPruner`
/// impl (game pruners) — they all expose `batch_is_valid`.
trait ConstraintPrunerLike {
    fn batch_is_valid(
        &mut self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    );
}

impl<P: ConstraintPruner> ConstraintPrunerLike for P {
    fn batch_is_valid(
        &mut self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        ConstraintPruner::batch_is_valid(self, depth, candidates, parent_tokens, results);
    }
}

/// Measure TTFT for a generate domain: time a single-step greedy run.
///
/// Greedy with `max_tokens == 1` commits exactly one action (the first valid
/// candidate) then stops, so its wall-clock duration is the cost of producing
/// the first output — the TTFT proxy.
fn ttft_generate<F>(run_one: F) -> u128
where
    F: FnOnce() -> GenerateResult,
{
    let start = Instant::now();
    let result = run_one();
    let elapsed = start.elapsed().as_micros();
    assert!(
        !result.actions.is_empty(),
        "TTFT run produced no action — first-step pruner rejected every candidate"
    );
    elapsed
}

/// Measure TTFT for a decode domain: time a single-step greedy run.
fn ttft_decode<F>(run_one: F) -> u128
where
    F: FnOnce() -> DecodeResult,
{
    let start = Instant::now();
    let result = run_one();
    let elapsed = start.elapsed().as_micros();
    assert!(
        !result.tokens.is_empty(),
        "TTFT run produced no token — first-step pruner rejected every candidate"
    );
    elapsed
}

// ── Directional draft models (fluency only; pruner owns correctness) ──

/// `(x, y)` from a bomber cell index given `width`.
fn xy(index: usize, width: usize) -> (usize, usize) {
    (index % width, index / width)
}

/// Manhattan distance between two bomber cell indices given `width`.
fn manhattan(a: usize, b: usize, width: usize) -> usize {
    let (ax, ay) = xy(a, width);
    let (bx, by) = xy(b, width);
    (ax as isize - bx as isize).unsigned_abs() + (ay as isize - by as isize).unsigned_abs()
}

/// `(dx, dy)` for a bomber move action.
fn move_delta(action: TokenId) -> (isize, isize) {
    match action {
        ACTION_MOVE_N => (0, -1),
        ACTION_MOVE_S => (0, 1),
        ACTION_MOVE_E => (1, 0),
        ACTION_MOVE_W => (-1, 0),
        _ => (0, 0),
    }
}

/// Bomber draft: ranks moves by exit-distance reduction (fluency). Legality is
/// enforced by [`BomberActionPruner`].
struct TowardExitDraft {
    width: usize,
    exit: usize,
    vocab: usize,
}

impl TowardExitDraft {
    fn new(cfg: &BomberConfig) -> Self {
        Self {
            width: cfg.width,
            exit: cfg.exit_index().unwrap_or(0),
            vocab: 6,
        }
    }

    fn player_index(&self, actions: &[TokenId]) -> usize {
        let (mut x, mut y) = (0usize, 0usize);
        for &a in actions {
            match a {
                ACTION_MOVE_N => y = y.saturating_sub(1),
                ACTION_MOVE_S => y += 1,
                ACTION_MOVE_E => x += 1,
                ACTION_MOVE_W => x = x.saturating_sub(1),
                _ => {}
            }
        }
        y * self.width + x
    }
}

impl DraftModel for TowardExitDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, context: &[TokenId]) -> Logits {
        let here = self.player_index(context);
        let (hx, hy) = xy(here, self.width);
        let here_d = manhattan(here, self.exit, self.width);
        let mut logits = vec![0.0_f32; self.vocab];
        for a in 0..self.vocab as TokenId {
            let score = match a {
                ACTION_MOVE_N | ACTION_MOVE_S | ACTION_MOVE_E | ACTION_MOVE_W => {
                    let (dx, dy) = move_delta(a);
                    let nx = (hx as isize) + dx;
                    let ny = (hy as isize) + dy;
                    if nx < 0 || ny < 0 {
                        -10.0
                    } else {
                        let next = (ny as usize) * self.width + (nx as usize);
                        let next_d = manhattan(next, self.exit, self.width);
                        // Closer to exit → higher logit. The tiny `-a as f32 *
                        // 1e-3` tiebreaker makes the draft a STRICT total
                        // order, so `shuffle_tied_groups` has no ties to
                        // randomize — the path is deterministic under both
                        // greedy and backtracking.
                        (here_d as isize - next_d as isize) as f32 - (a as f32) * 1e-3
                    }
                }
                _ => -10.0,
            };
            logits[a as usize] = score;
        }
        logits
    }
}

/// Quest draft: ranks ACCEPT then COMPLETE on low-index quests highest; FAIL
/// lowest. Legality (prerequisites, lifecycle) is enforced by
/// [`QuestActionPruner`].
struct TowardGoalDraft {
    num_quests: usize,
    vocab: usize,
}

impl TowardGoalDraft {
    fn new(cfg: &QuestConfig) -> Self {
        Self {
            num_quests: cfg.num_quests,
            vocab: cfg.num_quests * 3,
        }
    }
}

impl DraftModel for TowardGoalDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        let mut logits = vec![0.0_f32; self.vocab];
        for q in 0..self.num_quests {
            let accept = encode_quest_action(q, QUEST_ACTION_ACCEPT) as usize;
            let complete = encode_quest_action(q, QUEST_ACTION_COMPLETE) as usize;
            let fail = encode_quest_action(q, QUEST_ACTION_FAIL) as usize;
            let base = (self.num_quests - q) as f32;
            logits[accept] = base + 1.0;
            logits[complete] = base + 0.5;
            logits[fail] = -10.0;
        }
        logits
    }
}

/// JSON draft: deprioritizes whitespace and ranks `"` highest so greedy
/// produces the minimal complete document `""`. Validity is enforced by
/// [`JsonSchemaPruner`].
struct StructuralJsonDraft {
    vocab: usize,
}

impl StructuralJsonDraft {
    fn new(vocab: usize) -> Self {
        Self { vocab }
    }
}

impl DraftModel for StructuralJsonDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        let mut logits = vec![0.0_f32; self.vocab];
        for (t, slot) in logits.iter_mut().enumerate() {
            if t < 128 && matches!(t as u8, b' ' | b'\t' | b'\n' | b'\r') {
                *slot = -1.0;
            }
        }
        if self.vocab > b'"' as usize {
            logits[b'"' as usize] = 1.0;
        }
        logits
    }
}

// ── Per-domain benchmarks ──────────────────────────────────────

fn bench_bomber() -> Row {
    let cfg = BomberConfig::default();

    // TTFT: single-step greedy.
    let ttft = ttft_generate(|| {
        let initial = BomberState::initial(cfg.clone());
        let draft = TowardExitDraft::new(&cfg);
        let mut pruner = BomberActionPruner::new(cfg.clone());
        let one_step = DecodeConfig {
            backtrack: false,
            max_tokens: 1,
            top_k: 6,
            seed: 42,
            max_attempts: 100,
        };
        speculative_generate(&initial, &draft, &mut pruner, &one_step)
    });

    // Full solve (greedy) for correctness + output count. Greedy is
    // deterministic under a strict-total-order draft and reaches the goal
    // without the random-walk behavior backtracking exhibits on tie-free grids.
    let initial = BomberState::initial(cfg.clone());
    let draft = TowardExitDraft::new(&cfg);
    let mut pruner = BomberActionPruner::new(cfg);
    let full = DecodeConfig {
        backtrack: false,
        max_tokens: 20,
        top_k: 6,
        seed: 42,
        max_attempts: 1_000,
    };
    let result = speculative_generate(&initial, &draft, &mut pruner, &full);

    // Acceptance: fraction of the draft's top-k proposals the pruner accepts,
    // measured along the solution trajectory with a fresh pruner.
    let mut measurer = BomberActionPruner::new(BomberConfig::default());
    let acceptance = acceptance_rate(&result.actions, &draft, &mut measurer, 6);

    Row {
        domain: "bomber",
        ttft_us: ttft,
        acceptance,
        correct: result.goal,
        output_len: result.actions.len(),
    }
}

fn bench_quest() -> Row {
    let cfg = QuestConfig::default();

    let ttft = ttft_generate(|| {
        let initial = QuestState::initial(cfg.clone());
        let draft = TowardGoalDraft::new(&cfg);
        let mut pruner = QuestActionPruner::new(cfg.clone());
        let one_step = DecodeConfig {
            backtrack: false,
            max_tokens: 1,
            top_k: draft.vocab_size(),
            seed: 42,
            max_attempts: 100,
        };
        speculative_generate(&initial, &draft, &mut pruner, &one_step)
    });

    let initial = QuestState::initial(cfg.clone());
    let draft = TowardGoalDraft::new(&cfg);
    let mut pruner = QuestActionPruner::new(cfg);
    let full = DecodeConfig {
        backtrack: false,
        max_tokens: 20,
        top_k: draft.vocab_size(),
        seed: 42,
        max_attempts: 1_000,
    };
    let vocab = draft.vocab_size();
    let result = speculative_generate(&initial, &draft, &mut pruner, &full);

    let mut measurer = QuestActionPruner::new(QuestConfig::default());
    let acceptance = acceptance_rate(&result.actions, &draft, &mut measurer, vocab);

    Row {
        domain: "quest",
        ttft_us: ttft,
        acceptance,
        correct: result.goal,
        output_len: result.actions.len(),
    }
}

fn bench_json_schema() -> Row {
    // TTFT: single-step greedy decode.
    let ttft = ttft_decode(|| {
        let mut pruner = JsonSchemaPruner::from_vocab_size(128);
        let draft = StructuralJsonDraft::new(128);
        let one_step = DecodeConfig {
            backtrack: false,
            max_tokens: 1,
            top_k: 128,
            seed: 42,
            max_attempts: 100,
        };
        speculative_decode(&draft, &mut pruner, &one_step)
    });

    // Full decode (greedy) for correctness + output count.
    let mut pruner = JsonSchemaPruner::from_vocab_size(128);
    let draft = StructuralJsonDraft::new(128);
    let full = DecodeConfig {
        max_tokens: 20,
        top_k: 128,
        seed: 42,
        backtrack: false,
        max_attempts: 1_000,
    };
    let result = speculative_decode(&draft, &mut pruner, &full);
    let correct = pruner.is_complete(&result.tokens);

    let mut measurer = JsonSchemaPruner::from_vocab_size(128);
    let acceptance = acceptance_rate(&result.tokens, &draft, &mut measurer, 128);

    Row {
        domain: "json-schema",
        ttft_us: ttft,
        acceptance,
        correct,
        output_len: result.tokens.len(),
    }
}

// ── Harness ────────────────────────────────────────────────────

/// Run every Phase 5 domain benchmark, print the table, and assert each domain
/// meets its correctness objective. TTFT and acceptance rate are reported only.
#[test]
fn phase5_benchmark_suite() {
    let rows = vec![bench_bomber(), bench_quest(), bench_json_schema()];

    print_table(&rows);

    // Correctness is the machine-independent invariant — assert it per domain.
    for row in &rows {
        assert!(
            row.correct,
            "domain {domain} failed its correctness objective",
            domain = row.domain
        );
    }

    // Every domain must have produced real output (non-zero committed count).
    for row in &rows {
        assert!(
            row.output_len > 0,
            "domain {domain} produced zero output",
            domain = row.domain
        );
    }

    // Acceptance rate is always in [0, 1] by construction; sanity-check bounds.
    for row in &rows {
        assert!(
            (0.0..=1.0).contains(&row.acceptance),
            "domain {domain} acceptance {acc} out of [0,1]",
            domain = row.domain,
            acc = row.acceptance
        );
    }
}
