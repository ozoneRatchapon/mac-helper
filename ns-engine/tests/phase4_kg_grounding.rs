//! Phase 4 end-to-end proof: KG-grounded generation and NIAH retrieval.
//!
//! Two stories from `.plans/006` Phase 4:
//!
//! - **Checkbox 6 (KG-grounded generation)**: facts stored as triples are
//!   recalled verbatim through a KG-augmented decode loop. `DomainLatent`
//!   biases the next-token logits toward KG objects, and a chain of facts
//!   is reproduced exactly — the modelless thesis in action (uniform draft
//!   with zero knowledge + symbolic KG facts → structured output).
//!
//! - **Checkbox 7 (NIAH retrieval)**: a specific fact buried among many is
//!   retrieved via `KgStore::lookup` with 100% accuracy — no hallucination,
//!   no false positives, exact discrete recall. This is the "needle in a
//!   haystack" guarantee: KG lookup is precise, not approximate.
//!
//! Run with:
//!   cargo test --test phase4_kg_grounding -- --nocapture

use ns_engine::draft::UniformDraftModel;
use ns_engine::kg::{DomainLatent, InMemoryKgStore, ShardEmbedding};
use ns_engine::traits::{DraftModel, KgStore};
use ns_engine::types::{KgTriple, TokenId};

// ── Helpers ────────────────────────────────────────────────────

/// Find the index of the highest-value entry in `logits`.
///
/// Ties broken by lowest index (first occurrence wins). Returns `None` for
/// an empty slice. Mirrors the decode loop's candidate selection in its
/// simplest single-best form.
fn argmax(logits: &[f32]) -> Option<TokenId> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        match best {
            Some((_, bv)) if v <= bv => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i as TokenId)
}

/// Mini decode loop with KG grounding applied at every step.
///
/// Starts from `seed`, runs `max_tokens` steps. At each step:
/// 1. Draft model produces logits over the vocabulary.
/// 2. `DomainLatent::apply_bias` boosts logits for KG-grounded objects.
/// 3. `argmax` selects the highest-logit token.
///
/// With a uniform draft model (all logits equal), any positive bias makes
/// the KG object win — demonstrating that symbolic facts drive generation
/// when the "model" contributes nothing.
fn kg_grounded_decode(
    draft: &dyn DraftModel,
    domain_latent: &DomainLatent<InMemoryKgStore>,
    seed: &[TokenId],
    max_tokens: usize,
    boost: f32,
) -> Vec<TokenId> {
    let mut tokens: Vec<TokenId> = seed.to_vec();
    for _ in 0..max_tokens {
        let mut logits = draft.log_probs(&tokens);
        domain_latent.apply_bias(&mut logits, &tokens, boost);
        match argmax(&logits) {
            Some(token) => tokens.push(token),
            None => break,
        }
    }
    tokens
}

/// Build an `InMemoryKgStore` from a flat list of `(subject, predicate, object)`
/// tuples. Deduplicates on insert.
fn build_kg_with_facts(facts: &[(TokenId, TokenId, TokenId)]) -> InMemoryKgStore {
    let mut kg = InMemoryKgStore::new(32);
    for &(s, p, o) in facts {
        kg.insert(KgTriple::new(s, p, o));
    }
    kg
}

// ── Constants ──────────────────────────────────────────────────

/// Predicate token used for chain/next-fact relations.
const NEXT: TokenId = 100;

// ════════════════════════════════════════════════════════════════
// Checkbox 6: KG-grounded generation — facts recalled verbatim
// ════════════════════════════════════════════════════════════════

#[test]
fn kg_grounded_generation_recalls_chain_verbatim() {
    // Build a chain: 1 → 2 → 3 → 4 → 5 via predicate NEXT.
    // Each subject has exactly one "next" object.
    let kg = build_kg_with_facts(&[(1, NEXT, 2), (2, NEXT, 3), (3, NEXT, 4), (4, NEXT, 5)]);

    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, NEXT);
    let draft = UniformDraftModel::new(10); // vocab 0..9; chain uses 1..5

    // Seed with entity 1; generate 4 tokens (one per chain link).
    let output = kg_grounded_decode(&draft, &dl, &[1], 4, 10.0);

    assert_eq!(
        output,
        vec![1, 2, 3, 4, 5],
        "chain must be recalled verbatim, got {output:?}"
    );
}

#[test]
fn kg_grounded_generation_single_fact_recalled() {
    // Single fact: (alice=1, knows=10, bob=2).
    let kg = build_kg_with_facts(&[(1, 10, 2)]);

    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, 10);
    let draft = UniformDraftModel::new(10);

    let output = kg_grounded_decode(&draft, &dl, &[1], 1, 5.0);

    assert_eq!(output, vec![1, 2], "single fact must be recalled verbatim");
}

#[test]
fn kg_grounded_generation_overrides_uniform_draft() {
    // Without KG bias, uniform draft picks token 0 (first of equal logits).
    // With KG bias, the grounded object must win.
    let kg = build_kg_with_facts(&[(1, 10, 7)]);

    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, 10);
    let draft = UniformDraftModel::new(10);

    // No bias → token 0 (uniform, lowest index wins on ties).
    let logits = draft.log_probs(&[1]);
    match argmax(&logits) {
        Some(t) => assert_eq!(
            t, 0,
            "uniform draft without bias should pick token 0, got {t}"
        ),
        None => panic!("uniform draft produced empty logits"),
    }

    // With bias → token 7 (KG grounded object).
    let mut biased_logits = draft.log_probs(&[1]);
    dl.apply_bias(&mut biased_logits, &[1], 10.0);
    match argmax(&biased_logits) {
        Some(t) => assert_eq!(
            t, 7,
            "KG bias should override uniform draft to pick token 7, got {t}"
        ),
        None => panic!("biased logits empty"),
    }
}

#[test]
fn kg_grounded_generation_respects_predicate() {
    // Same subject, two predicates → different recalled objects.
    let facts = [(1, 10, 2), (1, 20, 3)];
    let draft = UniformDraftModel::new(10);

    // Predicate 10 → recalls token 2.
    let kg_10 = build_kg_with_facts(&facts);
    let dl_10 = DomainLatent::new(kg_10, ShardEmbedding::new(32, 16, 42), 0, 10);
    let out_10 = kg_grounded_decode(&draft, &dl_10, &[1], 1, 5.0);
    assert_eq!(out_10, vec![1, 2], "predicate 10 must recall token 2");

    // Predicate 20 → recalls token 3.
    let kg_20 = build_kg_with_facts(&facts);
    let dl_20 = DomainLatent::new(kg_20, ShardEmbedding::new(32, 16, 42), 0, 20);
    let out_20 = kg_grounded_decode(&draft, &dl_20, &[1], 1, 5.0);
    assert_eq!(out_20, vec![1, 3], "predicate 20 must recall token 3");
}

#[test]
fn kg_grounded_generation_multi_object_distributes_boost() {
    // One subject, one predicate, two objects (alice knows [bob, carol]).
    // Boost makes both objects rank above non-grounded tokens.
    let kg = build_kg_with_facts(&[(1, 10, 2), (1, 10, 3)]);

    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, 10);

    let mut logits = vec![0.0f32; 10];
    dl.apply_bias(&mut logits, &[1], 10.0);

    // Both grounded tokens must be strictly above the uniform baseline.
    assert!(logits[2] > 0.0, "bob (token 2) must receive positive bias");
    assert!(
        logits[3] > 0.0,
        "carol (token 3) must receive positive bias"
    );
    // Non-grounded tokens must be untouched.
    assert!(logits[0] == 0.0, "token 0 must remain at baseline");
    assert!(logits[5] == 0.0, "token 5 must remain at baseline");
    // Total bias equals the boost (weights sum to 1.0).
    let total_bias: f32 = logits.iter().sum();
    assert!(
        (total_bias - 10.0).abs() < 1e-4,
        "total bias must equal boost, got {total_bias}"
    );
}

// ════════════════════════════════════════════════════════════════
// Checkbox 7: NIAH retrieval — 100% accuracy via KG lookup
// ════════════════════════════════════════════════════════════════

#[test]
fn niah_retrieval_finds_needle_in_haystack() {
    // Haystack: 100 decoy facts with distinct subjects.
    // Needle: (150, 205, 9999) — a unique, recognizable fact.
    let mut facts: Vec<(TokenId, TokenId, TokenId)> = Vec::new();
    for i in 0..100u32 {
        facts.push((i + 200, 200, i + 300)); // decoys
    }
    facts.push((150, 205, 9999)); // the needle

    let kg = build_kg_with_facts(&facts);

    // Direct lookup — exact retrieval, no approximation.
    let result = kg.lookup(150, 205);
    assert_eq!(
        result,
        vec![9999],
        "needle must be retrieved exactly from haystack"
    );

    // DomainLatent grounding — needle must be the sole candidate.
    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, 205);
    let groundings = dl.ground(&[150]);
    assert_eq!(
        groundings.len(),
        1,
        "exactly one object for the needle subject"
    );
    match groundings.first() {
        Some(&(token, _)) => assert_eq!(token, 9999, "needle object must be retrieved"),
        None => panic!("needle not found via DomainLatent"),
    }
}

#[test]
fn niah_retrieval_100_percent_accuracy_1000_facts() {
    // 1000 facts, each with a distinct subject → exact object.
    // Every single lookup must return the correct object.
    const HAS_VALUE: TokenId = 100;
    let mut kg = InMemoryKgStore::new(32);
    for i in 0..1000u32 {
        kg.insert(KgTriple::new(i, HAS_VALUE, i + 10000));
    }

    for i in 0..1000u32 {
        let expected = i + 10000;
        let result = kg.lookup(i, HAS_VALUE);
        assert_eq!(
            result,
            vec![expected],
            "fact {i}: expected object {expected}, recall must be exact"
        );
    }
    // If we reach here, all 1000 lookups passed → 100% accuracy.
}

#[test]
fn niah_retrieval_100_percent_via_domain_latent() {
    // Same 1000 facts, but retrieval goes through DomainLatent::ground.
    // Each entity has exactly one fact → ground() returns one object with
    // weight 1.0. Every entity must be retrieved correctly.
    const HAS_VALUE: TokenId = 100;
    let mut kg = InMemoryKgStore::new(32);
    for i in 0..1000u32 {
        kg.insert(KgTriple::new(i, HAS_VALUE, i + 10000));
    }

    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, HAS_VALUE);

    for i in 0..1000u32 {
        let expected = i + 10000;
        let groundings = dl.ground(&[i]);
        assert_eq!(
            groundings.len(),
            1,
            "entity {i} must have exactly one grounding"
        );
        match groundings.first() {
            Some(&(token, weight)) => {
                assert_eq!(token, expected, "entity {i}: wrong object retrieved");
                assert!(
                    (weight - 1.0).abs() < 1e-5,
                    "entity {i}: single-object weight must be 1.0, got {weight}"
                );
            }
            None => panic!("entity {i}: no grounding returned"),
        }
    }
}

#[test]
fn niah_retrieval_no_false_positives() {
    // Unknown subjects and predicates must return empty — no hallucination.
    let kg = build_kg_with_facts(&[(1, 10, 2)]);

    // Subject not in KG → empty.
    let unknown_subject = kg.lookup(999, 10);
    assert!(
        unknown_subject.is_empty(),
        "unknown subject must return empty, got {unknown_subject:?}"
    );

    // Predicate not in KG → empty.
    let unknown_predicate = kg.lookup(1, 999);
    assert!(
        unknown_predicate.is_empty(),
        "unknown predicate must return empty, got {unknown_predicate:?}"
    );

    // Both unknown → empty.
    let both_unknown = kg.lookup(888, 999);
    assert!(
        both_unknown.is_empty(),
        "unknown subject+predicate must return empty, got {both_unknown:?}"
    );
}

#[test]
fn niah_retrieval_predicate_isolation() {
    // Entity 1 has facts under two predicates. Each lookup must return only
    // the objects for that predicate — no cross-contamination.
    let kg = build_kg_with_facts(&[
        (1, 10, 2), // predicate 10
        (1, 10, 3),
        (1, 20, 4), // predicate 20
        (1, 20, 5),
    ]);

    // Predicate 10 → exactly [2, 3].
    let mut knows = kg.lookup(1, 10);
    knows.sort();
    assert_eq!(
        knows,
        vec![2, 3],
        "predicate 10 must isolate to objects [2, 3], got {knows:?}"
    );

    // Predicate 20 → exactly [4, 5].
    let mut works_with = kg.lookup(1, 20);
    works_with.sort();
    assert_eq!(
        works_with,
        vec![4, 5],
        "predicate 20 must isolate to objects [4, 5], got {works_with:?}"
    );
}

#[test]
fn niah_retrieval_needle_among_many_predicates() {
    // Entity 1 has 50 decoy facts (haystack) under various predicates,
    // plus one needle fact under a unique predicate.
    const NEEDLE_PRED: TokenId = 999;
    const NEEDLE_OBJ: TokenId = 8888;

    let mut facts: Vec<(TokenId, TokenId, TokenId)> = Vec::new();
    for p in 200..250u32 {
        facts.push((1, p, p + 1000)); // 50 decoy facts
    }
    facts.push((1, NEEDLE_PRED, NEEDLE_OBJ)); // the needle

    let kg = build_kg_with_facts(&facts);

    // The needle is isolated by its unique predicate.
    let result = kg.lookup(1, NEEDLE_PRED);
    assert_eq!(
        result,
        vec![NEEDLE_OBJ],
        "needle must be isolated by predicate from 50 decoys"
    );

    // Via DomainLatent with the needle predicate.
    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, NEEDLE_PRED);
    let groundings = dl.ground(&[1]);
    assert_eq!(groundings.len(), 1, "exactly one needle grounding");
    match groundings.first() {
        Some(&(token, _)) => assert_eq!(token, NEEDLE_OBJ, "needle object must match"),
        None => panic!("needle not found among 50 decoy predicates"),
    }
}

#[test]
fn niah_retrieval_via_decode_recall_needle() {
    // End-to-end: the decode loop recalls the needle fact verbatim.
    // Entity 1 has the needle (1, SECRET, 42). Decode starting from [1]
    // with predicate SECRET must emit token 42 as the next token.
    const SECRET: TokenId = 200;
    const NEEDLE_OBJ: TokenId = 42;

    // Add decoys so the KG is non-trivial.
    let mut facts: Vec<(TokenId, TokenId, TokenId)> = Vec::new();
    for i in 0..20u32 {
        facts.push((i + 50, 100, i + 200)); // unrelated decoys
    }
    facts.push((1, SECRET, NEEDLE_OBJ)); // the needle

    let kg = build_kg_with_facts(&facts);
    let shard = ShardEmbedding::new(32, 16, 42);
    let dl = DomainLatent::new(kg, shard, 0, SECRET);
    let draft = UniformDraftModel::new(64); // vocab large enough for token 42

    let output = kg_grounded_decode(&draft, &dl, &[1], 1, 10.0);

    assert_eq!(
        output,
        vec![1, NEEDLE_OBJ],
        "decode must recall the needle fact verbatim, got {output:?}"
    );
}
