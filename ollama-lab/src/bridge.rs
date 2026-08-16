//! Phase 3 bridge: local Ollama model as an ns-engine `DraftModel`.
//!
//! Ollama's HTTP API exposes no per-token logits, so the LLM can't sit in the
//! decode loop the classic way. Instead it fills the DRAFT seat as a ranker:
//! `DraftModel::log_probs` only needs "any ordering signal", so we ask the LLM
//! to rank candidate next actions and convert rank → pseudo-logits. Legality
//! stays with `QuestActionPruner`, dynamics with `QuestState` — the LLM is
//! pure fluency, exactly the ownership boundary ns-engine draws.
//!
//! Run: `cargo run -- bridge [model]` (default gemma4:26b).

use ns_engine::pruners::{
    decode_quest_action, QuestActionPruner, QuestConfig, QuestState, ACTIONS_PER_QUEST,
    QUEST_ACTION_COMPLETE,
};
use ns_engine::draft::UniformDraftModel;
use ns_engine::{speculative_generate, DecodeConfig, DraftModel, Logits, TokenId};
use std::sync::Mutex;

const ACTION_NAMES: [&str; 3] = ["accept", "complete", "fail"];

fn action_label(token: TokenId) -> String {
    let (quest, kind) = decode_quest_action(token);
    format!("{}_quest_{}", ACTION_NAMES[kind as usize], quest)
}

#[derive(Default)]
struct BridgeStats {
    llm_calls: u64,
    parse_failures: u64,
}

/// LLM-backed draft model: ranks the action vocabulary via a local Ollama
/// model; unranked actions get a low floor logit.
struct OllamaDraftModel {
    model: String,
    domain_prompt: String,
    vocab: usize,
    client: reqwest::blocking::Client,
    stats: Mutex<BridgeStats>,
}

impl OllamaDraftModel {
    fn new(model: &str, config: &QuestConfig) -> Self {
        let mut lines = vec![format!(
            "You plan actions in a quest game with {} quests (ids 0..{}).",
            config.num_quests, config.num_quests
        )];
        for (q, prereqs) in config.prerequisites.iter().enumerate() {
            lines.push(format!("Quest {q} requires completed prerequisites: {prereqs:?}."));
        }
        lines.push(format!(
            "Goal: complete quests {:?}. Rules: a quest must be accepted before it can be \
             completed or failed; prerequisites must be completed before accepting; \
             failing a goal quest loses the game.",
            config.goal_quests
        ));
        lines.push(format!(
            "Actions are numbered: token = quest_id*{ACTIONS_PER_QUEST} + kind, where kind \
             0=accept 1=complete 2=fail. Example: token {} = complete quest 1.",
            encode(1, QUEST_ACTION_COMPLETE)
        ));
        Self {
            model: model.to_string(),
            domain_prompt: lines.join("\n"),
            vocab: config.num_quests * ACTIONS_PER_QUEST,
            client: reqwest::blocking::Client::new(),
            stats: Mutex::new(BridgeStats::default()),
        }
    }

    fn ask_ranking(&self, context: &[TokenId]) -> Option<Vec<TokenId>> {
        let trajectory: Vec<String> = context.iter().map(|&t| action_label(t)).collect();
        let prompt = format!(
            "{}\n\nActions taken so far, in order: {:?}\n\nRank ALL action tokens 0..{} from \
             most promising to least promising as the NEXT action toward the goal. Output ONLY \
             a JSON object: {{\"ranking\": [token, token, ...]}} containing each token exactly once.",
            self.domain_prompt, trajectory, self.vocab
        );
        let body = serde_json::json!({
            "model": self.model, "prompt": prompt, "stream": false,
            "think": false, "format": "json",
            "options": {"temperature": 0.0}
        });
        let resp: serde_json::Value = self
            .client
            .post("http://127.0.0.1:11434/api/generate")
            .json(&body)
            .send()
            .ok()?
            .json()
            .ok()?;
        let parsed: serde_json::Value = serde_json::from_str(resp["response"].as_str()?).ok()?;
        let ranking: Vec<TokenId> = parsed["ranking"]
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|t| t as TokenId))
            .filter(|&t| (t as usize) < self.vocab)
            .collect();
        (!ranking.is_empty()).then_some(ranking)
    }
}

const fn encode(quest: usize, kind: TokenId) -> TokenId {
    quest as TokenId * ACTIONS_PER_QUEST as TokenId + kind
}

impl DraftModel for OllamaDraftModel {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, context: &[TokenId]) -> Logits {
        let mut stats = self.stats.lock().unwrap();
        stats.llm_calls += 1;
        drop(stats);

        // Floor logit for unranked/unparseable: everything ties, which
        // degrades gracefully to uniform-draft behavior.
        let mut logits = vec![0.0f32; self.vocab];
        match self.ask_ranking(context) {
            Some(ranking) => {
                let n = ranking.len() as f32;
                for (rank, &token) in ranking.iter().enumerate() {
                    logits[token as usize] = n - rank as f32; // best rank → highest logit
                }
            }
            None => {
                self.stats.lock().unwrap().parse_failures += 1;
            }
        }
        logits
    }
}

fn quest_domain() -> QuestConfig {
    // Diamond prerequisite graph over 5 quests: 0 → {1,2} → 3 → 4.
    QuestConfig {
        num_quests: 5,
        prerequisites: vec![vec![], vec![0], vec![0], vec![1, 2], vec![3]],
        goal_quests: vec![4],
    }
}

/// 12-quest decoy domain: the goal chain is 0 → 4 → 8 → 11; the other eight
/// quests have no prerequisites and are pure decoys — always legal, never
/// useful. Uniform DFS burns attempts exploring them; a strategic ranker
/// should walk the chain directly.
fn decoy_domain() -> QuestConfig {
    let mut prerequisites = vec![vec![]; 12];
    prerequisites[4] = vec![0];
    prerequisites[8] = vec![4];
    prerequisites[11] = vec![8];
    QuestConfig {
        num_quests: 12,
        prerequisites,
        goal_quests: vec![11],
    }
}

pub fn run(model: &str, big: bool) {
    let config = if big { decoy_domain() } else { quest_domain() };
    let vocab = config.num_quests * ACTIONS_PER_QUEST;
    let decode_cfg = DecodeConfig {
        max_tokens: 40,
        top_k: vocab,
        seed: 42,
        backtrack: true,
        max_attempts: 10_000,
    };

    if big {
        println!("domain: 12-quest decoy field, goal chain 0 -> 4 -> 8 -> 11");
    } else {
        println!("domain: 5-quest diamond (0 -> 1,2 -> 3 -> 4), goal = complete quest 4");
    }

    // Baseline: uniform draft (pure symbolic search).
    let uniform = UniformDraftModel::new(vocab);
    let mut pruner = QuestActionPruner::new(config.clone());
    let initial = QuestState::initial(config.clone());
    let base = speculative_generate(&initial, &uniform, &mut pruner, &decode_cfg);
    println!(
        "uniform draft:  goal={} steps={} attempts={} reward={}",
        base.goal, base.actions.len(), base.attempts, base.reward
    );

    // LLM draft: local Ollama model ranks actions.
    let llm = OllamaDraftModel::new(model, &config);
    let mut pruner = QuestActionPruner::new(config.clone());
    let res = speculative_generate(&initial, &llm, &mut pruner, &decode_cfg);
    let stats = llm.stats.lock().unwrap();
    println!(
        "{} draft: goal={} steps={} attempts={} reward={} llm_calls={} parse_failures={}",
        model, res.goal, res.actions.len(), res.attempts, res.reward,
        stats.llm_calls, stats.parse_failures
    );
    let path: Vec<String> = res.actions.iter().map(|&t| action_label(t)).collect();
    println!("trajectory: {path:?}");
}
