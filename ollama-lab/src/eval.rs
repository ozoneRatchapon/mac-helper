//! `ollama-lab eval` — a fixed, scored task suite for local models.
//!
//! All calls hit 127.0.0.1:11434 (local-first rule). Runs synchronously on a
//! plain OS thread (reqwest::blocking), the same pattern as `bridge`.
//!
//! Base tier (7 tasks, all-or-nothing-ish):
//!   rust_codegen   — program compiles under rustc and prints the right value
//!   json_unassisted — strict JSON schema, no format hint
//!   json_format    — strict JSON schema with format=json
//!   arithmetic     — two integer multi-step questions
//!   tool_call      — emit a correct tool-call JSON
//!   needle         — recall a planted fact from a ~15K-token haystack
//!   multi_turn     — remember a fact across two chat turns
//!
//! Hard tier (6 tasks, partial credit, designed to differentiate models):
//!   arith_words       — word problems incl. a distractor value
//!   codegen_strict    — Rust program with stdin, scored per contract
//!   json_deep         — nested JSON schema, six independent checks
//!   tool_choice       — pick the right tool of two similar ones
//!   needle_deep       — decoys + master fact at ~75% depth
//!   multi_turn_update — a fact stated, then corrected; keep the correction
//!
//! Results print as two matrices (base / hard subtotals + overall) and are
//! saved to `ollama-lab/results/eval-<ts>.json` so successive runs form a
//! ratchet; each task also records wall-clock elapsed_ms (per-model total in
//! `total_elapsed_ms`).

use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const OLLAMA: &str = "http://127.0.0.1:11434";

/// Where the suite sends its requests.
///
/// Default is the local Ollama native API. `--openai <base-url>` switches to a
/// generic OpenAI-compatible `/chat/completions`, so the SAME task definitions
/// and scorers can measure a remote endpoint — the only way to compare, say, a
/// BF16 hosted build against the local quantized one without two divergent
/// implementations of the suite.
pub struct Backend {
    pub base: String,
    pub openai: bool,
}

static BACKEND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();

fn backend() -> &'static Backend {
    BACKEND.get_or_init(|| Backend {
        base: OLLAMA.to_string(),
        openai: false,
    })
}

/// Point the suite at an OpenAI-compatible endpoint. Must be called before the
/// first request; later calls are ignored.
pub fn set_openai_backend(base_url: &str) {
    let _ = BACKEND.set(Backend {
        base: base_url.trim_end_matches('/').to_string(),
        openai: true,
    });
}

/// Extract `choices[0].message.content` from an OpenAI-compatible reply.
///
/// Thinking models can return an empty `content` with the text in `reasoning`;
/// treat that as a failure rather than scoring an empty answer, since the two
/// look identical to the scorers otherwise.
fn openai_content(v: &Value) -> Result<String, String> {
    let msg = v
        .pointer("/choices/0/message")
        .ok_or_else(|| format!("no choices[0].message: {}", &v.to_string()[..200.min(v.to_string().len())]))?;
    let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
    if content.is_empty() {
        let reasoning_len = msg
            .get("reasoning")
            .or_else(|| msg.get("reasoning_content"))
            .and_then(Value::as_str)
            .map(str::len)
            .unwrap_or(0);
        return Err(format!(
            "empty content (reasoning={reasoning_len} chars) — model spent its budget thinking"
        ));
    }
    Ok(content.to_string())
}

/// One OpenAI-compatible chat request shared by `generate` and `chat`.
fn openai_chat(
    c: &reqwest::blocking::Client,
    model: &str,
    messages: Vec<Value>,
    json_format: bool,
) -> Result<String, String> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "temperature": 0.0,
        "max_tokens": 4096,
        "reasoning_effort": "none",
    });
    if json_format {
        body["response_format"] = serde_json::json!({ "type": "json_object" });
    }
    let url = format!("{}/chat/completions", backend().base);
    let resp = c
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| format!("send: {e}"))?;
    let status = resp.status();
    let text = resp.text().map_err(|e| format!("read: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", &text[..300.min(text.len())]));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;
    openai_content(&v)
}

fn user_msg(content: &str) -> Value {
    serde_json::json!({ "role": "user", "content": content })
}
const RESULTS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
const TIMEOUT: Duration = Duration::from_secs(600);

/// One scored task result. Transport/compile errors land here as 0.0 + detail
/// rather than aborting the whole model's run.
pub struct TaskScore {
    pub name: &'static str,
    pub score: f64,
    pub detail: String,
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .map_err(|e| format!("client: {e}"))
}

/// POST /api/generate (raw, like `bridge`), optionally with `format: "json"`.
fn generate(
    c: &reqwest::blocking::Client,
    model: &str,
    prompt: &str,
    json_format: bool,
) -> Result<String, String> {
    if backend().openai {
        return openai_chat(c, model, vec![user_msg(prompt)], json_format);
    }
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "think": false,
        "options": { "temperature": 0.0 }
    });
    if json_format {
        body["format"] = Value::String("json".into());
    }
    let resp = c
        .post(format!("{OLLAMA}/api/generate"))
        .json(&body)
        .send()
        .map_err(|e| format!("send: {e}"))?;
    let status = resp.status();
    let text = resp.text().map_err(|e| format!("read: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;
    v.get("response")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "no `response` field in /api/generate".into())
}

/// POST /api/chat with a message list; returns the assistant reply.
fn chat(
    c: &reqwest::blocking::Client,
    model: &str,
    messages: &[(&str, &str)],
) -> Result<String, String> {
    if backend().openai {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect();
        return openai_chat(c, model, msgs, false);
    }
    let msgs: Vec<Value> = messages
        .iter()
        .map(|(role, content)| {
            Value::Object(serde_json::Map::from_iter([
                ("role".into(), Value::String((*role).into())),
                ("content".into(), Value::String((*content).into())),
            ]))
        })
        .collect();
    let body =
        serde_json::json!({ "model": model, "messages": msgs, "stream": false, "think": false });
    let resp = c
        .post(format!("{OLLAMA}/api/chat"))
        .json(&body)
        .send()
        .map_err(|e| format!("send: {e}"))?;
    let status = resp.status();
    let text = resp.text().map_err(|e| format!("read: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;
    v.pointer("/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "no /message/content in /api/chat".into())
}

// ---------- answer extraction / scoring helpers ----------

/// First balanced top-level JSON object in `text`; tolerates prose and fences.
fn extract_json(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, b) in text.bytes().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&text[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// All base-10 integers appearing in `text` (runs of digits split on anything else).
fn ints_in(text: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(n) = cur.parse::<i64>() {
                out.push(n);
            }
            cur.clear();
        }
    }
    if !cur.is_empty()
        && let Ok(n) = cur.parse::<i64>()
    {
        out.push(n);
    }
    out
}

/// Strict schema: name:string, age:integer, hobbies:2-3 strings.
fn score_json_task(resp: &str) -> (f64, String) {
    let Some(v) = extract_json(resp) else {
        return (0.0, "no parseable JSON object in response".into());
    };
    let Some(o) = v.as_object() else {
        return (0.0, "top-level JSON is not an object".into());
    };
    let name_ok = o.get("name").is_some_and(Value::is_string);
    let age_ok = o.get("age").map(Value::is_i64).unwrap_or(false);
    let hobbies_ok = o.get("hobbies").is_some_and(|h| {
        h.as_array()
            .is_some_and(|a| (2..=3).contains(&a.len()) && a.iter().all(Value::is_string))
    });
    if name_ok && age_ok && hobbies_ok {
        (1.0, "schema satisfied".into())
    } else {
        let bad = [
            (!name_ok, "name not a string"),
            (!age_ok, "age not an integer"),
            (!hobbies_ok, "hobbies not 2-3 strings"),
        ]
        .into_iter()
        .filter(|(fail, _)| *fail)
        .map(|(_, why)| why)
        .collect::<Vec<_>>()
        .join("; ");
        (0.5, format!("valid object but {bad}"))
    }
}

/// Tool-call shape: get_weather(city="Osaka", unit="fahrenheit").
fn score_tool_call(resp: &str) -> (f64, String) {
    let Some(v) = extract_json(resp) else {
        return (0.0, "no JSON object in response".into());
    };
    let name = v
        .get("name")
        .or_else(|| v.get("tool"))
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/function/name").and_then(Value::as_str))
        .unwrap_or("");
    let args = v
        .get("arguments")
        .or_else(|| v.get("args"))
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("input"))
        .or_else(|| v.pointer("/function/arguments"))
        .or_else(|| v.pointer("/function/parameters"));
    let city = args
        .and_then(|a| a.get("city"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    let unit = args
        .and_then(|a| a.get("unit"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    let name_ok = name == "get_weather";
    let city_ok = city == "osaka";
    let unit_ok = unit == "fahrenheit";
    match (name_ok, city_ok, unit_ok) {
        (true, true, true) => (1.0, "correct tool call".into()),
        (true, true, false) => (0.5, format!("right tool+city, unit={unit:?}")),
        (_, true, _) => (0.25, "city right, tool/unit off".into()),
        _ => (
            0.0,
            format!("wrong call: name={name:?} city={city:?} unit={unit:?}"),
        ),
    }
}

// ---------- tasks ----------

const CODEGEN_PROMPT: &str = "Write a complete single-file Rust program (a fn main) that computes the sum of all even numbers from 2 to 100 inclusive, and prints only that number with no other output. Return the code in a single ```rust block and nothing else.";
const CODEGEN_ANSWER: &str = "2550";

fn extract_code_block(text: &str) -> Option<String> {
    let fence = text.find("```rust").or_else(|| text.find("```"))?;
    let rest = &text[fence..];
    let after = rest.find('\n')?;
    let body = &rest[after..];
    let end = body.find("```")?;
    Some(body[..end].trim().to_string())
}

/// Monotonic counter for per-invocation temp dirs (avoids clobbering a shared dir).
static RUN_SEQ: AtomicU64 = AtomicU64::new(0);

/// Compile with rustc, optionally feed stdin, run, return stdout (or the failure stage).
fn rustc_run(code: &str, stdin: Option<&str>) -> Result<String, String> {
    use std::io::Write;

    let seq = RUN_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("ollama_lab_eval_{}_{}", std::process::id(), seq));
    std::fs::create_dir_all(&dir).map_err(|e| format!("tempdir: {e}"))?;
    let src = dir.join("main.rs");
    let bin = dir.join("prog");
    let mut f = std::fs::File::create(&src).map_err(|e| format!("write src: {e}"))?;
    f.write_all(code.as_bytes())
        .map_err(|e| format!("write src: {e}"))?;
    drop(f);

    let compile = std::process::Command::new("rustc")
        .args(["-O", "--edition", "2021"])
        .arg("-o")
        .arg(&bin)
        .arg(&src)
        .output()
        .map_err(|e| format!("rustc spawn: {e}"))?;
    if !compile.status.success() {
        let err = String::from_utf8_lossy(&compile.stderr);
        let tail = err
            .lines()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!("compile failed: {tail}"));
    }

    let mut child = std::process::Command::new(&bin)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("run spawn: {e}"))?;
    if let (Some(input), Some(mut pipe)) = (stdin, child.stdin.take()) {
        pipe.write_all(input.as_bytes())
            .map_err(|e| format!("stdin write: {e}"))?;
    }
    let out = child.wait_with_output().map_err(|e| format!("wait: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "run failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn task_rust_codegen(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "rust_codegen";
    let resp = match generate(c, model, CODEGEN_PROMPT, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let code = match extract_code_block(&resp) {
        Some(code) => code,
        None => {
            return TaskScore {
                name,
                score: 0.0,
                detail: "no code block in response".into(),
            };
        }
    };
    match rustc_run(&code, None) {
        Ok(out) => {
            if out.contains(CODEGEN_ANSWER) {
                TaskScore {
                    name,
                    score: 1.0,
                    detail: format!(
                        "compiles, output contains {CODEGEN_ANSWER} (stdout={:?})",
                        out.trim()
                    ),
                }
            } else {
                TaskScore {
                    name,
                    score: 0.25,
                    detail: format!("runs but wrong output: {:?}", out.trim()),
                }
            }
        }
        Err(e) => TaskScore {
            name,
            score: 0.0,
            detail: e,
        },
    }
}

const JSON_PROMPT: &str = "Respond with ONLY a JSON object (no markdown, no prose) that has exactly these fields:\n- \"name\": a string, the name of a fictional bird\n- \"age\": an integer, its age in years\n- \"hobbies\": an array of 2 or 3 strings";

fn task_json_unassisted(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "json_unassisted";
    let resp = match generate(c, model, JSON_PROMPT, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let (score, detail) = score_json_task(&resp);
    TaskScore {
        name,
        score,
        detail,
    }
}

fn task_json_format(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "json_format";
    let resp = match generate(c, model, JSON_PROMPT, true) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let (score, detail) = score_json_task(&resp);
    TaskScore {
        name,
        score,
        detail,
    }
}

const ARITH_TASKS: [(&str, i64); 2] = [
    ("What is 47 + 38 * 2? Answer with only the number.", 123),
    (
        "A farmer has 6 fields. Each field has 9 rows. Each row has 5 plants. How many plants are there in total? Answer with only the number.",
        270,
    ),
];

/// Shared arithmetic scoring loop: one point per problem where the expected
/// integer appears in the response. Score = fraction of problems passed.
fn task_arith(
    c: &reqwest::blocking::Client,
    model: &str,
    name: &'static str,
    tasks: &[(&str, i64)],
) -> TaskScore {
    let mut total = 0.0f64;
    let mut details = Vec::new();
    for (prompt, expected) in tasks {
        let detail = match generate(c, model, prompt, false) {
            Ok(resp) if ints_in(&resp).contains(expected) => {
                total += 1.0;
                format!("\"{prompt}\" -> correct ({expected})")
            }
            Ok(resp) => {
                let tail: String = resp.chars().take(80).collect();
                format!("\"{prompt}\" -> expected {expected}, got {tail:?}")
            }
            Err(e) => {
                format!("\"{prompt}\" -> {e}")
            }
        };
        details.push(detail);
    }
    TaskScore {
        name,
        score: total / tasks.len() as f64,
        detail: details.join("; "),
    }
}

fn task_arithmetic(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    task_arith(c, model, "arithmetic", &ARITH_TASKS)
}

/// Hard tier: multi-step word problems. Second problem contains a distractor
/// value (50) that must NOT be included in the answer.
const ARITH_WORD_TASKS: [(&str, i64); 2] = [
    (
        "A bakery has 13 crates, each holding 90 eggs. 250 eggs are broken and discarded. The remaining good eggs are split equally into 2 bins. How many eggs are in one bin? Answer with only the number.",
        460,
    ),
    (
        "A shop sold 40 loaves of bread on Monday, 55 loaves on Tuesday, and 110 loaves on Wednesday. It also had 50 loaves left in storage at the end of the week, which it did not sell. How many loaves did it sell in total over those three days? Answer with only the number.",
        205,
    ),
];

fn task_arith_words(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    task_arith(c, model, "arith_words", &ARITH_WORD_TASKS)
}

// ---------------------------------------------------------------------------
// Hard tier (partial credit, designed to differentiate models)
// ---------------------------------------------------------------------------

const CODEGEN_STRICT_PROMPT: &str = "Write a complete Rust program (fn main, no external crates) that reads ONE line of comma-separated integers from standard input and prints the sum of the squares of the negative numbers. If the line contains no negative numbers, print the word none instead. Print nothing else. Do not panic on well-formed input. Answer with only the code.";

const CODEGEN_STRICT_CONTRACTS: [(&str, &str); 3] =
    [("1,-2,3,-4", "20"), ("5,7,9", "none"), ("-3,10,-1", "10")];

/// Hard tier: Rust program with stdin input; scored per contract (partial credit).
fn task_codegen_strict(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "codegen_strict";
    let resp = match generate(c, model, CODEGEN_STRICT_PROMPT, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let code = match extract_code_block(&resp) {
        Some(code) => code,
        None => resp.trim().to_string(),
    };
    let mut passed = 0usize;
    let mut details = Vec::new();
    for (input, expected) in CODEGEN_STRICT_CONTRACTS {
        let detail = match rustc_run(&code, Some(input)) {
            Ok(out) if out.trim() == expected => {
                passed += 1;
                format!("stdin={input:?} -> {expected:?} ok")
            }
            Ok(out) => {
                let tail: String = out.trim().chars().take(40).collect();
                format!("stdin={input:?} -> expected {expected:?}, got {tail:?}")
            }
            Err(e) => format!("stdin={input:?} -> {e}"),
        };
        details.push(detail);
    }
    TaskScore {
        name,
        score: passed as f64 / CODEGEN_STRICT_CONTRACTS.len() as f64,
        detail: details.join("; "),
    }
}

/// Hand-rolled semver shape check: DIGITS[.DIGITS[.DIGITS]], no empty parts.
fn looks_semver(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    !parts.is_empty()
        && parts.len() <= 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

/// Hand-rolled ISO-8601 date shape check: YYYY-MM-DD, month 01-12, day 01-31.
fn looks_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let digits = |lo: usize, hi: usize| b[lo..hi].iter().all(|c| c.is_ascii_digit());
    if !digits(0, 4) || !digits(5, 7) || !digits(8, 10) {
        return false;
    }
    let num = |lo: usize, hi: usize| std::str::from_utf8(&b[lo..hi]).unwrap_or("").parse::<u8>();
    matches!(num(5, 7), Ok(1..=12)) && matches!(num(8, 10), Ok(1..=31))
}

/// Six independent schema checks; score = fraction passed (partial credit).
fn score_json_deep(resp: &str) -> (f64, String) {
    let Some(v) = extract_json(resp) else {
        return (0.0, "no JSON object found in response".into());
    };
    let mut passed = 0usize;
    let mut details = Vec::new();
    let record = |ok: bool, label: &str, passed: &mut usize, details: &mut Vec<String>| {
        if ok {
            *passed += 1;
        }
        details.push(if ok {
            format!("{label} ok")
        } else {
            format!("{label} FAIL")
        });
    };

    let name_ok = v
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty());
    record(name_ok, "name", &mut passed, &mut details);

    let version_ok = v
        .get("version")
        .and_then(Value::as_str)
        .is_some_and(looks_semver);
    record(version_ok, "version(semver)", &mut passed, &mut details);

    let date_ok = v
        .get("date")
        .and_then(Value::as_str)
        .is_some_and(looks_date);
    record(date_ok, "date(ISO)", &mut passed, &mut details);

    let yanked_ok = v.get("yanked").is_some_and(Value::is_boolean);
    record(yanked_ok, "yanked(bool)", &mut passed, &mut details);

    let changes = v.get("changes");
    let two_objs = changes
        .and_then(Value::as_array)
        .is_some_and(|a| a.len() == 2 && a.iter().all(Value::is_object));
    record(two_objs, "changes=2 objects", &mut passed, &mut details);

    let items_ok = changes.and_then(Value::as_array).is_some_and(|a| {
        a.iter().all(|item| {
            let ty = item.get("type").and_then(Value::as_str).unwrap_or("");
            let ty_ok = matches!(ty, "feature" | "fix" | "perf");
            let words = item
                .get("summary")
                .and_then(Value::as_str)
                .map(|s| s.split_whitespace().count())
                .unwrap_or(0);
            ty_ok && (3..=10).contains(&words)
        })
    });
    record(
        items_ok,
        "items type+summary(3-10 words)",
        &mut passed,
        &mut details,
    );

    (passed as f64 / 6.0, details.join("; "))
}

const JSON_DEEP_PROMPT: &str = "Produce a JSON object (and nothing else) describing a software release with exactly these fields:\n- \"name\": non-empty string, the package name\n- \"version\": semver string like \"1.2.3\"\n- \"date\": ISO 8601 date like \"2026-01-15\"\n- \"yanked\": boolean\n- \"changes\": array of exactly 2 objects; each object has \"type\" (one of feature, fix, perf) and \"summary\" (a string of 3 to 10 words)\nRespond with ONLY the JSON object, no prose.";

/// Hard tier: nested JSON schema, six independent checks (partial credit).
fn task_json_deep(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "json_deep";
    let resp = match generate(c, model, JSON_DEEP_PROMPT, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let (score, detail) = score_json_deep(&resp);
    TaskScore {
        name,
        score,
        detail,
    }
}

const TOOL_PROMPT: &str = "You have exactly one tool:\n{\n  \"name\": \"get_weather\",\n  \"description\": \"Get the current weather for a city\",\n  \"parameters\": {\n    \"city\": \"string, required, the city name\",\n    \"unit\": \"string, 'celsius' or 'fahrenheit', defaults to 'celsius'\"\n  }\n}\nRespond with ONLY the JSON tool call (an object with the tool name and its arguments) for this user request. No prose.\nUser: What's the weather in Osaka in fahrenheit?";

fn task_tool_call(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "tool_call";
    let resp = match generate(c, model, TOOL_PROMPT, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let (score, detail) = score_tool_call(&resp);
    TaskScore {
        name,
        score,
        detail,
    }
}

const NEEDLE_CODE: &str = "ZEPHYR-9271";

/// Deterministic ~15K-token haystack with the needle planted mid-text.
/// Fixed content (seeded by counter, not RNG) so every model sees the same text.
fn needle_prompt() -> (String, String) {
    const TOPICS: [&str; 6] = [
        "compilers",
        "memory management",
        "concurrency",
        "toolchains",
        "linters",
        "release profiles",
    ];
    let mut text = String::with_capacity(70_000);
    for (i, topic) in TOPICS.iter().cycle().take(280).enumerate() {
        text.push_str(&format!(
            "Section {i}. This passage reviews {topic} in systems programming, covering \
             history, design trade-offs, community adoption, performance, portability, and \
             ergonomics across many release cycles.\n"
        ));
        if i == 140 {
            text.push_str(&format!("FACT: The vault code is {NEEDLE_CODE}.\n"));
        }
    }
    (
        format!(
            "Context:\n{text}\n\nQuestion: What is the vault code mentioned in the context? \
             Answer with only the code."
        ),
        NEEDLE_CODE.to_string(),
    )
}

fn task_needle(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "needle";
    let (prompt, expected) = needle_prompt();
    let resp = match generate(c, model, &prompt, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let lower = resp.to_lowercase();
    let expected_lower = expected.to_lowercase();
    if lower.contains(&expected_lower) {
        TaskScore {
            name,
            score: 1.0,
            detail: format!("recalled {expected}"),
        }
    } else if lower.contains("zephyr") {
        TaskScore {
            name,
            score: 0.5,
            detail: format!("partial recall, got {:?}", resp.trim()),
        }
    } else {
        TaskScore {
            name,
            score: 0.0,
            detail: format!("no recall, got {:?}", resp.trim()),
        }
    }
}

const MULTITURN_FACTS: &str =
    "Remember two things: my lucky number is 74 and my favorite color is teal. Just say OK.";
const MULTITURN_QUESTION: &str = "What is my lucky number? Reply with only the number.";
const MULTITURN_ANSWER: i64 = 74;

fn task_multi_turn(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "multi_turn";
    let first = match chat(c, model, &[("user", MULTITURN_FACTS)]) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let reply = match chat(
        c,
        model,
        &[
            ("user", MULTITURN_FACTS),
            ("assistant", first.as_str()),
            ("user", MULTITURN_QUESTION),
        ],
    ) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    if ints_in(&reply).contains(&MULTITURN_ANSWER) {
        TaskScore {
            name,
            score: 1.0,
            detail: format!("remembered {MULTITURN_ANSWER}"),
        }
    } else {
        TaskScore {
            name,
            score: 0.0,
            detail: format!("lost the fact, got {:?}", reply.trim()),
        }
    }
}

const TOOL_CHOICE_PROMPT: &str = "You have exactly two tools:\n{\n  \"name\": \"send_message\",\n  \"description\": \"Send a short instant message (SMS-like) to a contact\",\n  \"parameters\": {\n    \"to\": \"string, required, the recipient's contact identifier\",\n    \"text\": \"string, required, the message body\"\n  }\n}\n{\n  \"name\": \"send_email\",\n  \"description\": \"Send an email to an address\",\n  \"parameters\": {\n    \"to\": \"string, required, the recipient's email address\",\n    \"body\": \"string, required, the email body\"\n  }\n}\nRespond with ONLY the JSON tool call (an object with the tool name and its arguments) for this user request. No prose.\nUser: Email maria@example.com and say 'see you at 5'.";

/// Hard tier: two similar tools, one must be picked by intent ("Email").
/// 0.5 correct tool + 0.25 `to` + 0.25 payload (partial credit).
fn score_tool_choice(resp: &str) -> (f64, String) {
    let Some(v) = extract_json(resp) else {
        return (0.0, format!("no JSON object found: {:?}", resp.trim()));
    };
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| v.get("tool").and_then(Value::as_str))
        .or_else(|| v.pointer("/function/name").and_then(Value::as_str))
        .unwrap_or("");
    let mut passed = 0usize;
    let mut details = Vec::new();
    let record = |ok: bool, passed: &mut usize, details: &mut Vec<String>, label: &str| {
        if ok {
            *passed += 1;
        }
        details.push(if ok {
            format!("{label} ok")
        } else {
            format!("{label} FAIL")
        });
    };
    let tool_ok = name == "send_email";
    record(
        tool_ok,
        &mut passed,
        &mut details,
        &format!("tool (want send_email, got {name})"),
    );
    let args = v
        .get("arguments")
        .or_else(|| v.get("args"))
        .or_else(|| v.pointer("/function/arguments"))
        .unwrap_or(&Value::Null);
    let to = args.get("to").and_then(Value::as_str).unwrap_or("");
    let to_ok = to.contains("maria@example.com");
    record(
        to_ok,
        &mut passed,
        &mut details,
        &format!("to (got {to:?})"),
    );
    let payload = args
        .get("body")
        .and_then(Value::as_str)
        .or_else(|| args.get("text").and_then(Value::as_str))
        .unwrap_or("");
    let payload_ok = payload.to_lowercase().contains("see you at 5");
    record(
        payload_ok,
        &mut passed,
        &mut details,
        &format!("payload (got {payload:?})"),
    );
    // Weighted: tool 0.5, to 0.25, payload 0.25 (partial credit).
    let mut score = 0.0f64;
    if tool_ok {
        score += 0.5;
    }
    if to_ok {
        score += 0.25;
    }
    if payload_ok {
        score += 0.25;
    }
    (score, details.join("; "))
}

fn task_tool_choice(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "tool_choice";
    let resp = match generate(c, model, TOOL_CHOICE_PROMPT, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let (score, detail) = score_tool_choice(&resp);
    TaskScore {
        name,
        score,
        detail,
    }
}

const NEEDLE_DEEP_MASTER: &str = "ATLAS-4815";

/// Hard tier: ~280-section haystack. Three ZEPHYR decoys are planted in the
/// first ~54%; the real ATLAS-4815 sits at section 210 (~75% depth). Fixed
/// content, same text for every model.
fn needle_deep_prompt() -> String {
    const TOPICS: [&str; 6] = [
        "compilers",
        "memory management",
        "concurrency",
        "toolchains",
        "linters",
        "release profiles",
    ];
    const SECTIONS: usize = 280;
    let mut text = String::with_capacity(75_000);
    for (i, topic) in TOPICS.iter().cycle().take(SECTIONS).enumerate() {
        text.push_str(&format!(
            "Section {i}. This passage reviews {topic} in systems programming, covering \
             history, design trade-offs, community adoption, performance, portability, and \
             ergonomics across many release cycles.\n"
        ));
        match i {
            40 => text.push_str("FACT: The vault code is ZEPHYR-1101.\n"),
            100 => text.push_str("FACT: The vault code is ZEPHYR-2202.\n"),
            150 => text.push_str("FACT: The vault code is ZEPHYR-3303.\n"),
            210 => text.push_str(&format!("FACT: The vault code is {NEEDLE_DEEP_MASTER}.\n")),
            _ => {}
        }
    }
    format!(
        "Context:\n{text}\n\nQuestion: What is the vault code mentioned in the context? \
         Answer with only the code."
    )
}

/// Hard tier: recall must beat decoys and come from ~75% depth. Partial credit
/// for a partial master recall; decoy recall scores low.
fn task_needle_deep(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "needle_deep";
    let prompt = needle_deep_prompt();
    let resp = match generate(c, model, &prompt, false) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let lower = resp.to_lowercase();
    if lower.contains("atlas-4815") {
        TaskScore {
            name,
            score: 1.0,
            detail: format!("recalled {NEEDLE_DEEP_MASTER}"),
        }
    } else if lower.contains("atlas") {
        TaskScore {
            name,
            score: 0.5,
            detail: format!("partial master recall, got {:?}", resp.trim()),
        }
    } else if lower.contains("zephyr") {
        TaskScore {
            name,
            score: 0.25,
            detail: format!("decoy, got {:?}", resp.trim()),
        }
    } else {
        TaskScore {
            name,
            score: 0.0,
            detail: format!("no recall, got {:?}", resp.trim()),
        }
    }
}

const MTU_FACT: &str = "Remember: my lucky number is 74. Just say OK.";
const MTU_CORRECTION: &str = "Correction: I misspoke, my lucky number is 96, not 74. Just say OK.";
const MTU_QUESTION: &str = "What is my lucky number? Reply with only the number.";
const MTU_STALE: i64 = 74;
const MTU_TRUE: i64 = 96;

/// Hard tier: a fact is stated, then corrected. 96-only = 1.0, both = 0.5,
/// stale 74-only = 0.25, neither = 0.0.
fn score_multi_turn_update(reply: &str) -> (f64, String) {
    let nums = ints_in(reply);
    let has_true = nums.contains(&MTU_TRUE);
    let has_stale = nums.contains(&MTU_STALE);
    match (has_true, has_stale) {
        (true, false) => (1.0, format!("updated to {MTU_TRUE}, dropped {MTU_STALE}")),
        (true, true) => (
            0.5,
            format!("kept both {MTU_STALE} and {MTU_TRUE}: {:?}", reply.trim()),
        ),
        (false, true) => (0.25, format!("stale {MTU_STALE} only: {:?}", reply.trim())),
        (false, false) => (0.0, format!("neither, got {:?}", reply.trim())),
    }
}

fn task_multi_turn_update(c: &reqwest::blocking::Client, model: &str) -> TaskScore {
    let name = "multi_turn_update";
    let turn1 = match chat(c, model, &[("user", MTU_FACT)]) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let turn2 = match chat(
        c,
        model,
        &[
            ("user", MTU_FACT),
            ("assistant", &turn1),
            ("user", MTU_CORRECTION),
        ],
    ) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let reply = match chat(
        c,
        model,
        &[
            ("user", MTU_FACT),
            ("assistant", &turn1),
            ("user", MTU_CORRECTION),
            ("assistant", &turn2),
            ("user", MTU_QUESTION),
        ],
    ) {
        Ok(r) => r,
        Err(e) => {
            return TaskScore {
                name,
                score: 0.0,
                detail: e,
            };
        }
    };
    let (score, detail) = score_multi_turn_update(&reply);
    TaskScore {
        name,
        score,
        detail,
    }
}

// ---------- runner ----------

/// Number of base-tier tasks; hard tier is everything else.
const BASE_LEN: usize = 7;

type Timed = (TaskScore, u64); // (score, wall-clock ms for that task)

fn run_timed(
    f: fn(&reqwest::blocking::Client, &str) -> TaskScore,
    c: &reqwest::blocking::Client,
    model: &str,
) -> Timed {
    let t0 = std::time::Instant::now();
    let s = f(c, model);
    (s, t0.elapsed().as_millis() as u64)
}

fn base_tasks(c: &reqwest::blocking::Client, model: &str) -> Vec<Timed> {
    [
        task_rust_codegen,
        task_json_unassisted,
        task_json_format,
        task_arithmetic,
        task_tool_call,
        task_needle,
        task_multi_turn,
    ]
    .iter()
    .map(|f| run_timed(*f, c, model))
    .collect()
}

fn hard_tasks(c: &reqwest::blocking::Client, model: &str) -> Vec<Timed> {
    [
        task_arith_words,
        task_codegen_strict,
        task_json_deep,
        task_tool_choice,
        task_needle_deep,
        task_multi_turn_update,
    ]
    .iter()
    .map(|f| run_timed(*f, c, model))
    .collect()
}

fn subtotal(ts: &[Timed]) -> f64 {
    ts.iter().map(|(t, _)| t.score).sum::<f64>() / ts.len() as f64
}

fn print_block(results: &[(String, Vec<Timed>, Vec<Timed>)], hard: bool) {
    let title = if hard { "hard" } else { "base" };
    let first = &results[0];
    let head: &[Timed] = if hard { &first.2 } else { &first.1 };
    print!("\n[{title} tier]");
    print!("{:<24}", "model");
    for (t, _) in head {
        print!("{:<17}", t.name);
    }
    println!("{title:<17}{:<17}", "overall");
    for (model, base, hardts) in results {
        let row: &[Timed] = if hard { hardts } else { base };
        let overall: f64 = base
            .iter()
            .chain(hardts.iter())
            .map(|(t, _)| t.score)
            .sum::<f64>()
            / (base.len() + hardts.len()) as f64;
        print!("{model:<24}");
        for (t, _) in row {
            print!("{:<17}", format!("{:.2}", t.score));
        }
        println!(
            "{:<17}{:<17}",
            format!("{:.2}", subtotal(row)),
            format!("{overall:.2}")
        );
    }
}

fn stamp() -> String {
    let out = std::process::Command::new("date")
        .args(["+%Y%m%d_%H%M%S"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() { epoch_secs() } else { s }
        }
        _ => epoch_secs(),
    }
}

fn epoch_secs() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

/// Run the full suite for each model, print the matrix, save a timestamped JSON.
pub fn run(models: &[String]) {
    if models.is_empty() {
        eprintln!("eval: no models given");
        return;
    }
    let Ok(c) = client() else {
        eprintln!("eval: failed to build HTTP client");
        return;
    };

    let mut results: Vec<(String, Vec<Timed>, Vec<Timed>)> = Vec::new();
    for model in models {
        eprintln!("eval: running suite for {model} ...");
        let t0 = std::time::Instant::now();
        let base = base_tasks(&c, model);
        let hard = hard_tasks(&c, model);
        debug_assert_eq!(base.len(), BASE_LEN);
        eprintln!("eval: {model} done in {} ms", t0.elapsed().as_millis());
        results.push((model.clone(), base, hard));
    }

    // ---- print matrices (base tier, then hard tier) ----
    print_block(&results, false);
    print_block(&results, true);

    // ---- details ----
    for (model, base, hard) in &results {
        println!();
        println!("{model}:");
        for (t, ms) in base.iter().chain(hard.iter()) {
            println!(
                "  - {:<20} {:.2} ({} ms) — {}",
                t.name, t.score, ms, t.detail
            );
        }
    }

    // ---- persist ----
    let dir = std::path::Path::new(RESULTS_DIR);
    let fname = format!("eval-{}.json", stamp());
    let doc = serde_json::json!({
        "server": backend().base,
        // Which transport produced these numbers. Local Ollama runs form the
        // ratchet; an `openai` row came from a remote endpoint and must not be
        // compared against them as if it were the same setup (different
        // hardware, quantization and contention).
        "backend": if backend().openai { "openai" } else { "ollama-native" },
        "stamp": fname,
        "base_tasks": results[0].1.iter().map(|(t, _)| t.name).collect::<Vec<_>>(),
        "hard_tasks": results[0].2.iter().map(|(t, _)| t.name).collect::<Vec<_>>(),
        "models": results.iter().map(|(m, base, hard)| {
            let base_score = subtotal(base);
            let hard_score = subtotal(hard);
            let overall: f64 = base
                .iter()
                .chain(hard.iter())
                .map(|(t, _)| t.score)
                .sum::<f64>()
                / (base.len() + hard.len()) as f64;
            let total_ms: u64 = base.iter().chain(hard.iter()).map(|(_, ms)| ms).sum();
            serde_json::json!({
                "model": m,
                "base": base_score,
                "hard": hard_score,
                "overall": overall,
                "total_elapsed_ms": total_ms,
                "tasks": base.iter().chain(hard.iter()).map(|(t, ms)| serde_json::json!({
                    "name": t.name, "score": t.score, "detail": t.detail, "elapsed_ms": ms
                })).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    });
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("eval: cannot create results dir {RESULTS_DIR}: {e}");
        return;
    }
    let path = dir.join(&fname);
    match std::fs::write(
        &path,
        serde_json::to_string_pretty(&doc).unwrap_or_default(),
    ) {
        Ok(()) => println!("\nsaved: {}", path.display()),
        Err(e) => eprintln!("eval: failed to write {}: {e}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_tolerates_prose_and_fences() {
        let v = extract_json(
            "Sure! Here it is:\n```json\n{\"name\":\"Kite\",\"age\":4,\"hobbies\":[\"a\",\"b\"]}\n```\nEnjoy!",
        )
        .expect("should parse");
        assert_eq!(v["name"], "Kite");
    }

    #[test]
    fn extract_json_handles_escaped_quotes_and_nested_objects() {
        let text = r#"noise {"a":"he said \"hi\"","b":{"c":1}} tail"#;
        let v = extract_json(text).expect("should parse");
        assert_eq!(v["b"]["c"], 1);
    }

    #[test]
    fn extract_json_rejects_garbage() {
        assert!(extract_json("no braces here").is_none());
        assert!(extract_json("{ unbalanced").is_none());
        assert!(extract_json("").is_none());
    }

    #[test]
    fn ints_in_extracts_all_number_runs() {
        assert_eq!(ints_in("I think the answer is 123. Maybe 7?"), vec![123, 7]);
        assert!(ints_in("no numbers").is_empty());
    }

    #[test]
    fn score_json_task_requires_strict_types() {
        let (s, _) = score_json_task("{\"name\":\"Kite\",\"age\":4,\"hobbies\":[\"x\",\"y\"]}");
        assert_eq!(s, 1.0);
        // age as string, hobbies too short
        let (s, _) = score_json_task("{\"name\":\"Kite\",\"age\":\"4\",\"hobbies\":[\"x\"]}");
        assert!(s < 1.0);
        let (s, _) = score_json_task("not json at all");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn score_tool_call_accepts_common_shapes() {
        let (s, _) = score_tool_call(
            r#"{"name":"get_weather","arguments":{"city":"Osaka","unit":"fahrenheit"}}"#,
        );
        assert_eq!(s, 1.0);
        // qwen3.8:latest emits the tool name under a `tool` key (regression: it
        // scored 0.25 on a semantically correct call).
        let (s, _) = score_tool_call(
            r#"{"tool":"get_weather","arguments":{"city":"Osaka","unit":"fahrenheit"}}"#,
        );
        assert_eq!(s, 1.0);
        let (s, _) = score_tool_call(
            r#"{"function":{"name":"get_weather","arguments":{"city":"Osaka","unit":"fahrenheit"}}}"#,
        );
        assert_eq!(s, 1.0);
        let (s, _) = score_tool_call(r#"{"name":"get_weather","args":{"city":"Tokyo"}}"#);
        assert!(s < 1.0);
    }

    #[test]
    fn extract_code_block_pulls_rust_fence() {
        let text = "Here you go:\n```rust\nfn main() { println!(\"hi\"); }\n```\ndone";
        let code = extract_code_block(text).expect("should find block");
        assert!(code.contains("fn main"));
        assert!(!code.contains("```"));
    }

    #[test]
    fn needle_prompt_plants_the_code_midway() {
        let (p, expected) = needle_prompt();
        assert!(p.contains(&expected));
        assert!(
            p.chars().count() > 40_000,
            "haystack should be long, got {}",
            p.chars().count()
        );
        // needle is not in the first or last quarter
        let total = p.chars().count();
        let first_q = p.chars().take(total / 4).collect::<String>();
        let last_q: String = p.chars().skip(total - total / 4).collect();
        assert!(!first_q.contains(&expected));
        assert!(!last_q.contains(&expected));
    }

    #[test]
    fn looks_semver_and_date_reject_sloppy_shapes() {
        // semver: DIGITS[.DIGITS[.DIGITS]], <=3 non-empty all-digit parts
        assert!(looks_semver("1.2.3"));
        assert!(looks_semver("10.20"));
        assert!(!looks_semver("v1.2.3"));
        assert!(!looks_semver("1.2.3.4"));
        assert!(!looks_semver("1..3"));
        assert!(!looks_semver(""));
        // date: exactly 10 bytes, dashes at 4 and 7, month 01-12, day 01-31
        assert!(looks_date("2026-01-15"));
        assert!(looks_date("2026-12-31"));
        assert!(!looks_date("2026-13-01"));
        assert!(!looks_date("2026-00-10"));
        assert!(!looks_date("2026-01-32"));
        assert!(!looks_date("26-01-15"));
        assert!(!looks_date("2026/01/15"));
    }

    #[test]
    fn score_json_deep_grants_partial_credit() {
        let perfect = r#"{
            "name": "acme",
            "version": "1.2.3",
            "date": "2026-01-15",
            "yanked": false,
            "changes": [
                {"type": "feature", "summary": "added a brand new parser"},
                {"type": "fix", "summary": "fixed the crash in io"}
            ]
        }"#;
        let (s, _) = score_json_deep(perfect);
        assert!((s - 1.0).abs() < 1e-9, "perfect should be 1.0, got {s}");
        // same object minus `yanked` -> exactly 5 of 6 checks pass
        let no_yanked = r#"{
            "name": "acme",
            "version": "1.2.3",
            "date": "2026-01-15",
            "changes": [
                {"type": "feature", "summary": "added a brand new parser"},
                {"type": "fix", "summary": "fixed the crash in io"}
            ]
        }"#;
        let (s, _) = score_json_deep(no_yanked);
        assert!(
            (s - 5.0 / 6.0).abs() < 1e-9,
            "missing yanked should be 5/6, got {s}"
        );
        let (s, _) = score_json_deep("not json at all");
        assert!(s < 1e-9, "non-JSON should be 0.0, got {s}");
    }

    #[test]
    fn score_tool_choice_penalizes_wrong_tool() {
        let perfect =
            r#"{"name":"send_email","arguments":{"to":"maria@example.com","body":"see you at 5"}}"#;
        let (s, _) = score_tool_choice(perfect);
        assert!((s - 1.0).abs() < 1e-9, "perfect should be 1.0, got {s}");
        // right tool, bad payload -> tool 0.5 + to 0.25 = 0.75
        let bad_payload =
            r#"{"name":"send_email","arguments":{"to":"maria@example.com","body":"hello"}}"#;
        let (s, _) = score_tool_choice(bad_payload);
        assert!(
            (s - 0.75).abs() < 1e-9,
            "right tool + bad payload should be 0.75, got {s}"
        );
        // wrong tool, but to + payload ok -> to 0.25 + payload 0.25 = 0.5
        let wrong_tool = r#"{"name":"send_message","arguments":{"to":"maria@example.com","text":"see you at 5"}}"#;
        let (s, _) = score_tool_choice(wrong_tool);
        assert!(
            (s - 0.5).abs() < 1e-9,
            "wrong tool + good to/payload should be 0.5, got {s}"
        );
        // all wrong -> 0.0
        let all_wrong =
            r#"{"name":"send_message","arguments":{"to":"someone else","text":"nothing"}}"#;
        let (s, _) = score_tool_choice(all_wrong);
        assert!(s < 1e-9, "all wrong should be 0.0, got {s}");
        // non-JSON -> 0.0
        let (s, _) = score_tool_choice("no json here");
        assert!(s < 1e-9, "non-JSON should be 0.0, got {s}");
    }

    #[test]
    fn needle_deep_prompt_plants_decoys_then_master() {
        let p = needle_deep_prompt();
        for decoy in ["ZEPHYR-1101", "ZEPHYR-2202", "ZEPHYR-3303"] {
            assert_eq!(
                p.matches(decoy).count(),
                1,
                "{decoy} should appear exactly once"
            );
        }
        assert_eq!(
            p.matches(NEEDLE_DEEP_MASTER).count(),
            1,
            "master should appear exactly once"
        );
        let pos = p.find(NEEDLE_DEEP_MASTER).expect("master present");
        assert!(
            pos > p.len() / 2,
            "master should sit past the midpoint (pos {pos}, len {})",
            p.len()
        );
        for section in [0usize, 40, 100, 150, 210, 279] {
            let marker = format!("Section {section}. This");
            assert!(
                p.contains(&marker),
                "section {section} marker {marker:?} should be present"
            );
        }
    }

    #[test]
    fn rustc_run_supports_stdin_contracts() {
        // Reference program implementing the codegen_strict contract. Kept as a
        // raw string so no `\"` / `\\` sequences appear in this source.
        let program = r#"
use std::io::Read;
fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let line = s.lines().next().unwrap_or("");
    let mut sum = 0i64;
    let mut any = false;
    for tok in line.split(',') {
        if let Ok(n) = tok.trim().parse::<i64>() {
            if n < 0 { sum += n * n; any = true; }
        }
    }
    if any { println!("{sum}"); } else { println!("none"); }
}
"#;
        let out = rustc_run(program, Some("1,-2,3,-4")).expect("should compile+run");
        assert_eq!(out.trim(), "20");
        let out = rustc_run(program, Some("5,7,9")).expect("should compile+run");
        assert_eq!(out.trim(), "none");
        let out = rustc_run(program, Some("-3,10,-1")).expect("should compile+run");
        assert_eq!(out.trim(), "10");
    }

    #[test]
    fn arith_word_answers_check_out() {
        // Re-derive the expected answers independently of the task table.
        assert_eq!((13 * 90 - 250) / 2, 460);
        assert_eq!(40 + 55 + 110, 205);
        let answers: Vec<i64> = ARITH_WORD_TASKS.iter().map(|(_, a)| *a).collect();
        assert_eq!(answers, vec![460, 205]);
        let (p0, _) = &ARITH_WORD_TASKS[0];
        for tok in ["13", "90", "250"] {
            assert!(p0.contains(tok), "prompt 0 should contain {tok:?}");
        }
        let (p1, _) = &ARITH_WORD_TASKS[1];
        for tok in ["40", "55", "110", "50"] {
            assert!(p1.contains(tok), "prompt 1 should contain {tok:?}");
        }
    }
}
