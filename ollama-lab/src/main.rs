//! ollama-lab — local Ollama integration lab (all inference stays on 127.0.0.1:11434).
//!
//! Usage: ollama-lab <health|generate|stream|chat|embed|bridge|rag|eval> [args...]
//! Model selection: OLLAMA_MODEL env var (default: qwen3.8).

mod bridge;
mod eval;
mod rag;

use ollama_rs::Ollama;
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest};
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

fn model() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.8".to_string())
}

/// Fail fast with a clear message if the server isn't up — the article's
/// "common mistakes" section calls out surfacing connection-refused properly.
async fn health() -> Result<(), Box<dyn std::error::Error>> {
    let resp = reqwest::get("http://127.0.0.1:11434/").await;
    match resp {
        Ok(r) if r.status().is_success() => {
            println!("ok: {}", r.text().await?);
            Ok(())
        }
        Ok(r) => Err(format!("server responded with {}", r.status()).into()),
        Err(_) => Err(
            "Ollama server unreachable at 127.0.0.1:11434 — start it with `ollama serve`".into(),
        ),
    }
}

async fn generate(prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::default();
    let res = ollama
        .generate(GenerationRequest::new(model(), prompt.to_string()))
        .await?;
    println!("{}", res.response);
    Ok(())
}

async fn stream(prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::default();
    let mut stream = ollama
        .generate_stream(GenerationRequest::new(model(), prompt.to_string()))
        .await?;
    let mut stdout = tokio::io::stdout();
    while let Some(Ok(chunk)) = stream.next().await {
        for part in chunk {
            stdout.write_all(part.response.as_bytes()).await?;
            stdout.flush().await?;
        }
    }
    println!();
    Ok(())
}

async fn chat() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::default();
    let mut history = vec![ChatMessage::system(
        "You are a concise Rust tutor. Always show code examples.".to_string(),
    )];
    let questions = [
        "What is the difference between Box<T> and Rc<T>? One paragraph.",
        "When would I use Arc<T> instead? One paragraph.",
    ];
    for q in questions {
        history.push(ChatMessage::user(q.to_string()));
        let res = ollama
            .send_chat_messages(ChatMessageRequest::new(model(), history.clone()))
            .await?;
        println!("Q: {q}\nA: {}\n---", res.message.content);
        history.push(res.message);
    }
    Ok(())
}

async fn embed() -> Result<(), Box<dyn std::error::Error>> {
    use rag::cosine;

    let ollama = Ollama::default();
    let texts = [
        "Rust ownership and borrowing",
        "memory management in systems programming",
        "how to bake sourdough bread",
    ];
    let mut vectors = Vec::new();
    for t in texts {
        let res = ollama
            .generate_embeddings(GenerateEmbeddingsRequest::new(
                "nomic-embed-text".to_string(),
                EmbeddingsInput::Single(t.to_string()),
            ))
            .await?;
        vectors.push(res.embeddings.into_iter().next().unwrap_or_default());
    }
    println!(
        "sim(ownership, memory-mgmt) = {:.3}",
        cosine(&vectors[0], &vectors[1])
    );
    println!(
        "sim(ownership, sourdough)   = {:.3}",
        cosine(&vectors[0], &vectors[2])
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    // `args.get(2..)` rather than `args[2..]`: running the binary with no
    // arguments leaves a slice of length 1 and indexing it panics before the
    // usage message can print.
    let prompt = args.get(2..).unwrap_or(&[]).join(" ");
    let default_prompt = "Explain Rust's borrow checker in two sentences.";
    let prompt = if prompt.is_empty() {
        default_prompt
    } else {
        &prompt
    };

    health().await?;
    match cmd {
        "health" => {}
        "generate" => generate(prompt).await?,
        "stream" => stream(prompt).await?,
        "chat" => chat().await?,
        "embed" => embed().await?,
        "bridge" => {
            // reqwest::blocking may not run on a tokio runtime thread — hand
            // the whole sync ns-engine loop to a plain OS thread.
            // Skip flags when reading the positional model, so `bridge --big`
            // doesn't send "--big" to Ollama as the model name.
            let model = args
                .get(2..)
                .unwrap_or(&[])
                .iter()
                .find(|a| !a.starts_with("--"))
                .cloned()
                .unwrap_or_else(|| "gemma4:26b".to_string());
            let big = args.iter().any(|a| a == "--big");
            std::thread::spawn(move || bridge::run(&model, big))
                .join()
                .unwrap();
        }
        "rag" => {
            let q = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "What is ns-engine's core thesis?".to_string());
            let m = args
                .get(3)
                .cloned()
                .unwrap_or_else(|| "gemma4:26b".to_string());
            rag::run(&q, &m).await?;
        }
        "eval" => {
            // Fixed scored task suite; reqwest::blocking runs on a plain OS
            // thread (same pattern as `bridge`).
            // `--openai <base-url>` retargets the suite at any
            // OpenAI-compatible endpoint, so the same tasks and scorers can
            // measure a hosted build against the local one.
            let rest = args.get(2..).unwrap_or(&[]);
            if let Some(i) = rest.iter().position(|a| a == "--openai") {
                match rest.get(i + 1) {
                    Some(url) => eval::set_openai_backend(url),
                    None => {
                        eprintln!("--openai needs a base URL, e.g. --openai https://host/v1");
                        return Ok(());
                    }
                }
            }
            let skip_next = rest
                .iter()
                .position(|a| a == "--openai")
                .map(|i| i + 1)
                .unwrap_or(usize::MAX);
            let models: Vec<String> = rest
                .iter()
                .enumerate()
                .filter(|(i, a)| !a.starts_with("--") && *i != skip_next)
                .map(|(_, a)| a.clone())
                .collect();
            let models = if models.is_empty() {
                vec![
                    "gemma4:26b".to_string(),
                    "qwen3-coder:30b".to_string(),
                    "qwen3.8:latest".to_string(),
                ]
            } else {
                models
            };
            let handle = std::thread::spawn(move || eval::run(&models));
            if handle.join().is_err() {
                eprintln!("eval thread panicked");
            }
        }
        _ => eprintln!(
            "usage: ollama-lab <health|generate|stream|chat|embed|bridge|rag|eval> [args...]"
        ),
    }
    Ok(())
}
