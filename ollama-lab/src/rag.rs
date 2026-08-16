//! Phase 4 — fully-local RAG: nomic-embed-text retrieval + local generation.
//!
//! Corpus = markdown files from this repo and katgpt-rs. Chunk by paragraph
//! groups, embed locally, cosine top-k, answer with a local chat model.
//!
//! Run: `cargo run -- rag "<question>" [model]` (default gemma4:26b).

use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest};
use ollama_rs::Ollama;
use std::path::{Path, PathBuf};

const EMBED_MODEL: &str = "nomic-embed-text";
const CHUNK_TARGET: usize = 1200; // chars per chunk, paragraph-aligned
const TOP_K: usize = 4;
const MAX_FILE_BYTES: u64 = 200_000;

struct Chunk {
    source: String,
    text: String,
    vector: Vec<f32>,
}

fn collect_markdown(roots: &[&Path]) -> Vec<(String, String)> {
    let mut docs = Vec::new();
    let mut stack: Vec<PathBuf> = roots.iter().map(|p| p.to_path_buf()).collect();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                // Skip build output, VCS internals, and dot-dirs.
                if !name.starts_with('.') && name != "target" && name != "node_modules" {
                    stack.push(path);
                }
            } else if name.ends_with(".md")
                && entry.metadata().map(|m| m.len() <= MAX_FILE_BYTES).unwrap_or(false)
            {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    docs.push((path.display().to_string(), text));
                }
            }
        }
    }
    docs
}

fn chunk(source: &str, text: &str) -> Vec<(String, String)> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for para in text.split("\n\n") {
        if !current.is_empty() && current.len() + para.len() > CHUNK_TARGET {
            chunks.push((source.to_string(), std::mem::take(&mut current)));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.trim().is_empty() {
        chunks.push((source.to_string(), current));
    }
    chunks
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

async fn embed_batch(
    ollama: &Ollama,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let res = ollama
        .generate_embeddings(GenerateEmbeddingsRequest::new(
            EMBED_MODEL.to_string(),
            EmbeddingsInput::Multiple(texts),
        ))
        .await?;
    Ok(res.embeddings)
}

pub async fn run(question: &str, model: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::default();
    let home = std::env::var("HOME")?;
    let base = PathBuf::from(&home).join("mac helper");
    let roots = [base.as_path()];

    let t0 = std::time::Instant::now();
    let docs = collect_markdown(&roots);
    let mut pieces: Vec<(String, String)> = Vec::new();
    for (source, text) in &docs {
        pieces.extend(chunk(source, text));
    }
    println!("corpus: {} files -> {} chunks", docs.len(), pieces.len());

    // Embed in batches to keep request bodies bounded.
    let mut chunks: Vec<Chunk> = Vec::with_capacity(pieces.len());
    for batch in pieces.chunks(32) {
        let texts: Vec<String> = batch.iter().map(|(_, t)| t.clone()).collect();
        let vectors = embed_batch(&ollama, texts).await?;
        for ((source, text), vector) in batch.iter().cloned().zip(vectors) {
            chunks.push(Chunk { source, text, vector });
        }
    }
    println!("indexed in {:.1}s (local {})", t0.elapsed().as_secs_f32(), EMBED_MODEL);

    let qv = embed_batch(&ollama, vec![question.to_string()]).await?.remove(0);
    let mut scored: Vec<(f32, &Chunk)> = chunks.iter().map(|c| (cosine(&qv, &c.vector), c)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top = &scored[..TOP_K.min(scored.len())];

    println!("\ntop-{TOP_K} retrieved:");
    let mut context = String::new();
    for (score, c) in top {
        println!("  {score:.3}  {} ({} chars)", c.source, c.text.len());
        context.push_str(&format!("--- from {} ---\n{}\n\n", c.source, c.text));
    }

    let prompt = format!(
        "Answer the question using ONLY the context below. If the context is insufficient, say so.\n\n\
         CONTEXT:\n{context}\nQUESTION: {question}\n\nAnswer concisely."
    );
    let res = ollama
        .generate(GenerationRequest::new(model.to_string(), prompt))
        .await?;
    println!("\n{model} answers:\n{}", res.response);
    Ok(())
}
