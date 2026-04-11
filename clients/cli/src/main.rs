mod account;
mod debug;
mod edit;
mod imex;
mod input;
mod lb_fs;
mod list;
mod migrate;
mod share;
mod stream;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use account::ApiUrl;
use cli_rs::arg::Arg;
use cli_rs::cli_error::{CliError, CliResult};
use cli_rs::command::Command;
use cli_rs::flag::Flag;
use cli_rs::parser::Cmd;
use colored::Colorize;
use input::FileInput;
use lb_rs::model::core_config::Config;
use lb_rs::model::errors::LbErrKind;
use lb_rs::model::path_ops::Filter;
use lb_rs::{Lb, Uuid};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

// ── Model registry ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum ModelType {
    Embedder,
    Reranker,
}

struct ModelFile {
    filename:  &'static str,
    repo_id:   &'static str,
    subfolder: &'static str,
}

struct ModelMetadata {
    name:  &'static str,
    kind:  ModelType,
    files: &'static [ModelFile],
}

static MODELS: &[ModelMetadata] = &[
    ModelMetadata {
        name: "multilingual-e5-large",
        kind: ModelType::Embedder,
        files: &[
            ModelFile {
                filename:  "model.onnx",
                repo_id:   "Xenova/multilingual-e5-large",
                subfolder: "onnx",
            },
            ModelFile {
                filename:  "model.onnx_data",
                repo_id:   "Xenova/multilingual-e5-large",
                subfolder: "onnx",
            },
            ModelFile {
                filename:  "tokenizer.json",
                repo_id:   "Xenova/multilingual-e5-large",
                subfolder: "",
            },
        ],
    },
    ModelMetadata {
        name: "ms-marco-MiniLM-L-6-v2",
        kind: ModelType::Reranker,
        files: &[
            ModelFile {
                filename:  "model.onnx",
                repo_id:   "Xenova/ms-marco-MiniLM-L-6-v2",
                subfolder: "onnx",
            },
            ModelFile {
                filename:  "tokenizer.json",
                repo_id:   "Xenova/ms-marco-MiniLM-L-6-v2",
                subfolder: "",
            },
        ],
    },
];

// ── Index structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ChunkRecord {
    file_path:    String,
    file_name:    String,
    parent_chunk: String,
    child_text:   String,
}

#[derive(Serialize, Deserialize)]
struct IndexEntry {
    hmac:    Vec<u8>,
    chunks:  Vec<ChunkRecord>,
    vectors: Vec<f32>,
}

type Index = HashMap<String, IndexEntry>;

// ── Constants ─────────────────────────────────────────────────────────────────

const BI_HIDDEN:               usize = 1024;
const BI_MAX_LEN:              usize = 128;
const CHILD_CHARS:             usize = 512;
const PARENT_CHARS:            usize = 2048;
const CHILD_OVERLAP:           usize = 51;
const PARENT_OVERLAP:          usize = 205;
const TOP_K:                   usize = 5;
const RERANK_K:                usize = 100;
const MAX_CHUNKS_PER_FILE:     usize = 10_000;
const MAX_CHUNKS_FOR_INDEXING: usize = 10_000;
const BATCH_SIZE:              usize = 32;
const RERANKER_MAX_LEN:        usize = 512;
const PER_FILE_TIMEOUT_SECS:   u64   = 1800; // 30 min — enough for 8985 chunks on CPU
const SLOW_BATCH_MS:           u128  = 2000;

// ── Core init ─────────────────────────────────────────────────────────────────

pub async fn core() -> CliResult<Lb> {
    Lb::init(Config::cli_config("cli"))
        .await
        .map_err(|err| CliError::from(err.to_string()))
}

// ── Build a CUDA session — no intra_threads, let CUDA own the execution ───────
//
// Key insight: with_intra_threads() is a CPU parallelism hint. Setting it
// while using CUDA can cause ONNX Runtime to route ops back to CPU kernels.
// For GPU sessions we set inter_op_threads=1 and let CUDA handle everything.
//
fn build_cuda_session(model_path: &Path) -> Result<Session, String> {
    let builder = Session::builder()
        .map_err(|e| format!("Session builder error: {}", e))?;

    let cuda = CUDAExecutionProvider::default().build();
    let builder = builder
        .with_execution_providers([cuda])
        .map_err(|e| format!("CUDA provider error: {}", e))?;

    // Do NOT call with_intra_threads for GPU sessions —
    // it interferes with CUDA kernel dispatch.
    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("Opt level error: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("Model load error: {}", e))
}

// ── Build a CPU session — use all threads for parallelism ─────────────────────
fn build_cpu_session(model_path: &Path) -> CliResult<Session> {
    Session::builder()
        .map_err(|e| CliError::from(format!("Session builder error: {}", e)))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| CliError::from(format!("Opt level error: {}", e)))?
        .with_intra_threads(8)
        .map_err(|e| CliError::from(format!("Thread error: {}", e)))?
        .commit_from_file(model_path)
        .map_err(|e| CliError::from(format!("Model load error: {}", e)))
}

// ── Session builder used for reranker (small model, GPU fine with threads) ────
fn build_session(model_path: &Path) -> CliResult<Session> {
    let builder = Session::builder()
        .map_err(|e| CliError::from(format!("Session builder error: {}", e)))?;

    let cuda    = CUDAExecutionProvider::default().build();
    let builder = match builder.with_execution_providers([cuda]) {
        Ok(b)  => { println!("  ✓ GPU acceleration available"); b }
        Err(_) => {
            println!("  ⚠ No GPU, using CPU");
            Session::builder()
                .map_err(|e| CliError::from(format!("Session builder error: {}", e)))?
        }
    };

    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| CliError::from(format!("Opt level error: {}", e)))?
        .with_intra_threads(4)
        .map_err(|e| CliError::from(format!("Thread error: {}", e)))?
        .commit_from_file(model_path)
        .map_err(|e| CliError::from(format!("Model load error: {}", e)))
}

// ── GPU probe + embedder loader ───────────────────────────────────────────────
//
// Tries to build a pure CUDA session (no intra_threads), runs a real-size
// batch to confirm GPU is being used, returns the session if fast enough.
// Falls back to CPU session if GPU fails or is suspiciously slow.
//
fn load_embedder(model_dir: &Path) -> CliResult<(Session, bool)> {
    let model_path = model_dir
        .join("multilingual-e5-large")
        .join("model.onnx");

    println!("🔬 Probing GPU / cuDNN...");

    // Try building a CUDA session
    let cuda_session = build_cuda_session(&model_path);

    let mut session = match cuda_session {
        Err(e) => {
            println!("  ✗ CUDA session failed: {}", e);
            println!("  → Falling back to CPU");
            let s = build_cpu_session(&model_path)?;
            return Ok((s, false));
        }
        Ok(s) => s,
    };

    // Run a real-size batch — same shape as actual indexing work.
    // batch=8 is small enough to be fast but large enough to trigger
    // any GPU/CPU routing decision the runtime makes.
    let test_batch = 8usize;
    let dummy_ids  = vec![0i64; test_batch * BI_MAX_LEN];
    let dummy_mask = vec![1i64; test_batch * BI_MAX_LEN];

    let probe_start = std::time::Instant::now();

    let result = session.run(ort::inputs![
        "input_ids"      => Value::from_array(([test_batch, BI_MAX_LEN], dummy_ids)).unwrap(),
        "attention_mask" => Value::from_array(([test_batch, BI_MAX_LEN], dummy_mask)).unwrap(),
    ]);

    let ms = probe_start.elapsed().as_millis();
    let probe_err = result.err().map(|e| e.to_string());

    if let Some(e) = probe_err {
        println!("  ✗ GPU inference error: {}", e);
        println!("  → Falling back to CPU");
        let s = build_cpu_session(&model_path)?;
        return Ok((s, false));
    }

    if ms > 3000 {
        // 3s+ for a batch of 8 = definitely CPU, not GPU
        println!("  ✗ GPU probe took {}ms — CUDA session fell back to CPU", ms);
        println!("  → Building explicit CPU session instead");
        let s = build_cpu_session(&model_path)?;
        return Ok((s, false));
    }

    println!("  ✓ GPU confirmed ({}ms for batch of {}) — using CUDA session", ms, test_batch);
    Ok((session, true))
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn run() -> CliResult<()> {
    Command::name("lockbook")
        .description("The private, polished note-taking platform.")
        .version(env!("CARGO_PKG_VERSION"))
        .subcommand(
            Command::name("account")
                .description("account management commands")
                .subcommand(
                    Command::name("new")
                        .input(Arg::str("username").description("your desired username."))
                        .input(Flag::<ApiUrl>::new("api_url")
                            .description("location of the lockbook server you're trying to use. If not provided will check the API_URL env var, and then fall back to https://api.prod.lockbook.net"))
                        .handler(|username, api_url| {
                            account::new(username.get(), api_url.get())
                        })
                )
                .subcommand(
                    Command::name("import").description("import an existing account by piping in the account string")
                        .handler(account::import)
                )
                .subcommand(
                    Command::name("export").description("reveal your account's private key")
                        .input(Flag::bool("skip-check").description("don't ask for confirmation to reveal the private key"))
                        .handler(|skip_check| account::export(skip_check.get()))
                )
                .subcommand(
                    Command::name("subscribe").description("start a monthly subscription for massively increased storage")
                        .handler(account::subscribe)
                )
                .subcommand(
                    Command::name("unsubscribe").description("cancel an existing subscription")
                        .handler(account::unsubscribe)
                )
                .subcommand(
                    Command::name("status").description("show your account status")
                        .handler(account::status)
                )
        )
        .subcommand(
            Command::name("copy").description("import files from your file system into lockbook")
                .input(Arg::<PathBuf>::name("disk-path").description("path of file on disk"))
                .input(Arg::<FileInput>::name("dest")
                       .description("the path or id of a folder within lockbook to place the file.")
                       .completor(|prompt| input::file_completor(prompt, Some(Filter::FoldersOnly))))
                .handler(|disk, parent| imex::copy(disk.get(), parent.get()))
        )
        .subcommand(
            Command::name("debug").description("investigative commands")
                .subcommand(
                    Command::name("validate").description("helps find invalid states within your lockbook")
                        .handler(debug::validate)
                )
                .subcommand(
                    Command::name("info").description("print metadata associated with a file")
                        .input(Arg::<FileInput>::name("target").description("id or path of file to debug")
                            .completor(|prompt| input::file_completor(prompt, None)))
                        .handler(|target| debug::info(target.get()))
                )
                .subcommand(
                    Command::name("whoami").description("print who is logged into this lockbook")
                        .handler(debug::whoami)
                )
                .subcommand(
                    Command::name("whereami").description("print information about where this lockbook is stored and it's server url")
                        .handler(debug::whereami)
                )
                .subcommand(
                    Command::name("debuginfo").description("retrieve the debug-info string to help a lockbook engineer diagnose a problem")
                        .handler(debug::debug_info)
                )
        )
        .subcommand(
            Command::name("delete").description("delete a file")
                .input(Flag::bool("force"))
                .input(Arg::<FileInput>::name("target").description("path of id of file to delete")
                            .completor(|prompt| input::file_completor(prompt, None)))
                .handler(|force, target| {
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        delete(force.get(), target.get()).await
                    })
                })
        )
        .subcommand(
            Command::name("edit").description("edit a document")
                .input(edit::editor_flag())
                .input(Arg::<FileInput>::name("target").description("path or id of file to edit")
                            .completor(|prompt| input::file_completor(prompt, None)))
                .handler(|editor, target| edit::edit(editor.get(), target.get()))
        )
        .subcommand(
            Command::name("export").description("export a lockbook file to your file system")
                .input(Arg::<FileInput>::name("target")
                            .completor(|prompt| input::file_completor(prompt, None)))
                .input(Arg::<PathBuf>::name("dest"))
                .handler(|target, dest| imex::export(target.get(), dest.get()))
        )
        .subcommand(
            Command::name("fs")
                .description("use your lockbook files with your local filesystem by mounting an NFS drive to /tmp/lockbook")
                .handler(lb_fs::mount)
        )
        .subcommand(
            Command::name("list").description("list files and file information")
                .input(Flag::bool("long").description("'long listing format': displays id and sharee information in table format"))
                .input(Flag::bool("recursive").description("include all children of the given directory, recursively. Implicitly enables --paths"))
                .input(Flag::bool("paths").description("display the full path of any children"))
                .input(Arg::<FileInput>::name("target").description("file path location whose files will be listed")
                            .completor(|prompt| input::file_completor(prompt, Some(Filter::FoldersOnly)))
                            .default(FileInput::Path("/".to_string())))
                .handler(|long, recur, paths, target| list::list(long.get(), recur.get(), paths.get(), target.get()))
        )
        .subcommand(
            Command::name("move").description("move a file to a new parent")
                .input(Arg::<FileInput>::name("src-target").description("lockbook file path or ID of the file to move")
                            .completor(|prompt| input::file_completor(prompt, None)))
                .input(Arg::<FileInput>::name("dest").description("lockbook file path or ID of the new parent folder")
                            .completor(|prompt| input::file_completor(prompt, Some(Filter::FoldersOnly))))
                .handler(|src, dst| {
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        move_file(src.get(), dst.get()).await
                    })
                })
        )
        .subcommand(
            Command::name("new").description("create a new file at the given path or do nothing if it exists")
                .input(Arg::<FileInput>::name("path").description("create a new file at the given path or do nothing if it exists")
                            .completor(|prompt| input::file_completor(prompt, Some(Filter::FoldersOnly))))
                .handler(|target| {
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        create_file(target.get()).await
                    })
                })
        )
        .subcommand(
            Command::name("stream").description("interact with stdout and stdin")
                .subcommand(
                    Command::name("out")
                        .description("print a document to stdout")
                        .input(Arg::<FileInput>::name("target").description("lockbook file path or ID")
                            .completor(|prompt| input::file_completor(prompt, None)))
                        .handler(|target| stream::stdout(target.get()))
                )
                .subcommand(
                    Command::name("in")
                        .description("write stdin to a document")
                        .input(Arg::<FileInput>::name("target").description("lockbook file path or ID")
                            .completor(|prompt| input::file_completor(prompt, None)))
                        .input(Flag::bool("append").description("don't overwrite the specified lb file, append to it"))
                        .handler(|target, append| stream::stdin(target.get(), append.get()))
                )
        )
        .subcommand(
            Command::name("rename").description("rename a file")
                .input(Arg::<FileInput>::name("target").description("lockbook file path or ID of file to rename")
                            .completor(|prompt| input::file_completor(prompt, None)))
                .input(Arg::str("new_name"))
                .handler(|target, new_name| {
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        rename(target.get(), new_name.get()).await
                    })
                })
        )
        .subcommand(
            Command::name("share").description("sharing related commands")
                .subcommand(
                    Command::name("new").description("share a file with someone")
                        .input(Arg::<FileInput>::name("target").description("lockbook file path or ID of file to rename")
                            .completor(|prompt| input::file_completor(prompt, None)))
                        .input(Arg::str("username")
                            .completor(input::username_completor))
                        .input(Flag::bool("read-only"))
                        .handler(|target, username, ro| share::new(target.get(), username.get(), ro.get()))
                )
                .subcommand(
                    Command::name("pending").description("list pending shares")
                        .handler(share::pending)
                )
                .subcommand(
                    Command::name("accept").description("accept a pending share by adding it to your file tree")
                        .input(Arg::<Uuid>::name("pending-share-id").description("ID of pending share")
                                    .completor(share::pending_share_completor))
                        .input(Arg::<FileInput>::name("target").description("lockbook file path or ID of the folder you want to place this shared file")
                            .completor(|prompt| input::file_completor(prompt, Some(Filter::FoldersOnly))))
                        .handler(|id, dest| share::accept(&id.get(), dest.get()))
                )
                .subcommand(
                    Command::name("delete").description("delete a pending share")
                        .input(Arg::<Uuid>::name("share-id").description("ID of pending share to delete")
                               .completor(share::pending_share_completor))
                        .handler(|target| share::delete(target.get()))
                )
        )
        .subcommand(
            Command::name("search").description("search document contents")
                .subcommand(
                    Command::name("semantic").description("semantic search using neural embeddings")
                        .input(Arg::str("query"))
                        .handler(|query| {
                            tokio::runtime::Runtime::new().unwrap().block_on(async {
                                search_semantic(&query.get()).await
                            })
                        })
                )
        )
        .subcommand(
            Command::name("migrate-from").description("transfer files from an existing platform")
                .subcommand(
                    Command::name("bear").description("migrate your files from https://bear.app/ Export as md and using the 'export attachments' option.")
                        .input(Arg::<PathBuf>::name("disk-path").description("location of a bear export of files."))
                        .handler(|path| migrate::bear(path.get()))
                )
        )
        .subcommand(
            Command::name("sync").description("sync your local changes back to lockbook servers")
                .handler(|| {
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        sync().await
                    })
                })
        )
        .with_completions()
        .parse()
}

// ── Semantic search ───────────────────────────────────────────────────────────

pub async fn search_semantic(query: &str) -> CliResult<()> {
    let start_time = std::time::Instant::now();
    println!("{}", "🔍 Semantic Search".cyan().bold());
    println!("Query: {}\n", query);

    let lb = &core().await?;
    ensure_account_and_root(lb).await?;

    let base       = PathBuf::from(&lb.config.writeable_path).join("search");
    let model_dir  = base.join("models");
    let index_path = base.join("index.json");

    // ── Step 1: Ensure models are downloaded ──────────────────────────────────
    ensure_models_downloaded(&model_dir).await?;

    // ── Step 2: Load embedder (GPU if available, CPU fallback) ────────────────
    println!("🧠 Loading embedder...");
    let load_start = std::time::Instant::now();

    let (mut bi_session, using_gpu) = load_embedder(&model_dir)
    .map_err(|e| CliError::from(e))?;

    println!("  ✓ Embedder ready in {:.2}s ({})",
        load_start.elapsed().as_secs_f32(),
        if using_gpu { "GPU" } else { "CPU — will be slow" }
    );

    if !using_gpu {
        println!("  ⚠ GPU not being used. Press Enter to continue on CPU, or Ctrl+C to abort.");
        let mut inp = String::new();
        std::io::stdin().read_line(&mut inp).ok();
    }

    let bi_tokenizer = Tokenizer::from_file(
        &model_dir.join("multilingual-e5-large").join("tokenizer.json"),
    ).map_err(|e| CliError::from(format!("Tokenizer error: {}", e)))?;

    // ── Step 3: Scan files + read HMACs ──────────────────────────────────────
    println!("📁 Scanning files...");

    let all_files = lb
        .get_and_get_children_recursively(&lb.root().await?.id)
        .await?;

    struct ScannedFile {
        id:      Uuid,
        path:    String,
        name:    String,
        content: String,
        hmac:    Vec<u8>,
    }

    let mut scanned: Vec<ScannedFile> = Vec::new();

    for file in &all_files {
        if file.is_folder() {
            continue;
        }
        let lower = file.name.to_lowercase();
        if !lower.ends_with(".md") && !lower.ends_with(".txt") {
            continue;
        }

        let (hmac_opt, raw) = match lb.read_document_with_hmac(file.id, false).await {
            Ok(pair) => pair,
            Err(_)   => continue,
        };

        let content = String::from_utf8_lossy(&raw).to_string();
        if content.trim().len() < 20 {
            continue;
        }

        let path = lb
            .get_path_by_id(file.id)
            .await
            .unwrap_or_else(|_| file.name.clone());

        let name = Path::new(&file.name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&file.name)
            .replace(['-', '_'], " ");

        let hmac = hmac_opt.map(|h| h.to_vec()).unwrap_or_default();

        scanned.push(ScannedFile { id: file.id, path, name, content, hmac });
    }

    if scanned.is_empty() {
        println!("No documents found. Add some .txt or .md files first.");
        return Ok(());
    }

    println!("  Found {} indexable file(s)", scanned.len());

    // ── Step 4: Load existing index ───────────────────────────────────────────
    let mut index: Index = if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path).unwrap_or_default();
        serde_json::from_str(&raw).unwrap_or_default()
    } else {
        HashMap::new()
    };

    // ── Step 5: Determine which files need (re)indexing ───────────────────────
    let current_ids: HashSet<String> = scanned.iter().map(|f| f.id.to_string()).collect();
    index.retain(|id, _| current_ids.contains(id));

    let needs_index: Vec<&ScannedFile> = scanned
        .iter()
        .filter(|f| match index.get(&f.id.to_string()) {
            None        => true,
            Some(entry) => entry.hmac != f.hmac,
        })
        .collect();

    // ── Step 6: Embed changed/new files ──────────────────────────────────────
    if !needs_index.is_empty() {
        println!("📚 Indexing {} file(s)...", needs_index.len());

        std::fs::create_dir_all(&base)
            .map_err(|e| CliError::from(format!("Cannot create search dir: {}", e)))?;

        for (file_idx, sf) in needs_index.iter().enumerate() {
            // ── Progress bar ─────────────────────────────────────────────────
            {
                use std::io::Write;
                let pct     = (file_idx as f32 / needs_index.len() as f32 * 100.0) as usize;
                let bar_len = pct / 2;
                let bar     = "█".repeat(bar_len);
                let empty   = "░".repeat(50usize.saturating_sub(bar_len));
                let short   = if sf.name.len() > 28 {
                    format!("{}...", &sf.name[..28])
                } else {
                    sf.name.clone()
                };
                print!(
                    "\r\x1B[K  [{bar}{empty}] {}/{} ({pct}%) - {short}",
                    file_idx + 1,
                    needs_index.len(),
                );
                let _ = std::io::stdout().flush();
            }

            let file_start = std::time::Instant::now();

            // ── Chunking ─────────────────────────────────────────────────────
            let sections = split_markdown_sections(&sf.content);
            let mut chunks: Vec<ChunkRecord> = Vec::new();

            'chunking: for section in &sections {
                let parent_chunks =
                    chunk_text(section, PARENT_CHARS, PARENT_OVERLAP, MAX_CHUNKS_PER_FILE);
                for parent in &parent_chunks {
                    let child_chunks =
                        chunk_text(parent, CHILD_CHARS, CHILD_OVERLAP, MAX_CHUNKS_PER_FILE);
                    for child in child_chunks {
                        if child.trim().is_empty() {
                            continue;
                        }
                        chunks.push(ChunkRecord {
                            file_path:    sf.path.clone(),
                            file_name:    sf.name.clone(),
                            parent_chunk: parent.clone(),
                            child_text:   child,
                        });
                        if chunks.len() >= MAX_CHUNKS_FOR_INDEXING {
                            eprintln!(
                                "\n  ⚠ '{}' hit chunk cap ({}) — truncating",
                                sf.name, MAX_CHUNKS_FOR_INDEXING
                            );
                            break 'chunking;
                        }
                    }
                }
            }

            if chunks.is_empty() {
                continue;
            }

            if chunks.len() > 1000 {
                eprintln!(
                    "\n  ℹ Large file: '{}' — {} chunks (~{} batches)",
                    sf.name,
                    chunks.len(),
                    (chunks.len() + BATCH_SIZE - 1) / BATCH_SIZE
                );
            }

            // File-name anchor chunk
            let intro: String = sf.content.chars().take(200).collect();
            chunks.push(ChunkRecord {
                file_path:    sf.path.clone(),
                file_name:    sf.name.clone(),
                parent_chunk: format!("{}\n{}", sf.name, intro),
                child_text:   sf.name.clone(),
            });

            // ── Pre-format all texts, then batch-embed ────────────────────────
            let all_texts: Vec<String> = chunks
                .iter()
                .map(|c| format!("passage: {} | {}", c.file_name, c.child_text))
                .collect();

            let total_batches = (all_texts.len() + BATCH_SIZE - 1) / BATCH_SIZE;
            let mut vectors: Vec<f32> = Vec::with_capacity(chunks.len() * BI_HIDDEN);
            let mut embed_failed        = false;
            let mut cpu_fallback_warned = false;

            for (batch_idx, batch_texts) in all_texts.chunks(BATCH_SIZE).enumerate() {
                if file_start.elapsed().as_secs() > PER_FILE_TIMEOUT_SECS {
                    eprintln!(
                        "\n  ⚠ '{}' timed out after {}s — partial index saved",
                        sf.name, PER_FILE_TIMEOUT_SECS
                    );
                    embed_failed = true;
                    break;
                }

                let batch_start = std::time::Instant::now();

                match embed_batch(&mut bi_session, &bi_tokenizer, batch_texts, BI_HIDDEN, BI_MAX_LEN) {
                    Ok(vecs) => {
                        let batch_ms = batch_start.elapsed().as_millis();

                        if batch_ms > SLOW_BATCH_MS && !cpu_fallback_warned {
                            eprintln!(
                                "\n  ⚠ Batch {}/{} took {}ms — GPU not being used!",
                                batch_idx + 1, total_batches, batch_ms
                            );
                            cpu_fallback_warned = true;
                        }

                        for v in vecs {
                            vectors.extend_from_slice(&v);
                        }
                    }
                    Err(e) => {
                        eprintln!("\n  ⚠ Embed failed for '{}': {:?}", sf.name, e);
                        embed_failed = true;
                        break;
                    }
                }
            }

            if embed_failed {
                continue;
            }

            let elapsed = file_start.elapsed().as_secs_f32();
            let chunks_per_sec = chunks.len() as f32 / elapsed;
            eprintln!(
                "\n  ✓ '{}' done in {:.1}s ({:.0} chunks/s)",
                sf.name, elapsed, chunks_per_sec
            );

            // ── Insert + incremental save ─────────────────────────────────────
            index.insert(
                sf.id.to_string(),
                IndexEntry { hmac: sf.hmac.clone(), chunks, vectors },
            );

            let json = serde_json::to_string(&index)
                .map_err(|e| CliError::from(format!("Cannot serialize index: {}", e)))?;
            std::fs::write(&index_path, &json)
                .map_err(|e| CliError::from(format!("Cannot write index: {}", e)))?;
        }

        println!();
        let total_chunks: usize = index.values().map(|e| e.chunks.len()).sum();
        println!("  ✓ Indexing complete ({} total chunks)", total_chunks);
    } else {
        let total_chunks: usize = index.values().map(|e| e.chunks.len()).sum();
        println!("📖 Index up to date ({} chunks across {} files)",
            total_chunks, index.len());
    }

    // ── Step 7: Load reranker ─────────────────────────────────────────────────
    println!("🧠 Loading reranker...");
    let reranker_start = std::time::Instant::now();

    let reranker_dir = model_dir.join("ms-marco-MiniLM-L-6-v2");
    let mut reranker_session = build_session(&reranker_dir.join("model.onnx"))?;
    let reranker_tokenizer = Tokenizer::from_file(&reranker_dir.join("tokenizer.json"))
        .map_err(|e| CliError::from(format!("Reranker tokenizer error: {}", e)))?;

    println!("  ✓ Reranker loaded in {:.2}s", reranker_start.elapsed().as_secs_f32());

    // ── Step 8: Flatten index into searchable slices ──────────────────────────
    let mut all_chunks:  Vec<&ChunkRecord> = Vec::new();
    let mut all_vectors: Vec<f32>          = Vec::new();

    for entry in index.values() {
        for (i, chunk) in entry.chunks.iter().enumerate() {
            let base = i * BI_HIDDEN;
            if base + BI_HIDDEN <= entry.vectors.len() {
                all_chunks.push(chunk);
                all_vectors.extend_from_slice(&entry.vectors[base..base + BI_HIDDEN]);
            }
        }
    }

    let num_chunks = all_chunks.len();
    if num_chunks == 0 {
        println!("No chunks in index.");
        return Ok(());
    }

    println!("🔎 Searching {} chunks...", num_chunks);

    // ── Step 9: Embed query ───────────────────────────────────────────────────
    let query_vec = embed_single(
        &mut bi_session, &bi_tokenizer, query, "query", BI_HIDDEN, BI_MAX_LEN,
    )?;

    // ── Step 10: Bi-encoder scoring ───────────────────────────────────────────
    let mut scores: Vec<(usize, f32)> = (0..num_chunks)
        .map(|i| {
            let base = i * BI_HIDDEN;
            let sim: f32 = (0..BI_HIDDEN)
                .map(|j| query_vec[j] * all_vectors[base + j])
                .sum();
            (i, sim)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let candidates: Vec<(usize, f32)> = scores.into_iter().take(RERANK_K).collect();

    // ── Step 11: Rerank ───────────────────────────────────────────────────────
    println!("  ✓ Reranking {} candidates...", candidates.len());
    let rerank_start = std::time::Instant::now();

    let mut reranked: Vec<(String, String, f32)> = Vec::new();

    for (idx, _) in &candidates {
        let chunk = all_chunks[*idx];
        let score = rerank_pair(
            &mut reranker_session,
            &reranker_tokenizer,
            query,
            &chunk.parent_chunk,
            RERANKER_MAX_LEN,
        )?;
        reranked.push((chunk.file_path.clone(), chunk.child_text.clone(), score));
    }

    reranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    println!("  ✓ Reranking done in {:.2}s", rerank_start.elapsed().as_secs_f32());

    // ── Step 12: Deduplicate by file ──────────────────────────────────────────
    let mut seen:    HashSet<String>             = HashSet::new();
    let mut results: Vec<(String, String, f32)> = Vec::new();

    for (path, snippet, score) in &reranked {
        if seen.insert(path.clone()) {
            results.push((path.clone(), snippet.clone(), *score));
        }
        if results.len() >= TOP_K {
            break;
        }
    }

    // ── Step 13: Display ──────────────────────────────────────────────────────
    if results.is_empty() {
        println!("\n{}", "No results found.".yellow());
    } else {
        println!("\n{}", "Top Results:".green().bold());
        println!("{}", "============".green());

        for (i, (path, snippet, score)) in results.iter().enumerate() {
            let sig           = 1.0f32 / (1.0 + (-score).exp());
            let score_percent = (sig * 100.0).clamp(0.0, 100.0) as usize;
            let bar_len       = score_percent / 2;
            let bar           = "█".repeat(bar_len);
            let empty         = "░".repeat(50usize.saturating_sub(bar_len));

            let score_colored = if sig > 0.7 {
                format!("{:.4}", sig).green()
            } else if sig > 0.4 {
                format!("{:.4}", sig).yellow()
            } else {
                format!("{:.4}", sig).red()
            };

            let preview: String = snippet.chars().take(150).collect();
            let preview = preview.trim().replace('\n', " ");

            println!("\n{}. {}", i + 1, path.cyan().bold());
            println!("   Score (logit={:.4}, sigmoid={} {:.1}%)",
                score, score_colored, score_percent);
            println!("   Relevance: [{}{}]", bar, empty);
            println!("   ↳ \"{}...\"", preview.dimmed());
        }
    }

    println!("\n✅ Search completed in {:.2}s", start_time.elapsed().as_secs_f32());
    Ok(())
}

// ── Model downloader ──────────────────────────────────────────────────────────
async fn ensure_models_downloaded(model_dir: &Path) -> CliResult<()> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| CliError::from(format!("HTTP client error: {}", e)))?;

    for model in MODELS {
        let local_dir = model_dir.join(model.name);
        std::fs::create_dir_all(&local_dir)
            .map_err(|e| CliError::from(format!("Cannot create model dir: {}", e)))?;

        for file in model.files {
            let dest = local_dir.join(file.filename);
            if dest.exists() {
                continue;
            }

            let url = if file.subfolder.is_empty() {
                format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    file.repo_id, file.filename
                )
            } else {
                format!(
                    "https://huggingface.co/{}/resolve/main/{}/{}",
                    file.repo_id, file.subfolder, file.filename
                )
            };

            println!("📥 Downloading {}/{}...", model.name, file.filename);

            let resp = client
                .get(&url)
                .send()
                .await
                .map_err(|e| CliError::from(format!("Request failed: {}", e)))?;

            if !resp.status().is_success() {
                return Err(CliError::from(format!(
                    "HTTP {} fetching {}/{}",
                    resp.status().as_u16(),
                    model.name,
                    file.filename
                )));
            }

            let bytes = resp
                .bytes()
                .await
                .map_err(|e| CliError::from(format!("Read failed: {}", e)))?;

            std::fs::write(&dest, &bytes)
                .map_err(|e| CliError::from(format!("Write failed: {}", e)))?;

            println!(
                "  ✓ {}/{} ({:.1} MB)",
                model.name,
                file.filename,
                bytes.len() as f64 / 1_048_576.0
            );
        }
    }

    Ok(())
}

// ── Markdown-aware section splitter ──────────────────────────────────────────
fn split_markdown_sections(text: &str) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if line.starts_with('#') && !current.trim().is_empty() {
            sections.push(current.trim().to_string());
            current = String::new();
        }
        current.push_str(line);
        current.push('\n');
    }

    if !current.trim().is_empty() {
        sections.push(current.trim().to_string());
    }

    if sections.is_empty() {
        sections.push(text.trim().to_string());
    }

    sections
}

// ── Reranker: score (query, passage) pair ────────────────────────────────────
fn rerank_pair(
    session:   &mut Session,
    tokenizer: &Tokenizer,
    query:     &str,
    passage:   &str,
    max_len:   usize,
) -> CliResult<f32> {
    let enc = tokenizer
        .encode((query, passage), true)
        .map_err(|e| CliError::from(format!("Reranker tokenization failed: {}", e)))?;

    let ids   = enc.get_ids();
    let mask  = enc.get_attention_mask();
    let types = enc.get_type_ids();
    let len   = ids.len().min(max_len);

    let mut padded_ids   = vec![0i64; max_len];
    let mut padded_mask  = vec![0i64; max_len];
    let mut padded_types = vec![0i64; max_len];

    for i in 0..len {
        padded_ids[i]   = ids[i]   as i64;
        padded_mask[i]  = mask[i]  as i64;
        padded_types[i] = types[i] as i64;
    }

    let outputs = session
        .run(ort::inputs![
            "input_ids"      => Value::from_array(([1usize, max_len], padded_ids))
                .map_err(|e| CliError::from(format!("Tensor error: {}", e)))?,
            "attention_mask" => Value::from_array(([1usize, max_len], padded_mask))
                .map_err(|e| CliError::from(format!("Tensor error: {}", e)))?,
            "token_type_ids" => Value::from_array(([1usize, max_len], padded_types))
                .map_err(|e| CliError::from(format!("Tensor error: {}", e)))?,
        ])
        .map_err(|e| CliError::from(format!("Reranker inference failed: {}", e)))?;

    let (_, logits) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| CliError::from(format!("Reranker extract failed: {}", e)))?;

    Ok(logits[0])
}

// ── Bi-encoder: embed a batch ─────────────────────────────────────────────────
fn embed_batch(
    session:   &mut Session,
    tokenizer: &Tokenizer,
    texts:     &[String],
    hidden:    usize,
    max_len:   usize,
) -> CliResult<Vec<Vec<f32>>> {
    let batch_size = texts.len();
    if batch_size == 0 {
        return Ok(vec![]);
    }

    let mut all_ids  = vec![0i64; batch_size * max_len];
    let mut all_mask = vec![0i64; batch_size * max_len];
    let mut lengths  = vec![0usize; batch_size];

    for (b, text) in texts.iter().enumerate() {
        let enc = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| CliError::from(format!("Tokenization failed: {}", e)))?;

        let ids  = enc.get_ids();
        let mask = enc.get_attention_mask();
        let len  = ids.len().min(max_len);
        lengths[b] = len;

        let offset = b * max_len;
        for i in 0..len {
            all_ids[offset + i]  = ids[i]  as i64;
            all_mask[offset + i] = mask[i] as i64;
        }
    }

    let outputs = session
        .run(ort::inputs![
            "input_ids"      => Value::from_array(([batch_size, max_len], all_ids))
                .map_err(|e| CliError::from(format!("Tensor error: {}", e)))?,
            "attention_mask" => Value::from_array(([batch_size, max_len], all_mask))
                .map_err(|e| CliError::from(format!("Tensor error: {}", e)))?,
        ])
        .map_err(|e| CliError::from(format!("Inference failed: {}", e)))?;

    let (_, emb) = outputs["last_hidden_state"]
        .try_extract_tensor::<f32>()
        .map_err(|e| CliError::from(format!("Extract failed: {}", e)))?;

    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let len = lengths[b];
        let mut pooled = vec![0.0f32; hidden];

        for i in 0..len {
            let base = b * max_len * hidden + i * hidden;
            for j in 0..hidden {
                pooled[j] += emb[base + j];
            }
        }
        if len > 0 {
            for v in &mut pooled {
                *v /= len as f32;
            }
        }

        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut pooled {
                *v /= norm;
            }
        }

        results.push(pooled);
    }

    Ok(results)
}

// ── Bi-encoder: embed a single query ─────────────────────────────────────────
fn embed_single(
    session:   &mut Session,
    tokenizer: &Tokenizer,
    text:      &str,
    prefix:    &str,
    hidden:    usize,
    max_len:   usize,
) -> CliResult<Vec<f32>> {
    let prefixed = format!("{}: {}", prefix, text);
    let enc = tokenizer
        .encode(prefixed.as_str(), true)
        .map_err(|e| CliError::from(format!("Tokenization failed: {}", e)))?;

    let ids  = enc.get_ids();
    let mask = enc.get_attention_mask();
    let qlen = ids.len().min(max_len);

    let mut padded_ids  = vec![0i64; max_len];
    let mut padded_mask = vec![0i64; max_len];

    for i in 0..qlen {
        padded_ids[i]  = ids[i]  as i64;
        padded_mask[i] = mask[i] as i64;
    }

    let outputs = session
        .run(ort::inputs![
            "input_ids"      => Value::from_array(([1usize, max_len], padded_ids))
                .map_err(|e| CliError::from(format!("Tensor error: {}", e)))?,
            "attention_mask" => Value::from_array(([1usize, max_len], padded_mask))
                .map_err(|e| CliError::from(format!("Tensor error: {}", e)))?,
        ])
        .map_err(|e| CliError::from(format!("Inference failed: {}", e)))?;

    let (_, q_emb) = outputs["last_hidden_state"]
        .try_extract_tensor::<f32>()
        .map_err(|e| CliError::from(format!("Extract failed: {}", e)))?;

    let mut vec = vec![0.0f32; hidden];
    for i in 0..qlen {
        let base = i * hidden;
        for j in 0..hidden {
            vec[j] += q_emb[base + j];
        }
    }
    if qlen > 0 {
        for v in &mut vec {
            *v /= qlen as f32;
        }
    }

    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for v in &mut vec {
            *v /= norm;
        }
    }

    Ok(vec)
}

// ── Unicode-safe char-based chunking ─────────────────────────────────────────
fn chunk_text(text: &str, size: usize, overlap: usize, max_chunks: usize) -> Vec<String> {
    let chars: Vec<char> = text.trim().chars().collect();
    let total = chars.len();

    if total == 0 {
        return vec![];
    }
    if total <= size {
        return vec![chars.iter().collect()];
    }

    let mut chunks = Vec::new();
    let mut start  = 0usize;
    let step       = size.saturating_sub(overlap).max(1);

    while start < total && chunks.len() < max_chunks {
        let end   = (start + size).min(total);
        let chunk: String = chars[start..end].iter().collect();
        let chunk = chunk.trim().to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        if end >= total {
            break;
        }
        start += step;
    }

    chunks
}

// ── Other commands ────────────────────────────────────────────────────────────

async fn delete(force: bool, target: FileInput) -> Result<(), CliError> {
    let lb = &core().await?;
    ensure_account_and_root(lb).await?;

    let f = target.find(lb).await?;

    if !force {
        let mut phrase = format!("delete '{target}'");

        if f.is_folder() {
            let count = lb
                .get_and_get_children_recursively(&f.id)
                .await
                .unwrap_or_default()
                .len() as u64
                - 1;
            match count {
                0 => {}
                1 => phrase = format!("{phrase} and its 1 child"),
                _ => phrase = format!("{phrase} and its {count} children"),
            };
        }

        let answer: String = input::std_in(format!("are you sure you want to {phrase}? [y/n]: "))?;
        if answer != "y" && answer != "Y" {
            println!("aborted.");
            return Ok(());
        }
    }

    lb.delete(&f.id).await?;
    Ok(())
}

async fn move_file(src: FileInput, dest: FileInput) -> CliResult<()> {
    let lb = &core().await?;
    ensure_account_and_root(lb).await?;

    let src  = src.find(lb).await?;
    let dest = dest.find(lb).await?;
    lb.move_file(&src.id, &dest.id).await?;
    Ok(())
}

async fn create_file(path: FileInput) -> CliResult<()> {
    let lb = &core().await?;
    ensure_account_and_root(lb).await?;

    let FileInput::Path(path) = path else {
        return Err(CliError::from("cannot create a file using ids"));
    };

    match lb.get_by_path(&path).await {
        Ok(_f) => Ok(()),
        Err(err) => match err.kind {
            LbErrKind::FileNonexistent => match lb.create_at_path(&path).await {
                Ok(_f) => Ok(()),
                Err(err) => Err(err.into()),
            },
            _ => Err(err.into()),
        },
    }
}

async fn rename(target: FileInput, new_name: String) -> Result<(), CliError> {
    let lb = &core().await?;
    ensure_account_and_root(lb).await?;

    let id = target.find(lb).await?.id;
    lb.rename_file(&id, &new_name).await?;
    Ok(())
}

fn ensure_account(lb: &Lb) -> CliResult<()> {
    if let Err(e) = lb.get_account() {
        if e.kind == LbErrKind::AccountNonexistent {
            return Err(CliError::from("no account found, run lockbook account import"));
        }
    }
    Ok(())
}

async fn ensure_account_and_root(lb: &Lb) -> CliResult<()> {
    ensure_account(lb)?;
    if let Err(e) = lb.root().await {
        if e.kind == LbErrKind::RootNonexistent {
            return Err(CliError::from("no root found, have you synced yet?"));
        }
    }
    Ok(())
}

async fn sync() -> CliResult<()> {
    let lb = &core().await?;
    ensure_account_and_root(lb).await?;
    lb.sync(None).await?;
    println!("Sync complete!");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {:?}", e);
        std::process::exit(1);
    }
}