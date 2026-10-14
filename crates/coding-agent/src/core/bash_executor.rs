//! Bash command execution with streaming support and cancellation, ported
//! from upstream `src/core/bash-executor.ts`.
//!
//! The unified execution path `AgentSession.executeBash()` and the modes
//! calling bash directly run through. Output chunks pass the same
//! sanitize pipeline the tool uses — strip ANSI, replace binary garbage,
//! drop carriage returns — into a rolling buffer; once the raw stream
//! passes the truncation threshold the full output spools to a temp file.

use std::io::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::types::AgentToolError;

use crate::tools::bash::{BashExecOptions, BashOperations, OnDataListener};
use crate::tools::output_accumulator::StreamingDecoder;
use crate::tools::truncate::{DEFAULT_MAX_BYTES, TruncationOptions, truncate_tail};

/// The sanitized streaming chunk callback, upstream's `onChunk`.
pub type OnChunkListener = Arc<dyn Fn(&str) + Send + Sync>;

/// The executor options, upstream's `BashExecutorOptions`.
#[derive(Clone, Default)]
pub struct BashExecutorOptions {
    /// The callback for sanitized streaming output chunks.
    pub on_chunk: Option<OnChunkListener>,
    /// The abort signal cancelling the command.
    pub signal: Option<AbortSignal>,
}

impl std::fmt::Debug for BashExecutorOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashExecutorOptions")
            .field("on_chunk", &self.on_chunk.is_some())
            .field("signal", &self.signal)
            .finish()
    }
}

/// The execution outcome, upstream's `BashResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BashResult {
    /// The combined stdout + stderr output (sanitized, possibly truncated).
    pub output: String,
    /// The process exit code, absent when killed or cancelled.
    pub exit_code: Option<i32>,
    /// Whether the command was cancelled via the signal.
    pub cancelled: bool,
    /// Whether the output was truncated.
    pub truncated: bool,
    /// The temp file holding the full output, when the output exceeded the
    /// truncation threshold.
    pub full_output_path: Option<String>,
}

/// The UTF-16 code-unit length, upstream's `String.length` at the rolling
/// buffer's budget (the temp-file threshold uses real byte counts).
fn utf16_length(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

/// The executor's shared state, upstream's closure-captured locals.
#[derive(Default)]
struct ExecutorState {
    output_chunks: std::sync::Mutex<Vec<String>>,
    output_bytes: AtomicUsize,
    temp_file_path: std::sync::Mutex<Option<std::path::PathBuf>>,
    temp_file: std::sync::Mutex<Option<std::fs::File>>,
    total_bytes: AtomicUsize,
}

/// Open the spool and backfill the chunks collected before it opened,
/// upstream's `ensureTempFile`.
fn ensure_temp_file(state: &ExecutorState) -> std::io::Result<()> {
    let file = {
        let mut path_slot = state
            .temp_file_path
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if path_slot.is_some() {
            return Ok(());
        }
        let id = crate::tools::random_bytes_hex(8);
        let path = std::env::temp_dir().join(format!("pi-bash-{id}.log"));
        let mut file = std::fs::File::create(&path)?;
        for chunk in state
            .output_chunks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
        {
            file.write_all(chunk.as_bytes())?;
        }
        *path_slot = Some(path);
        file
    };
    *state
        .temp_file
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(file);
    Ok(())
}

/// Flush the spool to disk, upstream's awaited `tempFileStream.end()`.
fn flush_temp_file(state: &ExecutorState) -> std::io::Result<()> {
    if let Some(file) = state
        .temp_file
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_mut()
    {
        file.flush()?;
        file.sync_all()?;
    }
    Ok(())
}

/// The collected output, upstream's `outputChunks.join("")`.
fn full_output(state: &ExecutorState) -> String {
    state
        .output_chunks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .join("")
}

/// The recorded spool path, upstream's `tempFilePath`.
fn recorded_path(state: &ExecutorState) -> Option<String> {
    state
        .temp_file_path
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned())
}

/// Execute a bash command using custom [`BashOperations`], upstream's
/// `executeBashWithOperations`. Used for remote execution (SSH, containers,
/// etc.).
///
/// # Errors
/// The operations' own failure when the command was not cancelled, upstream
///'s rethrow.
#[expect(
    clippy::too_many_lines,
    reason = "the executor mirrors upstream's single body: the rolling buffer, the temp-file spool, and the two settle arms read the same state"
)]
pub async fn execute_bash_with_operations(
    command: &str,
    cwd: &str,
    operations: &BashOperations,
    options: Option<BashExecutorOptions>,
) -> Result<BashResult, AgentToolError> {
    let options = options.unwrap_or_default();
    let state = Arc::new(ExecutorState::default());
    let max_output_bytes = DEFAULT_MAX_BYTES * 2;

    // The chunk handler, upstream's `onData`: decode with the streaming
    // decoder, sanitize, spool past the threshold, bound the rolling
    // buffer, and stream to the callback.
    let on_data: OnDataListener = {
        let state = Arc::clone(&state);
        let on_chunk = options.on_chunk.clone();
        let decoder = Arc::new(std::sync::Mutex::new(StreamingDecoder::default()));
        Arc::new(move |data: &[u8]| {
            let text = decoder
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .decode(data);
            // Sanitize: strip ANSI, replace binary garbage, normalize
            // newlines.
            let text =
                crate::utils::shell::sanitize_binary_output(&crate::utils::ansi::strip_ansi(&text))
                    .replace('\r', "");

            state.total_bytes.fetch_add(data.len(), Ordering::SeqCst);
            // Start writing to temp file if the stream exceeds the threshold.
            if state.total_bytes.load(Ordering::SeqCst) > DEFAULT_MAX_BYTES {
                let _opened = ensure_temp_file(&state);
            }

            if let Some(file) = state
                .temp_file
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_mut()
            {
                let _written = file.write_all(text.as_bytes());
            }

            // Keep the rolling buffer bounded.
            let chunk_len = utf16_length(&text);
            let mut bytes = state.output_bytes.load(Ordering::SeqCst) + chunk_len;
            let mut chunks = state
                .output_chunks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            chunks.push(text.clone());
            while bytes > max_output_bytes && chunks.len() > 1 {
                let removed = chunks.remove(0);
                bytes -= utf16_length(&removed);
            }
            state.output_bytes.store(bytes, Ordering::SeqCst);
            drop(chunks);

            if let Some(on_chunk) = on_chunk.as_ref() {
                on_chunk(&text);
            }
        })
    };

    let outcome = (operations.exec)(
        command,
        cwd,
        BashExecOptions {
            on_data,
            signal: options.signal.clone(),
            timeout: None,
            env: None,
        },
    )
    .await;

    match outcome {
        Ok(settled) => {
            let full_output = full_output(&state);
            let truncation = truncate_tail(&full_output, TruncationOptions::default());
            if truncation.truncated {
                ensure_temp_file(&state).map_err(AgentToolError::from)?;
            }
            flush_temp_file(&state).map_err(AgentToolError::from)?;
            let cancelled = options.signal.as_ref().is_some_and(AbortSignal::aborted);
            Ok(BashResult {
                output: if truncation.truncated {
                    truncation.content
                } else {
                    full_output
                },
                exit_code: if cancelled { None } else { settled.exit_code },
                cancelled,
                truncated: truncation.truncated,
                full_output_path: recorded_path(&state),
            })
        }
        Err(error) => {
            if options.signal.as_ref().is_some_and(AbortSignal::aborted) {
                let full_output = full_output(&state);
                let truncation = truncate_tail(&full_output, TruncationOptions::default());
                if truncation.truncated {
                    ensure_temp_file(&state).map_err(AgentToolError::from)?;
                }
                let _flushed = flush_temp_file(&state);
                return Ok(BashResult {
                    output: if truncation.truncated {
                        truncation.content
                    } else {
                        full_output
                    },
                    exit_code: None,
                    cancelled: true,
                    truncated: truncation.truncated,
                    full_output_path: recorded_path(&state),
                });
            }
            Err(error)
        }
    }
}
