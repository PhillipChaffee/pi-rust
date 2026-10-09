//! Serializes file mutations targeting the same file, ported from upstream
//! `src/core/tools/file-mutation-queue.ts`.
//!
//! The coding-agent tool surface; the harness's env-keyed restatement lives
//! in `pi_agent_core::harness::tools::file_mutation_queue`.
//!
//! Upstream keys a process-global `Map` by the file's `realpath`, falling
//! back to the resolved absolute path while the file does not exist, and
//! serializes key computation through one registration chain. Rust restate:
//! a process-global registry of per-path `Notify` tails behind one
//! registration lock — each caller registers its own `Notify` as the path's
//! tail and waits on the tail it displaced, the exact hand-off shape
//! upstream's promise chain builds, with the release running from a drop
//! guard so aborts and panics release the chain like upstream's `finally`.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::{Arc, Mutex};

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::types::AgentToolError;

use tokio::sync::Notify;

/// The process-global registry, upstream's `fileMutationQueues`.
fn registry() -> &'static Mutex<HashMap<String, Arc<Notify>>> {
    static REGISTRY: std::sync::OnceLock<Mutex<HashMap<String, Arc<Notify>>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The registration lock serializing key computation and tail registration,
/// upstream's `registrationQueue` promise chain.
fn registration_lock() -> &'static tokio::sync::Mutex<()> {
    static REGISTRATION: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    REGISTRATION.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Resolve a path against the process cwd and lexically normalize it,
/// upstream's `resolve(filePath)`.
fn resolve_for_key(path: &str) -> String {
    let joined = if Path::new(path).is_absolute() {
        path.to_owned()
    } else {
        let cwd = std::env::current_dir()
            .map_or_else(|_| String::new(), |dir| dir.to_string_lossy().into_owned());
        format!("{cwd}/{path}")
    };
    crate::utils::paths::normalize_path(
        &joined,
        &crate::utils::paths::PathInputOptions {
            expand_tilde: false,
            ..crate::utils::paths::PathInputOptions::default()
        },
    )
    .unwrap_or(joined)
}

/// The canonical-path queue key, upstream's `getMutationQueueKey`: the
/// file's realpath; when the file is missing (`ENOENT`/`ENOTDIR`) the
/// resolved absolute path.
///
/// # Errors
/// Any other canonicalization failure, upstream's rethrow.
async fn mutation_queue_key(path: &str) -> Result<String, AgentToolError> {
    let resolved_path = resolve_for_key(path);
    match tokio::fs::canonicalize(&resolved_path).await {
        Ok(real_path) => Ok(real_path.to_string_lossy().into_owned()),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            Ok(resolved_path)
        }
        Err(error) => Err(Box::new(error)),
    }
}

/// The drop guard releasing the caller's chain position, upstream's
/// `finally { releaseNext(); ... }`.
struct QueueSlotGuard {
    notify: Arc<Notify>,
    path_key: String,
}

impl Drop for QueueSlotGuard {
    fn drop(&mut self) {
        self.notify.notify_one();
        let mut queues = registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if queues
            .get(&self.path_key)
            .is_some_and(|tail| Arc::ptr_eq(tail, &self.notify))
        {
            queues.remove(&self.path_key);
        }
    }
}

/// The abort probe the file tools poll between awaits, upstream's
/// `throwIfAborted`.
///
/// The mutation queue must not release from an abort event listener while
/// an in-flight filesystem operation may still finish, so the tools check
/// `signal.aborted` after each await instead: the same aborts are observed
/// while the queue stays locked until the current operation has settled.
///
/// # Errors
/// The aborted rejection, upstream's throw.
pub(crate) fn throw_if_aborted(signal: Option<&AbortSignal>) -> Result<(), AgentToolError> {
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(super::io_error("Operation aborted"));
    }
    Ok(())
}

/// Run `fn` while holding the mutation queue for the canonical path,
/// upstream's `withFileMutationQueue`. Operations for different files still
/// run in parallel.
///
/// # Errors
/// The canonicalization failure; `fn`'s own outcome is its return value.
pub async fn with_file_mutation_queue<T, F>(path: &str, fn_: F) -> Result<T, AgentToolError>
where
    F: Future<Output = T>,
{
    let registration_guard = registration_lock().lock().await;
    let path_key = mutation_queue_key(path).await?;
    let slot = Arc::new(Notify::new());
    let wait_on = {
        let mut queues = registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let wait_on = queues
            .get(&path_key)
            .cloned()
            // The first caller for the path has no predecessor: a
            // pre-armed permit is the resolved `Promise.resolve()`.
            .unwrap_or_else(|| {
                let resolved = Arc::new(Notify::new());
                resolved.notify_one();
                resolved
            });
        queues.insert(path_key.clone(), Arc::clone(&slot));
        wait_on
    };
    drop(registration_guard);

    wait_on.notified().await;
    let _guard = QueueSlotGuard {
        notify: slot,
        path_key,
    };
    Ok(fn_.await)
}
