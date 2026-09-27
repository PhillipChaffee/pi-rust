//! Serializes file mutations per canonical path, ported from upstream
//! `src/harness/tools/file-mutation-queue.ts`.
//!
//! Upstream keys a `WeakMap<ExecutionEnv, MutationQueueState>` by the
//! environment object itself, so concurrent tool executions sharing one
//! environment serialize through it even when they address the same file by
//! different (canonical-equivalent) paths. Rust restate: a process-global
//! registry keyed by the environment's `Arc` allocation address with a
//! `Weak` back-reference — dead environments are reaped on the next access,
//! which is the `WeakMap` entry lifetime; two environments at the same
//! address cannot coexist because a live `Arc` pinning the address is held
//! for the queue's duration.
//!
//! The promise chain restates as a per-path chain of `Notify`s: each caller
//! registers its own `Notify` as the path's tail and waits on the tail it
//! displaced — the exact hand-off shape upstream's `chainedQueue` builds,
//! with the release running from a drop guard so aborts and panics release
//! the chain like upstream's `finally`.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::Notify;

use crate::harness::context::Context;
use crate::harness::types::{ExecutionEnv, FileError, FileErrorCode};
use crate::types::AgentToolError;

/// The process-global registry: the environments' `Arc` addresses to their
/// weak back-reference and queue state, upstream's `WeakMap<ExecutionEnv,
/// MutationQueueState>`.
type MutationRegistry = HashMap<usize, (Weak<dyn ExecutionEnv>, MutationQueueState)>;

/// The per-environment queue state, upstream's `MutationQueueState`.
struct MutationQueueState {
    /// The tail `Notify` per canonical path, upstream's `queues: Map<string,
    /// Promise<void>>`.
    queues: HashMap<String, Arc<Notify>>,
    /// The registration lock serializing key computation and tail
    /// registration per environment, upstream's `registration` promise.
    registration: Arc<tokio::sync::Mutex<()>>,
}

impl MutationQueueState {
    fn new() -> Self {
        Self {
            queues: HashMap::new(),
            registration: Arc::new(tokio::sync::Mutex::new(())),
        }
    }
}

/// The process-global registry, upstream's `WeakMap` states.
fn registry() -> &'static Mutex<MutationRegistry> {
    static REGISTRY: std::sync::OnceLock<Mutex<MutationRegistry>> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The registry key for one environment, its `Arc` allocation address.
fn environment_key(env: &Arc<dyn ExecutionEnv>) -> usize {
    Arc::as_ptr(env).cast::<()>() as usize
}

/// Returns the environment's registration lock, reaping entries whose
/// environment died, upstream's `getState`.
#[expect(
    clippy::significant_drop_tightening,
    reason = "the reap, insert, and clone are one registry transaction; splitting them would race the entry's creation"
)]
fn get_state(env: &Arc<dyn ExecutionEnv>) -> Arc<tokio::sync::Mutex<()>> {
    let key = environment_key(env);
    let mut states = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    states.retain(|_, (weak, _)| weak.upgrade().is_some());
    let (_, state) = states
        .entry(key)
        .or_insert_with(|| (Arc::downgrade(env), MutationQueueState::new()));
    Arc::clone(&state.registration)
}

/// The canonical-path queue key, upstream's `getMutationQueueKey`: canonical
/// paths serialize; when canonicalization is unavailable the absolute path
/// is the fallback key.
async fn mutation_queue_key(
    env: &Arc<dyn ExecutionEnv>,
    path: &str,
    context: &Context,
) -> Result<String, AgentToolError> {
    let absolute_path = env
        .absolute_path(path, context)
        .await
        .map_err(Box::<FileError>::from)?;
    match env.canonical_path(&absolute_path, context).await {
        Ok(canonical_path) => Ok(canonical_path),
        Err(error)
            if error.code == FileErrorCode::NotFound
                || error.code == FileErrorCode::NotSupported =>
        {
            Ok(absolute_path)
        }
        Err(error) => Err(Box::new(error)),
    }
}

/// The drop guard releasing the caller's chain position, upstream's
/// `finally { releaseNext(); ... }`: it wakes the next registered caller
/// and removes the path's tail entry when this guard's `Notify` is still
/// the registered tail.
struct QueueSlotGuard {
    notify: Arc<Notify>,
    environment_key: usize,
    path_key: String,
}

impl Drop for QueueSlotGuard {
    fn drop(&mut self) {
        self.notify.notify_one();
        if let Some((_, state)) = registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_mut(&self.environment_key)
            && state
                .queues
                .get(&self.path_key)
                .is_some_and(|tail| Arc::ptr_eq(tail, &self.notify))
        {
            state.queues.remove(&self.path_key);
        }
    }
}

/// Runs `fn` while holding the mutation queue for the environment and
/// canonical path, upstream's `withFileMutationQueue`.
///
/// # Errors
/// The canonicalization failure, or `fn`'s own failure; both release the
/// queue position before returning.
#[expect(
    clippy::significant_drop_tightening,
    reason = "the tail read, insert, and swap are one registry transaction; splitting them would race the chain's tail"
)]
pub async fn with_file_mutation_queue<T, F>(
    env: &Arc<dyn ExecutionEnv>,
    path: &str,
    fn_: F,
    context: &Context,
) -> Result<T, AgentToolError>
where
    F: Future<Output = Result<T, AgentToolError>>,
{
    let registration = get_state(env);
    // Held across the key computation and tail registration: upstream's
    // `state.registration` chain serializes the same span.
    let registration_guard = registration.lock().await;
    let path_key = mutation_queue_key(env, path, context).await?;
    let (wait_on, slot) = {
        let mut states = registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some((_, state)) = states.get_mut(&environment_key(env)) else {
            unreachable!("the entry for the live environment");
        };
        let wait_on = state
            .queues
            .get(&path_key)
            .cloned()
            // The first caller for the path has no predecessor: a
            // pre-armed permit is the resolved `Promise.resolve()`.
            .unwrap_or_else(|| {
                let resolved = Arc::new(Notify::new());
                resolved.notify_one();
                resolved
            });
        let slot = Arc::new(Notify::new());
        state.queues.insert(path_key.clone(), Arc::clone(&slot));
        (wait_on, slot)
    };
    drop(registration_guard);

    wait_on.notified().await;
    let guard = QueueSlotGuard {
        notify: slot,
        environment_key: environment_key(env),
        path_key,
    };
    let result = fn_.await;
    drop(guard);
    result
}
