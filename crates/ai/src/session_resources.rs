//! The process-wide session-resource registry, ported from
//! `packages/ai/src/session-resources.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream keeps one module-level `Set` of cleanup callbacks; registrations
//! persist until their unregister closure runs, and cleanup invokes every
//! registered callback once, aggregating failures. Rust cannot throw, so a
//! cleanup reports failure through its `Result` — the restatement that turns
//! upstream's thrown values into the returned error's failure list.
//!
//! Cleanups run while the registry lock is held; a cleanup that registers or
//! unregisters synchronously would deadlock, the Rust-native restatement of
//! upstream's mutation-during-iteration allowance.

use std::error::Error as StdError;
use std::fmt;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// A registered cleanup: invoked with the session id being torn down, when a
/// session id is known, upstream's `SessionResourceCleanup`.
pub type SessionResourceCleanup =
    Box<dyn Fn(Option<&str>) -> Result<(), Box<dyn StdError + Send + Sync>> + Send>;

struct Registry {
    next_id: usize,
    cleanups: Vec<(usize, SessionResourceCleanup)>,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(Registry {
            next_id: 1,
            cleanups: Vec::new(),
        })
    })
}

/// Locks the registry, recovering from poisoning so cleanup stays passive.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The handle a registration returns; calling [`SessionResourceGuard::unregister`]
/// removes the cleanup, upstream's returned closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionResourceGuard {
    id: usize,
}

impl SessionResourceGuard {
    /// Removes the registered cleanup. Later cleanups keep their registration
    /// order.
    pub fn unregister(self) {
        let mut registry = lock(registry());
        registry.cleanups.retain(|(id, _)| *id != self.id);
    }
}

/// Registers a session-resource cleanup and returns the handle that removes
/// it. Cleanups run in registration order.
#[must_use]
pub fn register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> SessionResourceGuard {
    let id = {
        let mut registry = lock(registry());
        let id = registry.next_id;
        registry.next_id += 1;
        registry.cleanups.push((id, cleanup));
        id
    };
    SessionResourceGuard { id }
}

/// All cleanup failures from one [`cleanup_session_resources`] call, upstream's
/// `AggregateError(errors, "Failed to cleanup session resources")`.
#[derive(Debug)]
pub struct SessionCleanupError {
    /// One description per failed cleanup, in registration order.
    pub failures: Vec<String>,
}

impl fmt::Display for SessionCleanupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to cleanup session resources")?;
        for failure in &self.failures {
            write!(f, ": {failure}")?;
        }
        Ok(())
    }
}

impl StdError for SessionCleanupError {}

/// Invokes every registered cleanup with `session_id` and aggregates the
/// failures.
///
/// # Errors
/// One [`SessionCleanupError`] listing every failure's description when any
/// cleanup failed; every cleanup still runs, in registration order.
pub fn cleanup_session_resources(session_id: Option<&str>) -> Result<(), SessionCleanupError> {
    let registry = lock(registry());
    let mut failures = Vec::new();
    for (_, cleanup) in &registry.cleanups {
        if let Err(error) = cleanup(session_id) {
            failures.push(error.to_string());
        }
    }
    drop(registry);
    if failures.is_empty() {
        Ok(())
    } else {
        Err(SessionCleanupError { failures })
    }
}

#[cfg(test)]
mod tests;
