//! The process-global default [`StreamFn`], upstream's `src/stream-fn.ts`.
//!
//! Configure the fallback the `Agent` and low-level loops use when callers
//! omit their stream function: hosts that provide a default model runtime
//! install its stream function here without making `pi-agent-core` depend on
//! a provider catalog or compatibility layer.

use std::sync::RwLock;

use crate::types::StreamFn;

static DEFAULT_STREAM_FN: RwLock<Option<StreamFn>> = RwLock::new(None);

/// The error [`get_default_stream_fn`] raises when no default is configured,
/// upstream's thrown `Error`.
#[derive(Debug)]
pub struct NoDefaultStreamFn;

impl std::fmt::Display for NoDefaultStreamFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The message text is the upstream contract verbatim.
        write!(
            f,
            "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
        )
    }
}

impl std::error::Error for NoDefaultStreamFn {}

/// Read-lock the cell under the house poison-recovery policy: the cell holds
/// owned plain data, so a poisoned lock recovers and keeps the global usable.
fn read() -> std::sync::RwLockReadGuard<'static, Option<StreamFn>> {
    DEFAULT_STREAM_FN
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Write-lock the cell; the section is a single assignment.
fn write() -> std::sync::RwLockWriteGuard<'static, Option<StreamFn>> {
    DEFAULT_STREAM_FN
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Configure the fallback used by `Agent` and low-level loops when callers
/// omit their stream function, upstream's `setDefaultStreamFn`. `None`
/// clears it.
pub fn set_default_stream_fn(stream_fn: Option<StreamFn>) {
    *write() = stream_fn;
}

/// The configured default stream function.
///
/// # Errors
/// `NoDefaultStreamFn` when no default is configured — pass `stream_fn`
/// explicitly or call [`set_default_stream_fn`] first, per the upstream
/// contract.
pub fn get_default_stream_fn() -> Result<StreamFn, NoDefaultStreamFn> {
    read().clone().ok_or(NoDefaultStreamFn)
}
