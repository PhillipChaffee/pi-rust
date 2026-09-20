//! Credential storage, ported from
//! `packages/ai/src/auth/credential-store.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The default store keys one credential per provider id; writes serialize
//! per provider through a tokio mutex, the promise-chain port. Reads are
//! unsynchronized snapshots, matching upstream's immediate `read`.
//!
//! Porting restatements: `list` order is sorted by provider id where
//! TypeScript's map preserves insertion order; the trait is object-safe with
//! boxed futures so hosts hold `Arc<dyn CredentialStore>`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Mutex as AsyncMutex;

use crate::auth::types::{AuthError, AuthOptions, Credential, CredentialInfo, CredentialModifyFn};
use crate::types::BoxedFuture;
use crate::utils::abort::{AbortError, operation_signal};

/// The abort failure every early-exit reports, boxed at the auth error type.
fn aborted() -> AuthError {
    Box::new(AbortError)
}

/// App-owned credential storage keyed by `Provider.id`, one credential per
/// provider, upstream's `CredentialStore`.
///
/// `modify` is the only write path, so every mutation is a serialized
/// read-modify-write; `Models.getAuth()` runs OAuth refresh inside `modify`
/// so concurrent requests cannot double-refresh a rotated token.
///
/// Error semantics: `read` resolves `None` for missing entries. Methods fail
/// only on storage failure; Models wraps such failures in
/// [`ModelsError`](crate::auth::resolve::ModelsError) with the `auth` code.
/// Best-effort stores that serve an in-memory view and record persistence
/// errors internally are valid implementations.
pub trait CredentialStore: Send + Sync {
    /// Read the stored credential, possibly expired. Display/status use;
    /// resolved request auth comes from `Models.getAuth()`.
    ///
    /// # Errors
    /// A storage failure, or an already-aborted signal.
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>>;

    /// List stored credential metadata without resolving or exposing
    /// secrets. Implementations must not execute configured API-key commands
    /// while listing.
    ///
    /// # Errors
    /// A storage failure, or an already-aborted signal.
    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>>;

    /// Serialized write — the only write path. `f` sees the current
    /// credential because correct writes (refresh, login-during-refresh)
    /// depend on it; return the new credential, or `None` to leave the entry
    /// unchanged. Mutual exclusion per provider id. Resolves with the
    /// post-write credential; `f`'s failures propagate.
    ///
    /// # Errors
    /// A storage failure, `f`'s own failure, or an aborted signal — an
    /// aborted write is never applied.
    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>>;

    /// Remove a credential (logout). Serialized against `modify`.
    ///
    /// # Errors
    /// A storage failure, or an already-aborted signal.
    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>>;
}

/// Default in-memory credential store, upstream's `InMemoryCredentialStore`.
/// Apps inject persistent stores. Writes serialize per provider id through
/// one tokio mutex each.
#[derive(Debug, Default)]
pub struct InMemoryCredentialStore {
    credentials: Mutex<BTreeMap<String, Credential>>,
    chains: Mutex<BTreeMap<String, Arc<AsyncMutex<()>>>>,
}

impl InMemoryCredentialStore {
    /// The serialized write lock for a provider, created on first use.
    fn chain(&self, provider_id: &str) -> Arc<AsyncMutex<()>> {
        let mut chains = self
            .chains
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        chains.entry(provider_id.to_owned()).or_default().clone()
    }
}

impl CredentialStore for InMemoryCredentialStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(async move {
            let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
            if signal.is_cancelled() {
                return Err(aborted());
            }
            let credentials = self
                .credentials
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(credentials.get(provider_id).cloned())
        })
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        Box::pin(async move {
            let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
            if signal.is_cancelled() {
                return Err(aborted());
            }
            let entries = {
                let credentials = self
                    .credentials
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                credentials
                    .iter()
                    .map(|(provider_id, credential)| CredentialInfo {
                        provider_id: provider_id.clone(),
                        auth_type: credential.auth_type(),
                    })
                    .collect::<Vec<_>>()
            };
            Ok(entries)
        })
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        let chain = self.chain(provider_id);
        Box::pin(async move {
            // Queue behind the previous write; an aborted waiter never runs.
            let _guard = tokio::select! {
                () = signal.cancelled() => return Err(aborted()),
                guard = chain.lock() => guard,
            };
            if signal.is_cancelled() {
                return Err(aborted());
            }
            let current = self
                .credentials
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(provider_id)
                .cloned();
            let next = f(current.clone()).await?;
            let _ = &current;
            // Upstream checks the signal after `fn` settles so an aborted
            // mutation never persists its result.
            if signal.is_cancelled() {
                return Err(aborted());
            }
            if let Some(next) = &next {
                self.credentials
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(provider_id.to_owned(), next.clone());
            }
            Ok(next.or(current))
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        let chain = self.chain(provider_id);
        Box::pin(async move {
            let _guard = tokio::select! {
                () = signal.cancelled() => return Err(aborted()),
                guard = chain.lock() => guard,
            };
            if signal.is_cancelled() {
                return Err(aborted());
            }
            self.credentials
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(provider_id);
            Ok(())
        })
    }
}
