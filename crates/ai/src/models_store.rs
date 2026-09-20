//! Catalog persistence, ported from
//! `packages/ai/src/models-store.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::types::{BoxedFuture, Model};
use crate::utils::abort::AbortError;

/// One provider's persisted catalog, upstream's `ModelsStoreEntry`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelsStoreEntry {
    /// The catalog models.
    pub models: Vec<Model>,
    /// Unix timestamp from the remote catalog's Last-Modified header.
    pub last_modified: Option<i64>,
    /// Unix timestamp of the last completed remote check.
    pub checked_at: Option<i64>,
    /// Opaque validator from the remote catalog's `ETag` header, stored
    /// verbatim (quotes included) and echoed back as If-None-Match.
    pub etag: Option<String>,
}

/// Options for catalog-store operations, upstream's
/// `ModelsStoreOperationOptions`.
#[derive(Clone, Debug, Default)]
pub struct ModelsStoreOptions {
    /// Cancellation for the operation.
    pub signal: Option<tokio_util::sync::CancellationToken>,
}

/// Persistent model catalogs keyed by provider ID, upstream's `ModelsStore`.
///
/// `read` resolves `None` for missing entries; methods fail only on storage
/// failure. Models treats a store failure as a refresh error.
pub trait ModelsStore: Send + Sync {
    /// Read the entry for a provider.
    ///
    /// # Errors
    /// A storage failure, or an already-aborted signal.
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<Option<ModelsStoreEntry>, ModelsStoreError>>;

    /// Write the entry for a provider, replacing any previous one.
    ///
    /// # Errors
    /// A storage failure, or an already-aborted signal.
    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: ModelsStoreEntry,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>>;

    /// Remove the entry for a provider.
    ///
    /// # Errors
    /// A storage failure, or an already-aborted signal.
    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>>;
}

/// The failure a [`ModelsStore`] operation reports.
pub type ModelsStoreError = Box<dyn std::error::Error + Send + Sync>;

/// The in-memory models store, upstream's `InMemoryModelsStore`.
///
/// Hosts inject persistent stores; this default clones entries on read and
/// write the way upstream's `structuredClone` did, so callers cannot mutate
/// stored state.
#[derive(Debug, Default)]
pub struct InMemoryModelsStore {
    entries: Mutex<BTreeMap<String, ModelsStoreEntry>>,
}

fn aborted() -> ModelsStoreError {
    Box::new(AbortError)
}

impl ModelsStore for InMemoryModelsStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<Option<ModelsStoreEntry>, ModelsStoreError>> {
        Box::pin(async move {
            if options.is_some_and(|options| {
                options
                    .signal
                    .as_ref()
                    .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
            }) {
                return Err(aborted());
            }
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(entries.get(provider_id).cloned())
        })
    }

    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: ModelsStoreEntry,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>> {
        Box::pin(async move {
            if options.is_some_and(|options| {
                options
                    .signal
                    .as_ref()
                    .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
            }) {
                return Err(aborted());
            }
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(provider_id.to_owned(), entry);
            Ok(())
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>> {
        Box::pin(async move {
            if options.is_some_and(|options| {
                options
                    .signal
                    .as_ref()
                    .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
            }) {
                return Err(aborted());
            }
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(provider_id);
            Ok(())
        })
    }
}
