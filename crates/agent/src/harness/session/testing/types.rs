//! The conformance-surface types, ported from upstream
//! `src/harness/session/testing/types.ts`.

use pi_ai::types::BoxedFuture;

use crate::harness::context::background_context;
use crate::harness::session::types::Storage;
use std::sync::Arc;

/// A fresh backend storage instance owned by one conformance case, upstream's
/// `StorageFixture`.
///
/// Upstream's fixture carries an `Symbol.asyncDispose` closer; every fixture
/// in-tree closes its storage, so the port's dispose does exactly that.
#[derive(Clone)]
pub struct StorageFixture {
    /// The storage the case runs against.
    pub storage: Arc<dyn Storage>,
}

impl std::fmt::Debug for StorageFixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageFixture").finish_non_exhaustive()
    }
}

impl StorageFixture {
    /// A fixture closing the storage on dispose.
    #[must_use]
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }

    /// Release the fixture, upstream's `asyncDispose`.
    pub async fn dispose(&self) {
        let _ = self.storage.close(&background_context()).await;
    }
}

/// A runner-independent conformance case any test framework registers,
/// upstream's `ConformanceCase`.
#[derive(Clone)]
pub struct ConformanceCase {
    /// The group the case reports under.
    pub group: String,
    /// The case name.
    pub name: String,
    run: Arc<dyn Fn() -> BoxedFuture<'static, ()> + Send + Sync>,
}

impl std::fmt::Debug for ConformanceCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConformanceCase")
            .field("group", &self.group)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ConformanceCase {
    /// One case over the async body; failures surface as test panics,
    /// upstream's rejected `run` promise.
    pub fn new<F, G, N>(group: G, name: N, run: F) -> Self
    where
        F: Fn() -> BoxedFuture<'static, ()> + Send + Sync + 'static,
        G: Into<String>,
        N: Into<String>,
    {
        Self {
            group: group.into(),
            name: name.into(),
            run: Arc::new(run),
        }
    }

    /// Run the case against its fresh fixture, upstream's `run`.
    pub async fn run(&self) {
        (self.run)().await;
    }
}

/// The repository context one repo conformance case runs against, upstream's
/// `RepoCaseContext`.
pub struct RepoFixture {
    /// The fresh backend repository instance.
    pub repo: Arc<dyn crate::harness::session::types::SessionRepo>,
    /// The close hook the case runs after its body, upstream's `close?`.
    pub close: Option<Arc<dyn Fn() -> BoxedFuture<'static, ()> + Send + Sync>>,
}

impl std::fmt::Debug for RepoFixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepoFixture").finish_non_exhaustive()
    }
}

impl RepoFixture {
    /// A fixture with the optional close hook.
    #[must_use]
    pub fn new(
        repo: Arc<dyn crate::harness::session::types::SessionRepo>,
        close: Option<Arc<dyn Fn() -> BoxedFuture<'static, ()> + Send + Sync>>,
    ) -> Self {
        Self { repo, close }
    }

    /// Release the fixture, upstream's `finally { await context.close?.() }`.
    pub async fn dispose(&self) {
        if let Some(close) = &self.close {
            close().await;
        }
    }
}
