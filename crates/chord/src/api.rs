//! Factory functions mirroring upstream `src/api.ts`.
//!
//! Facet and service definition with the reserved `$chord.` prefix rules,
//! the facet host entry point, remote service binding construction, the
//! replicated state factory, and the loader combinators. The crate root
//! re-exports this module's surface as the crate's single import point,
//! mirroring upstream's `src/index.ts` export list.

use std::cell::Cell;
use std::rc::Rc;

use crate::consumer::RemoteServiceBinding;
use crate::errors::ChordError;
use crate::facets::host::{FacetKernel, FacetKernelOptions};
use crate::future::{LocalBoxFuture, boxed};
use crate::handle::{Disposal, ErrorReporter};
use crate::services::state::MutableReplicatedState;
use crate::types::{FacetDef, FacetLoader, LoadedFacets};

/// A facet host over one complete active generation, upstream's
/// `FacetHost`.
#[derive(Clone)]
pub struct FacetHost {
    kernel: Rc<FacetKernel>,
}

impl std::fmt::Debug for FacetHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetHost").finish_non_exhaustive()
    }
}

/// Creates an active host for one complete set of facets.
///
/// # Errors
/// [`ChordError`] when the kernel rejects the facet graph or any stage of
/// activation fails; cleanup failures aggregate into the report.
pub async fn create_facet_host(options: FacetKernelOptions) -> Result<FacetHost, ChordError> {
    let kernel = FacetKernel::new(options)?;
    kernel.activate().await?;
    Ok(FacetHost {
        kernel: Rc::new(kernel),
    })
}

impl FacetHost {
    /// The host's provider, the remote surface consumers of the host use.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is not assembled.
    pub fn services(&self) -> Result<crate::services::provider::RemoteServiceProvider, ChordError> {
        self.kernel.provider()
    }

    /// Reloads the listed facets, replacing generations with matching IDs
    /// without disconnecting consumer service handles.
    ///
    /// # Errors
    /// [`ChordError`] when the host is not active, a facet is unknown, or
    /// any reload stage fails.
    pub async fn reload(&self, facets: Vec<FacetDef>) -> Result<(), ChordError> {
        self.kernel.reload(facets).await
    }

    /// Tears the host down.
    ///
    /// # Errors
    /// [`ChordError`] aggregating the failures collected along the way.
    pub async fn dispose(&self) -> Result<(), ChordError> {
        self.kernel.dispose().await
    }
}

/// Builds one facet, upstream's `defineFacet` (which is identity there);
/// the port's constructor adds nothing because the setup closure is
/// synchronous by type.
#[must_use]
pub const fn define_facet(facet: FacetDef) -> FacetDef {
    facet
}

/// Defines one remotable service identity.
///
/// # Errors
/// [`ChordError`] when the ID is empty or begins with the reserved
/// `$chord.` namespace.
pub fn define_service(id: &str) -> Result<crate::types::Service, ChordError> {
    define_service_options(id, false)
}

/// Defines one process-local service identity.
///
/// # Errors
/// [`ChordError`] when the ID is empty or reserved.
pub fn define_local_service(id: &str) -> Result<crate::types::Service, ChordError> {
    define_service_options(id, true)
}

fn define_service_options(id: &str, local: bool) -> Result<crate::types::Service, ChordError> {
    if id.is_empty() {
        return Err(ChordError::Message("Service ID must not be empty".to_string()));
    }
    if id.starts_with("$chord.") {
        return Err(ChordError::Message(
            "Service IDs beginning with $chord. are reserved".to_string(),
        ));
    }
    Ok(crate::types::Service {
        id: id.to_string(),
        local,
    })
}

/// Creates a remote service binding, upstream's
/// `createRemoteServiceBinding`.
///
/// # Errors
/// [`ChordError`] when the service list has duplicate IDs.
pub fn create_remote_service_binding(
    options: crate::consumer::RemoteServiceBindingOptions,
) -> Result<RemoteServiceBinding, ChordError> {
    crate::consumer::create_remote_service_binding(options)
}

/// Creates initialized mutable state, upstream's `replicatedState`.
#[must_use]
pub fn replicated_state(initial: crate::types::JsonValue) -> MutableReplicatedState {
    MutableReplicatedState::new(initial)
}

/// Creates a loader that loads one static facet list on every call,
/// upstream's `createStaticFacetLoader`.
#[must_use]
pub fn create_static_facet_loader(facets: Vec<FacetDef>) -> impl FacetLoader {
    StaticFacetLoader { facets }
}

struct StaticFacetLoader {
    facets: Vec<FacetDef>,
}

impl FacetLoader for StaticFacetLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let facets = self.facets.clone();
        boxed(std::future::ready(Ok(LoadedFacets {
            facets,
            dispose: no_disposal(),
        })))
    }
}

/// A disposal that does nothing, the static loader's teardown.
#[must_use]
pub fn no_disposal() -> Disposal {
    crate::handle::sync_disposal(|| Ok(()))
}

/// Combines loaders so their facets load in loader order and every
/// generation disposes in reverse order, upstream's `combineFacetLoaders`.
#[must_use]
pub fn combine_facet_loaders(loaders: Vec<Rc<dyn FacetLoader>>) -> impl FacetLoader {
    CombinedLoader(Rc::new(loaders))
}

struct CombinedLoader(Rc<Vec<Rc<dyn FacetLoader>>>);

impl FacetLoader for CombinedLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let loaders = self.0.clone();
        boxed(async move {
            let mut facets: Vec<FacetDef> = Vec::new();
            let mut disposals: Vec<Disposal> = Vec::new();
            for loader in loaders.iter() {
                match loader.load().await {
                    Ok(LoadedFacets { facets: loaded, dispose }) => {
                        facets.extend(loaded);
                        disposals.push(dispose);
                    }
                    Err(error) => {
                        let mut errors = Vec::new();
                        for dispose in disposals.into_iter().rev() {
                            if let Err(cleanup) = dispose().await {
                                errors.push(cleanup);
                            }
                        }
                        if errors.is_empty() {
                            return Err(error);
                        }
                        return Err(ChordError::Aggregate(
                            std::iter::once(error).chain(errors).collect(),
                            "Facet loading and cleanup failed".to_string(),
                        ));
                    }
                }
            }
            Ok(LoadedFacets {
                facets,
                dispose: combined_disposal(disposals),
            })
        })
    }
}

fn combined_disposal(disposals: Vec<Disposal>) -> Disposal {
    let disposed = Rc::new(Cell::new(false));
    Box::new(move || {
        boxed(async move {
            if disposed.get() {
                return Ok(());
            }
            disposed.set(true);
            let mut failures = Vec::new();
            for dispose in disposals.into_iter().rev() {
                if let Err(cleanup) = dispose().await {
                    failures.push(cleanup);
                }
            }
            crate::errors::collect_errors(failures, "Failed to dispose loaded facets").map_or(Ok(()), Err)
        })
    })
}

/// The error reporter a host or binding gets when none is supplied, the
/// upstream `onError ?? (() => {})` default.
#[must_use]
pub fn default_on_error() -> ErrorReporter {
    crate::handle::no_error_reporter()
}