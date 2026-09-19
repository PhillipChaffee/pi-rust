//! The facet host: the kernel that validates the facet graph, binds
//! service handles, and drives activation, disposal, and reload, ported
//! from upstream `src/facets/`.

pub mod host;

pub use host::{
    FacetEnvironment, FacetKernel, FacetKernelOptions, GenerationPhase, KeyedSource,
    LocalKeyedServiceRegistry, StagedServiceSpawner,
};
