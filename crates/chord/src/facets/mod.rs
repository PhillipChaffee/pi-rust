//! Facet hosting, ported from upstream `src/facets/*`: the kernel that
//! validates the dependency graph, binds service handles, activates
//! providers before consumers, and disposes in reverse order; the lifecycle
//! state machine gating handle use; the staged-candidate reload cutover; and
//! the loader combination surface upstream places in `facets/loader.ts`.
