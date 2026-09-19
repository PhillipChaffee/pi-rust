//! Core data model of the runtime, ported from upstream `src/types.ts`: the
//! owned `JsonValue` tree, the service vocabulary (`ServiceType` singleton
//! and keyed variants, `ServiceCall`, `ServiceProviderUpdate`,
//! `ServiceCatalogueEntry`, `ServiceInstanceAddress`), the `Service<T>` and
//! `Context` contracts, and the facet-host plus remote-transport surfaces.
//! Contracts TypeScript enforces only at compile time (`RemoteServiceContract`,
//! `JsonRepresentation`) restate here as trait bounds over owned wire-safe
//! data.
