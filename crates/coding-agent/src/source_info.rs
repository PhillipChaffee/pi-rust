//! Resource provenance, upstream's `src/core/source-info.ts`.
//!
//! Where a loaded resource came from: the source label (a local file, an
//! npm package, an extension), its trust scope, and whether a package
//! manifest named it. The metadata itself is the package manager's
//! [`PathMetadata`]; this module is
//! the per-resource view of it.

use crate::package_manager::PathMetadata;

/// The trust scope a resource loads under, upstream's `SourceScope`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceScope {
    /// The user's agent dir.
    User,
    /// The project's `.pi` tree (trust-gated).
    Project,
    /// A CLI flag, an extension, or another ephemeral source.
    Temporary,
}

/// Whether a resource came from a package manifest or a top-level
/// declaration, upstream's `SourceOrigin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceOrigin {
    /// Declared by a package's `pi` manifest (or an installed package's
    /// default layout).
    Package,
    /// Declared at the top level — settings, an auto-discovered directory,
    /// or a CLI path.
    TopLevel,
}

/// Where a loaded resource came from, upstream's `SourceInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceInfo {
    /// The resource's own path.
    pub path: String,
    /// The source label, upstream's `source` (`local`, `auto`,
    /// `npm:<name>`, `git:<url>`, `extension:<name>`, …).
    pub source: String,
    /// The trust scope the resource loaded under.
    pub scope: SourceScope,
    /// Whether a package manifest named the resource.
    pub origin: SourceOrigin,
    /// The directory the resource was declared against, upstream's
    /// `baseDir` — the agent dir, the project `.pi` dir, or the package
    /// root.
    pub base_dir: Option<String>,
}

/// Build a resource's provenance from its package metadata, upstream's
/// `createSourceInfo`.
#[must_use]
pub fn create_source_info(path: &str, metadata: &PathMetadata) -> SourceInfo {
    SourceInfo {
        path: path.to_string(),
        source: metadata.source.clone(),
        scope: metadata.scope,
        origin: metadata.origin,
        base_dir: metadata.base_dir.clone(),
    }
}

/// Build synthetic provenance without package metadata, upstream's
/// `createSyntheticSourceInfo`. The scope and origin default to
/// temporary/top-level, the way upstream's optionals do.
#[must_use]
pub fn create_synthetic_source_info(path: &str, options: &SyntheticSourceOptions) -> SourceInfo {
    SourceInfo {
        path: path.to_string(),
        source: options.source.clone(),
        scope: options.scope.unwrap_or(SourceScope::Temporary),
        origin: options.origin.unwrap_or(SourceOrigin::TopLevel),
        base_dir: options.base_dir.clone(),
    }
}

/// The synthetic-provenance options, upstream's `createSyntheticSourceInfo`
/// options object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SyntheticSourceOptions {
    /// The source label.
    pub source: String,
    /// The trust scope; defaults to temporary.
    pub scope: Option<SourceScope>,
    /// The origin; defaults to top-level.
    pub origin: Option<SourceOrigin>,
    /// The directory the resource was declared against.
    pub base_dir: Option<String>,
}
