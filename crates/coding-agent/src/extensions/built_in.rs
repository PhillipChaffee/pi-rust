//! First-party compile-in extensions, upstream's
//! `packages/coding-agent/src/extensions/index.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! A first-party extension compiles into the binary and registers directly
//! through the [`ExtensionApi`](crate::extensions::api::ExtensionApi) its
//! factory receives at load (ADR 0007). Upstream's only built-in entry is
//! the llama.cpp extension, whose provider registration rides the
//! native-provider queue's ticket and whose `/llama` command rides #132's
//! UI surface, so the list lands empty until that session compiles it in.

use crate::extensions::api::InlineExtension;

/// The compile-in first-party extensions, upstream's `builtInExtensions`.
///
/// Each entry loads through the loader's inline-factory arm with its name
/// in the `<inline:name>` label and — named entries only — can hide itself
/// from user-facing lists.
#[must_use]
pub const fn built_in_extensions() -> Vec<InlineExtension> {
    Vec::new()
}

/// The host build path's factory list, upstream `main`'s merge.
///
/// The list is `[...builtInExtensions, ...options.extensionFactories]`:
/// first-party entries load ahead of the caller's inline factories, so
/// their registrations win name collisions downstream.
#[must_use]
pub fn extension_factories(caller_factories: Vec<InlineExtension>) -> Vec<InlineExtension> {
    let mut factories = built_in_extensions();
    factories.extend(caller_factories);
    factories
}

#[cfg(test)]
mod tests;
