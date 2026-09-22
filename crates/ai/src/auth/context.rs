//! The default auth context, ported from
//! `packages/ai/src/auth/context.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::auth::types::AuthContext;

/// The default auth context: environment variables from the process, and file
/// existence through the filesystem, expanding a leading `~` against the home
/// directory.
///
/// Porting restatement: the browser fallbacks are gone — a Rust host always
/// has a filesystem and process env, so the probes are synchronous and never
/// report "unavailable".
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAuthContext;

impl AuthContext for DefaultAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        let value = std::env::var(name).ok()?;
        (!value.trim().is_empty()).then_some(value)
    }

    fn file_exists(&self, path: &str) -> bool {
        std::fs::metadata(resolve_home(path)).is_ok()
    }
}

/// Expand a leading `~` against `$HOME`; other paths pass through verbatim.
fn resolve_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix('~')
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest.trim_start_matches('/'));
    }
    PathBuf::from(path)
}

/// The default provider auth context, upstream's
/// `defaultProviderAuthContext()`.
#[must_use]
pub fn default_provider_auth_context() -> Arc<dyn AuthContext> {
    Arc::new(DefaultAuthContext)
}
