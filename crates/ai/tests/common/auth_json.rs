//! Test helper resolving API keys from `~/.pi/agent/auth.json`, ported from
//! `packages/ai/test/oauth.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Supports both API key and OAuth credentials: api-key credentials return
//! the stored key; OAuth credentials return the access token, refreshing an
//! expired one first and saving the (rotated) credential back to `auth.json`.
//!
//! Porting restatements this module records:
//!
//! - The path is injectable ([`AuthJsonStore::auth_path`]) so suites run
//!   hermetically; [`default_path`] reproduces upstream's
//!   `~/.pi/agent/auth.json` for live use.
//! - Upstream looks the OAuth flow up through `builtinProviders()`;
//!   [`resolve_api_key`] takes the flow as a parameter instead, so suites
//!   inject the flow under test without the provider registry.
//! - Upstream logs a refresh error's serialized object to stdout; the port
//!   drops the log line so nothing can carry key material to the output.
//! - [`AuthJsonStore::save`] is best-effort: io failures are dropped, where
//!   upstream's `writeFileSync` would throw through the caller. The write
//!   payload is byte-identical to upstream's (pretty JSON, no trailing
//!   newline).

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use tokio_util::sync::CancellationToken;

use pi_ai::auth::clock::{AuthClock, SystemClock};
use pi_ai::auth::types::{Credential, OAuthAuth};

/// The auth file's name, the tail of upstream's `homedir()`-joined
/// `~/.pi/agent/auth.json` path.
pub const AUTH_FILE_NAME: &str = "auth.json";

/// The auth.json credential store, parameterized for hermetic tests.
#[derive(Debug)]
pub struct AuthJsonStore {
    /// The file the store reads and writes; [`default_path`] supplies
    /// upstream's path when a suite wants the live location.
    pub auth_path: PathBuf,
}

/// Upstream's `~/.pi/agent/auth.json` path: the home directory joined into
/// `.pi/agent/`. When no home directory is known, the path degrades to
/// `.pi/agent/auth.json` relative to the process working directory.
#[must_use]
pub fn default_path() -> PathBuf {
    home_dir().map_or_else(
        || PathBuf::from(".pi").join("agent").join(AUTH_FILE_NAME),
        |home| home.join(".pi").join("agent").join(AUTH_FILE_NAME),
    )
}

/// The user's home directory, upstream's `homedir()` join root; `None` when
/// the platform cannot determine one.
#[must_use]
pub fn home_dir() -> Option<PathBuf> {
    std::env::home_dir()
}

impl AuthJsonStore {
    /// Read the credential store, upstream's `loadAuthStorage`: a missing
    /// file or a parse failure yields an empty store, and entries parse as
    /// the tagged [`Credential`] union keyed on the `type` field.
    #[must_use]
    pub fn load(&self) -> BTreeMap<String, Credential> {
        std::fs::read_to_string(&self.auth_path)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
            .unwrap_or_default()
    }

    /// Write the whole store, upstream's `saveAuthStorage`: create the parent
    /// directory with 0700 when missing, write pretty JSON, and set the file
    /// to 0600 so stored key material stays owner-only. Persistence failures
    /// are dropped (see the module doc).
    pub fn save(&self, storage: &BTreeMap<String, Credential>) {
        if let Some(parent) = self.auth_path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
        let Ok(serialized) = serde_json::to_string_pretty(storage) else {
            return;
        };
        if std::fs::write(&self.auth_path, serialized).is_err() {
            return;
        }
        let _ = std::fs::set_permissions(&self.auth_path, std::fs::Permissions::from_mode(0o600));
    }
}

/// Resolve the api key or OAuth access token for `provider` from the store,
/// upstream's `resolveApiKey`. An api-key credential returns the stored key.
/// An OAuth credential resolves the flow from the `oauth` argument (upstream
/// looks it up through `builtinProviders()`; the parameter replaces that
/// lookup), refreshes it when the epoch-millisecond expiry has passed
/// (`now >= expires`, the wall clock over `std::time` behind
/// [`SystemClock`]), writes the store back either way, and returns
/// `to_auth`'s api key. A refresh failure returns `None` (upstream logs the
/// serialized error; the port drops the log).
pub async fn resolve_api_key(
    store: &AuthJsonStore,
    provider: &str,
    oauth: Option<&OAuthAuth>,
) -> Option<String> {
    let mut storage = store.load();
    let entry = storage.get(provider)?.clone();
    match entry {
        Credential::ApiKey(api_key) => api_key.key,
        Credential::OAuth(stored) => {
            // Upstream's builtinProviders() lookup; the injected flow takes
            // its place so no registry is needed here.
            let oauth = oauth?;
            let mut credential = stored;
            if SystemClock.now_ms() >= credential.expires {
                credential =
                    match (oauth.refresh)(credential.clone(), CancellationToken::new()).await {
                        Ok(refreshed) => refreshed,
                        Err(_) => return None,
                    };
            }
            let api_key = (oauth.to_auth)(credential.clone())
                .await
                .expect("to_auth derives the api key")
                .api_key;
            storage.insert(provider.to_owned(), Credential::OAuth(credential));
            store.save(&storage);
            api_key
        }
    }
}
