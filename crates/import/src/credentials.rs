//! The credentials leg: TS pi's `auth.json` migrated into the parity-mirror
//! store the Rust pi reads.
//!
//! The source is a provider-id keyed record of `api_key` and `oauth`
//! entries with `!cmd`/`${VAR}` indirection; the write goes through the
//! Rust store's own locked path (parent directory at 0700, the lock
//! discipline, the 0600 file mode).
//!
//! The decision's rules bind the leg: indirections carry verbatim and never
//! execute (import resolves nothing), key material never prints (the
//! report's items carry provider ids and kinds only), and the source files
//! stay untouched — the legacy `oauth.json`/`settings.json#apiKeys` fold
//! reads them the way upstream's `migrateAuthToAuthJson` merges them
//! (oauth entries win, api-keys fill the gaps, string values only) without
//! the renames and rewrites a running TS pi would do itself.
//!
//! Validation runs each entry through pi-ai's [`Credential`], the typed read
//! the Rust store performs. The one normalization the tool makes: a
//! fractional `expires` floors to an integer, because Rust's `i64` rejects
//! the JSON float upstream's `Number.isFinite` accepted. Entries that
//! cannot validate fail and do not land in the target store; in place they
//! stay in the file untouched (the Rust store reports the malformed entry
//! for that provider alone), so import is never destructive.

use std::path::Path;

use pi_ai::auth::types::Credential;
use pi_coding_agent::auth_storage::FileAuthStorageBackend;
use pi_coding_agent::utils::text::strip_bom;
use serde_json::{Map, Value};

use crate::discovery::Discovery;
use crate::plan::{OpTarget, PlannedOp};
use crate::report::{CredentialItem, CredentialOutcome, ImportReport};

/// The credentials leg's outcome: its report items plus the fold flag.
///
/// The flag tells the settings leg to drop `settings.json`'s `apiKeys`,
/// the file shape upstream's migration removes from the settings after the
/// fold into the auth store.
#[derive(Debug, Default)]
pub struct CredentialsLeg {
    /// The report index of every credential item the store write carries,
    /// the `AuthWrite` op's failure targets.
    pub landed_items: Vec<usize>,
    /// The write to plan, when the target store's content changes.
    pub write: Option<AuthWritePlan>,
    /// Whether the fold consumed `settings.json`'s `apiKeys` object.
    pub settings_api_keys_consumed: bool,
}

/// The auth store write the leg computed: the store's path and the merged
/// map's serialized form.
pub struct AuthWritePlan {
    /// The store's path.
    pub auth_path: String,
    /// The next file content, the merged map pretty-serialized; the Debug
    /// form hides it, the house rule's print discipline.
    pub next: String,
}

impl std::fmt::Debug for AuthWritePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The `next` content is the merged store, key material included;
        // the debug form names the path only, the house rule's print
        // discipline.
        f.debug_struct("AuthWritePlan")
            .field("auth_path", &self.auth_path)
            .finish_non_exhaustive()
    }
}

/// Run the credentials leg against the discovery, appending items and
/// returning the leg's outcome for the settings leg.
pub fn run_credentials(discovery: &Discovery, report: &mut ImportReport) -> CredentialsLeg {
    let mut leg = CredentialsLeg::default();
    let source_auth_path = discovery.source.join("auth.json");
    let mut sources = Vec::new();

    // The source entries: auth.json when present, else the legacy fold.
    let source_map = if source_auth_path.exists() {
        sources.push("auth.json".to_string());
        match read_json_object(&source_auth_path) {
            Ok(map) => map,
            Err(reason) => {
                report.push_failure(
                    source_auth_path.display().to_string(),
                    format!("auth.json is unreadable: {reason}"),
                );
                return leg;
            }
        }
    } else {
        fold_legacy_sources(discovery, report, &mut sources, &mut leg)
    };
    report.credential_sources = sources;

    // Validate and normalize every source entry, carrying the raw value so
    // unknown fields ride verbatim.
    let mut entries: Vec<(usize, String, Value, bool)> = Vec::new();
    for (provider, value) in &source_map {
        let index = report.credentials.len();
        report.credentials.push(CredentialItem {
            provider: provider.clone(),
            kind: value
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string),
            outcome: CredentialOutcome::Imported,
        });
        if value.is_null() {
            report.credentials[index].outcome = CredentialOutcome::Skipped {
                reason: "null entry".to_string(),
            };
            continue;
        }
        let (value, normalized) = normalize_expires(value.clone());
        if serde_json::from_value::<Credential>(value.clone()).is_ok() {
            report.credentials[index].outcome = if normalized {
                CredentialOutcome::Normalized
            } else {
                CredentialOutcome::Imported
            };
            entries.push((index, provider.clone(), value, normalized));
        } else {
            // The reason stays value-free: serde's own message echoes the
            // offending value, and a malformed entry's fields are exactly
            // the material the house rule never prints.
            let reason = "invalid credential: does not match the api_key or oauth shape";
            report.credentials[index].outcome = CredentialOutcome::Failed {
                reason: reason.to_string(),
            };
            report.push_failure(provider.clone(), reason);
        }
    }

    // Merge into the target store. In place the target file is the source
    // file when it exists: only the normalized entries change anything,
    // malformed ones stay untouched. The legacy fold lands even in place —
    // the target store does not exist yet. Cross-dir: existing providers
    // are kept, missing ones land.
    let target_auth_path = discovery.target.join("auth.json");
    let target_existed = target_auth_path.exists();
    let Ok(mut target_map) = read_target_map(&target_auth_path, report) else {
        // The target refused: nothing lands, so no validated entry may keep
        // its Imported outcome.
        for (index, _, _, _) in &entries {
            report.credentials[*index].outcome = CredentialOutcome::Failed {
                reason: "target auth.json is not a readable store".to_string(),
            };
        }
        return leg;
    };
    let mut changed = false;
    for (index, provider, value, normalized) in entries {
        if discovery.in_place && target_existed && !normalized {
            report.credentials[index].outcome = CredentialOutcome::Kept;
            continue;
        }
        if !discovery.in_place && target_map.contains_key(&provider) {
            report.credentials[index].outcome = CredentialOutcome::Kept;
            continue;
        }
        target_map.insert(provider, value);
        changed = true;
        leg.landed_items.push(index);
    }

    // The write: only when the target store's content changes, never for a
    // fully-present source. The backend rides the infallible constructor —
    // the tool's paths are plain filesystem paths, so the `file://`
    // normalization the default constructor serves has no input here.
    if changed {
        // Serializing a JSON map cannot fail; the default stands in for the
        // unreachable arm.
        let next = serde_json::to_string_pretty(&target_map).unwrap_or_default();
        leg.write = Some(AuthWritePlan {
            auth_path: target_auth_path.display().to_string(),
            next,
        });
    }
    leg
}

/// The write operation the runner plans from the leg's outcome, wiring the
/// failure targets.
#[must_use]
pub fn auth_write_op(plan: &AuthWritePlan, leg: &CredentialsLeg) -> PlannedOp {
    PlannedOp::AuthWrite {
        backend: FileAuthStorageBackend::with_lock_strategy(
            &plan.auth_path,
            std::sync::Arc::new(pi_coding_agent::file_lock::MkdirLock),
        ),
        next: plan.next.clone(),
        targets: leg
            .landed_items
            .iter()
            .map(|index| OpTarget::Credential(*index))
            .collect(),
    }
}

/// The legacy fold, upstream's `migrateAuthToAuthJson` merge rules without
/// its source mutations: `oauth.json` entries first, then
/// `settings.json#apiKeys` string values filling the gaps.
fn fold_legacy_sources(
    discovery: &Discovery,
    report: &mut ImportReport,
    sources: &mut Vec<String>,
    leg: &mut CredentialsLeg,
) -> Map<String, Value> {
    let mut folded = Map::new();
    let oauth_path = discovery.source.join("oauth.json");
    if oauth_path.exists() {
        match read_json_object(&oauth_path) {
            Ok(map) => {
                sources.push("oauth.json".to_string());
                for (provider, cred) in map {
                    let mut entry = Map::new();
                    entry.insert("type".to_string(), Value::String("oauth".to_string()));
                    if let Value::Object(fields) = cred {
                        // Upstream's spread order: the credential's own
                        // fields override the inserted tag.
                        for (field, value) in fields {
                            entry.insert(field, value);
                        }
                    }
                    folded.insert(provider, Value::Object(entry));
                }
            }
            Err(reason) => {
                report.push_failure(
                    oauth_path.display().to_string(),
                    format!("oauth.json is unreadable: {reason}"),
                );
            }
        }
    }
    let settings_path = discovery.source.join("settings.json");
    if settings_path.exists() {
        match read_json_object(&settings_path) {
            Ok(settings) => {
                if let Some(api_keys) = settings.get("apiKeys").and_then(Value::as_object) {
                    leg.settings_api_keys_consumed = true;
                    sources.push("settings.json apiKeys".to_string());
                    for (provider, key) in api_keys {
                        if folded.contains_key(provider) || !key.is_string() {
                            continue;
                        }
                        let mut entry = Map::new();
                        entry.insert("type".to_string(), Value::String("api_key".to_string()));
                        entry.insert("key".to_string(), key.clone());
                        folded.insert(provider.clone(), Value::Object(entry));
                    }
                }
            }
            Err(reason) => {
                report.push_failure(
                    settings_path.display().to_string(),
                    format!("settings.json apiKeys fold skipped: {reason}"),
                );
            }
        }
    }
    folded
}

/// The fractional-`expires` normalization: a float `expires` floors to the
/// integer the Rust typed read requires, in `i64` range; integral numbers
/// ride unchanged and out-of-range floats stay for validation to fail them.
fn normalize_expires(mut value: Value) -> (Value, bool) {
    if !value.get("expires").is_some_and(Value::is_f64) {
        return (value, false);
    }
    let is_oauth = value
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|tag| tag == "oauth");
    if !is_oauth {
        return (value, false);
    }
    let expires = value.get("expires").and_then(Value::as_f64).unwrap_or(0.0);
    let floored = expires.floor();
    // The i64 range's float-exact bound, 2^63: beyond it the floor cannot
    // fit and validation fails the entry instead.
    let limit = (2.0f64).powi(63);
    if floored < -limit || floored >= limit {
        return (value, false);
    }
    if let Some(object) = value.as_object_mut() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "floored is already integral; the cast carries it to the integer type the store reads"
        )]
        let integer = floored as i64;
        object.insert("expires".to_string(), Value::from(integer));
    }
    (value, true)
}

/// A JSON object file read, BOM-stripped.
fn read_json_object(path: &Path) -> Result<Map<String, Value>, String> {
    let content = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: Value =
        serde_json::from_str(strip_bom(&content)).map_err(|error| error.to_string())?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| "not a JSON object".to_string())
}

/// The target store's map, empty when absent; a present non-object target
/// records a leg-level failure and refuses the merge.
fn read_target_map(
    target_auth_path: &Path,
    report: &mut ImportReport,
) -> Result<Map<String, Value>, ()> {
    if !target_auth_path.exists() {
        return Ok(Map::new());
    }
    read_json_object(target_auth_path).map_err(|reason| {
        report.push_failure(
            target_auth_path.display().to_string(),
            format!("target auth.json is not a readable store: {reason}"),
        );
    })
}
