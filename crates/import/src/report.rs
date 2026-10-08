//! The migration report, the tool's answer to the report-UX decision
//! (import-tool ticket #13).
//!
//! The default render is a human-readable per-artifact summary, `--json`
//! emits the machine-readable form, and failures are itemized so a re-run
//! can target them.
//!
//! The outcome vocabularies name what the tool did per artifact — `copied`
//! and `present` for sessions, `imported`, `normalized`, and `kept` for
//! credential entries, `written` and `unchanged` for whole-file artifacts.
//! Skipped and failed outcomes carry the reason. The failures list is the
//! single store: the legs record each failure beside the item it came from,
//! so the rendered failures section cannot drift from the items.

use serde::Serialize;

/// One session file's import result, the item the sessions leg reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionItem {
    /// The session file's path, relative to the source agent dir.
    pub path: String,
    /// The header's format version; absent on v1 sessions.
    pub version: Option<i64>,
    /// What the tool did with the file.
    pub outcome: SessionOutcome,
}

/// What the tool did with one session file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SessionOutcome {
    /// Copied (or placed, for agent-root strays) into the target sessions
    /// tree; the string is the destination path.
    #[serde(rename_all = "camelCase")]
    Copied {
        /// The destination path the file landed at.
        placed_to: String,
    },
    /// Already at the target, validated only.
    Present,
    /// Not a session the tool carries; the string says why.
    #[serde(rename_all = "camelCase")]
    Skipped {
        /// Why the artifact was not carried.
        reason: String,
    },
    /// The tool could not carry the file; the string says why.
    #[serde(rename_all = "camelCase")]
    Failed {
        /// Why the tool could not carry the artifact.
        reason: String,
    },
}

/// One credential entry's import result, the item the credentials leg
/// reports. Provider ids and kinds only — never key material.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialItem {
    /// The provider id the credential is stored under.
    pub provider: String,
    /// The credential's type tag (`api_key` or `oauth`), when the entry
    /// validated far enough to have one.
    pub kind: Option<String>,
    /// What the tool did with the entry.
    pub outcome: CredentialOutcome,
}

/// What the tool did with one credential entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CredentialOutcome {
    /// Stored in the target auth.json verbatim.
    Imported,
    /// Stored after the fractional `expires` was floored to an integer, the
    /// one normalization the Rust typed read requires.
    Normalized,
    /// The target store already has this provider; the source entry was
    /// left out.
    Kept,
    /// The tool could not carry the entry; the string says why.
    #[serde(rename_all = "camelCase")]
    Failed {
        /// Why the tool could not carry the artifact.
        reason: String,
    },
    /// Nothing to carry; the string says why.
    #[serde(rename_all = "camelCase")]
    Skipped {
        /// Why the artifact was not carried.
        reason: String,
    },
}

/// One settings scope's import result, the item the settings leg reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsItem {
    /// The scope, `global` or `project`.
    pub scope: String,
    /// The file the scope's outcome is about — the target path when
    /// written, the source path otherwise.
    pub path: String,
    /// What the tool did with the scope.
    pub outcome: SettingsOutcome,
    /// The legacy-key migrations applied, `queueMode -> steeringMode` form.
    pub merged: Vec<String>,
    /// The keys removed without a replacement, `apiKeys (folded into
    /// auth.json)` form.
    pub dropped: Vec<String>,
    /// The key count the migrated settings carry.
    pub carried_keys: u64,
}

/// What the tool did with one settings scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SettingsOutcome {
    /// The migrated settings were written (or would be, under `--dry-run`).
    Written,
    /// The source settings already match what the target carries.
    Unchanged,
    /// The scope was not migrated; the string says why.
    #[serde(rename_all = "camelCase")]
    Skipped {
        /// Why the artifact was not carried.
        reason: String,
    },
    /// The tool could not migrate the scope; the string says why.
    #[serde(rename_all = "camelCase")]
    Failed {
        /// Why the tool could not carry the artifact.
        reason: String,
    },
}

/// The trust store's import result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrustItem {
    /// The trust.json path the outcome is about — the target path when
    /// written, the source path otherwise.
    pub path: String,
    /// What the tool did with the store.
    pub outcome: TrustOutcome,
}

/// What the tool did with the trust store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TrustOutcome {
    /// Copied to the target; the count is the validated entry count.
    #[serde(rename_all = "camelCase")]
    Written {
        /// The validated entry count.
        entries: u64,
    },
    /// Already at the target, validated only; the count is the validated
    /// entry count.
    #[serde(rename_all = "camelCase")]
    Unchanged {
        /// The validated entry count.
        entries: u64,
    },
    /// The store was not migrated; the string says why.
    #[serde(rename_all = "camelCase")]
    Skipped {
        /// Why the artifact was not carried.
        reason: String,
    },
    /// The tool could not migrate the store; the string says why.
    #[serde(rename_all = "camelCase")]
    Failed {
        /// Why the tool could not carry the artifact.
        reason: String,
    },
}

/// One TS artifact the Rust pi cannot carry, reported as a skipped item per
/// the import-tool decision: TS extensions and npm packages (ADR 0007).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkippedItem {
    /// The artifact's source — the path or the entry's identity.
    pub source: String,
    /// The artifact's class, `extension` or `package`.
    pub kind: String,
    /// Why the Rust pi cannot carry it.
    pub reason: String,
}

/// One itemized failure, the re-run target the report-UX decision names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Failure {
    /// The artifact that failed, the report's path for it.
    pub artifact: String,
    /// Why the tool failed.
    pub detail: String,
}

/// The whole migration report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ImportReport {
    /// The source agent dir.
    pub source: String,
    /// The target agent dir.
    pub target: String,
    /// The project dir when `--project` was given.
    pub project: Option<String>,
    /// Whether the run reported without writing.
    pub dry_run: bool,
    /// The sessions leg's items.
    pub sessions: Vec<SessionItem>,
    /// The credentials leg's items.
    pub credentials: Vec<CredentialItem>,
    /// The settings leg's items.
    pub settings: Vec<SettingsItem>,
    /// The trust leg's item, when the source has a trust store.
    pub trust: Option<TrustItem>,
    /// The TS extensions and npm packages reported as skipped.
    pub skipped: Vec<SkippedItem>,
    /// The source files the credential fold read from, `auth.json` or the
    /// legacy pair, in fold order.
    pub credential_sources: Vec<String>,
    /// The itemized failures; the legs keep each one beside the item it
    /// came from.
    pub(crate) failures: Vec<Failure>,
}

impl ImportReport {
    /// Record an itemized failure; the rendered report lists it in the
    /// failures section.
    pub fn push_failure(&mut self, artifact: impl Into<String>, detail: impl Into<String>) {
        self.failures.push(Failure {
            artifact: artifact.into(),
            detail: detail.into(),
        });
    }

    /// The itemized failures.
    #[must_use]
    pub fn failures(&self) -> &[Failure] {
        &self.failures
    }

    /// Whether anything failed, the non-zero-exit condition the report-UX
    /// decision fixes.
    #[must_use]
    pub const fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    /// Downgrade one session item to a failure, the write-failure path.
    pub fn fail_session(&mut self, index: usize, reason: &str) {
        let Some(item) = self.sessions.get_mut(index) else {
            return;
        };
        let path = item.path.clone();
        item.outcome = SessionOutcome::Failed {
            reason: reason.to_string(),
        };
        self.push_failure(path, reason);
    }

    /// Downgrade one credential item to a failure, the write-failure path.
    pub fn fail_credential(&mut self, index: usize, reason: &str) {
        let Some(item) = self.credentials.get_mut(index) else {
            return;
        };
        let provider = item.provider.clone();
        item.outcome = CredentialOutcome::Failed {
            reason: reason.to_string(),
        };
        self.push_failure(provider, reason);
    }

    /// Downgrade one settings item to a failure, the write-failure path.
    pub fn fail_settings(&mut self, index: usize, reason: &str) {
        let Some(item) = self.settings.get_mut(index) else {
            return;
        };
        let path = item.path.clone();
        item.outcome = SettingsOutcome::Failed {
            reason: reason.to_string(),
        };
        self.push_failure(path, reason);
    }

    /// Downgrade the trust item to a failure, the write-failure path.
    pub fn fail_trust(&mut self, reason: &str) {
        let Some(item) = self.trust.as_mut() else {
            return;
        };
        let path = item.path.clone();
        item.outcome = TrustOutcome::Failed {
            reason: reason.to_string(),
        };
        self.push_failure(path, reason);
    }

    /// The report as the machine-readable JSON value, `--json`'s output.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}
