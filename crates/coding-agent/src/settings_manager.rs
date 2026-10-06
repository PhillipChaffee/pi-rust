//! The settings manager, upstream's
//! `packages/coding-agent/src/core/settings-manager.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Global settings deep-merge with project `<cwd>/.pi/settings.json` behind
//! per-scope file locks, with external-edit preservation — only the fields a
//! session modified override the file on save, nested-key tracked.
//!
//! Porting restatements this module records:
//!
//! - The typed `Settings` interface is a compile-time view over an arbitrary
//!   JSON object upstream; the port carries the same JSON map and rides the
//!   accessors for every typed read and write, so unknown keys in a user's
//!   file survive every save the way upstream's spread-and-stringify does.
//!   Key order is preserved (`preserve_order`), the `JSON.stringify`
//!   insertion order the saved files carry.
//! - The write queue restates as synchronous persistence: upstream's queued
//!   tasks serialize promise-chain writes over async fs, and the port's
//!   writes are sync, so `save` persists inline and `flush` waits for
//!   nothing. The queue's error path is kept — a failed write records the
//!   error and leaves the modified-field tracking intact, the retry the
//!   catch-less chain implies.
//! - `randomUUID` restates on the `uuid` crate for the analytics tracking
//!   identifier. The environment reads (`VISUAL`/`EDITOR`,
//!   `PI_CLEAR_ON_SHRINK`, `PI_HARDWARE_CURSOR`) are injectable through the
//!   `_with_env` getters; the plain getters read the process environment.
//! - The http-dispatcher slice this module consumes —
//!   [`DEFAULT_HTTP_IDLE_TIMEOUT_MS`] and [`parse_http_idle_timeout_ms`] —
//!   carries here; the undici dispatcher machinery and the choice
//!   formatting ride their consumers.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};
use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::Transport;
use pi_ai::utils::retry::DEFAULT_MAX_AGENT_RETRY_DELAY_MS;
use pi_tui::components::scroll_view::ScrollViewScrollbar;
use pi_tui::terminal_image::CapabilityOverrides;
use pi_tui::tui::TuiMode;
use serde_json::Value;

use crate::config::{CONFIG_DIR_NAME, EnvLookup, default_env_lookup};
use crate::file_lock::{acquire_sync_retrying, lock_dir_for};
use crate::utils::paths::{PathInputOptions, normalize_path, resolve_path};
use crate::utils::text::strip_bom;

/// The default HTTP idle timeout, upstream's
/// `core/http-dispatcher.ts` constant: five minutes.
pub const DEFAULT_HTTP_IDLE_TIMEOUT_MS: i64 = 300_000;

/// Parse an HTTP idle timeout value, upstream's
/// `core/http-dispatcher.ts` `parseHttpIdleTimeoutMs`.
///
/// A `"disabled"` string is zero, a blank string is unset, anything
/// non-finite or negative is unset, and everything else floors.
#[must_use]
pub fn parse_http_idle_timeout_ms(value: &Value) -> Option<i64> {
    if let Value::String(text) = value {
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("disabled") {
            return Some(0);
        }
        if trimmed.is_empty() {
            return None;
        }
        return trimmed
            .parse::<f64>()
            .ok()
            .and_then(parse_finite_timeout_ms);
    }
    parse_finite_timeout_ms(value.as_f64()?)
}

fn parse_finite_timeout_ms(value: f64) -> Option<i64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Math.floor restatement: the value is checked finite and non-negative immediately above"
    )]
    Some(value.floor() as i64)
}

/// The compaction token settings, upstream's `CompactionSettings` with the
/// built-in defaults.
const DEFAULT_COMPACTION_RESERVE_TOKENS: i64 = 16384;
const DEFAULT_COMPACTION_KEEP_RECENT_TOKENS: i64 = 20000;

/// The model a per-model setting resolves for, upstream's
/// `Pick<Model<string>, "provider" | "id">` — the provider id and model id
/// pair the override keys spell `"provider/modelId"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelKey {
    /// The provider id.
    pub provider: String,
    /// The model id.
    pub id: String,
}

impl ModelKey {
    /// The override key, upstream's `${model.provider}/${model.id}`.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }
}

/// The compaction settings' resolved shape, upstream's
/// `getCompactionSettings` return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionSettings {
    /// Whether compaction runs.
    pub enabled: bool,
    /// Tokens reserved for prompt and LLM response.
    pub reserve_tokens: i64,
    /// Recent tokens a compaction keeps.
    pub keep_recent_tokens: i64,
}

/// The branch-summary settings' resolved shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchSummarySettings {
    /// Tokens reserved for prompt and LLM response.
    pub reserve_tokens: i64,
    /// Whether the "Summarize branch?" prompt is skipped.
    pub skip_prompt: bool,
}

/// The retry settings' resolved shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrySettings {
    /// Whether agent-loop retries run.
    pub enabled: bool,
    /// SDK/provider retry attempts.
    pub max_retries: i64,
    /// Exponential-backoff base delay.
    pub base_delay_ms: i64,
    /// The agent retry delay cap.
    pub max_agent_delay_ms: i64,
}

/// The provider retry settings' resolved shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderRetrySettings {
    /// The SDK/provider request timeout, when set.
    pub timeout_ms: Option<i64>,
    /// The SDK/provider retry attempts, when set.
    pub max_retries: Option<i64>,
    /// The max server-requested delay before failing.
    pub max_retry_delay_ms: i64,
}

/// The steering (and follow-up) queue mode, upstream's `"all" |
/// "one-at-a-time"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMode {
    /// All queued items accepted.
    All,
    /// One queued item at a time.
    OneAtATime,
}

impl QueueMode {
    /// The wire string, upstream's stored value.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::OneAtATime => "one-at-a-time",
        }
    }

    fn parse(value: &Value) -> Option<Self> {
        match value.as_str() {
            Some("all") => Some(Self::All),
            Some("one-at-a-time") => Some(Self::OneAtATime),
            _ => None,
        }
    }
}

/// The default project-trust posture, upstream's `DefaultProjectTrust`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultProjectTrust {
    /// Ask on every untrusted project.
    Ask,
    /// Always trust.
    Always,
    /// Never trust.
    Never,
}

impl DefaultProjectTrust {
    /// The wire string, upstream's stored value.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Always => "always",
            Self::Never => "never",
        }
    }
}

/// The fullscreen exit output, upstream's `FullscreenExitOutput`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullscreenExitOutput {
    /// Show the transcript on exit.
    Transcript,
    /// Show the resume hint on exit.
    ResumeHint,
}

/// The mermaid rendering mode, upstream's `MermaidRenderingMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MermaidRenderingMode {
    /// Never render.
    Off,
    /// Render when a message completes.
    Final,
    /// Render while streaming.
    Streaming,
}

/// One scope of settings, upstream's `SettingsScope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsScope {
    /// The agent dir's `settings.json`.
    Global,
    /// The project's `.pi/settings.json`.
    Project,
}

/// The create options, upstream's `SettingsManagerCreateOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SettingsManagerCreateOptions {
    /// Whether the project's settings load at all; `true` when unset.
    pub project_trusted: Option<bool>,
}

/// One settings failure, upstream's `SettingsError`: the scope, the file
/// path when the storage reported one, and the error message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsError {
    /// The scope the failure came from.
    pub scope: SettingsScope,
    /// The file path, when the storage is file-backed.
    pub path: Option<String>,
    /// The failure message, upstream's `error.message`.
    pub message: String,
}

/// The locked operation's closure, upstream's `(current) => string |
/// undefined`.
///
/// Read the scope's serialized content, return the next content or `None`
/// to leave the file untouched. The `Err` half carries the operation's own
/// failure, upstream's throw out of the fn.
pub type SettingsLockFn<'a> = &'a mut dyn FnMut(Option<&str>) -> Result<Option<String>, String>;

/// The storage surface, upstream's `SettingsStorage`: a locked read of the
/// scope's serialized content with an optional write-back.
pub trait SettingsStorage: Send + Sync {
    /// Run the locked operation.
    ///
    /// # Errors
    /// A lock, read, or write failure — the caller records them.
    fn with_lock(&self, scope: SettingsScope, f: SettingsLockFn<'_>) -> Result<(), String>;
}

/// The settings paths the error records carry, upstream's `SettingsPaths`.
#[derive(Debug, Default, Clone)]
pub struct SettingsPaths {
    /// The global settings file path, when file-backed.
    pub global: Option<String>,
    /// The project settings file path, when file-backed.
    pub project: Option<String>,
}

/// The file-backed storage, upstream's `FileSettingsStorage`: the global
/// path under the agent dir, the project path under `<cwd>/.pi`.
pub struct FileSettingsStorage {
    global_settings_path: String,
    project_settings_path: String,
}

impl std::fmt::Debug for FileSettingsStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSettingsStorage")
            .field("global_settings_path", &self.global_settings_path)
            .field("project_settings_path", &self.project_settings_path)
            .finish()
    }
}

impl FileSettingsStorage {
    /// The storage over a resolved cwd and agent dir, upstream's
    /// constructor.
    #[must_use]
    pub fn new(cwd: &str, agent_dir: &str) -> Self {
        Self {
            global_settings_path: format!("{agent_dir}/settings.json"),
            project_settings_path: format!("{cwd}/{CONFIG_DIR_NAME}/settings.json"),
        }
    }

    fn path_for(&self, scope: SettingsScope) -> &str {
        match scope {
            SettingsScope::Global => &self.global_settings_path,
            SettingsScope::Project => &self.project_settings_path,
        }
    }
}

impl SettingsStorage for FileSettingsStorage {
    fn with_lock(&self, scope: SettingsScope, f: SettingsLockFn<'_>) -> Result<(), String> {
        let path = self.path_for(scope);
        let lock_dir = lock_dir_for(path);
        // Only create a directory and lock when the file exists or a write
        // is coming, upstream's lazy locking.
        let mut release = None;
        let file_exists = Path::new(path).exists();
        if file_exists {
            release = Some(acquire_sync_retrying(&lock_dir).map_err(|error| error.to_string())?);
        }
        // The outcome is computed before the release so a failed callback or
        // a failed read does not leave the lock directory behind — upstream
        // releases in a `finally`.
        let outcome = (|| {
            let current = if file_exists {
                Some(std::fs::read_to_string(path).map_err(|error| error.to_string())?)
            } else {
                None
            };
            let next = f(current.as_deref())?;
            if let Some(next) = next {
                if let Some(parent) = Path::new(path).parent()
                    && !parent.exists()
                {
                    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                }
                if release.is_none() {
                    release =
                        Some(acquire_sync_retrying(&lock_dir).map_err(|error| error.to_string())?);
                }
                std::fs::write(path, next).map_err(|error| error.to_string())?;
            }
            Ok::<(), String>(())
        })();
        if let Some(guard) = release {
            guard.release().map_err(|error| error.to_string())?;
        }
        outcome
    }
}

/// The in-memory storage, upstream's `InMemorySettingsStorage`.
#[derive(Debug, Default)]
pub struct InMemorySettingsStorage {
    global: MutexSlot,
    project: MutexSlot,
}

type MutexSlot = std::sync::Mutex<Option<String>>;

impl InMemorySettingsStorage {
    const fn slot(&self, scope: SettingsScope) -> &MutexSlot {
        match scope {
            SettingsScope::Global => &self.global,
            SettingsScope::Project => &self.project,
        }
    }

    /// Seed a scope's content directly, the test-fixture form upstream's
    /// `storage.withLock(scope, () => content)` calls take.
    pub fn seed(&self, scope: SettingsScope, content: String) {
        *self
            .slot(scope)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(content);
    }
}

impl SettingsStorage for InMemorySettingsStorage {
    fn with_lock(&self, scope: SettingsScope, f: SettingsLockFn<'_>) -> Result<(), String> {
        let mut slot = self
            .slot(scope)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let next = f(slot.as_deref())?;
        if let Some(next) = next {
            *slot = Some(next);
        }
        drop(slot);
        Ok(())
    }
}

/// Deep-merge two JSON objects, upstream's `deepMergeObjects`: nested objects
/// merge recursively, everything else — including arrays and `null` —
/// replaces.
fn deep_merge_objects(base: &Settings, overrides: &Settings) -> Settings {
    let mut result = base.clone();
    for (key, override_value) in overrides {
        let Some(base_value) = base.get(key) else {
            result.insert(key.clone(), override_value.clone());
            continue;
        };
        let merged = match (base_value.as_object(), override_value.as_object()) {
            (Some(base_object), Some(override_object)) => {
                Value::Object(deep_merge_objects(base_object, override_object))
            }
            _ => override_value.clone(),
        };
        result.insert(key.clone(), merged);
    }
    result
}

/// Deep-merge settings, upstream's `deepMergeSettings`: project/overrides
/// take precedence, nested objects merge recursively.
fn deep_merge_settings(base: &Settings, overrides: &Settings) -> Settings {
    deep_merge_objects(base, overrides)
}

/// Upstream's `String(value)` for the validation messages: numbers format
/// JS-style through serde's display, objects read `[object Object]`, arrays
/// read empty.
fn js_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(inner) => if *inner { "true" } else { "false" }.to_string(),
        Value::Number(inner) => inner.to_string(),
        Value::String(inner) => inner.clone(),
        Value::Object(_) => "[object Object]".to_string(),
        Value::Array(_) => String::new(),
    }
}

/// Whether the value merges as an object, upstream's `isMergeableObject`.
fn is_mergeable_object(value: &Value) -> bool {
    value.is_object()
}

/// Whether the number is a non-negative safe integer, upstream's
/// `Number.isSafeInteger` bound plus the sign check.
fn is_non_negative_safe_integer(value: &Value) -> bool {
    let Some(number) = value.as_i64() else {
        return false;
    };
    (0..(1i64 << 53)).contains(&number)
}

/// Migrate old settings formats, upstream's `migrateSettings`: queueMode →
/// steeringMode, the websockets boolean → transport, the skills object →
/// array, and retry.maxDelayMs → retry.provider.maxRetryDelayMs.
fn migrate_settings(settings: &mut Settings) {
    if settings.contains_key("queueMode") && !settings.contains_key("steeringMode") {
        let queue_mode = settings.get("queueMode").cloned();
        if let Some(queue_mode) = queue_mode {
            settings.insert("steeringMode".to_string(), queue_mode);
        }
        settings.shift_remove("queueMode");
    }

    if !settings.contains_key("transport")
        && settings.get("websockets").is_some_and(Value::is_boolean)
    {
        let websockets = settings.get("websockets").and_then(Value::as_bool);
        settings.insert(
            "transport".to_string(),
            Value::String(
                if websockets.unwrap_or(false) {
                    "websocket"
                } else {
                    "sse"
                }
                .to_string(),
            ),
        );
        settings.shift_remove("websockets");
    }

    if settings
        .get("skills")
        .is_some_and(|skills| skills.is_object() && !skills.is_array())
    {
        let skills_settings = settings.get("skills").and_then(Value::as_object).cloned();
        if let Some(skills_settings) = skills_settings {
            let enable = skills_settings.get("enableSkillCommands").cloned();
            if let Some(enable) = enable
                && !settings.contains_key("enableSkillCommands")
            {
                settings.insert("enableSkillCommands".to_string(), enable);
            }
            let custom = skills_settings
                .get("customDirectories")
                .and_then(Value::as_array)
                .cloned();
            match custom {
                Some(custom) if !custom.is_empty() => {
                    settings.insert("skills".to_string(), Value::Array(custom));
                }
                _ => {
                    settings.shift_remove("skills");
                }
            }
        }
    }

    if settings
        .get("retry")
        .is_some_and(|retry| retry.is_object() && !retry.is_array())
    {
        let retry_object = settings.get_mut("retry").and_then(Value::as_object_mut);
        if let Some(retry_object) = retry_object {
            // The number is carried as its original JSON value, not refloated
            // through f64: upstream copies the JS number, whose stringify
            // keeps `500` integral, and a refloat would both rewrite the
            // saved text and drop the value out of the `as_i64` readers.
            let max_delay = retry_object.get("maxDelayMs").cloned();
            let provider = retry_object
                .get("provider")
                .and_then(Value::as_object)
                .cloned();
            let provider_has_override = provider
                .as_ref()
                .and_then(|provider| provider.get("maxRetryDelayMs"))
                .is_some_and(|value| !value.is_null());
            if let Some(max_delay) = max_delay.filter(Value::is_number)
                && !provider_has_override
            {
                let mut next_provider = provider.unwrap_or_default();
                next_provider.insert("maxRetryDelayMs".to_string(), max_delay);
                retry_object.insert("provider".to_string(), Value::Object(next_provider));
            }
            // Upstream deletes maxDelayMs whenever retry is an object —
            // migrated or not, over a provider override or not.
            retry_object.shift_remove("maxDelayMs");
        }
    }
}

/// The modified-field tracking, upstream's `modifiedFields` set and
/// `modifiedNestedFields` map: insertion-ordered, nested keys per field.
#[derive(Debug, Default, Clone)]
struct ModifiedFields {
    fields: IndexSet<String>,
    nested: IndexMap<String, IndexSet<String>>,
}

impl ModifiedFields {
    fn mark(&mut self, field: &str, nested_key: Option<&str>) {
        self.fields.insert(field.to_string());
        if let Some(nested_key) = nested_key {
            self.nested
                .entry(field.to_string())
                .or_default()
                .insert(nested_key.to_string());
        }
    }

    fn clear(&mut self) {
        self.fields.clear();
        self.nested.clear();
    }
}

/// The manager, upstream's `SettingsManager`.
pub struct SettingsManager<S: SettingsStorage> {
    storage: Arc<S>,
    global_settings: Settings,
    project_settings: Settings,
    settings: Settings,
    project_trusted: bool,
    modified: ModifiedFields,
    modified_project: ModifiedFields,
    global_load_error: Option<String>,
    project_load_error: Option<String>,
    errors: Vec<SettingsError>,
    settings_paths: SettingsPaths,
}

impl<S: SettingsStorage> std::fmt::Debug for SettingsManager<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingsManager")
            .field("project_trusted", &self.project_trusted)
            .finish_non_exhaustive()
    }
}

/// A settings object, the JSON map the accessors read and write.
pub type Settings = serde_json::Map<String, Value>;

impl<S: SettingsStorage> SettingsManager<S> {
    fn new(
        storage: Arc<S>,
        global: LoadedSettings,
        project: LoadedSettings,
        initial_errors: Vec<SettingsError>,
        project_trusted: bool,
        settings_paths: SettingsPaths,
    ) -> Self {
        let LoadedSettings {
            settings: global_settings,
            error: global_load_error,
        } = global;
        let LoadedSettings {
            settings: project_settings,
            error: project_load_error,
        } = project;
        let settings = deep_merge_settings(&global_settings, &project_settings);
        Self {
            storage,
            global_settings,
            project_settings,
            settings,
            project_trusted,
            modified: ModifiedFields::default(),
            modified_project: ModifiedFields::default(),
            global_load_error,
            project_load_error,
            errors: initial_errors,
            settings_paths,
        }
    }

    /// The manager over an arbitrary storage, upstream's `fromStorage`.
    #[must_use]
    pub fn from_storage(storage: S, options: SettingsManagerCreateOptions) -> Self {
        Self::from_storage_with_paths(storage, options, SettingsPaths::default())
    }

    /// The manager over an arbitrary storage with the error-reporting paths,
    /// upstream's private `fromStorageWithPaths`.
    #[must_use]
    pub fn from_storage_with_paths(
        storage: S,
        options: SettingsManagerCreateOptions,
        settings_paths: SettingsPaths,
    ) -> Self {
        let storage = Arc::new(storage);
        let project_trusted = options.project_trusted.unwrap_or(true);
        let global_load = Self::try_load_from_storage(&storage, SettingsScope::Global, true);
        let project_load =
            Self::try_load_from_storage(&storage, SettingsScope::Project, project_trusted);
        let mut initial_errors = Vec::new();
        if let Some(error) = &global_load.error {
            initial_errors.push(SettingsError {
                scope: SettingsScope::Global,
                path: settings_paths.global.clone(),
                message: error.clone(),
            });
        }
        if let Some(error) = &project_load.error {
            initial_errors.push(SettingsError {
                scope: SettingsScope::Project,
                path: settings_paths.project.clone(),
                message: error.clone(),
            });
        }
        Self::new(
            storage,
            global_load,
            project_load,
            initial_errors,
            project_trusted,
            settings_paths,
        )
    }

    fn load_from_storage(
        storage: &S,
        scope: SettingsScope,
        project_trusted: bool,
    ) -> Result<Settings, String> {
        if scope == SettingsScope::Project && !project_trusted {
            return Ok(Settings::new());
        }
        let mut content: Option<String> = None;
        storage.with_lock(scope, &mut |current| {
            content = current.map(str::to_string);
            Ok(None)
        })?;
        let Some(content) = content else {
            return Ok(Settings::new());
        };
        let parsed: Value = serde_json::from_str(strip_bom(&content))
            .map_err(|error| format!("Failed to parse settings: {error}"))?;
        let mut settings = match parsed {
            Value::Object(settings) => settings,
            // Upstream's cast tolerates a non-object by the same accident its
            // spread does: no keys survive into the merge.
            _ => Settings::new(),
        };
        migrate_settings(&mut settings);
        Ok(settings)
    }

    fn try_load_from_storage(
        storage: &S,
        scope: SettingsScope,
        project_trusted: bool,
    ) -> LoadedSettings {
        match Self::load_from_storage(storage, scope, project_trusted) {
            Ok(settings) => LoadedSettings {
                settings,
                error: None,
            },
            Err(error) => LoadedSettings {
                settings: Settings::new(),
                error: Some(error),
            },
        }
    }

    /// The merged settings view, upstream's private `settings`.
    #[must_use]
    pub const fn merged(&self) -> &Settings {
        &self.settings
    }

    /// The global settings clone, upstream's `getGlobalSettings`.
    #[must_use]
    pub fn get_global_settings(&self) -> Settings {
        self.global_settings.clone()
    }

    /// The project settings clone, upstream's `getProjectSettings`.
    #[must_use]
    pub fn get_project_settings(&self) -> Settings {
        self.project_settings.clone()
    }

    /// Whether project settings load and write, upstream's
    /// `isProjectTrusted`.
    #[must_use]
    pub const fn is_project_trusted(&self) -> bool {
        self.project_trusted
    }

    /// Flip the project trust, upstream's `setProjectTrusted`: untrusting
    /// drops the project view, trusting reloads it.
    pub fn set_project_trusted(&mut self, trusted: bool) {
        if self.project_trusted == trusted {
            return;
        }
        self.project_trusted = trusted;
        self.modified_project.clear();
        if !trusted {
            self.project_settings = Settings::new();
            self.project_load_error = None;
            self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
            return;
        }
        let project_load =
            Self::try_load_from_storage(&self.storage, SettingsScope::Project, trusted);
        self.project_settings = project_load.settings;
        self.project_load_error = project_load.error.clone();
        if let Some(error) = project_load.error {
            self.record_error(SettingsScope::Project, error);
        }
        self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
    }

    /// Reload both scopes from storage, upstream's `reload`: a failed scope
    /// keeps its previous settings and records the error.
    pub fn reload(&mut self) {
        let global_load = Self::try_load_from_storage(&self.storage, SettingsScope::Global, true);
        if global_load.error.is_none() {
            self.global_settings = global_load.settings;
            self.global_load_error = None;
        } else {
            self.global_load_error = global_load.error.clone();
            if let Some(error) = global_load.error {
                self.record_error(SettingsScope::Global, error);
            }
        }

        self.modified.clear();
        self.modified_project.clear();

        let project_load = Self::try_load_from_storage(
            &self.storage,
            SettingsScope::Project,
            self.project_trusted,
        );
        if project_load.error.is_none() {
            self.project_settings = project_load.settings;
            self.project_load_error = None;
        } else {
            self.project_load_error = project_load.error.clone();
            if let Some(error) = project_load.error {
                self.record_error(SettingsScope::Project, error);
            }
        }

        self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
    }

    /// Apply additional overrides on top of the current settings, upstream's
    /// `applyOverrides`.
    pub fn apply_overrides(&mut self, overrides: &Settings) {
        self.settings = deep_merge_settings(&self.settings, overrides);
    }

    /// Wait for pending writes, upstream's `flush`: the port's writes are
    /// synchronous, so this is the shape the awaits keep.
    #[expect(
        clippy::unused_async,
        reason = "the async shape keeps upstream's awaitable flush contract at the call sites"
    )]
    pub async fn flush(&self) {}

    /// Drain the recorded errors, upstream's `drainErrors`.
    pub fn drain_errors(&mut self) -> Vec<SettingsError> {
        std::mem::take(&mut self.errors)
    }

    fn record_error(&mut self, scope: SettingsScope, message: String) {
        self.errors.push(SettingsError {
            scope,
            path: match scope {
                SettingsScope::Global => self.settings_paths.global.clone(),
                SettingsScope::Project => self.settings_paths.project.clone(),
            },
            message,
        });
    }

    fn assert_project_trusted_for_write(&self) -> Result<(), String> {
        if !self.project_trusted {
            return Err("Project is not trusted; refusing to write project settings".to_string());
        }
        Ok(())
    }

    fn persist_scoped_settings(
        &self,
        scope: SettingsScope,
        snapshot: &Settings,
        modified: &ModifiedFields,
    ) -> Result<(), String> {
        let mut write = |current: Option<&str>| -> Result<Option<String>, String> {
            let current_file_settings = match current {
                Some(content) if !content.is_empty() => {
                    let parsed: Value = serde_json::from_str(strip_bom(content))
                        .map_err(|error| format!("Failed to parse settings: {error}"))?;
                    match parsed {
                        Value::Object(settings) => {
                            let mut settings = settings;
                            migrate_settings(&mut settings);
                            settings
                        }
                        _ => Settings::new(),
                    }
                }
                _ => Settings::new(),
            };
            let mut merged = current_file_settings.clone();
            for field in &modified.fields {
                let value = snapshot.get(field);
                if modified.nested.contains_key(field) && value.is_some_and(Value::is_object) {
                    let nested_modified = &modified.nested[field];
                    let base_nested = current_file_settings
                        .get(field)
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    let in_memory_nested = value.and_then(Value::as_object);
                    let mut merged_nested = base_nested;
                    for nested_key in nested_modified {
                        match in_memory_nested.and_then(|nested| nested.get(nested_key)) {
                            Some(nested_value) => {
                                merged_nested.insert(nested_key.clone(), nested_value.clone());
                            }
                            // Upstream assigns undefined for a nested key the
                            // session removed, and JSON.stringify omits it.
                            None => {
                                merged_nested.shift_remove(nested_key);
                            }
                        }
                    }
                    merged.insert(field.clone(), Value::Object(merged_nested));
                } else {
                    match value {
                        Some(value) => {
                            merged.insert(field.clone(), value.clone());
                        }
                        // Upstream assigns undefined for a field the session
                        // removed, and JSON.stringify omits it.
                        None => {
                            merged.shift_remove(field);
                        }
                    }
                }
            }
            Ok(Some(
                serde_json::to_string_pretty(&merged).unwrap_or_else(|_| "{}".to_string()),
            ))
        };
        self.storage.with_lock(scope, &mut write)
    }

    /// Persist the global scope, upstream's `save`: a load error suppresses
    /// the write; a write failure records and keeps the modified tracking
    /// for the next attempt.
    fn save(&mut self) {
        self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
        if self.global_load_error.is_some() {
            return;
        }
        let snapshot = self.global_settings.clone();
        let modified = self.modified.clone();
        match self.persist_scoped_settings(SettingsScope::Global, &snapshot, &modified) {
            Ok(()) => self.modified.clear(),
            Err(message) => self.record_error(SettingsScope::Global, message),
        }
    }

    /// Persist the project scope with new settings, upstream's
    /// `saveProjectSettings`.
    fn save_project_settings(&mut self, settings: Settings) -> Result<(), String> {
        self.assert_project_trusted_for_write()?;
        self.project_settings = settings;
        self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
        if self.project_load_error.is_some() {
            return Ok(());
        }
        let snapshot = self.project_settings.clone();
        let modified = self.modified_project.clone();
        let outcome = self.persist_scoped_settings(SettingsScope::Project, &snapshot, &modified);
        match outcome {
            Ok(()) => self.modified_project.clear(),
            Err(message) => self.record_error(SettingsScope::Project, message),
        }
        Ok(())
    }

    /// Update one project field through a mutated clone, upstream's
    /// `updateProjectSettings`.
    fn update_project_settings(
        &mut self,
        field: &str,
        update: impl FnOnce(&mut Settings),
    ) -> Result<(), String> {
        self.assert_project_trusted_for_write()?;
        let mut project_settings = self.project_settings.clone();
        update(&mut project_settings);
        self.modified_project.mark(field, None);
        self.save_project_settings(project_settings)
    }

    // =========================================================================
    // Typed accessors
    // =========================================================================

    fn set_global(&mut self, key: &str, value: Value, nested: Option<&str>) {
        match nested {
            None => {
                self.global_settings.insert(key.to_string(), value);
            }
            Some(nested_key) => {
                let entry = self
                    .global_settings
                    .entry(key.to_string())
                    .or_insert_with(|| Value::Object(Settings::new()));
                if let Some(object) = entry.as_object_mut() {
                    object.insert(nested_key.to_string(), value);
                }
            }
        }
        self.modified.mark(key, nested);
        self.save();
    }

    /// Clear a global field, upstream's optional setters assigning `undefined`:
    /// the key drops from the saved file instead of serializing as `null`.
    fn remove_global(&mut self, key: &str) {
        self.global_settings.shift_remove(key);
        self.modified.mark(key, None);
        self.save();
    }

    fn merged_get(&self, key: &str) -> Option<&Value> {
        self.settings.get(key)
    }

    fn merged_get_nested(&self, key: &str, nested_key: &str) -> Option<&Value> {
        self.settings
            .get(key)
            .and_then(|value| value.get(nested_key))
    }

    /// The last-seen changelog version, upstream's `getLastChangelogVersion`.
    #[must_use]
    pub fn get_last_changelog_version(&self) -> Option<String> {
        self.merged_get("lastChangelogVersion")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// Record the changelog version, upstream's `setLastChangelogVersion`.
    pub fn set_last_changelog_version(&mut self, version: &str) {
        self.set_global(
            "lastChangelogVersion",
            Value::String(version.to_string()),
            None,
        );
    }

    /// The session directory, upstream's `getSessionDir` — tilde-expanded
    /// through `normalizePath` when set.
    #[must_use]
    pub fn get_session_dir(&self) -> Option<String> {
        self.merged_get("sessionDir")
            .and_then(Value::as_str)
            .map(|session_dir| {
                normalize_path(session_dir, &PathInputOptions::default())
                    .unwrap_or_else(|_| session_dir.to_string())
            })
    }

    /// The default provider, upstream's `getDefaultProvider`.
    #[must_use]
    pub fn get_default_provider(&self) -> Option<String> {
        self.merged_get("defaultProvider")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// The default model, upstream's `getDefaultModel`.
    #[must_use]
    pub fn get_default_model(&self) -> Option<String> {
        self.merged_get("defaultModel")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// Set the default provider, upstream's `setDefaultProvider`.
    pub fn set_default_provider(&mut self, provider: &str) {
        self.set_global("defaultProvider", Value::String(provider.to_string()), None);
    }

    /// Set the default model, upstream's `setDefaultModel`.
    pub fn set_default_model(&mut self, model_id: &str) {
        self.set_global("defaultModel", Value::String(model_id.to_string()), None);
    }

    /// Set both defaults, upstream's `setDefaultModelAndProvider`.
    pub fn set_default_model_and_provider(&mut self, provider: &str, model_id: &str) {
        self.set_global("defaultProvider", Value::String(provider.to_string()), None);
        self.set_global("defaultModel", Value::String(model_id.to_string()), None);
    }

    /// The steering mode, upstream's `getSteeringMode`.
    #[must_use]
    pub fn get_steering_mode(&self) -> QueueMode {
        self.merged_get("steeringMode")
            .and_then(QueueMode::parse)
            .unwrap_or(QueueMode::OneAtATime)
    }

    /// Set the steering mode, upstream's `setSteeringMode`.
    pub fn set_steering_mode(&mut self, mode: QueueMode) {
        self.set_global(
            "steeringMode",
            Value::String(mode.as_str().to_string()),
            None,
        );
    }

    /// The follow-up mode, upstream's `getFollowUpMode`.
    #[must_use]
    pub fn get_follow_up_mode(&self) -> QueueMode {
        self.merged_get("followUpMode")
            .and_then(QueueMode::parse)
            .unwrap_or(QueueMode::OneAtATime)
    }

    /// Set the follow-up mode, upstream's `setFollowUpMode`.
    pub fn set_follow_up_mode(&mut self, mode: QueueMode) {
        self.set_global(
            "followUpMode",
            Value::String(mode.as_str().to_string()),
            None,
        );
    }

    /// The raw theme setting, upstream's `getThemeSetting`: any non-string
    /// reads as unset.
    #[must_use]
    pub fn get_theme_setting(&self) -> Option<String> {
        self.merged_get("theme")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// The fixed theme name, upstream's `getTheme`: slash-separated
    /// automatic theme settings read as unset.
    #[must_use]
    pub fn get_theme(&self) -> Option<String> {
        self.get_theme_setting()
            .filter(|theme| !theme.contains('/'))
    }

    /// Set the theme, upstream's `setTheme`.
    pub fn set_theme(&mut self, theme: &str) {
        self.set_global("theme", Value::String(theme.to_string()), None);
    }

    /// The default thinking level, upstream's `getDefaultThinkingLevel`.
    #[must_use]
    pub fn get_default_thinking_level(&self) -> Option<ThinkingLevel> {
        serde_json::from_value(self.merged_get("defaultThinkingLevel").cloned()?).ok()
    }

    /// Set the default thinking level, upstream's
    /// `setDefaultThinkingLevel`.
    pub fn set_default_thinking_level(&mut self, level: ThinkingLevel) {
        let value = serde_json::to_value(level).unwrap_or(Value::Null);
        self.set_global("defaultThinkingLevel", value, None);
    }

    /// One model's thinking-level override, upstream's
    /// `getModelThinkingLevel`.
    #[must_use]
    pub fn get_model_thinking_level(
        &self,
        provider: &str,
        model_id: &str,
    ) -> Option<ThinkingLevel> {
        let levels = self.merged_get("modelThinkingLevels")?;
        serde_json::from_value(levels.get(format!("{provider}/{model_id}")).cloned()?).ok()
    }

    /// Every model's thinking-level overrides, upstream's
    /// `getAllModelThinkingLevels`.
    #[must_use]
    pub fn get_all_model_thinking_levels(&self) -> Settings {
        self.merged_get("modelThinkingLevels")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    }

    /// Set one model's thinking-level override, upstream's
    /// `setModelThinkingLevel`.
    pub fn set_model_thinking_level(
        &mut self,
        provider: &str,
        model_id: &str,
        level: ThinkingLevel,
    ) {
        let value = serde_json::to_value(level).unwrap_or(Value::Null);
        self.set_global(
            "modelThinkingLevels",
            value,
            Some(&format!("{provider}/{model_id}")),
        );
    }

    /// Remove one model's thinking-level override, upstream's
    /// `removeModelThinkingLevel`.
    pub fn remove_model_thinking_level(&mut self, provider: &str, model_id: &str) {
        let key = format!("{provider}/{model_id}");
        if let Some(levels) = self
            .global_settings
            .get_mut("modelThinkingLevels")
            .and_then(Value::as_object_mut)
        {
            levels.shift_remove(&key);
            if levels.is_empty() {
                self.global_settings.shift_remove("modelThinkingLevels");
            }
        }
        self.modified.mark("modelThinkingLevels", Some(&key));
        self.save();
    }

    /// The transport, upstream's `getTransport`.
    #[must_use]
    pub fn get_transport(&self) -> Transport {
        serde_json::from_value(
            self.merged_get("transport")
                .cloned()
                .unwrap_or(Value::String("auto".to_string())),
        )
        .unwrap_or(Transport::Auto)
    }

    /// Set the transport, upstream's `setTransport`.
    pub fn set_transport(&mut self, transport: Transport) {
        let value = serde_json::to_value(transport).unwrap_or(Value::Null);
        self.set_global("transport", value, None);
    }

    /// Whether compaction runs, upstream's `getCompactionEnabled`.
    #[must_use]
    pub fn get_compaction_enabled(&self) -> bool {
        self.merged_get_nested("compaction", "enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Set the compaction toggle, upstream's `setCompactionEnabled`.
    pub fn set_compaction_enabled(&mut self, enabled: bool) {
        self.set_global("compaction", Value::Bool(enabled), Some("enabled"));
    }

    fn get_compaction_token_setting(
        &self,
        field: &str,
        model: Option<&ModelKey>,
    ) -> Result<i64, String> {
        // Upstream validates with a `!== undefined` gate ahead of the `??`
        // chain, so a JSON `null` present in the file is invalid input, not
        // an unset setting — dropping the key is the unset spelling.
        let compaction = self.merged_get("compaction");
        let ordinary = compaction.and_then(|compaction| compaction.get(field));
        if let Some(ordinary) = ordinary
            && !is_non_negative_safe_integer(ordinary)
        {
            return Err(format!(
                "Invalid compaction.{field} setting: {}. Expected a non-negative safe integer.",
                js_string(ordinary)
            ));
        }

        let model_key = model.map(ModelKey::key);
        let entry = model_key.as_deref().and_then(|model_key| {
            compaction
                .and_then(|compaction| compaction.get("modelOverrides"))
                .and_then(|overrides| overrides.get(model_key))
        });
        if let Some(entry) = entry
            && !is_mergeable_object(entry)
        {
            return Err(format!(
                "Invalid compaction.modelOverrides[\"{}\"] setting: {}. Expected an object.",
                model_key.unwrap_or_default(),
                js_string(entry)
            ));
        }
        let override_value = entry.and_then(|entry| entry.get(field));
        if let Some(override_value) = override_value
            && !is_non_negative_safe_integer(override_value)
        {
            return Err(format!(
                "Invalid compaction.modelOverrides[\"{}\"].{field} setting: {}. Expected a non-negative safe integer.",
                model_key.unwrap_or_default(),
                js_string(override_value)
            ));
        }
        let resolved = override_value
            .and_then(Value::as_i64)
            .or_else(|| ordinary.and_then(Value::as_i64));
        Ok(resolved.unwrap_or(if field == "reserveTokens" {
            DEFAULT_COMPACTION_RESERVE_TOKENS
        } else {
            DEFAULT_COMPACTION_KEEP_RECENT_TOKENS
        }))
    }

    /// The compaction reserve for a model, upstream's
    /// `getCompactionReserveTokens`.
    ///
    /// # Errors
    /// The invalid-setting messages upstream throws.
    pub fn get_compaction_reserve_tokens(&self, model: Option<&ModelKey>) -> Result<i64, String> {
        self.get_compaction_token_setting("reserveTokens", model)
    }

    /// The compaction recent-keep for a model, upstream's
    /// `getCompactionKeepRecentTokens`.
    ///
    /// # Errors
    /// The invalid-setting messages upstream throws.
    pub fn get_compaction_keep_recent_tokens(
        &self,
        model: Option<&ModelKey>,
    ) -> Result<i64, String> {
        self.get_compaction_token_setting("keepRecentTokens", model)
    }

    /// The resolved compaction settings, upstream's `getCompactionSettings`.
    ///
    /// # Errors
    /// The invalid-setting messages upstream throws.
    pub fn get_compaction_settings(
        &self,
        model: Option<&ModelKey>,
    ) -> Result<CompactionSettings, String> {
        Ok(CompactionSettings {
            enabled: self.get_compaction_enabled(),
            reserve_tokens: self.get_compaction_reserve_tokens(model)?,
            keep_recent_tokens: self.get_compaction_keep_recent_tokens(model)?,
        })
    }

    /// The branch-summary settings, upstream's `getBranchSummarySettings`.
    #[must_use]
    pub fn get_branch_summary_settings(&self) -> BranchSummarySettings {
        BranchSummarySettings {
            reserve_tokens: self
                .merged_get_nested("branchSummary", "reserveTokens")
                .and_then(Value::as_i64)
                .unwrap_or(16384),
            skip_prompt: self
                .merged_get_nested("branchSummary", "skipPrompt")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }

    /// Whether the branch-summary prompt is skipped, upstream's
    /// `getBranchSummarySkipPrompt`.
    #[must_use]
    pub fn get_branch_summary_skip_prompt(&self) -> bool {
        self.get_branch_summary_settings().skip_prompt
    }

    /// Whether agent retries run, upstream's `getRetryEnabled`.
    #[must_use]
    pub fn get_retry_enabled(&self) -> bool {
        self.merged_get_nested("retry", "enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Set the retry toggle, upstream's `setRetryEnabled`.
    pub fn set_retry_enabled(&mut self, enabled: bool) {
        self.set_global("retry", Value::Bool(enabled), Some("enabled"));
    }

    /// The agent retry settings, upstream's `getRetrySettings`.
    #[must_use]
    pub fn get_retry_settings(&self) -> RetrySettings {
        RetrySettings {
            enabled: self.get_retry_enabled(),
            max_retries: self
                .merged_get_nested("retry", "maxRetries")
                .and_then(Value::as_i64)
                .unwrap_or(3),
            base_delay_ms: self
                .merged_get_nested("retry", "baseDelayMs")
                .and_then(Value::as_i64)
                .unwrap_or(2000),
            max_agent_delay_ms: self
                .merged_get_nested("retry", "maxAgentDelayMs")
                .and_then(Value::as_i64)
                .unwrap_or(DEFAULT_MAX_AGENT_RETRY_DELAY_MS.cast_signed()),
        }
    }

    /// The provider retry settings, upstream's `getProviderRetrySettings`.
    #[must_use]
    pub fn get_provider_retry_settings(&self) -> ProviderRetrySettings {
        ProviderRetrySettings {
            timeout_ms: self
                .merged_get_nested("retry", "provider")
                .and_then(|provider| provider.get("timeoutMs"))
                .and_then(Value::as_i64),
            max_retries: self
                .merged_get_nested("retry", "provider")
                .and_then(|provider| provider.get("maxRetries"))
                .and_then(Value::as_i64),
            max_retry_delay_ms: self
                .merged_get_nested("retry", "provider")
                .and_then(|provider| provider.get("maxRetryDelayMs"))
                .and_then(Value::as_i64)
                .unwrap_or(60000),
        }
    }

    fn parse_timeout_setting(&self, key: &str, setting_name: &str) -> Result<Option<i64>, String> {
        let value = self.merged_get(key);
        if let Some(value) = value {
            if let Some(timeout) = parse_http_idle_timeout_ms(value) {
                return Ok(Some(timeout));
            }
            return Err(format!(
                "Invalid {setting_name} setting: {}",
                js_string(value)
            ));
        }
        Ok(None)
    }

    /// The HTTP idle timeout, upstream's `getHttpIdleTimeoutMs`.
    ///
    /// # Errors
    /// The invalid-setting message upstream throws.
    pub fn get_http_idle_timeout_ms(&self) -> Result<i64, String> {
        Ok(self
            .parse_timeout_setting("httpIdleTimeoutMs", "httpIdleTimeoutMs")?
            .unwrap_or(DEFAULT_HTTP_IDLE_TIMEOUT_MS))
    }

    /// Set the HTTP idle timeout, upstream's `setHttpIdleTimeoutMs`.
    ///
    /// # Errors
    /// A non-finite or negative timeout.
    pub fn set_http_idle_timeout_ms(&mut self, timeout_ms: f64) -> Result<(), String> {
        if !timeout_ms.is_finite() || timeout_ms < 0.0 {
            return Err(format!("Invalid httpIdleTimeoutMs setting: {timeout_ms}"));
        }
        let floored = timeout_ms.floor();
        // Upstream stores the floored JS number, which stringify writes
        // without a decimal; the integer representation keeps the saved file
        // byte-identical. Floors at or beyond 2^63 stay floats.
        let value = if floored < 2f64.powi(63) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the bound above keeps the integral floor inside i64's exact range"
            )]
            Value::Number(serde_json::Number::from(floored as i64))
        } else {
            serde_json::Number::from_f64(floored).map_or(Value::Null, Value::Number)
        };
        self.set_global("httpIdleTimeoutMs", value, None);
        Ok(())
    }

    /// The WebSocket connect timeout, upstream's
    /// `getWebSocketConnectTimeoutMs`.
    ///
    /// # Errors
    /// The invalid-setting message upstream throws.
    pub fn get_websocket_connect_timeout_ms(&self) -> Result<Option<i64>, String> {
        self.parse_timeout_setting("websocketConnectTimeoutMs", "websocketConnectTimeoutMs")
    }

    /// Whether the thinking block hides, upstream's `getHideThinkingBlock`.
    #[must_use]
    pub fn get_hide_thinking_block(&self) -> bool {
        self.merged_get("hideThinkingBlock")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Set the thinking-block hide, upstream's `setHideThinkingBlock`.
    pub fn set_hide_thinking_block(&mut self, hide: bool) {
        self.set_global("hideThinkingBlock", Value::Bool(hide), None);
    }

    /// Whether cache-miss notices show, upstream's `getShowCacheMissNotices`.
    #[must_use]
    pub fn get_show_cache_miss_notices(&self) -> bool {
        self.merged_get("showCacheMissNotices")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Set the cache-miss notices, upstream's `setShowCacheMissNotices`.
    pub fn set_show_cache_miss_notices(&mut self, show: bool) {
        self.set_global("showCacheMissNotices", Value::Bool(show), None);
    }

    /// The external editor command, upstream's `getExternalEditorCommand`:
    /// the setting first, then `VISUAL`/`EDITOR`, then the platform default.
    #[must_use]
    pub fn get_external_editor_command(&self) -> String {
        self.get_external_editor_command_with(&default_env_lookup())
    }

    /// [`get_external_editor_command`](Self::get_external_editor_command)
    /// over an injected environment lookup.
    #[must_use]
    pub fn get_external_editor_command_with(&self, env: &EnvLookup) -> String {
        if let Some(editor) = self.merged_get("externalEditor").and_then(Value::as_str)
            && !editor.trim().is_empty()
        {
            return editor.to_string();
        }
        if let Some(editor) = env("VISUAL").filter(|editor| !editor.is_empty()) {
            return editor;
        }
        if let Some(editor) = env("EDITOR").filter(|editor| !editor.is_empty()) {
            return editor;
        }
        if cfg!(windows) {
            "notepad".to_string()
        } else {
            "nano".to_string()
        }
    }

    /// The shell path, upstream's `getShellPath` — tilde-expanded through
    /// `normalizePath` when set.
    #[must_use]
    pub fn get_shell_path(&self) -> Option<String> {
        self.merged_get("shellPath")
            .and_then(Value::as_str)
            .map(|shell_path| {
                normalize_path(shell_path, &PathInputOptions::default())
                    .unwrap_or_else(|_| shell_path.to_string())
            })
    }

    /// Set the shell path, upstream's `setShellPath`.
    pub fn set_shell_path(&mut self, path: Option<&str>) {
        match path {
            Some(path) => self.set_global("shellPath", Value::String(path.to_string()), None),
            None => self.remove_global("shellPath"),
        }
    }

    /// Whether startup is quiet, upstream's `getQuietStartup`.
    #[must_use]
    pub fn get_quiet_startup(&self) -> bool {
        self.merged_get("quietStartup")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Set the quiet startup, upstream's `setQuietStartup`.
    pub fn set_quiet_startup(&mut self, quiet: bool) {
        self.set_global("quietStartup", Value::Bool(quiet), None);
    }

    /// The default project trust, upstream's `getDefaultProjectTrust` —
    /// read from the global scope only, invalid values reading as ask.
    #[must_use]
    pub fn get_default_project_trust(&self) -> DefaultProjectTrust {
        match self
            .global_settings
            .get("defaultProjectTrust")
            .and_then(Value::as_str)
        {
            Some("always") => DefaultProjectTrust::Always,
            Some("never") => DefaultProjectTrust::Never,
            _ => DefaultProjectTrust::Ask,
        }
    }

    /// Set the default project trust, upstream's `setDefaultProjectTrust`.
    pub fn set_default_project_trust(&mut self, default_project_trust: DefaultProjectTrust) {
        self.set_global(
            "defaultProjectTrust",
            Value::String(default_project_trust.as_str().to_string()),
            None,
        );
    }

    /// The shell command prefix, upstream's `getShellCommandPrefix`.
    #[must_use]
    pub fn get_shell_command_prefix(&self) -> Option<String> {
        self.merged_get("shellCommandPrefix")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// Set the shell command prefix, upstream's `setShellCommandPrefix`.
    pub fn set_shell_command_prefix(&mut self, prefix: Option<&str>) {
        match prefix {
            Some(prefix) => {
                self.set_global(
                    "shellCommandPrefix",
                    Value::String(prefix.to_string()),
                    None,
                );
            }
            None => self.remove_global("shellCommandPrefix"),
        }
    }

    /// The npm command argv, upstream's `getNpmCommand`.
    #[must_use]
    pub fn get_npm_command(&self) -> Option<Vec<String>> {
        self.merged_get("npmCommand")
            .and_then(Value::as_array)
            .map(|command| {
                command
                    .iter()
                    .filter_map(|part| part.as_str().map(str::to_string))
                    .collect()
            })
    }

    /// Set the npm command argv, upstream's `setNpmCommand`.
    pub fn set_npm_command(&mut self, command: Option<&[String]>) {
        match command {
            Some(command) => {
                self.set_global(
                    "npmCommand",
                    Value::Array(
                        command
                            .iter()
                            .map(|part| Value::String(part.clone()))
                            .collect(),
                    ),
                    None,
                );
            }
            None => self.remove_global("npmCommand"),
        }
    }

    /// Whether the changelog collapses, upstream's `getCollapseChangelog`.
    #[must_use]
    pub fn get_collapse_changelog(&self) -> bool {
        self.merged_get("collapseChangelog")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Set the changelog collapse, upstream's `setCollapseChangelog`.
    pub fn set_collapse_changelog(&mut self, collapse: bool) {
        self.set_global("collapseChangelog", Value::Bool(collapse), None);
    }

    /// Whether the install ping sends, upstream's
    /// `getEnableInstallTelemetry`.
    #[must_use]
    pub fn get_enable_install_telemetry(&self) -> bool {
        self.merged_get("enableInstallTelemetry")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Set the install ping, upstream's `setEnableInstallTelemetry`.
    pub fn set_enable_install_telemetry(&mut self, enabled: bool) {
        self.set_global("enableInstallTelemetry", Value::Bool(enabled), None);
    }

    /// Whether analytics share, upstream's `getEnableAnalytics`.
    #[must_use]
    pub fn get_enable_analytics(&self) -> bool {
        self.merged_get("enableAnalytics")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// The analytics tracking identifier, upstream's `getTrackingId`.
    #[must_use]
    pub fn get_tracking_id(&self) -> Option<String> {
        self.merged_get("trackingId")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// Set the analytics opt-in, upstream's `setEnableAnalytics`: the first
    /// opt-in generates the tracking identifier.
    pub fn set_enable_analytics(&mut self, enabled: bool) {
        self.set_global("enableAnalytics", Value::Bool(enabled), None);
        if enabled
            && self
                .global_settings
                .get("trackingId")
                .is_none_or(Value::is_null)
        {
            let tracking_id = uuid::Uuid::new_v4().to_string();
            self.set_global("trackingId", Value::String(tracking_id), None);
        }
    }

    /// The package sources, upstream's `getPackages`.
    #[must_use]
    pub fn get_packages(&self) -> Vec<Value> {
        self.merged_get("packages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    }

    /// Set the package sources, upstream's `setPackages`.
    pub fn set_packages(&mut self, packages: &[Value]) {
        self.set_global("packages", Value::Array(packages.to_vec()), None);
    }

    /// Set the project package sources, upstream's `setProjectPackages`.
    ///
    /// # Errors
    /// The untrusted-project refusal.
    pub fn set_project_packages(&mut self, packages: &[Value]) -> Result<(), String> {
        self.update_project_settings("packages", |settings| {
            settings.insert("packages".to_string(), Value::Array(packages.to_vec()));
        })
    }

    /// The extension paths, upstream's `getExtensionPaths`.
    #[must_use]
    pub fn get_extension_paths(&self) -> Vec<String> {
        string_array(self.merged_get("extensions"))
    }

    /// Set the extension paths, upstream's `setExtensionPaths`.
    pub fn set_extension_paths(&mut self, paths: &[String]) {
        self.set_global(
            "extensions",
            Value::Array(
                paths
                    .iter()
                    .map(|path| Value::String(path.clone()))
                    .collect(),
            ),
            None,
        );
    }

    /// Set the project extension paths, upstream's
    /// `setProjectExtensionPaths`.
    ///
    /// # Errors
    /// The untrusted-project refusal.
    pub fn set_project_extension_paths(&mut self, paths: &[String]) -> Result<(), String> {
        self.update_project_settings("extensions", |settings| {
            settings.insert(
                "extensions".to_string(),
                Value::Array(
                    paths
                        .iter()
                        .map(|path| Value::String(path.clone()))
                        .collect(),
                ),
            );
        })
    }

    /// The skill paths, upstream's `getSkillPaths`.
    #[must_use]
    pub fn get_skill_paths(&self) -> Vec<String> {
        string_array(self.merged_get("skills"))
    }

    /// Set the skill paths, upstream's `setSkillPaths`.
    pub fn set_skill_paths(&mut self, paths: &[String]) {
        self.set_global(
            "skills",
            Value::Array(
                paths
                    .iter()
                    .map(|path| Value::String(path.clone()))
                    .collect(),
            ),
            None,
        );
    }

    /// Set the project skill paths, upstream's `setProjectSkillPaths`.
    ///
    /// # Errors
    /// The untrusted-project refusal.
    pub fn set_project_skill_paths(&mut self, paths: &[String]) -> Result<(), String> {
        self.update_project_settings("skills", |settings| {
            settings.insert(
                "skills".to_string(),
                Value::Array(
                    paths
                        .iter()
                        .map(|path| Value::String(path.clone()))
                        .collect(),
                ),
            );
        })
    }

    /// The prompt-template paths, upstream's `getPromptTemplatePaths`.
    #[must_use]
    pub fn get_prompt_template_paths(&self) -> Vec<String> {
        string_array(self.merged_get("prompts"))
    }

    /// Set the prompt-template paths, upstream's `setPromptTemplatePaths`.
    pub fn set_prompt_template_paths(&mut self, paths: &[String]) {
        self.set_global(
            "prompts",
            Value::Array(
                paths
                    .iter()
                    .map(|path| Value::String(path.clone()))
                    .collect(),
            ),
            None,
        );
    }

    /// Set the project prompt-template paths, upstream's
    /// `setProjectPromptTemplatePaths`.
    ///
    /// # Errors
    /// The untrusted-project refusal.
    pub fn set_project_prompt_template_paths(&mut self, paths: &[String]) -> Result<(), String> {
        self.update_project_settings("prompts", |settings| {
            settings.insert(
                "prompts".to_string(),
                Value::Array(
                    paths
                        .iter()
                        .map(|path| Value::String(path.clone()))
                        .collect(),
                ),
            );
        })
    }

    /// The theme paths, upstream's `getThemePaths`.
    #[must_use]
    pub fn get_theme_paths(&self) -> Vec<String> {
        string_array(self.merged_get("themes"))
    }

    /// Set the theme paths, upstream's `setThemePaths`.
    pub fn set_theme_paths(&mut self, paths: &[String]) {
        self.set_global(
            "themes",
            Value::Array(
                paths
                    .iter()
                    .map(|path| Value::String(path.clone()))
                    .collect(),
            ),
            None,
        );
    }

    /// Set the project theme paths, upstream's `setProjectThemePaths`.
    ///
    /// # Errors
    /// The untrusted-project refusal.
    pub fn set_project_theme_paths(&mut self, paths: &[String]) -> Result<(), String> {
        self.update_project_settings("themes", |settings| {
            settings.insert(
                "themes".to_string(),
                Value::Array(
                    paths
                        .iter()
                        .map(|path| Value::String(path.clone()))
                        .collect(),
                ),
            );
        })
    }

    /// Whether skills register as commands, upstream's
    /// `getEnableSkillCommands`.
    #[must_use]
    pub fn get_enable_skill_commands(&self) -> bool {
        self.merged_get("enableSkillCommands")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Set the skill-command registration, upstream's
    /// `setEnableSkillCommands`.
    pub fn set_enable_skill_commands(&mut self, enabled: bool) {
        self.set_global("enableSkillCommands", Value::Bool(enabled), None);
    }

    /// The thinking budgets, upstream's `getThinkingBudgets`.
    #[must_use]
    pub fn get_thinking_budgets(&self) -> Option<Settings> {
        self.merged_get("thinkingBudgets")
            .and_then(Value::as_object)
            .cloned()
    }

    /// The terminal capability overrides, upstream's
    /// `getTerminalCapabilityOverrides`: explicit values map, auto values
    /// omit.
    #[must_use]
    pub fn get_terminal_capability_overrides(&self) -> CapabilityOverrides {
        let terminal = self.merged_get("terminal");
        // Upstream spreads `{ images: null }` only when `images === false`;
        // every other value — absent, "auto", or a string the protocol list
        // does not name — omits the field, so the raw value must be read
        // rather than its string projection.
        let images_override = match terminal.and_then(|terminal| terminal.get("images")) {
            Some(Value::String(protocol)) if protocol == "kitty" => {
                Some(Some(pi_tui::terminal_image::ImageProtocol::Kitty))
            }
            Some(Value::String(protocol)) if protocol == "iterm2" => {
                Some(Some(pi_tui::terminal_image::ImageProtocol::Iterm2))
            }
            Some(Value::Bool(false)) => Some(None),
            _ => None,
        };
        let true_color = terminal
            .and_then(|terminal| terminal.get("trueColor"))
            .and_then(Value::as_bool);
        let hyperlinks = terminal
            .and_then(|terminal| terminal.get("hyperlinks"))
            .and_then(Value::as_bool);
        CapabilityOverrides {
            images: images_override,
            true_color,
            hyperlinks,
        }
    }

    /// Whether terminal images show, upstream's `getShowImages`.
    #[must_use]
    pub fn get_show_images(&self) -> bool {
        self.merged_get_nested("terminal", "showImages")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Set the terminal images, upstream's `setShowImages`.
    pub fn set_show_images(&mut self, show: bool) {
        self.set_global("terminal", Value::Bool(show), Some("showImages"));
    }

    /// The inline image width in cells, upstream's `getImageWidthCells`.
    #[must_use]
    pub fn get_image_width_cells(&self) -> i64 {
        let width = self
            .merged_get_nested("terminal", "imageWidthCells")
            .and_then(Value::as_f64);
        match width {
            Some(width) if width.is_finite() => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "Math.floor restatement: the width is checked finite immediately above"
                )]
                let floored = width.floor() as i64;
                std::cmp::max(1, floored)
            }
            _ => 60,
        }
    }

    /// Set the inline image width, upstream's `setImageWidthCells`.
    pub fn set_image_width_cells(&mut self, width: f64) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Math.floor restatement: the fraction drops the way upstream's floor does"
        )]
        let floored = std::cmp::max(1, width.floor() as i64);
        self.set_global(
            "terminal",
            Value::Number(serde_json::Number::from(floored)),
            Some("imageWidthCells"),
        );
    }

    /// Whether shrinking clears empty rows, upstream's `getClearOnShrink`:
    /// the setting first, then the `PI_CLEAR_ON_SHRINK` environment.
    #[must_use]
    pub fn get_clear_on_shrink(&self) -> bool {
        self.get_clear_on_shrink_with(&default_env_lookup())
    }

    /// [`get_clear_on_shrink`](Self::get_clear_on_shrink) over an injected
    /// environment lookup.
    #[must_use]
    pub fn get_clear_on_shrink_with(&self, env: &EnvLookup) -> bool {
        if let Some(enabled) = self.merged_get_nested("terminal", "clearOnShrink") {
            return enabled.as_bool().unwrap_or(false);
        }
        env("PI_CLEAR_ON_SHRINK").is_some_and(|value| value == "1")
    }

    /// Set the shrink clearing, upstream's `setClearOnShrink`.
    pub fn set_clear_on_shrink(&mut self, enabled: bool) {
        self.set_global("terminal", Value::Bool(enabled), Some("clearOnShrink"));
    }

    /// Whether terminal progress shows, upstream's
    /// `getShowTerminalProgress`.
    #[must_use]
    pub fn get_show_terminal_progress(&self) -> bool {
        self.merged_get_nested("terminal", "showTerminalProgress")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Set the terminal progress, upstream's `setShowTerminalProgress`.
    pub fn set_show_terminal_progress(&mut self, enabled: bool) {
        self.set_global(
            "terminal",
            Value::Bool(enabled),
            Some("showTerminalProgress"),
        );
    }

    /// The TUI mode, upstream's `getTuiMode`.
    #[must_use]
    pub fn get_tui_mode(&self) -> TuiMode {
        match self.merged_get("tuiMode").and_then(Value::as_str) {
            Some("fullscreen") => TuiMode::Fullscreen,
            _ => TuiMode::Regular,
        }
    }

    /// Set the TUI mode, upstream's `setTuiMode`.
    pub fn set_tui_mode(&mut self, mode: TuiMode) {
        let value = match mode {
            TuiMode::Fullscreen => "fullscreen",
            TuiMode::Regular => "regular",
        };
        self.set_global("tuiMode", Value::String(value.to_string()), None);
    }

    /// The fullscreen exit output, upstream's `getFullscreenExitOutput`.
    #[must_use]
    pub fn get_fullscreen_exit_output(&self) -> FullscreenExitOutput {
        match self
            .merged_get("fullscreenExitOutput")
            .and_then(Value::as_str)
        {
            Some("resume-hint") => FullscreenExitOutput::ResumeHint,
            _ => FullscreenExitOutput::Transcript,
        }
    }

    /// Set the fullscreen exit output, upstream's `setFullscreenExitOutput`.
    pub fn set_fullscreen_exit_output(&mut self, output: FullscreenExitOutput) {
        let value = match output {
            FullscreenExitOutput::ResumeHint => "resume-hint",
            FullscreenExitOutput::Transcript => "transcript",
        };
        self.set_global(
            "fullscreenExitOutput",
            Value::String(value.to_string()),
            None,
        );
    }

    /// The fullscreen scrollbar mode, upstream's `getFullscreenScrollbar`.
    #[must_use]
    pub fn get_fullscreen_scrollbar(&self) -> ScrollViewScrollbar {
        match self
            .merged_get("fullscreenScrollbar")
            .and_then(Value::as_str)
        {
            Some("always") => ScrollViewScrollbar::Always,
            Some("hidden") => ScrollViewScrollbar::Hidden,
            _ => ScrollViewScrollbar::Auto,
        }
    }

    /// Set the fullscreen scrollbar, upstream's `setFullscreenScrollbar`.
    pub fn set_fullscreen_scrollbar(&mut self, mode: ScrollViewScrollbar) {
        let value = match mode {
            ScrollViewScrollbar::Always => "always",
            ScrollViewScrollbar::Hidden => "hidden",
            ScrollViewScrollbar::Auto => "auto",
        };
        self.set_global(
            "fullscreenScrollbar",
            Value::String(value.to_string()),
            None,
        );
    }

    /// Whether selection copies, upstream's `getFullscreenCopyOnSelect`.
    #[must_use]
    pub fn get_fullscreen_copy_on_select(&self) -> bool {
        self.merged_get("fullscreenCopyOnSelect")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Set the selection copy, upstream's `setFullscreenCopyOnSelect`.
    pub fn set_fullscreen_copy_on_select(&mut self, enabled: bool) {
        self.set_global("fullscreenCopyOnSelect", Value::Bool(enabled), None);
    }

    /// Whether images auto-resize, upstream's `getImageAutoResize`.
    #[must_use]
    pub fn get_image_auto_resize(&self) -> bool {
        self.merged_get_nested("images", "autoResize")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Set the image auto-resize, upstream's `setImageAutoResize`.
    pub fn set_image_auto_resize(&mut self, enabled: bool) {
        self.set_global("images", Value::Bool(enabled), Some("autoResize"));
    }

    /// Whether images block, upstream's `getBlockImages`.
    #[must_use]
    pub fn get_block_images(&self) -> bool {
        self.merged_get_nested("images", "blockImages")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Set the image block, upstream's `setBlockImages`.
    pub fn set_block_images(&mut self, blocked: bool) {
        self.set_global("images", Value::Bool(blocked), Some("blockImages"));
    }

    /// The enabled model patterns, upstream's `getEnabledModels`.
    #[must_use]
    pub fn get_enabled_models(&self) -> Option<Vec<String>> {
        self.merged_get("enabledModels")
            .map(|value| string_array(Some(value)))
    }

    /// Set the enabled model patterns, upstream's `setEnabledModels`.
    pub fn set_enabled_models(&mut self, patterns: Option<&[String]>) {
        match patterns {
            Some(patterns) => {
                self.set_global(
                    "enabledModels",
                    Value::Array(
                        patterns
                            .iter()
                            .map(|pattern| Value::String(pattern.clone()))
                            .collect(),
                    ),
                    None,
                );
            }
            None => self.remove_global("enabledModels"),
        }
    }

    /// The initial built-in tool selection, upstream's `getDefaultTools`.
    #[must_use]
    pub fn get_default_tools(&self) -> Option<Vec<String>> {
        self.merged_get("defaultTools")
            .map(|value| string_array(Some(value)))
    }

    /// Set the built-in tool selection, upstream's `setDefaultTools`.
    pub fn set_default_tools(&mut self, tools: Option<&[String]>) {
        match tools {
            Some(tools) => {
                self.set_global(
                    "defaultTools",
                    Value::Array(
                        tools
                            .iter()
                            .map(|tool| Value::String(tool.clone()))
                            .collect(),
                    ),
                    None,
                );
            }
            None => self.remove_global("defaultTools"),
        }
    }

    /// The double-escape action, upstream's `getDoubleEscapeAction`.
    #[must_use]
    pub fn get_double_escape_action(&self) -> &str {
        match self
            .merged_get("doubleEscapeAction")
            .and_then(Value::as_str)
        {
            Some("fork") => "fork",
            Some("none") => "none",
            _ => "tree",
        }
    }

    /// Set the double-escape action, upstream's `setDoubleEscapeAction`.
    pub fn set_double_escape_action(&mut self, action: &str) {
        self.set_global(
            "doubleEscapeAction",
            Value::String(action.to_string()),
            None,
        );
    }

    /// The tree filter mode, upstream's `getTreeFilterMode`: invalid values
    /// read as the default.
    #[must_use]
    pub fn get_tree_filter_mode(&self) -> &str {
        match self.merged_get("treeFilterMode").and_then(Value::as_str) {
            Some("no-tools") => "no-tools",
            Some("user-only") => "user-only",
            Some("labeled-only") => "labeled-only",
            Some("all") => "all",
            _ => "default",
        }
    }

    /// Set the tree filter mode, upstream's `setTreeFilterMode`.
    pub fn set_tree_filter_mode(&mut self, mode: &str) {
        self.set_global("treeFilterMode", Value::String(mode.to_string()), None);
    }

    /// Whether the hardware cursor shows, upstream's
    /// `getShowHardwareCursor`.
    #[must_use]
    pub fn get_show_hardware_cursor(&self) -> bool {
        self.get_show_hardware_cursor_with(&default_env_lookup())
    }

    /// [`get_show_hardware_cursor`](Self::get_show_hardware_cursor) over an
    /// injected environment lookup.
    #[must_use]
    pub fn get_show_hardware_cursor_with(&self, env: &EnvLookup) -> bool {
        if let Some(enabled) = self
            .merged_get("showHardwareCursor")
            .and_then(Value::as_bool)
        {
            return enabled;
        }
        env("PI_HARDWARE_CURSOR").is_some_and(|value| value == "1")
    }

    /// Set the hardware cursor, upstream's `setShowHardwareCursor`.
    pub fn set_show_hardware_cursor(&mut self, enabled: bool) {
        self.set_global("showHardwareCursor", Value::Bool(enabled), None);
    }

    /// The editor horizontal padding, upstream's `getEditorPaddingX`.
    #[must_use]
    pub fn get_editor_padding_x(&self) -> i64 {
        self.merged_get("editorPaddingX")
            .and_then(Value::as_i64)
            .unwrap_or(0)
    }

    /// Set the editor padding, upstream's `setEditorPaddingX` — clamped to
    /// `0..=3`.
    pub fn set_editor_padding_x(&mut self, padding: f64) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Math.floor restatement: the fraction drops the way upstream's floor does"
        )]
        let clamped = (padding.floor() as i64).clamp(0, 3);
        self.set_global(
            "editorPaddingX",
            Value::Number(serde_json::Number::from(clamped)),
            None,
        );
    }

    /// The chat output padding, upstream's `getOutputPad`.
    #[must_use]
    pub fn get_output_pad(&self) -> u8 {
        u8::from(self.merged_get("outputPad").and_then(Value::as_i64) != Some(0))
    }

    /// Set the chat output padding, upstream's `setOutputPad`.
    pub fn set_output_pad(&mut self, padding: u8) {
        self.set_global(
            "outputPad",
            Value::Number(serde_json::Number::from(padding)),
            None,
        );
    }

    /// The autocomplete dropdown cap, upstream's `getAutocompleteMaxVisible`.
    #[must_use]
    pub fn get_autocomplete_max_visible(&self) -> i64 {
        self.merged_get("autocompleteMaxVisible")
            .and_then(Value::as_i64)
            .unwrap_or(5)
    }

    /// Set the autocomplete cap, upstream's `setAutocompleteMaxVisible` —
    /// clamped to `3..=20`.
    pub fn set_autocomplete_max_visible(&mut self, max_visible: f64) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Math.floor restatement: the fraction drops the way upstream's floor does"
        )]
        let clamped = (max_visible.floor() as i64).clamp(3, 20);
        self.set_global(
            "autocompleteMaxVisible",
            Value::Number(serde_json::Number::from(clamped)),
            None,
        );
    }

    /// The code-block indent, upstream's `getCodeBlockIndent`.
    #[must_use]
    pub fn get_code_block_indent(&self) -> String {
        self.merged_get_nested("markdown", "codeBlockIndent")
            .and_then(Value::as_str)
            .unwrap_or("  ")
            .to_string()
    }

    /// The mermaid rendering mode, upstream's `getMermaidRenderingMode`.
    #[must_use]
    pub fn get_mermaid_rendering_mode(&self) -> MermaidRenderingMode {
        match self
            .merged_get_nested("markdown", "mermaid")
            .and_then(Value::as_str)
        {
            Some("off") => MermaidRenderingMode::Off,
            Some("final") => MermaidRenderingMode::Final,
            _ => MermaidRenderingMode::Streaming,
        }
    }

    /// Set the mermaid rendering mode, upstream's `setMermaidRenderingMode`.
    pub fn set_mermaid_rendering_mode(&mut self, mode: MermaidRenderingMode) {
        let value = match mode {
            MermaidRenderingMode::Off => "off",
            MermaidRenderingMode::Final => "final",
            MermaidRenderingMode::Streaming => "streaming",
        };
        self.set_global(
            "markdown",
            Value::String(value.to_string()),
            Some("mermaid"),
        );
    }

    /// The warning toggles, upstream's `getWarnings`.
    #[must_use]
    pub fn get_warnings(&self) -> Settings {
        self.merged_get("warnings")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    }

    /// Set the warning toggles, upstream's `setWarnings`.
    pub fn set_warnings(&mut self, warnings: &Settings) {
        self.set_global("warnings", Value::Object(warnings.clone()), None);
    }
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// One scope's load outcome, upstream's `{ settings, error }`.
struct LoadedSettings {
    settings: Settings,
    error: Option<String>,
}

/// The in-memory manager, upstream's `SettingsManager.inMemory`: the
/// settings migrate once and seed the global scope.
impl SettingsManager<InMemorySettingsStorage> {
    /// The in-memory manager, upstream's `inMemory`.
    #[must_use]
    pub fn in_memory(settings: &Settings, options: SettingsManagerCreateOptions) -> Self {
        let storage = InMemorySettingsStorage::default();
        let mut initial = settings.clone();
        migrate_settings(&mut initial);
        storage.seed(
            SettingsScope::Global,
            serde_json::to_string_pretty(&initial).unwrap_or_else(|_| "{}".to_string()),
        );
        Self::from_storage(storage, options)
    }
}

/// The file-backed manager, upstream's `SettingsManager.create`: the
/// resolved cwd/agent-dir storage with the error-reporting paths attached.
impl SettingsManager<FileSettingsStorage> {
    /// The manager over the files, upstream's `create`.
    ///
    /// # Errors
    /// Never — load failures record as errors, matching upstream's
    /// try/catch.
    #[must_use]
    pub fn create(cwd: &str, agent_dir: &str, options: SettingsManagerCreateOptions) -> Self {
        let resolved_cwd = resolve_path(cwd, &process_cwd(), &crate::config::home_dir());
        let resolved_agent_dir =
            resolve_path(agent_dir, &process_cwd(), &crate::config::home_dir());
        let storage = FileSettingsStorage::new(&resolved_cwd, &resolved_agent_dir);
        let settings_paths = SettingsPaths {
            global: Some(format!("{resolved_agent_dir}/settings.json")),
            project: Some(format!("{resolved_cwd}/{CONFIG_DIR_NAME}/settings.json")),
        };
        Self::from_storage_with_paths(storage, options, settings_paths)
    }
}

/// Upstream's `process.cwd()` default for the resolver.
fn process_cwd() -> String {
    std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// The nested-key tracking's ordered key set type, exposed for the manager's
/// modified-field bookkeeping.
pub type NestedKeys = BTreeSet<String>;
