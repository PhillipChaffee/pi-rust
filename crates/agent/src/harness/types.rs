//! The harness capability and option types, ported from upstream
//! `src/harness/types.ts`.
//!
//! Upstream's hand-rolled `Result`/`ok`/`err` restates as [`std::result::Result`]:
//! `ok(value)` is `Ok`, `err(error)` is `Err`, `getOrUndefined(result)` is
//! [`Result::ok`], and `getOrThrow(result)` is [`Result::expect`] at the test
//! and adapter-boundary call sites that used it. `toError` normalizes JS
//! thrown values into `Error` instances; Rust errors are typed values, so the
//! normalizer has no counterpart and `cause` chains carry typed sources.
//!
//! The `FileSystem`/`Shell`/`ExecutionEnv` capability traits keep upstream's
//! non-throwing contract: every operation method encodes all failures —
//! including unexpected backend failures — in the returned
//! [`Result::Err`], and `cleanup` is best-effort and never fails.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use pi_ai::types::{BoxedFuture, CacheRetention, DeferredRequest, ProviderHeaders, Transport};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::types::{AgentToolResult, ToolExecutionMode, ToolReplay};

use super::context::Context;

/// A skill loaded from a `SKILL.md` file or provided by an application,
/// upstream's `Skill`.
///
/// `name`, `description`, and `file_path` are inserted into the system
/// prompt in an XML-formatted block as suggested by agentskills.io; the
/// skills child generates the spec-compatible system prompt block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Skill {
    /// Stable skill name used for lookup and model-visible listings.
    pub name: String,
    /// Short model-visible description of when to use the skill.
    pub description: String,
    /// Full skill instructions.
    pub content: String,
    /// Absolute path to the skill file, used for model-visible location and
    /// resolving relative references.
    pub file_path: String,
    /// Exclude this skill from model-visible skill lists while still
    /// allowing explicit application invocation.
    #[serde(default, skip_serializing_if = "std::option::Option::is_none")]
    pub disable_model_invocation: Option<bool>,
}

/// A prompt template that can be formatted into a prompt for explicit
/// invocation, upstream's `PromptTemplate`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTemplate {
    /// Stable template name used for lookup or application command routing.
    pub name: String,
    /// Optional description for command lists or autocomplete.
    #[serde(default, skip_serializing_if = "std::option::Option::is_none")]
    pub description: Option<String>,
    /// Template content. Argument placeholders are formatted by the
    /// skills child's invocation formatter.
    pub content: String,
}

/// Resources made available to explicit invocation methods and
/// system-prompt callbacks, upstream's `AgentHarnessResources`.
///
/// Upstream parameterizes the skill and template types; the port carries
/// the canonical shapes, and applications with richer skill types convert
/// at their boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessResources {
    /// Prompt templates available for explicit invocation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_templates: Vec<PromptTemplate>,
    /// Skills available to the model and explicit skill invocation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<Skill>,
}

/// Options for one live harness tool progress update, upstream's
/// `AgentHarnessToolUpdateOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentHarnessToolUpdateOptions {
    /// Request replacement of this invocation's durable recovery checkpoint.
    pub checkpoint: bool,
}

/// Synchronous full-snapshot progress callback supplied to harness-native
/// tools, upstream's `AgentHarnessToolUpdateCallback<TDetails>`.
pub type AgentHarnessToolUpdateCallback<'a, TDetails = JsonValue> =
    &'a (dyn Fn(&AgentToolResult<TDetails>, Option<AgentHarnessToolUpdateOptions>) + Send + Sync);

/// Stable harness identity for one logical tool call, unchanged during safe
/// replay, upstream's `AgentHarnessToolInvocation`.
///
/// The memo accessors are the invocation-scoped durable replay memo: a
/// `None` value deletes the memo.
pub trait AgentHarnessToolInvocation: fmt::Debug + Send + Sync {
    /// Opaque session-unique id equal to the call's reserved result-entry id.
    fn invocation_id(&self) -> &str;

    /// The durable operation id the invocation runs under.
    fn operation_id(&self) -> &str;

    /// The invocation-local turn id the call belongs to.
    fn turn_id(&self) -> &str;

    /// Read one invocation-scoped durable replay memo.
    fn get_memo(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<JsonValue>, String>>;

    /// Set or delete one invocation-scoped durable replay memo.
    fn set_memo(
        &self,
        name: &str,
        value: Option<JsonValue>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), String>>;
}

/// The erased application-defined context a harness tool executes with,
/// upstream's `TContext extends object | undefined` type parameter.
///
/// TypeScript parameterizes the harness over the application's context
/// type; the erased runtime path carries it behind an `Arc` and tools
/// downcast to their own type. `undefined` restates as `None`.
pub type ToolContext = Option<Arc<dyn std::any::Any + Send + Sync>>;

/// The harness tool executor, upstream's
/// `AgentHarnessTool<TContext>["execute"]`.
///
/// Arguments arrive validated against the tool's schema document; the
/// update callback publishes full-snapshot progress; the invocation carries
/// the stable identity and durable memo surface; the chord `Context` is the
/// invocation's cancellation and value scope.
pub type AgentHarnessToolExecuteFn = dyn for<'a> Fn(
        &'a str,
        &'a JsonValue,
        Option<AgentHarnessToolUpdateCallback<'a>>,
        ToolContext,
        &'a dyn AgentHarnessToolInvocation,
        &'a Context,
    ) -> BoxedFuture<'a, Result<AgentToolResult, crate::types::AgentToolError>>
    + Send
    + Sync;

/// Tool definition executed by a harness with an application-defined
/// context, upstream's `AgentHarnessTool<TContext, TParameters, TDetails>`.
///
/// The declarative surface mirrors the agent runtime's [`crate::types::AgentTool`];
/// the erased parameterization restates over the tool's JSON Schema document
/// and `JsonValue` arguments.
#[derive(Clone)]
pub struct AgentHarnessTool {
    /// The pi-ai tool surface: name, description, parameter JSON Schema
    /// document, constrained-sampling setting.
    pub tool: pi_ai::types::Tool,
    /// Human-readable label for UI display.
    pub label: String,
    /// Optional compatibility shim for raw tool-call arguments before schema
    /// validation; must return an object that matches the tool's schema.
    pub prepare_arguments: Option<crate::types::AgentToolPrepareArguments>,
    /// Execute the tool call with the context resolved for the current turn
    /// snapshot.
    pub execute: Arc<AgentHarnessToolExecuteFn>,
    /// Recovery policy for an effect whose outcome is unknown.
    pub replay: Option<ToolReplay>,
    /// Per-tool execution-mode override; omitted, the config's mode applies.
    pub execution_mode: Option<ToolExecutionMode>,
}

impl fmt::Debug for AgentHarnessTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The executor does not debug; the declarative surface does, and the
        // non-exhaustive finish names the executor's presence.
        f.debug_struct("AgentHarnessTool")
            .field("tool", &self.tool)
            .field("label", &self.label)
            .field("prepare_arguments", &self.prepare_arguments.is_some())
            .field("execute", &())
            .field("replay", &self.replay)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

impl AgentHarnessTool {
    /// The tool name, upstream's `AgentHarnessTool.name` inherited from
    /// `Tool`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.tool.name
    }
}

/// The resolved-context provider, upstream's `toolContext` provider over
/// one turn snapshot.
pub type ToolContextProvider =
    Arc<dyn Fn(&Context) -> BoxedFuture<'static, ToolContext> + Send + Sync>;

/// Static tool context or provider resolved for each turn snapshot,
/// upstream's `AgentHarnessToolContextSource<TContext>`.
///
/// The resolved context is the erased [`ToolContext`]; the provider receives
/// the chord `Context` and returns the turn's context value or `None`.
pub enum AgentHarnessToolContextSource {
    /// One static context value shared by every turn.
    Static(ToolContext),
    /// A provider resolved per turn snapshot.
    Resolved(ToolContextProvider),
}

impl fmt::Debug for AgentHarnessToolContextSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(..) => f.write_str("AgentHarnessToolContextSource::Static(..)"),
            Self::Resolved(..) => f.write_str("AgentHarnessToolContextSource::Resolved(..)"),
        }
    }
}

/// Curated provider request options owned by the harness and snapshotted
/// per turn, upstream's `AgentHarnessStreamOptions`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessStreamOptions {
    /// Preferred transport forwarded to the stream function.
    pub transport: Option<Transport>,
    /// Provider request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum provider retry attempts.
    pub max_retries: Option<u32>,
    /// Optional cap for provider-requested retry delays.
    pub max_retry_delay_ms: Option<u64>,
    /// Additional request headers merged with auth and lifecycle headers.
    pub headers: Option<BTreeMap<String, String>>,
    /// Provider metadata forwarded with requests.
    pub metadata: Option<BTreeMap<String, JsonValue>>,
    /// Provider cache retention hint.
    pub cache_retention: Option<CacheRetention>,
    /// Ask a capable provider to continue generation asynchronously,
    /// upstream's `deferred?: boolean | { window? }`.
    pub deferred: Option<DeferredRequest>,
}

/// Per-request stream option patch returned by provider hooks, upstream's
/// `AgentHarnessStreamOptionsPatch`.
///
/// Each field is `None` when absent, `Some(None)` to delete the base
/// value, and `Some(Some(value))` to set it. `headers` and `metadata`
/// merge per key, where a `None` value deletes one key; a map-level
/// `Some(None)` clears the whole map.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessStreamOptionsPatch {
    /// Preferred transport patch.
    pub transport: Option<Option<Transport>>,
    /// Timeout patch, in milliseconds.
    pub timeout_ms: Option<Option<u64>>,
    /// Retry-attempt patch.
    pub max_retries: Option<Option<u32>>,
    /// Retry-delay cap patch, in milliseconds.
    pub max_retry_delay_ms: Option<Option<u64>>,
    /// Cache-retention patch.
    pub cache_retention: Option<Option<CacheRetention>>,
    /// Deferred-execution patch.
    pub deferred: Option<Option<DeferredRequest>>,
    /// Header patch; per-key `None` deletes, map-level `None` clears all.
    pub headers: Option<Option<ProviderHeaders>>,
    /// Metadata patch; per-key `None` deletes, map-level `None` clears all.
    pub metadata: Option<Option<BTreeMap<String, Option<JsonValue>>>>,
}

/// Which kind of filesystem object a path addresses, upstream's `FileKind`.
/// Symlinks are not followed automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileKind {
    /// A regular file.
    File,
    /// A directory.
    Directory,
    /// A symbolic link; targets are never followed implicitly.
    Symlink,
}

/// Stable, backend-independent file error codes returned by [`FileSystem`]
/// file operations, upstream's `FileErrorCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileErrorCode {
    /// The operation was aborted through the context signal.
    Aborted,
    /// The addressed path does not exist.
    NotFound,
    /// The OS denied the operation.
    PermissionDenied,
    /// A parent component of the path is not a directory.
    NotDirectory,
    /// The addressed path is a directory where a file was required.
    IsDirectory,
    /// The arguments or addressed object are invalid for the operation.
    Invalid,
    /// The backend does not support the operation.
    NotSupported,
    /// Any other backend failure.
    Unknown,
}

/// Error returned by [`FileSystem`] file operations, upstream's `FileError`.
///
/// The `name` field upstream carries is the type identity itself; the
/// backend-independent `code` is the stable discriminator tests match on.
#[derive(Debug)]
pub struct FileError {
    /// Backend-independent error code.
    pub code: FileErrorCode,
    /// The human-readable failure message.
    pub message: String,
    /// Absolute addressed path associated with the failure, when available.
    pub path: Option<String>,
    /// The underlying backend error, when one produced this failure.
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl FileError {
    /// Builds a file error with its code, message, addressed path, and
    /// optional cause.
    #[must_use]
    pub fn new(
        code: FileErrorCode,
        message: impl Into<String>,
        path: Option<String>,
        cause: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            path,
            source: cause,
        }
    }
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| {
            let coerced: &(dyn std::error::Error + 'static) = &**source;
            coerced
        })
    }
}

/// Stable, backend-independent execution error codes returned by
/// [`Shell::exec`], upstream's `ExecutionErrorCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionErrorCode {
    /// The command was aborted through the context signal.
    Aborted,
    /// The command exceeded its timeout.
    Timeout,
    /// No usable shell is configured or discoverable.
    ShellUnavailable,
    /// The shell process could not spawn.
    SpawnError,
    /// An output callback supplied by the caller failed.
    CallbackError,
    /// Any other backend failure.
    Unknown,
}

/// Error returned by [`Shell::exec`], upstream's `ExecutionError`.
#[derive(Debug)]
pub struct ExecutionError {
    /// Backend-independent error code.
    pub code: ExecutionErrorCode,
    /// The human-readable failure message.
    pub message: String,
    /// The underlying error, when one produced this failure.
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl ExecutionError {
    /// Builds an execution error with its code, message, and optional
    /// cause.
    #[must_use]
    pub fn new(
        code: ExecutionErrorCode,
        message: impl Into<String>,
        cause: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            source: cause,
        }
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| {
            let coerced: &(dyn std::error::Error + 'static) = &**source;
            coerced
        })
    }
}

/// Stable compaction error codes returned by compaction helpers, upstream's
/// `CompactionErrorCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionErrorCode {
    /// The compaction was aborted through the context signal.
    Aborted,
    /// The summarization model call failed.
    SummarizationFailed,
}

/// Error returned by compaction helpers, upstream's `CompactionError`.
#[derive(Debug)]
pub struct CompactionError {
    /// Backend-independent error code.
    pub code: CompactionErrorCode,
    /// The human-readable failure message.
    pub message: String,
    /// The underlying error, when one produced this failure.
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl CompactionError {
    /// Builds a compaction error with its code, message, and optional
    /// cause.
    #[must_use]
    pub fn new(
        code: CompactionErrorCode,
        message: impl Into<String>,
        cause: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            source: cause,
        }
    }
}

impl fmt::Display for CompactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CompactionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| {
            let coerced: &(dyn std::error::Error + 'static) = &**source;
            coerced
        })
    }
}

/// Stable branch-summary error codes returned by branch summarization
/// helpers, upstream's `BranchSummaryErrorCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchSummaryErrorCode {
    /// The summarization was aborted through the context signal.
    Aborted,
    /// The summarization model call failed.
    SummarizationFailed,
}

/// Error returned by branch summarization helpers, upstream's
/// `BranchSummaryError`.
#[derive(Debug)]
pub struct BranchSummaryError {
    /// Backend-independent error code.
    pub code: BranchSummaryErrorCode,
    /// The human-readable failure message.
    pub message: String,
    /// The underlying error, when one produced this failure.
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl BranchSummaryError {
    /// Builds a branch-summary error with its code, message, and optional
    /// cause.
    #[must_use]
    pub fn new(
        code: BranchSummaryErrorCode,
        message: impl Into<String>,
        cause: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            source: cause,
        }
    }
}

impl fmt::Display for BranchSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BranchSummaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| {
            let coerced: &(dyn std::error::Error + 'static) = &**source;
            coerced
        })
    }
}

/// Metadata for one filesystem object in a [`FileSystem`], upstream's
/// `FileInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileInfo {
    /// Basename of [`path`](FileInfo::path).
    pub name: String,
    /// Absolute, syntactically normalized addressed path in the execution
    /// environment. Symlinks are not followed.
    pub path: String,
    /// Object kind. Symlink targets are not followed; use
    /// [`FileSystem::canonical_path`] explicitly.
    pub kind: FileKind,
    /// Size in bytes for the addressed filesystem object.
    pub size: u64,
    /// Modification time as milliseconds since the Unix epoch.
    pub mtime_ms: i64,
}

/// One UTF-8 line read from a text file, upstream's `TextLine`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextLine {
    /// The line's text without its terminator.
    pub text: String,
    /// Whether the line ended with `\n`; callers use this to discard a torn
    /// final record.
    pub terminated: bool,
}

/// Options for [`FileSystem::read_text_lines`], upstream's
/// `{ maxLines?: number }`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadTextLinesOptions {
    /// Stop after this many lines.
    pub max_lines: Option<u64>,
}

/// Pull-based UTF-8 line reader that preserves final-line termination,
/// upstream's `TextLineReader`.
pub trait TextLineReader: Send {
    /// Read the next line, or `None` at end of file. A torn final record
    /// reports `terminated: false`.
    fn read_line<'a>(
        &'a mut self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Option<TextLine>, FileError>>;

    /// Release the open file. Best-effort; never fails.
    fn close<'a>(&'a mut self, context: &'a Context) -> BoxedFuture<'a, ()>;
}

/// Options for [`FileSystem::create_dir`], upstream's
/// `{ recursive?: boolean }`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CreateDirOptions {
    /// Create missing parent directories. Defaults to `true`.
    pub recursive: Option<bool>,
}

/// Options for [`FileSystem::remove`], upstream's
/// `{ recursive?: boolean; force?: boolean }`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RemoveOptions {
    /// Remove directories with their contents. Defaults to `false`.
    pub recursive: Option<bool>,
    /// Ignore missing paths instead of reporting `not_found`. Defaults to
    /// `false`.
    pub force: Option<bool>,
}

/// Options for [`FileSystem::create_temp_file`], upstream's
/// `{ prefix?: string; suffix?: string }`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TempFileOptions {
    /// Filename prefix. Defaults to `""`.
    pub prefix: Option<String>,
    /// Filename suffix. Defaults to `""`.
    pub suffix: Option<String>,
}

/// Filesystem capability used by the harness, upstream's `FileSystem`.
///
/// Paths passed to methods may be absolute or relative to [`cwd`](FileSystem::cwd).
/// Paths returned by file operations are addressed paths in the filesystem
/// namespace, but are not canonicalized through symlinks unless returned by
/// [`canonical_path`](FileSystem::canonical_path).
///
/// Operation methods must never panic or reject. All filesystem failures,
/// including unexpected backend failures, must be encoded in the returned
/// [`Result::Err`]. Implementations must preserve this invariant.
pub trait FileSystem: Send + Sync {
    /// Current working directory for relative paths.
    fn cwd(&self) -> &str;

    /// Return an absolute addressed path without requiring it to exist and
    /// without resolving symlinks.
    fn absolute_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>>;

    /// Join path segments in the filesystem namespace without requiring the
    /// result to exist.
    fn join_path<'a>(
        &'a self,
        parts: &'a [String],
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>>;

    /// Read a UTF-8 text file.
    fn read_text_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>>;

    /// Open a UTF-8 text file for pull-based line reading.
    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Box<dyn TextLineReader>, FileError>>;

    /// Read UTF-8 text lines. Implementations should stop once `max_lines`
    /// lines have been read.
    fn read_text_lines<'a>(
        &'a self,
        path: &'a str,
        options: Option<ReadTextLinesOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<String>, FileError>>;

    /// Read a binary file.
    fn read_binary_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<u8>, FileError>>;

    /// Create or overwrite a file, creating parent directories when
    /// supported.
    fn write_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>>;

    /// Create or append to a file, creating parent directories when
    /// supported.
    fn append_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>>;

    /// Atomically rename a file, replacing the destination when it exists.
    /// Does not copy across filesystems.
    fn rename_file<'a>(
        &'a self,
        source_path: &'a str,
        destination_path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>>;

    /// Return metadata for the addressed path without following symlinks.
    fn file_info<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<FileInfo, FileError>>;

    /// List direct children of a directory without following symlinks.
    fn list_dir<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<FileInfo>, FileError>>;

    /// Return the canonical path for an existing path, resolving symlinks
    /// where supported.
    fn canonical_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>>;

    /// Return `false` for missing paths. Other errors, such as permission
    /// failures, return a [`FileError`].
    fn exists<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<bool, FileError>>;

    /// Create a directory. Defaults to `recursive: true`.
    fn create_dir<'a>(
        &'a self,
        path: &'a str,
        options: Option<CreateDirOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>>;

    /// Remove a file or directory. Defaults to `recursive: false` and
    /// `force: false`.
    fn remove<'a>(
        &'a self,
        path: &'a str,
        options: Option<RemoveOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>>;

    /// Create a temporary directory and return its absolute path. Defaults
    /// to prefix `"tmp-"`.
    fn create_temp_dir<'a>(
        &'a self,
        prefix: Option<&'a str>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>>;

    /// Create a temporary file and return its absolute path. Defaults to
    /// prefix `""` and suffix `""`.
    fn create_temp_file<'a>(
        &'a self,
        options: Option<TempFileOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>>;

    /// Release filesystem resources. Best-effort; never fails.
    fn cleanup<'a>(&'a self, context: &'a Context) -> BoxedFuture<'a, ()>;
}

/// File content for the write operations, upstream's `string | Uint8Array`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileContent {
    /// UTF-8 text.
    Text(String),
    /// Raw bytes.
    Bytes(Vec<u8>),
}

impl From<&str> for FileContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for FileContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<u8>> for FileContent {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

/// Which portion of bounded output survives after the limit is crossed,
/// upstream's `ShellOutputRetention`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShellOutputRetention {
    /// Keep the first lines and bytes.
    Head,
    /// Keep the last lines. The default.
    #[default]
    Tail,
}

/// Source-side limits for one combined shell output view, upstream's
/// `ShellOutputLimits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShellOutputLimits {
    /// Maximum retained bytes.
    pub max_bytes: u64,
    /// Maximum retained lines.
    pub max_lines: u64,
    /// Which portion survives after the limit is crossed. Defaults to
    /// [`ShellOutputRetention::Tail`].
    pub retain: Option<ShellOutputRetention>,
}

/// Bounded shell capture requested by the caller, upstream's
/// `ShellOutputCaptureOptions`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShellOutputCaptureOptions {
    /// The bounds of the retained view.
    pub limits: ShellOutputLimits,
    /// Preserve complete output in an execution-environment-local file
    /// after the limits are crossed.
    pub spill: bool,
}

/// Truncation metadata without a duplicate copy of the retained text,
/// upstream's `ShellOutputTruncation = Omit<TruncationResult, "content">`.
///
/// The fields mirror pi-ai's truncate belt's `TruncationResult` minus its
/// `content` copy; `TruncationResult` itself derefs to this type.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellOutputTruncation {
    /// Whether truncation occurred.
    pub truncated: bool,
    /// Which limit was hit: lines or bytes; `None` when not truncated.
    pub truncated_by: Option<TruncatedBy>,
    /// Total number of lines in the original content.
    pub total_lines: u64,
    /// Total number of bytes in the original content.
    pub total_bytes: u64,
    /// Number of complete lines in the truncated output.
    pub output_lines: u64,
    /// Number of bytes in the truncated output.
    pub output_bytes: u64,
    /// Whether the last line was partially truncated, for the tail
    /// truncation edge case only.
    pub last_line_partial: bool,
    /// Whether the first line exceeded the byte limit, for head truncation.
    pub first_line_exceeds_limit: bool,
    /// The max-lines limit that was applied.
    pub max_lines: u64,
    /// The max-bytes limit that was applied.
    pub max_bytes: u64,
}

/// Which independent limit a truncation hit, upstream's
/// `"lines" | "bytes" | null`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncatedBy {
    /// The line limit was hit.
    Lines,
    /// The byte limit was hit.
    Bytes,
}

/// Metadata accompanying a bounded shell output view, upstream's
/// `ShellOutputMetadata`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellOutputMetadata {
    /// Truncation metadata without a duplicate copy of the retained text.
    pub truncation: ShellOutputTruncation,
    /// The execution-environment-local file preserving complete output,
    /// when the spill was created.
    pub spill_path: Option<String>,
    /// Bytes of the partially truncated last line, when the tail retention
    /// cut mid-line.
    pub last_line_bytes: Option<u64>,
}

/// Complete bounded shell output view, upstream's `ShellOutputView`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellOutputView {
    /// Metadata accompanying the view.
    pub metadata: ShellOutputMetadata,
    /// The retained, sanitized output text.
    pub text: String,
}

/// Incremental source-side change to one bounded shell output view,
/// upstream's `ShellOutputUpdate`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ShellOutputUpdate {
    /// The view was replaced wholesale.
    Replace {
        /// The complete new view.
        output: ShellOutputView,
    },
    /// Bytes appended past the previous view's end.
    Append {
        /// The appended, sanitized text.
        text: String,
        /// The view's metadata after the append.
        metadata: ShellOutputMetadata,
    },
    /// The window slid: `drop` leading characters left the view and `text`
    /// appended behind them.
    Slide {
        /// Number of leading characters that left the window.
        drop: usize,
        /// The appended, sanitized text.
        text: String,
        /// The view's metadata after the slide.
        metadata: ShellOutputMetadata,
    },
    /// Only metadata changed; the text is unchanged.
    Metadata {
        /// The view's new metadata.
        metadata: ShellOutputMetadata,
    },
}

/// Bounded shell completion, upstream's `ShellExecResult`. Output text is
/// delivered through [`ShellExecOptions::on_update`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellExecResult {
    /// Process exit code, or `128 + signal` for a signal-killed process.
    pub exit_code: i32,
    /// Truncation metadata without a duplicate copy of the retained text.
    pub truncation: ShellOutputTruncation,
    /// The spill file path, when the capture spilled.
    pub spill_path: Option<String>,
    /// Bytes of the partially truncated last line, when present.
    pub last_line_bytes: Option<u64>,
}

/// The output callback [`Shell::exec`] reports bounded view changes
/// through, upstream's `ShellExecOptions.onUpdate`.
pub type OnShellOutputUpdate = dyn Fn(&ShellOutputUpdate, &Context) + Send + Sync;

/// Options for [`Shell::exec`], upstream's `ShellExecOptions`.
#[derive(Clone, Default)]
pub struct ShellExecOptions {
    /// Working directory for the command. Relative paths are resolved
    /// against the execution environment's cwd. Defaults to the
    /// environment's cwd.
    pub cwd: Option<String>,
    /// Environment variables for the command. Values override inherited
    /// defaults when `inherit_env` is true.
    pub env: Option<BTreeMap<String, String>>,
    /// Whether to inherit the execution environment's default variables.
    /// Defaults to `true`.
    pub inherit_env: Option<bool>,
    /// Timeout in seconds. Implementations return a timeout error when the
    /// command exceeds this duration. Defaults to no timeout.
    pub timeout: Option<f64>,
    /// Source-side bounded capture. Output is discarded when this and
    /// `on_update` are both absent.
    pub capture: Option<ShellOutputCaptureOptions>,
    /// Called with bounded output changes.
    pub on_update: Option<Arc<OnShellOutputUpdate>>,
}

impl fmt::Debug for ShellExecOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShellExecOptions")
            .field("cwd", &self.cwd)
            .field("env", &self.env)
            .field("inherit_env", &self.inherit_env)
            .field("timeout", &self.timeout)
            .field("capture", &self.capture)
            .field("on_update", &self.on_update.is_some())
            .finish()
    }
}

/// Shell execution capability used by the harness, upstream's `Shell`.
pub trait Shell: Send + Sync {
    /// Execute a shell command in the filesystem cwd unless
    /// `options.cwd` is provided.
    fn exec<'a>(
        &'a self,
        command: &'a str,
        options: Option<ShellExecOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<ShellExecResult, ExecutionError>>;

    /// Release shell resources. Best-effort; never fails.
    fn cleanup<'a>(&'a self, context: &'a Context) -> BoxedFuture<'a, ()>;
}

/// Filesystem and process execution environment used by the harness,
/// upstream's `ExecutionEnv extends FileSystem, Shell`.
pub trait ExecutionEnv: FileSystem + Shell {}

/// The number of provider retry attempts the harness defaults to, restated
/// over pi-ai's `RetryPolicy` by [`crate::harness::config::default_retry_policy`].
pub use pi_ai::utils::retry::DEFAULT_MAX_AGENT_RETRY_DELAY_MS;

#[cfg(test)]
mod tests;
