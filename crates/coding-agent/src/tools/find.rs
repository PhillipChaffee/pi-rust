//! The find tool, ported from upstream `src/core/tools/find.ts` with this
//! slice's recorded restatement.
//!
//! The `fd` spawn (its binary, downloaded by the tools manager) restates on
//! the `ignore` crate's walk — fd's own gitignore engine — with fd's glob
//! semantics carried by `globset` and the basename/full-path split plus the
//! `**/` full-path prefixing the `3302-find-path-glob` regression pins. The
//! hierarchical `.gitignore` scoping the `3303-find-nested-gitignore`
//! regression pins is the walker's native behavior, with fd's
//! `--no-require-git` outside git repos ported as the walk-up `.git` probe.
//! Windows is out of scope for this effort (map ticket "Decide the Rust
//! stack"), so the win32 separator branch rests on that ruling.

use std::fmt::Write as _;
use std::sync::Arc;

use globset::GlobBuilder;
use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::AgentHarnessTool;
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, TextContent};
use serde_json::{Value, json};

use crate::extensions::types::{ExtensionContext, ToolDefinition};

use super::bash::SystemPromptContribution;
use super::io_error;
use super::path_utils::{path_exists, resolve_to_cwd};
use super::tool_definition_wrapper::wrap_tool_definition;
use super::truncate::{DEFAULT_MAX_BYTES, TruncationOptions, TruncationResult, truncate_head};

/// The default result limit, upstream's `DEFAULT_LIMIT`.
pub const DEFAULT_LIMIT: usize = 1000;

/// The find tool's input, upstream's `FindToolInput`.
#[derive(Clone, Debug, PartialEq)]
pub struct FindToolInput {
    /// The glob pattern to match files.
    pub pattern: String,
    /// The directory to search in.
    pub path: Option<String>,
    /// The maximum number of results.
    pub limit: Option<f64>,
}

/// The find tool's details, upstream's `FindToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FindToolDetails {
    /// The truncation metadata, when the output truncated.
    pub truncation: Option<TruncationResult>,
    /// The result limit that was reached, serialized as
    /// `resultLimitReached`.
    pub result_limit_reached: Option<usize>,
}

impl FindToolDetails {
    /// The wire shape with absent members dropped.
    pub(crate) fn to_wire(&self) -> Value {
        let mut object = serde_json::Map::new();
        if let Some(truncation) = &self.truncation {
            object.insert(
                "truncation".to_owned(),
                serde_json::to_value(truncation).unwrap_or_default(),
            );
        }
        if let Some(result_limit_reached) = self.result_limit_reached {
            object.insert("resultLimitReached".to_owned(), json!(result_limit_reached));
        }
        Value::Object(object)
    }
}

/// Relativize a find result against the search root and normalize it to
/// posix separators, upstream's `relativizeFindResultPath`.
///
/// Its `pathModule` parameter carries the win32 separator; the posix module
/// is the only one this effort targets.
#[must_use]
pub fn relativize_find_result_path(result_path: &str, search_path: &str) -> String {
    const SEPARATOR: char = '/';
    let had_trailing_separator = result_path.ends_with(SEPARATOR);
    let relative_path = if std::path::Path::new(result_path).is_absolute() {
        path_relative(result_path, search_path)
    } else {
        result_path.to_owned()
    };
    let posix_path = relative_path.replace(SEPARATOR, "/");
    if had_trailing_separator && !posix_path.ends_with('/') {
        format!("{posix_path}/")
    } else {
        posix_path
    }
}

/// Node's `path.relative(from, to)` for the absolute paths the walk yields,
/// over the lexical posix shapes.
fn path_relative(to: &str, from: &str) -> String {
    let normalize = |path: &str| -> Vec<String> {
        path.split('/')
            .filter(|segment| !segment.is_empty() && *segment != ".")
            .fold(Vec::new(), |mut stack, segment| {
                if segment == ".." {
                    stack.pop();
                } else {
                    stack.push(segment.to_owned());
                }
                stack
            })
    };
    let to_parts = normalize(to);
    let from_parts = normalize(from);
    let mut common = 0usize;
    while common < to_parts.len()
        && common < from_parts.len()
        && to_parts[common] == from_parts[common]
    {
        common += 1;
    }
    let mut parts: Vec<String> = Vec::new();
    for _ in common..from_parts.len() {
        parts.push("..".to_owned());
    }
    parts.extend(to_parts[common..].iter().cloned());
    if parts.is_empty() {
        ".".to_owned()
    } else {
        parts.join("/")
    }
}

/// The pluggable operations for the find tool, upstream's
/// `FindOperations`. Override to delegate file search to remote systems.
#[derive(Clone)]
pub struct FindOperations {
    /// Check the path exists, upstream's `exists`.
    pub exists: FindExistsFn,
    /// Find files matching the glob pattern, upstream's `glob` (the custom
    /// backends return relative or absolute paths).
    pub glob: Option<FindGlobFn>,
}

/// The erased `exists` operation.
pub type FindExistsFn = Arc<dyn Fn(String) -> BoxedFuture<'static, bool> + Send + Sync>;
/// The erased `glob` operation.
pub type FindGlobFn =
    Arc<dyn Fn(String, String, FindGlobOptions) -> BoxedFuture<'static, Vec<String>> + Send + Sync>;

/// The glob call's options, upstream's `{ ignore, limit }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FindGlobOptions {
    /// The ignore patterns the backend applies.
    pub ignore: Vec<String>,
    /// The result cap.
    pub limit: usize,
}

impl std::fmt::Debug for FindOperations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FindOperations")
            .field("glob", &self.glob.is_some())
            .finish_non_exhaustive()
    }
}

fn default_find_operations() -> FindOperations {
    FindOperations {
        exists: Arc::new(|path| Box::pin(async move { path_exists(&path).await })),
        // The placeholder: the default glob execution happens in the
        // execute body, upstream's comment.
        glob: None,
    }
}

/// The find tool's options, upstream's `FindToolOptions`.
#[derive(Clone, Default)]
pub struct FindToolOptions {
    /// Custom operations for find; default, local filesystem plus the
    /// native walk.
    pub operations: Option<FindOperations>,
}

impl std::fmt::Debug for FindToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FindToolOptions")
            .field("operations", &self.operations.is_some())
            .finish()
    }
}

/// The find tool's system-prompt contribution, upstream's
/// `findToolSystemPromptContribution`.
pub const FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION: SystemPromptContribution =
    SystemPromptContribution {
        snippet: "Find files by glob pattern (respects .gitignore)",
        guidelines: &[],
    };

/// The find tool's schema, upstream's `findSchema`.
fn find_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of results (default: 1000)"
            }
        },
        "required": ["pattern"]
    })
}

fn parse_input(params: &Value) -> Result<FindToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        pattern: String,
        path: Option<String>,
        limit: Option<f64>,
    }
    serde_json::from_value::<RawInput>(params.clone())
        .map(|raw| FindToolInput {
            pattern: raw.pattern,
            path: raw.path,
            limit: raw.limit,
        })
        .map_err(|error| io_error(error.to_string()))
}

/// Compile the find glob, fd's two modes: a pattern containing `/` matches
/// the full relative path with the `**/` prefix fd's `--full-path` mode
/// needs, otherwise the basename. Returns the compiled matcher.
///
/// # Errors
/// The glob-compile failure, upstream's `error parsing glob` rejection.
fn compile_find_glob(pattern: &str) -> Result<Option<globset::GlobMatcher>, String> {
    let effective_pattern = if pattern.contains('/')
        && !pattern.starts_with('/')
        && !pattern.starts_with("**/")
        && pattern != "**"
    {
        format!("**/{pattern}")
    } else {
        pattern.to_owned()
    };
    let matcher = GlobBuilder::new(&effective_pattern)
        .literal_separator(true)
        .build()
        .map_err(|error| error.to_string())?
        .compile_matcher();
    Ok(Some(matcher))
}

/// Whether the walked entry matches the compiled glob, fd's basename /
/// full-path split over the search-root-relative path.
fn matches_entry(matcher: Option<&globset::GlobMatcher>, relative: &str) -> bool {
    let Some(matcher) = matcher else { return false };
    if matcher.glob().glob().contains('/') {
        // Full-path mode.
        return matcher.is_match(relative);
    }
    // Basename mode, fd's default.
    let basename = relative.rsplit('/').next().unwrap_or(relative);
    matcher.is_match(basename)
}

/// The native walk, upstream's fd spawn: the `ignore` crate's walker with
/// `--hidden` (hidden files included), hierarchical gitignore scoping, and
/// fd's git-requirement default outside repos. Yields the
/// search-root-relative paths in the walker's order, stopping once the
/// result cap is met.
fn walk_candidates(
    search_path: &str,
    matcher: Option<&globset::GlobMatcher>,
    effective_limit: usize,
) -> Vec<String> {
    let inside_git = walk_up_inside_git_repo(search_path);
    let mut builder = ignore::WalkBuilder::new(search_path);
    builder.hidden(false); // fd --hidden
    builder.git_ignore(true);
    builder.require_git(inside_git);
    builder.parents(true);
    let walker = builder.build();
    let mut relativized: Vec<String> = Vec::new();
    for entry in walker {
        if relativized.len() >= effective_limit {
            break;
        }
        let Ok(entry) = entry else { continue };
        let path = entry.path().to_string_lossy();
        let Some(relative) = path.strip_prefix(search_path) else {
            continue;
        };
        let relative = relative.trim_start_matches('/');
        if relative.is_empty() {
            continue;
        }
        if matches_entry(matcher, relative) {
            relativized.push(relative.replace(std::path::MAIN_SEPARATOR, "/"));
        }
    }
    relativized
}

/// The walk-up `.git` probe on the blocking thread the walk already runs
/// on, upstream's walk-up loop.
fn walk_up_inside_git_repo(search_path: &str) -> bool {
    let mut current = std::path::PathBuf::from(search_path);
    loop {
        if std::path::Path::new(&current.join(".git")).exists() {
            return true;
        }
        let Some(parent) = current.parent() else {
            return false;
        };
        if parent == current {
            return false;
        }
        current = parent.to_path_buf();
    }
}

/// The result formatting the custom-glob and default branches share,
/// upstream's duplicated settle body. `with_hint` selects the default
/// branch's longer limit notice.
fn format_results(
    relativized: &[String],
    effective_limit: usize,
    with_hint: bool,
) -> (String, Option<Value>) {
    let result_limit_reached = relativized.len() >= effective_limit;
    let raw_output = relativized.join("\n");
    let truncation = truncate_head(
        &raw_output,
        TruncationOptions {
            max_lines: Some(usize::MAX),
            ..TruncationOptions::default()
        },
    );
    let mut result_output = truncation.content.clone();
    let mut details = FindToolDetails::default();
    let mut notices: Vec<String> = Vec::new();
    if result_limit_reached {
        if with_hint {
            notices.push(format!(
                "{effective_limit} results limit reached. Use limit={} for more, or refine pattern",
                effective_limit * 2
            ));
        } else {
            notices.push(format!("{effective_limit} results limit reached"));
        }
        details.result_limit_reached = Some(effective_limit);
    }
    if truncation.truncated {
        notices.push(format!(
            "{} limit reached",
            super::truncate::format_size(DEFAULT_MAX_BYTES)
        ));
        details.truncation = Some(truncation);
    }
    if !notices.is_empty() {
        let _ = write!(result_output, "\n\n[{}]", notices.join(". "));
    }
    (
        result_output,
        (details.truncation.is_some() || details.result_limit_reached.is_some())
            .then(|| details.to_wire()),
    )
}

/// The find tool's execution body, upstream's `createFindToolDefinition`
/// execute.
async fn execute_find_tool(
    params: &Value,
    signal: Option<&AbortSignal>,
    ctx: Option<&dyn ExtensionContext>,
    ops: &FindOperations,
    cwd: &str,
) -> Result<AgentToolResult, AgentToolError> {
    let input = parse_input(params)?;
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }
    let effective_cwd = ctx
        .map(ExtensionContext::cwd)
        .filter(|ctx_cwd| !ctx_cwd.is_empty())
        .unwrap_or(cwd);
    let search_path = resolve_to_cwd(input.path.as_deref().unwrap_or("."), effective_cwd)
        .map_err(|error| io_error(error.to_string()))?;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a fractional limit floors to a whole result count, upstream's Math.max(0, limit)"
    )]
    let effective_limit = input
        .limit
        .map_or(DEFAULT_LIMIT, |limit| limit.max(0.0) as usize)
        .max(1);

    // If custom operations provide glob(), use that instead of the native
    // walk, upstream's custom-ops branch.
    if let Some(glob) = ops.glob.as_ref() {
        if !(ops.exists)(search_path.clone()).await {
            return Err(io_error(format!("Path not found: {search_path}")));
        }
        if signal.is_some_and(AbortSignal::aborted) {
            return Err(io_error("Operation aborted"));
        }
        let results = glob(
            input.pattern.clone(),
            search_path.clone(),
            FindGlobOptions {
                ignore: vec!["**/node_modules/**".to_owned(), "**/.git/**".to_owned()],
                limit: effective_limit,
            },
        )
        .await;
        if signal.is_some_and(AbortSignal::aborted) {
            return Err(io_error("Operation aborted"));
        }
        if results.is_empty() {
            return Ok(AgentToolResult {
                content: vec![AgentToolContent::Text(TextContent {
                    text: "No files found matching pattern".to_owned(),
                    text_signature: None,
                })],
                details: Value::Null,
                usage: None,
                added_tool_names: None,
                terminate: None,
            });
        }

        // Relativize paths against the search root for stable output.
        let relativized: Vec<String> = results
            .iter()
            .map(|path| relativize_find_result_path(path, &search_path))
            .collect();
        let (result_output, details) = format_results(&relativized, effective_limit, false);
        return Ok(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: result_output,
                text_signature: None,
            })],
            details: details.unwrap_or(Value::Null),
            usage: None,
            added_tool_names: None,
            terminate: None,
        });
    }

    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }

    // The native walk restatement of the fd spawn.
    let matcher = compile_find_glob(&input.pattern)
        .map_err(|error| io_error(format!("error parsing glob: {error}")))?;
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }
    let relativized = walk_candidates(&search_path, matcher.as_ref(), effective_limit);

    if relativized.is_empty() {
        return Ok(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: "No files found matching pattern".to_owned(),
                text_signature: None,
            })],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            terminate: None,
        });
    }

    let (result_output, details) = format_results(&relativized, effective_limit, true);
    Ok(AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: result_output,
            text_signature: None,
        })],
        details: details.unwrap_or(Value::Null),
        usage: None,
        added_tool_names: None,
        terminate: None,
    })
}

/// Build the find tool definition, upstream's `createFindToolDefinition`.
#[must_use]
pub fn create_find_tool_definition(cwd: &str, options: Option<FindToolOptions>) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let ops = Arc::new(options.operations.unwrap_or_else(default_find_operations));
    let cwd = Arc::new(cwd.to_owned());

    ToolDefinition {
        name: "find".to_owned(),
        label: "find".to_owned(),
        description: format!(
            "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or {}KB (whichever is hit first).",
            DEFAULT_MAX_BYTES / 1024
        ),
        prompt_snippet: Some(FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet.to_owned()),
        prompt_guidelines: Some(
            FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION
                .guidelines
                .iter()
                .map(|guideline| (*guideline).to_owned())
                .collect(),
        ),
        parameters: find_schema(),
        constrained_sampling: None,
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(
            move |_tool_call_id: &str, params: &Value, signal, _on_update, ctx| {
                let ops = Arc::clone(&ops);
                let cwd = Arc::clone(&cwd);
                Box::pin(async move { execute_find_tool(params, signal, ctx, &ops, &cwd).await })
            },
        ),
    }
}

/// Build the find tool, upstream's `createFindTool`.
#[must_use]
pub fn create_find_tool(cwd: &str, options: Option<FindToolOptions>) -> AgentHarnessTool {
    let definition = create_find_tool_definition(cwd, options);
    wrap_tool_definition::<crate::extensions::types::CwdContext>(definition, None)
}
