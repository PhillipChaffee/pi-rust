//! The grep tool, ported from upstream `src/core/tools/grep.ts` with this
//! slice's recorded restatement.
//!
//! The `rg` spawn restates on the `ignore` crate's walk (ripgrep's own
//! gitignore engine, its git-requirement default carried) plus the `regex`
//! crate — ripgrep's own regex engine — with the JSON-event stream
//! collapsing into a direct scan. The binary skip, the flag-like pattern
//! safety (the injected `-e` argv can no longer reach a shell), and the
//! match-limit kill restate as the scan's caps.

use std::fmt::Write as _;
use std::sync::Arc;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::AgentHarnessTool;
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, TextContent};
use serde_json::{Value, json};

use crate::extensions::types::{ExtensionContext, ToolDefinition};

use super::bash::SystemPromptContribution;
use super::io_error;
use super::path_utils::resolve_to_cwd;
use super::tool_definition_wrapper::wrap_tool_definition;
use super::truncate::{
    DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH, TruncationOptions, TruncationResult, truncate_head,
    truncate_line_default,
};

/// The default match limit, upstream's `DEFAULT_LIMIT`.
pub const DEFAULT_LIMIT: usize = 100;

/// One scan hit, the collapsed shape of rg's JSON `match` event.
struct Match {
    file_path: String,
    line_number: usize,
    line_text: Option<String>,
}

/// The grep tool's input, upstream's `GrepToolInput`.
#[derive(Clone, Debug, PartialEq)]
pub struct GrepToolInput {
    /// The search pattern (regex or literal string).
    pub pattern: String,
    /// The directory or file to search.
    pub path: Option<String>,
    /// The glob filter.
    pub glob: Option<String>,
    /// Case-insensitive search.
    pub ignore_case: Option<bool>,
    /// Treat the pattern as a literal string.
    pub literal: Option<bool>,
    /// Context lines before and after each match.
    pub context: Option<f64>,
    /// The maximum number of matches.
    pub limit: Option<f64>,
}

/// The grep tool's details, upstream's `GrepToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrepToolDetails {
    /// The truncation metadata, when the output truncated.
    pub truncation: Option<TruncationResult>,
    /// The match limit that was reached, serialized as `matchLimitReached`.
    pub match_limit_reached: Option<usize>,
    /// Whether long lines were truncated, serialized as `linesTruncated`.
    pub lines_truncated: Option<bool>,
}

impl GrepToolDetails {
    /// The wire shape with absent members dropped.
    pub(crate) fn to_wire(&self) -> Value {
        let mut object = serde_json::Map::new();
        if let Some(truncation) = &self.truncation {
            object.insert(
                "truncation".to_owned(),
                serde_json::to_value(truncation).unwrap_or_default(),
            );
        }
        if let Some(match_limit_reached) = self.match_limit_reached {
            object.insert("matchLimitReached".to_owned(), json!(match_limit_reached));
        }
        if let Some(lines_truncated) = self.lines_truncated {
            object.insert("linesTruncated".to_owned(), json!(lines_truncated));
        }
        Value::Object(object)
    }
}

/// The pluggable operations for the grep tool, upstream's
/// `GrepOperations`. Override to delegate search to remote systems.
#[derive(Clone)]
pub struct GrepOperations {
    /// Check the path is a directory (throws when the path does not
    /// exist), upstream's `isDirectory`.
    pub is_directory: GrepIsDirectoryFn,
    /// Read file contents for context lines, upstream's `readFile`.
    pub read_file: GrepReadFileFn,
}

/// The erased `isDirectory` operation: `None` models the throw upstream's
/// catch turns into `Path not found`.
pub type GrepIsDirectoryFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Option<bool>> + Send + Sync>;
/// The erased `readFile` operation: the error string models the throw
/// upstream's `getFileLines` turns into an empty file.
pub type GrepReadFileFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<String, String>> + Send + Sync>;

impl std::fmt::Debug for GrepOperations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrepOperations").finish_non_exhaustive()
    }
}

fn default_grep_operations() -> GrepOperations {
    GrepOperations {
        is_directory: Arc::new(|path| {
            Box::pin(async move {
                tokio::fs::metadata(&path)
                    .await
                    .ok()
                    .map(|metadata| metadata.is_dir())
            })
        }),
        read_file: Arc::new(|path| {
            Box::pin(async move {
                let bytes = tokio::fs::read(&path)
                    .await
                    .map_err(|error| error.to_string())?;
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            })
        }),
    }
}

/// The grep tool's options, upstream's `GrepToolOptions`.
#[derive(Clone, Default)]
pub struct GrepToolOptions {
    /// Custom operations for grep; default, local filesystem plus the
    /// native scan.
    pub operations: Option<GrepOperations>,
}

impl std::fmt::Debug for GrepToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrepToolOptions")
            .field("operations", &self.operations.is_some())
            .finish()
    }
}

/// The grep tool's system-prompt contribution, upstream's
/// `grepToolSystemPromptContribution`.
pub const GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION: SystemPromptContribution =
    SystemPromptContribution {
        snippet: "Search file contents for patterns (respects .gitignore)",
        guidelines: &[],
    };

/// The grep tool's schema, upstream's `grepSchema`.
fn grep_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Search pattern (regex or literal string)"
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: current directory)"
            },
            "glob": {
                "type": "string",
                "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"
            },
            "ignoreCase": {
                "type": "boolean",
                "description": "Case-insensitive search (default: false)"
            },
            "literal": {
                "type": "boolean",
                "description": "Treat pattern as literal string instead of regex (default: false)"
            },
            "context": {
                "type": "number",
                "description": "Number of lines to show before and after each match (default: 0)"
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of matches to return (default: 100)"
            }
        },
        "required": ["pattern"]
    })
}

fn parse_input(params: &Value) -> Result<GrepToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        pattern: String,
        path: Option<String>,
        glob: Option<String>,
        #[serde(rename = "ignoreCase")]
        ignore_case: Option<bool>,
        literal: Option<bool>,
        context: Option<f64>,
        limit: Option<f64>,
    }
    serde_json::from_value::<RawInput>(params.clone())
        .map(|raw| GrepToolInput {
            pattern: raw.pattern,
            path: raw.path,
            glob: raw.glob,
            ignore_case: raw.ignore_case,
            literal: raw.literal,
            context: raw.context,
            limit: raw.limit,
        })
        .map_err(|error| io_error(error.to_string()))
}

/// Compile the scan regex, upstream's rg argv assembly: the literal flag
/// escapes the pattern into a fixed-strings needle, and the case flag sets
/// the matcher's case-insensitivity.
///
/// # Errors
/// The regex-compile failure, upstream's `Failed to run ripgrep` shape for
/// a bad pattern.
fn compile_scan_regex(
    pattern: &str,
    literal: bool,
    ignore_case: bool,
) -> Result<regex::Regex, String> {
    let needle = if literal {
        regex::escape(pattern)
    } else {
        pattern.to_owned()
    };
    regex::RegexBuilder::new(&needle)
        .case_insensitive(ignore_case)
        .build()
        .map_err(|error| error.to_string())
}

/// The glob filter the walk applies per entry, upstream's `--glob` argv:
/// a pattern without `/` matches the basename (gitignore semantics), with
/// `/` it anchors to the search-root-relative path.
fn glob_filter_matches(glob: &str, relative: &str) -> bool {
    if glob.contains('/') {
        crate::utils::minimatch::matches(relative, glob, false)
    } else {
        let basename = relative.rsplit('/').next().unwrap_or(relative);
        crate::utils::minimatch::matches(basename, glob, false)
    }
}

/// The line-split the context reads run, upstream's
/// `content.replace(/\r\n/g, "\n").replace(/\r/g, "").split("\n")`.
fn split_lines(content: &str) -> Vec<String> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "");
    normalized.split('\n').map(<str>::to_owned).collect()
}

/// Whether the file content marks it binary, ripgrep's NUL-byte detection.
fn is_binary(content: &[u8]) -> bool {
    content.contains(&0)
}

/// The grep tool's execution body, upstream's `createGrepToolDefinition`
/// execute.
#[expect(
    clippy::too_many_lines,
    reason = "the body mirrors upstream's single execute: the scan, the per-match formatting, and the notice assembly"
)]
async fn execute_grep_tool(
    params: &Value,
    signal: Option<&AbortSignal>,
    ctx: Option<&dyn ExtensionContext>,
    ops: &GrepOperations,
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
        reason = "the validated numbers are non-negative; a fractional context floors to a whole line count"
    )]
    let context_value =
        input.context.map_or(
            0usize,
            |context| {
                if context > 0.0 { context as usize } else { 0 }
            },
        );
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a fractional limit floors to a whole match count, upstream's Math.max(1, limit)"
    )]
    let effective_limit = input
        .limit
        .map_or(DEFAULT_LIMIT, |limit| limit.max(1.0) as usize);

    let is_directory = (ops.is_directory)(search_path.clone())
        .await
        .ok_or_else(|| io_error(format!("Path not found: {search_path}")))?;

    let format_path = |file_path: &str| -> String {
        if is_directory && let Some(relative) = file_path.strip_prefix(&search_path) {
            let relative = relative.trim_start_matches('/');
            if !relative.is_empty() {
                return relative.replace(std::path::MAIN_SEPARATOR, "/");
            }
        }
        file_path
            .rsplit(std::path::MAIN_SEPARATOR)
            .next()
            .unwrap_or(file_path)
            .to_owned()
    };

    let regex = compile_scan_regex(
        &input.pattern,
        input.literal.unwrap_or(false),
        input.ignore_case.unwrap_or(false),
    )
    .map_err(|error| io_error(format!("Failed to run ripgrep: {error}")))?;

    // The scan, upstream's rg JSON stream collapsed into a direct walk:
    // matches stream per file, the limit kills the walk the way the
    // tool killed the child.
    let mut matches: Vec<Match> = Vec::new();
    let mut match_limit_reached = false;

    if is_directory {
        let mut builder = ignore::WalkBuilder::new(&search_path);
        builder.hidden(true); // rg --hidden
        builder.git_ignore(true);
        builder.parents(true);
        let walker = builder.build();
        for entry in walker {
            if matches.len() >= effective_limit {
                match_limit_reached = true;
                break;
            }
            if signal.is_some_and(AbortSignal::aborted) {
                return Err(io_error("Operation aborted"));
            }
            let Ok(entry) = entry else { continue };
            let path = entry.path().to_string_lossy().into_owned();
            if entry
                .file_type()
                .is_some_and(|file_type| file_type.is_dir())
            {
                continue;
            }
            if let Some(glob) = input.glob.as_deref() {
                let relative = path
                    .strip_prefix(&search_path)
                    .unwrap_or(&path)
                    .trim_start_matches('/');
                if !glob_filter_matches(glob, relative) {
                    continue;
                }
            }
            let Ok(bytes) = tokio::fs::read(&path).await else {
                continue;
            };
            if is_binary(&bytes) {
                continue;
            }
            let content = String::from_utf8_lossy(&bytes).into_owned();
            let lines = split_lines(&content);
            for (index, line) in lines.iter().enumerate() {
                if matches.len() >= effective_limit {
                    match_limit_reached = true;
                    break;
                }
                if regex.is_match(line) {
                    matches.push(Match {
                        file_path: path.clone(),
                        line_number: index + 1,
                        line_text: Some((*line).clone()),
                    });
                }
            }
        }
    } else {
        // Single-file search, upstream's rg path argument: no walk, no
        // gitignore scoping.
        if let Ok(bytes) = tokio::fs::read(&search_path).await
            && !is_binary(&bytes)
        {
            let content = String::from_utf8_lossy(&bytes).into_owned();
            let lines = split_lines(&content);
            for (index, line) in lines.iter().enumerate() {
                if matches.len() >= effective_limit {
                    match_limit_reached = true;
                    break;
                }
                if regex.is_match(line) {
                    matches.push(Match {
                        file_path: search_path.clone(),
                        line_number: index + 1,
                        line_text: Some((*line).clone()),
                    });
                }
            }
            if matches.len() >= effective_limit {
                match_limit_reached = match_limit_reached || matches.len() == effective_limit;
            }
        }
    }

    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }

    if matches.is_empty() {
        return Ok(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: "No matches found".to_owned(),
                text_signature: None,
            })],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            terminate: None,
        });
    }

    // Format matches after the scan finishes so custom readFile() backends
    // can be async, upstream's post-close formatting loop.
    let mut output_lines: Vec<String> = Vec::new();
    let mut lines_truncated = false;
    let file_cache = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
        String,
        Vec<String>,
    >::new()));
    let get_file_lines = {
        let ops = Arc::new(ops.clone());
        let file_cache = Arc::clone(&file_cache);
        move |file_path: String| -> BoxedFuture<'static, Vec<String>> {
            let ops: Arc<GrepOperations> = ops.clone();
            let file_cache = Arc::clone(&file_cache);
            Box::pin(async move {
                let cached = file_cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(&file_path)
                    .cloned();
                let lines: Vec<String> = match cached {
                    Some(lines) => lines,
                    None => (ops.read_file)(file_path.clone())
                        .await
                        .map_or_else(|_| Vec::new(), |content| split_lines(&content)),
                };
                file_cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(file_path, lines.clone());
                lines
            })
        }
    };

    for match_item in &matches {
        let relative_path = format_path(&match_item.file_path);
        if context_value == 0 {
            let sanitized = match_item
                .line_text
                .as_deref()
                .unwrap_or_default()
                .replace("\r\n", "\n")
                .replace('\r', "");
            let sanitized = sanitized.strip_suffix('\n').unwrap_or(&sanitized);
            let truncated = truncate_line_default(sanitized);
            if truncated.was_truncated {
                lines_truncated = true;
            }
            output_lines.push(format!(
                "{relative_path}:{}: {}",
                match_item.line_number, truncated.text
            ));
        } else {
            let lines = get_file_lines(match_item.file_path.clone()).await;
            if lines.is_empty() {
                output_lines.push(format!(
                    "{relative_path}:{}: (unable to read file)",
                    match_item.line_number
                ));
                continue;
            }
            let start = if context_value > 0 {
                match_item.line_number.saturating_sub(context_value).max(1)
            } else {
                match_item.line_number
            };
            let end = if context_value > 0 {
                std::cmp::min(lines.len(), match_item.line_number + context_value)
            } else {
                match_item.line_number
            };
            for current in start..=end {
                let line_text = lines.get(current - 1).map_or("", String::as_str);
                let sanitized = line_text.replace('\r', "");
                let is_match_line = current == match_item.line_number;
                let truncated = truncate_line_default(&sanitized);
                if truncated.was_truncated {
                    lines_truncated = true;
                }
                if is_match_line {
                    output_lines.push(format!("{relative_path}:{current}: {}", truncated.text));
                } else {
                    output_lines.push(format!("{relative_path}-{current}- {}", truncated.text));
                }
            }
        }
    }

    let raw_output = output_lines.join("\n");
    // Apply byte truncation. There is no line limit here because the match
    // limit already capped rows.
    let truncation = truncate_head(
        &raw_output,
        TruncationOptions {
            max_lines: Some(usize::MAX),
            ..TruncationOptions::default()
        },
    );
    let mut output = truncation.content.clone();
    let mut details = GrepToolDetails::default();
    // Build actionable notices for truncation and match limits.
    let mut notices: Vec<String> = Vec::new();
    if match_limit_reached {
        notices.push(format!(
            "{effective_limit} matches limit reached. Use limit={} for more, or refine pattern",
            effective_limit * 2
        ));
        details.match_limit_reached = Some(effective_limit);
    }
    if truncation.truncated {
        notices.push(format!(
            "{} limit reached",
            super::truncate::format_size(DEFAULT_MAX_BYTES)
        ));
        details.truncation = Some(truncation);
    }
    if lines_truncated {
        notices.push(format!(
            "Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
        ));
        details.lines_truncated = Some(true);
    }
    if !notices.is_empty() {
        let _ = write!(output, "\n\n[{}]", notices.join(". "));
    }

    Ok(AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: output,
            text_signature: None,
        })],
        details: details.to_wire(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    })
}

/// Build the grep tool definition, upstream's `createGrepToolDefinition`.
#[must_use]
pub fn create_grep_tool_definition(cwd: &str, options: Option<GrepToolOptions>) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let ops = Arc::new(options.operations.unwrap_or_else(default_grep_operations));
    let cwd = Arc::new(cwd.to_owned());

    ToolDefinition {
        name: "grep".to_owned(),
        label: "grep".to_owned(),
        description: format!(
            "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or {}KB (whichever is hit first). Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars.",
            DEFAULT_MAX_BYTES / 1024
        ),
        prompt_snippet: Some(GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet.to_owned()),
        prompt_guidelines: Some(
            GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION
                .guidelines
                .iter()
                .map(|guideline| (*guideline).to_owned())
                .collect(),
        ),
        parameters: grep_schema(),
        constrained_sampling: None,
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(
            move |_tool_call_id: &str, params: &Value, signal, _on_update, ctx| {
                let ops = Arc::clone(&ops);
                let cwd = Arc::clone(&cwd);
                Box::pin(async move { execute_grep_tool(params, signal, ctx, &ops, &cwd).await })
            },
        ),
    }
}

/// Build the grep tool, upstream's `createGrepTool`.
#[must_use]
pub fn create_grep_tool(cwd: &str, options: Option<GrepToolOptions>) -> AgentHarnessTool {
    let definition = create_grep_tool_definition(cwd, options);
    wrap_tool_definition::<crate::extensions::types::CwdContext>(definition, None)
}
