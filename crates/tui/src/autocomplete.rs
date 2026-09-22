//! Autocomplete providers, ported from `packages/tui/src/autocomplete.ts`
//! in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#49).
//!
//! [`CombinedAutocompleteProvider`] answers slash-command, `@`-file, and
//! path-prefix queries; its recursive fuzzy walk shells out to `fd`,
//! exactly upstream — a `None` fd path degrades the `@` search to no
//! suggestions.
//!
//! Restatements against upstream:
//!
//! - Text offsets are byte offsets into the Rust `String` rather than UTF-16
//!   code units, the editor's convention (#47); cursor columns and delimiter
//!   scans are byte offsets end to end.
//! - `AbortSignal` becomes [`tokio_util::sync::CancellationToken`] per the
//!   stack decision (map ticket "Decide the Rust stack"); the provider
//!   receives it in [`AutocompleteQueryOptions`] and checks
//!   `is_cancelled()` around its blocking work.
//! - `getSuggestions`/`applyCompletion` are synchronous trait methods: the
//!   editor runs `getSuggestions` on its autocomplete worker thread, so a
//!   provider may block on the `fd` subprocess the way upstream's async
//!   provider awaits it.
//! - `child_process.spawn` becomes [`std::process::Command`]; the abort
//!   watcher kills the child, mirroring upstream's `SIGKILL` on abort.
//! - `os.homedir()` becomes an injected [`HomeLookup`] (constructor seam,
//!   the #41 env-read pattern), defaulting to the `HOME` (unix) /
//!   `USERPROFILE` (windows) environment read.
//! - `localeCompare` restates to lexicographic byte comparison; the sorts
//!   are stable, matching JS `Array.prototype.sort`.
//! - `path.join`/`dirname`/`basename` restate to small string helpers over
//!   the same `'/'`-separated display paths; the filesystem-facing joins go
//!   through [`std::path::PathBuf`], where `.` and `./` segments resolve
//!   identically.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use regex::Regex;
use tokio_util::sync::CancellationToken;

use crate::fuzzy::fuzzy_filter;

/// Characters that end a token for path extraction, upstream
/// `PATH_DELIMITERS`.
const PATH_DELIMITERS: &[u8; 5] = b" \t\"'=";

/// The attachment-autocomplete debounce, upstream
/// `ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS`.
///
/// Symbol-triggered (`@`, `#`, or a provider's own characters) suggestions
/// wait this long after the last keystroke before the provider runs.
pub const ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS: u64 = 20;

/// The editor's always-on trigger characters, upstream
/// `DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS`.
pub const DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS: [&str; 2] = ["@", "#"];

/// Upper bound on `@` fuzzy suggestions, upstream's per-walk `maxResults`
/// and the 20-item cut.
const FUZZY_MAX_RESULTS: usize = 100;
const FUZZY_TOP_COUNT: usize = 20;

/// Normalize Windows separators to `/`, upstream `toDisplayPath`.
fn to_display_path(value: &str) -> String {
    value.replace('\\', "/")
}

/// Escape a string for literal use inside a regex character class, upstream
/// `escapeRegex`.
fn escape_regex(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '.' | '*' | '+' | '?' | '^' | '$' | '{' | '}' | '(' | ')' | '|' | '[' | ']' | '\\'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Build the regex fragment `fd` matches file paths against, upstream
/// `buildFdPathQuery`: bare segments match basenames, a `/`-containing
/// query becomes a `--full-path` pattern whose separators match either
/// separator kind.
fn build_fd_path_query(query: &str) -> String {
    let normalized = to_display_path(query);
    if !normalized.contains('/') {
        return normalized;
    }

    let has_trailing_separator = normalized.ends_with('/');
    let trimmed = normalized.trim_matches('/');
    if trimmed.is_empty() {
        return normalized;
    }

    #[expect(
        clippy::items_after_statements,
        reason = "upstream declares separatorPattern inline inside buildFdPathQuery; the port keeps the shape"
    )]
    const SEPARATOR_PATTERN: &str = "[\\\\/]";
    let segments: Vec<String> = trimmed
        .split('/')
        .filter(|segment| !segment.is_empty())
        .map(escape_regex)
        .collect();
    if segments.is_empty() {
        return normalized;
    }

    let mut pattern = segments.join(SEPARATOR_PATTERN);
    if has_trailing_separator {
        pattern += SEPARATOR_PATTERN;
    }
    pattern
}

/// The last path-delimiter byte, upstream `findLastDelimiter`.
fn find_last_delimiter(text: &str) -> Option<usize> {
    text.bytes()
        .rposition(|byte| PATH_DELIMITERS.contains(&byte))
}

/// The byte index of the last unclosed `"`, upstream
/// `findUnclosedQuoteStart`.
fn find_unclosed_quote_start(text: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut quote_start = None;
    for (index, character) in text.char_indices() {
        if character == '"' {
            in_quotes = !in_quotes;
            if in_quotes {
                quote_start = Some(index);
            }
        }
    }
    if in_quotes { quote_start } else { None }
}

/// Whether `index` begins a token — at the text start or right after a
/// delimiter, upstream `isTokenStart`.
fn is_token_start(text: &str, index: usize) -> bool {
    index == 0
        || text
            .as_bytes()
            .get(index - 1)
            .is_some_and(|byte| PATH_DELIMITERS.contains(byte))
}

/// The token from the last unclosed quote, upstream
/// `extractQuotedPrefix`. Only a token-starting quote counts: `@"…` or
/// `"…` at the start of the token.
fn extract_quoted_prefix(text: &str) -> Option<String> {
    let quote_start = find_unclosed_quote_start(text)?;

    if quote_start > 0 && text.as_bytes().get(quote_start - 1) == Some(&b'@') {
        if !is_token_start(text, quote_start - 1) {
            return None;
        }
        return Some(text[quote_start - 1..].to_string());
    }

    if !is_token_start(text, quote_start) {
        return None;
    }

    Some(text[quote_start..].to_string())
}

/// The extracted prefix's raw path and `@`/quote shape, upstream
/// `parsePathPrefix`'s return object.
///
/// The field names mirror upstream's, so the `_prefix` postfixes are the
/// wire's names.
#[expect(
    clippy::struct_field_names,
    reason = "the fields mirror upstream's parsePathPrefix shape; renaming them would detach the port from the wire"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PathPrefixParts<'a> {
    raw_prefix: &'a str,
    is_at_prefix: bool,
    is_quoted_prefix: bool,
}

fn parse_path_prefix(prefix: &str) -> PathPrefixParts<'_> {
    if let Some(raw) = prefix.strip_prefix("@\"") {
        return PathPrefixParts {
            raw_prefix: raw,
            is_at_prefix: true,
            is_quoted_prefix: true,
        };
    }
    if let Some(raw) = prefix.strip_prefix('"') {
        return PathPrefixParts {
            raw_prefix: raw,
            is_at_prefix: false,
            is_quoted_prefix: true,
        };
    }
    if let Some(raw) = prefix.strip_prefix('@') {
        return PathPrefixParts {
            raw_prefix: raw,
            is_at_prefix: true,
            is_quoted_prefix: false,
        };
    }
    PathPrefixParts {
        raw_prefix: prefix,
        is_at_prefix: false,
        is_quoted_prefix: false,
    }
}

/// The replacement text for a completion, upstream `buildCompletionValue`:
/// quote when the prefix was quoted or the path contains a space.
fn build_completion_value(path: &str, options: PathPrefixParts<'_>) -> String {
    let needs_quotes = options.is_quoted_prefix || path.contains(' ');
    let prefix = if options.is_at_prefix { "@" } else { "" };

    if !needs_quotes {
        return format!("{prefix}{path}");
    }

    format!("{prefix}\"{path}\"")
}

/// node `path.dirname` for `'/'`-separated display paths: everything before
/// the last `'/'`, `"."` when there is none.
fn dirname_str(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) => "/",
        Some(index) => &path[..index],
        None => ".",
    }
}

/// node `path.basename` for `'/'`-separated display paths: everything after
/// the last `'/'`.
fn basename_str(path: &str) -> &str {
    path.rfind('/').map_or(path, |index| &path[index + 1..])
}

/// node `path.join` for `'/'`-separated display paths: a single separator
/// between the parts, the second part kept verbatim.
fn join_display_path(base: &str, tail: &str) -> String {
    if base.is_empty() {
        return tail.to_string();
    }
    let base = base.strip_suffix('/').unwrap_or(base);
    format!("{base}/{tail}")
}

/// Escape a trigger character for literal use inside a regex character
/// class, upstream `escapeCharacterClass`.
fn escape_character_class(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '\\' | '^'
                | '$'
                | '.'
                | '*'
                | '+'
                | '?'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '|'
                | '-'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// The token-boundary trigger pattern, upstream `buildTriggerPattern`.
///
/// A trigger character right after the text start or whitespace, with
/// non-whitespace to the cursor.
///
/// # Panics
/// Panics if the escaped characters do not form a valid pattern —
/// impossible for single characters, which the editor's
/// `setAutocompleteTriggerCharacters` filter guarantees.
#[must_use]
#[expect(
    clippy::expect_used,
    reason = "the escaped single-character class forms a valid pattern; a build failure is a programmer error"
)]
pub fn build_trigger_pattern(trigger_characters: &[String]) -> Regex {
    let escaped: String = trigger_characters
        .iter()
        .map(String::as_str)
        .map(escape_character_class)
        .collect();
    Regex::new(&format!("(?:^|[\\s])[{escaped}][^\\s]*$"))
        .expect("trigger characters form a valid pattern")
}

/// The debounce-decision pattern, upstream `buildDebouncePattern`.
///
/// The `@` alternative (including an open quoted path) and the other
/// trigger characters, at a token boundary to the cursor. `@` always keeps
/// the attachment debounce even when the provider replaces the defaults.
///
/// # Panics
/// Panics only if the escaped characters cannot form a valid pattern —
/// impossible for single characters, which the editor's
/// `setAutocompleteTriggerCharacters` filter guarantees.
#[must_use]
#[expect(
    clippy::expect_used,
    reason = "the escaped single-character class forms a valid pattern; a build failure is a programmer error"
)]
pub fn build_debounce_pattern(trigger_characters: &[String]) -> Regex {
    let escaped_without_at: String = trigger_characters
        .iter()
        .filter(|character| character.as_str() != "@")
        .map(|character| escape_character_class(character))
        .collect();
    Regex::new(&format!(
        "(?:^|[ \t])(?:@(?:\"[^\"]*|[^\\s]*)|[{escaped_without_at}][^\\s]*)$"
    ))
    .expect("debounce characters form a valid pattern")
}

/// One walked path, upstream's `{ path, isDirectory }` entries.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FdEntry {
    path: String,
    is_directory: bool,
}

/// Walk the directory tree with `fd`, upstream `walkDirectoryWithFd`:
/// fast, respects .gitignore, follows symlinks, includes hidden entries
/// except `.git`. Aborting kills the child, upstream's `SIGKILL`; spawn or
/// exit failures degrade to no results. Output is bounded by
/// `--max-results`, so the stdout pipe cannot fill while the child runs.
fn walk_directory_with_fd(
    base_dir: &Path,
    fd_path: &Path,
    query: &str,
    max_results: usize,
    signal: &CancellationToken,
    max_depth: Option<u32>,
) -> Vec<FdEntry> {
    let mut args: Vec<String> = vec![
        "--base-directory".into(),
        base_dir.to_string_lossy().to_string(),
        "--max-results".into(),
        max_results.to_string(),
        "--type".into(),
        "f".into(),
        "--type".into(),
        "d".into(),
        "--follow".into(),
        "--hidden".into(),
        "--exclude".into(),
        ".git".into(),
        "--exclude".into(),
        ".git/*".into(),
        "--exclude".into(),
        ".git/**".into(),
    ];

    if let Some(max_depth) = max_depth {
        args.push("--max-depth".into());
        args.push(max_depth.to_string());
    }

    if to_display_path(query).contains('/') {
        args.push("--full-path".into());
    }

    if !query.is_empty() {
        args.push(build_fd_path_query(query));
    }

    if signal.is_cancelled() {
        return Vec::new();
    }

    let child = Command::new(fd_path)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let Ok(mut child) = child else {
        return Vec::new();
    };

    // Abort watcher, upstream's `onAbort` listener: kill the child so the
    // walk stops promptly; a natural exit reaps the same way.
    loop {
        if signal.is_cancelled() {
            let _ = child.kill();
        }
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if !signal.is_cancelled() => {
                thread::sleep(Duration::from_millis(2));
            }
            Ok(None) => {}
            Err(_) => return Vec::new(),
        }
    }

    let Ok(output) = child.wait_with_output() else {
        return Vec::new();
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    if signal.is_cancelled() || !output.status.success() || stdout.is_empty() {
        return Vec::new();
    }

    let lines = stdout.trim().lines().filter(|line| !line.is_empty());
    let mut results: Vec<FdEntry> = Vec::new();

    for line in lines {
        let display_line = to_display_path(line);
        let has_trailing_separator = display_line.ends_with('/');
        let normalized_path = display_line.strip_suffix('/').unwrap_or(&display_line);
        if normalized_path == ".git"
            || normalized_path.starts_with(".git/")
            || normalized_path.contains("/.git/")
        {
            continue;
        }

        results.push(FdEntry {
            path: display_line,
            is_directory: has_trailing_separator,
        });
    }

    results
}

/// One suggestion row, upstream `AutocompleteItem`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutocompleteItem {
    /// Value applied on selection, upstream `value`.
    pub value: String,
    /// Display label, upstream `label`.
    pub label: String,
    /// Optional description, upstream `description`.
    pub description: Option<String>,
}

impl From<AutocompleteItem> for crate::components::select_list::SelectItem {
    fn from(item: AutocompleteItem) -> Self {
        Self {
            value: item.value,
            label: item.label,
            description: item.description,
        }
    }
}

impl From<crate::components::select_list::SelectItem> for AutocompleteItem {
    fn from(item: crate::components::select_list::SelectItem) -> Self {
        Self {
            value: item.value,
            label: item.label,
            description: item.description,
        }
    }
}

/// A slash command registered with the provider, upstream `SlashCommand`.
pub struct SlashCommand {
    /// Command name without the leading `/`, upstream `name`.
    pub name: String,
    /// Description shown in the dropdown, upstream `description`.
    pub description: Option<String>,
    /// Argument hint prepended to the description, upstream
    /// `argumentHint`.
    pub argument_hint: Option<String>,
    /// Argument completion hook, upstream `getArgumentCompletions`:
    /// `None` when the command offers none.
    pub get_argument_completions: Option<SlashArgumentCompletionsFn>,
}

impl std::fmt::Debug for SlashCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlashCommand")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("argument_hint", &self.argument_hint)
            .finish_non_exhaustive()
    }
}

/// The argument-completion hook, upstream
/// `SlashCommand.getArgumentCompletions`: `None` when no argument
/// completion is available.
pub type SlashArgumentCompletionsFn =
    Box<dyn Fn(&str) -> Option<Vec<AutocompleteItem>> + Send + Sync>;

/// A provider entry, upstream `(SlashCommand | AutocompleteItem)[]`: a
/// slash command or a plain item (whose `value` is the command name).
#[derive(Debug)]
pub enum CommandEntry {
    /// Upstream's `SlashCommand` variant.
    Slash(SlashCommand),
    /// Upstream's plain `AutocompleteItem` entry.
    Item(AutocompleteItem),
}

impl CommandEntry {
    /// The command name matched against the typed text, upstream's
    /// `"name" in cmd ? cmd.name : cmd.value` lookup.
    fn name(&self) -> &str {
        match self {
            Self::Slash(command) => &command.name,
            Self::Item(item) => &item.value,
        }
    }

    fn description(&self) -> Option<&str> {
        match self {
            Self::Slash(command) => command.description.as_deref(),
            Self::Item(item) => item.description.as_deref(),
        }
    }

    fn argument_hint(&self) -> Option<&str> {
        match self {
            Self::Slash(command) => command.argument_hint.as_deref(),
            Self::Item(_) => None,
        }
    }
}

/// One mapped command suggestion, upstream's inline `{name, label,
/// description}` shape inside `getSuggestions`.
struct CommandSuggestion {
    name: String,
    label: String,
    description: Option<String>,
}

/// A suggestion set, upstream `AutocompleteSuggestions`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutocompleteSuggestions {
    /// The suggestion rows, upstream `items`.
    pub items: Vec<AutocompleteItem>,
    /// What the suggestions match against (e.g. `/` or `src/`), upstream
    /// `prefix`.
    pub prefix: String,
}

/// Per-query options, upstream `getSuggestions`' `options`.
#[derive(Debug)]
pub struct AutocompleteQueryOptions {
    /// Cancellation for the in-flight query, upstream `signal`.
    pub signal: CancellationToken,
    /// Force a file-completion query even without a natural trigger,
    /// upstream `force`.
    pub force: bool,
}

/// The replacement produced by [`AutocompleteProvider::apply_completion`],
/// upstream `applyCompletion`'s return shape.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutocompleteApplyResult {
    /// The new text lines, upstream `lines`.
    pub lines: Vec<String>,
    /// The new cursor line, upstream `cursorLine`.
    pub cursor_line: usize,
    /// The new cursor column (byte offset), upstream `cursorCol`.
    pub cursor_col: usize,
}

/// The autocomplete source, upstream `AutocompleteProvider`.
pub trait AutocompleteProvider: Send + Sync {
    /// Characters that naturally trigger this provider at token boundaries,
    /// upstream `triggerCharacters`.
    fn trigger_characters(&self) -> Vec<String> {
        Vec::new()
    }

    /// Suggestions for the current text/cursor position, upstream
    /// `getSuggestions`. `None` when nothing is available.
    fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        options: &AutocompleteQueryOptions,
    ) -> Option<AutocompleteSuggestions>;

    /// Apply the selected item, upstream `applyCompletion`.
    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> AutocompleteApplyResult;

    /// Whether an explicit Tab completion should trigger file completion,
    /// upstream `shouldTriggerFileCompletion?`. Default: yes.
    fn should_trigger_file_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> bool {
        let _ = (lines, cursor_line, cursor_col);
        true
    }
}

/// The home-directory lookup, upstream `os.homedir()`; injected so tests
/// can control expansion (the #41 env-read seam pattern).
pub type HomeLookup = Box<dyn Fn() -> Option<String> + Send + Sync>;

/// The default home lookup: the `HOME` environment variable on unix,
/// `USERPROFILE` on windows, `None` when unset.
fn default_home_lookup() -> Option<String> {
    #[cfg(unix)]
    {
        std::env::var("HOME").ok()
    }
    #[cfg(not(unix))]
    {
        std::env::var("USERPROFILE").ok()
    }
}

/// The combined provider, upstream `CombinedAutocompleteProvider`: slash
/// commands and file paths.
pub struct CombinedAutocompleteProvider {
    commands: Vec<CommandEntry>,
    base_path: String,
    fd_path: Option<PathBuf>,
    home_lookup: HomeLookup,
}

impl std::fmt::Debug for CombinedAutocompleteProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedAutocompleteProvider")
            .field("commands", &self.commands)
            .field("base_path", &self.base_path)
            .field("fd_path", &self.fd_path)
            .finish_non_exhaustive()
    }
}

/// A scoped fuzzy query, upstream `resolveScopedFuzzyQuery`'s return
/// shape: a narrowed search root, the remaining query, and the display
/// base the results are prefixed back with.
struct ScopedQuery {
    base_dir: String,
    query: String,
    display_base: String,
}

impl CombinedAutocompleteProvider {
    /// Upstream's `new CombinedAutocompleteProvider(commands, basePath,
    /// fdPath)`.
    #[must_use]
    pub fn new(
        commands: Vec<CommandEntry>,
        base_path: impl Into<String>,
        fd_path: Option<PathBuf>,
    ) -> Self {
        Self {
            commands,
            base_path: base_path.into(),
            fd_path,
            home_lookup: Box::new(default_home_lookup),
        }
    }

    /// The constructor with the home lookup seam injected, for tests that
    /// exercise `~` expansion.
    #[must_use]
    pub fn with_home_lookup(
        commands: Vec<CommandEntry>,
        base_path: impl Into<String>,
        fd_path: Option<PathBuf>,
        home_lookup: HomeLookup,
    ) -> Self {
        Self {
            commands,
            base_path: base_path.into(),
            fd_path,
            home_lookup,
        }
    }

    /// The suggestion query, upstream `getSuggestions`: `@`-prefixed fuzzy
    /// file search first, then slash commands (unless forced), then
    /// path-prefix completion.
    fn get_suggestions_impl(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        options: &AutocompleteQueryOptions,
    ) -> Option<AutocompleteSuggestions> {
        let current_line = lines.get(cursor_line).cloned().unwrap_or_default();
        let cursor_col = cursor_col.min(current_line.len());
        let text_before_cursor = &current_line[..cursor_col];

        let at_prefix = self.extract_at_prefix(text_before_cursor);
        if let Some(at_prefix) = at_prefix {
            let parts = parse_path_prefix(&at_prefix);
            let suggestions = self.get_fuzzy_file_suggestions(
                parts.raw_prefix,
                parts.is_quoted_prefix,
                &options.signal,
            );
            if suggestions.is_empty() {
                return None;
            }

            return Some(AutocompleteSuggestions {
                items: suggestions,
                prefix: at_prefix,
            });
        }

        if !options.force && text_before_cursor.starts_with('/') {
            let Some(space_index) = text_before_cursor.find(' ') else {
                let prefix = &text_before_cursor[1..];
                let command_items: Vec<CommandSuggestion> = self
                    .commands
                    .iter()
                    .map(|command| {
                        let name = command.name();
                        let hint = command.argument_hint().filter(|hint| !hint.is_empty());
                        let description = command.description().unwrap_or("");
                        let full_description = match hint {
                            Some(hint) if !description.is_empty() => {
                                format!("{hint} — {description}")
                            }
                            Some(hint) => hint.to_string(),
                            None => description.to_string(),
                        };
                        CommandSuggestion {
                            name: name.to_string(),
                            label: name.to_string(),
                            description: if full_description.is_empty() {
                                None
                            } else {
                                Some(full_description)
                            },
                        }
                    })
                    .collect();

                let filtered = fuzzy_filter(&command_items, prefix, |item| item.name.clone());
                if filtered.is_empty() {
                    return None;
                }

                return Some(AutocompleteSuggestions {
                    items: filtered
                        .iter()
                        .map(|item| AutocompleteItem {
                            value: item.name.clone(),
                            label: item.label.clone(),
                            description: item.description.clone(),
                        })
                        .collect(),
                    prefix: text_before_cursor.to_string(),
                });
            };

            let command_name = &text_before_cursor[1..space_index];
            let argument_text = &text_before_cursor[space_index + 1..];

            let command = self
                .commands
                .iter()
                .find(|command| command.name() == command_name);
            let get_argument_completions = command.and_then(|command| match command {
                CommandEntry::Slash(command) => command.get_argument_completions.as_deref(),
                CommandEntry::Item(_) => None,
            });
            let get_argument_completions = get_argument_completions?;
            let argument_suggestions = get_argument_completions(argument_text)?;
            if argument_suggestions.is_empty() {
                return None;
            }

            return Some(AutocompleteSuggestions {
                items: argument_suggestions,
                prefix: argument_text.to_string(),
            });
        }

        let path_match = self.extract_path_prefix(text_before_cursor, options.force)?;
        let suggestions = self.get_file_suggestions(&path_match);
        if suggestions.is_empty() {
            return None;
        }

        Some(AutocompleteSuggestions {
            items: suggestions,
            prefix: path_match,
        })
    }

    /// The completion application, upstream `applyCompletion`: command
    /// names get a trailing space, `@` attachments keep the cursor inside
    /// directories, quoted prefixes swallow the trailing quote.
    fn apply_completion_impl(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> AutocompleteApplyResult {
        let current_line = lines.get(cursor_line).cloned().unwrap_or_default();
        let before_prefix = current_line[..cursor_col.saturating_sub(prefix.len())].to_string();
        let after_cursor = current_line[cursor_col.min(current_line.len())..].to_string();
        let is_quoted_prefix = prefix.starts_with('"') || prefix.starts_with("@\"");
        let has_leading_quote_after_cursor = after_cursor.starts_with('"');
        let has_trailing_quote_in_item = item.value.ends_with('"');
        let adjusted_after_cursor =
            if is_quoted_prefix && has_trailing_quote_in_item && has_leading_quote_after_cursor {
                after_cursor[1..].to_string()
            } else {
                after_cursor
            };

        // Check if we're completing a slash command (prefix starts with
        // "/" but NOT a file path): at the start of the line, no path
        // separators after the first /
        let is_slash_command = prefix.starts_with('/')
            && before_prefix.trim().is_empty()
            && !prefix[1..].contains('/');
        if is_slash_command {
            // This is a command name completion
            let new_line = format!("{before_prefix}/{} {adjusted_after_cursor}", item.value);
            let mut new_lines = lines.to_vec();
            new_lines[cursor_line] = new_line;

            // +2 for "/" and space
            return AutocompleteApplyResult {
                lines: new_lines,
                cursor_line,
                cursor_col: before_prefix.len() + item.value.len() + 2,
            };
        }

        // Check if we're completing a file attachment (prefix starts with
        // "@"): no space after directories so the user can keep completing
        if prefix.starts_with('@') {
            let is_directory = item.label.ends_with('/');
            let suffix = if is_directory { "" } else { " " };
            let new_line = format!(
                "{before_prefix}{}{suffix}{adjusted_after_cursor}",
                item.value
            );
            let mut new_lines = lines.to_vec();
            new_lines[cursor_line] = new_line;

            let has_trailing_quote = item.value.ends_with('"');
            let cursor_offset = if is_directory && has_trailing_quote {
                item.value.len().saturating_sub(1)
            } else {
                item.value.len()
            };

            return AutocompleteApplyResult {
                lines: new_lines,
                cursor_line,
                cursor_col: before_prefix.len() + cursor_offset + suffix.len(),
            };
        }

        // Check if we're in a slash command context ("/command " before the
        // cursor) — likely a command argument completion. The file-path
        // fall-through below is the same splice, upstream included.
        self.apply_completion_splice(
            lines,
            cursor_line,
            &before_prefix,
            &adjusted_after_cursor,
            item,
        )
    }

    /// The shared `beforePrefix + item.value + adjustedAfterCursor` splice
    /// for argument and file-path completions, upstream's duplicated tail
    /// branches.
    #[expect(
        clippy::unused_self,
        reason = "upstream's duplicated applyCompletion tail is a provider method; the port keeps the shape"
    )]
    fn apply_completion_splice(
        &self,
        lines: &[String],
        cursor_line: usize,
        before_prefix: &str,
        adjusted_after_cursor: &str,
        item: &AutocompleteItem,
    ) -> AutocompleteApplyResult {
        let mut new_lines = lines.to_vec();
        new_lines[cursor_line] = format!("{before_prefix}{}{adjusted_after_cursor}", item.value);

        let is_directory = item.label.ends_with('/');
        let has_trailing_quote = item.value.ends_with('"');
        let cursor_offset = if is_directory && has_trailing_quote {
            item.value.len().saturating_sub(1)
        } else {
            item.value.len()
        };

        AutocompleteApplyResult {
            lines: new_lines,
            cursor_line,
            cursor_col: before_prefix.len() + cursor_offset,
        }
    }

    /// The `@` prefix for fuzzy file suggestions, upstream
    /// `extractAtPrefix`: an unclosed `@"` token wins, else a token-started
    /// `@`.
    #[expect(
        clippy::unused_self,
        reason = "upstream's extractAtPrefix is a provider method; the port keeps the shape"
    )]
    fn extract_at_prefix(&self, text: &str) -> Option<String> {
        let quoted_prefix = extract_quoted_prefix(text);
        if let Some(quoted_prefix) = quoted_prefix
            && quoted_prefix.starts_with("@\"")
        {
            return Some(quoted_prefix);
        }

        let last_delimiter_index = find_last_delimiter(text);
        let token_start = last_delimiter_index.map_or(0, |index| index + 1);

        if text.as_bytes().get(token_start) == Some(&b'@') {
            return Some(text[token_start..].to_string());
        }

        None
    }

    /// A path-like prefix from the text before cursor, upstream
    /// `extractPathPrefix`: quoted prefixes always extract; forced
    /// extraction (Tab) always returns the token; natural triggers need a
    /// path shape.
    #[expect(
        clippy::unused_self,
        reason = "upstream's extractPathPrefix is a provider method; the port keeps the shape"
    )]
    fn extract_path_prefix(&self, text: &str, force_extract: bool) -> Option<String> {
        let quoted_prefix = extract_quoted_prefix(text);
        if let Some(quoted_prefix) = quoted_prefix {
            return Some(quoted_prefix);
        }

        let last_delimiter_index = find_last_delimiter(text);
        let path_prefix = last_delimiter_index.map_or(text, |index| &text[index + 1..]);

        // For forced extraction (Tab key), always return something
        if force_extract {
            return Some(path_prefix.to_string());
        }

        // For natural triggers, return if it looks like a path: contains
        // `/`, or starts with `.` or `~/`
        if path_prefix.contains('/')
            || path_prefix.starts_with('.')
            || path_prefix.starts_with("~/")
        {
            return Some(path_prefix.to_string());
        }

        // Return an empty prefix only after a space (not for completely
        // empty text) — empty text waits for forced Tab completion
        if path_prefix.is_empty() && text.ends_with(' ') {
            return Some(path_prefix.to_string());
        }

        None
    }

    /// Expand the home directory (`~/`) to the actual home path, upstream
    /// `expandHomePath`. The trailing slash is preserved.
    fn expand_home_path(&self, path: &str) -> String {
        if let Some(rest) = path.strip_prefix("~/") {
            let Some(home) = (self.home_lookup)() else {
                return path.to_string();
            };
            let expanded = if home.ends_with('/') || rest.is_empty() {
                format!("{home}{rest}")
            } else {
                format!("{home}/{rest}")
            };
            if path.ends_with('/') && !expanded.ends_with('/') {
                return format!("{expanded}/");
            }
            return expanded;
        } else if path == "~" {
            return (self.home_lookup)().unwrap_or_else(|| path.to_string());
        }
        path.to_string()
    }

    /// Split a scoped fuzzy query at its last `/`, upstream
    /// `resolveScopedFuzzyQuery`: the prefix becomes the search root
    /// (relative to the base path, or absolute / home-expanded), the
    /// suffix the narrowed query.
    fn resolve_scoped_fuzzy_query(&self, raw_query: &str) -> Option<ScopedQuery> {
        let normalized_query = to_display_path(raw_query);
        let slash_index = normalized_query.rfind('/')?;
        let display_base = normalized_query[..=slash_index].to_string();
        let query = normalized_query[slash_index + 1..].to_string();

        let base_dir = if display_base.starts_with("~/") {
            self.expand_home_path(&display_base)
        } else if display_base.starts_with('/') {
            display_base.clone()
        } else {
            join_display_path(&self.base_path, &display_base)
        };

        if !std::fs::metadata(&base_dir).is_ok_and(|metadata| metadata.is_dir()) {
            return None;
        }

        Some(ScopedQuery {
            base_dir,
            query,
            display_base,
        })
    }

    /// Prefix a scoped walk result back into display form, upstream
    /// `scopedPathForDisplay`.
    fn scoped_path_for_display(display_base: &str, relative_path: &str) -> String {
        let normalized_relative_path = to_display_path(relative_path);
        if display_base == "/" {
            return format!("/{normalized_relative_path}");
        }
        format!(
            "{}{}",
            to_display_path(display_base),
            normalized_relative_path
        )
    }

    /// File/directory suggestions for a given path prefix, upstream
    /// `getFileSuggestions`: a synchronous readdir over the prefix's
    /// directory, directories first.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's getFileSuggestions branch for branch; splitting it would detach the port from the upstream method it mirrors"
    )]
    fn get_file_suggestions(&self, prefix: &str) -> Vec<AutocompleteItem> {
        let parts = parse_path_prefix(prefix);
        let mut expanded_prefix = parts.raw_prefix.to_string();

        // Handle home directory expansion
        if expanded_prefix.starts_with('~') {
            expanded_prefix = self.expand_home_path(&expanded_prefix);
        }

        let is_root_prefix = parts.raw_prefix.is_empty()
            || parts.raw_prefix == "./"
            || parts.raw_prefix == "../"
            || parts.raw_prefix == "~"
            || parts.raw_prefix == "~/"
            || parts.raw_prefix == "/"
            || (parts.is_at_prefix && parts.raw_prefix.is_empty());

        let (search_dir, search_prefix) = if is_root_prefix {
            // Complete from the specified position
            if parts.raw_prefix.starts_with('~') || expanded_prefix.starts_with('/') {
                (PathBuf::from(&expanded_prefix), String::new())
            } else {
                (
                    Path::new(&self.base_path).join(&expanded_prefix),
                    String::new(),
                )
            }
        } else if parts.raw_prefix.ends_with('/') {
            // If prefix ends with /, show the contents of that directory
            if parts.raw_prefix.starts_with('~') || expanded_prefix.starts_with('/') {
                (PathBuf::from(&expanded_prefix), String::new())
            } else {
                (
                    Path::new(&self.base_path).join(&expanded_prefix),
                    String::new(),
                )
            }
        } else {
            // Split into directory and file prefix
            let dir = dirname_str(&expanded_prefix);
            let file = basename_str(&expanded_prefix);
            let search_dir =
                if parts.raw_prefix.starts_with('~') || expanded_prefix.starts_with('/') {
                    PathBuf::from(dir)
                } else {
                    Path::new(&self.base_path).join(dir)
                };
            (search_dir, file.to_string())
        };

        let Ok(entries) = std::fs::read_dir(&search_dir) else {
            // Directory doesn't exist or not accessible
            return Vec::new();
        };
        let lowered_search_prefix = search_prefix.to_lowercase();
        let mut suggestions: Vec<AutocompleteItem> = Vec::new();

        for entry in entries {
            let Ok(entry) = entry else { continue };
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.to_lowercase().starts_with(&lowered_search_prefix) {
                continue;
            }

            // Check if entry is a directory (or a symlink pointing to a
            // directory); broken symlinks degrade to files
            let Ok(entry_type) = entry.file_type() else {
                continue;
            };
            let mut is_directory = entry_type.is_dir();
            if !is_directory && entry_type.is_symlink() {
                is_directory =
                    std::fs::metadata(entry.path()).is_ok_and(|metadata| metadata.is_dir());
            }

            let display_prefix = parts.raw_prefix;
            #[expect(
                clippy::option_if_let_else,
                reason = "mirrors upstream's nested strip_prefix branch shape; flattening to map_or_else would invert the upstream if/else order"
            )]
            let relative_path: String = if display_prefix.ends_with('/') {
                // If prefix ends with /, append entry to the prefix
                format!("{display_prefix}{name}")
            } else if display_prefix.contains('/') || display_prefix.contains('\\') {
                if let Some(home_relative) = display_prefix.strip_prefix("~/") {
                    // Preserve ~/ format for home directory paths
                    let dir = dirname_str(home_relative);
                    if dir == "." {
                        format!("~/{name}")
                    } else {
                        format!("~/{dir}/{name}")
                    }
                } else if display_prefix.starts_with('/') {
                    // Absolute path - construct properly
                    let dir = dirname_str(display_prefix);
                    if dir == "/" {
                        format!("/{name}")
                    } else {
                        format!("{dir}/{name}")
                    }
                } else {
                    let dir = dirname_str(display_prefix);
                    let mut relative = if dir == "." {
                        name.clone()
                    } else {
                        format!("{dir}/{name}")
                    };
                    // path.join normalizes away ./ prefix, preserve it
                    if display_prefix.starts_with("./") && !relative.starts_with("./") {
                        relative = format!("./{relative}");
                    }
                    relative
                }
            } else {
                // For standalone entries, preserve ~/ if the original
                // prefix was ~/
                if display_prefix.starts_with('~') {
                    format!("~/{name}")
                } else {
                    name.clone()
                }
            };

            let relative_path = to_display_path(&relative_path);
            let path_value = if is_directory {
                format!("{relative_path}/")
            } else {
                relative_path
            };
            let value = build_completion_value(&path_value, parts);

            suggestions.push(AutocompleteItem {
                value,
                label: if is_directory {
                    format!("{name}/")
                } else {
                    name
                },
                description: None,
            });
        }

        // Sort directories first, then alphabetically
        suggestions.sort_by(
            |a, b| match (a.value.ends_with('/'), b.value.ends_with('/')) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.label.cmp(&b.label),
            },
        );

        suggestions
    }

    /// Score an entry against the query (higher = better), upstream
    /// `scoreEntry`: exact filename, then prefix, then substring, then
    /// full-path substring; directories get a bonus to appear first.
    fn score_entry(file_path: &str, query: &str, is_directory: bool) -> i32 {
        let file_name = basename_str(file_path);
        let lower_file_name = file_name.to_lowercase();
        let lower_query = query.to_lowercase();

        let mut score: i32 = 0;

        // Exact filename match (highest)
        if lower_file_name == lower_query {
            score = 100;
        }
        // Filename starts with query
        else if lower_file_name.starts_with(&lower_query) {
            score = 80;
        }
        // Substring match in filename
        else if lower_file_name.contains(&lower_query) {
            score = 50;
        }
        // Substring match in full path
        else if file_path.to_lowercase().contains(&lower_query) {
            score = 30;
        }

        // Directories get a bonus to appear first
        if is_directory && score > 0 {
            score += 10;
        }

        score
    }

    /// The direct-children walk for the base directory, upstream
    /// `getBaseDirSuggestions`: bounded to depth 1.
    fn get_base_dir_suggestions(
        &self,
        base_dir: &Path,
        query: &str,
        signal: &CancellationToken,
    ) -> Vec<FdEntry> {
        let Some(fd_path) = &self.fd_path else {
            return Vec::new();
        };
        if signal.is_cancelled() {
            return Vec::new();
        }

        walk_directory_with_fd(base_dir, fd_path, query, FUZZY_MAX_RESULTS, signal, Some(1))
    }

    /// Fuzzy file search using fd, upstream `getFuzzyFileSuggestions`:
    /// depth-1 direct children first, then the recursive walk, deduped,
    /// scored, and cut to the top 20.
    fn get_fuzzy_file_suggestions(
        &self,
        query: &str,
        is_quoted_prefix: bool,
        signal: &CancellationToken,
    ) -> Vec<AutocompleteItem> {
        let Some(fd_path) = &self.fd_path else {
            return Vec::new();
        };
        if signal.is_cancelled() {
            return Vec::new();
        }

        let scoped_query = self.resolve_scoped_fuzzy_query(query);
        let fd_base_dir = scoped_query.as_ref().map_or_else(
            || PathBuf::from(&self.base_path),
            |scoped| PathBuf::from(&scoped.base_dir),
        );
        let fd_query = scoped_query
            .as_ref()
            .map_or(query, |scoped| scoped.query.as_str());
        let base_dir_entries = self.get_base_dir_suggestions(&fd_base_dir, fd_query, signal);
        let recursive_entries = walk_directory_with_fd(
            &fd_base_dir,
            fd_path,
            fd_query,
            FUZZY_MAX_RESULTS,
            signal,
            None,
        );
        let mut seen_paths: HashSet<String> = base_dir_entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect();
        let mut entries = base_dir_entries;
        for entry in recursive_entries {
            if seen_paths.contains(&entry.path) {
                continue;
            }
            seen_paths.insert(entry.path.clone());
            entries.push(entry);
        }
        if signal.is_cancelled() {
            return Vec::new();
        }

        let mut scored_entries: Vec<(FdEntry, i32)> = entries
            .into_iter()
            .map(|entry| {
                let score = if fd_query.is_empty() {
                    1
                } else {
                    Self::score_entry(&entry.path, fd_query, entry.is_directory)
                };
                (entry, score)
            })
            .filter(|(_entry, score)| *score > 0)
            .collect();

        scored_entries.sort_by(|(a_entry, a_score), (b_entry, b_score)| {
            let a_path = &a_entry.path;
            let b_path = &b_entry.path;
            b_score
                .cmp(a_score)
                .then_with(|| path_depth(a_path).cmp(&path_depth(b_path)))
                .then_with(|| a_path.len().cmp(&b_path.len()))
                .then_with(|| a_path.cmp(b_path))
        });
        let top_entries = scored_entries.into_iter().take(FUZZY_TOP_COUNT);

        let mut suggestions: Vec<AutocompleteItem> = Vec::new();
        for (entry, _score) in top_entries {
            let path_without_slash = if entry.is_directory {
                entry.path.strip_suffix('/').unwrap_or(&entry.path)
            } else {
                &entry.path
            };
            let display_path = scoped_query.as_ref().map_or_else(
                || path_without_slash.to_string(),
                |scoped| Self::scoped_path_for_display(&scoped.display_base, path_without_slash),
            );
            let entry_name = basename_str(path_without_slash);
            let completion_path = if entry.is_directory {
                format!("{display_path}/")
            } else {
                display_path.clone()
            };
            let value = build_completion_value(
                &completion_path,
                PathPrefixParts {
                    raw_prefix: "",
                    is_at_prefix: true,
                    is_quoted_prefix,
                },
            );

            suggestions.push(AutocompleteItem {
                value,
                label: if entry.is_directory {
                    format!("{entry_name}/")
                } else {
                    entry_name.to_string()
                },
                description: Some(display_path),
            });
        }

        suggestions
    }

    /// Whether Tab should trigger file completion for the text before the
    /// cursor, upstream `shouldTriggerFileCompletion`: not while a bare
    /// slash command is being typed.
    #[expect(
        clippy::unused_self,
        reason = "upstream's shouldTriggerFileCompletion is a provider method; the port keeps the shape"
    )]
    fn should_trigger_file_completion_impl(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> bool {
        let current_line = lines.get(cursor_line).cloned().unwrap_or_default();
        let text_before_cursor = &current_line[..cursor_col.min(current_line.len())];

        // Don't trigger if we're typing a slash command at the start of the
        // line
        if text_before_cursor.trim().starts_with('/') && !text_before_cursor.trim().contains(' ') {
            return false;
        }

        true
    }
}

/// Path depth for the fuzzy ranking, upstream's
/// `toDisplayPath(path).split("/").filter(Boolean).length`.
fn path_depth(path: &str) -> usize {
    to_display_path(path)
        .split('/')
        .filter(|part| !part.is_empty())
        .count()
}

impl AutocompleteProvider for CombinedAutocompleteProvider {
    fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        options: &AutocompleteQueryOptions,
    ) -> Option<AutocompleteSuggestions> {
        self.get_suggestions_impl(lines, cursor_line, cursor_col, options)
    }

    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> AutocompleteApplyResult {
        self.apply_completion_impl(lines, cursor_line, cursor_col, item, prefix)
    }

    fn should_trigger_file_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> bool {
        self.should_trigger_file_completion_impl(lines, cursor_line, cursor_col)
    }
}
