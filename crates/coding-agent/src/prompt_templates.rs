//! The prompt-template loader, upstream's `src/core/prompt-templates.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! This is the coding-agent's own surface — a sibling of pi-agent-core's
//! harness loader (#95) that carries provenance, argument hints, the
//! `${N:-default}` placeholder defaults, and bash-style argument slicing
//! in one single-pass substitution, upstream's single global regex with a
//! replacement callback (no rescan of inserted text).

use std::path::Path;

use crate::source_info::{
    SourceInfo, SourceScope, SyntheticSourceOptions, create_synthetic_source_info,
};
use crate::utils::frontmatter::{frontmatter_string, parse_frontmatter};
use crate::utils::paths::{
    PathInputOptions, basename_posix, dirname_posix, is_under_path, resolve_path, resolve_path_with,
};

/// A prompt template loaded from a markdown file, upstream's
/// `PromptTemplate`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptTemplate {
    /// The template name — the file's base name without `.md`.
    pub name: String,
    /// The frontmatter description, or the first non-blank body line
    /// clipped at sixty characters with an ellipsis.
    pub description: String,
    /// The frontmatter `argument-hint` when one is set.
    pub argument_hint: Option<String>,
    /// The template body after the frontmatter.
    pub content: String,
    /// Where the template came from.
    pub source_info: SourceInfo,
    /// The absolute path of the template file.
    pub file_path: String,
}

/// Parse command arguments respecting quoted strings, upstream's
/// `parseCommandArgs`.
///
/// Quotes toggle and never terminate early and carry no escape; unquoted
/// whitespace (any `\s` run — spaces, tabs, newlines) splits; an
/// unterminated quote keeps its content.
#[must_use]
pub fn parse_command_args(args_string: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quote: Option<char> = None;

    for character in args_string.chars() {
        if let Some(quote) = in_quote {
            if character == quote {
                in_quote = None;
            } else {
                current.push(character);
            }
        } else if character == '"' || character == '\'' {
            in_quote = Some(character);
        } else if character.is_whitespace() {
            // Upstream splits on JavaScript's `\s`; the port splits on
            // Unicode's White_Space property, identical on the separators
            // templates carry.
            if !current.is_empty() {
                args.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }

    if !current.is_empty() {
        args.push(current);
    }

    args
}

/// Substitute argument placeholders in template content, upstream's
/// `substituteArgs`.
///
/// One scan, upstream's single global regex with a replacement callback:
/// `${N:-default}` / `${@:-default}` / `${ARGUMENTS:-default}` give
/// positional and all-argument defaults (an empty or missing argument
/// falls back to the default), `${@:N}` and `${@:N:L}` slice the argument
/// list bash-style (N is 1-based, 0 clamps to the first), and `$N`/`$@`/
/// `$ARGUMENTS` substitute directly (out-of-range positions read empty).
/// Inserted text is literal — argument and default values carrying
/// placeholder patterns are not rescanned, upstream's callback return
/// semantics.
#[must_use]
pub fn substitute_args(content: &str, args: &[String]) -> String {
    let all_args = args.join(" ");
    let mut out = String::with_capacity(content.len());
    let rest = content;
    let mut cursor = 0;

    while let Some(at) = rest[cursor..].find('$') {
        let at = cursor + at;
        out.push_str(&rest[cursor..at]);
        let after_dollar = &rest[at + 1..];

        let replacement: Option<String> = match after_dollar.strip_prefix('{') {
            Some(inside) => match parse_braced(inside) {
                Some(BracedForm::Default { target, default }) => {
                    let value = match target {
                        DefaultTarget::All => {
                            // An empty all-args string falls back to the
                            // default, upstream's `value ? value :
                            // defaultValue` on the joined arguments.
                            if all_args.is_empty() {
                                None
                            } else {
                                Some(all_args.clone())
                            }
                        }
                        // A `0` target (upstream's `args[-1]`) reads
                        // missing, the default falling back the same way.
                        DefaultTarget::Positional(number) => number
                            .checked_sub(1)
                            .and_then(|index| args.get(index))
                            // An empty argument falls back to the
                            // default, upstream's `value ? value :
                            // defaultValue`.
                            .filter(|value| !value.is_empty())
                            .cloned(),
                    };
                    Some(value.unwrap_or_else(|| default.to_string()))
                }
                Some(BracedForm::Slice { start, length }) => {
                    // 1-based, clamped to the first argument at zero,
                    // upstream's `if (start < 0) start = 0`.
                    let start = start.saturating_sub(1).min(args.len());
                    let end = length.map_or(args.len(), |length| {
                        start.saturating_add(length).min(args.len())
                    });
                    Some(args[start..end].join(" "))
                }
                None => None,
            },
            None if after_dollar.starts_with("ARGUMENTS") => Some(all_args.clone()),
            None if after_dollar.starts_with('@') => Some(all_args.clone()),
            None => {
                let digits_end = after_dollar
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(after_dollar.len());
                if digits_end == 0 {
                    None
                } else {
                    let index = after_dollar[..digits_end]
                        .parse::<usize>()
                        .unwrap_or(usize::MAX)
                        .checked_sub(1);
                    Some(
                        index
                            .and_then(|index| args.get(index))
                            .cloned()
                            .unwrap_or_default(),
                    )
                }
            }
        };

        if let Some(value) = replacement {
            out.push_str(&value);
            cursor = at + 1 + consumed_len(after_dollar);
        } else {
            out.push('$');
            cursor = at + 1;
        }
    }

    out.push_str(&rest[cursor..]);
    out
}

/// The `${...}` forms one substitution can carry.
enum BracedForm<'a> {
    /// `${N:-default}` / `${@:-default}` / `${ARGUMENTS:-default}`.
    Default {
        target: DefaultTarget,
        default: &'a str,
    },
    /// `${@:N}` / `${@:N:L}`.
    Slice { start: usize, length: Option<usize> },
}

/// The `${N:-default}` target, upstream's `(\d+|ARGUMENTS|@)`.
enum DefaultTarget {
    /// `@` or `ARGUMENTS` — every argument joined.
    All,
    /// A 1-based positional.
    Positional(usize),
}

/// Parse the text after `${` into one of the substitution forms, `None`
/// when the shape matches neither (the literal stays, upstream's failed
/// regex alternative).
fn parse_braced(inside: &str) -> Option<BracedForm<'_>> {
    if let Some((target_part, default)) = inside.split_once(":-") {
        let default = default.split('}').next()?;
        // `[^}]*` cannot carry a closing brace; a missing one fails the
        // alternative the same way.
        let closing = inside.contains('}');
        if !closing {
            return None;
        }
        let target = if target_part == "@" || target_part == "ARGUMENTS" {
            DefaultTarget::All
        } else if target_part.chars().all(|c| c.is_ascii_digit()) && !target_part.is_empty() {
            // Upstream's `parseInt` saturates past `usize`; an
            // out-of-range position reads missing either way.
            DefaultTarget::Positional(target_part.parse::<usize>().unwrap_or(usize::MAX))
        } else {
            return None;
        };
        return Some(BracedForm::Default { target, default });
    }

    let rest = inside.strip_prefix("@:")?;
    let close_at = rest.find('}')?;
    let body = &rest[..close_at];
    let (start_part, length_part) = if let Some((start, length)) = body.split_once(':') {
        (start, Some(length))
    } else {
        (body, None)
    };
    let start = if start_part.chars().all(|c| c.is_ascii_digit()) && !start_part.is_empty() {
        start_part.parse::<usize>().unwrap_or(usize::MAX)
    } else {
        return None;
    };
    let length = match length_part {
        Some(length) => {
            if length.chars().all(|c| c.is_ascii_digit()) && !length.is_empty() {
                Some(length.parse::<usize>().unwrap_or(usize::MAX))
            } else {
                return None;
            }
        }
        None => None,
    };
    Some(BracedForm::Slice { start, length })
}

/// How many bytes of `after_dollar` one matched alternative consumed.
/// Only the matched forms reach it — the caller advances past a match, so
/// every arm has a length.
fn consumed_len(after_dollar: &str) -> usize {
    if let Some(inside) = after_dollar.strip_prefix('{') {
        let close = inside.find('}').unwrap_or(inside.len());
        return close + 2;
    }
    if let Some(rest) = after_dollar.strip_prefix("ARGUMENTS") {
        return after_dollar.len() - rest.len();
    }
    if after_dollar.starts_with('@') {
        return 1;
    }
    after_dollar
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_dollar.len())
}

/// Load one template file, upstream's `loadTemplateFromFile`. A file that
/// cannot be read or whose frontmatter does not parse loads as nothing,
/// upstream's bare catch.
fn load_template_from_file(file_path: &str, source_info: SourceInfo) -> Option<PromptTemplate> {
    let Ok(raw_content) = std::fs::read_to_string(file_path) else {
        return None;
    };
    let Ok(parsed) = parse_frontmatter(&raw_content) else {
        return None;
    };

    let name = basename_posix(file_path);
    let name = name.strip_suffix(".md").unwrap_or(&name).to_string();

    // The description comes from the frontmatter or the first non-blank
    // body line, clipped at sixty characters with an ellipsis.
    let mut description = frontmatter_string(&parsed.frontmatter, "description")
        .unwrap_or_default()
        .to_string();
    if description.is_empty()
        && let Some(first_line) = parsed.body.split('\n').find(|line| !line.trim().is_empty())
    {
        description = first_line.chars().take(60).collect();
        if first_line.chars().count() > 60 {
            description.push_str("...");
        }
    }

    Some(PromptTemplate {
        name,
        description,
        argument_hint: frontmatter_string(&parsed.frontmatter, "argument-hint")
            .filter(|hint| !hint.is_empty())
            .map(str::to_string),
        content: parsed.body,
        source_info,
        file_path: file_path.to_string(),
    })
}

/// Scan a directory's direct `.md` children as prompt templates,
/// upstream's `loadTemplatesFromDir`.
///
/// Non-recursive; broken symlinks skip.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's md ends-with probe is case-sensitive; the 1:1 shape keeps it"
)]
fn load_templates_from_dir(
    dir: &str,
    get_source_info: &dyn Fn(&str) -> SourceInfo,
) -> Vec<PromptTemplate> {
    let mut templates = Vec::new();
    let Ok(read) = std::fs::read_dir(dir) else {
        return templates;
    };

    for entry in read.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let is_file = if file_type.is_symlink() {
            match crate::utils::paths::symlink_resolved_kind(&entry.path()) {
                Some((_, is_file)) => is_file,
                None => continue,
            }
        } else {
            file_type.is_file()
        };

        let full_path = entry.path().to_string_lossy().into_owned();
        if is_file
            && name.ends_with(".md")
            && let Some(template) = load_template_from_file(&full_path, get_source_info(&full_path))
        {
            templates.push(template);
        }
    }

    templates
}

/// The load options, upstream's `LoadPromptTemplatesOptions`.
#[derive(Debug)]
pub struct LoadPromptTemplatesOptions<'a> {
    /// The working directory for project-local templates.
    pub cwd: &'a str,
    /// The agent config directory for global templates.
    pub agent_dir: &'a str,
    /// The explicit prompt template paths (files or directories).
    pub prompt_paths: &'a [String],
    /// Whether the default prompt directories load.
    pub include_defaults: bool,
}

/// Load all prompt templates, upstream's `loadPromptTemplates`.
///
/// The global `prompts/` dir, the project `.pi/prompts/` dir when
/// defaults are on, then the explicit paths. Read failures and
/// non-markdown files skip silently, upstream's bare catches.
#[must_use]
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's md ends-with probe is case-sensitive; the 1:1 shape keeps it"
)]
pub fn load_prompt_templates(options: &LoadPromptTemplatesOptions<'_>) -> Vec<PromptTemplate> {
    let home = crate::config::home_dir();
    let resolved_cwd = resolve_path(options.cwd, &crate::config::process_cwd(), &home);
    let resolved_agent_dir = resolve_path(options.agent_dir, &crate::config::process_cwd(), &home);

    let mut templates = Vec::new();

    let global_prompts_dir = Path::new(&resolved_agent_dir)
        .join("prompts")
        .to_string_lossy()
        .into_owned();
    let project_prompts_dir = Path::new(&resolved_cwd)
        .join(crate::config::CONFIG_DIR_NAME)
        .join("prompts")
        .to_string_lossy()
        .into_owned();

    let get_source_info = |resolved_path: &str| -> SourceInfo {
        if is_under_path(resolved_path, &global_prompts_dir) {
            return create_synthetic_source_info(
                resolved_path,
                &SyntheticSourceOptions {
                    source: "local".to_string(),
                    scope: Some(SourceScope::User),
                    origin: None,
                    base_dir: Some(global_prompts_dir.clone()),
                },
            );
        }
        if is_under_path(resolved_path, &project_prompts_dir) {
            return create_synthetic_source_info(
                resolved_path,
                &SyntheticSourceOptions {
                    source: "local".to_string(),
                    scope: Some(SourceScope::Project),
                    origin: None,
                    base_dir: Some(project_prompts_dir.clone()),
                },
            );
        }
        let is_directory = std::fs::metadata(resolved_path).is_ok_and(|stats| stats.is_dir());
        create_synthetic_source_info(
            resolved_path,
            &SyntheticSourceOptions {
                source: "local".to_string(),
                scope: None,
                origin: None,
                base_dir: Some(if is_directory {
                    resolved_path.to_string()
                } else {
                    dirname_posix(resolved_path)
                }),
            },
        )
    };

    if options.include_defaults {
        templates.extend(load_templates_from_dir(
            &global_prompts_dir,
            &get_source_info,
        ));
        templates.extend(load_templates_from_dir(
            &project_prompts_dir,
            &get_source_info,
        ));
    }

    for raw_path in options.prompt_paths {
        let resolved_path = resolve_path_with(
            raw_path,
            &resolved_cwd,
            &PathInputOptions {
                trim: true,
                ..PathInputOptions::default()
            },
        )
        .unwrap_or_else(|_| raw_path.clone());
        if !Path::new(&resolved_path).exists() {
            continue;
        }

        let Ok(stats) = std::fs::metadata(&resolved_path) else {
            continue;
        };
        if stats.is_dir() {
            templates.extend(load_templates_from_dir(&resolved_path, &get_source_info));
        } else if stats.is_file()
            && resolved_path.ends_with(".md")
            && let Some(template) =
                load_template_from_file(&resolved_path, get_source_info(&resolved_path))
        {
            templates.push(template);
        }
    }

    templates
}

/// Expand a prompt template invocation, upstream's `expandPromptTemplate`:
/// `/name args` against the loaded templates' names, expanding to the
/// substituted body; anything else passes through.
#[must_use]
pub fn expand_prompt_template(text: &str, templates: &[PromptTemplate]) -> String {
    if !text.starts_with('/') {
        return text.to_string();
    }

    // Upstream's `/^\/([^\s]+)(?:\s+([\s\S]*))?$/`: the first
    // non-whitespace run is the template name, the remainder the
    // arguments.
    let body = &text[1..];
    let name_end = body.find(char::is_whitespace).unwrap_or(body.len());
    if name_end == 0 {
        return text.to_string();
    }
    let template_name = &body[..name_end];
    let args_string = body[name_end..].trim_start().to_string();

    if let Some(template) = templates.iter().find(|t| t.name == template_name) {
        let args = parse_command_args(&args_string);
        return substitute_args(&template.content, &args);
    }

    text.to_string()
}
