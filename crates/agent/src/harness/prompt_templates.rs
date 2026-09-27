//! The prompt-template loader and invocation formatting, ported from
//! upstream `src/harness/prompt-templates.ts`.
//!
//! `loadPromptTemplates` reads `.md` files through the
//! [`ExecutionEnv`] capability:
//! directory inputs load direct `.md` children non-recursively, file inputs
//! load explicit `.md` files, missing paths and non-markdown files are
//! skipped, and read/parse failures come back as diagnostics. A template
//! with no frontmatter description takes its first non-blank body line,
//! clipped at sixty characters with an ellipsis.

use crate::harness::context::Context;
use crate::harness::fs_scan::{
    LoadDiagnostic, SourcedDiagnostic, SourcedPath, frontmatter_string, parse_frontmatter,
    resolve_kind, sort_entries_by_name,
};
use crate::harness::types::{ExecutionEnv, FileErrorCode, FileKind, PromptTemplate};

/// The stable prompt-template diagnostic codes, upstream's
/// `PromptTemplateDiagnosticCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptTemplateDiagnosticCode {
    /// A file-info or canonical-path probe failed outside `not_found`.
    FileInfoFailed,
    /// Listing a directory failed.
    ListFailed,
    /// Reading a file failed.
    ReadFailed,
    /// The frontmatter does not parse.
    ParseFailed,
}

/// A warning produced while loading prompt templates, upstream's
/// `PromptTemplateDiagnostic`.
pub type PromptTemplateDiagnostic = LoadDiagnostic<PromptTemplateDiagnosticCode>;

/// The prompt templates and diagnostics one load produced, upstream's
/// `{ promptTemplates, diagnostics }` return object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadedPromptTemplates {
    /// The templates that loaded, in path then name order.
    pub prompt_templates: Vec<PromptTemplate>,
    /// The warnings the load produced.
    pub diagnostics: Vec<PromptTemplateDiagnostic>,
}

/// A prompt template paired with the source it was loaded for, upstream's
/// `{ promptTemplate, source }` element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourcedPromptTemplate<TPromptTemplate, TSource> {
    /// The template, mapped when the caller supplied a mapper.
    pub prompt_template: TPromptTemplate,
    /// The source value the loader input carried.
    pub source: TSource,
}

/// The sourced prompt templates and diagnostics one load produced, upstream's
/// `{ promptTemplates, diagnostics }` return object of
/// `loadSourcedPromptTemplates`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedSourcedPromptTemplates<TPromptTemplate, TSource> {
    /// The sourced templates, in input then name order.
    pub prompt_templates: Vec<SourcedPromptTemplate<TPromptTemplate, TSource>>,
    /// The warnings with their sources attached.
    pub diagnostics: Vec<SourcedDiagnostic<PromptTemplateDiagnostic, TSource>>,
}

/// The optional template mapper, upstream's `mapPromptTemplate` parameter:
/// given the loaded template, the input's source, and the context, it
/// produces the application's template type.
pub type PromptTemplateMapper<'map, TSource, TPromptTemplate> =
    &'map dyn Fn(&PromptTemplate, &TSource, &Context) -> TPromptTemplate;

/// Load prompt templates from one or more paths, upstream's
/// `loadPromptTemplates`.
///
/// Directory inputs load direct `.md` children non-recursively. File inputs
/// load explicit `.md` files. Missing paths and non-markdown files are
/// skipped. Read and parse failures are returned as diagnostics.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's `info.name.endsWith(\".md\")` is case-sensitive; the 1:1 shape keeps it"
)]
pub async fn load_prompt_templates(
    env: &dyn ExecutionEnv,
    paths: &[String],
    context: &Context,
) -> LoadedPromptTemplates {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    for path in paths {
        let info = match env.file_info(path, context).await {
            Err(error) => {
                if error.code != FileErrorCode::NotFound {
                    diagnostics.push(PromptTemplateDiagnostic::warning(
                        PromptTemplateDiagnosticCode::FileInfoFailed,
                        error.message,
                        path.clone(),
                    ));
                }
                continue;
            }
            Ok(info) => info,
        };
        match resolve_kind(
            env,
            &info,
            &mut diagnostics,
            PromptTemplateDiagnosticCode::FileInfoFailed,
            context,
        )
        .await
        {
            Some(FileKind::Directory) => {
                let (loaded, loaded_diagnostics) =
                    load_templates_from_dir(env, &info.path, context).await;
                prompt_templates.extend(loaded);
                diagnostics.extend(loaded_diagnostics);
            }
            Some(FileKind::File) if info.name.ends_with(".md") => {
                let (template, loaded_diagnostics) =
                    load_template_from_file(env, &info.path, &info.name, context).await;
                if let Some(template) = template {
                    prompt_templates.push(template);
                }
                diagnostics.extend(loaded_diagnostics);
            }
            _ => {}
        }
    }
    LoadedPromptTemplates {
        prompt_templates,
        diagnostics,
    }
}

/// Load prompt templates from source-tagged paths, upstream's
/// `loadSourcedPromptTemplates` with no mapper.
///
/// Source values are preserved exactly and attached to every loaded prompt
/// template and diagnostic. The agent package does not interpret source
/// values; applications define their own provenance shape. Upstream's
/// optional mapper and default `TPromptTemplate` restates as the identity
/// variant here and [`load_sourced_prompt_templates_mapped`] beside it:
/// TypeScript's `TPromptTemplate extends PromptTemplate` has no structural
/// subtyping equivalent, so the identity case clones through a
/// pass-through mapper.
pub async fn load_sourced_prompt_templates<TSource: Clone>(
    env: &dyn ExecutionEnv,
    inputs: &[SourcedPath<TSource>],
    context: &Context,
) -> LoadedSourcedPromptTemplates<PromptTemplate, TSource> {
    load_sourced_prompt_templates_mapped(
        env,
        inputs,
        &|prompt_template, _source, _context| prompt_template.clone(),
        context,
    )
    .await
}

/// Load prompt templates from source-tagged paths with a mapper, upstream's
/// `loadSourcedPromptTemplates` with `mapPromptTemplate`.
///
/// The mapper receives the loaded template, the input's source, and the
/// context, and produces the application's template type.
pub async fn load_sourced_prompt_templates_mapped<TSource: Clone, TPromptTemplate>(
    env: &dyn ExecutionEnv,
    inputs: &[SourcedPath<TSource>],
    map_prompt_template: PromptTemplateMapper<'_, TSource, TPromptTemplate>,
    context: &Context,
) -> LoadedSourcedPromptTemplates<TPromptTemplate, TSource> {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    for input in inputs {
        let result = load_prompt_templates(env, std::slice::from_ref(&input.path), context).await;
        for prompt_template in result.prompt_templates {
            prompt_templates.push(SourcedPromptTemplate {
                prompt_template: map_prompt_template(&prompt_template, &input.source, context),
                source: input.source.clone(),
            });
        }
        for diagnostic in result.diagnostics {
            diagnostics.push(SourcedDiagnostic {
                diagnostic,
                source: input.source.clone(),
            });
        }
    }
    LoadedSourcedPromptTemplates {
        prompt_templates,
        diagnostics,
    }
}

/// Load a directory's direct `.md` children, upstream's `loadTemplatesFromDir`.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's `entry.name.endsWith(\".md\")` is case-sensitive; the 1:1 shape keeps it"
)]
async fn load_templates_from_dir(
    env: &dyn ExecutionEnv,
    dir: &str,
    context: &Context,
) -> (Vec<PromptTemplate>, Vec<PromptTemplateDiagnostic>) {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    let mut entries = match env.list_dir(dir, context).await {
        Err(error) => {
            diagnostics.push(PromptTemplateDiagnostic::warning(
                PromptTemplateDiagnosticCode::ListFailed,
                error.message,
                dir,
            ));
            return (prompt_templates, diagnostics);
        }
        Ok(entries) => entries,
    };

    sort_entries_by_name(&mut entries);
    for entry in &entries {
        if resolve_kind(
            env,
            entry,
            &mut diagnostics,
            PromptTemplateDiagnosticCode::FileInfoFailed,
            context,
        )
        .await
            != Some(FileKind::File)
            || !entry.name.ends_with(".md")
        {
            continue;
        }
        let (template, loaded_diagnostics) =
            load_template_from_file(env, &entry.path, &entry.name, context).await;
        if let Some(template) = template {
            prompt_templates.push(template);
        }
        diagnostics.extend(loaded_diagnostics);
    }
    (prompt_templates, diagnostics)
}

/// Load one template file, upstream's `loadTemplateFromFile`.
///
/// The description comes from the frontmatter, falling back to the first
/// non-blank body line clipped at sixty characters with an ellipsis.
async fn load_template_from_file(
    env: &dyn ExecutionEnv,
    file_path: &str,
    file_name: &str,
    context: &Context,
) -> (Option<PromptTemplate>, Vec<PromptTemplateDiagnostic>) {
    let mut diagnostics = Vec::new();
    let raw_content = match env.read_text_file(file_path, context).await {
        Err(error) => {
            diagnostics.push(PromptTemplateDiagnostic::warning(
                PromptTemplateDiagnosticCode::ReadFailed,
                error.message,
                file_path,
            ));
            return (None, diagnostics);
        }
        Ok(raw_content) => raw_content,
    };

    let parsed = match parse_frontmatter(&raw_content) {
        Err(error) => {
            diagnostics.push(PromptTemplateDiagnostic::warning(
                PromptTemplateDiagnosticCode::ParseFailed,
                error.0,
                file_path,
            ));
            return (None, diagnostics);
        }
        Ok(parsed) => parsed,
    };

    let body = parsed.body;
    let first_line = body.split('\n').find(|line| !line.trim().is_empty());
    let mut description = frontmatter_string(&parsed.frontmatter, "description")
        .unwrap_or_default()
        .to_owned();
    if description.is_empty()
        && let Some(first_line) = first_line
    {
        description = first_line.chars().take(60).collect();
        if first_line.chars().count() > 60 {
            description.push_str("...");
        }
    }
    (
        Some(PromptTemplate {
            // The strip is case-insensitive, upstream's `/\.md$/i`; the
            // selection checks elsewhere stay case-sensitive, as upstream's
            // `endsWith(".md")` checks do.
            name: file_name
                .strip_suffix(".md")
                .or_else(|| file_name.strip_suffix(".MD"))
                .or_else(|| file_name.strip_suffix(".Md"))
                .or_else(|| file_name.strip_suffix(".mD"))
                .unwrap_or(file_name)
                .to_owned(),
            description: Some(description),
            content: body,
        }),
        diagnostics,
    )
}

/// Parse an argument string using simple shell-style single and double
/// quotes, upstream's `parseCommandArgs`.
///
/// Quotes toggle and never terminate early; space and tab split; an
/// unterminated quote keeps its content.
#[must_use]
pub fn parse_command_args(args_string: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quote: Option<char> = None;

    for character in args_string.chars() {
        match in_quote {
            Some(quote) => {
                if character == quote {
                    in_quote = None;
                } else {
                    current.push(character);
                }
            }
            None if character == '"' || character == '\'' => in_quote = Some(character),
            None if character == ' ' || character == '\t' => {
                if !current.is_empty() {
                    args.push(std::mem::take(&mut current));
                }
            }
            None => current.push(character),
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

/// Substitute prompt template placeholders (`$1`, `$@`, `$ARGUMENTS`,
/// `${@:N}`, `${@:N:L}`) with command arguments, upstream's
/// `substituteArgs`.
///
/// The passes run in upstream's order — positional `$N` first, then the
/// `${@:N}` and `${@:N:L}` slices, then `$ARGUMENTS`, then `$@` — and
/// substituted text is rescanned by the later passes, so an argument value
/// carrying a placeholder participates like upstream's sequential
/// `String.replace` calls. `$0` and out-of-range positions read empty;
/// `${@:0}` clamps to the first argument. Substituted values insert
/// literally, upstream's replacement-string `$` patterns notwithstanding.
#[must_use]
pub fn substitute_args(content: &str, args: &[String]) -> String {
    let result = substitute_positional(content, args);
    let result = substitute_slices(&result, args);
    let all_args = args.join(" ");
    result
        .replace("$ARGUMENTS", &all_args)
        .replace("$@", &all_args)
}

/// Replace `$N` with the Nth argument (1-based; `$0` and out-of-range read
/// empty), upstream's `/\$(\d+)/g` pass.
fn substitute_positional(content: &str, args: &[String]) -> String {
    let mut out = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(at) = rest.find('$') {
        out.push_str(&rest[..at]);
        rest = &rest[at + 1..];
        let digits_end = rest
            .find(|character: char| !character.is_ascii_digit())
            .unwrap_or(rest.len());
        let digits = &rest[..digits_end];
        if digits.is_empty() {
            out.push('$');
            continue;
        }
        // Upstream's `parseInt` saturates past `usize`; the out-of-range
        // index reads empty either way.
        let number = digits.parse::<usize>().unwrap_or(usize::MAX);
        let value = number
            .checked_sub(1)
            .and_then(|index| args.get(index))
            .map(String::as_str)
            .unwrap_or_default();
        out.push_str(value);
        rest = &rest[digits_end..];
    }
    out.push_str(rest);
    out
}

/// Replace `${@:N}` and `${@:N:L}` with the argument slice joined by
/// spaces, upstream's `/\$\{@:(\d+)(?::(\d+))?\}/g` pass.
///
/// `N` is 1-based and clamps to the first argument at zero; JavaScript
/// `slice`'s clamping restates as saturation against the argument list. A
/// shape the regex cannot match (no leading digits, a missing or
/// non-digit length, a missing closing brace) stays literal, and scanning
/// resumes after the `${@:` prefix the way the global regex rescans from
/// the next byte.
fn substitute_slices(content: &str, args: &[String]) -> String {
    let mut out = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(at) = rest.find("${@:") {
        out.push_str(&rest[..at]);
        let after = &rest[at + "${@:".len()..];
        let start_digits_end = after
            .find(|character: char| !character.is_ascii_digit())
            .unwrap_or(after.len());
        let start_digits = &after[..start_digits_end];
        let after_start = &after[start_digits_end..];
        let pattern = if start_digits.is_empty() {
            None
        } else if let Some(after_brace) = after_start.strip_prefix('}') {
            Some((None, after_brace))
        } else if let Some(after_colon) = after_start.strip_prefix(':') {
            let length_end = after_colon
                .find(|character: char| !character.is_ascii_digit())
                .unwrap_or(after_colon.len());
            let length_digits = &after_colon[..length_end];
            match after_colon[length_end..].strip_prefix('}') {
                Some(after_brace) if !length_digits.is_empty() => {
                    Some((Some(length_digits), after_brace))
                }
                _ => None,
            }
        } else {
            None
        };
        if let Some((length_digits, after_pattern)) = pattern {
            // Upstream's `parseInt` saturates past `usize`; the clamped
            // slice reads empty either way.
            let start = start_digits
                .parse::<usize>()
                .unwrap_or(usize::MAX)
                .saturating_sub(1)
                .min(args.len());
            let end = length_digits.map_or(args.len(), |length| {
                start
                    .saturating_add(length.parse::<usize>().unwrap_or(usize::MAX))
                    .min(args.len())
            });
            out.push_str(&args[start..end].join(" "));
            rest = after_pattern;
        } else {
            out.push_str("${@:");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Format a prompt template invocation with positional arguments, upstream's
/// `formatPromptTemplateInvocation`.
#[must_use]
pub fn format_prompt_template_invocation(template: &PromptTemplate, args: &[String]) -> String {
    substitute_args(&template.content, args)
}

#[cfg(test)]
mod tests;
