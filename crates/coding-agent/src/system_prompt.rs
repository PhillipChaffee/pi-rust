//! System prompt construction, upstream's `src/core/system-prompt.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::config::{get_docs_path, get_examples_path, get_readme_path};
use crate::skills::{Skill, format_skills_for_prompt};

/// The system-prompt options, upstream's `BuildSystemPromptOptions`.
#[derive(Debug, Default)]
pub struct BuildSystemPromptOptions {
    /// A custom system prompt that replaces the default.
    pub custom_prompt: Option<String>,
    /// The tools to name in the prompt. Defaults to
    /// `["read", "bash", "edit", "write"]`.
    pub selected_tools: Option<Vec<String>>,
    /// One-line snippets keyed by tool name; a tool appears in
    /// `Available tools` only when it carries one.
    pub tool_snippets: Option<std::collections::HashMap<String, String>>,
    /// Additional guideline bullets appended to the default guidelines.
    pub prompt_guidelines: Option<Vec<String>>,
    /// Text appended after the main prompt.
    pub append_system_prompt: Option<String>,
    /// The working directory.
    pub cwd: String,
    /// Pre-loaded context files.
    pub context_files: Option<Vec<ContextFile>>,
    /// Pre-loaded skills.
    pub skills: Option<Vec<Skill>>,
}

/// A project context file, upstream's `{ path, content }` element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFile {
    /// The absolute path the file loaded from.
    pub path: String,
    /// The file's text.
    pub content: String,
}

/// Build the system prompt with tools, guidelines, and context, upstream's
/// `buildSystemPrompt`.
///
/// A custom prompt replaces the default body but keeps the append,
/// project-context, and skills sections.
///
/// The default body names the tools that carry snippets, states the
/// shell/file-exploration guidelines that match the tool set, and points
/// at the pi docs and examples the binary ships.
#[expect(
    clippy::too_many_lines,
    reason = "the 1:1 restatement of upstream's buildSystemPrompt reads as one block per section"
)]
#[must_use]
pub fn build_system_prompt(options: &BuildSystemPromptOptions) -> String {
    let prompt_cwd = options.cwd.replace('\\', "/");

    let append_section = options
        .append_system_prompt
        .as_ref()
        .map(|append| format!("\n\n{append}"))
        .unwrap_or_default();

    let context_files = options.context_files.clone().unwrap_or_default();
    let skills = options.skills.clone().unwrap_or_default();
    let tools = options.selected_tools.clone().unwrap_or_else(|| {
        ["read", "bash", "edit", "write"]
            .iter()
            .map(|tool| (*tool).to_string())
            .collect()
    });
    // The first of read/bash the tool set carries decides how the skills
    // block tells the model to load skill files.
    let skill_file_read_tool = ["read", "bash"]
        .iter()
        .find(|tool| tools.contains(&(*tool).to_string()))
        .map(|tool| (*tool).to_string());

    if let Some(custom_prompt) = &options.custom_prompt {
        let mut prompt = custom_prompt.clone();

        if !append_section.is_empty() {
            prompt.push_str(&append_section);
        }

        // Append project context files.
        if !context_files.is_empty() {
            prompt.push_str("\n\n<project_context>\n\n");
            prompt.push_str("Project-specific instructions and guidelines:\n\n");
            for ContextFile { path, content } in &context_files {
                prompt.push_str("<project_instructions path=\"");
                prompt.push_str(path);
                prompt.push_str("\">\n");
                prompt.push_str(content);
                prompt.push_str("\n</project_instructions>\n\n");
            }
            prompt.push_str("</project_context>\n");
        }

        // Append skills when a tool capable of reading their files is
        // available.
        if skill_file_read_tool.is_some() && !skills.is_empty() {
            prompt.push_str(&format_skills_for_prompt(
                &skills,
                skill_file_read_tool.as_deref().unwrap_or("read"),
            ));
        }

        prompt.push_str("\nCurrent working directory: ");
        prompt.push_str(&prompt_cwd);
        prompt.push('\n');

        return prompt;
    }

    // Absolute paths to the documentation and examples the binary ships.
    let readme_path = get_readme_path();
    let docs_path = get_docs_path();
    let examples_path = get_examples_path();

    // Build the tools list: a tool appears in Available tools only when
    // the caller provides a one-line snippet.
    let snippets = options.tool_snippets.clone().unwrap_or_default();
    let visible_tools: Vec<&String> = tools
        .iter()
        .filter(|name| snippets.contains_key((*name).as_str()))
        .collect();
    let tools_list = if visible_tools.is_empty() {
        "(none)".to_string()
    } else {
        visible_tools
            .iter()
            .map(|name| format!("- {name}: {}", snippets[(*name).as_str()]))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Build the guidelines based on which tools are actually available;
    // a guideline states once.
    let mut guidelines_list: Vec<String> = Vec::new();
    let mut guidelines_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    let add_guideline =
        |guideline: &str,
         guidelines_list: &mut Vec<String>,
         guidelines_set: &mut std::collections::HashSet<String>| {
            if guidelines_set.contains(guideline) {
                return;
            }
            guidelines_set.insert(guideline.to_string());
            guidelines_list.push(guideline.to_string());
        };

    let has_bash = tools.iter().any(|tool| tool == "bash");
    let has_power_shell = tools.iter().any(|tool| tool == "powershell");
    let has_grep = tools.iter().any(|tool| tool == "grep");
    let has_find = tools.iter().any(|tool| tool == "find");
    let has_ls = tools.iter().any(|tool| tool == "ls");

    // File exploration guidelines.
    if (has_bash || has_power_shell) && !has_grep && !has_find && !has_ls {
        if has_bash && has_power_shell {
            add_guideline(
                "Use bash or PowerShell for file operations like listing, searching, and finding files",
                &mut guidelines_list,
                &mut guidelines_set,
            );
        } else if has_power_shell {
            add_guideline(
                "Use PowerShell for file operations like listing, searching, and finding files",
                &mut guidelines_list,
                &mut guidelines_set,
            );
        } else {
            add_guideline(
                "Use bash for file operations like ls, rg, find",
                &mut guidelines_list,
                &mut guidelines_set,
            );
        }
    }

    for guideline in options.prompt_guidelines.iter().flatten() {
        let normalized = guideline.trim();
        if !normalized.is_empty() {
            add_guideline(normalized, &mut guidelines_list, &mut guidelines_set);
        }
    }

    // Always include these.
    add_guideline(
        "Be concise in your responses",
        &mut guidelines_list,
        &mut guidelines_set,
    );
    add_guideline(
        "Show file paths clearly when working with files",
        &mut guidelines_list,
        &mut guidelines_set,
    );

    let guidelines = guidelines_list
        .iter()
        .map(|g| format!("- {g}"))
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt = format!(
        "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n\
\n\
Available tools:\n\
{tools_list}\n\
\n\
In addition to the tools above, you may have access to other custom tools depending on the project.\n\
\n\
Guidelines:\n\
{guidelines}\n\
\n\
Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n\
- Main documentation: {readme_path}\n\
- Additional docs: {docs_path}\n\
- Examples: {examples_path} (extensions, custom tools, SDK)\n\
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory\n\
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md)\n\
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing\n\
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)"
    );

    if !append_section.is_empty() {
        prompt.push_str(&append_section);
    }

    // Append project context files.
    if !context_files.is_empty() {
        prompt.push_str("\n\n<project_context>\n\n");
        prompt.push_str("Project-specific instructions and guidelines:\n\n");
        for ContextFile { path, content } in &context_files {
            prompt.push_str("<project_instructions path=\"");
            prompt.push_str(path);
            prompt.push_str("\">\n");
            prompt.push_str(content);
            prompt.push_str("\n</project_instructions>\n\n");
        }
        prompt.push_str("</project_context>\n");
    }

    // Append skills when a tool capable of reading their files is
    // available.
    if skill_file_read_tool.is_some() && !skills.is_empty() {
        prompt.push_str(&format_skills_for_prompt(
            &skills,
            skill_file_read_tool.as_deref().unwrap_or("read"),
        ));
    }

    prompt.push_str("\nCurrent working directory: ");
    prompt.push_str(&prompt_cwd);

    prompt
}
