//! Upstream `test/system-prompt.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! The suite builds skills through the loader's shape; the doc-path
//! assertions ride the build-time `get_package_dir` restatement.

use std::collections::HashMap;

use pi_coding_agent::skills::Skill;
use pi_coding_agent::source_info::{SyntheticSourceOptions, create_synthetic_source_info};
use pi_coding_agent::system_prompt::{BuildSystemPromptOptions, build_system_prompt};

fn test_skill() -> Skill {
    Skill {
        name: "test-skill".to_string(),
        description: "A test skill.".to_string(),
        file_path: "/skills/test-skill/SKILL.md".to_string(),
        base_dir: "/skills/test-skill".to_string(),
        source_info: create_synthetic_source_info(
            "/skills/test-skill/SKILL.md",
            &SyntheticSourceOptions {
                source: "test".to_string(),
                ..SyntheticSourceOptions::default()
            },
        ),
        disable_model_invocation: false,
    }
}

fn base_options() -> BuildSystemPromptOptions {
    BuildSystemPromptOptions {
        cwd: std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        ..BuildSystemPromptOptions::default()
    }
}

// === empty tools ============================================================

#[test]
fn shows_none_for_empty_tools_list() {
    let mut options = base_options();
    options.selected_tools = Some(Vec::new());
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("Available tools:\n(none)"));
}

#[test]
fn shows_file_paths_guideline_even_with_no_tools() {
    let mut options = base_options();
    options.selected_tools = Some(Vec::new());
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("Show file paths clearly"));
}

// === default tools ==========================================================

#[test]
fn includes_all_default_tools_when_snippets_are_provided() {
    let mut options = base_options();
    let mut snippets = HashMap::new();
    snippets.insert("read".to_string(), "Read file contents".to_string());
    snippets.insert("bash".to_string(), "Execute bash commands".to_string());
    snippets.insert("edit".to_string(), "Make surgical edits".to_string());
    snippets.insert("write".to_string(), "Create or overwrite files".to_string());
    options.tool_snippets = Some(snippets);
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("- read:"));
    assert!(prompt.contains("- bash:"));
    assert!(prompt.contains("- edit:"));
    assert!(prompt.contains("- write:"));
}

#[test]
fn uses_shell_specific_guidance_for_powershell_only() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["powershell".to_string()]);
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("Use PowerShell for file operations"));
}

#[test]
fn uses_shell_specific_guidance_for_bash_and_powershell() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["bash".to_string(), "powershell".to_string()]);
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("Use bash or PowerShell for file operations"));
}

#[test]
fn instructs_models_to_resolve_pi_docs_and_examples_under_absolute_base_paths() {
    let options = base_options();

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains(
        "- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory"
    ));
    assert!(prompt.contains("environment variables (docs/environment-variables.md)"));
}

// === custom tool snippets ===================================================

#[test]
fn includes_custom_tools_in_available_tools_section_when_snippet_is_provided() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["read".to_string(), "dynamic_tool".to_string()]);
    let mut snippets = HashMap::new();
    snippets.insert(
        "dynamic_tool".to_string(),
        "Run dynamic test behavior".to_string(),
    );
    options.tool_snippets = Some(snippets);
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("- dynamic_tool: Run dynamic test behavior"));
}

#[test]
fn omits_custom_tools_from_available_tools_section_when_snippet_is_not_provided() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["read".to_string(), "dynamic_tool".to_string()]);
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(!prompt.contains("dynamic_tool"));
}

// === prompt guidelines ======================================================

#[test]
fn appends_prompt_guidelines_to_default_guidelines() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["read".to_string(), "dynamic_tool".to_string()]);
    options.prompt_guidelines = Some(vec!["Use dynamic_tool for project summaries.".to_string()]);
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("- Use dynamic_tool for project summaries."));
}

#[test]
fn deduplicates_and_trims_prompt_guidelines() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["read".to_string(), "dynamic_tool".to_string()]);
    options.prompt_guidelines = Some(vec![
        "Use dynamic_tool for summaries.".to_string(),
        "  Use dynamic_tool for summaries.  ".to_string(),
        "   ".to_string(),
    ]);
    options.context_files = Some(Vec::new());
    options.skills = Some(Vec::new());

    let prompt = build_system_prompt(&options);

    assert_eq!(
        prompt.matches("- Use dynamic_tool for summaries.").count(),
        1
    );
}

// === skills =================================================================

#[test]
fn includes_skills_with_only_bash_in_the_default_prompt() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["bash".to_string()]);
    options.context_files = Some(Vec::new());
    options.skills = Some(vec![test_skill()]);

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("<available_skills>"));
    assert!(prompt.contains("<name>test-skill</name>"));
    assert!(prompt.contains("Use bash to load a skill's file"));
}

#[test]
fn includes_skills_with_only_bash_in_a_custom_prompt() {
    let mut options = base_options();
    options.custom_prompt = Some("Custom system prompt".to_string());
    options.selected_tools = Some(vec!["bash".to_string()]);
    options.context_files = Some(Vec::new());
    options.skills = Some(vec![test_skill()]);

    let prompt = build_system_prompt(&options);

    assert!(prompt.contains("<available_skills>"));
    assert!(prompt.contains("<name>test-skill</name>"));
    assert!(prompt.contains("Use bash to load a skill's file"));
}

#[test]
fn omits_skills_without_read_or_bash() {
    let mut options = base_options();
    options.selected_tools = Some(vec!["write".to_string()]);
    options.context_files = Some(Vec::new());
    options.skills = Some(vec![test_skill()]);

    let prompt = build_system_prompt(&options);

    assert!(!prompt.contains("<available_skills>"));
}
