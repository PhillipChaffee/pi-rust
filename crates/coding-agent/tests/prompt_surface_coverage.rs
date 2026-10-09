//! Coverage-closing cases for the prompt surface at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the system prompt's
//! append + project-context rendering in both the custom and the default
//! branch, and the skill loader's validation and filesystem-warning arms
//! the 1:1 suites leave open. Deliberately uncovered, mirroring
//! upstream's reachability: the `read_dir` and `file_type` stat-failure
//! arms (only a filesystem race reaches them) and the "failed to read
//! skill path" metadata warning for the same reason.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use pi_coding_agent::diagnostics::ResourceDiagnosticKind;
use pi_coding_agent::skills::{LoadSkillsOptions, load_skills};
use pi_coding_agent::source_info::SourceScope;
use pi_coding_agent::system_prompt::{BuildSystemPromptOptions, ContextFile, build_system_prompt};

struct SkillsEnv {
    root: PathBuf,
    agent_dir: String,
    cwd: String,
}

impl SkillsEnv {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("tempdir").keep();
        let agent_dir = root.join("agent");
        let cwd = root.join("project");
        std::fs::create_dir_all(&agent_dir).expect("mkdir");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        Self {
            agent_dir: agent_dir.to_string_lossy().into_owned(),
            cwd: cwd.to_string_lossy().into_owned(),
            root,
        }
    }

    fn mkdir(&self, relative: &str) {
        std::fs::create_dir_all(self.root.join(relative)).expect("mkdir");
    }

    fn write(&self, relative: &str, content: &str) -> String {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&path, content).expect("write");
        path.to_string_lossy().into_owned()
    }
}

fn make_unreadable(path: &Path) {
    let mut permissions = std::fs::metadata(path).expect("stat").permissions();
    permissions.set_mode(0o000);
    std::fs::set_permissions(path, permissions).expect("chmod");
}

// === system prompt ===========================================================

#[test]
fn custom_prompt_renders_the_append_and_project_context_sections() {
    let prompt = build_system_prompt(&BuildSystemPromptOptions {
        custom_prompt: Some("Custom body.".to_string()),
        append_system_prompt: Some("Appendix text.".to_string()),
        cwd: "/tmp/cwd".to_string(),
        context_files: Some(vec![ContextFile {
            path: "/repo/AGENTS.md".to_string(),
            content: "Be tidy.".to_string(),
        }]),
        ..BuildSystemPromptOptions::default()
    });

    assert!(prompt.starts_with("Custom body."));
    assert!(prompt.contains("Appendix text."));
    assert!(prompt.contains("<project_context>"));
    assert!(prompt.contains("Project-specific instructions and guidelines:"));
    assert!(prompt.contains("<project_instructions path=\"/repo/AGENTS.md\">"));
    assert!(prompt.contains("Be tidy."));
    assert!(prompt.contains("</project_instructions>"));
    assert!(prompt.contains("</project_context>"));
}

#[test]
fn default_prompt_renders_the_append_and_project_context_sections() {
    let prompt = build_system_prompt(&BuildSystemPromptOptions {
        append_system_prompt: Some("Appendix text.".to_string()),
        cwd: "/tmp/cwd".to_string(),
        context_files: Some(vec![ContextFile {
            path: "/repo/AGENTS.md".to_string(),
            content: "Be tidy.".to_string(),
        }]),
        ..BuildSystemPromptOptions::default()
    });

    assert!(prompt.contains("Appendix text."));
    assert!(prompt.contains("<project_context>"));
    assert!(prompt.contains("<project_instructions path=\"/repo/AGENTS.md\">"));
    assert!(prompt.contains("Be tidy."));
    assert!(prompt.contains("</project_context>"));
}

// === skill loader validation and warning arms ================================

#[test]
fn name_and_description_validation_arms_report_their_messages() {
    let env = SkillsEnv::new();
    let hyphen = env.write(
        "agent/skills/hyphen/SKILL.md",
        "---\nname: -lead\ndescription: fine\n---\nbody",
    );
    let empty_description = env.write(
        "agent/skills/empty/SKILL.md",
        "---\nname: empty-desc\ndescription: \"   \"\n---\nbody",
    );
    let long_description = env.write(
        "agent/skills/long/SKILL.md",
        &format!(
            "---\nname: long-desc\ndescription: {}\n---\nbody",
            "x".repeat(1025)
        ),
    );

    let result = load_skills(&LoadSkillsOptions {
        cwd: &env.cwd,
        agent_dir: &env.agent_dir,
        skill_paths: &[],
        include_defaults: true,
    });

    let messages: Vec<&str> = result
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        messages.contains(&"name must not start or end with a hyphen"),
        "{messages:?}"
    );
    assert!(messages.contains(&"description is required"));
    assert!(
        messages
            .iter()
            .any(|m| m.starts_with("description exceeds 1024 characters (1025)")),
        "{messages:?}"
    );
    let _ = (hyphen, empty_description, long_description);
}

#[test]
fn unreadable_skill_files_warn_and_root_md_without_description_stays_silent() {
    let env = SkillsEnv::new();
    let unreadable = env.write(
        "agent/skills/locked/SKILL.md",
        "---\nname: locked\ndescription: nope\n---\nbody",
    );
    make_unreadable(Path::new(&unreadable));
    // A root `.md` skill without a description neither loads nor warns
    // (the description rule applies to declared SKILL.md files only).
    let _plain = env.write("agent/skills/plain.md", "no frontmatter at all");

    let result = load_skills(&LoadSkillsOptions {
        cwd: &env.cwd,
        agent_dir: &env.agent_dir,
        skill_paths: &[],
        include_defaults: true,
    });

    assert!(result.diagnostics.iter().any(|d| {
        d.kind == ResourceDiagnosticKind::Warning
            && d.message == format!("failed to read skill file: {unreadable}")
            && d.path.as_deref() == Some(unreadable.as_str())
    }));
    assert!(!result.skills.iter().any(|skill| skill.name == "plain"));
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| d.path.as_deref().is_some_and(|p| p.ends_with("plain.md")))
    );
}

#[test]
fn symlinked_skill_files_load_dangling_links_are_skipped() {
    let env = SkillsEnv::new();
    let target = env.write(
        "agent/skills/real/SKILL.md",
        "---\nname: real\ndescription: target\n---\nbody",
    );
    env.mkdir("agent/skills/link");
    env.mkdir("agent/skills/dangling");
    env.mkdir("agent/skills/dirlink");
    std::os::unix::fs::symlink(
        Path::new(&target),
        env.root.join("agent/skills/link/SKILL.md"),
    )
    .expect("symlink");
    std::os::unix::fs::symlink(
        env.root.join("agent/skills/missing/SKILL.md"),
        env.root.join("agent/skills/dangling/SKILL.md"),
    )
    .expect("symlink");
    // A symlink named SKILL.md resolving to a directory is not a file.
    std::os::unix::fs::symlink(
        env.root.join("agent/skills/real"),
        env.root.join("agent/skills/dirlink/SKILL.md"),
    )
    .expect("symlink");

    let result = load_skills(&LoadSkillsOptions {
        cwd: &env.cwd,
        agent_dir: &env.agent_dir,
        skill_paths: &[],
        include_defaults: true,
    });

    let names: Vec<&str> = result
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert!(names.contains(&"real"), "{names:?}");
    assert!(!names.contains(&"dangling"), "{names:?}");
    // Exactly one skill survives: the dangling and directory links
    // contribute nothing.
    assert_eq!(names.len(), 1, "{names:?}");
}

#[test]
fn explicit_path_kinds_warn_by_shape() {
    let env = SkillsEnv::new();
    let good = env.write(
        "good/SKILL.md",
        "---\nname: good\ndescription: fine\n---\nbody",
    );
    let not_markdown = env.write("shapes/notes.txt", "plain text");
    let broken = env.write("broken/SKILL.md", "---\nname: [unclosed\n---\nbody");

    let result = load_skills(&LoadSkillsOptions {
        cwd: &env.cwd,
        agent_dir: &env.agent_dir,
        skill_paths: &[good, not_markdown.clone(), broken],
        include_defaults: false,
    });

    let names: Vec<&str> = result
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, vec!["good"]);
    assert!(result.diagnostics.iter().any(|d| {
        d.kind == ResourceDiagnosticKind::Warning
            && d.message == "skill path is not a markdown file"
            && d.path.as_deref() == Some(not_markdown.as_str())
    }));
    // The declared SKILL.md with unparseable frontmatter warns with the
    // parser's message.
    assert!(result.diagnostics.iter().any(|d| {
        d.kind == ResourceDiagnosticKind::Warning
            && d.path
                .as_deref()
                .is_some_and(|p| p.ends_with("broken/SKILL.md"))
    }));
    assert!(
        result
            .skills
            .iter()
            .all(|skill| matches!(skill.source_info.scope, SourceScope::Temporary))
    );
}
