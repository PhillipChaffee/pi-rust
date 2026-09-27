//! The `skills.test.ts` suite ported 1:1, plus boundary tests binding the
//! restated surfaces upstream's suite does not reach: the ignore-file
//! machinery (npm `ignore` parity on the Rust `ignore` crate), the
//! first-`SKILL.md` early return, the silent skips, the metadata
//! validation messages, and the invocation formatter's path shapes.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use crate::harness::context::background_context;
use crate::harness::fs_scan::{SourcedDiagnostic, SourcedPath};
use crate::harness::skills::{
    SkillDiagnostic, SkillDiagnosticCode, SourcedSkill, format_skill_invocation, load_skills,
    load_sourced_skills, load_sourced_skills_mapped,
};
use crate::harness::test_support::{TestSource, env_for, file_path, mkdir, write};
use crate::harness::types::Skill;

/// Loads SKILL.md files through the execution environment, upstream's
/// "loads SKILL.md files through the execution environment".
#[tokio::test]
async fn loads_skill_md_files_through_the_execution_environment() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, ".agents/skills/example", &context).await;
    write(
        &env,
        ".agents/skills/example/SKILL.md",
        "---\nname: example\ndescription: Example skill\ndisable-model-invocation: true\n---\nUse this skill.\n",
        &context,
    )
    .await;

    let dirs = vec![".agents/skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert!(loaded.diagnostics.is_empty());
    assert_eq!(
        loaded.skills,
        vec![Skill {
            name: "example".to_owned(),
            description: "Example skill".to_owned(),
            content: "Use this skill.".to_owned(),
            file_path: file_path(root.path(), ".agents/skills/example/SKILL.md"),
            disable_model_invocation: Some(true),
        }]
    );
}

/// Loads skills through symlinked directories, upstream's "loads skills
/// through symlinked directories".
#[tokio::test]
async fn loads_skills_through_symlinked_directories() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "actual/example", &context).await;
    write(
        &env,
        "actual/example/SKILL.md",
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
        &context,
    )
    .await;
    std::os::unix::fs::symlink(
        file_path(root.path(), "actual"),
        file_path(root.path(), "skills-link"),
    )
    .expect("symlink");

    let dirs = vec!["skills-link".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let names: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, ["example"]);
    assert_eq!(
        loaded.skills[0].file_path,
        file_path(root.path(), "skills-link/example/SKILL.md")
    );
}

/// Preserves source info for sourced skills, upstream's "preserves source
/// info for sourced skills".
#[tokio::test]
async fn preserves_source_info_for_sourced_skills() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "user/example", &context).await;
    write(
        &env,
        "user/example/SKILL.md",
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
        &context,
    )
    .await;

    let inputs = vec![SourcedPath {
        path: "user".to_owned(),
        source: TestSource::User,
    }];
    let loaded = load_sourced_skills(&env, &inputs, &context).await;

    assert!(loaded.diagnostics.is_empty());
    assert_eq!(
        loaded.skills,
        vec![SourcedSkill {
            skill: Skill {
                name: "example".to_owned(),
                description: "Example skill".to_owned(),
                content: "Use this skill.".to_owned(),
                file_path: file_path(root.path(), "user/example/SKILL.md"),
                disable_model_invocation: Some(false),
            },
            source: TestSource::User,
        }]
    );
}

/// Attaches source info to diagnostics, upstream's "attaches source info to
/// diagnostics".
#[tokio::test]
async fn attaches_source_info_to_diagnostics() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "user/broken", &context).await;
    write(
        &env,
        "user/broken/SKILL.md",
        "---\nname: broken\n---\nMissing description.",
        &context,
    )
    .await;

    let inputs = vec![SourcedPath {
        path: "user".to_owned(),
        source: TestSource::User,
    }];
    let loaded = load_sourced_skills(&env, &inputs, &context).await;

    assert!(loaded.skills.is_empty());
    assert_eq!(
        loaded.diagnostics,
        vec![SourcedDiagnostic {
            diagnostic: SkillDiagnostic::warning(
                SkillDiagnosticCode::InvalidMetadata,
                "description is required",
                file_path(root.path(), "user/broken/SKILL.md"),
            ),
            source: TestSource::User,
        }]
    );
}

/// Loads direct markdown children only from the root directory, upstream's
/// "loads direct markdown children only from the root directory".
#[tokio::test]
async fn loads_direct_markdown_children_only_from_the_root_directory() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/nested", &context).await;
    write(
        &env,
        "skills/root.md",
        "---\ndescription: Root skill\n---\nRoot content",
        &context,
    )
    .await;
    write(
        &env,
        "skills/nested/ignored.md",
        "---\ndescription: Ignored\n---\nIgnored content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let names: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, ["skills"]);
    assert_eq!(loaded.skills[0].content, "Root content");
}

/// Ignores root markdown docs that do not declare skills, upstream's
/// "ignores root markdown docs that do not declare skills".
#[tokio::test]
async fn ignores_root_markdown_docs_that_do_not_declare_skills() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/nested-skill", &context).await;
    write(
        &env,
        "skills/README.md",
        "# Shared skills\n\nDocumentation.",
        &context,
    )
    .await;
    write(
        &env,
        "skills/AGENTS.md",
        "# Agent notes\n\nDocumentation.",
        &context,
    )
    .await;
    write(
        &env,
        "skills/CLAUDE.md",
        "---\ndescription: [invalid\n---\n\nDocumentation.",
        &context,
    )
    .await;
    write(
        &env,
        "skills/root.md",
        "---\ndescription: Root skill\n---\nRoot content",
        &context,
    )
    .await;
    write(
        &env,
        "skills/nested-skill/SKILL.md",
        "---\nname: nested-skill\ndescription: Nested skill\n---\nNested content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert!(loaded.diagnostics.is_empty());
    let mut names: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["nested-skill", "skills"]);
}

/// An ignore file at the traversal root keeps an ignored subtree out,
/// including the `SKILL.md` files inside it.
#[tokio::test]
async fn a_root_ignore_file_keeps_an_ignored_subtree_out() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/ignored/deep", &context).await;
    mkdir(&env, "skills/keep", &context).await;
    write(&env, "skills/.gitignore", "ignored/", &context).await;
    write(
        &env,
        "skills/ignored/deep/SKILL.md",
        "---\nname: ignored\ndescription: Ignored\n---\nIgnored content",
        &context,
    )
    .await;
    write(
        &env,
        "skills/keep/SKILL.md",
        "---\nname: keep\ndescription: Keep\n---\nKeep content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let names: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, ["keep"]);
    assert!(loaded.diagnostics.is_empty());
}

/// A whitelist cannot re-include under an ignored parent directory, npm
/// `ignore`'s parent-first rule.
#[tokio::test]
async fn a_whitelist_cannot_reinclude_under_an_ignored_parent() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/private/deep", &context).await;
    write(
        &env,
        "skills/.gitignore",
        "private/\n!private/deep",
        &context,
    )
    .await;
    write(
        &env,
        "skills/private/deep/SKILL.md",
        "---\nname: deep\ndescription: Deep\n---\nDeep content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert!(loaded.skills.is_empty());
}

/// A whitelist at the ignored directory itself rescues its subtree, the
/// shape npm's `!private` after `private/` produces.
#[tokio::test]
async fn a_whitelist_on_the_ignored_directory_itself_rescues_its_subtree() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/private/deep", &context).await;
    write(&env, "skills/.gitignore", "private/\n!private", &context).await;
    write(
        &env,
        "skills/private/deep/SKILL.md",
        "---\nname: deep\ndescription: Deep\n---\nDeep content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let names: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, ["deep"]);
}

/// A subdirectory's ignore file applies with the directory's relative path
/// prefixed, so its patterns anchor at that depth: they reach the subtree
/// below it but not a same-named subtree elsewhere.
#[tokio::test]
async fn a_subdirectory_ignore_file_anchors_at_its_own_depth() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/a/kill-dir", &context).await;
    mkdir(&env, "skills/deeper/kill-dir", &context).await;
    write(&env, "skills/a/.gitignore", "kill-dir/", &context).await;
    write(
        &env,
        "skills/a/kill-dir/SKILL.md",
        "---\nname: kill-dir\ndescription: Killed\n---\nKilled content",
        &context,
    )
    .await;
    write(
        &env,
        "skills/deeper/kill-dir/SKILL.md",
        "---\nname: kill-dir\ndescription: Deeper\n---\nDeeper content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let descriptions: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.description.as_str())
        .collect();
    assert_eq!(descriptions, ["Deeper"]);
    assert!(loaded.diagnostics.is_empty());
}

/// Ignore files match case-insensitively, npm `ignore`'s default.
#[tokio::test]
async fn ignore_files_match_case_insensitively() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/Build", &context).await;
    write(&env, "skills/.gitignore", "build/", &context).await;
    write(
        &env,
        "skills/Build/SKILL.md",
        "---\nname: build\ndescription: Build\n---\nBuild content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert!(loaded.skills.is_empty());
}

/// The first `SKILL.md` owns its directory: sibling root files and deeper
/// subdirectories never load, upstream's in-loop `return`.
#[tokio::test]
async fn the_first_skill_md_owns_its_directory() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/example/nested/other", &context).await;
    write(
        &env,
        "skills/example/SKILL.md",
        "---\nname: example\ndescription: Example\n---\nExample content",
        &context,
    )
    .await;
    write(
        &env,
        "skills/example/root.md",
        "---\ndescription: Root\n---\nRoot content",
        &context,
    )
    .await;
    write(
        &env,
        "skills/example/nested/other/SKILL.md",
        "---\nname: other\ndescription: Other\n---\nOther content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let names: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, ["example"]);
}

/// Missing input directories are skipped silently; a plain file input is
/// skipped silently too.
#[tokio::test]
async fn missing_and_non_directory_inputs_are_skipped_silently() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    write(&env, "plain.md", "Not a skill", &context).await;

    let dirs = vec!["missing".to_owned(), "plain.md".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert!(loaded.skills.is_empty());
    assert!(loaded.diagnostics.is_empty());
}

/// A root `.md` file whose frontmatter name differs from the parent
/// directory still loads, with the mismatch as a warning.
#[tokio::test]
async fn a_root_markdown_name_mismatch_warns_and_still_loads() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills", &context).await;
    write(
        &env,
        "skills/root.md",
        "---\nname: custom\ndescription: Custom\n---\nCustom content",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let names: Vec<&str> = loaded
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, ["custom"]);
    assert_eq!(
        loaded.diagnostics,
        vec![SkillDiagnostic::warning(
            SkillDiagnosticCode::InvalidMetadata,
            "name \"custom\" does not match parent directory \"skills\"",
            file_path(root.path(), "skills/root.md"),
        )]
    );
}

/// The name validation messages arrive in upstream's order: a declared
/// skill directory named past the limit reports the length; an uppercase
/// basename reports the characters; hyphen edges report their rules.
#[tokio::test]
async fn the_name_validation_messages_match_upstream() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    let long_name = "a".repeat(65);
    mkdir(&env, &format!("skills/{long_name}"), &context).await;
    mkdir(&env, "skills/Bad", &context).await;
    mkdir(&env, "skills/lead-hyphen", &context).await;
    write(
        &env,
        &format!("skills/{long_name}/SKILL.md"),
        "---\ndescription: Long\n---\nBody",
        &context,
    )
    .await;
    write(
        &env,
        "skills/Bad/SKILL.md",
        "---\ndescription: Bad\n---\nBody",
        &context,
    )
    .await;
    write(
        &env,
        "skills/lead-hyphen/SKILL.md",
        "---\nname: -lead\ndescription: Lead\n---\nBody",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    let messages: Vec<&str> = loaded
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect();
    assert_eq!(
        messages,
        [
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)",
            "name exceeds 64 characters (65)",
            "name \"-lead\" does not match parent directory \"lead-hyphen\"",
            "name must not start or end with a hyphen",
        ]
    );
    // Skills load despite the warnings; the leading-hyphen name was
    // rejected by nothing but its own message.
    assert_eq!(loaded.skills.len(), 3);
}

/// An empty frontmatter name falls back to the parent directory name,
/// upstream's `frontmatterName || parentDirName`.
#[tokio::test]
async fn an_empty_frontmatter_name_falls_back_to_the_parent_directory() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/example", &context).await;
    write(
        &env,
        "skills/example/SKILL.md",
        "---\nname: \"\"\ndescription: Example\n---\nBody",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert_eq!(loaded.skills[0].name, "example");
    assert!(loaded.diagnostics.is_empty());
}

/// A non-string description reads as absent, so a declared skill reports
/// the required-description diagnostic.
#[tokio::test]
async fn a_non_string_description_reads_as_absent() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/example", &context).await;
    write(
        &env,
        "skills/example/SKILL.md",
        "---\nname: example\ndescription: 42\n---\nBody",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert!(loaded.skills.is_empty());
    assert_eq!(loaded.diagnostics[0].message, "description is required");
}

/// A description past the limit warns and the skill still loads.
#[tokio::test]
async fn a_description_past_the_limit_warns_and_still_loads() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/example", &context).await;
    let description = "x".repeat(1025);
    write(
        &env,
        "skills/example/SKILL.md",
        &format!("---\nname: example\ndescription: {description}\n---\nBody"),
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert_eq!(loaded.skills.len(), 1);
    assert_eq!(
        loaded.diagnostics[0].message,
        "description exceeds 1024 characters (1025)"
    );
}

/// A string `"disable-model-invocation"` is not the boolean `true`, so the
/// skill stays model-visible.
#[tokio::test]
async fn a_string_disable_flag_keeps_the_skill_visible() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "skills/example", &context).await;
    write(
        &env,
        "skills/example/SKILL.md",
        "---\nname: example\ndescription: Example\ndisable-model-invocation: \"true\"\n---\nBody",
        &context,
    )
    .await;

    let dirs = vec!["skills".to_owned()];
    let loaded = load_skills(&env, &dirs, &context).await;

    assert_eq!(loaded.skills[0].disable_model_invocation, Some(false));
}

/// The sourced mapper converts to the application's skill type; the source
/// rides through untouched.
#[tokio::test]
async fn the_sourced_mapper_converts_to_the_application_skill_type() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct RichSkill {
        name: String,
        origin: &'static str,
    }

    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "user/example", &context).await;
    write(
        &env,
        "user/example/SKILL.md",
        "---\nname: example\ndescription: Example\n---\nBody",
        &context,
    )
    .await;

    let inputs = vec![SourcedPath {
        path: "user".to_owned(),
        source: TestSource::User,
    }];
    let loaded = load_sourced_skills_mapped(
        &env,
        &inputs,
        &|skill, source, _context| RichSkill {
            name: skill.name.clone(),
            origin: match source {
                TestSource::User => "user",
                TestSource::Project => "project",
            },
        },
        &context,
    )
    .await;

    assert_eq!(
        loaded.skills,
        vec![SourcedSkill {
            skill: RichSkill {
                name: "example".to_owned(),
                origin: "user",
            },
            source: TestSource::User,
        }]
    );
}

/// The invocation formatter names the skill and file, points relative
/// references at the file's directory, and appends instructions when given.
#[test]
fn the_invocation_formatter_names_the_skill_and_its_directory() {
    let skill = Skill {
        name: "example".to_owned(),
        description: "Example".to_owned(),
        content: "Use this skill.".to_owned(),
        file_path: "/skills/example/SKILL.md".to_owned(),
        disable_model_invocation: None,
    };
    assert_eq!(
        format_skill_invocation(&skill, None),
        "<skill name=\"example\" location=\"/skills/example/SKILL.md\">\nReferences are relative to /skills/example.\n\nUse this skill.\n</skill>"
    );
    assert_eq!(
        format_skill_invocation(&skill, Some("extra instructions")),
        "<skill name=\"example\" location=\"/skills/example/SKILL.md\">\nReferences are relative to /skills/example.\n\nUse this skill.\n</skill>\n\nextra instructions"
    );
}

/// The directory part of the invocation covers the path shapes upstream's
/// `dirnameEnvPath` handles: Windows separators, drive roots, and separator
/// free names.
#[test]
fn the_invocation_directory_covers_the_path_shapes() {
    let skill = |file_path: &str| Skill {
        name: "example".to_owned(),
        description: "Example".to_owned(),
        content: "Body".to_owned(),
        file_path: file_path.to_owned(),
        disable_model_invocation: None,
    };
    assert!(
        format_skill_invocation(&skill("C:\\skills\\SKILL.md"), None)
            .contains("References are relative to C:\\skills.")
    );
    assert!(
        format_skill_invocation(&skill("SKILL.md"), None).contains("References are relative to /.")
    );
}
