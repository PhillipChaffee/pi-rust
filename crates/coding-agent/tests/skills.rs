//! Upstream `test/skills.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! The invalid-YAML case asserts the failure family the port reports —
//! upstream asserts npm `yaml`'s `at line` scanner message, the port
//! surfaces yaml-rust2's scanner message for the same malformed input
//! (the belt frontmatter suite records the same restatement).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::path::Path;

use pi_coding_agent::diagnostics::ResourceDiagnostic;
use pi_coding_agent::skills::{
    LoadSkillsFromDirOptions, LoadSkillsOptions, Skill, format_skills_for_prompt, load_skills,
    load_skills_from_dir,
};
use pi_coding_agent::source_info::{SyntheticSourceOptions, create_synthetic_source_info};

fn fixtures_dir() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/skills")
        .to_string_lossy()
        .into_owned()
}

fn collision_fixtures_dir() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/skills-collision")
        .to_string_lossy()
        .into_owned()
}

fn create_test_skill(options: SkillOptions) -> Skill {
    let source_info = create_synthetic_source_info(
        &options.file_path,
        &SyntheticSourceOptions {
            source: options.source.unwrap_or_else(|| "test".to_string()),
            ..SyntheticSourceOptions::default()
        },
    );
    Skill {
        name: options.name,
        description: options.description,
        file_path: options.file_path,
        base_dir: options.base_dir,
        source_info,
        disable_model_invocation: options.disable_model_invocation,
    }
}

struct SkillOptions {
    name: String,
    description: String,
    file_path: String,
    base_dir: String,
    disable_model_invocation: bool,
    source: Option<String>,
}

// === loadSkillsFromDir ======================================================

#[test]
fn loads_a_valid_skill() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/valid-skill"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "valid-skill");
    assert_eq!(
        result.skills[0].description,
        "A valid skill for testing purposes."
    );
    assert_eq!(result.skills[0].source_info.source, "test");
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn allows_names_that_dont_match_parent_directory() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/name-mismatch"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "different-name");
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d: &ResourceDiagnostic| d.message.contains("does not match parent directory"))
    );
}

#[test]
fn warns_when_name_contains_invalid_characters() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/invalid-name-chars"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("invalid characters"))
    );
}

#[test]
fn warns_when_name_exceeds_64_characters() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/long-name"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("exceeds 64 characters"))
    );
}

#[test]
fn warns_and_skips_skill_when_description_is_missing() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/missing-description"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 0);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("description is required"))
    );
}

#[test]
fn ignores_unknown_frontmatter_fields() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/unknown-field"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn loads_nested_skills_recursively() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/nested"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "child-skill");
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn prefers_a_directory_root_skill_md_over_nested_skill_md_files() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/root-skill-preferred"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "root-skill-preferred");
    assert_eq!(result.skills[0].description, "Root skill should win.");
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn skips_files_without_frontmatter() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/no-frontmatter"),
        source: "test",
    });

    // no-frontmatter has no description, so it should be skipped
    assert_eq!(result.skills.len(), 0);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("description is required"))
    );
}

#[test]
fn warns_and_skips_skill_when_yaml_frontmatter_is_invalid() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/invalid-yaml"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 0);
    // The malformed-input family the `yaml` parse throws on: the port
    // surfaces yaml-rust2's scanner message for the same input.
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("flow sequence")),
        "{:?}",
        result.diagnostics
    );
}

#[test]
fn preserves_multiline_descriptions_from_yaml() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/multiline-description"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(result.skills[0].description.contains('\n'));
    assert!(
        result.skills[0]
            .description
            .contains("This is a multiline description.")
    );
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn warns_when_name_contains_consecutive_hyphens() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/consecutive-hyphens"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("consecutive hyphens"))
    );
}

#[test]
fn loads_all_skills_from_fixture_directory() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &fixtures,
        source: "test",
    });

    // Should load all skills that have descriptions (even with warnings)
    // valid-skill, name-mismatch, invalid-name-chars, long-name,
    // unknown-field, nested/child-skill, consecutive-hyphens
    // NOT: missing-description, no-frontmatter (both missing descriptions)
    assert!(result.skills.len() >= 6);
}

#[test]
fn returns_empty_for_non_existent_directory() {
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: "/non/existent/path",
        source: "test",
    });

    assert_eq!(result.skills.len(), 0);
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn uses_parent_directory_name_when_name_not_in_frontmatter() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/valid-skill"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "valid-skill");
}

#[test]
fn parses_disable_model_invocation_frontmatter_field() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/disable-model-invocation"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "disable-model-invocation");
    assert!(result.skills[0].disable_model_invocation);
    // Should not warn about unknown field
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("unknown frontmatter field"))
    );
}

#[test]
fn defaults_disable_model_invocation_to_false_when_not_specified() {
    let fixtures = fixtures_dir();
    let result = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{fixtures}/valid-skill"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(!result.skills[0].disable_model_invocation);
}

// === formatSkillsForPrompt ==================================================

fn skill_named(name: &str, description: &str, file_path: &str, base_dir: &str) -> Skill {
    create_test_skill(SkillOptions {
        name: name.to_string(),
        description: description.to_string(),
        file_path: file_path.to_string(),
        base_dir: base_dir.to_string(),
        disable_model_invocation: false,
        source: None,
    })
}

#[test]
fn formats_no_skills_as_the_empty_string() {
    assert_eq!(format_skills_for_prompt(&[], "read"), "");
}

#[test]
fn formats_skills_as_xml() {
    let skills = vec![skill_named(
        "test-skill",
        "A test skill.",
        "/path/to/skill/SKILL.md",
        "/path/to/skill",
    )];

    let result = format_skills_for_prompt(&skills, "read");

    assert!(result.contains("<available_skills>"));
    assert!(result.contains("</available_skills>"));
    assert!(result.contains("<skill>"));
    assert!(result.contains("<name>test-skill</name>"));
    assert!(result.contains("<description>A test skill.</description>"));
    assert!(result.contains("<location>/path/to/skill/SKILL.md</location>"));
}

#[test]
fn includes_intro_text_before_xml() {
    let skills = vec![skill_named(
        "test-skill",
        "A test skill.",
        "/path/to/skill/SKILL.md",
        "/path/to/skill",
    )];

    let result = format_skills_for_prompt(&skills, "read");
    let xml_start = result.find("<available_skills>").expect("xml start");
    let intro_text = &result[..xml_start];

    assert!(intro_text.contains("The following skills provide specialized instructions"));
    assert!(intro_text.contains("Use the read tool to load a skill's file"));
}

#[test]
fn escapes_xml_special_characters() {
    let skills = vec![skill_named(
        "test-skill",
        "A skill with <special> & \"characters\".",
        "/path/to/skill/SKILL.md",
        "/path/to/skill",
    )];

    let result = format_skills_for_prompt(&skills, "read");

    assert!(result.contains("&lt;special&gt;"));
    assert!(result.contains("&amp;"));
    assert!(result.contains("&quot;characters&quot;"));
}

#[test]
fn formats_multiple_skills() {
    let skills = vec![
        skill_named(
            "skill-one",
            "First skill.",
            "/path/one/SKILL.md",
            "/path/one",
        ),
        skill_named(
            "skill-two",
            "Second skill.",
            "/path/two/SKILL.md",
            "/path/two",
        ),
    ];

    let result = format_skills_for_prompt(&skills, "read");

    assert!(result.contains("<name>skill-one</name>"));
    assert!(result.contains("<name>skill-two</name>"));
    assert_eq!(result.matches("<skill>").count(), 2);
}

#[test]
fn excludes_skills_with_disable_model_invocation_from_prompt() {
    let visible = skill_named(
        "visible-skill",
        "A visible skill.",
        "/path/visible/SKILL.md",
        "/path/visible",
    );
    let hidden_skill = Skill {
        disable_model_invocation: true,
        ..skill_named(
            "hidden-skill",
            "A hidden skill.",
            "/path/hidden/SKILL.md",
            "/path/hidden",
        )
    };
    let skills = vec![visible, hidden_skill];

    let result = format_skills_for_prompt(&skills, "read");

    assert!(result.contains("<name>visible-skill</name>"));
    assert!(!result.contains("<name>hidden-skill</name>"));
    assert_eq!(result.matches("<skill>").count(), 1);
}

#[test]
fn formats_the_empty_string_when_all_skills_have_disable_model_invocation() {
    let skills = vec![Skill {
        disable_model_invocation: true,
        ..skill_named(
            "hidden-skill",
            "A hidden skill.",
            "/path/hidden/SKILL.md",
            "/path/hidden",
        )
    }];

    assert_eq!(format_skills_for_prompt(&skills, "read"), "");
}

// === loadSkills with options ================================================

#[test]
fn loads_from_explicit_skill_paths() {
    let fixtures = fixtures_dir();
    let result = load_skills(&LoadSkillsOptions {
        agent_dir: &format!("{fixtures}/nonexistent-agent"),
        cwd: &format!("{fixtures}/nonexistent-cwd"),
        skill_paths: &[format!("{fixtures}/valid-skill")],
        include_defaults: true,
    });
    assert_eq!(result.skills.len(), 1);
    assert!(matches!(
        result.skills[0].source_info.scope,
        pi_coding_agent::source_info::SourceScope::Temporary
    ));
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn warns_when_skill_path_does_not_exist() {
    let fixtures = fixtures_dir();
    let result = load_skills(&LoadSkillsOptions {
        agent_dir: &format!("{fixtures}/nonexistent-agent"),
        cwd: &format!("{fixtures}/nonexistent-cwd"),
        skill_paths: &["/non/existent/path".to_string()],
        include_defaults: true,
    });
    assert_eq!(result.skills.len(), 0);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("does not exist"))
    );
}

#[test]
fn expands_tilde_in_skill_paths() {
    let home_skills_dir = format!(
        "{}/.pi/agent/skills",
        std::env::home_dir().unwrap_or_default().to_string_lossy()
    );
    let fixtures = fixtures_dir();
    let with_tilde = load_skills(&LoadSkillsOptions {
        agent_dir: &format!("{fixtures}/nonexistent-agent"),
        cwd: &format!("{fixtures}/nonexistent-cwd"),
        skill_paths: &["~/.pi/agent/skills".to_string()],
        include_defaults: true,
    });
    let without_tilde = load_skills(&LoadSkillsOptions {
        agent_dir: &format!("{fixtures}/nonexistent-agent"),
        cwd: &format!("{fixtures}/nonexistent-cwd"),
        skill_paths: &[home_skills_dir],
        include_defaults: true,
    });
    assert_eq!(with_tilde.skills.len(), without_tilde.skills.len());
}

// === collision handling =====================================================

#[test]
fn detects_name_collisions_and_keeps_first_skill() {
    let collision_fixtures = collision_fixtures_dir();
    // Load from first directory
    let first = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{collision_fixtures}/first"),
        source: "first",
    });

    let second = load_skills_from_dir(&LoadSkillsFromDirOptions {
        dir: &format!("{collision_fixtures}/second"),
        source: "second",
    });

    // Simulate the collision behavior from loadSkills()
    let mut skill_map: std::collections::HashMap<String, Skill> = std::collections::HashMap::new();
    let mut collision_warnings: Vec<(String, String)> = Vec::new();

    for skill in first.skills {
        skill_map.insert(skill.name.clone(), skill);
    }

    for skill in second.skills {
        if let Some(existing) = skill_map.get(&skill.name) {
            collision_warnings.push((
                skill.file_path.clone(),
                format!(
                    "name collision: \"{}\" already loaded from {}",
                    skill.name, existing.file_path
                ),
            ));
        } else {
            skill_map.insert(skill.name.clone(), skill);
        }
    }

    assert_eq!(skill_map.len(), 1);
    assert_eq!(
        skill_map
            .get("calendar")
            .map(|s| s.source_info.source.clone()),
        Some("first".to_string())
    );
    assert_eq!(collision_warnings.len(), 1);
    assert!(collision_warnings[0].1.contains("name collision"));
}
