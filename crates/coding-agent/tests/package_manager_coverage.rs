//! Coverage-closing cases for the static package-manager slice at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the pattern machinery's
//! include/exclude/force arms, the collectors' index fallbacks, symlink
//! and skip rules, the manifest entry expansion, and the object-form
//! package sources. The npm/git install arm and its suites ride #129.
//!
//! Deliberately uncovered, mirroring upstream's reachability: the stat
//! and `file_type` failure arms (only a filesystem race reaches them)
//! and `add_resource`'s empty-path guard (no collector produces an
//! empty path).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::path::PathBuf;

use pi_coding_agent::package_manager::{
    DefaultPackageManager, DefaultPackageManagerOptions, ResourceType, SkillDiscoveryMode,
    collect_auto_extension_entries, collect_auto_skill_entries, collect_resource_files,
};
use pi_coding_agent::pi_manifest::read_pi_manifest;
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};

fn manager_for(cwd: &str, agent_dir: &str) -> DefaultPackageManager {
    DefaultPackageManager::new(&DefaultPackageManagerOptions {
        cwd: cwd.to_string(),
        agent_dir: agent_dir.to_string(),
    })
}

fn in_memory_manager() -> SettingsManager<InMemorySettingsStorage> {
    SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default())
}

struct Env {
    root: PathBuf,
    agent_dir: PathBuf,
    cwd: PathBuf,
}

impl Env {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("tempdir").keep();
        let agent_dir = root.join("agent");
        let cwd = root.join("project");
        std::fs::create_dir_all(&agent_dir).expect("mkdir");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        Self {
            root,
            agent_dir,
            cwd,
        }
    }

    fn write(&self, relative: &str, content: &str) -> PathBuf {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&path, content).expect("write");
        path
    }

    fn mkdir(&self, relative: &str) -> PathBuf {
        let path = self.root.join(relative);
        std::fs::create_dir_all(&path).expect("mkdir");
        path
    }
}

// === the pattern machinery ===================================================

#[test]
fn manifest_patterns_drive_include_exclude_and_force_arms() {
    let env = Env::new();
    let package_root = env.mkdir("pkg");
    env.write("pkg/skills/a/SKILL.md", "");
    env.write("pkg/skills/b/SKILL.md", "");
    env.write("pkg/skills/c/x.md", "");
    let manifest = r#"{"name":"pkg","pi":{"skills":[
            "skills/a", "skills/b", "skills/c/x.md",
            "!skills/a/SKILL.md", "+skills/a/SKILL.md", "-skills/b"
        ]}}"#;
    env.write("pkg/package.json", manifest);

    let manager = manager_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    let mut settings = in_memory_manager();
    settings.set_packages(&[serde_json::Value::String(
        package_root.to_string_lossy().into_owned(),
    )]);
    let resolved = manager.resolve(&settings);

    // The plain parent-dir include ("skills/a"/"skills/b") matches each
    // SKILL.md through the SKILL.md parent arm; the `!` exclude removes
    // A's, the `+` force-include revives it, and the `-` force-exclude
    // drops B's from the enabled set entirely (manifest flows add only
    // enabled paths).
    let enabled = |suffix: &str| {
        resolved
            .skills
            .iter()
            .find(|resource| resource.path.ends_with(suffix))
            .map(|resource| resource.enabled)
    };
    assert_eq!(enabled("skills/a/SKILL.md"), Some(true));
    assert_eq!(enabled("skills/c/x.md"), Some(true));
    assert_eq!(enabled("skills/b/SKILL.md"), None);
}

#[test]
fn settings_override_patterns_force_include_against_excludes() {
    let env = Env::new();
    env.write("agent/skills/a/SKILL.md", "");
    env.write("agent/skills/b/SKILL.md", "");

    let manager = manager_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    let mut settings = in_memory_manager();
    settings.set_skill_paths(&["+skills/b/SKILL.md".to_string(), "!**/SKILL.md".to_string()]);
    let resolved = manager.resolve(&settings);

    // The `!` exclude disables both SKILL.md files; the `+` force-include
    // re-enables B's through the exact-match arm.
    let enabled = |suffix: &str| {
        resolved
            .skills
            .iter()
            .find(|resource| resource.path.ends_with(suffix))
            .map(|resource| resource.enabled)
    };
    assert_eq!(enabled("skills/a/SKILL.md"), Some(false));
    assert_eq!(enabled("skills/b/SKILL.md"), Some(true));
}

// === the collectors ==========================================================

#[test]
fn extension_entry_discovery_reads_indexes_and_skips_hidden_and_node_modules() {
    let env = Env::new();
    let _plain = env.write("agent/extensions/plain.ts", "");
    let _index_ts = env.write("agent/extensions/with-index/index.ts", "");
    let _index_js = env.write("agent/extensions/with-js/index.js", "");
    let _nested = env.write("agent/extensions/deep/nested.ts", "");
    env.write("agent/extensions/node_modules/junk.ts", "");
    env.write("agent/extensions/.hidden/junk.ts", "");
    let symlink_dir = env.root.join("agent/extensions/linked");
    std::os::unix::fs::symlink(env.root.join("agent/extensions/with-index"), &symlink_dir)
        .expect("symlink");

    let entries =
        collect_auto_extension_entries(env.agent_dir.join("extensions").to_string_lossy().as_ref());
    let contains = |suffix: &str| entries.iter().any(|entry| entry.ends_with(suffix));
    assert!(contains("plain.ts"), "{entries:?}");
    assert!(contains("with-index/index.ts"), "{entries:?}");
    assert!(contains("with-js/index.js"), "{entries:?}");
    assert!(contains("linked/index.ts"), "{entries:?}");
    assert!(!contains("junk.ts"), "{entries:?}");
    // A child directory contributes only its own explicit entries: a
    // plain `.ts` file below a child without an index or manifest stays
    // undiscovered.
    assert!(!contains("deep/nested.ts"), "{entries:?}");
}

#[test]
fn resource_file_walks_recurse_and_follow_symlinks() {
    let env = Env::new();
    let nested = env.write("agent/prompts/sub/inner.md", "");
    let top = env.write("agent/prompts/top.md", "");
    let symlink_file = env.root.join("agent/prompts/linked.md");
    std::os::unix::fs::symlink(env.root.join("agent/prompts/top.md"), &symlink_file)
        .expect("symlink");
    let symlink_dir = env.root.join("agent/prompts/linked-dir");
    std::os::unix::fs::symlink(env.root.join("agent/prompts/sub"), &symlink_dir).expect("symlink");

    let files = collect_resource_files(
        env.agent_dir.join("prompts").to_string_lossy().as_ref(),
        ResourceType::Prompts,
    );
    let contains = |suffix: &str| files.iter().any(|f| f.ends_with(suffix));
    assert!(contains("top.md"), "{files:?}");
    assert!(contains("sub/inner.md"), "{files:?}");
    assert!(contains("linked.md"), "{files:?}");
    assert!(contains("linked-dir/inner.md"), "{files:?}");
    let _ = (nested, top);
}

#[test]
fn skill_entry_walk_collects_skill_files_and_agents_roots() {
    let env = Env::new();
    let pi_skill = env.write("agent/skills/pi-mode/SKILL.md", "");
    let agents_skill = env.write("project/.agents/skills/sub/SKILL.md", "");

    let pi_entries = collect_auto_skill_entries(
        env.agent_dir.join("skills").to_string_lossy().as_ref(),
        SkillDiscoveryMode::Pi,
    );
    assert!(pi_entries.iter().any(|e| e.ends_with("pi-mode/SKILL.md")));

    let agents_entries = collect_auto_skill_entries(
        env.cwd.join(".agents/skills").to_string_lossy().as_ref(),
        SkillDiscoveryMode::Agents,
    );
    assert!(
        agents_entries
            .iter()
            .any(|e| e.ends_with(".agents/skills/sub/SKILL.md"))
    );
    let _ = (pi_skill, agents_skill);
}

// === package sources =========================================================

#[test]
fn non_local_package_sources_are_skipped_until_the_install_arm_lands() {
    let env = Env::new();
    let manager = manager_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    let mut settings = in_memory_manager();
    settings.set_packages(&[serde_json::Value::String("npm:left-pad".to_string())]);
    let resolved = manager.resolve(&settings);
    assert!(resolved.extensions.is_empty());
    assert!(resolved.skills.is_empty());
}

#[test]
fn object_form_package_sources_read_their_source_field() {
    let env = Env::new();
    let package_root = env.mkdir("pkg");
    env.write("pkg/skills/SKILL.md", "");

    let manager = manager_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    let mut settings = in_memory_manager();
    settings.set_packages(&[serde_json::json!({ "source": package_root.to_string_lossy() })]);
    let resolved = manager.resolve(&settings);
    assert!(
        resolved
            .skills
            .iter()
            .any(|resource| resource.path.ends_with("pkg/skills/SKILL.md"))
    );
}

#[test]
fn local_package_directories_resolve_convention_dirs_without_a_manifest() {
    let env = Env::new();
    let package_root = env.mkdir("pkg");
    env.write("pkg/skills/SKILL.md", "");
    env.write("pkg/extensions/tool.ts", "");

    let manager = manager_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    let mut settings = in_memory_manager();
    settings.set_packages(&[serde_json::Value::String(
        package_root.to_string_lossy().into_owned(),
    )]);
    let resolved = manager.resolve(&settings);
    assert!(
        resolved
            .skills
            .iter()
            .any(|resource| resource.path.ends_with("pkg/skills/SKILL.md"))
    );
    assert!(
        resolved
            .extensions
            .iter()
            .any(|resource| resource.path.ends_with("pkg/extensions/tool.ts"))
    );
}

#[test]
fn manifest_entries_expand_directories_and_drop_missing_paths() {
    let env = Env::new();
    let package_root = env.mkdir("pkg");
    env.write("pkg/skills-dir/nested/SKILL.md", "");
    env.write(
        "pkg/package.json",
        r#"{"name":"pkg","pi":{"skills":["skills-dir", "no-such-file.md"]}}"#,
    );

    let manifest = read_pi_manifest(package_root.join("package.json").to_string_lossy().as_ref())
        .expect("manifest");
    let entries = manifest.skills.expect("skills entries");
    assert_eq!(entries, vec!["skills-dir", "no-such-file.md"]);

    // The same expansion through the manager: the directory entry expands
    // to its SKILL.md, the missing path contributes nothing.
    let manager = manager_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    let mut settings = in_memory_manager();
    settings.set_packages(&[serde_json::Value::String(
        package_root.to_string_lossy().into_owned(),
    )]);
    let resolved = manager.resolve(&settings);
    let paths: Vec<String> = resolved.skills.iter().map(|r| r.path.clone()).collect();
    assert!(
        paths
            .iter()
            .any(|p| p.ends_with("skills-dir/nested/SKILL.md")),
        "{paths:?}"
    );
    assert!(
        !paths.iter().any(|p| p.contains("no-such-file")),
        "{paths:?}"
    );
}
