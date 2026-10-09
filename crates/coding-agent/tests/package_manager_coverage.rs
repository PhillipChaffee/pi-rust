//! Coverage-closing cases for the package manager at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the pattern machinery's
//! include/exclude/force arms, the collectors' index fallbacks, symlink
//! and skip rules, the manifest entry expansion, and the object-form
//! package sources.
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
use std::sync::{Arc, Mutex};

use pi_coding_agent::package_manager::{
    DefaultPackageManager, PackageManagerOptions, ResourceType, SkillDiscoveryMode,
    collect_auto_extension_entries, collect_auto_skill_entries, collect_resource_files,
};
use pi_coding_agent::pi_manifest::read_pi_manifest;
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};

type SettingsHandle = Arc<Mutex<SettingsManager<InMemorySettingsStorage>>>;

fn rig_for(
    cwd: &str,
    agent_dir: &str,
) -> (
    DefaultPackageManager<InMemorySettingsStorage>,
    SettingsHandle,
) {
    let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
        &Settings::new(),
        SettingsManagerCreateOptions::default(),
    )));
    let manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd: cwd.to_string(),
        agent_dir: agent_dir.to_string(),
        settings: Arc::clone(&settings),
        command_runner: None,
        env: None,
        http_client: None,
    });
    (manager, settings)
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

    /// Write a file carrying the execute bit, the extension-entry filter's
    /// selection (ADR 0007's executability restatement of upstream's
    /// `.ts`/`.js` file pattern).
    fn write_exec(&self, relative: &str, content: &str) -> PathBuf {
        let path = self.write(relative, content);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
        path
    }

    fn mkdir(&self, relative: &str) -> PathBuf {
        let path = self.root.join(relative);
        std::fs::create_dir_all(&path).expect("mkdir");
        path
    }
}

// === the pattern machinery ===================================================

#[tokio::test]
async fn manifest_patterns_drive_include_exclude_and_force_arms() {
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

    let (manager, settings) = rig_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String(
            package_root.to_string_lossy().into_owned(),
        )]);
    let resolved = manager.resolve(None).await.expect("resolve");

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

#[tokio::test]
async fn settings_override_patterns_force_include_against_excludes() {
    let env = Env::new();
    env.write("agent/skills/a/SKILL.md", "");
    env.write("agent/skills/b/SKILL.md", "");

    let (manager, settings) = rig_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_skill_paths(&["+skills/b/SKILL.md".to_string(), "!**/SKILL.md".to_string()]);
    let resolved = manager.resolve(None).await.expect("resolve");

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
    let _plain = env.write_exec("agent/extensions/plain", "");
    let _index = env.write_exec("agent/extensions/with-index/index", "");
    let _nested = env.write_exec("agent/extensions/deep/nested", "");
    env.write("agent/extensions/node_modules/junk", "");
    env.write("agent/extensions/.hidden/junk", "");
    let symlink_dir = env.root.join("agent/extensions/linked");
    std::os::unix::fs::symlink(env.root.join("agent/extensions/with-index"), &symlink_dir)
        .expect("symlink");

    let entries = collect_auto_extension_entries(&env.agent_dir.join("extensions"));
    let contains = |suffix: &str| entries.iter().any(|entry| entry.ends_with(suffix));
    assert!(contains("plain"), "{entries:?}");
    assert!(contains("with-index/index"), "{entries:?}");
    assert!(contains("linked/index"), "{entries:?}");
    assert!(!contains("junk"), "{entries:?}");
    // A child directory contributes only its own explicit entries: an
    // executable below a child without an index or manifest stays
    // undiscovered.
    assert!(!contains("deep/nested"), "{entries:?}");
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

    let files = collect_resource_files(&env.agent_dir.join("prompts"), ResourceType::Prompts);
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

    let pi_entries =
        collect_auto_skill_entries(&env.agent_dir.join("skills"), SkillDiscoveryMode::Pi);
    assert!(pi_entries.iter().any(|e| e.ends_with("pi-mode/SKILL.md")));

    let agents_entries =
        collect_auto_skill_entries(&env.cwd.join(".agents/skills"), SkillDiscoveryMode::Agents);
    assert!(
        agents_entries
            .iter()
            .any(|e| e.ends_with(".agents/skills/sub/SKILL.md"))
    );
    let _ = (pi_skill, agents_skill);
}

// === package sources =========================================================

#[tokio::test]
async fn npm_package_sources_are_parse_errors() {
    let env = Env::new();
    let (manager, settings) = rig_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String("npm:left-pad".to_string())]);
    // The npm channel drops (ADR 0007): a settings package carrying an
    // `npm:` source fails the resolve with the migration message instead
    // of skipping.
    let error = manager
        .resolve(None)
        .await
        .expect_err("an npm source is a parse error");
    assert!(
        error.0.contains("npm package sources are not supported"),
        "{error:?}"
    );
}

#[tokio::test]
async fn object_form_package_sources_read_their_source_field() {
    let env = Env::new();
    let package_root = env.mkdir("pkg");
    env.write("pkg/skills/SKILL.md", "");

    let (manager, settings) = rig_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::json!({ "source": package_root.to_string_lossy() })]);
    let resolved = manager.resolve(None).await.expect("resolve");
    assert!(
        resolved
            .skills
            .iter()
            .any(|resource| resource.path.ends_with("pkg/skills/SKILL.md"))
    );
}

#[tokio::test]
async fn local_package_directories_resolve_convention_dirs_without_a_manifest() {
    let env = Env::new();
    let package_root = env.mkdir("pkg");
    env.write("pkg/skills/SKILL.md", "");
    let _tool = env.write_exec("pkg/extensions/tool", "");

    let (manager, settings) = rig_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String(
            package_root.to_string_lossy().into_owned(),
        )]);
    let resolved = manager.resolve(None).await.expect("resolve");
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
            .any(|resource| resource.path.ends_with("pkg/extensions/tool"))
    );
}

#[tokio::test]
async fn manifest_entries_expand_directories_and_drop_missing_paths() {
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
    let (manager, settings) = rig_for(
        env.cwd.to_string_lossy().as_ref(),
        env.agent_dir.to_string_lossy().as_ref(),
    );
    settings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .set_packages(&[serde_json::Value::String(
            package_root.to_string_lossy().into_owned(),
        )]);
    let resolved = manager.resolve(None).await.expect("resolve");
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
