//! The suites' shared fixtures: temp agent dirs with the recognized
//! artifacts, session-file writers, and the settings/auth shapes the legs
//! read.

#![expect(
    clippy::expect_used,
    reason = "test fixtures panic on their own broken setup, the test body's failure"
)]
#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique temp dir, the suite's scratch root.
#[must_use]
pub fn temp_dir(label: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "pi-import-test-{label}-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Remove a temp dir, the suite's cleanup.
pub fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Write a file under the dir, creating parents.
pub fn write(dir: &Path, relative: &str, content: &str) -> PathBuf {
    let path = dir.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(&path, content).expect("write file");
    path
}

/// A TS-pi source dir with one recognized artifact, the discovery
/// check's minimum.
#[must_use]
pub fn source_dir(label: &str) -> PathBuf {
    let dir = temp_dir(label);
    write(&dir, "settings.json", "{}\n");
    dir
}

/// A v3 session header line, the tree files' first line.
#[must_use]
pub fn session_header(id: &str, cwd: &str, version: Option<i64>) -> String {
    let version_field = version.map_or_else(String::new, |value| format!("\"version\":{value},"));
    format!(
        "{{\"type\":\"session\",{version_field}\"id\":\"{id}\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{cwd}\"}}"
    )
}

/// A message entry line, the tree files' body.
#[must_use]
pub fn message_entry(id: &str, parent: Option<&str>) -> String {
    let parent_field = parent.map_or_else(
        || "\"parentId\":null,".to_string(),
        |value| format!("\"parentId\":\"{value}\","),
    );
    format!(
        "{{\"type\":\"message\",{parent_field}\"id\":\"{id}\",\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{{\"role\":\"user\",\"content\":\"hi\"}}}}"
    )
}

/// A session file under `sessions/<encoded>/`, returning the file path.
pub fn write_session(
    agent_dir: &Path,
    encoded: &str,
    name: &str,
    header: &str,
    entries: &[String],
) -> PathBuf {
    let mut content = format!("{header}\n");
    for entry in entries {
        content.push_str(entry);
        content.push('\n');
    }
    write(agent_dir, &format!("sessions/{encoded}/{name}"), &content)
}

/// An `api_key` credential entry's JSON.
#[must_use]
pub fn api_key_entry(key: &str, env: Option<&str>) -> String {
    let env_field = env
        .map(|value| format!(",\"env\":{value}"))
        .unwrap_or_default();
    format!("{{\"type\":\"api_key\",\"key\":\"{key}\"{env_field}}}")
}

/// An `oauth` credential entry's JSON.
#[must_use]
pub fn oauth_entry(access: &str, refresh: &str, expires: &str, extra: Option<&str>) -> String {
    let extra_field = extra.map(|value| format!(",{value}")).unwrap_or_default();
    format!(
        "{{\"type\":\"oauth\",\"access\":\"{access}\",\"refresh\":\"{refresh}\",\"expires\":{expires}{extra_field}}}"
    )
}

/// A `settings.json` object's JSON string from pairs.
#[must_use]
pub fn settings_object(pairs: &[(&str, &str)]) -> String {
    let body = pairs
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{body}}}")
}
