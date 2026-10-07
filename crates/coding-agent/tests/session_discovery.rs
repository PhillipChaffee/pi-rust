//! Session discovery and listing at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the #7497 symlink regression
//! ported 1:1, the modified-timestamp suite ported 1:1, and the cwd-filter
//! and continue-recent boundaries.
//!
//! The `PI_CODING_AGENT_DIR` stubs upstream drives with `vi.stubEnv` run as
//! probe children instead — `set_var` is forbidden in this workspace, so the
//! parent composes the child's environment at spawn time (the same pattern
//! the auth-core suite uses).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use std::fs;
use std::os::unix::fs::symlink;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    AssistantBlock, AssistantMessage, KnownApi, Message, ProviderId, StopReason, Usage, UsageCost,
};

use pi_coding_agent::session_manager::SessionManager;

/// The probe key the child-process runs key their scenario on.
const ENV_PROBE: &str = "PI_CODING_AGENT_DISCOVERY_PROBE";
/// The fixture root the parent hands the probe child.
const ENV_TEMP: &str = "PI_CODING_AGENT_DISCOVERY_TEMP";

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_millis()
        .try_into()
        .expect("epoch millis fit i64")
}

fn assistant_message(text: &str, timestamp: i64) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: vec![AssistantBlock::Text(pi_ai::types::TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: KnownApi::OpenaiCompletions.into(),
        provider: ProviderId("openai".to_owned()),
        model: "test".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 2,
            cost: UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp,
    }))
}

fn write_session(dir: &str, id: &str, cwd: &str) -> String {
    fs::create_dir_all(dir).expect("dir");
    let path = format!("{dir}/{id}.jsonl");
    fs::write(
        &path,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"{id}\",\"timestamp\":\"2026-08-03T00:00:00.000Z\",\"cwd\":\"{cwd}\"}}\n"
        ),
    )
    .expect("write session");
    path
}

/// Run this suite's own binary as a child with `PI_CODING_AGENT_DIR` pointed
/// at the fixture's agent dir, the `vi.stubEnv` stand-in.
fn probe_child(mode: &str, temp: &str) {
    let mut command =
        std::process::Command::new(std::env::current_exe().expect("the test binary path"));
    command
        .args([
            "--exact",
            "the_discovery_probe",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ENV_PROBE, mode)
        .env(ENV_TEMP, temp)
        .env("PI_CODING_AGENT_DIR", format!("{temp}/agent"));
    let output = command.output().expect("the probe child runs");
    assert!(
        output.status.success(),
        "the {mode:?} probe child passes: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// Regression #7497: discover sessions through symlinked directories.
// ---------------------------------------------------------------------------

#[test]
fn discovers_a_session_through_a_directory_link_and_preserves_the_alias_path() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let sessions_dir = format!("{temp}/agent/sessions");
    fs::create_dir_all(&sessions_dir).expect("sessions dir");

    let target_dir = format!("{temp}/linked-sessions");
    write_session(&target_dir, "linked", &format!("{temp}/project"));
    let alias_dir = format!("{sessions_dir}/--linked--");
    symlink(&target_dir, &alias_dir).expect("symlink");

    probe_child("alias-path", &temp);
}

#[test]
fn ignores_a_broken_directory_link_without_hiding_valid_sessions() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let sessions_dir = format!("{temp}/agent/sessions");
    fs::create_dir_all(&sessions_dir).expect("sessions dir");

    write_session(
        &format!("{sessions_dir}/--regular--"),
        "regular",
        &format!("{temp}/project"),
    );
    let target_dir = format!("{temp}/removed-sessions");
    fs::create_dir(&target_dir).expect("target");
    symlink(&target_dir, format!("{sessions_dir}/--broken--")).expect("symlink");
    fs::remove_dir_all(&target_dir).expect("remove target");

    probe_child("broken-link", &temp);
}

#[test]
fn ignores_links_to_files() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let sessions_dir = format!("{temp}/agent/sessions");
    fs::create_dir_all(&sessions_dir).expect("sessions dir");

    write_session(
        &format!("{sessions_dir}/--regular--"),
        "regular",
        &format!("{temp}/project"),
    );
    let target_file = format!("{temp}/not-a-directory");
    fs::write(&target_file, "").expect("file");
    symlink(&target_file, format!("{sessions_dir}/--file--")).expect("symlink");

    probe_child("file-link", &temp);
}

/// The probe child: each mode runs one env-dependent discovery scenario with
/// `PI_CODING_AGENT_DIR` already in the environment.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "each mode is one self-contained upstream scenario; splitting the probe would scatter the fixtures"
)]
fn the_discovery_probe() {
    let Ok(mode) = std::env::var(ENV_PROBE) else {
        return;
    };
    let temp = std::env::var(ENV_TEMP).expect("the fixture root");
    let run = tokio_block_on(SessionManager::list_all(None));

    match mode.as_str() {
        "alias-path" => {
            assert_eq!(
                run.iter()
                    .map(|session| session.id.as_str())
                    .collect::<Vec<_>>(),
                ["linked"]
            );
            assert_eq!(
                run[0].path,
                format!("{temp}/agent/sessions/--linked--/linked.jsonl"),
                "the alias path is preserved"
            );
        }
        "broken-link" | "file-link" => {
            assert_eq!(
                run.iter()
                    .map(|session| session.id.as_str())
                    .collect::<Vec<_>>(),
                ["regular"]
            );
        }
        "progress" => {
            let project = format!("{temp}/project");
            write_session(
                &pi_coding_agent::session_manager::get_default_session_dir(&project)
                    .expect("dir")
                    .display()
                    .to_string(),
                "seeded",
                &project,
            );
            let progress: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
            let sessions = tokio_block_on(SessionManager::list(
                &project,
                None,
                Some(&|loaded, total| {
                    progress.lock().expect("progress").push((loaded, total));
                }),
            ))
            .expect("list");
            assert_eq!(
                sessions
                    .iter()
                    .map(|session| session.id.as_str())
                    .collect::<Vec<_>>(),
                ["seeded"]
            );
            assert_eq!(
                *progress.into_inner().expect("progress"),
                vec![(1, 1)],
                "one file, one progress tick"
            );
        }
        "create-default" => {
            let project = format!("{temp}/project");
            let session =
                SessionManager::create(&project, None, None).expect("create in the default dir");
            let file = session
                .session_file()
                .expect("the created session persists")
                .to_owned();
            assert!(
                file.starts_with(&format!("{temp}/agent/sessions/")),
                "{file}"
            );
        }
        "continue-default" => {
            let project = format!("{temp}/project");
            let mut created = SessionManager::create(&project, None, None).expect("create");
            // The buffer stays in memory until the first assistant message,
            // so the file only exists on disk after the flush gate.
            created
                .append_message(assistant_message("flush", now_millis()))
                .expect("append");
            let continued = SessionManager::continue_recent(&project, None).expect("continue");
            assert_eq!(
                continued.session_file(),
                created.session_file(),
                "the recent session reopens"
            );
        }
        "list-all-progress" => {
            let project = format!("{temp}/project");
            write_session(
                &pi_coding_agent::session_manager::get_default_session_dir(&project)
                    .expect("dir")
                    .display()
                    .to_string(),
                "seeded",
                &project,
            );
            let progress: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
            let sessions = tokio_block_on(SessionManager::list_all(Some(&|loaded, total| {
                progress.lock().expect("progress").push((loaded, total));
            })));
            assert_eq!(
                sessions
                    .iter()
                    .map(|session| session.id.as_str())
                    .collect::<Vec<_>>(),
                ["seeded"]
            );
            assert_eq!(*progress.into_inner().expect("progress"), vec![(1, 1)]);
        }
        "list-all-file" => {
            let sessions_dir = format!("{temp}/agent/sessions");
            fs::create_dir_all(format!("{temp}/agent")).expect("agent dir");
            // A sibling probe mode may have created the directory already;
            // the unreadable root stands in for upstream's stat race.
            if fs::exists(&sessions_dir).unwrap_or(false) {
                fs::remove_dir_all(&sessions_dir).expect("remove sessions dir");
            }
            fs::write(&sessions_dir, "a file, not the sessions directory").expect("write file");
            assert!(
                tokio_block_on(SessionManager::list_all(None)).is_empty(),
                "an unreadable sessions directory lists nothing"
            );
        }
        "fork-default" => {
            let source = format!("{temp}/source.jsonl");
            fs::write(
                &source,
                "{\"type\":\"session\",\"version\":3,\"id\":\"src\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"/tmp\"}\n",
            )
            .expect("write source");
            let project = format!("{temp}/project");
            let forked = SessionManager::fork_from(&source, &project, None, None).expect("fork");
            let dir = forked.session_dir().to_owned();
            assert!(
                dir.starts_with(&format!("{temp}/agent/sessions/")),
                "the fork lands in the default sessions directory: {dir}"
            );
            assert!(
                fs::exists(&dir).unwrap_or(false),
                "the default directory is created"
            );
        }
        "create-default-fails" => {
            let sessions_dir = format!("{temp}/agent/sessions");
            fs::create_dir_all(format!("{temp}/agent")).expect("agent dir");
            if fs::exists(&sessions_dir).unwrap_or(false) {
                fs::remove_dir_all(&sessions_dir).expect("remove sessions dir");
            }
            fs::write(&sessions_dir, "a file, not the sessions directory").expect("write file");
            let project = format!("{temp}/project");
            let error = SessionManager::create(&project, None, None)
                .expect_err("the default directory cannot be created");
            assert!(
                matches!(
                    error,
                    pi_coding_agent::session_manager::SessionManagerError::Io(_)
                ),
                "{error}"
            );
        }
        other => panic!("unknown probe mode: {other}"),
    }
}

#[test]
fn list_creates_the_default_directory_and_reports_progress() {
    let dir = tempfile::tempdir().expect("temp dir");
    probe_child("progress", &dir.path().display().to_string());
}

// ---------------------------------------------------------------------------
// The default-directory lifecycle arms: create/continueRecent without a
// directory override, and listAll's progress and unreadable-root arms.
// ---------------------------------------------------------------------------

#[test]
fn create_and_continue_recent_fall_back_to_the_default_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    probe_child("create-default", &temp);
    probe_child("continue-default", &temp);
    probe_child("fork-default", &temp);
}

#[test]
fn create_fails_when_the_default_directory_cannot_be_created() {
    let dir = tempfile::tempdir().expect("temp dir");
    probe_child("create-default-fails", &dir.path().display().to_string());
}

#[test]
fn list_all_reports_progress_and_survives_an_unreadable_sessions_root() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    probe_child("list-all-progress", &temp);
    probe_child("list-all-file", &temp);
}

// ---------------------------------------------------------------------------
// SessionInfo.modified: the last user/assistant timestamp, not the mtime.
// ---------------------------------------------------------------------------

#[test]
fn uses_last_message_timestamp_instead_of_file_mtime() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let file_path = format!("{temp}/session.jsonl");
    fs::write(
        &file_path,
        "{\"type\":\"session\",\"id\":\"test-session\",\"version\":3,\"timestamp\":\"1970-01-01T00:00:00.000Z\",\"cwd\":\"/tmp\"}\n",
    )
    .expect("write");

    // SessionManager only persists once it has seen at least one assistant
    // message. Add a minimal assistant entry so subsequent appends persist.
    {
        let mut setup = SessionManager::open(&file_path, None, None).expect("open");
        setup
            .append_message(assistant_message("hi", now_millis()))
            .expect("append");
    }

    let before_mtime = fs::metadata(&file_path)
        .expect("stat")
        .modified()
        .expect("mtime");
    std::thread::sleep(std::time::Duration::from_millis(15));

    let mut session = SessionManager::open(&file_path, None, None).expect("open");
    let msg_time = now_millis();
    session
        .append_message(assistant_message("later", msg_time))
        .expect("append");

    let sessions = tokio_block_on(SessionManager::list("/tmp", Some(&temp), None)).expect("list");
    let info = sessions
        .iter()
        .find(|session| session.path == file_path)
        .expect("found");
    assert_eq!(info.modified, msg_time, "the last message timestamp wins");
    let mtime: i64 = before_mtime
        .duration_since(UNIX_EPOCH)
        .expect("epoch")
        .as_millis()
        .try_into()
        .expect("epoch millis fit i64");
    assert_ne!(info.modified, mtime, "it is not the file mtime");
}

// ---------------------------------------------------------------------------
// Discovery boundaries: cwd filters, most-recent, continue-recent.
// ---------------------------------------------------------------------------

#[test]
fn list_filters_by_cwd_when_the_directory_differs_from_the_default() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let project_a = format!("{temp}/a");
    let project_b = format!("{temp}/b");
    let sessions_dir = format!("{temp}/sessions");
    write_session(&sessions_dir, "in-a", &project_a);
    write_session(&sessions_dir, "in-b", &project_b);

    let sessions =
        tokio_block_on(SessionManager::list(&project_a, Some(&sessions_dir), None)).expect("list");
    assert_eq!(
        sessions
            .iter()
            .map(|session| session.id.as_str())
            .collect::<Vec<_>>(),
        ["in-a"]
    );

    let all = tokio_block_on(SessionManager::list_all_from_dir(&sessions_dir, None)).expect("all");
    assert_eq!(all.len(), 2, "the override form does not filter by cwd");
}

#[test]
fn find_most_recent_session_filters_by_cwd_and_picks_the_newest_mtime() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let project_a = format!("{temp}/a");
    let project_b = format!("{temp}/b");
    let sessions_dir = format!("{temp}/sessions");
    let _older = write_session(&sessions_dir, "older-a", &project_a);
    std::thread::sleep(std::time::Duration::from_millis(20));
    let newer = write_session(&sessions_dir, "newer-a", &project_a);
    let _other = write_session(&sessions_dir, "other-b", &project_b);

    assert_eq!(
        pi_coding_agent::session_manager::find_most_recent_session(&sessions_dir, Some(&project_a)),
        Some(newer)
    );
    assert!(
        pi_coding_agent::session_manager::find_most_recent_session(&sessions_dir, None).is_some(),
        "without a cwd filter the newest mtime wins regardless of project"
    );
    assert_eq!(
        pi_coding_agent::session_manager::find_most_recent_session(
            &format!("{sessions_dir}/missing"),
            None
        ),
        None
    );
}

#[test]
fn continue_recent_reopens_the_newest_session_or_creates_fresh() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let project = format!("{temp}/project");
    let sessions_dir = format!("{temp}/sessions");
    let mut created = SessionManager::create(&project, Some(&sessions_dir), None).expect("create");
    created
        .append_message(assistant_message("hi", now_millis()))
        .expect("append");
    std::thread::sleep(std::time::Duration::from_millis(20));

    let continued =
        SessionManager::continue_recent(&project, Some(&sessions_dir)).expect("continue");
    assert_eq!(
        continued.session_file(),
        created.session_file(),
        "the most recent session reopens"
    );

    let fresh = SessionManager::continue_recent(&project, Some(&format!("{temp}/empty")))
        .expect("continue");
    assert_ne!(
        fresh.session_id(),
        continued.session_id(),
        "no prior session means a fresh one"
    );
}

/// The single-threaded runtime the async discovery calls need.
fn tokio_block_on<T>(future: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(future)
}
