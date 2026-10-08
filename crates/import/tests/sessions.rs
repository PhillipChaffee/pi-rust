#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]
#![expect(
    clippy::panic,
    reason = "the outcome assertions panic when the tool misbehaves"
)]

//! The sessions leg's suite: in-place validation, cross-dir placement,
//! agent-root strays, and the dry-run no-write branch.

mod common;

use std::path::Path;

use common::{
    api_key_entry, cleanup, message_entry, session_header, source_dir, temp_dir, write,
    write_session,
};
use pi_import::discovery::{ImportOptions, discover_with_target};
use pi_import::report::SessionOutcome;
use pi_import::{run_import_with_target, sessions};

fn options(source: &Path) -> ImportOptions {
    ImportOptions {
        source: Some(source.to_path_buf()),
        ..ImportOptions::default()
    }
}

#[test]
fn in_place_validates_without_writing() {
    let dir = source_dir("sessions-in-place");
    let cwd = dir.join("proj");
    std::fs::create_dir_all(&cwd).unwrap();
    let encoded = pi_coding_agent::config::encode_session_cwd(&cwd.display().to_string());
    let v3 = write_session(
        &dir,
        &encoded,
        "20260101T000000-000Z_a.jsonl",
        &session_header("a", &cwd.display().to_string(), Some(3)),
        &[message_entry("m1", None)],
    );
    write_session(
        &dir,
        &encoded,
        "20260101T000000-000Z_b.jsonl",
        &session_header("b", &cwd.display().to_string(), None),
        &[],
    );
    let before = std::fs::read_to_string(&v3).unwrap();

    let discovery = discover_with_target(&options(&dir), &dir).expect("source recognized");
    assert!(discovery.in_place);
    let mut report = pi_import::report::ImportReport::default();
    let ops = sessions::run_sessions(&discovery, &mut report);

    assert!(ops.is_empty());
    assert_eq!(report.sessions.len(), 2);
    assert_eq!(report.sessions[0].outcome, SessionOutcome::Present);
    assert_eq!(report.sessions[0].version, Some(3));
    assert_eq!(report.sessions[1].version, None);
    assert!(!report.has_failures());
    assert_eq!(std::fs::read_to_string(&v3).unwrap(), before);
    cleanup(&dir);
}

#[test]
fn in_place_skips_headerless_files() {
    let dir = source_dir("sessions-headerless");
    write(
        &dir,
        "sessions/--x--/broken.jsonl",
        "{\"type\":\"message\",\"id\":\"m1\"}\n",
    );
    write(
        &dir,
        "sessions/--x--/garbage.jsonl",
        "not json\nstill not\n",
    );
    let discovery = discover_with_target(&options(&dir), &dir).expect("recognized");
    let mut report = pi_import::report::ImportReport::default();
    sessions::run_sessions(&discovery, &mut report);

    assert_eq!(report.sessions.len(), 2);
    assert_eq!(
        report.sessions[0].outcome,
        SessionOutcome::Skipped {
            reason: "no session header".to_string()
        }
    );
    assert_eq!(
        report.sessions[1].outcome,
        SessionOutcome::Skipped {
            reason: "no session header".to_string()
        }
    );
    assert!(!report.has_failures());
    cleanup(&dir);
}

#[test]
fn in_place_fails_scan_limit_files() {
    let dir = source_dir("sessions-scan-limit");
    let garbage = "not json\n".repeat(300_000);
    write(&dir, "sessions/--x--/huge.jsonl", &garbage);
    let discovery = discover_with_target(&options(&dir), &dir).expect("recognized");
    let mut report = pi_import::report::ImportReport::default();
    sessions::run_sessions(&discovery, &mut report);

    assert_eq!(report.sessions.len(), 1);
    let SessionOutcome::Failed { reason } = &report.sessions[0].outcome else {
        panic!("expected failure, got {:?}", report.sessions[0].outcome);
    };
    assert!(reason.contains("scan limit"), "reason: {reason}");
    assert!(report.has_failures());
    cleanup(&dir);
}

#[test]
fn cross_dir_copies_into_the_encoded_placement() {
    let source = source_dir("sessions-cross");
    let target = temp_dir("sessions-cross-target");
    let cwd = source.join("proj");
    std::fs::create_dir_all(&cwd).unwrap();
    let content = format!(
        "{}\n{}\n",
        session_header("a", &cwd.display().to_string(), Some(3)),
        message_entry("m1", None)
    );
    write(
        &source,
        "sessions/--s--/20260101T000000-000Z_a.jsonl",
        &content,
    );
    // An already-present copy skips.
    let encoded = pi_coding_agent::config::encode_session_cwd(&cwd.display().to_string());
    write_session(
        &target,
        &encoded,
        "20260101T000000-000Z_present.jsonl",
        &session_header("p", &cwd.display().to_string(), Some(3)),
        &[],
    );
    write(
        &source,
        "sessions/--s--/20260101T000000-000Z_present.jsonl",
        &format!(
            "{}\n",
            session_header("p", &cwd.display().to_string(), Some(3))
        ),
    );

    let options = options(&source);
    let report = run_import_with_target(&options, &target).expect("run");

    let copied = report
        .sessions
        .iter()
        .find(|item| item.path.contains("_a.jsonl"))
        .expect("copied item");
    let SessionOutcome::Copied { placed_to } = &copied.outcome else {
        panic!("expected copied, got {:?}", copied.outcome);
    };
    let placed = Path::new(placed_to);
    assert_eq!(
        placed
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy(),
        encoded
    );
    assert_eq!(std::fs::read_to_string(placed).unwrap(), content);
    let present = report
        .sessions
        .iter()
        .find(|item| item.path.contains("_present.jsonl"))
        .expect("present item");
    assert_eq!(present.outcome, SessionOutcome::Present);
    assert!(!report.has_failures());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn cross_dir_carries_agent_root_strays_and_skips_the_rest() {
    let source = source_dir("strays-cross");
    let target = temp_dir("strays-cross-target");
    let cwd = source.join("proj");
    std::fs::create_dir_all(&cwd).unwrap();
    let header = session_header("s1", &cwd.display().to_string(), Some(3));
    write(&source, "stray.jsonl", &format!("{header}\n"));
    write(&source, "noheader.jsonl", "{\"type\":\"message\"}\n");
    write(
        &source,
        "nocwd.jsonl",
        "{\"type\":\"session\",\"id\":\"x\",\"timestamp\":\"2026-01-01T00:00:00.000Z\"}\n",
    );

    let options = options(&source);
    let report = run_import_with_target(&options, &target).expect("run");

    let placed = report
        .sessions
        .iter()
        .find(|item| item.path == "stray.jsonl")
        .expect("stray item");
    let SessionOutcome::Copied { placed_to } = &placed.outcome else {
        panic!("expected copied, got {:?}", placed.outcome);
    };
    assert_eq!(
        std::fs::read_to_string(Path::new(placed_to)).unwrap(),
        format!("{header}\n")
    );
    let noheader = report
        .sessions
        .iter()
        .find(|item| item.path == "noheader.jsonl")
        .expect("noheader item");
    assert_eq!(
        noheader.outcome,
        SessionOutcome::Skipped {
            reason: "no session header".to_string()
        }
    );
    let nocwd = report
        .sessions
        .iter()
        .find(|item| item.path == "nocwd.jsonl")
        .expect("nocwd item");
    assert_eq!(
        nocwd.outcome,
        SessionOutcome::Skipped {
            reason: "session header has no cwd".to_string()
        }
    );
    // The source strays stay put cross-dir.
    assert!(source.join("stray.jsonl").exists());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn in_place_moves_agent_root_strays() {
    let dir = source_dir("strays-in-place");
    let cwd = dir.join("proj");
    std::fs::create_dir_all(&cwd).unwrap();
    let header = session_header("s1", &cwd.display().to_string(), Some(3));
    write(&dir, "stray.jsonl", &format!("{header}\n"));
    // An existing placement target skips the move.
    let encoded = pi_coding_agent::config::encode_session_cwd(&cwd.display().to_string());
    write_session(
        &dir,
        &encoded,
        "taken.jsonl",
        &session_header("s2", &cwd.display().to_string(), Some(3)),
        &[],
    );
    write(
        &dir,
        "taken.jsonl",
        &format!(
            "{}\n",
            session_header("s2", &cwd.display().to_string(), Some(3))
        ),
    );

    let options = options(&dir);
    let report = run_import_with_target(&options, &dir).expect("run");

    let moved = report
        .sessions
        .iter()
        .find(|item| item.path == "stray.jsonl")
        .expect("stray item");
    let SessionOutcome::Copied { placed_to } = &moved.outcome else {
        panic!("expected placed, got {:?}", moved.outcome);
    };
    assert!(!dir.join("stray.jsonl").exists());
    assert!(Path::new(placed_to).exists());
    let taken = report
        .sessions
        .iter()
        .find(|item| item.path == "taken.jsonl")
        .expect("taken item");
    assert_eq!(taken.outcome, SessionOutcome::Present);
    assert!(dir.join("taken.jsonl").exists());
    cleanup(&dir);
}

#[test]
fn dry_run_plays_sessions_without_writing() {
    let source = source_dir("sessions-dry");
    let target = temp_dir("sessions-dry-target");
    let cwd = source.join("proj");
    std::fs::create_dir_all(&cwd).unwrap();
    let content = format!(
        "{}\n",
        session_header("a", &cwd.display().to_string(), Some(3))
    );
    write(
        &source,
        "sessions/--s--/20260101T000000-000Z_a.jsonl",
        &content,
    );
    write(&source, "auth.json", &api_key_entry("sk-source", None));

    let options = ImportOptions {
        source: Some(source.clone()),
        dry_run: true,
        ..ImportOptions::default()
    };
    let report = run_import_with_target(&options, &target).expect("run");

    assert!(report.dry_run);
    let copied = report
        .sessions
        .iter()
        .find(|item| item.path.ends_with("_a.jsonl"))
        .expect("copied item");
    assert!(matches!(copied.outcome, SessionOutcome::Copied { .. }));
    assert!(!target.join("sessions").exists());
    assert_eq!(
        std::fs::read_to_string(source.join("sessions/--s--/20260101T000000-000Z_a.jsonl"))
            .unwrap(),
        content
    );
    assert_eq!(
        std::fs::read_to_string(source.join("auth.json")).unwrap(),
        api_key_entry("sk-source", None)
    );
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn scan_limit_strays_fail_the_leg() {
    let dir = source_dir("strays-scan-limit");
    write(&dir, "huge.jsonl", &"not json\n".repeat(300_000));
    let discovery = discover_with_target(&options(&dir), &dir).expect("recognized");
    let mut report = pi_import::report::ImportReport::default();
    sessions::run_sessions(&discovery, &mut report);

    assert_eq!(report.sessions.len(), 1);
    assert!(matches!(
        report.sessions[0].outcome,
        SessionOutcome::Failed { .. }
    ));
    assert!(report.has_failures());
    cleanup(&dir);
}

#[test]
fn relative_paths_outside_the_base_carry_their_full_form() {
    let base = source_dir("relative-base");
    let outside = temp_dir("relative-outside").join("elsewhere.jsonl");
    assert_eq!(
        sessions::relative_path(&base, &outside),
        outside.display().to_string()
    );
    let inside = base.join("sessions/x.jsonl");
    assert_eq!(sessions::relative_path(&base, &inside), "sessions/x.jsonl");
    cleanup(&base);
}

#[test]
fn unreadable_directories_skip_their_entries() {
    use std::os::unix::fs::PermissionsExt;

    let dir = source_dir("unreadable-dirs");
    let cwd = dir.join("proj");
    std::fs::create_dir_all(&cwd).unwrap();
    write_session(
        &dir,
        "--hidden--",
        "20260101T000000-000Z_h.jsonl",
        &session_header("h", &cwd.display().to_string(), Some(3)),
        &[],
    );
    let subdir = dir.join("sessions/--hidden--");
    let mut perms = std::fs::metadata(&subdir).unwrap().permissions();
    perms.set_mode(0o000);
    std::fs::set_permissions(&subdir, perms).unwrap();

    let discovery = discover_with_target(&options(&dir), &dir).expect("recognized");
    let mut report = pi_import::report::ImportReport::default();
    sessions::run_sessions(&discovery, &mut report);

    // The unreadable subdir's files never list; no items, no failures.
    assert!(report.sessions.is_empty());
    assert!(!report.has_failures());

    let mut perms = std::fs::metadata(&subdir).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&subdir, perms).unwrap();
    cleanup(&dir);
}
