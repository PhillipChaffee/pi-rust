//! `test/agent-session-compaction.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream gates the suite on `API_KEY`; the port keeps that gate as
//! `#[ignore]` so a credentialed shell opts in with `--ignored`.

#![expect(
    clippy::expect_used,
    clippy::print_stderr,
    reason = "the tests pin outcomes and log their credential skip the way upstream does"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use pi_agent_core::types::AgentMessage;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::Settings;

use common::agent_session::SessionSetup;
use common::agent_session::authorize_with;
use common::agent_session::build;
use common::agent_session::live_stream;
use common::agent_session::live_token;

fn keep_recent_one() -> Settings {
    serde_json::json!({ "compaction": { "keepRecentTokens": 1 } })
        .as_object()
        .expect("settings object")
        .clone()
}

async fn create_session(
    root: &str,
    in_memory: bool,
    token: &str,
) -> Arc<
    pi_coding_agent::core::agent_session::AgentSession<
        pi_coding_agent::settings_manager::InMemorySettingsStorage,
    >,
> {
    std::fs::create_dir_all(root).expect("root");
    let manager = if in_memory {
        SessionManager::in_memory(Some(root), None, None).expect("session")
    } else {
        SessionManager::create(root, Some(&format!("{root}/sessions")), None).expect("session")
    };
    let mut setup = SessionSetup::new(root, manager);
    setup.settings = keep_recent_one();
    setup.stream = Some(live_stream(token.to_owned()));
    let session = build(setup).await;
    authorize_with(&session, token).await;
    session
}

fn message_roles(
    session: &pi_coding_agent::core::agent_session::AgentSession<
        pi_coding_agent::settings_manager::InMemorySettingsStorage,
    >,
) -> Vec<String> {
    session
        .agent()
        .state()
        .messages
        .iter()
        .map(|message| match message {
            AgentMessage::Standard(pi_ai::types::Message::User(_)) => "user".to_owned(),
            AgentMessage::Standard(pi_ai::types::Message::Assistant(_)) => "assistant".to_owned(),
            AgentMessage::Standard(pi_ai::types::Message::ToolResult(_)) => "toolResult".to_owned(),
            AgentMessage::Custom(custom) => custom.role.clone(),
        })
        .collect()
}

fn exported_entries(path: &str) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .expect("export")
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .collect()
}

fn compaction_entry_count(
    session: &pi_coding_agent::core::agent_session::AgentSession<
        pi_coding_agent::settings_manager::InMemorySettingsStorage,
    >,
    root: &str,
) -> usize {
    let path = format!("{root}/export.jsonl");
    session.export_to_jsonl(Some(&path)).expect("export");
    exported_entries(&path)
        .into_iter()
        .filter(|entry| entry.get("type").and_then(serde_json::Value::as_str) == Some("compaction"))
        .count()
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn triggers_manual_compaction_via_compact() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-compaction-e2e";
    let session = create_session(root, false, &token).await;

    session
        .prompt("What is 2+2? Reply with just the number.".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("What is 3+3? Reply with just the number.".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;

    let result = session.compact(None).await.expect("compact");
    assert!(!result.summary.is_empty());
    assert!(result.tokens_before > 0);
    assert_eq!(
        message_roles(&session).first().map(String::as_str),
        Some("compactionSummary")
    );
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn maintains_valid_session_state_after_compaction() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-compaction-e2e-state";
    let session = create_session(root, false, &token).await;
    session
        .prompt(
            "What is the capital of France? One word answer.".to_owned(),
            None,
        )
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt(
            "What is the capital of Germany? One word answer.".to_owned(),
            None,
        )
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;
    session.compact(None).await.expect("compact");
    session
        .prompt(
            "What is the capital of Italy? One word answer.".to_owned(),
            None,
        )
        .await
        .expect("third prompt");
    session.agent().wait_for_idle().await;

    let roles = message_roles(&session);
    assert!(!roles.is_empty());
    assert!(roles.iter().any(|role| role == "assistant"));
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn persists_compaction_to_the_session_file() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-compaction-e2e-persist";
    let session = create_session(root, false, &token).await;
    session
        .prompt("Say hello".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("Say goodbye".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;
    session.compact(None).await.expect("compact");

    assert_eq!(compaction_entry_count(&session, root), 1);
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn works_with_no_session_mode() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-compaction-e2e-memory";
    let session = create_session(root, true, &token).await;
    session
        .prompt("What is 2+2? Reply with just the number.".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("What is 3+3? Reply with just the number.".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;
    let result = session.compact(None).await.expect("compact");
    assert!(!result.summary.is_empty());
    assert_eq!(compaction_entry_count(&session, root), 1);
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn emits_compaction_events_during_manual_compaction() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-compaction-e2e-events";
    let session = create_session(root, false, &token).await;
    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        let label = match event {
            AgentSessionEvent::CompactionStart { reason } => format!("start:{reason:?}"),
            AgentSessionEvent::CompactionEnd {
                aborted,
                will_retry,
                ..
            } => {
                format!("end:{aborted}:{will_retry}")
            }
            AgentSessionEvent::MessageEnd { .. } => "message_end".to_owned(),
            _ => return,
        };
        sink.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(label);
    }));

    session
        .prompt("Say hello".to_owned(), None)
        .await
        .expect("prompt");
    session.agent().wait_for_idle().await;
    session.compact(None).await.expect("compact");

    let events = events
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let compaction_events: Vec<&String> = events
        .iter()
        .filter(|event| event.starts_with("start:") || event.starts_with("end:"))
        .collect();
    assert_eq!(compaction_events.len(), 2);
    assert_eq!(compaction_events[0], "start:Manual");
    assert_eq!(compaction_events[1], "end:false:false");
    assert!(events.iter().any(|event| event == "message_end"));
}
