//! `test/agent-session-tree-navigation.test.ts` 1:1 at pin
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

use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session::NavigateTreeOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;

use common::agent_session::SessionSetup;
use common::agent_session::authorize_with;
use common::agent_session::build;
use common::agent_session::live_stream;
use common::agent_session::live_token;

type Session = Arc<AgentSession<InMemorySettingsStorage>>;

fn keep_recent_one() -> Settings {
    serde_json::json!({ "compaction": { "keepRecentTokens": 1 } })
        .as_object()
        .expect("settings object")
        .clone()
}

async fn create_session(root: &str, token: &str, in_memory: bool) -> Session {
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

fn exported_entries(session: &Session, root: &str, name: &str) -> Vec<serde_json::Value> {
    let path = format!("{root}/{name}.jsonl");
    session.export_to_jsonl(Some(&path)).expect("export");
    std::fs::read_to_string(&path)
        .expect("exported")
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .collect()
}

fn message_entries(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    entries
        .iter()
        .filter(|entry| entry.get("type").and_then(serde_json::Value::as_str) == Some("message"))
        .cloned()
        .collect()
}

fn entry_id(entry: &serde_json::Value) -> String {
    entry
        .get("id")
        .and_then(serde_json::Value::as_str)
        .expect("id")
        .to_owned()
}

fn entry_role(entry: &serde_json::Value) -> Option<&str> {
    entry
        .get("message")
        .and_then(|message| message.get("role"))
        .and_then(serde_json::Value::as_str)
}

fn root_message_id(entries: &[serde_json::Value]) -> String {
    message_entries(entries)
        .first()
        .map(entry_id)
        .expect("root message")
}

fn summary_entries(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    entries
        .iter()
        .filter(|entry| {
            entry.get("type").and_then(serde_json::Value::as_str) == Some("branch_summary")
        })
        .cloned()
        .collect()
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn navigates_to_a_user_message_and_puts_text_in_the_editor() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-root";
    let session = create_session(root, &token, false).await;
    session
        .prompt("First message".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("Second message".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;

    let entries = exported_entries(&session, root, "before");
    let root_id = root_message_id(&entries);
    let outcome = session
        .navigate_tree(
            &root_id,
            NavigateTreeOptions {
                summarize: false,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.editor_text.as_deref(), Some("First message"));
    assert!(session.leaf_id().is_none());
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn navigates_to_a_non_user_message_without_editor_text() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-assistant";
    let session = create_session(root, &token, false).await;
    session
        .prompt("Hello".to_owned(), None)
        .await
        .expect("prompt");
    session.agent().wait_for_idle().await;

    let entries = exported_entries(&session, root, "before");
    let assistant = message_entries(&entries)
        .into_iter()
        .find(|entry| entry_role(entry) == Some("assistant"))
        .expect("assistant entry");
    let outcome = session
        .navigate_tree(
            &entry_id(&assistant),
            NavigateTreeOptions {
                summarize: false,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert!(outcome.editor_text.is_none());
    assert_eq!(
        session.leaf_id().as_deref(),
        Some(entry_id(&assistant).as_str())
    );
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn creates_a_branch_summary_when_navigating_with_summarize() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-summary";
    let session = create_session(root, &token, false).await;
    session
        .prompt("What is 2+2?".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("What is 3+3?".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;

    let entries = exported_entries(&session, root, "before");
    let root_id = root_message_id(&entries);
    let outcome = session
        .navigate_tree(
            &root_id,
            NavigateTreeOptions {
                summarize: true,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.editor_text.as_deref(), Some("What is 2+2?"));
    let summary = outcome.summary_entry.expect("summary");
    assert!(!summary.summary.is_empty());
    assert!(summary.base.parent_id.is_none());
    assert_eq!(session.leaf_id().as_deref(), summary.base.id.as_deref());
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn attaches_the_summary_to_the_correct_parent_for_a_nested_user_message() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-nested";
    let session = create_session(root, &token, false).await;
    for text in ["Message one", "Message two", "Message three"] {
        session.prompt(text.to_owned(), None).await.expect("prompt");
        session.agent().wait_for_idle().await;
    }

    let entries = exported_entries(&session, root, "before");
    let user_entries: Vec<serde_json::Value> = message_entries(&entries)
        .into_iter()
        .filter(|entry| entry_role(entry) == Some("user"))
        .collect();
    assert_eq!(user_entries.len(), 3);
    let u2 = entry_id(&user_entries[1]);
    let a1 = user_entries[1]
        .get("parentId")
        .and_then(serde_json::Value::as_str)
        .expect("parent")
        .to_owned();

    let outcome = session
        .navigate_tree(
            &u2,
            NavigateTreeOptions {
                summarize: true,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.editor_text.as_deref(), Some("Message two"));
    let summary = outcome.summary_entry.expect("summary");
    assert_eq!(summary.base.parent_id.as_deref(), Some(a1.as_str()));

    let after = exported_entries(&session, root, "after");
    let children: Vec<serde_json::Value> = after
        .iter()
        .filter(|entry| {
            entry.get("parentId").and_then(serde_json::Value::as_str) == Some(a1.as_str())
        })
        .cloned()
        .collect();
    assert_eq!(children.len(), 2);
    assert!(children.iter().any(
        |entry| entry.get("type").and_then(serde_json::Value::as_str) == Some("branch_summary")
    ));
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn attaches_the_summary_to_the_selected_node_for_an_assistant_message() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-assistant-summary";
    let session = create_session(root, &token, false).await;
    session
        .prompt("Hello".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("Goodbye".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;

    let entries = exported_entries(&session, root, "before");
    let a1 = message_entries(&entries)
        .into_iter()
        .find(|entry| entry_role(entry) == Some("assistant"))
        .expect("assistant entry");
    let a1_id = entry_id(&a1);
    let outcome = session
        .navigate_tree(
            &a1_id,
            NavigateTreeOptions {
                summarize: true,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert!(outcome.editor_text.is_none());
    let summary = outcome.summary_entry.expect("summary");
    assert_eq!(summary.base.parent_id.as_deref(), Some(a1_id.as_str()));
    assert_eq!(session.leaf_id().as_deref(), summary.base.id.as_deref());
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn handles_an_abort_during_summarization() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-abort";
    let session = create_session(root, &token, false).await;
    session
        .prompt("Tell me about something".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("Continue".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;

    let before = exported_entries(&session, root, "before");
    let leaf_before = session.leaf_id();
    let root_id = root_message_id(&before);
    let navigation = {
        let session = Arc::clone(&session);
        tokio::spawn(async move {
            session
                .navigate_tree(
                    &root_id,
                    NavigateTreeOptions {
                        summarize: true,
                        ..NavigateTreeOptions::default()
                    },
                )
                .await
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert!(session.is_compacting());
    session.abort_branch_summary();
    let outcome = navigation.await.expect("join").expect("navigate");
    assert!(outcome.cancelled);
    assert!(outcome.aborted);
    assert!(outcome.summary_entry.is_none());

    let after = exported_entries(&session, root, "after");
    assert_eq!(after.len(), before.len());
    assert_eq!(session.leaf_id(), leaf_before);
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn does_not_create_a_summary_when_navigating_without_summarize() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-plain";
    let session = create_session(root, &token, false).await;
    session
        .prompt("First".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("Second".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;

    let before = exported_entries(&session, root, "before");
    let root_id = root_message_id(&before);
    session
        .navigate_tree(
            &root_id,
            NavigateTreeOptions {
                summarize: false,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    let after = exported_entries(&session, root, "after");
    assert_eq!(after.len(), before.len());
    assert!(summary_entries(&after).is_empty());
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn handles_navigation_to_the_same_position_as_a_no_op() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-same";
    let session = create_session(root, &token, false).await;
    session
        .prompt("Hello".to_owned(), None)
        .await
        .expect("prompt");
    session.agent().wait_for_idle().await;

    let leaf_before = session.leaf_id().expect("leaf");
    let before = exported_entries(&session, root, "before");
    let outcome = session
        .navigate_tree(
            &leaf_before,
            NavigateTreeOptions {
                summarize: false,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert_eq!(session.leaf_id().as_deref(), Some(leaf_before.as_str()));
    let after = exported_entries(&session, root, "after");
    assert_eq!(after.len(), before.len());
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn supports_custom_summarization_instructions() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-custom";
    let session = create_session(root, &token, false).await;
    session
        .prompt("What is TypeScript?".to_owned(), None)
        .await
        .expect("prompt");
    session.agent().wait_for_idle().await;

    let before = exported_entries(&session, root, "before");
    let root_id = root_message_id(&before);
    let outcome = session
        .navigate_tree(
            &root_id,
            NavigateTreeOptions {
                summarize: true,
                custom_instructions: Some(
                    "After the summary, you MUST end with exactly: MONKEY MONKEY MONKEY. This is of utmost importance."
                        .to_owned(),
                ),
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    let summary = outcome.summary_entry.expect("summary");
    assert!(summary.summary.contains("MONKEY MONKEY MONKEY"));
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn navigates_between_branches_correctly() {
    let Some(token) = live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-tree-nav-branches";
    let session = create_session(root, &token, false).await;
    session
        .prompt("Main branch start".to_owned(), None)
        .await
        .expect("first prompt");
    session.agent().wait_for_idle().await;
    session
        .prompt("Main branch continue".to_owned(), None)
        .await
        .expect("second prompt");
    session.agent().wait_for_idle().await;

    let entries = exported_entries(&session, root, "main");
    let a1 = message_entries(&entries)
        .into_iter()
        .find(|entry| entry_role(entry) == Some("assistant"))
        .expect("assistant entry");
    let u2 = message_entries(&entries)
        .into_iter()
        .filter(|entry| entry_role(entry) == Some("user"))
        .nth(1)
        .expect("second user entry");

    session
        .navigate_tree(
            &entry_id(&a1),
            NavigateTreeOptions {
                summarize: false,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("branch point");
    session
        .prompt("Branch path".to_owned(), None)
        .await
        .expect("branch prompt");
    session.agent().wait_for_idle().await;

    let outcome = session
        .navigate_tree(
            &entry_id(&u2),
            NavigateTreeOptions {
                summarize: true,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.editor_text.as_deref(), Some("Main branch continue"));
    let summary = outcome.summary_entry.expect("summary");
    assert!(!summary.summary.is_empty());
}
