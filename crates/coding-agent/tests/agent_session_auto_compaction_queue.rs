//! `test/agent-session-auto-compaction-queue.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    clippy::type_complexity,
    reason = "the tests pin outcomes and name their event-collector shapes inline"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::CustomAgentMessage;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::reasons::CompactionReason;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::Settings;

use common::agent_session::SessionSetup;
use common::agent_session::anthropic_model;
use common::agent_session::assistant_with_usage;
use common::agent_session::authorize;
use common::agent_session::build;
use common::agent_session::error_message;
use common::agent_session::text_stream;
use common::agent_session::usage;
use common::agent_session::user_message;

fn settings_with_keep_recent_tokens(tokens: i64) -> Settings {
    serde_json::json!({ "compaction": { "keepRecentTokens": tokens } })
        .as_object()
        .expect("settings object")
        .clone()
}

fn in_memory_manager(root: &str) -> SessionManager {
    SessionManager::in_memory(Some(root), None, None).expect("session")
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(0)
        })
}

fn queued_custom_message(text: &str) -> AgentMessage {
    let mut data = serde_json::Map::new();
    data.insert(
        "customType".to_owned(),
        serde_json::Value::String("test".to_owned()),
    );
    data.insert(
        "content".to_owned(),
        serde_json::json!([{ "type": "text", "text": text }]),
    );
    data.insert("display".to_owned(), serde_json::Value::Bool(false));
    AgentMessage::Custom(CustomAgentMessage {
        role: "custom".to_owned(),
        timestamp: now_ms(),
        data,
    })
}

fn compaction_events() -> (
    Arc<Mutex<Vec<(bool, Option<String>)>>>,
    Arc<dyn Fn(AgentSessionEvent) + Send + Sync>,
) {
    let events: Arc<Mutex<Vec<(bool, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let listener: Arc<dyn Fn(AgentSessionEvent) + Send + Sync> = Arc::new(move |event| {
        if let AgentSessionEvent::CompactionEnd {
            aborted,
            error_message,
            ..
        } = event
        {
            sink.lock().expect("lock").push((aborted, error_message));
        }
    });
    (events, listener)
}

fn compaction_starts() -> (
    Arc<Mutex<Vec<CompactionReason>>>,
    Arc<dyn Fn(AgentSessionEvent) + Send + Sync>,
) {
    let reasons: Arc<Mutex<Vec<CompactionReason>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&reasons);
    let listener: Arc<dyn Fn(AgentSessionEvent) + Send + Sync> = Arc::new(move |event| {
        if let AgentSessionEvent::CompactionStart { reason } = event {
            sink.lock().expect("lock").push(reason);
        }
    });
    (reasons, listener)
}

#[tokio::test]
async fn resumes_after_threshold_compaction_when_only_agent_level_queued_messages_exist() {
    let root = "/tmp/pi-session-auto-compaction-queue-resume";
    let now = now_ms();
    let mut manager = in_memory_manager(root);
    manager
        .append_message(user_message("message to compact", now - 1000))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(pi_ai::types::Message::Assistant(
            assistant_with_usage("assistant response to compact", usage(100, 0), now - 500),
        )))
        .expect("append");
    let messages = manager.build_session_context().messages;
    let mut setup = SessionSetup::new(root, manager);
    setup.settings = settings_with_keep_recent_tokens(1);
    setup.messages = messages;
    setup.stream = Some(text_stream("compacted"));
    let session = build(setup).await;
    authorize(&session).await;

    session
        .agent()
        .follow_up(queued_custom_message("Queued custom"));
    assert_eq!(session.pending_message_count(), 0);
    assert!(session.agent().has_queued_messages());

    let resumed = session
        .run_auto_compaction(CompactionReason::Threshold, false)
        .await;
    assert!(resumed);
}

#[tokio::test]
async fn does_not_compact_repeatedly_after_overflow_recovery_was_already_attempted() {
    let root = "/tmp/pi-session-auto-compaction-overflow";
    let session = build(SessionSetup::new(root, in_memory_manager(root))).await;
    authorize(&session).await;
    let (events, listener) = compaction_events();
    let _unsubscribe = session.subscribe(listener);

    let overflow = error_message("prompt is too long", now_ms());
    assert!(!session.check_compaction(&overflow, false).await);
    assert!(!session.check_compaction(&overflow, false).await);

    assert_eq!(
        events.lock().expect("lock").clone(),
        vec![(
            false,
            Some(
                "Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model."
                    .to_owned()
            )
        )]
    );
}

#[tokio::test]
async fn ignores_stale_pre_compaction_assistant_usage_on_pre_prompt_checks() {
    let root = "/tmp/pi-session-auto-compaction-stale";
    let timestamp = now_ms() - 10_000;
    let stale = assistant_with_usage(
        "large response before compaction",
        usage(600_000, 10_000),
        timestamp,
    );
    let mut manager = in_memory_manager(root);
    manager
        .append_message(user_message("before compaction", timestamp - 1000))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(pi_ai::types::Message::Assistant(
            stale.clone(),
        )))
        .expect("append");
    let first_kept_entry_id = manager
        .entries()
        .iter()
        .find_map(|entry| entry.entry_id().map(str::to_owned))
        .expect("entry id");
    manager
        .append_compaction("summary", &first_kept_entry_id, 610_000, None, None, None)
        .expect("compaction");
    manager
        .append_message(user_message("session recovery payload", now_ms()))
        .expect("append");
    let messages = manager.build_session_context().messages;
    let mut setup = SessionSetup::new(root, manager);
    setup.messages = messages;
    let session = build(setup).await;
    authorize(&session).await;
    let (starts, listener) = compaction_starts();
    let _unsubscribe = session.subscribe(listener);

    assert!(!session.check_compaction(&stale, false).await);
    assert!(starts.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn triggers_threshold_compaction_for_error_messages_using_the_last_successful_usage() {
    let root = "/tmp/pi-session-auto-compaction-usage";
    let model = anthropic_model();
    let threshold_tokens =
        i64::try_from(model.context_window).expect("context window") - 16_384 + 1;
    let successful = assistant_with_usage(
        "large successful response",
        usage(
            u64::try_from(threshold_tokens - 10_000).expect("input"),
            10_000,
        ),
        now_ms(),
    );
    let error = error_message("529 overloaded", now_ms() + 1000);
    let messages = vec![
        user_message("hello", now_ms() - 1000),
        AgentMessage::Standard(pi_ai::types::Message::Assistant(successful)),
        user_message("another prompt", now_ms() + 500),
        AgentMessage::Standard(pi_ai::types::Message::Assistant(error.clone())),
    ];
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.messages = messages;
    setup.stream = Some(text_stream("summary"));
    let session = build(setup).await;
    authorize(&session).await;
    let (starts, listener) = compaction_starts();
    let _unsubscribe = session.subscribe(listener);

    session.check_compaction(&error, true).await;

    assert_eq!(
        starts.lock().expect("lock").clone(),
        vec![CompactionReason::Threshold]
    );
}

#[tokio::test]
async fn does_not_trigger_threshold_compaction_for_error_messages_when_no_prior_usage_exists() {
    let root = "/tmp/pi-session-auto-compaction-no-usage";
    let error = error_message("529 overloaded", now_ms());
    let messages = vec![
        user_message("hello", now_ms() - 1000),
        AgentMessage::Standard(pi_ai::types::Message::Assistant(error.clone())),
    ];
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.messages = messages;
    let session = build(setup).await;
    authorize(&session).await;
    let (starts, listener) = compaction_starts();
    let _unsubscribe = session.subscribe(listener);

    session.check_compaction(&error, true).await;

    assert!(starts.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn does_not_trigger_threshold_compaction_for_error_messages_when_only_kept_pre_compaction_usage_exists()
 {
    let root = "/tmp/pi-session-auto-compaction-kept";
    let pre_compaction_timestamp = now_ms() - 10_000;
    let kept = assistant_with_usage(
        "kept response from before compaction",
        usage(180_000, 10_000),
        pre_compaction_timestamp,
    );
    let mut manager = in_memory_manager(root);
    manager
        .append_message(user_message(
            "before compaction",
            pre_compaction_timestamp - 1000,
        ))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(pi_ai::types::Message::Assistant(
            kept.clone(),
        )))
        .expect("append");
    let first_kept_entry_id = manager
        .entries()
        .iter()
        .find_map(|entry| entry.entry_id().map(str::to_owned))
        .expect("entry id");
    manager
        .append_compaction("summary", &first_kept_entry_id, 190_000, None, None, None)
        .expect("compaction");
    let error = error_message("529 overloaded", now_ms());
    let messages = vec![
        user_message("kept user msg", pre_compaction_timestamp - 1000),
        AgentMessage::Standard(pi_ai::types::Message::Assistant(kept)),
        user_message("new prompt", now_ms() - 500),
        AgentMessage::Standard(pi_ai::types::Message::Assistant(error.clone())),
    ];
    let mut setup = SessionSetup::new(root, manager);
    setup.messages = messages;
    let session = build(setup).await;
    authorize(&session).await;
    let (starts, listener) = compaction_starts();
    let _unsubscribe = session.subscribe(listener);

    session.check_compaction(&error, true).await;

    assert!(starts.lock().expect("lock").is_empty());
}
