//! `test/interactive-mode-compaction.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements:
//! - Upstream's fake `this` objects become the real frame; `renderSessionItems`
//!   call assertions become the rendered transcript order the user sees.
//! - `sessionManager.buildContextEntries` is the real manager seeded through
//!   the harness with typed entries.
//! - The `showWorkingStatusIndicator`/`clearStatusIndicator` spies become the
//!   active indicator kind plus the terminal's OSC 9;4 progress writes, the
//!   port's `terminal.setProgress` restatement.
//! - `session.prompt`/`steer` spies become the real session queues.

#![expect(
    clippy::expect_used,
    reason = "the fixture entries and session are pinned; a rejected append is a test bug"
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::interactive::{Harness, HarnessOptions};
use pi_ai::types::{Usage, UsageCost};
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::session_manager::{FileEntry, SessionManager};
use pi_coding_agent::settings_manager::Settings;

fn usage(input: u64, output: u64, cache_read: u64, cache_write: u64, total: f64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        cost: UsageCost {
            total,
            ..UsageCost::default()
        },
        ..Usage::default()
    }
}

fn compaction_entry(id: &str, parent: Option<&str>, summary: &str, usage: Usage) -> FileEntry {
    serde_json::from_value(serde_json::json!({
        "type": "compaction",
        "id": id,
        "parentId": parent,
        "timestamp": "2025-01-02T00:00:00Z",
        "summary": summary,
        "firstKeptEntryId": "kept",
        "tokensBefore": 123,
        "usage": serde_json::to_value(usage).expect("usage serializes")
    }))
    .expect("compaction entry parses")
}

#[tokio::test]
async fn uses_the_cache_miss_notice_setting_for_compaction_and_branch_summary_costs() {
    let mut settings = Settings::new();
    settings.insert("showCacheMissNotices".to_owned(), true.into());
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;
    let usage = usage(10, 20, 30, 40, 0.125);

    harness
        .frame
        .add_compaction_cost_notice("Compaction", &usage);
    harness
        .frame
        .add_compaction_cost_notice("Branch summary", &usage);

    let text = harness.chat_text();
    assert!(text.contains("Compaction: 100 tokens billed (~$0.13)"));
    assert!(text.contains("Branch summary: 100 tokens billed (~$0.13)"));
}

#[tokio::test]
async fn hides_compaction_costs_when_cache_miss_notices_are_disabled() {
    let mut settings = Settings::new();
    settings.insert("showCacheMissNotices".to_owned(), false.into());
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;

    harness
        .frame
        .add_compaction_cost_notice("Compaction", &usage(10, 20, 30, 40, 0.125));

    assert_eq!(harness.frame.chat_container().children.borrow().len(), 0);
}

#[tokio::test]
async fn renders_each_compaction_cost_after_its_summary() {
    let mut settings = Settings::new();
    settings.insert("showCacheMissNotices".to_owned(), true.into());
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.set_tools_expanded(true);
    let current = compaction_entry(
        "current",
        Some("previous"),
        "current summary",
        usage(10, 20, 30, 40, 0.1),
    );
    let previous = compaction_entry(
        "previous",
        None,
        "previous summary",
        usage(1, 2, 3, 4, 0.01),
    );

    harness
        .frame
        .render_session_entries(&[current, previous], false, false);

    let text = harness.chat_text();
    let current_summary = text.find("current summary").expect("current summary");
    let current_cost = text
        .find("Compaction: 100 tokens billed")
        .expect("current cost");
    let previous_summary = text.find("previous summary").expect("previous summary");
    let previous_cost = text
        .find("Compaction: 10 tokens billed")
        .expect("previous cost");
    assert!(current_summary < current_cost, "{text}");
    assert!(current_cost < previous_summary, "{text}");
    assert!(previous_summary < previous_cost, "{text}");
}

#[tokio::test]
async fn renders_retained_entries_and_appends_the_latest_summary_cost_at_the_bottom() {
    let mut settings = Settings::new();
    settings.insert("showCacheMissNotices".to_owned(), true.into());
    let usage = usage(10, 20, 30, 40, 0.1);
    let seed_usage = usage;
    let harness = Harness::new(HarnessOptions {
        settings,
        seed_manager: Some(Box::new(move |manager: &mut SessionManager| {
            let first_kept = manager
                .append_message(common::agent_session::user_message("kept prompt", 1))
                .expect("user");
            manager
                .append_message(pi_agent_core::types::AgentMessage::Standard(
                    pi_ai::types::Message::Assistant(common::agent_session::assistant_message(
                        "kept reply",
                        2,
                    )),
                ))
                .expect("assistant");
            manager
                .append_compaction("summary", &first_kept, 123, None, None, Some(seed_usage))
                .expect("compaction");
        })),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.set_tools_expanded(true);
    harness.frame.show_error("stale row");

    harness
        .frame
        .handle_event(AgentSessionEvent::CompactionEnd {
            reason: pi_coding_agent::extensions::reasons::CompactionReason::Manual,
            result: Some(pi_coding_agent::compaction::CompactionResult {
                summary: "new summary".to_owned(),
                first_kept_entry_id: "kept".to_owned(),
                tokens_before: 123,
                estimated_tokens_after: None,
                usage,
                details: None,
            }),
            aborted: false,
            will_retry: false,
            error_message: None,
        });

    let text = harness.chat_text();
    assert!(
        !text.contains("stale row"),
        "the transcript was cleared: {text}"
    );
    assert!(text.contains("new summary"), "{text}");
    assert!(
        text.contains("Compaction: 100 tokens billed (~$0.10)"),
        "{text}"
    );
    assert!(text.contains("kept prompt"), "{text}");
    assert_eq!(
        harness.frame.get_all_queued_messages(),
        (Vec::new(), Vec::new())
    );
}

#[tokio::test]
async fn updates_the_working_state_when_the_same_agent_run_resumes_after_compaction() {
    let mut settings = Settings::new();
    settings.insert(
        "terminal".to_owned(),
        serde_json::json!({ "showTerminalProgress": true }),
    );
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.handle_event(AgentSessionEvent::TurnStart);

    assert!(harness.terminal.writes().contains("\x1b]9;4;3\x07"));
    assert_eq!(
        harness.frame.active_status_kind(),
        Some(
            pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind::Working
        )
    );

    harness.frame.set_working_visible(false);
    harness.frame.handle_event(AgentSessionEvent::TurnStart);

    assert_eq!(harness.frame.active_status_kind(), None);
}

#[tokio::test]
async fn preserves_steering_behavior_when_flushing_into_an_active_agent_run() {
    let release = Arc::new(tokio::sync::Notify::new());
    let stream = gated_stream(Arc::clone(&release));
    let harness = Harness::new(HarnessOptions {
        stream: Some(stream),
        ..HarnessOptions::default()
    })
    .await;
    let session = harness.frame.session();
    let prompt = session.prompt("first".to_owned(), None);
    tokio::pin!(prompt);
    tokio::select! {
        _ = &mut prompt => panic!("the gated stream must stay open"),
        () = tokio::time::sleep(Duration::from_millis(20)) => {}
    }
    assert!(session.is_streaming());

    harness.frame.queue_compaction_message(
        "change direction",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::Steer,
    );
    harness.frame.flush_compaction_queue(false);
    tokio::select! {
        _ = &mut prompt => {}
        () = tokio::time::sleep(Duration::from_millis(30)) => {}
    }

    assert!(
        session
            .steering_messages()
            .contains(&"change direction".to_owned())
    );
    assert_eq!(
        harness.frame.get_all_queued_messages(),
        (vec!["change direction".to_owned()], Vec::new())
    );
    release.notify_one();
    tokio::select! {
        _ = &mut prompt => {}
        () = tokio::time::sleep(Duration::from_millis(50)) => {}
    }
}

/// A stream that opens immediately and settles only when `release` notifies.
fn gated_stream(release: Arc<tokio::sync::Notify>) -> pi_agent_core::types::StreamFn {
    use pi_ai::types::{AssistantMessageEvent, StopReason};
    Arc::new(move |model, _context, _options| {
        let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
        let mut message = common::sdk_fixture::done_message();
        message.provider = model.provider.clone();
        message.model.clone_from(&model.id);
        stream.push(AssistantMessageEvent::Start {
            partial: pi_ai::types::AssistantMessage {
                content: Vec::new(),
                ..message.clone()
            },
        });
        let settle = stream.clone();
        let release = Arc::clone(&release);
        tokio::spawn(async move {
            release.notified().await;
            settle.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            });
        });
        stream
    })
}
