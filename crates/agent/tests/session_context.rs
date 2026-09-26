//! The session-context projection suite, ported 1:1 from upstream
//! `test/harness/session-context.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod session_common;
use session_common::*;

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::context::{SessionContextBuildOptions, build_session_context};
use pi_agent_core::harness::session::types::{
    BranchSummaryEntryBody, CompactionEntryBody, CustomEntryBody, Entry, EntryProjector,
    MessageEntry, SessionError,
};
use pi_agent_core::types::AgentMessage;

fn message_entry(id: &str, parent_id: Option<&str>, seq: u64, message: AgentMessage) -> Entry {
    Entry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        seq,
        timestamp: NOW,
        body: Box::new(MessageEntry {
            message,
            terminate: None,
        }),
    }
}

fn compaction_entry(
    id: &str,
    parent_id: Option<&str>,
    seq: u64,
    summary: &str,
    retained_tail: Vec<AgentMessage>,
    tokens_before: i64,
) -> Entry {
    Entry::Compaction {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        seq,
        timestamp: NOW,
        body: CompactionEntryBody {
            summary: summary.to_owned(),
            retained_tail,
            tokens_before,
            details: None,
            usage: None,
            from_hook: false,
        },
    }
}

fn branch_summary_entry(
    id: &str,
    parent_id: Option<&str>,
    seq: u64,
    from_id: Option<&str>,
    summary: &str,
) -> Entry {
    Entry::BranchSummary {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        seq,
        timestamp: NOW,
        body: BranchSummaryEntryBody {
            from_id: from_id.map(str::to_owned),
            summary: summary.to_owned(),
            details: None,
            usage: None,
            from_hook: false,
        },
    }
}

fn custom_entry(id: &str, parent_id: Option<&str>, seq: u64, custom_type: &str) -> Entry {
    Entry::Custom {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        seq,
        timestamp: NOW,
        body: CustomEntryBody {
            custom_type: custom_type.to_owned(),
            data: None,
        },
    }
}

/// The summary-role message the projections produce, upstream's
/// `{ role: "compactionSummary" | "branchSummary", ... }` literals.
fn summary_message(role: &str, data: serde_json::Value, timestamp: i64) -> AgentMessage {
    let fields = match data {
        serde_json::Value::Object(fields) => fields,
        _ => serde_json::Map::new(),
    };
    AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage {
        role: role.to_owned(),
        timestamp,
        data: fields,
    })
}

#[tokio::test]
async fn filters_non_context_assistant_response_entries_while_preserving_valid_messages() {
    let user = user_message("question");
    let stopped = assistant_message("stop", "answer");
    let length = assistant_message("length", "truncated answer");
    let tool_use = assistant_message("toolUse", "");
    let failed = assistant_message("error", "failed");
    let aborted = assistant_message("aborted", "aborted");
    let deferred = assistant_message("deferred", "");
    let entries = vec![
        message_entry("user", None, 1, user.clone()),
        message_entry("failed", Some("user"), 2, failed),
        message_entry("stopped", Some("failed"), 3, stopped.clone()),
        message_entry("aborted", Some("stopped"), 4, aborted),
        message_entry("tool-use", Some("aborted"), 5, tool_use.clone()),
        message_entry("deferred", Some("tool-use"), 6, deferred),
        message_entry("length", Some("deferred"), 7, length.clone()),
    ];

    assert_eq!(
        build_session_context(&entries, None, &background_context())
            .await
            .expect("context"),
        [user, stopped, tool_use, length],
    );
}

#[tokio::test]
async fn projects_branch_summaries_in_branch_order() {
    let before = message_entry("before", None, 1, user_message("before summary"));
    let summary = branch_summary_entry(
        "branch-summary",
        Some("before"),
        2,
        Some("source-leaf"),
        "work on the abandoned branch",
    );
    let after = message_entry(
        "after",
        Some("branch-summary"),
        3,
        user_message("after summary"),
    );

    assert_eq!(
        build_session_context(&[before, summary, after], None, &background_context())
            .await
            .expect("context"),
        [
            user_message("before summary"),
            summary_message(
                "branchSummary",
                serde_json::json!({
                    "summary": "work on the abandoned branch",
                    "fromId": "source-leaf",
                }),
                NOW,
            ),
            user_message("after summary"),
        ],
    );
}

#[tokio::test]
async fn filters_retained_tail_responses_without_hiding_the_compaction_summary() {
    let user = user_message("kept user");
    let stopped = assistant_message("stop", "kept answer");
    let tool_use = assistant_message("toolUse", "");
    let length = assistant_message("length", "kept truncated answer");
    let failed = assistant_message("error", "failed");
    let aborted = assistant_message("aborted", "aborted");
    let deferred = assistant_message("deferred", "");
    let compaction = compaction_entry(
        "compaction",
        None,
        1,
        "summary",
        vec![
            failed,
            user.clone(),
            aborted,
            stopped.clone(),
            deferred,
            tool_use.clone(),
            length.clone(),
        ],
        100,
    );

    assert_eq!(
        build_session_context(&[compaction], None, &background_context())
            .await
            .expect("context"),
        [
            summary_message(
                "compactionSummary",
                serde_json::json!({ "summary": "summary", "tokensBefore": 100 }),
                NOW,
            ),
            user,
            stopped,
            tool_use,
            length,
        ],
    );
}

#[tokio::test]
async fn uses_only_the_latest_compaction_checkpoint_and_entries_after_it() {
    let before_first = message_entry("before-first", None, 1, user_message("before first"));
    let first = compaction_entry(
        "first-compaction",
        Some("before-first"),
        2,
        "stale summary",
        vec![user_message("stale tail")],
        100,
    );
    let between = message_entry(
        "between",
        Some("first-compaction"),
        3,
        user_message("between compactions"),
    );
    let latest = compaction_entry(
        "latest-compaction",
        Some("between"),
        4,
        "latest summary",
        vec![user_message("latest tail")],
        200,
    );
    let after = message_entry(
        "after",
        Some("latest-compaction"),
        5,
        user_message("after latest"),
    );

    assert_eq!(
        build_session_context(
            &[before_first, first, between, latest, after],
            None,
            &background_context(),
        )
        .await
        .expect("context"),
        [
            summary_message(
                "compactionSummary",
                serde_json::json!({ "summary": "latest summary", "tokensBefore": 200 }),
                NOW,
            ),
            user_message("latest tail"),
            user_message("after latest"),
        ],
    );
}

#[tokio::test]
async fn projects_custom_entries_through_synchronous_and_asynchronous_projectors_in_branch_order() {
    let old_custom = custom_entry("old-custom", None, 1, "sync");
    let compaction = compaction_entry(
        "compaction",
        Some("old-custom"),
        2,
        "summary",
        Vec::new(),
        100,
    );
    let sync_custom = custom_entry("sync-custom", Some("compaction"), 3, "sync");
    let omitted_custom = custom_entry("omitted-custom", Some("sync-custom"), 4, "omitted");
    let async_custom = custom_entry("async-custom", Some("omitted-custom"), 5, "async");
    let projected_ids = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

    let sync_projector = {
        let projected_ids = projected_ids.clone();
        EntryProjector(Arc::new(
            move |entry: &Entry,
                  _context|
                  -> pi_ai::types::BoxedFuture<
                '_,
                Result<Option<Vec<AgentMessage>>, SessionError>,
            > {
                let projected_ids = projected_ids.clone();
                let id = entry.id().to_owned();
                Box::pin(async move {
                    projected_ids
                        .lock()
                        .expect("projected ids")
                        .push(id.clone());
                    Ok(Some(vec![user_message(&format!("projected:{id}"))]))
                })
            },
        ))
    };
    let async_projector = {
        let projected_ids = projected_ids.clone();
        EntryProjector(Arc::new(
            move |entry: &Entry,
                  _context|
                  -> pi_ai::types::BoxedFuture<
                '_,
                Result<Option<Vec<AgentMessage>>, SessionError>,
            > {
                let projected_ids = projected_ids.clone();
                let id = entry.id().to_owned();
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    projected_ids
                        .lock()
                        .expect("projected ids")
                        .push(id.clone());
                    Ok(Some(vec![user_message(&format!("projected:{id}"))]))
                })
            },
        ))
    };
    let entry_projectors = BTreeMap::from([
        ("sync".to_owned(), sync_projector),
        ("async".to_owned(), async_projector),
    ]);

    let messages = build_session_context(
        &[
            old_custom,
            compaction,
            sync_custom.clone(),
            omitted_custom,
            async_custom.clone(),
        ],
        Some(&SessionContextBuildOptions {
            entry_projectors: Some(entry_projectors),
        }),
        &background_context(),
    )
    .await
    .expect("context");

    assert_eq!(
        *projected_ids.lock().expect("projected ids"),
        [sync_custom.id().to_owned(), async_custom.id().to_owned()],
    );
    assert_eq!(
        messages,
        [
            summary_message(
                "compactionSummary",
                serde_json::json!({ "summary": "summary", "tokensBefore": 100 }),
                NOW,
            ),
            user_message(&format!("projected:{}", sync_custom.id())),
            user_message(&format!("projected:{}", async_custom.id())),
        ],
    );
}

#[tokio::test]
async fn propagates_custom_projector_failures() {
    let custom = custom_entry("custom", None, 1, "broken");
    let failure = SessionError::Message("projector failed".to_owned());
    let expected = failure.clone();

    let projector =
        EntryProjector(Arc::new(
            move |_entry: &Entry,
                  _context|
                  -> pi_ai::types::BoxedFuture<
                '_,
                Result<Option<Vec<AgentMessage>>, SessionError>,
            > {
                let failure = expected.clone();
                Box::pin(std::future::ready(Err(failure)))
            },
        ));
    let rejected = build_session_context(
        &[custom],
        Some(&SessionContextBuildOptions {
            entry_projectors: Some(BTreeMap::from([("broken".to_owned(), projector)])),
        }),
        &background_context(),
    )
    .await;

    assert_eq!(rejected.expect_err("projector failure"), failure);
}
