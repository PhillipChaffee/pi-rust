//! The transcript helpers suite: upstream's `src/harness/runtime/transcript.ts`
//! behaviors — the parent chaining, the committed entries' lifecycle
//! events, the bounded reads, and the queue/pending reads' error paths —
//! against the landed session layer.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; a violated expectation panics the test by design"
)]

use std::any::Any;
use std::sync::Arc;

use pi_ai::types::Message;
use serde_json::json;

use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::LaneQueuedItem;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::agent_harness::PromptMessagesPayload;
use crate::harness::context::background_context;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::test_support::empty_lane_snapshot;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::noop_emit_batch;
use crate::harness::runtime::test_support::noop_hook_reporter;
use crate::harness::runtime::test_support::passthrough_fault_handler;
use crate::harness::runtime::test_support::recording_watch_installer;
use crate::harness::runtime::test_support::runtime_config;
use crate::harness::runtime::test_support::seed_main_lane_values;
use crate::harness::runtime::transcript::chain_entries;
use crate::harness::runtime::transcript::committed_entry_events;
use crate::harness::runtime::transcript::entry_lifecycle_events;
use crate::harness::runtime::transcript::read_bounded_context;
use crate::harness::runtime::transcript::read_bounded_entries;
use crate::harness::runtime::transcript::read_lane_queues;
use crate::harness::runtime::transcript::read_pending_messages;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::session::memory::MemoryStorage;
use crate::harness::session::memory::MemoryStorageOptions;
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::Entry;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionStats;
use crate::harness::session::values as stored_values;
use crate::harness::session::values::ValueAddress;
use crate::harness::session::values::ValueSetWrite;
use crate::harness::session::values::Write;
use crate::types::AgentMessage;

fn next_session_id() -> String {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    format!(
        "runtime-transcript-{}",
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    )
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text(text.to_owned()),
        timestamp: 1,
    }))
}

fn user_new_entry(id: &str, text: &str) -> NewEntry {
    NewEntry::Message {
        id: id.to_owned(),
        parent_id: None,
        body: Box::new(MessageEntry {
            message: user_message(text),
            terminate: None,
        }),
    }
}

fn commit_result(entries: usize) -> CommitResult {
    CommitResult {
        first_seq: 1,
        seqs: (0..entries + 2)
            .map(|index| u64::try_from(index + 1).expect("seq"))
            .collect(),
        timestamp: 42,
        stats: SessionStats {
            message_count: u64::try_from(entries).expect("count"),
            usage: pi_ai::types::Usage::default(),
        },
    }
}

fn committed_entries(events: &[crate::harness::agent_harness::HarnessEvent]) -> Vec<Entry> {
    events
        .iter()
        .filter_map(|event| match &event.payload {
            HarnessEventPayload::EntryAdded { entry } => Some(entry.clone()),
            _ => None,
        })
        .collect()
}

async fn commit_writes(session: &Arc<StorageBackedSession>, writes: Vec<Write>) {
    session
        .mutate(
            Box::new(move |mutator, context| {
                let writes = writes.clone();
                Box::pin(async move {
                    mutator.commit(writes, context).await?;
                    let payload: Box<dyn Any + Send> = Box::new(());
                    Ok(payload)
                })
            }),
            &background_context(),
        )
        .await
        .expect("the commit settles");
}

/// The raw value-set write the malformed fixtures build: the serialized
/// payload bypasses the typed setter, mirroring a corrupted store.
fn raw_write(address: &ValueAddress, value: serde_json::Value) -> Write {
    Write::ValueSet(ValueSetWrite {
        kind: "value".to_owned(),
        op: "set".to_owned(),
        namespace: address.namespace.clone(),
        key: address.key.clone(),
        value,
    })
}

/// The lane the bounded-read fixtures drive: restored from the seeded
/// session over the memory backend.
async fn lane_fixture() -> (Arc<Lane>, Arc<StorageBackedSession>) {
    let session = Arc::new(StorageBackedSession::new(
        crate::harness::runtime::test_support::runtime_session_metadata(next_session_id()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        crate::harness::session::session::StorageBackedSessionOptions::default(),
    ));
    seed_main_lane_values(&session, None)
        .await
        .expect("seed commit");
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Arc::new(Lane::new(
        "main",
        session.clone(),
        Arc::new(pi_ai::models::create_models(None)),
        Arc::new(crate::harness::hooks::HookRegistry::new(
            noop_hook_reporter(),
        )),
        restored,
        passthrough_fault_handler(),
        noop_emit_batch(),
        recording_watch_installer(empty_lane_snapshot("main", &lane_configuration())),
        Arc::new(runtime_config),
    ));
    (lane, session)
}

#[test]
fn chains_entries_into_one_parent_line() {
    let chained = chain_entries(
        None,
        vec![
            user_new_entry("first", "one"),
            user_new_entry("second", "two"),
        ],
    );
    let parents: Vec<Option<&str>> = chained
        .iter()
        .map(|entry| match entry {
            NewEntry::Message { parent_id, .. } => parent_id.as_deref(),
            other => panic!("the message entry: {other:?}"),
        })
        .collect();
    assert_eq!(
        parents,
        vec![None, Some("first")],
        "each entry names its predecessor",
    );

    let from_parent = chain_entries(
        Some("root".to_owned()),
        vec![
            user_new_entry("first", "one"),
            user_new_entry("second", "two"),
        ],
    );
    let parents: Vec<Option<&str>> = from_parent
        .iter()
        .map(|entry| match entry {
            NewEntry::Message { parent_id, .. } => parent_id.as_deref(),
            other => panic!("the message entry: {other:?}"),
        })
        .collect();
    assert_eq!(
        parents,
        vec![Some("root"), Some("first")],
        "the chain starts at the given parent",
    );
}

#[test]
fn announces_message_entries_start_end_and_commit() {
    let entry = Entry::Message {
        id: "entry".to_owned(),
        parent_id: None,
        seq: 1,
        timestamp: 1,
        body: Box::new(MessageEntry {
            message: user_message("hello"),
            terminate: None,
        }),
    };
    let events = entry_lifecycle_events(entry.clone(), "main", None);
    assert_eq!(events.len(), 3, "start, end, and the commit");
    match &events[0].payload {
        HarnessEventPayload::MessageStart { run_id, message } => {
            assert_eq!(*run_id, None, "the run-less commit announces no run");
            assert_eq!(message, &user_message("hello"));
        }
        other => panic!("the first event: {other:?}"),
    }
    match &events[1].payload {
        HarnessEventPayload::MessageEnd {
            run_id,
            message,
            entry_id,
        } => {
            assert_eq!(*run_id, None);
            assert_eq!(message, &user_message("hello"));
            assert_eq!(entry_id.as_deref(), Some("entry"));
        }
        other => panic!("the second event: {other:?}"),
    }
    match &events[2].payload {
        HarnessEventPayload::EntryAdded { entry: added } => assert_eq!(added, &entry),
        other => panic!("the third event: {other:?}"),
    }
    for event in &events {
        assert_eq!(
            event.lane.as_deref(),
            Some("main"),
            "the events are lane-scoped"
        );
    }

    let run_events = entry_lifecycle_events(entry, "main", Some("run"));
    let start = match &run_events[0].payload {
        HarnessEventPayload::MessageStart { run_id, .. } => run_id.as_deref(),
        other => panic!("the first event: {other:?}"),
    };
    let end = match &run_events[1].payload {
        HarnessEventPayload::MessageEnd { run_id, .. } => run_id.as_deref(),
        other => panic!("the second event: {other:?}"),
    };
    assert_eq!(
        (start, end),
        (Some("run"), Some("run")),
        "the run's commits name the run",
    );

    let compaction = Entry::Compaction {
        id: "compact".to_owned(),
        parent_id: None,
        seq: 2,
        timestamp: 2,
        body: crate::harness::session::types::CompactionEntryBody {
            summary: "summary".to_owned(),
            retained_tail: Vec::new(),
            tokens_before: 0,
            details: None,
            usage: None,
            from_hook: false,
        },
    };
    let events = entry_lifecycle_events(compaction, "main", None);
    assert_eq!(
        events.len(),
        1,
        "non-message entries announce the commit only",
    );
    assert!(matches!(
        events[0].payload,
        HarnessEventPayload::EntryAdded { .. }
    ));
}

#[test]
fn materializes_committed_entries_with_their_storage_metadata() {
    let entries = chain_entries(
        None,
        vec![
            user_new_entry("first", "one"),
            user_new_entry("second", "two"),
        ],
    );
    let events = committed_entry_events(&entries, &commit_result(2), "main", None, 0);
    assert_eq!(events.len(), 6, "two entries, three events each");
    let materialized = committed_entries(&events);
    let mut pairs = Vec::new();
    for entry in &materialized {
        match entry {
            Entry::Message {
                id,
                seq,
                timestamp,
                parent_id,
                ..
            } => pairs.push((id.as_str(), *seq, *timestamp, parent_id.as_deref())),
            other => panic!("the materialized entry: {other:?}"),
        }
    }
    assert_eq!(
        pairs,
        vec![("first", 1, 42, None), ("second", 2, 42, Some("first"))],
        "each entry materializes its commit sequence and timestamp",
    );

    let events = committed_entry_events(&entries, &commit_result(4), "main", None, 2);
    let materialized = committed_entries(&events);
    let seqs: Vec<u64> = materialized
        .iter()
        .map(|entry| match entry {
            Entry::Message { seq, .. } => *seq,
            other => panic!("the materialized entry: {other:?}"),
        })
        .collect();
    assert_eq!(
        seqs,
        vec![3, 4],
        "the first-write index offsets the sequence window",
    );
}

/// The operation admission the bounded reads drive: one accepted run whose
/// tip commits the prompt's user message.
async fn admitted_run() -> (Arc<Lane>, Arc<StorageBackedSession>, Drive) {
    let (lane, session) = lane_fixture().await;
    let admission = lane
        .accept(
            OperationRequest::Prompt {
                operation_id: None,
                prompt: Box::new(PromptMessagesPayload::Text {
                    prompt: "hello".to_owned(),
                    images: None,
                }),
            },
            &background_context(),
        )
        .await
        .expect("accept serves")
        .expect("the admission");
    let drive = Drive::new(
        &DriveOptions {
            operation_id: admission.operation_id,
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    );
    (lane, session, drive)
}

/// The tipless-operation lane the no-tip invariant drives: the seeded
/// starting operation carries no branch tip.
async fn tipless_operation_lane() -> (Arc<Lane>, Drive) {
    let session = Arc::new(StorageBackedSession::new(
        crate::harness::runtime::test_support::runtime_session_metadata(next_session_id()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        crate::harness::session::session::StorageBackedSessionOptions::default(),
    ));
    seed_main_lane_values(&session, Some("operation"))
        .await
        .expect("seed commit");
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Arc::new(Lane::new(
        "main",
        session,
        Arc::new(pi_ai::models::create_models(None)),
        Arc::new(crate::harness::hooks::HookRegistry::new(
            noop_hook_reporter(),
        )),
        restored,
        passthrough_fault_handler(),
        noop_emit_batch(),
        recording_watch_installer(empty_lane_snapshot("main", &lane_configuration())),
        Arc::new(runtime_config),
    ));
    let drive = Drive::new(
        &DriveOptions {
            operation_id: "operation".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    );
    (lane, drive)
}

fn accepted_prompt_content(message: &AgentMessage) -> Vec<pi_ai::types::UserBlock> {
    match message {
        AgentMessage::Standard(Message::User(user)) => match &user.content {
            pi_ai::types::UserContent::Blocks(blocks) => blocks.clone(),
            other @ pi_ai::types::UserContent::Text(_) => panic!("the blocks content: {other:?}"),
        },
        other => panic!("the user message: {other:?}"),
    }
}

fn accepted_prompt_text() -> Vec<pi_ai::types::UserBlock> {
    vec![pi_ai::types::UserBlock::Text(pi_ai::types::TextContent {
        text: "hello".to_owned(),
        text_signature: None,
    })]
}

#[tokio::test]
async fn reads_the_run_bounded_entries_oldest_first() {
    let (lane, _session, drive) = admitted_run().await;
    let entries = match read_bounded_entries(&lane, &drive).await {
        Ok(ContinueOperationResult::Result { value }) => value,
        other => panic!("the bounded read: {other:?}"),
    };
    assert_eq!(
        entries.len(),
        1,
        "the run's bounded path holds the accepted prompt only",
    );
    match &entries[0] {
        Entry::Message { body, .. } => assert_eq!(
            accepted_prompt_content(&body.message),
            accepted_prompt_text(),
            "the accepted prompt committed as the user message",
        ),
        other => panic!("the entry: {other:?}"),
    }
    let tip = lane.state().tip_id.expect("the accepted tip");
    assert_eq!(entries[0].id(), tip, "the path ends at the tip");
}

#[tokio::test]
async fn rejects_the_bounded_read_when_the_operation_has_no_tip() {
    let (lane, drive) = tipless_operation_lane().await;
    let read = read_bounded_entries(&lane, &drive).await;
    let error = match read {
        Err(error) => error,
        other => panic!("the tipless read: {other:?}"),
    };
    assert!(
        error
            .to_string()
            .contains("Run operation has no Branch tip"),
        "the no-tip invariant names the operation: {error}",
    );
}

#[tokio::test]
async fn reads_the_run_bounded_context_messages() {
    let (lane, _session, drive) = admitted_run().await;
    let messages = match read_bounded_context(&lane, &drive).await {
        Ok(ContinueOperationResult::Result { value }) => value,
        other => panic!("the bounded context: {other:?}"),
    };
    assert_eq!(
        messages.len(),
        1,
        "the bounded context holds the transcript"
    );
    assert_eq!(
        accepted_prompt_content(&messages[0]),
        accepted_prompt_text(),
        "the accepted prompt rides the context",
    );
}

/// The queued fixtures' session: the seeded lane with the steer and custom
/// pending payloads committed.
async fn queued_session() -> Arc<StorageBackedSession> {
    let session = Arc::new(StorageBackedSession::new(
        crate::harness::runtime::test_support::runtime_session_metadata(next_session_id()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        crate::harness::session::session::StorageBackedSessionOptions::default(),
    ));
    seed_main_lane_values(&session, None)
        .await
        .expect("seed commit");
    commit_writes(
        &session,
        vec![
            crate::harness::session::values::set_value_write(
                &stored_values::pending_entry("steer-1"),
                PendingEntry::Message {
                    payload: Box::new(user_message("steer")),
                },
            )
            .expect("steer pending write"),
            crate::harness::session::values::set_value_write(
                &stored_values::pending_entry("write-1"),
                PendingEntry::Custom {
                    custom_type: "note".to_owned(),
                    payload: Some(json!({ "k": "v" })),
                },
            )
            .expect("custom pending write"),
        ],
    )
    .await;
    session
}

#[tokio::test]
async fn reads_the_lane_queues_from_the_inbox_payloads() {
    let session = queued_session().await;
    let inbox = vec![
        InboxItem {
            entry_id: "steer-1".to_owned(),
            kind: InboxItemKind::Steer,
        },
        InboxItem {
            entry_id: "write-1".to_owned(),
            kind: InboxItemKind::Write,
        },
    ];
    let queues = read_lane_queues(session.as_ref(), &inbox, &background_context())
        .await
        .expect("the queues read");
    assert_eq!(queues.len(), 2);
    match &queues[0] {
        LaneQueuedItem::Message {
            entry_id,
            kind,
            message,
        } => {
            assert_eq!(entry_id, "steer-1");
            assert_eq!(*kind, InboxItemKind::Steer);
            assert_eq!(message.as_ref(), &user_message("steer"));
        }
        other @ LaneQueuedItem::Custom { .. } => panic!("the custom queue item: {other:?}"),
    }
    match &queues[1] {
        LaneQueuedItem::Custom {
            entry_id,
            kind,
            custom_type,
            data,
        } => {
            assert_eq!(entry_id, "write-1");
            assert_eq!(*kind, InboxItemKind::Write);
            assert_eq!(custom_type, "note");
            assert_eq!(data.as_ref(), Some(&json!({ "k": "v" })));
        }
        other @ LaneQueuedItem::Message { .. } => panic!("the message queue item: {other:?}"),
    }
}

#[tokio::test]
async fn reports_the_queue_read_invariants() {
    let session = queued_session().await;
    commit_raw(
        &session,
        &stored_values::pending_entry("malformed").address,
        json!(42),
    )
    .await;
    commit_raw(
        &session,
        &stored_values::pending_entry("custom-on-steer").address,
        serde_json::to_value(PendingEntry::Custom {
            custom_type: "note".to_owned(),
            payload: None,
        })
        .expect("custom wire"),
    )
    .await;

    let missing = read_lane_queues(
        session.as_ref(),
        &[InboxItem {
            entry_id: "absent".to_owned(),
            kind: InboxItemKind::Steer,
        }],
        &background_context(),
    )
    .await
    .expect_err("the missing payload rejects");
    assert!(
        missing
            .to_string()
            .contains("Pending steer entry absent is missing its payload"),
        "the missing payload names the queue and the entry: {missing}",
    );

    let malformed = read_lane_queues(
        session.as_ref(),
        &[InboxItem {
            entry_id: "malformed".to_owned(),
            kind: InboxItemKind::Steer,
        }],
        &background_context(),
    )
    .await
    .expect_err("the malformed payload rejects");
    assert!(
        malformed
            .to_string()
            .contains("Pending entry payload is malformed"),
        "the malformed payload names its parse failure: {malformed}",
    );

    let non_message = read_lane_queues(
        session.as_ref(),
        &[InboxItem {
            entry_id: "custom-on-steer".to_owned(),
            kind: InboxItemKind::Steer,
        }],
        &background_context(),
    )
    .await
    .expect_err("the custom payload on a message queue rejects");
    assert!(
        non_message
            .to_string()
            .contains("Pending steer entry custom-on-steer is not a message"),
        "the non-message payload names the queue: {non_message}",
    );
}

async fn commit_raw(
    session: &Arc<StorageBackedSession>,
    address: &ValueAddress,
    value: serde_json::Value,
) {
    commit_writes(session, vec![raw_write(address, value)]).await;
}

#[tokio::test]
async fn reads_the_pending_message_payloads() {
    let session = queued_session().await;
    commit_raw(
        &session,
        &stored_values::pending_entry("custom-1").address,
        serde_json::to_value(PendingEntry::Custom {
            custom_type: "note".to_owned(),
            payload: None,
        })
        .expect("custom wire"),
    )
    .await;
    commit_raw(
        &session,
        &stored_values::pending_entry("broken-1").address,
        json!(42),
    )
    .await;

    let messages = read_pending_messages(
        session.as_ref(),
        &["steer-1".to_owned()],
        "The cancelled steer",
        &background_context(),
    )
    .await
    .expect("the pending read");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, "steer-1");
    assert_eq!(messages[0].1, user_message("steer"));

    let missing = read_pending_messages(
        session.as_ref(),
        &["absent".to_owned()],
        "The cancelled steer",
        &background_context(),
    )
    .await
    .expect_err("the missing payload rejects");
    assert!(
        missing
            .to_string()
            .contains("The cancelled steer absent is missing its message payload"),
        "the missing payload carries the caller's description: {missing}",
    );

    let non_message = read_pending_messages(
        session.as_ref(),
        &["custom-1".to_owned()],
        "The cancelled steer",
        &background_context(),
    )
    .await
    .expect_err("the custom payload rejects");
    assert!(
        non_message
            .to_string()
            .contains("missing its message payload")
    );

    let malformed = read_pending_messages(
        session.as_ref(),
        &["broken-1".to_owned()],
        "The cancelled steer",
        &background_context(),
    )
    .await
    .expect_err("the malformed payload rejects");
    assert!(
        malformed
            .to_string()
            .contains("missing its message payload")
    );
}
#[test]
fn chains_the_structural_and_custom_entry_shapes() {
    let compaction = NewEntry::Compaction {
        id: "compact".to_owned(),
        parent_id: None,
        body: crate::harness::session::types::CompactionEntryBody {
            summary: "summary".to_owned(),
            retained_tail: Vec::new(),
            tokens_before: 0,
            details: None,
            usage: None,
            from_hook: false,
        },
    };
    let branch_summary = NewEntry::BranchSummary {
        id: "summary".to_owned(),
        parent_id: None,
        body: crate::harness::session::types::BranchSummaryEntryBody {
            from_id: None,
            summary: "summary".to_owned(),
            details: None,
            usage: None,
            from_hook: false,
        },
    };
    let chained = chain_entries(
        None,
        vec![compaction, branch_summary, user_new_entry("tail", "tail")],
    );
    let parents: Vec<Option<&str>> = chained
        .iter()
        .map(|entry| match entry {
            NewEntry::Message { parent_id, .. }
            | NewEntry::Compaction { parent_id, .. }
            | NewEntry::BranchSummary { parent_id, .. }
            | NewEntry::Custom { parent_id, .. } => parent_id.as_deref(),
        })
        .collect();
    assert_eq!(
        parents,
        vec![None, Some("compact"), Some("summary")],
        "every entry shape links into the chain",
    );
}

#[tokio::test]
async fn reads_the_bounded_entries_as_cancel_requested_once_cancelled() {
    let (lane, _session, drive) = admitted_run().await;
    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the operation");
    let requested =
        AgentLane::request_abort(lane.as_ref(), &operation_id, &background_context()).await;
    requested.expect("abort serves").expect("the request");
    let read = read_bounded_entries(&lane, &drive).await;
    assert!(
        matches!(read, Ok(ContinueOperationResult::CancelRequested)),
        "the cancelled run's read never runs the planner",
    );
    let context = read_bounded_context(&lane, &drive).await;
    assert!(
        matches!(context, Ok(ContinueOperationResult::CancelRequested)),
        "the cancelled run's context never builds",
    );
}

#[tokio::test]
async fn the_queue_invariants_name_the_write_and_next_run_queues() {
    let session = queued_session().await;
    for (entry_id, kind) in [
        ("absent-write", InboxItemKind::Write),
        ("absent-next", InboxItemKind::NextRun),
    ] {
        let missing = read_lane_queues(
            session.as_ref(),
            &[InboxItem {
                entry_id: entry_id.to_owned(),
                kind,
            }],
            &background_context(),
        )
        .await
        .expect_err("the missing payload rejects");
        assert!(
            missing
                .to_string()
                .contains(&format!("entry {entry_id} is missing its payload")),
            "the {kind:?} queue named the entry: {missing}",
        );
    }
}
