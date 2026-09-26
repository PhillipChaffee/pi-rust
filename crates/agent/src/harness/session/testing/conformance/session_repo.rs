//! The `SessionRepo` conformance cases, ported from upstream
//! `src/harness/session/testing/conformance/session-repo.ts`.
//!
//! They cover lifecycle, ownership, messages, fork behavior, fork
//! coordination, and the streaming-fork subset. Upstream parameterizes
//! every creator over the repo's metadata and
//! option generics so partial backends can run groups independently; the
//! port erases to the object-safe
//! [`SessionRepo`] surface,
//! and a partial backend implements the trait with panicking stubs for the
//! methods it does not support yet. The bodies assert with `assert*!`
//! (upstream's `node:assert`).

#![expect(
    clippy::expect_used,
    reason = "conformance fixtures construct fixed addresses and fixed writes whose Results are infallible; a failure is a bug the case panics on"
)]
#![expect(
    clippy::panic,
    reason = "conformance cases panic on a broken fixture contract, upstream's node:assert throws"
)]
#![expect(
    clippy::too_many_lines,
    reason = "each creator mirrors one upstream case list; the bodies are the cases"
)]

use std::sync::Arc;

use pi_ai::types::{BoxedFuture, Usage};
use serde_json::{Value as JsonValue, json};

use crate::harness::context::background_context;
use crate::harness::session::testing::conformance::{
    asc_query, assert_lane_config, assert_lane_state, assert_session_list_values,
    assert_session_value_absent, branch_fork, close_session, create_repo_session,
    downcast_commit_result, find_entry_ids, fork_branch_session, fork_tree_session,
    insert_entry_write, insert_usage_write, label_write, lane_config_write, lane_state_write,
    seed_custom_entry, stored_values, tip_write, tree_fork,
};
use crate::harness::session::testing::types::{ConformanceCase, RepoFixture};
use crate::harness::session::types::{
    Entry, ForkPosition, LaneConfiguration, LaneState, OperationMeta, OperationResultRecord,
    Session, SessionError, SessionRepo,
};
use crate::harness::session::values::Write;
use crate::harness::session::values::{
    append_list as append_list_write, branch_tip, entry_label, generic_value, lane_config,
    lane_state, operation_meta, operation_preparation, operation_result, operation_state,
    operation_tool_args, pending_entry, session_name, set_value as set_value_write,
};
use crate::types::ThinkingLevel;

/// The factory one repo fixture case builds through, upstream's
/// `() => Promise<TRepo>`.
pub type RepoFixtureFactory = Arc<dyn Fn() -> BoxedFuture<'static, RepoFixture> + Send + Sync>;

/// The per-case body, upstream's `test: (context) => Promise<void>`.
pub type RepoCaseTest =
    Arc<dyn for<'a> Fn(&'a dyn SessionRepo) -> BoxedFuture<'a, ()> + Send + Sync>;

const ROOT_ID: &str = "00000000-0000-7000-8000-000000000001";
const CHILD_ID: &str = "00000000-0000-7000-8000-000000000002";
const SIBLING_ID: &str = "00000000-0000-7000-8000-000000000003";
const USAGE_ID: &str = "00000000-0000-7000-8000-000000000004";
const OPERATION_ID: &str = "00000000-0000-7000-8000-000000000005";
const PENDING_ID: &str = "00000000-0000-7000-8000-000000000006";
const UNKNOWN_ID: &str = "00000000-0000-7000-8000-000000000007";

fn idle_lane_state() -> LaneState {
    LaneState::default()
}

fn configuration() -> LaneConfiguration {
    LaneConfiguration {
        model: crate::harness::session::types::ModelIdentity {
            provider: "provider".to_owned(),
            model_id: "model".to_owned(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: vec!["read".to_owned()],
    }
}

fn application_value() -> crate::harness::session::values::Value<JsonValue> {
    generic_value("test.application.value", "")
}

fn application_list() -> crate::harness::session::values::ValueList<JsonValue> {
    stored_values::generic_list("test.application.list", "")
}

fn events_list() -> crate::harness::session::values::ValueList<JsonValue> {
    stored_values::generic_list("test.application.events", "")
}

/// The branch tip one session carries for the named branch, upstream's
/// `getBranchTip`; `None` when the session has no such branch.
async fn get_branch_tip(session: &dyn Session, name: &str) -> Result<Option<String>, SessionError> {
    match session.branch(name, &background_context()).await? {
        Some(branch) => branch.get_tip_id(&background_context()).await,
        None => Ok(None),
    }
}

fn assistant_message(stop_reason: &str) -> crate::types::AgentMessage {
    let content = if stop_reason == "toolUse" {
        json!([{ "type": "toolCall", "id": "call", "name": "read", "arguments": {} }])
    } else {
        json!([{ "type": "text", "text": stop_reason }])
    };
    let mut wire = json!({
        "role": "assistant",
        "content": content,
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
        },
        "stopReason": stop_reason,
        "timestamp": 1,
    });
    if stop_reason == "deferred" {
        wire["deferred"] = json!({
            "provider": "anthropic",
            "modelId": "claude-sonnet-4-5",
            "api": "anthropic-messages",
            "id": "job",
        });
    }
    serde_json::from_value(wire).expect("assistant message wire")
}

fn settled_usage() -> Usage {
    Usage {
        input: 1,
        output: 2,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 3,
        cost: pi_ai::types::UsageCost::default(),
    }
}

fn zero_usage() -> Usage {
    Usage::default()
}

/// The callback committing one transaction through the exclusive mutator,
/// upstream's `(mutator) => mutator.commit([...])` closures.
fn commit_writes(writes: Vec<Write>) -> crate::harness::session::types::SessionMutationCallback {
    crate::harness::session::session::StorageBackedSession::commit_writes_callback(writes)
}

/// The stop reasons the message conformance cycles, upstream's
/// `Record<Exclude<StopReason, "pending">, AssistantMessage>` keys.
const SETTLED_STOP_REASONS: [&str; 6] =
    ["stop", "length", "toolUse", "error", "aborted", "deferred"];

/// Creates lifecycle cases for repositories that support creation,
/// discovery, open, and deletion, upstream's
/// `createSessionRepoLifecycleConformance`.
#[must_use]
///
/// # Panics
/// A case body panics on its first broken assertion, upstream's `node:assert` throw.
pub fn create_session_repo_lifecycle_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    vec![
        repo_case(
            factory,
            "lifecycle",
            "creates a session with no implicit branch and rejects duplicate ids",
            Arc::new(|repo| {
                Box::pin(async move {
                    let session = create_repo_session(repo, Some("session"), None).await;

                    assert_eq!(session.metadata().id, "session");
                    assert_ne!(session.metadata().created_at, 0);
                    assert_eq!(session.metadata().storage_version, 1);
                    assert!(
                        session
                            .branch("main", &background_context())
                            .await
                            .expect("branch")
                            .is_none(),
                    );
                    assert_session_value_absent(session.as_ref(), &lane_state("main").address)
                        .await;
                    assert_session_value_absent(session.as_ref(), &lane_config("main").address)
                        .await;
                    let rejected = repo
                        .create(
                            crate::harness::session::types::SessionCreateOptions {
                                id: Some("session".to_owned()),
                                ..Default::default()
                            },
                            &background_context(),
                        )
                        .await;
                    assert!(rejected.is_err(), "expected the duplicate id to reject");
                    close_session(session.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "lifecycle",
            "close drains an acquired scope and rejects a queued mutation callback",
            Arc::new(|repo| {
                Box::pin(async move {
                    let session = create_repo_session(repo, Some("session"), None).await;
                    let active = session
                        .begin_mutation(&background_context())
                        .await
                        .expect("begin");
                    let queued_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let queued_flag = queued_started.clone();
                    let queued = session.mutate(
                        Box::new(
                            move |_mutator: &dyn crate::harness::session::types::SessionMutator,
                                  _context| {
                                queued_flag.store(true, std::sync::atomic::Ordering::Release);
                                let done: Box<dyn std::any::Any + Send> = Box::new(());
                                Box::pin(std::future::ready(Ok(done)))
                            },
                        ),
                        &background_context(),
                    );
                    let closing = session.close(&background_context());

                    active.end(&background_context()).await.expect("end");
                    assert!(
                        queued.await.is_err(),
                        "expected the queued callback to reject"
                    );
                    closing.await.expect("close");
                    assert!(
                        !queued_started.load(std::sync::atomic::Ordering::Acquire),
                        "the queued callback must not have started",
                    );
                })
            }),
        ),
        repo_case(
            factory,
            "lifecycle",
            "lists metadata and preserves state across close and reopen",
            Arc::new(|repo| {
                Box::pin(async move {
                    let first = repo
                        .create(
                            crate::harness::session::types::SessionCreateOptions {
                                id: Some("first".to_owned()),
                                ..Default::default()
                            },
                            &background_context(),
                        )
                        .await
                        .expect("create first");
                    first
                        .set_name(Some("preserved".to_owned()), &background_context())
                        .await
                        .expect("set name");
                    let second = create_repo_session(repo, Some("second"), Some("parent")).await;
                    close_session(second.as_ref()).await;

                    let listed = repo.list(&background_context()).await.expect("list");
                    let listed: Vec<(String, Option<String>)> = listed
                        .into_iter()
                        .map(|metadata| (metadata.id.clone(), metadata.parent_session_id))
                        .collect();
                    let mut listed = listed;
                    listed.sort_by(|left, right| left.0.cmp(&right.0));
                    assert_eq!(
                        listed,
                        vec![
                            ("first".to_owned(), None),
                            ("second".to_owned(), Some("parent".to_owned())),
                        ],
                    );
                    close_session(first.as_ref()).await;
                    let rejected = first.get_name(&background_context()).await;
                    assert!(rejected.is_err(), "expected the closed session to reject");
                    let reopened = repo
                        .open(first.metadata(), &background_context())
                        .await
                        .expect("reopen");
                    assert_eq!(
                        reopened
                            .get_name(&background_context())
                            .await
                            .expect("name"),
                        Some("preserved".to_owned()),
                    );
                    close_session(reopened.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "lifecycle",
            "deletes closed sessions without affecting other sessions",
            Arc::new(|repo| {
                Box::pin(async move {
                    let removed = repo
                        .create(
                            crate::harness::session::types::SessionCreateOptions {
                                id: Some("removed".to_owned()),
                                ..Default::default()
                            },
                            &background_context(),
                        )
                        .await
                        .expect("create removed");
                    let retained = repo
                        .create(
                            crate::harness::session::types::SessionCreateOptions {
                                id: Some("retained".to_owned()),
                                ..Default::default()
                            },
                            &background_context(),
                        )
                        .await
                        .expect("create retained");
                    close_session(removed.as_ref()).await;
                    close_session(retained.as_ref()).await;

                    repo.delete(removed.metadata(), &background_context())
                        .await
                        .expect("delete");
                    assert_eq!(
                        repo.list(&background_context())
                            .await
                            .expect("list")
                            .into_iter()
                            .map(|metadata| metadata.id)
                            .collect::<Vec<_>>(),
                        ["retained"],
                    );
                    let rejected = repo.open(removed.metadata(), &background_context()).await;
                    assert!(
                        rejected.is_err(),
                        "expected the deleted session to reject reopen"
                    );
                    let rejected = repo.delete(removed.metadata(), &background_context()).await;
                    assert!(
                        rejected.is_err(),
                        "expected the deleted session to reject delete"
                    );
                })
            }),
        ),
    ]
}

/// Creates exclusive-open cases for repositories that own active session
/// handles, upstream's `createSessionRepoOwnershipConformance`.
#[must_use]
///
/// # Panics
/// A case body panics on its first broken assertion, upstream's `node:assert` throw.
pub fn create_session_repo_ownership_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    vec![repo_case(
        factory,
        "ownership",
        "rejects opening an already-open session",
        Arc::new(|repo| {
            Box::pin(async move {
                let session = create_repo_session(repo, Some("session"), None).await;
                let rejected = repo.open(session.metadata(), &background_context()).await;
                assert!(
                    rejected.is_err(),
                    "expected the open session to reject reopen"
                );
                close_session(session.as_ref()).await;

                let reopened = repo
                    .open(session.metadata(), &background_context())
                    .await
                    .expect("reopen");
                let rejected = repo.open(session.metadata(), &background_context()).await;
                assert!(
                    rejected.is_err(),
                    "expected the reopened session to reject reopen"
                );
                close_session(reopened.as_ref()).await;
            })
        }),
    )]
}

/// Creates message cases for repositories that support session creation,
/// upstream's `createSessionRepoMessageConformance`.
#[must_use]
///
/// # Panics
/// A case body panics on its first broken assertion, upstream's `node:assert` throw.
pub fn create_session_repo_message_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    vec![
        repo_case(
            factory,
            "messages",
            "rejects pending assistant messages without changing the tree",
            Arc::new(|repo| {
                Box::pin(async move {
                    let session = create_repo_session(repo, Some("session"), None).await;
                    let branch = session
                        .create_branch("main", None, &background_context())
                        .await
                        .expect("create branch");

                    let rejected = branch
                        .append_message(assistant_message("pending"), &background_context())
                        .await;
                    assert!(rejected.is_err(), "expected the pending message to reject");

                    assert_eq!(
                        get_branch_tip(session.as_ref(), "main").await.expect("tip"),
                        None,
                    );
                    assert!(
                        session
                            .find_entries(None, &background_context())
                            .await
                            .expect("entries")
                            .is_empty(),
                    );
                    close_session(session.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "messages",
            "preserves every settled assistant stop reason",
            Arc::new(|repo| {
                Box::pin(async move {
                    let session = create_repo_session(repo, Some("session"), None).await;
                    let messages: Vec<crate::types::AgentMessage> = SETTLED_STOP_REASONS
                        .iter()
                        .map(|reason| assistant_message(reason))
                        .collect();
                    let branch = session
                        .create_branch("main", None, &background_context())
                        .await
                        .expect("create branch");
                    let mut ids: Vec<String> = Vec::new();
                    for message in &messages {
                        ids.push(
                            branch
                                .append_message(message.clone(), &background_context())
                                .await
                                .expect("append"),
                        );
                    }

                    let entries = session
                        .find_entries(
                            Some(&crate::harness::session::types::EntryQuery {
                                order: Some(crate::harness::session::types::EntryScanOrder::Asc),
                                kind: Some(crate::harness::session::types::EntryType::Message),
                                ..Default::default()
                            }),
                            &background_context(),
                        )
                        .await
                        .expect("entries");
                    assert_eq!(
                        entries
                            .iter()
                            .map(|entry| entry.id().to_owned())
                            .collect::<Vec<_>>(),
                        ids,
                    );
                    for (index, entry) in entries.iter().enumerate() {
                        let Entry::Message { body, .. } = entry else {
                            panic!("Expected message entry");
                        };
                        assert_eq!(body.message.clone(), messages[index]);
                    }
                    assert_eq!(
                        get_branch_tip(session.as_ref(), "main").await.expect("tip"),
                        ids.last().cloned(),
                    );
                    close_session(session.as_ref()).await;
                })
            }),
        ),
    ]
}

/// Creates fork-content cases that do not require concurrent repository
/// coordination, upstream's `createSessionRepoForkBehaviorConformance`.
#[must_use]
///
/// # Panics
/// A case body panics on its first broken assertion, upstream's `node:assert` throw.
pub fn create_session_repo_fork_behavior_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    vec![
        repo_case(
            factory,
            "forks",
            "tree-forks a fresh session before first attachment",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let fork = repo
                        .fork(source.metadata(), &tree_fork("fork"), &background_context())
                        .await
                        .expect("fork");

                    assert_eq!(fork.metadata().id, "fork");
                    assert_eq!(fork.metadata().parent_session_id.as_deref(), Some("source"));
                    assert!(
                        fork.branch("main", &background_context())
                            .await
                            .expect("branch")
                            .is_none(),
                    );
                    assert_session_value_absent(fork.as_ref(), &lane_config("main").address).await;
                    assert_session_value_absent(fork.as_ref(), &lane_state("main").address).await;
                    assert!(
                        fork.find_entries(None, &background_context())
                            .await
                            .expect("entries")
                            .is_empty(),
                    );
                    assert_eq!(
                        fork.get_stats(&background_context()).await.expect("stats"),
                        crate::harness::session::types::SessionStats {
                            message_count: 0,
                            usage: zero_usage(),
                        },
                    );
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "forks",
            "rejects a data-only branch and releases its destination id",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    source
                        .create_branch("data", None, &background_context())
                        .await
                        .expect("create branch");

                    let rejected = repo
                        .fork(
                            source.metadata(),
                            &branch_fork("data", None, None, Some("destination")),
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the data-only branch fork to reject"
                    );
                    assert_eq!(
                        repo.list(&background_context())
                            .await
                            .expect("list")
                            .into_iter()
                            .map(|metadata| metadata.id)
                            .collect::<Vec<_>>(),
                        ["source"],
                    );

                    let destination = repo
                        .create(
                            crate::harness::session::types::SessionCreateOptions {
                                id: Some("destination".to_owned()),
                                ..Default::default()
                            },
                            &background_context(),
                        )
                        .await
                        .expect("create destination");
                    close_session(source.as_ref()).await;
                    close_session(destination.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "forks",
            "forks one named configured branch with scoped values and a zero ledger",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    source
                        .mutate(
                            commit_writes(vec![
                                seed_custom_entry(ROOT_ID, None, "root"),
                                insert_entry_write(user_entry_fixture(CHILD_ID, Some(ROOT_ID.to_owned()), "child")),
                                seed_custom_entry(SIBLING_ID, Some(ROOT_ID), "sibling"),
                                tip_write("main", Some(SIBLING_ID)),
                                tip_write("review", Some(CHILD_ID)),
                                lane_config_write("review", &configuration()),
                                lane_state_write("review", &occupied_lane_state()),
                                Write::ValueSet(set_value_write(
                                    &operation_result("previous"),
                                    OperationResultRecord {
                                        operation_id: "previous".to_owned(),
                                        kind: crate::harness::session::types::OperationKind::Navigation,
                                        status: crate::harness::session::types::TerminalStatus::Completed,
                                        error: None,
                                        from_tip_id: Some(ROOT_ID.to_owned()),
                                        tip_id: Some(CHILD_ID.to_owned()),
                                        started_at: 1,
                                        ended_at: 2,
                                    },
                                )
                                .expect("write")),
                                Write::ValueSet(set_value_write(&session_name(), "source name".to_owned()).expect("write")),
                                Write::ValueSet(set_value_write(&application_value(), json!({ "copied": false })).expect("write")),
                                Write::ListAppend(append_list_write(&application_list(), json!({ "copied": false })).expect("write")),
                                label_write(ROOT_ID, "root label"),
                                label_write(SIBLING_ID, "sibling label"),
                                Write::ValueSet(set_value_write(
                                    &pending_entry(PENDING_ID),
                                    crate::harness::session::types::PendingEntry::Custom {
                                        custom_type: "pending".to_owned(),
                                        payload: None,
                                    },
                                )
                                .expect("write")),
                                Write::ValueSet(set_value_write(
                                    &operation_meta(OPERATION_ID),
                                    OperationMeta {
                                        operation_id: OPERATION_ID.to_owned(),
                                        lane: "review".to_owned(),
                                        source_tip_id: Some(CHILD_ID.to_owned()),
                                        started_at: 1,
                                        intent: crate::harness::session::types::OperationIntent::Compaction {
                                            custom_instructions: None,
                                        },
                                    },
                                )
                                .expect("write")),
                                Write::ValueSet(set_value_write(
                                    &operation_state(OPERATION_ID),
                                    summary_deciding_state(),
                                )
                                .expect("write")),
                                Write::ValueSet(set_value_write(
                                    &operation_tool_args(OPERATION_ID, ROOT_ID, 0),
                                    json!({ "argument": true })
                                        .as_object()
                                        .cloned()
                                        .expect("tool args object"),
                                )
                                .expect("write")),
                                Write::ValueSet(set_value_write(
                                    &operation_preparation(OPERATION_ID, OPERATION_ID),
                                    compaction_preparation(),
                                )
                                .expect("write")),
                                insert_usage_write(USAGE_ID, settled_usage(), true, None),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("seed commit");

                    let fork = repo
                        .fork(
                            source.metadata(),
                            &branch_fork(
                                "review",
                                Some(CHILD_ID),
                                Some(ForkPosition::At),
                                Some("fork"),
                            ),
                            &background_context(),
                        )
                        .await
                        .expect("fork");

                    assert_eq!(
                        find_entry_ids(
                            &*fork,
                            &crate::harness::session::types::EntryQuery {
                                order: Some(crate::harness::session::types::EntryScanOrder::Asc),
                                ..Default::default()
                            }
                        )
                        .await,
                        [ROOT_ID, CHILD_ID],
                    );
                    assert!(
                        fork.branch("main", &background_context())
                            .await
                            .expect("branch")
                            .is_none(),
                    );
                    assert_eq!(
                        get_branch_tip(fork.as_ref(), "review").await.expect("tip"),
                        Some(CHILD_ID.to_owned()),
                    );
                    assert_lane_config(fork.as_ref(), "review", &configuration()).await;
                    assert_lane_state(fork.as_ref(), "review", &idle_lane_state()).await;
                    assert_session_value_absent(fork.as_ref(), &lane_config("main").address).await;
                    assert_session_value_absent(fork.as_ref(), &lane_state("main").address).await;
                    assert_eq!(
                        fork.get_name(&background_context()).await.expect("name"),
                        Some("source name".to_owned()),
                    );
                    assert_session_value_absent(fork.as_ref(), &application_value().address).await;
                    assert!(
                        fork.read_list(&application_list().address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                    assert_eq!(
                        fork.get_label(ROOT_ID, &background_context())
                            .await
                            .expect("label"),
                        Some("root label".to_owned()),
                    );
                    assert!(
                        fork.get_label(SIBLING_ID, &background_context())
                            .await
                            .expect("label")
                            .is_none(),
                    );
                    assert!(
                        fork.get_value(
                            &operation_result("previous").address,
                            &background_context()
                        )
                        .await
                        .expect("value")
                        .is_none(),
                    );
                    assert_session_value_absent(fork.as_ref(), &pending_entry(PENDING_ID).address)
                        .await;
                    assert!(
                        fork.get_value(
                            &operation_meta(OPERATION_ID).address,
                            &background_context()
                        )
                        .await
                        .expect("value")
                        .is_none(),
                    );
                    assert!(
                        fork.get_value(
                            &operation_state(OPERATION_ID).address,
                            &background_context()
                        )
                        .await
                        .expect("value")
                        .is_none(),
                    );
                    assert!(
                        fork.get_value(
                            &operation_tool_args(OPERATION_ID, ROOT_ID, 0).address,
                            &background_context()
                        )
                        .await
                        .expect("value")
                        .is_none(),
                    );
                    assert!(
                        fork.get_value(
                            &operation_preparation(OPERATION_ID, OPERATION_ID).address,
                            &background_context()
                        )
                        .await
                        .expect("value")
                        .is_none(),
                    );
                    let stats = fork.get_stats(&background_context()).await.expect("stats");
                    assert_eq!(stats.message_count, 1);
                    assert_eq!(stats.usage, zero_usage());
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "forks",
            "enforces branch ancestry for at and before placement",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    source
                        .mutate(
                            commit_writes(vec![
                                seed_custom_entry(ROOT_ID, None, "root"),
                                seed_custom_entry(CHILD_ID, Some(ROOT_ID), "child"),
                                seed_custom_entry(SIBLING_ID, Some(ROOT_ID), "sibling"),
                                tip_write("main", Some(CHILD_ID)),
                                lane_config_write("main", &configuration()),
                                lane_state_write("main", &idle_lane_state()),
                                Write::ValueSet(
                                    set_value_write(&branch_tip("empty"), None).expect("write"),
                                ),
                                lane_config_write("empty", &configuration()),
                                lane_state_write("empty", &idle_lane_state()),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("seed commit");

                    let before = repo
                        .fork(
                            source.metadata(),
                            &branch_fork(
                                "main",
                                Some(CHILD_ID),
                                Some(ForkPosition::Before),
                                Some("before"),
                            ),
                            &background_context(),
                        )
                        .await
                        .expect("fork before");
                    assert_eq!(
                        get_branch_tip(before.as_ref(), "main").await.expect("tip"),
                        Some(ROOT_ID.to_owned())
                    );
                    assert_eq!(
                        find_entry_ids(before.as_ref(), &asc_query()).await,
                        [ROOT_ID],
                    );

                    let mid = repo
                        .fork(
                            source.metadata(),
                            &branch_fork(
                                "main",
                                Some(ROOT_ID),
                                Some(ForkPosition::At),
                                Some("mid"),
                            ),
                            &background_context(),
                        )
                        .await
                        .expect("fork mid");
                    assert_eq!(
                        get_branch_tip(mid.as_ref(), "main").await.expect("tip"),
                        Some(ROOT_ID.to_owned())
                    );
                    assert_eq!(
                        find_entry_ids(
                            &*mid,
                            &crate::harness::session::types::EntryQuery {
                                order: Some(crate::harness::session::types::EntryScanOrder::Asc),
                                ..Default::default()
                            }
                        )
                        .await,
                        [ROOT_ID],
                    );

                    let before_root = repo
                        .fork(
                            source.metadata(),
                            &branch_fork(
                                "main",
                                Some(ROOT_ID),
                                Some(ForkPosition::Before),
                                Some("before-root"),
                            ),
                            &background_context(),
                        )
                        .await
                        .expect("fork before root");
                    assert_eq!(
                        get_branch_tip(before_root.as_ref(), "main")
                            .await
                            .expect("tip"),
                        None
                    );
                    assert!(
                        before_root
                            .find_entries(
                                Some(&crate::harness::session::types::EntryQuery {
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Asc
                                    ),
                                    ..Default::default()
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("entries")
                            .is_empty(),
                    );

                    let empty =
                        fork_branch_session(repo, source.as_ref(), "empty", None, Some("empty"))
                            .await;
                    assert_eq!(
                        get_branch_tip(empty.as_ref(), "empty").await.expect("tip"),
                        None
                    );
                    assert!(
                        empty
                            .find_entries(
                                Some(&crate::harness::session::types::EntryQuery {
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Asc
                                    ),
                                    ..Default::default()
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("entries")
                            .is_empty(),
                    );

                    for (id, branch_name, entry_id) in [
                        ("off-branch", "main", SIBLING_ID),
                        ("unknown", "main", UNKNOWN_ID),
                        ("null-tip", "empty", ROOT_ID),
                    ] {
                        let rejected = repo
                            .fork(
                                source.metadata(),
                                &crate::harness::session::types::ForkOptions::Branch {
                                    branch: branch_name.to_owned(),
                                    entry_id: Some(entry_id.to_owned()),
                                    position: None,
                                    id: Some(id.to_owned()),
                                },
                                &background_context(),
                            )
                            .await;
                        assert!(rejected.is_err(), "expected the {id} fork to reject");
                    }
                    let mut listed = repo
                        .list(&background_context())
                        .await
                        .expect("list")
                        .into_iter()
                        .map(|metadata| metadata.id)
                        .collect::<Vec<_>>();
                    listed.sort();
                    assert_eq!(listed, ["before", "before-root", "empty", "mid", "source"]);
                    for session in [source, before, before_root, empty, mid] {
                        close_session(session.as_ref()).await;
                    }
                })
            }),
        ),
        repo_case(
            factory,
            "forks",
            "forks a closed source session",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    source
                        .mutate(
                            commit_writes(vec![
                                seed_custom_entry(ROOT_ID, None, "root"),
                                tip_write("main", Some(ROOT_ID)),
                                lane_config_write("main", &configuration()),
                                lane_state_write("main", &idle_lane_state()),
                                Write::ValueSet(
                                    set_value_write(&application_value(), json!("excluded"))
                                        .expect("write"),
                                ),
                                Write::ListAppend(
                                    append_list_write(&application_list(), json!("excluded"))
                                        .expect("write"),
                                ),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("seed commit");
                    close_session(source.as_ref()).await;

                    let fork =
                        fork_branch_session(repo, source.as_ref(), "main", None, Some("fork"))
                            .await;
                    assert_eq!(
                        get_branch_tip(fork.as_ref(), "main").await.expect("tip"),
                        Some(ROOT_ID.to_owned())
                    );
                    assert_lane_config(fork.as_ref(), "main", &configuration()).await;
                    assert_lane_state(fork.as_ref(), "main", &idle_lane_state()).await;
                    assert_session_value_absent(fork.as_ref(), &application_value().address).await;
                    assert!(
                        fork.read_list(&application_list().address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                    close_session(fork.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "forks",
            "forks the whole configured tree with fresh lane state",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    source
                        .mutate(
                            commit_writes(vec![
                                seed_custom_entry(ROOT_ID, None, "root"),
                                seed_custom_entry(CHILD_ID, Some(ROOT_ID), "child"),
                                seed_custom_entry(SIBLING_ID, Some(ROOT_ID), "sibling"),
                                tip_write("main", Some(CHILD_ID)),
                                lane_config_write("main", &configuration()),
                                lane_state_write("main", &idle_lane_state()),
                                Write::ValueSet(
                                    set_value_write(
                                        &branch_tip("review"),
                                        Some(SIBLING_ID.to_owned()),
                                    )
                                    .expect("write"),
                                ),
                                lane_config_write("review", &configuration()),
                                lane_state_write("review", &idle_lane_state()),
                                tip_write("notes", Some(ROOT_ID)),
                                Write::ValueSet(
                                    set_value_write(
                                        &application_value(),
                                        json!({ "copied": true }),
                                    )
                                    .expect("write"),
                                ),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("seed commit");

                    let fork = repo
                        .fork(source.metadata(), &tree_fork("fork"), &background_context())
                        .await
                        .expect("fork");

                    assert_eq!(
                        find_entry_ids(
                            &*fork,
                            &crate::harness::session::types::EntryQuery {
                                order: Some(crate::harness::session::types::EntryScanOrder::Asc),
                                ..Default::default()
                            }
                        )
                        .await,
                        [ROOT_ID, CHILD_ID, SIBLING_ID],
                    );
                    assert_eq!(
                        get_branch_tip(fork.as_ref(), "main").await.expect("tip"),
                        Some(CHILD_ID.to_owned())
                    );
                    assert_eq!(
                        get_branch_tip(fork.as_ref(), "review").await.expect("tip"),
                        Some(SIBLING_ID.to_owned())
                    );
                    assert_eq!(
                        get_branch_tip(fork.as_ref(), "notes").await.expect("tip"),
                        Some(ROOT_ID.to_owned())
                    );
                    assert_session_value_absent(fork.as_ref(), &lane_config("notes").address).await;
                    assert_session_value_absent(fork.as_ref(), &lane_state("notes").address).await;
                    assert_lane_config(fork.as_ref(), "review", &configuration()).await;
                    assert_lane_state(fork.as_ref(), "review", &idle_lane_state()).await;
                    assert_eq!(
                        fork.get_value(&application_value().address, &background_context())
                            .await
                            .expect("value")
                            .expect("value")
                            .value,
                        json!({ "copied": true }),
                    );
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "forks",
            "rejects only surviving unknown reserved scalar state",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    source
                        .create_branch("main", None, &background_context())
                        .await
                        .expect("create branch");
                    source
                        .mutate(
                            commit_writes(vec![
                                lane_config_write("main", &configuration()),
                                lane_state_write("main", &idle_lane_state()),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("seed commit");

                    for namespace in ["pi", "pi.unknown"] {
                        let address = generic_value(namespace, "");
                        source
                            .set_value(&address.address, json!(true), &background_context())
                            .await
                            .expect("set value");
                        let rejected = repo
                            .fork(source.metadata(), &tree_fork("tree"), &background_context())
                            .await;
                        assert!(
                            rejected.is_err(),
                            "expected the tree fork to reject {namespace}"
                        );
                        let rejected = repo
                            .fork(
                                source.metadata(),
                                &branch_fork("main", None, None, Some("branch")),
                                &background_context(),
                            )
                            .await;
                        assert!(
                            rejected.is_err(),
                            "expected the branch fork to reject {namespace}"
                        );
                        source
                            .delete_value(&address.address, &background_context())
                            .await
                            .expect("delete value");
                    }
                    close_session(source.as_ref()).await;

                    let tree = repo
                        .fork(source.metadata(), &tree_fork("tree"), &background_context())
                        .await
                        .expect("tree fork");
                    let branch =
                        fork_branch_session(repo, source.as_ref(), "main", None, Some("branch"))
                            .await;
                    close_session(tree.as_ref()).await;
                    close_session(branch.as_ref()).await;
                })
            }),
        ),
    ]
}

/// Creates every fork coordination case, upstream's
/// `createSessionRepoForkCoordinationConformance`.
#[must_use]
pub fn create_session_repo_fork_coordination_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    [
        create_session_repo_fork_destination_reservation_conformance(factory),
        create_session_repo_fork_source_snapshot_conformance(factory),
    ]
    .concat()
}

/// The fork-point fork the streaming-fork cases issue: the optional source
/// close, the branch fork at the entry, and the tip check, upstream's
/// `if (sourceState === "closed") ...; fork = await repo.fork(...); assert tip`.
///
/// # Panics
/// When the fork fails or the tip differs.
async fn fork_at_fork_point(
    repo: &dyn SessionRepo,
    source: &dyn Session,
    branch: &str,
    entry_id: &str,
    close_first: bool,
) -> Box<dyn Session> {
    if close_first {
        close_session(source).await;
    }
    let fork = fork_branch_session(repo, source, branch, Some(entry_id), None).await;
    assert_eq!(
        get_branch_tip(fork.as_ref(), branch).await.expect("tip"),
        Some(entry_id.to_owned()),
    );
    fork
}

/// Creates fork cases that require destination reservation across create and
/// fork, upstream's
/// `createSessionRepoForkDestinationReservationConformance`.
#[must_use]
///
/// # Panics
/// A case body panics on its first broken assertion, upstream's `node:assert` throw.
pub fn create_session_repo_fork_destination_reservation_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    vec![
        repo_case(
            factory,
            "fork coordination",
            "publishes create when it reserves a shared destination id first",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let create = repo.create(
                        crate::harness::session::types::SessionCreateOptions {
                            id: Some("destination".to_owned()),
                            ..Default::default()
                        },
                        &background_context(),
                    );
                    let fork = repo.fork(
                        source.metadata(),
                        &tree_fork("destination"),
                        &background_context(),
                    );
                    let (created, forked) = tokio::join!(create, fork);

                    let created = created.expect("create must win the reservation");
                    assert!(forked.is_err(), "expected the fork to lose the reservation");
                    close_session(created.as_ref()).await;
                    close_session(source.as_ref()).await;
                })
            }),
        ),
        repo_case(
            factory,
            "fork coordination",
            "publishes fork when it reserves a shared destination id first",
            Arc::new(|repo| {
                Box::pin(async move {
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let fork = repo.fork(
                        source.metadata(),
                        &tree_fork("destination"),
                        &background_context(),
                    );
                    let create = repo.create(
                        crate::harness::session::types::SessionCreateOptions {
                            id: Some("destination".to_owned()),
                            ..Default::default()
                        },
                        &background_context(),
                    );
                    let (forked, created) = tokio::join!(fork, create);

                    let forked = forked.expect("fork must win the reservation");
                    assert!(
                        created.is_err(),
                        "expected the create to lose the reservation"
                    );
                    close_session(forked.as_ref()).await;
                    close_session(source.as_ref()).await;
                })
            }),
        ),
    ]
}

/// Creates fork cases that require a snapshot boundary on an active source
/// storage queue, upstream's
/// `createSessionRepoForkSourceSnapshotConformance`.
#[must_use]
///
/// # Panics
/// A case body panics on its first broken assertion, upstream's `node:assert` throw.
pub fn create_session_repo_fork_source_snapshot_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    vec![repo_case(
        factory,
        "fork coordination",
        "captures one coherent boundary between source commits",
        Arc::new(|repo| {
            Box::pin(async move {
                let source = create_repo_session(repo, Some("source"), None).await;
                let first_mutation = source
                    .begin_mutation(&background_context())
                    .await
                    .expect("begin");
                let first_commit = first_mutation.commit(
                    vec![
                        seed_custom_entry(ROOT_ID, None, "first"),
                        tip_write("main", Some(ROOT_ID)),
                        lane_config_write("main", &configuration()),
                        lane_state_write("main", &idle_lane_state()),
                        Write::ValueSet(
                            set_value_write(&session_name(), "first name".to_owned())
                                .expect("write"),
                        ),
                        Write::ValueSet(
                            set_value_write(&entry_label(ROOT_ID), "first label".to_owned())
                                .expect("write"),
                        ),
                    ],
                    &background_context(),
                );
                let fork = repo.fork(
                    source.metadata(),
                    &branch_fork("main", None, None, Some("fork")),
                    &background_context(),
                );
                let second_commit = source.mutate(
                    commit_writes(vec![
                        seed_custom_entry(CHILD_ID, Some(ROOT_ID), "second"),
                        tip_write("main", Some(CHILD_ID)),
                        Write::ValueSet(
                            set_value_write(&session_name(), "second name".to_owned())
                                .expect("write"),
                        ),
                        Write::ValueSet(
                            set_value_write(&entry_label(ROOT_ID), "second label".to_owned())
                                .expect("write"),
                        ),
                    ]),
                    &background_context(),
                );

                let (_, forked) = tokio::join!(first_commit, fork);
                let forked = forked.expect("fork between commits");
                first_mutation
                    .end(&background_context())
                    .await
                    .expect("end");
                second_commit.await.expect("second commit");
                assert_eq!(
                    get_branch_tip(forked.as_ref(), "main").await.expect("tip"),
                    Some(ROOT_ID.to_owned())
                );
                assert_eq!(
                    find_entry_ids(forked.as_ref(), &asc_query()).await,
                    [ROOT_ID],
                );
                assert_eq!(
                    forked.get_name(&background_context()).await.expect("name"),
                    Some("first name".to_owned()),
                );
                assert_eq!(
                    forked
                        .get_label(ROOT_ID, &background_context())
                        .await
                        .expect("label"),
                    Some("first label".to_owned()),
                );
                close_session(source.as_ref()).await;
                close_session(forked.as_ref()).await;
            })
        }),
    )]
}

/// Creates every fork conformance case, upstream's
/// `createSessionRepoForkConformance`.
#[must_use]
pub fn create_session_repo_fork_conformance(factory: &RepoFixtureFactory) -> Vec<ConformanceCase> {
    [
        create_session_repo_fork_behavior_conformance(factory),
        create_session_repo_fork_coordination_conformance(factory),
    ]
    .concat()
}

/// The subset of fork cases the streaming backends enable while SQLite's
/// streaming fork implementation is pending, upstream's
/// `createSessionRepoStreamingForkConformance`.
#[must_use]
pub fn create_session_repo_streaming_fork_conformance(
    factory: &RepoFixtureFactory,
) -> Vec<ConformanceCase> {
    // Upstream merges this into the fork conformance once SQLite supports
    // these cases; the port carries the same split for the JSONL/SQLite
    // children to land the same way.
    [
        fork_application_list_conformance(factory),
        branch_fork_application_state_conformance(factory),
        fork_lane_validation_conformance(factory),
    ]
    .concat()
}

fn fork_lane_validation_conformance(factory: &RepoFixtureFactory) -> Vec<ConformanceCase> {
    vec![repo_case(
        factory,
        "fork lane validation",
        "ignores malformed unrelated lanes",
        Arc::new(|repo| {
            Box::pin(async move {
                let source = create_repo_session(repo, Some("source"), None).await;
                source
                    .create_branch("main", None, &background_context())
                    .await
                    .expect("create branch");
                source
                    .mutate(
                        commit_writes(vec![
                            lane_config_write("main", &configuration()),
                            lane_state_write("main", &idle_lane_state()),
                            lane_config_write("unrelated", &configuration()),
                        ]),
                        &background_context(),
                    )
                    .await
                    .expect("seed commit");

                let tree = repo
                    .fork(source.metadata(), &tree_fork("tree"), &background_context())
                    .await
                    .expect("tree fork");
                let branch =
                    fork_branch_session(repo, source.as_ref(), "main", None, Some("branch")).await;
                assert_eq!(
                    get_branch_tip(branch.as_ref(), "main").await.expect("tip"),
                    None
                );
                close_session(source.as_ref()).await;
                close_session(tree.as_ref()).await;
                close_session(branch.as_ref()).await;
            })
        }),
    )]
}

fn fork_application_list_conformance(factory: &RepoFixtureFactory) -> Vec<ConformanceCase> {
    let mut cases = Vec::new();
    for source_state in ["open", "closed"] {
        cases.push(repo_case(
            factory,
            format!("fork application lists ({source_state} source)"),
            "tree fork copies lists at distinct addresses",
            Arc::new(move |repo| {
                Box::pin(async move {
                    // Each list keeps its own contents: keys "" and "other" in one
                    // namespace must not merge. A different namespace also stays
                    // separate; pi2.events is application-owned, not reserved.
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let events = events_list();
                    let sibling = stored_values::generic_list(&events.address.namespace, "other");
                    let other_namespace = stored_values::generic_list("pi2.events", "");
                    let absent = stored_values::generic_list(&events.address.namespace, "absent");
                    source
                        .append_list(&events.address, json!("event"), &background_context())
                        .await
                        .expect("append");
                    source
                        .append_list(&sibling.address, json!("sibling"), &background_context())
                        .await
                        .expect("append");
                    source
                        .append_list(
                            &other_namespace.address,
                            json!("other namespace"),
                            &background_context(),
                        )
                        .await
                        .expect("append");

                    let fork =
                        fork_tree_session(repo, source.as_ref(), "fork", source_state == "closed")
                            .await;

                    assert_session_list_values(fork.as_ref(), &events.address, &[json!("event")])
                        .await;
                    assert_session_list_values(
                        fork.as_ref(),
                        &sibling.address,
                        &[json!("sibling")],
                    )
                    .await;
                    assert_session_list_values(
                        fork.as_ref(),
                        &other_namespace.address,
                        &[json!("other namespace")],
                    )
                    .await;
                    assert!(
                        fork.read_list(&absent.address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ));
        cases.push(repo_case(
            factory,
            format!("fork application lists ({source_state} source)"),
            "tree fork copies only survivors after list deletion and reappend",
            Arc::new(move |repo| {
                Box::pin(async move {
                    // Append old -> delete -> append new must copy only new, even
                    // when changes share a transaction. A list deleted without a
                    // later append must remain empty in the fork.
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let events = events_list();
                    let deleted = stored_values::generic_list(&events.address.namespace, "deleted");
                    source
                        .append_list(&events.address, json!("old"), &background_context())
                        .await
                        .expect("append");
                    source
                        .append_list(&deleted.address, json!("removed"), &background_context())
                        .await
                        .expect("append");
                    source
                        .mutate(
                            commit_writes(vec![
                                Write::ListDelete(stored_values::delete_list(&events)),
                                Write::ListAppend(
                                    append_list_write(&events, json!("temporary")).expect("write"),
                                ),
                                Write::ListDelete(stored_values::delete_list(&events)),
                                Write::ListAppend(
                                    append_list_write(&events, json!("first survivor"))
                                        .expect("write"),
                                ),
                                Write::ListDelete(stored_values::delete_list(&deleted)),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("commit");
                    source
                        .append_list(
                            &events.address,
                            json!("second survivor"),
                            &background_context(),
                        )
                        .await
                        .expect("append");

                    let fork =
                        fork_tree_session(repo, source.as_ref(), "fork", source_state == "closed")
                            .await;

                    assert_session_list_values(
                        fork.as_ref(),
                        &events.address,
                        &[json!("first survivor"), json!("second survivor")],
                    )
                    .await;
                    assert!(
                        fork.read_list(&deleted.address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ));
        cases.push(repo_case(
            factory,
            format!("fork application lists ({source_state} source)"),
            "tree fork preserves list element sequences including gaps",
            Arc::new(move |repo| {
                Box::pin(async move {
                    // Other writes consume sequences too. List elements at seq 2
                    // and 4 must stay at 2 and 4 in the fork, not be renumbered to
                    // 1 and 2.
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let events = events_list();
                    let committed = source
                        .mutate(
                            commit_writes(vec![
                                Write::ValueSet(
                                    set_value_write(&session_name(), "before".to_owned())
                                        .expect("write"),
                                ),
                                Write::ListAppend(
                                    append_list_write(&events, json!("first")).expect("write"),
                                ),
                                Write::ValueSet(
                                    set_value_write(&session_name(), "between".to_owned())
                                        .expect("write"),
                                ),
                                Write::ListAppend(
                                    append_list_write(&events, json!("second")).expect("write"),
                                ),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("commit");

                    let fork =
                        fork_tree_session(repo, source.as_ref(), "fork", source_state == "closed")
                            .await;

                    let committed = downcast_commit_result(committed);
                    assert_eq!(
                        fork.read_list(&events.address, None, &background_context())
                            .await
                            .expect("list"),
                        vec![
                            crate::harness::session::values::ListElement {
                                seq: committed.seqs[1],
                                value: json!("first"),
                            },
                            crate::harness::session::values::ListElement {
                                seq: committed.seqs[3],
                                value: json!("second"),
                            },
                        ],
                    );
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ));
        for order in ["asc", "desc"] {
            let order_enum = if order == "asc" {
                crate::harness::session::types::EntryScanOrder::Asc
            } else {
                crate::harness::session::types::EntryScanOrder::Desc
            };
            cases.push(repo_case(
                factory,
                format!("fork application lists ({source_state} source)"),
                format!("tree fork continues {order} pagination using source cursors"),
                Arc::new(move |repo| {
                    Box::pin(async move {
                        // A cursor from the source must resume at the same place in
                        // the fork. After reading [first, second] ascending, continue
                        // with [third]; descending does the reverse. Continuing after
                        // the final element must return an empty page, not repeat that
                        // element.
                        let source = create_repo_session(repo, Some("source"), None).await;
                        let events = events_list();
                        for item in ["first", "second", "third"] {
                            source
                                .append_list(&events.address, json!(item), &background_context())
                                .await
                                .expect("append");
                        }
                        let first_page = source
                            .read_list(
                                &events.address,
                                Some(stored_values::ListReadOptions {
                                    order: Some(order_enum),
                                    limit: Some(2),
                                    ..Default::default()
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("page");
                        assert_eq!(first_page.len(), 2);
                        let cursor = crate::harness::session::values::ListCursor {
                            seq: first_page[1].seq,
                        };
                        let last_page = source
                            .read_list(
                                &events.address,
                                Some(stored_values::ListReadOptions {
                                    order: Some(order_enum),
                                    cursor: Some(cursor),
                                    limit: Some(2),
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("page");
                        assert_eq!(
                            last_page
                                .iter()
                                .map(|element| element.value.clone())
                                .collect::<Vec<_>>(),
                            [json!(if order == "asc" { "third" } else { "first" })],
                        );
                        let end_cursor = crate::harness::session::values::ListCursor {
                            seq: last_page[0].seq,
                        };

                        if source_state == "closed" {
                            close_session(source.as_ref()).await;
                        }
                        let fork = repo
                            .fork(source.metadata(), &tree_fork("fork"), &background_context())
                            .await
                            .expect("fork");

                        assert_eq!(
                            fork.read_list(
                                &events.address,
                                Some(stored_values::ListReadOptions {
                                    order: Some(order_enum),
                                    limit: Some(2),
                                    ..Default::default()
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("page"),
                            first_page,
                        );
                        assert_eq!(
                            fork.read_list(
                                &events.address,
                                Some(stored_values::ListReadOptions {
                                    order: Some(order_enum),
                                    cursor: Some(cursor),
                                    limit: Some(2),
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("page"),
                            last_page,
                        );
                        assert!(
                            fork.read_list(
                                &events.address,
                                Some(stored_values::ListReadOptions {
                                    order: Some(order_enum),
                                    cursor: Some(end_cursor),
                                    limit: Some(2),
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("page")
                            .is_empty(),
                        );
                        close_session(source.as_ref()).await;
                        close_session(fork.as_ref()).await;
                    })
                }),
            ));
        }
    }
    cases
}

fn branch_fork_application_state_conformance(factory: &RepoFixtureFactory) -> Vec<ConformanceCase> {
    let mut cases = Vec::new();
    for source_state in ["open", "closed"] {
        cases.push(repo_case(
            factory,
            format!("branch fork application state ({source_state} source)"),
            "excludes overwritten and unchanged application values",
            Arc::new(move |repo| {
                Box::pin(async move {
                    // WP08 §1.1: set v1 -> fork point -> overwrite with v2. A branch
                    // fork copies neither version: it cannot reconstruct v1 and must
                    // not copy v2. Even unchanged older values are excluded, so
                    // filtering current values by the fork point's sequence is not
                    // enough.
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let state = generic_value("test.application.state", "");
                    let unchanged = generic_value("test.application.settings", "");
                    let branch = source
                        .create_branch("review", None, &background_context())
                        .await
                        .expect("create branch");
                    source
                        .mutate(
                            commit_writes(vec![
                                lane_config_write("review", &configuration()),
                                lane_state_write("review", &idle_lane_state()),
                                Write::ValueSet(
                                    set_value_write(&state, json!("v1")).expect("write"),
                                ),
                                Write::ValueSet(
                                    set_value_write(&unchanged, json!("predates fork point"))
                                        .expect("write"),
                                ),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("seed commit");
                    let entry_id = branch
                        .append_custom_entry("fork-point", None, &background_context())
                        .await
                        .expect("append");
                    source
                        .set_value(&state.address, json!("v2"), &background_context())
                        .await
                        .expect("set v2");
                    assert_eq!(
                        source
                            .get_value(&state.address, &background_context())
                            .await
                            .expect("value")
                            .expect("state")
                            .value,
                        json!("v2"),
                    );

                    let fork = fork_at_fork_point(
                        repo,
                        source.as_ref(),
                        "review",
                        &entry_id,
                        source_state == "closed",
                    )
                    .await;
                    assert!(
                        fork.get_value(&state.address, &background_context())
                            .await
                            .expect("value")
                            .is_none(),
                    );
                    // A survivor older than the fork point must also be excluded,
                    // not copied by a seq cutoff.
                    assert!(
                        fork.get_value(&unchanged.address, &background_context())
                            .await
                            .expect("value")
                            .is_none(),
                    );
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ));
        cases.push(repo_case(
            factory,
            format!("branch fork application state ({source_state} source)"),
            "excludes deleted/reappended and untouched application lists",
            Arc::new(move |repo| {
                Box::pin(async move {
                    // WP08 §1.1: append old -> fork point -> delete -> append new. A
                    // branch fork copies neither the deleted elements nor the new
                    // ones. A separate untouched list is also excluded, even though
                    // its elements still survive from before the fork point.
                    let source = create_repo_session(repo, Some("source"), None).await;
                    let events = events_list();
                    let untouched =
                        stored_values::generic_list(&events.address.namespace, "untouched");
                    let branch = source
                        .create_branch("review", None, &background_context())
                        .await
                        .expect("create branch");
                    source
                        .mutate(
                            commit_writes(vec![
                                lane_config_write("review", &configuration()),
                                lane_state_write("review", &idle_lane_state()),
                                Write::ListAppend(
                                    append_list_write(&events, json!("old first")).expect("write"),
                                ),
                                Write::ListAppend(
                                    append_list_write(&events, json!("old second")).expect("write"),
                                ),
                                Write::ListAppend(
                                    append_list_write(&untouched, json!("predates fork point"))
                                        .expect("write"),
                                ),
                            ]),
                            &background_context(),
                        )
                        .await
                        .expect("seed commit");
                    let entry_id = branch
                        .append_custom_entry("fork-point", None, &background_context())
                        .await
                        .expect("append");
                    source
                        .delete_list(&events.address, &background_context())
                        .await
                        .expect("delete list");
                    source
                        .append_list(&events.address, json!("new"), &background_context())
                        .await
                        .expect("append");
                    assert_eq!(
                        source
                            .read_list(&events.address, None, &background_context())
                            .await
                            .expect("list")
                            .into_iter()
                            .map(|element| element.value)
                            .collect::<Vec<_>>(),
                        [json!("new")],
                    );

                    let fork = fork_at_fork_point(
                        repo,
                        source.as_ref(),
                        "review",
                        &entry_id,
                        source_state == "closed",
                    )
                    .await;
                    assert!(
                        fork.read_list(&events.address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                    // Even elements still surviving from before the fork point are
                    // excluded.
                    assert!(
                        fork.read_list(&untouched.address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                    close_session(source.as_ref()).await;
                    close_session(fork.as_ref()).await;
                })
            }),
        ));
    }
    cases
}

/// Creates every `SessionRepo` conformance case, upstream's
/// `createSessionRepoConformance`.
#[must_use]
pub fn create_session_repo_conformance(factory: &RepoFixtureFactory) -> Vec<ConformanceCase> {
    [
        create_session_repo_lifecycle_conformance(factory),
        create_session_repo_ownership_conformance(factory),
        create_session_repo_message_conformance(factory),
        create_session_repo_fork_conformance(factory),
    ]
    .concat()
}

/// The user-message entry fixture the repo suites build, upstream's
/// `{ role: "user", content: "child", timestamp: 1 }` entries.
fn user_entry_fixture(
    id: &str,
    parent_id: Option<String>,
    text: &str,
) -> crate::harness::session::types::NewEntry {
    crate::harness::session::types::NewEntry::Message {
        id: id.to_owned(),
        parent_id,
        body: Box::new(crate::harness::session::types::MessageEntry {
            message: crate::types::AgentMessage::Standard(pi_ai::types::Message::User(
                pi_ai::types::UserMessage {
                    content: pi_ai::types::UserContent::Text(text.to_owned()),
                    timestamp: 1,
                },
            )),
            terminate: None,
        }),
    }
}

/// The occupied lane state the configured-branch fork seeds, upstream's
/// `{ currentOperationId, lastOperationId, inbox }` fixture.
fn occupied_lane_state() -> LaneState {
    LaneState {
        current_operation_id: Some(OPERATION_ID.to_owned()),
        last_operation_id: Some("previous".to_owned()),
        inbox: vec![crate::harness::session::types::InboxItem {
            entry_id: PENDING_ID.to_owned(),
            kind: crate::harness::session::types::InboxItemKind::Write,
        }],
    }
}

/// The `summary.deciding` operation state the configured-branch fork seeds,
/// upstream's `{ at: "summary.deciding", ... }` fixture in the port's
/// nested-scope wire shape.
fn summary_deciding_state() -> crate::harness::session::types::OperationState {
    crate::harness::session::types::OperationState::SummaryDeciding(
        crate::harness::session::types::SummaryDecidingOperation {
            scope: crate::harness::session::types::OperationScope {
                control: crate::harness::session::types::Control::Running,
                settings: crate::harness::session::types::RunSettings {
                    compaction: crate::harness::compaction::types::CompactionSettings {
                        enabled: true,
                        reserve_tokens: 1,
                        keep_recent_tokens: 1,
                    },
                    steering_mode: crate::types::QueueMode::All,
                    follow_up_mode: crate::types::QueueMode::All,
                    tool_execution: crate::types::ToolExecutionMode::Sequential,
                },
                latest_assistant_entry_id: None,
            },
            task: crate::harness::session::types::SummaryTask {
                task_id: OPERATION_ID.to_owned(),
                reason: Some(crate::harness::session::types::CompactionReason::Manual),
                custom_instructions: None,
                boundary: crate::harness::session::types::ResultBoundary::Finish,
            },
        },
    )
}

/// The compaction preparation the configured-branch fork seeds, upstream's
/// `{ kind: "compaction", ... }` fixture.
fn compaction_preparation() -> crate::harness::session::types::DurableStructuralPreparation {
    crate::harness::session::types::DurableStructuralPreparation::Compaction {
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: Vec::new(),
        retained_tail: Vec::new(),
        is_split_turn: false,
        tokens_before: 0,
        previous_summary: None,
        file_ops: crate::harness::compaction::types::FileOperations::default(),
        settings: crate::harness::compaction::types::CompactionSettings {
            enabled: true,
            reserve_tokens: 1,
            keep_recent_tokens: 1,
        },
    }
}

fn repo_case<G, N>(
    factory: &Arc<dyn Fn() -> BoxedFuture<'static, RepoFixture> + Send + Sync>,
    group: G,
    name: N,
    test: RepoCaseTest,
) -> ConformanceCase
where
    G: Into<String>,
    N: Into<String>,
{
    let factory = factory.clone();
    ConformanceCase::new(group, name, move || {
        let factory = factory.clone();
        let test = test.clone();
        Box::pin(async move {
            let fixture = factory().await;
            test(fixture.repo.as_ref()).await;
            fixture.dispose().await;
        })
    })
}
