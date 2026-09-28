//! The harness container's boundary cases, the uncovered-arm sweep the
//! coverage gate drove: the fault/closed latch reads and their Display and
//! source chains, the memoized close's second-close route, the mutation
//! catches' three routes (closed wins, caller-error passthrough, fault),
//! the name/label setters' delete arms and their failure surfaces, the
//! watch-session slice stub, the config setters' validation rejections,
//! the `Events`/`Hooks` closed-registry rethrows, the configured-lane
//! restore path, the `createAt` `UnknownTarget` passthrough, the malformed
//! branch-tip fault, and the creation errors' validation and fault halves.
//!
//! Upstream at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` carries no
//! suite pins for most of these arms (its harness tests stay on the happy
//! choreography), so each case pins the arm through the public surface
//! that reaches it and keeps the assertion on the observable outcome only.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;
use pi_ai::utils::retry::RetryPolicy;
use serde_json::json;

use super::common;
use crate::harness::agent_harness::AcquireLaneOptions;
use crate::harness::agent_harness::AgentHarness;
use crate::harness::agent_harness::CreateAt;
use crate::harness::agent_harness::Events;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::{HookOptions, HookResult, LaneOperationError, Resources};
use crate::harness::context::{Context, background_context};
use crate::harness::result::HarnessError;
use crate::harness::runtime::harness::{Harness, HarnessCreationError, create_agent_harness};
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::lane_state_write;
use crate::harness::runtime::test_support::{lock, raw_write, runtime_session, user_text_message};
use crate::harness::runtime::types::lane_error;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::Branch;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryQuery;
use crate::harness::session::types::IdGenerator;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RunSettings;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionMetadata;
use crate::harness::session::types::SessionMutation;
use crate::harness::session::types::SessionMutationCallback;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::SessionStats;
use crate::harness::session::types::{StorageBranchScan, SummaryDecidingOperation, SummaryTask};
use crate::harness::session::values as stored_values;
use crate::harness::session::values::ListAddress;
use crate::harness::session::values::ListElement;
use crate::harness::session::values::ListReadOptions;
use crate::harness::session::values::{StoredValue, ValueAddress, Write, set_value_write};
use crate::harness::types::AgentHarnessTool;
use crate::harness::types::AgentHarnessToolExecuteFn;
use crate::harness::types::AgentHarnessToolInvocation;
use crate::harness::types::{AgentHarnessToolUpdateCallback, ToolContext};
use crate::types::{AgentToolResult, QueueMode, ThinkingLevel, ToolExecutionMode};

/// JavaScript's `Number.MAX_SAFE_INTEGER` plus one, the first value the
/// safe-integer validators reject.
const OVER_SAFE_INTEGER: u64 = 9_007_199_254_740_992;

/// The closed error's message, [`crate::harness::result::HarnessClosed`]'s
/// display the closed latch rethrows.
const CLOSED_MESSAGE: &str = "AgentHarness was closed while the operation was active";

/// The storage-error message the armed commit failures carry.
const COMMIT_FAILED: &str = "commit failed";

/// The harness fault's fixed message, upstream's
/// `"AgentHarness storage or invariant fault"` throw.
const FAULT_MESSAGE: &str = "AgentHarness storage or invariant fault";

/// Builds one declarative tool whose executor never runs; the validation
/// cases reject before any dispatch.
fn named_tool(name: &str) -> AgentHarnessTool {
    AgentHarnessTool {
        tool: pi_ai::types::Tool {
            name: name.to_owned(),
            description: name.to_owned(),
            parameters: json!({ "type": "object" }),
            constrained_sampling: None,
        },
        label: name.to_owned(),
        prepare_arguments: None,
        execute: never_execute(),
        replay: None,
        execution_mode: None,
    }
}

/// The never-executed executor, the erased execute surface the tool
/// literal requires.
fn never_execute() -> Arc<AgentHarnessToolExecuteFn> {
    Arc::new(
        |_tool_call_id: &str,
         _args: &serde_json::Value,
         _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
         _tool_context: ToolContext,
         _invocation: &dyn AgentHarnessToolInvocation,
         _context: &Context| {
            Box::pin(async {
                Ok(AgentToolResult {
                    content: Vec::new(),
                    details: json!({}),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    )
}

/// Opens one session over a controlled memory backend and records it in
/// the ledger; the controlled storage arms the commit hooks and failures
/// the mutation-catch cases drive.
fn controlled_session(
    sessions: &mut common::SessionLedger,
) -> (Arc<StorageBackedSession>, Arc<ControlledStorage>) {
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(runtime_session(
        format!("harness-boundary-{}", sessions.len()),
        storage.clone(),
    ));
    sessions.push(Arc::clone(&session));
    (session, storage)
}

/// The session whose `mutate` returns a foreign payload once armed: the
/// port's erased `Any` payload is a live contract a foreign `Session`
/// implementation can breach (upstream's typed `mutate<T>` carries no such
/// surface), and the mutation line's rejection arm binds through it. A
/// second arm resolves `mutate` without running the callback at all, the
/// only surface that reaches the not-published guard upstream and here.
struct ForeignMutationPayloadSession {
    /// The real session every other method forwards to.
    inner: Arc<StorageBackedSession>,
    /// Whether `mutate` returns the foreign payload.
    foreign: AtomicBool,
    /// Whether `mutate` resolves without running the callback.
    skip_callback: AtomicBool,
}

/// The payload marker the armed `mutate` returns instead of the mutation
/// outcome: a type the mutation line's downcast cannot recognize.
struct ForeignPayload;

impl ForeignMutationPayloadSession {
    /// Arms the foreign payload for the later mutations.
    fn arm_foreign_payload(&self) {
        self.foreign.store(true, Ordering::SeqCst);
    }

    /// Arms the skip-callback resolution for the later mutations.
    fn arm_skip_callback(&self) {
        self.skip_callback.store(true, Ordering::SeqCst);
    }
}

impl SessionReader for ForeignMutationPayloadSession {
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        self.inner.get_entries(ids, context)
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        self.inner.get_stats(context)
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        self.inner.get_value(address, context)
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        self.inner.scan_values(prefix, context)
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        self.inner.read_list(address, options, context)
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.inner.scan_branch(query, context)
    }
}

impl Session for ForeignMutationPayloadSession {
    fn metadata(&self) -> &SessionMetadata {
        self.inner.metadata()
    }

    fn id_generator(&self) -> &dyn IdGenerator {
        self.inner.id_generator()
    }

    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        self.inner.get_entry(id, context)
    }

    fn get_name(&self, context: &Context) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        self.inner.get_name(context)
    }

    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        self.inner.get_label(target_id, context)
    }

    fn find_entries(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.inner.find_entries(query, context)
    }

    fn find_entry(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        self.inner.find_entry(query, context)
    }

    fn branch(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Box<dyn Branch>>, SessionError>> {
        self.inner.branch(name, context)
    }

    fn create_branch(
        &self,
        name: &str,
        at: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Branch>, SessionError>> {
        self.inner.create_branch(name, at, context)
    }

    fn begin_mutation(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn SessionMutation>, SessionError>> {
        self.inner.begin_mutation(context)
    }

    fn mutate(
        &self,
        mutation: SessionMutationCallback,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
        if self.skip_callback.load(Ordering::SeqCst) {
            // The publication the callback would have carried, resolved
            // without the callback: upstream's `mutate` resolving without
            // publishing the lane, the not-published guard's only route.
            let payload = crate::harness::runtime::types::any_payload(
                crate::harness::runtime::harness::MutationOutcome::Published { delivery: None },
            );
            return Box::pin(std::future::ready(Ok(payload)));
        }
        if !self.foreign.load(Ordering::SeqCst) {
            return self.inner.mutate(mutation, context);
        }
        let payload: Box<dyn Any + Send> = Box::new(ForeignPayload);
        Box::pin(std::future::ready(Ok(payload)))
    }

    fn set_value(
        &self,
        address: &ValueAddress,
        next: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.inner.set_value(address, next, context)
    }

    fn delete_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.inner.delete_value(address, context)
    }

    fn append_list(
        &self,
        address: &ListAddress,
        element: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.inner.append_list(address, element, context)
    }

    fn delete_list(
        &self,
        address: &ListAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.inner.delete_list(address, context)
    }

    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.inner.set_name(name, context)
    }

    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.inner.set_label(target_id, label, context)
    }

    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.inner.close(context)
    }
}

/// Commits the configured `main` lane's durable values with one tip entry,
/// the restore-path seed.
async fn seed_configured_lane(session: &Arc<StorageBackedSession>, tip: Option<&str>) {
    let mut writes: Vec<Write> = vec![
        set_value_write(&stored_values::branch_tip("main"), tip.map(str::to_owned))
            .expect("the tip write"),
        set_value_write(&stored_values::lane_config("main"), common::configured())
            .expect("the config write"),
        lane_state_write("main", None, None, Vec::new()).expect("the lane state write"),
    ];
    if let Some(tip) = tip {
        writes.push(Write::Entry(Box::new(insert_entry(NewEntry::Message {
            id: tip.to_owned(),
            parent_id: None,
            body: Box::new(MessageEntry {
                message: user_text_message("history"),
                terminate: None,
            }),
        }))));
    }
    commit_writes(session, writes).await;
}

/// Upstream pin: none (the arms are the poison pill's latch reads; the
/// upstream suite stays on the happy choreography). Faults the harness
/// once, re-faults, closes, re-faults, and reads the fault's Display and
/// source chain: the latch returns the stored error object and the boxed
/// cause rides the chain.
#[tokio::test]
async fn fault_latch_returns_the_stored_error_and_its_cause_chain_holds() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();

    assert!(
        !harness.session().metadata().id.is_empty(),
        "the session accessor serves the attached session"
    );
    assert!(
        harness.models().model("faux", "faux-1").is_some(),
        "the models runtime resolves the seeded provider model"
    );

    let first = harness.fault(
        lane_error(SessionError::Message("boom".to_owned())),
        &context,
    );
    assert_eq!(first.to_string(), FAULT_MESSAGE, "the fault message");
    let cause = std::error::Error::source(first.as_ref()).expect("the fault carries its cause");
    assert_eq!(cause.to_string(), "boom", "the boxed cause displays");
    assert!(
        std::error::Error::source(cause).is_none(),
        "the message cause ends the chain"
    );

    let second = harness.fault(
        lane_error(SessionError::Message("other".to_owned())),
        &context,
    );
    assert!(
        Arc::ptr_eq(&first, &second),
        "the re-fault returns the latched error object"
    );

    common::close_sessions(&mut sessions).await;

    // The closed latch answers only while the fault latch is empty: the
    // fault read precedes the closed read, upstream's `faultError`-first
    // `assertOpen` order.
    let mut sessions = common::SessionLedger::new();
    let closed_harness = common::create_harness(&mut sessions).await;
    closed_harness
        .close(&background_context())
        .await
        .expect("the close settles");
    let after_close = closed_harness.fault(
        lane_error(SessionError::Message("late".to_owned())),
        &background_context(),
    );
    assert_eq!(
        after_close.to_string(),
        CLOSED_MESSAGE,
        "the closed latch wins the fault race"
    );
    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's `closePromise` memoization is exercised
/// only through concurrent closes; the port binds the memoized route
/// directly). Closes twice sequentially: the loser claims nothing and
/// awaits the winner's memoized completion.
#[tokio::test]
async fn close_memoizes_one_completion_across_closes() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();

    let first = harness.close(&context);
    first.await.expect("the first close settles");
    let second = harness.close(&context);
    second.await.expect("the memoized close settles");

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` (implicit): a configured durable lane restores through
/// the acquisition mutation when the container starts empty — the
/// `ClassifiedLaneStorage::Lane` restore path the `Harness::new` seed
/// bypasses.
#[tokio::test]
async fn lane_acquisition_restores_a_configured_lane_from_durable_state() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    seed_configured_lane(&session, Some("e1")).await;
    let (options, _faux) = common::harness_options(Arc::clone(&session));
    let harness = Harness::new(options, common::configured(), BTreeMap::new());
    let context = background_context();

    let lane = harness
        .lane("main", &context)
        .await
        .expect("the lane restores");
    let second = harness
        .lane("main", &context)
        .await
        .expect("the lane returns");
    assert_eq!(lane.name(), "main", "the restored lane carries its name");
    let tip = lane.get_tip_id(&context).await.expect("the tip read");
    assert_eq!(
        tip.as_deref(),
        Some("e1"),
        "the restored lane adopts the stored tip"
    );
    assert!(
        Arc::ptr_eq(&lane, &second),
        "the same instance returns for every acquisition"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (the invariant is the port's typed parse of the
/// stored tip; upstream's restore rejects the same shape). A malformed
/// branch tip faults the acquisition, latches the fault, and every later
/// operation rethrows it.
#[tokio::test]
async fn malformed_branch_tip_faults_acquisition_and_latches_the_fault() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    commit_writes(
        &session,
        vec![raw_write(
            &stored_values::branch_tip("main").address,
            json!(42),
        )],
    )
    .await;
    let (options, _faux) = common::harness_options(session);
    let harness = Harness::new(options, common::configured(), BTreeMap::new());
    let context = background_context();

    let Err(error) = harness.lane("main", &context).await else {
        panic!("the malformed tip faults the acquisition");
    };
    assert_eq!(
        error.to_string(),
        FAULT_MESSAGE,
        "the catch faults the acquisition",
    );
    let cause = std::error::Error::source(error.as_ref()).expect("the fault carries its cause");
    assert!(
        cause.to_string().contains("tip is malformed"),
        "the parse invariant rides the cause chain: {cause}",
    );
    let Err(latched) = harness.lane("main", &context).await else {
        panic!("the fault latches");
    };
    assert_eq!(
        latched.to_string(),
        FAULT_MESSAGE,
        "later acquisitions rethrow the latched fault"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: the `createAt` target validation's error half. The
/// `UnknownTarget` caller error passes through the catch without
/// faulting: the harness stays open afterwards.
#[tokio::test]
async fn unknown_target_passes_through_without_faulting() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();

    let Err(error) = harness
        .lane_with_options(
            "main",
            AcquireLaneOptions {
                create_at: Some(CreateAt::Entry("ghost".to_owned())),
            },
            &context,
        )
        .await
    else {
        panic!("the unknown target rejects");
    };
    let tag = error
        .as_ref()
        .downcast_ref::<HarnessError>()
        .map(HarnessError::tag)
        .expect("the caller error keeps its tag");
    assert_eq!(tag, "UnknownTarget", "the passthrough keeps the tag");
    assert!(
        harness.lanes(&context).await.is_ok(),
        "the harness stays open after the passthrough"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's catch reads the closed error first; the
/// port binds the same route). With the closed latch set mid-mutation, the
/// lane and name catches return the closed error instead of faulting.
#[tokio::test]
async fn the_catches_return_the_closed_latch_over_the_mutation_failure() {
    let mut sessions = common::SessionLedger::new();
    let context = background_context();

    let (session, storage) = controlled_session(&mut sessions);
    let (options, _faux) = common::harness_options(session);
    let name_harness = create_agent_harness(options, &context)
        .await
        .expect("the harness creates")
        .harness;
    {
        let harness = name_harness.clone();
        storage.set_before_next_commit(Some(Box::new(move || {
            Box::pin(async move {
                // The close's synchronous prefix sets the latch before the
                // commit failure unwinds, upstream's close-during-mutation
                // race; the dropped future leaves the drain to its task.
                drop(harness.close(&background_context()));
                Err(SessionError::Message(COMMIT_FAILED.to_owned()))
            })
        })));
    }
    let closed = name_harness
        .set_name(Some("named".to_owned()), &context)
        .await
        .expect_err("the failed mutation returns the closed latch");
    assert_eq!(
        closed.to_string(),
        CLOSED_MESSAGE,
        "the name catch returns the closed error"
    );

    let (session, storage) = controlled_session(&mut sessions);
    let (options, _faux) = common::harness_options(session);
    let lane_harness = create_agent_harness(options, &context)
        .await
        .expect("the harness creates")
        .harness;
    {
        let harness = lane_harness.clone();
        storage.set_before_next_commit(Some(Box::new(move || {
            Box::pin(async move {
                drop(harness.close(&background_context()));
                Err(SessionError::Message(COMMIT_FAILED.to_owned()))
            })
        })));
    }
    let Err(closed) = lane_harness.lane("fresh", &context).await else {
        panic!("the failed acquisition returns the closed latch");
    };
    assert_eq!(
        closed.to_string(),
        CLOSED_MESSAGE,
        "the lane catch returns the closed error"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's `setName` catch faults the same way).
/// A commit failure faults the harness through the name and label catches,
/// and every later operation rethrows the latched fault.
#[tokio::test]
async fn name_and_label_mutation_failures_fault_the_harness() {
    let mut sessions = common::SessionLedger::new();
    let (session, storage) = controlled_session(&mut sessions);
    let (options, _faux) = common::harness_options(session);
    let harness = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates")
        .harness;
    let context = background_context();
    storage.set_failure(Some(SessionError::Message(COMMIT_FAILED.to_owned())));

    let error = harness
        .set_name(Some("named".to_owned()), &context)
        .await
        .expect_err("the failed name write faults");
    assert_eq!(error.to_string(), FAULT_MESSAGE, "the catch faults");
    let error = harness
        .set_label("entry", Some("label".to_owned()), &context)
        .await
        .expect_err("the failed label write faults");
    assert_eq!(error.to_string(), FAULT_MESSAGE, "the catch faults");
    let latched = harness
        .set_name(None, &context)
        .await
        .expect_err("the fault latches");
    assert_eq!(
        latched.to_string(),
        FAULT_MESSAGE,
        "later setters rethrow the latched fault"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's `setLabel` catch faults the same way).
/// A label commit failure faults the harness through the label catch, and
/// the latched fault answers the later calls.
#[tokio::test]
async fn a_failed_label_write_faults_the_harness() {
    let mut sessions = common::SessionLedger::new();
    let (session, storage) = controlled_session(&mut sessions);
    let (options, _faux) = common::harness_options(session);
    let harness = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates")
        .harness;
    let context = background_context();
    storage.set_failure(Some(SessionError::Message(COMMIT_FAILED.to_owned())));

    let error = harness
        .set_label("entry", Some("label".to_owned()), &context)
        .await
        .expect_err("the failed label write faults");
    assert_eq!(error.to_string(), FAULT_MESSAGE, "the catch faults");
    let latched = harness
        .set_label("entry", None, &context)
        .await
        .expect_err("the fault latches");
    assert_eq!(
        latched.to_string(),
        FAULT_MESSAGE,
        "later setters rethrow the latched fault"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's session mutate rejects the same way; the
/// port's catch faults it). A session-level mutation failure faults the
/// harness through the name catch.
#[tokio::test]
async fn a_closed_session_faults_the_harness_through_the_mutation_line() {
    let mut sessions = common::SessionLedger::new();
    let (session, _storage) = controlled_session(&mut sessions);
    let (options, _faux) = common::harness_options(Arc::clone(&session));
    let harness = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates")
        .harness;
    session
        .close(&background_context())
        .await
        .expect("the session closes");

    let error = harness
        .set_name(Some("named".to_owned()), &background_context())
        .await
        .expect_err("the closed session faults the mutation");
    assert_eq!(error.to_string(), FAULT_MESSAGE, "the catch faults");

    common::close_sessions(&mut sessions).await;
}

/// The foreign-payload harness over the wrapped session, the payload
/// contract's probes: the ledger, the wrapper, the harness, and the
/// context. The wrapper arms nothing — the probes arm after.
fn foreign_payload_harness() -> (
    common::SessionLedger,
    Arc<ForeignMutationPayloadSession>,
    Harness,
    Context,
) {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let foreign = Arc::new(ForeignMutationPayloadSession {
        inner: Arc::clone(&session),
        foreign: AtomicBool::new(false),
        skip_callback: AtomicBool::new(false),
    });
    let (mut options, _faux) = common::harness_options(Arc::clone(&session));
    options.session = foreign.clone();
    let harness = Harness::new(options, common::configured(), BTreeMap::new());
    let context = background_context();
    (sessions, foreign, harness, context)
}

/// Upstream pin: none (upstream's typed `mutate<T>` carries no erased
/// payload; the port's `Any` surface is a live contract a foreign `Session`
/// implementation can breach). A mutation whose session returns a foreign
/// payload faults the harness through the name catch, and the latched
/// fault answers the later calls.
#[tokio::test]
async fn a_foreign_mutation_payload_faults_the_harness() {
    let (mut sessions, foreign, harness, context) = foreign_payload_harness();
    foreign.arm_foreign_payload();

    let error = harness
        .set_name(Some("named".to_owned()), &context)
        .await
        .expect_err("the foreign payload faults the mutation");
    assert_eq!(error.to_string(), FAULT_MESSAGE, "the catch faults");
    let cause = std::error::Error::source(error.as_ref()).expect("the fault carries its cause");
    assert!(
        cause
            .to_string()
            .contains("the mutation callback returns a mutation outcome"),
        "the payload contract's rejection rides the cause chain: {cause}"
    );
    let latched = harness
        .set_name(None, &context)
        .await
        .expect_err("the fault latches");
    assert_eq!(
        latched.to_string(),
        FAULT_MESSAGE,
        "later setters rethrow the latched fault"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: `preserves_value_update_publication_and_delivery`'s
/// clearing half. Clearing the session name and an entry label commits the
/// delete writes and reads back empty.
#[tokio::test]
async fn clearing_the_name_and_labels_deletes_the_stored_values() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();

    harness
        .set_name(Some("named".to_owned()), &context)
        .await
        .expect("the name sets");
    harness
        .set_label("entry", Some("label".to_owned()), &context)
        .await
        .expect("the label sets");
    harness
        .set_name(None, &context)
        .await
        .expect("the name clears");
    harness
        .set_label("entry", None, &context)
        .await
        .expect("the label clears");
    assert_eq!(
        harness.get_name(&context).await.expect("the name read"),
        None,
        "the cleared name reads back empty"
    );
    assert_eq!(
        harness
            .get_label("entry", &context)
            .await
            .expect("the label read"),
        None,
        "the cleared label reads back empty"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's `watchSession` stays unlanded in this
/// slice). The session-watch operation reports the slice stub on both the
/// inherent and trait surfaces without faulting the harness.
#[tokio::test]
async fn watch_session_reports_the_unlanded_slice() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();

    let Err(error) = harness.watch_session(&context) else {
        panic!("the inherent watch_session rejects");
    };
    assert!(
        error.to_string().contains("watchSession"),
        "the stub names the slice: {error}"
    );
    let Err(error) = AgentHarness::watch_session(&harness, &context).await else {
        panic!("the trait watch_session rejects");
    };
    let LaneOperationError::Closed(closed) = &error;
    assert!(
        closed.to_string().contains("watchSession"),
        "the trait stub names the slice: {closed}"
    );
    assert!(
        harness.lanes(&context).await.is_ok(),
        "the harness stays open after the stub"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: the validators' throw messages. The trait config setters
/// reject duplicate tool names, an over-ceiling retry delay, and
/// over-ceiling compaction counts without faulting the harness.
#[tokio::test]
async fn trait_config_setters_reject_invalid_values() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();

    let error = AgentHarness::set_tools(
        &harness,
        vec![named_tool("read"), named_tool("read")],
        &context,
    )
    .await
    .expect_err("the duplicate tool name rejects");
    let LaneOperationError::Closed(rejected) = &error;
    assert_eq!(
        rejected.message(),
        "Duplicate tool name: \"read\"",
        "the duplicate rejection carries the validator's message"
    );
    let error = AgentHarness::set_retry_policy(
        &harness,
        RetryPolicy {
            enabled: true,
            max_retries: 0,
            base_delay_ms: OVER_SAFE_INTEGER,
            max_agent_delay_ms: None,
        },
        &context,
    )
    .await
    .expect_err("the over-ceiling delay rejects");
    let LaneOperationError::Closed(rejected) = &error;
    assert_eq!(
        rejected.message(),
        "Retry policy values must be finite non-negative safe integers",
        "the retry rejection carries the validator's message"
    );
    let error = AgentHarness::set_compaction_settings(
        &harness,
        crate::harness::compaction::types::CompactionSettings {
            enabled: true,
            reserve_tokens: OVER_SAFE_INTEGER,
            keep_recent_tokens: 1,
        },
        &context,
    )
    .await
    .expect_err("the over-ceiling reserve rejects");
    let LaneOperationError::Closed(rejected) = &error;
    assert_eq!(
        rejected.message(),
        "Compaction token counts must be finite non-negative safe integers",
        "the compaction rejection carries the validator's message"
    );
    let tools = AgentHarness::get_tools(&harness, &context)
        .await
        .expect("the tools read");
    assert!(
        tools.is_empty(),
        "the harness stays open after the rejections"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: `publishes_previous_and_current_data_bearing_global_
/// configuration`'s resources half. The trait resources setter stores the
/// replacement and publishes the config-update event.
#[tokio::test]
async fn the_trait_resources_setter_stores_and_publishes() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();
    let seen: Arc<Mutex<Vec<crate::harness::agent_harness::ConfigUpdateKind>>> =
        Arc::new(Mutex::new(Vec::new()));
    {
        let seen = Arc::clone(&seen);
        harness
            .events()
            .on(
                crate::harness::agent_harness::HarnessEventType::ConfigUpdate,
                Arc::new(move |event: &crate::harness::agent_harness::HarnessEvent, _context| {
                    let seen = Arc::clone(&seen);
                    Box::pin(async move {
                        if let crate::harness::agent_harness::HarnessEventPayload::ConfigUpdate {
                            property,
                        } = &event.payload
                        {
                            lock(&seen).push(property.clone());
                        }
                        Ok(())
                    })
                }),
            )
            .expect("the listener subscribes");
    }
    let resources = Resources {
        prompt_templates: Vec::new(),
        skills: Vec::new(),
    };

    AgentHarness::set_resources(&harness, resources.clone(), &context)
        .await
        .expect("the resources set");
    let stored = AgentHarness::get_resources(&harness, &context)
        .await
        .expect("the resources read");
    assert_eq!(stored, resources, "the resources round-trip");
    assert_eq!(
        lock(&seen).len(),
        1,
        "the resources update publishes one config-update event"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: `closes_every_lane_and_rejects_later_acquisition`'s trait
/// surface. After the close, the trait lane and label calls wrap the
/// closed error for the `LaneOperationError` surface.
#[tokio::test]
async fn the_trait_surface_wraps_the_closed_error() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();
    harness.close(&context).await.expect("the close settles");

    let Err(error) = AgentHarness::lane(&harness, "main", &context).await else {
        panic!("the closed lane call rejects");
    };
    let LaneOperationError::Closed(closed) = &error;
    assert_eq!(
        closed.to_string(),
        CLOSED_MESSAGE,
        "the closed rejection carries the closed message"
    );
    let error = AgentHarness::set_label(&harness, "entry", None, &context)
        .await
        .expect_err("the closed label call rejects");
    let LaneOperationError::Closed(closed) = &error;
    assert_eq!(
        closed.to_string(),
        CLOSED_MESSAGE,
        "the label rejection carries the closed message"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: the closed-registry/bus `on` throws. After the fault,
/// the `Events` trait `on` rethrows the closed error as the panic the
/// no-error-channel surface carries.
#[tokio::test]
#[should_panic(expected = "AgentHarness storage or invariant fault")]
async fn events_on_rethrows_the_closed_bus() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    harness.fault(
        lane_error(SessionError::Message("boom".to_owned())),
        &background_context(),
    );
    drop(Events::on(
        harness.events().as_ref(),
        crate::harness::agent_harness::HarnessEventType::ValueUpdate,
        Arc::new(
            |_event: &crate::harness::agent_harness::HarnessEvent, _context| {
                Box::pin(std::future::ready(Ok(())))
            },
        ),
    ));
    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: the closed-registry `on` throw. Same rethrow over the
/// hook registry's trait surface.
#[tokio::test]
#[should_panic(expected = "AgentHarness storage or invariant fault")]
async fn hooks_on_rethrows_the_closed_registry() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    harness.fault(
        lane_error(SessionError::Message("boom".to_owned())),
        &background_context(),
    );
    drop(crate::harness::agent_harness::Hooks::on(
        harness.hooks().as_ref(),
        HookName::BeforeRun,
        Arc::new(|_invocation: &HookInvocation, _context| {
            Box::pin(std::future::ready(Ok(HookResult::BeforeRun(None))))
        }),
        HookOptions::default(),
    ));
    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: the validators' create-time rejection. Duplicate tool
/// names reject the creation with the validator's message and no source.
#[tokio::test]
async fn creation_rejects_duplicate_tool_names_as_a_validation_error() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let (mut options, _faux) = common::harness_options(session);
    options.tools = Some(vec![named_tool("read"), named_tool("read")]);

    let error = create_agent_harness(options, &background_context())
        .await
        .expect_err("the duplicate names reject the creation");
    assert!(
        matches!(error, HarnessCreationError::Validation(_)),
        "the rejection is a validation error: {error:?}"
    );
    assert_eq!(
        error.to_string(),
        "Duplicate tool name: \"read\"",
        "the validation message surfaces"
    );
    assert!(
        std::error::Error::source(&error).is_none(),
        "the validation error carries no source"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: `rejects_partial_durable_lane_state_as_a_harness_fault`'s
/// source half. A restore read failure wraps in the creation fault whose
/// source carries the storage error.
#[tokio::test]
async fn creation_wraps_a_restore_read_failure_in_a_fresh_fault() {
    let mut sessions = common::SessionLedger::new();
    let (session, storage) = controlled_session(&mut sessions);
    storage.arm_read_failure(None, None, SessionError::Message("read failed".to_owned()));
    let (options, _faux) = common::harness_options(session);

    let error = create_agent_harness(options, &background_context())
        .await
        .expect_err("the read failure faults the creation");
    assert!(
        matches!(error, HarnessCreationError::Fault(_)),
        "the rejection is a fault: {error:?}"
    );
    assert_eq!(
        error.to_string(),
        FAULT_MESSAGE,
        "the fault message surfaces"
    );
    let fault = std::error::Error::source(&error).expect("the fault carries its cause");
    let cause = std::error::Error::source(fault).expect("the fault's cause carries its source");
    assert!(
        cause.to_string().contains("read failed"),
        "the source carries the storage error: {cause}"
    );

    common::close_sessions(&mut sessions).await;
}

/// Commits one live operation's durable seed over the `e1` history entry,
/// the open-view cases' shared seeding: the branch tip, the lane config,
/// the lane record pointing the operation, the tip entry, and the
/// operation's meta and state.
async fn seed_live_operation(
    session: &Arc<StorageBackedSession>,
    operation_id: &str,
    intent: OperationIntent,
    state: OperationState,
) {
    let meta = OperationMeta {
        operation_id: operation_id.to_owned(),
        lane: "main".to_owned(),
        source_tip_id: Some("e1".to_owned()),
        started_at: 1,
        intent,
    };
    commit_writes(
        session,
        vec![
            set_value_write(&stored_values::branch_tip("main"), Some("e1".to_owned()))
                .expect("the tip write"),
            set_value_write(&stored_values::lane_config("main"), common::configured())
                .expect("the config write"),
            lane_state_write("main", Some(operation_id), None, Vec::new())
                .expect("the lane state write"),
            Write::Entry(Box::new(insert_entry(NewEntry::Message {
                id: "e1".to_owned(),
                parent_id: None,
                body: Box::new(MessageEntry {
                    message: user_text_message("history"),
                    terminate: None,
                }),
            }))),
            set_value_write(&stored_values::operation_meta(operation_id), meta)
                .expect("the meta write"),
            set_value_write(&stored_values::operation_state(operation_id), state)
                .expect("the state write"),
        ],
    )
    .await;
}

/// Upstream pin: none (upstream's `open` view is exercised over run
/// leaves; the navigation leaf rides the same view). A restored
/// `navigation.ready_to_commit` operation reports `Navigation` in the
/// open-operation view.
#[tokio::test]
async fn a_restored_navigation_operation_reports_in_the_open_view() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let operation_id = "01950000-0000-7000-8000-00000000000a".to_owned();
    let scope = OperationScope {
        control: crate::harness::session::types::Control::Running,
        settings: RunSettings {
            compaction: crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::All,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: None,
    };
    seed_live_operation(
        &session,
        &operation_id,
        OperationIntent::Navigation {
            target_id: Some("e1".to_owned()),
            summarize: false,
            label: None,
            custom_instructions: None,
        },
        OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
            scope,
            target_id: Some("e1".to_owned()),
            label: None,
        }),
    )
    .await;
    let (options, _faux) = common::harness_options(session);

    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    assert_eq!(created.open.len(), 1, "the navigation operation is open");
    let open = &created.open[0];
    assert_eq!(open.lane, "main", "the open operation names its lane");
    assert_eq!(
        open.kind,
        OperationKind::Navigation,
        "the open operation carries the navigation kind"
    );
    assert_eq!(open.operation_id, operation_id, "the id rides through");

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's `open` view is exercised over run
/// leaves; the compaction leaf rides the same view). A restored
/// compaction-intent operation reports `Compaction` in the open-operation
/// view.
#[tokio::test]
async fn a_restored_compaction_operation_reports_in_the_open_view() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let operation_id = "01950000-0000-7000-8000-00000000000b".to_owned();
    let scope = OperationScope {
        control: crate::harness::session::types::Control::Running,
        settings: RunSettings {
            compaction: crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::All,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: None,
    };
    seed_live_operation(
        &session,
        &operation_id,
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        OperationState::SummaryDeciding(SummaryDecidingOperation {
            scope,
            task: SummaryTask {
                task_id: "task".to_owned(),
                reason: Some(CompactionReason::Manual),
                custom_instructions: None,
                boundary: ResultBoundary::Finish,
            },
        }),
    )
    .await;
    let (options, _faux) = common::harness_options(session);

    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    assert_eq!(created.open.len(), 1, "the compaction operation is open");
    let open = &created.open[0];
    assert_eq!(open.lane, "main", "the open operation names its lane");
    assert_eq!(
        open.kind,
        OperationKind::Compaction,
        "the open operation carries the compaction kind"
    );
    assert_eq!(open.operation_id, operation_id, "the id rides through");

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's lane reads carry the live config store;
/// the suite pins the store's read-modify-write at the lane surface). A
/// lane acquired before a config set reads the later store through its
/// live `read_config` closure.
#[tokio::test]
async fn the_config_store_serves_lane_reads_across_later_sets() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();

    let lane = harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    AgentHarness::set_retry_policy(
        &harness,
        RetryPolicy {
            enabled: false,
            max_retries: 0,
            base_delay_ms: 5,
            max_agent_delay_ms: None,
        },
        &context,
    )
    .await
    .expect("the policy sets");
    let tip = lane.get_tip_id(&context).await.expect("the tip read");
    assert_eq!(tip, None, "the fresh lane stays at the root");
    let stored = harness
        .session()
        .get_value(&stored_values::lane_config("main").address, &context)
        .await
        .expect("the config read")
        .expect("the config is stored");
    assert_eq!(
        stored.value["thinkingLevel"],
        json!(ThinkingLevel::Medium),
        "the seeded lane config persists the options' thinking level"
    );

    common::close_sessions(&mut sessions).await;
}

/// The closed rejection one closed-harness trait call returns, the
/// sweep's shared unpack.
fn expect_closed_rejection<T>(result: Result<T, LaneOperationError>, what: &str) {
    let Err(error) = result else {
        panic!("{what} accepts on the closed harness");
    };
    let LaneOperationError::Closed(rejected) = error;
    assert_eq!(
        rejected.message(),
        CLOSED_MESSAGE,
        "{what} carries the closed message"
    );
}

/// Upstream pin: none (upstream's getters run `assertOpen` first and the
/// closed error throws through the same call; the port's trait surface
/// carries the closed wrap per method, so this sweep drives every read's
/// wrap). After the close, every trait read rejects with the closed error.
#[tokio::test]
async fn the_trait_reads_wrap_the_closed_error() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();
    harness.close(&context).await.expect("the close settles");

    expect_closed_rejection(
        AgentHarness::get_name(&harness, &context).await,
        "the name read",
    );
    expect_closed_rejection(
        AgentHarness::get_label(&harness, "entry", &context).await,
        "the label read",
    );
    expect_closed_rejection(
        AgentHarness::get_tools(&harness, &context).await,
        "the tools read",
    );
    expect_closed_rejection(
        AgentHarness::get_resources(&harness, &context).await,
        "the resources read",
    );
    expect_closed_rejection(
        AgentHarness::get_stream_options(&harness, &context).await,
        "the stream-options read",
    );
    expect_closed_rejection(
        AgentHarness::get_retry_policy(&harness, &context).await,
        "the retry read",
    );
    expect_closed_rejection(
        AgentHarness::get_compaction_settings(&harness, &context).await,
        "the compaction read",
    );
    expect_closed_rejection(
        AgentHarness::get_steering_mode(&harness, &context).await,
        "the steering read",
    );
    expect_closed_rejection(
        AgentHarness::get_follow_up_mode(&harness, &context).await,
        "the follow-up read",
    );
    expect_closed_rejection(
        AgentHarness::lanes(&harness, &context).await,
        "the lanes read",
    );
    expect_closed_rejection(
        AgentHarness::lane_with_options(&harness, "main", AcquireLaneOptions::default(), &context)
            .await,
        "the options acquisition",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's setters run `assertOpen` first and the
/// closed error throws through the same call; the port's trait surface
/// carries the closed wrap per method, so this sweep drives every config
/// setter's wrap). After the close, every config setter rejects with the
/// closed error.
#[tokio::test]
async fn the_trait_config_setters_wrap_the_closed_error() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();
    harness.close(&context).await.expect("the close settles");

    expect_closed_rejection(
        AgentHarness::set_name(&harness, Some("named".to_owned()), &context).await,
        "the name write",
    );
    expect_closed_rejection(
        AgentHarness::set_tools(&harness, vec![named_tool("read")], &context).await,
        "the tools write",
    );
    expect_closed_rejection(
        AgentHarness::set_resources(
            &harness,
            Resources {
                prompt_templates: Vec::new(),
                skills: Vec::new(),
            },
            &context,
        )
        .await,
        "the resources write",
    );
    expect_closed_rejection(
        AgentHarness::set_stream_options(
            &harness,
            crate::harness::types::AgentHarnessStreamOptions::default(),
            &context,
        )
        .await,
        "the stream-options write",
    );
    expect_closed_rejection(
        AgentHarness::set_retry_policy(
            &harness,
            RetryPolicy {
                enabled: false,
                max_retries: 0,
                base_delay_ms: 5,
                max_agent_delay_ms: None,
            },
            &context,
        )
        .await,
        "the retry write",
    );
    expect_closed_rejection(
        AgentHarness::set_compaction_settings(
            &harness,
            crate::harness::compaction::types::CompactionSettings {
                enabled: true,
                reserve_tokens: 1_000,
                keep_recent_tokens: 10,
            },
            &context,
        )
        .await,
        "the compaction write",
    );
    expect_closed_rejection(
        AgentHarness::set_steering_mode(&harness, QueueMode::All, &context).await,
        "the steering write",
    );
    expect_closed_rejection(
        AgentHarness::set_follow_up_mode(&harness, QueueMode::All, &context).await,
        "the follow-up write",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: `createAgentHarness`'s seed — the
/// `options.activeToolNames ?? tools.map((tool) => tool.name)` fallback.
/// Without explicit active tool names the seed derives them from the
/// tools list, and the first acquisition persists them onto the lane.
#[tokio::test]
async fn the_seed_derives_the_active_tool_names_from_the_tools_list() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let (mut options, _faux) = common::harness_options(session);
    options.tools = Some(vec![named_tool("read")]);
    options.active_tool_names = None;

    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    let context = background_context();
    created
        .harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    let stored = created
        .harness
        .session()
        .get_value(&stored_values::lane_config("main").address, &context)
        .await
        .expect("the config read")
        .expect("the config is stored");
    assert_eq!(
        stored.value["activeToolNames"],
        json!(["read"]),
        "the seed derives the active tool names from the tools list"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: `lane`'s not-published guard (`Lane
/// ${JSON.stringify(name)} was not published`). A session whose `mutate`
/// resolves without running the callback publishes nothing, so the
/// acquisition faults with the guard's invariant and latches the fault.
#[tokio::test]
async fn an_unpublished_mutation_faults_the_acquisition() {
    let (mut sessions, foreign, harness, context) = foreign_payload_harness();
    foreign.arm_skip_callback();

    let Err(error) = harness.lane("main", &context).await else {
        panic!("the unpublished acquisition faults");
    };
    assert_eq!(
        error.to_string(),
        FAULT_MESSAGE,
        "the catch faults the acquisition"
    );
    let cause = std::error::Error::source(error.as_ref()).expect("the fault carries its cause");
    assert!(
        cause
            .to_string()
            .contains("Lane \"main\" was not published"),
        "the not-published invariant rides the cause chain: {cause}",
    );
    let Err(latched) = harness.lane("main", &context).await else {
        panic!("the fault latches");
    };
    assert_eq!(
        latched.to_string(),
        FAULT_MESSAGE,
        "later acquisitions rethrow the latched fault"
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream pin: none (upstream's `await delivery` no-ops the same way
/// when `mutate` resolves without running the callback). The name and
/// label setters resolve `Ok` with nothing published and nothing stored.
#[tokio::test]
async fn the_name_and_label_setters_resolve_without_a_delivery_when_nothing_publishes() {
    let (mut sessions, foreign, harness, context) = foreign_payload_harness();
    foreign.arm_skip_callback();

    harness
        .set_name(Some("named".to_owned()), &context)
        .await
        .expect("the name resolves without a delivery");
    harness
        .set_label("entry", Some("label".to_owned()), &context)
        .await
        .expect("the label resolves without a delivery");
    assert_eq!(
        harness.get_name(&context).await.expect("the name read"),
        None,
        "the skipped mutation stores nothing"
    );
    assert!(
        harness.lanes(&context).await.is_ok(),
        "the harness stays open"
    );

    common::close_sessions(&mut sessions).await;
}

// The sweep's remaining arms stay uncovered, verified against upstream at
// pin `60e7e76`: the `unreachable!` construction guards (the harness-global
// and lane-scoped event builds, the close drainer's dropped receiver, the
// watch installer's closed bus — the lane's `readLane` rejects a sealed
// lane before the installer runs —, the restored-lane and lane-created
// match arms, the losing close's memoized receiver, and `quoted`'s
// serialization fallback a string input cannot reach), the
// `Lane ... was not published` guard's original stub and the
// `set_name`/`set_label`
// if-let closing braces (the setters always publish a delivery, so the
// implicit-else region never runs), and the trait `close` wrap whose
// error half needs a failing session close (the memory-backed fixtures'
// close never fails).
