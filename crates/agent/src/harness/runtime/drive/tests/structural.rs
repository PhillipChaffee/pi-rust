//! The structural drive suite, ported 1:1 from upstream
//! `test/harness/runtime/drive-structural.test.ts` ("runtime structural
//! drive") at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's `it.each(["steer", "followUp"])` pair drives as one
//!   case-table loop inside a single test.
//! - upstream's fake-timer choreography (the two delayed-retry cases'
//!   `vi.useFakeTimers` + `vi.setSystemTime`) restates through the drive
//!   clock seam ([`super::common::pin_clock`] and
//!   [`crate::harness::runtime::clock::set_test_now`]): `waitForRetry`
//!   stays off so no tokio timer exists to pause, and the deadline is read
//!   from the wire state.
//! - the per-case queued-write commands rest on
//!   [`super::common::queue_pending_entries`]; every case closes its
//!   session at the end (upstream's `afterEach`).
//! - the commit-attempt and event-sequence assertions read
//!   [`crate::harness::session::testing::InstrumentedStorage`] and the
//!   fixture's collected events, upstream's `storage.getCommitAttempts()`
//!   and `events` reads.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; unexpected results and violated expectations panic the test by design"
)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxResponseStep;
use pi_ai::providers::faux::{FauxTokenSize, RegisterFauxProviderOptions, faux_assistant_message};
use pi_ai::types::{Message, SimpleStreamOptions, StopReason, Usage, UsageCost};
use pi_ai::utils::retry::RetryPolicy;
use serde_json::json;

use crate::harness::agent_harness::CompactionEndStatus;
use crate::harness::agent_harness::CompactionHookResult;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HarnessEventType;
use crate::harness::agent_harness::HookFailure;
use crate::harness::agent_harness::HookHandler;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookOptions;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::{NavigationEndStatus, NavigationHookResult, RunEndStatus};
use crate::harness::compaction::types::CompactResult;
use crate::harness::compaction::types::{CompactionSettings, DEFAULT_COMPACTION_SETTINGS};
use crate::harness::context::{Context, background_context};
use crate::harness::runtime::clock::set_test_now;
use crate::harness::runtime::drive::checkpoint::run_checkpoint;
use crate::harness::runtime::drive::generation::{GenerationLeaf, run_generation};
use crate::harness::runtime::drive::structural::commit_navigation;
use crate::harness::runtime::drive::structural::prepare_overflow_compaction;
use crate::harness::runtime::drive::structural::recover_structural_generation;
use crate::harness::runtime::drive::structural::run_structural_decision;
use crate::harness::runtime::drive::structural::run_structural_generation;
use crate::harness::runtime::drive::structural::run_structural_retry_wait;
use crate::harness::runtime::lane::{Lane, operation_state_with_scope};
use crate::harness::runtime::test_support::{deferred, lock, settle_events};
use crate::harness::runtime::types::{
    Drive, LaneCommand, LaneState, LiveOperation, ProcedureResult,
};
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::CompactionEntryBody;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DurableStructuralPreparation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryType;
use crate::harness::session::types::GenerationContext;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::SummaryContext;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryEffectPendingOperation;
use crate::harness::session::types::SummaryEffectRequest;
use crate::harness::session::types::SummaryGenerationScope;
use crate::harness::session::types::SummaryReadyOperation;
use crate::harness::session::types::{SummaryRetryWaitOperation, SummaryTask, TerminalStatus};
use crate::harness::session::values as stored_values;
use crate::harness::session::values::Write;
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::{AgentMessage, QueueMode};

use super::common;
use super::common::InstallOptions;
use super::common::OPERATION_ID;
use super::common::PreparationWrite;
use super::common::assistant_entry;
use super::common::branch_preparation;
use super::common::cancel_operation;
use super::common::compaction_preparation;
use super::common::create_drive_fixture;
use super::common::current_state;
use super::common::deciding_run;
use super::common::deciding_standalone;
use super::common::install_deciding_compaction;
use super::common::install_deciding_run;
use super::common::install_operation;
use super::common::install_standalone_summary_ready;
use super::common::navigation_summary_task;
use super::common::pin_clock;
use super::common::queue_pending_entries;
use super::common::run_compaction_task;
use super::common::run_scope;
use super::common::scope_with_control;
use super::common::{ClockPin, DriveFixture, DriveFixtureSpec};
use super::common::{standalone_compaction_task, summary_context, summary_ready, user, user_entry};
use crate::harness::runtime::types::LaneError;

/// The structural suite's fixture spec, upstream's `createFixture` body:
/// the faux provider's one-token stream bound, the
/// `{ enabled: true, maxRetries: 1, baseDelayMs: 10 }` retry policy, the
/// default stream options, the fixed-100-clock backend, and the
/// hand-seeded idle lane (no admission).
fn structural_spec() -> DriveFixtureSpec {
    DriveFixtureSpec {
        suite: "structural",
        watch_suite: "structural",
        faux: RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..RegisterFauxProviderOptions::default()
        },
        retry_policy: RetryPolicy {
            enabled: true,
            max_retries: 1,
            base_delay_ms: 10,
            max_agent_delay_ms: None,
        },
        stream_options: AgentHarnessStreamOptions::default(),
        backend: None,
        admit_prompt: false,
        system_prompt: None,
    }
}

/// Builds one fixture, upstream's `await createFixture()`.
async fn create_fixture() -> DriveFixture {
    create_drive_fixture(structural_spec()).await
}

/// The checkpoint leaf the threshold cases build, upstream's
/// `{ ...runScope(settings), at: "checkpoint", continuation, triggerEntryId }`
/// literals.
fn checkpoint_leaf(settings: CompactionSettings, trigger_entry_id: &str) -> CheckpointOperation {
    CheckpointOperation {
        scope: run_scope(settings),
        checkpoint: CheckpointData {
            continuation: Continuation::NeedAssistant {
                overflow_recovery_used: false,
            },
            trigger_entry_id: trigger_entry_id.to_owned(),
        },
    }
}

/// The hook handler returning one fixed result, upstream's
/// `fixture.hooks.on(name, () => result)` registrations.
fn fixed_hook(result: HookResult) -> HookHandler {
    Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
        Box::pin(std::future::ready(Ok(result.clone())))
    })
}

/// The `before_compaction` decline, upstream's `() => ({ decline: true })`.
fn decline_hook() -> HookHandler {
    fixed_hook(HookResult::BeforeCompaction(Some(CompactionHookResult {
        decline: Some(true),
        compaction: None,
    })))
}

/// The `before_run_end` observer counting invocations, upstream's
/// `before_run_end` `finishHooks++` counter.
fn finish_hook_counter(counter: Arc<AtomicUsize>) -> HookHandler {
    Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::ready(Ok(HookResult::BeforeRunEnd(None))))
    })
}

/// Registers the `before_compaction` decline hook, the threshold-decline
/// cases' `fixture.hooks.on` block.
fn register_decline_hook(fixture: &DriveFixture) {
    fixture
        .hooks
        .on(
            HookName::BeforeCompaction,
            decline_hook(),
            HookOptions::default(),
        )
        .expect("the decline hook registers");
}

/// Registers the `before_compaction` decline hook plus the counting
/// `before_run_end` hook, the decline+finish-counter pair the threshold
/// suites install; returns the counter.
fn register_declining_finish_hook(fixture: &DriveFixture) -> Arc<AtomicUsize> {
    let finish_hooks = Arc::new(AtomicUsize::new(0));
    register_decline_hook(fixture);
    fixture
        .hooks
        .on(
            HookName::BeforeRunEnd,
            finish_hook_counter(Arc::clone(&finish_hooks)),
            HookOptions::default(),
        )
        .expect("the finish hook registers");
    finish_hooks
}

/// Drives one checkpoint and pins the Continue routing; a wrong routing
/// panics the test by design.
async fn checkpoint_continues(fixture: &DriveFixture, checkpoint: &CheckpointOperation, why: &str) {
    let result = run_checkpoint(&fixture.lane, &fixture.drive, checkpoint)
        .await
        .expect("the checkpoint serves");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "{why}: {result:?}"
    );
}

/// Spawns one structural generation attempt over the fixture's lane and
/// drive, the attempt the cancellation suites race against.
fn spawn_generation_attempt(
    fixture: &DriveFixture,
    ready: &SummaryReadyOperation,
) -> tokio::task::JoinHandle<Result<ProcedureResult, LaneError>> {
    let lane = Arc::clone(&fixture.lane);
    let drive = Arc::clone(&fixture.drive);
    let ready = ready.clone();
    tokio::spawn(async move { run_structural_generation(&lane, &drive, &ready).await })
}

/// Signals the started gate and parks on the release gate, the parked
/// factory body the cancellation suites race against.
async fn release_parked(
    started: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    release: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
    why: &str,
) {
    let started = lock(&started).take();
    if let Some(started) = started {
        let _ = started.send(());
    }
    let release = lock(&release).take().expect(why);
    let _ = release.await;
}

/// The overflow-recovery deciding leaf, upstream's
/// `decidingRun(Continuation::NeedAssistant { overflowRecoveryUsed: true })`
/// constructions.
fn overflow_deciding() -> SummaryDecidingOperation {
    SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: run_compaction_task(
            CompactionReason::Overflow,
            CheckpointData {
                continuation: Continuation::NeedAssistant {
                    overflow_recovery_used: true,
                },
                trigger_entry_id: "tip".to_owned(),
            },
        ),
    }
}

/// The navigation-end status the events' last entry carries, the tail
/// reads the navigation suites pin; a non-end tail panics the test by
/// design.
fn last_navigation_end<'a>(events: &'a [HarnessEvent], why: &str) -> &'a NavigationEndStatus {
    let last = events.last().expect(why);
    let HarnessEvent {
        payload: HarnessEventPayload::NavigationEnd { status, .. },
        ..
    } = last
    else {
        panic!("the navigation ends: {:?}", last.payload.event_type());
    };
    status
}

/// Runs the tip-less decision's choreography: the deciding leaf installed
/// over the tip history without a usable tip (`tipId: null`), the decline
/// hook registered, and the decision expected to reject, the tip-less
/// invariants' shared fixture.
///
/// # Panics
/// The install's or the hook registration's failure, or the decision not
/// rejecting.
async fn run_tip_less_decision(
    fixture: &DriveFixture,
    deciding: &SummaryDecidingOperation,
    intent: OperationIntent,
    expect: &str,
) -> LaneError {
    install_operation(
        fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        intent,
        InstallOptions {
            entries: vec![user_entry("tip", None, "history")],
            tip_id: Some(None),
            preparation: Some(PreparationWrite {
                task_id: "task".to_owned(),
                value: compaction_preparation(),
            }),
            ..InstallOptions::default()
        },
    )
    .await;
    register_decline_hook(fixture);
    run_structural_decision(&fixture.lane, &fixture.drive, deciding)
        .await
        .expect_err(expect)
}

/// The retry-wait boundary fixture: the pinned clock, the installed
/// retry-wait leaf over the tip history, and the pass over
/// `wait_for_retry`, the retry-wait boundary cases' shared prelude. The
/// returned pin keeps the clock frozen for the test's duration.
///
/// # Panics
/// The install's failure.
async fn retry_wait_boundary_fixture() -> (
    DriveFixture,
    Arc<Drive>,
    SummaryRetryWaitOperation,
    ClockPin,
) {
    let clock = pin_clock(1_000);
    let fixture = create_fixture().await;
    let retry = retry_leaf(&fixture, 1_030);
    install_operation(
        &fixture,
        OperationState::SummaryRetryWait(retry.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        InstallOptions {
            entries: vec![user_entry("tip", None, "history")],
            ..InstallOptions::default()
        },
    )
    .await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: OPERATION_ID.to_owned(),
            wait_for_retry: Some(true),
            poll_deferred: None,
        },
        &background_context(),
    ));
    (fixture, drive, retry, clock)
}

/// The settled record one procedure result carries, upstream's
/// `result.outcome` reads on the `{ kind: "settled" }` arms.
///
/// # Panics
/// The result not being a settlement.
/// Elapses one retry wait to ready, upstream's
/// `vi.setSystemTime(retry.retryWait.notBefore)` + the resumed wait: the
/// clock moves to the deadline, the wait runs, and the ready leaf returns.
async fn elapsed_retry_ready(
    fixture: &DriveFixture,
    retry: &SummaryRetryWaitOperation,
) -> SummaryReadyOperation {
    set_test_now(retry.retry_wait.not_before);
    let result = run_structural_retry_wait(&fixture.lane, &fixture.drive, retry)
        .await
        .expect("the wait serves");
    let _ = result;
    match current_state(fixture) {
        OperationState::SummaryReady(second) => second,
        other => panic!("elapsed retry did not become ready: {:?}", other.at()),
    }
}

fn settled(result: ProcedureResult, context: &str) -> OperationResultRecord {
    match result {
        ProcedureResult::Settled { outcome } => outcome,
        other => panic!("{context}: {other:?}"),
    }
}

/// Upstream `it`: routes a declined threshold directly to assistant generation.
#[tokio::test]
async fn routes_a_declined_threshold_directly_to_assistant_generation() {
    let fixture = create_fixture().await;
    let model = fixture.faux.first_model();
    let settings = CompactionSettings {
        enabled: true,
        reserve_tokens: model.context_window,
        keep_recent_tokens: 1,
    };
    let checkpoint = checkpoint_leaf(settings, "assistant");
    install_operation(
        &fixture,
        OperationState::Checkpoint(checkpoint.clone()),
        OperationIntent::Run {
            prompt_entry_ids: vec!["user".to_owned()],
        },
        InstallOptions {
            entries: vec![
                user_entry("user", None, "question"),
                assistant_entry(
                    "assistant",
                    Some("user"),
                    faux_assistant_message("answer", FauxAssistantMessageOptions::default()),
                ),
            ],
            ..InstallOptions::default()
        },
    )
    .await;

    checkpoint_continues(&fixture, &checkpoint, "the threshold routes").await;
    let deciding = match current_state(&fixture) {
        OperationState::SummaryDeciding(deciding) => deciding,
        other => panic!("threshold did not enter compaction: {:?}", other.at()),
    };
    assert!(
        matches!(
            deciding.task.boundary,
            ResultBoundary::ResumeCheckpoint { .. }
        ),
        "threshold has wrong boundary"
    );
    let preparation = common::stored_preparation(&fixture, &deciding.task.task_id).await;
    assert!(
        preparation.is_some(),
        "the threshold persists its preparation"
    );

    register_decline_hook(&fixture);
    common::decision_continues(&fixture, &deciding, "the decline routes").await;
    let routed = match current_state(&fixture) {
        OperationState::AssistantReady(routed) => routed,
        other => panic!(
            "threshold decline did not route to generation: {:?}",
            other.at()
        ),
    };
    assert_eq!(routed.generation_context.trigger_entry_id, "assistant");
    assert!(!routed.generation_context.overflow_recovery_used);
    let events = lock(&fixture.events).clone();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::CompactionStart)
            .count(),
        1,
        "the threshold starts compaction once"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::CompactionEnd)
            .count(),
        1,
        "the decline ends compaction once"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: uses a newer compaction entry as the durable threshold guard.
#[tokio::test]
async fn uses_a_newer_compaction_entry_as_the_durable_threshold_guard() {
    let fixture = create_fixture().await;
    let model = fixture.faux.first_model();
    let checkpoint = checkpoint_leaf(
        CompactionSettings {
            enabled: true,
            reserve_tokens: model.context_window,
            keep_recent_tokens: 1,
        },
        "trigger",
    );
    install_operation(
        &fixture,
        OperationState::Checkpoint(checkpoint.clone()),
        OperationIntent::Run {
            prompt_entry_ids: vec!["trigger".to_owned()],
        },
        InstallOptions {
            entries: vec![
                user_entry("trigger", None, "history"),
                NewEntry::Compaction {
                    id: "compacted".to_owned(),
                    parent_id: Some("trigger".to_owned()),
                    body: CompactionEntryBody {
                        summary: "already compacted".to_owned(),
                        retained_tail: Vec::new(),
                        tokens_before: i64::try_from(model.context_window)
                            .expect("the window fits"),
                        details: None,
                        usage: None,
                        from_hook: false,
                    },
                },
            ],
            ..InstallOptions::default()
        },
    )
    .await;

    checkpoint_continues(&fixture, &checkpoint, "the guard skips the threshold").await;
    let ready = match current_state(&fixture) {
        OperationState::AssistantReady(ready) => ready,
        other => panic!(
            "newer compaction did not guard threshold re-entry: {:?}",
            other.at()
        ),
    };
    assert_eq!(ready.generation_context.trigger_entry_id, "trigger");
    let events = lock(&fixture.events).clone();
    assert!(
        !events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::CompactionStart),
        "the guarded checkpoint never starts compaction"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: rejects a missing threshold trigger when no newer compaction guards it.
#[tokio::test]
async fn rejects_a_missing_threshold_trigger_when_no_newer_compaction_guards_it() {
    let fixture = create_fixture().await;
    let model = fixture.faux.first_model();
    let checkpoint = checkpoint_leaf(
        CompactionSettings {
            enabled: true,
            reserve_tokens: model.context_window,
            keep_recent_tokens: 1,
        },
        "missing-trigger",
    );
    install_operation(
        &fixture,
        OperationState::Checkpoint(checkpoint.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        InstallOptions {
            entries: vec![user_entry("tip", None, "history")],
            ..InstallOptions::default()
        },
    )
    .await;

    let error = run_checkpoint(&fixture.lane, &fixture.drive, &checkpoint)
        .await
        .expect_err("the missing trigger rejects");
    assert!(
        error
            .to_string()
            .contains("Checkpoint trigger missing-trigger is missing from its Branch"),
        "the missing trigger carries its invariant: {error}"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: finishes a may-finish run directly after threshold decline.
#[tokio::test]
async fn finishes_a_may_finish_run_directly_after_threshold_decline() {
    let fixture = create_fixture().await;
    let deciding = deciding_run(Continuation::MayFinish {
        include_final_assistant: false,
    });
    install_deciding_run(&fixture, &deciding).await;
    let finish_hooks = register_declining_finish_hook(&fixture);

    let result = run_structural_decision(&fixture.lane, &fixture.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the declined threshold finishes");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    assert_eq!(outcome.tip_id.as_deref(), Some("tip"));
    assert_eq!(
        fixture.storage.get_commit_attempts().len(),
        1,
        "the finish commits once"
    );
    assert_eq!(
        finish_hooks.load(Ordering::SeqCst),
        1,
        "the finish hook runs once"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the finished operation clears"
    );
    let events = lock(&fixture.events).clone();
    let tail = &events[events.len().saturating_sub(2)..];
    assert_eq!(tail.len(), 2, "the tail pairs the decline with the run end");
    let HarnessEvent {
        payload: HarnessEventPayload::CompactionEnd { reason, status, .. },
        ..
    } = &tail[0]
    else {
        panic!(
            "the decline ends compaction: {:?}",
            tail[0].payload.event_type()
        );
    };
    assert_eq!(*reason, CompactionReason::Threshold);
    assert!(matches!(status, CompactionEndStatus::Declined));
    let HarnessEvent {
        payload: HarnessEventPayload::RunEnd { status, .. },
        ..
    } = &tail[1]
    else {
        panic!("the run ends: {:?}", tail[1].payload.event_type());
    };
    assert!(matches!(status, RunEndStatus::Completed));

    common::close_session(&fixture).await;
}

/// Upstream `it`: routes queued follow-up before `before_run_end` after threshold decline.
#[tokio::test]
async fn routes_queued_follow_up_before_before_run_end_after_threshold_decline() {
    let fixture = create_fixture().await;
    let deciding = deciding_run(Continuation::MayFinish {
        include_final_assistant: false,
    });
    install_deciding_run(&fixture, &deciding).await;
    queue_pending_entries(
        &fixture,
        vec![(
            "follow-up",
            PendingEntry::Message {
                payload: Box::new(user("continue", 1)),
            },
            InboxItemKind::FollowUp,
        )],
    )
    .await;
    fixture.storage.clear_commit_attempts();
    let finish_hooks = register_declining_finish_hook(&fixture);

    common::decision_continues(&fixture, &deciding, "the queued follow-up routes").await;
    assert_eq!(
        fixture.storage.get_commit_attempts().len(),
        1,
        "the routing commits once"
    );
    assert_eq!(
        finish_hooks.load(Ordering::SeqCst),
        0,
        "the queued follow-up precedes the finish hook"
    );
    let ready = match current_state(&fixture) {
        OperationState::AssistantReady(ready) => ready,
        other => panic!("follow-up did not route to generation: {:?}", other.at()),
    };
    assert_eq!(ready.generation_context.trigger_entry_id, "follow-up");
    let events = lock(&fixture.events).clone();
    assert!(
        events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::CompactionEnd),
        "the decline ends compaction"
    );

    common::close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the it.each pair is one case table the loop drives case by case"
)]
/// Upstream `it.each(["steer", "followUp"])`: continues to an assistant turn
/// when steer arrives during in-run compaction / continues to an assistant
/// turn when followUp arrives during in-run compaction.
#[tokio::test]
async fn continues_to_an_assistant_turn_when_queued_input_arrives_during_in_run_compaction() {
    for (queue_name, queue_kind) in [
        ("steer", InboxItemKind::Steer),
        ("followUp", InboxItemKind::FollowUp),
    ] {
        let fixture = create_fixture().await;
        let deciding = deciding_run(Continuation::MayFinish {
            include_final_assistant: true,
        });
        install_deciding_run(&fixture, &deciding).await;
        let (hook_started_tx, hook_started_rx) = deferred();
        let (hook_release_tx, hook_release_rx) = deferred();
        let started = Arc::new(Mutex::new(Some(hook_started_tx)));
        let release = Arc::new(Mutex::new(Some(hook_release_rx)));
        fixture
            .hooks
            .on(
                HookName::BeforeCompaction,
                Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                    let started = Arc::clone(&started);
                    let release = Arc::clone(&release);
                    Box::pin(async move {
                        release_parked(started, release, "the parked hook releases once").await;
                        Ok(HookResult::BeforeCompaction(Some(CompactionHookResult {
                            decline: None,
                            compaction: Some(CompactResult {
                                summary: "hook summary".to_owned(),
                                tokens_before: 1_000,
                                usage: None,
                                retained_tail: vec![user("tail", 1)],
                                details: None,
                            }),
                        })))
                    })
                }),
                HookOptions::default(),
            )
            .expect("the parked hook registers");
        let running = {
            let lane = Arc::clone(&fixture.lane);
            let drive = Arc::clone(&fixture.drive);
            let deciding = deciding.clone();
            tokio::spawn(async move { run_structural_decision(&lane, &drive, &deciding).await })
        };
        hook_started_rx.await.expect("the parked hook starts");
        queue_pending_entries(
            &fixture,
            vec![(
                "queued",
                PendingEntry::Message {
                    payload: Box::new(user(&format!("{queue_name} during compaction"), 1)),
                },
                queue_kind,
            )],
        )
        .await;
        let _ = hook_release_tx.send(());
        let result = running
            .await
            .expect("the decision joins")
            .expect("the decision serves");
        assert!(
            matches!(result, ProcedureResult::Continue),
            "the {queue_name} routes: {result:?}"
        );

        if queue_kind == InboxItemKind::Steer {
            let routed = match current_state(&fixture) {
                OperationState::AssistantReady(routed) => routed,
                other => panic!(
                    "steer did not route directly to generation: {:?}",
                    other.at()
                ),
            };
            assert_eq!(routed.generation_context.trigger_entry_id, "queued");
            assert!(!routed.generation_context.overflow_recovery_used);
            assert!(fixture.lane.state().inbox.is_empty(), "the steer consumes");
            let queued = common::stored_pending_entry(&fixture, "queued").await;
            assert!(queued.is_none(), "the consumed pending entry deletes");
            let entry = fixture
                .session
                .get_entry("queued", &background_context())
                .await
                .expect("the entry reads")
                .expect("the steer entry places");
            let parent_id = entry.parent_id().expect("the steer entry parents");
            assert_eq!(entry.entry_type(), EntryType::Message);
            let parent = fixture
                .session
                .get_entry(parent_id, &background_context())
                .await
                .expect("the parent reads")
                .expect("the steer's parent places");
            assert_eq!(parent.entry_type(), EntryType::Compaction);
        } else {
            let checkpoint = match current_state(&fixture) {
                OperationState::Checkpoint(checkpoint) => checkpoint,
                other => panic!(
                    "follow-up did not reach the finish checkpoint: {:?}",
                    other.at()
                ),
            };
            assert_eq!(
                fixture.lane.state().inbox,
                vec![InboxItem {
                    entry_id: "queued".to_owned(),
                    kind: InboxItemKind::FollowUp,
                }],
                "the follow-up stays queued"
            );
            checkpoint_continues(&fixture, &checkpoint, "the queued follow-up routes").await;
            let ready = match current_state(&fixture) {
                OperationState::AssistantReady(ready) => ready,
                other => panic!(
                    "follow-up did not route directly to generation: {:?}",
                    other.at()
                ),
            };
            assert_eq!(ready.generation_context.trigger_entry_id, "queued");
            assert!(!ready.generation_context.overflow_recovery_used);
            assert!(
                fixture.lane.state().inbox.is_empty(),
                "the follow-up consumes"
            );
        }

        common::close_session(&fixture).await;
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the admission-ordered commit across output and mixed queued input"
)]
/// Upstream `it`: publishes structural output and mixed write/steer input in one admission-ordered commit.
#[tokio::test]
async fn publishes_structural_output_and_mixed_write_steer_input_in_one_admission_ordered_commit() {
    let fixture = create_fixture().await;
    let mut scope = run_scope(DEFAULT_COMPACTION_SETTINGS);
    scope.settings.steering_mode = QueueMode::OneAtATime;
    let deciding = SummaryDecidingOperation {
        scope,
        task: run_compaction_task(
            CompactionReason::Threshold,
            CheckpointData {
                continuation: Continuation::NeedAssistant {
                    overflow_recovery_used: false,
                },
                trigger_entry_id: "tip".to_owned(),
            },
        ),
    };
    install_deciding_run(&fixture, &deciding).await;
    queue_pending_entries(
        &fixture,
        vec![
            (
                "write-1",
                PendingEntry::Custom {
                    custom_type: "note".to_owned(),
                    payload: Some(json!({"order": 1})),
                },
                InboxItemKind::Write,
            ),
            (
                "steer",
                PendingEntry::Message {
                    payload: Box::new(user("steer", 1)),
                },
                InboxItemKind::Steer,
            ),
            (
                "write-2",
                PendingEntry::Custom {
                    custom_type: "note".to_owned(),
                    payload: Some(json!({"order": 2})),
                },
                InboxItemKind::Write,
            ),
            (
                "steer-2",
                PendingEntry::Message {
                    payload: Box::new(user("next steer", 1)),
                },
                InboxItemKind::Steer,
            ),
        ],
    )
    .await;
    fixture.storage.clear_commit_attempts();
    fixture
        .hooks
        .on(
            HookName::BeforeCompaction,
            fixed_hook(HookResult::BeforeCompaction(Some(CompactionHookResult {
                decline: None,
                compaction: Some(CompactResult {
                    summary: "summary".to_owned(),
                    tokens_before: 1_000,
                    usage: None,
                    retained_tail: vec![user("tail", 1)],
                    details: None,
                }),
            }))),
            HookOptions::default(),
        )
        .expect("the compaction hook registers");

    common::decision_continues(&fixture, &deciding, "the mixed input routes").await;
    assert_eq!(
        fixture.storage.get_commit_attempts().len(),
        1,
        "the mixed input commits once"
    );
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id: "steer-2".to_owned(),
            kind: InboxItemKind::Steer,
        }],
        "one-at-a-time steering holds the second steer"
    );
    let queued = common::stored_pending_entry(&fixture, "steer-2").await;
    assert!(queued.is_some(), "the held steer's pending entry stays");
    let routed = match current_state(&fixture) {
        OperationState::AssistantReady(routed) => routed,
        other => panic!("mixed input did not route to generation: {:?}", other.at()),
    };
    assert_eq!(routed.generation_context.trigger_entry_id, "steer");
    let write_1 = fixture
        .session
        .get_entry("write-1", &background_context())
        .await
        .expect("the entry reads")
        .expect("the first write places");
    assert!(write_1.parent_id().is_some(), "the first write parents");
    let steer = fixture
        .session
        .get_entry("steer", &background_context())
        .await
        .expect("the entry reads")
        .expect("the steer places");
    assert_eq!(
        steer.parent_id(),
        Some("write-1"),
        "the steer follows the first write"
    );
    let write_2 = fixture
        .session
        .get_entry("write-2", &background_context())
        .await
        .expect("the entry reads")
        .expect("the second write places");
    assert_eq!(
        write_2.parent_id(),
        Some("steer"),
        "the second write follows the steer"
    );
    assert_eq!(fixture.lane.state().tip_id.as_deref(), Some("write-2"));

    common::close_session(&fixture).await;
}

/// Upstream `it`: queues writes during standalone structural work without changing the operation.
#[tokio::test]
async fn queues_writes_during_standalone_structural_work_without_changing_the_operation() {
    let fixture = create_fixture().await;
    let deciding = deciding_standalone();
    install_deciding_compaction(&fixture, &deciding).await;

    let entry_id = fixture
        .lane
        .append_custom_entry_impl(
            "note",
            Some(json!({"pending": true})),
            &background_context(),
        )
        .await
        .expect("the append serves");

    assert_eq!(
        current_state(&fixture),
        OperationState::SummaryDeciding(deciding),
        "the append leaves the operation"
    );
    assert_eq!(fixture.lane.state().tip_id.as_deref(), Some("tip"));
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id: entry_id.clone(),
            kind: InboxItemKind::Write,
        }],
        "the queued write joins the inbox"
    );
    let durable = crate::harness::runtime::test_support::stored_value(
        &fixture.session,
        &stored_values::lane_state("main").address,
        &background_context(),
    )
    .await;
    let durable: crate::harness::session::types::LaneState =
        serde_json::from_value(durable.value).expect("the lane state parses");
    assert_eq!(
        durable,
        crate::harness::session::types::LaneState {
            current_operation_id: Some(OPERATION_ID.to_owned()),
            last_operation_id: None,
            inbox: vec![InboxItem {
                entry_id: entry_id.clone(),
                kind: InboxItemKind::Write,
            }],
        },
        "the durable lane record matches the projection"
    );
    let queued = common::stored_pending_entry(&fixture, &entry_id).await;
    assert!(queued.is_some(), "the queued write stores");
    let events = lock(&fixture.events).clone();
    let last = events.last().expect("the append publishes");
    let HarnessEvent {
        payload: HarnessEventPayload::QueueUpdate { queues },
        ..
    } = last
    else {
        panic!(
            "the append publishes queue_update: {:?}",
            last.payload.event_type()
        );
    };
    assert_eq!(
        queues,
        &vec![crate::harness::agent_harness::LaneQueuedItem::Custom {
            entry_id: entry_id.clone(),
            kind: InboxItemKind::Write,
            custom_type: "note".to_owned(),
            data: Some(json!({"pending": true})),
        }],
        "the queue view dereferences the write"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: publishes overflow preparation with the normalized response settlement.
#[tokio::test]
async fn publishes_overflow_preparation_with_the_normalized_response_settlement() {
    let fixture = create_fixture().await;
    let ready = AssistantReadyOperation {
        scope: run_scope(CompactionSettings {
            enabled: true,
            reserve_tokens: 1_000,
            keep_recent_tokens: 1,
        }),
        generation_context: GenerationContext {
            step_id: "step".to_owned(),
            trigger_entry_id: "tip".to_owned(),
            configuration: fixture.configuration.clone(),
            stream_options: AgentHarnessStreamOptions::default(),
            retry_policy: NormalizedRetryPolicy {
                max_attempts: 2,
                base_delay_ms: 10,
                max_agent_delay_ms: 30_000,
            },
            overflow_recovery_used: false,
        },
        next_attempt: 1,
    };
    install_operation(
        &fixture,
        OperationState::AssistantReady(ready.clone()),
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        InstallOptions {
            entries: vec![user_entry("tip", None, "large prompt")],
            ..InstallOptions::default()
        },
    )
    .await;
    fixture
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "",
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("prompt exceeds the context window".to_owned()),
                ..FauxAssistantMessageOptions::default()
            },
        ))]);

    let result = run_generation(&fixture.lane, &fixture.drive, GenerationLeaf::Ready(ready))
        .await
        .expect("the generation serves");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the overflow settles into compaction: {result:?}"
    );
    let deciding = match current_state(&fixture) {
        OperationState::SummaryDeciding(deciding) => deciding,
        other => panic!("overflow did not enter compaction: {:?}", other.at()),
    };
    assert_eq!(deciding.task.reason, Some(CompactionReason::Overflow));
    let ResultBoundary::ResumeCheckpoint { resume_after } = &deciding.task.boundary else {
        panic!("overflow has wrong boundary");
    };
    assert!(
        matches!(
            resume_after.continuation,
            Continuation::NeedAssistant {
                overflow_recovery_used: true
            }
        ),
        "the overflow recovery bound carries"
    );
    assert_eq!(resume_after.trigger_entry_id, "tip");
    let preparation = common::stored_preparation(&fixture, &deciding.task.task_id).await;
    assert!(
        preparation.is_some(),
        "the overflow persists its preparation"
    );
    let tip = fixture
        .lane
        .state()
        .tip_id
        .expect("the error response places");
    let response = fixture
        .session
        .get_entry(&tip, &background_context())
        .await
        .expect("the entry reads")
        .expect("the response places");
    let Entry::Message { body, .. } = &response else {
        panic!("the response places as a message");
    };
    let AgentMessage::Standard(Message::Assistant(message)) = &body.message else {
        panic!("the response message is an assistant message");
    };
    assert_eq!(message.stop_reason, StopReason::Error);
    let events = lock(&fixture.events).clone();
    assert!(
        events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::CompactionStart),
        "the overflow starts compaction"
    );

    common::close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the hook compaction and the terminal cleanup in one commit"
)]
/// Upstream `it`: publishes a hook compaction and terminal cleanup atomically without assistant lifecycle.
#[tokio::test]
async fn publishes_a_hook_compaction_and_terminal_cleanup_atomically_without_assistant_lifecycle() {
    let fixture = create_fixture().await;
    let deciding = SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: SummaryTask {
            task_id: "task".to_owned(),
            reason: None,
            custom_instructions: Some("focus".to_owned()),
            boundary: ResultBoundary::Finish,
        },
    };
    install_operation(
        &fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Compaction {
            custom_instructions: Some("focus".to_owned()),
        },
        common::tip_install_options(),
    )
    .await;
    let queued_id = fixture
        .lane
        .append_custom_entry_impl(
            "retained",
            Some(json!({"after": "compaction"})),
            &background_context(),
        )
        .await
        .expect("the append serves");
    fixture
        .hooks
        .on(
            HookName::BeforeCompaction,
            fixed_hook(HookResult::BeforeCompaction(Some(CompactionHookResult {
                decline: None,
                compaction: Some(CompactResult {
                    summary: "hook summary".to_owned(),
                    tokens_before: 1_000,
                    usage: None,
                    retained_tail: vec![user("tail", 1)],
                    details: Some(json!({"source": "hook"})),
                }),
            }))),
            HookOptions::default(),
        )
        .expect("the compaction hook registers");

    let result = run_structural_decision(&fixture.lane, &fixture.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the hook compaction settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Compaction);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    assert!(
        fixture.lane.state().operation.is_none(),
        "the settled operation clears"
    );
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id: queued_id.clone(),
            kind: InboxItemKind::Write,
        }],
        "the queued write survives the cleanup"
    );
    let queued = common::stored_pending_entry(&fixture, &queued_id).await;
    assert!(queued.is_some(), "the queued write's payload survives");
    let tip = fixture.lane.state().tip_id.expect("the compaction places");
    let entry = fixture
        .session
        .get_entry(&tip, &background_context())
        .await
        .expect("the entry reads")
        .expect("the compaction entry places");
    let Entry::Compaction { body, .. } = &entry else {
        panic!("the compaction entry places");
    };
    assert_eq!(body.summary, "hook summary");
    assert!(body.from_hook, "the hook produced the entry");
    assert!(entry.seq() > 0, "the entry's seq assigns");
    assert_eq!(entry.timestamp(), 100, "the storage clock stamps the entry");
    let meta = fixture
        .session
        .get_value(
            &stored_values::operation_meta(OPERATION_ID).address,
            &background_context(),
        )
        .await
        .expect("the meta reads");
    assert!(meta.is_none(), "the cleanup deletes the operation meta");
    let preparation = common::stored_preparation(&fixture, "task").await;
    assert!(preparation.is_none(), "the cleanup deletes the preparation");
    let events = lock(&fixture.events).clone();
    assert!(
        !events.iter().any(|event| matches!(
            event.payload,
            HarnessEventPayload::MessageStart { .. } | HarnessEventPayload::MessageEnd { .. }
        )),
        "the hook compaction never carries the assistant lifecycle"
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::EntryAdded),
        "the compaction entry publishes"
    );
    let last = events.last().expect("the settlement publishes");
    let HarnessEvent {
        payload: HarnessEventPayload::CompactionEnd { reason, status, .. },
        ..
    } = last
    else {
        panic!("the compaction ends: {:?}", last.payload.event_type());
    };
    assert_eq!(*reason, CompactionReason::Manual);
    assert!(matches!(status, CompactionEndStatus::Completed { .. }));

    common::close_session(&fixture).await;
}

/// Upstream `it`: terminal-declines standalone compaction without publishing an entry.
#[tokio::test]
async fn terminal_declines_standalone_compaction_without_publishing_an_entry() {
    let fixture = create_fixture().await;
    let deciding = deciding_standalone();
    install_deciding_compaction(&fixture, &deciding).await;
    register_decline_hook(&fixture);

    let result = run_structural_decision(&fixture.lane, &fixture.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the decline settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Compaction);
    assert_eq!(outcome.status, TerminalStatus::Declined);
    assert_eq!(outcome.tip_id.as_deref(), Some("tip"));
    assert!(
        fixture.lane.state().operation.is_none(),
        "the declined operation clears"
    );
    assert_eq!(fixture.lane.state().tip_id.as_deref(), Some("tip"));
    let events = lock(&fixture.events).clone();
    let last = events.last().expect("the decline publishes");
    let HarnessEvent {
        payload: HarnessEventPayload::CompactionEnd { status, .. },
        ..
    } = last
    else {
        panic!("the compaction ends: {:?}", last.payload.event_type());
    };
    assert!(matches!(status, CompactionEndStatus::Declined));

    common::close_session(&fixture).await;
}

/// Upstream `it`: terminal-fails overflow decline while preserving lane-owned input.
#[tokio::test]
async fn terminal_fails_overflow_decline_while_preserving_lane_owned_input() {
    let fixture = create_fixture().await;
    let deciding = overflow_deciding();
    install_deciding_run(&fixture, &deciding).await;
    let queued_id = fixture
        .lane
        .append_custom_entry_impl(
            "retained",
            Some(json!({"value": true})),
            &background_context(),
        )
        .await
        .expect("the append serves");
    register_decline_hook(&fixture);

    let result = run_structural_decision(&fixture.lane, &fixture.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the overflow decline settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome
            .error
            .as_ref()
            .expect("the failure carries its error")
            .code,
        "compaction_declined"
    );
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id: queued_id.clone(),
            kind: InboxItemKind::Write,
        }],
        "the lane-owned input survives"
    );
    let queued = common::stored_pending_entry(&fixture, &queued_id).await;
    assert!(queued.is_some(), "the queued write's payload survives");
    let events = lock(&fixture.events).clone();
    let tail = &events[events.len().saturating_sub(2)..];
    assert_eq!(tail.len(), 2, "the tail pairs the decline with the run end");
    let HarnessEvent {
        payload: HarnessEventPayload::CompactionEnd { status, .. },
        ..
    } = &tail[0]
    else {
        panic!("the compaction ends: {:?}", tail[0].payload.event_type());
    };
    assert!(matches!(status, CompactionEndStatus::Declined));
    let HarnessEvent {
        payload: HarnessEventPayload::RunEnd { status, .. },
        ..
    } = &tail[1]
    else {
        panic!("the run ends: {:?}", tail[1].payload.event_type());
    };
    assert!(matches!(status, RunEndStatus::Failed { .. }));

    common::close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins one intent and usage row per split-turn request"
)]
/// Upstream `it`: gives each split-turn provider request its own durable intent and usage row.
#[tokio::test]
async fn gives_each_split_turn_provider_request_its_own_durable_intent_and_usage_row() {
    let fixture = create_fixture().await;
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        standalone_compaction_task(None),
        &fixture.configuration,
    );
    let mut preparation = compaction_preparation();
    match &mut preparation {
        DurableStructuralPreparation::Compaction {
            messages_to_summarize,
            turn_prefix_messages,
            is_split_turn,
            ..
        } => {
            *messages_to_summarize = vec![user("old history", 1)];
            *turn_prefix_messages = vec![user("large turn", 1)];
            *is_split_turn = true;
        }
        DurableStructuralPreparation::BranchSummary { .. } => {
            panic!("the compaction preparation builds");
        }
    }
    common::install_standalone_summary_ready_with(
        &fixture,
        InstallOptions {
            preparation: Some(PreparationWrite {
                task_id: "task".to_owned(),
                value: preparation,
            }),
            ..common::run_tip_install_options()
        },
    )
    .await;
    fixture.faux.set_responses([
        FauxResponseStep::Message(faux_assistant_message(
            "history summary",
            FauxAssistantMessageOptions::default(),
        )),
        FauxResponseStep::Message(faux_assistant_message(
            "turn prefix summary",
            FauxAssistantMessageOptions::default(),
        )),
    ]);

    let result = run_structural_generation(&fixture.lane, &fixture.drive, &ready)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the split turn settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Compaction);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    let attempts = fixture.storage.get_commit_attempts();
    let mut request_indices = Vec::new();
    for write in attempts.iter().flatten() {
        if let Write::ValueSet(value) = write
            && value.namespace == "pi.op.state"
            && let Ok(state) = serde_json::from_value::<OperationState>(value.value.clone())
            && let OperationState::SummaryEffectPending(effect) = &state
            && let Some(request) = &effect.request
        {
            request_indices.push(request.index);
        }
    }
    assert_eq!(
        request_indices,
        vec![0, 1],
        "each request carries its index"
    );
    let usage_writes = attempts
        .iter()
        .flatten()
        .filter(|write| matches!(write, Write::Usage(_)))
        .count();
    assert_eq!(usage_writes, 2, "each request writes its usage row");
    let frame_writes = attempts.iter().flatten().any(|write| match write {
        Write::ListAppend(list) => list.namespace == "pi.pending.assistant_frame",
        Write::ListDelete(list) => list.namespace == "pi.pending.assistant_frame",
        _ => false,
    });
    assert!(
        !frame_writes,
        "the structural requests never write assistant frames"
    );
    let events = lock(&fixture.events).clone();
    assert!(
        !events.iter().any(|event| matches!(
            event.payload,
            HarnessEventPayload::MessageStart { .. }
                | HarnessEventPayload::MessageUpdate { .. }
                | HarnessEventPayload::MessageEnd { .. }
        )),
        "the structural generation never publishes message lifecycle"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::Usage)
            .count(),
        2,
        "each request publishes its usage"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: preserves the overflow recovery bound when compaction resumes generation.
#[tokio::test]
async fn preserves_the_overflow_recovery_bound_when_compaction_resumes_generation() {
    let fixture = create_fixture().await;
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        run_compaction_task(
            CompactionReason::Overflow,
            CheckpointData {
                continuation: Continuation::NeedAssistant {
                    overflow_recovery_used: true,
                },
                trigger_entry_id: "tip".to_owned(),
            },
        ),
        &fixture.configuration,
    );
    common::install_ready_run(&fixture, &ready).await;
    fixture
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "generated summary",
            FauxAssistantMessageOptions::default(),
        ))]);

    common::generation_continues(&fixture, &ready, "the generated summary routes").await;
    let routed = match current_state(&fixture) {
        OperationState::AssistantReady(routed) => routed,
        other => panic!(
            "generated compaction did not route to generation: {:?}",
            other.at()
        ),
    };
    assert_eq!(routed.generation_context.trigger_entry_id, "tip");
    assert!(
        routed.generation_context.overflow_recovery_used,
        "the bound carries"
    );
    assert_eq!(
        fixture.lane.state().tip_id.as_deref(),
        Some("summary-entry")
    );
    let entry = fixture
        .session
        .get_entry("summary-entry", &background_context())
        .await
        .expect("the entry reads")
        .expect("the summary entry places");
    let Entry::Compaction { body, .. } = &entry else {
        panic!("the summary entry places");
    };
    assert_eq!(body.summary, "generated summary");
    assert!(!body.from_hook, "the generation produced the entry");

    common::close_session(&fixture).await;
}

/// Upstream `it`: settles structural usage without faulting when durable cancellation aborts the request.
#[tokio::test]
async fn settles_structural_usage_without_faulting_when_durable_cancellation_aborts_the_request() {
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    let (started_tx, started_rx) = deferred();
    let (release_tx, release_rx) = deferred();
    let started = Arc::new(Mutex::new(Some(started_tx)));
    let release = Arc::new(Mutex::new(Some(release_rx)));
    fixture
        .faux
        .set_responses([FauxResponseStep::Factory(Arc::new(
            move |_context: &pi_ai::types::Context,
                  options: Option<&SimpleStreamOptions>,
                  _state: &pi_ai::providers::faux::FauxProviderState,
                  _model: &pi_ai::types::Model| {
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                Box::pin(async move {
                    release_parked(started, release, "the request releases once").await;
                    let aborted = options.is_some_and(|options| {
                        options
                            .transport_options
                            .signal
                            .as_ref()
                            .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
                    });
                    Ok(faux_assistant_message(
                        "",
                        FauxAssistantMessageOptions {
                            stop_reason: Some(if aborted {
                                StopReason::Aborted
                            } else {
                                StopReason::Stop
                            }),
                            error_message: aborted.then(|| "cancelled".to_owned()),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ))
                })
            },
        ))]);

    let running = spawn_generation_attempt(&fixture, &ready);
    started_rx.await.expect("the request starts");
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(());
    fixture.drive.begin_abort(cancel_rx);
    cancel_operation(&fixture).await;
    let _ = cancel_tx.send(());
    fixture.drive.signal_abort();
    // The request's cancellation token links to the gate signal through a
    // spawned task; let it cancel before the factory resumes, upstream's
    // synchronous `signal.aborted` read.
    settle_events().await;
    let _ = release_tx.send(());

    let result = running
        .await
        .expect("the attempt joins")
        .expect("the attempt serves");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled attempt continues: {result:?}"
    );
    let cancelled = match current_state(&fixture) {
        OperationState::SummaryEffectPending(cancelled) => cancelled,
        other => panic!(
            "cancelled generation did not remain effect-pending for reconciliation: {:?}",
            other.at()
        ),
    };
    assert!(
        matches!(cancelled.scope.control, Control::CancelRequested { .. }),
        "the cancellation carries"
    );
    assert!(cancelled.request.is_none(), "the settled request clears");
    assert_eq!(cancelled.usage_ids.len(), 1, "the usage row settles");
    let usage_writes = common::usage_write_count(&fixture);
    assert_eq!(usage_writes, 1, "the usage row writes once");

    common::close_session(&fixture).await;
}

/// Upstream `it`: fails missing in-run structural models with configuration provenance.
#[tokio::test]
async fn fails_missing_in_run_structural_models_with_configuration_provenance() {
    let fixture = create_fixture().await;
    let missing = LaneConfiguration {
        model: ModelIdentity {
            provider: "missing".to_owned(),
            model_id: "missing".to_owned(),
        },
        ..fixture.configuration.clone()
    };
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        run_compaction_task(
            CompactionReason::Threshold,
            CheckpointData {
                continuation: Continuation::NeedAssistant {
                    overflow_recovery_used: false,
                },
                trigger_entry_id: "tip".to_owned(),
            },
        ),
        &missing,
    );
    common::install_ready_run(&fixture, &ready).await;

    let result = run_structural_generation(&fixture.lane, &fixture.drive, &ready)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the in-run attempt fails");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome
            .error
            .as_ref()
            .expect("the failure carries its error")
            .code,
        "model_unavailable"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the failed operation clears"
    );
    let usage_writes = common::usage_write_count(&fixture);
    assert_eq!(usage_writes, 0, "the missing model never reserves usage");

    let standalone = create_fixture().await;
    let standalone_missing = LaneConfiguration {
        model: ModelIdentity {
            provider: "missing".to_owned(),
            model_id: "missing".to_owned(),
        },
        ..standalone.configuration.clone()
    };
    let standalone_ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        standalone_compaction_task(None),
        &standalone_missing,
    );
    install_operation(
        &standalone,
        OperationState::SummaryReady(standalone_ready.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        InstallOptions {
            entries: vec![user_entry("standalone-tip", None, "history")],
            preparation: Some(PreparationWrite {
                task_id: "task".to_owned(),
                value: compaction_preparation(),
            }),
            ..InstallOptions::default()
        },
    )
    .await;
    let result = run_structural_generation(&standalone.lane, &standalone.drive, &standalone_ready)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the standalone attempt fails");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Compaction);
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome
            .error
            .as_ref()
            .expect("the failure carries its error")
            .code,
        "model_unavailable"
    );
    assert!(
        standalone.lane.state().operation.is_none(),
        "the failed standalone operation clears"
    );

    common::close_session(&fixture).await;
    common::close_session(&standalone).await;
}

/// Upstream `it`: durably brackets a delayed structural retry through success.
#[tokio::test]
async fn durably_brackets_a_delayed_structural_retry_through_success() {
    let _clock = pin_clock(1_000);
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    fixture.faux.set_responses([error_response()]);

    common::generation_continues(&fixture, &ready, "retryable failure did not wait").await;
    let retry = match current_state(&fixture) {
        OperationState::SummaryRetryWait(retry) => retry,
        other => panic!("retryable failure did not wait: {:?}", other.at()),
    };
    assert_eq!(retry.retry_wait.next_attempt, 2);
    let result = run_structural_retry_wait(&fixture.lane, &fixture.drive, &retry)
        .await
        .expect("the wait serves");
    match result {
        ProcedureResult::Waiting { outcome } => {
            let DriveOutcome::Waiting {
                operation_id,
                reason,
            } = outcome
            else {
                panic!("the wait waits: {outcome:?}");
            };
            assert_eq!(operation_id, OPERATION_ID);
            let DriveWaitReason::Retry { not_before } = reason else {
                panic!("the wait reasons retry: {reason:?}");
            };
            assert_eq!(
                not_before, retry.retry_wait.not_before,
                "the durable deadline echoes"
            );
        }
        other => panic!("retryable failure did not wait: {other:?}"),
    }
    set_test_now(retry.retry_wait.not_before);
    let result = run_structural_retry_wait(&fixture.lane, &fixture.drive, &retry)
        .await
        .expect("the wait serves");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the elapsed wait becomes ready: {result:?}"
    );
    let second = match current_state(&fixture) {
        OperationState::SummaryReady(second) => second,
        other => panic!("elapsed retry did not become ready: {:?}", other.at()),
    };
    fixture
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "summary",
            FauxAssistantMessageOptions::default(),
        ))]);
    let result = run_structural_generation(&fixture.lane, &fixture.drive, &second)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the retried attempt settles");
    assert_eq!(outcome.status, TerminalStatus::Completed);
    let events = lock(&fixture.events).clone();
    let retry_events: Vec<(&'static str, Option<bool>)> = events
        .iter()
        .filter_map(|event| match &event.payload {
            HarnessEventPayload::RetryScheduled { .. } => Some(("retry_scheduled", None)),
            HarnessEventPayload::RetryStart { .. } => Some(("retry_start", None)),
            HarnessEventPayload::RetryEnd { success, .. } => Some(("retry_end", Some(*success))),
            _ => None,
        })
        .collect();
    assert_eq!(
        retry_events,
        vec![
            ("retry_scheduled", None),
            ("retry_start", None),
            ("retry_end", Some(true)),
        ],
        "the retry lifecycle brackets the bracket"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: closes an exhausted structural retry with its final error.
#[tokio::test]
async fn closes_an_exhausted_structural_retry_with_its_final_error() {
    let _clock = pin_clock(2_000);
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    fixture.faux.set_responses([error_response()]);
    run_structural_generation(&fixture.lane, &fixture.drive, &ready)
        .await
        .expect("the attempt serves");
    let retry = match current_state(&fixture) {
        OperationState::SummaryRetryWait(retry) => retry,
        other => panic!("retryable failure did not wait: {:?}", other.at()),
    };
    let second = elapsed_retry_ready(&fixture, &retry).await;
    fixture.faux.set_responses([error_response()]);

    let result = run_structural_generation(&fixture.lane, &fixture.drive, &second)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the exhausted retry settles");
    assert_eq!(outcome.status, TerminalStatus::Failed);
    let events = lock(&fixture.events).clone();
    let retry_ends: Vec<&HarnessEventPayload> = events
        .iter()
        .filter(|event| event.event_type() == HarnessEventType::RetryEnd)
        .map(|event| &event.payload)
        .collect();
    let [only] = retry_ends.as_slice() else {
        panic!("the exhausted retry ends once: {}", retry_ends.len())
    };
    let HarnessEventPayload::RetryEnd {
        attempt,
        success,
        final_error,
        ..
    } = only
    else {
        panic!("the retry end carries its attempt");
    };
    assert_eq!(*attempt, 2);
    assert!(!success);
    let final_error = final_error.as_ref().expect("the final error carries");
    assert!(
        final_error.contains("rate limit exceeded"),
        "the final error carries the failure: {final_error}"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: finishes a standalone structural failure at the retry cap.
#[tokio::test]
async fn finishes_a_standalone_structural_failure_at_the_retry_cap() {
    let fixture = create_fixture().await;
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        standalone_compaction_task(None),
        &fixture.configuration,
    );
    let mut ready = ready;
    ready.generation.summary_context.retry_policy = NormalizedRetryPolicy {
        max_attempts: 1,
        base_delay_ms: 10,
        max_agent_delay_ms: 30_000,
    };
    install_operation(
        &fixture,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        common::tip_install_options(),
    )
    .await;
    fixture.faux.set_responses([error_response()]);

    let result = run_structural_generation(&fixture.lane, &fixture.drive, &ready)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the capped attempt settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Compaction);
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome
            .error
            .as_ref()
            .expect("the failure carries its error")
            .code,
        "summarization_failed"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the failed operation clears"
    );
    let events = lock(&fixture.events).clone();
    assert!(
        !events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::RetryScheduled),
        "the capped attempt never schedules a retry"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: rejects invalid unsummarized navigation state without committing.
#[tokio::test]
async fn rejects_invalid_unsummarized_navigation_state_without_committing() {
    for invalid in ["missing", "source", "root_label"] {
        let fixture = create_fixture().await;
        let (target_id, label) = match invalid {
            "missing" => (Some("missing"), None),
            "source" => (Some("source"), None),
            "root_label" => (None, Some("invalid")),
            other => panic!("the invalid case builds: {other}"),
        };
        let navigation = NavigationReadyToCommitOperation {
            scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
            target_id: target_id.map(str::to_owned),
            label: label.map(str::to_owned),
        };
        install_operation(
            &fixture,
            OperationState::NavigationReadyToCommit(navigation.clone()),
            OperationIntent::Navigation {
                target_id: target_id.map(str::to_owned),
                summarize: false,
                label: label.map(str::to_owned),
                custom_instructions: None,
            },
            InstallOptions {
                entries: vec![user_entry("source", None, "source")],
                tip_id: Some(Some("source".to_owned())),
                ..InstallOptions::default()
            },
        )
        .await;

        let error = common::navigation_rejects(&fixture, &navigation, "navigation").await;
        assert!(
            !error.to_string().is_empty(),
            "the {invalid} navigation rejects with its invariant: {error}"
        );
        assert!(
            fixture.storage.get_commit_attempts().is_empty(),
            "the {invalid} navigation never commits"
        );
        assert_eq!(
            fixture.lane.state().tip_id.as_deref(),
            Some("source"),
            "the {invalid} navigation never moves the tip"
        );

        common::close_session(&fixture).await;
    }
}

/// Upstream `it`: moves an unsummarized navigation and cleans up in one terminal transaction.
#[tokio::test]
async fn moves_an_unsummarized_navigation_and_cleans_up_in_one_terminal_transaction() {
    let fixture = create_fixture().await;
    let navigation = NavigationReadyToCommitOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        target_id: Some("target".to_owned()),
        label: Some("chosen".to_owned()),
    };
    install_operation(
        &fixture,
        OperationState::NavigationReadyToCommit(navigation.clone()),
        OperationIntent::Navigation {
            target_id: Some("target".to_owned()),
            summarize: false,
            label: Some("chosen".to_owned()),
            custom_instructions: None,
        },
        InstallOptions {
            entries: vec![
                user_entry("root", None, "root"),
                user_entry("source", Some("root"), "source"),
                user_entry("target", Some("root"), "target"),
            ],
            tip_id: Some(Some("source".to_owned())),
            ..InstallOptions::default()
        },
    )
    .await;

    let result = commit_navigation(&fixture.lane, &fixture.drive, &navigation)
        .await
        .expect("the navigation serves");
    let outcome = settled(result, "the navigation settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Navigation);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    assert_eq!(outcome.from_tip_id.as_deref(), Some("source"));
    assert_eq!(outcome.tip_id.as_deref(), Some("target"));
    assert_eq!(fixture.lane.state().tip_id.as_deref(), Some("target"));
    let label = fixture
        .session
        .get_label("target", &background_context())
        .await
        .expect("the label reads");
    assert_eq!(label.as_deref(), Some("chosen"), "the label lands");
    let attempts = fixture.storage.get_commit_attempts();
    let writes = attempts.last().expect("the terminal transaction commits");
    assert!(
        writes.iter().any(
            |write| matches!(write, Write::ValueDelete(value) if value.namespace == "pi.op.state")
        ),
        "the cleanup deletes the operation state"
    );
    assert!(
        writes
            .iter()
            .any(|write| matches!(write, Write::ValueSet(value) if value.namespace == "pi.result")),
        "the terminal transaction writes the result record"
    );
    assert!(
        writes.iter().any(
            |write| matches!(write, Write::ValueSet(value) if value.namespace == "pi.lane.state")
        ),
        "the terminal transaction writes the lane record"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: publishes a hook navigation summary with the target parent and source identity.
#[tokio::test]
async fn publishes_a_hook_navigation_summary_with_the_target_parent_and_source_identity() {
    let fixture = create_fixture().await;
    let deciding = common::install_navigation_deciding(&fixture).await;
    fixture
        .hooks
        .on(
            HookName::BeforeNavigation,
            fixed_hook(HookResult::BeforeNavigation(Some(NavigationHookResult {
                decline: None,
                summary: Some(crate::harness::compaction::types::BranchSummaryResult {
                    summary: "branch summary".to_owned(),
                    usage: None,
                    read_files: vec!["read.ts".to_owned()],
                    modified_files: vec!["edit.ts".to_owned()],
                }),
            }))),
            HookOptions::default(),
        )
        .expect("the summary hook registers");

    let result = run_structural_decision(&fixture.lane, &fixture.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the navigation settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Navigation);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    let tip = fixture.lane.state().tip_id.expect("the summary places");
    let entry = fixture
        .session
        .get_entry(&tip, &background_context())
        .await
        .expect("the entry reads")
        .expect("the summary entry places");
    let Entry::BranchSummary { body, .. } = &entry else {
        panic!("the branch summary places");
    };
    assert_eq!(
        entry.parent_id(),
        Some("target"),
        "the summary parents the target"
    );
    assert_eq!(
        body.from_id.as_deref(),
        Some("source"),
        "the summary names its source"
    );
    assert_eq!(body.summary, "branch summary");
    assert!(body.from_hook, "the hook produced the summary");

    common::close_session(&fixture).await;
}

/// Upstream `it`: terminal-declines summarized navigation without moving the tip.
#[tokio::test]
async fn terminal_declines_summarized_navigation_without_moving_the_tip() {
    let fixture = create_fixture().await;
    let deciding = common::install_navigation_deciding(&fixture).await;
    fixture
        .hooks
        .on(
            HookName::BeforeNavigation,
            fixed_hook(HookResult::BeforeNavigation(Some(NavigationHookResult {
                decline: Some(true),
                summary: None,
            }))),
            HookOptions::default(),
        )
        .expect("the decline hook registers");

    let result = run_structural_decision(&fixture.lane, &fixture.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the navigation decline settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Navigation);
    assert_eq!(outcome.status, TerminalStatus::Declined);
    assert_eq!(outcome.tip_id.as_deref(), Some("source"));
    assert_eq!(fixture.lane.state().tip_id.as_deref(), Some("source"));
    let events = lock(&fixture.events).clone();
    let status = last_navigation_end(&events, "the decline publishes");
    assert!(matches!(status, NavigationEndStatus::Declined));

    common::close_session(&fixture).await;
}

/// Upstream `it`: generates and atomically publishes a navigation summary.
#[tokio::test]
async fn generates_and_atomically_publishes_a_navigation_summary() {
    let fixture = create_fixture().await;
    let ready = common::install_navigation_summary_ready(&fixture).await;
    fixture
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "generated branch summary",
            FauxAssistantMessageOptions::default(),
        ))]);

    let result = run_structural_generation(&fixture.lane, &fixture.drive, &ready)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the navigation settles");
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Navigation);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    assert_eq!(outcome.from_tip_id.as_deref(), Some("source"));
    assert_eq!(outcome.tip_id.as_deref(), Some("summary-entry"));
    let entry = fixture
        .session
        .get_entry("summary-entry", &background_context())
        .await
        .expect("the entry reads")
        .expect("the summary entry places");
    let Entry::BranchSummary { body, .. } = &entry else {
        panic!("the branch summary places");
    };
    assert_eq!(entry.parent_id(), Some("target"));
    assert_eq!(body.from_id.as_deref(), Some("source"));
    assert!(!body.from_hook, "the generation produced the summary");
    assert_eq!(
        common::usage_event_count(&fixture),
        1,
        "the generation publishes its usage",
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: consumes an orphaned structural attempt and never resumes its nested request.
#[tokio::test]
async fn consumes_an_orphaned_structural_attempt_and_never_resumes_its_nested_request() {
    let fixture = create_fixture().await;
    let effect = SummaryEffectPendingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        generation: SummaryGenerationScope {
            task: standalone_compaction_task(None),
            summary_context: SummaryContext {
                result_entry_id: "summary-entry".to_owned(),
                configuration: fixture.configuration.clone(),
                stream_options: AgentHarnessStreamOptions::default(),
                retry_policy: NormalizedRetryPolicy {
                    max_attempts: 2,
                    base_delay_ms: 10,
                    max_agent_delay_ms: 30_000,
                },
            },
        },
        attempt: 1,
        request: Some(SummaryEffectRequest {
            index: 1,
            usage_id: "abandoned-usage".to_owned(),
        }),
        usage_ids: vec!["settled-usage".to_owned()],
    };
    common::install_effect_pending_compaction(&fixture, &effect).await;

    common::recovery_continues(&fixture, &effect, "the orphan consumes").await;
    let retry = match current_state(&fixture) {
        OperationState::SummaryRetryWait(retry) => retry,
        other => panic!("orphan did not enter retry wait: {:?}", other.at()),
    };
    assert_eq!(retry.retry_wait.next_attempt, 2);
    // The retry-wait leaf has no nested-request field to resume — the type
    // drops it where upstream deletes the property.
    let usage_writes = common::usage_write_count(&fixture);
    assert_eq!(usage_writes, 0, "the orphan never writes usage");
    let events = lock(&fixture.events).clone();
    let last = events.last().expect("the recovery publishes");
    assert!(last.recovery, "the orphan's schedule flags recovery");
    let HarnessEvent {
        payload: HarnessEventPayload::RetryScheduled { attempt, .. },
        ..
    } = last
    else {
        panic!("the recovery schedules: {:?}", last.payload.event_type());
    };
    assert_eq!(*attempt, 2);

    common::close_session(&fixture).await;
}

/// Upstream `it`: terminal-fails an orphaned structural attempt at the retry cap.
#[tokio::test]
async fn terminal_fails_an_orphaned_structural_attempt_at_the_retry_cap() {
    let fixture = create_fixture().await;
    let effect = SummaryEffectPendingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        generation: SummaryGenerationScope {
            task: standalone_compaction_task(None),
            summary_context: SummaryContext {
                result_entry_id: "summary-entry".to_owned(),
                configuration: fixture.configuration.clone(),
                stream_options: AgentHarnessStreamOptions::default(),
                retry_policy: NormalizedRetryPolicy {
                    max_attempts: 1,
                    base_delay_ms: 10,
                    max_agent_delay_ms: 30_000,
                },
            },
        },
        attempt: 1,
        request: Some(SummaryEffectRequest {
            index: 0,
            usage_id: "abandoned-usage".to_owned(),
        }),
        usage_ids: Vec::new(),
    };
    common::install_effect_pending_compaction(&fixture, &effect).await;

    let result = recover_structural_generation(&fixture.lane, &fixture.drive, &effect)
        .await
        .expect("the recovery serves");
    let outcome = settled(result, "the capped orphan settles");
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome
            .error
            .as_ref()
            .expect("the failure carries its error")
            .code,
        "structural_interrupted"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the failed operation clears"
    );
    let events = lock(&fixture.events).clone();
    assert!(
        !events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::RetryScheduled),
        "the capped orphan never schedules a retry"
    );

    common::close_session(&fixture).await;
}

/// Upstream `it`: rejects a preparation whose durable kind contradicts the structural state.
#[tokio::test]
async fn rejects_a_preparation_whose_durable_kind_contradicts_the_structural_state() {
    let fixture = create_fixture().await;
    let deciding = deciding_standalone();
    install_operation(
        &fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        InstallOptions {
            entries: vec![user_entry("tip", None, "history")],
            preparation: Some(PreparationWrite {
                task_id: "task".to_owned(),
                value: branch_preparation(),
            }),
            ..InstallOptions::default()
        },
    )
    .await;

    let error = common::decision_rejects(&fixture, &deciding, "deciding").await;
    assert!(
        error
            .to_string()
            .contains("Structural task task is missing its compaction preparation"),
        "the contradiction carries its invariant: {error}"
    );

    common::close_session(&fixture).await;
}

/// The closed hook registry faults the structural attempt through the
/// request closure's fault recording, upstream's closed-registry throw
/// reaching the request closure and the catch rethrowing it.
#[tokio::test]
async fn a_closed_hook_registry_faults_the_structural_attempt_through_the_request() {
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    fixture.hooks.close("closed".to_owned());

    let error = common::generation_rejects(&fixture, &ready, "ready").await;
    assert_eq!(
        error.to_string(),
        "closed",
        "the registry's closed error carries",
    );
    common::close_session(&fixture).await;
}

/// The gate-aborted request settles cancelled: the aborted gate turns the
/// request's admission into the cancelled settlement, upstream's
/// `gate.admit` throwing `AbortRequested` inside the request closure.
#[tokio::test]
async fn an_aborted_gate_settles_the_structural_request_cancelled() {
    let fixture = create_fixture().await;
    let deciding = deciding_standalone();
    install_deciding_compaction(&fixture, &deciding).await;
    common::register_aborting_hook(
        &fixture,
        HookName::BeforeCompaction,
        HookResult::BeforeCompaction(None),
    );
    fixture
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "summary",
            FauxAssistantMessageOptions::default(),
        ))]);

    common::decision_continues(&fixture, &deciding, "the aborted decision continues").await;
    let state = current_state(&fixture);
    let OperationState::SummaryReady(ready) = state else {
        panic!("the decision readies: {:?}", state.at())
    };
    common::generation_continues(&fixture, &ready, "the cancelled attempt continues").await;
    let state = current_state(&fixture);
    assert_eq!(state.at(), "summary.effect_pending", "the leaf reconciles");
    common::close_session(&fixture).await;
}

// The remaining cases are port-local boundary tests: they bind uncovered
// arms the upstream suites do not reach. The upstream source at pin
// `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` stays the behavior oracle —
// every bound arm exists there, and invariant strings carry verbatim.

/// Replaces the live operation's control with `cancel_requested` for hooks
/// and faux factories that cannot borrow the fixture, the same commit
/// [`super::common::cancel_operation`] makes.
///
/// # Panics
/// The absence of a live operation or the commit's failure.
async fn flip_control_to_cancelled(lane: &Arc<Lane>, operation_id: &str) {
    let operation = lane.state().operation.expect("fixture has no operation");
    let state = operation_state_with_scope(
        &operation.state,
        OperationScope {
            control: Control::CancelRequested { requested_at: 2 },
            ..crate::harness::session::types::operation_scope_of(&operation.state)
        },
    );
    let operation_id = operation_id.to_owned();
    lane.command::<(), _>(
        move |projection, _session, _context| {
            let meta = operation.meta.clone();
            let state = state.clone();
            let operation_id = operation_id.clone();
            Box::pin(async move {
                let next = LaneState {
                    operation: Some(LiveOperation {
                        meta: meta.clone(),
                        state: state.clone(),
                    }),
                    ..projection.clone()
                };
                Ok(LaneCommand::Commit {
                    writes: vec![
                        stored_values::set_value_write(
                            &stored_values::operation_state(&operation_id),
                            state.clone(),
                        )
                        .expect("cancel write"),
                    ],
                    next,
                    materialize: Arc::new(|_commit| ()),
                    events: None,
                })
            })
        },
        &background_context(),
    )
    .await
    .expect("the cancel commits");
}

/// The summary retry-wait leaf the wait cases hand-drive, upstream's
/// `SummaryRetryWaitOperation` fixture with the case's deadline.
fn retry_leaf(fixture: &DriveFixture, not_before: i64) -> SummaryRetryWaitOperation {
    SummaryRetryWaitOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        generation: SummaryGenerationScope {
            task: standalone_compaction_task(None),
            summary_context: summary_context(&fixture.configuration),
        },
        retry_wait: RetryWait {
            next_attempt: 2,
            not_before,
            error_message: "rate limit exceeded".to_owned(),
        },
    }
}

/// The error-stop faux response, upstream's
/// `{ stopReason: "error", errorMessage: "rate limit exceeded" }` fixture.
fn error_response() -> FauxResponseStep {
    FauxResponseStep::Message(faux_assistant_message(
        "",
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some("rate limit exceeded".to_owned()),
            ..FauxAssistantMessageOptions::default()
        },
    ))
}

/// The raw malformed preparation write the malformed-preparation cases seed,
/// bypassing the typed serializer with a JSON scalar.
fn malformed_preparation_write() -> Write {
    let address = stored_values::operation_preparation(OPERATION_ID, "task");
    let address = address.address;
    Write::ValueSet(stored_values::ValueSetWrite {
        kind: "value".to_owned(),
        op: "set".to_owned(),
        namespace: address.namespace,
        key: address.key,
        value: json!("not a preparation"),
    })
}

/// Port-local boundary: the missing durable preparation invariants on both
/// structural leaves, upstream's `readStructuralPreparation` and
/// `readAttemptPreparation` absent-value arms.
#[tokio::test]
async fn binds_missing_durable_preparations_on_both_structural_leaves() {
    let fixture = create_fixture().await;
    let deciding =
        common::install_deciding_standalone_with(&fixture, common::run_tip_install_options()).await;
    let error = common::decision_rejects(&fixture, &deciding, "deciding").await;
    assert!(
        error
            .to_string()
            .contains("Structural task task is missing its compaction preparation"),
        "the missing preparation carries its invariant: {error}"
    );

    let ready_fixture = create_fixture().await;
    let ready = common::install_standalone_summary_ready_with(
        &ready_fixture,
        common::run_tip_install_options(),
    )
    .await;
    let error = common::generation_rejects(&ready_fixture, &ready, "ready").await;
    assert!(
        error
            .to_string()
            .contains("Structural task task has invalid durable preparation"),
        "the missing attempt preparation carries its invariant: {error}"
    );
    assert!(
        ready_fixture.storage.get_commit_attempts().is_empty(),
        "the failed reads never commit"
    );

    common::close_session(&ready_fixture).await;
}

/// Port-local boundary: the malformed stored value and the ready leaf's
/// kind-mismatched preparation, upstream's
/// `Structural preparation {operationId}:{taskId} is malformed` and
/// `{taskId} has invalid durable preparation` arms.
#[tokio::test]
async fn binds_malformed_and_kind_mismatched_durable_preparations() {
    let fixture = create_fixture().await;
    let deciding = common::install_deciding_standalone_with(
        &fixture,
        InstallOptions {
            extra_writes: vec![malformed_preparation_write()],
            ..common::run_tip_install_options()
        },
    )
    .await;
    let error = common::decision_rejects(&fixture, &deciding, "deciding").await;
    assert!(
        error.to_string().contains("is malformed"),
        "the malformed preparation carries its invariant: {error}"
    );

    let ready_fixture = create_fixture().await;
    let ready = common::install_standalone_summary_ready_with(
        &ready_fixture,
        InstallOptions {
            extra_writes: vec![malformed_preparation_write()],
            ..common::run_tip_install_options()
        },
    )
    .await;
    let error = common::generation_rejects(&ready_fixture, &ready, "ready").await;
    assert!(
        error.to_string().contains("is malformed"),
        "the malformed attempt preparation carries its invariant: {error}"
    );

    let mismatched = create_fixture().await;
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        standalone_compaction_task(None),
        &mismatched.configuration,
    );
    common::install_standalone_summary_ready_with(
        &mismatched,
        InstallOptions {
            preparation: Some(PreparationWrite {
                task_id: "task".to_owned(),
                value: branch_preparation(),
            }),
            ..common::run_tip_install_options()
        },
    )
    .await;
    let error = common::generation_rejects(&mismatched, &ready, "ready").await;
    assert!(
        error
            .to_string()
            .contains("Structural task task has invalid durable preparation"),
        "the kind-mismatched preparation carries its invariant: {error}"
    );

    for fixture in [&fixture, &ready_fixture, &mismatched] {
        assert!(
            fixture.storage.get_commit_attempts().is_empty(),
            "the failed reads never commit"
        );
        common::close_session(fixture).await;
    }
}

/// Port-local boundary: the decision's missing-navigation-target invariant,
/// upstream's `` `Navigation target {targetId} is missing` `` arm in
/// `readStructuralPreparation`.
#[tokio::test]
async fn binds_the_missing_navigation_target_invariant_in_the_decision() {
    let fixture = create_fixture().await;
    let deciding = SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: navigation_summary_task("nav-target", None),
    };
    install_operation(
        &fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Navigation {
            target_id: Some("nav-target".to_owned()),
            summarize: true,
            label: None,
            custom_instructions: None,
        },
        InstallOptions {
            entries: vec![user_entry("source", None, "source")],
            preparation: Some(PreparationWrite {
                task_id: "task".to_owned(),
                value: branch_preparation(),
            }),
            ..InstallOptions::default()
        },
    )
    .await;

    let error = common::decision_rejects(&fixture, &deciding, "deciding").await;
    assert!(
        error
            .to_string()
            .contains("Navigation target nav-target is missing"),
        "the missing target carries its invariant: {error}"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the failed read never commits"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: the in-run compaction reason invariant, upstream's
/// `` `In-run compaction task {taskId} is missing its reason` `` arm — a
/// corrupted durable task whose reason vanished.
#[tokio::test]
async fn binds_the_missing_in_run_compaction_reason_invariant() {
    let fixture = create_fixture().await;
    let deciding = SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: SummaryTask {
            task_id: "task".to_owned(),
            reason: None,
            custom_instructions: None,
            boundary: ResultBoundary::ResumeCheckpoint {
                resume_after: CheckpointData {
                    continuation: Continuation::NeedAssistant {
                        overflow_recovery_used: false,
                    },
                    trigger_entry_id: "tip".to_owned(),
                },
            },
        },
    };
    install_deciding_run(&fixture, &deciding).await;

    let error = common::decision_rejects(&fixture, &deciding, "deciding").await;
    assert!(
        error
            .to_string()
            .contains("In-run compaction task task is missing its reason"),
        "the missing reason carries its invariant: {error}"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: the aborted-response invariant, upstream's
/// `` `Structural provider response is aborted while durable control is
/// running` `` — a nested request settled `aborted` with no cancellation
/// recorded.
#[tokio::test]
async fn binds_the_aborted_response_invariant_under_running_control() {
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    fixture.faux.set_responses([common::aborted_response()]);

    let error = common::generation_rejects(&fixture, &ready, "ready").await;
    assert!(
        error
            .to_string()
            .contains("Structural provider response is aborted while durable control is running"),
        "the aborted response carries its invariant: {error}"
    );
    common::close_session(&fixture).await;

    let navigation = create_fixture().await;
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        navigation_summary_task("target", None),
        &navigation.configuration,
    );
    install_operation(
        &navigation,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Navigation {
            target_id: Some("target".to_owned()),
            summarize: true,
            label: None,
            custom_instructions: None,
        },
        common::navigation_install_options(),
    )
    .await;
    navigation
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "",
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Aborted),
                error_message: Some("cancelled".to_owned()),
                ..FauxAssistantMessageOptions::default()
            },
        ))]);
    let error = common::generation_rejects(&navigation, &ready, "ready").await;
    assert!(
        error
            .to_string()
            .contains("Structural provider response is aborted while durable control is running"),
        "the branch summary's aborted response carries its invariant: {error}"
    );

    common::close_session(&navigation).await;
}

/// Port-local boundary: a failed branch-summary generation flows through
/// the retry budget, upstream's `generateBranchSummaryWithRequest` error
/// path and `publishAttemptResult`'s retry-scheduling arm.
#[tokio::test]
async fn binds_a_failed_branch_summary_generation_through_the_retry_budget() {
    let fixture = create_fixture().await;
    let ready = common::install_navigation_summary_ready(&fixture).await;
    fixture.faux.set_responses([error_response()]);

    common::generation_continues(&fixture, &ready, "the failed branch summary waits").await;
    let retry = match current_state(&fixture) {
        OperationState::SummaryRetryWait(retry) => retry,
        other => panic!("failed branch summary did not wait: {:?}", other.at()),
    };
    assert_eq!(retry.retry_wait.next_attempt, 2);
    assert!(
        retry
            .retry_wait
            .error_message
            .contains("rate limit exceeded"),
        "the wait carries the failure: {}",
        retry.retry_wait.error_message
    );
    let events = lock(&fixture.events).clone();
    let scheduled = events
        .iter()
        .find(|event| event.event_type() == HarnessEventType::RetryScheduled)
        .expect("the retry schedules");
    let HarnessEvent {
        payload:
            HarnessEventPayload::RetryScheduled {
                attempt,
                max_attempts,
                delay_ms,
                error_message,
                ..
            },
        ..
    } = scheduled
    else {
        panic!("the schedule carries its fields");
    };
    assert_eq!(*attempt, 2);
    assert_eq!(*max_attempts, 2);
    assert_eq!(*delay_ms, 10);
    assert!(
        error_message.contains("rate limit exceeded"),
        "the schedule carries the failure: {error_message}"
    );

    common::close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case drives both summary outcomes' usage-read arms"
)]
/// Port-local boundary: hook results carrying usage publish the hook usage
/// row, upstream's `hookUsageId` reservation and the usage event riding the
/// base events — the compaction and branch-summary arms of
/// `StructuralOutcome`'s usage read.
#[tokio::test]
async fn binds_hook_usage_rows_on_both_summary_outcomes() {
    let usage = Usage {
        input: 1,
        output: 2,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 3,
        cost: UsageCost::default(),
    };

    let fixture = create_fixture().await;
    let deciding = deciding_standalone();
    install_deciding_compaction(&fixture, &deciding).await;
    fixture
        .hooks
        .on(
            HookName::BeforeCompaction,
            fixed_hook(HookResult::BeforeCompaction(Some(CompactionHookResult {
                decline: None,
                compaction: Some(CompactResult {
                    summary: "hook summary".to_owned(),
                    tokens_before: 1_000,
                    usage: Some(usage),
                    retained_tail: vec![user("tail", 1)],
                    details: None,
                }),
            }))),
            HookOptions::default(),
        )
        .expect("the compaction hook registers");
    let result = run_structural_decision(&fixture.lane, &fixture.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the hook compaction settles");
    assert_eq!(outcome.status, TerminalStatus::Completed);
    assert_eq!(
        common::usage_event_count(&fixture),
        1,
        "the hook usage publishes",
    );
    assert!(
        fixture
            .storage
            .get_commit_attempts()
            .iter()
            .flatten()
            .any(|write| matches!(write, Write::Usage(_))),
        "the hook usage row writes"
    );
    common::close_session(&fixture).await;

    let navigation = create_fixture().await;
    let deciding = SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: navigation_summary_task("target", None),
    };
    install_operation(
        &navigation,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Navigation {
            target_id: Some("target".to_owned()),
            summarize: true,
            label: None,
            custom_instructions: None,
        },
        common::navigation_install_options(),
    )
    .await;
    navigation
        .hooks
        .on(
            HookName::BeforeNavigation,
            fixed_hook(HookResult::BeforeNavigation(Some(NavigationHookResult {
                decline: None,
                summary: Some(crate::harness::compaction::types::BranchSummaryResult {
                    summary: "branch summary".to_owned(),
                    usage: Some(usage),
                    read_files: vec!["read.ts".to_owned()],
                    modified_files: vec!["edit.ts".to_owned()],
                }),
            }))),
            HookOptions::default(),
        )
        .expect("the summary hook registers");
    let result = run_structural_decision(&navigation.lane, &navigation.drive, &deciding)
        .await
        .expect("the decision serves");
    let outcome = settled(result, "the hook navigation settles");
    assert_eq!(outcome.status, TerminalStatus::Completed);
    let events = lock(&navigation.events).clone();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::Usage)
            .count(),
        1,
        "the hook usage publishes"
    );
    assert!(
        navigation
            .storage
            .get_commit_attempts()
            .iter()
            .flatten()
            .any(|write| matches!(write, Write::Usage(_))),
        "the hook usage row writes"
    );

    common::close_session(&navigation).await;
}

/// Port-local boundary: a missing model terminal-fails the navigation
/// boundary, upstream's `model_unavailable` error flowing through the
/// `commit_navigation` arm's failed `navigation_end`.
#[tokio::test]
async fn binds_a_missing_model_failing_the_navigation_terminal() {
    let fixture = create_fixture().await;
    let missing = LaneConfiguration {
        model: ModelIdentity {
            provider: "missing".to_owned(),
            model_id: "missing".to_owned(),
        },
        ..fixture.configuration.clone()
    };
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        navigation_summary_task("target", None),
        &missing,
    );
    common::install_navigation_ready(&fixture, &ready).await;

    let result = run_structural_generation(&fixture.lane, &fixture.drive, &ready)
        .await
        .expect("the attempt serves");
    let outcome = settled(result, "the missing model fails");
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome
            .error
            .as_ref()
            .expect("the failure carries its error")
            .code,
        "model_unavailable"
    );
    assert_eq!(outcome.tip_id.as_deref(), Some("source"));
    assert_eq!(fixture.lane.state().tip_id.as_deref(), Some("source"));
    let events = lock(&fixture.events).clone();
    let status = last_navigation_end(&events, "the failure publishes");
    assert!(
        matches!(status, NavigationEndStatus::Failed { .. }),
        "the navigation ends failed"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: every structural leaf downgrades a durable
/// `cancel_requested` control to `Continue` before its transaction, upstream's
/// `continueOperation` cancel check restated at each leaf's first reader.
#[tokio::test]
async fn binds_cancelled_controls_downgrading_the_structural_leaves_to_continue() {
    let cancelled_scope = scope_with_control(
        Control::CancelRequested { requested_at: 2 },
        DEFAULT_COMPACTION_SETTINGS,
    );

    let fixture = create_fixture().await;
    let deciding = SummaryDecidingOperation {
        scope: cancelled_scope.clone(),
        task: standalone_compaction_task(None),
    };
    install_deciding_compaction(&fixture, &deciding).await;
    common::decision_continues(&fixture, &deciding, "the cancelled decision continues").await;
    assert_eq!(
        current_state(&fixture),
        OperationState::SummaryDeciding(deciding),
        "the leaf stays"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the cancelled decision never commits"
    );
    common::close_session(&fixture).await;

    let fixture = create_fixture().await;
    let ready = SummaryReadyOperation {
        scope: cancelled_scope.clone(),
        generation: SummaryGenerationScope {
            task: standalone_compaction_task(None),
            summary_context: summary_context(&fixture.configuration),
        },
        next_attempt: 1,
    };
    install_operation(
        &fixture,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        common::tip_install_options(),
    )
    .await;
    common::generation_continues(&fixture, &ready, "the cancelled attempt continues").await;
    assert_eq!(
        current_state(&fixture),
        OperationState::SummaryReady(ready),
        "the leaf stays"
    );
    common::close_session(&fixture).await;

    let fixture = create_fixture().await;
    let effect = SummaryEffectPendingOperation {
        scope: cancelled_scope.clone(),
        generation: SummaryGenerationScope {
            task: standalone_compaction_task(None),
            summary_context: summary_context(&fixture.configuration),
        },
        attempt: 1,
        request: Some(SummaryEffectRequest {
            index: 0,
            usage_id: "abandoned-usage".to_owned(),
        }),
        usage_ids: Vec::new(),
    };
    common::install_effect_pending_compaction(&fixture, &effect).await;
    common::recovery_continues(&fixture, &effect, "the cancelled recovery continues").await;
    assert_eq!(
        current_state(&fixture),
        OperationState::SummaryEffectPending(effect),
        "the leaf stays"
    );
    common::close_session(&fixture).await;

    let fixture = create_fixture().await;
    let navigation = NavigationReadyToCommitOperation {
        scope: cancelled_scope,
        target_id: Some("target".to_owned()),
        label: None,
    };
    common::install_navigation_commit_leaf(&fixture, &navigation).await;
    common::navigation_continues(&fixture, &navigation, "the cancelled navigation continues").await;
    assert_eq!(
        current_state(&fixture),
        OperationState::NavigationReadyToCommit(navigation),
        "the leaf stays"
    );
    assert_eq!(fixture.lane.state().tip_id.as_deref(), Some("source"));
    common::close_session(&fixture).await;
}

/// Port-local boundary: the gate aborting between the `before_request` hook
/// and the provider admission turns the request cancelled, upstream's
/// `gate.admit` throwing `AbortRequested` inside the request closure.
#[tokio::test]
async fn binds_the_gate_aborting_between_the_hook_and_the_request() {
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    common::register_aborting_hook(
        &fixture,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(None),
    );

    common::generation_continues(&fixture, &ready, "the aborted admission continues").await;
    let effect = match current_state(&fixture) {
        OperationState::SummaryEffectPending(effect) => effect,
        other => panic!("the cancelled attempt lost its leaf: {:?}", other.at()),
    };
    assert!(
        effect.request.is_some(),
        "the unsettled request stays pending"
    );
    assert!(effect.usage_ids.is_empty(), "no usage settles");
    let usage_writes = common::usage_write_count(&fixture);
    assert_eq!(usage_writes, 0, "the aborted request writes no usage");

    common::close_session(&fixture).await;
}

/// Port-local boundary: durable cancellation arriving between the
/// `before_request` hook and the nested intent downgrades the request to
/// cancelled, upstream's `publishNestedRequestIntent` returning
/// `cancel_requested`; the hook's stream-options patch rides the same
/// run, upstream's `applyStreamOptionsPatch` call site.
#[tokio::test]
async fn binds_the_nested_intent_cancelled_inside_the_request() {
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    let lane_for_hook = Arc::clone(&fixture.lane);
    let operation_id = fixture.operation_id.clone();
    fixture
        .hooks
        .on(
            HookName::BeforeRequest,
            Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                let lane = Arc::clone(&lane_for_hook);
                let operation_id = operation_id.clone();
                Box::pin(async move {
                    flip_control_to_cancelled(&lane, &operation_id).await;
                    Ok::<HookResult, HookFailure>(HookResult::BeforeRequest(Some(
                        crate::harness::agent_harness::BeforeRequestResult {
                            stream_options:
                                crate::harness::types::AgentHarnessStreamOptionsPatch::default(),
                        },
                    )))
                })
            }),
            HookOptions::default(),
        )
        .expect("the cancelling hook registers");

    common::generation_continues(&fixture, &ready, "the cancelled intent continues").await;
    let effect = match current_state(&fixture) {
        OperationState::SummaryEffectPending(effect) => effect,
        other => panic!("the cancelled intent lost its leaf: {:?}", other.at()),
    };
    assert!(effect.request.is_none(), "the intent never lands");
    assert!(effect.usage_ids.is_empty(), "no usage settles");
    let usage_writes = common::usage_write_count(&fixture);
    assert_eq!(usage_writes, 0, "the cancelled request writes no usage");

    common::close_session(&fixture).await;
}

/// Port-local boundary: durable cancellation arriving while the nested
/// request runs downgrades the retry schedule to `Continue`, upstream's
/// retry-scheduling `continueOperation` returning `cancel_requested`.
#[tokio::test]
async fn binds_the_retry_schedule_cancelled_while_the_request_runs() {
    let fixture = create_fixture().await;
    let ready = install_standalone_summary_ready(&fixture).await;
    let (started_tx, started_rx) = deferred();
    let (release_tx, release_rx) = deferred();
    let started = Arc::new(Mutex::new(Some(started_tx)));
    let release = Arc::new(Mutex::new(Some(release_rx)));
    let lane_for_factory = Arc::clone(&fixture.lane);
    let operation_id = fixture.operation_id.clone();
    fixture
        .faux
        .set_responses([FauxResponseStep::Factory(Arc::new(
            move |_context: &pi_ai::types::Context,
                  _options: Option<&SimpleStreamOptions>,
                  _state: &pi_ai::providers::faux::FauxProviderState,
                  _model: &pi_ai::types::Model| {
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                let lane = Arc::clone(&lane_for_factory);
                let operation_id = operation_id.clone();
                Box::pin(async move {
                    flip_control_to_cancelled(&lane, &operation_id).await;
                    release_parked(started, release, "the request releases once").await;
                    Ok(faux_assistant_message(
                        "",
                        FauxAssistantMessageOptions {
                            stop_reason: Some(StopReason::Error),
                            error_message: Some("rate limit exceeded".to_owned()),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ))
                })
            },
        ))]);

    let running = spawn_generation_attempt(&fixture, &ready);
    started_rx.await.expect("the request starts");
    let _ = release_tx.send(());

    let result = running
        .await
        .expect("the attempt joins")
        .expect("the attempt serves");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled schedule continues: {result:?}"
    );
    let effect = match current_state(&fixture) {
        OperationState::SummaryEffectPending(effect) => effect,
        other => panic!("the cancelled schedule lost its leaf: {:?}", other.at()),
    };
    assert!(effect.request.is_none(), "the settled request clears");
    assert_eq!(effect.usage_ids.len(), 1, "the usage row settles");
    let events = lock(&fixture.events).clone();
    assert!(
        !events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::RetryScheduled),
        "the cancelled schedule never schedules a retry"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: an abort begun before the wait rejects the retry
/// wait's admitted sleep, upstream's `gate.admit` refusing the
/// `waitUntil` call.
#[tokio::test]
async fn binds_the_retry_waits_admission_rejection_on_abort() {
    let (fixture, drive, retry, _clock) = retry_wait_boundary_fixture().await;
    let (_, cancel_rx) = tokio::sync::watch::channel(());
    drive.begin_abort(cancel_rx);
    drive.signal_abort();

    let error = run_structural_retry_wait(&fixture.lane, &drive, &retry)
        .await
        .expect_err("the aborted wait rejects");
    assert!(
        error.to_string().contains("Abort requested"),
        "the abort rejection carries: {error}"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: durable cancellation arriving mid-wait downgrades
/// the retry wait's advance to `Continue`, upstream's wait
/// `continueOperation` returning `cancel_requested`.
#[tokio::test]
async fn binds_the_retry_waits_cancelled_downgrade_mid_wait() {
    let (fixture, drive, retry, _clock) = retry_wait_boundary_fixture().await;

    let running = {
        let lane = Arc::clone(&fixture.lane);
        let retry = retry.clone();
        tokio::spawn(async move { run_structural_retry_wait(&lane, &drive, &retry).await })
    };
    cancel_operation(&fixture).await;
    set_test_now(1_100);
    let result = running
        .await
        .expect("the wait joins")
        .expect("the wait serves");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled wait continues: {result:?}"
    );
    let leaf = match current_state(&fixture) {
        OperationState::SummaryRetryWait(leaf) => leaf,
        other => panic!("the leaf moved: {:?}", other.at()),
    };
    assert_eq!(
        leaf.generation, retry.generation,
        "the leaf's generation stays"
    );
    assert_eq!(leaf.retry_wait, retry.retry_wait, "the leaf's wait stays");
    assert!(
        matches!(leaf.scope.control, Control::CancelRequested { .. }),
        "the cancellation carries"
    );
    let events = lock(&fixture.events).clone();
    assert!(
        !events
            .iter()
            .any(|event| event.event_type() == HarnessEventType::RetryStart),
        "the cancelled wait never restarts"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: the tip-less structural invariants, upstream's
/// `` `Run compaction has no Branch tip` ``, `` `Failed run has no Branch
/// tip` ``, and `` `Standalone compaction has no Branch tip` ``.
#[tokio::test]
async fn binds_the_tip_less_structural_invariants() {
    let fixture = create_fixture().await;
    let deciding = SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: run_compaction_task(
            CompactionReason::Threshold,
            CheckpointData {
                continuation: Continuation::NeedAssistant {
                    overflow_recovery_used: false,
                },
                trigger_entry_id: "tip".to_owned(),
            },
        ),
    };
    let error = run_tip_less_decision(
        &fixture,
        &deciding,
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        "the tip-less threshold rejects",
    )
    .await;
    assert!(
        error
            .to_string()
            .contains("Run compaction has no Branch tip"),
        "the tip-less threshold carries its invariant: {error}"
    );
    common::close_session(&fixture).await;

    let fixture = create_fixture().await;
    let deciding = overflow_deciding();
    let error = run_tip_less_decision(
        &fixture,
        &deciding,
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        "the tip-less overflow rejects",
    )
    .await;
    assert!(
        error.to_string().contains("Failed run has no Branch tip"),
        "the tip-less overflow carries its invariant: {error}"
    );
    common::close_session(&fixture).await;

    let fixture = create_fixture().await;
    let deciding = deciding_standalone();
    let error = run_tip_less_decision(
        &fixture,
        &deciding,
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        "the tip-less standalone rejects",
    )
    .await;
    assert!(
        error
            .to_string()
            .contains("Standalone compaction has no Branch tip"),
        "the tip-less standalone carries its invariant: {error}"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: a closed hook registry faults the decision's
/// structural hooks, upstream's non-abort hook failures propagating through
/// `runStructuralDecision`'s two boundary branches (handler failures
/// restate as declines/results, so only the closed registry reaches the
/// fault arms).
#[tokio::test]
async fn binds_closed_hook_registries_faulting_the_decision_hooks() {
    let fixture = create_fixture().await;
    let deciding = common::install_navigation_deciding(&fixture).await;
    fixture.hooks.close("closed".to_owned());
    let error = common::decision_rejects(&fixture, &deciding, "deciding").await;
    assert!(
        error.to_string().contains("closed"),
        "the navigation hook fault propagates: {error}"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the faulted decision never commits"
    );
    common::close_session(&fixture).await;

    let fixture = create_fixture().await;
    let deciding = deciding_standalone();
    install_deciding_compaction(&fixture, &deciding).await;
    fixture.hooks.close("closed".to_owned());
    let error = common::decision_rejects(&fixture, &deciding, "deciding").await;
    assert!(
        error.to_string().contains("closed"),
        "the compaction hook fault propagates: {error}"
    );

    common::close_session(&fixture).await;
}

/// Port-local boundary: durable cancellation downgrades the checkpoint
/// leaf's threshold preparation to `None` before its planner, upstream's
/// `readBoundedEntries` cancel propagating through
/// `prepareCompactionThreshold`.
#[tokio::test]
async fn binds_the_checkpoint_cancelled_downgrade_before_the_threshold() {
    let fixture = create_fixture().await;
    let checkpoint = CheckpointOperation {
        scope: scope_with_control(
            Control::CancelRequested { requested_at: 2 },
            DEFAULT_COMPACTION_SETTINGS,
        ),
        checkpoint: CheckpointData {
            continuation: Continuation::NeedAssistant {
                overflow_recovery_used: false,
            },
            trigger_entry_id: "tip".to_owned(),
        },
    };
    install_operation(
        &fixture,
        OperationState::Checkpoint(checkpoint.clone()),
        common::run_tip_intent(),
        common::run_tip_install_options(),
    )
    .await;

    checkpoint_continues(&fixture, &checkpoint, "the cancelled checkpoint continues").await;
    assert_eq!(
        current_state(&fixture),
        OperationState::Checkpoint(checkpoint),
        "the leaf stays"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the cancelled threshold never commits"
    );

    common::close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case drives the bound, cancelled-bounded, tipped, and prepared-return arms"
)]
/// Port-local boundary: the overflow preparation's three suppression arms
/// and its prepared arm, upstream's `prepareOverflowCompaction`: the
/// overflow-recovery bound, the cancelled bounded read, the
/// compaction-tipped path, and the prepared return.
#[tokio::test]
async fn binds_the_overflow_preparation_boundaries() {
    let fixture = create_fixture().await;
    let faux_model = fixture.faux.first_model();
    let bound = overflow_generation(
        &fixture.configuration,
        &faux_model,
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        true,
    );
    install_operation(
        &fixture,
        OperationState::AssistantEffectPending(bound.clone()),
        common::run_tip_intent(),
        common::run_tip_install_options(),
    )
    .await;
    let prepared = prepare_overflow_compaction(&fixture.lane, &fixture.drive, &bound)
        .await
        .expect("the preparation serves");
    assert!(
        prepared.is_none(),
        "the overflow recovery bound suppresses preparation"
    );
    common::close_session(&fixture).await;

    let cancelled = create_fixture().await;
    let faux_model = cancelled.faux.first_model();
    let bound = overflow_generation(
        &cancelled.configuration,
        &faux_model,
        scope_with_control(
            Control::CancelRequested { requested_at: 2 },
            DEFAULT_COMPACTION_SETTINGS,
        ),
        false,
    );
    install_operation(
        &cancelled,
        OperationState::AssistantEffectPending(bound.clone()),
        common::run_tip_intent(),
        common::run_tip_install_options(),
    )
    .await;
    let prepared = prepare_overflow_compaction(&cancelled.lane, &cancelled.drive, &bound)
        .await
        .expect("the preparation serves");
    assert!(
        prepared.is_none(),
        "the cancelled bounded read skips the compaction"
    );
    common::close_session(&cancelled).await;

    let tipped = create_fixture().await;
    let faux_model = tipped.faux.first_model();
    let bound = overflow_generation(
        &tipped.configuration,
        &faux_model,
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        false,
    );
    install_operation(
        &tipped,
        OperationState::AssistantEffectPending(bound.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        InstallOptions {
            entries: vec![NewEntry::Compaction {
                id: "compacted".to_owned(),
                parent_id: None,
                body: CompactionEntryBody {
                    summary: "already compacted".to_owned(),
                    retained_tail: Vec::new(),
                    tokens_before: i64::try_from(faux_model.context_window)
                        .expect("the window fits"),
                    details: None,
                    usage: None,
                    from_hook: false,
                },
            }],
            ..InstallOptions::default()
        },
    )
    .await;
    let prepared = prepare_overflow_compaction(&tipped.lane, &tipped.drive, &bound)
        .await
        .expect("the preparation serves");
    assert!(
        prepared.is_none(),
        "the compaction-tipped path prepares nothing"
    );
    common::close_session(&tipped).await;

    let prepared_path = create_fixture().await;
    let faux_model = prepared_path.faux.first_model();
    let bound = overflow_generation(
        &prepared_path.configuration,
        &faux_model,
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        false,
    );
    install_operation(
        &prepared_path,
        OperationState::AssistantEffectPending(bound.clone()),
        common::run_tip_intent(),
        common::run_tip_install_options(),
    )
    .await;
    let prepared = prepare_overflow_compaction(&prepared_path.lane, &prepared_path.drive, &bound)
        .await
        .expect("the preparation serves");
    let prepared = prepared.expect("the overflow prepares");
    assert!(
        matches!(
            prepared.preparation,
            DurableStructuralPreparation::Compaction { .. }
        ),
        "the overflow preparation is a compaction"
    );
    common::close_session(&prepared_path).await;
}

/// The assistant effect leaf the overflow cases hand-drive, upstream's
/// `AssistantEffectPendingOperation` fixture over the faux model's
/// metadata: the model's own `maxTokens`/`contextWindow` ride the context.
fn overflow_generation(
    configuration: &LaneConfiguration,
    model: &pi_ai::types::Model,
    scope: OperationScope,
    overflow_recovery_used: bool,
) -> AssistantEffectPendingOperation {
    AssistantEffectPendingOperation {
        scope,
        generation_context: GenerationContext {
            step_id: "step".to_owned(),
            trigger_entry_id: "tip".to_owned(),
            configuration: configuration.clone(),
            stream_options: AgentHarnessStreamOptions::default(),
            retry_policy: NormalizedRetryPolicy {
                max_attempts: 2,
                base_delay_ms: 10,
                max_agent_delay_ms: 30_000,
            },
            overflow_recovery_used,
        },
        attempt: 1,
        response_entry_id: "response".to_owned(),
        usage_id: "usage".to_owned(),
        intended_output_limit: model.max_tokens,
        context_window: model.context_window,
    }
}

/// Port-local boundary: the live lane configuration's headers reach the
/// nested summary request, upstream's `requestStreamOptions` header
/// adapter.
#[tokio::test]
async fn forwards_the_live_stream_options_headers_into_the_nested_request() {
    let mut headers = BTreeMap::new();
    headers.insert("x-drive-suite".to_owned(), "structural".to_owned());
    let fixture = create_drive_fixture(DriveFixtureSpec {
        stream_options: AgentHarnessStreamOptions {
            headers: Some(headers.clone()),
            ..AgentHarnessStreamOptions::default()
        },
        ..structural_spec()
    })
    .await;
    let captured: Arc<Mutex<Option<pi_ai::types::ProviderHeaders>>> = Arc::new(Mutex::new(None));
    let captured_for_factory = Arc::clone(&captured);
    fixture
        .faux
        .set_responses([FauxResponseStep::Factory(Arc::new(
            move |_context: &pi_ai::types::Context,
                  options: Option<&SimpleStreamOptions>,
                  _state: &pi_ai::providers::faux::FauxProviderState,
                  _model: &pi_ai::types::Model| {
                let captured = Arc::clone(&captured_for_factory);
                Box::pin(async move {
                    *lock(&captured) = options.and_then(|options| options.headers.clone());
                    Ok(faux_assistant_message(
                        "summary",
                        FauxAssistantMessageOptions::default(),
                    ))
                })
            },
        ))]);

    // The decision path's `publish_structural_ready` re-reads the lane's
    // live stream options into the summary context, so the headers flow
    // through the ready transition, upstream's `summaryContext` read.
    let deciding = deciding_standalone();
    install_deciding_compaction(&fixture, &deciding).await;
    common::decision_continues(&fixture, &deciding, "the decision readies").await;
    let ready = match current_state(&fixture) {
        OperationState::SummaryReady(ready) => ready,
        other => panic!("the decision did not ready: {:?}", other.at()),
    };
    let result = run_structural_generation(&fixture.lane, &fixture.drive, &ready)
        .await
        .expect("the attempt serves");
    let _ = settled(result, "the headered attempt settles");
    let seen = lock(&captured)
        .clone()
        .expect("the nested request carries headers");
    let expected: pi_ai::types::ProviderHeaders = headers
        .into_iter()
        .map(|(key, value)| (key, Some(value)))
        .collect();
    assert_eq!(seen, expected, "the headers reach the nested request");

    common::close_session(&fixture).await;
}
