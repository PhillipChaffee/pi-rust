//! The total-drive and cancellation-reconciliation suite, ported 1:1 from
//! upstream `test/harness/runtime/drive-reconcile.test.ts` ("runtime total
//! drive" / "runtime cancellation reconciliation") at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the tests bind:
//! - the `Date.now() + 100_000` matrix leaves seed a fixed `notBefore` far
//!   past any wall clock the test can observe: those rows carry
//!   `cancel_requested` control, so the wait never engages and the instant
//!   stays inert;
//! - `vi.useFakeTimers`/`vi.setSystemTime` restates as
//!   `#[tokio::test(start_paused = true)]` + `tokio::time::advance`, with
//!   the retry-wait leaf's `notBefore` seeded one second past `now_ms()` —
//!   the wall clock `retry::wait_until` compares against, so paused tokio
//!   time can never fire the deadline and the abort marker alone resolves
//!   the wait;
//! - upstream's `setTimeout` spy (the timer-installation proof) restates by
//!   behavior: the pass parks on its wait while the abort marker commits,
//!   and no `retry_start` publishes;
//! - upstream's `cancelDeferred` override restates as a `ProviderStreams`
//!   wrapper over the faux core whose deferred cancellation always fails,
//!   registered under the same provider id;
//! - upstream's fail-closed `before_drive` rejection carries the handler's
//!   normalized error (`blocked drive`); the landed hook registry reports
//!   the original through its error reporter and rejects with the
//!   `before_drive handler failed` wrapper, so that case pins the wrapper.
//!
//! The boundary cases at the file's tail bind the reconciliation branches
//! upstream's suites never reach, against the oracle
//! `src/harness/runtime/drive/reconcile.ts` at the same pin: the
//! no-matching-operation and not-cancelled invariants, the run-intent
//! summary leaves' `compaction_end` matrix rows and their invalid-boundary
//! invariant, the missing-model skip, and the close-linked cancellation
//! token that tracks `drive.closeSignal` through the best-effort remote
//! cancellation.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_ai::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::FauxCore;
use pi_ai::providers::faux::FauxDeferredOptions;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_tool_call};
use pi_ai::types::BoxedFuture;
use pi_ai::types::DeferredCancelOptions;
use pi_ai::types::DeferredFetchOptions;
use pi_ai::types::DeferredHandle;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::ProviderHeaders;
use pi_ai::types::{ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions};
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::provider_retry::ProviderRequestError;
use pi_ai::utils::retry::RetryPolicy;
use tokio_util::sync::CancellationToken;

use crate::harness::agent_harness::AbortRequestOutcome;
use crate::harness::agent_harness::CompactionEndStatus;
use crate::harness::agent_harness::CompactionHookResult;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HarnessEventType;
use crate::harness::agent_harness::{
    HookFailure, HookInvocation, HookName, HookOptions, HookResult,
};
use crate::harness::compaction::types::CompactResult;
use crate::harness::compaction::types::{DEFAULT_COMPACTION_SETTINGS, create_file_ops};
use crate::harness::context::{Context, background_context};
use crate::harness::result::HarnessError;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::drive_operation;
use crate::harness::runtime::drive::reconcile::reconcile_operation;
use crate::harness::runtime::drive::structural::run_structural_decision;
use crate::harness::runtime::test_support::provider_streams_pass_through;
use crate::harness::runtime::test_support::{deferred, lane_state_write, lock};
use crate::harness::runtime::types::{Drive, LaneCommand, LaneState, ProcedureResult};
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::AssistantRetryWaitOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::DeferredSuspendedOperation;
use crate::harness::session::types::DurableStructuralPreparation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryQuery;
use crate::harness::session::types::EntryType;
use crate::harness::session::types::GenerationContext;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::StartingOperation;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryEffectPendingOperation;
use crate::harness::session::types::SummaryEffectRequest;
use crate::harness::session::types::SummaryGenerationScope;
use crate::harness::session::types::SummaryReadyOperation;
use crate::harness::session::types::SummaryRetryWaitOperation;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::types::{TerminalStatus, ToolBatch, ToolCall, ToolsOperation};
use crate::harness::session::values::Write;
use crate::harness::session::values::append_list_write;
use crate::harness::session::values::operation_meta;
use crate::harness::session::values::operation_preparation;
use crate::harness::session::values::operation_preparation_prefix;
use crate::harness::session::values::operation_state;
use crate::harness::session::values::operation_tool_args_prefix;
use crate::harness::session::values::operation_tool_memo_prefix;
use crate::harness::session::values::{pending_assistant_frames, pending_entry, set_value_write};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::AgentMessage;

use super::common;
use super::common::DriveFixture;
use super::common::DriveFixtureSpec;
use super::common::InstallOptions;
use super::common::OPERATION_ID;
use super::common::cancelled_control;
use super::common::create_drive_fixture;
use super::common::{install_operation, run_scope, scope_with_control, summary_context, user};

/// The retry deadline the matrix's waiting leaves seed, upstream's
/// `Date.now() + 100_000`: a fixed instant far past any wall clock the test
/// can observe, so the wait outlives the case without reading the clock.
const MATRIX_NOT_BEFORE: i64 = 9_000_000_000_000;

/// The fixture spec the reconciliation suite builds, upstream's
/// `createFixture` literals: the `faux` api with the zero-pending deferred
/// behavior and the `{ maxRetries: 1, baseDelayMs: 10 }` retry policy.
fn reconcile_spec() -> DriveFixtureSpec {
    DriveFixtureSpec {
        suite: "reconcile",
        watch_suite: "reconciliation",
        faux: RegisterFauxProviderOptions {
            api: Some("faux".to_owned()),
            deferred: Some(FauxDeferredOptions {
                pending_fetches: Some(0),
                ..FauxDeferredOptions::default()
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

/// The cancelled scope the reconciliation leaves carry, upstream's
/// `scope(control = cancelledControl())` default.
fn cancelled_scope() -> OperationScope {
    scope_with_control(cancelled_control(), DEFAULT_COMPACTION_SETTINGS)
}

/// The generation inputs the run leaves carry, upstream's `cases()`'s
/// `generation` literal: the `step`/`tip` ids over the `{ maxAttempts: 2,
/// baseDelayMs: 10, maxAgentDelayMs: 30_000 }` policy.
fn generation_context(configuration: &LaneConfiguration) -> GenerationContext {
    GenerationContext {
        step_id: "step".to_owned(),
        trigger_entry_id: "tip".to_owned(),
        configuration: configuration.clone(),
        stream_options: AgentHarnessStreamOptions::default(),
        retry_policy: NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: 10,
            max_agent_delay_ms: 30_000,
        },
        overflow_recovery_used: false,
    }
}

/// The default transcript entry the installs seed, upstream's
/// `installed.entries ?? [{ id: "tip", ... user("history") }]`.
fn history_entry() -> NewEntry {
    NewEntry::Message {
        id: "tip".to_owned(),
        parent_id: None,
        body: Box::new(MessageEntry {
            message: user("history", 1),
            terminate: None,
        }),
    }
}

/// The run intent over the default tip, upstream's
/// `{ kind: "run", promptEntryIds: ["tip"] }`.
fn run_intent() -> OperationIntent {
    OperationIntent::Run {
        prompt_entry_ids: vec!["tip".to_owned()],
    }
}

/// The deferred leaves' seeds, upstream's `deferredOperation(fixture,
/// effectPending)`: the `deferred-job` handle over the faux model, the
/// `deferred-source` entry carrying it under the `deferred` stop reason,
/// and the leaf at `poll` 0 (suspended) or 1 (effect pending).
/// One matrix row with the shared intent, entries, and no writes,
/// upstream's `cases()` rows: the state and terminal event types are the
/// row's differences.
fn row(state: OperationState, terminal_events: Vec<&'static str>) -> InstalledOperation {
    InstalledOperation {
        state,
        intent: run_intent(),
        entries: vec![history_entry()],
        writes: Vec::new(),
        terminal_events,
    }
}

struct DeferredSeeds {
    /// The durable leaf.
    state: OperationState,
    /// The source entry the leaf reads its handle from.
    entry: NewEntry,
}

/// Builds one deferred leaf's seeds, upstream's `deferredOperation`.
fn deferred_operation(fixture: &DriveFixture, effect_pending: bool) -> DeferredSeeds {
    let model = fixture.faux.first_model();
    let handle = DeferredHandle {
        provider: model.provider.0,
        model_id: model.id.clone(),
        api: model.api.0,
        id: "deferred-job".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    };
    let entry = NewEntry::Message {
        id: "deferred-source".to_owned(),
        parent_id: None,
        body: Box::new(MessageEntry {
            message: AgentMessage::Standard(Message::Assistant(faux_assistant_message(
                Vec::<FauxContentBlock>::new(),
                FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::Deferred),
                    deferred: Some(handle),
                    ..FauxAssistantMessageOptions::default()
                },
            ))),
            terminate: None,
        }),
    };
    let scope = DeferredScope {
        scope: cancelled_scope(),
        step_id: "step".to_owned(),
        source_entry_id: entry.id().to_owned(),
        poll: u64::from(effect_pending),
        configuration: fixture.configuration.clone(),
        stream_options: AgentHarnessStreamOptions::default(),
    };
    let state = if effect_pending {
        OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
            scope,
            response_entry_id: "deferred-response".to_owned(),
            usage_id: "deferred-usage".to_owned(),
        })
    } else {
        OperationState::DeferredSuspended(DeferredSuspendedOperation { deferred: scope })
    };
    DeferredSeeds { state, entry }
}

/// One hand-seeded leaf install, upstream's `InstalledOperation`.
struct InstalledOperation {
    /// The durable leaf.
    state: OperationState,
    /// The intent meta the install commits.
    intent: OperationIntent,
    /// The entries the install inserts ahead of the operation's writes.
    entries: Vec<NewEntry>,
    /// The writes the install commits between the tip and the operation's
    /// own writes, upstream's `installed.writes`.
    writes: Vec<Write>,
    /// The terminal event types the leaf's settle publishes, upstream's
    /// `terminalEvents`.
    terminal_events: Vec<&'static str>,
}

/// The 13-leaf cancellation matrix, upstream's `cases(fixture)`: every
/// durable leaf in dispatch order with its intent and terminal event types.
#[expect(
    clippy::too_many_lines,
    reason = "the table restates upstream's single cases() body one row per leaf"
)]
fn cases(fixture: &DriveFixture) -> Vec<InstalledOperation> {
    let generation = generation_context(&fixture.configuration);
    let run_task = SummaryTask {
        task_id: "run-summary".to_owned(),
        reason: Some(CompactionReason::Threshold),
        custom_instructions: None,
        boundary: ResultBoundary::ResumeCheckpoint {
            resume_after: CheckpointData {
                continuation: Continuation::NeedAssistant {
                    overflow_recovery_used: false,
                },
                trigger_entry_id: "tip".to_owned(),
            },
        },
    };
    let compaction_task = SummaryTask {
        task_id: "compaction".to_owned(),
        reason: Some(CompactionReason::Manual),
        custom_instructions: None,
        boundary: ResultBoundary::Finish,
    };
    let navigation_task = SummaryTask {
        task_id: "navigation".to_owned(),
        reason: None,
        custom_instructions: None,
        boundary: ResultBoundary::CommitNavigation {
            target_id: "target".to_owned(),
            label: None,
        },
    };
    let suspended = deferred_operation(fixture, false);
    let deferred_effect = deferred_operation(fixture, true);
    let planned = serde_json::from_value::<ToolCall>(serde_json::json!({
        "sourceIndex": 0,
        "resultEntryId": "tool-result",
        "status": "planned",
    }))
    .expect("the planned call wire");
    vec![
        row(
            OperationState::Starting(StartingOperation {
                scope: cancelled_scope(),
            }),
            vec!["run_end"],
        ),
        row(
            OperationState::Checkpoint(CheckpointOperation {
                scope: cancelled_scope(),
                checkpoint: CheckpointData {
                    continuation: Continuation::NeedAssistant {
                        overflow_recovery_used: false,
                    },
                    trigger_entry_id: "tip".to_owned(),
                },
            }),
            vec!["run_end"],
        ),
        row(
            OperationState::AssistantReady(AssistantReadyOperation {
                scope: cancelled_scope(),
                generation_context: generation.clone(),
                next_attempt: 1,
            }),
            vec!["run_end"],
        ),
        InstalledOperation {
            state: OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
                scope: cancelled_scope(),
                generation_context: generation.clone(),
                attempt: 1,
                response_entry_id: "assistant-response".to_owned(),
                usage_id: "assistant-usage".to_owned(),
                intended_output_limit: 100,
                context_window: 1_000,
            }),
            intent: run_intent(),
            entries: vec![history_entry()],
            writes: vec![
                append_list_write(
                    &pending_assistant_frames(OPERATION_ID, "assistant-response"),
                    AssistantMessageFrame::TextDelta {
                        content_index: 0,
                        delta: "partial".to_owned(),
                    },
                )
                .expect("the frame write"),
            ],
            terminal_events: vec!["run_end"],
        },
        row(
            OperationState::AssistantRetryWait(AssistantRetryWaitOperation {
                scope: cancelled_scope(),
                generation_context: generation,
                retry_wait: RetryWait {
                    next_attempt: 2,
                    not_before: MATRIX_NOT_BEFORE,
                    error_message: "retry".to_owned(),
                },
            }),
            vec!["run_end"],
        ),
        InstalledOperation {
            state: OperationState::Tools(ToolsOperation {
                scope: cancelled_scope(),
                batch: ToolBatch {
                    assistant_entry_id: "assistant".to_owned(),
                    configuration: fixture.configuration.clone(),
                    turn_id: "turn".to_owned(),
                    calls: vec![planned],
                },
            }),
            intent: OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            entries: vec![NewEntry::Message {
                id: "assistant".to_owned(),
                parent_id: None,
                body: Box::new(MessageEntry {
                    message: AgentMessage::Standard(Message::Assistant(faux_assistant_message(
                        faux_tool_call("tool", serde_json::Map::new(), None),
                        FauxAssistantMessageOptions {
                            stop_reason: Some(StopReason::ToolUse),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ))),
                    terminate: None,
                }),
            }],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
        InstalledOperation {
            state: suspended.state,
            intent: OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            entries: vec![suspended.entry],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
        InstalledOperation {
            state: deferred_effect.state,
            intent: OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            entries: vec![deferred_effect.entry],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
        InstalledOperation {
            state: OperationState::SummaryDeciding(SummaryDecidingOperation {
                scope: cancelled_scope(),
                task: run_task.clone(),
            }),
            intent: run_intent(),
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: vec!["compaction_end", "run_end"],
        },
        InstalledOperation {
            state: OperationState::SummaryReady(SummaryReadyOperation {
                scope: cancelled_scope(),
                generation: SummaryGenerationScope {
                    task: compaction_task,
                    summary_context: summary_context(&fixture.configuration),
                },
                next_attempt: 1,
            }),
            intent: OperationIntent::Compaction {
                custom_instructions: None,
            },
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: vec!["compaction_end"],
        },
        InstalledOperation {
            state: OperationState::SummaryEffectPending(SummaryEffectPendingOperation {
                scope: cancelled_scope(),
                generation: SummaryGenerationScope {
                    task: navigation_task,
                    summary_context: summary_context(&fixture.configuration),
                },
                attempt: 1,
                request: Some(SummaryEffectRequest {
                    index: 0,
                    usage_id: "usage".to_owned(),
                }),
                usage_ids: Vec::new(),
            }),
            intent: OperationIntent::Navigation {
                target_id: Some("target".to_owned()),
                summarize: true,
                label: None,
                custom_instructions: None,
            },
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: vec!["navigation_end"],
        },
        InstalledOperation {
            state: OperationState::SummaryRetryWait(SummaryRetryWaitOperation {
                scope: cancelled_scope(),
                generation: SummaryGenerationScope {
                    task: run_task,
                    summary_context: summary_context(&fixture.configuration),
                },
                retry_wait: RetryWait {
                    next_attempt: 2,
                    not_before: MATRIX_NOT_BEFORE,
                    error_message: "retry".to_owned(),
                },
            }),
            intent: run_intent(),
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: vec!["compaction_end", "run_end"],
        },
        InstalledOperation {
            state: OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
                scope: cancelled_scope(),
                target_id: Some("target".to_owned()),
                label: None,
            }),
            intent: OperationIntent::Navigation {
                target_id: Some("target".to_owned()),
                summarize: false,
                label: None,
                custom_instructions: None,
            },
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: vec!["navigation_end"],
        },
    ]
}

/// Installs one matrix row, upstream's `installOperation(fixture, installed)`
/// call sites: the reconciliation installs preserve the projection's
/// `lastOperationId` and inbox.
async fn install(fixture: &DriveFixture, installed: &InstalledOperation) {
    install_operation(
        fixture,
        installed.state.clone(),
        installed.intent.clone(),
        InstallOptions {
            entries: installed.entries.clone(),
            extra_writes: installed.writes.clone(),
            keep_last_operation_id: true,
            ..InstallOptions::default()
        },
    )
    .await;
}

/// Registers one call-counting `before_drive` observer, upstream's
/// `const beforeDrive = vi.fn(); fixture.hooks.on("before_drive",
/// beforeDrive)`.
fn count_before_drive(fixture: &DriveFixture) -> Arc<AtomicUsize> {
    let calls = Arc::new(AtomicUsize::new(0));
    fixture
        .hooks
        .on(
            HookName::BeforeDrive,
            Arc::new({
                let calls = Arc::clone(&calls);
                move |_event: &HookInvocation, _context: &Context| {
                    let calls = Arc::clone(&calls);
                    Box::pin(async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        Ok(HookResult::BeforeDrive)
                    })
                }
            }),
            HookOptions::default(),
        )
        .expect("register");
    calls
}

/// Upstream "drives an ordinary run through all direct procedures with one
/// `before_drive` hook".
#[tokio::test]
async fn drives_an_ordinary_run_through_all_direct_procedures_with_one_before_drive_hook() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    install(
        &fixture,
        &InstalledOperation {
            state: OperationState::Starting(StartingOperation {
                scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
            }),
            intent: run_intent(),
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
    )
    .await;
    let before_drive = count_before_drive(&fixture);
    fixture.faux.set_responses([faux_assistant_message(
        "answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let outcome = common::settled_pass(
        &fixture.lane,
        &fixture.drive,
        "the pass serves",
        "the pass settles",
    )
    .await;
    assert_eq!(outcome.operation_id, OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    assert_eq!(before_drive.load(Ordering::SeqCst), 1, "one before_drive");
    assert_eq!(fixture.faux.state().call_count(), 1, "one provider call");
    assert!(
        fixture.lane.state().operation.is_none(),
        "the operation cleared"
    );
    common::close_session(&fixture).await;
}

/// Upstream "leaves durable state unchanged when `before_drive` fails closed".
#[tokio::test]
async fn leaves_durable_state_unchanged_when_before_drive_fails_closed() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let starting = OperationState::Starting(StartingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
    });
    install(
        &fixture,
        &InstalledOperation {
            state: starting.clone(),
            intent: run_intent(),
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
    )
    .await;
    fixture
        .hooks
        .on(
            HookName::BeforeDrive,
            Arc::new(|_event: &HookInvocation, _context: &Context| {
                let failure: HookFailure =
                    Box::new(SessionError::Message("blocked drive".to_owned()));
                Box::pin(async move { Err(failure) })
            }),
            HookOptions::default(),
        )
        .expect("register");

    let error = drive_operation(&fixture.lane, &fixture.drive)
        .await
        .expect_err("the pass faults");
    assert!(
        error.to_string().contains("before_drive handler failed"),
        "the fail-closed rejection carried: {error}"
    );
    assert_eq!(
        fixture.lane.state().operation.expect("the operation").state,
        starting,
        "the starting leaf stayed durable",
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "no commit published"
    );
    common::close_session(&fixture).await;
}

/// Upstream "reconciles every durable leaf without ordinary hook admission":
/// the 13-leaf matrix, one fixture per row, upstream's index loop.
#[expect(
    clippy::too_many_lines,
    reason = "the matrix drives thirteen leaves' full settle-and-cleanup contract in one table loop"
)]
#[tokio::test]
async fn reconciles_every_durable_leaf_without_ordinary_hook_admission() {
    for index in 0..13 {
        let fixture = create_drive_fixture(reconcile_spec()).await;
        let installed = cases(&fixture)
            .into_iter()
            .nth(index)
            .expect("the 13-leaf table carries the row");
        install(&fixture, &installed).await;
        let before_drive = count_before_drive(&fixture);

        let outcome = common::settled_pass(
            &fixture.lane,
            &fixture.drive,
            "the pass serves",
            "the pass settles",
        )
        .await;
        assert_eq!(outcome.operation_id, OPERATION_ID);
        assert_eq!(
            outcome.status,
            TerminalStatus::Aborted,
            "leaf {index} aborts"
        );
        assert_eq!(
            before_drive.load(Ordering::SeqCst),
            0,
            "leaf {index} admits no hook"
        );
        assert!(
            fixture.lane.state().operation.is_none(),
            "leaf {index} cleared the operation"
        );
        let record = fixture
            .lane
            .get_result_impl(OPERATION_ID, &background_context())
            .await
            .expect("the result read")
            .unwrap_or_else(|| unreachable!("leaf {index} recorded its result"));
        assert_eq!(record.status, TerminalStatus::Aborted);
        assert!(
            fixture
                .session
                .get_value(&operation_meta(OPERATION_ID).address, &background_context())
                .await
                .expect("the meta read")
                .is_none(),
            "leaf {index} cleared the meta",
        );
        assert!(
            fixture
                .session
                .get_value(
                    &operation_state(OPERATION_ID).address,
                    &background_context()
                )
                .await
                .expect("the state read")
                .is_none(),
            "leaf {index} cleared the state",
        );
        for (family, scanned) in [
            (
                "tool args",
                fixture
                    .session
                    .scan_values(
                        &operation_tool_args_prefix(OPERATION_ID, None).address,
                        &background_context(),
                    )
                    .await
                    .expect("the args scan"),
            ),
            (
                "tool memos",
                fixture
                    .session
                    .scan_values(
                        &operation_tool_memo_prefix(OPERATION_ID, None).address,
                        &background_context(),
                    )
                    .await
                    .expect("the memo scan"),
            ),
            (
                "preparations",
                fixture
                    .session
                    .scan_values(
                        &operation_preparation_prefix(OPERATION_ID).address,
                        &background_context(),
                    )
                    .await
                    .expect("the preparation scan"),
            ),
        ] {
            assert!(scanned.is_empty(), "leaf {index} cleared its {family}");
        }
        if matches!(
            installed.state,
            OperationState::DeferredSuspended(_) | OperationState::DeferredEffectPending(_)
        ) {
            assert_eq!(
                fixture.faux.state().cancelled_deferred().len(),
                1,
                "leaf {index} cancelled its deferred handle once",
            );
        }
        let events = lock(&fixture.events).clone();
        let seen: Vec<&str> = events
            .iter()
            .filter(|event| {
                installed
                    .terminal_events
                    .contains(&event.event_type().as_str())
            })
            .map(|event| event.event_type().as_str())
            .collect();
        let tail_start = seen.len().saturating_sub(installed.terminal_events.len());
        assert_eq!(
            &seen[tail_start..],
            installed.terminal_events.as_slice(),
            "leaf {index} settled through its terminal events",
        );
        common::close_session(&fixture).await;
    }
}

/// Upstream "durably drains abortable input once and preserves lane-owned
/// input through terminal cleanup".
#[expect(
    clippy::too_many_lines,
    reason = "the case drives the abort request's three calls and the settle in one choreography"
)]
#[tokio::test]
async fn durably_drains_abortable_input_once_and_preserves_lane_owned_input_through_terminal_cleanup()
 {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    install(
        &fixture,
        &row(
            OperationState::Checkpoint(CheckpointOperation {
                scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
                checkpoint: CheckpointData {
                    continuation: Continuation::NeedAssistant {
                        overflow_recovery_used: false,
                    },
                    trigger_entry_id: "tip".to_owned(),
                },
            }),
            vec!["run_end"],
        ),
    )
    .await;
    let queued: Vec<(&str, InboxItemKind, AgentMessage)> = vec![
        ("steer", InboxItemKind::Steer, user("steer", 1)),
        ("write", InboxItemKind::Write, user("write", 1)),
        ("follow", InboxItemKind::FollowUp, user("follow", 1)),
        ("next", InboxItemKind::NextRun, user("next", 1)),
    ];
    let queued_writes = queued.clone();
    fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let queued = queued_writes.clone();
                Box::pin(async move {
                    let mut writes: Vec<Write> = queued
                        .iter()
                        .map(|(entry_id, _kind, message)| {
                            set_value_write(
                                &pending_entry(entry_id),
                                PendingEntry::Message {
                                    payload: Box::new(message.clone()),
                                },
                            )
                            .expect("the pending write")
                        })
                        .collect();
                    let inbox: Vec<InboxItem> = queued
                        .iter()
                        .map(|(entry_id, kind, _)| InboxItem {
                            entry_id: (*entry_id).to_owned(),
                            kind: *kind,
                        })
                        .collect();
                    writes.push(
                        lane_state_write("main", Some(OPERATION_ID), None, inbox.clone())
                            .expect("the lane state write"),
                    );
                    Ok(LaneCommand::Commit {
                        writes,
                        next: LaneState { inbox, ..state },
                        materialize: Arc::new(|_: &CommitResult| ()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the queue commits");
    fixture.storage.clear_commit_attempts();

    let requested = fixture
        .lane
        .request_operation_abort(OPERATION_ID, &background_context())
        .await
        .expect("the request serves")
        .expect("the abort serves");
    assert_eq!(
        requested,
        AbortRequestOutcome {
            operation_id: OPERATION_ID.to_owned(),
            newly_requested: true,
            steer: vec![user("steer", 1)],
            follow_up: vec![user("follow", 1)],
        },
    );
    assert!(
        fixture.drive.gate.signal().aborted(),
        "the gate signal aborted"
    );
    assert!(
        !fixture.drive.close_signal.aborted(),
        "the close signal stayed live",
    );
    assert_eq!(
        fixture.lane.state().inbox,
        vec![
            InboxItem {
                entry_id: "write".to_owned(),
                kind: InboxItemKind::Write,
            },
            InboxItem {
                entry_id: "next".to_owned(),
                kind: InboxItemKind::NextRun,
            },
        ],
        "the abortable input drained",
    );
    for entry_id in ["steer", "follow"] {
        assert!(
            fixture
                .session
                .get_value(&pending_entry(entry_id).address, &background_context())
                .await
                .expect("the pending read")
                .is_none(),
            "the {entry_id} pending entry deleted",
        );
    }
    let aborts: Vec<HarnessEventPayload> = lock(&fixture.events)
        .iter()
        .filter(|event| event.event_type() == HarnessEventType::OperationAbort)
        .map(|event| event.payload.clone())
        .collect();
    assert_eq!(
        aborts,
        vec![HarnessEventPayload::OperationAbort {
            operation_id: OPERATION_ID.to_owned(),
            steer: vec![user("steer", 1)],
            follow_up: vec![user("follow", 1)],
        }],
    );
    let commit_count = fixture.storage.get_commit_attempts().len();
    let event_count = lock(&fixture.events).len();
    let requested_again = fixture
        .lane
        .request_operation_abort(OPERATION_ID, &background_context())
        .await
        .expect("the request serves")
        .expect("the abort serves");
    assert_eq!(
        requested_again,
        AbortRequestOutcome {
            operation_id: OPERATION_ID.to_owned(),
            newly_requested: false,
            steer: Vec::new(),
            follow_up: Vec::new(),
        },
    );
    assert_eq!(
        fixture.storage.get_commit_attempts().len(),
        commit_count,
        "the repeat request wrote nothing",
    );
    assert_eq!(lock(&fixture.events).len(), event_count, "no repeat events");

    let outcome = common::settled_pass(
        &fixture.lane,
        &fixture.drive,
        "the pass serves",
        "the pass settles",
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Aborted);
    assert_eq!(
        fixture.lane.state().inbox,
        vec![
            InboxItem {
                entry_id: "write".to_owned(),
                kind: InboxItemKind::Write,
            },
            InboxItem {
                entry_id: "next".to_owned(),
                kind: InboxItemKind::NextRun,
            },
        ],
        "the lane-owned input survived",
    );
    for entry_id in ["write", "next"] {
        assert!(
            fixture
                .session
                .get_value(&pending_entry(entry_id).address, &background_context())
                .await
                .expect("the pending read")
                .is_some(),
            "the {entry_id} pending entry survived",
        );
    }
    fixture.storage.clear_commit_attempts();
    let stale = fixture
        .lane
        .request_operation_abort(OPERATION_ID, &background_context())
        .await
        .expect("the request serves");
    let Err(error) = stale else {
        unreachable!("the stale abort rejects")
    };
    assert!(
        matches!(error, HarnessError::OperationMismatch { .. }),
        "the stale abort reports the mismatch: {error}"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the stale abort wrote nothing",
    );
    common::close_session(&fixture).await;
}

/// Upstream "marks cancellation without installing a Drive".
#[tokio::test]
async fn marks_cancellation_without_installing_a_drive() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    install(
        &fixture,
        &row(
            OperationState::Starting(StartingOperation {
                scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
            }),
            vec!["run_end"],
        ),
    )
    .await;
    fixture.lane.set_active_drive(None);

    let requested = fixture
        .lane
        .request_operation_abort(OPERATION_ID, &background_context())
        .await
        .expect("the request serves")
        .expect("the abort serves");
    assert!(requested.newly_requested, "the abort newly requested");
    assert!(fixture.lane.active_drive().is_none(), "no drive installed");
    let state = fixture.lane.state().operation.expect("the operation").state;
    let OperationState::Starting(leaf) = state else {
        unreachable!("the starting leaf stayed durable")
    };
    assert!(
        matches!(leaf.scope.control, Control::CancelRequested { .. }),
        "the control flipped: {:?}",
        leaf.scope.control,
    );
    common::close_session(&fixture).await;
}

/// Upstream "cancels an admitted retry timer only after the abort marker
/// commits".
#[tokio::test(start_paused = true)]
async fn cancels_an_admitted_retry_timer_only_after_the_abort_marker_commits() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let not_before = now_ms() + 1_000;
    install(
        &fixture,
        &row(
            OperationState::AssistantRetryWait(AssistantRetryWaitOperation {
                scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
                generation_context: generation_context(&fixture.configuration),
                retry_wait: RetryWait {
                    next_attempt: 2,
                    not_before,
                    error_message: "retry".to_owned(),
                },
            }),
            vec!["run_end"],
        ),
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
    fixture.lane.set_active_drive(Some(Arc::clone(&drive)));
    let before_drive = count_before_drive(&fixture);
    let pass_lane = Arc::clone(&fixture.lane);
    let pass_drive = Arc::clone(&drive);
    let pass = tokio::spawn(async move { drive_operation(&pass_lane, &pass_drive).await });

    // The timer-installation proof, upstream's `vi.waitFor(() =>
    // expect(setTimeoutSpy).toHaveBeenCalled())`: the pass parks on its
    // wait, observed by the drive hook having run. The yields never move
    // tokio's paused clock.
    let mut parked = false;
    for _ in 0..200 {
        if before_drive.load(Ordering::SeqCst) == 1 {
            parked = true;
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(parked, "the pass never reached its wait");
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    tokio::time::advance(Duration::from_millis(999)).await;
    assert!(
        !lock(&fixture.events)
            .iter()
            .any(|event| event.event_type() == HarnessEventType::RetryStart),
        "the wait stayed parked before the deadline",
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the wait committed nothing before the deadline",
    );

    let requested = fixture
        .lane
        .request_operation_abort(OPERATION_ID, &background_context())
        .await
        .expect("the request serves")
        .expect("the abort serves");
    assert!(requested.newly_requested, "the abort newly requested");

    let outcome = pass
        .await
        .expect("the pass joins")
        .expect("the pass serves");
    let DriveOutcome::Settled { outcome } = outcome else {
        unreachable!("the pass settles: {outcome:?}")
    };
    assert_eq!(outcome.status, TerminalStatus::Aborted);
    assert!(
        !lock(&fixture.events)
            .iter()
            .any(|event| event.event_type() == HarnessEventType::RetryStart),
        "the timer cancelled without a retry start",
    );
    common::close_session(&fixture).await;
}

/// Upstream "drops a stale structural hook result when cancellation commits
/// first".
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single case and its stale-hook choreography"
)]
#[tokio::test]
async fn drops_a_stale_structural_hook_result_when_cancellation_commits_first() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let task = SummaryTask {
        task_id: "task".to_owned(),
        reason: Some(CompactionReason::Manual),
        custom_instructions: None,
        boundary: ResultBoundary::Finish,
    };
    let deciding = SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: task.clone(),
    };
    install(
        &fixture,
        &InstalledOperation {
            state: OperationState::SummaryDeciding(deciding.clone()),
            intent: OperationIntent::Compaction {
                custom_instructions: None,
            },
            entries: vec![history_entry()],
            writes: vec![
                set_value_write(
                    &operation_preparation(OPERATION_ID, "task"),
                    DurableStructuralPreparation::Compaction {
                        messages_to_summarize: vec![user("history", 1)],
                        turn_prefix_messages: Vec::new(),
                        retained_tail: Vec::new(),
                        is_split_turn: false,
                        tokens_before: 100,
                        previous_summary: None,
                        file_ops: create_file_ops(),
                        settings: DEFAULT_COMPACTION_SETTINGS,
                    },
                )
                .expect("the preparation write"),
            ],
            terminal_events: vec!["compaction_end"],
        },
    )
    .await;
    let (started_tx, started_rx) = deferred();
    let (release_tx, release_rx) = deferred();
    let started_slot = Arc::new(Mutex::new(Some(started_tx)));
    let release_slot = Arc::new(Mutex::new(Some(release_rx)));
    fixture
        .hooks
        .on(
            HookName::BeforeCompaction,
            Arc::new(move |_event: &HookInvocation, _context: &Context| {
                let started_tx = lock(&started_slot).take();
                let release_rx = lock(&release_slot).take();
                Box::pin(async move {
                    if let Some(started_tx) = started_tx {
                        let _ = started_tx.send(());
                    }
                    if let Some(release_rx) = release_rx {
                        let _ = release_rx.await;
                    }
                    Ok(HookResult::BeforeCompaction(Some(CompactionHookResult {
                        decline: None,
                        compaction: Some(CompactResult {
                            summary: "stale".to_owned(),
                            tokens_before: 100,
                            usage: None,
                            retained_tail: Vec::new(),
                            details: None,
                        }),
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register");
    let pass_lane = Arc::clone(&fixture.lane);
    let pass_drive = Arc::clone(&fixture.drive);
    let pass_deciding = deciding.clone();
    let pass = tokio::spawn(async move {
        run_structural_decision(&pass_lane, &pass_drive, &pass_deciding).await
    });

    started_rx.await.expect("the hook started");
    fixture
        .lane
        .request_operation_abort(OPERATION_ID, &background_context())
        .await
        .expect("the request serves")
        .expect("the abort serves");
    release_tx.send(()).expect("the hook released");

    let running = pass
        .await
        .expect("the decision joins")
        .expect("the decision serves");
    assert!(
        matches!(running, ProcedureResult::Continue),
        "the stale hook result dropped: {running:?}"
    );
    let compactions = fixture
        .session
        .find_entries(
            Some(&EntryQuery {
                kind: Some(EntryType::Compaction),
                ..EntryQuery::default()
            }),
            &background_context(),
        )
        .await
        .expect("the compaction scan");
    assert!(
        compactions.is_empty(),
        "no compaction entry published: {compactions:?}"
    );

    let outcome = common::settled_pass(
        &fixture.lane,
        &fixture.drive,
        "the pass serves",
        "the pass settles",
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Aborted);
    let events = lock(&fixture.events).clone();
    let HarnessEventPayload::CompactionEnd { status, .. } =
        &events.last().expect("the final event").payload
    else {
        unreachable!(
            "the final event is the compaction end: {:?}",
            events.last().expect("the final event").payload
        )
    };
    assert_eq!(*status, CompactionEndStatus::Aborted);
    common::close_session(&fixture).await;
}

/// Upstream "keeps the deferred cleanup signal separate from operation
/// abort".
#[tokio::test]
async fn keeps_the_deferred_cleanup_signal_separate_from_operation_abort() {
    let drive = Drive::new(
        &DriveOptions {
            operation_id: OPERATION_ID.to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    );
    // The settled cancellation, upstream's `beginAbort(Promise.resolve())`:
    // dropping the sender resolves the receiver immediately.
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);
    drive.signal_abort();
    assert!(drive.gate.signal().aborted(), "the gate signal aborted");
    assert!(
        !drive.close_signal.aborted(),
        "the close signal stayed live",
    );
    let closed =
        crate::harness::runtime::types::lane_error(SessionError::Message("closed".to_owned()));
    drive.close_gate(Arc::clone(&closed));
    assert!(drive.close_signal.aborted(), "the close signal aborted");
    assert_eq!(
        drive.close_signal.reason(),
        Some(pi_chord::context::AbortReason::Caller("closed".to_owned())),
        "the close carried its reason",
    );
}

/// Upstream "can crash between cancelled response settlement and terminal
/// cleanup".
#[tokio::test]
async fn can_crash_between_cancelled_response_settlement_and_terminal_cleanup() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let installed = cases(&fixture)
        .into_iter()
        .nth(3)
        .expect("the effect-pending row");
    install(&fixture, &installed).await;

    let continued = reconcile_operation(&fixture.lane, &fixture.drive)
        .await
        .expect("the reconcile serves");
    assert!(
        matches!(continued, ProcedureResult::Continue),
        "the reconcile continues: {continued:?}"
    );
    assert_eq!(
        fixture
            .lane
            .state()
            .operation
            .expect("the operation")
            .state
            .at(),
        "checkpoint",
        "the settle landed its checkpoint",
    );
    let entry = fixture
        .session
        .get_entry("assistant-response", &background_context())
        .await
        .expect("the entry read")
        .expect("the response entry");
    let Entry::Message { body, .. } = entry else {
        unreachable!("the response entry is a message: {entry:?}")
    };
    let AgentMessage::Standard(Message::Assistant(message)) = body.message else {
        unreachable!("the response entry carries the assistant message")
    };
    assert_eq!(
        message.stop_reason,
        StopReason::Aborted,
        "the response settled aborted"
    );
    assert!(
        fixture
            .lane
            .get_result_impl(OPERATION_ID, &background_context())
            .await
            .expect("the result read")
            .is_none(),
        "no result record published",
    );
    let frames = fixture
        .session
        .read_list(
            &pending_assistant_frames(OPERATION_ID, "assistant-response").address,
            None,
            &background_context(),
        )
        .await
        .expect("the frame read");
    assert!(frames.is_empty(), "the frames deleted");

    let resumed = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: OPERATION_ID.to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    fixture.lane.set_active_drive(Some(Arc::clone(&resumed)));
    let outcome = common::settled_pass(
        &fixture.lane,
        &resumed,
        "the pass serves",
        "the pass settles",
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Aborted);
    common::close_session(&fixture).await;
}

/// The provider streams whose deferred cancellation always fails, upstream's
/// `{ ...fixture.faux.provider, cancelDeferred: async () => { throw ... } }`
/// spread: the faux core streams every method, the cancel rejects with the
/// remote failure's message.
struct FailingCancelStreams {
    inner: FauxCore,
}

impl ProviderStreams for FailingCancelStreams {
    provider_streams_pass_through!(inner);

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: Option<&'a DeferredCancelOptions>,
    ) -> BoxedFuture<'a, Result<(), ProviderRequestError>> {
        Box::pin(async {
            Err(ProviderRequestError::new(
                None,
                None,
                "remote cancellation failed",
            ))
        })
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

/// Upstream "ignores deferred-provider cancellation failure".
#[tokio::test]
async fn ignores_deferred_provider_cancellation_failure() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let deferred_seeds = deferred_operation(&fixture, false);
    let provider = create_provider(CreateProviderOptions {
        id: Provider::id(&fixture.faux.provider).to_owned(),
        name: Some(Provider::name(&fixture.faux.provider).to_owned()),
        base_url: None,
        headers: None,
        auth: Provider::auth(&fixture.faux.provider).clone(),
        models: fixture.faux.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: ProviderApi::Single(Arc::new(FailingCancelStreams {
            inner: fixture.faux.core().clone(),
        })),
    });
    fixture.models.set_provider(Arc::new(provider));
    install(
        &fixture,
        &InstalledOperation {
            state: deferred_seeds.state,
            intent: OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            entries: vec![deferred_seeds.entry],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
    )
    .await;

    let outcome = common::settled_pass(
        &fixture.lane,
        &fixture.drive,
        "the pass serves",
        "the pass settles",
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Aborted);
    common::close_session(&fixture).await;
}

/// Upstream `reconcileOperation` invariant: a pass whose operation is
/// absent or foreign faults the reconcile.
#[tokio::test]
async fn reconcile_requires_a_matching_operation() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let idle = reconcile_operation(&fixture.lane, &fixture.drive)
        .await
        .expect_err("the idle lane faults");
    assert!(
        idle.to_string().contains(&format!(
            "Drive {OPERATION_ID} has no matching operation to reconcile"
        )),
        "the idle invariant carried: {idle}"
    );

    let installed = cases(&fixture).into_iter().next().expect("the first row");
    install(&fixture, &installed).await;
    let foreign = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "foreign-pass".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    let error = reconcile_operation(&fixture.lane, &foreign)
        .await
        .expect_err("the foreign pass faults");
    assert!(
        error
            .to_string()
            .contains("Drive foreign-pass has no matching operation to reconcile"),
        "the foreign invariant carried: {error}"
    );
    common::close_session(&fixture).await;
}

/// Upstream `reconcileOperation` invariant: a leaf whose durable control is
/// not `cancel_requested` faults the reconcile.
#[tokio::test]
async fn reconcile_requires_cancelled_durable_control() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    install(
        &fixture,
        &InstalledOperation {
            state: OperationState::AssistantReady(AssistantReadyOperation {
                scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
                generation_context: generation_context(&fixture.configuration),
                next_attempt: 1,
            }),
            intent: run_intent(),
            entries: vec![history_entry()],
            writes: Vec::new(),
            terminal_events: Vec::new(),
        },
    )
    .await;

    let error = reconcile_operation(&fixture.lane, &fixture.drive)
        .await
        .expect_err("the running leaf faults");
    assert!(
        error
            .to_string()
            .contains(&format!("Operation {OPERATION_ID} is not cancelled")),
        "the control invariant carried: {error}"
    );
    common::close_session(&fixture).await;
}

/// The run-intent summary leaves the upstream matrix never carries: their
/// reconciliation publishes the aborted `compaction_end` ahead of the
/// aborted `run_end`, upstream's `publishAbortedTerminal` run-summary arm.
#[tokio::test]
async fn run_intent_summary_leaves_publish_their_aborted_compaction_end() {
    for leaf in ["summary.ready", "summary.effect_pending"] {
        let fixture = create_drive_fixture(reconcile_spec()).await;
        let resume_task = SummaryTask {
            task_id: "run-summary".to_owned(),
            reason: Some(CompactionReason::Threshold),
            custom_instructions: None,
            boundary: ResultBoundary::ResumeCheckpoint {
                resume_after: CheckpointData {
                    continuation: Continuation::NeedAssistant {
                        overflow_recovery_used: false,
                    },
                    trigger_entry_id: "tip".to_owned(),
                },
            },
        };
        let state = match leaf {
            "summary.ready" => OperationState::SummaryReady(SummaryReadyOperation {
                scope: cancelled_scope(),
                generation: SummaryGenerationScope {
                    task: resume_task,
                    summary_context: summary_context(&fixture.configuration),
                },
                next_attempt: 1,
            }),
            _ => OperationState::SummaryEffectPending(SummaryEffectPendingOperation {
                scope: cancelled_scope(),
                generation: SummaryGenerationScope {
                    task: resume_task,
                    summary_context: summary_context(&fixture.configuration),
                },
                attempt: 1,
                request: Some(SummaryEffectRequest {
                    index: 0,
                    usage_id: "usage".to_owned(),
                }),
                usage_ids: Vec::new(),
            }),
        };
        install(&fixture, &row(state, vec!["compaction_end", "run_end"])).await;

        let outcome = reconcile_operation(&fixture.lane, &fixture.drive)
            .await
            .expect("the reconcile serves");
        let ProcedureResult::Settled { outcome } = outcome else {
            unreachable!("the reconcile settles: {outcome:?}")
        };
        assert_eq!(outcome.status, TerminalStatus::Aborted, "{leaf} aborts");
        let seen: Vec<&str> = lock(&fixture.events)
            .iter()
            .filter(|event| matches!(event.event_type().as_str(), "compaction_end" | "run_end"))
            .map(|event| event.event_type().as_str())
            .collect();
        assert_eq!(
            seen,
            ["compaction_end", "run_end"],
            "{leaf} settles through its aborted pair"
        );
        common::close_session(&fixture).await;
    }
}

/// Upstream `publishAbortedTerminal` invariant: a run-intent summary leaf
/// whose task boundary is not a reasoned `resume_checkpoint` faults the
/// reconcile.
#[tokio::test]
async fn cancelled_run_summary_requires_a_reasoned_resume_checkpoint_boundary() {
    for (label, reason) in [
        ("finish", Some(CompactionReason::Manual)),
        ("reasonless", None),
    ] {
        let fixture = create_drive_fixture(reconcile_spec()).await;
        let task = SummaryTask {
            task_id: "run-summary".to_owned(),
            reason,
            custom_instructions: None,
            boundary: match label {
                "finish" => ResultBoundary::Finish,
                _ => ResultBoundary::ResumeCheckpoint {
                    resume_after: CheckpointData {
                        continuation: Continuation::NeedAssistant {
                            overflow_recovery_used: false,
                        },
                        trigger_entry_id: "tip".to_owned(),
                    },
                },
            },
        };
        install(
            &fixture,
            &InstalledOperation {
                state: OperationState::SummaryReady(SummaryReadyOperation {
                    scope: cancelled_scope(),
                    generation: SummaryGenerationScope {
                        task,
                        summary_context: summary_context(&fixture.configuration),
                    },
                    next_attempt: 1,
                }),
                intent: run_intent(),
                entries: vec![history_entry()],
                writes: Vec::new(),
                terminal_events: Vec::new(),
            },
        )
        .await;

        let error = reconcile_operation(&fixture.lane, &fixture.drive)
            .await
            .expect_err("the boundary faults");
        assert!(
            error
                .to_string()
                .contains("Cancelled run summary has an invalid result boundary"),
            "the {label} boundary invariant carried: {error}"
        );
        common::close_session(&fixture).await;
    }
}

/// Upstream `cancelDeferredBestEffort` arm: a leaf whose captured model no
/// longer resolves in the process skips the remote cancellation and still
/// reconciles its aborted terminal.
#[tokio::test]
async fn missing_model_skips_the_remote_cancellation() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let seeds = deferred_operation(&fixture, false);
    install(
        &fixture,
        &InstalledOperation {
            state: seeds.state,
            intent: OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            entries: vec![seeds.entry],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
    )
    .await;
    fixture
        .models
        .delete_provider(Provider::id(&fixture.faux.provider));

    let outcome = reconcile_operation(&fixture.lane, &fixture.drive)
        .await
        .expect("the reconcile serves");
    let ProcedureResult::Settled { outcome } = outcome else {
        unreachable!("the reconcile settles: {outcome:?}")
    };
    assert_eq!(outcome.status, TerminalStatus::Aborted, "the run aborts");
    assert!(
        fixture.faux.state().cancelled_deferred().is_empty(),
        "the missing model skips the remote cancellation"
    );
    common::close_session(&fixture).await;
}

/// The provider streams whose deferred cancellation parks on the request's
/// cancellation token, upstream's `signal: drive.closeSignal` handoff: the
/// faux core streams every method; `cancel_deferred` captures the token and
/// the carried headers, then resolves only once the token cancels.
/// The captured cancellation options, upstream's captured `fetchOptions`:
/// the close-linked token and the carried headers.
type CapturedCancelOptions = Option<(Option<CancellationToken>, Option<ProviderHeaders>)>;

struct CapturingCancelStreams {
    inner: FauxCore,
    captured: Arc<Mutex<CapturedCancelOptions>>,
}

impl ProviderStreams for CapturingCancelStreams {
    provider_streams_pass_through!(inner);

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        options: Option<&'a DeferredCancelOptions>,
    ) -> BoxedFuture<'a, Result<(), ProviderRequestError>> {
        let captured = Arc::clone(&self.captured);
        let options = options.cloned().unwrap_or_default();
        Box::pin(async move {
            let token = options.transport_options.signal.clone();
            *lock(&captured) = Some((token.clone(), options.headers.clone()));
            if let Some(token) = token {
                token.cancelled().await;
            }
            Ok(())
        })
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

/// Upstream `cancelDeferredBestEffort` binding: the provider request's
/// cancellation token links to the pass's close signal — the reconciliation
/// parks on the remote cancellation until the close signal aborts, and the
/// captured options carry the leaf's headers, upstream's
/// `signal: drive.closeSignal` and `headers: deferred.streamOptions.headers`.
#[tokio::test]
async fn the_close_signal_cancels_the_parked_deferred_cancellation() {
    let fixture = create_drive_fixture(reconcile_spec()).await;
    let captured: Arc<Mutex<CapturedCancelOptions>> = Arc::new(Mutex::new(None));
    let provider = create_provider(CreateProviderOptions {
        id: Provider::id(&fixture.faux.provider).to_owned(),
        name: Some(Provider::name(&fixture.faux.provider).to_owned()),
        base_url: None,
        headers: None,
        auth: Provider::auth(&fixture.faux.provider).clone(),
        models: fixture.faux.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: ProviderApi::Single(Arc::new(CapturingCancelStreams {
            inner: fixture.faux.core().clone(),
            captured: Arc::clone(&captured),
        })),
    });
    fixture.models.set_provider(Arc::new(provider));
    let seeds = deferred_operation(&fixture, false);
    let OperationState::DeferredSuspended(mut leaf) = seeds.state else {
        unreachable!("the suspended seeds carry the suspended leaf")
    };
    leaf.deferred.stream_options.headers =
        Some(BTreeMap::from([("x-a".to_owned(), "b".to_owned())]));
    install(
        &fixture,
        &InstalledOperation {
            state: OperationState::DeferredSuspended(leaf),
            intent: OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            entries: vec![seeds.entry],
            writes: Vec::new(),
            terminal_events: vec!["run_end"],
        },
    )
    .await;

    let pass_lane = Arc::clone(&fixture.lane);
    let pass_drive = Arc::clone(&fixture.drive);
    let pass = tokio::spawn(async move { reconcile_operation(&pass_lane, &pass_drive).await });
    common::wait_for(|| lock(&captured).is_some()).await;
    let mut parked = true;
    for _ in 0..20 {
        if pass.is_finished() {
            parked = false;
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(parked, "the reconcile parks on the remote cancellation");

    let closed =
        crate::harness::runtime::types::lane_error(SessionError::Message("closed".to_owned()));
    fixture.drive.close_gate(Arc::clone(&closed));
    let outcome = pass
        .await
        .expect("the reconcile joins")
        .expect("the reconcile serves");
    let ProcedureResult::Settled { outcome } = outcome else {
        unreachable!("the reconcile settles: {outcome:?}")
    };
    assert_eq!(outcome.status, TerminalStatus::Aborted, "the run aborts");
    let (token, headers) = lock(&captured)
        .take()
        .expect("the cancellation captured its options");
    assert!(
        token.expect("the token").is_cancelled(),
        "the close signal cancelled the request token"
    );
    let headers = headers.expect("the options carry the headers");
    assert_eq!(
        headers.get("x-a").map(Option::as_deref),
        Some(Some("b")),
        "the captured headers carry the leaf's headers"
    );
    common::close_session(&fixture).await;
}
