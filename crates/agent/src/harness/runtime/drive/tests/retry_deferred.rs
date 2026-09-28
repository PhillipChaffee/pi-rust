//! The assistant retry wait and deferred polling suite, ported 1:1 from
//! upstream `test/harness/runtime/drive-retry-deferred.test.ts` ("runtime
//! assistant retry wait" / "runtime deferred polling") at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's fake-timer cases pin `Date.now` at absolute epochs
//!   (`setSystemTime(1_000)` against `notBefore: 1_100`); the ported
//!   retry wait derives its remaining sleep from the real epoch clock
//!   ([`now_ms`]), so the absolute epochs restate as wall-clock-relative
//!   deadlines — the no-timer case seeds a future deadline, the due case
//!   seeds a due one, and the timer case proves no commit before the
//!   deadline and the commit at it (upstream's 99/1 advance split).
//! - the `setTimeout` spy on the disabled-waiting case restates as the
//!   returned durable waiting outcome with zero commit attempts: the
//!   disabled arm returns before any sleep runs.
//! - the wrapped provider's `fetchOptions.signal === drive.gate.signal`
//!   identity assert restates to presence plus a post-run linkage proof:
//!   the port hands the transport a cancellation token linked to the pass's
//!   gate signal (the abort signal cannot cross the token-typed transport
//!   field), so the case aborts the gate after the observations and reads
//!   the captured token cancelling.
//! - `expect(driveOptions).toEqual({ operationId, pollDeferred: true })`
//!   guards upstream's captured-options object alias; the port copies the
//!   options into the pass at construction, so nothing can observe a
//!   mutation and the assert has no restatement.
//! - the discarded-intent rejection: upstream drops the parked commit with
//!   `"commit discarded"`; the ported gate reports the discarded commit as
//!   `"commit rejected: storage discarded"`, so the pin is the shared
//!   `"storage discarded"` text.
//! - the durable-backend read after the discard rides the session's reader
//!   surface — the same stored record the backend holds, reads never gate.
//! - `afterEach` timer restore and session close restate as a close at
//!   each test's end; the fixtures carry no timers to restore.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the fixture guards raise with upstream's throw texts; the tests assert by panicking"
)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::auth::types::{Credential, ProviderAuth};
use pi_ai::models::CreateProviderOptions;
use pi_ai::models::Provider;
use pi_ai::models::ProviderApi;
use pi_ai::models::ProviderModelError;
use pi_ai::models::{ProviderError, ProviderImpl, RefreshModelsContext, create_provider};
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::FauxCore;
use pi_ai::providers::faux::FauxDeferredOptions;
use pi_ai::providers::faux::FauxResponseStep;
use pi_ai::providers::faux::FauxTokenSize;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_tool_call};
use pi_ai::types::Context as AiContext;
use pi_ai::types::DeferredCancelOptions;
use pi_ai::types::DeferredFetchOptions;
use pi_ai::types::DeferredHandle;
use pi_ai::types::DeferredRequest;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::ProviderHeaders;
use pi_ai::types::{AssistantMessage, AssistantMessageEvent, BoxedFuture};
use pi_ai::types::{ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions};
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_ai::utils::retry::RetryPolicy;
use tokio_util::sync::CancellationToken;

use super::common;
use crate::harness::agent_harness::BeforeRequestResult;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookFailure;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::{HookName, HookOptions, HookResult, PayloadResult, StepKind};
use crate::harness::context::{Context, background_context};
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::checkpoint::{run_checkpoint, start_run};
use crate::harness::runtime::drive::deferred::DeferredLeaf;
use crate::harness::runtime::drive::deferred::recover_deferred_poll;
use crate::harness::runtime::drive::deferred::{run_deferred, run_deferred_suspended};
use crate::harness::runtime::drive::generation::{GenerationLeaf, run_generation};
use crate::harness::runtime::test_support::deferred;
use crate::harness::runtime::test_support::provider_pass_through;
use crate::harness::runtime::test_support::{lock, patch_live_state, settle_events};
use crate::harness::runtime::types::{
    Drive, LaneCommand, LaneState, LiveOperation, ProcedureResult,
};
use crate::harness::session::commit::insert_entry;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::testing::GatingStorage;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::AssistantRetryWaitOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::Control;
use crate::harness::session::types::CustomEntryBody;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::DeferredSuspendedOperation;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::Storage;
use crate::harness::session::types::{
    TerminalStatus, ToolCallStatus, UsageScan, operation_scope_of,
};
use crate::harness::session::values::Write;
use crate::harness::session::values::append_list_write;
use crate::harness::session::values::{operation_state, pending_assistant_frames, set_value_write};
use crate::harness::types::{AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch};

/// The per-case fixture differences, upstream's `createFixture` options
/// object.
#[derive(Clone, Copy)]
struct FixtureOptions {
    /// Whether the backend sits under a [`GatingStorage`], upstream's
    /// `options.gated`.
    gated: bool,
    /// The faux provider's pending-fetch count, upstream's
    /// `options.pendingFetches`.
    pending_fetches: Option<u64>,
    /// Whether the stream options ask for deferred submission, upstream's
    /// `options.deferredSubmission ?? true`.
    deferred_submission: bool,
}

/// The `createFixture()` default: ungated, no pending fetches, deferred
/// submission on.
fn fixture_options() -> FixtureOptions {
    FixtureOptions {
        gated: false,
        pending_fetches: None,
        deferred_submission: true,
    }
}

/// The suite fixture, upstream's `Fixture`: the shared drive skeleton plus
/// the gated backend handle the gated cases drive through.
struct Fixture {
    /// The shared drive fixture skeleton, upstream's fixture fields.
    base: common::DriveFixture,
    /// The commit gate, upstream's `gating` field; `None` unless
    /// [`FixtureOptions::gated`].
    gating: Option<Arc<GatingStorage>>,
}

/// Builds one retry/deferred fixture, upstream's `createFixture`: the
/// optionally gated fixed-clock backend under the instrumented storage, the
/// faux provider carrying the deferred behavior, and the admitted `question`
/// run with its pass installed.
///
/// # Panics
/// The seed commit's, restore's, or admission's failure.
async fn create_fixture(options: FixtureOptions) -> Fixture {
    let gating = options.gated.then(|| {
        Arc::new(GatingStorage::new(Arc::new(MemoryStorage::new(
            MemoryStorageOptions {
                now: Some(Arc::new(|| 100)),
            },
        ))))
    });
    let backend: Option<Arc<dyn Storage>> = gating.as_ref().map(|gating| {
        let gated: Arc<dyn Storage> = gating.clone();
        gated
    });
    let spec = common::DriveFixtureSpec {
        suite: "retry-deferred",
        watch_suite: "retry/deferred",
        faux: RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            deferred: Some(FauxDeferredOptions {
                pending_fetches: options.pending_fetches,
                poll_after_ms: None,
            }),
            ..RegisterFauxProviderOptions::default()
        },
        retry_policy: RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 10,
            max_agent_delay_ms: None,
        },
        stream_options: AgentHarnessStreamOptions {
            deferred: options
                .deferred_submission
                .then_some(DeferredRequest::Enabled(true)),
            ..AgentHarnessStreamOptions::default()
        },
        backend,
        admit_prompt: true,
        system_prompt: None,
    };
    Fixture {
        base: common::create_drive_fixture(spec).await,
        gating,
    }
}

/// The settled response `submit_deferred` drives to suspension when the
/// case supplies none, upstream's default
/// `fauxAssistantMessage("done", { timestamp: 20 })`.
fn default_deferred_response() -> AssistantMessage {
    faux_assistant_message(
        "done",
        FauxAssistantMessageOptions {
            timestamp: Some(20),
            ..FauxAssistantMessageOptions::default()
        },
    )
}

/// The live operation's durable state leaf, upstream's `currentRun`.
/// The retry-wait leaf over the ready run, the deadline suites' seed; the
/// deadline rides `not_before` and the error message is upstream's
/// `"retry"`.
fn retry_wait_leaf(
    ready: &AssistantReadyOperation,
    not_before: i64,
) -> AssistantRetryWaitOperation {
    AssistantRetryWaitOperation {
        scope: ready.scope.clone(),
        generation_context: ready.generation_context.clone(),
        retry_wait: RetryWait {
            next_attempt: 2,
            not_before,
            error_message: "retry".to_owned(),
        },
    }
}

fn current_run(fixture: &Fixture) -> OperationState {
    common::current_state(&fixture.base)
}

/// Drives the admitted run from its starting leaf to the ready generation,
/// upstream's `advanceToReady`.
///
/// # Panics
/// Any intermediate leaf the boundaries did not produce, upstream's
/// `run did not start at its initial boundary` / `run did not reach
/// checkpoint` / `run did not reach ready generation` throws.
async fn advance_to_ready(fixture: &Fixture) -> AssistantReadyOperation {
    let OperationState::Starting(starting) = current_run(fixture) else {
        panic!("run did not start at its initial boundary");
    };
    start_run(&fixture.base.lane, &fixture.base.drive, &starting)
        .await
        .expect("the start settles");
    let OperationState::Checkpoint(checkpoint) = current_run(fixture) else {
        panic!("run did not reach checkpoint");
    };
    run_checkpoint(&fixture.base.lane, &fixture.base.drive, &checkpoint)
        .await
        .expect("the checkpoint settles");
    let OperationState::AssistantReady(ready) = current_run(fixture) else {
        panic!("run did not reach ready generation");
    };
    ready
}

/// Swaps the live operation's durable state with optional extra writes,
/// upstream's `replaceRunState`: one lane command committing the extras
/// ahead of the state write.
///
/// # Panics
/// The live operation's absence or the commit's failure.
async fn replace_run_state(
    fixture: &Fixture,
    next_state: OperationState,
    extra_writes: Vec<Write>,
) {
    let operation_id = fixture.base.operation_id.clone();
    let meta = fixture
        .base
        .lane
        .state()
        .operation
        .map(|operation| operation.meta)
        .expect("fixture has no operation");
    fixture
        .base
        .lane
        .command::<(), _>(
            move |projection, _session, _context| {
                let operation_id = operation_id.clone();
                let meta = meta.clone();
                let next_state = next_state.clone();
                let extra_writes = extra_writes.clone();
                Box::pin(async move {
                    let mut writes = extra_writes;
                    writes.push(
                        set_value_write(&operation_state(&operation_id), next_state.clone())
                            .expect("the state write"),
                    );
                    let next = LaneState {
                        operation: Some(LiveOperation {
                            meta,
                            state: next_state,
                        }),
                        ..projection
                    };
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(|_: &CommitResult| ()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the state patch commits");
}

/// Drives the admitted run through its initial deferred submission,
/// upstream's `submitDeferred`.
///
/// # Panics
/// Any intermediate boundary or the suspension's absence, upstream's
/// `initial request did not suspend` throw.
async fn submit_deferred(
    fixture: &Fixture,
    response: AssistantMessage,
) -> DeferredSuspendedOperation {
    fixture
        .base
        .faux
        .set_responses([FauxResponseStep::Message(response)]);
    let ready = advance_to_ready(fixture).await;
    run_generation(
        &fixture.base.lane,
        &fixture.base.drive,
        GenerationLeaf::Ready(ready),
    )
    .await
    .expect("the generation settles");
    let OperationState::DeferredSuspended(suspended) = current_run(fixture) else {
        panic!("initial request did not suspend");
    };
    suspended
}

/// Seeds one unknown-outcome poll at the next poll number with fresh ids
/// and one stale frame, upstream's `installUnknownPoll`.
///
/// # Panics
/// The live operation's absence or the patch's failure.
async fn install_unknown_poll(
    fixture: &Fixture,
    suspended: &DeferredSuspendedOperation,
) -> DeferredEffectPendingOperation {
    let effect_pending = DeferredEffectPendingOperation {
        scope: DeferredScope {
            scope: operation_scope_of(&current_run(fixture)),
            step_id: suspended.deferred.step_id.clone(),
            source_entry_id: suspended.deferred.source_entry_id.clone(),
            poll: suspended.deferred.poll + 1,
            configuration: suspended.deferred.configuration.clone(),
            stream_options: suspended.deferred.stream_options.clone(),
        },
        response_entry_id: fixture.base.session.id_generator().next(None),
        usage_id: fixture.base.session.id_generator().next(None),
    };
    replace_run_state(
        fixture,
        OperationState::DeferredEffectPending(effect_pending.clone()),
        vec![
            append_list_write(
                &pending_assistant_frames(
                    &fixture.base.operation_id,
                    &effect_pending.response_entry_id,
                ),
                AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "old".to_owned(),
                },
            )
            .expect("the stale frame write"),
        ],
    )
    .await;
    effect_pending
}

/// The live operation's deferred leaf, upstream's `currentDeferred`.
///
/// # Panics
/// The leaf's absence, upstream's `fixture has no deferred phase` throw.
fn current_deferred(fixture: &Fixture) -> DeferredLeaf {
    match current_run(fixture) {
        OperationState::DeferredSuspended(leaf) => DeferredLeaf::Suspended(leaf),
        OperationState::DeferredEffectPending(leaf) => DeferredLeaf::EffectPending(leaf),
        _ => panic!("fixture has no deferred phase"),
    }
}

/// Installs one fresh pass with the given wait options and makes it the
/// lane's owner, upstream's `installDrive`.
fn install_drive(
    fixture: &Fixture,
    wait_for_retry: Option<bool>,
    poll_deferred: Option<bool>,
) -> Arc<Drive> {
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: common::OPERATION_ID.to_owned(),
            wait_for_retry,
            poll_deferred,
        },
        &background_context(),
    ));
    fixture.base.lane.set_active_drive(Some(Arc::clone(&drive)));
    drive
}

/// Closes the fixture's session, upstream's `afterEach` close.
async fn close_fixture(fixture: &Fixture) {
    Session::close(fixture.base.session.as_ref(), &background_context())
        .await
        .expect("close");
}

/// The fetch options one poll's transport carries, upstream's captured
/// `fetchOptions`: the long-poll wait, the gate-linked cancellation token,
/// and the merged request headers.
#[derive(Default)]
struct CapturedFetch {
    /// The long-poll wait, upstream's `fetchOptions.wait`.
    wait: Option<u64>,
    /// The cancellation token, upstream's `fetchOptions.signal`.
    signal: Option<CancellationToken>,
    /// The request headers, upstream's `fetchOptions.headers`.
    headers: Option<ProviderHeaders>,
}

/// The provider wrapper one poll observation installs, upstream's object
/// spread `{ ...provider, fetchDeferred: (model, handle, options) => {...} }`:
/// every method delegates to the faux provider; `fetch_deferred` records
/// the options first.
struct FetchObservingProvider {
    /// The delegated faux provider.
    inner: ProviderImpl,
    /// The captured fetch options, upstream's `fetchOptions` assignment.
    captured: Arc<Mutex<Option<CapturedFetch>>>,
}

impl Provider for FetchObservingProvider {
    provider_pass_through!(inner);

    fn base_url(&self) -> Option<&str> {
        self.inner.base_url()
    }

    fn headers(&self) -> Option<&ProviderHeaders> {
        self.inner.headers()
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxedFuture<'_, Result<(), ProviderError>> {
        self.inner.refresh_models(context)
    }

    fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
        self.inner.filter_models(models, credential)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &AiContext,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        self.inner.stream_simple(model, context, options)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        if let Some(options) = options {
            *lock(&self.captured) = Some(CapturedFetch {
                wait: options.wait,
                signal: options.transport_options.signal.clone(),
                headers: options.headers.clone(),
            });
        }
        self.inner.fetch_deferred(model, handle, options)
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        options: Option<&'a DeferredCancelOptions>,
    ) -> Option<BoxedFuture<'a, Result<(), pi_ai::utils::provider_retry::ProviderRequestError>>>
    {
        self.inner.cancel_deferred(model, handle, options)
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.inner.supports_fetch_deferred()
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.inner.supports_cancel_deferred()
    }
}

/// Upstream `it`: classifies a live retryable provider error into durable
/// retry wait.
#[tokio::test]
async fn classifies_a_live_retryable_provider_error_into_durable_retry_wait() {
    let fixture = create_fixture(FixtureOptions {
        deferred_submission: false,
        ..fixture_options()
    })
    .await;
    let ready = advance_to_ready(&fixture).await;
    fixture
        .base
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("503 service unavailable".to_owned()),
                timestamp: Some(10),
                ..FauxAssistantMessageOptions::default()
            },
        ))]);

    let result = run_generation(
        &fixture.base.lane,
        &fixture.base.drive,
        GenerationLeaf::Ready(ready),
    )
    .await
    .expect("the generation settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the retryable failure continues into its wait"
    );
    let OperationState::AssistantRetryWait(retry) = current_run(&fixture) else {
        panic!("the retryable failure waits");
    };
    assert_eq!(
        retry.retry_wait.next_attempt, 2,
        "the wait advances the attempt"
    );
    assert_eq!(
        retry.retry_wait.error_message, "503 service unavailable",
        "the wait carries the error verbatim"
    );
    let last = lock(&fixture.base.events).last().cloned();
    let HarnessEventPayload::RetryScheduled { attempt, .. } =
        &last.expect("the retry schedules").payload
    else {
        panic!("the last event schedules the retry");
    };
    assert_eq!(*attempt, 2, "the schedule announces attempt 2");
    common::expect_projection_restores(&fixture.base).await;
    close_fixture(&fixture).await;
}

/// Upstream `it`: returns a durable waiting outcome without a timer or
/// write when local waiting is disabled.
#[tokio::test]
async fn returns_a_durable_waiting_outcome_without_a_timer_or_write_when_local_waiting_is_disabled()
{
    let fixture = create_fixture(FixtureOptions {
        deferred_submission: false,
        ..fixture_options()
    })
    .await;
    let ready = advance_to_ready(&fixture).await;
    // Upstream pins `Date.now` at 1_000 against `notBefore: 1_100`; the
    // wait's clock is the real epoch, so the deadline seeds 100 ms ahead.
    let not_before = now_ms() + 100;
    let retry_wait = AssistantRetryWaitOperation {
        scope: ready.scope.clone(),
        generation_context: ready.generation_context.clone(),
        retry_wait: RetryWait {
            next_attempt: 2,
            not_before,
            error_message: "retry".to_owned(),
        },
    };
    patch_live_state(
        &fixture.base.lane,
        OperationState::AssistantRetryWait(retry_wait.clone()),
    )
    .await;
    fixture.base.storage.clear_commit_attempts();

    let result = run_generation(
        &fixture.base.lane,
        &fixture.base.drive,
        GenerationLeaf::RetryWait(retry_wait),
    )
    .await
    .expect("the wait settles");
    let ProcedureResult::Waiting { outcome } = result else {
        panic!("the disabled wait reports waiting");
    };
    let DriveOutcome::Waiting {
        operation_id,
        reason: DriveWaitReason::Retry {
            not_before: reported,
        },
    } = outcome
    else {
        panic!("the wait rides the retry reason");
    };
    assert_eq!(
        operation_id.as_str(),
        common::OPERATION_ID,
        "the wait names the run"
    );
    assert_eq!(
        reported, not_before,
        "the wait reports its durable deadline"
    );
    assert!(
        fixture.base.storage.get_commit_attempts().is_empty(),
        "the disabled wait commits nothing"
    );
    close_fixture(&fixture).await;
}

/// Upstream `it`: commits ready at the deadline and emits retry lifecycle
/// around the next attempt.
#[tokio::test]
async fn commits_ready_at_the_deadline_and_emits_retry_lifecycle_around_the_next_attempt() {
    let fixture = create_fixture(FixtureOptions {
        deferred_submission: false,
        ..fixture_options()
    })
    .await;
    let ready = advance_to_ready(&fixture).await;
    // Upstream pins `Date.now` at 2_000 against `notBefore: 2_000` — a due
    // deadline; the real epoch seeds the due instant directly.
    let retry_wait = retry_wait_leaf(&ready, now_ms());
    patch_live_state(
        &fixture.base.lane,
        OperationState::AssistantRetryWait(retry_wait.clone()),
    )
    .await;
    fixture.base.storage.clear_commit_attempts();

    let result = run_generation(
        &fixture.base.lane,
        &fixture.base.drive,
        GenerationLeaf::RetryWait(retry_wait),
    )
    .await
    .expect("the wait settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the elapsed retry continues"
    );
    let OperationState::AssistantReady(next) = current_run(&fixture) else {
        panic!("retry did not become ready");
    };
    assert_eq!(next.next_attempt, 2, "the readiness keeps the attempt");
    let last = lock(&fixture.base.events).last().cloned();
    let HarnessEventPayload::RetryStart { attempt, .. } = &last.expect("the retry starts").payload
    else {
        panic!("the last event starts the retry");
    };
    assert_eq!(*attempt, 2, "the start announces attempt 2");
    fixture
        .base
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "retried",
            FauxAssistantMessageOptions {
                timestamp: Some(30),
                ..FauxAssistantMessageOptions::default()
            },
        ))]);
    run_generation(
        &fixture.base.lane,
        &fixture.base.drive,
        GenerationLeaf::Ready(next),
    )
    .await
    .expect("the retried generation settles");
    let ends_successfully = {
        let events = lock(&fixture.base.events);
        events.iter().any(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::RetryEnd {
                    attempt: 2,
                    success: true,
                    ..
                }
            )
        })
    };
    assert!(ends_successfully, "the retried attempt ends successfully");
    common::expect_projection_restores(&fixture.base).await;
    close_fixture(&fixture).await;
}

/// Upstream `it`: admits an abort-aware timer only for local waiting.
///
/// Upstream advances 99 of the 100 ms and reads no commit, then advances 1
/// and reads the readiness; the wait's clock is the real epoch, so the
/// split restates as no commit while the timer holds and the readiness once
/// the deadline passes.
#[tokio::test]
async fn admits_an_abort_aware_timer_only_for_local_waiting() {
    let fixture = create_fixture(FixtureOptions {
        deferred_submission: false,
        ..fixture_options()
    })
    .await;
    let ready = advance_to_ready(&fixture).await;
    // Upstream pins `Date.now` at 3_000 against `notBefore: 3_100` — a
    // 100 ms wait; the real epoch seeds the same offset.
    let retry_wait = retry_wait_leaf(&ready, now_ms() + 100);
    patch_live_state(
        &fixture.base.lane,
        OperationState::AssistantRetryWait(retry_wait.clone()),
    )
    .await;
    let drive = install_drive(&fixture, Some(true), None);
    fixture.base.storage.clear_commit_attempts();
    let lane = Arc::clone(&fixture.base.lane);
    let task_drive = Arc::clone(&drive);
    let polling = tokio::spawn(async move {
        run_generation(&lane, &task_drive, GenerationLeaf::RetryWait(retry_wait)).await
    });
    settle_events().await;
    assert!(
        fixture.base.storage.get_commit_attempts().is_empty(),
        "the timer holds before the deadline"
    );

    let result = polling
        .await
        .expect("the wait joins")
        .expect("the wait settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the elapsed timer continues"
    );
    let OperationState::AssistantReady(next) = current_run(&fixture) else {
        panic!("the elapsed timer becomes ready");
    };
    assert_eq!(next.next_attempt, 2, "the readiness advances the attempt");
    close_fixture(&fixture).await;
}

/// Upstream `it`: waits without a permit, then performs at most one pending
/// poll from captured options.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the case mirrors upstream's single `it` and its capture choreography"
)]
async fn waits_without_a_permit_then_performs_at_most_one_pending_poll_from_captured_options() {
    let fixture = create_fixture(FixtureOptions {
        pending_fetches: Some(1),
        ..fixture_options()
    })
    .await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    fixture.base.storage.clear_commit_attempts();

    let result = run_deferred_suspended(&fixture.base.lane, &fixture.base.drive, suspended.clone())
        .await
        .expect("the suspended pass settles");
    let ProcedureResult::Waiting { outcome } = result else {
        panic!("the permit-less pass waits");
    };
    let DriveOutcome::Waiting {
        operation_id,
        reason: DriveWaitReason::Deferred { deferred },
    } = outcome
    else {
        panic!("the wait rides the deferred handle");
    };
    assert_eq!(
        operation_id.as_str(),
        common::OPERATION_ID,
        "the wait names the run"
    );
    assert!(!deferred.id.is_empty(), "the wait carries the handle");
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the wait performs no fetch"
    );
    assert!(
        fixture.base.storage.get_commit_attempts().is_empty(),
        "the wait commits nothing"
    );

    let hook_options: Arc<Mutex<Option<AgentHarnessStreamOptions>>> = Arc::new(Mutex::new(None));
    let capture = Arc::clone(&hook_options);
    fixture
        .base
        .hooks
        .on(
            HookName::BeforeRequest,
            Arc::new(move |event: &HookInvocation, _context: &Context| {
                let capture = Arc::clone(&capture);
                Box::pin(async move {
                    if let HookEvent::BeforeRequest {
                        step: StepKind::Deferred,
                        stream_options,
                        ..
                    } = &event.event
                    {
                        *lock(&capture) = Some(stream_options.clone());
                    }
                    Ok(HookResult::BeforeRequest(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("the hook registers");
    let captured_fetch: Arc<Mutex<Option<CapturedFetch>>> = Arc::new(Mutex::new(None));
    fixture
        .base
        .models
        .set_provider(Arc::new(FetchObservingProvider {
            inner: fixture.base.faux.provider.clone(),
            captured: Arc::clone(&captured_fetch),
        }));
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred(&fixture.base.lane, &drive, current_deferred(&fixture))
        .await
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the poll continues"
    );
    assert_eq!(
        drive.deferred_permits.load(Ordering::SeqCst),
        0,
        "the poll spends its permit"
    );
    let hook_options = lock(&hook_options).take().expect("the deferred hook ran");
    assert_eq!(
        hook_options.deferred,
        Some(DeferredRequest::Enabled(false)),
        "the poll's hook options disable deferred submission"
    );
    let captured = lock(&captured_fetch).take().expect("the poll fetched");
    assert_eq!(
        captured.wait,
        Some(0),
        "the fetch performs one status check"
    );
    assert!(
        captured.signal.is_some(),
        "the fetch carries a cancellation token linked to the pass's gate signal"
    );
    let OperationState::DeferredSuspended(suspended) = current_run(&fixture) else {
        panic!("the pending poll suspends again");
    };
    assert_eq!(suspended.deferred.poll, 1, "the poll advances the counter");
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        1,
        "the poll fetches once"
    );
    let tail: Vec<String> = {
        let events = lock(&fixture.base.events);
        events[events.len() - 2..]
            .iter()
            .map(|event| event.event_type().as_str().to_owned())
            .collect()
    };
    assert_eq!(
        tail,
        ["turn_end", "run_suspend"],
        "the suspension publishes its tail"
    );

    let result = run_deferred(&fixture.base.lane, &drive, current_deferred(&fixture))
        .await
        .expect("the second pass settles");
    assert!(
        matches!(
            result,
            ProcedureResult::Waiting {
                outcome: DriveOutcome::Waiting {
                    reason: DriveWaitReason::Deferred { .. },
                    ..
                },
            }
        ),
        "the permit-less pass waits again"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        1,
        "the second wait performs no fetch"
    );

    // The fetch options' signal tracks the pass's gate signal; the gate's
    // abort cancels it, the ported identity assert.
    drive.begin_abort(tokio::sync::watch::channel(()).1);
    drive.signal_abort();
    let captured_token = captured.signal.expect("the captured token");
    common::wait_for(|| captured_token.is_cancelled()).await;

    common::expect_projection_restores(&fixture.base).await;
    close_fixture(&fixture).await;
}

/// Upstream `it`: declines poll intent when cancellation wins preparation.
#[tokio::test]
async fn declines_poll_intent_when_cancellation_wins_preparation() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let drive = install_drive(&fixture, None, Some(true));
    let (hook_started_tx, hook_started_rx) = deferred();
    let (release_tx, release_rx) = deferred();
    let started_cell = Arc::new(Mutex::new(Some(hook_started_tx)));
    let release_cell = Arc::new(Mutex::new(Some(release_rx)));
    fixture
        .base
        .hooks
        .on(
            HookName::BeforeRequest,
            Arc::new(move |event: &HookInvocation, _context: &Context| {
                let started_cell = Arc::clone(&started_cell);
                let release_cell = Arc::clone(&release_cell);
                Box::pin(async move {
                    if let HookEvent::BeforeRequest { step, .. } = &event.event
                        && *step == StepKind::Deferred
                    {
                        let started = lock(&started_cell).take();
                        if let Some(started) = started {
                            let _ = started.send(());
                        }
                        // The guard ends before the park; the held lock must
                        // not cross the await.
                        let release = lock(&release_cell).take();
                        if let Some(release) = release {
                            let _ = release.await;
                        }
                    }
                    Ok(HookResult::BeforeRequest(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("the hook registers");

    let lane = Arc::clone(&fixture.base.lane);
    let polling =
        tokio::spawn(async move { run_deferred_suspended(&lane, &drive, suspended).await });
    hook_started_rx.await.expect("the hook parks");
    common::cancel_operation(&fixture.base).await;
    let _ = release_tx.send(());

    let result = polling
        .await
        .expect("the poll joins")
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled preparation continues"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the declined poll performs no fetch"
    );
    let OperationState::DeferredSuspended(leaf) = current_run(&fixture) else {
        panic!("the declined poll stays suspended");
    };
    assert!(
        matches!(leaf.deferred.scope.control, Control::CancelRequested { .. }),
        "the suspension carries the cancellation"
    );
    close_fixture(&fixture).await;
}

/// Upstream `it`: plans ready deferred tool calls with the poll turn
/// identity and follower result ids.
#[tokio::test]
async fn plans_ready_deferred_tool_calls_with_the_poll_turn_identity_and_follower_result_ids() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(
        &fixture,
        faux_assistant_message(
            faux_tool_call(
                "lookup",
                serde_json::json!({ "query": "value" })
                    .as_object()
                    .expect("the tool arguments")
                    .clone(),
                None,
            ),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                timestamp: Some(20),
                ..FauxAssistantMessageOptions::default()
            },
        ),
    )
    .await;
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred_suspended(&fixture.base.lane, &drive, suspended.clone())
        .await
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the tool poll continues"
    );
    let OperationState::Tools(run) = current_run(&fixture) else {
        panic!("deferred tool response did not create a batch");
    };
    assert_eq!(
        run.batch.turn_id,
        format!("{}:poll:1", suspended.deferred.step_id),
        "the batch carries the poll turn identity"
    );
    assert_eq!(run.batch.calls.len(), 1, "the batch plans one call");
    assert_eq!(
        run.batch.calls[0].source_index, 0,
        "the call keeps its source index"
    );
    assert!(
        matches!(run.batch.calls[0].status(), ToolCallStatus::Planned),
        "the planned call stays planned"
    );
    let latest = run
        .scope
        .latest_assistant_entry_id
        .as_deref()
        .expect("the latest assistant entry");
    assert_eq!(
        run.batch.calls[0].result_entry_id.get(..13),
        latest.get(..13),
        "the follower result id shares the poll response's id seed"
    );
    let settles_on_usage = {
        let events = lock(&fixture.base.events);
        matches!(
            events.last().expect("events").payload,
            HarnessEventPayload::Usage { .. }
        )
    };
    assert!(settles_on_usage, "the batch settles on its usage event");
    common::expect_projection_restores(&fixture.base).await;
    close_fixture(&fixture).await;
}

/// Upstream `it`: consumes its permit only after the fresh intent commit
/// lands.
#[tokio::test]
async fn consumes_its_permit_only_after_the_fresh_intent_commit_lands() {
    let fixture = create_fixture(FixtureOptions {
        gated: true,
        ..fixture_options()
    })
    .await;
    let suspended = submit_deferred(
        &fixture,
        faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("failed".to_owned()),
                timestamp: Some(20),
                ..FauxAssistantMessageOptions::default()
            },
        ),
    )
    .await;
    let gating = fixture.gating.clone().expect("fixture is not gated");
    let drive = install_drive(&fixture, None, Some(true));
    gating.arm();
    let lane = Arc::clone(&fixture.base.lane);
    let task_drive = Arc::clone(&drive);
    let polling =
        tokio::spawn(async move { run_deferred_suspended(&lane, &task_drive, suspended).await });

    gating.wait_pending(1).await.expect("the intent parks");
    assert_eq!(
        drive.deferred_permits.load(Ordering::SeqCst),
        1,
        "the permit stays unspent while the intent parks"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the parked intent performs no fetch"
    );
    gating.next(1).await.expect("the intent lands");
    assert_eq!(
        drive.deferred_permits.load(Ordering::SeqCst),
        0,
        "the permit spends at the landed intent"
    );
    common::wait_for(|| fixture.base.faux.state().deferred_fetch_count() == 1).await;
    gating.next(2).await.expect("the frame and settlement land");
    let result = polling
        .await
        .expect("the poll joins")
        .expect("the poll settles");
    let ProcedureResult::Settled { outcome } = result else {
        panic!("the failing poll settles the run");
    };
    assert_eq!(
        outcome.operation_id.as_str(),
        common::OPERATION_ID,
        "the record names the run"
    );
    assert_eq!(
        outcome.kind,
        OperationKind::Run,
        "the record keeps the run kind"
    );
    assert_eq!(
        outcome.status,
        TerminalStatus::Failed,
        "the poll fails the run"
    );
    assert!(
        fixture.base.lane.state().operation.is_none(),
        "the settled run clears the operation"
    );
    close_fixture(&fixture).await;
}

/// Upstream `it`: leaves suspended state unchanged when the fresh intent is
/// discarded.
#[tokio::test]
async fn leaves_suspended_state_unchanged_when_the_fresh_intent_is_discarded() {
    let fixture = create_fixture(FixtureOptions {
        gated: true,
        ..fixture_options()
    })
    .await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let gating = fixture.gating.clone().expect("fixture is not gated");
    let drive = install_drive(&fixture, None, Some(true));
    gating.arm();
    let lane = Arc::clone(&fixture.base.lane);
    let task_drive = Arc::clone(&drive);
    let polling =
        tokio::spawn(async move { run_deferred_suspended(&lane, &task_drive, suspended).await });

    gating.wait_pending(1).await.expect("the intent parks");
    gating.discard();
    let joined = polling.await.expect("the poll joins");
    let error = joined.expect_err("the discarded intent rejects the poll");
    assert!(
        error.to_string().contains("storage discarded"),
        "the discard carries the storage-discard text: {error}"
    );
    assert_eq!(
        drive.deferred_permits.load(Ordering::SeqCst),
        1,
        "the discarded intent leaves the permit unspent"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the discarded intent performs no fetch"
    );
    let durable = crate::harness::runtime::test_support::stored_value(
        &fixture.base.session,
        &operation_state(common::OPERATION_ID).address,
        &background_context(),
    )
    .await;
    assert_eq!(
        durable.value.get("at"),
        Some(&serde_json::json!("deferred.suspended")),
        "the durable state stays suspended"
    );
    assert_eq!(
        durable.value.get("poll"),
        Some(&serde_json::json!(0)),
        "the poll stays 0"
    );
    close_fixture(&fixture).await;
}

/// Upstream `it`: replaces an unknown poll under fresh ids at the same poll
/// number and deletes old frames.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the case mirrors upstream's single `it` and its two-pass recovery"
)]
async fn replaces_an_unknown_poll_under_fresh_ids_at_the_same_poll_number_and_deletes_old_frames() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let unknown = install_unknown_poll(&fixture, &suspended).await;
    fixture.base.storage.clear_commit_attempts();

    let no_permit = install_drive(&fixture, None, None);
    let result = recover_deferred_poll(&fixture.base.lane, &no_permit, unknown.clone())
        .await
        .expect("the recovery settles");
    assert!(
        matches!(
            result,
            ProcedureResult::Waiting {
                outcome: DriveOutcome::Waiting {
                    reason: DriveWaitReason::Deferred { .. },
                    ..
                },
            }
        ),
        "the permit-less recovery waits"
    );
    let frames = fixture
        .base
        .session
        .read_list(
            &pending_assistant_frames(common::OPERATION_ID, &unknown.response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("the frame read");
    assert_eq!(
        frames.len(),
        1,
        "the old frame list survives without a permit"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the permit-less recovery performs no fetch"
    );

    let replacement = install_drive(&fixture, None, Some(true));
    let result = recover_deferred_poll(&fixture.base.lane, &replacement, unknown.clone())
        .await
        .expect("the replacement settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the permitted replacement continues"
    );
    let intent = fixture
        .base
        .storage
        .get_commit_attempts()
        .iter()
        .find(|attempt| {
            attempt.iter().any(|write| {
                matches!(write, Write::ListDelete(list)
                    if list.kind == "list"
                        && list.op == "delete"
                        && list.namespace == "pi.pending.assistant_frame"
                        && list.key.ends_with(&unknown.response_entry_id))
            })
        })
        .expect("the intent commit")
        .clone();
    let families: Vec<String> = intent
        .iter()
        .map(|write| match write {
            Write::Entry(_) | Write::Usage(_) => "insert".to_owned(),
            Write::ValueSet(value) => format!("{}:{}", value.kind, value.op),
            Write::ValueDelete(value) => format!("{}:{}", value.kind, value.op),
            Write::ListAppend(value) => format!("{}:{}", value.kind, value.op),
            Write::ListDelete(value) => format!("{}:{}", value.kind, value.op),
        })
        .collect();
    assert_eq!(
        families,
        ["list:delete".to_owned(), "value:set".to_owned()],
        "the intent commits the frame delete and the fresh state"
    );
    let intent_state = intent
        .iter()
        .find_map(|write| match write {
            Write::ValueSet(value) if value.namespace == "pi.op.state" => Some(value.value.clone()),
            _ => None,
        })
        .expect("the state write");
    let parsed = serde_json::from_value::<OperationState>(intent_state).expect("the state parses");
    assert_eq!(
        parsed.at(),
        "deferred.effect_pending",
        "the replacement stays in the deferred phase"
    );
    let OperationState::DeferredEffectPending(intent_leaf) = parsed else {
        panic!("the intent state is the effect-pending leaf");
    };
    assert_eq!(
        intent_leaf.scope.poll, unknown.scope.poll,
        "the replacement keeps the poll number"
    );
    assert_ne!(
        intent_leaf.response_entry_id, unknown.response_entry_id,
        "the replacement reserves a fresh response id"
    );
    assert_ne!(
        intent_leaf.usage_id, unknown.usage_id,
        "the replacement reserves a fresh usage id"
    );

    let entry = fixture
        .base
        .session
        .get_entry(&unknown.response_entry_id, &background_context())
        .await
        .expect("the entry read");
    assert!(entry.is_none(), "the old response entry never lands");
    let usage = fixture
        .base
        .storage
        .scan_usage(&UsageScan::default(), &background_context())
        .await
        .expect("the usage scan");
    assert!(
        !usage.iter().any(|row| row.id == unknown.usage_id),
        "the old usage row stays absent"
    );
    let frames = fixture
        .base
        .session
        .read_list(
            &pending_assistant_frames(common::OPERATION_ID, &unknown.response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("the frame read");
    assert!(frames.is_empty(), "the old frames delete with the intent");
    assert_eq!(
        current_run(&fixture).at(),
        "checkpoint",
        "the replacement run continues"
    );
    let publishes_recovery = {
        let events = lock(&fixture.base.events);
        events.iter().any(|event| event.recovery)
    };
    assert!(publishes_recovery, "a recovery-flagged event publishes");
    common::expect_projection_restores(&fixture.base).await;
    close_fixture(&fixture).await;
}

/// Upstream `it`: abandons unknown ids and frames into configuration
/// failure when the captured model is unavailable.
#[tokio::test]
async fn abandons_unknown_ids_and_frames_into_configuration_failure_when_the_captured_model_is_unavailable()
 {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let unknown = install_unknown_poll(&fixture, &suspended).await;
    fixture
        .base
        .models
        .delete_provider(fixture.base.faux.provider.id());
    fixture.base.storage.clear_commit_attempts();
    let drive = install_drive(&fixture, None, Some(true));

    let result = recover_deferred_poll(&fixture.base.lane, &drive, unknown.clone())
        .await
        .expect("the recovery settles");
    let ProcedureResult::Settled { outcome } = result else {
        panic!("the unavailable model settles the run");
    };
    assert_eq!(
        outcome.operation_id.as_str(),
        common::OPERATION_ID,
        "the record names the run"
    );
    assert_eq!(
        outcome.kind,
        OperationKind::Run,
        "the record keeps the run kind"
    );
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    let error = outcome.error.expect("the failure carries its error");
    assert_eq!(
        error.code, "model_unavailable",
        "the failure names the unavailable model"
    );
    assert_eq!(
        drive.deferred_permits.load(Ordering::SeqCst),
        1,
        "the unspent permit stays"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the unavailable model performs no fetch"
    );
    assert!(
        fixture.base.lane.state().operation.is_none(),
        "the failed run clears the operation"
    );
    let last = fixture
        .base
        .storage
        .get_commit_attempts()
        .last()
        .expect("the terminal commit")
        .clone();
    let families: Vec<String> = last
        .iter()
        .map(|write| match write {
            Write::Entry(_) => "entry".to_owned(),
            Write::Usage(_) => "usage".to_owned(),
            Write::ValueSet(value) => format!("{}:{}:{}", value.kind, value.op, value.namespace),
            Write::ValueDelete(value) => format!("{}:{}:{}", value.kind, value.op, value.namespace),
            Write::ListAppend(value) => format!("{}:{}:{}", value.kind, value.op, value.namespace),
            Write::ListDelete(value) => format!("{}:{}:{}", value.kind, value.op, value.namespace),
        })
        .collect();
    assert_eq!(
        families,
        [
            "value:delete:pi.op.meta".to_owned(),
            "value:delete:pi.op.state".to_owned(),
            "list:delete:pi.pending.assistant_frame".to_owned(),
            "value:set:pi.result".to_owned(),
            "value:set:pi.lane.state".to_owned(),
        ],
        "the terminal transaction deletes the unknown ids and frames exactly"
    );
    assert!(
        !fixture
            .base
            .storage
            .get_commit_attempts()
            .iter()
            .flatten()
            .any(|write| matches!(write, Write::Entry(_) | Write::Usage(_))),
        "no entry or usage write lands"
    );
    common::expect_projection_restores(&fixture.base).await;
    close_fixture(&fixture).await;
}

/// Upstream `pollDeferred` configuration-failure arm over the suspended
/// leaf: the unavailable captured model abandons the run before any id is
/// reserved, upstream's `publishConfigurationFailure` with the suspended
/// capability.
#[tokio::test]
async fn suspended_poll_abandons_into_configuration_failure_when_the_model_is_unavailable() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    fixture
        .base
        .models
        .delete_provider(fixture.base.faux.provider.id());
    fixture.base.storage.clear_commit_attempts();
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred_suspended(&fixture.base.lane, &drive, suspended.clone())
        .await
        .expect("the recovery settles");
    let ProcedureResult::Settled { outcome } = result else {
        panic!("the unavailable model settles the run");
    };
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    let error = outcome.error.expect("the failure carries its error");
    assert_eq!(
        error.code, "model_unavailable",
        "the failure names the unavailable model"
    );
    assert_eq!(
        drive.deferred_permits.load(Ordering::SeqCst),
        1,
        "the unspent permit stays"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the unavailable model performs no fetch"
    );
    assert!(
        fixture.base.lane.state().operation.is_none(),
        "the failed run clears the operation"
    );
    close_fixture(&fixture).await;
}

/// Upstream `readDeferredSourceHandle` invariant: a source entry that is
/// not an assistant message faults the poll's preparation.
#[tokio::test]
async fn poll_declines_when_the_source_entry_is_not_an_assistant_message() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let mut leaf = suspended.clone();
    leaf.deferred.source_entry_id = "wrong-source".to_owned();
    replace_run_state(
        &fixture,
        OperationState::DeferredSuspended(leaf.clone()),
        vec![Write::Entry(Box::new(insert_entry(common::user_entry(
            "wrong-source",
            None,
            "not an assistant",
        ))))],
    )
    .await;
    let drive = install_drive(&fixture, None, Some(true));

    let error = run_deferred_suspended(&fixture.base.lane, &drive, leaf)
        .await
        .expect_err("the foreign source faults");
    assert!(
        error
            .to_string()
            .contains("Deferred source wrong-source is missing its assistant handle"),
        "the source invariant carried: {error}"
    );
    close_fixture(&fixture).await;
}

/// Upstream `readDeferredSourceHandle` invariant: a source entry that is
/// not a message entry faults the poll's preparation.
#[tokio::test]
async fn poll_declines_when_the_source_entry_is_not_a_message_entry() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let mut leaf = suspended.clone();
    leaf.deferred.source_entry_id = "custom-source".to_owned();
    replace_run_state(
        &fixture,
        OperationState::DeferredSuspended(leaf.clone()),
        vec![Write::Entry(Box::new(insert_entry(NewEntry::Custom {
            id: "custom-source".to_owned(),
            parent_id: None,
            body: CustomEntryBody {
                custom_type: "note".to_owned(),
                data: None,
            },
        })))],
    )
    .await;
    let drive = install_drive(&fixture, None, Some(true));

    let error = run_deferred_suspended(&fixture.base.lane, &drive, leaf)
        .await
        .expect_err("the custom source faults");
    assert!(
        error
            .to_string()
            .contains("Deferred source custom-source is missing its assistant handle"),
        "the source invariant carried: {error}"
    );
    close_fixture(&fixture).await;
}

/// Upstream `readDeferredSourceHandle` invariant: a deferred handle whose
/// identity mismatches the leaf's captured configuration faults the poll's
/// preparation.
#[tokio::test]
async fn poll_declines_when_the_source_handle_mismatches_the_leaf_configuration() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let mut leaf = suspended.clone();
    leaf.deferred.source_entry_id = "other-source".to_owned();
    let handle = DeferredHandle {
        provider: "other-provider".to_owned(),
        model_id: "other-model".to_owned(),
        api: "other-api".to_owned(),
        id: "job".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    };
    replace_run_state(
        &fixture,
        OperationState::DeferredSuspended(leaf.clone()),
        vec![Write::Entry(Box::new(insert_entry(NewEntry::Message {
            id: "other-source".to_owned(),
            parent_id: None,
            body: Box::new(MessageEntry {
                message: crate::types::AgentMessage::Standard(Message::Assistant(
                    faux_assistant_message(
                        Vec::<FauxContentBlock>::new(),
                        FauxAssistantMessageOptions {
                            stop_reason: Some(StopReason::Deferred),
                            deferred: Some(handle),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ),
                )),
                terminate: None,
            }),
        })))],
    )
    .await;
    let drive = install_drive(&fixture, None, Some(true));

    let error = run_deferred_suspended(&fixture.base.lane, &drive, leaf)
        .await
        .expect_err("the mismatched handle faults");
    assert!(
        error
            .to_string()
            .contains("Deferred source other-source has an invalid handle"),
        "the handle invariant carried: {error}"
    );
    close_fixture(&fixture).await;
}

/// Upstream `pollDeferred` arm: cancellation requested before the source
/// read downgrades the poll to a plain continue, upstream's
/// `continueOperation` contract.
#[tokio::test]
async fn cancellation_wins_the_poll_source_read() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    common::cancel_operation(&fixture.base).await;
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred_suspended(&fixture.base.lane, &drive, suspended.clone())
        .await
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled poll continues: {result:?}"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the cancelled poll performs no fetch"
    );
    let OperationState::DeferredSuspended(leaf) = current_run(&fixture) else {
        panic!("the declined poll stays suspended");
    };
    assert!(
        matches!(leaf.deferred.scope.control, Control::CancelRequested { .. }),
        "the suspension carries the cancellation"
    );
    close_fixture(&fixture).await;
}

/// Upstream `prepareDeferredPoll` binding: a closed hook registry faults
/// the poll's preparation, upstream's rejected `runWithGate` promise.
#[tokio::test]
async fn a_closed_hook_registry_faults_the_poll_preparation() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let drive = install_drive(&fixture, None, Some(true));
    fixture.base.hooks.close("closed for the poll".to_owned());

    let error = run_deferred_suspended(&fixture.base.lane, &drive, suspended.clone())
        .await
        .expect_err("the closed registry faults");
    assert!(
        error.to_string().contains("closed for the poll"),
        "the closed registry's error carried: {error}"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the faulted preparation performs no fetch",
    );
    close_fixture(&fixture).await;
}

/// Upstream `prepareDeferredPoll` binding: a pass whose abort began before
/// the hook faults the poll's preparation with the abort request,
/// upstream's `AbortRequested` throw.
#[tokio::test]
async fn an_aborting_pass_faults_the_poll_preparation() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let drive = install_drive(&fixture, None, Some(true));
    // The abort begins without firing the signal, so the hook's admission
    // refuses while the gate's own signal stays silent, upstream's
    // `beginAbort` before the hook.
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);

    let error = run_deferred_suspended(&fixture.base.lane, &drive, suspended.clone())
        .await
        .expect_err("the aborting pass faults");
    assert_eq!(
        error.to_string(),
        "Abort requested",
        "the abort request carried: {error}"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the faulted preparation performs no fetch",
    );
    close_fixture(&fixture).await;
}

/// Upstream `prepareDeferredPoll` binding: a `before_request` patch
/// rewrites the poll's fetch options, upstream's
/// `applyStreamOptionsPatch(baseOptions, beforeRequest.streamOptions)`.
#[tokio::test]
async fn a_before_request_patch_rewrites_the_poll_options() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    fixture
        .base
        .hooks
        .on(
            HookName::BeforeRequest,
            Arc::new(|_event: &HookInvocation, _context: &Context| {
                Box::pin(async move {
                    Ok(HookResult::BeforeRequest(Some(BeforeRequestResult {
                        stream_options: AgentHarnessStreamOptionsPatch {
                            headers: Some(Some(ProviderHeaders::from([(
                                "x-patch".to_owned(),
                                Some("patched".to_owned()),
                            )]))),
                            ..AgentHarnessStreamOptionsPatch::default()
                        },
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("the hook registers");
    let captured_fetch: Arc<Mutex<Option<CapturedFetch>>> = Arc::new(Mutex::new(None));
    fixture
        .base
        .models
        .set_provider(Arc::new(FetchObservingProvider {
            inner: fixture.base.faux.provider.clone(),
            captured: Arc::clone(&captured_fetch),
        }));
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred_suspended(&fixture.base.lane, &drive, suspended.clone())
        .await
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the patched poll continues"
    );
    let captured = lock(&captured_fetch).take().expect("the poll fetched");
    let headers = captured.headers.expect("the fetch carries headers");
    assert_eq!(
        headers.get("x-patch").map(Option::as_deref),
        Some(Some("patched")),
        "the patch's header rides the fetch"
    );
    close_fixture(&fixture).await;
}

/// Upstream `performDeferredPoll` option assembly: the leaf's captured
/// headers ride the poll's fetch, upstream's
/// `headers: prepared.streamOptions.headers`.
#[tokio::test]
async fn the_leaf_headers_ride_the_poll_fetch() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let mut leaf = suspended.clone();
    leaf.deferred.stream_options.headers =
        Some(BTreeMap::from([("x-leaf".to_owned(), "leaf".to_owned())]));
    replace_run_state(
        &fixture,
        OperationState::DeferredSuspended(leaf.clone()),
        Vec::new(),
    )
    .await;
    let captured_fetch: Arc<Mutex<Option<CapturedFetch>>> = Arc::new(Mutex::new(None));
    fixture
        .base
        .models
        .set_provider(Arc::new(FetchObservingProvider {
            inner: fixture.base.faux.provider.clone(),
            captured: Arc::clone(&captured_fetch),
        }));
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred_suspended(&fixture.base.lane, &drive, leaf)
        .await
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the poll continues"
    );
    let captured = lock(&captured_fetch).take().expect("the poll fetched");
    let headers = captured.headers.expect("the fetch carries headers");
    assert_eq!(
        headers.get("x-leaf").map(Option::as_deref),
        Some(Some("leaf")),
        "the leaf's header rides the fetch"
    );
    close_fixture(&fixture).await;
}

/// The provider streams whose deferred fetch runs the request's payload
/// hook before its events, upstream's `onPayload` invocation: the faux core
/// streams every method; `fetch_deferred` invokes the captured `onPayload`
/// hook behind the optional hold gate, then streams a settled `stop`
/// message.
struct PayloadObservingStreams {
    inner: FauxCore,
    invoked: Arc<AtomicUsize>,
    hold: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl ProviderStreams for PayloadObservingStreams {
    fn stream(
        &self,
        model: &Model,
        context: &AiContext,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &AiContext,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        self.inner.stream_simple(model, context, options)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        _handle: &DeferredHandle,
        options: Option<&DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        let on_payload = options
            .and_then(|options| options.transport_options.on_payload.clone())
            .expect("the poll carries its payload hook");
        self.invoked.fetch_add(1, Ordering::SeqCst);
        let message = faux_assistant_message(
            "resumed",
            FauxAssistantMessageOptions {
                timestamp: Some(30),
                ..FauxAssistantMessageOptions::default()
            },
        );
        let stream = assistant_message_event_stream();
        let stream_for_push = stream.clone();
        let model = model.clone();
        let hold = lock(&self.hold).take();
        tokio::spawn(async move {
            if let Some(hold) = hold {
                let _ = hold.await;
            }
            let _ = on_payload
                .call(serde_json::json!({"probe": true}), model)
                .await;
            stream_for_push.push(AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            stream_for_push.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            });
        });
        Some(stream)
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        options: Option<&'a DeferredCancelOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::utils::provider_retry::ProviderRequestError>> {
        self.inner.cancel_deferred(model, handle, options)
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

/// Installs the payload-observing provider and its invocation counter, the
/// payload cases' shared setup.
fn install_payload_observing_provider(fixture: &Fixture) -> Arc<AtomicUsize> {
    let invoked = Arc::new(AtomicUsize::new(0));
    let provider = create_provider(CreateProviderOptions {
        id: Provider::id(&fixture.base.faux.provider).to_owned(),
        name: Some(Provider::name(&fixture.base.faux.provider).to_owned()),
        base_url: None,
        headers: None,
        auth: Provider::auth(&fixture.base.faux.provider).clone(),
        models: fixture.base.faux.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: ProviderApi::Single(Arc::new(PayloadObservingStreams {
            inner: fixture.base.faux.core().clone(),
            invoked: Arc::clone(&invoked),
            hold: Mutex::new(None),
        })),
    });
    fixture.base.models.set_provider(Arc::new(provider));
    invoked
}

/// Upstream `performDeferredPoll` binding: the request's payload hook runs
/// the `before_payload` hook through the pass's gate, upstream's
/// `onPayload: async (payload, requestModel) => ...` body.
#[tokio::test]
async fn the_polls_payload_hook_runs_the_before_payload_hook() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let invoked = install_payload_observing_provider(&fixture);
    let payload_seen = Arc::new(Mutex::new(None::<serde_json::Value>));
    let seen = Arc::clone(&payload_seen);
    fixture
        .base
        .hooks
        .on(
            HookName::BeforePayload,
            Arc::new(move |event: &HookInvocation, _context: &Context| {
                let seen = Arc::clone(&seen);
                Box::pin(async move {
                    if let HookEvent::BeforePayload { payload, .. } = &event.event {
                        *lock(&seen) = Some(payload.clone());
                    }
                    Ok(HookResult::BeforePayload(Some(PayloadResult {
                        payload: serde_json::json!({"replaced": true}),
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("the hook registers");
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred_suspended(&fixture.base.lane, &drive, suspended.clone())
        .await
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the polled response continues"
    );
    assert_eq!(invoked.load(Ordering::SeqCst), 1, "one payload hook ran");
    assert_eq!(
        lock(&payload_seen).as_ref(),
        Some(&serde_json::json!({"probe": true})),
        "the hook saw the request's payload"
    );
    close_fixture(&fixture).await;
}

/// Upstream `performDeferredPoll` binding: a faulting `before_payload` hook
/// hands the payload back unchanged, upstream's swallowed onPayload throw.
///
/// The before-payload aggregate reports handler failures and keeps going,
/// so the arm rides the gate's mid-request abort: the pass aborts while the
/// fetch parks, and the payload hook's refusal hands `None` back to the
/// transport.
#[tokio::test]
async fn a_refused_payload_hook_hands_the_payload_back() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let invoked = Arc::new(AtomicUsize::new(0));
    let (hold_tx, hold_rx) = tokio::sync::oneshot::channel::<()>();
    let provider = create_provider(CreateProviderOptions {
        id: Provider::id(&fixture.base.faux.provider).to_owned(),
        name: Some(Provider::name(&fixture.base.faux.provider).to_owned()),
        base_url: None,
        headers: None,
        auth: Provider::auth(&fixture.base.faux.provider).clone(),
        models: fixture.base.faux.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: ProviderApi::Single(Arc::new(PayloadObservingStreams {
            inner: fixture.base.faux.core().clone(),
            invoked: Arc::clone(&invoked),
            hold: Mutex::new(Some(hold_rx)),
        })),
    });
    fixture.base.models.set_provider(Arc::new(provider));
    fixture
        .base
        .hooks
        .on(
            HookName::BeforePayload,
            Arc::new(|_event: &HookInvocation, _context: &Context| {
                let failure: HookFailure =
                    Box::new(SessionError::Message("blocked payload".to_owned()));
                Box::pin(async move { Err(failure) })
            }),
            HookOptions::default(),
        )
        .expect("the hook registers");
    let drive = install_drive(&fixture, None, Some(true));

    let lane = Arc::clone(&fixture.base.lane);
    let task_drive = Arc::clone(&drive);
    let polling =
        tokio::spawn(async move { run_deferred_suspended(&lane, &task_drive, suspended).await });
    common::wait_for(|| invoked.load(Ordering::SeqCst) == 1).await;
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);
    drive.signal_abort();
    let _ = hold_tx.send(());

    let result = polling
        .await
        .expect("the poll joins")
        .expect("the poll settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the payload refusal does not fault the poll: {result:?}"
    );
    assert_eq!(invoked.load(Ordering::SeqCst), 1, "one payload hook ran");
    close_fixture(&fixture).await;
}

/// Parks the deferred `before_request` hook between its started and release
/// gates so the test can flip the pass's gate mid-preparation, the gate
/// rejection cases' shared choreography.
///
/// # Panics
/// The registration's failure.
fn park_deferred_before_request(
    fixture: &Fixture,
) -> (
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
) {
    let (started_tx, started_rx) = deferred();
    let (release_tx, release_rx) = deferred();
    let started_slot = Arc::new(Mutex::new(Some(started_tx)));
    let release_slot = Arc::new(Mutex::new(Some(release_rx)));
    fixture
        .base
        .hooks
        .on(
            HookName::BeforeRequest,
            Arc::new(move |event: &HookInvocation, _context: &Context| {
                let started_slot = Arc::clone(&started_slot);
                let release_slot = Arc::clone(&release_slot);
                Box::pin(async move {
                    if let HookEvent::BeforeRequest { step, .. } = &event.event
                        && *step == StepKind::Deferred
                    {
                        let started = lock(&started_slot).take();
                        if let Some(started) = started {
                            let _ = started.send(());
                        }
                        let release = lock(&release_slot).take();
                        if let Some(release) = release {
                            let _ = release.await;
                        }
                    }
                    Ok(HookResult::BeforeRequest(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("the hook registers");
    (started_rx, release_tx)
}

/// Upstream `performDeferredPoll` arm: a pass whose abort wins after the
/// hook faults the stream's admission, upstream's `admit` rejection.
#[tokio::test]
async fn an_aborting_pass_rejects_the_poll_stream_admission() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let drive = install_drive(&fixture, None, Some(true));
    let (started_rx, release_tx) = park_deferred_before_request(&fixture);

    let lane = Arc::clone(&fixture.base.lane);
    let task_drive = Arc::clone(&drive);
    let polling =
        tokio::spawn(async move { run_deferred_suspended(&lane, &task_drive, suspended).await });
    started_rx.await.expect("the hook parks");
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);
    let _ = release_tx.send(());

    let error = polling
        .await
        .expect("the poll joins")
        .expect_err("the aborting pass rejects the admission");
    assert_eq!(
        error.to_string(),
        "Abort requested",
        "the admission rejection carried: {error}"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        0,
        "the rejected admission performs no fetch"
    );
    close_fixture(&fixture).await;
}

/// Upstream `performDeferredPoll` arm: a closed pass rejects the stream's
/// admission with the gate's error, upstream's `Gate.closed` rejection.
#[tokio::test]
async fn a_closed_pass_rejects_the_poll_stream_admission() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let drive = install_drive(&fixture, None, Some(true));
    let (started_rx, release_tx) = park_deferred_before_request(&fixture);

    let lane = Arc::clone(&fixture.base.lane);
    let task_drive = Arc::clone(&drive);
    let polling =
        tokio::spawn(async move { run_deferred_suspended(&lane, &task_drive, suspended).await });
    started_rx.await.expect("the hook parks");
    let closed =
        crate::harness::runtime::types::lane_error(SessionError::Message("closed".to_owned()));
    drive.close_gate(Arc::clone(&closed));
    let _ = release_tx.send(());

    let error = polling
        .await
        .expect("the poll joins")
        .expect_err("the closed pass rejects the admission");
    assert!(
        error.to_string().contains("closed"),
        "the closed rejection carried: {error}"
    );
    close_fixture(&fixture).await;
}

/// Upstream `performDeferredPoll` binding: the poll's close error replaces
/// the settled response, upstream's `finally` throw — the gated backend's
/// discarded frame write surfaces through the close.
#[tokio::test]
async fn a_discarded_frame_write_faults_the_poll_through_the_close() {
    let fixture = create_fixture(FixtureOptions {
        gated: true,
        ..fixture_options()
    })
    .await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    let gating = fixture.gating.clone().expect("fixture is not gated");
    let drive = install_drive(&fixture, None, Some(true));
    gating.arm();
    let lane = Arc::clone(&fixture.base.lane);
    let task_drive = Arc::clone(&drive);
    let polling =
        tokio::spawn(async move { run_deferred_suspended(&lane, &task_drive, suspended).await });

    gating.wait_pending(1).await.expect("the intent parks");
    gating.next(1).await.expect("the intent lands");
    // The lane's mutation line serializes commands, so the poll's first
    // frame write is the one commit that parks; the discard rejects it and
    // every later write outright.
    gating.wait_pending(1).await.expect("the frame parks");
    gating.discard();

    let joined = polling.await.expect("the poll joins");
    let error = joined.expect_err("the discarded frame faults the poll");
    assert!(
        error.to_string().contains("storage discarded"),
        "the discard carried the storage-discard text: {error}"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        1,
        "the poll fetched before the discard"
    );
    assert_eq!(
        current_run(&fixture).at(),
        "deferred.effect_pending",
        "the poll's intent stayed durable"
    );
    close_fixture(&fixture).await;
}

/// Upstream `runDeferred` binding: the effect-pending leaf routes to the
/// replacement poll, upstream's `recoverDeferredPoll` dispatch.
#[tokio::test]
async fn run_deferred_replaces_an_unknown_poll() {
    let fixture = create_fixture(fixture_options()).await;
    let suspended = submit_deferred(&fixture, default_deferred_response()).await;
    install_unknown_poll(&fixture, &suspended).await;
    let drive = install_drive(&fixture, None, Some(true));

    let result = run_deferred(&fixture.base.lane, &drive, current_deferred(&fixture))
        .await
        .expect("the replacement settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the replacement continues"
    );
    assert_eq!(
        fixture.base.faux.state().deferred_fetch_count(),
        1,
        "the replacement fetched once"
    );
    assert_eq!(
        current_run(&fixture).at(),
        "checkpoint",
        "the replacement run continues"
    );
    close_fixture(&fixture).await;
}
