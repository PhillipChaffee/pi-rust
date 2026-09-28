//! Boundary tests binding the response-settlement branches upstream's
//! generation/deferred suites never reach, against the oracle
//! `src/harness/runtime/drive/response.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//!
//! - the classification ladder's failure rungs (aborted-while-running,
//!   overflow without recovery preparation, invalid deferred handle,
//!   tool-use without calls) and the error-stop fallback texts
//!   (`provider_error`'s and the retry wait's), upstream's `publishResponse`
//!   arms;
//! - the reserved-UUIDv7 timestamp invariant, upstream's `uuidV7Timestamp`;
//! - `publishConfigurationFailure`'s no-tip invariant and its
//!   cancel-before-planner downgrade, upstream's `continueOperation`
//!   contract;
//! - the stream lifecycle's dead-stream restatement: a first stream failure
//!   is held, later callbacks no-op, and the close capability raises it
//!   after the seal and drain (upstream's throw aborting the stream);
//! - the `after_response` binding's hook-failure surfaces, upstream's
//!   `runWithGate` rejection;
//! - the lifecycle's `Debug` rendering.
//!
//! The settled messages ride [`publish_response`] directly: the
//! classification ladder is the procedure itself, and the faux provider
//! cannot produce these responses on the wire without scripting a second
//! provider implementation.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the fixture helpers raise deliberately, upstream's thrown fixture errors"
)]

use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_tool_call};
use pi_ai::types::{AssistantMessage, Message, StopReason};
use pi_ai::utils::retry::RetryPolicy;

use crate::harness::agent_harness::{HarnessEventPayload, RunEndStatus};
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::background_context;
use crate::harness::execution::assistant::AssistantResponseMetadata;
use crate::harness::runtime::drive::response::AssistantResponseLifecycle;
use crate::harness::runtime::drive::response::ResponseIntent;
use crate::harness::runtime::drive::response::ResponseOptions;
use crate::harness::runtime::drive::response::open_assistant_response;
use crate::harness::runtime::drive::response::{publish_configuration_failure, publish_response};
use crate::harness::runtime::test_support::lock;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::CompactionEntryBody;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::Entry;
use crate::harness::session::types::GenerationContext;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationError;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::Session;
use crate::harness::session::types::{SettledAssistantMessage, SettledStopReason, TerminalStatus};
use crate::types::AgentMessage;

use super::common;

/// The fixture spec the response suite builds: the plain shared skeleton
/// with an unadmitted lane — the settled responses are published by hand,
/// upstream's direct `publishResponse` calls.
#[must_use]
fn response_spec() -> common::DriveFixtureSpec {
    common::DriveFixtureSpec {
        suite: "response",
        watch_suite: "response",
        faux: RegisterFauxProviderOptions::default(),
        retry_policy: RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 10,
            max_agent_delay_ms: None,
        },
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        backend: None,
        admit_prompt: false,
        system_prompt: None,
    }
}

/// The generation inputs the assistant leaves carry, the shared suite
/// fixtures' `generation` literal over the `{ maxAttempts: 2, baseDelayMs:
/// 10, maxAgentDelayMs: 30_000 }` policy.
#[must_use]
fn generation_context(fixture: &common::DriveFixture) -> GenerationContext {
    GenerationContext {
        step_id: "step".to_owned(),
        trigger_entry_id: "tip".to_owned(),
        configuration: fixture.configuration.clone(),
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        retry_policy: NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: 10,
            max_agent_delay_ms: 30_000,
        },
        overflow_recovery_used: false,
    }
}

/// The assistant effect-pending leaf one settlement runs under, upstream's
/// `AssistantEffectPendingOperation` fixture with the reserved ids passed
/// in.
#[must_use]
fn assistant_leaf(
    fixture: &common::DriveFixture,
    response_entry_id: String,
    usage_id: String,
) -> AssistantEffectPendingOperation {
    AssistantEffectPendingOperation {
        scope: common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        generation_context: generation_context(fixture),
        attempt: 1,
        response_entry_id,
        usage_id,
        intended_output_limit: 100,
        context_window: 1_000,
    }
}

/// The deferred effect-pending leaf one settlement runs under, upstream's
/// `DeferredEffectPendingOperation` fixture at poll 0.
#[must_use]
fn deferred_leaf(
    fixture: &common::DriveFixture,
    response_entry_id: String,
    usage_id: String,
) -> DeferredEffectPendingOperation {
    DeferredEffectPendingOperation {
        scope: DeferredScope {
            scope: common::run_scope(DEFAULT_COMPACTION_SETTINGS),
            step_id: "step".to_owned(),
            source_entry_id: "deferred-source".to_owned(),
            poll: 0,
            configuration: fixture.configuration.clone(),
            stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        },
        response_entry_id,
        usage_id,
    }
}

/// Installs the intent's leaf over the `tip` history so the settle runs
/// under it, the shared suites' `installOperation` call.
///
/// # Panics
/// The install's failure.
async fn install_intent(fixture: &common::DriveFixture, intent: &ResponseIntent) {
    let state = match intent {
        ResponseIntent::Assistant(leaf) => OperationState::AssistantEffectPending(leaf.clone()),
        ResponseIntent::Deferred(leaf) => OperationState::DeferredEffectPending(leaf.clone()),
    };
    common::install_operation(
        fixture,
        state,
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        common::InstallOptions {
            entries: vec![common::user_entry("tip", None, "history")],
            ..common::InstallOptions::default()
        },
    )
    .await;
    fixture.storage.clear_commit_attempts();
}

/// Publishes one settled response under its freshly installed leaf and
/// returns the failed record, the ladder's failure-rung shape.
///
/// # Panics
/// The install's or the settle's failure, or the settle continuing.
async fn settled_failure(
    fixture: &common::DriveFixture,
    intent: ResponseIntent,
    response: SettledAssistantMessage,
    options: ResponseOptions,
) -> OperationResultRecord {
    install_intent(fixture, &intent).await;
    let result = publish_response(&fixture.lane, &fixture.drive, intent, response, options)
        .await
        .expect("the response settles");
    let ProcedureResult::Settled { outcome } = result else {
        panic!("the failure settles the run: {result:?}")
    };
    outcome
}

/// Settles one assistant message, upstream's
/// `{ message, stopReason }` literal.
///
/// # Panics
/// A pending stop reason, which never settles.
fn settled(message: AssistantMessage) -> SettledAssistantMessage {
    let stop_reason = SettledStopReason::from_stop_reason(message.stop_reason)
        .unwrap_or_else(|| panic!("the fixture settles only non-pending stops"));
    SettledAssistantMessage {
        message,
        stop_reason,
    }
}

/// The reserved ids one fixture hands its intents, upstream's
/// `lane.session.idGenerator.next()` pair.
fn reserved_ids(fixture: &common::DriveFixture) -> (String, String) {
    (
        fixture.session.id_generator().next(None),
        fixture.session.id_generator().next(None),
    )
}

/// The committed response entry's assistant message, upstream's entry read
/// after the settle.
///
/// # Panics
/// The entry's absence or its not being an assistant message.
async fn committed_response(
    fixture: &common::DriveFixture,
    response_entry_id: &str,
) -> AssistantMessage {
    let entry = fixture
        .session
        .get_entry(response_entry_id, &background_context())
        .await
        .expect("the entry read")
        .unwrap_or_else(|| panic!("the response entry {response_entry_id} lands"));
    let Entry::Message { body, .. } = entry else {
        panic!("the response entry is a message")
    };
    let AgentMessage::Standard(Message::Assistant(message)) = body.message else {
        panic!("the response entry carries an assistant message")
    };
    message
}

/// Whether one published event is the failed `run_end` with the given
/// error text, upstream's `events.find((event) => event.type === "run_end")`
/// pin.
fn failed_run_end(
    events: &[crate::harness::agent_harness::HarnessEvent],
    error_text: &str,
) -> bool {
    events.iter().any(|event| {
        matches!(
            &event.payload,
            HarnessEventPayload::RunEnd {
                status: RunEndStatus::Failed { error },
                ..
            } if error.message == error_text
        )
    })
}

/// Upstream `publishResponse` arm: an aborted response under running
/// durable control is the invariant.
#[tokio::test]
async fn aborted_response_while_durable_control_is_running_is_the_invariant() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let intent = ResponseIntent::Assistant(assistant_leaf(&fixture, response_entry_id, usage_id));
    install_intent(&fixture, &intent).await;

    let error = publish_response(
        &fixture.lane,
        &fixture.drive,
        intent,
        settled(faux_assistant_message(
            "aborted",
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Aborted),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await
    .expect_err("the aborted response faults");
    assert!(
        error
            .to_string()
            .contains("Assistant response is aborted while durable control is running"),
        "the invariant carried: {error}"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the invariant commits nothing"
    );
    close_fixture(&fixture).await;
}

/// Upstream `publishResponse` arm: an overflow response the generation
/// already recovered from fails the run, and the normalized error's
/// fallback text rides the committed entry.
#[tokio::test]
async fn overflow_without_recovery_preparation_fails_the_run() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let mut leaf = assistant_leaf(&fixture, response_entry_id.clone(), usage_id);
    leaf.generation_context.overflow_recovery_used = true;
    let outcome = settled_failure(
        &fixture,
        ResponseIntent::Assistant(leaf),
        settled(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Length),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    let message = committed_response(&fixture, &response_entry_id).await;
    assert_eq!(
        message.stop_reason,
        StopReason::Error,
        "the response normalized to the error stop"
    );
    assert_eq!(
        message.error_message.as_deref(),
        Some("Assistant request exceeded the context window"),
        "the overflow fallback text rides the entry"
    );
    assert!(
        failed_run_end(
            &lock(&fixture.events),
            "Assistant request exceeded the context window"
        ),
        "the run end carries the overflow failure"
    );
    close_fixture(&fixture).await;
}

/// Upstream `publishResponse` arm: a deferred stop whose settled message
/// carries no handle fails the run with the invalid-handle text.
#[tokio::test]
async fn deferred_stop_without_a_handle_fails_the_run() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let leaf = assistant_leaf(&fixture, response_entry_id.clone(), usage_id);
    let outcome = settled_failure(
        &fixture,
        ResponseIntent::Assistant(leaf),
        settled(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Deferred),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    let message = committed_response(&fixture, &response_entry_id).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider returned an invalid deferred handle"),
        "the invalid-handle text rides the entry"
    );
    assert!(
        failed_run_end(
            &lock(&fixture.events),
            "Provider returned an invalid deferred handle"
        ),
        "the run end carries the invalid-handle failure"
    );
    close_fixture(&fixture).await;
}

/// Upstream `publishResponse` arm: a tool-use stop without any tool calls
/// fails the run with the no-calls text.
#[tokio::test]
async fn tool_use_without_any_tool_calls_fails_the_run() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let leaf = assistant_leaf(&fixture, response_entry_id.clone(), usage_id);
    let outcome = settled_failure(
        &fixture,
        ResponseIntent::Assistant(leaf),
        settled(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    let message = committed_response(&fixture, &response_entry_id).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider reported tool use without any tool calls"),
        "the no-calls text rides the entry"
    );
    close_fixture(&fixture).await;
}

/// Upstream `publishResponse` arm: an error stop past the attempt budget
/// fails the run, and the fallback template names the request and the stop
/// reason on both the record and the retry end.
#[tokio::test]
async fn error_stop_past_the_budget_fails_with_the_fallback_text() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let mut leaf = assistant_leaf(&fixture, response_entry_id, usage_id);
    leaf.attempt = 2;
    let outcome = settled_failure(
        &fixture,
        ResponseIntent::Assistant(leaf),
        settled(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Error),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    assert_eq!(
        outcome
            .error
            .expect("the failure carries its error")
            .message,
        "Assistant request ended with error",
        "the record carries the fallback text"
    );
    let retry_ended = lock(&fixture.events).iter().any(|event| {
        matches!(
            &event.payload,
            HarnessEventPayload::RetryEnd {
                attempt: 2,
                success: false,
                final_error: Some(final_error),
                ..
            } if final_error == "Assistant request ended with error"
        )
    });
    assert!(
        retry_ended,
        "the failed attempt's retry end carries the fallback"
    );
    close_fixture(&fixture).await;
}

/// Upstream `publishResponse` arm: a retryable error stop without an error
/// message schedules the wait under the fallback text.
#[tokio::test]
async fn error_stop_without_a_message_schedules_the_fallback_text() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let intent = ResponseIntent::Assistant(assistant_leaf(&fixture, response_entry_id, usage_id));
    install_intent(&fixture, &intent).await;

    let result = publish_response(
        &fixture.lane,
        &fixture.drive,
        intent,
        settled(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Error),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions { recovery: true },
    )
    .await
    .expect("the recovery settles");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the recovery schedules its wait: {result:?}"
    );
    let OperationState::AssistantRetryWait(retry) = common::current_state(&fixture) else {
        panic!("the recovery scheduled its wait");
    };
    assert_eq!(
        retry.retry_wait.error_message, "Assistant request failed",
        "the wait carries the fallback text"
    );
    assert_eq!(
        retry.retry_wait.next_attempt, 2,
        "the wait advances the attempt"
    );
    close_fixture(&fixture).await;
}

/// Upstream `publishResponse` arm: a deferred poll's error stop without an
/// error message fails with the deferred fallback template.
#[tokio::test]
async fn deferred_error_stop_fails_with_the_deferred_fallback_text() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let intent = ResponseIntent::Deferred(deferred_leaf(&fixture, response_entry_id, usage_id));
    let outcome = settled_failure(
        &fixture,
        intent,
        settled(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Error),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await;
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    assert_eq!(
        outcome
            .error
            .expect("the failure carries its error")
            .message,
        "Deferred request ended with error",
        "the deferred template names its request family"
    );
    close_fixture(&fixture).await;
}

/// Upstream `uuidV7Timestamp` invariant: a tool-call response whose
/// reserved entry id is not a UUIDv7 faults the settle before any write.
#[tokio::test]
async fn tool_calls_require_a_reserved_uuidv7_entry_id() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let usage_id = fixture.session.id_generator().next(None);
    let intent = ResponseIntent::Assistant(assistant_leaf(
        &fixture,
        "handwritten-id".to_owned(),
        usage_id,
    ));
    install_intent(&fixture, &intent).await;

    let error = publish_response(
        &fixture.lane,
        &fixture.drive,
        intent,
        settled(faux_assistant_message(
            faux_tool_call("lookup", serde_json::Map::new(), None),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await
    .expect_err("the malformed reserved id faults");
    assert!(
        error
            .to_string()
            .contains("Invalid reserved UUIDv7 handwritten-id"),
        "the timestamp invariant carried: {error}"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the malformed id commits nothing"
    );
    close_fixture(&fixture).await;
}

/// The ready leaf one configuration failure faults, upstream's
/// `AssistantReadyOperation` fixture; `tip_none` seeds the tip-less install
/// the invariant case drives.
///
/// # Panics
/// The install's failure.
async fn install_ready_leaf(
    fixture: &common::DriveFixture,
    tip_none: bool,
) -> AssistantReadyOperation {
    let ready = AssistantReadyOperation {
        scope: common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        generation_context: generation_context(fixture),
        next_attempt: 1,
    };
    common::install_operation(
        fixture,
        OperationState::AssistantReady(ready.clone()),
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        common::InstallOptions {
            entries: if tip_none {
                Vec::new()
            } else {
                vec![common::user_entry("tip", None, "history")]
            },
            tip_id: tip_none.then_some(None),
            ..common::InstallOptions::default()
        },
    )
    .await;
    ready
}

/// The request-configuration failure the two cases publish, upstream's
/// `configurationError` shape.
fn configuration_error() -> OperationError {
    OperationError {
        code: "model_unavailable".to_owned(),
        message: "The configured model is unavailable in this process".to_owned(),
        details: None,
    }
}

/// Upstream `publishConfigurationFailure` contract: cancellation requested
/// before the planner downgrades to a plain continue.
#[tokio::test]
async fn configuration_failure_downgrades_to_a_plain_continue_under_cancellation() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let ready = install_ready_leaf(&fixture, false).await;
    common::cancel_operation(&fixture).await;
    fixture.storage.clear_commit_attempts();

    let result = publish_configuration_failure(
        &fixture.lane,
        &fixture.drive,
        &OperationState::AssistantReady(ready),
        configuration_error(),
    )
    .await
    .expect("the cancelled failure serves");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled planner downgrades: {result:?}"
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the downgrade commits nothing"
    );
    close_fixture(&fixture).await;
}

/// Upstream `publishConfigurationFailure` invariant: a failed run without a
/// branch tip faults the planner.
#[tokio::test]
async fn failed_run_requires_a_branch_tip() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let ready = install_ready_leaf(&fixture, true).await;

    let error = publish_configuration_failure(
        &fixture.lane,
        &fixture.drive,
        &OperationState::AssistantReady(ready),
        configuration_error(),
    )
    .await
    .expect_err("the tip-less run faults");
    assert!(
        error.to_string().contains("Failed run has no Branch tip"),
        "the tip invariant carried: {error}"
    );
    close_fixture(&fixture).await;
}

/// Upstream `openAssistantResponse` binding: the lifecycle renders as an
/// opaque value.
#[tokio::test]
async fn the_lifecycle_renders_opaquely() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let lifecycle = open_assistant_response(&fixture.lane, &fixture.drive, "resp", false);
    assert_eq!(
        format!("{lifecycle:?}"),
        "AssistantResponseLifecycle(..)",
        "the lifecycle renders opaquely"
    );
    close_fixture(&fixture).await;
}

/// The no-op publications the dead-stream callbacks leave behind: only the
/// first start published, upstream's dead stream dropping later events.
///
/// # Panics
/// Any later publication.
fn assert_noop_publications(fixture: &common::DriveFixture) {
    let published: Vec<String> = lock(&fixture.events)
        .iter()
        .map(|event| event.event_type().as_str().to_owned())
        .collect();
    assert_eq!(
        published,
        ["message_start".to_owned()],
        "only the first start published"
    );
}

/// The close raise the held stream failure carries, upstream's `finally`
/// throw after the seal and drain.
///
/// # Panics
/// The close's success or the error text's mismatch.
async fn assert_close_raise(lifecycle: &AssistantResponseLifecycle, expected_error: &str) {
    let error = ((lifecycle.close)())
        .await
        .expect_err("the held failure raises at close");
    assert!(
        error.to_string().contains(expected_error),
        "the frame encoder's failure raised: {error}"
    );
}

/// The stream lifecycle's dead-stream restatement: the first held failure
/// makes later callbacks no-op and the close capability raises the failure
/// after the seal and drain, upstream's throw aborting the stream.
#[tokio::test]
async fn a_held_stream_failure_silences_later_callbacks_and_raises_at_close() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let lifecycle = open_assistant_response(&fixture.lane, &fixture.drive, "resp", false);
    let message = faux_assistant_message("partial", FauxAssistantMessageOptions::default());
    let start = pi_ai::types::AssistantMessageEvent::Start {
        partial: message.clone(),
    };
    // The first start publishes; the delta at an absent block index fails
    // the frame encoder and holds the stream's failure.
    lifecycle
        .observer
        .start(message.clone(), &start, &background_context())
        .await;
    let bad_delta = pi_ai::types::AssistantMessageEvent::TextDelta {
        content_index: 5,
        delta: "late".to_owned(),
        partial: message.clone(),
    };
    lifecycle
        .observer
        .update(message.clone(), &bad_delta, &background_context())
        .await;

    // Every later callback no-ops: a second update, a second start, and the
    // settlement all publish nothing.
    lifecycle
        .observer
        .update(message.clone(), &bad_delta, &background_context())
        .await;
    lifecycle
        .observer
        .start(message.clone(), &start, &background_context())
        .await;
    lifecycle
        .observer
        .end(&settled(message.clone()), &background_context())
        .await;
    assert_noop_publications(&fixture);
    assert_close_raise(&lifecycle, "Assistant message text block 5 has not started").await;
    close_fixture(&fixture).await;
}

/// Upstream `publishResponse` arm: an overflow response whose bounded path
/// ends at a compaction entry fails the run — the preparation comes back
/// empty and the settle takes the no-preparation failure, upstream's
/// `prepareOverflowCompaction` returning `undefined`.
#[tokio::test]
async fn overflow_with_a_compacted_tip_fails_the_run() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let (response_entry_id, usage_id) = reserved_ids(&fixture);
    let leaf = assistant_leaf(&fixture, response_entry_id.clone(), usage_id);
    common::install_operation(
        &fixture,
        OperationState::AssistantEffectPending(leaf.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions {
            entries: vec![NewEntry::Compaction {
                id: "summary".to_owned(),
                parent_id: None,
                body: CompactionEntryBody {
                    summary: "already summarized".to_owned(),
                    retained_tail: Vec::new(),
                    tokens_before: 1_000,
                    details: None,
                    usage: None,
                    from_hook: false,
                },
            }],
            tip_id: Some(Some("summary".to_owned())),
            ..common::InstallOptions::default()
        },
    )
    .await;
    fixture.storage.clear_commit_attempts();
    let result = publish_response(
        &fixture.lane,
        &fixture.drive,
        ResponseIntent::Assistant(leaf),
        settled(faux_assistant_message(
            Vec::<FauxContentBlock>::new(),
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::Length),
                ..FauxAssistantMessageOptions::default()
            },
        )),
        ResponseOptions::default(),
    )
    .await
    .expect("the overflow settles");
    let ProcedureResult::Settled { outcome } = result else {
        panic!("the overflow failure settles the run: {result:?}")
    };
    assert_eq!(outcome.status, TerminalStatus::Failed, "the run fails");
    assert!(
        failed_run_end(
            &lock(&fixture.events),
            "Assistant request exceeded the context window"
        ),
        "the run end carries the overflow failure"
    );
    close_fixture(&fixture).await;
}

/// The stream lifecycle's dead-stream restatement, start rung: a duplicate
/// start's frame-encoder failure is held, later callbacks no-op, and the
/// close capability raises it, upstream's throw aborting the stream.
#[tokio::test]
async fn a_duplicate_start_holds_the_encoder_failure_and_raises_at_close() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let lifecycle = open_assistant_response(&fixture.lane, &fixture.drive, "resp", false);
    let message = faux_assistant_message("partial", FauxAssistantMessageOptions::default());
    let start = pi_ai::types::AssistantMessageEvent::Start {
        partial: message.clone(),
    };
    lifecycle
        .observer
        .start(message.clone(), &start, &background_context())
        .await;
    // The duplicate start fails the frame encoder and holds the failure.
    lifecycle
        .observer
        .start(message.clone(), &start, &background_context())
        .await;
    // Every later callback no-ops.
    lifecycle
        .observer
        .end(&settled(message.clone()), &background_context())
        .await;
    assert_noop_publications(&fixture);
    assert_close_raise(
        &lifecycle,
        "Assistant message stream contains more than one start event",
    )
    .await;
    close_fixture(&fixture).await;
}

/// The settled message the after-response cases carry, upstream's default
/// `fauxAssistantMessage("done")` settlement.
fn settled_done() -> SettledAssistantMessage {
    settled(faux_assistant_message(
        "done",
        FauxAssistantMessageOptions::default(),
    ))
}

/// Upstream `afterResponse` binding: a closed hook registry faults the
/// settlement, upstream's rejected `runWithGate` promise.
#[tokio::test]
async fn a_closed_hook_registry_faults_the_after_response_binding() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let lifecycle = open_assistant_response(&fixture.lane, &fixture.drive, "resp", false);
    fixture.hooks.close("closed for the test".to_owned());

    let error = (lifecycle.after_response)(
        settled_done(),
        AssistantResponseMetadata::default(),
        background_context(),
    )
    .await
    .expect_err("the closed registry faults");
    assert!(
        error.to_string().contains("closed for the test"),
        "the closed registry's error carried: {error}"
    );
    close_fixture(&fixture).await;
}

/// Upstream `afterResponse` binding: an aborting pass faults the settlement
/// with the abort request, upstream's `AbortRequested` throw.
#[tokio::test]
async fn an_aborting_pass_faults_the_after_response_binding() {
    let fixture = common::create_drive_fixture(response_spec()).await;
    let lifecycle = open_assistant_response(&fixture.lane, &fixture.drive, "resp", false);
    // The abort begins without firing the signal, so the hook's admission
    // refuses while the gate's own signal stays silent, upstream's
    // `beginAbort` before the hook.
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    fixture.drive.begin_abort(cancellation);

    let error = (lifecycle.after_response)(
        settled_done(),
        AssistantResponseMetadata::default(),
        background_context(),
    )
    .await
    .expect_err("the aborting pass faults");
    assert_eq!(
        error.to_string(),
        "Abort requested",
        "the abort request carried: {error}"
    );
    close_fixture(&fixture).await;
}

/// Closes the fixture's session, upstream's `afterEach` close.
///
/// # Panics
/// The close's failure.
async fn close_fixture(fixture: &common::DriveFixture) {
    fixture
        .session
        .close(&background_context())
        .await
        .expect("the session closes");
}
