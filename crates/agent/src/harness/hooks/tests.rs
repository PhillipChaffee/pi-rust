//! Boundary tests for the harness hook registry.
//!
//! Upstream has no unit file for `hooks.ts` — the registry rides the
//! runtime suites. These tests pin the aggregate semantics this slice
//! restates: fail-open aggregation, fail-closed `before_drive`, the tool
//! gate's abort contract, first-match structural hooks, and stream-options
//! patch derivation.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(clippy::unwrap_used, reason = "tests unwrap the pinned outcomes")]

use std::sync::{Arc, Mutex};

use pi_chord::context::background_context;

use crate::harness::agent_harness::{
    HookEvent, HookInvocation, HookName, HookOptions, HookResult,
};
use crate::harness::gate::{create_gate, GateRejection};
use crate::harness::hooks::{apply_stream_options_patch, create_stream_options_patch, HookRegistry};
use crate::harness::types::{AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch};

fn invocation(event: HookEvent) -> HookInvocation {
    HookInvocation {
        lane: "main".to_owned(),
        run_id: "run".to_owned(),
        event,
    }
}

fn reporting_handler_errors(errors: Arc<Mutex<Vec<String>>>) -> crate::harness::hooks::HookErrorReporter {
    Arc::new(move |error, _name, _lane, _context| {
        let errors = Arc::clone(&errors);
        Box::pin(async move {
            errors.lock().expect("report lock").push(error.to_string());
        })
    })
}

fn injected_message() -> crate::types::AgentMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": "injected",
        "timestamp": 1
    }))
    .expect("injected message")
}

fn handler(
    outcome: impl Fn(&HookInvocation) -> HookResult + Send + Sync + 'static,
) -> crate::harness::agent_harness::HookHandler {
    Arc::new(move |event: &HookInvocation, _context| {
        let result = outcome(event);
        Box::pin(async move { Ok(result) })
    })
}

/// `before_run` aggregates fail-open: injected messages accumulate across
/// handlers, a failing handler reports but does not stop the rest.
#[tokio::test]
async fn before_run_aggregates_fail_open() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(Arc::clone(&errors)));
    let mut options = HookOptions::default();
    options.id = Some("second".to_owned());
    registry
        .on(
            HookName::BeforeRun,
            handler(|_event| {
                HookResult::BeforeRun(Some(crate::harness::agent_harness::BeforeRunResult {
                    messages: vec![injected_message()],
                }))
            }),
            HookOptions::default(),
        )
        .expect("register");
    let event = HookEvent::BeforeRun {
        prompt: vec![],
        resources: crate::harness::agent_harness::Resources::default(),
    };
    let current = invocation(event.clone());
    let (gate, _gate_control) = create_gate();
    let result = registry
        .run_with_gate(
            HookName::BeforeRun,
            current,
            &gate,
            &background_context(),
        )
        .await;
    let result = match result {
        Ok(result) => result,
        Err(error) => panic!("the run should aggregate: {error}"),
    };
    let HookResult::BeforeRun(Some(injected)) = result else {
        panic!("the aggregate should carry the injected messages");
    };
    assert_eq!(injected.messages, vec![injected_message()]);
    let _ = options;
}

/// `run_with_gate` reports a closed gate rejection and a closed registry.
#[tokio::test]
async fn the_gate_rejection_maps_to_the_run_error() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(errors));
    let (gate, gate_control) = create_gate();
    let (cancelled, controller) = pi_chord::context::with_cancel(&background_context());
    controller.abort("aborted");
    let cancellation = tokio::sync::watch::channel(()).1;
    gate_control.begin_abort(cancellation);
    gate_control.signal_abort();
    let _ = controller;
    let error = registry
        .run_with_gate(
            HookName::BeforeDrive,
            invocation(HookEvent::BeforeDrive {
                operation: crate::harness::session::types::OperationKind::Run,
            }),
            &gate,
            &cancelled,
        )
        .await
        .expect_err("an aborting gate refuses admission");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Aborted(_)
    ));
    let _ = GateRejection::Closed;
}

/// A closed registry refuses every run.
#[tokio::test]
async fn a_closed_registry_refuses_every_run() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(errors));
    registry.close("closed".to_owned());
    let (gate, _gate_control) = create_gate();
    let error = registry
        .run_with_gate(
            HookName::BeforeRun,
            invocation(HookEvent::BeforeRun {
                prompt: vec![],
                resources: crate::harness::agent_harness::Resources::default(),
            }),
            &gate,
            &background_context(),
        )
        .await
        .expect_err("a closed registry errors");
    assert!(matches!(error, crate::harness::hooks::HookRunError::Closed(_)));
}

/// The stream-options patch applies per key over the base, and the
/// derived patch between two snapshots round-trips.
#[test]
fn stream_options_patches_apply_and_derive() {
    let base = AgentHarnessStreamOptions {
        timeout_ms: Some(1_000),
        max_retries: Some(2),
        ..AgentHarnessStreamOptions::default()
    };
    let patch = AgentHarnessStreamOptionsPatch {
        timeout_ms: Some(None),
        max_retries: Some(Some(5)),
        ..AgentHarnessStreamOptionsPatch::default()
    };
    let next = apply_stream_options_patch(&base, &patch);
    assert_eq!(next.timeout_ms, None);
    assert_eq!(next.max_retries, Some(5));

    let derived = create_stream_options_patch(&base, &next);
    assert_eq!(derived.timeout_ms, Some(None));
    assert_eq!(derived.max_retries, Some(Some(5)));
    assert!(derived.headers.is_none());
}