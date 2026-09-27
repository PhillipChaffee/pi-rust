//! The runtime command vocabulary and drive-pass suite: the Debug surfaces
//! the shared vocabulary renders, [`DriveCompletion`]'s one-shot
//! settlement, and the drive pass's abort/close gate behavior, upstream's
//! `src/harness/runtime/types.ts` restated over the gate module.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; a violated expectation panics the test by design"
)]

use std::sync::Arc;

use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::gate::GateRejection;
use crate::harness::runtime::test_support::drive_pass;
use crate::harness::runtime::test_support::runtime_config;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::DriveCompletion;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::TerminalStatus;

fn settled_record() -> OperationResultRecord {
    OperationResultRecord {
        operation_id: "operation".to_owned(),
        kind: crate::harness::session::types::OperationKind::Run,
        status: TerminalStatus::Completed,
        error: None,
        from_tip_id: None,
        tip_id: None,
        started_at: 1,
        ended_at: 2,
    }
}

fn waiting_outcome() -> DriveOutcome {
    DriveOutcome::Waiting {
        operation_id: "operation".to_owned(),
        reason: DriveWaitReason::Retry { not_before: 10 },
    }
}

#[test]
fn the_configuration_debug_stays_summary_only() {
    let debug = format!("{:?}", runtime_config());
    assert!(debug.starts_with("Config {"), "the config's debug header");
    assert!(debug.contains("tools: 0"), "the tool count renders");
    assert!(debug.contains(".."), "the closures stay behind the summary");
    assert!(
        !debug.contains("to_provider_messages:"),
        "the boxed closures render only as the summary tail"
    );
}

#[test]
fn the_command_vocabulary_renders_its_debug_surfaces() {
    let commit: LaneCommand<()> = LaneCommand::Commit {
        writes: Vec::new(),
        next: crate::harness::runtime::types::LaneState {
            tip_id: None,
            configuration: crate::harness::runtime::test_support::lane_configuration(),
            inbox: Vec::new(),
            last_operation_id: None,
            operation: None,
        },
        materialize: Arc::new(|_commit| ()),
        events: None,
    };
    assert!(
        format!("{commit:?}").starts_with("Commit {"),
        "the commit header"
    );
    assert!(
        format!("{commit:?}").contains(".."),
        "the closures stay behind the summary"
    );

    let returned: LaneCommand<()> = LaneCommand::Return { result: () };
    assert_eq!(format!("{returned:?}"), "Return(..)");

    let rejected: LaneCommand<()> = LaneCommand::Reject {
        error: crate::harness::runtime::test_support::commit_failure("declined"),
    };
    assert!(
        format!("{rejected:?}").starts_with("Reject("),
        "the rejection carries its error: {rejected:?}",
    );

    let cancel = ContinueOperationResult::<()>::CancelRequested;
    assert_eq!(format!("{cancel:?}"), "CancelRequested");
    let result = ContinueOperationResult::Result { value: () };
    assert_eq!(format!("{result:?}"), "Result(..)");

    let commit: OperationCommand<()> = OperationCommand::Commit {
        writes: Vec::new(),
        operation_state: crate::harness::runtime::test_support::starting_run_state(),
        lane: None,
        materialize: Arc::new(|_commit| ()),
        events: None,
    };
    assert!(
        format!("{commit:?}").starts_with("Commit {"),
        "the operation commit header"
    );
    let finish: OperationCommand<()> = OperationCommand::Finish {
        writes: Vec::new(),
        record: settled_record(),
        lane: None,
        materialize: Arc::new(|_commit| ()),
        events: None,
    };
    assert!(
        format!("{finish:?}").starts_with("Finish {"),
        "the finish header"
    );
    let returned: OperationCommand<()> = OperationCommand::Return { result: () };
    assert_eq!(format!("{returned:?}"), "Return(..)");
}

#[test]
fn the_drive_completion_clones_and_renders_its_states() {
    let pending = DriveCompletion::Pending;
    assert_eq!(format!("{pending:?}"), "Pending");
    assert!(matches!(pending, DriveCompletion::Pending));

    let outcome = waiting_outcome();
    let settled = DriveCompletion::Settled(outcome.clone());
    match settled.clone() {
        DriveCompletion::Settled(carried) => assert_eq!(carried, outcome),
        other => panic!("the settled completion: {other:?}"),
    }
    assert!(
        format!("{settled:?}").starts_with("Settled("),
        "the settlement renders its outcome: {settled:?}",
    );

    let error: LaneError = crate::harness::runtime::test_support::commit_failure("boom");
    let failed = DriveCompletion::Failed(Arc::clone(&error));
    let cloned = failed.clone();
    match cloned {
        DriveCompletion::Failed(carried) => {
            assert!(
                Arc::ptr_eq(&carried, &error),
                "the failed completion clones the shared error",
            );
        }
        other => panic!("the failed completion: {other:?}"),
    }
    assert!(
        format!("{failed:?}").contains("boom"),
        "the failure renders its message"
    );
}

#[test]
fn the_drive_debug_names_its_pass() {
    let debug = format!("{:?}", drive_pass("operation"));
    assert!(
        debug.contains("operation_id: \"operation\""),
        "the pass names its operation"
    );
    assert!(
        debug.contains(".."),
        "the gate and completion stay behind the summary"
    );
}

#[tokio::test]
async fn begin_abort_gates_admission_and_carries_the_cancellation() {
    let pass = drive_pass("operation");
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(());
    pass.begin_abort(cancel_rx);
    pass.begin_abort(tokio::sync::watch::channel(()).1);

    let rejection = pass
        .gate
        .admit(|| ())
        .expect_err("the aborting gate refuses admission");
    match rejection {
        GateRejection::AbortRequested(mut abort) => {
            let _ = cancel_tx.send(());
            assert!(
                abort.cancellation.changed().await.is_ok(),
                "the carried cancellation settles with the abort work",
            );
        }
        GateRejection::Closed(error) => panic!("the gate closed: {error}"),
    }
}

#[test]
fn signal_abort_fires_only_once_aborting() {
    let pass = drive_pass("operation");
    pass.signal_abort();
    assert!(
        !pass.gate.signal().aborted(),
        "the signal without a recorded abort stays quiet",
    );

    let (_, cancel_rx) = tokio::sync::watch::channel(());
    pass.begin_abort(cancel_rx);
    pass.signal_abort();
    assert!(
        pass.gate.signal().aborted(),
        "the aborting gate fires its signal"
    );
}

#[tokio::test]
async fn close_gate_closes_once_and_fails_the_completion() {
    let pass = drive_pass("operation");
    let error: LaneError = crate::harness::runtime::test_support::commit_failure("closed");
    pass.close_gate(Arc::clone(&error));

    let rejection = pass
        .gate
        .admit(|| ())
        .expect_err("the closed gate refuses admission");
    match rejection {
        GateRejection::Closed(closed) => {
            assert_eq!(closed.0, "closed", "the gate carries the error");
        }
        GateRejection::AbortRequested(_) => panic!("the closed gate refuses with its error"),
    }
    assert!(
        pass.gate.signal().aborted(),
        "the close aborts the live signal"
    );
    assert!(pass.close_signal.aborted(), "the pass's close signal fires");
    let failed = pass.completion().await.expect_err("the failed completion");
    assert!(
        Arc::ptr_eq(&failed, &error),
        "the completion carries the close error"
    );

    pass.close_gate(crate::harness::runtime::test_support::commit_failure(
        "late",
    ));
    let still = pass
        .completion()
        .await
        .expect_err("the first settlement wins");
    assert!(
        Arc::ptr_eq(&still, &error),
        "the second close is a no-op on the completion",
    );
}

/// The one-shot settlement the completion watch carries, restated against
/// the vocabulary's `settle`/`fail` pair.
#[tokio::test]
async fn the_completion_settles_exactly_once() {
    let pass = drive_pass("operation");
    let outcome = waiting_outcome();
    pass.settle(outcome.clone());
    pass.fail(lane_error(SessionError::Message("late".to_owned())));
    assert_eq!(
        pass.completion().await.expect("the settlement wins"),
        outcome,
        "the later failure is a no-op",
    );
}
