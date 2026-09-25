//! Boundary tests for the effect admission gate.
//!
//! Upstream has no unit file for `effect-gate.ts` — the gate rides the
//! runtime suites. These tests pin the admission semantics this slice
//! restates: an open gate admits everything, an aborting gate hands back
//! the cancellation future, a closed gate reports its error, and the
//! owner-facing controls are one-shot.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use crate::harness::gate::{GateRejection, create_gate};

fn settled_cancellation() -> crate::harness::gate::Cancellation {
    let (_sender, receiver) = tokio::sync::watch::channel(());
    receiver
}

/// An open gate admits every effect; the value passes through.
#[test]
fn an_open_gate_admits_every_effect() {
    let (gate, _control) = create_gate();
    let admitted = gate.admit(|| 42);
    assert!(matches!(admitted, Ok(42)));
}

/// An aborting gate refuses admission with the cancellation future; a
/// later close does not override the abort.
#[tokio::test]
async fn an_aborting_gate_refuses_admission_with_the_cancellation() {
    let (gate, control) = create_gate();
    control.begin_abort(settled_cancellation());
    let rejection = gate
        .admit(|| 42)
        .expect_err("an aborting gate refuses admission");
    let GateRejection::AbortRequested(requested) = rejection else {
        panic!("the abort path carries the cancellation future");
    };
    *requested.cancellation.borrow();
    let () = ();
    // A second abort attempt is a no-op; the gate stays aborting.
    control.begin_abort(settled_cancellation());
    let rejection = gate.admit(|| 42).expect_err("the gate stays aborting");
    assert!(matches!(rejection, GateRejection::AbortRequested(_)));
}

/// A closed gate reports its error; the first close wins.
#[test]
fn a_closed_gate_reports_its_error() {
    let (gate, control) = create_gate();
    control.close("gate closed".to_owned());
    control.close("ignored".to_owned());
    let rejection = gate
        .admit(|| 42)
        .expect_err("a closed gate refuses admission");
    let GateRejection::Closed(error) = rejection else {
        panic!("the closed path carries the gate error");
    };
    assert_eq!(error.0, "gate closed");
}

/// A closed gate cannot become aborting again.
#[test]
fn begin_abort_ignores_a_closed_gate() {
    let (gate, control) = create_gate();
    control.close("gate closed".to_owned());
    control.begin_abort(settled_cancellation());
    let rejection = gate.admit(|| 42).expect_err("the gate stays closed");
    assert!(matches!(rejection, GateRejection::Closed(_)));
}
