//! The `MutationLine` suite, ported 1:1 from upstream
//! `test/harness/mutation-line.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The upstream async functions
//! start eagerly; Rust futures start lazily, so each job is spawned.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_agent_core::harness::session::mutation_line::MutationLine;
use pi_agent_core::harness::session::types::SessionError;

/// The release gate upstream's `deferred()` restates.
fn deferred() -> (
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    tokio::sync::oneshot::channel()
}

#[tokio::test]
async fn serializes_every_session_mutation() {
    let line = Arc::new(MutationLine::new());
    let (gate_tx, gate_rx) = deferred();
    let order = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let order_first = order.clone();
    let first = {
        let line = line.clone();
        let order = order_first;
        tokio::spawn(async move {
            line.run(move || {
                let order = order.clone();
                async move {
                    order.lock().expect("order").push("first:start");
                    gate_rx.await.expect("gate");
                    order.lock().expect("order").push("first:end");
                    Ok("first")
                }
            })
            .await
        })
    };
    let second = {
        let line = line.clone();
        let order = order.clone();
        tokio::spawn(async move {
            line.run(move || {
                let order = order.clone();
                async move {
                    order.lock().expect("order").push("second");
                    Ok("second")
                }
            })
            .await
        })
    };

    tokio::task::yield_now().await;
    assert_eq!(*order.lock().expect("order"), ["first:start"]);
    let _ = gate_tx.send(());
    assert_eq!(first.await.expect("join").expect("first"), "first");
    assert_eq!(second.await.expect("join").expect("second"), "second");
    assert_eq!(
        *order.lock().expect("order"),
        ["first:start", "first:end", "second"]
    );
}

#[tokio::test]
async fn continues_after_a_failed_job_while_preserving_the_original_failure() {
    let line = MutationLine::new();
    let rejection = SessionError::Message("mutation failed".to_owned());

    let failed: Result<String, SessionError> = line
        .run(|| async { Err(SessionError::Message("mutation failed".to_owned())) })
        .await;
    assert_eq!(failed.expect_err("failed job"), rejection);
    let next = line
        .run(|| async { Ok("next".to_owned()) })
        .await
        .expect("next job");
    assert_eq!(next, "next");
}

#[tokio::test]
async fn seals_queued_and_future_jobs_while_draining_the_running_job() {
    let line = Arc::new(MutationLine::new());
    let (gate_tx, gate_rx) = deferred();
    let running = {
        let line = line.clone();
        tokio::spawn(async move {
            line.run(move || async move {
                gate_rx.await.expect("gate");
                Ok("running")
            })
            .await
        })
    };
    let queued = {
        let line = line.clone();
        tokio::spawn(async move { line.run(|| async { Ok("queued".to_owned()) }).await })
    };
    tokio::task::yield_now().await;
    let closed = SessionError::Message("closed".to_owned());

    let drained = line.seal(closed.clone());
    let late = line.run(|| async { Ok("late".to_owned()) }).await;
    assert_eq!(late.expect_err("late job"), closed);
    let _ = gate_tx.send(());
    assert_eq!(
        running.await.expect("join").expect("running job"),
        "running"
    );
    assert_eq!(queued.await.expect("join").expect_err("queued job"), closed);
    drained.await;
}
