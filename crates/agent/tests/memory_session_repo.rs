//! The `MemorySessionRepo` suite, ported 1:1 from upstream
//! `test/harness/memory-session-repo.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod session_common;
use session_common::*;

use pi_agent_core::harness::context::background_context;
use std::sync::Arc;

use pi_agent_core::harness::session::memory::{MemorySessionRepo, MemorySessionRepoOptions};
use pi_agent_core::harness::session::types::SessionRepo;

/// The uuidv7 timestamp the generated id carries, upstream's
/// `uuidTimestamp(id)`.
fn uuid_timestamp(id: &str) -> u64 {
    u64::from_str_radix(&id.replace('-', "")[..12], 16).expect("uuid timestamp")
}

#[tokio::test]
async fn uses_its_injected_clock_for_generated_session_identity_and_metadata() {
    let repo = MemorySessionRepo::new(MemorySessionRepoOptions {
        now: Some(fixed_clock(NOW)),
    });
    let session = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions::default(),
            &background_context(),
        )
        .await
        .expect("create");

    assert_eq!(session.metadata().created_at, NOW);
    assert_eq!(uuid_timestamp(&session.metadata().id), NOW as u64);
    session
        .close(&background_context())
        .await
        .expect("close session");
    repo.close(&background_context()).await.expect("close repo");
}

#[tokio::test]
async fn returns_a_fresh_facade_after_close_while_retaining_one_session_and_storage() {
    let repo = MemorySessionRepo::new(MemorySessionRepoOptions {
        now: Some(fixed_clock(NOW)),
    });
    let first = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some("session".to_owned()),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("create");
    let first_branch = first
        .create_branch("main", None, &background_context())
        .await
        .expect("create branch");
    let admitted_write = first.set_name(Some("preserved".to_owned()), &background_context());

    let rejected = repo.open(first.metadata(), &background_context()).await;
    assert!(
        rejected
            .err()
            .expect("reopen while open")
            .to_string()
            .contains("already open"),
    );
    admitted_write.await.expect("admitted write");
    first.close(&background_context()).await.expect("close");
    let rejected = first.get_name(&background_context()).await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
    let rejected = first
        .scan_branch(
            &pi_agent_core::harness::session::types::StorageBranchScan {
                start: "entry".to_owned(),
                ..Default::default()
            },
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
    let rejected = first_branch.get_tip_id(&background_context()).await;
    assert!(
        rejected
            .expect_err("closed branch")
            .to_string()
            .contains("Session is closed"),
    );

    let second = repo
        .open(first.metadata(), &background_context())
        .await
        .expect("reopen");
    // Fresh facade: the reopened handle is a distinct value.
    assert_eq!(second.metadata().id, "session");
    assert_eq!(
        second.get_name(&background_context()).await.expect("name"),
        Some("preserved".to_owned()),
    );
    second.close(&background_context()).await.expect("close");
    repo.close(&background_context()).await.expect("close repo");
}

#[tokio::test]
async fn waits_for_an_explicit_mutation_before_closing_its_facade() {
    let repo = MemorySessionRepo::new(MemorySessionRepoOptions {
        now: Some(fixed_clock(NOW)),
    });
    let session = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some("session".to_owned()),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("create");
    let mutation = session
        .begin_mutation(&background_context())
        .await
        .expect("begin");
    let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let closed_flag = closed.clone();
    let metadata = session.metadata().clone();
    let closing = tokio::spawn(async move {
        session.close(&background_context()).await.expect("close");
        closed_flag.store(true, std::sync::atomic::Ordering::Release);
    });

    tokio::task::yield_now().await;
    assert!(
        !closed.load(std::sync::atomic::Ordering::Acquire),
        "close must not finish while the explicit mutation holds",
    );
    mutation.end(&background_context()).await.expect("end");
    closing.await.expect("closing task");
    assert!(closed.load(std::sync::atomic::Ordering::Acquire));

    let reopened = repo
        .open(&metadata, &background_context())
        .await
        .expect("reopen");
    reopened
        .close(&background_context())
        .await
        .expect("close reopened");
    repo.close(&background_context()).await.expect("close repo");
}

#[tokio::test]
async fn rejects_an_explicit_scope_that_had_not_acquired_before_facade_close() {
    let repo = MemorySessionRepo::new(MemorySessionRepoOptions {
        now: Some(fixed_clock(NOW)),
    });
    let session = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some("session".to_owned()),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("create");
    let first = session
        .begin_mutation(&background_context())
        .await
        .expect("begin");
    let metadata = session.metadata().clone();
    let queued = session.begin_mutation(&background_context());
    // The close call seals the line while the queued scope is still parked.
    let closing = session.close(&background_context());

    first.end(&background_context()).await.expect("end");
    let queued = queued.await;
    assert!(
        queued
            .err()
            .expect("queued scope")
            .to_string()
            .contains("Session is closed"),
    );
    closing.await.expect("close");

    let reopened = repo
        .open(&metadata, &background_context())
        .await
        .expect("reopen");
    reopened
        .close(&background_context())
        .await
        .expect("close reopened");
    repo.close(&background_context()).await.expect("close repo");
}
