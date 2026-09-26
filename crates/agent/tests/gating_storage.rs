//! The `GatingStorage` suite, ported 1:1 from upstream
//! `test/harness/gating-storage.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod session_common;
use session_common::memory_storage;
use session_common::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::testing::{GatingStorage, InstrumentedStorage};
use pi_agent_core::harness::session::types::{SessionError, Storage};
use pi_agent_core::harness::session::values::{
    self as stored_values, Write, set_value as set_value_write,
};

/// The FIFO test's delegate: upstream's `ControlledLandingStorage` —
/// every commit signals started, waits for the shared release latch, then
/// reaches the memory backend. The test builds it through the shared
/// [`HookedStorage`](session_common::HookedStorage) and keeps the watch
/// handles itself.
/// A name write, the suites' shorthand.
fn name_write(value: &str) -> Write {
    Write::ValueSet(
        set_value_write(&stored_values::session_name(), value.to_owned()).expect("write"),
    )
}

/// The name value the suites read back.
async fn stored_name(storage: &dyn Storage) -> serde_json::Value {
    storage
        .get_value(
            &stored_values::session_name().address,
            &background_context(),
        )
        .await
        .expect("value")
        .expect("stored")
        .value
}

#[tokio::test]
async fn bypasses_setup_writes_until_armed_and_waits_for_commits_parked_after_wait_pending() {
    let storage = Arc::new(GatingStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions {
            now: Some(fixed_clock(10)),
        },
    ))));
    storage
        .commit(vec![name_write("setup")], &background_context())
        .await
        .expect("setup");
    assert_eq!(storage.pending(), 0);

    storage.arm();
    let waiting_resolved = Arc::new(AtomicBool::new(false));
    let waiting_flag = waiting_resolved.clone();
    let waiting = {
        let storage = storage.clone();
        tokio::spawn(async move {
            storage.wait_pending(1).await.expect("wait pending");
            waiting_flag.store(true, Ordering::Release);
        })
    };
    tokio::task::yield_now().await;
    assert!(!waiting_resolved.load(Ordering::Acquire));
    let commit = {
        let storage = storage.clone();
        tokio::spawn(async move {
            storage
                .commit(vec![name_write("parked")], &background_context())
                .await
                .expect("parked commit")
        })
    };
    waiting.await.expect("waiting task");
    assert!(waiting_resolved.load(Ordering::Acquire));
    assert_eq!(storage.pending(), 1);
    assert_eq!(
        stored_name(storage.as_ref()).await,
        serde_json::json!("setup")
    );

    storage.next(1).await.expect("release");
    commit.await.expect("commit task");
    assert_eq!(storage.pending(), 0);
    assert_eq!(
        stored_name(storage.as_ref()).await,
        serde_json::json!("parked")
    );
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn releases_in_fifo_order_and_next_resolves_only_after_the_backend_write_lands() {
    let (started_tx, started_rx) = tokio::sync::watch::channel(false);
    let (released_tx, _released_rx) = tokio::sync::watch::channel(false);
    let delegate: Arc<dyn Storage> = Arc::new(HookedStorage {
        base: memory_storage(),
        commit_hook: Box::new({
            let started = started_tx.clone();
            let released = released_tx.clone();
            move |base: Arc<MemoryStorage>, writes: Vec<Write>, context| {
                let started = started.clone();
                let released = released.subscribe();
                Box::pin(async move {
                    let _ = started.send(true);
                    let mut released = released;
                    while !*released.borrow_and_update() {
                        released.changed().await.map_err(|_| {
                            SessionError::Message("release sender dropped".to_owned())
                        })?;
                    }
                    base.commit(writes, &context).await
                })
            }
        }),
    });
    let storage = Arc::new(GatingStorage::new(delegate.clone()));
    storage.arm();
    let first = {
        let storage = storage.clone();
        tokio::spawn(async move {
            storage
                .commit(vec![name_write("first")], &background_context())
                .await
                .expect("first commit")
        })
    };
    let second = {
        let storage = storage.clone();
        tokio::spawn(async move {
            storage
                .commit(vec![name_write("second")], &background_context())
                .await
                .expect("second commit")
        })
    };
    storage.wait_pending(2).await.expect("both parked");
    assert_eq!(storage.pending(), 2);

    let released = Arc::new(AtomicBool::new(false));
    let released_flag = released.clone();
    let next = {
        let storage = storage.clone();
        tokio::spawn(async move {
            storage.next(1).await.expect("release one");
            released_flag.store(true, Ordering::Release);
        })
    };
    let mut started_watch = started_rx.clone();
    started_watch
        .wait_for(|started| *started)
        .await
        .expect("started");
    assert!(*started_watch.borrow());
    tokio::task::yield_now().await;
    assert!(!released.load(Ordering::Acquire));
    assert_eq!(storage.pending(), 1);
    let _ = released_tx.send(true);
    next.await.expect("next task");
    assert!(released.load(Ordering::Acquire));
    first.await.expect("first commit task");
    assert_eq!(
        stored_name(storage.as_ref()).await,
        serde_json::json!("first")
    );

    storage.next(1).await.expect("release two");
    second.await.expect("second commit task");
    assert_eq!(
        stored_name(storage.as_ref()).await,
        serde_json::json!("second")
    );
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn permanently_rejects_parked_waiting_and_later_commits_after_discard() {
    let storage = Arc::new(GatingStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    storage.arm();
    let waiting = {
        let storage = storage.clone();
        tokio::spawn(async move { storage.wait_pending(2).await })
    };
    let parked = {
        let storage = storage.clone();
        tokio::spawn(async move {
            storage
                .commit(vec![name_write("lost")], &background_context())
                .await
        })
    };
    storage.wait_pending(1).await.expect("first parked");
    assert_eq!(storage.pending(), 1);

    storage.discard();
    assert_eq!(storage.pending(), 0);
    let parked = parked.await.expect("parked task");
    assert!(matches!(
        parked.expect_err("parked rejected"),
        SessionError::CommitDiscarded(_)
    ),);
    let waiting = waiting.await.expect("waiting task");
    assert!(matches!(
        waiting.expect_err("waiting rejected"),
        SessionError::CommitDiscarded(_)
    ),);
    let next = storage.next(1).await;
    assert!(matches!(
        next.expect_err("next rejected"),
        SessionError::CommitDiscarded(_)
    ));
    let later = storage.commit(Vec::new(), &background_context()).await;
    assert!(matches!(
        later.expect_err("later rejected"),
        SessionError::CommitDiscarded(_)
    ));
    storage.close(&background_context()).await.expect("close");

    let unarmed = GatingStorage::new(Arc::new(
        MemoryStorage::new(MemoryStorageOptions::default()),
    ));
    unarmed.discard();
    let later = unarmed.commit(Vec::new(), &background_context()).await;
    assert!(matches!(
        later.expect_err("unarmed rejected"),
        SessionError::CommitDiscarded(_)
    ));
    unarmed.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn records_attempts_before_gating_parks_them() {
    let gating = Arc::new(GatingStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let storage = Arc::new(InstrumentedStorage::new(gating.clone()));
    gating.arm();
    let writes = vec![name_write("recorded")];

    let commit = {
        let storage = storage.clone();
        let writes = writes.clone();
        tokio::spawn(async move {
            storage
                .commit(writes, &background_context())
                .await
                .expect("commit")
        })
    };
    // The spawned task's poll runs the record; yield once to reach it,
    // upstream's synchronous call.
    tokio::task::yield_now().await;
    assert_eq!(storage.get_commit_attempts(), vec![writes.clone()]);
    gating.wait_pending(1).await.expect("parked");
    assert_eq!(gating.pending(), 1);
    gating.next(1).await.expect("released");
    commit.await.expect("commit task");
    storage.close(&background_context()).await.expect("close");
}
