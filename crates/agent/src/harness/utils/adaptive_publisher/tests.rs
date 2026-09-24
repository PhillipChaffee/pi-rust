//! The adaptive publisher suite, ported from upstream
//! `test/harness/adaptive-publisher.test.ts`.
//!
//! Upstream drives vi's fake timers; the port drives tokio's pausable
//! clock (`tokio::time::pause`), which the module doc pins as the
//! fake-timer substitute.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(clippy::unwrap_used, reason = "tests unwrap the pinned outcomes")]

use std::sync::{Arc, Mutex};

use crate::harness::utils::adaptive_publisher::{AdaptivePublisher, AdaptivePublisherOptions};

fn sink<T: Send + 'static>() -> (Arc<Mutex<Vec<T>>>, Arc<dyn Fn(T) + Send + Sync>) {
    let sink: Arc<Mutex<Vec<T>>> = Arc::new(Mutex::new(Vec::new()));
    let handle = Arc::clone(&sink);
    (
        sink,
        Arc::new(move |update: T| {
            handle.lock().expect("updates lock").push(update);
        }),
    )
}

/// The trailing timer is a spawned task on the paused clock: advancing the
/// clock wakes registered timers, and yields let the task poll before
/// assertions. The pump after each `mark_dirty` registers the freshly
/// armed timer's deadline before the clock advances.
async fn pump() {
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
}

/// The first dirty state publishes immediately; a large publication buys a
/// proportional delay; the minimum interval bounds event count and a
/// trailing timer guarantees the eventual publication.
#[tokio::test(start_paused = true)]
async fn bounds_event_count_and_spaces_large_publications() {
    let value: Arc<Mutex<String>> = Arc::new(Mutex::new("a".to_owned()));
    let (updates, publish) = sink::<String>();
    let publisher = AdaptivePublisher::new(AdaptivePublisherOptions {
        snapshot: {
            let value = Arc::clone(&value);
            Arc::new(move || value.lock().expect("value lock").clone())
        },
        update: Arc::new(|_previous, current| Some(current.clone())),
        measure: Arc::new(|update: &String| u64::try_from(update.len()).unwrap_or(u64::MAX)),
        publish,
        on_error: Arc::new(|error| panic!("the publisher errored: {error}")),
        min_interval_ms: Some(100),
        target_bytes_per_second: Some(100),
    });

    publisher.mark_dirty();
    pump().await;
    *value.lock().expect("value lock") = "x".repeat(100);
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    pump().await;
    assert_eq!(
        updates.lock().expect("updates lock").clone(),
        vec!["a".to_owned(), "x".repeat(100)]
    );

    *value.lock().expect("value lock") = "held".to_owned();
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(999)).await;
    pump().await;
    assert_eq!(updates.lock().expect("updates lock").len(), 2);
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    pump().await;
    assert_eq!(
        updates.lock().expect("updates lock").clone(),
        vec!["a".to_owned(), "x".repeat(100), "held".to_owned()]
    );
}

/// A consumer failure after the baseline commit keeps the commit; the
/// commit-before-delivery ordering means the next flush carries the
/// correct previous value, upstream's throw-after-apply contract.
#[tokio::test(start_paused = true)]
async fn commits_its_baseline_before_a_consumer_failure() {
    let value: Arc<Mutex<String>> = Arc::new(Mutex::new("a".to_owned()));
    let (updates, publish) = sink::<(Option<String>, String)>();
    let publisher = AdaptivePublisher::new(AdaptivePublisherOptions {
        snapshot: {
            let value = Arc::clone(&value);
            Arc::new(move || value.lock().expect("value lock").clone())
        },
        update: Arc::new(|previous: Option<&String>, current: &String| {
            Some((previous.cloned(), current.clone()))
        }),
        measure: Arc::new(|_: &(Option<String>, String)| 1),
        publish,
        on_error: Arc::new(|_error| {}),
        min_interval_ms: Some(100),
        target_bytes_per_second: Some(100),
    });

    publisher.mark_dirty();
    pump().await;
    *value.lock().expect("value lock") = "ab".to_owned();
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    pump().await;
    assert_eq!(
        updates.lock().expect("updates lock").clone(),
        vec![(None, "a".to_owned()), (Some("a".to_owned()), "ab".to_owned())]
    );

    *value.lock().expect("value lock") = "abc".to_owned();
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    pump().await;
    let published = updates.lock().expect("updates lock").clone();
    assert_eq!(published.last().map(|record| record.1.clone()), Some("abc".to_owned()));
}

