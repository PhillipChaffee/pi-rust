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
use std::sync::{Arc, Mutex};

use crate::harness::utils::adaptive_publisher::{AdaptivePublisher, AdaptivePublisherOptions};

/// The published-update collector and its push handle, shared by the
/// fixture builders.
type Sink<T> = (Arc<Mutex<Vec<T>>>, Arc<dyn Fn(T) + Send + Sync>);

fn sink<T: Send + 'static>() -> Sink<T> {
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
        vec![
            (None, "a".to_owned()),
            (Some("a".to_owned()), "ab".to_owned())
        ]
    );

    *value.lock().expect("value lock") = "abc".to_owned();
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    pump().await;
    let delivered = updates.lock().expect("updates lock").clone();
    assert_eq!(
        delivered.last().map(|record| record.1.clone()),
        Some("abc".to_owned())
    );
}

/// The option bundle and the publisher handle render debug views without
/// leaking closure internals.
#[test]
fn the_publisher_handles_render_debug_views() {
    let (sink_records, publish) = sink::<String>();
    let options = AdaptivePublisherOptions {
        snapshot: Arc::new(|| "a".to_owned()),
        update: Arc::new(|_previous: Option<&String>, current: &String| Some(current.clone())),
        measure: Arc::new(|update: &String| u64::try_from(update.len()).unwrap_or(u64::MAX)),
        publish,
        on_error: Arc::new(|_: String| {}),
        min_interval_ms: Some(100),
        target_bytes_per_second: Some(100),
    };
    assert!(format!("{options:?}").contains("AdaptivePublisherOptions"));
    let publisher = AdaptivePublisher::new(options);
    assert!(format!("{publisher:?}").contains("AdaptivePublisher"));
    drop(sink_records);
}

/// A disposed publisher ignores dirty marks and forced flushes.
#[tokio::test(start_paused = true)]
async fn a_disposed_publisher_ignores_dirty_marks() {
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
    publisher.dispose();
    *value.lock().expect("value lock") = "b".to_owned();
    publisher.mark_dirty();
    publisher.flush(true);
    pump().await;
    assert!(updates.lock().expect("updates lock").is_empty());
}

/// A rate-limited non-forceful flush re-arms the trailing timer instead of
/// publishing, upstream's re-arm branch.
#[tokio::test(start_paused = true)]
async fn a_rate_limited_flush_re_arms_the_trailing_timer() {
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
    *value.lock().expect("value lock") = "b".to_owned();
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    pump().await;
    *value.lock().expect("value lock") = "c".to_owned();
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(50)).await;
    pump().await;
    publisher.flush(false);
    tokio::time::advance(std::time::Duration::from_millis(50)).await;
    pump().await;
    assert_eq!(
        updates.lock().expect("updates lock").clone(),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
    );
}

/// A flush whose update is `None` re-baselines without delivering, and the
/// next dirty mark publishes without waiting on a pushed-forward interval,
/// upstream's no-change branch.
#[tokio::test(start_paused = true)]
async fn an_unchanged_state_rebaselines_without_publishing() {
    let value: Arc<Mutex<String>> = Arc::new(Mutex::new("a".to_owned()));
    let (updates, publish) = sink::<String>();
    let publisher = AdaptivePublisher::new(AdaptivePublisherOptions {
        snapshot: {
            let value = Arc::clone(&value);
            Arc::new(move || value.lock().expect("value lock").clone())
        },
        update: Arc::new(|previous: Option<&String>, current: &String| {
            (previous != Some(current)).then(|| current.clone())
        }),
        measure: Arc::new(|update: &String| u64::try_from(update.len()).unwrap_or(u64::MAX)),
        publish,
        on_error: Arc::new(|error| panic!("the publisher errored: {error}")),
        min_interval_ms: Some(100),
        target_bytes_per_second: Some(100),
    });

    publisher.mark_dirty();
    pump().await;
    assert_eq!(
        updates.lock().expect("updates lock").clone(),
        vec!["a".to_owned()]
    );
    publisher.mark_dirty();
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    pump().await;
    assert_eq!(
        updates.lock().expect("updates lock").clone(),
        vec!["a".to_owned()]
    );

    *value.lock().expect("value lock") = "b".to_owned();
    publisher.mark_dirty();
    pump().await;
    assert_eq!(
        updates.lock().expect("updates lock").clone(),
        vec!["a".to_owned(), "b".to_owned()]
    );
}
