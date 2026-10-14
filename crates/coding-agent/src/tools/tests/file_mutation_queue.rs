//! The mutation-queue half of upstream's `file-mutation-queue.test.ts` —
//! the built-in-tool describes ride the edit/write suites' shared file; the
//! queue's own serialization semantics bind here.

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::time::Duration;

use crate::tools::file_mutation_queue::with_file_mutation_queue;

/// The turn-based recorder the upstream suite's `order` array stands for.
#[derive(Default)]
struct Order {
    events: std::sync::Mutex<Vec<&'static str>>,
}

impl Order {
    fn push(&self, event: &'static str) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
    }

    fn recorded(&self) -> Vec<&'static str> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[tokio::test]
async fn serializes_operations_for_the_same_file() {
    let order = std::sync::Arc::new(Order::default());

    let first_order = std::sync::Arc::clone(&order);
    let second_order = std::sync::Arc::clone(&order);

    let first = tokio::spawn(with_file_mutation_queue(
        "/tmp/file-mutation-queue-same",
        async move {
            first_order.push("first:start");
            tokio::time::sleep(Duration::from_millis(30)).await;
            first_order.push("first:end");
        },
    ));
    let second = tokio::spawn(with_file_mutation_queue(
        "/tmp/file-mutation-queue-same",
        async move {
            second_order.push("second:start");
            second_order.push("second:end");
        },
    ));

    let (first, second) = tokio::join!(first, second);
    first.unwrap().unwrap();
    second.unwrap().unwrap();
    assert_eq!(
        order.recorded(),
        vec!["first:start", "first:end", "second:start", "second:end"]
    );
}

#[tokio::test]
async fn allows_different_files_to_proceed_in_parallel() {
    let order = std::sync::Arc::new(Order::default());
    // A's finish waits on B's start so the interleaving the upstream suite
    // times out with 30ms delays pins deterministically.
    let (b_started, mut b_started_rx) = tokio::sync::watch::channel(false);

    let order_a = std::sync::Arc::clone(&order);
    let order_b = std::sync::Arc::clone(&order);

    let (a, b) = tokio::join!(
        with_file_mutation_queue("/tmp/file-mutation-queue-a", async move {
            order_a.push("a:start");
            while !*b_started_rx.borrow() {
                if b_started_rx.changed().await.is_err() {
                    break;
                }
            }
            order_a.push("a:end");
        }),
        with_file_mutation_queue("/tmp/file-mutation-queue-b", async move {
            order_b.push("b:start");
            let _signalled = b_started.send(true);
            tokio::time::sleep(Duration::from_millis(30)).await;
            order_b.push("b:end");
        })
    );
    a.unwrap();
    b.unwrap();
    let recorded = order.recorded();
    let index_of = |event: &str| recorded.iter().position(|e| *e == event).unwrap();
    assert!(index_of("a:start") < index_of("a:end"));
    assert!(index_of("b:start") < index_of("b:end"));
    assert!(index_of("b:start") < index_of("a:end"));
}

#[tokio::test]
async fn uses_the_same_queue_for_symlink_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let target_path = dir.path().join("target.txt");
    let symlink_path = dir.path().join("alias.txt");
    std::fs::write(&target_path, "hello\n").unwrap();
    tokio::fs::symlink(&target_path, &symlink_path)
        .await
        .unwrap();

    let order = std::sync::Arc::new(Order::default());
    let order_target = std::sync::Arc::clone(&order);
    let order_alias = std::sync::Arc::clone(&order);
    let target_key = target_path.to_string_lossy().into_owned();
    let alias_key = symlink_path.to_string_lossy().into_owned();

    let (target, alias) = tokio::join!(
        with_file_mutation_queue(&target_key, async move {
            order_target.push("target:start");
            tokio::time::sleep(Duration::from_millis(30)).await;
            order_target.push("target:end");
        }),
        with_file_mutation_queue(&alias_key, async move {
            order_alias.push("alias:start");
            order_alias.push("alias:end");
        })
    );
    target.unwrap();
    alias.unwrap();
    assert_eq!(
        order.recorded(),
        vec!["target:start", "target:end", "alias:start", "alias:end"]
    );
}

#[tokio::test]
async fn a_released_queue_slot_wakes_the_next_caller() {
    // The release path: the first operation's completion hands the queue to
    // the second without an explicit open, upstream's chained promise.
    let order = std::sync::Arc::new(Order::default());
    let order_clone = std::sync::Arc::clone(&order);
    let order_second = std::sync::Arc::clone(&order);
    let (first, second) = tokio::join!(
        with_file_mutation_queue("/tmp/file-mutation-queue-chain", async {
            order_clone.push("first");
        }),
        with_file_mutation_queue("/tmp/file-mutation-queue-chain", async {
            order_second.push("second");
        })
    );
    first.unwrap();
    second.unwrap();
    assert_eq!(order.recorded(), vec!["first", "second"]);
}
