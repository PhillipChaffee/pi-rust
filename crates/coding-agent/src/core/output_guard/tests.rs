use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn takeover_flips_and_restores_idempotently() {
    restore_stdout();
    assert!(!stdout_is_taken_over());
    take_over_stdout();
    take_over_stdout();
    assert!(stdout_is_taken_over());
    restore_stdout();
    restore_stdout();
    assert!(!stdout_is_taken_over());
}

#[tokio::test]
async fn raw_writes_serialize_on_the_tail_and_flush_drains() {
    let writes = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&writes);
    let previous = set_raw_writer_for_test(Some(Arc::new(move |text: &str| {
        // The flush's empty chunk passes through the raw writer too,
        // upstream's `writeRawStdoutChunk("")`.
        assert!(text.is_empty() || text.contains("chunk"));
        if !text.is_empty() {
            counter.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    })));
    write_raw_stdout("chunk-a");
    write_raw_stdout("chunk-b");
    flush_raw_stdout().await;
    assert_eq!(writes.load(Ordering::SeqCst), 2);
    set_raw_writer_for_test(previous);
}

#[test]
fn stdout_write_delegates_to_the_taken_over_or_system_stream() {
    restore_stdout();
    write_stdout("plain text stdout\n");
    take_over_stdout();
    write_stdout("taken over text\n");
    restore_stdout();
    write_stdout("plain text again\n");
}

#[test]
fn the_backpressure_class_is_would_block_only() {
    restore_stdout();
    assert!(!is_transient_backpressure(&std::io::Error::other("no")));
    assert!(is_transient_backpressure(&std::io::Error::from(
        std::io::ErrorKind::WouldBlock
    )));
}

#[tokio::test]
async fn raw_writes_retry_transient_backpressure_and_queue_through_the_tail() {
    set_raw_writer_for_test(None);
    // The empty chunk returns before queueing anything.
    write_raw_stdout("");
    // One transient failure, then the write succeeds through the tail.
    let retried = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&retried);
    let previous = set_raw_writer_for_test(Some(Arc::new(move |text: &str| {
        if text.is_empty() {
            // The flush's empty chunk passes through but never counts.
            return Ok(());
        }
        counter.fetch_add(1, Ordering::SeqCst);
        if counter.load(Ordering::SeqCst) == 1 {
            Err(std::io::Error::from(std::io::ErrorKind::WouldBlock))
        } else {
            Ok(())
        }
    })));
    write_raw_stdout("chunk-retry");
    wait_for_raw_stdout_backpressure().await;
    flush_raw_stdout().await;
    assert_eq!(retried.load(Ordering::SeqCst), 2);
    write_raw_stdout(""); // the flush's empty chunk passes through too
    set_raw_writer_for_test(previous);
}
