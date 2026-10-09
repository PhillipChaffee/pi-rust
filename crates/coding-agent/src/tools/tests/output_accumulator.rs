//! Boundary tests for the output accumulator, upstream's
//! `output-accumulator.ts` — the streaming decoder, the line/byte counters,
//! the temp-file preservation, and the finish flush.

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use crate::tools::output_accumulator::{OutputAccumulator, OutputAccumulatorOptions};

fn accumulator() -> OutputAccumulator {
    OutputAccumulator::new(OutputAccumulatorOptions::default())
}

#[test]
fn decodes_utf8_split_across_chunks() {
    let accumulator = accumulator();
    let euro = "€\n".as_bytes();
    accumulator.append(&euro[..1]);
    accumulator.append(&euro[1..]);
    accumulator.finish();
    let snapshot = accumulator.snapshot(false);
    assert_eq!(snapshot.content, "€\n");
    assert!(!snapshot.truncation.truncated);
    assert_eq!(snapshot.truncation.total_lines, 1);
    assert!(snapshot.full_output_path.is_none());
}

#[test]
fn flush_surfaces_an_incomplete_trailing_sequence() {
    let accumulator = accumulator();
    accumulator.append(&[0xe2, 0x82]); // incomplete €
    accumulator.finish();
    let snapshot = accumulator.snapshot(false);
    // The flushed U+FFFD counts its own UTF-8 length toward the budget.
    assert_eq!(snapshot.content, "\u{FFFD}");
    assert_eq!(snapshot.truncation.total_bytes, "\u{FFFD}".len());
}

#[test]
fn ill_formed_bytes_surface_one_replacement_per_subpart() {
    let accumulator = accumulator();
    accumulator.append(&[0xff, 0xff, b'a']);
    accumulator.finish();
    let snapshot = accumulator.snapshot(false);
    assert_eq!(snapshot.content, "\u{FFFD}\u{FFFD}a");
}

#[test]
fn counters_track_lines_and_bytes() {
    let accumulator = accumulator();
    accumulator.append(b"one\ntwo\nthr");
    accumulator.append(b"ee\n");
    accumulator.finish();
    let snapshot = accumulator.snapshot(false);
    assert_eq!(snapshot.content, "one\ntwo\nthree\n");
    assert_eq!(snapshot.truncation.total_lines, 3);
    assert_eq!(snapshot.truncation.total_bytes, 14);
    assert_eq!(accumulator.get_last_line_bytes(), 0);
}

#[test]
fn open_line_bytes_track_the_partial_tail() {
    let accumulator = accumulator();
    accumulator.append(b"one\ntwo");
    assert_eq!(accumulator.get_last_line_bytes(), 3);
    accumulator.append(b"ree");
    assert_eq!(accumulator.get_last_line_bytes(), 6);
    accumulator.append(b"\nfour");
    assert_eq!(accumulator.get_last_line_bytes(), 4);
}

#[test]
fn truncated_snapshot_overrides_with_full_counters() {
    let accumulator = OutputAccumulator::new(OutputAccumulatorOptions {
        max_lines: Some(3),
        ..OutputAccumulatorOptions::default()
    });
    accumulator.append(b"1\n2\n3\n4\n5\n6\n");
    accumulator.finish();
    let snapshot = accumulator.snapshot(true);
    assert!(snapshot.truncation.truncated);
    assert_eq!(snapshot.truncation.total_lines, 6);
    assert_eq!(snapshot.truncation.output_lines, 3);
    assert_eq!(snapshot.truncation.max_lines, 3);
    assert_eq!(
        snapshot.truncation.truncated_by,
        Some(crate::tools::truncate::TruncatedBy::Lines)
    );
    // The tail keeps the last lines, truncated from the head.
    assert!(snapshot.content.starts_with("4\n"));
    // The temp file was persisted and carries the raw bytes.
    let path = snapshot.full_output_path.expect("persisted");
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(raw, b"1\n2\n3\n4\n5\n6\n");
    let _ = std::fs::remove_file(path);
}

#[test]
fn byte_truncation_falls_back_to_the_byte_cause() {
    let accumulator = OutputAccumulator::new(OutputAccumulatorOptions {
        max_bytes: Some(8),
        ..OutputAccumulatorOptions::default()
    });
    accumulator.append(b"a-long-chunk\n");
    accumulator.finish();
    let snapshot = accumulator.snapshot(false);
    assert!(snapshot.truncation.truncated);
    assert_eq!(
        snapshot.truncation.truncated_by,
        Some(crate::tools::truncate::TruncatedBy::Bytes)
    );
    assert_eq!(snapshot.truncation.total_bytes, 13);
}

#[test]
fn append_after_finish_panics() {
    let accumulator = accumulator();
    accumulator.finish();
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| accumulator.append(b"x")));
    assert!(result.is_err());
}

#[test]
fn close_temp_file_flushes_the_pending_writes() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    rt.block_on(async {
        let accumulator = OutputAccumulator::new(OutputAccumulatorOptions {
            max_bytes: Some(4),
            ..OutputAccumulatorOptions::default()
        });
        accumulator.append(b"chunked-output\n");
        let snapshot = accumulator.snapshot(true);
        let path = snapshot.full_output_path.expect("persisted");
        accumulator.close_temp_file().await.unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw, b"chunked-output\n");
        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn close_without_a_temp_file_is_a_noop() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    rt.block_on(async {
        let accumulator = accumulator();
        accumulator.close_temp_file().await.unwrap();
    });
}
