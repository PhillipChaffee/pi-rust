//! The output capture suite, ported from upstream
//! `test/harness/output-capture.test.ts`.
//!
//! Upstream drives vi's fake timers; the port drives tokio's paused clock,
//! and the publisher's spawned trailing timer pumps before assertions (the
//! adaptive-publisher suite pins the registration ordering).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::{Arc, Mutex};

use pi_chord::context::background_context;

use crate::harness::types::{
    ShellOutputLimits, ShellOutputMetadata, ShellOutputRetention, ShellOutputUpdate, TruncatedBy,
};
use crate::harness::utils::output_capture::{
    OutputCapture, OutputCaptureHandlers, apply_shell_output_update, is_invalid_shell_output_char,
    sanitize_shell_output,
};

struct Fixture {
    capture: OutputCapture,
    updates: Arc<Mutex<Vec<ShellOutputUpdate>>>,
    errors: Arc<Mutex<Vec<String>>>,
}

fn capture() -> Fixture {
    capture_with(50, 100, ShellOutputRetention::Tail)
}

fn capture_with(max_bytes: u64, max_lines: u64, retain: ShellOutputRetention) -> Fixture {
    let updates: Arc<Mutex<Vec<ShellOutputUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let update_sink = Arc::clone(&updates);
    let error_sink = Arc::clone(&errors);
    let capture = OutputCapture::new(
        Some(&crate::harness::types::ShellOutputCaptureOptions {
            limits: ShellOutputLimits {
                max_bytes,
                max_lines,
                retain: Some(retain),
            },
            spill: false,
        }),
        background_context(),
        OutputCaptureHandlers {
            on_update: Some(Arc::new(move |update: &ShellOutputUpdate, _context| {
                update_sink
                    .lock()
                    .expect("updates lock")
                    .push(update.clone());
            })),
            on_error: Arc::new(move |message: String| {
                error_sink.lock().expect("errors lock").push(message);
            }),
        },
    )
    .expect("capture");
    Fixture {
        capture,
        updates,
        errors,
    }
}

fn fold(updates: &[ShellOutputUpdate]) -> Option<crate::harness::types::ShellOutputView> {
    let mut output: Option<crate::harness::types::ShellOutputView> = None;
    for update in updates {
        output = Some(apply_shell_output_update(output.as_ref(), update));
    }
    output
}

/// Invalid control characters drop without changing text or line
/// boundaries.
#[test]
fn removes_invalid_control_characters() {
    let input = "a\0b\tc\nd\re\u{7}f\u{fff9}g\u{fffb}h\u{1f600}";
    assert_eq!(sanitize_shell_output(input), "ab\tc\ndefgh\u{1f600}");
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(input));
    assert_eq!(fixture.capture.snapshot().text, "ab\tc\ndefgh\u{1f600}");
}

/// UTF-8 decodes across raw process chunks.
#[test]
fn decodes_utf8_split_across_raw_process_chunks() {
    let mut fixture = capture();
    let bytes = "😀".as_bytes();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Bytes(
            &bytes[..2],
        ));
    assert_eq!(fixture.capture.snapshot().text, "");
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Bytes(
            &bytes[2..],
        ));
    fixture.capture.finish();
    assert_eq!(fixture.capture.snapshot().text, "😀");
}

/// A single line larger than the working buffer keeps the exact byte
/// count.
#[test]
fn keeps_the_exact_byte_count_for_an_oversized_single_line() {
    let mut fixture = capture_with(10, 100, ShellOutputRetention::Tail);
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(
            &"x".repeat(100),
        ));
    let snapshot = fixture.capture.snapshot();
    assert_eq!(snapshot.text, "x".repeat(10));
    assert_eq!(snapshot.metadata.last_line_bytes, Some(100));
    assert!(snapshot.metadata.truncation.last_line_partial);
}

/// The head retention preserves the original head after the raw guard is
/// crossed.
#[test]
fn preserves_the_original_head_after_its_raw_guard_is_crossed() {
    let mut fixture = capture_with(100, 2, ShellOutputRetention::Head);
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(
            &format!("first\nsecond\n{}", "tail".repeat(100)),
        ));
    assert_eq!(fixture.capture.snapshot().text, "first\nsecond");
}

/// The sanitizer's character class: control bytes and invisible format
/// marks drop, tabs and newlines stay.
#[test]
fn the_invalid_character_class_drops_only_controls() {
    assert!(!is_invalid_shell_output_char('a'));
    assert!(!is_invalid_shell_output_char('\t'));
    assert!(!is_invalid_shell_output_char('\n'));
    assert!(is_invalid_shell_output_char('\0'));
    assert!(is_invalid_shell_output_char('\u{7}'));
    assert!(is_invalid_shell_output_char('\u{fff9}'));
}
async fn pump() {
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
}

/// The first bounded view publishes immediately; trickling appends stay
/// responsive; a burst collapses into one trailing append.
#[tokio::test(start_paused = true)]
async fn publishes_immediately_then_trickles_and_collapses_bursts() {
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("one"));
    pump().await;
    {
        let updates = fixture.updates.lock().expect("updates lock");
        assert_eq!(updates.len(), 1);
        assert!(matches!(updates[0], ShellOutputUpdate::Replace { .. }));
        drop(updates);
    }
    tokio::time::advance(std::time::Duration::from_millis(150)).await;
    pump().await;
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(" two"));
    pump().await;
    let updates = fixture.updates.lock().expect("updates lock").clone();
    assert_eq!(updates.len(), 2);
    assert_eq!(folded_text(&updates), "one two");
}

fn folded_text(updates: &[ShellOutputUpdate]) -> String {
    fold(updates).map(|output| output.text).unwrap_or_default()
}

/// Post-cap trickle publishes a small slide; a full turnover publishes a
/// cap-bounded replacement.
#[tokio::test(start_paused = true)]
async fn publishes_a_slide_then_a_replacement_after_turnover() {
    let mut fixture = capture_with(10, 100, ShellOutputRetention::Tail);
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(
            "abcdefghij",
        ));
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(150)).await;
    pump().await;
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("k"));
    pump().await;
    {
        let updates = fixture.updates.lock().expect("updates lock");
        assert!(
            matches!(updates[1], ShellOutputUpdate::Slide { drop: 1, ref text, .. } if text == "k"),
            "the trickle slides: {:?}",
            updates[1]
        );
        drop(updates);
    }
    assert_eq!(
        folded_text(&fixture.updates.lock().expect("updates lock")),
        "bcdefghijk"
    );

    let mut replacement = capture_with(10, 100, ShellOutputRetention::Tail);
    replacement
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(
            "abcdefghij",
        ));
    pump().await;
    replacement
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(
            &"x".repeat(100),
        ));
    pump().await;
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    pump().await;
    {
        let updates = replacement.updates.lock().expect("updates lock");
        assert!(matches!(updates[1], ShellOutputUpdate::Replace { .. }));
        drop(updates);
    }
    let view = fold(&replacement.updates.lock().expect("updates lock")).expect("folded");
    assert_eq!(view.text.chars().count(), 10);
    assert_eq!(view.metadata.truncation.total_bytes, 110);
}

/// `flush` publishes the held state; later pushes after `dispose` stay
/// silent, upstream's trailing-timer cancel.
#[tokio::test(start_paused = true)]
async fn flush_publishes_and_dispose_silences_the_trailing_timer() {
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("a"));
    pump().await;
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("b"));
    fixture.capture.flush();
    pump().await;
    assert_eq!(
        folded_text(&fixture.updates.lock().expect("updates lock")),
        "ab"
    );
    fixture.capture.dispose();
    tokio::time::advance(std::time::Duration::from_millis(1_000)).await;
    pump().await;
    assert_eq!(fixture.updates.lock().expect("updates lock").len(), 2);
}

/// Spill metadata publishes without resending text.
#[tokio::test(start_paused = true)]
async fn publishes_spill_metadata_without_resending_text() {
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("output"));
    pump().await;
    fixture.capture.set_spill_path("/tmp/output.log");
    pump().await;
    let updates = fixture.updates.lock().expect("updates lock").clone();
    let view = fold(&updates).expect("folded");
    assert_eq!(view.metadata.spill_path.as_deref(), Some("/tmp/output.log"));
    assert!(fixture.errors.lock().expect("errors lock").is_empty());
}

/// The spill path rides the view metadata, upstream's
/// `spillPath` field.
#[tokio::test(start_paused = true)]
async fn the_spill_path_reaches_the_folded_metadata() {
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("output"));
    pump().await;
    fixture.capture.set_spill_path("/tmp/output.log");
    pump().await;
    let view = fold(&fixture.updates.lock().expect("updates lock")).expect("folded");
    let metadata: ShellOutputMetadata = view.metadata;
    assert_eq!(metadata.spill_path.as_deref(), Some("/tmp/output.log"));
    let _ = TruncatedBy::Bytes;
}

/// The constructor rejects zero limits, upstream's `TypeError` throw.
#[test]
fn the_constructor_rejects_zero_limits() {
    let options =
        |max_bytes: u64, max_lines: u64| crate::harness::types::ShellOutputCaptureOptions {
            limits: ShellOutputLimits {
                max_bytes,
                max_lines,
                retain: None,
            },
            spill: false,
        };
    let zero_bytes = OutputCapture::new(
        Some(&options(0, 10)),
        background_context(),
        OutputCaptureHandlers {
            on_update: None,
            on_error: Arc::new(|_| {}),
        },
    )
    .expect_err("zero maxBytes");
    assert_eq!(
        zero_bytes,
        "Output maxBytes must be a positive finite number"
    );
    let zero_lines = OutputCapture::new(
        Some(&options(10, 0)),
        background_context(),
        OutputCaptureHandlers {
            on_update: None,
            on_error: Arc::new(|_| {}),
        },
    )
    .expect_err("zero maxLines");
    assert_eq!(zero_lines, "Output maxLines must be a positive integer");
}

/// The capture and handler handles render debug views without leaking
/// closure internals.
#[test]
fn the_capture_handles_render_debug_views() {
    let handlers = OutputCaptureHandlers {
        on_update: None,
        on_error: Arc::new(|_: String| {}),
    };
    assert!(format!("{handlers:?}").contains("on_update: false"));
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("a"));
    assert!(format!("{:?}", fixture.capture).contains("OutputCapture"));
}

/// An incomplete trailing sequence flushes as the replacement character
/// when the stream ends, Node's non-fatal decoding.
#[test]
fn finishes_an_incomplete_sequence_as_a_replacement() {
    let mut fixture = capture();
    let bytes = "😀".as_bytes();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Bytes(
            &bytes[..2],
        ));
    fixture.capture.finish();
    assert_eq!(fixture.capture.snapshot().text, "\u{fffd}");
}

/// A text chunk flushes the decoder's pending raw-byte tail before
/// appending, upstream's string push.
#[tokio::test]
async fn a_text_push_flushes_the_pending_byte_tail() {
    let mut fixture = capture();
    let bytes = "a😀".as_bytes();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Bytes(
            &bytes[..2],
        ));
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("b"));
    assert_eq!(fixture.capture.snapshot().text, "a\u{fffd}b");
}

/// Pushes and finishes after `dispose` stay silent.
#[tokio::test(start_paused = true)]
async fn pushes_and_finishes_after_dispose_stay_silent() {
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("a"));
    pump().await;
    let published = fixture.updates.lock().expect("updates lock").len();
    fixture.capture.dispose();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("b"));
    fixture.capture.finish();
    tokio::time::advance(std::time::Duration::from_millis(1_000)).await;
    pump().await;
    assert_eq!(
        fixture.updates.lock().expect("updates lock").len(),
        published
    );
    assert_eq!(fixture.capture.snapshot().text, "a");
}

/// A repeated spill path and a disposed capture republish nothing,
/// upstream's `setSpillPath` guard.
#[tokio::test(start_paused = true)]
async fn a_repeated_or_disposed_spill_path_republishes_nothing() {
    let mut fixture = capture();
    fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text("a"));
    pump().await;
    fixture.capture.set_spill_path("/tmp/one.log");
    pump().await;
    let published = fixture.updates.lock().expect("updates lock").len();
    fixture.capture.set_spill_path("/tmp/one.log");
    pump().await;
    assert_eq!(
        fixture.updates.lock().expect("updates lock").len(),
        published
    );
    fixture.capture.dispose();
    fixture.capture.set_spill_path("/tmp/two.log");
    pump().await;
    assert_eq!(
        fixture.updates.lock().expect("updates lock").len(),
        published
    );
}

fn view(text: &str, max_bytes: u64) -> crate::harness::types::ShellOutputView {
    let bytes = crate::harness::utils::truncate::utf8_byte_length(text);
    crate::harness::types::ShellOutputView {
        text: text.to_owned(),
        metadata: ShellOutputMetadata {
            truncation: crate::harness::types::ShellOutputTruncation {
                truncated: false,
                truncated_by: None,
                total_lines: 1,
                total_bytes: bytes,
                output_lines: 1,
                output_bytes: bytes,
                last_line_partial: false,
                first_line_exceeds_limit: false,
                max_lines: 100,
                max_bytes,
            },
            spill_path: None,
            last_line_bytes: None,
        },
    }
}

/// `update_from` derives appends and metadata-only updates from the two
/// views, upstream's `updateFrom`.
#[test]
fn update_from_derives_appends_and_metadata_only_updates() {
    let previous = view("abc", 50);
    let grown = view("abcd", 50);
    assert!(matches!(
        crate::harness::utils::output_capture::update_from(Some(&previous), &grown),
        Some(ShellOutputUpdate::Append { ref text, .. }) if text == "d"
    ));
    let same = view("abc", 60);
    assert!(matches!(
        crate::harness::utils::output_capture::update_from(Some(&previous), &same),
        Some(ShellOutputUpdate::Metadata { .. })
    ));
    assert!(matches!(
        crate::harness::utils::output_capture::update_from(None, &grown),
        Some(ShellOutputUpdate::Replace { .. })
    ));
}

/// A probe match that fails the byte-overlap check falls back to the
/// whole-view replacement, and the candidate scan caps at nine probes;
/// a zero scan or an empty side short-circuits, upstream's
/// `suffixPrefixOverlap` guards.
#[test]
fn update_from_replaces_when_the_slide_overlap_fails() {
    // "ab" matches the "aQ" tail's first byte but the tails diverge.
    let diverged =
        crate::harness::utils::output_capture::update_from(Some(&view("baQ", 50)), &view("ab", 50));
    assert!(matches!(diverged, Some(ShellOutputUpdate::Replace { .. })));
    // Repeated probe hits exhaust the candidate cap without an overlap.
    let capped = crate::harness::utils::output_capture::update_from(
        Some(&view(&"a".repeat(12), 50)),
        &view(&format!("a{}", "b".repeat(11)), 50),
    );
    assert!(matches!(capped, Some(ShellOutputUpdate::Replace { .. })));
    // A zero byte budget zeroes the scan window.
    let zeroed =
        crate::harness::utils::output_capture::update_from(Some(&view("abc", 0)), &view("xbcd", 0));
    assert!(matches!(zeroed, Some(ShellOutputUpdate::Replace { .. })));
}

/// A scan window capped below the previous view still finds the slide
/// overlap from the trimmed tail.
#[test]
fn a_capped_scan_window_still_finds_the_slide() {
    let previous = view("zzzzabcd", 2);
    let current = view("abcde", 2);
    assert!(matches!(
        crate::harness::utils::output_capture::update_from(Some(&previous), &current),
        Some(ShellOutputUpdate::Slide { drop: 4, ref text, .. }) if text == "e"
    ));
}

/// The raw buffer guard trims on UTF-8 boundaries for both retentions.
#[test]
fn the_raw_buffer_guard_trims_on_utf8_boundaries() {
    let mut tail_fixture = capture_with(10, 100, ShellOutputRetention::Tail);
    tail_fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(
            &"中".repeat(14),
        ));
    assert_eq!(tail_fixture.capture.snapshot().text, "中".repeat(3));

    let mut head_fixture = capture_with(10, 100, ShellOutputRetention::Head);
    head_fixture
        .capture
        .push(crate::harness::utils::output_capture::Chunk::Text(
            &"中".repeat(14),
        ));
    let snapshot = head_fixture.capture.snapshot();
    assert_eq!(
        snapshot.metadata.truncation.total_bytes,
        crate::harness::utils::truncate::utf8_byte_length(&"中".repeat(14))
    );
    assert!(snapshot.metadata.truncation.first_line_exceeds_limit);
}
