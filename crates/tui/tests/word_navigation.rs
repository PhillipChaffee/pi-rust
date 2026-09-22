//! The word-navigation suite, ported 1:1 from
//! `packages/tui/test/word-navigation.test.ts` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! Restatements against upstream:
//!
//! - Text offsets are byte offsets into the Rust `str`; the CJK expectations
//!   restate the upstream UTF-16 offsets to byte offsets.
//! - Upstream's `Intl.Segmenter` applies locale dictionary segmentation for
//!   Han text (Node groups 你好 and 世界 as single word-like segments); the
//!   port's UAX #29 segmenter yields one segment per ideograph, so the CJK
//!   expectations walk the same boundaries one step at a time. The fullwidth
//!   comma stays a non-word punctuation segment in both.
use std::collections::HashMap;

use pi_tui::word_navigation::{
    SegmentData, WordNavigationOptions, find_word_backward, find_word_forward,
};

#[test]
fn basic_words_hello_world() {
    let text = "hello world";
    assert_eq!(find_word_backward(text, 11, None), 6);
    assert_eq!(find_word_backward(text, 6, None), 0);
}

#[test]
fn dotted_foo_bar() {
    let text = "foo.bar";
    assert_eq!(find_word_backward(text, 7, None), 4);
    assert_eq!(find_word_backward(text, 4, None), 3);
    assert_eq!(find_word_backward(text, 3, None), 0);
}

#[test]
fn colon_foo_bar() {
    let text = "foo:bar";
    assert_eq!(find_word_backward(text, 7, None), 4);
    assert_eq!(find_word_backward(text, 4, None), 3);
    assert_eq!(find_word_backward(text, 3, None), 0);
}

#[test]
fn path_to_file() {
    let text = "path/to/file";
    assert_eq!(find_word_backward(text, 12, None), 8);
    assert_eq!(find_word_backward(text, 8, None), 7);
    // "/to" is one word-like segment with "/" as punctuation boundary
    assert_eq!(find_word_backward(text, 7, None), 5);
    assert_eq!(find_word_backward(text, 5, None), 4);
    assert_eq!(find_word_backward(text, 4, None), 0);
}

#[test]
fn cjk_mixed_backward() {
    let text = "你好世界 test";
    // "test" starts at byte 13.
    assert_eq!(find_word_backward(text, text.len(), None), 13);
    // The UAX #29 segmenter yields one word-like segment per ideograph, so
    // each backward step crosses one character (byte 3 each): 13 -> 9 -> 6.
    assert_eq!(find_word_backward(text, 13, None), 9);
    assert_eq!(find_word_backward(text, 9, None), 6);
    assert_eq!(find_word_backward(text, 6, None), 3);
    assert_eq!(find_word_backward(text, 3, None), 0);
}

#[test]
fn whitespace_at_boundaries_backward() {
    let text = "  hello  ";
    assert_eq!(find_word_backward(text, 9, None), 2);
    assert_eq!(find_word_backward(text, 2, None), 0);
}

#[test]
fn punctuation_run_foo_bar_backward() {
    let text = "foo...bar";
    assert_eq!(find_word_backward(text, 9, None), 6);
    assert_eq!(find_word_backward(text, 6, None), 3);
    assert_eq!(find_word_backward(text, 3, None), 0);
}

#[test]
fn cursor_at_0_returns_0() {
    assert_eq!(find_word_backward("hello", 0, None), 0);
}

#[test]
fn basic_words_hello_world_forward() {
    let text = "hello world";
    assert_eq!(find_word_forward(text, 0, None), 5);
    assert_eq!(find_word_forward(text, 5, None), 11);
}

#[test]
fn dotted_foo_bar_forward() {
    let text = "foo.bar";
    assert_eq!(find_word_forward(text, 0, None), 3);
    assert_eq!(find_word_forward(text, 3, None), 4);
    assert_eq!(find_word_forward(text, 4, None), 7);
}

#[test]
fn colon_foo_bar_forward() {
    let text = "foo:bar";
    assert_eq!(find_word_forward(text, 0, None), 3);
    assert_eq!(find_word_forward(text, 3, None), 4);
    assert_eq!(find_word_forward(text, 4, None), 7);
}

#[test]
fn path_to_file_forward() {
    let text = "path/to/file";
    assert_eq!(find_word_forward(text, 0, None), 4);
    assert_eq!(find_word_forward(text, 4, None), 5);
    assert_eq!(find_word_forward(text, 5, None), 7);
    assert_eq!(find_word_forward(text, 7, None), 8);
    assert_eq!(find_word_forward(text, 8, None), 12);
}

#[test]
fn cjk_mixed_forward() {
    let text = "你好世界 test";
    let first_end = find_word_forward(text, 0, None);
    assert!(first_end > 0);
    assert!(first_end <= 3);
    // Walk to end
    let mut position = 0;
    while position < text.len() {
        let next = find_word_forward(text, position, None);
        if next == position {
            break;
        }
        position = next;
    }
    assert_eq!(position, text.len());
}

#[test]
fn whitespace_at_boundaries_forward() {
    let text = "  hello  ";
    assert_eq!(find_word_forward(text, 0, None), 7);
    assert_eq!(find_word_forward(text, 7, None), 9);
}

#[test]
fn punctuation_run_foo_bar_forward() {
    let text = "foo...bar";
    assert_eq!(find_word_forward(text, 0, None), 3);
    assert_eq!(find_word_forward(text, 3, None), 6);
    assert_eq!(find_word_forward(text, 6, None), 9);
}

#[test]
fn cursor_at_end_returns_end() {
    assert_eq!(find_word_forward("hello", 5, None), 5);
}

#[test]
fn atomic_segments_skip_as_one_unit() {
    let marker = "[paste #1 +5 lines]";
    let text = format!("hello {marker} world");
    let is_atomic = |segment: &str| segment == marker;

    // The functions slice text before calling segment(), so we map each
    // expected substring to its pre-split segments.
    let full: Vec<SegmentData> = vec![
        segment("hello", 0, true),
        segment(" ", 5, false),
        segment(marker, 6, true),
        segment(" ", 25, false),
        segment("world", 26, true),
    ];
    let mut segment_map: HashMap<String, Vec<SegmentData>> = HashMap::new();
    segment_map.insert(text.clone(), full.clone());
    // backward from end: slice(0, 31) = full text
    segment_map.insert(text.clone(), full);
    // backward from 26: "hello [paste #1 +5 lines] "
    segment_map.insert(
        text[..26].to_string(),
        vec![
            segment("hello", 0, true),
            segment(" ", 5, false),
            segment(marker, 6, true),
            segment(" ", 25, false),
        ],
    );
    // forward from 6: "[paste #1 +5 lines] world"
    segment_map.insert(
        text[6..].to_string(),
        vec![
            segment(marker, 0, true),
            segment(" ", 19, false),
            segment("world", 20, true),
        ],
    );

    let segment_fn = |input: &str| segment_map.get(input).cloned().unwrap_or_default();
    let options = WordNavigationOptions {
        segment: Some(&segment_fn),
        is_atomic_segment: Some(&is_atomic),
    };

    assert_eq!(find_word_backward(&text, text.len(), Some(&options)), 26);
    assert_eq!(find_word_backward(&text, 26, Some(&options)), 6);
    assert_eq!(
        find_word_forward(&text, 6, Some(&options)),
        6 + marker.len()
    );
}

/// Upstream's inline `Intl.SegmentData` object literals.
fn segment(text: &str, index: usize, is_word_like: bool) -> SegmentData {
    SegmentData {
        segment: text.to_string(),
        index,
        is_word_like: Some(is_word_like),
    }
}
