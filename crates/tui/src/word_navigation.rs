//! Word-boundary cursor navigation, ported from
//! `packages/tui/src/word-navigation.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! Restatements against upstream:
//!
//! - Text offsets are byte offsets into the Rust `str` rather than UTF-16
//!   code units; callers pass byte offsets and receive byte offsets.
//! - `Intl.Segmenter` word granularity becomes [`crate::utils::word_segments`]
//!   with [`crate::utils::is_word_like`]; a custom segmenter supplies
//!   [`SegmentData`] values directly.

use crate::utils::{is_punctuation_char, is_whitespace_char, is_word_like, word_segments};

impl std::fmt::Debug for WordNavigationOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WordNavigationOptions")
            .finish_non_exhaustive()
    }
}

/// One segment of a word-segmentation pass, upstream `Intl.SegmentData`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentData {
    /// The segment text, upstream `segment`.
    pub segment: String,
    /// Byte offset of the segment in the segmented text, upstream `index`.
    pub index: usize,
    /// Upstream `isWordLike`. `None` mirrors a producer that omits it —
    /// the editor's paste-marker-merged segments — and reads as not
    /// word-like.
    pub is_word_like: Option<bool>,
}

/// A custom word segmenter, upstream `WordNavigationOptions.segment`.
pub type SegmentFn<'a> = &'a dyn Fn(&str) -> Vec<SegmentData>;

/// An atomic-segment predicate, upstream `isAtomicSegment`.
pub type AtomicSegmentFn<'a> = &'a dyn Fn(&str) -> bool;

/// Options for word navigation, upstream `WordNavigationOptions`.
///
/// When `segment` is omitted, the default `Intl.Segmenter`-equivalent word
/// segmentation applies.
#[derive(Default)]
pub struct WordNavigationOptions<'a> {
    /// Custom segmenter returning word segments for the given text,
    /// upstream `segment`.
    pub segment: Option<SegmentFn<'a>>,
    /// Predicate identifying atomic segments that should be treated as
    /// single units (e.g. paste markers), upstream `isAtomicSegment`.
    pub is_atomic_segment: Option<AtomicSegmentFn<'a>>,
}

/// Whether a segment reads as whitespace, upstream testing the whole segment
/// with `isWhitespaceChar`'s `/\s/` any-position match.
fn segment_is_whitespace(segment: &str) -> bool {
    segment.chars().any(is_whitespace_char)
}

/// Whether a segment is atomic under the options, upstream `isAtomic?.(...)`.
fn is_atomic(options: Option<&WordNavigationOptions<'_>>, segment: &str) -> bool {
    options
        .and_then(|options| options.is_atomic_segment)
        .is_some_and(|is_atomic_segment| is_atomic_segment(segment))
}

/// Whether a segment reads as word-like, upstream `isWordLike`.
fn segment_is_word_like(segment: &SegmentData) -> bool {
    segment.is_word_like.unwrap_or(false)
}

/// Default word segmentation, upstream's module-global `wordSegmenter`.
fn default_word_segments(text: &str) -> Vec<SegmentData> {
    let mut index = 0;
    word_segments(text)
        .map(|segment| {
            let data = SegmentData {
                segment: segment.to_string(),
                index,
                is_word_like: Some(is_word_like(segment)),
            };
            index += segment.len();
            data
        })
        .collect()
}

/// Byte offset of the first character in the ASCII-punctuation set within
/// `segment`, upstream `PUNCTUATION_REGEX.exec(...).index` in
/// `find_word_forward`.
fn first_punctuation_offset(segment: &str) -> Option<usize> {
    segment
        .char_indices()
        .find(|(_, ch)| is_punctuation_char(*ch))
        .map(|(offset, _)| offset)
}

/// Byte offset just past the last punctuation character in `segment`,
/// upstream `lastMatch.index + lastMatch[0].length` in `find_word_backward`.
fn after_last_punctuation_offset(segment: &str) -> Option<usize> {
    segment
        .char_indices()
        .rfind(|(_, ch)| is_punctuation_char(*ch))
        .map(|(offset, ch)| offset + ch.len_utf8())
}

/// Find the cursor position after moving one word backward from `cursor` in
/// `text`. Skips trailing whitespace, then stops at the next
/// word/punctuation boundary.
///
/// Pure function — does not mutate any state.
#[must_use]
pub fn find_word_backward(
    text: &str,
    cursor: usize,
    options: Option<&WordNavigationOptions<'_>>,
) -> usize {
    if cursor == 0 {
        return 0;
    }
    let cursor = cursor.min(text.len());

    let text_before_cursor = &text[..cursor];
    let segment_fn = options.and_then(|options| options.segment);
    let mut segments: Vec<SegmentData> = segment_fn.map_or_else(
        || default_word_segments(text_before_cursor),
        |segment| segment(text_before_cursor),
    );
    let mut new_cursor = cursor;

    // Skip trailing whitespace
    while let Some(last) = segments.last() {
        if is_atomic(options, &last.segment) || !segment_is_whitespace(&last.segment) {
            break;
        }
        new_cursor = new_cursor.saturating_sub(last.segment.len());
        segments.pop();
    }

    if segments.is_empty() {
        return new_cursor;
    }

    let Some(last) = segments.last() else {
        return new_cursor;
    };

    if is_atomic(options, &last.segment) {
        // Skip one atomic segment.
        new_cursor -= last.segment.len();
    } else if segment_is_word_like(last) {
        // Skip inside one word-like segment, preserving ASCII punctuation
        // boundaries.
        let segment = &last.segment;
        new_cursor -= after_last_punctuation_offset(segment)
            .map_or(segment.len(), |after_last| segment.len() - after_last);
    } else {
        // Skip non-word non-whitespace run (punctuation)
        while let Some(last) = segments.last() {
            if is_atomic(options, &last.segment)
                || segment_is_word_like(last)
                || segment_is_whitespace(&last.segment)
            {
                break;
            }
            new_cursor = new_cursor.saturating_sub(last.segment.len());
            segments.pop();
        }
    }

    new_cursor
}

/// Find the cursor position after moving one word forward from `cursor` in
/// `text`. Skips leading whitespace, then stops at the next
/// word/punctuation boundary.
///
/// Pure function — does not mutate any state.
#[must_use]
pub fn find_word_forward(
    text: &str,
    cursor: usize,
    options: Option<&WordNavigationOptions<'_>>,
) -> usize {
    if cursor >= text.len() {
        return text.len();
    }

    let text_after_cursor = &text[cursor..];
    let segment_fn = options.and_then(|options| options.segment);
    let segments: Vec<SegmentData> = segment_fn.map_or_else(
        || default_word_segments(text_after_cursor),
        |segment| segment(text_after_cursor),
    );
    let mut position = 0;
    let mut new_cursor = cursor;

    // Skip leading whitespace
    while let Some(next) = segments.get(position) {
        if is_atomic(options, &next.segment) || !segment_is_whitespace(&next.segment) {
            break;
        }
        new_cursor += next.segment.len();
        position += 1;
    }

    let Some(next) = segments.get(position) else {
        return new_cursor;
    };

    if is_atomic(options, &next.segment) {
        // Skip one atomic segment.
        new_cursor += next.segment.len();
    } else if segment_is_word_like(next) {
        // Skip inside one word-like segment, preserving ASCII punctuation
        // boundaries.
        new_cursor += first_punctuation_offset(&next.segment).unwrap_or(next.segment.len());
    } else {
        // Skip non-word non-whitespace run (punctuation)
        while let Some(next) = segments.get(position) {
            if is_atomic(options, &next.segment)
                || segment_is_word_like(next)
                || segment_is_whitespace(&next.segment)
            {
                break;
            }
            new_cursor += next.segment.len();
            position += 1;
        }
    }

    new_cursor
}
