//! Editor component, ported from `packages/tui/src/components/editor.ts`
//! in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! The multi-line text editor: word-wrapped layout with scroll indicators,
//! grapheme-wise cursor movement over wrapped lines, paste markers that
//! behave as atomic units, prompt history, kill ring, undo, page scroll,
//! sticky-column vertical movement, and character jumps.
//!
//! Restatements against upstream:
//!
//! - Text offsets are byte offsets into the Rust `String` rather than UTF-16
//!   code units; cursors, chunk boundaries, and paste-marker spans are byte
//!   offsets end to end.
//! - The autocomplete machinery — provider hooks, trigger/debounce
//!   patterns, the request pipeline, and the `SelectList` dropdown —
//!   integrates [`crate::autocomplete::AutocompleteProvider`] (#49).
//!   Restatements: the request pipeline runs on one worker thread per
//!   editor (upstream's `setTimeout` debounce + `autocompleteRequestTask`
//!   serialization), whose results the host drains through
//!   [`Editor::drain_autocomplete`] / [`Editor::autocomplete_idle`] —
//!   upstream resolves them on the event loop before the next paint;
//!   `AbortSignal` becomes a `CancellationToken` per the stack decision;
//!   `this.autocompleteAbort` becomes the worker's active-token slot; and
//!   `onSelect` on the dropdown list is applied by the editor after the
//!   mouse dispatch, because the callback would need a handle on the
//!   non-`Rc` editor.
//! - `Intl.Segmenter` becomes [`crate::utils::grapheme_segments`] /
//!   [`crate::utils::word_segments`] wrapped by
//!   [`crate::word_navigation::SegmentData`]; the editor's paste-marker
//!   merging produces those [`SegmentData`] values directly.
//! - The vertical-move snap test `seg.segment.length <= 1` reads
//!   "single code unit" in UTF-16 terms; the port tests for a single
//!   grapheme instead, which is the documented intent ("single-grapheme
//!   segments don't need snapping") and keeps combining-mark text from
//!   snapping to itself.
//! - `insertCharacter`'s `skipUndoCoalescing` parameter has no caller
//!   upstream and is dropped.
//! - `String.replace` with a string argument replaces only the first
//!   occurrence; the port uses the same single replacement.
//! - Paste-marker renumbering maps an absent suffix group to the empty
//!   string; upstream's template literal would emit the text `undefined`
//!   for a hand-typed suffix-less marker, a latent bug the port does not
//!   reproduce.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::LazyLock;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use regex::{Captures, Regex};
use tokio_util::sync::CancellationToken;

use crate::autocomplete::{
    ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS, AutocompleteItem, AutocompleteProvider,
    AutocompleteQueryOptions, AutocompleteSuggestions, DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS,
    build_debounce_pattern, build_trigger_pattern,
};
use crate::components::select_list::{
    SelectItem, SelectList, SelectListLayoutOptions, SelectListTheme,
};
use crate::keybindings::get_keybindings;
use crate::keys::{KeyParser, decode_printable_key};
use crate::kill_ring::{KillRing, KillRingPushOptions};
use crate::tui::{
    CURSOR_MARKER, Component, Focusable, Tui, TuiMouseButton, TuiMouseEvent, TuiMouseEventResult,
    TuiMouseEventType,
};
use crate::undo_stack::UndoStack;
use crate::utils::{
    grapheme_segments, is_whitespace_char, is_word_like, slice_by_column, static_regex,
    visible_width, word_segments,
};
use crate::word_navigation::{
    SegmentData, WordNavigationOptions, find_word_backward, find_word_forward,
};

/// Regex matching paste markers like `[paste #1 +123 lines]` or
/// `[paste #2 1234 chars]`, upstream `PASTE_MARKER_REGEX`.
static PASTE_MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]"));

/// Non-global version for single-segment testing, upstream
/// `PASTE_MARKER_SINGLE`.
static PASTE_MARKER_SINGLE_RE: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"^\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]\z"));

/// Terminals re-encoding control bytes inside bracketed paste as CSI-u
/// Ctrl+<letter>, decoded by [`Editor::handle_paste`].
static CSI_U_CTRL_RE: LazyLock<Regex> = LazyLock::new(|| static_regex(r"\x1b\[(\d+);5u"));

/// Check if a segment is a paste marker (i.e. was merged by
/// [`segment_with_markers`]), upstream `isPasteMarker`.
fn is_paste_marker(segment: &str) -> bool {
    segment.len() >= 10 && PASTE_MARKER_SINGLE_RE.is_match(segment)
}

/// Whether a segment reads as whitespace, upstream testing the whole segment
/// with `isWhitespaceChar`'s `/\s/` any-position match.
fn segment_is_whitespace(segment: &str) -> bool {
    segment.chars().any(is_whitespace_char)
}

/// Which segmenter a [`SegmentData`] pass runs in, upstream
/// `Editor.segment`'s `"word" | "grapheme"` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentMode {
    Word,
    Grapheme,
}

/// A segmenter that wraps the base segmenters and merges graphemes that
/// fall within paste markers into single atomic segments. This makes cursor
/// movement, deletion, word-wrap, etc. treat paste markers as single units.
///
/// Only markers whose numeric ID exists in `valid_ids` are merged, upstream
/// `segmentWithMarkers`.
fn segment_with_markers(
    text: &str,
    mode: SegmentMode,
    valid_ids: &HashSet<u32>,
) -> Vec<SegmentData> {
    // Fast path: no paste markers in the text or no valid IDs.
    if valid_ids.is_empty() || !text.contains("[paste #") {
        return base_segments(text, mode);
    }

    // Find all marker spans with valid IDs.
    let mut markers: Vec<(usize, usize)> = Vec::new();
    for caps in PASTE_MARKER_RE.captures_iter(text) {
        let Some(whole) = caps.get(0) else {
            continue;
        };
        let id: u32 = match caps[1].parse() {
            Ok(id) => id,
            Err(_) => continue,
        };
        if !valid_ids.contains(&id) {
            continue;
        }
        markers.push((whole.start(), whole.end()));
    }
    if markers.is_empty() {
        return base_segments(text, mode);
    }

    // Build merged segment list.
    let base = base_segments(text, mode);
    let mut result: Vec<SegmentData> = Vec::new();
    let mut marker_index = 0;

    for seg in base {
        // Skip past markers that are entirely before this segment.
        while markers
            .get(marker_index)
            .is_some_and(|(_, end)| *end <= seg.index)
        {
            marker_index += 1;
        }

        let marker = markers.get(marker_index).copied();

        if marker.is_some_and(|(start, end)| seg.index >= start && seg.index < end) {
            // This segment falls inside a marker: emit one merged segment at
            // its first base segment, skip the rest.
            if let Some((start, end)) = marker
                && seg.index == start
            {
                result.push(SegmentData {
                    segment: text[start..end].to_string(),
                    index: start,
                    is_word_like: None,
                });
            }
        } else {
            result.push(seg);
        }
    }

    result
}

/// The base segmentation the marker merge wraps, upstream's
/// `wordSegmenter` / `graphemeSegmenter` segments.
fn base_segments(text: &str, mode: SegmentMode) -> Vec<SegmentData> {
    let mut index = 0;
    match mode {
        SegmentMode::Word => word_segments(text)
            .map(|segment| {
                let data = SegmentData {
                    segment: segment.to_string(),
                    index,
                    is_word_like: Some(is_word_like(segment)),
                };
                index += segment.len();
                data
            })
            .collect(),
        SegmentMode::Grapheme => grapheme_segments(text)
            .map(|segment| {
                let data = SegmentData {
                    segment: segment.to_string(),
                    index,
                    is_word_like: None,
                };
                index += segment.len();
                data
            })
            .collect(),
    }
}

/// A chunk of text for word-wrap layout, upstream `TextChunk`. Tracks both
/// the text content and its position in the original line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChunk {
    /// The chunk text.
    pub text: String,
    /// Byte offset of the chunk start in the wrapped line.
    pub start_index: usize,
    /// Byte offset just past the chunk end in the wrapped line.
    pub end_index: usize,
}

/// Split a line into word-wrapped chunks, upstream `wordWrapLine`.
/// Wraps at word boundaries when possible, falling back to character-level
/// wrapping for words longer than the available width.
///
/// `pre_segmented` carries graphemes segmented with paste-marker awareness;
/// when omitted the default segmenter applies.
/// # Panics
///
/// Panics when a `pre_segmented` entry carries byte offsets that fall
/// outside `line` — the wrap arithmetic indexes `line` with them. Callers
/// derive the segments from the same line, which keeps every span inside.
#[must_use]
pub fn word_wrap_line(
    line: &str,
    max_width: usize,
    pre_segmented: Option<&[SegmentData]>,
) -> Vec<TextChunk> {
    if line.is_empty() || max_width == 0 {
        return vec![TextChunk {
            text: String::new(),
            start_index: 0,
            end_index: 0,
        }];
    }

    let line_width = visible_width(line);
    if line_width <= max_width {
        return vec![TextChunk {
            text: line.to_string(),
            start_index: 0,
            end_index: line.len(),
        }];
    }

    let mut chunks: Vec<TextChunk> = Vec::new();
    let segments: Vec<SegmentData> =
        pre_segmented.map_or_else(|| base_segments(line, SegmentMode::Grapheme), Vec::from);

    let mut current_width = 0;
    let mut chunk_start = 0;

    // Wrap opportunity: the position after the last whitespace before a
    // non-whitespace grapheme, i.e. where a line break is allowed.
    let mut wrap_opp_index: Option<usize> = None;
    let mut wrap_opp_width = 0;

    for (position, seg) in segments.iter().enumerate() {
        let grapheme = &seg.segment;
        let g_width = visible_width(grapheme);
        let char_index = seg.index;
        let is_ws = !is_paste_marker(grapheme) && segment_is_whitespace(grapheme);

        // Overflow check before advancing.
        if current_width + g_width > max_width {
            let backtrack_opp = match wrap_opp_index {
                Some(wrap_opp) if current_width - wrap_opp_width + g_width <= max_width => {
                    Some(wrap_opp)
                }
                _ => None,
            };
            if let Some(wrap_opp) = backtrack_opp {
                // Backtrack to last wrap opportunity (the remaining content
                // plus the current grapheme still fits within maxWidth).
                chunks.push(TextChunk {
                    text: line[chunk_start..wrap_opp].to_string(),
                    start_index: chunk_start,
                    end_index: wrap_opp,
                });
                chunk_start = wrap_opp;
                current_width -= wrap_opp_width;
            } else if chunk_start < char_index {
                // No viable wrap opportunity: force-break at current
                // position. This also handles the case where backtracking to
                // a word boundary wouldn't help because the remaining content
                // plus the current grapheme (e.g. a wide character) still
                // exceeds maxWidth.
                chunks.push(TextChunk {
                    text: line[chunk_start..char_index].to_string(),
                    start_index: chunk_start,
                    end_index: char_index,
                });
                chunk_start = char_index;
                current_width = 0;
            }
            wrap_opp_index = None;
        }

        if g_width > max_width {
            // Single atomic segment wider than maxWidth (e.g. paste marker
            // in a narrow terminal). Re-wrap it at grapheme granularity.
            //
            // The segment remains logically atomic for cursor movement /
            // editing — the split is purely visual for word-wrap layout.
            let sub_chunks = word_wrap_line(grapheme, max_width, None);
            for sub_chunk in &sub_chunks[..sub_chunks.len().saturating_sub(1)] {
                chunks.push(TextChunk {
                    text: sub_chunk.text.clone(),
                    start_index: char_index + sub_chunk.start_index,
                    end_index: char_index + sub_chunk.end_index,
                });
            }
            if let Some(last) = sub_chunks.last() {
                chunk_start = char_index + last.start_index;
                current_width = visible_width(&last.text);
            }
            wrap_opp_index = None;
            continue;
        }

        // Advance.
        current_width += g_width;

        // Record wrap opportunity: whitespace followed by non-whitespace
        // (multiple spaces join; the break point is after the last space),
        // or at a boundary where either side is CJK (CJK allows breaking
        // between any adjacent characters).
        if let Some(next) = segments.get(position + 1) {
            let next_is_ws =
                !is_paste_marker(&next.segment) && segment_is_whitespace(&next.segment);
            if is_ws && !next_is_ws {
                wrap_opp_index = Some(next.index);
                wrap_opp_width = current_width;
            } else if !is_ws && !next_is_ws {
                let is_cjk =
                    !is_paste_marker(grapheme) && crate::utils::is_cjk_break_segment(grapheme);
                let next_is_cjk = !is_paste_marker(&next.segment)
                    && crate::utils::is_cjk_break_segment(&next.segment);
                if is_cjk || next_is_cjk {
                    wrap_opp_index = Some(next.index);
                    wrap_opp_width = current_width;
                }
            }
        }
    }

    // Push final chunk.
    chunks.push(TextChunk {
        text: line[chunk_start..].to_string(),
        start_index: chunk_start,
        end_index: line.len(),
    });

    chunks
}

/// The editor's text state, upstream `EditorState`.
#[derive(Debug, Clone, Default)]
struct EditorState {
    lines: Vec<String>,
    cursor_line: usize,
    cursor_col: usize,
}

/// Undo snapshot: editor text state plus the paste registry, upstream
/// `EditorSnapshot`.
#[derive(Debug, Clone)]
struct EditorSnapshot {
    state: EditorState,
    pastes: HashMap<u32, String>,
    paste_counter: u32,
}

/// One wrapped display row, upstream `LayoutLine`.
#[derive(Debug, Clone)]
struct LayoutLine {
    text: String,
    has_cursor: bool,
    cursor_pos: Option<usize>,
}

/// A wrapped segment of a logical line, upstream's anonymous visual-line
/// entries `{ logicalLine, startCol, length }`.
#[derive(Debug, Clone, Copy)]
struct VisualLine {
    logical_line: usize,
    start_col: usize,
    length: usize,
}

/// Cursor position accessor result, upstream `getCursor`'s
/// `{ line, col }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorPosition {
    /// Logical line index, upstream `line`.
    pub line: usize,
    /// Byte column within the line, upstream `col`.
    pub col: usize,
}

/// Border and accent color callback, upstream `(str: string) => string`.
pub type EditorColorFn = Rc<dyn Fn(&str) -> String>;

/// The submit/change callback, upstream `onSubmit?` / `onChange?`.
pub type EditorCallback = Rc<dyn Fn(&str)>;

/// The editor theme, upstream `EditorTheme`.
#[derive(Clone)]
pub struct EditorTheme {
    /// Border color function, upstream `borderColor`.
    pub border_color: EditorColorFn,
    /// The dropdown theme for autocomplete suggestions, upstream
    /// `selectList`.
    pub select_list: SelectListTheme,
}

/// Construction options, upstream `EditorOptions`.
#[derive(Debug, Clone, Copy)]
pub struct EditorOptions {
    /// Horizontal padding, upstream `paddingX`.
    pub padding_x: usize,
    /// Maximum rows visible in the autocomplete dropdown, upstream
    /// `autocompleteMaxVisible`; clamped into `[3, 20]`, default 5.
    pub autocomplete_max_visible: usize,
}

impl Default for EditorOptions {
    fn default() -> Self {
        Self {
            padding_x: 0,
            autocomplete_max_visible: 5,
        }
    }
}

/// The last mutation the editor performed, upstream `lastAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LastAction {
    #[default]
    None,
    Kill,
    Yank,
    TypeWord,
}

/// The pending character-jump direction, upstream
/// `"forward" | "backward"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JumpDirection {
    Forward,
    Backward,
}

/// Whether the open autocomplete picker came from a forced (Tab) or a
/// natural trigger, upstream `"regular" | "force" | null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutocompleteMode {
    Regular,
    Force,
}

/// The options of one autocomplete request, upstream's inline
/// `{ force, explicitTab }` objects.
#[derive(Debug, Clone, Copy)]
struct AutocompleteRequestOptions {
    force: bool,
    explicit_tab: bool,
}

/// One queued autocomplete request, upstream's `requestAutocomplete` call
/// plus the snapshot `startAutocompleteRequest` captures.
struct AutocompleteRequest {
    generation: u64,
    provider: Arc<dyn AutocompleteProvider>,
    lines: Vec<String>,
    cursor_line: usize,
    cursor_col: usize,
    force: bool,
    explicit_tab: bool,
    token: CancellationToken,
    debounce_ms: u64,
    snapshot_text: String,
    snapshot_line: usize,
    snapshot_col: usize,
}

/// A completed request handed back to the editor, upstream
/// `runAutocompleteRequest`'s continuation.
struct AutocompleteResult {
    generation: u64,
    token: CancellationToken,
    force: bool,
    explicit_tab: bool,
    suggestions: Option<AutocompleteSuggestions>,
    snapshot_text: String,
    snapshot_line: usize,
    snapshot_col: usize,
}

/// Worker state behind the mutex, upstream's debounce timer,
/// `autocompleteRequestTask` serialization, and `autocompleteAbort` slot.
struct AutocompleteWorkerState {
    pending: Option<AutocompleteRequest>,
    generation: u64,
    active_token: Option<CancellationToken>,
    shutdown: bool,
    busy: bool,
}

/// The shared worker handles, upstream's event-loop scheduling.
struct AutocompleteWorkerShared {
    state: Mutex<AutocompleteWorkerState>,
    signal: Condvar,
    results_tx: mpsc::Sender<AutocompleteResult>,
}

impl AutocompleteWorkerShared {
    /// Lock the worker mutex. A poisoned mutex means a worker-thread panic
    /// already happened; upstream's event loop has no poisoning concept, so
    /// the port surfaces it rather than guessing.
    #[expect(
        clippy::expect_used,
        reason = "a poisoned worker mutex means a thread panicked mid-request; surfacing beats silently desynchronizing the editor"
    )]
    fn lock(&self) -> std::sync::MutexGuard<'_, AutocompleteWorkerState> {
        self.state
            .lock()
            .expect("autocomplete worker mutex poisoned")
    }

    /// Wait on the shared condvar, re-locking on wake.
    #[expect(
        clippy::expect_used,
        reason = "the same poisoned-mutex contract as lock()"
    )]
    fn wait<'a>(
        &self,
        guard: std::sync::MutexGuard<'a, AutocompleteWorkerState>,
    ) -> std::sync::MutexGuard<'a, AutocompleteWorkerState> {
        self.signal
            .wait(guard)
            .expect("autocomplete worker mutex poisoned")
    }

    /// Wait with a timeout, re-locking on wake.
    #[expect(
        clippy::expect_used,
        reason = "the same poisoned-mutex contract as lock()"
    )]
    fn wait_timeout<'a>(
        &self,
        guard: std::sync::MutexGuard<'a, AutocompleteWorkerState>,
        timeout: Duration,
    ) -> std::sync::MutexGuard<'a, AutocompleteWorkerState> {
        self.signal
            .wait_timeout(guard, timeout)
            .expect("autocomplete worker mutex poisoned")
            .0
    }
}

/// The editor's channel to its worker thread, upstream's `setTimeout` +
/// promise chain; dropped with the editor, which shuts the worker down.
struct AutocompleteChannel {
    shared: Arc<AutocompleteWorkerShared>,
    results: mpsc::Receiver<AutocompleteResult>,
}

/// The slash-command dropdown layout, upstream
/// `SLASH_COMMAND_SELECT_LIST_LAYOUT`.
const SLASH_COMMAND_SELECT_LIST_LAYOUT: SelectListLayoutOptions = SelectListLayoutOptions {
    min_primary_column_width: Some(12),
    max_primary_column_width: Some(32),
    truncate_primary: None,
};

/// The editor's initial trigger characters, upstream's
/// `[...DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS]` spread.
fn default_trigger_characters() -> Vec<String> {
    DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS
        .iter()
        .map(|char| (*char).to_string())
        .collect()
}

/// Whether the in-flight request was replaced (a newer request queued) or
/// abandoned (cancel, shutdown): the generation moved on, a newer request
/// sits queued, or the worker is shutting down.
const fn is_superseded(state: &AutocompleteWorkerState, request_generation: u64) -> bool {
    state.generation != request_generation || state.pending.is_some() || state.shutdown
}

/// The autocomplete worker: one thread per editor, upstream's debounce
/// timer and request-task serialization on the event loop. Requests are
/// replaced (debounce restart) or abandoned (cancel) through the
/// generation counter; provider calls are serialized; results flow to the
/// editor's receiver where the host drains them.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the thread entry point owns its Arc handle; taking a reference would borrow from a local"
)]
#[expect(
    clippy::significant_drop_tightening,
    reason = "the condvar wait releases the mutex while parked; the guard drops at each loop exit"
)]
fn run_autocomplete_worker(shared: Arc<AutocompleteWorkerShared>) {
    loop {
        // Take the newest pending request, waiting for work.
        let request = {
            let mut state = shared.lock();
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(request) = state.pending.take() {
                    state.busy = true;
                    break request;
                }
                state = shared.wait(state);
            }
        };
        let request_generation = request.generation;

        // The debounce, upstream's setTimeout: a newer request or a cancel
        // abandons this one before the provider runs.
        if request.debounce_ms > 0 {
            let deadline = Instant::now() + Duration::from_millis(request.debounce_ms);
            let mut superseded = false;
            while !superseded {
                let mut state = shared.lock();
                if is_superseded(&state, request_generation) {
                    superseded = true;
                    state.busy = false;
                    shared.signal.notify_all();
                    drop(state);
                    break;
                }
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    drop(state);
                    break;
                };
                let mut state = shared.wait_timeout(state, remaining);
                if is_superseded(&state, request_generation) {
                    superseded = true;
                    state.busy = false;
                    shared.signal.notify_all();
                    drop(state);
                    break;
                }
                drop(state);
            }
            if superseded {
                continue;
            }
        }

        if request.token.is_cancelled() {
            let mut state = shared.lock();
            state.busy = false;
            shared.signal.notify_all();
            drop(state);
            continue;
        }

        let options = AutocompleteQueryOptions {
            signal: request.token.clone(),
            force: request.force,
        };
        let suggestions = request.provider.get_suggestions(
            &request.lines,
            request.cursor_line,
            request.cursor_col,
            &options,
        );

        // Upstream clears `autocompleteAbort` once the provider returns; a
        // newer request owns the slot by then.
        {
            let mut state = shared.lock();
            if state.generation == request_generation {
                state.active_token = None;
            }
            drop(state);
        }

        let _ = shared.results_tx.send(AutocompleteResult {
            generation: request_generation,
            token: request.token.clone(),
            force: request.force,
            explicit_tab: request.explicit_tab,
            suggestions,
            snapshot_text: request.snapshot_text,
            snapshot_line: request.snapshot_line,
            snapshot_col: request.snapshot_col,
        });

        let mut state = shared.lock();
        state.busy = false;
        shared.signal.notify_all();
        drop(state);
    }
}

/// Spawn the worker thread for a fresh editor, upstream's implicit
/// event-loop scheduling; the editor's `Drop` shuts it down.
fn spawn_autocomplete_worker(
    results_tx: mpsc::Sender<AutocompleteResult>,
) -> Arc<AutocompleteWorkerShared> {
    let shared = Arc::new(AutocompleteWorkerShared {
        state: Mutex::new(AutocompleteWorkerState {
            pending: None,
            generation: 0,
            active_token: None,
            shutdown: false,
            busy: false,
        }),
        signal: Condvar::new(),
        results_tx,
    });
    let thread_shared = Arc::clone(&shared);
    let spawned = thread::Builder::new()
        .name("editor-autocomplete".to_string())
        .spawn(move || run_autocomplete_worker(thread_shared));
    if spawned.is_err() {
        // No worker thread: mark the shared state shut down so requests
        // queue nowhere and the drain sees nothing.
        let mut state = shared.lock();
        state.shutdown = true;
    }
    shared
}

/// Cursor placement when replacing the whole text, upstream
/// `setTextInternal`'s `"start" | "end"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CursorPlacement {
    Start,
    End,
}

/// The multi-line text editor, upstream `Editor`.
pub struct Editor {
    state: RefCell<EditorState>,
    /// Focusable interface — set by the TUI when focus changes.
    focused: Cell<bool>,
    /// The owning TUI, held weakly so the editor can live inside the TUI's
    /// component tree without an `Rc` cycle. A dead handle degrades the
    /// terminal-row reads to zero rows.
    tui: RefCell<Weak<Tui>>,
    padding_x: Cell<usize>,

    // Store last render geometry for cursor navigation and mouse hit-testing.
    last_width: Cell<usize>,
    rendered_visible_line_count: Cell<usize>,
    rendered_autocomplete_height: Cell<usize>,

    // Vertical scrolling support
    scroll_offset: Cell<usize>,

    /// Border color function (can be changed dynamically), upstream
    /// `borderColor`.
    pub border_color: EditorColorFn,
    /// The construction theme, upstream `this.theme`: its dropdown theme
    /// colors every autocomplete list the editor opens.
    theme: EditorTheme,

    // Autocomplete support, upstream's `autocomplete*` fields: the provider
    // handle, the trigger/debounce patterns rebuilt per provider, the open
    // dropdown list, and the generation counter discarding stale results.
    autocomplete_provider: RefCell<Option<Arc<dyn AutocompleteProvider>>>,
    autocomplete_trigger_characters: RefCell<Vec<String>>,
    autocomplete_trigger_pattern: RefCell<Option<Regex>>,
    autocomplete_debounce_pattern: RefCell<Option<Regex>>,
    autocomplete_list: RefCell<Option<SelectList>>,
    autocomplete_state: Cell<Option<AutocompleteMode>>,
    autocomplete_prefix: RefCell<String>,
    autocomplete_max_visible: Cell<usize>,
    autocomplete_start_token: Cell<u64>,
    autocomplete_channel: RefCell<Option<AutocompleteChannel>>,

    // Paste tracking for large pastes
    pastes: RefCell<HashMap<u32, String>>,
    paste_counter: Cell<u32>,

    // Bracketed paste mode buffering
    paste_buffer: RefCell<String>,
    is_in_paste: Cell<bool>,

    // Prompt history for up/down navigation
    history: RefCell<Vec<String>>,
    /// -1 = not browsing, 0 = most recent, 1 = older, etc., upstream
    /// `historyIndex`.
    history_index: Cell<i64>,
    history_draft: RefCell<Option<EditorState>>,

    // Kill ring for Emacs-style kill/yank operations
    kill_ring: RefCell<KillRing>,
    last_action: Cell<LastAction>,

    // Character jump mode
    jump_mode: Cell<Option<JumpDirection>>,

    // Preferred visual column for vertical cursor movement (sticky column)
    preferred_visual_col: Cell<Option<usize>>,

    // When the cursor is snapped to the start of an atomic segment, e.g. a
    // paste marker, cursorCol no longer reflects where the cursor would
    // have landed. This field stores the pre-snap cursorCol so that the
    // next vertical move can resolve it to a visual column on whatever VL
    // it belongs to.
    snapped_from_cursor_col: Cell<Option<usize>>,

    // Undo support
    undo_stack: RefCell<UndoStack<EditorSnapshot>>,

    /// Called when user submits (e.g. Enter key), upstream `onSubmit`.
    pub on_submit: RefCell<Option<EditorCallback>>,
    /// Called when text changes, upstream `onChange`.
    pub on_change: RefCell<Option<EditorCallback>>,
    /// Disable the submit path, upstream `disableSubmit`.
    pub disable_submit: Cell<bool>,

    parser: KeyParser,
}

impl Editor {
    /// Upstream's default-argument constructor.
    #[must_use]
    pub fn new(tui: &Rc<Tui>, theme: EditorTheme) -> Self {
        Self::with_options(tui, theme, EditorOptions::default())
    }

    /// Upstream's `new Editor(tui, theme, options)`.
    #[must_use]
    pub fn with_options(tui: &Rc<Tui>, theme: EditorTheme, options: EditorOptions) -> Self {
        let max_visible = options.autocomplete_max_visible.clamp(3, 20);
        Self {
            state: RefCell::new(EditorState {
                lines: vec![String::new()],
                cursor_line: 0,
                cursor_col: 0,
            }),
            focused: Cell::new(false),
            tui: RefCell::new(Rc::downgrade(tui)),
            padding_x: Cell::new(options.padding_x),
            last_width: Cell::new(80),
            rendered_visible_line_count: Cell::new(1),
            rendered_autocomplete_height: Cell::new(0),
            scroll_offset: Cell::new(0),
            border_color: theme.border_color.clone(),
            theme,
            autocomplete_provider: RefCell::new(None),
            autocomplete_trigger_characters: RefCell::new(default_trigger_characters()),
            autocomplete_trigger_pattern: RefCell::new(Some(build_trigger_pattern(
                &default_trigger_characters(),
            ))),
            autocomplete_debounce_pattern: RefCell::new(Some(build_debounce_pattern(
                &default_trigger_characters(),
            ))),
            autocomplete_list: RefCell::new(None),
            autocomplete_state: Cell::new(None),
            autocomplete_prefix: RefCell::new(String::new()),
            autocomplete_max_visible: Cell::new(max_visible),
            autocomplete_start_token: Cell::new(0),
            autocomplete_channel: RefCell::new(None),
            pastes: RefCell::new(HashMap::new()),
            paste_counter: Cell::new(0),
            paste_buffer: RefCell::new(String::new()),
            is_in_paste: Cell::new(false),
            history: RefCell::new(Vec::new()),
            history_index: Cell::new(-1),
            history_draft: RefCell::new(None),
            kill_ring: RefCell::new(KillRing::default()),
            last_action: Cell::new(LastAction::None),
            jump_mode: Cell::new(None),
            preferred_visual_col: Cell::new(None),
            snapped_from_cursor_col: Cell::new(None),
            undo_stack: RefCell::new(UndoStack::new()),
            on_submit: RefCell::new(None),
            on_change: RefCell::new(None),
            disable_submit: Cell::new(false),
            parser: KeyParser::new(),
        }
    }

    /// Set of currently valid paste IDs, for marker-aware segmentation,
    /// upstream `validPasteIds`.
    fn valid_paste_ids(&self) -> HashSet<u32> {
        self.pastes.borrow().keys().copied().collect()
    }

    /// Segment text with paste-marker awareness, only merging markers with
    /// valid IDs, upstream `segment`.
    fn segment(&self, text: &str, mode: SegmentMode) -> Vec<SegmentData> {
        segment_with_markers(text, mode, &self.valid_paste_ids())
    }

    /// The horizontal padding, upstream `getPaddingX`.
    #[must_use]
    pub const fn get_padding_x(&self) -> usize {
        self.padding_x.get()
    }

    /// Set the horizontal padding, upstream `setPaddingX`.
    pub fn set_padding_x(&self, padding: usize) {
        if self.padding_x.get() != padding {
            self.padding_x.set(padding);
            self.request_render();
        }
    }

    // ============================================================
    // Autocomplete support, upstream's autocomplete methods (#49)
    // ============================================================

    /// The dropdown's maximum visible rows, upstream
    /// `getAutocompleteMaxVisible`.
    #[must_use]
    pub const fn get_autocomplete_max_visible(&self) -> usize {
        self.autocomplete_max_visible.get()
    }

    /// Set the dropdown's maximum visible rows, upstream
    /// `setAutocompleteMaxVisible`: clamped into `[3, 20]`, re-render on
    /// change.
    pub fn set_autocomplete_max_visible(&self, max_visible: usize) {
        let new_max_visible = max_visible.clamp(3, 20);
        if self.autocomplete_max_visible.get() != new_max_visible {
            self.autocomplete_max_visible.set(new_max_visible);
            self.request_render();
        }
    }

    /// Install the autocomplete provider, upstream
    /// `setAutocompleteProvider`: cancels any open picker, stores the
    /// provider, and rebuilds the trigger characters.
    pub fn set_autocomplete_provider(&self, provider: Arc<dyn AutocompleteProvider>) {
        self.ensure_autocomplete_worker();
        self.cancel_autocomplete();
        *self.autocomplete_provider.borrow_mut() = Some(provider);
        let trigger_characters = self
            .autocomplete_provider
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |provider| provider.trigger_characters());
        self.set_autocomplete_trigger_characters(&trigger_characters);
    }

    /// Whether a picker is open, upstream `isShowingAutocomplete`.
    #[must_use]
    pub const fn is_showing_autocomplete(&self) -> bool {
        self.autocomplete_state.get().is_some()
    }

    /// Whether the autocomplete request pipeline has nothing queued or
    /// running. Upstream's equivalent is awaiting the event loop; the
    /// host's flush helper (and tests) poll this before draining.
    ///
    /// # Panics
    /// Panics if the worker mutex was poisoned — a worker thread panicked
    /// mid-request.
    #[must_use]
    pub fn autocomplete_idle(&self) -> bool {
        let channel_guard = self.autocomplete_channel.borrow();
        let Some(channel) = channel_guard.as_ref() else {
            return true;
        };
        let state = channel.shared.lock();
        state.shutdown || (!state.busy && state.pending.is_none())
    }

    /// Apply completed autocomplete results, upstream's promise
    /// continuations running on the event loop: stale generations and
    /// stale snapshots are discarded, an empty suggestion set closes the
    /// picker, a single forced Tab suggestion auto-applies without a menu,
    /// and everything else opens the dropdown. Called before each render
    /// (the pre-paint resumption point) and callable by the host pump.
    pub fn drain_autocomplete(&self) {
        let results: Vec<AutocompleteResult> = {
            let channel_guard = self.autocomplete_channel.borrow();
            let Some(channel) = channel_guard.as_ref() else {
                return;
            };
            let mut results = Vec::new();
            while let Ok(result) = channel.results.try_recv() {
                results.push(result);
            }
            results
        };

        for result in results {
            // Upstream's `isAutocompleteRequestCurrent`: not aborted, still
            // the newest request, and the editor text/cursor unchanged
            // since.
            if result.token.is_cancelled()
                || result.generation != self.autocomplete_start_token.get()
            {
                continue;
            }
            let current = {
                let state = self.state.borrow();
                state.cursor_line == result.snapshot_line
                    && state.cursor_col == result.snapshot_col
                    && self.get_text() == result.snapshot_text
            };
            if !current {
                continue;
            }

            let Some(suggestions) = result.suggestions else {
                self.cancel_autocomplete();
                self.request_render();
                continue;
            };
            if suggestions.items.is_empty() {
                self.cancel_autocomplete();
                self.request_render();
                continue;
            }

            // A forced Tab query resolving to exactly one suggestion
            // applies it directly without showing the menu.
            if result.force && result.explicit_tab && suggestions.items.len() == 1 {
                let item = &suggestions.items[0];
                self.apply_selected_completion(item, &suggestions.prefix);
                self.fire_on_change();
                self.request_render();
                continue;
            }

            let mode = if result.force {
                AutocompleteMode::Force
            } else {
                AutocompleteMode::Regular
            };
            self.apply_autocomplete_suggestions(&suggestions, mode);
            self.request_render();
        }
    }

    /// Spawn the worker thread on first provider install, upstream's
    /// implicit event-loop scheduling.
    fn ensure_autocomplete_worker(&self) {
        if self.autocomplete_channel.borrow().is_some() {
            return;
        }
        let (results_tx, results) = mpsc::channel();
        let shared = spawn_autocomplete_worker(results_tx);
        *self.autocomplete_channel.borrow_mut() = Some(AutocompleteChannel { shared, results });
    }

    /// Set the trigger characters, upstream `setAutocompleteTriggerCharacters`:
    /// the defaults plus single non-`/`, non-whitespace additions; the
    /// trigger and debounce patterns rebuild with them.
    fn set_autocomplete_trigger_characters(&self, trigger_characters: &[String]) {
        let mut next: Vec<String> = DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS
            .iter()
            .map(|char| (*char).to_string())
            .collect();
        for character in trigger_characters {
            let single_char = character.chars().count() == 1;
            if !single_char
                || character == "/"
                || character.chars().next().is_some_and(is_whitespace_char)
                || next.contains(character)
            {
                continue;
            }
            next.push(character.clone());
        }
        self.autocomplete_trigger_characters
            .borrow_mut()
            .clone_from(&next);
        *self.autocomplete_trigger_pattern.borrow_mut() = Some(build_trigger_pattern(&next));
        *self.autocomplete_debounce_pattern.borrow_mut() = Some(build_debounce_pattern(&next));
    }

    /// The debounce delay for a request, upstream
    /// `getAutocompleteDebounceMs`: explicit Tab and forced queries run
    /// immediately; symbol-triggered tokens wait
    /// [`ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS`].
    fn get_autocomplete_debounce_ms(&self, options: AutocompleteRequestOptions) -> u64 {
        if options.explicit_tab || options.force {
            return 0;
        }

        let current_line = {
            let state = self.state.borrow();
            state.lines[state.cursor_line].clone()
        };
        let text_before_cursor =
            &current_line[..self.state.borrow().cursor_col.min(current_line.len())];
        let is_attachment = self
            .autocomplete_debounce_pattern
            .borrow()
            .as_ref()
            .is_some_and(|pattern| pattern.is_match(text_before_cursor));
        if is_attachment {
            ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS
        } else {
            0
        }
    }

    /// Queue an autocomplete request, upstream `requestAutocomplete`:
    /// forced queries pass the provider's file-completion gate, then the
    /// previous request is cancelled and the new one queued (debounced or
    /// immediate).
    fn request_autocomplete(&self, options: AutocompleteRequestOptions) {
        let Some(provider) = self.autocomplete_provider.borrow().clone() else {
            return;
        };

        if options.force {
            let (lines, cursor_line, cursor_col) = {
                let state = self.state.borrow();
                (state.lines.clone(), state.cursor_line, state.cursor_col)
            };
            let should_trigger =
                provider.should_trigger_file_completion(&lines, cursor_line, cursor_col);
            if !should_trigger {
                return;
            }
        }

        self.cancel_autocomplete_request();
        let generation = self.autocomplete_start_token.get() + 1;
        self.autocomplete_start_token.set(generation);

        let debounce_ms = self.get_autocomplete_debounce_ms(options);
        let (snapshot_text, snapshot_line, snapshot_col) = {
            let state = self.state.borrow();
            (self.get_text(), state.cursor_line, state.cursor_col)
        };
        let lines = self.state.borrow().lines.clone();
        let request = AutocompleteRequest {
            generation,
            provider,
            lines,
            cursor_line: snapshot_line,
            cursor_col: snapshot_col,
            force: options.force,
            explicit_tab: options.explicit_tab,
            token: CancellationToken::new(),
            debounce_ms,
            snapshot_text,
            snapshot_line,
            snapshot_col,
        };
        let token = request.token.clone();

        let channel_guard = self.autocomplete_channel.borrow();
        let Some(channel) = channel_guard.as_ref() else {
            return;
        };
        let mut state = channel.shared.lock();
        state.pending = Some(request);
        state.generation = generation;
        state.active_token = Some(token);
        channel.shared.signal.notify_all();
        drop(state);
    }

    /// Invalidate the in-flight request, upstream
    /// `cancelAutocompleteRequest`: bump the generation, clear the pending
    /// queue, abort the active token.
    fn cancel_autocomplete_request(&self) {
        self.autocomplete_start_token
            .set(self.autocomplete_start_token.get() + 1);
        if let Some(channel) = self.autocomplete_channel.borrow().as_ref() {
            let mut state = channel.shared.lock();
            state.pending = None;
            if let Some(token) = state.active_token.take() {
                token.cancel();
            }
            channel.shared.signal.notify_all();
            drop(state);
        }
    }

    /// Drop the picker UI, upstream `clearAutocompleteUi`.
    fn clear_autocomplete_ui(&self) {
        self.autocomplete_state.set(None);
        *self.autocomplete_list.borrow_mut() = None;
        *self.autocomplete_prefix.borrow_mut() = String::new();
    }

    /// Cancel the request and the picker, upstream `cancelAutocomplete`.
    fn cancel_autocomplete(&self) {
        self.cancel_autocomplete_request();
        self.clear_autocomplete_ui();
    }

    /// Natural trigger, upstream `tryTriggerAutocomplete`.
    fn try_trigger_autocomplete(&self, explicit_tab: bool) {
        self.request_autocomplete(AutocompleteRequestOptions {
            force: false,
            explicit_tab,
        });
    }

    /// Tab completion dispatch, upstream `handleTabCompletion`: a
    /// token-started slash command re-runs the command query, everything
    /// else forces a file query.
    fn handle_tab_completion(&self) {
        if self.autocomplete_provider.borrow().is_none() {
            return;
        }

        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();
        let before_cursor =
            current_line[..self.state.borrow().cursor_col.min(current_line.len())].to_string();

        if self.is_in_slash_command_context(&before_cursor)
            && !before_cursor.trim_start().contains(' ')
        {
            self.handle_slash_command_completion();
        } else {
            self.force_file_autocomplete(true);
        }
    }

    /// Slash-command Tab completion, upstream
    /// `handleSlashCommandCompletion`.
    fn handle_slash_command_completion(&self) {
        self.request_autocomplete(AutocompleteRequestOptions {
            force: false,
            explicit_tab: true,
        });
    }

    /// Forced file query, upstream `forceFileAutocomplete`.
    fn force_file_autocomplete(&self, explicit_tab: bool) {
        self.request_autocomplete(AutocompleteRequestOptions {
            force: true,
            explicit_tab,
        });
    }

    /// Refresh an open picker for the new text/cursor, upstream
    /// `updateAutocomplete`.
    fn update_autocomplete(&self) {
        if self.autocomplete_state.get().is_none() || self.autocomplete_provider.borrow().is_none()
        {
            return;
        }
        let force = self.autocomplete_state.get() == Some(AutocompleteMode::Force);
        self.request_autocomplete(AutocompleteRequestOptions {
            force,
            explicit_tab: false,
        });
    }

    /// The best preselected row for a suggestion set, upstream
    /// `getBestAutocompleteMatchIndex`: exact value wins, then the first
    /// value starting with the prefix, else none (case-sensitive,
    /// `value` only).
    #[expect(
        clippy::unused_self,
        reason = "upstream's getBestAutocompleteMatchIndex is an editor method; the port keeps the shape"
    )]
    fn get_best_autocomplete_match_index(
        &self,
        items: &[AutocompleteItem],
        prefix: &str,
    ) -> Option<usize> {
        if prefix.is_empty() {
            return None;
        }

        let mut first_prefix_index = None;

        for (index, item) in items.iter().enumerate() {
            if item.value == prefix {
                return Some(index); // Exact match always wins
            }
            if first_prefix_index.is_none() && item.value.starts_with(prefix) {
                first_prefix_index = Some(index);
            }
        }

        first_prefix_index
    }

    /// Build the dropdown list, upstream `createAutocompleteList`: the
    /// slash-command layout for `/` prefixes. Upstream wires the list's
    /// `onSelect` to apply the clicked completion; the port applies it in
    /// the editor's mouse dispatch because the callback would need a
    /// handle on the non-`Rc` editor.
    fn create_autocomplete_list(&self, prefix: &str, items: &[AutocompleteItem]) -> SelectList {
        let layout = if prefix.starts_with('/') {
            SLASH_COMMAND_SELECT_LIST_LAYOUT
        } else {
            SelectListLayoutOptions::default()
        };
        let select_items: Vec<SelectItem> = items.iter().map(|item| item.clone().into()).collect();
        SelectList::with_layout(
            select_items,
            self.autocomplete_max_visible.get(),
            self.theme.select_list.clone(),
            layout,
        )
    }

    /// Apply a suggestion set to the UI, upstream
    /// `applyAutocompleteSuggestions`.
    fn apply_autocomplete_suggestions(
        &self,
        suggestions: &AutocompleteSuggestions,
        mode: AutocompleteMode,
    ) {
        self.autocomplete_prefix
            .borrow_mut()
            .clone_from(&suggestions.prefix);
        let list = self.create_autocomplete_list(&suggestions.prefix, &suggestions.items);

        if let Some(best_match_index) =
            self.get_best_autocomplete_match_index(&suggestions.items, &suggestions.prefix)
        {
            list.set_selected_index(best_match_index);
        }

        *self.autocomplete_list.borrow_mut() = Some(list);
        self.autocomplete_state.set(Some(mode));
    }

    /// Splice a selected completion into the text, upstream's
    /// `applyCompletion` call shared by the Tab, confirm, click, and
    /// auto-apply paths: undo snapshot, action reset, splice, cursor. The
    /// prefix is the open picker's (`autocompletePrefix`) except for the
    /// auto-apply path, which passes the fresh suggestion set's prefix —
    /// upstream's exact arguments.
    fn apply_selected_completion(&self, item: &AutocompleteItem, prefix: &str) {
        let Some(provider) = self.autocomplete_provider.borrow().clone() else {
            return;
        };
        self.push_undo_snapshot();
        self.last_action.set(LastAction::None);
        let (lines, cursor_line, cursor_col) = {
            let state = self.state.borrow();
            (state.lines.clone(), state.cursor_line, state.cursor_col)
        };
        let result = provider.apply_completion(&lines, cursor_line, cursor_col, item, prefix);
        {
            let mut state = self.state.borrow_mut();
            state.lines = result.lines;
            state.cursor_line = result.cursor_line;
        }
        self.set_cursor_col(result.cursor_col);
    }

    /// Whether the cursor sits where a slash command can start, upstream
    /// `isAtStartOfMessage`.
    fn is_at_start_of_message(&self) -> bool {
        if !self.is_slash_menu_allowed() {
            return false;
        }
        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();
        let before_cursor =
            current_line[..self.state.borrow().cursor_col.min(current_line.len())].to_string();
        let trimmed = before_cursor.trim();
        trimmed.is_empty() || trimmed == "/"
    }

    /// Slash menus live on the first editor line only, upstream
    /// `isSlashMenuAllowed`.
    fn is_slash_menu_allowed(&self) -> bool {
        self.state.borrow().cursor_line == 0
    }

    /// Whether the text before the cursor is inside a slash command
    /// context, upstream `isInSlashCommandContext`.
    fn is_in_slash_command_context(&self, text_before_cursor: &str) -> bool {
        self.is_slash_menu_allowed() && text_before_cursor.trim_start().starts_with('/')
    }

    /// Add a prompt to history for up/down arrow navigation, upstream
    /// `addToHistory`. Called after successful submission.
    pub fn add_to_history(&self, text: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        let mut history = self.history.borrow_mut();
        // Don't add consecutive duplicates
        if history.first().is_some_and(|first| first == trimmed) {
            return;
        }
        history.insert(0, trimmed.to_string());
        // Limit history size
        if history.len() > 100 {
            history.pop();
        }
    }

    fn is_editor_empty(&self) -> bool {
        let state = self.state.borrow();
        state.lines.len() == 1 && state.lines[0].is_empty()
    }

    fn is_on_first_visual_line(&self) -> bool {
        let visual_lines = self.build_visual_line_map(self.last_width.get());
        self.find_current_visual_line(&visual_lines) == 0
    }

    fn is_on_last_visual_line(&self) -> bool {
        let visual_lines = self.build_visual_line_map(self.last_width.get());
        self.find_current_visual_line(&visual_lines) == visual_lines.len() - 1
    }

    /// Browse the prompt history, upstream `navigateHistory`; direction -1
    /// is up (older), 1 is down (newer).
    fn navigate_history(&self, direction: i32) {
        self.last_action.set(LastAction::None);
        let history_len = i64::try_from(self.history.borrow().len()).unwrap_or(i64::MAX);
        if history_len == 0 {
            return;
        }

        let history_index = self.history_index.get();
        let new_index = history_index - i64::from(direction); // Up(-1) increases index, Down(1) decreases
        if new_index < -1 || new_index >= history_len {
            return;
        }

        // Capture state when first entering history browsing mode
        if history_index == -1 && new_index >= 0 {
            self.push_undo_snapshot();
            *self.history_draft.borrow_mut() = Some(self.state.borrow().clone());
        }

        self.history_index.set(new_index);

        if new_index == -1 {
            let draft = self.history_draft.borrow_mut().take();
            if let Some(draft) = draft {
                *self.state.borrow_mut() = draft;
                self.preferred_visual_col.set(None);
                self.snapped_from_cursor_col.set(None);
                self.scroll_offset.set(0);
                self.fire_on_change();
            } else {
                self.set_text_internal("", CursorPlacement::End);
            }
        } else {
            let entry = self
                .history
                .borrow()
                .get(usize::try_from(new_index).unwrap_or_default())
                .cloned()
                .unwrap_or_default();
            self.set_text_internal(
                &entry,
                if direction == -1 {
                    CursorPlacement::Start
                } else {
                    CursorPlacement::End
                },
            );
        }
    }

    fn exit_history_browsing(&self) {
        self.history_index.set(-1);
        *self.history_draft.borrow_mut() = None;
    }

    /// Internal setText that doesn't reset history state — used by
    /// navigateHistory, upstream `setTextInternal`.
    fn set_text_internal(&self, text: &str, cursor_placement: CursorPlacement) {
        let lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let mut state = self.state.borrow_mut();
        state.lines = lines;
        let placement_start = cursor_placement == CursorPlacement::Start;
        state.cursor_line = if placement_start {
            0
        } else {
            state.lines.len() - 1
        };
        let cursor_col = if placement_start {
            0
        } else {
            state.lines[state.cursor_line].len()
        };
        drop(state);
        self.set_cursor_col(cursor_col);
        // Reset scroll - render() will adjust to show cursor
        self.scroll_offset.set(0);

        self.fire_on_change();
    }

    /// The top border, scroll indicator when content is hidden above,
    /// upstream `renderTopBorder`.
    fn render_top_border(&self, width: usize, hidden_line_count: usize) -> String {
        let border = if hidden_line_count > 0 {
            create_scroll_border("↑", hidden_line_count, width)
        } else {
            "─".repeat(width)
        };
        (self.border_color)(&border)
    }

    /// The bottom border, scroll indicator when content is hidden below,
    /// upstream `renderBottomBorder`.
    fn render_bottom_border(&self, width: usize, hidden_line_count: usize) -> String {
        let border = if hidden_line_count > 0 {
            create_scroll_border("↓", hidden_line_count, width)
        } else {
            "─".repeat(width)
        };
        (self.border_color)(&border)
    }

    /// The editor frame, upstream `Editor.render`: top and bottom borders,
    /// the wrapped visible rows with the inverse-video cursor, and the
    /// scroll-adjusted slice of layout lines.
    fn render_impl(&self, width: usize) -> Vec<String> {
        // Completed autocomplete requests resume before the paint, the
        // event loop's continuation slot.
        self.drain_autocomplete();

        let max_padding = (width.saturating_sub(1)) / 2;
        let padding_x = self.padding_x.get().min(max_padding);
        let content_width = (width.saturating_sub(padding_x * 2)).max(1);

        // Layout width: with padding the cursor can overflow into it,
        // without padding we reserve 1 column for the cursor.
        let layout_width = (content_width.saturating_sub(usize::from(padding_x == 0))).max(1);

        // Store for cursor navigation (must match wrapping width)
        self.last_width.set(layout_width);

        // Layout the text
        let layout_lines = self.layout_text(layout_width);

        // Calculate max visible lines: 30% of terminal height, minimum 5 lines
        let terminal_rows = usize::from(self.terminal_rows());
        let max_visible_lines = 5.max(terminal_rows * 30 / 100);

        // Find the cursor line index in layoutLines
        let cursor_line_index = layout_lines
            .iter()
            .position(|line| line.has_cursor)
            .unwrap_or(0);

        // Adjust scroll offset to keep cursor visible
        if cursor_line_index < self.scroll_offset.get() {
            self.scroll_offset.set(cursor_line_index);
        } else if cursor_line_index >= self.scroll_offset.get() + max_visible_lines {
            self.scroll_offset
                .set(cursor_line_index - max_visible_lines + 1);
        }

        // Clamp scroll offset to valid range
        let max_scroll_offset = layout_lines.len().saturating_sub(max_visible_lines);
        self.scroll_offset
            .set(self.scroll_offset.get().min(max_scroll_offset));

        // Get visible lines slice
        let scroll_offset = self.scroll_offset.get();
        let visible_lines: Vec<LayoutLine> = layout_lines
            .iter()
            .skip(scroll_offset)
            .take(max_visible_lines)
            .cloned()
            .collect();
        self.rendered_visible_line_count.set(visible_lines.len());

        let mut result: Vec<String> = Vec::new();
        let left_padding = " ".repeat(padding_x);
        let right_padding = left_padding.clone();

        // Render top border (with scroll indicator if scrolled down)
        result.push(self.render_top_border(width, scroll_offset));

        // Render each visible layout line. Emit the hardware cursor marker
        // when focused so the TUI can position the hardware cursor for IME
        // candidate-window placement even while an overlay is visible.
        let emit_cursor_marker = self.focused.get();

        for layout_line in &visible_lines {
            let mut display_text = layout_line.text.clone();
            let mut line_visible_width = visible_width(&layout_line.text);
            let mut cursor_in_padding = false;

            // Add cursor if this line has it
            if layout_line.has_cursor
                && let Some(cursor_pos) = layout_line.cursor_pos
            {
                let before = display_text[..cursor_pos].to_string();
                let after = display_text[cursor_pos..].to_string();

                // Hardware cursor marker (zero-width, emitted before fake
                // cursor for IME positioning)
                let marker = if emit_cursor_marker {
                    CURSOR_MARKER
                } else {
                    ""
                };

                if after.is_empty() {
                    // Cursor is at the end - add highlighted space
                    let cursor = "\x1b[7m \x1b[0m";
                    display_text = format!("{before}{marker}{cursor}");
                    line_visible_width += 1;
                    // If cursor overflows content width into the padding,
                    // flag it
                    if line_visible_width > content_width && padding_x > 0 {
                        cursor_in_padding = true;
                    }
                } else {
                    // Cursor is on a character (grapheme) - replace it
                    // with highlighted version
                    let first_grapheme = grapheme_segments(&after)
                        .next()
                        .map_or_else(String::new, str::to_string);
                    let rest_after = after[first_grapheme.len()..].to_string();
                    let cursor = format!("\x1b[7m{first_grapheme}\x1b[0m");
                    display_text = format!("{before}{marker}{cursor}{rest_after}");
                    // lineVisibleWidth stays the same - we're replacing,
                    // not adding
                }
            }

            // Calculate padding based on actual visible width
            let padding = " ".repeat(content_width.saturating_sub(line_visible_width));
            let line_right_padding = if cursor_in_padding {
                right_padding.get(1..).unwrap_or_default().to_string()
            } else {
                right_padding.clone()
            };

            // Render the line (no side borders, just horizontal lines above
            // and below)
            result.push(format!(
                "{left_padding}{display_text}{padding}{line_right_padding}"
            ));
        }

        // Render bottom border (with scroll indicator if more content below)
        let lines_below = layout_lines
            .len()
            .saturating_sub(scroll_offset + visible_lines.len());
        result.push(self.render_bottom_border(width, lines_below));

        // Add autocomplete list if active
        self.rendered_autocomplete_height.set(0);
        if self.autocomplete_state.get().is_some()
            && let Some(list) = self.autocomplete_list.borrow().as_ref()
        {
            let autocomplete_result = list.render(content_width);
            self.rendered_autocomplete_height
                .set(autocomplete_result.len());
            for line in autocomplete_result {
                let line_width = visible_width(&line);
                let line_padding = " ".repeat(content_width.saturating_sub(line_width));
                result.push(format!("{left_padding}{line}{line_padding}{right_padding}"));
            }
        }

        result
    }

    /// The text layout, upstream `layoutText`: one display row per logical
    /// line, or word-wrapped chunks with the cursor mapped into its chunk.
    fn layout_text(&self, content_width: usize) -> Vec<LayoutLine> {
        let mut layout_lines: Vec<LayoutLine> = Vec::new();

        let state = self.state.borrow();
        if state.lines.is_empty() || (state.lines.len() == 1 && state.lines[0].is_empty()) {
            // Empty editor
            layout_lines.push(LayoutLine {
                text: String::new(),
                has_cursor: true,
                cursor_pos: Some(0),
            });
            return layout_lines;
        }

        // Process each logical line
        for (index, line) in state.lines.iter().enumerate() {
            let line = line.as_str();
            let is_current_line = index == state.cursor_line;
            let line_visible_width = visible_width(line);

            if line_visible_width <= content_width {
                // Line fits in one layout line
                if is_current_line {
                    layout_lines.push(LayoutLine {
                        text: line.to_string(),
                        has_cursor: true,
                        cursor_pos: Some(state.cursor_col),
                    });
                } else {
                    layout_lines.push(LayoutLine {
                        text: line.to_string(),
                        has_cursor: false,
                        cursor_pos: None,
                    });
                }
            } else {
                // Line needs wrapping - use word-aware wrapping
                let pre_segmented = self.segment(line, SegmentMode::Grapheme);
                let chunks = word_wrap_line(line, content_width, Some(&pre_segmented));

                for (chunk_index, chunk) in chunks.iter().enumerate() {
                    let cursor_pos = state.cursor_col;
                    let is_last_chunk = chunk_index == chunks.len() - 1;

                    // Determine if cursor is in this chunk. For word-wrapped
                    // chunks, we need to handle the case where cursor might be
                    // in trimmed whitespace at end of chunk.
                    let mut has_cursor_in_chunk = false;
                    let mut adjusted_cursor_pos = 0;

                    if is_current_line {
                        if is_last_chunk {
                            // Last chunk: cursor belongs here if >= startIndex
                            has_cursor_in_chunk = cursor_pos >= chunk.start_index;
                            adjusted_cursor_pos = cursor_pos - chunk.start_index;
                        } else {
                            // Non-last chunk: cursor belongs here if in range
                            // [startIndex, endIndex), clamped to the chunk text
                            // in case the cursor was in trimmed whitespace.
                            has_cursor_in_chunk =
                                cursor_pos >= chunk.start_index && cursor_pos < chunk.end_index;
                            if has_cursor_in_chunk {
                                adjusted_cursor_pos = cursor_pos - chunk.start_index;
                                adjusted_cursor_pos = adjusted_cursor_pos.min(chunk.text.len());
                            }
                        }
                    }

                    if has_cursor_in_chunk {
                        layout_lines.push(LayoutLine {
                            text: chunk.text.clone(),
                            has_cursor: true,
                            cursor_pos: Some(adjusted_cursor_pos),
                        });
                    } else {
                        layout_lines.push(LayoutLine {
                            text: chunk.text.clone(),
                            has_cursor: false,
                            cursor_pos: None,
                        });
                    }
                }
            }
        }

        layout_lines
    }

    /// The full text, upstream `getText`.
    #[must_use]
    pub fn get_text(&self) -> String {
        self.state.borrow().lines.join("\n")
    }

    /// Replace paste markers with their stored content, upstream
    /// `expandPasteMarkers`.
    fn expand_paste_markers(&self, text: &str) -> String {
        let pastes = self.pastes.borrow();
        let mut result = String::new();
        let mut last_end = 0;
        for caps in PASTE_MARKER_RE.captures_iter(text) {
            let Some(whole) = caps.get(0) else {
                continue;
            };
            let id: u32 = caps[1].parse().unwrap_or(u32::MAX);
            let Some(content) = pastes.get(&id) else {
                continue;
            };
            result.push_str(&text[last_end..whole.start()]);
            result.push_str(content);
            last_end = whole.end();
        }
        result.push_str(&text[last_end..]);
        result
    }

    /// Get text with paste markers expanded to their actual content, upstream
    /// `getExpandedText`. Use this when you need the full content (e.g. for
    /// an external editor).
    #[must_use]
    pub fn get_expanded_text(&self) -> String {
        let joined = self.state.borrow().lines.join("\n");
        self.expand_paste_markers(&joined)
    }

    /// A copy of the logical lines, upstream `getLines`.
    #[must_use]
    pub fn get_lines(&self) -> Vec<String> {
        self.state.borrow().lines.clone()
    }

    /// The cursor position, upstream `getCursor`.
    #[must_use]
    pub fn get_cursor(&self) -> CursorPosition {
        let state = self.state.borrow();
        CursorPosition {
            line: state.cursor_line,
            col: state.cursor_col,
        }
    }

    /// Replace the whole text, upstream `setText`.
    pub fn set_text(&self, text: &str) {
        self.cancel_autocomplete();
        self.last_action.set(LastAction::None);
        self.exit_history_browsing();
        let normalized = self.normalize_text(text);
        // Push undo snapshot if content differs (makes programmatic changes
        // undoable)
        if self.get_text() != normalized {
            self.push_undo_snapshot();
        }
        self.pastes.borrow_mut().clear();
        self.paste_counter.set(0);
        self.set_text_internal(&normalized, CursorPlacement::End);
    }

    /// Insert text at the current cursor position, upstream
    /// `insertTextAtCursor`. Used for programmatic insertion (e.g. clipboard
    /// image markers). This is atomic for undo — a single undo restores the
    /// entire pre-insert state.
    pub fn insert_text_at_cursor(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.cancel_autocomplete();
        self.push_undo_snapshot();
        self.last_action.set(LastAction::None);
        self.exit_history_browsing();
        self.insert_text_at_cursor_internal(text);
    }

    /// Normalize text for editor storage, upstream `normalizeText`:
    /// `\r\n` and `\r` become `\n`, tabs become four spaces.
    #[expect(
        clippy::unused_self,
        reason = "upstream's normalizeText is an editor method; the port keeps the shape"
    )]
    fn normalize_text(&self, text: &str) -> String {
        text.replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\t', "    ")
    }

    /// Internal text insertion at cursor. Handles single and multi-line
    /// text. Does not push undo snapshots — the caller is responsible.
    /// Normalizes line endings and calls onChange once at the end, upstream
    /// `insertTextAtCursorInternal`.
    fn insert_text_at_cursor_internal(&self, text: &str) {
        if text.is_empty() {
            return;
        }

        // Normalize line endings and tabs
        let normalized = self.normalize_text(text);
        let inserted_lines: Vec<String> = normalized.split('\n').map(str::to_string).collect();

        let mut state = self.state.borrow_mut();
        let current_line = state.lines[state.cursor_line].clone();
        let before_cursor = current_line[..state.cursor_col].to_string();
        let after_cursor = current_line[state.cursor_col..].to_string();

        if inserted_lines.len() == 1 {
            // Single line - insert at cursor position
            let cursor_line = state.cursor_line;
            state.lines[cursor_line] =
                format!("{before_cursor}{}{after_cursor}", inserted_lines[0]);
            let new_col = state.cursor_col + normalized.len();
            drop(state);
            self.set_cursor_col(new_col);
        } else {
            // Multi-line insertion
            let mut new_lines: Vec<String> = Vec::new();
            new_lines.extend_from_slice(&state.lines[..state.cursor_line]);
            new_lines.push(format!("{before_cursor}{}", inserted_lines[0]));
            new_lines.extend_from_slice(&inserted_lines[1..inserted_lines.len() - 1]);
            new_lines.push(format!(
                "{}{after_cursor}",
                inserted_lines[inserted_lines.len() - 1]
            ));
            new_lines.extend_from_slice(&state.lines[state.cursor_line + 1..]);

            state.lines = new_lines;
            state.cursor_line += inserted_lines.len() - 1;
            let new_col = inserted_lines[inserted_lines.len() - 1].len();
            drop(state);
            self.set_cursor_col(new_col);
        }

        self.fire_on_change();
    }

    /// Insert one character at the cursor, upstream `insertCharacter`:
    /// fish-style undo coalescing (consecutive word chars coalesce into one
    /// undo unit; each space is separately undoable), then the insert, the
    /// change notification, and the autocomplete trigger tail.
    fn insert_character(&self, char: &str) {
        self.exit_history_browsing();

        if char.chars().any(is_whitespace_char) || self.last_action.get() != LastAction::TypeWord {
            self.push_undo_snapshot();
        }
        self.last_action.set(LastAction::TypeWord);

        {
            let mut state = self.state.borrow_mut();
            let cursor_line = state.cursor_line;
            let cursor_col = state.cursor_col;
            let line = &mut state.lines[cursor_line];
            let before = line[..cursor_col].to_string();
            let after = line[cursor_col..].to_string();
            *line = format!("{before}{char}{after}");
        }
        let new_col = self.state.borrow().cursor_col + char.len();
        self.set_cursor_col(new_col);

        self.fire_on_change();

        // Check if we should trigger or update autocomplete
        if self.autocomplete_state.get().is_none() {
            let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();
            let text_before_cursor =
                current_line[..self.state.borrow().cursor_col.min(current_line.len())].to_string();

            // Auto-trigger for "/" at the start of a line (slash commands)
            if char == "/" && self.is_at_start_of_message() {
                self.try_trigger_autocomplete(false);
            }
            // Auto-trigger for symbol-based completion like @, #, or
            // provider triggers at token boundaries
            else if self
                .autocomplete_trigger_characters
                .borrow()
                .iter()
                .any(|trigger| trigger == char)
            {
                let symbol_len = char.len();
                let before_symbol =
                    &text_before_cursor[..text_before_cursor.len().saturating_sub(symbol_len)];
                let char_before_symbol = before_symbol.chars().last();
                if before_symbol.is_empty()
                    || char_before_symbol == Some(' ')
                    || char_before_symbol == Some('\t')
                {
                    self.try_trigger_autocomplete(false);
                }
            }
            // Also auto-trigger when typing letters in a slash command or
            // symbol completion context
            else if char
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || ".-_".contains(character))
                && !char.is_empty()
            {
                // A slash command (with or without arguments) or a
                // symbol-trigger (@, #, provider) context
                if self.is_in_slash_command_context(&text_before_cursor)
                    || self
                        .autocomplete_trigger_pattern
                        .borrow()
                        .as_ref()
                        .is_some_and(|pattern| pattern.is_match(&text_before_cursor))
                {
                    self.try_trigger_autocomplete(false);
                }
            }
        } else {
            self.update_autocomplete();
        }
    }

    /// Route a bracketed paste through the editor, upstream `handlePaste`:
    /// control-byte decoding, cleaning, the file-path space, and the
    /// large-paste marker registry.
    fn handle_paste(&self, pasted_text: &str) {
        self.cancel_autocomplete();
        self.exit_history_browsing();
        self.last_action.set(LastAction::None);

        self.push_undo_snapshot();

        // Some terminals (e.g. tmux popups with extended-keys-format=csi-u)
        // re-encode control bytes inside bracketed paste as CSI-u
        // Ctrl+<letter> sequences (ESC [ <codepoint> ; 5 u). Decode those
        // back to their literal byte so the per-char filter below preserves
        // newlines instead of stripping ESC and leaking the printable tail
        // (e.g. "[106;5u") into the editor.
        let decoded_text = CSI_U_CTRL_RE.replace_all(pasted_text, |caps: &Captures| {
            let Ok(codepoint) = caps[1].parse::<u32>() else {
                return caps[0].to_string();
            };
            if (97..=122).contains(&codepoint) {
                return char::from_u32(codepoint - 96)
                    .map_or_else(|| caps[0].to_string(), String::from);
            }
            if (65..=90).contains(&codepoint) {
                return char::from_u32(codepoint - 64)
                    .map_or_else(|| caps[0].to_string(), String::from);
            }
            caps[0].to_string()
        });

        // Clean the pasted text: normalize line endings, expand tabs
        let clean_text = self.normalize_text(&decoded_text);

        // Filter out non-printable characters except newlines
        let filtered_text: String = clean_text
            .chars()
            .filter(|char| *char == '\n' || u32::from(*char) >= 32)
            .collect();

        // If pasting a file path (starts with /, ~, or .) and the character
        // before the cursor is a word character, prepend a space for better
        // readability
        let mut filtered_text = filtered_text;
        if filtered_text.starts_with('/')
            || filtered_text.starts_with('~')
            || filtered_text.starts_with('.')
        {
            let state = self.state.borrow();
            let current_line = &state.lines[state.cursor_line];
            let char_before_cursor = state.cursor_col > 0
                && current_line[..state.cursor_col]
                    .chars()
                    .last()
                    .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_');
            drop(state);
            if char_before_cursor {
                filtered_text = format!(" {filtered_text}");
            }
        }

        // Split into lines to check for large paste
        let pasted_lines: Vec<&str> = filtered_text.split('\n').collect();

        // Check if this is a large paste (> 10 lines or > 1000 characters)
        let total_chars = filtered_text.len();
        if pasted_lines.len() > 10 || total_chars > 1000 {
            // Store the paste and insert a marker
            let paste_id = self.paste_counter.get() + 1;
            self.paste_counter.set(paste_id);
            self.pastes
                .borrow_mut()
                .insert(paste_id, filtered_text.clone());

            // Insert marker like "[paste #1 +123 lines]" or
            // "[paste #1 1234 chars]"
            let marker = if pasted_lines.len() > 10 {
                format!("[paste #{paste_id} +{} lines]", pasted_lines.len())
            } else {
                format!("[paste #{paste_id} {total_chars} chars]")
            };
            self.insert_text_at_cursor_internal(&marker);
            return;
        }

        // Single and multi-line small pastes insert atomically (no
        // autocomplete trigger during paste)
        self.insert_text_at_cursor_internal(&filtered_text);
    }

    /// Split the current line at the cursor, upstream `addNewLine`.
    fn add_new_line(&self) {
        self.cancel_autocomplete();
        self.exit_history_browsing();
        self.last_action.set(LastAction::None);

        self.push_undo_snapshot();

        let mut state = self.state.borrow_mut();
        let current_line = state.lines[state.cursor_line].clone();

        let before = current_line[..state.cursor_col].to_string();
        let after = current_line[state.cursor_col..].to_string();

        // Split current line
        let cursor_line = state.cursor_line;
        state.lines[cursor_line] = before;
        state.lines.insert(cursor_line + 1, after);

        // Move cursor to start of new line
        state.cursor_line = cursor_line + 1;
        drop(state);
        self.set_cursor_col(0);

        self.fire_on_change();
    }

    /// Whether a backslash directly before the cursor converts Enter into a
    /// newline — the Shift+Enter workaround, upstream
    /// `shouldSubmitOnBackslashEnter`.
    fn should_submit_on_backslash_enter(&self, data: &str) -> bool {
        if self.disable_submit.get() {
            return false;
        }
        if !self.parser.matches_key(data, "enter") {
            return false;
        }
        let submit_keys = get_keybindings().get_keys("tui.input.submit");
        let has_shift_enter = submit_keys
            .iter()
            .any(|key| key == "shift+enter" || key == "shift+return");
        if !has_shift_enter {
            return false;
        }

        let state = self.state.borrow();
        let current_line = &state.lines[state.cursor_line];
        state.cursor_col > 0
            && current_line
                .get(state.cursor_col - 1..state.cursor_col)
                .is_some_and(|char| char == "\\")
    }

    /// Submit the current text, upstream `submitValue`: paste markers
    /// expand, the editor resets to empty, and undo history clears.
    fn submit_value(&self) {
        self.cancel_autocomplete();
        let joined = self.state.borrow().lines.join("\n");
        let result = self.expand_paste_markers(&joined).trim().to_string();

        *self.state.borrow_mut() = EditorState {
            lines: vec![String::new()],
            cursor_line: 0,
            cursor_col: 0,
        };
        self.pastes.borrow_mut().clear();
        self.paste_counter.set(0);
        self.exit_history_browsing();
        self.scroll_offset.set(0);
        self.undo_stack.borrow_mut().clear();
        self.last_action.set(LastAction::None);

        let on_change = self.on_change.borrow().clone();
        if let Some(on_change) = on_change {
            on_change("");
        }
        let on_submit = self.on_submit.borrow().clone();
        if let Some(on_submit) = on_submit {
            on_submit(&result);
        }
    }

    /// Delete one grapheme before the cursor (or merge with the previous
    /// line at column 0), upstream `handleBackspace`, including the
    /// paste-marker registry shift when the deleted grapheme is a marker,
    /// and the autocomplete update-or-retrigger tail.
    fn handle_backspace(&self) {
        self.exit_history_browsing();
        self.last_action.set(LastAction::None);

        let cursor_col = self.state.borrow().cursor_col;
        let cursor_line = self.state.borrow().cursor_line;

        if cursor_col > 0 {
            self.push_undo_snapshot();

            // Delete grapheme before cursor (handles emojis, combining
            // characters, etc.)
            let before_cursor = self.state.borrow().lines[cursor_line][..cursor_col].to_string();

            // Find the last grapheme in the text before cursor
            let last_grapheme = self
                .segment(&before_cursor, SegmentMode::Grapheme)
                .pop()
                .map(|segment| segment.segment);
            let grapheme_length = last_grapheme.as_deref().map_or(1, str::len);
            let is_paste_marker_segment = last_grapheme
                .as_deref()
                .is_some_and(|segment| PASTE_MARKER_SINGLE_RE.is_match(segment));

            if is_paste_marker_segment {
                // This contains the id part, e.g. 4 from
                // [paste #4 +123 lines]
                let target_id: u32 = last_grapheme
                    .as_deref()
                    .and_then(|segment| PASTE_MARKER_SINGLE_RE.captures(segment))
                    .and_then(|caps| caps[1].parse().ok())
                    .unwrap_or(u32::MAX);
                self.pastes.borrow_mut().remove(&target_id);
                self.paste_counter
                    .set(self.paste_counter.get().saturating_sub(1));

                // Shift registry entries down in ascending id order,
                // independent of marker order in the text
                // ([paste #3] becomes [paste #2] when [paste #1] is
                // removed).
                let mut higher_ids: Vec<u32> = self
                    .pastes
                    .borrow()
                    .keys()
                    .copied()
                    .filter(|id| *id > target_id)
                    .collect();
                higher_ids.sort_unstable();
                for id in higher_ids {
                    let content = self.pastes.borrow_mut().remove(&id);
                    if let Some(content) = content {
                        self.pastes.borrow_mut().insert(id - 1, content);
                    }
                }

                // Renumber markers with ids greater than the removed one.
                let mut state = self.state.borrow_mut();
                for line in &mut state.lines {
                    *line = renumber_paste_markers(line, target_id);
                }
                drop(state);
            }

            let line = self.state.borrow().lines[cursor_line].clone();
            let merged = format!(
                "{}{}",
                &line[..cursor_col - grapheme_length],
                &line[cursor_col..]
            );
            self.state.borrow_mut().lines[cursor_line] = merged;
            self.set_cursor_col(cursor_col - grapheme_length);
        } else if cursor_line > 0 {
            self.push_undo_snapshot();

            // Merge with previous line
            let current_line = self.state.borrow().lines[cursor_line].clone();
            let previous_line = self.state.borrow().lines[cursor_line - 1].clone();

            {
                let mut state = self.state.borrow_mut();
                state.lines[cursor_line - 1] = format!("{previous_line}{current_line}");
                state.lines.remove(cursor_line);
                state.cursor_line -= 1;
            }
            self.set_cursor_col(previous_line.len());
        }

        self.fire_on_change();

        // Update or re-trigger autocomplete after backspace
        if self.autocomplete_state.get().is_some() {
            self.update_autocomplete();
        } else {
            // If autocomplete was cancelled (no matches), re-trigger if
            // we're in a completable context
            let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();
            let text_before_cursor =
                current_line[..self.state.borrow().cursor_col.min(current_line.len())].to_string();
            // Slash-command or symbol-trigger (@, #, provider) context
            if self.is_in_slash_command_context(&text_before_cursor)
                || self
                    .autocomplete_trigger_pattern
                    .borrow()
                    .as_ref()
                    .is_some_and(|pattern| pattern.is_match(&text_before_cursor))
            {
                self.try_trigger_autocomplete(false);
            }
        }
    }

    /// Set the cursor column and clear the sticky-column state, upstream
    /// `setCursorCol`. Use this for all non-vertical cursor movements to
    /// reset sticky column behavior.
    fn set_cursor_col(&self, col: usize) {
        let mut state = self.state.borrow_mut();
        state.cursor_col = col;
        drop(state);
        self.preferred_visual_col.set(None);
        self.snapped_from_cursor_col.set(None);
    }

    /// Move the cursor onto a target visual line with sticky-column logic,
    /// upstream `moveToVisualLine`. Shared by [`Editor::move_cursor`] and
    /// [`Editor::page_scroll`].
    fn move_to_visual_line(
        &self,
        visual_lines: &[VisualLine],
        current_visual_line: usize,
        target_visual_line: usize,
    ) {
        let Some(current_vl) = visual_lines.get(current_visual_line) else {
            return;
        };
        let Some(target_vl) = visual_lines.get(target_visual_line) else {
            return;
        };

        // When the cursor was snapped to a segment start, resolve the
        // pre-snap position against the VL it belongs to. This gives the
        // correct visual column even after a resize reshuffles VLs.
        let current_visual_col = self.snapped_from_cursor_col.get().map_or_else(
            || {
                self.state
                    .borrow()
                    .cursor_col
                    .saturating_sub(current_vl.start_col)
            },
            |snapped_from| {
                let vl_index =
                    self.find_visual_line_at(visual_lines, current_vl.logical_line, snapped_from);
                snapped_from.saturating_sub(visual_lines[vl_index].start_col)
            },
        );

        // For non-last segments, clamp to length-1 to stay within the segment
        let is_last_source_segment = current_visual_line == visual_lines.len() - 1
            || visual_lines
                .get(current_visual_line + 1)
                .is_none_or(|next| next.logical_line != current_vl.logical_line);
        let source_max_visual_col = if is_last_source_segment {
            current_vl.length
        } else {
            current_vl.length.saturating_sub(1)
        };

        let is_last_target_segment = target_visual_line == visual_lines.len() - 1
            || visual_lines
                .get(target_visual_line + 1)
                .is_none_or(|next| next.logical_line != target_vl.logical_line);
        let target_max_visual_col = if is_last_target_segment {
            target_vl.length
        } else {
            target_vl.length.saturating_sub(1)
        };

        let move_to_visual_col = self.compute_vertical_move_column(
            current_visual_col,
            source_max_visual_col,
            target_max_visual_col,
        );

        // Set cursor position
        let logical_line = self.state.borrow().lines[target_vl.logical_line].clone();
        let target_col = target_vl.start_col + move_to_visual_col;
        let new_col = target_col.min(logical_line.len());
        {
            let mut state = self.state.borrow_mut();
            state.cursor_line = target_vl.logical_line;
            state.cursor_col = new_col;
        }

        // Snap cursor to atomic segment boundary (e.g. paste markers) so the
        // cursor never lands in the middle of a multi-grapheme unit.
        // Single-grapheme segments don't need snapping.
        let segments = self.segment(&logical_line, SegmentMode::Grapheme);
        for seg in &segments {
            if seg.index > new_col {
                break;
            }
            if grapheme_segments(&seg.segment).count() <= 1 {
                continue;
            }
            if new_col < seg.index + seg.segment.len() {
                let is_continuation = seg.index < target_vl.start_col;
                let is_moving_down = target_visual_line > current_visual_line;

                if is_continuation && is_moving_down {
                    // The segment started on a previous visual line, and we
                    // already visited it on the way down. Skip all remaining
                    // continuation VLs and land on the first VL past it.
                    let segment_end = seg.index + seg.segment.len();
                    let mut next = target_visual_line + 1;
                    while next < visual_lines.len()
                        && visual_lines[next].logical_line == target_vl.logical_line
                        && visual_lines[next].start_col < segment_end
                    {
                        next += 1;
                    }
                    if next < visual_lines.len() {
                        self.move_to_visual_line(visual_lines, current_visual_line, next);
                        return;
                    }
                }

                // Snap to the start of the segment so it gets highlighted.
                // Store the pre-snap position so the next vertical move can
                // resolve it to the correct visual column.
                self.snapped_from_cursor_col.set(Some(new_col));
                self.state.borrow_mut().cursor_col = seg.index;
                return;
            }
        }

        // No snap occurred – we moved out of the atomic segment.
        self.snapped_from_cursor_col.set(None);
    }

    /// Compute the target visual column for vertical cursor movement,
    /// upstream `computeVerticalMoveColumn`'s sticky-column decision table:
    ///
    /// | P | S | T | U | Scenario                                             | Set Preferred | Move To     |
    /// |---|---|---|---| ---------------------------------------------------- |---------------|-------------|
    /// | 0 | * | 0 | - | Start nav, target fits                               | null          | current     |
    /// | 0 | * | 1 | - | Start nav, target shorter                            | current       | target end  |
    /// | 1 | 0 | 0 | 0 | Clamped, target fits preferred                       | null          | preferred   |
    /// | 1 | 0 | 0 | 1 | Clamped, target longer but still can't fit preferred | keep          | target end  |
    /// | 1 | 0 | 1 | - | Clamped, target even shorter                         | keep          | target end  |
    /// | 1 | 1 | 0 | - | Rewrapped, target fits current                       | null          | current     |
    /// | 1 | 1 | 1 | - | Rewrapped, target shorter than current               | current       | target end  |
    ///
    /// Where:
    /// - P = preferred col is set
    /// - S = cursor in middle of source line (not clamped to end)
    /// - T = target line shorter than current visual col
    /// - U = target line shorter than preferred col
    fn compute_vertical_move_column(
        &self,
        current_visual_col: usize,
        source_max_visual_col: usize,
        target_max_visual_col: usize,
    ) -> usize {
        let has_preferred = self.preferred_visual_col.get().is_some(); // P
        let cursor_in_middle = current_visual_col < source_max_visual_col; // S
        let target_too_short = target_max_visual_col < current_visual_col; // T

        if !has_preferred || cursor_in_middle {
            if target_too_short {
                // Cases 2 and 7
                self.preferred_visual_col.set(Some(current_visual_col));
                return target_max_visual_col;
            }

            // Cases 1 and 6
            self.preferred_visual_col.set(None);
            return current_visual_col;
        }

        let target_cant_fit_preferred =
            target_max_visual_col < self.preferred_visual_col.get().unwrap_or(0); // U
        if target_too_short || target_cant_fit_preferred {
            // Cases 4 and 5
            return target_max_visual_col;
        }

        // Case 3
        let result = self.preferred_visual_col.get().unwrap_or(0);
        self.preferred_visual_col.set(None);
        result
    }

    fn move_to_line_start(&self) {
        self.last_action.set(LastAction::None);
        self.set_cursor_col(0);
    }

    fn move_to_line_end(&self) {
        self.last_action.set(LastAction::None);
        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();
        self.set_cursor_col(current_line.len());
    }

    /// Delete from line start to the cursor (or merge with the previous line
    /// at column 0), pushing the deleted text onto the kill ring, upstream
    /// `deleteToStartOfLine`.
    fn delete_to_start_of_line(&self) {
        self.exit_history_browsing();

        let state = self.state.borrow();
        let current_line = state.lines[state.cursor_line].clone();
        drop(state);

        if self.state.borrow().cursor_col > 0 {
            self.push_undo_snapshot();

            // Calculate text to be deleted and save to kill ring (backward
            // deletion = prepend)
            let deleted_text = current_line[..self.state.borrow().cursor_col].to_string();
            self.kill_ring.borrow_mut().push(
                &deleted_text,
                KillRingPushOptions {
                    prepend: true,
                    accumulate: self.last_action.get() == LastAction::Kill,
                },
            );
            self.last_action.set(LastAction::Kill);

            // Delete from start of line up to cursor
            let new_line = current_line[self.state.borrow().cursor_col..].to_string();
            let cursor_line = self.state.borrow().cursor_line;
            let mut state = self.state.borrow_mut();
            state.lines[cursor_line] = new_line;
            drop(state);
            self.set_cursor_col(0);
        } else if self.state.borrow().cursor_line > 0 {
            self.push_undo_snapshot();

            // At start of line - merge with previous line, treating newline
            // as deleted text
            self.kill_ring.borrow_mut().push(
                "\n",
                KillRingPushOptions {
                    prepend: true,
                    accumulate: self.last_action.get() == LastAction::Kill,
                },
            );
            self.last_action.set(LastAction::Kill);

            let mut state = self.state.borrow_mut();
            let previous_line = state.lines[state.cursor_line - 1].clone();
            let cursor_line = state.cursor_line;
            state.lines[cursor_line - 1] = format!("{previous_line}{current_line}");
            state.lines.remove(cursor_line);
            state.cursor_line -= 1;
            drop(state);
            self.set_cursor_col(previous_line.len());
        }

        self.fire_on_change();
    }

    /// Delete from the cursor to line end (or merge with the next line at
    /// line end), pushing the deleted text onto the kill ring, upstream
    /// `deleteToEndOfLine`.
    fn delete_to_end_of_line(&self) {
        self.exit_history_browsing();

        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();

        if self.state.borrow().cursor_col < current_line.len() {
            self.push_undo_snapshot();

            // Calculate text to be deleted and save to kill ring (forward
            // deletion = append)
            let deleted_text = current_line[self.state.borrow().cursor_col..].to_string();
            self.kill_ring.borrow_mut().push(
                &deleted_text,
                KillRingPushOptions {
                    prepend: false,
                    accumulate: self.last_action.get() == LastAction::Kill,
                },
            );
            self.last_action.set(LastAction::Kill);

            // Delete from cursor to end of line
            let cursor_col = self.state.borrow().cursor_col;
            let cursor_line = self.state.borrow().cursor_line;
            self.state.borrow_mut().lines[cursor_line] = current_line[..cursor_col].to_string();
        } else if self.state.borrow().cursor_line < self.state.borrow().lines.len() - 1 {
            self.push_undo_snapshot();

            // At end of line - merge with next line, treating newline as
            // deleted text
            self.kill_ring.borrow_mut().push(
                "\n",
                KillRingPushOptions {
                    prepend: false,
                    accumulate: self.last_action.get() == LastAction::Kill,
                },
            );
            self.last_action.set(LastAction::Kill);

            let next_line = self.state.borrow().lines[self.state.borrow().cursor_line + 1].clone();
            let cursor_line = self.state.borrow().cursor_line;
            let mut state = self.state.borrow_mut();
            state.lines[cursor_line] = format!("{current_line}{next_line}");
            state.lines.remove(cursor_line + 1);
        }

        self.fire_on_change();
    }

    /// Delete one word backward, upstream `deleteWordBackwards`: at column 0
    /// this behaves like backspace (merge with the previous line).
    fn delete_word_backwards(&self) {
        self.exit_history_browsing();

        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();

        // If at start of line, behave like backspace at column 0 (merge with
        // previous line)
        if self.state.borrow().cursor_col == 0 {
            if self.state.borrow().cursor_line > 0 {
                self.push_undo_snapshot();

                // Treat newline as deleted text (backward deletion = prepend)
                self.kill_ring.borrow_mut().push(
                    "\n",
                    KillRingPushOptions {
                        prepend: true,
                        accumulate: self.last_action.get() == LastAction::Kill,
                    },
                );
                self.last_action.set(LastAction::Kill);

                let mut state = self.state.borrow_mut();
                let previous_line = state.lines[state.cursor_line - 1].clone();
                let cursor_line = state.cursor_line;
                state.lines[cursor_line - 1] = format!("{previous_line}{current_line}");
                state.lines.remove(cursor_line);
                state.cursor_line -= 1;
                drop(state);
                self.set_cursor_col(previous_line.len());
            }
        } else {
            self.push_undo_snapshot();

            // Save lastAction before cursor movement (moveWordBackwards resets
            // it)
            let was_kill = self.last_action.get() == LastAction::Kill;

            let old_cursor_col = self.state.borrow().cursor_col;
            self.move_word_backwards();
            let delete_from = self.state.borrow().cursor_col;
            self.set_cursor_col(old_cursor_col);

            let deleted_text = current_line[delete_from..old_cursor_col].to_string();
            self.kill_ring.borrow_mut().push(
                &deleted_text,
                KillRingPushOptions {
                    prepend: true,
                    accumulate: was_kill,
                },
            );
            self.last_action.set(LastAction::Kill);

            let cursor_line = self.state.borrow().cursor_line;
            let mut state = self.state.borrow_mut();
            state.lines[cursor_line] = format!(
                "{}{}",
                &current_line[..delete_from],
                &current_line[old_cursor_col..]
            );
            drop(state);
            self.set_cursor_col(delete_from);
        }

        self.fire_on_change();
    }

    /// Delete one word forward, upstream `deleteWordForward`: at line end
    /// this merges with the next line (deleting the newline).
    fn delete_word_forward(&self) {
        self.exit_history_browsing();

        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();

        // If at end of line, merge with next line (delete the newline)
        if self.state.borrow().cursor_col >= current_line.len() {
            if self.state.borrow().cursor_line < self.state.borrow().lines.len() - 1 {
                self.push_undo_snapshot();

                // Treat newline as deleted text (forward deletion = append)
                self.kill_ring.borrow_mut().push(
                    "\n",
                    KillRingPushOptions {
                        prepend: false,
                        accumulate: self.last_action.get() == LastAction::Kill,
                    },
                );
                self.last_action.set(LastAction::Kill);

                let next_line =
                    self.state.borrow().lines[self.state.borrow().cursor_line + 1].clone();
                let cursor_line = self.state.borrow().cursor_line;
                let mut state = self.state.borrow_mut();
                state.lines[cursor_line] = format!("{current_line}{next_line}");
                state.lines.remove(cursor_line + 1);
            }
        } else {
            self.push_undo_snapshot();

            // Save lastAction before cursor movement (moveWordForwards resets
            // it)
            let was_kill = self.last_action.get() == LastAction::Kill;

            let old_cursor_col = self.state.borrow().cursor_col;
            self.move_word_forwards();
            let delete_to = self.state.borrow().cursor_col;
            self.set_cursor_col(old_cursor_col);

            let deleted_text = current_line[old_cursor_col..delete_to].to_string();
            self.kill_ring.borrow_mut().push(
                &deleted_text,
                KillRingPushOptions {
                    prepend: false,
                    accumulate: was_kill,
                },
            );
            self.last_action.set(LastAction::Kill);

            let cursor_line = self.state.borrow().cursor_line;
            let mut state = self.state.borrow_mut();
            state.lines[cursor_line] = format!(
                "{}{}",
                &current_line[..old_cursor_col],
                &current_line[delete_to..]
            );
        }

        self.fire_on_change();
    }

    /// Delete one grapheme at the cursor (or merge with the next line at
    /// line end), upstream `handleForwardDelete`.
    /// Delete one grapheme after the cursor (or merge with the next line at
    /// line end), upstream `handleForwardDelete`, with the same
    /// autocomplete update-or-retrigger tail as backspace.
    fn handle_forward_delete(&self) {
        self.exit_history_browsing();
        self.last_action.set(LastAction::None);

        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();

        if self.state.borrow().cursor_col < current_line.len() {
            self.push_undo_snapshot();

            // Delete grapheme at cursor position (handles emojis, combining
            // characters, etc.)
            let after_cursor = current_line[self.state.borrow().cursor_col..].to_string();

            // Find the first grapheme at cursor
            let first_grapheme = self
                .segment(&after_cursor, SegmentMode::Grapheme)
                .into_iter()
                .next()
                .map(|segment| segment.segment);
            let grapheme_length = first_grapheme.as_deref().map_or(1, str::len);

            let cursor_col = self.state.borrow().cursor_col;
            let before = current_line[..cursor_col].to_string();
            let after_end = (cursor_col + grapheme_length).min(current_line.len());
            let after = current_line[after_end..].to_string();
            let cursor_line = self.state.borrow().cursor_line;
            self.state.borrow_mut().lines[cursor_line] = format!("{before}{after}");
        } else if self.state.borrow().cursor_line < self.state.borrow().lines.len() - 1 {
            self.push_undo_snapshot();

            // At end of line - merge with next line
            let next_line = self.state.borrow().lines[self.state.borrow().cursor_line + 1].clone();
            let cursor_line = self.state.borrow().cursor_line;
            let mut state = self.state.borrow_mut();
            state.lines[cursor_line] = format!("{current_line}{next_line}");
            state.lines.remove(cursor_line + 1);
        }

        self.fire_on_change();

        // Update or re-trigger autocomplete after forward delete
        if self.autocomplete_state.get().is_some() {
            self.update_autocomplete();
        } else {
            let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();
            let text_before_cursor =
                current_line[..self.state.borrow().cursor_col.min(current_line.len())].to_string();
            // Slash-command or symbol-trigger (@, #, provider) context
            if self.is_in_slash_command_context(&text_before_cursor)
                || self
                    .autocomplete_trigger_pattern
                    .borrow()
                    .as_ref()
                    .is_some_and(|pattern| pattern.is_match(&text_before_cursor))
            {
                self.try_trigger_autocomplete(false);
            }
        }
    }

    /// Map visual lines to logical positions, upstream
    /// `buildVisualLineMap`: each entry carries the logical line index, the
    /// byte start column, and the byte length of the segment.
    fn build_visual_line_map(&self, width: usize) -> Vec<VisualLine> {
        let mut visual_lines: Vec<VisualLine> = Vec::new();

        let state = self.state.borrow();
        for (index, line) in state.lines.iter().enumerate() {
            let line_vis_width = visible_width(line);
            if line.is_empty() {
                // Empty line still takes one visual line
                visual_lines.push(VisualLine {
                    logical_line: index,
                    start_col: 0,
                    length: 0,
                });
            } else if line_vis_width <= width {
                visual_lines.push(VisualLine {
                    logical_line: index,
                    start_col: 0,
                    length: line.len(),
                });
            } else {
                // Line needs wrapping - use word-aware wrapping
                let pre_segmented = self.segment(line, SegmentMode::Grapheme);
                let chunks = word_wrap_line(line, width, Some(&pre_segmented));
                for chunk in &chunks {
                    visual_lines.push(VisualLine {
                        logical_line: index,
                        start_col: chunk.start_index,
                        length: chunk.end_index - chunk.start_index,
                    });
                }
            }
        }

        visual_lines
    }

    /// Find the visual line index that contains the given logical position,
    /// upstream `findVisualLineAt`.
    #[expect(
        clippy::unused_self,
        reason = "upstream's findVisualLineAt is an editor method; the port keeps the shape"
    )]
    fn find_visual_line_at(&self, visual_lines: &[VisualLine], line: usize, col: usize) -> usize {
        for (index, vl) in visual_lines.iter().enumerate() {
            if vl.logical_line != line || col < vl.start_col {
                continue;
            }
            let offset = col - vl.start_col;
            // Cursor is in this segment if it's within range. For the last
            // segment of a logical line, cursor can be at length (end
            // position)
            let is_last_segment_of_line = index == visual_lines.len() - 1
                || visual_lines
                    .get(index + 1)
                    .is_none_or(|next| next.logical_line != vl.logical_line);
            if offset < vl.length || (is_last_segment_of_line && offset == vl.length) {
                return index;
            }
        }
        visual_lines.len().saturating_sub(1)
    }

    /// Find the visual line index for the current cursor position, upstream
    /// `findCurrentVisualLine`.
    fn find_current_visual_line(&self, visual_lines: &[VisualLine]) -> usize {
        let state = self.state.borrow();
        self.find_visual_line_at(visual_lines, state.cursor_line, state.cursor_col)
    }

    /// Move the cursor by whole lines and/or one grapheme horizontally,
    /// upstream `moveCursor`, with the wrap-to-neighbor-line behavior at
    /// line boundaries.
    fn move_cursor(&self, delta_line: i32, delta_col: i32) {
        self.last_action.set(LastAction::None);
        let visual_lines = self.build_visual_line_map(self.last_width.get());
        let current_visual_line = self.find_current_visual_line(&visual_lines);

        if delta_line != 0 {
            let current = i64::try_from(current_visual_line).unwrap_or(i64::MAX);
            let target = current + i64::from(delta_line);
            let lines_len = i64::try_from(visual_lines.len()).unwrap_or(i64::MAX);
            if (0..lines_len).contains(&target) {
                let target = usize::try_from(target).unwrap_or_default();
                self.move_to_visual_line(&visual_lines, current_visual_line, target);
            }
        }

        if delta_col != 0 {
            let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();

            if delta_col > 0 {
                // Moving right - move by one grapheme (handles emojis,
                // combining characters, etc.)
                if self.state.borrow().cursor_col < current_line.len() {
                    let after_cursor = current_line[self.state.borrow().cursor_col..].to_string();
                    let first_grapheme = self
                        .segment(&after_cursor, SegmentMode::Grapheme)
                        .into_iter()
                        .next()
                        .map(|segment| segment.segment);
                    let step = first_grapheme.as_deref().map_or(1, str::len);
                    let new_col = self.state.borrow().cursor_col + step;
                    self.set_cursor_col(new_col);
                } else if self.state.borrow().cursor_line < self.state.borrow().lines.len() - 1 {
                    // Wrap to start of next logical line
                    self.state.borrow_mut().cursor_line += 1;
                    self.set_cursor_col(0);
                } else {
                    // At end of last line - can't move, but record the
                    // preferred visual col for up/down navigation
                    if let Some(current_vl) = visual_lines.get(current_visual_line) {
                        self.preferred_visual_col.set(Some(
                            self.state
                                .borrow()
                                .cursor_col
                                .saturating_sub(current_vl.start_col),
                        ));
                    }
                }
            } else {
                // Moving left - move by one grapheme (handles emojis,
                // combining characters, etc.)
                if self.state.borrow().cursor_col > 0 {
                    let before_cursor = current_line[..self.state.borrow().cursor_col].to_string();
                    let last_grapheme = self
                        .segment(&before_cursor, SegmentMode::Grapheme)
                        .pop()
                        .map(|segment| segment.segment);
                    let step = last_grapheme.as_deref().map_or(1, str::len);
                    let new_col = self.state.borrow().cursor_col - step;
                    self.set_cursor_col(new_col);
                } else if self.state.borrow().cursor_line > 0 {
                    // Wrap to end of previous logical line
                    self.state.borrow_mut().cursor_line -= 1;
                    let prev_line =
                        self.state.borrow().lines[self.state.borrow().cursor_line].clone();
                    self.set_cursor_col(prev_line.len());
                }
            }
        }

        // Keep an open autocomplete picker in sync with the new cursor
        // position: cursor movement changes the text before the cursor, so
        // a picker computed for the old position is stale. Re-query so it
        // refreshes — or closes when the new position yields no
        // suggestions — mirroring insertCharacter()/handleBackspace().
        // Without this, arrowing left from `/cmd ` back into the command
        // name leaves the argument picker showing against a `/cmd` prefix
        // (and a Tab there would concatenate the stale suggestion onto the
        // partial command name).
        if self.autocomplete_state.get().is_some() {
            self.update_autocomplete();
        }
    }

    /// Scroll by a page (direction: -1 for up, 1 for down), upstream
    /// `pageScroll`. Moves cursor by the page size while keeping it in
    /// bounds.
    fn page_scroll(&self, direction: i32) {
        self.last_action.set(LastAction::None);
        let terminal_rows = usize::from(self.terminal_rows());
        let page_size = 5.max(terminal_rows * 30 / 100);

        let visual_lines = self.build_visual_line_map(self.last_width.get());
        let current_visual_line = self.find_current_visual_line(&visual_lines);
        let current = i64::try_from(current_visual_line).unwrap_or(i64::MAX);
        let page_size = i64::try_from(page_size).unwrap_or(i64::MAX);
        let last_line = i64::try_from(visual_lines.len().saturating_sub(1)).unwrap_or(i64::MAX);
        let target_visual_line =
            usize::try_from((current + i64::from(direction) * page_size).clamp(0, last_line))
                .unwrap_or_default();

        self.move_to_visual_line(&visual_lines, current_visual_line, target_visual_line);
    }

    /// Move one word backward, upstream `moveWordBackwards`: at column 0
    /// this moves to the end of the previous line.
    fn move_word_backwards(&self) {
        self.last_action.set(LastAction::None);
        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();

        // If at start of line, move to end of previous line
        if self.state.borrow().cursor_col == 0 {
            if self.state.borrow().cursor_line > 0 {
                self.state.borrow_mut().cursor_line -= 1;
                let prev_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();
                self.set_cursor_col(prev_line.len());
            }
            return;
        }

        let segment_fn = |text: &str| self.segment(text, SegmentMode::Word);
        let atomic_fn = |segment: &str| is_paste_marker(segment);
        let options = WordNavigationOptions {
            segment: Some(&segment_fn),
            is_atomic_segment: Some(&atomic_fn),
        };
        let cursor_col = self.state.borrow().cursor_col;
        let new_col = find_word_backward(&current_line, cursor_col, Some(&options));
        self.set_cursor_col(new_col);
    }

    /// Yank (paste) the most recent kill ring entry at cursor position,
    /// upstream `yank`.
    fn yank(&self) {
        if self.kill_ring.borrow().is_empty() {
            return;
        }

        self.push_undo_snapshot();

        let text = self
            .kill_ring
            .borrow()
            .peek()
            .unwrap_or_default()
            .to_string();
        self.insert_yanked_text(&text);

        self.last_action.set(LastAction::Yank);
    }

    /// Cycle through kill ring (only works immediately after yank or
    /// yank-pop), upstream `yankPop`. Replaces the last yanked text with the
    /// previous entry in the ring.
    fn yank_pop(&self) {
        // Only works if we just yanked and have more than one entry
        if self.last_action.get() != LastAction::Yank || self.kill_ring.borrow().len() <= 1 {
            return;
        }

        self.push_undo_snapshot();

        // Delete the previously yanked text (still at end of ring before
        // rotation)
        self.delete_yanked_text();

        // Rotate the ring: move end to front
        self.kill_ring.borrow_mut().rotate();

        // Insert the new most recent entry (now at end after rotation)
        let text = self
            .kill_ring
            .borrow()
            .peek()
            .unwrap_or_default()
            .to_string();
        self.insert_yanked_text(&text);

        self.last_action.set(LastAction::Yank);
    }

    /// Insert text at cursor position (used by yank operations), upstream
    /// `insertYankedText`.
    fn insert_yanked_text(&self, text: &str) {
        self.exit_history_browsing();
        let lines: Vec<String> = text.split('\n').map(str::to_string).collect();

        let mut state = self.state.borrow_mut();
        let cursor_col = state.cursor_col;
        let cursor_line = state.cursor_line;
        let current_line = state.lines[cursor_line].clone();
        let before = current_line[..cursor_col].to_string();
        let after = current_line[cursor_col..].to_string();
        if lines.len() == 1 {
            // Single line - insert at cursor
            state.lines[cursor_line] = format!("{before}{}{after}", lines[0]);
            let new_col = cursor_col + text.len();
            drop(state);
            self.set_cursor_col(new_col);
        } else {
            // Multi-line insert

            // First line merges with text before cursor
            state.lines[cursor_line] = format!("{before}{}", lines[0]);

            // Insert middle lines
            for (offset, line) in lines[1..lines.len() - 1].iter().enumerate() {
                state.lines.insert(cursor_line + 1 + offset, line.clone());
            }

            // Last line merges with text after cursor
            let last_line_index = cursor_line + lines.len() - 1;
            state.lines.insert(
                last_line_index,
                format!("{}{after}", lines[lines.len() - 1]),
            );

            // Update cursor position
            state.cursor_line = last_line_index;
            let new_col = lines[lines.len() - 1].len();
            drop(state);
            self.set_cursor_col(new_col);
        }

        self.fire_on_change();
    }

    /// Delete the previously yanked text (used by yank-pop), upstream
    /// `deleteYankedText`. The yanked text is derived from the ring's most
    /// recent entry since it hasn't been rotated yet.
    fn delete_yanked_text(&self) {
        let Some(yanked_text) = self.kill_ring.borrow().peek().map(str::to_string) else {
            return;
        };

        let yank_lines: Vec<&str> = yanked_text.split('\n').collect();

        let mut state = self.state.borrow_mut();
        if yank_lines.len() == 1 {
            // Single line - delete backward from cursor
            let cursor_col = state.cursor_col;
            let delete_len = yanked_text.len();
            let cursor_line = state.cursor_line;
            let current_line = state.lines[cursor_line].clone();
            let before = current_line[..cursor_col - delete_len].to_string();
            let after = current_line[cursor_col..].to_string();
            state.lines[cursor_line] = format!("{before}{after}");
            let new_col = cursor_col - delete_len;
            drop(state);
            self.set_cursor_col(new_col);
        } else {
            // Multi-line delete - cursor is at end of last yanked line
            let cursor_line = state.cursor_line;
            let start_line = cursor_line - (yank_lines.len() - 1);
            let start_col = state.lines[start_line].len() - yank_lines[0].len();

            // Get text after cursor on current line
            let after_cursor = state.lines[cursor_line][state.cursor_col..].to_string();

            // Get text before yank start position
            let before_yank = state.lines[start_line][..start_col].to_string();

            // Remove all lines from startLine to cursorLine and replace with
            // merged line
            let merged = format!("{before_yank}{after_cursor}");
            state.lines.splice(start_line..=cursor_line, [merged]);

            // Update cursor
            state.cursor_line = start_line;
            let new_col = start_col;
            drop(state);
            self.set_cursor_col(new_col);
        }

        self.fire_on_change();
    }

    /// Push the current state plus the paste registry as an undo snapshot,
    /// upstream `pushUndoSnapshot`.
    fn push_undo_snapshot(&self) {
        let state = self.state.borrow().clone();
        let pastes = self.pastes.borrow().clone();
        let paste_counter = self.paste_counter.get();
        self.undo_stack.borrow_mut().push(&EditorSnapshot {
            state,
            pastes,
            paste_counter,
        });
    }

    /// Restore the most recent undo snapshot, upstream `undo`.
    fn undo(&self) {
        self.exit_history_browsing();
        let Some(snapshot) = self.undo_stack.borrow_mut().pop() else {
            return;
        };
        *self.state.borrow_mut() = snapshot.state;
        *self.pastes.borrow_mut() = snapshot.pastes;
        self.paste_counter.set(snapshot.paste_counter);
        self.last_action.set(LastAction::None);
        self.preferred_visual_col.set(None);
        self.fire_on_change();
    }

    /// Jump to the first occurrence of a character in the specified
    /// direction, upstream `jumpToChar`. Multi-line search. Case-sensitive.
    /// Skips the current cursor position.
    fn jump_to_char(&self, char: &str, direction: JumpDirection) {
        self.last_action.set(LastAction::None);
        let is_forward = direction == JumpDirection::Forward;
        let (cursor_line, cursor_col) = {
            let state = self.state.borrow();
            (state.cursor_line, state.cursor_col)
        };
        let lines = self.state.borrow().lines.clone();

        let indices: Vec<usize> = if is_forward {
            (cursor_line..lines.len()).collect()
        } else {
            (0..=cursor_line).rev().collect()
        };

        for line_idx in indices {
            let line = &lines[line_idx];
            let is_current_line = line_idx == cursor_line;

            // Current line: start after/before cursor; other lines: search
            // full line
            let found = if is_forward {
                let search_from = if is_current_line { cursor_col + 1 } else { 0 };
                line.get(search_from..)
                    .and_then(|tail| tail.find(char))
                    .map(|offset| offset + search_from)
            } else {
                let search_from = if is_current_line {
                    cursor_col.saturating_sub(1)
                } else {
                    line.len()
                };
                line.match_indices(char)
                    .map(|(offset, _)| offset)
                    .filter(|offset| *offset <= search_from)
                    .last()
            };

            if let Some(col) = found {
                self.state.borrow_mut().cursor_line = line_idx;
                self.set_cursor_col(col);
                return;
            }
        }
        // No match found - cursor stays in place
    }

    /// Move one word forward, upstream `moveWordForwards`: at line end this
    /// moves to the start of the next line.
    fn move_word_forwards(&self) {
        self.last_action.set(LastAction::None);
        let current_line = self.state.borrow().lines[self.state.borrow().cursor_line].clone();

        // If at end of line, move to start of next line
        if self.state.borrow().cursor_col >= current_line.len() {
            if self.state.borrow().cursor_line < self.state.borrow().lines.len() - 1 {
                self.state.borrow_mut().cursor_line += 1;
                self.set_cursor_col(0);
            }
            return;
        }

        let segment_fn = |text: &str| self.segment(text, SegmentMode::Word);
        let atomic_fn = |segment: &str| is_paste_marker(segment);
        let options = WordNavigationOptions {
            segment: Some(&segment_fn),
            is_atomic_segment: Some(&atomic_fn),
        };
        let cursor_col = self.state.borrow().cursor_col;
        let new_col = find_word_forward(&current_line, cursor_col, Some(&options));
        self.set_cursor_col(new_col);
    }

    /// The terminal row count the layout budget reads, upstream
    /// `this.tui.terminal.rows`. A detached editor (weak reference dead)
    /// reads zero rows, which the visible-line and page-size minimums
    /// absorb.
    fn terminal_rows(&self) -> u16 {
        self.tui
            .borrow()
            .upgrade()
            .map_or(0, |tui| tui.terminal_rows())
    }

    /// Request a render from the owning TUI; a detached editor drops the
    /// request, upstream `this.tui.requestRender()`.
    fn request_render(&self) {
        if let Some(tui) = self.tui.borrow().upgrade() {
            tui.request_render(false);
        }
    }

    /// Fire the change callback with the current text, upstream
    /// `this.onChange?.(this.getText())`.
    fn fire_on_change(&self) {
        let on_change = self.on_change.borrow().clone();
        if let Some(on_change) = on_change {
            let text = self.get_text();
            on_change(&text);
        }
    }

    /// The raw input dispatch, upstream `Editor.handleInput`. Completed
    /// autocomplete requests resume here (upstream's event-loop
    /// continuations run between keystrokes), then the autocomplete
    /// dispatch block intercepts the open-picker keys.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's one dispatch chain branch for branch; splitting it would detach the port from the upstream method it mirrors"
    )]
    pub fn handle_input(&self, data: &str) {
        self.drain_autocomplete();

        // Handle character jump mode (awaiting next character to jump to)
        if let Some(direction) = self.jump_mode.get() {
            // Cancel if the hotkey is pressed again
            if self.matches(data, "tui.editor.jumpForward")
                || self.matches(data, "tui.editor.jumpBackward")
            {
                self.jump_mode.set(None);
                return;
            }

            let printable = decode_printable_key(data).or_else(|| {
                data.bytes()
                    .next()
                    .filter(|byte| *byte >= 32)
                    .map(|_| data.to_string())
            });
            if let Some(printable) = printable {
                // Printable character - perform the jump
                self.jump_mode.set(None);
                self.jump_to_char(&printable, direction);
                return;
            }

            // Control character - cancel and fall through to normal handling
            self.jump_mode.set(None);
        }

        // Handle bracketed paste mode
        let mut data = data.to_string();
        if data.contains("\x1b[200~") {
            self.is_in_paste.set(true);
            *self.paste_buffer.borrow_mut() = String::new();
            data = data.replacen("\x1b[200~", "", 1);
        }

        if self.is_in_paste.get() {
            self.paste_buffer.borrow_mut().push_str(&data);
            let end_index = self.paste_buffer.borrow().find("\x1b[201~");
            if let Some(end_index) = end_index {
                let paste_content = self.paste_buffer.borrow()[..end_index].to_string();
                if !paste_content.is_empty() {
                    self.handle_paste(&paste_content);
                }
                self.is_in_paste.set(false);
                let remaining = self.paste_buffer.borrow()[end_index + 6..].to_string();
                *self.paste_buffer.borrow_mut() = String::new();
                if !remaining.is_empty() {
                    self.handle_input(&remaining);
                }
                return;
            }
            return;
        }

        // Ctrl+C - let parent handle (exit/clear)
        if self.matches(&data, "tui.input.copy") {
            return;
        }

        // Undo
        if self.matches(&data, "tui.editor.undo") {
            self.undo();
            return;
        }

        // Handle autocomplete mode
        if self.autocomplete_state.get().is_some() {
            if self.matches(&data, "tui.select.cancel") {
                self.cancel_autocomplete();
                return;
            }

            if self.matches(&data, "tui.select.up") || self.matches(&data, "tui.select.down") {
                if let Some(list) = self.autocomplete_list.borrow().as_ref() {
                    list.handle_input(&data);
                }
                return;
            }

            if self.matches(&data, "tui.input.tab") {
                let selected = self
                    .autocomplete_list
                    .borrow()
                    .as_ref()
                    .and_then(SelectList::get_selected_item);
                if let Some(selected) = selected
                    && self.autocomplete_provider.borrow().is_some()
                {
                    let prefix = self.autocomplete_prefix.borrow().clone();
                    self.apply_selected_completion(&selected.into(), &prefix);
                    self.cancel_autocomplete();
                    self.fire_on_change();
                }
                return;
            }

            if self.matches(&data, "tui.select.confirm") {
                let selected = self
                    .autocomplete_list
                    .borrow()
                    .as_ref()
                    .and_then(SelectList::get_selected_item);
                if let Some(selected) = selected
                    && self.autocomplete_provider.borrow().is_some()
                {
                    let prefix = self.autocomplete_prefix.borrow().clone();
                    self.apply_selected_completion(&selected.into(), &prefix);

                    self.cancel_autocomplete();
                    if self.autocomplete_prefix.borrow().starts_with('/') {
                        // Fall through to submit
                    } else {
                        self.fire_on_change();
                        return;
                    }
                }
            }
        }

        // Tab - trigger completion
        if self.matches(&data, "tui.input.tab") && self.autocomplete_state.get().is_none() {
            self.handle_tab_completion();
            return;
        }

        // Deletion actions
        if self.matches(&data, "tui.editor.deleteToLineEnd") {
            self.delete_to_end_of_line();
            return;
        }
        if self.matches(&data, "tui.editor.deleteToLineStart") {
            self.delete_to_start_of_line();
            return;
        }
        if self.matches(&data, "tui.editor.deleteWordBackward") {
            self.delete_word_backwards();
            return;
        }
        if self.matches(&data, "tui.editor.deleteWordForward") {
            self.delete_word_forward();
            return;
        }
        if self.matches(&data, "tui.editor.deleteCharBackward")
            || self.parser.matches_key(&data, "shift+backspace")
        {
            self.handle_backspace();
            return;
        }
        if self.matches(&data, "tui.editor.deleteCharForward")
            || self.parser.matches_key(&data, "shift+delete")
        {
            self.handle_forward_delete();
            return;
        }

        // Kill ring actions
        if self.matches(&data, "tui.editor.yank") {
            self.yank();
            return;
        }
        if self.matches(&data, "tui.editor.yankPop") {
            self.yank_pop();
            return;
        }

        // Dedicated history actions always browse entries instead of moving
        // the cursor.
        if self.matches(&data, "tui.editor.historyPrevious") {
            self.cancel_autocomplete();
            self.navigate_history(-1);
            return;
        }
        if self.matches(&data, "tui.editor.historyNext") {
            self.cancel_autocomplete();
            self.navigate_history(1);
            return;
        }

        // Cursor movement actions
        if self.matches(&data, "tui.editor.cursorLineStart") {
            self.move_to_line_start();
            return;
        }
        if self.matches(&data, "tui.editor.cursorLineEnd") {
            self.move_to_line_end();
            return;
        }
        if self.matches(&data, "tui.editor.cursorWordLeft") {
            self.move_word_backwards();
            return;
        }
        if self.matches(&data, "tui.editor.cursorWordRight") {
            self.move_word_forwards();
            return;
        }

        // New line
        if self.matches(&data, "tui.input.newLine")
            || (data.as_bytes().first() == Some(&10) && data.len() > 1)
            || data == "\x1b\r"
            || data == "\x1b[13;2~"
            || (data.len() > 1 && data.contains('\x1b') && data.contains('\r'))
            || (data == "\n" && data.len() == 1)
        {
            if self.should_submit_on_backslash_enter(&data) {
                self.handle_backspace();
                self.submit_value();
                return;
            }
            self.add_new_line();
            return;
        }

        // Submit (Enter)
        if self.matches(&data, "tui.input.submit") {
            if self.disable_submit.get() {
                return;
            }

            // Workaround for terminals without Shift+Enter support:
            // If char before cursor is \, delete it and insert newline instead
            // of submitting.
            let (cursor_col, current_line) = {
                let state = self.state.borrow();
                (state.cursor_col, state.lines[state.cursor_line].clone())
            };
            if cursor_col > 0
                && current_line
                    .get(cursor_col - 1..cursor_col)
                    .is_some_and(|char| char == "\\")
            {
                self.handle_backspace();
                self.add_new_line();
                return;
            }

            self.submit_value();
            return;
        }

        // Arrow key navigation (with history support)
        if self.matches(&data, "tui.editor.cursorUp") {
            if self.is_on_first_visual_line()
                && (self.is_editor_empty()
                    || self.history_index.get() > -1
                    || self.state.borrow().cursor_col == 0)
            {
                self.navigate_history(-1);
            } else if self.is_on_first_visual_line() {
                // Already at top - jump to start of line
                self.move_to_line_start();
            } else {
                self.move_cursor(-1, 0);
            }
            return;
        }
        if self.matches(&data, "tui.editor.cursorDown") {
            if self.history_index.get() > -1 && self.is_on_last_visual_line() {
                self.navigate_history(1);
            } else if self.is_on_last_visual_line() {
                // Already at bottom - jump to end of line
                self.move_to_line_end();
            } else {
                self.move_cursor(1, 0);
            }
            return;
        }
        if self.matches(&data, "tui.editor.cursorRight") {
            self.move_cursor(0, 1);
            return;
        }
        if self.matches(&data, "tui.editor.cursorLeft") {
            self.move_cursor(0, -1);
            return;
        }

        // Page up/down - scroll by page and move cursor
        if self.matches(&data, "tui.editor.pageUp") {
            self.page_scroll(-1);
            return;
        }
        if self.matches(&data, "tui.editor.pageDown") {
            self.page_scroll(1);
            return;
        }

        // Character jump mode triggers
        if self.matches(&data, "tui.editor.jumpForward") {
            self.jump_mode.set(Some(JumpDirection::Forward));
            return;
        }
        if self.matches(&data, "tui.editor.jumpBackward") {
            self.jump_mode.set(Some(JumpDirection::Backward));
            return;
        }

        // Shift+Space - insert regular space
        if self.parser.matches_key(&data, "shift+space") {
            self.insert_character(" ");
            return;
        }

        if let Some(printable) = decode_printable_key(&data) {
            self.insert_character(&printable);
            return;
        }

        // Regular characters
        if data.as_bytes().first().is_some_and(|byte| *byte >= 32) {
            self.insert_character(&data);
        }
    }

    /// Binding check through the editor's parser, upstream `kb.matches`.
    fn matches(&self, data: &str, action: &str) -> bool {
        get_keybindings().matches(&self.parser, data, action)
    }

    /// Mouse cursor placement, upstream `Editor.handleMouse`: clicks inside
    /// the visible editor rows map to a logical position; press/drag/release
    /// stay unhandled so the renderer's screen-level text selection can run
    /// over the editor rows. The autocomplete-dropdown region dispatch ships
    /// with the autocomplete child (#49).
    /// Mouse cursor placement, upstream `Editor.handleMouse`: the open
    /// autocomplete dropdown's region dispatches into its list (clicks
    /// apply the clicked suggestion); presses/drag/releases over the editor
    /// rows stay unhandled so the renderer's screen-level text selection
    /// can run (drag to select, release to copy); clicks inside the visible
    /// editor rows map to a logical position. The renderer synthesizes a
    /// click when press and release land on the same cell without
    /// movement, which is the gesture that positions the cursor.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's handleMouse branch for branch; splitting it would detach the port from the upstream method it mirrors"
    )]
    fn handle_mouse_impl(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        // The dropdown renders after the bottom border: top border row +
        // visible rows + bottom border = the picker's first row.
        let autocomplete_start_row = self.rendered_visible_line_count.get() + 2;
        if self.autocomplete_state.get().is_some()
            && self.autocomplete_list.borrow().is_some()
            && usize::from(event.y) >= autocomplete_start_row
            && usize::from(event.y)
                < autocomplete_start_row + self.rendered_autocomplete_height.get()
        {
            let max_padding = (usize::from(event.width).saturating_sub(1)) / 2;
            let padding_x = self.padding_x.get().min(max_padding);
            let content_width = usize::from(event.width)
                .saturating_sub(padding_x * 2)
                .max(1);
            let translated = TuiMouseEvent {
                x: event
                    .x
                    .saturating_sub(u16::try_from(padding_x).unwrap_or(u16::MAX)),
                y: event.y - u16::try_from(autocomplete_start_row).unwrap_or(u16::MAX),
                width: u16::try_from(content_width).unwrap_or(u16::MAX),
                height: u16::try_from(self.rendered_autocomplete_height.get()).unwrap_or(u16::MAX),
                ..event.clone()
            };
            let result = self
                .autocomplete_list
                .borrow()
                .as_ref()
                .and_then(|list| list.handle_mouse(&translated))?;
            // Upstream wires the list's `onSelect` to apply the clicked
            // completion; the port applies it after the dispatch because
            // the callback would need a handle on the non-`Rc` editor.
            if translated.event_type == TuiMouseEventType::Click {
                let selected = self
                    .autocomplete_list
                    .borrow()
                    .as_ref()
                    .and_then(SelectList::get_selected_item);
                if let Some(selected) = selected {
                    let prefix = self.autocomplete_prefix.borrow().clone();
                    self.apply_selected_completion(&selected.into(), &prefix);
                }
                self.cancel_autocomplete();
                self.fire_on_change();
            }
            return Some(TuiMouseEventResult {
                focus: true,
                ..result
            });
        }

        // Leave press/drag/release unhandled so the renderer's screen-level
        // text selection can run over the editor rows (drag to select,
        // release to copy). The renderer synthesizes a click when press and
        // release land on the same cell without movement, which is the
        // gesture that positions the cursor.
        if event.event_type != TuiMouseEventType::Click || event.button != TuiMouseButton::Left {
            return None;
        }
        if event.y == 0 || usize::from(event.y) > self.rendered_visible_line_count.get() {
            return Some(TuiMouseEventResult {
                handled: true,
                focus: true,
                ..TuiMouseEventResult::default()
            });
        }

        let visual_lines = self.build_visual_line_map(self.last_width.get());
        let visual_line_index = self.scroll_offset.get() + usize::from(event.y) - 1;
        let Some(visual_line) = visual_lines.get(visual_line_index) else {
            return Some(TuiMouseEventResult {
                handled: true,
                focus: true,
                ..TuiMouseEventResult::default()
            });
        };
        let logical_line = self.state.borrow().lines[visual_line.logical_line].clone();
        let chunk_end = visual_line.start_col + visual_line.length;
        let chunk = &logical_line[visual_line.start_col..chunk_end];
        let max_padding = (usize::from(event.width).saturating_sub(1)) / 2;
        let padding_x = self.padding_x.get().min(max_padding);
        let target_column = usize::from(event.x).saturating_sub(padding_x);
        let mut visible_column = 0;
        let mut target_index = chunk.len();
        let mut last_grapheme_index = 0;
        let mut offset = 0;
        for grapheme in self.segment(chunk, SegmentMode::Grapheme) {
            let next_column = visible_column + visible_width(&grapheme.segment);
            last_grapheme_index = offset;
            if target_column < next_column {
                target_index = offset;
                break;
            }
            visible_column = next_column;
            offset += grapheme.segment.len();
        }
        let is_last_segment = visual_line_index == visual_lines.len() - 1
            || visual_lines
                .get(visual_line_index + 1)
                .is_none_or(|next| next.logical_line != visual_line.logical_line);
        if !is_last_segment && target_index == chunk.len() && !chunk.is_empty() {
            target_index = last_grapheme_index;
        }

        self.state.borrow_mut().cursor_line = visual_line.logical_line;
        self.set_cursor_col(visual_line.start_col + target_index);
        self.last_action.set(LastAction::None);
        self.exit_history_browsing();
        if self.autocomplete_state.get().is_some() {
            self.update_autocomplete();
        }
        Some(TuiMouseEventResult {
            handled: true,
            focus: true,
            ..TuiMouseEventResult::default()
        })
    }
}

/// Renumber paste markers with ids greater than `target_id` down by one,
/// upstream's `PASTE_MARKER_REGEX` replace callback in `handleBackspace`.
/// A marker with no suffix — not producible by the editor itself — renumbers
/// without one; upstream's template literal would emit the text
/// `undefined` there, which the port does not reproduce.
fn renumber_paste_markers(line: &str, target_id: u32) -> String {
    PASTE_MARKER_RE
        .replace_all(line, |caps: &Captures| {
            let Ok(id) = caps[1].parse::<u32>() else {
                return caps[0].to_string();
            };
            if id <= target_id {
                return caps[0].to_string();
            }
            let suffix = caps
                .get(2)
                .map_or_else(String::new, |suffix| suffix.as_str().to_string());
            format!("[paste #{}{suffix}]", id - 1)
        })
        .to_string()
}

impl Component for Editor {
    fn render(&self, width: usize) -> Vec<String> {
        self.render_impl(width)
    }

    fn handle_input(&self, data: &str) {
        self.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.handle_mouse_impl(event)
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for Editor {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

impl Drop for Editor {
    fn drop(&mut self) {
        // Shut the autocomplete worker down: clear its queue, cancel the
        // active request, wake it. A provider call already running finishes
        // and the worker exits on the next loop.
        if let Some(channel) = self.autocomplete_channel.borrow().as_ref() {
            let mut state = channel.shared.lock();
            state.pending = None;
            state.shutdown = true;
            if let Some(token) = state.active_token.take() {
                token.cancel();
            }
            channel.shared.signal.notify_all();
            drop(state);
        }
    }
}

impl std::fmt::Debug for EditorTheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditorTheme").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Editor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Editor")
            .field("padding_x", &self.padding_x.get())
            .field("scroll_offset", &self.scroll_offset.get())
            .field("paste_counter", &self.paste_counter.get())
            .field("history_index", &self.history_index.get())
            .finish_non_exhaustive()
    }
}

/// The centered scroll indicator border, upstream `createScrollBorder`.
#[must_use]
fn create_scroll_border(direction: &str, hidden_line_count: usize, width: usize) -> String {
    let available_width = width;
    let label = format!(" {direction} {hidden_line_count} more ");
    let label_width = visible_width(&label);
    if label_width + 2 <= available_width {
        let left_width = (available_width - label_width) / 2;
        return format!(
            "{}{label}{}",
            "─".repeat(left_width),
            "─".repeat(available_width - left_width - label_width)
        );
    }

    let indicator = format!("─── {direction} {hidden_line_count} more ");
    let remaining = available_width.saturating_sub(visible_width(&indicator));
    if available_width >= visible_width(&indicator) {
        return format!("{indicator}{}", "─".repeat(remaining));
    }

    let ellipsis = "...".get(..available_width.min(3)).unwrap_or("...");
    let indicator_width = available_width.saturating_sub(visible_width(ellipsis));
    format!(
        "{}{ellipsis}",
        slice_by_column(&indicator, 0, indicator_width, true)
    )
}
