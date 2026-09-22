//! The transcript search of `packages/tui/src/alt-screen-search.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#46).
//!
//! It carries the corpus builder and match index the alternate-screen
//! renderer refreshes each frame, and the search overlay component itself.
//!
//! Restatements against upstream:
//!
//! - Upstream `Input` (`components/input.ts`) is the editor slice's scope
//!   ([#47](https://github.com/PhillipChaffee/pi-rust/issues/47)); the
//!   search box consumes only the single-line editing core, so
//!   the private `SearchInput` ports that surface — cancel/submit consume, printable
//!   and Kitty-printable insert, backspace, grapheme-wise cursor moves, and
//!   the full render including the horizontal-scroll window. Kill ring,
//!   undo stack, word motions, paste buffering, and mouse placement are
//!   #47's and stay out; #47's `Input` replaces this stand-in.
//! - Upstream's module-global key parser behind `matchesKey` becomes a
//!   local [`crate::keys::KeyParser`]: the input's binding checks are
//!   stateless key-shape matches, and key releases never reach it (dispatch
//!   filters releases for components that do not opt in).
//! - [`AltScreenSearchIndex::search`] returned the live matches array
//!   upstream, and the suite asserted reference identity on the cached
//!   result; the port returns owned matches, so the same assertion compares
//!   by value.
//! - `matchAll` over a case-insensitive Unicode regex becomes the `regex`
//!   crate with the `(?i)` flag; corpus offsets are byte offsets rather
//!   than UTF-16 code units, which the span arithmetic carries end to end.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::LazyLock;

use regex::Regex;

use crate::keybindings::get_keybindings;
use crate::keys::{KeyParser, decode_kitty_printable};
use crate::tui::{CURSOR_MARKER, Component, Focusable};
use crate::utils::{
    grapheme_segments, slice_by_column, static_regex, truncate_to_width, visible_width,
};

static WHITESPACE_RUN: LazyLock<Regex> = LazyLock::new(|| static_regex(r"\s+"));
static PRINTABLE_ASCII: LazyLock<Regex> = LazyLock::new(|| static_regex(r"^[\x20-\x7e]*$"));
static WHITESPACE_ONLY: LazyLock<Regex> = LazyLock::new(|| static_regex(r"^\s+$"));

/// One source span in the search corpus, upstream `SearchSourceSpan`.
///
/// Text offsets are byte offsets into [`SearchCorpus::text`]; the column
/// range is cells on the source row.
struct SearchSourceSpan {
    text_start: usize,
    text_end: usize,
    row: usize,
    start_col: usize,
    end_col: usize,
    linear_columns: bool,
}

/// The flattened searchable transcript, upstream `SearchCorpus`.
struct SearchCorpus {
    text: String,
    spans: Vec<SearchSourceSpan>,
}

/// One rendered-column segment of a match, upstream
/// `AltScreenSearchSegment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AltScreenSearchSegment {
    /// The source row.
    pub row: usize,
    /// First selected column.
    pub start_col: usize,
    /// One past the last selected column.
    pub end_col: usize,
}

/// One match as rendered-column segments, upstream `AltScreenSearchMatch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AltScreenSearchMatch {
    /// The segments, in row order.
    pub segments: Vec<AltScreenSearchSegment>,
}

fn append_separator(text: &mut String, pending: &mut bool) {
    if *pending {
        text.push(' ');
        *pending = false;
    }
}

/// Build the corpus, upstream `buildSearchCorpus`: strip each line, flatten
/// the non-space runs into the text, and map every run back to its row and
/// cell range.
fn build_search_corpus(lines: &[String]) -> SearchCorpus {
    let mut text = String::new();
    let mut spans = Vec::new();
    let mut pending_separator = false;

    for (row, source) in lines.iter().enumerate() {
        let line = crate::utils::strip_terminal_sequences(source);
        let mut column = 0;

        // Rendered transcripts are overwhelmingly ASCII. Index complete
        // non-space runs at once instead of segmenting and allocating one
        // mapping per cell.
        if PRINTABLE_ASCII.is_match(&line) {
            let bytes = line.as_bytes();
            let mut index = 0;
            while index < bytes.len() {
                if bytes[index] == 0x20 {
                    if !text.is_empty() {
                        pending_separator = true;
                    }
                    column += 1;
                    index += 1;
                    continue;
                }
                let mut end = index + 1;
                while end < bytes.len() && bytes[end] != 0x20 {
                    end += 1;
                }
                append_separator(&mut text, &mut pending_separator);
                let run = &line[index..end];
                spans.push(SearchSourceSpan {
                    text_start: text.len(),
                    text_end: text.len() + run.len(),
                    row,
                    start_col: column,
                    end_col: column + run.len(),
                    linear_columns: true,
                });
                text.push_str(run);
                column += run.len();
                index = end;
            }
        } else {
            for grapheme in grapheme_segments(&line) {
                let width = visible_width(grapheme);
                if WHITESPACE_ONLY.is_match(grapheme) {
                    if !text.is_empty() {
                        pending_separator = true;
                    }
                    column += width;
                    continue;
                }
                append_separator(&mut text, &mut pending_separator);
                spans.push(SearchSourceSpan {
                    text_start: text.len(),
                    text_end: text.len() + grapheme.len(),
                    row,
                    start_col: column,
                    end_col: column + width,
                    linear_columns: false,
                });
                text.push_str(grapheme);
                column += width;
            }
        }
        if !text.is_empty() {
            pending_separator = true;
        }
    }

    SearchCorpus { text, spans }
}

/// Collapse whitespace runs and trim the ends, upstream `normalizeQuery`.
fn normalize_query(query: &str) -> String {
    WHITESPACE_RUN.replace_all(query, " ").trim().to_string()
}

/// Find the corpus matches, upstream `findSearchCorpusMatches`: match the
/// escaped query case-insensitively and map each hit back through the spans
/// to rendered rows and columns, merging the segments a hit covers on one
/// row.
fn find_search_corpus_matches(
    corpus: &SearchCorpus,
    normalized_query: &str,
) -> Vec<AltScreenSearchMatch> {
    if normalized_query.is_empty() {
        return Vec::new();
    }
    // The query is regex-escaped, so the pattern builds for any input.
    #[expect(
        clippy::expect_used,
        reason = "the escaped query pattern is valid by construction; a build failure is a programmer error"
    )]
    let expression = Regex::new(&format!("(?i){}", regex::escape(normalized_query)))
        .expect("an escaped pattern builds");
    let mut matches = Vec::new();
    let mut span_index = 0;

    for hit in expression.find_iter(&corpus.text) {
        let start = hit.start();
        let end = hit.end();
        while span_index < corpus.spans.len() && corpus.spans[span_index].text_end <= start {
            span_index += 1;
        }

        let mut segments: Vec<AltScreenSearchSegment> = Vec::new();
        for span in &corpus.spans[span_index..] {
            if span.text_start >= end {
                break;
            }
            if span.text_end <= start {
                continue;
            }
            let (start_col, end_col) = if span.linear_columns {
                (
                    span.start_col + start.max(span.text_start) - span.text_start,
                    span.start_col + end.min(span.text_end) - span.text_start,
                )
            } else {
                (span.start_col, span.end_col)
            };
            match segments.last_mut() {
                Some(previous) if previous.row == span.row && start_col <= previous.end_col => {
                    previous.end_col = previous.end_col.max(end_col);
                }
                _ => segments.push(AltScreenSearchSegment {
                    row: span.row,
                    start_col,
                    end_col,
                }),
            }
        }
        while span_index < corpus.spans.len() && corpus.spans[span_index].text_end <= end {
            span_index += 1;
        }
        if !segments.is_empty() {
            matches.push(AltScreenSearchMatch { segments });
        }
    }

    matches
}

/// The result of one [`AltScreenSearchIndex::search`], upstream
/// `AltScreenSearchResult`.
#[derive(Debug, Clone)]
pub struct AltScreenSearchResult {
    /// The matches for the query.
    pub matches: Vec<AltScreenSearchMatch>,
    /// Whether the corpus or the query changed since the previous search.
    pub changed: bool,
}

/// Cache the searchable corpus and matches while rendered transcript lines
/// remain unchanged, upstream `class AltScreenSearchIndex`.
///
/// The corpus rides a manual [`Debug`] because the flattened text is opaque
/// rendered content; the cache shape is what matters.
#[derive(Default)]
pub struct AltScreenSearchIndex {
    source_lines: Option<Vec<String>>,
    corpus: Option<SearchCorpus>,
    normalized_query: Option<String>,
    matches: Vec<AltScreenSearchMatch>,
}

impl std::fmt::Debug for AltScreenSearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AltScreenSearchIndex")
            .field("source_lines", &self.source_lines.as_ref().map(Vec::len))
            .field(
                "corpus",
                &self.corpus.as_ref().map(|corpus| corpus.text.len()),
            )
            .field("normalized_query", &self.normalized_query)
            .field("matches", &self.matches.len())
            .finish()
    }
}

impl AltScreenSearchIndex {
    /// Search `lines` for `query`, reusing the cached corpus and matches
    /// while both stay unchanged, upstream `AltScreenSearchIndex.search`.
    ///
    /// Upstream returned the live matches array and the suite asserted
    /// reference identity on the cached result; the port returns owned
    /// matches, so the same assertion compares by value.
    pub fn search(&mut self, lines: &[String], query: &str) -> AltScreenSearchResult {
        let mut source_changed = self
            .source_lines
            .as_ref()
            .is_none_or(|source| source.len() != lines.len());
        if !source_changed && let Some(source) = &self.source_lines {
            source_changed = source
                .iter()
                .zip(lines.iter())
                .any(|(source, line)| source != line);
        }
        if source_changed || self.corpus.is_none() {
            self.source_lines = Some(lines.to_vec());
            self.corpus = Some(build_search_corpus(lines));
        }

        let normalized_query = normalize_query(query);
        let changed =
            source_changed || self.normalized_query.as_deref() != Some(normalized_query.as_str());
        if changed && let Some(corpus) = &self.corpus {
            self.normalized_query = Some(normalized_query.clone());
            self.matches = find_search_corpus_matches(corpus, &normalized_query);
        }
        AltScreenSearchResult {
            matches: self.matches.clone(),
            changed,
        }
    }
}

/// The matches for `query` over `lines`, upstream
/// `findAltScreenSearchMatches`: one-shot search without a cache.
#[must_use]
pub fn find_alt_screen_search_matches(lines: &[String], query: &str) -> Vec<AltScreenSearchMatch> {
    let normalized_query = normalize_query(query);
    if normalized_query.is_empty() {
        return Vec::new();
    }
    find_search_corpus_matches(&build_search_corpus(lines), &normalized_query)
}

/// The stable identity of a match — its first segment's start beside its
/// last segment's end, upstream `getAltScreenSearchMatchKey`.
#[must_use]
pub fn get_alt_screen_search_match_key(match_: &AltScreenSearchMatch) -> String {
    match (match_.segments.first(), match_.segments.last()) {
        (Some(first), Some(last)) => {
            format!(
                "{}:{}:{}:{}",
                first.row, first.start_col, last.row, last.end_col
            )
        }
        _ => String::new(),
    }
}

/// The style callbacks the search overlay consumes, upstream `(text:
/// string) => string`.
pub type SearchStyleFn = Rc<dyn Fn(&str) -> String>;

/// The navigation-button style, upstream `(text: string, hovered: boolean)
/// => string`.
pub type NavigationButtonStyleFn = Rc<dyn Fn(&str, bool) -> String>;

/// The single-line input the search box embeds, upstream
/// `components/input.ts`'s `Input`.
///
/// The full `Input` — kill ring, undo stack, word motions, paste buffering,
/// and mouse placement — is the editor slice's scope ([#47](https://github.com/PhillipChaffee/pi-rust/issues/47));
/// the search box consumes only the single-line editing surface, so this
/// stand-in ports exactly that: cancel/submit consume, printable and
/// Kitty-printable insert, backspace, grapheme-wise cursor movement, and
/// the full render including the horizontal-scroll window. #47's `Input`
/// replaces it.
struct SearchInput {
    value: RefCell<String>,
    cursor: RefCell<usize>,
    prompt: String,
    placeholder: String,
    placeholder_style: SearchStyleFn,
    focused: Cell<bool>,
    parser: RefCell<KeyParser>,
}

/// Construction options for [`SearchInput`], upstream `InputOptions`.
struct SearchInputOptions {
    prompt: Option<String>,
    placeholder: Option<String>,
    placeholder_style: Option<SearchStyleFn>,
}

impl SearchInput {
    fn new(options: SearchInputOptions) -> Self {
        Self {
            value: RefCell::new(String::new()),
            cursor: RefCell::new(0),
            prompt: options.prompt.unwrap_or_else(|| "> ".to_string()),
            placeholder: options.placeholder.unwrap_or_default(),
            placeholder_style: options
                .placeholder_style
                .unwrap_or_else(|| Rc::new(ToString::to_string)),
            focused: Cell::new(false),
            parser: RefCell::new(KeyParser::new()),
        }
    }

    fn get_value(&self) -> String {
        self.value.borrow().clone()
    }

    fn handle_input(&self, data: &str) {
        let matches = |action: &str| {
            let parser = self.parser.borrow();
            get_keybindings().matches(&parser, data, action)
        };
        if matches("tui.select.cancel") {
            return;
        }
        if matches("tui.input.submit") || data == "\n" {
            return;
        }
        if matches("tui.editor.deleteCharBackward") {
            self.handle_backspace();
            return;
        }
        if matches("tui.editor.cursorLeft") {
            self.move_cursor(-1);
            return;
        }
        if matches("tui.editor.cursorRight") {
            self.move_cursor(1);
            return;
        }

        // Kitty CSI-u printable character (e.g. \x1b[97u for 'a'): terminals
        // with Kitty protocol flag 1 send CSI-u for all keys, including
        // plain printable characters, and the sequences contain \x1b which
        // the control-char check would reject.
        if let Some(printable) = decode_kitty_printable(data) {
            self.insert_character(&printable);
            return;
        }

        // Regular character input — printable characters including Unicode,
        // but reject control characters (C0: 0x00-0x1F, DEL: 0x7F,
        // C1: 0x80-0x9F).
        let has_control_chars = data.chars().any(|ch| {
            let code = u32::from(ch);
            code < 32 || code == 0x7f || (0x80..=0x9f).contains(&code)
        });
        if !has_control_chars {
            self.insert_character(data);
        }
    }

    fn insert_character(&self, text: &str) {
        let mut value = self.value.borrow_mut();
        let mut cursor = self.cursor.borrow_mut();
        *cursor = (*cursor).min(value.len());
        value.insert_str(*cursor, text);
        *cursor += text.len();
    }

    fn handle_backspace(&self) {
        let mut value = self.value.borrow_mut();
        let mut cursor = self.cursor.borrow_mut();
        if *cursor == 0 {
            return;
        }
        let before_cursor = value[..*cursor].to_string();
        let last = grapheme_segments(&before_cursor)
            .next_back()
            .map_or(1, str::len);
        value.replace_range(*cursor - last..*cursor, "");
        *cursor -= last;
    }

    fn move_cursor(&self, direction: i32) {
        let value = self.value.borrow();
        let mut cursor = self.cursor.borrow_mut();
        *cursor = (*cursor).min(value.len());
        if direction < 0 {
            let before = &value[..*cursor];
            let last = grapheme_segments(before).next_back().map_or(1, str::len);
            *cursor -= last.min(*cursor);
        } else if *cursor < value.len() {
            let after = &value[*cursor..];
            let first = grapheme_segments(after).next().map_or(1, str::len);
            *cursor = (*cursor + first).min(value.len());
        }
    }

    /// Upstream's `Input.invalidate`: a documented no-op — the input holds no
    /// cached state to invalidate.
    #[expect(
        clippy::unused_self,
        clippy::missing_const_for_fn,
        reason = "upstream's Input.invalidate is a documented no-op; the port keeps the surface for the Component contract"
    )]
    fn invalidate(&self) {}

    /// The prompt-and-value line, upstream `Input.render`: the inverse-video
    /// fake cursor at the grapheme the cursor sits on, the hardware-cursor
    /// marker ahead of it when focused, and the horizontal-scroll window
    /// when the value overflows the available width.
    fn render(&self, width: usize) -> Vec<String> {
        let value = self.value.borrow().clone();
        let cursor = (*self.cursor.borrow()).min(value.len());
        let available_width = width.saturating_sub(visible_width(&self.prompt));

        if available_width == 0 {
            return vec![truncate_to_width(&self.prompt, width, "", false)];
        }

        if value.is_empty() && !self.placeholder.is_empty() {
            let placeholder = truncate_to_width(&self.placeholder, available_width, "", false);
            let at_cursor = grapheme_segments(&placeholder).next().unwrap_or(" ");
            let after_cursor = &placeholder[at_cursor.len()..];
            let marker = if self.focused.get() {
                CURSOR_MARKER
            } else {
                ""
            };
            let cursor_char = format!(
                "\x1b[7m{}{}\x1b[27m",
                (self.placeholder_style)(at_cursor),
                (self.placeholder_style)(after_cursor)
            );
            let text_with_cursor = format!("{marker}{cursor_char}");
            let padding =
                " ".repeat(available_width.saturating_sub(visible_width(&text_with_cursor)));
            return vec![format!("{}{text_with_cursor}{padding}", self.prompt)];
        }

        // The horizontal-scroll window, upstream Input's over-long branch:
        // reserve one column for the cursor when it sits at the end, then
        // center the cursor column in the visible window.
        let total_width = visible_width(&value);
        let visible_text;
        let mut cursor_display = cursor;
        if total_width < available_width {
            visible_text = value;
        } else {
            let scroll_width = if cursor == value.len() {
                available_width - 1
            } else {
                available_width
            };
            let cursor_col = visible_width(&value[..cursor]);
            if scroll_width > 0 {
                let half_width = scroll_width / 2;
                let start_col = if cursor_col < half_width {
                    0
                } else if cursor_col > total_width.saturating_sub(half_width) {
                    total_width.saturating_sub(scroll_width)
                } else {
                    cursor_col.saturating_sub(half_width)
                };
                let window = slice_by_column(&value, start_col, scroll_width, true);
                let before_cursor = slice_by_column(
                    &value,
                    start_col,
                    cursor_col.saturating_sub(start_col),
                    true,
                );
                cursor_display = before_cursor.len();
                visible_text = window;
            } else {
                cursor_display = 0;
                visible_text = String::new();
            }
        }

        let after = &visible_text[cursor_display.min(visible_text.len())..];
        let at_cursor = grapheme_segments(after).next().map_or(" ", |g| g);
        let before_cursor = &visible_text[..cursor_display.min(visible_text.len())];
        let after_cursor = after.get(at_cursor.len()..).unwrap_or_default();

        // Hardware cursor marker (zero-width, emitted before fake cursor for
        // IME positioning).
        let marker = if self.focused.get() {
            CURSOR_MARKER
        } else {
            ""
        };
        let cursor_char = format!("\x1b[7m{at_cursor}\x1b[27m");
        let text_with_cursor = format!("{before_cursor}{marker}{cursor_char}{after_cursor}");
        let visual_length = visible_width(&text_with_cursor);
        let padding = " ".repeat(available_width.saturating_sub(visual_length));
        vec![format!("{}{text_with_cursor}{padding}", self.prompt)]
    }
}

/// The transcript search overlay, upstream `class AltScreenSearchComponent`:
/// the input line beside the right-aligned navigation buttons, rendered as a
/// three-line box.
pub struct AltScreenSearchComponent {
    input: SearchInput,
    on_query_change: Rc<dyn Fn(&str)>,
    navigation_button_style: NavigationButtonStyleFn,
    result_count: Cell<usize>,
    result_index: Cell<i64>,
    previous_button_start: Cell<i64>,
    previous_button_end: Cell<i64>,
    next_button_start: Cell<i64>,
    next_button_end: Cell<i64>,
    hovered_navigation_direction: Cell<Option<i8>>,
    focused: Cell<bool>,
}

impl std::fmt::Debug for AltScreenSearchComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AltScreenSearchComponent")
            .field("result_index", &self.result_index.get())
            .field("result_count", &self.result_count.get())
            .finish_non_exhaustive()
    }
}

impl AltScreenSearchComponent {
    /// Upstream's constructor: `new AltScreenSearchComponent(onQueryChange,
    /// navigationButtonStyle?)`.
    #[must_use]
    pub fn new(
        on_query_change: Rc<dyn Fn(&str)>,
        navigation_button_style: Option<NavigationButtonStyleFn>,
    ) -> Rc<Self> {
        Rc::new(Self {
            input: SearchInput::new(SearchInputOptions {
                prompt: Some(" ".to_string()),
                placeholder: Some("Find in transcript".to_string()),
                placeholder_style: Some(Rc::new(|text| format!("\x1b[2m{text}\x1b[22m"))),
            }),
            on_query_change,
            navigation_button_style: navigation_button_style
                .unwrap_or_else(|| Rc::new(|text, _hovered| text.to_string())),
            result_count: Cell::new(0),
            result_index: Cell::new(-1),
            previous_button_start: Cell::new(-1),
            previous_button_end: Cell::new(-1),
            next_button_start: Cell::new(-1),
            next_button_end: Cell::new(-1),
            hovered_navigation_direction: Cell::new(None),
            focused: Cell::new(false),
        })
    }

    /// Report the current result index and total, upstream `setResult`.
    pub fn set_result(&self, index: i64, count: usize) {
        self.result_index.set(index);
        self.result_count.set(count);
    }

    /// The navigation direction at an overlay-local position, upstream
    /// `getNavigationDirectionAt`: the button row is row 2.
    #[must_use]
    pub const fn get_navigation_direction_at(&self, row: usize, column: i64) -> Option<i8> {
        if row != 2 {
            return None;
        }
        if column >= self.previous_button_start.get() && column < self.previous_button_end.get() {
            return Some(-1);
        }
        if column >= self.next_button_start.get() && column < self.next_button_end.get() {
            return Some(1);
        }
        None
    }

    /// Track the hovered navigation direction, upstream
    /// `setHoveredNavigationDirection`; answers whether it changed.
    pub fn set_hovered_navigation_direction(&self, direction: Option<i8>) -> bool {
        if direction == self.hovered_navigation_direction.get() {
            return false;
        }
        self.hovered_navigation_direction.set(direction);
        true
    }

    /// Feed the input and report a changed query, upstream `handleInput`.
    pub fn handle_input(&self, data: &str) {
        let previous = self.input.get_value();
        self.input.handle_input(data);
        let query = self.input.get_value();
        if query != previous {
            (self.on_query_change)(&query);
        }
    }

    /// The first key bound to `action`, formatted for display, upstream's
    /// `formatKey`: capitalized parts, `Option` for `alt` on darwin, and
    /// `Unbound` when the action carries no key.
    fn format_key(key: Option<&String>) -> String {
        let Some(key) = key else {
            return "Unbound".to_string();
        };
        key.split('+')
            .map(|part| {
                if cfg!(target_os = "macos") && part.eq_ignore_ascii_case("alt") {
                    return "Option".to_string();
                }
                let mut chars = part.chars();
                chars.next().map_or_else(String::new, |first| {
                    first.to_uppercase().collect::<String>() + chars.as_str()
                })
            })
            .collect::<Vec<_>>()
            .join("+")
    }

    /// The three-line search box, upstream `AltScreenSearchComponent.render`.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's render block for block; splitting it would break the 1:1 correspondence"
    )]
    fn render_box(&self, width: usize) -> Vec<String> {
        let safe_width = width.max(1);
        let inner_width = safe_width.saturating_sub(2);
        let (previous_key, next_key) = {
            let keybindings = get_keybindings();
            (
                Self::format_key(keybindings.get_keys("tui.altScreen.searchPrevious").first()),
                Self::format_key(keybindings.get_keys("tui.altScreen.searchNext").first()),
            )
        };
        let query = self.input.get_value();
        let result = if query.is_empty() {
            String::new()
        } else if self.result_count.get() == 0 {
            "No matches".to_string()
        } else {
            format!(
                "{}/{}",
                self.result_index.get() + 1,
                self.result_count.get()
            )
        };
        let result_space = inner_width.saturating_sub(3);
        let visible_result = truncate_to_width(&result, result_space, "", false);
        let result_text = if visible_result.is_empty() {
            String::new()
        } else {
            format!("\x1b[2m {visible_result} \x1b[22m")
        };
        let input_width = inner_width.saturating_sub(visible_width(&result_text));
        let input_line = truncate_to_width(
            self.input
                .render(input_width.max(1))
                .first()
                .map_or("", String::as_str),
            input_width,
            "",
            false,
        );
        let input_padding = " ".repeat(input_width.saturating_sub(visible_width(&input_line)));
        let content = format!("{input_line}{input_padding}{result_text}");

        let mut previous_button = format!("↑ {previous_key}");
        let mut next_button = format!("↓ {next_key}");
        let mut separator = " · ";
        let outer_gap_width = 1;
        let available_controls_width = inner_width.saturating_sub(outer_gap_width * 2 + 1);
        let mut controls_width = visible_width(&previous_button)
            + visible_width(separator)
            + visible_width(&next_button);
        if controls_width > available_controls_width {
            previous_button = "↑".to_string();
            next_button = "↓".to_string();
            separator = " ";
            controls_width = visible_width(&previous_button)
                + visible_width(separator)
                + visible_width(&next_button);
        }
        let show_buttons = controls_width <= available_controls_width;
        let rendered_buttons = if show_buttons {
            format!(
                "{}{separator}{}",
                (self.navigation_button_style)(
                    &previous_button,
                    self.hovered_navigation_direction.get() == Some(-1)
                ),
                (self.navigation_button_style)(
                    &next_button,
                    self.hovered_navigation_direction.get() == Some(1)
                )
            )
        } else {
            String::new()
        };
        let outer_gaps = if show_buttons { outer_gap_width * 2 } else { 0 };
        let right_rule_width =
            usize::from(!rendered_buttons.is_empty() && inner_width > controls_width + outer_gaps);
        let left_rule_width = inner_width
            .saturating_sub(if show_buttons { controls_width } else { 0 })
            .saturating_sub(outer_gaps)
            .saturating_sub(right_rule_width);
        let previous_start = 1 + left_rule_width + outer_gap_width;
        let previous_start_i64 = i64::try_from(previous_start).unwrap_or(i64::MAX);
        self.previous_button_start
            .set(if show_buttons { previous_start_i64 } else { -1 });
        self.previous_button_end.set(if show_buttons {
            i64::try_from(previous_start + visible_width(&previous_button)).unwrap_or(i64::MAX)
        } else {
            -1
        });
        self.next_button_start.set(if show_buttons {
            self.previous_button_end.get()
                + i64::try_from(visible_width(separator)).unwrap_or(i64::MAX)
        } else {
            -1
        });
        self.next_button_end.set(if show_buttons {
            self.next_button_start.get()
                + i64::try_from(visible_width(&next_button)).unwrap_or(i64::MAX)
        } else {
            -1
        });

        if safe_width == 1 {
            return vec!["┌".to_string(), "│".to_string(), "└".to_string()];
        }
        vec![
            format!("┌{}┐", "─".repeat(inner_width)),
            format!("│{content}│"),
            format!(
                "└{}{}{}{}{}┘",
                "─".repeat(left_rule_width),
                if rendered_buttons.is_empty() { "" } else { " " },
                rendered_buttons,
                if rendered_buttons.is_empty() { "" } else { " " },
                "─".repeat(right_rule_width),
            ),
        ]
    }
}

impl Component for AltScreenSearchComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.render_box(width)
    }

    fn handle_input(&self, data: &str) {
        self.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        self.input.invalidate();
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for AltScreenSearchComponent {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.input.focused.set(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}
