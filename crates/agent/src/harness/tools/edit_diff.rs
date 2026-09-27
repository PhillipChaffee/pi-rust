//! Shared diff computation for the edit tool, ported from upstream
//! `src/harness/tools/edit-diff.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The npm `diff` 8.0.4 dependency hand-ports onto this module, the same
//! shape as the CBOR codec's hand port: `Diff.diffLines` is jsdiff's Myers
//! D-path search (`base.js` `diffWithOptionsObj` plus `line.js`'s
//! line tokenizer) with jsdiff's own tie-breaks — the port keeps the
//! add-before-remove diagonal selection, the `extractCommon` run merging,
//! and the backward component chain; similar-style LCS libraries tie-break
//! ambiguities differently than jsdiff, so the port carries the dependency's
//! algorithm rather than a lookalike. `Diff.createTwoFilesPatch` with
//! `FILE_HEADERS_ONLY` is the structured-patch hunking plus `formatPatch`:
//! the `---`/`+++` headers (no index, no underline), the count-0
//! decrement quirk for `@@` starts, and the `\ No newline at end of file`
//! marker after each line that lacks its newline.
//!
//! String indices upstream are UTF-16 code units; the port's are byte
//! offsets, self-consistent across the match, span, and replacement
//! arithmetic. Upstream's `Error` throws restate as [`Result::Err`]
//! strings carrying the messages verbatim.

use unicode_normalization::UnicodeNormalization;

/// Which line ending a file uses, upstream's `"\r\n" | "\n"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineEnding {
    /// Carriage return plus line feed.
    Crlf,
    /// Line feed.
    Lf,
}

/// Detects the file's line ending: CRLF when the first CRLF precedes the
/// first bare LF, upstream's `detectLineEnding`.
#[must_use]
pub fn detect_line_ending(content: &str) -> LineEnding {
    let crlf_index = content.find("\r\n");
    let lf_index = content.find('\n');
    match (crlf_index, lf_index) {
        (Some(crlf), Some(lf)) if crlf < lf => LineEnding::Crlf,
        _ => LineEnding::Lf,
    }
}

/// Normalizes CRLF and lone CR to LF, upstream's `normalizeToLF`.
#[must_use]
pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Restores the file's original line ending over LF-normalized text,
/// upstream's `restoreLineEndings`.
#[must_use]
pub fn restore_line_endings(text: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::Crlf => text.replace('\n', "\r\n"),
        LineEnding::Lf => text.to_owned(),
    }
}

/// Normalizes text for fuzzy matching, upstream's `normalizeForFuzzyMatch`.
///
/// Applies NFKC, strips trailing whitespace per line, maps smart quotes to
/// ASCII, the Unicode dash run to ASCII hyphen, and the special Unicode
/// spaces to regular space.
#[must_use]
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let trimmed = nfkc
        .split('\n')
        .map(|line| line.trim_end_matches(is_js_trim_end_whitespace))
        .collect::<Vec<_>>()
        .join("\n");
    trimmed
        .chars()
        .map(|character| match character {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

/// The ECMAScript `trimEnd` whitespace set, which is wider than Rust's
/// `char::is_whitespace` (it adds U+00A0, U+1680, U+2000..U+200A, U+2028,
/// U+2029, U+202F, U+205F, U+3000, and U+FEFF).
const fn is_js_trim_end_whitespace(character: char) -> bool {
    matches!(
        character,
        '\t' | '\n' | '\u{000B}' | '\u{000C}' | '\r' | ' ' | '\u{00A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// Splits into lines keeping their `\n`, upstream's
/// `/[^\n]*\n|[^\n]+/g` match: a final line without its newline stays, a
/// trailing newline does not contribute an empty final line, and empty
/// lines between newlines are kept.
fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut rest = content;
    while let Some(position) = rest.find('\n') {
        lines.push(&rest[..=position]);
        rest = &rest[position + 1..];
    }
    if !rest.is_empty() {
        lines.push(rest);
    }
    lines
}

/// A line's byte span in the base content, upstream's `LineSpan`.
#[derive(Clone, Copy, Debug)]
struct LineSpan {
    start: usize,
    end: usize,
}

/// One match-driven replacement, upstream's `MatchedEdit` (the
/// `TextReplacement = Pick<MatchedEdit, ...>` projection shares the type:
/// the edit index is only read by the overlap diagnostics).
#[derive(Clone, Debug)]
struct MatchedEdit {
    edit_index: usize,
    match_index: usize,
    match_length: usize,
    new_text: String,
}

fn get_line_spans(content: &str) -> Vec<LineSpan> {
    let mut offset = 0;
    split_lines_with_endings(content)
        .into_iter()
        .map(|line| {
            let span = LineSpan {
                start: offset,
                end: offset + line.len(),
            };
            offset = span.end;
            span
        })
        .collect()
}

/// The line range one replacement touches, upstream's
/// `getReplacementLineRange`.
///
/// # Errors
/// `"Replacement range is outside the base content."` when the replacement
/// does not lie within the spans — an invariant violation.
fn get_replacement_line_range(
    lines: &[LineSpan],
    replacement: &MatchedEdit,
) -> Result<(usize, usize), String> {
    let replacement_start = replacement.match_index;
    let replacement_end = replacement.match_index + replacement.match_length;

    let start_line = lines
        .iter()
        .position(|line| replacement_start >= line.start && replacement_start < line.end)
        .ok_or_else(|| "Replacement range is outside the base content.".to_owned())?;

    let mut end_line = start_line;
    while end_line < lines.len() && lines[end_line].end < replacement_end {
        end_line += 1;
    }
    if end_line >= lines.len() {
        return Err("Replacement range is outside the base content.".to_owned());
    }

    Ok((start_line, end_line + 1))
}

fn apply_replacements(content: &str, replacements: &[MatchedEdit], offset: usize) -> String {
    let mut result = content.to_owned();
    for replacement in replacements.iter().rev() {
        #[allow(
            clippy::expect_used,
            reason = "the group slice was derived from the same matches; the subtraction is the port's invariant guard"
        )]
        let match_index = replacement
            .match_index
            .checked_sub(offset)
            .expect("the replacement lies within the group slice");
        let end = match_index + replacement.match_length;
        result = format!(
            "{}{}{}",
            &result[..match_index],
            replacement.new_text,
            &result[end..]
        );
    }
    result
}

/// Applies replacements matched against `base_content` to `original_content`
/// while preserving unchanged line blocks from the original, upstream's
/// `applyReplacementsPreservingUnchangedLines`.
///
/// Each replacement is widened to the lines it touches, those lines are
/// rewritten from the normalized base, and all other lines are copied back;
/// the actual replacement ranges drive preservation so duplicate normalized
/// lines cannot align to the wrong occurrence.
///
/// # Errors
/// The line-count mismatch message, or the replacement-outside-base
/// invariant.
fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[MatchedEdit],
) -> Result<String, String> {
    let original_lines = split_lines_with_endings(original_content);
    let base_lines = get_line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        return Err(
            "Cannot preserve unchanged lines because the base content has a different line count."
                .to_owned(),
        );
    }

    let mut sorted_replacements: Vec<&MatchedEdit> = replacements.iter().collect();
    sorted_replacements.sort_by_key(|replacement| replacement.match_index);
    let mut groups: Vec<(usize, usize, Vec<MatchedEdit>)> = Vec::new();
    for replacement in sorted_replacements {
        let (start_line, end_line) = get_replacement_line_range(&base_lines, replacement)?;
        match groups.last_mut() {
            Some((_, current_end, list)) if start_line < *current_end => {
                *current_end = (*current_end).max(end_line);
                list.push(replacement.clone());
            }
            _ => groups.push((start_line, end_line, vec![replacement.clone()])),
        }
    }

    let mut original_line_index = 0;
    let mut result = String::new();
    for (start_line, end_line, group_replacements) in &groups {
        result.push_str(&original_lines[original_line_index..*start_line].join(""));
        let group_start_offset = base_lines[*start_line].start;
        let group_end_offset = base_lines[end_line - 1].end;
        result.push_str(&apply_replacements(
            &base_content[group_start_offset..group_end_offset],
            group_replacements,
            group_start_offset,
        ));
        original_line_index = *end_line;
    }
    result.push_str(&original_lines[original_line_index..].join(""));

    Ok(result)
}

/// The fuzzy-find outcome, upstream's `FuzzyMatchResult`. When `found` is
/// false, the index and length fields carry no meaning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuzzyMatchResult {
    /// Whether a match was found.
    pub found: bool,
    /// Where the match starts, in the content used for replacement.
    pub index: usize,
    /// The matched text's length.
    pub match_length: usize,
    /// Whether fuzzy matching was used; `false` is an exact match.
    pub used_fuzzy_match: bool,
    /// The content to run replacement operations against: the original
    /// content on an exact match, the fuzzy-normalized content otherwise.
    pub content_for_replacement: String,
}

/// One old-text/new-text replacement pair, upstream's `Edit`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    /// The text to find.
    pub old_text: String,
    /// The replacement text.
    pub new_text: String,
}

/// The normalized before/after pair, upstream's `AppliedEditsResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedEditsResult {
    /// The content the matches ran against.
    pub base_content: String,
    /// The content after every replacement applied.
    pub new_content: String,
}

/// Finds `old_text` in `content`, exact first, then fuzzy, upstream's
/// `fuzzyFindText`.
///
/// On a fuzzy match the returned `content_for_replacement` is the
/// fuzzy-normalized content and the index and length are offsets into it —
/// callers compute replacements in normalized space and overlay the
/// line-level changes back onto the original.
#[must_use]
pub fn fuzzy_find_text(content: &str, old_text: &str) -> FuzzyMatchResult {
    if let Some(index) = content.find(old_text) {
        return FuzzyMatchResult {
            found: true,
            index,
            match_length: old_text.len(),
            used_fuzzy_match: false,
            content_for_replacement: content.to_owned(),
        };
    }

    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    #[allow(
        clippy::option_if_let_else,
        reason = "both arms build the full result struct; map_or_else buries the shape"
    )]
    match fuzzy_content.find(&fuzzy_old_text) {
        Some(index) => FuzzyMatchResult {
            found: true,
            index,
            match_length: fuzzy_old_text.len(),
            used_fuzzy_match: true,
            content_for_replacement: fuzzy_content,
        },
        None => FuzzyMatchResult {
            found: false,
            index: 0,
            match_length: 0,
            used_fuzzy_match: false,
            content_for_replacement: content.to_owned(),
        },
    }
}

/// The BOM split, upstream's `{ bom, text }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BomSplit {
    /// The BOM, empty when the content carries none.
    pub bom: String,
    /// The content without the BOM.
    pub text: String,
}

/// Strips a UTF-8 BOM, returning both the BOM and the text without it,
/// upstream's `stripBom`.
#[must_use]
pub fn strip_bom(content: &str) -> BomSplit {
    #[allow(
        clippy::option_if_let_else,
        reason = "both arms build the full BomSplit; map_or_else buries the shape"
    )]
    match content.strip_prefix('\u{FEFF}') {
        Some(text) => BomSplit {
            bom: "\u{FEFF}".to_owned(),
            text: text.to_owned(),
        },
        None => BomSplit {
            bom: String::new(),
            text: content.to_owned(),
        },
    }
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    fuzzy_content
        .split(&fuzzy_old_text)
        .count()
        .saturating_sub(1)
}

fn get_not_found_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
            "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
    } else {
        format!(
            "Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
        )
    }
}

fn get_duplicate_error(
    path: &str,
    edit_index: usize,
    total_edits: usize,
    occurrences: usize,
) -> String {
    if total_edits == 1 {
        format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
    } else {
        format!(
            "Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
    }
}

fn get_empty_old_text_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!("oldText must not be empty in {path}.")
    } else {
        format!("edits[{edit_index}].oldText must not be empty in {path}.")
    }
}

fn get_no_change_error(path: &str, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
            "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        )
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

/// Applies one or more exact-text replacements to LF-normalized content,
/// upstream's `applyEditsToNormalizedContent`.
///
/// All edits match against the same original content and apply in reverse
/// order so offsets stay stable. When any edit needed fuzzy matching, the
/// operation runs in fuzzy-normalized content space and overlays those
/// line-level changes onto the original so unchanged line blocks keep
/// their original bytes.
///
/// # Errors
/// The empty-`oldText`, not-found, duplicate, overlap, and no-change
/// rejections, and the preserving invariant; the messages restate
/// upstream's throws verbatim.
pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<AppliedEditsResult, String> {
    let normalized_edits: Vec<Edit> = edits
        .iter()
        .map(|edit| Edit {
            old_text: normalize_to_lf(&edit.old_text),
            new_text: normalize_to_lf(&edit.new_text),
        })
        .collect();

    for (index, edit) in normalized_edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(get_empty_old_text_error(
                path,
                index,
                normalized_edits.len(),
            ));
        }
    }

    let initial_matches: Vec<FuzzyMatchResult> = normalized_edits
        .iter()
        .map(|edit| fuzzy_find_text(normalized_content, &edit.old_text))
        .collect();
    let used_fuzzy_match = initial_matches.iter().any(|match_| match_.used_fuzzy_match);
    let replacement_base_content = if used_fuzzy_match {
        normalize_for_fuzzy_match(normalized_content)
    } else {
        normalized_content.to_owned()
    };

    let mut matched_edits: Vec<MatchedEdit> = Vec::new();
    for (index, edit) in normalized_edits.iter().enumerate() {
        let match_result = fuzzy_find_text(&replacement_base_content, &edit.old_text);
        if !match_result.found {
            return Err(get_not_found_error(path, index, normalized_edits.len()));
        }

        let occurrences = count_occurrences(&replacement_base_content, &edit.old_text);
        if occurrences > 1 {
            return Err(get_duplicate_error(
                path,
                index,
                normalized_edits.len(),
                occurrences,
            ));
        }

        matched_edits.push(MatchedEdit {
            edit_index: index,
            match_index: match_result.index,
            match_length: match_result.match_length,
            new_text: edit.new_text.clone(),
        });
    }

    matched_edits.sort_by_key(|matched| matched.match_index);
    for window in matched_edits.windows(2) {
        let (previous, current) = (&window[0], &window[1]);
        if previous.match_index + previous.match_length > current.match_index {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                previous.edit_index, current.edit_index
            ));
        }
    }

    let base_content = normalized_content.to_owned();
    let new_content = if used_fuzzy_match {
        apply_replacements_preserving_unchanged_lines(
            normalized_content,
            &replacement_base_content,
            &matched_edits,
        )?
    } else {
        apply_replacements(&replacement_base_content, &matched_edits, 0)
    };

    if base_content == new_content {
        return Err(get_no_change_error(path, normalized_edits.len()));
    }

    Ok(AppliedEditsResult {
        base_content,
        new_content,
    })
}

/// One change-run part of the line diff, npm `diff` 8.0.4's diffLines
/// component: `added` and `removed` are mutually exclusive and both false
/// for unchanged runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffPart {
    /// The part's lines joined the new content, `added: true`.
    pub added: bool,
    /// The part's lines joined from the old content, `removed: true`.
    pub removed: bool,
    /// The part's lines with their newlines.
    pub value: String,
}

impl DiffPart {
    const fn is_change(&self) -> bool {
        self.added || self.removed
    }
}

/// The line tokenizer, npm `diff` 8.0.4's `line.js` `tokenize` plus base's
/// `removeEmpty`: separators (`\n` or `\r\n`) merge into the preceding
/// content token and a trailing empty token drops.
fn line_tokens(value: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut rest = value;
    loop {
        match rest.find('\n') {
            None => {
                if !rest.is_empty() {
                    tokens.push(rest.to_owned());
                }
                break;
            }
            Some(newline_at) => {
                // The regex alternation `\n|\r\n` scans left to right: a
                // preceding `\r` joins the separator.
                let (content_end, separator_end) =
                    if newline_at > 0 && rest.as_bytes()[newline_at - 1] == b'\r' {
                        (newline_at - 1, newline_at + 1)
                    } else {
                        (newline_at, newline_at + 1)
                    };
                tokens.push(rest[..content_end].to_owned());
                if let Some(last) = tokens.last_mut() {
                    last.push_str(&rest[content_end..separator_end]);
                }
                rest = &rest[separator_end..];
            }
        }
    }
    tokens.retain(|token| !token.is_empty());
    tokens
}

/// One component of the edit script, jsdiff's backward-linked-list node.
struct Component {
    count: usize,
    added: bool,
    removed: bool,
    previous: Option<usize>,
}

/// One position on the edit graph's diagonal band, jsdiff's `bestPath`
/// entry.
#[derive(Clone, Copy)]
struct DiffPath {
    old_pos: isize,
    last_component: Option<usize>,
}

fn push_component(
    arena: &mut Vec<Component>,
    count: usize,
    added: bool,
    removed: bool,
    previous: Option<usize>,
) -> usize {
    arena.push(Component {
        count,
        added,
        removed,
        previous,
    });
    arena.len() - 1
}

/// Extends the path with one add or remove step, jsdiff's `addToPath`:
/// consecutive same-kind steps merge into one component.
fn add_to_path(
    arena: &mut Vec<Component>,
    path: DiffPath,
    added: bool,
    removed: bool,
    old_pos_inc: isize,
) -> DiffPath {
    let merges = path
        .last_component
        .is_some_and(|index| arena[index].added == added && arena[index].removed == removed);
    if merges {
        let Some(last) = path.last_component else {
            unreachable!("the merge condition implies a last component");
        };
        DiffPath {
            old_pos: path.old_pos + old_pos_inc,
            last_component: Some(push_component(
                arena,
                arena[last].count + 1,
                added,
                removed,
                arena[last].previous,
            )),
        }
    } else {
        DiffPath {
            old_pos: path.old_pos + old_pos_inc,
            last_component: Some(push_component(
                arena,
                1,
                added,
                removed,
                path.last_component,
            )),
        }
    }
}

/// Consumes the common run ahead of the path, jsdiff's `extractCommon`,
/// returning the new position.
fn extract_common(
    arena: &mut Vec<Component>,
    path: &mut DiffPath,
    diagonal_path: isize,
    old_tokens: &[String],
    new_tokens: &[String],
) -> isize {
    let old_len = old_tokens.len().cast_signed();
    let new_len = new_tokens.len().cast_signed();
    let mut old_pos = path.old_pos;
    let mut new_pos = old_pos - diagonal_path;
    let mut common_count = 0_isize;
    while new_pos + 1 < new_len
        && old_pos + 1 < old_len
        && old_tokens[(old_pos + 1).cast_unsigned()] == new_tokens[(new_pos + 1).cast_unsigned()]
    {
        new_pos += 1;
        old_pos += 1;
        common_count += 1;
    }
    if common_count > 0 {
        path.last_component = Some(push_component(
            arena,
            common_count.cast_unsigned(),
            false,
            false,
            path.last_component,
        ));
    }
    path.old_pos = old_pos;
    new_pos
}

/// Reverses the component chain into parts with their values, jsdiff's
/// `buildValues` (`useLongestToken` is false for line diffs).
fn build_values(
    arena: &[Component],
    last_component: Option<usize>,
    new_tokens: &[String],
    old_tokens: &[String],
) -> Vec<DiffPart> {
    let mut indices = Vec::new();
    let mut cursor = last_component;
    while let Some(index) = cursor {
        indices.push(index);
        cursor = arena[index].previous;
    }
    indices.reverse();

    let mut parts = Vec::with_capacity(indices.len());
    let mut new_pos = 0_usize;
    let mut old_pos = 0_usize;
    for index in indices {
        let component = &arena[index];
        if component.removed {
            let value = old_tokens[old_pos..old_pos + component.count].concat();
            old_pos += component.count;
            parts.push(DiffPart {
                added: false,
                removed: true,
                value,
            });
        } else {
            let value = new_tokens[new_pos..new_pos + component.count].concat();
            new_pos += component.count;
            if !component.added {
                old_pos += component.count;
            }
            parts.push(DiffPart {
                added: component.added,
                removed: false,
                value,
            });
        }
    }
    parts
}

/// The line-level diff, npm `diff` 8.0.4's `diffLines`: the Myers D-path
/// walk over line tokens with jsdiff's tie-breaks (add before remove when
/// the add path reaches farther), the diagonal-band pruning, and the
/// backward component chain.
pub(crate) fn line_diff(old_content: &str, new_content: &str) -> Vec<DiffPart> {
    let old_tokens = line_tokens(old_content);
    let new_tokens = line_tokens(new_content);
    let old_len = old_tokens.len().cast_signed();
    let new_len = new_tokens.len().cast_signed();
    let max_edit_length = old_len + new_len;
    let offset = max_edit_length + 1;
    let mut arena: Vec<Component> = Vec::new();
    let mut best_path: Vec<Option<DiffPath>> =
        vec![None; (2 * max_edit_length + 3).cast_unsigned()];

    best_path[offset.cast_unsigned()] = Some(DiffPath {
        old_pos: -1,
        last_component: None,
    });
    let (new_pos, seed) = {
        let Some(mut seed) = best_path[offset.cast_unsigned()] else {
            unreachable!("the seed path was just stored");
        };
        let new_pos = extract_common(&mut arena, &mut seed, 0, &old_tokens, &new_tokens);
        best_path[offset.cast_unsigned()] = Some(seed);
        let Some(seed) = best_path[offset.cast_unsigned()] else {
            unreachable!("the seed path was just stored back");
        };
        (new_pos, seed)
    };
    if seed.old_pos + 1 >= old_len && new_pos + 1 >= new_len {
        return build_values(&arena, seed.last_component, &new_tokens, &old_tokens);
    }

    let mut min_diagonal = isize::MIN;
    let mut max_diagonal = isize::MAX;
    let mut edit_length = 1_isize;
    while edit_length <= max_edit_length {
        let mut done: Option<Vec<DiffPart>> = None;
        let mut diagonal = min_diagonal.max(-edit_length);
        while diagonal <= max_diagonal.min(edit_length) {
            let remove_path = best_path[(diagonal - 1 + offset).cast_unsigned()].take();
            let add_path = best_path[(diagonal + 1 + offset).cast_unsigned()];
            let can_add = add_path.is_some_and(|path| {
                let add_path_new_pos = path.old_pos - diagonal;
                0 <= add_path_new_pos && add_path_new_pos < new_len
            });
            let can_remove = remove_path.is_some_and(|path| path.old_pos + 1 < old_len);
            if !can_add && !can_remove {
                best_path[(diagonal + offset).cast_unsigned()] = None;
            } else {
                let mut base = if can_remove {
                    match (remove_path, add_path) {
                        (Some(remove), Some(add)) if can_add && remove.old_pos < add.old_pos => {
                            add_to_path(&mut arena, add, true, false, 0)
                        }
                        (Some(remove), _) => add_to_path(&mut arena, remove, false, true, 1),
                        (None, _) => unreachable!("canRemove implies a present remove path"),
                    }
                } else {
                    let Some(add) = add_path else {
                        unreachable!("a pruned diagonal was filtered above");
                    };
                    add_to_path(&mut arena, add, true, false, 0)
                };
                let base_new_pos =
                    extract_common(&mut arena, &mut base, diagonal, &old_tokens, &new_tokens);
                if base.old_pos + 1 >= old_len && base_new_pos + 1 >= new_len {
                    done = Some(build_values(
                        &arena,
                        base.last_component,
                        &new_tokens,
                        &old_tokens,
                    ));
                    break;
                }
                best_path[(diagonal + offset).cast_unsigned()] = Some(base);
                if base.old_pos + 1 >= old_len {
                    max_diagonal = max_diagonal.min(diagonal - 1);
                }
                if base_new_pos + 1 >= new_len {
                    min_diagonal = min_diagonal.max(diagonal + 1);
                }
            }
            diagonal += 2;
        }
        if let Some(parts) = done {
            return parts;
        }
        edit_length += 1;
    }
    unreachable!("the walk reaches both ends at the latest at the maximum edit length")
}

/// A structured hunk, jsdiff's `structuredPatch` hunk object: starts and
/// counts pre-quirk, lines carrying their newlines and prefixes.
struct Hunk {
    old_start: usize,
    old_lines: usize,
    new_start: usize,
    new_lines: usize,
    lines: Vec<String>,
}

/// Builds the hunks from the diff parts, jsdiff's
/// `diffLinesResultToPatch`: the range state machine over parts (the
/// sentinel's empty part closes a trailing range with no context), the
/// `<= 2 * context` overlap merge for interior equal runs, and the
/// `\ No newline at end of file` marker pass.
fn structured_patch_hunks(parts: &[DiffPart], context: usize) -> Vec<Hunk> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut old_range_start = 0_usize;
    let mut new_range_start = 0_usize;
    let mut cur_range: Vec<String> = Vec::new();
    let mut old_line = 1_usize;
    let mut new_line = 1_usize;

    for (index, part) in parts.iter().enumerate() {
        let lines = split_lines_with_endings(&part.value);
        if part.is_change() {
            if old_range_start == 0 {
                old_range_start = old_line;
                new_range_start = new_line;
                if index > 0 {
                    let previous = split_lines_with_endings(&parts[index - 1].value);
                    if context > 0 {
                        cur_range = previous
                            .iter()
                            .rev()
                            .take(context)
                            .rev()
                            .map(|line| format!(" {line}"))
                            .collect();
                    }
                    old_range_start -= cur_range.len();
                    new_range_start -= cur_range.len();
                }
            }
            let prefix = if part.added { '+' } else { '-' };
            for line in &lines {
                cur_range.push(format!("{prefix}{line}"));
            }
            if part.added {
                new_line += lines.len();
            } else {
                old_line += lines.len();
            }
        } else if old_range_start != 0 {
            if lines.len() <= context * 2 && index < parts.len() - 1 {
                for line in &lines {
                    cur_range.push(format!(" {line}"));
                }
            } else {
                let context_size = lines.len().min(context);
                for line in &lines[..context_size] {
                    cur_range.push(format!(" {line}"));
                }
                hunks.push(Hunk {
                    old_start: old_range_start,
                    old_lines: old_line - old_range_start + context_size,
                    new_start: new_range_start,
                    new_lines: new_line - new_range_start + context_size,
                    lines: std::mem::take(&mut cur_range),
                });
                old_range_start = 0;
                new_range_start = 0;
            }
            old_line += lines.len();
            new_line += lines.len();
        } else {
            old_line += lines.len();
            new_line += lines.len();
        }
    }
    if old_range_start != 0 {
        // The sentinel's empty part closes a trailing range with no
        // context, upstream's `diff.push({ value: '', lines: [] })`.
        hunks.push(Hunk {
            old_start: old_range_start,
            old_lines: old_line - old_range_start,
            new_start: new_range_start,
            new_lines: new_line - new_range_start,
            lines: std::mem::take(&mut cur_range),
        });
    }

    for hunk in &mut hunks {
        let mut index = 0;
        while index < hunk.lines.len() {
            if hunk.lines[index].ends_with('\n') {
                hunk.lines[index].pop();
            } else {
                hunk.lines
                    .insert(index + 1, "\\ No newline at end of file".to_owned());
                index += 1;
            }
            index += 1;
        }
    }
    hunks
}

/// Generates a standard unified patch, upstream's `generateUnifiedPatch`
/// (`Diff.createTwoFilesPatch` with `FILE_HEADERS_ONLY` and a four-line
/// context default).
#[must_use]
pub fn generate_unified_patch(
    path: &str,
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> String {
    let parts = line_diff(old_content, new_content);
    let hunks = structured_patch_hunks(&parts, context_lines);
    let mut ret: Vec<String> = vec![format!("--- {path}"), format!("+++ {path}")];
    for hunk in hunks {
        // Unified Diff Format quirk: when a side's count is 0, its start is
        // one lower than the structured value, jsdiff's `formatPatch`.
        let old_start = if hunk.old_lines == 0 {
            hunk.old_start - 1
        } else {
            hunk.old_start
        };
        let new_start = if hunk.new_lines == 0 {
            hunk.new_start - 1
        } else {
            hunk.new_start
        };
        ret.push(format!(
            "@@ -{old_start},{} +{new_start},{} @@",
            hunk.old_lines, hunk.new_lines
        ));
        ret.extend(hunk.lines.iter().cloned());
    }
    format!("{}\n", ret.join("\n"))
}

/// The display diff, upstream's `generateDiffString` return.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedDiff {
    /// The diff text with line numbers and context elision.
    pub diff: String,
    /// The first changed line number in the new file, when any change
    /// exists.
    pub first_changed_line: Option<usize>,
}

/// Pushes one context line, advancing both counters, upstream's repeated
/// context-line block in `generateDiffString`.
fn push_context_line(
    output: &mut Vec<String>,
    line: &str,
    old_line_num: &mut usize,
    new_line_num: &mut usize,
    line_num_width: usize,
) {
    output.push(format!(" {old_line_num:>line_num_width$} {line}"));
    *old_line_num += 1;
    *new_line_num += 1;
}

fn elision_line(line_num_width: usize) -> String {
    format!(" {} ...", " ".repeat(line_num_width))
}

/// Generates a display-oriented diff string with line numbers and context,
/// upstream's `generateDiffString`.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "the formatter mirrors upstream's single generateDiffString loop: the four context-elision arms share the same counters"
)]
pub fn generate_diff_string(
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> GeneratedDiff {
    let parts = line_diff(old_content, new_content);
    let mut output: Vec<String> = Vec::new();

    // The split lengths include the trailing empty entry, upstream's
    // `split("\n")` array length driving the number width.
    let old_lines = old_content.split('\n').count();
    let new_lines = new_content.split('\n').count();
    let line_num_width = old_lines.max(new_lines).to_string().len();
    let pad = |number: usize| format!("{number:>line_num_width$}");

    let mut old_line_num = 1_usize;
    let mut new_line_num = 1_usize;
    let mut last_was_change = false;
    let mut first_changed_line: Option<usize> = None;

    for (index, part) in parts.iter().enumerate() {
        if part.is_change() {
            if first_changed_line.is_none() {
                first_changed_line = Some(new_line_num);
            }
            let (marker, lines, counter): (&str, Vec<&str>, &mut usize) = match part {
                DiffPart { added: true, .. } => {
                    ("+", run_lines_inner(&part.value), &mut new_line_num)
                }
                DiffPart { removed: true, .. } => {
                    ("-", run_lines_inner(&part.value), &mut old_line_num)
                }
                _ => unreachable!("the change branch excludes equal parts"),
            };
            for line in lines {
                output.push(format!("{marker}{} {line}", pad(*counter)));
                *counter += 1;
            }
            last_was_change = true;
            continue;
        }

        let raw = run_lines_inner(&part.value);
        let next_part_is_change = index < parts.len() - 1 && parts[index + 1].is_change();
        let has_leading_change = last_was_change;
        let has_trailing_change = next_part_is_change;

        if has_leading_change && has_trailing_change {
            if raw.len() <= context_lines * 2 {
                for line in &raw {
                    push_context_line(
                        &mut output,
                        line,
                        &mut old_line_num,
                        &mut new_line_num,
                        line_num_width,
                    );
                }
            } else {
                let leading_lines = &raw[..context_lines];
                let trailing_lines = &raw[raw.len() - context_lines..];
                let skipped_lines = raw.len() - leading_lines.len() - trailing_lines.len();

                for line in leading_lines {
                    push_context_line(
                        &mut output,
                        line,
                        &mut old_line_num,
                        &mut new_line_num,
                        line_num_width,
                    );
                }
                output.push(elision_line(line_num_width));
                old_line_num += skipped_lines;
                new_line_num += skipped_lines;

                for line in trailing_lines {
                    push_context_line(
                        &mut output,
                        line,
                        &mut old_line_num,
                        &mut new_line_num,
                        line_num_width,
                    );
                }
            }
        } else if has_leading_change {
            // Upstream's `slice(0, contextLines)` clamps to the array
            // length.
            let shown_lines = &raw[..raw.len().min(context_lines)];
            let skipped_lines = raw.len() - shown_lines.len();

            for line in shown_lines {
                push_context_line(
                    &mut output,
                    line,
                    &mut old_line_num,
                    &mut new_line_num,
                    line_num_width,
                );
            }
            if skipped_lines > 0 {
                output.push(elision_line(line_num_width));
                old_line_num += skipped_lines;
                new_line_num += skipped_lines;
            }
        } else if has_trailing_change {
            let skipped_lines = raw.len().saturating_sub(context_lines);
            if skipped_lines > 0 {
                output.push(elision_line(line_num_width));
                old_line_num += skipped_lines;
                new_line_num += skipped_lines;
            }
            for line in &raw[skipped_lines..] {
                push_context_line(
                    &mut output,
                    line,
                    &mut old_line_num,
                    &mut new_line_num,
                    line_num_width,
                );
            }
        } else {
            // Skip these context lines entirely.
            old_line_num += raw.len();
            new_line_num += raw.len();
        }

        last_was_change = false;
    }

    GeneratedDiff {
        diff: output.join("\n"),
        first_changed_line,
    }
}

/// The part's lines with the trailing empty split entry popped, upstream's
/// `part.value.split("\n")` pop in `generateDiffString`.
fn run_lines_inner(value: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = value.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}
