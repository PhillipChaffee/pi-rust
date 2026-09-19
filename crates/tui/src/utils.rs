//! ANSI-aware terminal text measurement and manipulation, ported from
//! `packages/tui/src/utils.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements against upstream, all surveyed in map ticket "Survey the tui
//! package" and pinned by the upstream golden tests:
//!
//! - `Intl.Segmenter` grapheme/word iteration becomes `unicode-segmentation`
//!   (both implement UAX #29 extended grapheme clusters / word boundaries);
//!   the shared-segmenter accessors become pure functions.
//! - `get-east-asian-width` becomes `unicode-width`: fullwidth/wide code
//!   points measure 2, everything else measures 1, matching the upstream
//!   defaults where ambiguous stays narrow; zero-width handling lives in this
//!   module's property logic, not in the width table, exactly as upstream.
//! - The `\p{RGI_Emoji}` property regex (not expressible with the `regex`
//!   crate) becomes an exact RGI-sequence lookup in the `emojis` crate.
//! - JS UTF-16 index arithmetic becomes UTF-8 byte indexing; every scan lands
//!   on `char` boundaries, so no slice can panic.
//! - The width cache becomes a mutex-guarded FIFO map (upstream: a single
//!   threaded `Map` with first-key eviction).

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

use regex::Regex;
use unicode_segmentation::{Graphemes, UWordBounds, UnicodeSegmentation};
use unicode_width::UnicodeWidthChar;

/// Upstream segmenter accessors share one `Intl.Segmenter`; segmentation is a
/// pure function of the input here, so the accessors become iterator-returning
/// functions.
///
/// UAX #29 extended grapheme clusters, matching the upstream granularity.
#[must_use]
pub fn grapheme_segments(text: &str) -> Graphemes<'_> {
    text.graphemes(true)
}

/// Word-boundary segmentation matching `Intl.Segmenter` word granularity
/// (UAX #29), consumed by the editor's word-navigation port.
#[must_use]
pub fn word_segments(text: &str) -> UWordBounds<'_> {
    text.split_word_bounds()
}

const WIDTH_CACHE_SIZE: usize = 512;

struct WidthCache {
    map: HashMap<String, usize>,
    order: VecDeque<String>,
}

static WIDTH_CACHE: LazyLock<Mutex<WidthCache>> = LazyLock::new(|| {
    Mutex::new(WidthCache {
        map: HashMap::new(),
        order: VecDeque::new(),
    })
});

#[expect(
    clippy::expect_used,
    reason = "every pattern here is a compile-time constant; a bad pattern is a programmer error that must surface, not be swallowed"
)]
fn static_regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("static regex pattern is compile-time verified")
}

/// Clusters whose every code point is default-ignorable, a control, or a mark
/// occupy no terminal cells. The upstream class also names `\p{Surrogate}`,
/// which is vacuous here: Rust `&str` is valid UTF-8 and cannot hold a
/// surrogate code point.
static ZERO_WIDTH_RE: LazyLock<Regex> = LazyLock::new(|| {
    static_regex("^(?:\\p{Default_Ignorable_Code_Point}|\\p{Control}|\\p{Mark})+$")
});

/// Leading code points invisible to terminals, stripped before measuring a
/// cluster's base character.
static LEADING_NON_PRINTING_RE: LazyLock<Regex> = LazyLock::new(|| {
    static_regex("^[\\p{Default_Ignorable_Code_Point}\\p{Control}\\p{Format}\\p{Mark}]+")
});

/// A single non-printing code point (default ignorable, control, format, or
/// mark; the upstream class's surrogate leg is vacuous in valid UTF-8).
static NON_PRINTING_RE: LazyLock<Regex> = LazyLock::new(|| {
    static_regex("^(?:\\p{Default_Ignorable_Code_Point}|\\p{Control}|\\p{Format}|\\p{Mark})$")
});

/// A single mark (`\p{Mark}`).
static MARK_RE: LazyLock<Regex> = LazyLock::new(|| static_regex("^\\p{Mark}$"));

/// The UCD `Spacing_Mark` property as inclusive code-point ranges, generated
/// from this port's ICU baseline (Node 26.8.2, Unicode 16) — the same property
/// engine the TypeScript upstream's `\p{Spacing_Mark}` regex runs on. The
/// `regex` crate carries only general categories, and `Spacing_Mark` is a
/// `PropList` property (`gc=Mc` minus exceptions plus additions), not a category,
/// so the set cannot be spelled in a pattern here.
const SPACING_MARK_RANGES: &[(u32, u32)] = &[
    (0x903, 0x903),
    (0x93b, 0x93b),
    (0x93e, 0x940),
    (0x949, 0x94c),
    (0x94e, 0x94f),
    (0x982, 0x983),
    (0x9be, 0x9c0),
    (0x9c7, 0x9c8),
    (0x9cb, 0x9cc),
    (0x9d7, 0x9d7),
    (0xa03, 0xa03),
    (0xa3e, 0xa40),
    (0xa83, 0xa83),
    (0xabe, 0xac0),
    (0xac9, 0xac9),
    (0xacb, 0xacc),
    (0xb02, 0xb03),
    (0xb3e, 0xb3e),
    (0xb40, 0xb40),
    (0xb47, 0xb48),
    (0xb4b, 0xb4c),
    (0xb57, 0xb57),
    (0xbbe, 0xbbf),
    (0xbc1, 0xbc2),
    (0xbc6, 0xbc8),
    (0xbca, 0xbcc),
    (0xbd7, 0xbd7),
    (0xc01, 0xc03),
    (0xc41, 0xc44),
    (0xc82, 0xc83),
    (0xcbe, 0xcbe),
    (0xcc0, 0xcc4),
    (0xcc7, 0xcc8),
    (0xcca, 0xccb),
    (0xcd5, 0xcd6),
    (0xcf3, 0xcf3),
    (0xd02, 0xd03),
    (0xd3e, 0xd40),
    (0xd46, 0xd48),
    (0xd4a, 0xd4c),
    (0xd57, 0xd57),
    (0xd82, 0xd83),
    (0xdcf, 0xdd1),
    (0xdd8, 0xddf),
    (0xdf2, 0xdf3),
    (0xf3e, 0xf3f),
    (0xf7f, 0xf7f),
    (0x102b, 0x102c),
    (0x1031, 0x1031),
    (0x1038, 0x1038),
    (0x103b, 0x103c),
    (0x1056, 0x1057),
    (0x1062, 0x1064),
    (0x1067, 0x106d),
    (0x1083, 0x1084),
    (0x1087, 0x108c),
    (0x108f, 0x108f),
    (0x109a, 0x109c),
    (0x1715, 0x1715),
    (0x1734, 0x1734),
    (0x17b6, 0x17b6),
    (0x17be, 0x17c5),
    (0x17c7, 0x17c8),
    (0x1923, 0x1926),
    (0x1929, 0x192b),
    (0x1930, 0x1931),
    (0x1933, 0x1938),
    (0x1a19, 0x1a1a),
    (0x1a55, 0x1a55),
    (0x1a57, 0x1a57),
    (0x1a61, 0x1a61),
    (0x1a63, 0x1a64),
    (0x1a6d, 0x1a72),
    (0x1b04, 0x1b04),
    (0x1b35, 0x1b35),
    (0x1b3b, 0x1b3b),
    (0x1b3d, 0x1b41),
    (0x1b43, 0x1b44),
    (0x1b82, 0x1b82),
    (0x1ba1, 0x1ba1),
    (0x1ba6, 0x1ba7),
    (0x1baa, 0x1baa),
    (0x1be7, 0x1be7),
    (0x1bea, 0x1bec),
    (0x1bee, 0x1bee),
    (0x1bf2, 0x1bf3),
    (0x1c24, 0x1c2b),
    (0x1c34, 0x1c35),
    (0x1ce1, 0x1ce1),
    (0x1cf7, 0x1cf7),
    (0x302e, 0x302f),
    (0xa823, 0xa824),
    (0xa827, 0xa827),
    (0xa880, 0xa881),
    (0xa8b4, 0xa8c3),
    (0xa952, 0xa953),
    (0xa983, 0xa983),
    (0xa9b4, 0xa9b5),
    (0xa9ba, 0xa9bb),
    (0xa9be, 0xa9c0),
    (0xaa2f, 0xaa30),
    (0xaa33, 0xaa34),
    (0xaa4d, 0xaa4d),
    (0xaa7b, 0xaa7b),
    (0xaa7d, 0xaa7d),
    (0xaaeb, 0xaaeb),
    (0xaaee, 0xaaef),
    (0xaaf5, 0xaaf5),
    (0xabe3, 0xabe4),
    (0xabe6, 0xabe7),
    (0xabe9, 0xabea),
    (0xabec, 0xabec),
    (0x11000, 0x11000),
    (0x11002, 0x11002),
    (0x11082, 0x11082),
    (0x110b0, 0x110b2),
    (0x110b7, 0x110b8),
    (0x1112c, 0x1112c),
    (0x11145, 0x11146),
    (0x11182, 0x11182),
    (0x111b3, 0x111b5),
    (0x111bf, 0x111c0),
    (0x111ce, 0x111ce),
    (0x1122c, 0x1122e),
    (0x11232, 0x11233),
    (0x11235, 0x11235),
    (0x112e0, 0x112e2),
    (0x11302, 0x11303),
    (0x1133e, 0x1133f),
    (0x11341, 0x11344),
    (0x11347, 0x11348),
    (0x1134b, 0x1134d),
    (0x11357, 0x11357),
    (0x11362, 0x11363),
    (0x113b8, 0x113ba),
    (0x113c2, 0x113c2),
    (0x113c5, 0x113c5),
    (0x113c7, 0x113ca),
    (0x113cc, 0x113cd),
    (0x113cf, 0x113cf),
    (0x11435, 0x11437),
    (0x11440, 0x11441),
    (0x11445, 0x11445),
    (0x114b0, 0x114b2),
    (0x114b9, 0x114b9),
    (0x114bb, 0x114be),
    (0x114c1, 0x114c1),
    (0x115af, 0x115b1),
    (0x115b8, 0x115bb),
    (0x115be, 0x115be),
    (0x11630, 0x11632),
    (0x1163b, 0x1163c),
    (0x1163e, 0x1163e),
    (0x116ac, 0x116ac),
    (0x116ae, 0x116af),
    (0x116b6, 0x116b6),
    (0x1171e, 0x1171e),
    (0x11720, 0x11721),
    (0x11726, 0x11726),
    (0x1182c, 0x1182e),
    (0x11838, 0x11838),
    (0x11930, 0x11935),
    (0x11937, 0x11938),
    (0x1193d, 0x1193d),
    (0x11940, 0x11940),
    (0x11942, 0x11942),
    (0x119d1, 0x119d3),
    (0x119dc, 0x119df),
    (0x119e4, 0x119e4),
    (0x11a39, 0x11a39),
    (0x11a57, 0x11a58),
    (0x11a97, 0x11a97),
    (0x11b61, 0x11b61),
    (0x11b65, 0x11b65),
    (0x11b67, 0x11b67),
    (0x11c2f, 0x11c2f),
    (0x11c3e, 0x11c3e),
    (0x11ca9, 0x11ca9),
    (0x11cb1, 0x11cb1),
    (0x11cb4, 0x11cb4),
    (0x11d8a, 0x11d8e),
    (0x11d93, 0x11d94),
    (0x11d96, 0x11d96),
    (0x11ef5, 0x11ef6),
    (0x11f03, 0x11f03),
    (0x11f34, 0x11f35),
    (0x11f3e, 0x11f3f),
    (0x11f41, 0x11f41),
    (0x1612a, 0x1612c),
    (0x16f51, 0x16f87),
    (0x16ff0, 0x16ff1),
    (0x1d165, 0x1d166),
    (0x1d16d, 0x1d172),
];

/// Marks that terminals allocate cells for when attached to a base character:
/// the UCD `Spacing_Mark` set minus the three tone marks terminals render
/// zero-width, plus the legacy `wcwidth` non-spacing exceptions.
fn is_terminal_spacing_mark_cp(ch: char) -> bool {
    let cp = u32::from(ch);
    if matches!(cp, 0x1734 | 0x302e | 0x302f) {
        return false;
    }
    let in_ucd_spacing_mark = SPACING_MARK_RANGES
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                Ordering::Greater
            } else if cp > hi {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok();
    in_ucd_spacing_mark
        || matches!(
            cp,
            0x065f | 0x0f7f | 0x102b | 0x102c | 0x1031 | 0x1033..=0x1035 | 0x1038 | 0x103a..=0x103e
        )
}

/// CJK scripts break lines between any two of their graphemes, not only at
/// spaces (used by the wrapping engine).
static CJK_BREAK_RE: LazyLock<Regex> = LazyLock::new(|| {
    static_regex(
        "[\\p{Script_Extensions=Han}\\p{Script_Extensions=Hiragana}\\p{Script_Extensions=Katakana}\\p{Script_Extensions=Hangul}\\p{Script_Extensions=Bopomofo}]",
    )
});

/// The OSC 8 hyperlink-open form extracted by [`extract_ansi_code`].
static OSC8_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| static_regex("^\\x1b\\]8;[^;]*;([^\\x07\\x1b]*)(?:\\x07|\\x1b\\\\)$"));

fn char_matches(re: &Regex, ch: char) -> bool {
    let mut buf = [0_u8; 4];
    re.is_match(ch.encode_utf8(&mut buf))
}

/// Fast pre-filter mirroring upstream's `couldBeEmoji`: only clusters that
/// could be RGI emoji reach the (costly) exact lookup.
fn could_be_emoji(segment: &str) -> bool {
    let cp = u32::from(segment.chars().next().unwrap_or('\0'));
    (0x1_f000..=0x1_fbff).contains(&cp)
        || (0x2300..=0x23ff).contains(&cp)
        || (0x2600..=0x27bf).contains(&cp)
        || (0x2b50..=0x2b55).contains(&cp)
        || segment.contains('\u{fe0f}')
        || segment.encode_utf16().count() > 2
}

fn is_rgi_emoji(segment: &str) -> bool {
    emojis::get(segment).is_some()
}

fn is_printable_ascii(text: &str) -> bool {
    text.bytes().all(|b| (0x20..=0x7e).contains(&b))
}

/// East Asian width on a visible code point, matching the upstream
/// `get-east-asian-width` defaults: fullwidth/wide measure 2, everything else
/// (including ambiguous, which upstream leaves narrow) measures 1.
fn east_asian_width(ch: char) -> usize {
    if ch.width() == Some(2) { 2 } else { 1 }
}

/// Measure one grapheme cluster's terminal-cell width.
fn grapheme_width(segment: &str) -> usize {
    if segment == "\t" {
        return 3;
    }

    if !segment.is_empty() && segment.chars().all(is_terminal_spacing_mark_cp) {
        return segment.chars().count();
    }

    if ZERO_WIDTH_RE.is_match(segment) {
        return 0;
    }

    if could_be_emoji(segment) && is_rgi_emoji(segment) {
        return 2;
    }

    let base = LEADING_NON_PRINTING_RE.replace(segment, "");
    let Some(cp) = base.chars().next() else {
        return 0;
    };

    // Regional indicators stream as isolated clusters before their pair
    // arrives; terminals render them full-width even alone, and measuring
    // narrow invites auto-wrap drift.
    if (0x1_f1e6..=0x1_f1ff).contains(&u32::from(cp)) {
        return 2;
    }

    let mut width = east_asian_width(cp);

    // A cluster can hold several terminal-spacing code points after the base;
    // count the ones terminals actually give cells to.
    let mut follows_mark = false;
    for ch in base.chars().skip(1) {
        if is_terminal_spacing_mark_cp(ch) {
            width += 1;
            follows_mark = false;
        } else if char_matches(&MARK_RE, ch) {
            follows_mark = true;
        } else if !char_matches(&NON_PRINTING_RE, ch) {
            let code = u32::from(ch);
            if follows_mark || (0xff00..=0xffef).contains(&code) {
                width += east_asian_width(ch);
            } else if ch == '\u{0e33}' || ch == '\u{0eb3}' {
                width += 1;
            }
            follows_mark = false;
        }
    }

    width
}

/// Measure the visible width of a line in terminal cells, counting tabs as 3
/// and skipping ANSI/OSC/APC sequences. Repeated non-ASCII lines hit a
/// bounded FIFO cache.
#[must_use]
pub fn visible_width(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    if is_printable_ascii(text) {
        return text.chars().count();
    }

    let cached = {
        let cache = WIDTH_CACHE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.map.get(text).copied()
    };
    if let Some(width) = cached {
        return width;
    }

    let mut clean = text.to_string();
    if text.contains('\t') {
        clean = clean.replace('\t', "   ");
    }
    if clean.contains('\x1b') {
        // Strip supported ANSI/OSC/APC escape sequences in one pass: CSI
        // styling/cursor codes, OSC hyperlinks and prompt markers, and APC
        // sequences like the cursor marker.
        let mut stripped = String::with_capacity(clean.len());
        let mut i = 0;
        while i < clean.len() {
            if let Some(ansi) = extract_ansi_code(&clean, i) {
                i += ansi.length;
                continue;
            }
            let end = next_char_end(&clean, i);
            stripped.push_str(&clean[i..end]);
            i = end;
        }
        clean = stripped;
    }

    let width: usize = grapheme_segments(&clean).map(grapheme_width).sum();

    let mut cache = WIDTH_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if cache.map.len() >= WIDTH_CACHE_SIZE
        && let Some(first) = cache.order.pop_front()
    {
        cache.map.remove(&first);
    }
    cache.map.insert(text.to_string(), width);
    cache.order.push_back(text.to_string());

    width
}

/// Remove ANSI, OSC, and APC control sequences while preserving visible text.
#[must_use]
pub fn strip_terminal_sequences(text: &str) -> String {
    if !text.contains('\x1b') {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if let Some(ansi) = extract_ansi_code(text, i) {
            i += ansi.length;
            continue;
        }
        let end = next_char_end(text, i);
        result.push_str(&text[i..end]);
        i = end;
    }
    result
}

/// The terminal-cell range occupied by one grapheme cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphemeCellRange {
    /// First column the cluster occupies.
    pub start: usize,
    /// One past the last column the cluster occupies.
    pub end: usize,
}

/// The cell range of the grapheme cluster covering a visible column, or
/// `None` when the column sits past the line's visible content.
#[must_use]
pub fn get_grapheme_cell_range(line: &str, column: usize) -> Option<GraphemeCellRange> {
    let mut current_col = 0;
    let mut i = 0;
    while i < line.len() {
        if let Some(ansi) = extract_ansi_code(line, i) {
            i += ansi.length;
            continue;
        }
        let text_end = scan_text_end(line, i);
        for segment in grapheme_segments(&line[i..text_end]) {
            let width = grapheme_width(segment);
            if width > 0 && column >= current_col && column < current_col + width {
                return Some(GraphemeCellRange {
                    start: current_col,
                    end: current_col + width,
                });
            }
            current_col += width;
        }
        i = text_end;
    }
    None
}

/// The OSC 8 hyperlink URL covering a visible terminal column, or `None`.
#[must_use]
pub fn get_osc8_link_at_column(line: &str, column: usize) -> Option<&str> {
    let mut active_url: Option<&str> = None;
    let mut current_col = 0;
    let mut i = 0;
    while i < line.len() {
        if let Some(ansi) = extract_ansi_code(line, i) {
            if let Some(hyperlink) = OSC8_OPEN_RE.captures(ansi.code) {
                let url = hyperlink.get(1).map_or("", |m| m.as_str());
                active_url = if url.is_empty() { None } else { Some(url) };
            }
            i += ansi.length;
            continue;
        }
        let text_end = scan_text_end(line, i);
        for segment in grapheme_segments(&line[i..text_end]) {
            let width = if segment == "\t" {
                3
            } else {
                grapheme_width(segment)
            };
            if column >= current_col && column < current_col + width {
                return active_url;
            }
            current_col += width;
        }
        i = text_end;
    }
    None
}

/// Some terminals render precomposed Thai/Lao AM vowels inconsistently during
/// differential repaint; their compatibility decompositions have the same cell
/// width but avoid stale-cell artifacts.
///
/// Visible tabs are expanded to the fixed 3-cell width the layout assumes so
/// terminal tab stops cannot wrap a logical line; tabs inside terminal
/// sequences stay byte-identical.
#[must_use]
pub fn normalize_terminal_output(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\u{0e33}' => normalized.push_str("\u{0e4d}\u{0e32}"),
            '\u{0eb3}' => normalized.push_str("\u{0ecd}\u{0eb2}"),
            other => normalized.push(other),
        }
    }
    if !normalized.contains('\t') {
        return normalized;
    }

    let mut result = String::with_capacity(normalized.len());
    let mut i = 0;
    while i < normalized.len() {
        if let Some(ansi) = extract_ansi_code(&normalized, i) {
            result.push_str(ansi.code);
            i += ansi.length;
            continue;
        }
        if normalized.as_bytes()[i] == b'\t' {
            result.push_str("   ");
            i += 1;
            continue;
        }
        let end = next_char_end(&normalized, i);
        result.push_str(&normalized[i..end]);
        i = end;
    }
    result
}

/// One escape sequence extracted at a byte position: the sequence's text and
/// its byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnsiCode<'a> {
    /// The escape sequence, starting with `ESC`.
    pub code: &'a str,
    /// Byte length of the sequence.
    pub length: usize,
}

/// Extract the ANSI escape sequence at byte position `pos`, if one starts
/// there, or `None` when `pos` holds no sequence start.
///
/// CSI closes at `m`/`G`/`K`/`H`/`J`; OSC and APC close at `BEL` or `ESC \`.
/// A truncated sequence yields `None` — the `ESC` byte is then ordinary text,
/// so malformed input cannot hang the scanner.
#[must_use]
pub fn extract_ansi_code(line: &str, pos: usize) -> Option<AnsiCode<'_>> {
    if pos >= line.len() || !line.is_char_boundary(pos) || line.as_bytes()[pos] != 0x1b {
        return None;
    }

    let bytes = line.as_bytes();
    let next = *bytes.get(pos + 1)?;

    let closed_at = |start: usize, terminator: fn(u8, u8) -> bool| -> Option<usize> {
        let mut j = start;
        while j < bytes.len() {
            if terminator(bytes[j], bytes.get(j + 1).copied().unwrap_or(0)) {
                return Some(j);
            }
            j += 1;
        }
        None
    };

    // CSI: ESC [ ... m/G/K/H/J — first terminator byte wins.
    if next == b'[' {
        let mut j = pos + 2;
        while j < bytes.len() && !matches!(bytes[j], b'm' | b'G' | b'K' | b'H' | b'J') {
            j += 1;
        }
        if j < bytes.len() {
            return Some(AnsiCode {
                code: &line[pos..=j],
                length: j + 1 - pos,
            });
        }
        return None;
    }

    let bel_or_st = |b: u8, following: u8| b == 0x07 || (b == 0x1b && following == b'\\');
    let terminated_len = |j: usize| -> usize {
        if bytes[j] == 0x07 {
            j + 1 - pos
        } else {
            j + 2 - pos
        }
    };

    // OSC: hyperlinks, window titles, prompt markers.
    if next == b']' {
        return closed_at(pos + 2, bel_or_st).map(|j| AnsiCode {
            code: &line[pos..pos + terminated_len(j)],
            length: terminated_len(j),
        });
    }

    // APC: the cursor marker and application-specific commands.
    if next == b'_' {
        return closed_at(pos + 2, bel_or_st).map(|j| AnsiCode {
            code: &line[pos..pos + terminated_len(j)],
            length: terminated_len(j),
        });
    }

    None
}

fn scan_text_end(line: &str, start: usize) -> usize {
    let mut end = start;
    while end < line.len() && extract_ansi_code(line, end).is_none() {
        end += 1;
    }
    end
}

fn next_char_end(text: &str, i: usize) -> usize {
    text[i..].chars().next().map_or(i, |c| i + c.len_utf8())
}

/// Parse state of one candidate OSC 8 sequence.
enum Osc8Parse {
    /// Not an OSC 8 sequence at all; the caller keeps SGR-parsing it.
    NotHyperlink,
    /// An OSC 8 with an empty URL: the close form, ending the active link.
    Closed,
    /// An OSC 8 open with a URL and its parameter field.
    Active(ActiveHyperlink),
}

/// The original terminator is preserved on reopen because some terminals only
/// make BEL-terminated links clickable — OAuth login URLs use BEL, and
/// reopening wrapped lines with ST left only the first physical line
/// clickable in those terminals.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveHyperlink {
    params: String,
    url: String,
    terminator: Osc8Terminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Osc8Terminator {
    Bel,
    St,
}

fn parse_osc8_hyperlink(ansi_code: &str) -> Osc8Parse {
    if !ansi_code.starts_with("\x1b]8;") {
        return Osc8Parse::NotHyperlink;
    }

    let terminator = if ansi_code.ends_with('\x07') {
        Osc8Terminator::Bel
    } else {
        Osc8Terminator::St
    };
    let body = match terminator {
        Osc8Terminator::Bel => &ansi_code[4..ansi_code.len() - 1],
        Osc8Terminator::St => &ansi_code[4..ansi_code.len() - 2],
    };
    let Some(separator_index) = body.find(';') else {
        return Osc8Parse::NotHyperlink;
    };

    let params = &body[..separator_index];
    let url = &body[separator_index + 1..];
    if url.is_empty() {
        return Osc8Parse::Closed;
    }
    Osc8Parse::Active(ActiveHyperlink {
        params: params.to_string(),
        url: url.to_string(),
        terminator,
    })
}

fn format_osc8_hyperlink(hyperlink: &ActiveHyperlink) -> String {
    let terminator = match hyperlink.terminator {
        Osc8Terminator::Bel => "\x07",
        Osc8Terminator::St => "\x1b\\",
    };
    format!(
        "\x1b]8;{};{}{}",
        hyperlink.params, hyperlink.url, terminator
    )
}

fn format_osc8_close(terminator: Osc8Terminator) -> String {
    match terminator {
        Osc8Terminator::Bel => "\x1b]8;;\x07".to_string(),
        Osc8Terminator::St => "\x1b]8;;\x1b\\".to_string(),
    }
}

fn get_active_osc8_close(prefix: &str) -> String {
    if !prefix.contains("\x1b]8;") {
        return String::new();
    }

    let mut active_hyperlink: Option<ActiveHyperlink> = None;
    let mut i = 0;
    while i < prefix.len() {
        if let Some(ansi) = extract_ansi_code(prefix, i) {
            match parse_osc8_hyperlink(ansi.code) {
                Osc8Parse::Active(hyperlink) => active_hyperlink = Some(hyperlink),
                Osc8Parse::Closed => active_hyperlink = None,
                Osc8Parse::NotHyperlink => {}
            }
            i += ansi.length;
        } else {
            i = next_char_end(prefix, i);
        }
    }
    active_hyperlink.map_or_else(String::new, |h| format_osc8_close(h.terminator))
}

/// Tracks active ANSI SGR attributes and the active OSC 8 hyperlink so styles
/// survive line breaks.
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool mirrors one upstream SGR attribute state; folding them into a bit set would obscure the 1:1 SGR-code mapping"
)]
#[derive(Debug, Default)]
struct AnsiCodeTracker {
    bold: bool,
    dim: bool,
    italic: bool,
    underline: bool,
    blink: bool,
    inverse: bool,
    hidden: bool,
    strikethrough: bool,
    fg_color: Option<String>,
    bg_color: Option<String>,
    active_hyperlink: Option<ActiveHyperlink>,
}

impl AnsiCodeTracker {
    fn process(&mut self, ansi_code: &str) {
        match parse_osc8_hyperlink(ansi_code) {
            Osc8Parse::Active(hyperlink) => {
                self.active_hyperlink = Some(hyperlink);
                return;
            }
            Osc8Parse::Closed => {
                self.active_hyperlink = None;
                return;
            }
            Osc8Parse::NotHyperlink => {}
        }

        if !ansi_code.ends_with('m') {
            return;
        }

        let Some(params) = ansi_code
            .strip_prefix("\x1b[")
            .and_then(|rest| rest.strip_suffix('m'))
            .filter(|params| params.bytes().all(|b| b.is_ascii_digit() || b == b';'))
        else {
            return;
        };

        if params.is_empty() || params == "0" {
            self.reset();
            return;
        }

        let parts: Vec<&str> = params.split(';').collect();
        let mut i = 0;
        while i < parts.len() {
            let Ok(code) = parts[i].parse::<u32>() else {
                i += 1;
                continue;
            };

            // 256-color (38;5;N) and RGB (38;2;R;G;B) forms consume several
            // parameters; the stored code keeps the full parameter list.
            if code == 38 || code == 48 {
                if parts.get(i + 1) == Some(&"5") && parts.len() > i + 2 {
                    let color = format!("{};{};{}", parts[i], parts[i + 1], parts[i + 2]);
                    self.set_color(code, color);
                    i += 3;
                    continue;
                }
                if parts.get(i + 1) == Some(&"2") && parts.len() > i + 4 {
                    let color = format!(
                        "{};{};{};{};{}",
                        parts[i],
                        parts[i + 1],
                        parts[i + 2],
                        parts[i + 3],
                        parts[i + 4]
                    );
                    self.set_color(code, color);
                    i += 5;
                    continue;
                }
            }

            match code {
                0 => self.reset(),
                1 => self.bold = true,
                2 => self.dim = true,
                3 => self.italic = true,
                4 => self.underline = true,
                5 => self.blink = true,
                7 => self.inverse = true,
                8 => self.hidden = true,
                9 => self.strikethrough = true,
                21 => self.bold = false,
                22 => {
                    self.bold = false;
                    self.dim = false;
                }
                23 => self.italic = false,
                24 => self.underline = false,
                25 => self.blink = false,
                27 => self.inverse = false,
                28 => self.hidden = false,
                29 => self.strikethrough = false,
                39 => self.fg_color = None,
                49 => self.bg_color = None,
                other => {
                    if (30..=37).contains(&other) || (90..=97).contains(&other) {
                        self.fg_color = Some(other.to_string());
                    } else if (40..=47).contains(&other) || (100..=107).contains(&other) {
                        self.bg_color = Some(other.to_string());
                    }
                }
            }
            i += 1;
        }
    }

    fn set_color(&mut self, code: u32, color: String) {
        if code == 38 {
            self.fg_color = Some(color);
        } else {
            self.bg_color = Some(color);
        }
    }

    fn reset(&mut self) {
        self.bold = false;
        self.dim = false;
        self.italic = false;
        self.underline = false;
        self.blink = false;
        self.inverse = false;
        self.hidden = false;
        self.strikethrough = false;
        self.fg_color = None;
        self.bg_color = None;
        // SGR reset does not affect OSC 8 hyperlink state.
    }

    /// The SGR prefix that reproduces the tracker's current state, plus a
    /// re-open of the active hyperlink (with its original terminator).
    fn get_active_codes(&self) -> String {
        let mut codes: Vec<&str> = Vec::new();
        if self.bold {
            codes.push("1");
        }
        if self.dim {
            codes.push("2");
        }
        if self.italic {
            codes.push("3");
        }
        if self.underline {
            codes.push("4");
        }
        if self.blink {
            codes.push("5");
        }
        if self.inverse {
            codes.push("7");
        }
        if self.hidden {
            codes.push("8");
        }
        if self.strikethrough {
            codes.push("9");
        }
        if let Some(fg) = &self.fg_color {
            codes.push(fg);
        }
        if let Some(bg) = &self.bg_color {
            codes.push(bg);
        }

        let mut result = if codes.is_empty() {
            String::new()
        } else {
            format!("\x1b[{}m", codes.join(";"))
        };
        if let Some(hyperlink) = &self.active_hyperlink {
            result.push_str(&format_osc8_hyperlink(hyperlink));
        }
        result
    }

    /// Only the background color currently active, for line padding.
    fn get_active_background_code(&self) -> String {
        self.bg_color
            .as_ref()
            .map_or(String::new(), |bg| format!("\x1b[{bg}m"))
    }

    /// Underline must close at line ends so it cannot bleed into padding, and
    /// the active hyperlink must close to be re-opened at the next line's
    /// start via [`Self::get_active_codes`].
    fn get_line_end_reset(&self) -> String {
        let mut result = String::new();
        if self.underline {
            result.push_str("\x1b[24m");
        }
        if let Some(hyperlink) = &self.active_hyperlink {
            result.push_str(&format_osc8_close(hyperlink.terminator));
        }
        result
    }
}

fn update_tracker_from_text(text: &str, tracker: &mut AnsiCodeTracker) {
    let mut i = 0;
    while i < text.len() {
        if let Some(ansi) = extract_ansi_code(text, i) {
            tracker.process(ansi.code);
            i += ansi.length;
        } else {
            i = next_char_end(text, i);
        }
    }
}

/// Only the background color active at the end of an ANSI-styled string.
#[must_use]
pub fn get_active_background_ansi(text: &str) -> String {
    let mut tracker = AnsiCodeTracker::default();
    update_tracker_from_text(text, &mut tracker);
    tracker.get_active_background_code()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Space,
    Word,
}

/// Split text into space/word tokens with ANSI codes attached to the next
/// visible content; CJK segments always stand alone so wrapping can break
/// between them.
fn split_into_tokens_with_ansi(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut pending_ansi = String::new();
    let mut current_kind: Option<TokenKind> = None;
    let mut i = 0;

    macro_rules! flush_current {
        () => {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        };
    }

    while i < text.len() {
        if let Some(ansi) = extract_ansi_code(text, i) {
            pending_ansi.push_str(ansi.code);
            i += ansi.length;
            continue;
        }

        let end = scan_text_end(text, i);
        for segment in grapheme_segments(&text[i..end]) {
            let segment_is_space = segment == " ";
            if !segment_is_space && is_cjk_break_segment(segment) {
                flush_current!();
                let token = std::mem::take(&mut pending_ansi);
                tokens.push(token + segment);
                continue;
            }

            let segment_kind = if segment_is_space {
                TokenKind::Space
            } else {
                TokenKind::Word
            };
            if !current.is_empty() && current_kind != Some(segment_kind) {
                flush_current!();
            }

            if !pending_ansi.is_empty() {
                current.push_str(&pending_ansi);
                pending_ansi.clear();
            }

            current_kind = Some(segment_kind);
            current.push_str(segment);
        }

        i = end;
    }

    if !pending_ansi.is_empty() {
        if !current.is_empty() {
            current.push_str(&pending_ansi);
        } else if let Some(last) = tokens.last_mut() {
            last.push_str(&pending_ansi);
        }
        // No token and no current means no visible content, which cannot
        // reach here: `wrap_single_line` early-returns on lines that fit.
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn is_cjk_break_segment(segment: &str) -> bool {
    CJK_BREAK_RE.is_match(segment)
}

/// Word-wrap a line of pre-styled ANSI text to a visible width.
///
/// Only word wrapping: no padding, no background application. Active ANSI
/// state carries across line breaks; hyperlinks close at line ends and
/// re-open on continuations.
#[must_use]
pub fn wrap_text_with_ansi(text: &str, width: usize) -> Vec<String> {
    // Newlines split the input; ANSI state carries across the literal breaks.
    static LINE_BREAK_RE: LazyLock<Regex> = LazyLock::new(|| static_regex("\r\n|\r|\n"));
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut result: Vec<String> = Vec::new();
    let mut tracker = AnsiCodeTracker::default();

    for input_line in LINE_BREAK_RE.split(text) {
        let prefix = if result.is_empty() {
            String::new()
        } else {
            tracker.get_active_codes()
        };
        let wrapped = wrap_single_line(&(prefix + input_line), width);
        result.extend(wrapped);
        update_tracker_from_text(input_line, &mut tracker);
    }

    // A non-empty input always splits into at least one line, so `result`
    // cannot be empty here.
    result
}

fn wrap_single_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }

    let visible_length = visible_width(line);
    if visible_length <= width {
        return vec![line.to_string()];
    }

    let mut wrapped: Vec<String> = Vec::new();
    let mut tracker = AnsiCodeTracker::default();
    let tokens = split_into_tokens_with_ansi(line);

    let mut current_line = String::new();
    let mut current_visible_length = 0;

    for token in &tokens {
        let token_visible_length = visible_width(token);
        let is_whitespace = token.trim().is_empty();

        // A single token wider than the line breaks across lines.
        if token_visible_length > width && !is_whitespace {
            if !current_line.is_empty() {
                let line_end_reset = tracker.get_line_end_reset();
                current_line.push_str(&line_end_reset);
                wrapped.push(std::mem::take(&mut current_line));
            }

            let mut pieces = break_long_word(token, width, &mut tracker);
            let last = pieces.pop().unwrap_or_default();
            wrapped.extend(pieces);
            current_line = last;
            current_visible_length = visible_width(&current_line);
            continue;
        }

        let total_needed = current_visible_length + token_visible_length;

        if total_needed > width && current_visible_length > 0 {
            let mut line_to_wrap = current_line.trim_end().to_string();
            let line_end_reset = tracker.get_line_end_reset();
            line_to_wrap.push_str(&line_end_reset);
            wrapped.push(line_to_wrap);
            if is_whitespace {
                // Never start a continuation line with whitespace.
                current_line = tracker.get_active_codes();
                current_visible_length = 0;
            } else {
                current_line = format!("{}{}", tracker.get_active_codes(), token);
                current_visible_length = token_visible_length;
            }
        } else {
            current_line.push_str(token);
            current_visible_length += token_visible_length;
        }

        update_tracker_from_text(token, &mut tracker);
    }

    if !current_line.is_empty() {
        // No reset on the final line: the caller owns continuation.
        wrapped.push(current_line);
    }

    // Trailing whitespace can push lines past the requested width. A line
    // that did not fit always yields at least one wrapped line.
    wrapped
        .into_iter()
        .map(|line| line.trim_end().to_string())
        .collect()
}

fn break_long_word(word: &str, width: usize, tracker: &mut AnsiCodeTracker) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current_line = tracker.get_active_codes();
    let mut current_width = 0;

    // Split the word into ANSI pieces and grapheme clusters.
    let mut segments: Vec<(bool, &str)> = Vec::new();
    let mut i = 0;
    while i < word.len() {
        if let Some(ansi) = extract_ansi_code(word, i) {
            segments.push((true, ansi.code));
            i += ansi.length;
        } else {
            let end = scan_text_end(word, i);
            for grapheme in grapheme_segments(&word[i..end]) {
                segments.push((false, grapheme));
            }
            i = end;
        }
    }

    for (is_ansi, piece) in segments {
        if is_ansi {
            current_line.push_str(piece);
            tracker.process(piece);
            continue;
        }

        let piece_width = visible_width(piece);
        if current_width + piece_width > width {
            let line_end_reset = tracker.get_line_end_reset();
            current_line.push_str(&line_end_reset);
            lines.push(std::mem::take(&mut current_line));
            current_line = tracker.get_active_codes();
            current_width = 0;
        }

        current_line.push_str(piece);
        current_width += piece_width;
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
}

/// Apply a background color to a line, padding it to the full width first so
/// the background reaches the right edge.
#[must_use]
pub fn apply_background_to_line(
    line: &str,
    width: usize,
    bg_fn: impl Fn(&str) -> String,
) -> String {
    let visible_len = visible_width(line);
    let padding_needed = width.saturating_sub(visible_len);
    let padding = " ".repeat(padding_needed);
    bg_fn(&(line.to_string() + &padding))
}

/// Whether a character counts as whitespace for token splitting.
#[must_use]
pub const fn is_whitespace_char(ch: char) -> bool {
    ch.is_whitespace()
}

/// Whether a character is in the punctuation set the wrapping and word
/// navigation treat as breakable non-word material.
#[must_use]
pub const fn is_punctuation_char(ch: char) -> bool {
    matches!(
        ch,
        '(' | ')'
            | '{'
            | '}'
            | '['
            | ']'
            | '<'
            | '>'
            | '.'
            | ','
            | ';'
            | ':'
            | '\''
            | '"'
            | '!'
            | '?'
            | '+'
            | '-'
            | '='
            | '*'
            | '/'
            | '\\'
            | '|'
            | '&'
            | '%'
            | '^'
            | '$'
            | '#'
            | '@'
            | '~'
            | '`'
    )
}

/// Truncate a line to `max_width` visible cells, appending `ellipsis` when
/// content is dropped.
///
/// ANSI codes never count toward width; the result resets styling around the
/// ellipsis and re-opens (then closes) any active OSC 8 hyperlink. With
/// `pad`, the result is padded with spaces to exactly `max_width` cells.
///
/// The kept prefix is contiguous: when the next grapheme would overflow the
/// prefix budget, the prefix stops there instead of skipping a wide grapheme
/// and resuming, so terminals never see a drifted layout mid-line.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "the function mirrors upstream truncateToWidth branch-for-branch; splitting the plain and ANSI+tab paths would hide the 1:1 correspondence"
)]
pub fn truncate_to_width(text: &str, max_width: usize, ellipsis: &str, pad: bool) -> String {
    if max_width == 0 {
        return String::new();
    }

    if text.is_empty() {
        return if pad {
            " ".repeat(max_width)
        } else {
            String::new()
        };
    }

    let ellipsis_width = visible_width(ellipsis);
    if ellipsis_width >= max_width {
        let text_width = visible_width(text);
        if text_width <= max_width {
            return if pad {
                format!("{text}{}", " ".repeat(max_width - text_width))
            } else {
                text.to_string()
            };
        }

        let clipped_ellipsis = truncate_fragment_to_width(ellipsis, max_width);
        if clipped_ellipsis.width == 0 {
            return if pad {
                " ".repeat(max_width)
            } else {
                String::new()
            };
        }
        return finalize_truncated_result(
            "",
            0,
            &clipped_ellipsis.text,
            clipped_ellipsis.width,
            max_width,
            pad,
        );
    }

    if is_printable_ascii(text) {
        if text.chars().count() <= max_width {
            return if pad {
                format!("{text}{}", " ".repeat(max_width - text.chars().count()))
            } else {
                text.to_string()
            };
        }
        let target_width = max_width - ellipsis_width;
        return finalize_truncated_result(
            &text[..target_width],
            target_width,
            ellipsis,
            ellipsis_width,
            max_width,
            pad,
        );
    }

    let target_width = max_width - ellipsis_width;
    let mut result = String::new();
    let mut pending_ansi = String::new();
    let mut visible_so_far = 0;
    let mut kept_width = 0;
    let mut keep_contiguous_prefix = true;
    let mut overflowed = false;
    let has_ansi = text.contains('\x1b');
    let has_tabs = text.contains('\t');
    let mut i = 0;

    if !has_ansi && !has_tabs {
        for segment in grapheme_segments(text) {
            let width = grapheme_width(segment);
            if keep_contiguous_prefix && kept_width + width <= target_width {
                result.push_str(segment);
                kept_width += width;
            } else {
                keep_contiguous_prefix = false;
            }
            visible_so_far += width;
            if visible_so_far > max_width {
                overflowed = true;
                break;
            }
        }
    } else {
        while i < text.len() {
            if let Some(ansi) = extract_ansi_code(text, i) {
                pending_ansi.push_str(ansi.code);
                i += ansi.length;
                continue;
            }

            if text.as_bytes()[i] == b'\t' {
                if keep_contiguous_prefix && kept_width + 3 <= target_width {
                    if !pending_ansi.is_empty() {
                        result.push_str(&pending_ansi);
                        pending_ansi.clear();
                    }
                    result.push('\t');
                    kept_width += 3;
                } else {
                    keep_contiguous_prefix = false;
                    pending_ansi.clear();
                }
                visible_so_far += 3;
                if visible_so_far > max_width {
                    overflowed = true;
                    break;
                }
                i += 1;
                continue;
            }

            let end = scan_until_tab_or_ansi(text, i);

            for segment in grapheme_segments(&text[i..end]) {
                let width = grapheme_width(segment);
                if keep_contiguous_prefix && kept_width + width <= target_width {
                    if !pending_ansi.is_empty() {
                        result.push_str(&pending_ansi);
                        pending_ansi.clear();
                    }
                    result.push_str(segment);
                    kept_width += width;
                } else {
                    keep_contiguous_prefix = false;
                    pending_ansi.clear();
                }

                visible_so_far += width;
                if visible_so_far > max_width {
                    overflowed = true;
                    break;
                }
            }
            if overflowed {
                break;
            }
            i = end;
        }
    }

    let exhausted_input = if has_ansi || has_tabs {
        i >= text.len()
    } else {
        !overflowed
    };

    if !overflowed && exhausted_input {
        return if pad {
            format!(
                "{text}{}",
                " ".repeat(max_width.saturating_sub(visible_so_far))
            )
        } else {
            text.to_string()
        };
    }

    finalize_truncated_result(
        &result,
        kept_width,
        ellipsis,
        ellipsis_width,
        max_width,
        pad,
    )
}

fn scan_until_tab_or_ansi(text: &str, start: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = start;
    while end < bytes.len() && bytes[end] != b'\t' && extract_ansi_code(text, end).is_none() {
        end += 1;
    }
    end
}

fn truncate_fragment_to_width(text: &str, max_width: usize) -> SlicedSegment {
    // Only called with a non-empty ellipsis and `max_width >= 1` (the caller
    // guards `max_width == 0` before reaching here).
    if is_printable_ascii(text) {
        let clipped = &text[..max_width.min(text.len())];
        return SlicedSegment {
            text: clipped.to_string(),
            width: clipped.len(),
        };
    }

    let has_ansi = text.contains('\x1b');
    let has_tabs = text.contains('\t');
    if !has_ansi && !has_tabs {
        let mut result = String::new();
        let mut width = 0;
        for segment in grapheme_segments(text) {
            let w = grapheme_width(segment);
            if width + w > max_width {
                break;
            }
            result.push_str(segment);
            width += w;
        }
        return SlicedSegment {
            text: result,
            width,
        };
    }

    let mut result = String::new();
    let mut width = 0;
    let mut i = 0;
    let mut pending_ansi = String::new();

    while i < text.len() {
        if let Some(ansi) = extract_ansi_code(text, i) {
            pending_ansi.push_str(ansi.code);
            i += ansi.length;
            continue;
        }

        if text.as_bytes()[i] == b'\t' {
            if width + 3 > max_width {
                break;
            }
            if !pending_ansi.is_empty() {
                result.push_str(&pending_ansi);
                pending_ansi.clear();
            }
            result.push('\t');
            width += 3;
            i += 1;
            continue;
        }

        let bytes = text.as_bytes();
        let mut end = i;
        while end < bytes.len() && bytes[end] != b'\t' {
            if extract_ansi_code(text, end).is_some() {
                break;
            }
            end += 1;
        }

        for segment in grapheme_segments(&text[i..end]) {
            let w = grapheme_width(segment);
            if width + w > max_width {
                return SlicedSegment {
                    text: result,
                    width,
                };
            }
            if !pending_ansi.is_empty() {
                result.push_str(&pending_ansi);
                pending_ansi.clear();
            }
            result.push_str(segment);
            width += w;
        }
        i = end;
    }

    SlicedSegment {
        text: result,
        width,
    }
}

fn finalize_truncated_result(
    prefix: &str,
    prefix_width: usize,
    ellipsis: &str,
    ellipsis_width: usize,
    max_width: usize,
    pad: bool,
) -> String {
    const RESET: &str = "\x1b[0m";
    let hyperlink_close = get_active_osc8_close(prefix);
    let visible_width_total = prefix_width + ellipsis_width;
    let mut result = if ellipsis.is_empty() {
        format!("{prefix}{hyperlink_close}{RESET}")
    } else {
        format!("{prefix}{hyperlink_close}{RESET}{ellipsis}{RESET}")
    };

    if pad {
        result.push_str(&" ".repeat(max_width.saturating_sub(visible_width_total)));
    }
    result
}

/// A slice of a line plus the visible width it actually occupies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlicedSegment {
    /// The extracted text, including any ANSI codes that style it.
    pub text: String,
    /// Visible cell width of [`Self::text`].
    pub width: usize,
}

/// Extract a visible-column range from a line, dropping wide graphemes at the
/// boundary when `strict` (so the result's width never exceeds `length`).
#[must_use]
pub fn slice_by_column(line: &str, start_col: usize, length: usize, strict: bool) -> String {
    slice_with_width(line, start_col, length, strict).text
}

/// Like [`slice_by_column`] but also returns the visible width of the slice.
#[must_use]
pub fn slice_with_width(
    line: &str,
    start_col: usize,
    length: usize,
    strict: bool,
) -> SlicedSegment {
    if length == 0 {
        return SlicedSegment {
            text: String::new(),
            width: 0,
        };
    }
    let end_col = start_col + length;
    let mut result = String::new();
    let mut result_width = 0;
    let mut current_col = 0;
    let mut i = 0;
    let mut pending_ansi = String::new();

    while i < line.len() {
        if let Some(ansi) = extract_ansi_code(line, i) {
            if current_col >= start_col && current_col < end_col {
                result.push_str(ansi.code);
            } else if current_col < start_col {
                pending_ansi.push_str(ansi.code);
            }
            i += ansi.length;
            continue;
        }

        let text_end = scan_text_end(line, i);
        for segment in grapheme_segments(&line[i..text_end]) {
            let w = grapheme_width(segment);
            let in_range = current_col >= start_col && current_col < end_col;
            let fits = !strict || current_col + w <= end_col;
            if in_range && fits {
                if !pending_ansi.is_empty() {
                    result.push_str(&pending_ansi);
                    pending_ansi.clear();
                }
                result.push_str(segment);
                result_width += w;
            }
            current_col += w;
            if current_col >= end_col {
                break;
            }
        }
        i = text_end;
        if current_col >= end_col {
            break;
        }
    }
    SlicedSegment {
        text: result,
        width: result_width,
    }
}

/// The before/after halves of a line split around an overlay region, from one
/// pass.
///
/// The "after" half inherits the styling active before the overlay, so
/// compositing does not leak style gaps at the overlay's right edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedSegments {
    /// Content before the overlay region.
    pub before: String,
    /// Visible width of [`Self::before`].
    pub before_width: usize,
    /// Content after the overlay region.
    pub after: String,
    /// Visible width of [`Self::after`].
    pub after_width: usize,
}

/// Extract the segments before column `before_end` and after column
/// `after_start` (spanning `after_len` visible cells) in a single pass.
///
/// With `strict_after`, wide graphemes crossing the after-region's end are
/// excluded so [`ExtractedSegments::after_width`] never overshoots
/// `after_len`. ANSI codes before the region are tracked so the "after"
/// segment opens with the styling that was active at the overlay boundary.
#[must_use]
pub fn extract_segments(
    line: &str,
    before_end: usize,
    after_start: usize,
    after_len: usize,
    strict_after: bool,
) -> ExtractedSegments {
    let mut before = String::new();
    let mut before_width = 0;
    let mut after = String::new();
    let mut after_width = 0;
    let mut current_col = 0;
    let mut i = 0;
    let mut pending_ansi_before = String::new();
    let mut after_started = false;
    let after_end = after_start + after_len;

    let mut tracker = AnsiCodeTracker::default();

    let done = |current_col: usize| {
        if after_len == 0 {
            current_col >= before_end
        } else {
            current_col >= after_end
        }
    };

    while i < line.len() {
        if let Some(ansi) = extract_ansi_code(line, i) {
            tracker.process(ansi.code);
            if current_col < before_end {
                pending_ansi_before.push_str(ansi.code);
            } else if current_col >= after_start && current_col < after_end && after_started {
                after.push_str(ansi.code);
            }
            i += ansi.length;
            continue;
        }

        let text_end = scan_text_end(line, i);
        for segment in grapheme_segments(&line[i..text_end]) {
            let w = grapheme_width(segment);

            if current_col < before_end && current_col + w <= before_end {
                if !pending_ansi_before.is_empty() {
                    before.push_str(&pending_ansi_before);
                    pending_ansi_before.clear();
                }
                before.push_str(segment);
                before_width += w;
            } else if current_col >= after_start && current_col < after_end {
                let fits = !strict_after || current_col + w <= after_end;
                if fits {
                    if !after_started {
                        after.push_str(&tracker.get_active_codes());
                        after_started = true;
                    }
                    after.push_str(segment);
                    after_width += w;
                }
            }

            current_col += w;
            if done(current_col) {
                break;
            }
        }
        i = text_end;
        if done(current_col) {
            break;
        }
    }

    ExtractedSegments {
        before,
        before_width,
        after,
        after_width,
    }
}
