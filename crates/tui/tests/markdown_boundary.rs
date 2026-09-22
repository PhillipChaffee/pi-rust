//! Boundary tests binding the markdown port's branches the upstream suite
//! leaves untested (upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`,
//! [#48](https://github.com/PhillipChaffee/pi-rust/issues/48)): the theme
//! function and default-text-style arms, the `getStylePrefix` sentinel, the
//! port-specific gfm autolink scanners, strict strikethrough, the
//! `preserveBackslashEscapes` gap re-emission, the inline and block LaTeX
//! scanner rejections and pending forms, the narrow-table fallback, the rule
//! width cap, the source list markers, the trailing style-prefix strip, the
//! render cache, image-line passthrough, empty sources, HTML passthrough,
//! and setext headings.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;

use pi_tui::components::DefaultTextStyle;
use pi_tui::components::Markdown;
use pi_tui::components::MarkdownOptions;
use pi_tui::terminal_image::TerminalCapabilities;
use pi_tui::terminal_image::reset_capabilities_cache;
use pi_tui::terminal_image::set_capabilities;
use pi_tui::tui::Component;
use tui_support::default_markdown_theme;
use tui_support::strip_ansi;

fn markdown(source: &str) -> Markdown {
    Markdown::new(source, 0, 0, default_markdown_theme())
}

fn styled(source: &str, style: DefaultTextStyle) -> Markdown {
    Markdown::with_style(source, 0, 0, default_markdown_theme(), style)
}

fn markdown_with_options(source: &str, options: MarkdownOptions) -> Markdown {
    Markdown::with_options(source, 0, 0, default_markdown_theme(), None, options)
}

fn trimmed(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| line.trim_end().to_string())
        .collect()
}

fn plain(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| strip_ansi(line).trim_end().to_string())
        .collect()
}

fn joined_output(lines: &[String]) -> String {
    lines.join("\n")
}

const fn hyperlinks_enabled() -> TerminalCapabilities {
    TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    }
}

/// One recorded highlighter call: the line and the fence language.
type HighlightCall = (String, Option<String>);

/// The recorded highlighter calls, read through a poisoned-lock fallback so
/// the tests never panic on a poisoned mutex.
fn highlight_calls(mutex: &Mutex<Vec<HighlightCall>>) -> MutexGuard<'_, Vec<HighlightCall>> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// =============================================================================
// Theme functions and the default text style
// =============================================================================

#[test]
fn theme_functions_render_their_byte_exact_sgr_over_one_document() {
    // One document fires every MarkdownTheme entry except the link pair,
    // which the hyperlink tests pin: heading (bold cyan), code (yellow),
    // codeBlock (green), codeBlockBorder / quoteBorder / hr (dim), quote
    // (italic), listBullet (cyan), and the plain decorations bold, italic,
    // strikethrough.
    let source = "### Head\nBody *em* **st** ~~del~~ `cd`\n- it\n> q\n---\n<div>x</div>";
    let markdown = markdown(source);
    let dashes = "─".repeat(80);
    let hr = format!("\x1b[2m{dashes}\x1b[22m");
    assert_eq!(
        trimmed(&markdown.render(80)),
        [
            "\x1b[1m\x1b[36m\x1b[1m### \x1b[22m\x1b[1m\x1b[39m\x1b[22m\x1b[1m\x1b[36m\x1b[1mHead\x1b[22m\x1b[1m\x1b[39m\x1b[22m",
            "",
            "Body \x1b[3mem\x1b[23m \x1b[1mst\x1b[22m \x1b[9mdel\x1b[29m \x1b[33mcd\x1b[39m",
            "\x1b[36m- \x1b[39mit",
            "\x1b[2m│ \x1b[22m\x1b[3m\x1b[3mq\x1b[23m\x1b[3m\x1b[23m",
            "",
            hr.as_str(),
            "",
            "<div>x</div>",
        ],
    );
}

#[test]
fn code_block_indent_swaps_the_code_line_prefix() {
    let mut theme = default_markdown_theme();
    theme.code_block_indent = Some("->".to_string());
    let markdown = Markdown::new("```\nhi\n```", 0, 0, theme);
    assert_eq!(
        trimmed(&markdown.render(80)),
        [
            "\x1b[2m```\x1b[22m",
            "->\x1b[32mhi\x1b[39m",
            "\x1b[2m```\x1b[22m"
        ],
    );
}

#[test]
fn highlight_code_replaces_the_code_block_lines_and_receives_the_fence_language() {
    let calls: Arc<Mutex<Vec<HighlightCall>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&calls);
    let mut theme = default_markdown_theme();
    theme.highlight_code = Some(Arc::new(move |code: &str, lang: Option<&str>| {
        highlight_calls(&recorded).push((code.to_string(), lang.map(String::from)));
        code.lines()
            .map(|line| format!("hl[{lang:?}]{line}"))
            .collect()
    }));

    // A labelled fence forwards its language and each highlighted line gets
    // the indent plus the highlighter's line — no codeBlock styling.
    let labelled = Markdown::new("```ts\nA\nB\n```", 0, 0, theme.clone());
    assert_eq!(
        trimmed(&labelled.render(80)),
        [
            "\x1b[2m```ts\x1b[22m",
            "  hl[Some(\"ts\")]A",
            "  hl[Some(\"ts\")]B",
            "\x1b[2m```\x1b[22m"
        ],
    );

    // An unlabelled fence forwards no language.
    let unlabelled = Markdown::new("```\nZ\n```", 0, 0, theme);
    assert_eq!(
        trimmed(&unlabelled.render(80)),
        ["\x1b[2m```\x1b[22m", "  hl[None]Z", "\x1b[2m```\x1b[22m"],
    );

    assert_eq!(
        highlight_calls(&calls).clone(),
        [
            ("A\nB".to_string(), Some("ts".to_string())),
            ("Z".to_string(), None),
        ],
    );
}

#[test]
fn each_default_text_style_arm_applies_its_own_theme_function() {
    let color = styled(
        "plain",
        DefaultTextStyle {
            color: Some(tui_support::chalk_cyan()),
            ..DefaultTextStyle::default()
        },
    );
    assert_eq!(trimmed(&color.render(80)), ["\x1b[36mplain\x1b[39m"]);

    let bold = styled(
        "plain",
        DefaultTextStyle {
            bold: true,
            ..DefaultTextStyle::default()
        },
    );
    assert_eq!(trimmed(&bold.render(80)), ["\x1b[1mplain\x1b[22m"]);

    let italic = styled(
        "plain",
        DefaultTextStyle {
            italic: true,
            ..DefaultTextStyle::default()
        },
    );
    assert_eq!(trimmed(&italic.render(80)), ["\x1b[3mplain\x1b[23m"]);

    let strikethrough = styled(
        "plain",
        DefaultTextStyle {
            strikethrough: true,
            ..DefaultTextStyle::default()
        },
    );
    assert_eq!(trimmed(&strikethrough.render(80)), ["\x1b[9mplain\x1b[29m"]);

    let underline = styled(
        "plain",
        DefaultTextStyle {
            underline: true,
            ..DefaultTextStyle::default()
        },
    );
    assert_eq!(trimmed(&underline.render(80)), ["\x1b[4mplain\x1b[24m"]);

    // The foreground color applies first, then the theme's text decorations.
    let color_then_bold = styled(
        "plain",
        DefaultTextStyle {
            color: Some(tui_support::chalk_cyan()),
            bold: true,
            ..DefaultTextStyle::default()
        },
    );
    assert_eq!(
        trimmed(&color_then_bold.render(80)),
        ["\x1b[1m\x1b[36mplain\x1b[39m\x1b[22m"]
    );
}

#[test]
fn background_color_extends_across_the_full_line_width() {
    let style = DefaultTextStyle {
        bg_color: Some(Arc::new(|text| format!("\x1b[41m{text}\x1b[49m"))),
        ..DefaultTextStyle::default()
    };
    let markdown = styled("plain", style);
    // The background wraps the line after it is padded to the full width, so
    // the color reaches the right edge.
    let pad = " ".repeat(15);
    assert_eq!(
        markdown.render(20),
        vec![format!("\x1b[41mplain{pad}\x1b[49m")]
    );
}

#[test]
fn without_a_default_text_style_rendering_is_the_identity_path() {
    let without = markdown("plain **b** `c`");
    let with_empty = styled("plain **b** `c`", DefaultTextStyle::default());
    assert_eq!(
        trimmed(&without.render(80)),
        trimmed(&with_empty.render(80))
    );
    assert_eq!(plain(&without.render(80)), ["plain b c"]);
}

#[test]
fn a_default_color_fn_that_drops_the_sentinel_renders_without_prefix_reapplication() {
    // getStylePrefix probes the style with a \u{0} sentinel and reads the
    // codes before it; a color fn that drops the sentinel yields an empty
    // prefix, so nested elements are followed by no reapplication.
    let drop_sentinel = DefaultTextStyle {
        color: Some(Arc::new(|text: &str| text.replace('\u{0}', ""))),
        ..DefaultTextStyle::default()
    };
    let markdown = Markdown::with_style("a `c` b", 0, 0, default_markdown_theme(), drop_sentinel);
    assert_eq!(trimmed(&markdown.render(80)), ["a \x1b[33mc\x1b[39m b"]);

    // Contrast: a sentinel-keeping color reopens the default style after the
    // codespan's reset.
    let keeping = DefaultTextStyle {
        color: Some(tui_support::chalk_cyan()),
        ..DefaultTextStyle::default()
    };
    let markdown = Markdown::with_style("a `c` b", 0, 0, default_markdown_theme(), keeping);
    let lines = trimmed(&markdown.render(80));
    assert!(
        lines[0].contains("\x1b[36m\x1b[36m"),
        "the style prefix must be reapplied after the codespan: {lines:?}"
    );
}

#[test]
fn consecutive_strong_spans_do_not_double_the_trailing_default_style_prefix() {
    let style = DefaultTextStyle {
        color: Some(tui_support::chalk_cyan()),
        ..DefaultTextStyle::default()
    };
    let markdown = Markdown::with_style("**a** **b**", 0, 0, default_markdown_theme(), style);
    let lines = trimmed(&markdown.render(80));
    assert_eq!(
        lines,
        [
            "\x1b[1m\x1b[36ma\x1b[39m\x1b[22m\x1b[36m\x1b[36m \x1b[39m\x1b[1m\x1b[36mb\x1b[39m\x1b[22m"
        ],
    );
    assert!(
        !lines[0].ends_with("\x1b[36m"),
        "the trailing style prefix must be stripped"
    );

    // A paragraph ending with an emphasis closes cleanly as well.
    let em = styled(
        "*em*",
        DefaultTextStyle {
            color: Some(tui_support::chalk_cyan()),
            ..DefaultTextStyle::default()
        },
    );
    assert_eq!(
        trimmed(&em.render(80)),
        ["\x1b[3m\x1b[36mem\x1b[39m\x1b[23m"]
    );
}

// =============================================================================
// marked's gfm autolink scanners
// =============================================================================

#[test]
fn the_gfm_autolink_scanners_render_their_href_decisions_as_osc8_hyperlinks() {
    set_capabilities(hyperlinks_enabled());

    // The backpedal strips a trailing punctuation run; the stripped
    // characters render as plain text after the link closes.
    let joined = joined_output(&markdown("Visit https://example.com.").render(80));
    assert!(
        joined.contains("\x1b]8;;https://example.com\x1b\\"),
        "{joined}"
    );
    assert!(!joined.contains("\x1b]8;;https://example.com."), "{joined}");
    assert!(
        joined.contains("\x1b]8;;\x1b\\."),
        "the trailing dot must stay plain: {joined}"
    );

    let joined = joined_output(&markdown("(see https://example.com)").render(80));
    assert!(
        joined.contains("\x1b]8;;https://example.com\x1b\\"),
        "{joined}"
    );
    assert!(
        joined.contains("\x1b]8;;\x1b\\)"),
        "the unbalanced paren must stay plain: {joined}"
    );

    let joined = joined_output(&markdown("Go https://example.com/x, then stop").render(80));
    assert!(
        joined.contains("\x1b]8;;\x1b\\,"),
        "the trailing comma must stay plain: {joined}"
    );

    // A balanced paren group inside the URL stays part of the link.
    let joined = joined_output(&markdown("Visit https://example.com/x(y) now").render(80));
    assert!(
        joined.contains("\x1b]8;;https://example.com/x(y)\x1b\\"),
        "{joined}"
    );

    // An underscore before the paren splits the text event at the emphasis
    // delimiter, so the scanner only sees the URL up to the underscore.
    let joined = joined_output(&markdown("See https://example.com/a_(b) here").render(80));
    assert!(
        joined.contains("\x1b]8;;https://example.com/a\x1b\\"),
        "{joined}"
    );
    assert!(
        !joined.contains("\x1b]8;;https://example.com/a_"),
        "{joined}"
    );

    // Uppercase schemes autolink verbatim, no case folding.
    let joined = joined_output(&markdown("Go HTTPS://Example.COM/Docs here").render(80));
    assert!(
        joined.contains("\x1b]8;;HTTPS://Example.COM/Docs\x1b\\"),
        "{joined}"
    );

    // The `www.` alternative autolinks like marked's url rule, the href
    // gaining the http:// scheme its tokenizer prepends.
    let source = "See www.example.com/x here";
    let joined = joined_output(&markdown(source).render(80));
    assert!(
        joined.contains("\x1b]8;;http://www.example.com/x\x1b\\"),
        "{joined}"
    );

    // A multi-tld email links in full.
    let joined = joined_output(&markdown("Mail a@b.c.d now").render(80));
    assert!(joined.contains("\x1b]8;;mailto:a@b.c.d\x1b\\"), "{joined}");

    // The trailing (?![-_]) lookahead rejects a domain whose tail runs
    // straight into a `-`; an underscore instead splits the text event at
    // the emphasis delimiter, so the scan never sees it and the shorter
    // email links.
    let joined = joined_output(&markdown("Mail a@b.c- now").render(80));
    assert!(
        !joined.contains("\x1b]8;;mailto"),
        "the dash must block the link: {joined}"
    );
    let joined = joined_output(&markdown("Mail a@b.c_ now").render(80));
    assert!(joined.contains("\x1b]8;;mailto:a@b.c\x1b\\"), "{joined}");

    // The same lookahead backtracks the match: with a `-` trailing the
    // domain, the link shrinks to the alphanumeric tail before it.
    let joined = joined_output(&markdown("Mail user@example.com- `x` now").render(80));
    assert!(
        joined.contains("\x1b]8;;mailto:user@example.co\x1b\\"),
        "{joined}"
    );
    assert!(
        !joined.contains("\x1b]8;;mailto:user@example.com"),
        "{joined}"
    );

    // `!` sits in marked's email flanking class but cannot start a local
    // part, so no scan position produces a match.
    let joined = joined_output(&markdown("see !xabc@def.gh end").render(80));
    assert!(!joined.contains("\x1b]8;;mailto"), "{joined}");
    assert!(joined.contains("!xabc@def.gh"), "{joined}");

    // Bare URLs autolink inside table cells.
    let joined =
        joined_output(&markdown("| Page |\n| --- |\n| https://example.com/doc |").render(80));
    assert!(
        joined.contains("\x1b]8;;https://example.com/doc\x1b\\"),
        "{joined}"
    );

    // Hyperlinks off: the paren fallback arms.
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    // The link and linkUrl theme entries: a blue underlined label and the
    // URL dim in parentheses.
    let joined = joined_output(&markdown("[txt](https://e.com)").render(80));
    assert!(
        joined.contains("\x1b[34m\x1b[4mtxt\x1b[24m\x1b[39m\x1b[2m (https://e.com)\x1b[22m"),
        "{joined}"
    );

    // An autolinked bare URL's label equals its href, so it renders once.
    let joined = joined_output(&markdown("Visit https://example.com for more").render(80));
    assert!(
        joined.contains("\x1b[34m\x1b[4mhttps://example.com\x1b[24m\x1b[39m"),
        "{joined}"
    );
    assert!(!joined.contains("(https://example.com)"), "{joined}");

    // An autolinked email drops the mailto: prefix in the comparison too.
    let joined = joined_output(&markdown("Contact user@example.com for help").render(80));
    assert!(joined.contains("user@example.com"), "{joined}");
    assert!(!joined.contains("mailto"), "{joined}");

    reset_capabilities_cache();
}

// =============================================================================
// Strict strikethrough
// =============================================================================

#[test]
fn single_tilde_runs_render_their_literal_source() {
    let markdown = markdown("use ~x~ here");
    let lines = markdown.render(80);
    assert!(
        !joined_output(&lines).contains("\x1b[9m"),
        "single tildes must not strike: {lines:?}"
    );
    assert_eq!(plain(&lines), ["use ~x~ here"]);
}

#[test]
fn double_tilde_strikes_including_adjacent_text() {
    assert_eq!(trimmed(&markdown("~~a~~").render(80)), ["\x1b[9ma\x1b[29m"]);
    assert_eq!(
        trimmed(&markdown("a~~b~~c").render(80)),
        ["a\x1b[9mb\x1b[29mc"]
    );
}

// =============================================================================
// Backslash escapes
// =============================================================================

#[test]
fn preserve_mode_reemits_escape_backslashes_mid_run() {
    let source = "a \\\"b";
    assert_eq!(plain(&markdown(source).render(80)), ["a \"b"]);

    let preserve = markdown_with_options(
        source,
        MarkdownOptions {
            preserve_backslash_escapes: true,
            ..MarkdownOptions::default()
        },
    );
    assert_eq!(plain(&preserve.render(80)), ["a \\\"b"]);
}

#[test]
fn an_escape_leading_a_paragraph_keeps_its_backslash_in_preserve_mode() {
    // The gap opens at the backslash run's own start even when no text
    // event precedes it, so a paragraph-leading escape keeps its backslash.
    let preserve = markdown_with_options(
        "\\\"x",
        MarkdownOptions {
            preserve_backslash_escapes: true,
            ..MarkdownOptions::default()
        },
    );
    assert_eq!(plain(&preserve.render(80)), ["\\\"x"]);

    let starred = markdown_with_options(
        "\\*x",
        MarkdownOptions {
            preserve_backslash_escapes: true,
            ..MarkdownOptions::default()
        },
    );
    assert_eq!(plain(&starred.render(80)), ["\\*x"]);
}

#[test]
fn escaped_dollar_never_becomes_latex_in_either_escape_mode() {
    let source = "see \\$x$ ok";
    assert_eq!(plain(&markdown(source).render(80)), ["see $x$ ok"]);

    let preserve = markdown_with_options(
        source,
        MarkdownOptions {
            preserve_backslash_escapes: true,
            ..MarkdownOptions::default()
        },
    );
    assert_eq!(plain(&preserve.render(80)), ["see \\$x$ ok"]);
}

// =============================================================================
// Inline LaTeX scanner rules (#50 seam: every expression renders raw)
// =============================================================================

#[test]
fn currency_dollar_spans_stay_raw() {
    assert_eq!(plain(&markdown("$5 and $10").render(80)), ["$5 and $10"]);
    assert_eq!(plain(&markdown("$8k–$12k").render(80)), ["$8k–$12k"]);
    assert_eq!(plain(&markdown("$x$5").render(80)), ["$x$5"]);
}

#[test]
fn all_caps_content_before_an_identifier_after_the_close_stays_raw() {
    assert_eq!(plain(&markdown("$FOO$bar").render(80)), ["$FOO$bar"]);
}

#[test]
fn a_backtick_in_dollar_content_blocks_the_inline_latex_claim() {
    let markdown = markdown("$a `b` c$");
    let lines = markdown.render(80);
    assert_eq!(plain(&lines), ["$a b c$"]);
    assert!(
        joined_output(&lines).contains("\x1b[33m"),
        "the backtick span still styles as code"
    );
}

#[test]
fn inline_double_dollar_tokenizes_and_stays_raw() {
    assert_eq!(plain(&markdown("$$x$$").render(80)), ["$$x$$"]);
}

#[test]
fn pending_paren_and_bracket_latex_forms_stay_raw_to_context_end() {
    assert_eq!(plain(&markdown("a \\(x y z").render(80)), ["a \\(x y z"]);
    assert_eq!(plain(&markdown("a \\[b c").render(80)), ["a \\[b c"]);
}

#[test]
fn a_streamed_dollar_without_a_closer_or_math_content_stays_plain() {
    assert_eq!(
        plain(&markdown("hello $x world").render(80)),
        ["hello $x world"]
    );
}

// =============================================================================
// Block LaTeX pre-pass (#50 seam: every expression renders raw)
// =============================================================================

#[test]
fn the_block_latex_pre_pass_cuts_a_display_block_out_of_the_document() {
    // The cut is observable through the raw passthrough: the block's inner
    // markdown stays raw instead of styling as a paragraph.
    let markdown = markdown("text\n$$\n**st**\n$$\nmore");
    let lines = markdown.render(80);
    assert_eq!(
        plain(&lines),
        ["text", "", "$$", "**st**", "$$", "", "more"]
    );
    assert!(
        !joined_output(&lines).contains("\x1b[1m"),
        "the block content must stay raw: {lines:?}"
    );
}

#[test]
fn an_unclosed_bracket_block_swallows_the_rest_of_the_document() {
    assert_eq!(
        plain(&markdown("\\[a\n# not a heading").render(80)),
        ["\\[a", "# not a heading"],
    );
}

#[test]
fn a_display_block_inside_a_code_fence_is_not_latex() {
    assert_eq!(
        plain(&markdown("```\n$$\n```\n").render(80)),
        ["```", "  $$", "```"]
    );
}

// =============================================================================
// Tables, rules, and list markers
// =============================================================================

#[test]
fn a_table_too_narrow_for_its_borders_falls_back_to_the_wrapped_raw_source() {
    // Three columns need 3*3+1 = 10 border cells; at width 10 nothing is
    // left for the cells, so the raw markdown renders instead.
    let markdown = markdown("| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |");
    let lines = markdown.render(10);
    let plain_lines = plain(&lines);
    for line in &plain_lines {
        assert!(
            pi_tui::utils::visible_width(line) <= 10,
            "fallback lines must wrap to the width: {plain_lines:?}"
        );
    }
    let joined_plain = plain_lines.join("\n");
    assert!(
        !joined_plain.contains('┌'),
        "the fallback must not draw table borders: {joined_plain}"
    );
    assert!(
        joined_plain.contains("| A |"),
        "the raw source must appear: {joined_plain}"
    );
    assert!(
        joined_plain.contains("---"),
        "the raw separator row must appear: {joined_plain}"
    );
}

#[test]
fn a_rule_caps_at_eighty_columns_and_fills_narrower_widths() {
    let wide = markdown("---");
    let dashes80 = "─".repeat(80);
    let pad = " ".repeat(20);
    assert_eq!(
        wide.render(100),
        vec![format!("\x1b[2m{dashes80}\x1b[22m{pad}")]
    );

    let narrow = markdown("---");
    let dashes40 = "─".repeat(40);
    assert_eq!(
        narrow.render(40),
        vec![format!("\x1b[2m{dashes40}\x1b[22m")]
    );
}

#[test]
fn a_ten_digit_ordered_marker_is_not_a_list_at_all() {
    let options = MarkdownOptions {
        preserve_ordered_list_markers: true,
        ..MarkdownOptions::default()
    };
    // CommonMark (and pulldown with it) caps list markers at nine digits, so
    // a ten-digit marker parses as a paragraph and renders verbatim in both
    // marker modes; getOrderedListMarker's ten-digit fallback never fires.
    let ten = markdown_with_options("1234567890. x\n1234567890. y", options.clone());
    assert_eq!(plain(&ten.render(80)), ["1234567890. x", "1234567890. y"]);
    let plain_default = markdown("1234567890. x");
    assert_eq!(plain(&plain_default.render(80)), ["1234567890. x"]);

    // Nine digits still form a list, and preserve mode keeps the raw marker
    // verbatim instead of renumbering the second item.
    let nine = markdown_with_options("123456789. a\n123456789. b", options);
    assert_eq!(plain(&nine.render(80)), ["123456789. a", "123456789. b"]);
}

#[test]
fn a_bare_plus_item_renders_a_plus_bullet_when_preserving_markers() {
    let options = MarkdownOptions {
        preserve_ordered_list_markers: true,
        ..MarkdownOptions::default()
    };
    let markdown = markdown_with_options("+\n+", options);
    assert_eq!(plain(&markdown.render(80)), ["+", "+"]);
}

// =============================================================================
// Render cache, empty sources, HTML passthrough, image lines, setext headings
// =============================================================================

#[test]
fn repeated_renders_are_cached_and_set_text_and_invalidate_refresh_the_output() {
    let markdown = markdown("alpha beta gamma delta");
    let first = markdown.render(80);
    assert_eq!(
        markdown.render(80),
        first,
        "a repeated render must serve the cached lines"
    );

    // The cache is keyed by width too: a narrower render rewraps.
    let narrow = plain(&markdown.render(12));
    assert_ne!(plain(&first), narrow);
    assert!(
        narrow.len() > 1,
        "a narrow render must wrap into several lines: {narrow:?}"
    );

    markdown.set_text("replaced text here");
    assert_eq!(plain(&markdown.render(12)), ["replaced", "text here"]);

    let before_invalidate = markdown.render(12);
    markdown.invalidate();
    assert_eq!(
        markdown.render(12),
        before_invalidate,
        "invalidate must re-render identical output"
    );
}

#[test]
fn empty_and_whitespace_only_sources_render_no_lines() {
    assert!(markdown("").render(80).is_empty());

    // The whitespace-only early return ignores the vertical padding.
    let whitespace = Markdown::new("  \n\t ", 0, 2, default_markdown_theme());
    assert!(whitespace.render(80).is_empty());
    assert_eq!(whitespace.render(80), Vec::<String>::new());
}

#[test]
fn html_blocks_render_trimmed_raw_and_inline_html_renders_raw_inline() {
    let block = markdown("<div>\nhello\n</div>\n");
    assert_eq!(trimmed(&block.render(80)), ["<div>", "hello", "</div>"]);

    let inline = markdown("a <span>x</span> b");
    assert_eq!(trimmed(&inline.render(80)), ["a <span>x</span> b"]);
}

#[test]
fn an_image_line_passes_through_wrapping_and_padding_untouched() {
    let image = tui_support::kitty_image("QUJD", 4, 2, 7);
    let markdown = markdown(&format!("before {image} after"));
    // The line is far wider than 20 columns, yet no wrap, margin, or padding
    // touches it: the isImageLine checks pass it through verbatim.
    assert_eq!(markdown.render(20), vec![format!("before {image} after")]);
}

#[test]
fn a_setext_heading_renders_as_a_depth_one_heading() {
    let setext = markdown("Title\n===");
    let atx = markdown("# Title");
    assert_eq!(trimmed(&setext.render(80)), trimmed(&atx.render(80)));
    assert_eq!(
        trimmed(&setext.render(80)),
        ["\x1b[1m\x1b[36m\x1b[1m\x1b[4mTitle\x1b[24m\x1b[22m\x1b[1m\x1b[39m\x1b[22m"],
    );
}
