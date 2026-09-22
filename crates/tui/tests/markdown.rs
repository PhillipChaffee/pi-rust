//! 1:1 port of `packages/tui/test/markdown.test.ts` (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`), with the upstream
//! rendered-output goldens as the oracle.
//!
//! Restatements: the LaTeX-rendering goldens restate to raw passthrough
//! until the latex ticket ([#50](https://github.com/PhillipChaffee/pi-rust/issues/50))
//! lands the renderer — the seam answers `None`, which is markdown's
//! documented degradation; the xterm cell-attribute reads run through the
//! emulator's per-cell SGR tracking.

#![expect(
    clippy::expect_used,
    reason = "the suite asserts on finds and locks like upstream's assert.ok(...) with messages; expecting keeps the failure modes readable"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pi_tui::components::{DefaultTextStyle, Markdown, MarkdownOptions, MarkdownTheme};
use pi_tui::terminal_image::{TerminalCapabilities, reset_capabilities_cache, set_capabilities};
use pi_tui::tui::Component;
use tui_support::{VirtualTerminal, default_markdown_theme, new_test_tui, strip_ansi};

fn stripped(lines: &[String]) -> Vec<String> {
    lines.iter().map(|line| strip_ansi(line)).collect()
}

fn plain_trimmed(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| strip_ansi(line).trim_end().to_string())
        .collect()
}

fn plain_trimmed_start(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| strip_ansi(line).trim().to_string())
        .collect()
}

fn markdown(source: &str) -> Markdown {
    Markdown::new(source, 0, 0, default_markdown_theme())
}

fn markdown_with_options(source: &str, options: MarkdownOptions) -> Markdown {
    Markdown::with_options(source, 0, 0, default_markdown_theme(), None, options)
}

fn joined_output(lines: &[String]) -> String {
    lines.join("\n")
}

/// xterm's `getCell(col)` and the emulator's per-cell reads index buffer
/// columns, while `str::find` returns byte offsets, so byte offsets convert
/// through the line's chars.
fn cell_col(line: &str, byte_col: usize) -> usize {
    line[..byte_col].chars().count()
}

// --- Transforms -------------------------------------------------------------

#[test]
fn caches_transformed_markdown_by_source_and_available_width() {
    let calls: Arc<Mutex<Vec<(String, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&calls);
    let markdown = Markdown::with_options(
        "source",
        2,
        0,
        default_markdown_theme(),
        None,
        MarkdownOptions {
            transform: Some(Arc::new(move |source: &str, available_width: usize| {
                recorded
                    .lock()
                    .expect("test-local lock")
                    .push((source.to_string(), available_width));
                format!("{source} {available_width}")
            })),
            ..MarkdownOptions::default()
        },
    );

    assert_eq!(
        plain_trimmed_start(&markdown.render(80)),
        ["source 76"],
        "first render transforms with the content width"
    );
    markdown.render(80);
    assert_eq!(plain_trimmed_start(&markdown.render(60)), ["source 56"]);
    assert_eq!(
        calls.lock().expect("test-local lock").clone(),
        [("source".to_string(), 76), ("source".to_string(), 56)]
    );

    markdown.set_text("updated");
    assert_eq!(plain_trimmed_start(&markdown.render(60)), ["updated 56"]);
    assert_eq!(
        calls.lock().expect("test-local lock").last().cloned(),
        Some(("updated".to_string(), 56))
    );

    markdown.invalidate();
    markdown.render(60);
    assert_eq!(
        calls.lock().expect("test-local lock").last().cloned(),
        Some(("updated".to_string(), 56))
    );
    assert_eq!(calls.lock().expect("test-local lock").len(), 4);
}

// --- Lists ------------------------------------------------------------------

#[test]
fn renders_simple_nested_list() {
    let markdown = markdown("- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2");
    let lines = markdown.render(80);
    assert!(!lines.is_empty());
    let plain_lines = stripped(&lines);
    assert!(plain_lines.iter().any(|line| line.contains("- Item 1")));
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("    - Nested 1.1"))
    );
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("    - Nested 1.2"))
    );
    assert!(plain_lines.iter().any(|line| line.contains("- Item 2")));
}

#[test]
fn renders_deeply_nested_list() {
    let markdown = markdown("- Level 1\n  - Level 2\n    - Level 3\n      - Level 4");
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    assert!(plain_lines.iter().any(|line| line.contains("- Level 1")));
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("    - Level 2"))
    );
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("        - Level 3"))
    );
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("            - Level 4"))
    );
}

#[test]
fn renders_ordered_nested_list() {
    let markdown = markdown("1. First\n   1. Nested first\n   2. Nested second\n2. Second");
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    assert!(plain_lines.iter().any(|line| line.contains("1. First")));
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("    1. Nested first"))
    );
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("    2. Nested second"))
    );
    assert!(plain_lines.iter().any(|line| line.contains("2. Second")));
}

#[test]
fn normalizes_ordered_list_markers_by_default() {
    let markdown = markdown("1. alpha\n1. beta\n1. gamma");
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        ["1. alpha", "2. beta", "3. gamma"]
    );
}

#[test]
fn preserves_source_list_markers_when_configured() {
    let markdown = markdown_with_options(
        "  4. forth\n  3. third\n\n10) ten\n7) seven\n\n+ plus\n* star\n- minus\n+",
        MarkdownOptions {
            preserve_ordered_list_markers: true,
            ..MarkdownOptions::default()
        },
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [
            "4. forth", "3. third", "", "10) ten", "7) seven", "", "+ plus", "* star", "- minus",
            "+"
        ]
    );
}

#[test]
fn renders_mixed_ordered_and_unordered_nested_lists() {
    let markdown = markdown(
        "1. Ordered item\n   - Unordered nested\n   - Another nested\n2. Second ordered\n   - More nested",
    );
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("1. Ordered item"))
    );
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("    - Unordered nested"))
    );
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("2. Second ordered"))
    );
}

#[test]
fn renders_blank_lines_between_loose_list_items() {
    let markdown = markdown(
        "1. Lorem ipsum dolor sit amet.\n\n   Ut enim ad minim veniam.\n\n2. Duis aute irure dolor.\n\n   Excepteur sint occaecat cupidatat.\n\n3. Beep boop",
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [
            "1. Lorem ipsum dolor sit amet.",
            "",
            "   Ut enim ad minim veniam.",
            "",
            "2. Duis aute irure dolor.",
            "",
            "   Excepteur sint occaecat cupidatat.",
            "",
            "3. Beep boop",
        ]
    );
}

#[test]
fn renders_task_list_markers() {
    let markdown = markdown("- [ ] beep\n- [x] boop");
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        ["- [ ] beep", "- [x] boop"]
    );
}

#[test]
fn maintains_numbering_when_code_blocks_are_not_indented() {
    // When code blocks aren't indented, marked parses each item as a
    // separate list; the list's start preserves the original numbering.
    let markdown = markdown(
        "1. First item\n\n```typescript\n// code block\n```\n\n2. Second item\n\n```typescript\n// another code block\n```\n\n3. Third item",
    );
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines)
        .iter()
        .map(|line| line.trim().to_string())
        .collect::<Vec<_>>();
    let numbered_lines: Vec<&String> = plain_lines
        .iter()
        .filter(|line| line.chars().next().is_some_and(|c| c.is_ascii_digit()))
        .collect();
    assert_eq!(
        numbered_lines.len(),
        3,
        "expected 3 numbered items, got: {numbered_lines:?}"
    );
    assert!(
        numbered_lines[0].starts_with("1."),
        "first item should be \"1.\", got: {}",
        numbered_lines[0]
    );
    assert!(
        numbered_lines[1].starts_with("2."),
        "second item should be \"2.\", got: {}",
        numbered_lines[1]
    );
    assert!(
        numbered_lines[2].starts_with("3."),
        "third item should be \"3.\", got: {}",
        numbered_lines[2]
    );
}

#[test]
fn indents_wrapped_unordered_list_lines() {
    let markdown = markdown("- alpha beta gamma delta epsilon");
    assert_eq!(
        plain_trimmed(&markdown.render(20)),
        ["- alpha beta gamma", "  delta epsilon"]
    );
}

#[test]
fn indents_wrapped_ordered_list_lines() {
    let markdown = markdown("1. alpha beta gamma delta epsilon");
    assert_eq!(
        plain_trimmed(&markdown.render(20)),
        ["1. alpha beta gamma", "   delta epsilon"]
    );
}

#[test]
fn indents_wrapped_ordered_list_lines_with_multi_digit_markers() {
    let markdown = markdown("10. alpha beta gamma delta epsilon");
    assert_eq!(
        plain_trimmed(&markdown.render(21)),
        ["10. alpha beta gamma", "    delta epsilon"]
    );
}

#[test]
fn indents_wrapped_nested_list_lines() {
    let markdown = markdown("- parent\n  - alpha beta gamma delta epsilon");
    assert_eq!(
        plain_trimmed(&markdown.render(24)),
        ["- parent", "    - alpha beta gamma", "      delta epsilon"]
    );
}

#[test]
fn indents_wrapped_nested_list_lines_under_ordered_parents() {
    let markdown = markdown("1. parent\n   - alpha beta gamma delta epsilon");
    assert_eq!(
        plain_trimmed(&markdown.render(24)),
        ["1. parent", "    - alpha beta gamma", "      delta epsilon"]
    );
}

#[test]
fn renders_and_wraps_blockquotes_inside_list_items() {
    let markdown = markdown("- > alpha beta gamma delta epsilon zeta");
    assert_eq!(
        plain_trimmed(&markdown.render(24)),
        ["- │ alpha beta gamma", "  │ delta epsilon zeta"]
    );
}

#[test]
fn renders_and_wraps_code_blocks_inside_list_items() {
    let markdown = markdown("- ```ts\n  alpha beta gamma delta epsilon zeta\n  ```");
    assert_eq!(
        plain_trimmed(&markdown.render(24)),
        [
            "- ```ts",
            "    alpha beta gamma",
            "  delta epsilon zeta",
            "  ```"
        ]
    );
}

// --- Tables -----------------------------------------------------------------

#[test]
fn renders_simple_table() {
    let markdown = markdown("| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |");
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    assert!(plain_lines.iter().any(|line| line.contains("Name")));
    assert!(plain_lines.iter().any(|line| line.contains("Age")));
    assert!(plain_lines.iter().any(|line| line.contains("Alice")));
    assert!(plain_lines.iter().any(|line| line.contains("Bob")));
    assert!(plain_lines.iter().any(|line| line.contains("│")));
    assert!(plain_lines.iter().any(|line| line.contains("─")));
}

#[test]
fn renders_row_dividers_between_data_rows() {
    let markdown = markdown("| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |");
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    let divider_lines = plain_lines.iter().filter(|line| line.contains("┼")).count();
    assert_eq!(divider_lines, 2, "expected header + row divider");
}

#[test]
fn keeps_column_width_at_least_the_longest_word() {
    let longest_word = "superlongword";
    let markdown = markdown(
        "| Column One | Column Two |\n| --- | --- |\n| superlongword short | otherword |\n| small | tiny |",
    );
    let lines = markdown.render(32);
    let plain_lines = stripped(&lines);
    let data_line = plain_lines
        .iter()
        .find(|line| line.contains(longest_word))
        .expect("data row containing longest word");
    let segments: Vec<&str> = data_line.split('│').collect();
    let first_segment = &segments[1];
    let first_column_width = first_segment.len() - 2;
    assert!(
        first_column_width >= longest_word.len(),
        "expected first column width >= {longest_word}, got {first_column_width}"
    );
}

#[test]
fn renders_table_with_alignment() {
    let markdown = markdown(
        "| Left | Center | Right |\n| :--- | :---: | ---: |\n| A | B | C |\n| Long text | Middle | End |",
    );
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    assert!(plain_lines.iter().any(|line| line.contains("Left")));
    assert!(plain_lines.iter().any(|line| line.contains("Center")));
    assert!(plain_lines.iter().any(|line| line.contains("Right")));
    assert!(plain_lines.iter().any(|line| line.contains("Long text")));
}

#[test]
fn handles_tables_with_varying_column_widths() {
    let markdown = markdown(
        "| Short | Very long column header |\n| --- | --- |\n| A | This is a much longer cell content |\n| B | Short |",
    );
    let lines = markdown.render(80);
    assert!(!lines.is_empty());
    let plain_lines = stripped(&lines);
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("Very long column header"))
    );
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("This is a much longer cell content"))
    );
}

#[test]
fn wraps_table_cells_when_table_exceeds_available_width() {
    let markdown = markdown(
        "| Command | Description | Example |\n| --- | --- | --- |\n| npm install | Install all dependencies | npm install |\n| npm run build | Build the project | npm run build |",
    );
    let lines = markdown.render(50);
    let plain_lines = plain_trimmed(&lines);
    for line in &plain_lines {
        assert!(
            line.chars().count() <= 50,
            "line exceeds width 50: \"{line}\" (length: {})",
            line.chars().count()
        );
    }
    let all_text = plain_lines.join(" ");
    assert!(all_text.contains("Command"), "should contain 'Command'");
    assert!(
        all_text.contains("Description"),
        "should contain 'Description'"
    );
    assert!(
        all_text.contains("npm install"),
        "should contain 'npm install'"
    );
    assert!(all_text.contains("Install"), "should contain 'Install'");
}

#[test]
fn does_not_leak_wrapped_link_styles_into_table_borders_or_plain_cells() {
    let _capability_lock = tui_support::capabilities_lock();
    let source = "| Link | Plain |\n| --- | --- |\n| [**one two three four five six**](https://example.com) | normal text |";
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    });
    assert_table_link_cells(24, 16, source, true);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    assert_table_link_cells(24, 16, source, false);
    reset_capabilities_cache();
}

fn assert_table_link_cells(columns: u16, rows: u16, source: &str, hyperlinks: bool) {
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks,
    });
    let terminal = VirtualTerminal::new(columns, rows);
    let tui = new_test_tui(terminal.clone());
    tui.add_child(Rc::new(markdown(source)));
    tui.start();
    tui_support::wait_for_render(&tui);
    let viewport = terminal.get_viewport();
    let row = viewport
        .iter()
        .position(|line| line.contains("one") && line.contains("norm"));
    assert_ne!(row, None, "missing wrapped table row: {viewport:?}");
    let row = row.expect("row found");
    let line = &viewport[row];
    let link_col = line.find("one").expect("link col");
    let separator_col = line[link_col..]
        .find("│")
        .map(|offset| link_col + offset)
        .expect("separator col");
    let plain_col = line.find("norm").expect("plain col");
    assert!(separator_col > link_col && plain_col > separator_col);
    assert!(
        !terminal.is_fg_default(row, cell_col(line, link_col)),
        "link cell should be styled"
    );
    assert!(
        terminal.is_fg_default(row, cell_col(line, separator_col)),
        "separator should be unstyled"
    );
    assert!(
        terminal.is_fg_default(row, cell_col(line, plain_col)),
        "plain cell should be unstyled"
    );
    assert!(
        terminal.is_bold(row, cell_col(line, link_col)),
        "link cell should be bold"
    );
    assert!(
        !terminal.is_bold(row, cell_col(line, separator_col)),
        "separator should not be bold"
    );
    assert!(
        !terminal.is_bold(row, cell_col(line, plain_col)),
        "plain cell should not be bold"
    );
    if !hyperlinks {
        let url_row = viewport.iter().position(|line| line.contains("https"));
        assert_ne!(url_row, None, "missing fallback URL row: {viewport:?}");
        let url_row = url_row.expect("url row");
        let url_line = &viewport[url_row];
        let url_col = url_line.find("https").expect("url col");
        let url_separator_col = url_line[url_col..]
            .find("│")
            .map(|offset| url_col + offset)
            .expect("url separator col");
        let url_border_col = url_line.rfind("│").expect("url border col");
        assert!(url_separator_col > url_col && url_border_col > url_separator_col);
        assert!(
            terminal.is_dim(url_row, cell_col(url_line, url_col)),
            "URL should be dim"
        );
        assert!(
            !terminal.is_dim(url_row, cell_col(url_line, url_separator_col)),
            "separator should not be dim"
        );
        assert!(
            !terminal.is_dim(url_row, cell_col(url_line, url_border_col)),
            "border should not be dim"
        );
    }
    tui_support::stop(&tui);
}

#[test]
fn restores_the_enclosing_style_after_a_wrapped_table_link() {
    let _capability_lock = tui_support::capabilities_lock();
    let quote_color: u32 = 0x12_34_56;
    let source = "> | Link | Plain |\n> | --- | --- |\n> | [one two three four five six](https://example.com) | normal text |";
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: true,
        hyperlinks: true,
    });
    let terminal = VirtualTerminal::new(28, 10);
    let tui = new_test_tui(terminal.clone());
    tui.add_child(Rc::new(Markdown::with_options(
        source,
        0,
        0,
        quote_link_theme(),
        None,
        MarkdownOptions::default(),
    )));
    tui.start();
    tui_support::wait_for_render(&tui);
    let viewport = terminal.get_viewport();
    let row = viewport
        .iter()
        .position(|line| line.contains("one") && line.contains("normal"));
    assert_ne!(
        row, None,
        "missing wrapped blockquote table row: {viewport:?}"
    );
    let row = row.expect("row found");
    let line = &viewport[row];
    let link_col = line.find("one").expect("link col");
    let separator_col = line[link_col..]
        .find("│")
        .map(|offset| link_col + offset)
        .expect("separator col");
    let plain_col = line.find("normal").expect("plain col");
    assert!(separator_col > link_col && plain_col > separator_col);
    assert_ne!(
        terminal.fg_color(row, cell_col(line, link_col)),
        Some(quote_color),
        "link cell should not carry the quote color"
    );
    assert_eq!(
        terminal.fg_color(row, cell_col(line, separator_col)),
        Some(quote_color),
        "separator should carry the quote color"
    );
    assert_eq!(
        terminal.fg_color(row, cell_col(line, plain_col)),
        Some(quote_color),
        "plain cell should carry the quote color"
    );

    let final_row = viewport.iter().position(|line| line.contains("five six"));
    assert_ne!(
        final_row, None,
        "missing final wrapped link row: {viewport:?}"
    );
    let final_row = final_row.expect("final row");
    let final_line = &viewport[final_row];
    let final_link_col = final_line.find("five six").expect("final link col");
    let final_separator_col = final_line[final_link_col..]
        .find("│")
        .map(|offset| final_link_col + offset)
        .expect("final separator col");
    let final_border_col = final_line.rfind("│").expect("final border col");
    assert!(final_separator_col > final_link_col && final_border_col > final_separator_col);
    assert_eq!(
        terminal.fg_color(final_row, cell_col(final_line, final_separator_col)),
        Some(quote_color)
    );
    assert_eq!(
        terminal.fg_color(final_row, cell_col(final_line, final_border_col)),
        Some(quote_color)
    );
    tui_support::stop(&tui);
    reset_capabilities_cache();
}

/// The test-25 theme: raw RGB wrappers that do not reopen after nested
/// resets, upstream's "basic wrapper" comment.
fn quote_link_theme() -> MarkdownTheme {
    let theme = default_markdown_theme();
    MarkdownTheme {
        quote: Arc::new(|text| format!("\x1b[38;2;18;52;86m{text}\x1b[39m")),
        link: Arc::new(|text| format!("\x1b[38;2;129;162;190m{text}\x1b[39m")),
        ..theme
    }
}

#[test]
fn wraps_long_cell_content_to_multiple_lines() {
    let markdown =
        markdown("| Header |\n| --- |\n| This is a very long cell content that should wrap |");
    let lines = markdown.render(25);
    let plain_lines = plain_trimmed(&lines);
    let data_rows = plain_lines
        .iter()
        .filter(|line| line.starts_with('│') && !line.contains('─'))
        .count();
    assert!(data_rows > 2, "expected wrapped rows, got {data_rows} rows");
    let all_text = plain_lines.join(" ");
    assert!(
        all_text.contains("very long"),
        "should preserve 'very long'"
    );
    assert!(
        all_text.contains("cell content"),
        "should preserve 'cell content'"
    );
    assert!(
        all_text.contains("should wrap"),
        "should preserve 'should wrap'"
    );
}

#[test]
fn wraps_long_unbroken_tokens_inside_table_cells() {
    let _capability_lock = tui_support::capabilities_lock();
    // Pin to no-hyperlinks so width checks work on plain text without
    // OSC 8 sequences.
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let url = "https://example.com/this/is/a/very/long/url/that/should/wrap";
    let markdown = markdown_with_url(url);
    let width = 30;
    let lines = markdown.render(width);
    reset_capabilities_cache();
    let plain_lines = plain_trimmed(&lines);
    for line in &plain_lines {
        assert!(
            line.chars().count() <= width,
            "line exceeds width {width}: \"{line}\""
        );
    }
    let table_lines: Vec<&String> = plain_lines
        .iter()
        .filter(|line| line.starts_with('│'))
        .collect();
    assert!(!table_lines.is_empty(), "expected table rows to render");
    for line in &table_lines {
        let border_count = line.matches('│').count();
        assert_eq!(
            border_count, 2,
            "expected 2 borders, got {border_count}: \"{line}\""
        );
    }
    let extracted = plain_lines
        .join("")
        .chars()
        .filter(|c| !matches!(c, '│' | '├' | '┤' | '─' | ' '))
        .collect::<String>();
    assert!(extracted.contains("prefix"), "should preserve 'prefix'");
    assert!(extracted.contains(url), "should preserve URL");
}

#[test]
fn wraps_styled_inline_code_inside_table_cells_without_breaking_borders() {
    let markdown = markdown("| Code |\n| --- |\n| `averyveryveryverylongidentifier` |");
    let width = 20;
    let lines = markdown.render(width);
    let joined_output = joined_output(&lines);
    assert!(
        joined_output.contains("\x1b[33m"),
        "inline code should be styled (yellow)"
    );
    let plain_lines = plain_trimmed(&lines);
    for line in &plain_lines {
        assert!(
            line.chars().count() <= width,
            "line exceeds width {width}: \"{line}\""
        );
    }
    let table_lines: Vec<&String> = plain_lines
        .iter()
        .filter(|line| line.starts_with('│'))
        .collect();
    for line in &table_lines {
        let border_count = line.matches('│').count();
        assert_eq!(border_count, 2, "expected 2 borders, got {border_count}");
    }
}

#[test]
fn handles_extremely_narrow_width_gracefully() {
    let markdown = markdown("| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |");
    let lines = markdown.render(15);
    let plain_lines = plain_trimmed(&lines);
    assert!(!lines.is_empty(), "should produce output");
    for line in &plain_lines {
        assert!(
            line.chars().count() <= 15,
            "line exceeds width 15: \"{line}\""
        );
    }
}

#[test]
fn renders_table_correctly_when_it_fits_naturally() {
    let markdown = markdown("| A | B |\n| --- | --- |\n| 1 | 2 |");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    let header_line = plain_lines
        .iter()
        .find(|line| line.contains('A') && line.contains('B'));
    assert!(header_line.is_some(), "should have header row");
    assert!(
        header_line.is_some_and(|line| line.contains('│')),
        "header should have borders"
    );
    let separator_line = plain_lines
        .iter()
        .find(|line| line.contains('├') && line.contains('┼'));
    assert!(separator_line.is_some(), "should have separator row");
    let data_line = plain_lines
        .iter()
        .find(|line| line.contains('1') && line.contains('2'));
    assert!(data_line.is_some(), "should have data row");
}

#[test]
fn respects_padding_x_when_calculating_table_width() {
    let markdown = Markdown::new(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,
        0,
        default_markdown_theme(),
    );
    let lines = markdown.render(40);
    let plain_lines = plain_trimmed(&lines);
    for line in &plain_lines {
        assert!(
            line.chars().count() <= 40,
            "line exceeds width 40: \"{line}\""
        );
    }
    let table_row = plain_lines.iter().find(|line| line.contains('│'));
    assert!(
        table_row.is_some_and(|line| line.starts_with("  ")),
        "table should have left padding"
    );
}

#[test]
fn does_not_add_a_trailing_blank_line_when_table_is_last_rendered_block() {
    let markdown = markdown("| Name |\n| --- |\n| Alice |");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    assert_ne!(
        plain_lines.last(),
        Some(&String::new()),
        "expected table to end without a blank line: {plain_lines:?}"
    );
}

// --- Combined features ------------------------------------------------------

#[test]
fn renders_lists_and_tables_together() {
    let markdown = markdown(
        "# Test Document\n\n- Item 1\n  - Nested item\n- Item 2\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
    );
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("Test Document"))
    );
    assert!(plain_lines.iter().any(|line| line.contains("- Item 1")));
    assert!(
        plain_lines
            .iter()
            .any(|line| line.contains("    - Nested item"))
    );
    assert!(plain_lines.iter().any(|line| line.contains("Col1")));
    assert!(plain_lines.iter().any(|line| line.contains('│')));
}

// --- LaTeX math -------------------------------------------------------------
// Restated (#50): the LaTeX renderer lands with the latex ticket; until then
// the seam answers `None` and every expression renders as its raw source —
// markdown's documented degradation. The tokenization decisions (currency
// suppression, pending streaming, delimiter boundaries) port 1:1.

#[test]
fn renders_inline_dollar_and_parenthesis_delimiters() {
    let markdown = markdown(
        r"A map $\mathbb{C}^3 \to \mathbb{C}^3$, $xy$, $x-y$, $-x$, $\frac{1}{2}$, and \(s \to \infty\).",
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [
            r"A map $\mathbb{C}^3 \to \mathbb{C}^3$, $xy$, $x-y$, $-x$, $\frac{1}{2}$, and \(s",
            r"\to \infty\).",
        ]
    );
}

#[test]
fn renders_display_dollar_delimiters_without_markdown_escape_corruption() {
    let markdown = markdown(
        r"Before

$$\{3x+2y,\; x \in \{0, \pm 1\}\}$$

after",
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [
            "Before",
            "",
            r"$$\{3x+2y,\; x \in \{0, \pm 1\}\}$$",
            "",
            "after"
        ]
    );
}

#[test]
fn renders_display_bracket_delimiters() {
    let markdown = markdown(
        r"Before

\[
E \approx \frac{0.1\ \text{lux}}{100\ \text{lm/W}}
\]

after",
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [
            "Before",
            "",
            r"\[",
            r"E \approx \frac{0.1\ \text{lux}}{100\ \text{lm/W}}",
            r"\]",
            "",
            "after"
        ]
    );
}

#[test]
fn aligns_matrix_rows_with_the_opening_delimiter() {
    let markdown = markdown(
        r"Consider the matrix

\[
A=
\begin{pmatrix}
\pi & 0\\
0 & \frac{1}{\pi}
\end{pmatrix}.
\]",
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [
            "Consider the matrix",
            "",
            r"\[",
            "A=",
            r"\begin{pmatrix}",
            r"\pi & 0\\",
            r"0 & \frac{1}{\pi}",
            r"\end{pmatrix}.",
            r"\]",
        ]
    );
}

#[test]
fn renders_lower_limits_beneath_display_operators() {
    let markdown = markdown(
        r"\[
\lim_{x\to 0}\frac{\frac{\sin x}{x}-1}{\frac{e^x-1}{x}-1}=0
\]",
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [
            r"\[",
            r"\lim_{x\to 0}\frac{\frac{\sin x}{x}-1}{\frac{e^x-1}{x}-1}=0",
            r"\]"
        ]
    );
}

#[test]
fn renders_math_inside_lists_and_tables() {
    let markdown = markdown(
        r"- Formula: $F_1 = u^2$

| Value |
| --- |
| $\mathbb{C}^3$ |",
    );
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    let output = plain_lines.join("\n");
    assert!(output.contains("- Formula: $F_1 = u^2$"));
    assert!(output.contains("$\\mathbb{C}^3$"));
}

#[test]
fn does_not_treat_currency_shell_variables_or_code_spans_as_math() {
    let source = "Costs $5 and $10 or $8k–$12k; use `$x$`, $HOME, and ${PATH}.";
    let currency_markdown = markdown(source);
    assert_eq!(
        plain_trimmed(&currency_markdown.render(80)),
        ["Costs $5 and $10 or $8k–$12k; use $x$, $HOME, and ${PATH}."]
    );

    let shell_variables = "Paths: $HOME/$USER and $XDG_CONFIG_HOME/$APP_CONFIG";
    let shell_lines = markdown(shell_variables).render(80);
    assert_eq!(plain_trimmed(&shell_lines), [shell_variables]);
}

#[test]
fn preserves_unsupported_and_incomplete_latex_exactly() {
    let cases = [
        r"Unknown $x + \unknown{y}$ after",
        r"Streaming $\mathbb{C}^3",
    ];
    for source in cases {
        let markdown = markdown(source);
        assert_eq!(plain_trimmed(&markdown.render(80)), [source]);
    }
}

#[test]
fn preserves_incomplete_backslash_delimiters_while_streaming() {
    let inline = markdown(r"Map \(\mathbb{C}^3");
    assert_eq!(plain_trimmed(&inline.render(80)), [r"Map \(\mathbb{C}^3"]);

    let display = markdown("\\[\nx^2");
    assert_eq!(plain_trimmed(&display.render(80)), ["\\[", "x^2"]);
}

#[test]
fn does_not_render_latex_inside_escaped_delimiters_or_code_fences() {
    let source = "Escaped \\$x-y\\$.\n\n```text\n$\\mathbb{C}^3$\n```";
    let markdown = markdown(source);
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        ["Escaped $x-y$.", "", "```text", r"  $\mathbb{C}^3$", "```"]
    );
}

#[test]
fn allows_latex_rendering_to_be_disabled() {
    let markdown = Markdown::with_options(
        r"Map $\mathbb{C}^3 \to \mathbb{C}^3$",
        0,
        0,
        default_markdown_theme(),
        None,
        MarkdownOptions {
            render_latex: false,
            ..MarkdownOptions::default()
        },
    );
    assert_eq!(
        plain_trimmed(&markdown.render(80)),
        [r"Map $\mathbb{C}^3 \to \mathbb{C}^3$"]
    );
}

#[test]
fn switches_from_raw_to_rendered_math_when_a_streamed_delimiter_closes() {
    let markdown = markdown(r"Map $\mathbb{C}^3");
    assert_eq!(plain_trimmed(&markdown.render(80)), [r"Map $\mathbb{C}^3"]);

    markdown.set_text(r"Map $\mathbb{C}^3$");
    // Restated (#50): with the renderer landing later, the closed delimiter
    // still renders raw here.
    assert_eq!(plain_trimmed(&markdown.render(80)), [r"Map $\mathbb{C}^3$"]);
}

// --- Backslash escapes ------------------------------------------------------

#[test]
fn normalizes_escaped_punctuation_by_default() {
    let markdown = markdown("\"\\\"");
    assert_eq!(plain_trimmed(&markdown.render(80)), ["\"\""]);
}

#[test]
fn preserves_source_backslash_escapes_when_configured() {
    let markdown = Markdown::with_options(
        "\"\\\"",
        0,
        0,
        default_markdown_theme(),
        None,
        MarkdownOptions {
            preserve_backslash_escapes: true,
            ..MarkdownOptions::default()
        },
    );
    assert_eq!(plain_trimmed(&markdown.render(80)), ["\"\\\""]);
}

// --- Pre-styled text (thinking traces) --------------------------------------

fn thinking_style(color: pi_tui::components::ColorFn) -> DefaultTextStyle {
    DefaultTextStyle {
        color: Some(color),
        bold: false,
        italic: true,
        ..DefaultTextStyle::default()
    }
}

#[test]
fn preserves_gray_italic_styling_after_inline_code() {
    // This replicates how thinking content is rendered in assistant-message.ts
    let markdown = Markdown::with_style(
        "This is thinking with `inline code` and more text after",
        1,
        0,
        default_markdown_theme(),
        thinking_style(tui_support::chalk_gray()),
    );
    let lines = markdown.render(80);
    let joined_output = joined_output(&lines);
    assert!(joined_output.contains("inline code"));
    assert!(
        joined_output.contains("\x1b[90m"),
        "should have gray color code"
    );
    assert!(joined_output.contains("\x1b[3m"), "should have italic code");
    assert!(
        joined_output.contains("\x1b[33m"),
        "should style inline code"
    );
}

#[test]
fn preserves_gray_italic_styling_after_bold_text() {
    let markdown = Markdown::with_style(
        "This is thinking with **bold text** and more after",
        1,
        0,
        default_markdown_theme(),
        thinking_style(tui_support::chalk_gray()),
    );
    let lines = markdown.render(80);
    let joined_output = joined_output(&lines);
    assert!(joined_output.contains("bold text"));
    assert!(
        joined_output.contains("\x1b[90m"),
        "should have gray color code"
    );
    assert!(joined_output.contains("\x1b[3m"), "should have italic code");
    assert!(joined_output.contains("\x1b[1m"), "should have bold code");
}

#[test]
fn does_not_leak_styles_into_following_lines_when_rendered_in_tui() {
    struct MarkdownWithInput {
        markdown: Markdown,
        markdown_line_count: RefCell<usize>,
    }

    impl Component for MarkdownWithInput {
        fn render(&self, width: usize) -> Vec<String> {
            let lines = self.markdown.render(width);
            *self.markdown_line_count.borrow_mut() = lines.len();
            let mut result = lines;
            result.push("INPUT".to_string());
            result
        }

        fn invalidate(&self) {
            self.markdown.invalidate();
        }
    }

    let markdown = Markdown::with_style(
        "This is thinking with `inline code`",
        1,
        0,
        default_markdown_theme(),
        thinking_style(tui_support::chalk_gray()),
    );
    let terminal = VirtualTerminal::new(80, 6);
    let tui = new_test_tui(terminal.clone());
    let component = Rc::new(MarkdownWithInput {
        markdown,
        markdown_line_count: RefCell::new(0),
    });
    tui.add_child(component.clone());
    tui.start();
    tui_support::wait_for_render(&tui);

    let line_count = *component.markdown_line_count.borrow();
    assert!(line_count > 0);
    assert!(
        !terminal.is_italic(line_count, 0),
        "the line after the markdown must not be italic"
    );
    tui_support::stop(&tui);
}

// --- Spacing after code blocks ---------------------------------------------

#[test]
fn has_only_one_blank_line_between_code_block_and_following_paragraph() {
    let markdown =
        markdown("hello world\n\n```js\nconst hello = \"world\";\n```\n\nagain, hello world");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    let closing_backticks_index = plain_lines.iter().position(|line| line == "```");
    assert!(
        closing_backticks_index.is_some(),
        "should have closing backticks"
    );
    let after_backticks = &plain_lines[closing_backticks_index.expect("index") + 1..];
    let empty_line_count = after_backticks.iter().position(|line| !line.is_empty());
    assert_eq!(
        empty_line_count,
        Some(1),
        "expected 1 empty line after code block, but found {empty_line_count:?}. Lines after backticks: {:?}",
        &after_backticks[..after_backticks.len().min(5)]
    );
}

#[test]
fn normalizes_paragraph_and_code_block_spacing_to_one_blank_line() {
    let cases = [
        "hello this is text\n```\ncode block\n```\nmore text",
        "hello this is text\n\n```\ncode block\n```\n\nmore text",
    ];
    let expected_lines = [
        "hello this is text",
        "",
        "```",
        "  code block",
        "```",
        "",
        "more text",
    ];
    for text in cases {
        let markdown = markdown(text);
        let plain_lines = plain_trimmed(&markdown.render(80));
        assert_eq!(
            plain_lines, expected_lines,
            "unexpected spacing for markdown: {text:?}"
        );
    }
}

#[test]
fn does_not_add_a_trailing_blank_line_when_code_block_is_last_rendered_block() {
    let cases = [
        "```js\nconst hello = 'world';\n```",
        "hello world\n\n```js\nconst hello = 'world';\n```",
    ];
    for text in cases {
        let markdown = markdown(text);
        let lines = markdown.render(80);
        let plain_lines = plain_trimmed(&lines);
        assert_ne!(
            plain_lines.last(),
            Some(&String::new()),
            "expected code block to end without a blank line: {plain_lines:?}"
        );
    }
}

// --- Spacing after dividers -------------------------------------------------

#[test]
fn has_only_one_blank_line_between_divider_and_following_paragraph() {
    let markdown = markdown("hello world\n\n---\n\nagain, hello world");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    let divider_index = plain_lines.iter().position(|line| line.contains('─'));
    assert!(divider_index.is_some(), "should have divider");
    let after_divider = &plain_lines[divider_index.expect("index") + 1..];
    let empty_line_count = after_divider.iter().position(|line| !line.is_empty());
    assert_eq!(
        empty_line_count,
        Some(1),
        "expected 1 empty line after divider, but found {empty_line_count:?}. Lines after divider: {:?}",
        &after_divider[..after_divider.len().min(5)]
    );
}

#[test]
fn does_not_add_a_trailing_blank_line_when_divider_is_last_rendered_block() {
    let markdown = markdown("---");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    assert_ne!(
        plain_lines.last(),
        Some(&String::new()),
        "expected divider to end without a blank line"
    );
}

// --- Spacing after headings -------------------------------------------------

#[test]
fn has_only_one_blank_line_between_heading_and_following_paragraph() {
    let markdown = markdown("# Hello\n\nThis is a paragraph");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    let heading_index = plain_lines.iter().position(|line| line.contains("Hello"));
    assert!(heading_index.is_some(), "should have heading");
    let after_heading = &plain_lines[heading_index.expect("index") + 1..];
    let empty_line_count = after_heading.iter().position(|line| !line.is_empty());
    assert_eq!(
        empty_line_count,
        Some(1),
        "expected 1 empty line after heading, but found {empty_line_count:?}. Lines after heading: {:?}",
        &after_heading[..after_heading.len().min(5)]
    );
}

#[test]
fn does_not_add_a_trailing_blank_line_when_heading_is_last_rendered_block() {
    let markdown = markdown("# Hello");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    assert_ne!(
        plain_lines.last(),
        Some(&String::new()),
        "expected heading to end without a blank line: {plain_lines:?}"
    );
}

// --- Spacing after blockquotes ----------------------------------------------

#[test]
fn has_only_one_blank_line_between_blockquote_and_following_paragraph() {
    let markdown = markdown("hello world\n\n> This is a quote\n\nagain, hello world");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    let quote_index = plain_lines
        .iter()
        .position(|line| line.contains("This is a quote"));
    assert!(quote_index.is_some(), "should have blockquote");
    let after_quote = &plain_lines[quote_index.expect("index") + 1..];
    let empty_line_count = after_quote.iter().position(|line| !line.is_empty());
    assert_eq!(
        empty_line_count,
        Some(1),
        "expected 1 empty line after blockquote, but found {empty_line_count:?}. Lines after quote: {:?}",
        &after_quote[..after_quote.len().min(5)]
    );
}

#[test]
fn does_not_add_a_trailing_blank_line_when_blockquote_is_last_rendered_block() {
    let markdown = markdown("> This is a quote");
    let lines = markdown.render(80);
    let plain_lines = plain_trimmed(&lines);
    assert_ne!(
        plain_lines.last(),
        Some(&String::new()),
        "expected blockquote to end without a blank line: {plain_lines:?}"
    );
}

// --- Blockquotes with multiline content --------------------------------------

#[test]
fn applies_consistent_styling_to_all_lines_in_lazy_continuation_blockquote() {
    // Markdown "lazy continuation" - second line without > is still part of
    // the quote.
    let markdown = Markdown::with_style(
        ">Foo\nbar",
        0,
        0,
        default_markdown_theme(),
        DefaultTextStyle {
            color: Some(tui_support::chalk_magenta()),
            ..DefaultTextStyle::default()
        },
    );
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    let quoted_line_count = plain_lines
        .iter()
        .filter(|line| line.starts_with("│ "))
        .count();
    assert_eq!(
        quoted_line_count, 2,
        "expected 2 quoted lines, got: {plain_lines:?}"
    );

    let foo_line = lines
        .iter()
        .find(|line| line.contains("Foo"))
        .expect("Foo line");
    let bar_line = lines
        .iter()
        .find(|line| line.contains("bar"))
        .expect("bar line");
    assert!(
        foo_line.contains("\x1b[3m"),
        "Foo line should have italic: {foo_line}"
    );
    assert!(
        bar_line.contains("\x1b[3m"),
        "bar line should have italic: {bar_line}"
    );
    assert!(
        !foo_line.contains("\x1b[35m"),
        "Foo line should NOT have magenta color: {foo_line}"
    );
    assert!(
        !bar_line.contains("\x1b[35m"),
        "bar line should NOT have magenta color: {bar_line}"
    );
}

#[test]
fn applies_consistent_styling_to_explicit_multiline_blockquote() {
    let markdown = Markdown::with_style(
        ">Foo\n>bar",
        0,
        0,
        default_markdown_theme(),
        DefaultTextStyle {
            color: Some(tui_support::chalk_cyan()),
            ..DefaultTextStyle::default()
        },
    );
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    let quoted_line_count = plain_lines
        .iter()
        .filter(|line| line.starts_with("│ "))
        .count();
    assert_eq!(
        quoted_line_count, 2,
        "expected 2 quoted lines, got: {plain_lines:?}"
    );

    let foo_line = lines
        .iter()
        .find(|line| line.contains("Foo"))
        .expect("Foo line");
    let bar_line = lines
        .iter()
        .find(|line| line.contains("bar"))
        .expect("bar line");
    assert!(
        foo_line.contains("\x1b[3m"),
        "Foo line should have italic: {foo_line}"
    );
    assert!(
        bar_line.contains("\x1b[3m"),
        "bar line should have italic: {bar_line}"
    );
    assert!(
        !foo_line.contains("\x1b[36m"),
        "Foo line should NOT have cyan color: {foo_line}"
    );
    assert!(
        !bar_line.contains("\x1b[36m"),
        "bar line should NOT have cyan color: {bar_line}"
    );
}

#[test]
fn renders_list_content_inside_blockquotes() {
    let markdown = markdown("> 1. bla bla\n> - nested bullet");
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    let quoted_lines: Vec<&String> = plain_lines
        .iter()
        .filter(|line| line.starts_with("│ "))
        .collect();
    assert!(
        quoted_lines.iter().any(|line| line.contains("1. bla bla")),
        "missing ordered list item: {quoted_lines:?}"
    );
    assert!(
        quoted_lines
            .iter()
            .any(|line| line.contains("- nested bullet")),
        "missing unordered list item: {quoted_lines:?}"
    );
}

#[test]
fn wraps_long_blockquote_lines_and_adds_border_to_each_wrapped_line() {
    let long_text =
        "This is a very long blockquote line that should wrap to multiple lines when rendered";
    let markdown = markdown(&format!("> {long_text}"));
    let lines = markdown.render(30);
    let plain_lines = plain_trimmed(&lines);
    let content_lines: Vec<&String> = plain_lines.iter().filter(|line| !line.is_empty()).collect();
    assert!(
        content_lines.len() > 1,
        "expected multiple wrapped lines, got: {plain_lines:?}"
    );
    for line in &content_lines {
        assert!(
            line.starts_with("│ "),
            "wrapped line should have quote border: \"{line}\""
        );
    }
    let all_text = content_lines
        .iter()
        .map(|line| line.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_text.contains("very long"),
        "should preserve 'very long'"
    );
    assert!(
        all_text.contains("blockquote"),
        "should preserve 'blockquote'"
    );
    assert!(all_text.contains("multiple"), "should preserve 'multiple'");
}

#[test]
fn properly_indents_wrapped_blockquote_lines_with_styling() {
    let markdown = Markdown::with_style(
        "> This is styled text that is long enough to wrap",
        0,
        0,
        default_markdown_theme(),
        DefaultTextStyle {
            color: Some(tui_support::chalk_yellow()),
            ..DefaultTextStyle::default()
        },
    );
    let lines = markdown.render(25);
    let plain_lines = plain_trimmed(&lines);
    let content_lines: Vec<&String> = plain_lines.iter().filter(|line| !line.is_empty()).collect();
    for line in &content_lines {
        assert!(
            line.starts_with("│ "),
            "line should have quote border: \"{line}\""
        );
    }
    let all_output = joined_output(&lines);
    assert!(all_output.contains("\x1b[3m"), "should have italic");
    assert!(
        !all_output.contains("\x1b[33m"),
        "should NOT have yellow color from default style"
    );
}

#[test]
fn renders_inline_formatting_inside_blockquotes_and_reapplies_quote_styling_after() {
    let markdown = markdown("> Quote with **bold** and `code`");
    let lines = markdown.render(80);
    let plain_lines = stripped(&lines);
    assert!(
        plain_lines.iter().any(|line| line.starts_with("│ ")),
        "should have quote border"
    );
    let all_plain = plain_lines.join(" ");
    assert!(
        all_plain.contains("Quote with"),
        "should preserve 'Quote with'"
    );
    assert!(all_plain.contains("bold"), "should preserve 'bold'");
    assert!(all_plain.contains("code"), "should preserve 'code'");
    let all_output = joined_output(&lines);
    assert!(all_output.contains("\x1b[1m"), "should have bold styling");
    assert!(
        all_output.contains("\x1b[33m"),
        "should have code styling (yellow)"
    );
    assert!(
        all_output.contains("\x1b[3m"),
        "should have italic from quote styling"
    );
}

// --- Heading with inline code -------------------------------------------------

#[test]
fn preserves_heading_styling_after_inline_code() {
    let markdown = markdown("### Why `sourceInfo` should not be optional");
    let lines = markdown.render(80);
    let joined_output = joined_output(&lines);
    assert!(
        joined_output.contains("\x1b[33m"),
        "should have yellow for inline code"
    );

    let after_code_index = joined_output.find("should not be optional");
    assert!(
        after_code_index.is_some(),
        "should contain text after inline code"
    );
    let after_code_index = after_code_index.expect("index");
    assert!(after_code_index > 0);
    let preceding_chunk = &joined_output[after_code_index.saturating_sub(40)..after_code_index];
    assert!(
        preceding_chunk.contains("\x1b[1m"),
        "should re-apply bold before text after code: {preceding_chunk}"
    );
    assert!(
        preceding_chunk.contains("\x1b[36m"),
        "should re-apply cyan before text after code: {preceding_chunk}"
    );
}

#[test]
fn preserves_heading_styling_after_inline_code_for_h1() {
    let markdown = markdown("# Title with `code` inside");
    let lines = markdown.render(80);
    let joined_output = joined_output(&lines);
    let after_code_index = joined_output.find("inside");
    assert!(
        after_code_index.is_some(),
        "should contain text after inline code"
    );
    let after_code_index = after_code_index.expect("index");
    assert!(after_code_index > 0);
    let preceding_chunk = &joined_output[after_code_index.saturating_sub(40)..after_code_index];
    assert!(
        preceding_chunk.contains("\x1b[1m"),
        "should re-apply bold for h1: {preceding_chunk}"
    );
    assert!(
        preceding_chunk.contains("\x1b[36m"),
        "should re-apply cyan for h1: {preceding_chunk}"
    );
    assert!(
        preceding_chunk.contains("\x1b[4m"),
        "should re-apply underline for h1: {preceding_chunk}"
    );
}

#[test]
fn does_not_leak_h1_underline_into_padding_when_inline_code_is_last_token() {
    let markdown = Rc::new(markdown("# Important distinction from `open()`"));
    let terminal = VirtualTerminal::new(80, 4);
    let tui = new_test_tui(terminal.clone());
    let component: Rc<dyn Component> = markdown.clone();
    tui.add_child(component);
    tui.start();
    tui_support::wait_for_render(&tui);

    let rendered_line = markdown.render(80)[0].clone();
    assert!(
        rendered_line.contains("distinction"),
        "should render heading line"
    );
    let content_width = strip_ansi(&rendered_line).trim_end().chars().count();
    assert!(content_width > 0, "should have visible heading content");
    for col in content_width..80 {
        assert!(
            !terminal.is_underline(0, col),
            "expected no underline in padding at col {col}"
        );
    }
    tui_support::stop(&tui);
}

#[test]
fn preserves_heading_styling_after_bold_text() {
    let markdown = markdown("## Heading with **bold** and more");
    let lines = markdown.render(80);
    let joined_output = joined_output(&lines);
    let after_bold_index = joined_output.find("and more");
    assert!(after_bold_index.is_some(), "should contain text after bold");
    let after_bold_index = after_bold_index.expect("index");
    assert!(after_bold_index > 0);
    let preceding_chunk = &joined_output[after_bold_index.saturating_sub(40)..after_bold_index];
    assert!(
        preceding_chunk.contains("\x1b[1m"),
        "should re-apply bold for h2: {preceding_chunk}"
    );
    assert!(
        preceding_chunk.contains("\x1b[36m"),
        "should re-apply cyan for h2: {preceding_chunk}"
    );
}

// --- Strikethrough syntax ------------------------------------------------------

#[test]
fn renders_strikethrough_syntax_as_strikethrough() {
    let markdown = markdown("Use ~~strikethrough~~ here");
    let lines = markdown.render(80);
    let joined_output = joined_output(&lines);
    let joined_plain = stripped(&lines).join(" ");
    assert!(
        joined_output.contains("\x1b[9m"),
        "should apply strikethrough styling"
    );
    assert!(
        joined_plain.contains("strikethrough"),
        "should include struck text content"
    );
    assert!(
        !joined_plain.contains("~~strikethrough~~"),
        "should not render delimiters as text"
    );
}

#[test]
fn keeps_single_tilde_text_as_plain_text() {
    let markdown = markdown("Use ~strikethrough~ literally");
    let lines = markdown.render(80);
    let joined_output = joined_output(&lines);
    let joined_plain = stripped(&lines).join(" ");
    assert!(
        joined_plain.contains("~strikethrough~"),
        "single-tilde delimiters should remain visible"
    );
    assert!(
        !joined_output.contains("\x1b[9m"),
        "single-tilde text should not use strikethrough styling"
    );
}

// --- Links -------------------------------------------------------------------

#[test]
fn does_not_duplicate_url_for_autolinked_emails() {
    let _capability_lock = tui_support::capabilities_lock();
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let markdown = markdown("Contact user@example.com for help");
    let lines = markdown.render(80);
    let joined_plain = stripped(&lines).join(" ");
    assert!(
        joined_plain.contains("user@example.com"),
        "should contain email"
    );
    assert!(
        !joined_plain.contains("mailto:"),
        "should not show mailto: prefix for autolinked emails"
    );
    reset_capabilities_cache();
}

#[test]
fn does_not_duplicate_url_for_bare_urls() {
    let _capability_lock = tui_support::capabilities_lock();
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let markdown = markdown("Visit https://example.com for more");
    let lines = markdown.render(80);
    let joined_plain = stripped(&lines).join(" ");
    let url_count = joined_plain.matches("https://example.com").count();
    assert_eq!(url_count, 1, "URL should appear exactly once");
    reset_capabilities_cache();
}

#[test]
fn shows_url_in_parentheses_when_hyperlinks_are_not_supported() {
    let _capability_lock = tui_support::capabilities_lock();
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let markdown = markdown("[click here](https://example.com)");
    let lines = markdown.render(80);
    let joined_plain = stripped(&lines).join(" ");
    assert!(
        joined_plain.contains("click here"),
        "should contain link text"
    );
    assert!(
        joined_plain.contains("(https://example.com)"),
        "should show URL in parentheses"
    );
    reset_capabilities_cache();
}

#[test]
fn shows_mailto_url_in_parentheses_when_hyperlinks_are_not_supported() {
    let _capability_lock = tui_support::capabilities_lock();
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let markdown = markdown("[Email me](mailto:test@example.com)");
    let lines = markdown.render(80);
    let joined_plain = stripped(&lines).join(" ");
    assert!(
        joined_plain.contains("Email me"),
        "should contain link text"
    );
    assert!(
        joined_plain.contains("(mailto:test@example.com)"),
        "should show mailto URL in parentheses"
    );
    reset_capabilities_cache();
}

#[test]
fn emits_osc_8_hyperlink_sequence_when_terminal_supports_hyperlinks() {
    let _capability_lock = tui_support::capabilities_lock();
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    });
    let markdown = markdown("[click here](https://example.com)");
    let lines = markdown.render(80);
    let joined = joined_output(&lines);
    assert!(
        joined.contains("\x1b]8;;https://example.com\x1b\\"),
        "should contain OSC 8 open sequence"
    );
    assert!(
        joined.contains("\x1b]8;;\x1b\\"),
        "should contain OSC 8 close sequence"
    );
    let plain_lines = stripped(&lines);
    assert!(
        plain_lines.join("").contains("click here"),
        "should contain link text"
    );
    let raw_plain = lines
        .iter()
        .map(|line| strip_ansi(line))
        .collect::<String>();
    assert!(
        !raw_plain.contains("(https://example.com)"),
        "URL should not appear inline in parentheses"
    );
    reset_capabilities_cache();
}

#[test]
fn uses_osc_8_for_mailto_links_when_terminal_supports_hyperlinks() {
    let _capability_lock = tui_support::capabilities_lock();
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    });
    let markdown = markdown("[Email me](mailto:test@example.com)");
    let lines = markdown.render(80);
    let joined = joined_output(&lines);
    assert!(
        joined.contains("\x1b]8;;mailto:test@example.com\x1b\\"),
        "should contain OSC 8 open with mailto URL"
    );
    assert!(
        joined.contains("\x1b]8;;\x1b\\"),
        "should contain OSC 8 close sequence"
    );
    reset_capabilities_cache();
}

#[test]
fn uses_osc_8_for_bare_urls_when_terminal_supports_hyperlinks() {
    let _capability_lock = tui_support::capabilities_lock();
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    });
    let markdown = markdown("Visit https://example.com for more");
    let lines = markdown.render(80);
    let joined = joined_output(&lines);
    assert!(
        joined.contains("\x1b]8;;https://example.com\x1b\\"),
        "should contain OSC 8 hyperlink"
    );
    let raw_plain = lines
        .iter()
        .map(|line| strip_ansi(line))
        .collect::<String>();
    assert!(
        !raw_plain.contains("(https://example.com)"),
        "URL should not appear twice"
    );
    reset_capabilities_cache();
}

// --- HTML-like tags in text ---------------------------------------------------

#[test]
fn renders_content_with_html_like_tags_as_text() {
    let markdown =
        markdown("This is text with <thinking>hidden content</thinking> that should be visible");
    let lines = markdown.render(80);
    let joined_plain = stripped(&lines).join(" ");
    assert!(
        joined_plain.contains("hidden content") || joined_plain.contains("<thinking>"),
        "should render HTML-like tags or their content as text, not hide them"
    );
}

#[test]
fn renders_html_tags_in_code_blocks_correctly() {
    let markdown = markdown("```html\n<div>Some HTML</div>\n```");
    let lines = markdown.render(80);
    let joined_plain = stripped(&lines).join("\n");
    assert!(
        joined_plain.contains("<div>") && joined_plain.contains("</div>"),
        "should render HTML in code blocks"
    );
}

// --- Streaming code fences -----------------------------------------------------

#[test]
fn stabilizes_partial_closing_fence_rendering() {
    let cases: [(&str, Vec<&str>); 6] = [
        (
            "```ts\nconst x = 1;\n```",
            vec!["```ts", "  const x = 1;", "```"],
        ),
        (
            "```md\nnot a closing fence:\n``\n```",
            vec!["```md", "  not a closing fence:", "  ``", "```"],
        ),
        ("```ts\n```", vec!["```ts", "", "```"]),
        ("````\n```", vec!["```", "", "```"]),
        ("~~~~~\n~~~~", vec!["```", "", "```"]),
        (
            "```md\nnot a closing fence:\n``\n```\n\nafter",
            vec![
                "```md",
                "  not a closing fence:",
                "  ``",
                "```",
                "",
                "after",
            ],
        ),
    ];
    for (input, expected) in cases {
        let markdown = markdown(input);
        let lines = plain_trimmed(&markdown.render(80));
        let expected: Vec<String> = expected.into_iter().map(String::from).collect();
        assert_eq!(
            lines, expected,
            "unexpected spacing for markdown: {input:?}"
        );
    }

    let partial = markdown("```ts\nconst x = 1;\n```");
    let complete = markdown("```ts\nconst x = 1;\n```");
    assert_eq!(partial.render(80).len(), complete.render(80).len());
}

// --- helpers -------------------------------------------------------------------

fn markdown_with_url(url: &str) -> Markdown {
    Markdown::new(
        format!("| Value |\n| --- |\n| prefix {url} |"),
        0,
        0,
        default_markdown_theme(),
    )
}
