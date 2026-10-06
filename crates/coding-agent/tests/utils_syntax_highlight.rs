//! The syntax-highlight suite, upstream's `test/syntax-highlight.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The renderer block ports 1:1 over [`render_highlighted_html`]. The
//! "theme syntax highlighting" block rides the interactive theme module
//! (#132) and stays with its consumer. The load test restates: highlight.js
//! lazy-loads its language index, the syntect engine ships its whole set at
//! init, so the eager twenty are available from the start along with the
//! deferred remainder — `ada` never has a deferred phase to await
//! (recorded with the ticket).

use std::collections::HashMap;
use std::sync::Arc;

use pi_coding_agent::utils::syntax_highlight::{
    HighlightFormatter, HighlightOptions, HighlightTheme, highlight, load_all_highlight_languages,
    render_highlighted_html, supports_language,
};

fn formatter(tag: &str) -> HighlightFormatter {
    let tag = tag.to_string();
    Arc::new(move |text: &str| format!("[{tag}:{text}]"))
}

#[tokio::test]
async fn loads_the_twenty_most_common_languages_at_startup() {
    for name in [
        "python",
        "java",
        "go",
        "javascript",
        "cpp",
        "typescript",
        "php",
        "ruby",
        "c",
        "csharp",
        "nix",
        "bash",
        "rust",
        "scala",
        "kotlin",
        "swift",
        "dart",
        "groovy",
        "perl",
        "lua",
    ] {
        assert!(supports_language(name), "{name}");
    }
    // The restated engine has no deferred index: the uncommon languages are
    // available from the start, upstream's post-`loadAll` state.
    assert!(supports_language("ada"));
    load_all_highlight_languages().await;
    assert!(supports_language("ada"));
    assert!(!supports_language("no-such-language"));
}

#[test]
fn renders_highlighted_spans_with_the_provided_theme() {
    let mut theme = HashMap::new();
    theme.insert("keyword", formatter("keyword"));
    let rendered = render_highlighted_html(
        "<span class=\"hljs-keyword\">const</span> value",
        &HighlightTheme::new(theme),
    );
    assert_eq!(rendered, "[keyword:const] value");
}

#[test]
fn decodes_html_entities_emitted_by_the_highlighter() {
    let rendered = render_highlighted_html(
        "&lt;tag attr=&quot;value&quot;&gt;&amp;#x41;&#65;&lt;/tag&gt;",
        &HighlightTheme::new(HashMap::new()),
    );
    assert_eq!(rendered, "<tag attr=\"value\">&#x41;A</tag>");
}

#[test]
fn inherits_parent_formatting_for_unmapped_nested_scopes() {
    let interpolation = "$".to_string() + "{x}";
    let mut theme = HashMap::new();
    theme.insert("string", formatter("string"));
    let rendered = render_highlighted_html(
        &format!(
            "<span class=\"hljs-string\">a<span class=\"hljs-subst\">{interpolation}</span>b</span>"
        ),
        &HighlightTheme::new(theme),
    );
    assert_eq!(
        rendered,
        format!("[string:a][string:{interpolation}][string:b]")
    );
}

#[test]
fn keeps_parent_formatting_across_unscoped_nested_spans() {
    let mut theme = HashMap::new();
    theme.insert("string", formatter("string"));
    let rendered = render_highlighted_html(
        "<span class=\"hljs-string\">a<span class=\"language-xml\">b</span>c</span>",
        &HighlightTheme::new(theme),
    );
    assert_eq!(rendered, "[string:a][string:b][string:c]");
}

#[test]
fn highlights_code_through_the_engine() {
    assert!(supports_language("typescript"));
    let mut theme = HashMap::new();
    theme.insert("keyword", formatter("keyword"));
    theme.insert("number", formatter("number"));
    let rendered = highlight(
        "const value = 1",
        HighlightOptions {
            language: Some("typescript"),
            ignore_illegals: true,
            language_subset: Vec::new(),
            theme: Some(HighlightTheme::new(theme)),
        },
    );
    assert!(rendered.contains("[keyword:const]"), "{rendered}");
    assert!(rendered.contains("[number:1]"), "{rendered}");
}

#[test]
fn auto_detection_formats_through_the_theme_default() {
    // The engine's first-line detection restates `highlightAuto`: a shebang
    // picks the shell syntax, and every token the theme does not map rides
    // the default formatter.
    let mut theme = HashMap::new();
    theme.insert("default", formatter("plain"));
    let rendered = highlight(
        "#!/bin/bash\necho hello",
        HighlightOptions {
            language: None,
            ignore_illegals: false,
            language_subset: Vec::new(),
            theme: Some(HighlightTheme::new(theme)),
        },
    );
    assert!(rendered.contains("[plain:"), "{rendered}");
}

#[test]
fn undetectable_code_passes_through() {
    let rendered = highlight(
        "shebang-less plain text",
        HighlightOptions {
            language: None,
            ignore_illegals: false,
            language_subset: Vec::new(),
            theme: None,
        },
    );
    assert_eq!(rendered, "shebang-less plain text");
}

#[test]
fn unknown_languages_pass_through() {
    let rendered = highlight(
        "text",
        HighlightOptions {
            language: Some("no-such-language"),
            ignore_illegals: false,
            language_subset: Vec::new(),
            theme: None,
        },
    );
    assert_eq!(rendered, "text");
}

// === boundary vectors over the restated renderer ============================

#[test]
fn the_prefix_rules_map_dotted_and_dashed_scopes() {
    let mut theme = HashMap::new();
    theme.insert("string", formatter("string"));
    theme.insert("title", formatter("title"));
    // A dotted scope inherits its prefix; a dashed one does too.
    let rendered = render_highlighted_html(
        "<span class=\"hljs-string quoted\">x</span><span class=\"hljs-title function\">f</span>",
        &HighlightTheme::new(theme),
    );
    assert_eq!(rendered, "[string:x][title:f]");
}

#[test]
fn the_innermost_mapped_scope_wins() {
    let mut theme = HashMap::new();
    theme.insert("string", formatter("string"));
    theme.insert("keyword", formatter("keyword"));
    let rendered = render_highlighted_html(
        "<span class=\"hljs-string\">a<span class=\"hljs-keyword\">kw</span>b</span>",
        &HighlightTheme::new(theme),
    );
    assert_eq!(rendered, "[string:a][keyword:kw][string:b]");
}

#[test]
fn the_extension_probe_answers_file_suffixes() {
    // The restated lookup resolves the file extensions syntect indexes
    // after the language names, the way the alias table carries csharp.
    assert!(supports_language("rs"));
}

#[test]
fn malformed_spans_flow_through_as_text() {
    let rendered = render_highlighted_html(
        "<span class=\"hljs-keyword\"",
        &HighlightTheme::new(HashMap::new()),
    );
    assert_eq!(rendered, "<span class=\"hljs-keyword\"");
}
