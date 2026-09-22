//! Markdown component, ported from
//! `packages/tui/src/components/markdown.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#48).
//!
//! A block-and-inline renderer that turns Markdown into styled terminal
//! lines: headings, paragraphs, fenced code, lists (nested, ordered, task),
//! tables with width-aware cell wrapping, blockquotes, horizontal rules,
//! HTML passthrough, and inline styles over a [`MarkdownTheme`].
//!
//! Restatements against upstream, which lexes with `marked` 18:
//!
//! - The parser is [`pulldown_cmark`] 0.13 and its offset event stream
//!   replaces `marked`'s token tree (survey flag 9), with the upstream
//!   rendered-output goldens as the oracle. `marked`'s `space` tokens have
//!   no event counterpart, so the blank-line spacing between blocks
//!   re-derives from the source lines between block ranges; a list's
//!   `loose` flag re-derives from item content wrapped in paragraph events;
//!   and the source-marker option re-reads each item's source range.
//! - `marked` has no pulldown extension API, so the LaTeX tokenizer runs as
//!   a pre-pass over the source for blocks and as a scanner over text
//!   events inline, with upstream's delimiter, rejection, and pending rules
//!   byte-for-byte. Rendering goes through [`crate::latex::render_latex`],
//!   and an expression the renderer declines degrades to its raw source.
//! - `marked`'s gfm autolinks (bare URLs and emails) are not pulldown
//!   behavior; the port carries the same url/email scanners over text
//!   spans, angle autolinks being core CommonMark that pulldown emits
//!   natively.
//! - Upstream's strict-strikethrough tokenizer rejects single-tilde
//!   delimiters; pulldown enables them, so single-tilde strikethrough
//!   spans render as their literal source text.
//! - Backslash escapes decode inline in pulldown (the backslash sits in the
//!   gap between text events), so
//!   [`MarkdownOptions::preserve_backslash_escapes`] reconstitutes the raw
//!   form from those gaps and the default keeps the decoded text, both
//!   matching upstream's escape tokens.
//! - Block-level LaTeX inside blockquotes and list items restates to the
//!   inline scanner's rules: single-line `$$…$$` spans render as inline
//!   LaTeX, multi-line spans stay raw source, where upstream would lex them
//!   as display blocks.
//! - Angle email autolinks keep pulldown's bare `dest_url`, so their href
//!   gains the `mailto:` prefix upstream's tokenizer adds.
//! - The width-less `getLongestWordWidth` overload is dead upstream (every
//!   caller passes the 30-column unbroken-word cap) and drops.

use std::cell::Cell;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::LazyLock;

use std::ops::Range;

use pulldown_cmark::CodeBlockKind;
use pulldown_cmark::Event;
use pulldown_cmark::HeadingLevel;
use pulldown_cmark::Options;
use pulldown_cmark::Parser;
use pulldown_cmark::Tag;
use pulldown_cmark::TagEnd;
use regex::Regex;

use crate::components::ColorFn;
use crate::latex::render_latex;
use crate::terminal_image::get_capabilities;
use crate::terminal_image::hyperlink;
use crate::terminal_image::is_image_line;
use crate::tui::Component as _;
use crate::utils::apply_background_to_line;
use crate::utils::static_regex;
use crate::utils::visible_width;
use crate::utils::wrap_text_with_ansi;

/// Default text styling for markdown content, upstream `DefaultTextStyle`.
/// Applied to all text unless overridden by markdown formatting.
#[expect(
    clippy::struct_excessive_bools,
    reason = "upstream's `DefaultTextStyle` is one optional function plus one flag per text decoration; the shape is the interface"
)]
#[derive(Clone, Default)]
pub struct DefaultTextStyle {
    /// Foreground color function.
    pub color: Option<ColorFn>,
    /// Background color function. Applied at the padding stage so it
    /// extends across the full line width, upstream's note.
    pub bg_color: Option<ColorFn>,
    /// Bold text.
    pub bold: bool,
    /// Italic text.
    pub italic: bool,
    /// Strikethrough text.
    pub strikethrough: bool,
    /// Underline text.
    pub underline: bool,
}

/// Theme functions for markdown elements, upstream `MarkdownTheme`. Each
/// function takes text and returns styled text with ANSI codes.
#[derive(Clone)]
pub struct MarkdownTheme {
    /// Upstream `heading`.
    pub heading: ColorFn,
    /// Upstream `link`.
    pub link: ColorFn,
    /// Upstream `linkUrl`.
    pub link_url: ColorFn,
    /// Upstream `code`.
    pub code: ColorFn,
    /// Upstream `codeBlock`.
    pub code_block: ColorFn,
    /// Upstream `codeBlockBorder`.
    pub code_block_border: ColorFn,
    /// Upstream `quote`.
    pub quote: ColorFn,
    /// Upstream `quoteBorder`.
    pub quote_border: ColorFn,
    /// Upstream `hr`.
    pub hr: ColorFn,
    /// Upstream `listBullet`.
    pub list_bullet: ColorFn,
    /// Upstream `bold`.
    pub bold: ColorFn,
    /// Upstream `italic`.
    pub italic: ColorFn,
    /// Upstream `strikethrough`.
    pub strikethrough: ColorFn,
    /// Upstream `underline`.
    pub underline: ColorFn,
    /// Optional syntax highlighter, upstream `highlightCode`.
    pub highlight_code: Option<HighlightFn>,
    /// Prefix applied to each rendered code block line, upstream
    /// `codeBlockIndent`. Defaults to `"  "`.
    pub code_block_indent: Option<String>,
}

/// The optional syntax highlighter, upstream
/// `highlightCode?: (code, lang?) => string[]`.
pub type HighlightFn = Arc<dyn Fn(&str, Option<&str>) -> Vec<String> + Send + Sync>;

/// Options for the markdown renderer, upstream `MarkdownOptions`.
#[derive(Clone)]
pub struct MarkdownOptions {
    /// Preserve source list markers instead of normalizing them, upstream
    /// `preserveOrderedListMarkers`.
    pub preserve_ordered_list_markers: bool,
    /// Preserve source backslash escapes instead of normalizing escaped
    /// punctuation, upstream `preserveBackslashEscapes`.
    pub preserve_backslash_escapes: bool,
    /// Transform source Markdown before parsing, with the exact width
    /// available for content, upstream `transform`.
    pub transform: Option<TransformFn>,
    /// Render supported LaTeX math expressions as Unicode text, upstream
    /// `renderLatex`. Defaults to `true`.
    pub render_latex: bool,
}

/// The pre-parse transform, upstream `(markdown, availableWidth) => string`.
pub type TransformFn = Arc<dyn Fn(&str, usize) -> String + Send + Sync>;

/// A manual impl because the derive would default `render_latex` to `false`,
/// diverging from upstream's documented default of `true`.
impl Default for MarkdownOptions {
    fn default() -> Self {
        Self {
            preserve_ordered_list_markers: false,
            preserve_backslash_escapes: false,
            transform: None,
            render_latex: true,
        }
    }
}

impl std::fmt::Debug for DefaultTextStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultTextStyle").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for MarkdownTheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarkdownTheme").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for MarkdownOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarkdownOptions")
            .field(
                "preserve_ordered_list_markers",
                &self.preserve_ordered_list_markers,
            )
            .field(
                "preserve_backslash_escapes",
                &self.preserve_backslash_escapes,
            )
            .field("transform", &self.transform.is_some())
            .field("render_latex", &self.render_latex)
            .finish()
    }
}

/// One inline event with its absolute source range.
type InlineEvent = (Event<'static>, Range<usize>);
/// One block's inline event run, the port's stand-in for marked's inline
/// token list.
type InlineEvents = Vec<InlineEvent>;

/// One parsed token with its absolute source range, the port's stand-in for
/// marked's token tree. Sub-tokens nest for blockquotes and list items.
enum Token {
    /// Upstream `paragraph`; `events` is the paragraph's inline event run.
    Paragraph {
        range: Range<usize>,
        events: InlineEvents,
    },
    /// Upstream `heading`, `depth` = 1..=6.
    Heading {
        depth: usize,
        range: Range<usize>,
        events: InlineEvents,
    },
    /// Upstream `code`: `text` in marked's token form (the trailing newline
    /// of a closed fence stripped), `range` the block's source span.
    CodeBlock {
        lang: String,
        text: String,
        range: Range<usize>,
    },
    /// Upstream `hr`.
    Rule { range: Range<usize> },
    /// Upstream `html` block, the raw source passthrough.
    Html { raw: String, range: Range<usize> },
    /// Upstream's `latexBlock` extension token.
    LatexBlock {
        text: String,
        pending: bool,
        raw: String,
        range: Range<usize>,
    },
    /// Upstream `list`: `start` is the first item number of ordered lists,
    /// `loose` re-derived from paragraph-wrapped item content.
    List {
        ordered: bool,
        start: Option<u64>,
        loose: bool,
        items: Vec<ListItem>,
        range: Range<usize>,
    },
    /// Upstream `table`; `head` and each row are per-cell inline event runs.
    Table {
        range: Range<usize>,
        head: Vec<InlineEvents>,
        rows: Vec<Vec<InlineEvents>>,
    },
    /// Upstream `blockquote`, children grouped recursively.
    BlockQuote {
        range: Range<usize>,
        blocks: Vec<Self>,
    },
}

struct ListItem {
    /// The item's source span, starting at its marker, upstream `item.raw`.
    range: Range<usize>,
    task: Option<bool>,
    blocks: Vec<Token>,
}

impl Token {
    const fn range(&self) -> &Range<usize> {
        match self {
            Self::Paragraph { range, .. }
            | Self::Heading { range, .. }
            | Self::CodeBlock { range, .. }
            | Self::Html { range, .. }
            | Self::LatexBlock { range, .. }
            | Self::List { range, .. }
            | Self::Table { range, .. }
            | Self::BlockQuote { range, .. }
            | Self::Rule { range } => range,
        }
    }

    const fn kind(&self) -> BlockKind {
        match self {
            Self::Paragraph { .. } => BlockKind::Paragraph,
            Self::Heading { .. } => BlockKind::Heading,
            Self::CodeBlock { .. } => BlockKind::Code,
            Self::Rule { .. } => BlockKind::Hr,
            Self::Html { .. } => BlockKind::Html,
            Self::LatexBlock { .. } => BlockKind::LatexBlock,
            Self::List { .. } => BlockKind::List,
            Self::Table { .. } => BlockKind::Table,
            Self::BlockQuote { .. } => BlockKind::BlockQuote,
        }
    }
}

/// What follows a rendered block in the source, upstream's `nextTokenType`
/// plus the `space` token marked emitted between blocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NextBlock {
    /// Nothing follows.
    None,
    /// A blank source line separates this block from the next, upstream's
    /// `space` token between them.
    Space,
    /// The next block kind, upstream's `nextTokenType`.
    Kind(BlockKind),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlockKind {
    Paragraph,
    Heading,
    Code,
    LatexBlock,
    List,
    Table,
    BlockQuote,
    Hr,
    Html,
}

/// A LaTeX token the inline/block scanners produce, upstream `LatexToken`.
struct LatexToken {
    raw: String,
    text: String,
    pending: bool,
}

/// A claimed inline LaTeX span over the context source, discovered by the
/// scanner before the event walk and rendered once when reached.
struct LatexClaim {
    start: usize,
    end: usize,
    token: LatexToken,
    rendered: Cell<bool>,
}

/// The markdown component, upstream `Markdown implements Component`.
pub struct Markdown {
    text: RefCell<String>,
    padding_x: usize,
    padding_y: usize,
    default_text_style: Option<DefaultTextStyle>,
    theme: MarkdownTheme,
    options: MarkdownOptions,
    default_style_prefix: RefCell<Option<String>>,
    cached_text: RefCell<Option<String>>,
    cached_width: RefCell<Option<usize>>,
    cached_lines: RefCell<Option<Vec<String>>>,
}

impl std::fmt::Debug for Markdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Markdown").finish_non_exhaustive()
    }
}

/// The inline style context, upstream `InlineStyleContext`: how plain text
/// inside a block styles itself, and which ANSI codes re-open that style
/// after a nested element's reset.
struct InlineStyleContext<'a> {
    apply_text: &'a dyn Fn(&str) -> String,
    style_prefix: &'a str,
}

impl Markdown {
    /// Upstream `new Markdown(text, paddingX, paddingY, theme)`.
    #[must_use]
    pub fn new(
        text: impl Into<String>,
        padding_x: usize,
        padding_y: usize,
        theme: MarkdownTheme,
    ) -> Self {
        Self::with_options(
            text,
            padding_x,
            padding_y,
            theme,
            None,
            MarkdownOptions::default(),
        )
    }

    /// Upstream's constructor with a default text style.
    #[must_use]
    pub fn with_style(
        text: impl Into<String>,
        padding_x: usize,
        padding_y: usize,
        theme: MarkdownTheme,
        default_text_style: DefaultTextStyle,
    ) -> Self {
        Self::with_options(
            text,
            padding_x,
            padding_y,
            theme,
            Some(default_text_style),
            MarkdownOptions::default(),
        )
    }

    /// Upstream `new Markdown(text, paddingX, paddingY, theme, defaultTextStyle?, options?)`.
    #[must_use]
    pub fn with_options(
        text: impl Into<String>,
        padding_x: usize,
        padding_y: usize,
        theme: MarkdownTheme,
        default_text_style: Option<DefaultTextStyle>,
        options: MarkdownOptions,
    ) -> Self {
        Self {
            text: RefCell::new(text.into()),
            padding_x,
            padding_y,
            default_text_style,
            theme,
            options,
            default_style_prefix: RefCell::new(None),
            cached_text: RefCell::new(None),
            cached_width: RefCell::new(None),
            cached_lines: RefCell::new(None),
        }
    }

    /// Upstream `setText`.
    pub fn set_text(&self, text: impl Into<String>) {
        *self.text.borrow_mut() = text.into();
        self.invalidate();
    }

    /// Apply the default text style to a string, upstream
    /// `applyDefaultStyle`: foreground color first, then the theme's text
    /// decorations. Background color is applied at the padding stage so it
    /// extends to the full line width.
    fn apply_default_style(&self, text: &str) -> String {
        let Some(style) = self.default_text_style.as_ref() else {
            return text.to_string();
        };
        let mut styled = text.to_string();
        if let Some(color) = style.color.as_ref() {
            styled = color(&styled);
        }
        if style.bold {
            styled = (self.theme.bold)(&styled);
        }
        if style.italic {
            styled = (self.theme.italic)(&styled);
        }
        if style.strikethrough {
            styled = (self.theme.strikethrough)(&styled);
        }
        if style.underline {
            styled = (self.theme.underline)(&styled);
        }
        styled
    }

    /// The ANSI codes a style function emits before its text, upstream
    /// `getStylePrefix`: the sentinel run's open codes.
    fn style_prefix_of(style_fn: &dyn Fn(&str) -> String) -> String {
        const SENTINEL: char = '\u{0}';
        let styled = style_fn(&SENTINEL.to_string());
        styled
            .find(SENTINEL)
            .map_or_else(String::new, |index| styled[..index].to_string())
    }

    /// The cached prefix of the default text style, upstream
    /// `getDefaultStylePrefix`.
    fn default_style_prefix(&self) -> String {
        if self.default_text_style.is_none() {
            return String::new();
        }
        if let Some(prefix) = self.default_style_prefix.borrow().as_ref() {
            return prefix.clone();
        }
        let prefix = Self::style_prefix_of(&|text| self.apply_default_style(text));
        *self.default_style_prefix.borrow_mut() = Some(prefix.clone());
        prefix
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the render pipeline is upstream's render(): cache, parse, per-token lines, wrap, margins, padding, cache"
    )]
    fn render(&self, width: usize) -> Vec<String> {
        // Check cache.
        {
            let cached_lines = self.cached_lines.borrow();
            if let Some(lines) = cached_lines.as_ref()
                && self.cached_text.borrow().as_deref() == Some(self.text.borrow().as_str())
                && *self.cached_width.borrow() == Some(width)
            {
                return lines.clone();
            }
        }

        // Calculate available width for content (subtract horizontal padding).
        let content_width = width.saturating_sub(self.padding_x * 2).max(1);
        let source_text = self.text.borrow().clone();
        let text = self.options.transform.as_ref().map_or_else(
            || source_text.clone(),
            |transform| transform(&source_text, content_width),
        );

        // Don't render anything if there's no actual text.
        if text.trim().is_empty() {
            *self.cached_text.borrow_mut() = Some(source_text);
            *self.cached_width.borrow_mut() = Some(width);
            *self.cached_lines.borrow_mut() = Some(Vec::new());
            return Vec::new();
        }

        // Replace tabs with 3 spaces for consistent rendering.
        let normalized = text.replace('\t', "   ");

        // Parse to blocks and render each one, deriving the blank-line
        // spacing marked's `space` tokens carried.
        let blocks = parse_document(&normalized);
        let mut rendered_lines: Vec<String> = Vec::new();
        if leading_blank_line(&blocks, &normalized) {
            rendered_lines.push(String::new());
        }
        for (index, block) in blocks.iter().enumerate() {
            let next = next_block_kind(&blocks, index, &normalized);
            self.render_block(
                block,
                content_width,
                next,
                None,
                &normalized,
                &mut rendered_lines,
            );
            if next == NextBlock::Space {
                rendered_lines.push(String::new());
            }
        }
        if trailing_blank_line(&blocks, &normalized) {
            rendered_lines.push(String::new());
        }

        // Wrap lines (NO padding, NO background yet).
        let mut wrapped_lines: Vec<String> = Vec::new();
        for line in &rendered_lines {
            if is_image_line(line) {
                wrapped_lines.push(line.clone());
            } else {
                wrapped_lines.extend(wrap_text_with_ansi(line, content_width));
            }
        }

        // Add margins and background to each wrapped line.
        let left_margin = " ".repeat(self.padding_x);
        let right_margin = " ".repeat(self.padding_x);
        let bg_fn = self
            .default_text_style
            .as_ref()
            .and_then(|style| style.bg_color.clone());
        let mut content_lines: Vec<String> = Vec::new();
        for line in &wrapped_lines {
            if is_image_line(line) {
                content_lines.push(line.clone());
                continue;
            }
            let line_with_margins = format!("{left_margin}{line}{right_margin}");
            if let Some(bg) = bg_fn.as_ref() {
                content_lines.push(apply_background_to_line(
                    &line_with_margins,
                    width,
                    bg.as_ref(),
                ));
            } else {
                // No background - just pad to width.
                let visible_len = visible_width(&line_with_margins);
                let padding_needed = width.saturating_sub(visible_len);
                content_lines.push(format!("{line_with_margins}{}", " ".repeat(padding_needed)));
            }
        }

        // Add top/bottom padding (empty lines).
        let empty_line = " ".repeat(width);
        let mut empty_lines: Vec<String> = Vec::new();
        for _ in 0..self.padding_y {
            if let Some(bg) = bg_fn.as_ref() {
                empty_lines.push(apply_background_to_line(&empty_line, width, bg.as_ref()));
            } else {
                empty_lines.push(empty_line.clone());
            }
        }

        // Combine top padding, content, and bottom padding.
        let mut result = empty_lines.clone();
        result.extend(content_lines);
        result.extend(empty_lines);

        // Update cache.
        *self.cached_text.borrow_mut() = Some(source_text);
        *self.cached_width.borrow_mut() = Some(width);
        *self.cached_lines.borrow_mut() = Some(result.clone());

        if result.is_empty() {
            vec![String::new()]
        } else {
            result
        }
    }

    /// Render one block, upstream `renderToken`.
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per token kind, kept side by side with upstream's renderToken switch"
    )]
    fn render_block(
        &self,
        block: &Token,
        width: usize,
        next: NextBlock,
        style_context: Option<&InlineStyleContext<'_>>,
        source: &str,
        lines: &mut Vec<String>,
    ) {
        match block {
            Token::Heading { depth, events, .. } => {
                let heading_prefix = format!("{} ", "#".repeat(*depth));
                // Build a heading-specific style context so inline tokens
                // (codespan, bold, etc.) restore heading styling after their
                // own ANSI resets instead of falling back to the default
                // text style.
                let heading_level = *depth;
                let heading_style = move |text: &str| -> String {
                    let inner = if heading_level == 1 {
                        (self.theme.bold)(&(self.theme.underline)(text))
                    } else {
                        (self.theme.bold)(text)
                    };
                    (self.theme.heading)(&inner)
                };
                let heading_style_prefix = Self::style_prefix_of(&heading_style);
                let heading_style_context = InlineStyleContext {
                    apply_text: &heading_style,
                    style_prefix: &heading_style_prefix,
                };
                let heading_text = self.render_inline(events, source, Some(&heading_style_context));
                let styled_heading = if *depth >= 3 {
                    format!("{}{}", heading_style(&heading_prefix), heading_text)
                } else {
                    heading_text
                };
                lines.push(styled_heading);
                if matches!(next, NextBlock::Kind(_)) {
                    lines.push(String::new()); // Add spacing after headings (unless space token follows)
                }
            }

            Token::Paragraph { events, .. } => {
                let paragraph_text = self.render_inline(events, source, style_context);
                lines.push(paragraph_text);
                // Don't add spacing if next token is space or list.
                if let NextBlock::Kind(kind) = next
                    && kind != BlockKind::List
                {
                    lines.push(String::new());
                }
            }

            Token::LatexBlock {
                text, pending, raw, ..
            } => {
                let rendered = if !*pending && self.options.render_latex {
                    render_latex(text, true).unwrap_or_else(|| raw.trim().to_string())
                } else {
                    raw.trim().to_string()
                };
                for line in rendered.split('\n') {
                    lines.push(style_context.map_or_else(
                        || self.apply_default_style(line),
                        |ctx| (ctx.apply_text)(line),
                    ));
                }
                if matches!(next, NextBlock::Kind(_)) {
                    lines.push(String::new());
                }
            }

            Token::CodeBlock { lang, text, range } => {
                let indent = self.theme.code_block_indent.as_deref().unwrap_or("  ");
                lines.push((self.theme.code_block_border)(&format!("```{lang}")));
                let code_text = trimmed_code_text(text, &source[range.clone()]);
                if let Some(highlight) = self.theme.highlight_code.as_ref() {
                    let lang = (!lang.is_empty()).then_some(lang.as_str());
                    for hl_line in highlight(code_text, lang) {
                        lines.push(format!("{indent}{hl_line}"));
                    }
                } else {
                    // Split code by newlines and style each line.
                    for code_line in code_text.split('\n') {
                        lines.push(format!("{indent}{}", (self.theme.code_block)(code_line)));
                    }
                }
                lines.push((self.theme.code_block_border)("```"));
                if matches!(next, NextBlock::Kind(_)) {
                    lines.push(String::new()); // Add spacing after code blocks (unless space token follows)
                }
            }

            Token::List { .. } => {
                self.render_list(block, 0, width, style_context, source, lines);
                // Don't add spacing after lists if a space token follows
                // (the space token will handle it).
            }

            Token::Table { .. } => {
                self.render_table(block, width, next, style_context, source, lines);
            }

            Token::BlockQuote {
                blocks: inner_blocks,
                ..
            } => {
                let quote_style =
                    |text: &str| -> String { (self.theme.quote)(&(self.theme.italic)(text)) };
                let quote_style_prefix = Self::style_prefix_of(&quote_style);
                let apply_quote_style = |line: &str| -> String {
                    if quote_style_prefix.is_empty() {
                        return quote_style(line);
                    }
                    let line_with_reapplied_style =
                        line.replace("\x1b[0m", &format!("\x1b[0m{quote_style_prefix}"));
                    quote_style(&line_with_reapplied_style)
                };

                // Calculate available width for quote content (subtract
                // border "│ " = 2 chars).
                let quote_content_width = width.saturating_sub(2).max(1);

                // Blockquotes contain block-level tokens (paragraph, list,
                // code, etc.), so render children with render_block instead
                // of the inline walker. Default message style should not
                // apply inside blockquotes.
                let noop_style = |text: &str| -> String { text.to_string() };
                let quote_inline_style_context = InlineStyleContext {
                    apply_text: &noop_style,
                    style_prefix: &quote_style_prefix,
                };
                let mut rendered_quote_lines: Vec<String> = Vec::new();
                for (index, quote_block) in inner_blocks.iter().enumerate() {
                    let next_quote = next_block_kind(inner_blocks, index, source);
                    self.render_block(
                        quote_block,
                        quote_content_width,
                        next_quote,
                        Some(&quote_inline_style_context),
                        source,
                        &mut rendered_quote_lines,
                    );
                    if next_quote == NextBlock::Space {
                        rendered_quote_lines.push(String::new());
                    }
                }

                // Avoid rendering an extra empty quote line before the
                // outer blockquote spacing.
                while rendered_quote_lines.last().is_some_and(String::is_empty) {
                    rendered_quote_lines.pop();
                }

                for quote_line in &rendered_quote_lines {
                    let styled_line = apply_quote_style(quote_line);
                    for wrapped_line in wrap_text_with_ansi(&styled_line, quote_content_width) {
                        lines.push(format!(
                            "{}{}",
                            (self.theme.quote_border)("│ "),
                            wrapped_line
                        ));
                    }
                }
                if matches!(next, NextBlock::Kind(_)) {
                    lines.push(String::new()); // Add spacing after blockquotes (unless space token follows)
                }
            }

            Token::Rule { .. } => {
                lines.push((self.theme.hr)(&"─".repeat(width.min(80))));
                if matches!(next, NextBlock::Kind(_)) {
                    lines.push(String::new()); // Add spacing after horizontal rules (unless space token follows)
                }
            }

            Token::Html { raw, .. } => {
                // Render HTML as plain text (styled).
                lines.push(self.apply_default_style(raw.trim()));
            }
        }
    }

    /// Render one block's inline content, upstream `renderInlineTokens`.
    fn render_inline(
        &self,
        events: &[(Event<'static>, Range<usize>)],
        source: &str,
        style_context: Option<&InlineStyleContext<'_>>,
    ) -> String {
        let default_prefix = self.default_style_prefix();
        // The fallback closure borrows self; map_or_else's returning
        // closure cannot hold that borrow across the call, so the match
        // form stays.
        #[allow(
            clippy::option_if_let_else,
            reason = "the fallback context borrows self, which a returning closure cannot capture"
        )]
        let ctx = match style_context {
            Some(ctx) => InlineStyleContext {
                apply_text: ctx.apply_text,
                style_prefix: ctx.style_prefix,
            },
            None => InlineStyleContext {
                apply_text: &|text: &str| self.apply_default_style(text),
                style_prefix: &default_prefix,
            },
        };
        let pairs = matching_ends(events);
        let claims = latex_claims(events, source);

        let mut result = String::new();
        self.render_inline_events(
            events,
            0,
            events.len(),
            source,
            &pairs,
            &claims,
            &ctx,
            &mut result,
        );

        // An element that appends the style prefix is always followed by
        // more styling in the same run, so a trailing repeat goes.
        while !ctx.style_prefix.is_empty() && result.ends_with(ctx.style_prefix) {
            result.truncate(result.len() - ctx.style_prefix.len());
        }

        result
    }

    /// Walk one event slice emitting styled output, the port of the inline
    /// token switch in `renderInlineTokens`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the inline walker threads the shared parse state: the event slice, source, match pairs, claims, and style context; the body mirrors upstream's single renderInlineTokens switch, and splitting it would scatter that state across helpers"
    )]
    fn render_inline_events(
        &self,
        events: &[(Event<'static>, Range<usize>)],
        from: usize,
        to: usize,
        source: &str,
        pairs: &[Option<usize>],
        claims: &[LatexClaim],
        ctx: &InlineStyleContext<'_>,
        out: &mut String,
    ) {
        let mut i = from;
        let mut prev_text_end: Option<usize> = None;
        while i < to {
            let (event, range) = &events[i];
            // Content inside an already-rendered LaTeX claim belongs to the
            // token's raw text; skip it. An event straddling the claim's end
            // still renders its tail.
            if let Some(index) = claim_covering_index(claims, range.start) {
                let claim = &claims[index];
                if claim.end >= range.end && claim.rendered.get() {
                    i += 1;
                    prev_text_end = None;
                    continue;
                }
            }
            match event {
                Event::Text(_) => {
                    self.emit_text_span(range.clone(), prev_text_end, source, claims, ctx, out);
                    prev_text_end = Some(range.end);
                }
                Event::Code(code) => {
                    out.push_str(&(self.theme.code)(code));
                    out.push_str(ctx.style_prefix);
                    prev_text_end = None;
                }
                Event::SoftBreak | Event::HardBreak => {
                    out.push('\n');
                    prev_text_end = None;
                }
                Event::InlineHtml(_) => {
                    let raw = &source[range.clone()];
                    out.push_str(&apply_text_with_newlines(ctx.apply_text, raw));
                    prev_text_end = None;
                }
                Event::Start(Tag::Strong) => {
                    let end = pairs[i].unwrap_or(to);
                    let mut inner = String::new();
                    self.render_inline_events(
                        events,
                        i + 1,
                        end,
                        source,
                        pairs,
                        claims,
                        ctx,
                        &mut inner,
                    );
                    out.push_str(&(self.theme.bold)(&inner));
                    out.push_str(ctx.style_prefix);
                    i = end;
                }
                Event::Start(Tag::Emphasis) => {
                    let end = pairs[i].unwrap_or(to);
                    let mut inner = String::new();
                    self.render_inline_events(
                        events,
                        i + 1,
                        end,
                        source,
                        pairs,
                        claims,
                        ctx,
                        &mut inner,
                    );
                    out.push_str(&(self.theme.italic)(&inner));
                    out.push_str(ctx.style_prefix);
                    i = end;
                }
                Event::Start(Tag::Strikethrough) => {
                    let end = pairs[i].unwrap_or(to);
                    // Upstream's strict tokenizer only delimits exact `~~…~~`;
                    // pulldown also enables single-tilde, which renders as
                    // its literal source.
                    let tilde_run = source[range.clone()]
                        .chars()
                        .take_while(|c| *c == '~')
                        .count();
                    if tilde_run == 2 {
                        let mut inner = String::new();
                        self.render_inline_events(
                            events,
                            i + 1,
                            end,
                            source,
                            pairs,
                            claims,
                            ctx,
                            &mut inner,
                        );
                        out.push_str(&(self.theme.strikethrough)(&inner));
                        out.push_str(ctx.style_prefix);
                    } else {
                        let raw = &source[range.clone()];
                        out.push_str(&apply_text_with_newlines(ctx.apply_text, raw));
                    }
                    i = end;
                }
                Event::Start(Tag::Link { dest_url, .. }) => {
                    let end = pairs[i].unwrap_or(to);
                    let mut inner = String::new();
                    self.render_inline_events(
                        events,
                        i + 1,
                        end,
                        source,
                        pairs,
                        claims,
                        ctx,
                        &mut inner,
                    );
                    // Upstream `token.text`: the raw label, reconstructed as
                    // the children's source span; angle email autolinks gain
                    // the `mailto:` prefix upstream's tokenizer adds.
                    let label = if end > i + 1 {
                        let label_start = events[i + 1].1.start;
                        let label_end = events[end - 1].1.end;
                        &source[label_start..label_end.max(label_start)]
                    } else {
                        ""
                    };
                    out.push_str(&self.render_link(&inner, label, dest_url, ctx));
                    i = end;
                }
                Event::Start(Tag::Image { .. }) => {
                    let end = pairs[i].unwrap_or(to);
                    // Upstream's default arm renders an image token as its
                    // alt text, the label's source span.
                    let label = if end > i + 1 {
                        let label_start = events[i + 1].1.start;
                        let label_end = events[end - 1].1.end;
                        &source[label_start..label_end.max(label_start)]
                    } else {
                        ""
                    };
                    out.push_str(&apply_text_with_newlines(ctx.apply_text, label));
                    i = end;
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Render one text event's source span: precomputed LaTeX claims take
    /// their spans, marked's gfm autolinks take theirs, and the rest styles
    /// as plain text.
    fn emit_text_span(
        &self,
        range: Range<usize>,
        prev_text_end: Option<usize>,
        source: &str,
        claims: &[LatexClaim],
        ctx: &InlineStyleContext<'_>,
        out: &mut String,
    ) {
        // pulldown leaves the escape backslashes in the gap between text
        // events; preserve mode re-emits them so the raw form surfaces. The
        // gap opens at the backslash run's own start, also when no text
        // event precedes it (paragraph-leading escape, or a break before).
        if self.options.preserve_backslash_escapes {
            let run_start = range.start
                - source[..range.start]
                    .bytes()
                    .rev()
                    .take_while(|byte| *byte == b'\\')
                    .count();
            let gap_start = run_start.max(prev_text_end.unwrap_or(0));
            if gap_start < range.start {
                flush_plain(gap_start, range.start, source, ctx, out);
            }
        }
        let mut pos = range.start;
        let mut plain_start = pos;
        while pos < range.end {
            if let Some(claim_index) = claim_covering_index(claims, pos) {
                flush_plain(plain_start, pos, source, ctx, out);
                let claim = &claims[claim_index];
                if !claim.rendered.get() {
                    claim.rendered.set(true);
                    out.push_str(&self.render_latex_token(&claim.token, false, ctx));
                }
                pos = claim.end.max(pos + 1);
                plain_start = pos.min(range.end);
                continue;
            }
            if let Some((match_end, link_text, href)) = try_autolink_at(pos, range.end, source) {
                flush_plain(plain_start, pos, source, ctx, out);
                let styled_inner = (ctx.apply_text)(link_text);
                out.push_str(&self.render_link(&styled_inner, link_text, &href, ctx));
                pos = match_end;
                plain_start = pos;
                continue;
            }
            pos += source[pos..range.end]
                .chars()
                .next()
                .map_or(1, char::len_utf8);
        }
        flush_plain(plain_start, range.end, source, ctx, out);
    }

    /// Render a claimed LaTeX token, upstream's `latex`/`latexBlock` arms:
    /// the renderer's answer, else the raw source.
    fn render_latex_token(
        &self,
        token: &LatexToken,
        display: bool,
        ctx: &InlineStyleContext<'_>,
    ) -> String {
        let rendered = if !token.pending && self.options.render_latex {
            render_latex(&token.text, display).unwrap_or_else(|| token.raw.clone())
        } else {
            token.raw.clone()
        };
        apply_text_with_newlines(ctx.apply_text, &rendered)
    }

    /// Render a link, upstream's `link` arm: OSC 8 when the terminal
    /// supports hyperlinks, otherwise the URL in parentheses when it
    /// differs from the visible label.
    fn render_link(
        &self,
        inner: &str,
        label: &str,
        dest_url: &str,
        ctx: &InlineStyleContext<'_>,
    ) -> String {
        let styled_link = (self.theme.link)(&(self.theme.underline)(inner));
        if get_capabilities().hyperlinks {
            // OSC 8: render as a clickable hyperlink. The URL is not printed
            // inline, so we always show only the link text regardless of
            // whether it matches href.
            format!("{}{}", hyperlink(&styled_link, dest_url), ctx.style_prefix)
        } else {
            // Fallback: print URL in parentheses when text differs from
            // href. Compare the raw label (not styled) against href; for
            // mailto: links strip the prefix (autolinked emails use
            // text="foo@bar.com" but href="mailto:foo@bar.com").
            let href_for_comparison = dest_url.strip_prefix("mailto:").unwrap_or(dest_url);
            if label == dest_url || label == href_for_comparison {
                format!("{}{}", styled_link, ctx.style_prefix)
            } else {
                format!(
                    "{}{}{}",
                    styled_link,
                    (self.theme.link_url)(&format!(" ({dest_url})")),
                    ctx.style_prefix
                )
            }
        }
    }

    /// Render a list with proper nesting support, upstream `renderList`.
    fn render_list(
        &self,
        list: &Token,
        depth: usize,
        width: usize,
        style_context: Option<&InlineStyleContext<'_>>,
        source: &str,
        lines: &mut Vec<String>,
    ) {
        let Token::List {
            ordered,
            start,
            loose,
            items,
            ..
        } = list
        else {
            return;
        };
        let indent = "    ".repeat(depth);
        // Use the list's start property (defaults to 1 for ordered lists).
        let start_number = start.unwrap_or(1);

        for (index, item) in items.iter().enumerate() {
            let is_last_item = index == items.len() - 1;
            let bullet = if *ordered {
                if self.options.preserve_ordered_list_markers {
                    get_ordered_list_marker(&source[item.range.clone()])
                        .unwrap_or_else(|| format!("{}. ", start_number + index as u64))
                } else {
                    format!("{}. ", start_number + index as u64)
                }
            } else if self.options.preserve_ordered_list_markers {
                get_unordered_list_marker(&source[item.range.clone()])
                    .unwrap_or_else(|| "- ".to_string())
            } else {
                "- ".to_string()
            };
            let task_marker = item.task.map_or(String::new(), |checked| {
                if checked {
                    "[x] ".to_string()
                } else {
                    "[ ] ".to_string()
                }
            });
            let marker = format!("{bullet}{task_marker}");
            let first_prefix = format!("{}{}", indent, (self.theme.list_bullet)(&marker));
            let continuation_prefix = format!("{}{}", indent, " ".repeat(visible_width(&marker)));
            let item_width = width.saturating_sub(visible_width(&first_prefix)).max(1);
            let mut rendered_any_line = false;

            for (item_index, item_block) in item.blocks.iter().enumerate() {
                if matches!(item_block, Token::List { .. }) {
                    self.render_list(item_block, depth + 1, width, style_context, source, lines);
                    rendered_any_line = true;
                    continue;
                }

                // Upstream renders item tokens with `nextTokenType`
                // undefined, so only a blank source line between the item's
                // own blocks produces a blank line here.
                let next_in_item =
                    item.blocks
                        .get(item_index + 1)
                        .map_or(NextBlock::None, |next| {
                            if blank_between(source, item_block.range().end, next.range().start) {
                                NextBlock::Space
                            } else {
                                NextBlock::None
                            }
                        });
                let mut item_lines: Vec<String> = Vec::new();
                self.render_block(
                    item_block,
                    item_width,
                    next_in_item,
                    style_context,
                    source,
                    &mut item_lines,
                );
                if next_in_item == NextBlock::Space {
                    item_lines.push(String::new());
                }
                for line in &item_lines {
                    for wrapped_line in wrap_text_with_ansi(line, item_width) {
                        let line_prefix = if rendered_any_line {
                            continuation_prefix.as_str()
                        } else {
                            first_prefix.as_str()
                        };
                        lines.push(format!("{line_prefix}{wrapped_line}"));
                        rendered_any_line = true;
                    }
                }
            }

            if !rendered_any_line {
                lines.push(first_prefix);
            }

            if *loose && !is_last_item {
                lines.push(String::new());
            }
        }
    }

    /// Wrap a table cell to fit into a column, upstream `wrapCellText`.
    ///
    /// Delegates to [`wrap_text_with_ansi`] so ANSI codes + long tokens are
    /// handled consistently with the rest of the renderer.
    fn wrap_cell_text(text: &str, max_width: usize, style_prefix: &str) -> Vec<String> {
        let lines = wrap_text_with_ansi(text, max_width.max(1));
        let fragment_count = lines.len();
        lines
            .into_iter()
            .enumerate()
            .map(|(index, line)| {
                // Reset text styles after each non-final fragment, then
                // restore the surrounding style before padding and borders.
                let style_reset = if index < fragment_count - 1 {
                    "\x1b[22;23;24;25;27;28;29;39m"
                } else {
                    ""
                };
                format!("{line}{style_reset}{style_prefix}")
            })
            .collect()
    }

    /// Render a table with width-aware cell wrapping, upstream
    /// `renderTable`. Cells that don't fit are wrapped to multiple lines.
    #[allow(
        clippy::too_many_lines,
        reason = "the column-sizing and painting stages stay 1:1 with upstream's renderTable"
    )]
    fn render_table(
        &self,
        table: &Token,
        available_width: usize,
        next: NextBlock,
        style_context: Option<&InlineStyleContext<'_>>,
        source: &str,
        lines: &mut Vec<String>,
    ) {
        let Token::Table { range, head, rows } = table else {
            return;
        };
        let num_cols = head.len();

        if num_cols == 0 {
            return;
        }

        let style_prefix = style_context.map_or("", |context| context.style_prefix);

        // Calculate border overhead: "│ " + (n-1) * " │ " + " │" = 3n + 1.
        let border_overhead = 3 * num_cols + 1;
        let available_for_cells = available_width.saturating_sub(border_overhead);
        if available_for_cells < num_cols {
            // Too narrow to render a stable table. Fall back to raw markdown.
            lines.extend(wrap_text_with_ansi(&source[range.clone()], available_width));
            if matches!(next, NextBlock::Kind(_)) {
                lines.push(String::new());
            }
            return;
        }

        let max_unbroken_word_width = 30;

        // Calculate natural column widths (what each column needs without
        // constraints).
        let mut natural_widths = vec![0usize; num_cols];
        let mut min_word_widths = vec![1usize; num_cols];
        for (i, cell) in head.iter().enumerate() {
            let header_text = self.render_inline(cell, source, style_context);
            natural_widths[i] = visible_width(&header_text);
            min_word_widths[i] = longest_word_width(&header_text, max_unbroken_word_width).max(1);
        }
        for row in rows {
            for (i, cell) in row.iter().enumerate().take(num_cols) {
                let cell_text = self.render_inline(cell, source, style_context);
                natural_widths[i] = natural_widths[i].max(visible_width(&cell_text));
                min_word_widths[i] =
                    min_word_widths[i].max(longest_word_width(&cell_text, max_unbroken_word_width));
            }
        }

        let mut min_column_widths = min_word_widths.clone();
        let mut min_cells_width: usize = min_column_widths.iter().sum();

        if min_cells_width > available_for_cells {
            min_column_widths = vec![1; num_cols];
            let remaining = available_for_cells.saturating_sub(num_cols);

            if remaining > 0 {
                let total_weight: usize = min_word_widths
                    .iter()
                    .map(|width| width.saturating_sub(1))
                    .sum();
                let growth: Vec<usize> = min_word_widths
                    .iter()
                    .map(|width| {
                        (width.saturating_sub(1) * remaining)
                            .checked_div(total_weight)
                            .unwrap_or(0)
                    })
                    .collect();

                for (i, growth) in growth.iter().enumerate() {
                    min_column_widths[i] += growth;
                }

                let allocated: usize = growth.iter().sum();
                let mut leftover = remaining.saturating_sub(allocated);
                for width in &mut min_column_widths {
                    if leftover == 0 {
                        break;
                    }
                    *width += 1;
                    leftover -= 1;
                }
            }

            min_cells_width = min_column_widths.iter().sum();
        }

        // Calculate column widths that fit within available width.
        let total_natural_width: usize = natural_widths.iter().sum::<usize>() + border_overhead;
        let column_widths: Vec<usize> = if total_natural_width <= available_width {
            // Everything fits naturally.
            natural_widths
                .iter()
                .enumerate()
                .map(|(index, width)| (*width).max(min_column_widths[index]))
                .collect()
        } else {
            // Need to shrink columns to fit.
            let total_grow_potential: usize = natural_widths
                .iter()
                .enumerate()
                .map(|(index, width)| width.saturating_sub(min_column_widths[index]))
                .sum();
            let extra_width = available_for_cells.saturating_sub(min_cells_width);
            let mut widths: Vec<usize> = min_column_widths
                .iter()
                .enumerate()
                .map(|(index, min_width)| {
                    let min_width_delta = natural_widths[index].saturating_sub(*min_width);
                    let grow = min_width_delta
                        .checked_mul(extra_width)
                        .and_then(|scaled| scaled.checked_div(total_grow_potential))
                        .unwrap_or(0);
                    min_width + grow
                })
                .collect();

            // Adjust for rounding errors - distribute remaining space.
            let allocated: usize = widths.iter().sum();
            let mut remaining = available_for_cells.saturating_sub(allocated);
            while remaining > 0 {
                let mut grew = false;
                for (index, width) in widths.iter_mut().enumerate() {
                    if remaining == 0 {
                        break;
                    }
                    if *width < natural_widths[index] {
                        *width += 1;
                        remaining -= 1;
                        grew = true;
                    }
                }
                if !grew {
                    break;
                }
            }
            widths
        };

        // Render top border.
        let top_border_cells: Vec<String> = column_widths
            .iter()
            .map(|width| "─".repeat(*width))
            .collect();
        lines.push(format!("┌─{}─┐", top_border_cells.join("─┬─")));

        // Render header with wrapping.
        let header_cell_lines: Vec<Vec<String>> = head
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let text = self.render_inline(cell, source, style_context);
                Self::wrap_cell_text(&text, column_widths[i], style_prefix)
            })
            .collect();
        let header_line_count = header_cell_lines.iter().map(Vec::len).max().unwrap_or(0);

        for line_index in 0..header_line_count {
            let row_parts: Vec<String> = header_cell_lines
                .iter()
                .enumerate()
                .map(|(col_index, cell_lines)| {
                    let text = cell_lines.get(line_index).cloned().unwrap_or_default();
                    let padded = format!(
                        "{text}{}",
                        " ".repeat(column_widths[col_index].saturating_sub(visible_width(&text)))
                    );
                    (self.theme.bold)(&padded)
                })
                .collect();
            lines.push(format!("│ {} │", row_parts.join(" │ ")));
        }

        // Render separator.
        let separator_cells: Vec<String> = column_widths
            .iter()
            .map(|width| "─".repeat(*width))
            .collect();
        let separator_line = format!("├─{}─┤", separator_cells.join("─┼─"));
        lines.push(separator_line.clone());

        // Render rows with wrapping.
        for (row_index, row) in rows.iter().enumerate() {
            let row_cell_lines: Vec<Vec<String>> = row
                .iter()
                .take(num_cols)
                .enumerate()
                .map(|(i, cell)| {
                    let text = self.render_inline(cell, source, style_context);
                    Self::wrap_cell_text(&text, column_widths[i], style_prefix)
                })
                .collect();
            let row_line_count = row_cell_lines.iter().map(Vec::len).max().unwrap_or(0);

            for line_index in 0..row_line_count {
                let row_parts: Vec<String> = row_cell_lines
                    .iter()
                    .enumerate()
                    .map(|(col_index, cell_lines)| {
                        let text = cell_lines.get(line_index).cloned().unwrap_or_default();
                        format!(
                            "{text}{}",
                            " ".repeat(
                                column_widths[col_index].saturating_sub(visible_width(&text))
                            )
                        )
                    })
                    .collect();
                lines.push(format!("│ {} │", row_parts.join(" │ ")));
            }

            if row_index < rows.len() - 1 {
                lines.push(separator_line.clone());
            }
        }

        // Render bottom border.
        let bottom_border_cells: Vec<String> = column_widths
            .iter()
            .map(|width| "─".repeat(*width))
            .collect();
        lines.push(format!("└─{}─┘", bottom_border_cells.join("─┴─")));

        if matches!(next, NextBlock::Kind(_)) {
            lines.push(String::new()); // Add spacing after table
        }
    }
}

/// Get the visible width of the longest word in a string, upstream
/// `getLongestWordWidth`, capped at `max_width`.
fn longest_word_width(text: &str, max_width: usize) -> usize {
    let mut longest = 0;
    for word in text.split_whitespace() {
        longest = longest.max(visible_width(word));
    }
    longest.min(max_width)
}

impl crate::tui::Component for Markdown {
    fn render(&self, width: usize) -> Vec<String> {
        self.render(width)
    }

    fn invalidate(&self) {
        *self.default_style_prefix.borrow_mut() = None;
        *self.cached_text.borrow_mut() = None;
        *self.cached_width.borrow_mut() = None;
        *self.cached_lines.borrow_mut() = None;
    }
}

/// Style one text piece with per-line application, upstream
/// `applyTextWithNewlines`.
fn apply_text_with_newlines(apply_text: &dyn Fn(&str) -> String, text: &str) -> String {
    text.split('\n')
        .map(apply_text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Emit the plain-text piece between styled spans, upstream's text token
/// application.
fn flush_plain(
    start: usize,
    end: usize,
    source: &str,
    ctx: &InlineStyleContext<'_>,
    out: &mut String,
) {
    if start < end {
        let piece = &source[start..end];
        out.push_str(&apply_text_with_newlines(ctx.apply_text, piece));
    }
}

// --- parsing ----------------------------------------------------------------

/// Parse a normalized document into blocks: the block-level LaTeX pre-pass
/// segments the source (skipping code-fenced lines), each markdown chunk
/// parses through pulldown, and the block lists stitch in order.
fn parse_document(source: &str) -> Vec<Token> {
    let mut blocks: Vec<Token> = Vec::new();
    let mut chunk_start = 0usize;
    let mut pos = 0usize;
    let mut fence = FenceTracker::default();
    while pos < source.len() {
        let line_end = source[pos..]
            .find('\n')
            .map_or(source.len(), |i| pos + i + 1);
        let line = &source[pos..line_end];
        if !fence.inside()
            && let Some(token) = tokenize_block_latex(&source[pos..])
        {
            if pos > chunk_start {
                blocks.extend(parse_chunk(&source[chunk_start..pos], chunk_start));
            }
            let raw_len = token.raw.len();
            blocks.push(Token::LatexBlock {
                text: token.text,
                pending: token.pending,
                raw: token.raw,
                range: pos..pos + raw_len,
            });
            pos += raw_len;
            chunk_start = pos;
            continue;
        }
        fence.feed(line);
        pos = line_end;
    }
    if chunk_start < source.len() {
        blocks.extend(parse_chunk(&source[chunk_start..], chunk_start));
    }
    blocks
}

/// Parse one markdown chunk, offsets shifted to absolute positions.
fn parse_chunk(chunk: &str, base: usize) -> Vec<Token> {
    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let events: Vec<(Event<'static>, Range<usize>)> = Parser::new_ext(chunk, options)
        .into_offset_iter()
        .map(|(event, range)| (owned_event(event), range.start + base..range.end + base))
        .collect();
    group_blocks(&events)
}

/// Rebuild one event with owned strings so block grouping can outlive the
/// parser borrow.
fn owned_event(event: Event<'_>) -> Event<'static> {
    match event {
        Event::Start(tag) => Event::Start(owned_tag(tag)),
        Event::End(end) => Event::End(end),
        Event::Text(text) => Event::Text(text.into_string().into()),
        Event::Code(text) => Event::Code(text.into_string().into()),
        Event::Html(text) => Event::Html(text.into_string().into()),
        Event::InlineHtml(text) => Event::InlineHtml(text.into_string().into()),
        Event::FootnoteReference(id) => Event::FootnoteReference(id.into_string().into()),
        Event::SoftBreak => Event::SoftBreak,
        Event::HardBreak => Event::HardBreak,
        Event::Rule => Event::Rule,
        Event::TaskListMarker(checked) => Event::TaskListMarker(checked),
        Event::InlineMath(text) => Event::InlineMath(text.into_string().into()),
        Event::DisplayMath(text) => Event::DisplayMath(text.into_string().into()),
    }
}

fn owned_tag(tag: Tag<'_>) -> Tag<'static> {
    match tag {
        Tag::Paragraph => Tag::Paragraph,
        Tag::Heading {
            level,
            id,
            classes,
            attrs,
        } => Tag::Heading {
            level,
            id: id.map(|id| id.into_string().into()),
            classes: classes
                .into_iter()
                .map(|c| c.into_string().into())
                .collect(),
            attrs: attrs
                .into_iter()
                .map(|(key, value)| {
                    (
                        key.into_string().into(),
                        value.map(|v| v.into_string().into()),
                    )
                })
                .collect(),
        },
        Tag::BlockQuote(kind) => Tag::BlockQuote(kind),
        Tag::CodeBlock(kind) => Tag::CodeBlock(match kind {
            CodeBlockKind::Indented => CodeBlockKind::Indented,
            CodeBlockKind::Fenced(info) => CodeBlockKind::Fenced(info.into_string().into()),
        }),
        Tag::List(start) => Tag::List(start),
        Tag::Item => Tag::Item,
        Tag::FootnoteDefinition(id) => Tag::FootnoteDefinition(id.into_string().into()),
        Tag::DefinitionList => Tag::DefinitionList,
        Tag::DefinitionListTitle => Tag::DefinitionListTitle,
        Tag::DefinitionListDefinition => Tag::DefinitionListDefinition,
        Tag::Table(alignments) => Tag::Table(alignments),
        Tag::TableHead => Tag::TableHead,
        Tag::TableRow => Tag::TableRow,
        Tag::TableCell => Tag::TableCell,
        Tag::Emphasis => Tag::Emphasis,
        Tag::Strong => Tag::Strong,
        Tag::Strikethrough => Tag::Strikethrough,
        Tag::Link {
            link_type,
            dest_url,
            title,
            id,
        } => Tag::Link {
            link_type,
            dest_url: dest_url.into_string().into(),
            title: title.into_string().into(),
            id: id.into_string().into(),
        },
        Tag::Image {
            link_type,
            dest_url,
            title,
            id,
        } => Tag::Image {
            link_type,
            dest_url: dest_url.into_string().into(),
            title: title.into_string().into(),
            id: id.into_string().into(),
        },
        Tag::HtmlBlock => Tag::HtmlBlock,
        Tag::MetadataBlock(kind) => Tag::MetadataBlock(kind),
        Tag::Superscript => Tag::Superscript,
        Tag::Subscript => Tag::Subscript,
    }
}

/// Group a flat event run into blocks, matching Start/End pairs.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per token kind, kept side by side with upstream's lexer cases"
)]
fn group_blocks(events: &[InlineEvent]) -> Vec<Token> {
    let mut blocks: Vec<Token> = Vec::new();
    let mut i = 0;
    while i < events.len() {
        if matches!(events[i].0, Event::Rule) {
            blocks.push(Token::Rule {
                range: events[i].1.clone(),
            });
            i += 1;
            continue;
        }
        if matches!(events[i].0, Event::TaskListMarker(_)) {
            // The item's task state renders in the marker prefix; the event
            // itself carries no content.
            i += 1;
            continue;
        }
        let block_start = match &events[i].0 {
            // Inline tags (strong, link, …) inside a tight item's bare run
            // open the same implicit paragraph, not a block.
            Event::Start(tag) if !is_inline_event(&events[i].0) => Some(tag),
            _ => None,
        };
        let Some(tag) = block_start else {
            // Bare inline events (tight list items carry their content
            // without a paragraph wrapper) group into an implicit paragraph.
            let end = implicit_paragraph_end(events, i);
            blocks.push(Token::Paragraph {
                range: events[i].1.start..events[end].1.end,
                events: events[i..=end].to_vec(),
            });
            i = end + 1;
            continue;
        };
        let want = TagEnd::from(tag.clone());
        match tag {
            Tag::Paragraph => {
                let Some(end) = find_matching_end(events, i, want) else {
                    break;
                };
                blocks.push(Token::Paragraph {
                    range: events[i].1.clone(),
                    events: events[i..=end].to_vec(),
                });
                i = end + 1;
            }
            Tag::Heading { level, .. } => {
                let Some(end) = find_matching_end(events, i, want) else {
                    break;
                };
                blocks.push(Token::Heading {
                    depth: heading_depth(*level),
                    range: events[i].1.clone(),
                    events: events[i..=end].to_vec(),
                });
                i = end + 1;
            }
            Tag::CodeBlock(kind) => {
                let Some(end) = find_matching_end(events, i, want) else {
                    break;
                };
                let lang = match kind {
                    CodeBlockKind::Fenced(info) => info.to_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                let text = events[i + 1..=end]
                    .iter()
                    .filter_map(|(event, _)| match event {
                        Event::Text(text) => Some(text.to_string()),
                        _ => None,
                    })
                    .collect::<String>();
                blocks.push(Token::CodeBlock {
                    lang,
                    text,
                    range: events[i].1.clone(),
                });
                i = end + 1;
            }
            Tag::List(start) => {
                let Some(list_end) = find_matching_end(events, i, want) else {
                    break;
                };
                let ordered = match &events[list_end].0 {
                    Event::End(TagEnd::List(ordered)) => *ordered,
                    _ => start.is_some(),
                };
                let mut items = Vec::new();
                let mut loose = false;
                let mut j = i + 1;
                while j < list_end {
                    let Event::Start(Tag::Item) = &events[j].0 else {
                        j += 1;
                        continue;
                    };
                    let Some(item_end) = find_matching_end(events, j, TagEnd::Item) else {
                        break;
                    };
                    let item_events = &events[j + 1..item_end];
                    let task = item_events.iter().find_map(|(event, _)| match event {
                        Event::TaskListMarker(checked) => Some(*checked),
                        _ => None,
                    });
                    loose |= item_events
                        .iter()
                        .any(|(event, _)| matches!(event, Event::Start(Tag::Paragraph)));
                    items.push(ListItem {
                        range: events[j].1.clone(),
                        task,
                        blocks: group_blocks(item_events),
                    });
                    j = item_end + 1;
                }
                blocks.push(Token::List {
                    ordered,
                    start: *start,
                    loose,
                    items,
                    range: events[i].1.clone(),
                });
                i = list_end + 1;
            }
            Tag::Table(_) => {
                let Some(table_end) = find_matching_end(events, i, want) else {
                    break;
                };
                let mut head = Vec::new();
                let mut rows: Vec<Vec<InlineEvents>> = Vec::new();
                let mut j = i + 1;
                while j < table_end {
                    match &events[j].0 {
                        Event::Start(Tag::TableHead) => {
                            let Some(head_end) = find_matching_end(events, j, TagEnd::TableHead)
                            else {
                                break;
                            };
                            head = collect_cells(&events[j + 1..head_end]);
                            j = head_end + 1;
                        }
                        Event::Start(Tag::TableRow) => {
                            let Some(row_end) = find_matching_end(events, j, TagEnd::TableRow)
                            else {
                                break;
                            };
                            rows.push(collect_cells(&events[j + 1..row_end]));
                            j = row_end + 1;
                        }
                        _ => j += 1,
                    }
                }
                blocks.push(Token::Table {
                    range: events[i].1.clone(),
                    head,
                    rows,
                });
                i = table_end + 1;
            }
            Tag::BlockQuote(_) => {
                let Some(end) = find_matching_end(events, i, want) else {
                    break;
                };
                blocks.push(Token::BlockQuote {
                    range: events[i].1.clone(),
                    blocks: group_blocks(&events[i + 1..end]),
                });
                i = end + 1;
            }
            Tag::HtmlBlock => {
                let Some(end) = find_matching_end(events, i, want) else {
                    break;
                };
                let raw = events[i + 1..=end]
                    .iter()
                    .filter_map(|(event, _)| match event {
                        Event::Html(html) => Some(html.to_string()),
                        _ => None,
                    })
                    .collect::<String>();
                blocks.push(Token::Html {
                    raw,
                    range: events[i].1.clone(),
                });
                i = end + 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    blocks
}

/// Whether one event belongs to an inline run: leaf inline events and the
/// Start/End of inline tags. Block-level tags end the run.
const fn is_inline_event(event: &Event<'_>) -> bool {
    match event {
        Event::Text(_)
        | Event::Code(_)
        | Event::SoftBreak
        | Event::HardBreak
        | Event::InlineHtml(_)
        | Event::FootnoteReference(_) => true,
        Event::Start(tag) => {
            matches!(
                tag,
                Tag::Strong
                    | Tag::Emphasis
                    | Tag::Strikethrough
                    | Tag::Link { .. }
                    | Tag::Image { .. }
            )
        }
        Event::End(end) => {
            matches!(
                end,
                TagEnd::Strong
                    | TagEnd::Emphasis
                    | TagEnd::Strikethrough
                    | TagEnd::Link
                    | TagEnd::Image
            )
        }
        _ => false,
    }
}

/// The last index of the maximal run of inline events starting at `from`:
/// tight list items carry their content bare, without a paragraph wrapper,
/// so the run groups into one paragraph token.
fn implicit_paragraph_end(events: &[InlineEvent], from: usize) -> usize {
    let mut end = from;
    for (index, (event, _)) in events.iter().enumerate().skip(from) {
        if is_inline_event(event) {
            end = index;
        } else {
            break;
        }
    }
    end
}

fn collect_cells(events: &[InlineEvent]) -> Vec<InlineEvents> {
    let mut cells = Vec::new();
    let mut i = 0;
    while i < events.len() {
        if matches!(events[i].0, Event::Start(Tag::TableCell)) {
            let Some(end) = find_matching_end(events, i, TagEnd::TableCell) else {
                break;
            };
            cells.push(events[i..=end].to_vec());
            i = end + 1;
        } else {
            i += 1;
        }
    }
    cells
}

fn find_matching_end(events: &[InlineEvent], start: usize, want: TagEnd) -> Option<usize> {
    let mut depth = 0usize;
    for (i, (event, _)) in events.iter().enumerate().skip(start + 1) {
        match event {
            Event::Start(tag) => {
                if TagEnd::from(tag.clone()) == want {
                    depth += 1;
                }
            }
            Event::End(end) if *end == want => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

const fn heading_depth(level: HeadingLevel) -> usize {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// For each `Start` event index, the matching `End` index.
fn matching_ends(events: &[InlineEvent]) -> Vec<Option<usize>> {
    let mut pairs = vec![None; events.len()];
    let mut stack: Vec<usize> = Vec::new();
    for (i, (event, _)) in events.iter().enumerate() {
        match event {
            Event::Start(_) => stack.push(i),
            Event::End(_) => {
                if let Some(start) = stack.pop() {
                    pairs[start] = Some(i);
                }
            }
            _ => {}
        }
    }
    pairs
}

// --- spacing helpers: marked's `space` tokens re-derived from source lines

/// The spans of the source's lines, each including its line terminator.
fn line_spans(source: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for (index, _) in source.match_indices('\n') {
        spans.push((start, index + 1));
        start = index + 1;
    }
    if start < source.len() {
        spans.push((start, source.len()));
    }
    spans
}

fn blank_between(source: &str, end_byte: usize, start_byte: usize) -> bool {
    line_spans(source).into_iter().any(|(start, end)| {
        start >= end_byte && end <= start_byte && source[start..end].trim().is_empty()
    })
}

fn leading_blank_line(blocks: &[Token], source: &str) -> bool {
    let Some(first) = blocks.first() else {
        return false;
    };
    blank_between(source, 0, first.range().start)
}

fn trailing_blank_line(blocks: &[Token], source: &str) -> bool {
    let Some(last) = blocks.last() else {
        return false;
    };
    blank_between(source, last.range().end, source.len())
}

fn next_block_kind(blocks: &[Token], index: usize, source: &str) -> NextBlock {
    let Some(next) = blocks.get(index + 1) else {
        return NextBlock::None;
    };
    let current = &blocks[index];
    // pulldown's list span swallows the blank lines marked's list raw
    // leaves out, so the spacing gap starts at the last item's content end.
    let gap_start = match current {
        Token::List { items, .. } => items.last().map(|item| {
            item.blocks
                .last()
                .map_or(item.range.end, |last| last.range().end)
        }),
        _ => None,
    }
    .unwrap_or_else(|| current.range().end);
    if blank_between(source, gap_start, next.range().start) {
        NextBlock::Space
    } else {
        NextBlock::Kind(next.kind())
    }
}

// --- LaTeX tokenizers, upstream's marked extension tokenizers

/// Upstream `looksLikePendingDollarMath`.
fn looks_like_pending_dollar_math(source: &str) -> bool {
    static PENDING_MATH_RE: LazyLock<Regex> =
        LazyLock::new(|| static_regex(r"\\[A-Za-z]+|[_^=+*/<>()\[\]|±≤≥≠≈∈→⇒∞∫∑√-]"));
    PENDING_MATH_RE.is_match(source)
}

/// Whether the character at `index` sits behind an odd backslash run,
/// upstream `isEscaped`.
fn is_escaped(source: &str, index: usize) -> bool {
    let bytes = source.as_bytes();
    let mut backslashes = 0usize;
    let mut position = index;
    while position > 0 && bytes.get(position - 1) == Some(&b'\\') {
        backslashes += 1;
        position -= 1;
    }
    backslashes % 2 == 1
}

/// The first unescaped `closing` at or after `from`, upstream
/// `findClosingDelimiter`.
fn find_closing_delimiter(haystack: &str, closing: &str, from: usize) -> Option<usize> {
    let mut index = haystack
        .get(from..)
        .and_then(|rest| rest.find(closing))
        .map(|i| from + i);
    while let Some(i) = index {
        if !is_escaped(haystack, i) {
            return Some(i);
        }
        index = haystack[i + closing.len()..]
            .find(closing)
            .map(|j| i + closing.len() + j);
    }
    None
}

/// Upstream `tokenizeInlineLatex`, over the source from a candidate
/// delimiter to the end of the inline context.
fn tokenize_inline_latex(src: &str) -> Option<LatexToken> {
    static DOLLAR_SPACE_RE: LazyLock<Regex> = LazyLock::new(|| static_regex(r"^\$\s"));
    static ALL_CAPS_RE: LazyLock<Regex> =
        LazyLock::new(|| static_regex(r"^[A-Z_][A-Z0-9_]*(?:[^A-Za-z0-9_\s])?$"));
    static IDENT_RE: LazyLock<Regex> = LazyLock::new(|| static_regex(r"^[A-Za-z_][A-Za-z0-9_]*"));

    let (opening, closing) = if src.starts_with("$$") {
        ("$$", "$$")
    } else if src.starts_with("\\(") {
        ("\\(", "\\)")
    } else if src.starts_with("\\[") {
        ("\\[", "\\]")
    } else if src.starts_with('$') && !DOLLAR_SPACE_RE.is_match(src) {
        ("$", "$")
    } else {
        return None;
    };

    let opening_len = opening.len();
    let closing_index = find_closing_delimiter(src, closing, opening_len);
    if let Some(ci) = closing_index
        && opening == "$"
    {
        let content = &src[opening_len..ci];
        let after = &src[ci + 1..];
        let trailing_whitespace = content.chars().last().is_some_and(char::is_whitespace);
        if trailing_whitespace
            || after.starts_with(|c: char| c.is_ascii_digit())
            || (ALL_CAPS_RE.is_match(content) && IDENT_RE.is_match(after))
            || content.contains('`')
        {
            return None;
        }
    }

    let Some(ci) = closing_index else {
        let pending_source = &src[opening_len..];
        if opening.starts_with('\\') || looks_like_pending_dollar_math(pending_source) {
            return Some(LatexToken {
                raw: src.to_string(),
                text: pending_source.to_string(),
                pending: true,
            });
        }
        return None;
    };

    let text = &src[opening_len..ci];
    if text.is_empty() || text.contains('\n') {
        return None;
    }
    Some(LatexToken {
        raw: src[..ci + closing.len()].to_string(),
        text: text.to_string(),
        pending: false,
    })
}

/// Upstream `tokenizeBlockLatex`, over the source from a candidate line
/// start.
fn tokenize_block_latex(src: &str) -> Option<LatexToken> {
    static DOLLAR_RE: LazyLock<Regex> =
        LazyLock::new(|| static_regex(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*?)\$\$[ \t]*(?:\n|$)"));
    static BRACKET_RE: LazyLock<Regex> =
        LazyLock::new(|| static_regex(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*?)\\\][ \t]*(?:\n|$)"));
    static PENDING_BRACKET_RE: LazyLock<Regex> =
        LazyLock::new(|| static_regex(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*)$"));
    static PENDING_DOLLAR_RE: LazyLock<Regex> =
        LazyLock::new(|| static_regex(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*)$"));

    if let Some(caps) = DOLLAR_RE.captures(src)
        && !caps[1].is_empty()
    {
        return Some(LatexToken {
            raw: caps[0].to_string(),
            text: caps[1].trim().to_string(),
            pending: false,
        });
    }
    if let Some(caps) = BRACKET_RE.captures(src)
        && !caps[1].is_empty()
    {
        return Some(LatexToken {
            raw: caps[0].to_string(),
            text: caps[1].trim().to_string(),
            pending: false,
        });
    }
    if let Some(caps) = PENDING_BRACKET_RE.captures(src) {
        return Some(LatexToken {
            raw: caps[0].to_string(),
            text: caps[1].to_string(),
            pending: true,
        });
    }
    if let Some(caps) = PENDING_DOLLAR_RE.captures(src)
        && !caps[1].is_empty()
        && looks_like_pending_dollar_math(&caps[1])
    {
        return Some(LatexToken {
            raw: caps[0].to_string(),
            text: caps[1].to_string(),
            pending: true,
        });
    }
    None
}

/// Scan one inline context for LaTeX spans, upstream's inline extension
/// `start` hook over the whole context: candidates sit in text events
/// (`\(`/`\[` open at their backslash, whose punctuation the following
/// text event covers), claims may close across events, and pending
/// claims run to the context end like marked's rest-of-source token.
fn latex_claims(events: &[InlineEvent], source: &str) -> Vec<LatexClaim> {
    let Some(first) = events.first() else {
        return Vec::new();
    };
    let context_start = first.1.start;
    let context_end = events
        .iter()
        .map(|(_, range)| range.end)
        .max()
        .unwrap_or(context_start);
    let text_ranges: Vec<Range<usize>> = events
        .iter()
        .filter_map(|(event, range)| match event {
            Event::Text(_) => Some(range.clone()),
            _ => None,
        })
        .collect();

    let mut claims: Vec<LatexClaim> = Vec::new();
    let mut pos = context_start;
    while pos < context_end {
        if let Some(claim) = claims
            .iter()
            .find(|claim| claim.start <= pos && pos < claim.end)
        {
            pos = claim.end;
            continue;
        }
        let bytes = source.as_bytes();
        let dollar = bytes.get(pos) == Some(&b'$')
            && text_covers(&text_ranges, pos)
            && !is_escaped(source, pos);
        let bracket = bytes.get(pos) == Some(&b'\\')
            && matches!(bytes.get(pos + 1), Some(b'(' | b'['))
            && !is_escaped(source, pos)
            && text_covers(&text_ranges, pos + 1);
        if (dollar || bracket)
            && let Some(token) = tokenize_inline_latex(&source[pos..context_end])
        {
            let end = if token.pending {
                context_end
            } else {
                pos + token.raw.len()
            };
            claims.push(LatexClaim {
                start: pos,
                end,
                token,
                rendered: Cell::new(false),
            });
            pos = end;
            continue;
        }
        pos += 1;
    }
    claims
}

fn text_covers(text_ranges: &[Range<usize>], pos: usize) -> bool {
    text_ranges
        .iter()
        .any(|range| range.start <= pos && pos < range.end)
}

fn claim_covering_index(claims: &[LatexClaim], pos: usize) -> Option<usize> {
    claims
        .iter()
        .position(|claim| claim.start <= pos && pos < claim.end)
}

// --- code fence helpers

/// The fenced block's code text in marked's `token.text` form: a closed
/// fence's trailing newline drops (marked strips it), an unclosed fence's
/// partial closing line comes off (upstream `trimPartialClosingFences`, see
/// earendil-works/pi#5825) so code blocks do not shrink when the final
/// fence character arrives.
fn trimmed_code_text<'a>(text: &'a str, raw: &str) -> &'a str {
    let lines: Vec<&str> = raw.split('\n').collect();
    let Some((marker_char, marker_len)) = lines.first().and_then(|line| fence_marker(line)) else {
        return text;
    };
    let last_line = lines.last().copied().unwrap_or("");
    // A closing fence is the marker run alone (up to three leading spaces,
    // only spaces or tabs after), which an unclosed block never has.
    let trimmed_last = last_line.trim_start_matches([' ', '\t']);
    let run = trimmed_last
        .chars()
        .take_while(|c| *c == marker_char)
        .count();
    let closed = run >= marker_len && trimmed_last[run..].chars().all(|c| c == ' ' || c == '\t');
    let text = if closed {
        text.strip_suffix('\n').unwrap_or(text)
    } else {
        text
    };

    // A partial closing fence is a strict prefix of the marker repeated.
    if last_line.is_empty() || last_line.len() >= marker_len {
        return text;
    }
    if !last_line.bytes().all(|c| c == marker_char as u8) {
        return text;
    }
    let trimmed = &text[..text.len().saturating_sub(last_line.len())];
    trimmed.strip_suffix('\n').unwrap_or(trimmed)
}

/// A fence line's marker run: `(char, count)` when the line opens with
/// three or more of one fence character after at most three spaces.
fn fence_marker(line: &str) -> Option<(char, usize)> {
    let leading = line.len() - line.trim_start_matches(' ').len();
    if leading > 3 {
        return None;
    }
    let rest = &line[leading..];
    let first = rest.chars().next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let count = rest.chars().take_while(|c| *c == first).count();
    (count >= 3).then_some((first, count))
}

/// Whether one emulated line sits inside an open code fence, so the
/// block-latex pre-pass leaves fenced content alone.
#[derive(Default)]
struct FenceTracker {
    open: Option<(u8, usize)>,
}

impl FenceTracker {
    const fn inside(&self) -> bool {
        self.open.is_some()
    }

    fn feed(&mut self, line: &str) {
        let bytes = line.as_bytes();
        let mut indent = 0;
        while indent < 3 && bytes.get(indent) == Some(&b' ') {
            indent += 1;
        }
        match self.open {
            Some((marker_char, marker_len)) => {
                let mut j = indent;
                while bytes.get(j) == Some(&marker_char) {
                    j += 1;
                }
                let trailing_blank = bytes[j..]
                    .iter()
                    .all(|c| matches!(c, b' ' | b'\t' | b'\r' | b'\n'));
                if j - indent >= marker_len && trailing_blank {
                    self.open = None;
                }
            }
            None => {
                if let Some(&first @ (b'`' | b'~')) = bytes.get(indent) {
                    let mut j = indent;
                    while bytes.get(j) == Some(&first) {
                        j += 1;
                    }
                    if j - indent >= 3 {
                        self.open = Some((first, j - indent));
                    }
                }
            }
        }
    }
}

// --- source list markers, upstream getOrderedListMarker/getUnorderedListMarker

/// `^(?: {0,3})(\d{1,9}[.)])[ \t]+` over the item's source, upstream
/// `getOrderedListMarker`.
fn get_ordered_list_marker(item_raw: &str) -> Option<String> {
    let bytes = item_raw.as_bytes();
    let mut i = 0;
    while i < 3 && bytes.get(i) == Some(&b' ') {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() && i - digits_start < 9 {
        i += 1;
    }
    if i == digits_start {
        return None;
    }
    match bytes.get(i) {
        Some(b'.' | b')') => i += 1,
        _ => return None,
    }
    let marker_end = i;
    let mut after = i;
    while after < bytes.len() && (bytes[after] == b' ' || bytes[after] == b'\t') {
        after += 1;
    }
    if after == i {
        return None;
    }
    Some(format!("{} ", &item_raw[digits_start..marker_end]))
}

/// `^(?: {0,3})([-+*])(?:[ \t]+|(?=\r?\n|$))` over the item's source,
/// upstream `getUnorderedListMarker`.
fn get_unordered_list_marker(item_raw: &str) -> Option<String> {
    let bytes = item_raw.as_bytes();
    let mut i = 0;
    while i < 3 && bytes.get(i) == Some(&b' ') {
        i += 1;
    }
    match bytes.get(i) {
        Some(b'-' | b'+' | b'*') => i += 1,
        _ => return None,
    }
    let marker_end = i;
    match bytes.get(i) {
        Some(b' ' | b'\t' | b'\r' | b'\n') | None => {
            Some(format!("{} ", &item_raw[marker_end - 1..marker_end]))
        }
        _ => None,
    }
}

// --- marked's gfm autolinks over text spans

static URL_RE: LazyLock<Regex> = LazyLock::new(|| {
    static_regex(r"(?:[hH][tT][tT][pP][sS]?|[fF][tT][pP])://(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*")
});

/// The `www.` alternative of marked's gfm url rule, over the same tail.
static WWW_RE: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"www\.(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*"));

/// The char set of marked's email lookbehind flanking class,
/// `[a-zA-Z0-9.!#$%&'*+/?=\u{60}{|}~-]` (slash, backtick, pipe, tilde, dash).
const fn is_email_flank(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'.' | b'!'
                | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'/'
                | b'='
                | b'?'
                | b'_'
                | b'`'
                | b'{'
                | b'|'
                | b'}'
                | b'~'
                | b'-'
        )
}

/// Try marked's gfm autolink rules at one position: bare `http(s)/ftp` and
/// `www.` URLs (with marked's backpedal), then bare emails. Angle autolinks
/// are core CommonMark that pulldown emits natively.
fn try_autolink_at(pos: usize, span_end: usize, source: &str) -> Option<(usize, &str, String)> {
    let slice = &source[pos..span_end];
    let matched = URL_RE
        .find(slice)
        .filter(|m| m.start() == 0)
        .or_else(|| WWW_RE.find(slice).filter(|m| m.start() == 0));
    if let Some(matched) = matched {
        let text = backpedal(matched.as_str());
        let href = if slice.len() >= 4 && slice[..4].eq_ignore_ascii_case("www.") {
            format!("http://{text}")
        } else {
            text.to_string()
        };
        return Some((pos + text.len(), text, href));
    }
    let bytes = source.as_bytes();
    let prev_ok = pos == 0 || !is_email_flank(bytes[pos - 1]);
    if prev_ok && let Some(end) = scan_email(slice) {
        let email = &slice[..end];
        return Some((pos + end, email, format!("mailto:{email}")));
    }
    None
}

/// Scan a bare email per marked's gfm email rule, with the trailing
/// `(?![-_])` lookahead reproduced by backtracking the match: the match
/// must end inside the final tld group's alphanumeric tail and not be
/// followed by `-` or `_`.
fn scan_email(src: &str) -> Option<usize> {
    const fn is_local(byte: u8) -> bool {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'+' | b'-')
    }
    const fn is_domain(byte: u8) -> bool {
        byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'
    }
    let bytes = src.as_bytes();
    let mut p = 0;
    while p < bytes.len() && is_local(bytes[p]) {
        p += 1;
    }
    if p == 0 || bytes.get(p) != Some(&b'@') {
        return None;
    }
    p += 1;
    let domain_start = p;
    while p < bytes.len() && is_domain(bytes[p]) {
        p += 1;
    }
    if p == domain_start {
        return None;
    }
    let mut groups: Vec<(usize, usize)> = Vec::new();
    while p < bytes.len() && bytes[p] == b'.' {
        let content_start = p + 1;
        let mut r = content_start;
        while r < bytes.len() && is_domain(bytes[r]) {
            r += 1;
        }
        if r == content_start {
            break;
        }
        groups.push((p, r));
        p = r;
    }
    if groups.is_empty() {
        return None;
    }
    // Candidate ends: the last tld group shrinks to any alphanumeric tail
    // before whole trailing groups drop; the longest surviving candidate
    // wins when it is not followed by `-` or `_`.
    let (last_dot, last_max) = groups.last().copied()?;
    let mut e = last_max;
    while e > last_dot + 1 {
        if bytes[e - 1].is_ascii_alphanumeric()
            && !bytes.get(e).is_some_and(|c| *c == b'-' || *c == b'_')
        {
            return Some(e);
        }
        e -= 1;
    }
    None
}

/// Upstream's `_backpedal`: strip trailing punctuation from a matched URL,
/// keeping balanced paren groups and `&` entities.
fn backpedal(url: &str) -> &str {
    const PUNCT_RUN: &[u8] = b"?!.,:;*_\'\"~)";
    const SPECIAL: &[u8] = b"?!.,:;*_\'\"~()&";
    fn is_run(c: u8) -> bool {
        PUNCT_RUN.contains(&c)
    }
    fn is_special(c: u8) -> bool {
        SPECIAL.contains(&c)
    }
    fn entity_tail(rest: &str) -> bool {
        match rest.find(';') {
            Some(i) if i > 0 && i + 1 == rest.len() => {
                rest[..i].bytes().all(|c| c.is_ascii_alphanumeric())
            }
            _ => false,
        }
    }
    let bytes = url.as_bytes();
    let mut pos = 0usize;
    loop {
        if pos >= bytes.len() {
            break;
        }
        let c = bytes[pos];
        if !is_special(c) {
            while pos < bytes.len() && !is_special(bytes[pos]) {
                pos += 1;
            }
        } else if c == b'(' {
            match url[pos + 1..].find(')') {
                Some(offset) => pos += 1 + offset + 1,
                None => break,
            }
        } else if c == b'&' {
            if entity_tail(&url[pos + 1..]) {
                break;
            }
            pos += 1;
        } else {
            // A punctuation run that must not reach the string end.
            let start = pos;
            while pos < bytes.len() && is_run(bytes[pos]) && pos + 1 < bytes.len() {
                pos += 1;
            }
            if pos == start {
                break;
            }
        }
    }
    &url[..pos]
}
