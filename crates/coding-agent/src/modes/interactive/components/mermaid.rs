//! Mermaid fences to Unicode diagrams, upstream
//! `src/modes/interactive/components/mermaid.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatement: upstream renders through `grok-mermaid`'s full layout engine.
//! The port carries a terminal renderer for the `flowchart LR` subset the
//! suite pins — chains of box/rounded/diamond nodes joined by a link — and
//! leaves every other diagram kind, direction, or branch routing as its raw
//! fence. Unsupported and oversized diagrams degrade exactly like upstream:
//! the fence stays in place.

use std::sync::Arc;

use crate::extensions::types::{MarkdownTransformContext, MarkdownTransformer};
use crate::modes::interactive::theme::theme::Theme;
use crate::settings_manager::MermaidRenderingMode;

/// One styled run of a diagram row, upstream's `Span`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// The semantic class, upstream's `cls`.
    pub cls: &'static str,
    /// The run's text.
    pub text: String,
}

/// One parsed flowchart before layout.
#[derive(Default)]
struct Flowchart {
    labels: Vec<String>,
    names: Vec<String>,
    warnings: Vec<String>,
}

/// A rendered diagram, upstream's `MermaidArt`.
#[derive(Debug)]
pub struct MermaidArt {
    /// The uncolored rows.
    pub plain: Vec<String>,
    /// The semantic spans per row.
    pub styled: Vec<Vec<Span>>,
    /// The widest row.
    pub width: usize,
    /// Parse warnings; non-empty keeps the fence in place.
    pub warnings: Vec<String>,
}

impl MermaidArt {
    /// Whether the art is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.plain.is_empty()
    }
}

/// A Markdown segment: prose or one fenced code block.
enum Segment<'a> {
    Markdown(&'a str),
    Mermaid { raw: &'a str, body: &'a str },
}

/// Split the source into prose and mermaid fences; every other fence stays in
/// the prose segment.
fn split_mermaid_fences(markdown: &str) -> Vec<Segment<'_>> {
    let mut segments = Vec::new();
    let mut prose_start = 0;
    let mut search_from = 0;
    let bytes = markdown.as_bytes();
    while search_from < bytes.len() {
        let Some(open) = markdown[search_from..].find("```") else {
            break;
        };
        let open = search_from + open;
        let header_end = markdown[open..]
            .find('\n')
            .map_or(markdown.len(), |offset| open + offset);
        let header = &markdown[open + 3..header_end];
        let language = header.split_whitespace().next().unwrap_or_default();
        if language != "mermaid" {
            // Leave the block in the prose segment; skip past its closing
            // fence so its contents are not scanned for diagrams.
            let body_start = header_end.min(markdown.len()).saturating_add(1);
            let Some(close_offset) = markdown[body_start..].find("\n```") else {
                break;
            };
            search_from = (body_start + close_offset + 4).min(markdown.len());
            continue;
        }
        if open > prose_start {
            segments.push(Segment::Markdown(&markdown[prose_start..open]));
        }
        let body_start = header_end.min(markdown.len()).saturating_add(1);
        let Some(close_offset) = markdown[body_start..].find("\n```") else {
            segments.push(Segment::Mermaid {
                raw: &markdown[open..],
                body: &markdown[body_start.min(markdown.len())..],
            });
            prose_start = markdown.len();
            break;
        };
        let close = body_start + close_offset;
        let after_close = (close + 4).min(markdown.len());
        let raw_end = if markdown[after_close..].starts_with('\n') {
            after_close + 1
        } else {
            after_close
        };
        segments.push(Segment::Mermaid {
            raw: &markdown[open..raw_end],
            body: &markdown[body_start..close],
        });
        prose_start = raw_end;
        search_from = raw_end;
    }
    if prose_start < markdown.len() {
        segments.push(Segment::Markdown(&markdown[prose_start..]));
    }
    segments
}

/// Parse a `flowchart LR` chain, collecting unsupported statements as
/// warnings.
fn parse_flowchart(body: &str) -> Option<Flowchart> {
    let mut lines = body.lines().filter(|line| {
        let line = line.trim();
        !line.is_empty() && !line.starts_with("%%")
    });
    let header = lines.next()?;
    let mut header_parts = header.split_whitespace();
    let kind = header_parts.next()?;
    if kind != "flowchart" && kind != "graph" {
        return None;
    }
    if header_parts.next()? != "LR" {
        return None;
    }

    let mut flowchart = Flowchart::default();
    for line in lines {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let mut index = 0;
        while index < tokens.len() {
            let Some((name, label)) = parse_node(tokens[index]) else {
                flowchart
                    .warnings
                    .push(format!("dropped, expected a link: \"{}\"", tokens[index]));
                index += 1;
                continue;
            };
            if !flowchart
                .names
                .iter()
                .any(|existing| existing.as_str() == name)
            {
                flowchart.names.push(name);
                flowchart.labels.push(label);
            }
            index += 1;
            if index >= tokens.len() {
                break;
            }
            if !is_link(tokens[index]) {
                flowchart
                    .warnings
                    .push(format!("dropped, expected a link: \"{}\"", tokens[index]));
                break;
            }
            index += 1;
            if index >= tokens.len() {
                break;
            }
            let Some((target_name, target_label)) = parse_node(tokens[index]) else {
                flowchart
                    .warnings
                    .push(format!("dropped, expected a link: \"{}\"", tokens[index]));
                break;
            };
            if !flowchart
                .names
                .iter()
                .any(|existing| existing.as_str() == target_name)
            {
                flowchart.names.push(target_name);
                flowchart.labels.push(target_label);
            }
            index += 1;
        }
    }
    Some(flowchart)
}

/// Parse one node token into its name and label; the shape characters decide
/// where the label sits, and an unterminated shape keeps the remaining text.
fn parse_node(token: &str) -> Option<(String, String)> {
    let token = token.split(":::").next().unwrap_or(token);
    if token.is_empty() {
        return None;
    }
    let shape = token.find(['[', '(', '{']);
    let Some(shape) = shape else {
        if token.starts_with(['-', '=', '<']) || token.starts_with('|') {
            return None;
        }
        return Some((token.to_string(), token.to_string()));
    };
    let name = &token[..shape];
    if name.is_empty() {
        return None;
    }
    let opener = token.as_bytes()[shape] as char;
    let closer = match opener {
        '[' => ']',
        '(' => ')',
        _ => '}',
    };
    let rest = &token[shape + 1..];
    let label = rest
        .find(closer)
        .map_or_else(|| rest.to_string(), |end| rest[..end].to_string());
    Some((name.to_string(), label))
}

/// Whether a token is a link operator.
fn is_link(token: &str) -> bool {
    (token.starts_with('-') || token.starts_with('='))
        && (token.contains('>') || token.contains("---"))
}

/// Lay a flowchart out left to right as one chain, upstream's flow layout
/// for the subset the port carries.
fn render_flowchart(flowchart: &Flowchart) -> MermaidArt {
    if flowchart.labels.is_empty() {
        return MermaidArt {
            plain: Vec::new(),
            styled: Vec::new(),
            width: 0,
            warnings: flowchart.warnings.clone(),
        };
    }
    let inner_widths: Vec<usize> = flowchart
        .labels
        .iter()
        .map(|label| label.chars().count() + 2)
        .collect();
    let gap = 4;
    let width = inner_widths.iter().map(|inner| inner + 2).sum::<usize>()
        + gap * flowchart.labels.len().saturating_sub(1);

    let top = row_with_boxes(&inner_widths, '┌', '─', '┐', gap);
    let bottom = row_with_boxes(&inner_widths, '└', '─', '┘', gap);
    let mut middle = Vec::new();
    for (index, label) in flowchart.labels.iter().enumerate() {
        middle.push(Span {
            cls: "border",
            text: "│".to_string(),
        });
        middle.push(Span {
            cls: "text",
            text: format!(" {label} "),
        });
        if index + 1 < flowchart.labels.len() {
            middle.push(Span {
                cls: "border",
                text: "├".to_string(),
            });
            middle.push(Span {
                cls: "edge",
                text: "───▶".to_string(),
            });
        } else {
            middle.push(Span {
                cls: "border",
                text: "│".to_string(),
            });
        }
    }

    let styled = vec![top, middle, bottom];
    let plain = styled
        .iter()
        .map(|row| {
            row.iter()
                .map(|span| span.text.as_str())
                .collect::<String>()
        })
        .collect();
    MermaidArt {
        plain,
        styled,
        width,
        warnings: flowchart.warnings.clone(),
    }
}

/// A border row with one box per node and plain gaps between them.
fn row_with_boxes(
    inner_widths: &[usize],
    left: char,
    fill: char,
    right: char,
    gap: usize,
) -> Vec<Span> {
    let mut row = Vec::new();
    for (index, inner) in inner_widths.iter().enumerate() {
        row.push(Span {
            cls: "border",
            text: format!("{left}{}{right}", fill.to_string().repeat(*inner)),
        });
        if index + 1 < inner_widths.len() {
            row.push(Span {
                cls: "none",
                text: " ".repeat(gap),
            });
        }
    }
    row
}

/// Encode one diagram row as an inline code span so Markdown preserves its
/// spacing and box drawing, upstream's `codeSpan`.
fn code_span(line: &str) -> String {
    let content = if line.is_empty() { "\u{a0}" } else { line };
    let mut longest_backtick_run = 0;
    let mut current_run = 0;
    for character in content.chars() {
        if character == '`' {
            current_run += 1;
            longest_backtick_run = longest_backtick_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    let fence = "`".repeat(longest_backtick_run + 1);
    let padding = if content.starts_with('`') || content.ends_with('`') {
        " "
    } else {
        ""
    };
    format!("{fence}{padding}{content}{padding}{fence}")
}

/// Map a span's semantic class through the active theme.
fn style_span(span: &Span, theme: &Theme) -> String {
    match span.cls {
        "border" => theme.fg("borderMuted", &span.text),
        "text" => theme.fg("text", &span.text),
        "edge" => theme.fg("accent", &span.text),
        "edgeLabel" => theme.fg("muted", &span.text),
        "title" => theme.fg(
            "accent",
            &crate::modes::interactive::theme::theme::bold(&span.text),
        ),
        _ => span.text.clone(),
    }
}

/// Replace the mermaid fences in `markdown`, upstream's transformer body.
fn transform_markdown(
    markdown: &str,
    context: &MarkdownTransformContext,
    mode: MermaidRenderingMode,
    theme: Option<&Theme>,
) -> String {
    if mode == MermaidRenderingMode::Off
        || context.message_type == crate::extensions::types::MarkdownMessageType::AssistantThinking
    {
        return markdown.to_string();
    }
    if context.is_streaming && mode != MermaidRenderingMode::Streaming {
        return markdown.to_string();
    }

    let mut output = String::new();
    for segment in split_mermaid_fences(markdown) {
        match segment {
            Segment::Markdown(text) => output.push_str(text),
            Segment::Mermaid { raw, body } => {
                let Some(flowchart) = parse_flowchart(body) else {
                    output.push_str(raw);
                    continue;
                };
                let art = render_flowchart(&flowchart);
                if art.is_empty() || art.width > context.available_width {
                    output.push_str(raw);
                    continue;
                }
                if !context.is_streaming && !art.warnings.is_empty() {
                    let suffix = if art.warnings.len() > 1 {
                        format!(" (+{} more)", art.warnings.len() - 1)
                    } else {
                        String::new()
                    };
                    let warning =
                        format!("Mermaid diagram not rendered: {}{suffix}", art.warnings[0]);
                    let styled_warning = theme
                        .map_or_else(|| warning.clone(), |theme| theme.fg("warning", &warning));
                    output.push_str(raw);
                    output.push('\n');
                    output.push_str(&code_span(&styled_warning));
                    output.push_str("  \n");
                    continue;
                }
                let lines: Vec<String> = if let Some(theme) = theme {
                    art.styled
                        .iter()
                        .map(|row| row.iter().map(|span| style_span(span, theme)).collect())
                        .collect()
                } else {
                    art.plain.clone()
                };
                output.push_str(
                    &lines
                        .iter()
                        .map(|line| code_span(line))
                        .collect::<Vec<_>>()
                        .join("  \n"),
                );
                output.push('\n');
            }
        }
    }
    output
}

/// Create the transformer that replaces top-level Mermaid fences with Unicode
/// terminal diagrams, upstream's `createMermaidMarkdownTransformer`.
#[must_use]
pub fn create_mermaid_markdown_transformer(
    get_mode: Arc<dyn Fn() -> MermaidRenderingMode + Send + Sync>,
    theme: Option<Arc<Theme>>,
) -> MarkdownTransformer {
    Arc::new(move |markdown, context| {
        transform_markdown(markdown, context, get_mode(), theme.as_deref())
    })
}
