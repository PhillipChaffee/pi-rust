//! The `mermaid.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported against the port's
//! `flowchart LR` restatement.

use std::sync::Arc;

use pi_coding_agent::extensions::types::{
    MarkdownMessageType, MarkdownTransformContext, MarkdownTransformer,
};
use pi_coding_agent::modes::interactive::components::mermaid::create_mermaid_markdown_transformer;
use pi_coding_agent::modes::interactive::theme::theme::{init_theme, theme};
use pi_coding_agent::settings_manager::MermaidRenderingMode;

struct TransformOptions {
    max_width: usize,
    is_streaming: bool,
    message_type: MarkdownMessageType,
    mode: MermaidRenderingMode,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            max_width: 100,
            is_streaming: false,
            message_type: MarkdownMessageType::Assistant,
            mode: MermaidRenderingMode::Streaming,
        }
    }
}

fn transform_mermaid(markdown: &str, options: &TransformOptions) -> String {
    let mode = options.mode;
    let transformer: MarkdownTransformer =
        create_mermaid_markdown_transformer(Arc::new(move || mode), None);
    transformer(
        markdown,
        &MarkdownTransformContext {
            message_type: options.message_type,
            is_streaming: options.is_streaming,
            available_width: options.max_width,
        },
    )
}

fn styled_transform_mermaid(markdown: &str) -> String {
    init_theme(Some("dark"), false);
    let transformer: MarkdownTransformer = create_mermaid_markdown_transformer(
        Arc::new(|| MermaidRenderingMode::Streaming),
        Some(theme()),
    );
    transformer(
        markdown,
        &MarkdownTransformContext {
            message_type: MarkdownMessageType::Assistant,
            is_streaming: false,
            available_width: 100,
        },
    )
}

#[test]
fn replaces_mermaid_code_blocks_with_unicode_diagrams() {
    let markdown = "Before\n\n```mermaid\nflowchart LR\n  A[Start] --> B[Done]\n```\nAfter";
    let rendered = transform_mermaid(markdown, &TransformOptions::default());

    assert!(rendered.contains("Before"));
    assert!(rendered.contains("┌───────┐"));
    assert!(rendered.contains("│ Start ├───▶│ Done │"));
    assert!(rendered.contains("└───────┘    └──────┘`\nAfter"));
    assert!(!rendered.contains("```mermaid"));
    assert!(rendered.contains("After"));
}

#[test]
fn leaves_unsupported_and_oversized_diagrams_unchanged() {
    let unsupported = "```mermaid\npie\n  title Pets\n  \"Dogs\" : 4\n```";
    let oversized = "```mermaid\nflowchart LR\n  A[Start] --> B[Done]\n```";

    assert_eq!(
        transform_mermaid(unsupported, &TransformOptions::default()),
        unsupported
    );
    assert_eq!(
        transform_mermaid(
            oversized,
            &TransformOptions {
                max_width: 10,
                ..TransformOptions::default()
            }
        ),
        oversized
    );
}

#[test]
fn maps_semantic_spans_through_the_pi_theme() {
    let rendered = styled_transform_mermaid("```mermaid\nflowchart LR\n  A --> B\n```");
    let active = theme();
    assert!(rendered.contains(active.get_fg_ansi("borderMuted")));
    assert!(rendered.contains(active.get_fg_ansi("accent")));
}

#[test]
fn renders_incomplete_mermaid_blocks_during_streaming() {
    let partial = "```mermaid\nflowchart LR\n  A --> B";
    let rendered = transform_mermaid(
        partial,
        &TransformOptions {
            is_streaming: true,
            ..TransformOptions::default()
        },
    );
    assert!(rendered.contains("───▶"));
}

#[test]
fn renders_diagrams_with_class_assignments() {
    let markdown = "```mermaid\nflowchart LR\n  A[Foo]:::highlight --> B[Bar]\n```";
    let rendered = transform_mermaid(markdown, &TransformOptions::default());

    assert!(!rendered.contains("```mermaid"));
    assert!(!rendered.contains("Mermaid diagram not rendered"));
    assert!(rendered.contains("│ Foo ├───▶│ Bar │"));
}

#[test]
fn falls_back_to_the_code_block_with_a_warning_after_streaming() {
    let markdown = "```mermaid\nflowchart LR\n  A[Foo] invalid\n```";
    let final_render = transform_mermaid(markdown, &TransformOptions::default());
    let followed_by_text = transform_mermaid(
        &format!("{markdown}\nFollowing text"),
        &TransformOptions::default(),
    );
    let streaming = transform_mermaid(
        markdown,
        &TransformOptions {
            is_streaming: true,
            ..TransformOptions::default()
        },
    );

    assert!(final_render.contains(markdown));
    assert!(final_render.contains("```\n`Mermaid diagram not rendered"));
    assert!(final_render.contains("dropped, expected a link: \"invalid\""));
    assert!(!final_render.contains("more)"));
    assert!(followed_by_text.contains("  \nFollowing text"));
    assert!(!streaming.contains("Mermaid diagram not rendered"));
    assert!(!streaming.contains("```mermaid"));
    assert!(streaming.contains("│ Foo │"));
}

#[test]
fn summarizes_additional_partial_render_warnings() {
    let markdown = "```mermaid\nflowchart LR\n  A[Foo] invalid\n  B[Bar] also-invalid\n```";
    let rendered = transform_mermaid(markdown, &TransformOptions::default());

    assert!(rendered.contains(markdown));
    assert!(rendered.contains("dropped, expected a link: \"invalid\""));
    assert!(rendered.contains("(+1 more)"));
    assert!(!rendered.contains("dropped, expected a link: \"also-invalid\""));
}

#[test]
fn respects_rendering_modes_and_skips_thinking_blocks() {
    let markdown = "```mermaid\nflowchart LR\n  A --> B\n```";
    let options =
        |mode: MermaidRenderingMode, is_streaming: bool, message_type: MarkdownMessageType| {
            TransformOptions {
                is_streaming,
                message_type,
                mode,
                ..TransformOptions::default()
            }
        };

    assert_eq!(
        transform_mermaid(
            markdown,
            &options(
                MermaidRenderingMode::Off,
                false,
                MarkdownMessageType::Assistant
            )
        ),
        markdown
    );
    assert_eq!(
        transform_mermaid(
            markdown,
            &options(
                MermaidRenderingMode::Final,
                true,
                MarkdownMessageType::Assistant
            )
        ),
        markdown
    );
    assert!(
        !transform_mermaid(
            markdown,
            &options(
                MermaidRenderingMode::Final,
                false,
                MarkdownMessageType::Assistant
            )
        )
        .contains("```mermaid")
    );
    assert_eq!(
        transform_mermaid(
            markdown,
            &options(
                MermaidRenderingMode::Streaming,
                false,
                MarkdownMessageType::AssistantThinking
            )
        ),
        markdown
    );
}

#[test]
fn styles_warnings_and_leaves_non_mermaid_fences_alone() {
    init_theme(Some("dark"), false);
    let transformer: MarkdownTransformer = create_mermaid_markdown_transformer(
        Arc::new(|| MermaidRenderingMode::Streaming),
        Some(theme()),
    );
    let rendered = transformer(
        "```mermaid\nflowchart LR\n  A[Foo] invalid\n```",
        &MarkdownTransformContext {
            message_type: MarkdownMessageType::Assistant,
            is_streaming: false,
            available_width: 100,
        },
    );
    assert!(rendered.contains(theme().get_fg_ansi("warning")));

    let mixed = transform_mermaid(
        "```rust\nfn x() {}\n```\n\n```mermaid\nflowchart LR\n  A --> B\n```",
        &TransformOptions::default(),
    );
    assert!(mixed.contains("fn x() {}"));
    assert!(mixed.contains("───▶"));
}
