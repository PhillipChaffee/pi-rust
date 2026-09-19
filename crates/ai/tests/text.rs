//! The `contentText` port, from `test/text.test.ts`.

mod common;

use pi_ai::types::{
    AssistantBlock, ImageContent, TextContent, ThinkingContent, ToolCall, ToolResultBlock,
    UserBlock, UserContent,
};
use pi_ai::utils::text::content_text;

fn assistant_content() -> Vec<AssistantBlock> {
    vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: String::from("reasoning"),
            thinking_signature: None,
            redacted: None,
        }),
        AssistantBlock::Text(TextContent {
            text: String::from("first"),
            text_signature: None,
        }),
        AssistantBlock::ToolCall(ToolCall {
            id: String::from("1"),
            name: String::from("read"),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        }),
        AssistantBlock::Text(TextContent {
            text: String::from("second"),
            text_signature: None,
        }),
    ]
}

#[test]
fn extracts_assistant_text_blocks() {
    let content = assistant_content();
    assert_eq!(content_text(&content.as_slice()), "first\nsecond");
}

#[test]
fn supports_custom_separators() {
    let content = assistant_content();
    assert_eq!(
        pi_ai::utils::text::content_text_with(&content.as_slice(), ""),
        "firstsecond"
    );
}

#[test]
fn passes_string_content_through() {
    assert_eq![content_text(&String::from("hello")), "hello"];
}

#[test]
fn extracts_text_from_tool_result_content() {
    let content = vec![
        ToolResultBlock::Text(TextContent {
            text: String::from("first"),
            text_signature: None,
        }),
        ToolResultBlock::Image(ImageContent {
            data: String::from("..."),
            mime_type: String::from("image/png"),
        }),
        ToolResultBlock::Text(TextContent {
            text: String::from("second"),
            text_signature: None,
        }),
    ];

    assert_eq!(
        pi_ai::utils::text::content_text_with(&content.as_slice(), ""),
        "firstsecond"
    );
}

#[test]
fn a_tool_result_image_block_is_skipped_like_any_non_text_block() {
    // Covers the assistant-side arm symmetrically: thinking blocks are
    // skipped too.
    let content = vec![AssistantBlock::Thinking(ThinkingContent {
        thinking: String::from("reasoning"),
        thinking_signature: None,
        redacted: None,
    })];
    assert_eq!(content_text(&content.as_slice()), "");
}

#[test]
fn a_plain_str_is_its_own_text() {
    // The `str` impl is unsized, so the generic helpers (whose parameter
    // type is sized) reach it only through a ?Sized reader.
    fn text_parts_of<T: pi_ai::utils::text::ContentTextSource + ?Sized>(content: &T) -> Vec<&str> {
        content.text_parts()
    }
    assert_eq![text_parts_of("plain"), vec!["plain"]];
}

#[test]
fn user_blocks_keep_only_their_text_parts() {
    let blocks = vec![
        UserBlock::Text(TextContent {
            text: String::from("first"),
            text_signature: None,
        }),
        UserBlock::Image(ImageContent {
            data: String::from("..."),
            mime_type: String::from("image/png"),
        }),
        UserBlock::Text(TextContent {
            text: String::from("second"),
            text_signature: None,
        }),
    ];
    assert_eq!(content_text(&blocks.as_slice()), "first\nsecond");

    let only_image = vec![UserBlock::Image(ImageContent {
        data: String::from("..."),
        mime_type: String::from("image/png"),
    })];
    assert_eq!(content_text(&only_image.as_slice()), "");
}

#[test]
fn user_content_reads_both_the_string_and_block_forms() {
    assert_eq!(
        content_text(&UserContent::Text(String::from("hello"))),
        "hello"
    );
    assert_eq!(
        content_text(&UserContent::Blocks(vec![
            UserBlock::Text(TextContent {
                text: String::from("first"),
                text_signature: None,
            }),
            UserBlock::Image(ImageContent {
                data: String::from("..."),
                mime_type: String::from("image/png"),
            }),
            UserBlock::Text(TextContent {
                text: String::from("second"),
                text_signature: None,
            }),
        ])),
        "first\nsecond"
    );
}
