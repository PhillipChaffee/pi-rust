//! GitHub Copilot request headers, ported from
//! `packages/ai/src/api/github-copilot-headers.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` through the conformance
//! suites that drive it; this file pins the module shapes directly.

use pi_ai::api::github_copilot_headers::{
    build_copilot_dynamic_headers, has_copilot_vision_input, infer_copilot_initiator,
};
use pi_ai::types::{
    ImageContent, Message, ToolResultBlock, ToolResultMessage, UserContent, UserMessage,
};

fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: 1,
    })
}

fn image_result() -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: "call_1".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![ToolResultBlock::Image(ImageContent {
            data: "aGVsbG8=".to_owned(),
            mime_type: "image/png".to_owned(),
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    })
}

#[test]
fn the_initiator_reads_user_and_agent_from_the_last_message() {
    assert_eq!(infer_copilot_initiator(&[]), "user");
    assert_eq!(infer_copilot_initiator(&[user_message("hi")]), "user");
    assert_eq!(infer_copilot_initiator(&[image_result()]), "agent");
}

#[test]
fn vision_input_rides_user_blocks_and_tool_result_images() {
    assert!(!has_copilot_vision_input(&[user_message("text only")]));
    let image_user = Message::User(UserMessage {
        content: UserContent::Blocks(vec![pi_ai::types::UserBlock::Image(ImageContent {
            data: "aGVsbG8=".to_owned(),
            mime_type: "image/png".to_owned(),
        })]),
        timestamp: 1,
    });
    assert!(has_copilot_vision_input(&[image_user]));
    assert!(has_copilot_vision_input(&[image_result()]));
}

#[test]
fn dynamic_headers_carry_the_intent_and_the_vision_flag() {
    let plain = build_copilot_dynamic_headers(&[user_message("hi")], false);
    assert_eq!(plain.get("X-Initiator").map(String::as_str), Some("user"));
    assert_eq!(
        plain.get("Openai-Intent").map(String::as_str),
        Some("conversation-edits"),
    );
    assert!(!plain.contains_key("Copilot-Vision-Request"));

    let with_images = build_copilot_dynamic_headers(&[image_result()], true);
    assert_eq!(
        with_images
            .get("Copilot-Vision-Request")
            .map(String::as_str),
        Some("true"),
    );
}
