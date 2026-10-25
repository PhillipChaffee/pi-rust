#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;
use pi_ai::types::Api;
use pi_ai::types::AssistantMessage;
use pi_ai::types::ProviderId;
use pi_ai::types::TextContent;
use pi_ai::types::ThinkingContent;
use pi_ai::types::Usage;

fn assistant_message(
    content: Vec<AssistantBlock>,
    stop_reason: StopReason,
    error_message: Option<&str>,
) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content,
        api: Api::from("test"),
        provider: ProviderId("test".to_owned()),
        model: "test-model".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason,
        deferred: None,
        error_message: error_message.map(str::to_owned),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }))
}

fn text_block(text: &str) -> AssistantBlock {
    AssistantBlock::Text(TextContent {
        text: text.to_owned(),
        text_signature: None,
    })
}

#[test]
fn text_mode_output_emits_assistant_text_blocks_in_order() {
    let message = assistant_message(
        vec![
            text_block("first"),
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "hidden".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            text_block("second"),
        ],
        StopReason::Stop,
        None,
    );

    assert_eq!(
        text_mode_output(&message),
        Some(Ok(vec!["first".to_owned(), "second".to_owned()]))
    );
}

#[test]
fn text_mode_output_reports_the_error_message() {
    let message = assistant_message(Vec::new(), StopReason::Error, Some("provider failure"));

    assert_eq!(
        text_mode_output(&message),
        Some(Err("provider failure".to_owned()))
    );
}

#[test]
fn text_mode_output_falls_back_to_the_stop_reason() {
    assert_eq!(
        text_mode_output(&assistant_message(Vec::new(), StopReason::Error, None)),
        Some(Err("Request error".to_owned()))
    );
    assert_eq!(
        text_mode_output(&assistant_message(Vec::new(), StopReason::Aborted, None)),
        Some(Err("Request aborted".to_owned()))
    );
}

#[test]
fn text_mode_output_ignores_non_assistant_messages() {
    let message = AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text("hello".to_owned()),
        timestamp: 1,
    }));

    assert_eq!(text_mode_output(&message), None);
}

#[tokio::test]
async fn prompt_or_signal_prefers_an_armed_signal() {
    let (sender, mut receiver) = mpsc::unbounded_channel();
    sender.send(143).unwrap();

    let outcome = prompt_or_signal(&mut receiver, async { Ok(()) }).await;

    assert!(matches!(outcome, PromptOutcome::Signal(143)));
}
