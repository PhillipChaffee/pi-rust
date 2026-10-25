//! The `custom-message.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::CustomAgentMessage;
use pi_coding_agent::extensions::types::{MessageRenderOptions, MessageRenderer};
use pi_coding_agent::modes::interactive::components::custom_message::CustomMessageComponent;
use pi_coding_agent::modes::interactive::theme::theme::{Theme, init_theme};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::components::Text;
use pi_tui::tui::Component;

fn custom_message() -> CustomAgentMessage {
    CustomAgentMessage {
        role: "custom".to_string(),
        timestamp: 0,
        data: serde_json::Map::from_iter([
            ("customType".to_string(), serde_json::json!("test")),
            ("content".to_string(), serde_json::json!("custom")),
            ("display".to_string(), serde_json::json!(true)),
        ]),
    }
}

#[test]
fn provides_output_padding_to_custom_renderers_and_updates_it() {
    init_theme(Some("dark"), false);
    let options_seen = Arc::new(std::sync::Mutex::new(Vec::<MessageRenderOptions>::new()));
    let seen = Arc::clone(&options_seen);
    let renderer: MessageRenderer = Arc::new(move |_message, options, _theme: &Theme| {
        seen.lock().expect("lock").push(*options);
        Some(Rc::new(Text::with_padding("custom", options.output_pad, 0)))
    });
    let component = CustomMessageComponent::new(custom_message(), Some(renderer), None, 1);

    assert_eq!(
        *options_seen.lock().expect("lock"),
        vec![MessageRenderOptions {
            expanded: false,
            output_pad: 1,
        }]
    );
    assert!(
        component
            .render(40)
            .into_iter()
            .map(|line| strip_ansi(&line))
            .any(|line| line.starts_with(" custom"))
    );

    component.set_output_pad(0);

    assert_eq!(
        *options_seen.lock().expect("lock"),
        vec![
            MessageRenderOptions {
                expanded: false,
                output_pad: 1,
            },
            MessageRenderOptions {
                expanded: false,
                output_pad: 0,
            },
        ]
    );
    assert!(
        component
            .render(40)
            .into_iter()
            .map(|line| strip_ansi(&line))
            .any(|line| line.starts_with("custom"))
    );
}
