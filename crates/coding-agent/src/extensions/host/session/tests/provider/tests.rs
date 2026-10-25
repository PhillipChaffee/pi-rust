//! Provider stream proxies.

use std::sync::Arc;

use pi_ai::types::{AssistantMessageEvent, StopReason};

use super::*;
use crate::extensions::runtime::ContextActions;

use support::{assistant_message, demo_provider};

async fn provider_session() -> (
    HostSession<DuplexStream>,
    Link<DuplexStream>,
    Arc<Mutex<Vec<ExtensionError>>>,
) {
    ready_session_with(vec![demo_provider()], ContextActions::default(), None).await
}

#[tokio::test]
async fn provider_streams_relay_events() {
    let (mut session, mut child, _errors) = provider_session().await;
    let (stream, model, call) = support::start_provider(&mut session, &mut child).await;
    let message = assistant_message(&model);
    child
        .write_frame(&Frame::StreamEvent {
            call,
            event: serde_json::to_value(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: message.clone(),
            })
            .expect("event"),
        })
        .await
        .expect("stream event");
    child
        .write_frame(&Frame::StreamEnd { call })
        .await
        .expect("stream end");
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Done { .. })
    ));
    assert_eq!(stream.result().await, message);
    assert_eq!(stream.next().await, None);
}

#[tokio::test]
async fn provider_failures_end_the_stream_with_an_error() {
    let (mut session, mut child, _errors) = provider_session().await;
    let (stream, _model, call) = support::start_provider(&mut session, &mut child).await;
    child
        .write_frame(&Frame::CallError {
            call,
            code: "provider_failed".to_string(),
            message: "down".to_string(),
            stack: None,
        })
        .await
        .expect("failure");
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Error { .. })
    ));
    assert_eq!(
        stream.result().await.error_message.as_deref(),
        Some("provider_failed: down")
    );
}
