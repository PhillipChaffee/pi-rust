//! Child crashes and stale calls.

use super::*;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::results::HandlerResult;
use crate::extensions::runtime::ContextActions;

#[tokio::test]
async fn a_crashed_child_fails_the_call_and_later_calls_without_writing() {
    let (session, mut child, errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let first_handler = handler(&session, "agent_start");
    let context = test_context(ContextActions::default()).await;
    let first =
        tokio::spawn(async move { first_handler(ExtensionEvent::AgentStart, context).await });
    let Some(Frame::CallEvent { .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call event");
    };
    drop(child);
    assert!(matches!(first.await.expect("join"), HandlerResult::None));
    assert!(session.is_dead());
    let captured = errors.lock().expect("errors").clone();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].error.contains("extension_exited"));
    let second = handler(&session, "agent_start");
    let context = test_context(ContextActions::default()).await;
    assert!(matches!(
        second(ExtensionEvent::AgentStart, context).await,
        HandlerResult::None
    ));
    assert_eq!(errors.lock().expect("errors").len(), 1);
}
