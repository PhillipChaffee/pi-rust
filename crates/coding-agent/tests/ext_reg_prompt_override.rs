//! `test/suite/regressions/6162-extension-active-tools-next-turn.test.ts`,
//! the system-prompt override block, at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_agent_core::types::StreamFn;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::BeforeAgentStartResult;
use pi_coding_agent::extensions::results::HandlerResult;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::manager;
use common::extension_regressions::switching_tools_extension;
use serde_json::json;

/// #6162: a `before_agent_start` system-prompt override survives a
/// mid-run tool change.
#[tokio::test]
async fn system_prompt_override_survives_mid_run_tool_changes() {
    let root = "/tmp/pi-ext-reg-prompt-override";
    let prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&prompts);
    let request = Arc::new(AtomicUsize::new(0));
    let stream: StreamFn = Arc::new(move |_model, context, _options| {
        captured
            .lock()
            .expect("lock")
            .push(context.system_prompt.clone().unwrap_or_default());
        let message = if request.fetch_add(1, Ordering::SeqCst) == 0 {
            assistant_with_calls(&[("call-switch", "switch_tools", json!({}))], "")
        } else {
            assistant_message("done", 0)
        };
        let stream = assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message,
        });
        stream
    });
    let override_extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.on(
                    "before_agent_start",
                    Arc::new(|event: ExtensionEvent, _ctx| {
                        let ExtensionEvent::BeforeAgentStart { system_prompt, .. } = event else {
                            return Box::pin(async { HandlerResult::None });
                        };
                        Box::pin(async move {
                            HandlerResult::BeforeAgentStart(BeforeAgentStartResult {
                                message: None,
                                system_prompt: Some(format!(
                                    "{system_prompt}\n\nkeep this run override"
                                )),
                            })
                        })
                    }),
                )
                .expect("before_agent_start handler");
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![
        switching_tools_extension(Arc::new(Mutex::new(None))),
        override_extension,
    ];
    setup.bind_extensions = true;
    setup.stream = Some(stream);
    let session = build(setup).await;
    authorize(&session).await;

    session.set_active_tools_by_name(&["switch_tools".to_owned()]);
    session
        .prompt("start".to_owned(), None)
        .await
        .expect("prompt");

    let prompts = prompts.lock().expect("lock").clone();
    assert_eq!(prompts.len(), 2);
    assert!(prompts[0].contains("keep this run override"));
    assert!(prompts[1].contains("keep this run override"));
}
