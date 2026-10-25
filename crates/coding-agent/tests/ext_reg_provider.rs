//! `test/suite/regressions/8964-extension-provider-streaming.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use pi_ai::types::Api;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Context;
use pi_ai::types::Message;
use pi_ai::types::Modality;
use pi_ai::types::ProviderId;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::provider_composer::ProviderConfigInput;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::build;
use common::extension_regressions::Session;
use common::extension_regressions::manager;

const PROVIDER: &str = "extension-provider";
const API: &str = "issue-8964-extension-api";

fn provider_config(received_key: Arc<Mutex<Option<String>>>) -> ProviderConfigInput {
    let mut model = common::sdk_fixture::provider_model("faux-1", false, vec![Modality::Text]);
    model.api = Some(Api::from(API));
    ProviderConfigInput {
        name: Some(PROVIDER.to_owned()),
        base_url: Some("https://extension.invalid/v1".to_owned()),
        api_key: Some("extension-key".to_owned()),
        api: Some(Api::from(API)),
        stream_simple: Some(Arc::new(move |_model, _context, options| {
            *received_key.lock().expect("lock") =
                options.and_then(|options| options.api_key.clone());
            let message = AssistantMessage {
                provider: ProviderId(PROVIDER.to_owned()),
                model: "faux-1".to_owned(),
                api: Api::from(API),
                content: vec![AssistantBlock::Text(TextContent {
                    text: "custom provider response".to_owned(),
                    text_signature: None,
                })],
                stop_reason: StopReason::Stop,
                ..assistant_message("", 0)
            };
            let stream = assistant_message_event_stream();
            stream.push(AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            stream.push(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "custom provider response".to_owned(),
                partial: message.clone(),
            });
            stream.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            });
            stream
        })),
        models: Some(vec![model]),
        ..ProviderConfigInput::default()
    }
}

async fn provider_streaming_session(
    root: &str,
    method: &'static str,
    received_key: Arc<Mutex<Option<String>>>,
    streamed: Arc<Mutex<String>>,
    result: Arc<Mutex<Option<AssistantMessage>>>,
) -> Session {
    let provider_extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let config = provider_config(Arc::clone(&received_key));
            Box::pin(async move {
                api.register_provider(PROVIDER.to_owned(), config)
                    .expect("provider registers");
                Ok(())
            })
        }),
    };
    let command_extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let streamed = Arc::clone(&streamed);
            let result = Arc::clone(&result);
            Box::pin(async move {
                api.register_command(
                    "stream-custom".to_owned(),
                    Some("Stream a response from the custom provider".to_owned()),
                    Arc::new(move |_args, ctx| {
                        let streamed = Arc::clone(&streamed);
                        let result = Arc::clone(&result);
                        Box::pin(async move {
                            let registry = ctx.model_registry().expect("registry");
                            let model = registry.find(PROVIDER, "faux-1").expect("provider model");
                            let context = Context {
                                system_prompt: None,
                                messages: vec![Message::User(UserMessage {
                                    content: UserContent::Text("Hello".to_owned()),
                                    timestamp: 0,
                                })],
                                tools: None,
                            };
                            let stream = if method == "stream" {
                                registry.stream(&model, &context, None)
                            } else {
                                registry.stream_simple(&model, &context, None)
                            };
                            let mut text = String::new();
                            while let Some(event) = stream.next().await {
                                if let AssistantMessageEvent::TextDelta { delta, .. } = event {
                                    text.push_str(&delta);
                                }
                            }
                            *streamed.lock().expect("lock") = text;
                            *result.lock().expect("lock") = Some(stream.result().await);
                        })
                    }),
                )
                .expect("command registers");
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![provider_extension, command_extension];
    setup.bind_extensions = true;
    let session = build(setup).await;
    session
        .prompt("/stream-custom".to_owned(), None)
        .await
        .expect("command");
    session
}

fn assert_streamed(
    received: &Arc<Mutex<Option<String>>>,
    streamed: &Arc<Mutex<String>>,
    result: &Arc<Mutex<Option<AssistantMessage>>>,
) {
    assert_eq!(
        received.lock().expect("lock").as_deref(),
        Some("extension-key")
    );
    assert_eq!(*streamed.lock().expect("lock"), "custom provider response");
    let message = result.lock().expect("lock").clone().expect("result");
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        message.content,
        vec![AssistantBlock::Text(TextContent {
            text: "custom provider response".to_owned(),
            text_signature: None,
        })]
    );
}

/// #8964: a command can stream through `ctx.modelRegistry.stream`.
#[tokio::test]
async fn extension_command_streams_through_model_registry_stream() {
    let received = Arc::new(Mutex::new(None));
    let streamed = Arc::new(Mutex::new(String::new()));
    let result = Arc::new(Mutex::new(None));
    let _session = provider_streaming_session(
        "/tmp/pi-ext-reg-provider-stream",
        "stream",
        Arc::clone(&received),
        Arc::clone(&streamed),
        Arc::clone(&result),
    )
    .await;

    assert_streamed(&received, &streamed, &result);
}

/// #8964: the same holds for `ctx.modelRegistry.streamSimple`.
#[tokio::test]
async fn extension_command_streams_through_model_registry_stream_simple() {
    let received = Arc::new(Mutex::new(None));
    let streamed = Arc::new(Mutex::new(String::new()));
    let result = Arc::new(Mutex::new(None));
    let _session = provider_streaming_session(
        "/tmp/pi-ext-reg-provider-stream-simple",
        "streamSimple",
        Arc::clone(&received),
        Arc::clone(&streamed),
        Arc::clone(&result),
    )
    .await;

    assert_streamed(&received, &streamed, &result);
}
