//! Wire-shape tests for the core types: every message, event, and model shape
//! round-trips through serde with the exact upstream field names, and the
//! id newtypes keep their open-string behavior. These are the verification
//! the upstream `types.ts` suite has no standalone file for.

#![allow(
    clippy::expect_used,
    reason = "fixture-heavy wire assertions use expect's failure message as the assertion text; a panic names the broken shape"
)]

use std::collections::BTreeMap;

use pi_ai::types::{
    AssistantBlock, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    CacheRetention, ChatTemplateKwargValue, ConstrainedSamplingSetting, Context, DeferredRequest,
    DeferredWindow, DiagnosticErrorInfo, GrammarFormat, ImageContent, ImagesStopReason, KnownApi,
    KnownProvider, Message, Modality, Model, ModelCompat, StopReason, TextContent, ThinkingLevel,
    ThinkingTemplateVar, Tool, Transport, UserBlock, UserContent, UserMessage,
};

const fn seed_usage() -> pi_ai::types::Usage {
    pi_ai::types::Usage {
        input: 100,
        output: 40,
        cache_read: 20,
        cache_write: 10,
        cache_write_1h: Some(5),
        reasoning: Some(15),
        total_tokens: 140,
        cost: pi_ai::types::UsageCost {
            input: 0.3,
            output: 0.12,
            cache_read: 0.02,
            cache_write: 0.05,
            total: 0.47,
        },
    }
}

fn seed_assistant() -> AssistantMessage {
    AssistantMessage {
        content: vec![
            AssistantBlock::Thinking(pi_ai::types::ThinkingContent {
                thinking: "ponder".to_string(),
                thinking_signature: Some("sig".to_string()),
                redacted: Some(false),
            }),
            AssistantBlock::Text(TextContent {
                text: "hello".to_string(),
                text_signature: None,
            }),
            AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                arguments: serde_json::Map::new(),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: KnownApi::AnthropicMessages.into(),
        provider: KnownProvider::Anthropic.into(),
        model: "claude-test".to_string(),
        response_model: None,
        response_id: Some("resp-1".to_string()),
        provider_thinking_level: Some("high".to_string()),
        diagnostics: None,
        usage: seed_usage(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: Some("tool_use".to_string()),
        end_turn: Some(true),
        timestamp: 1_710_000_000_000,
    }
}

#[test]
fn assistant_message_round_trips_with_exact_wire_field_names() {
    let message = seed_assistant();
    let wire = serde_json::to_value(&message).expect("message serializes");
    let object = wire.as_object().expect("assistant message is an object");
    // The `role` tag belongs to the `Message` envelope, not the bare struct.
    for name in [
        "content",
        "api",
        "provider",
        "model",
        "responseId",
        "providerThinkingLevel",
        "usage",
        "stopReason",
        "rawStopReason",
        "endTurn",
        "timestamp",
    ] {
        assert!(object.contains_key(name), "missing wire field {name}");
    }
    // The `role` tag belongs to the `Message` envelope; the bare object
    // carries the wire fields only.
    assert_eq!(object["api"], "anthropic-messages");
    assert_eq!(object["stopReason"], "toolUse");
    let usage = &object["usage"];
    for name in [
        "input",
        "output",
        "cacheRead",
        "cacheWrite",
        "cacheWrite1h",
        "reasoning",
        "totalTokens",
        "cost",
    ] {
        assert!(usage.get(name).is_some(), "missing usage field {name}");
    }

    let round: AssistantMessage = serde_json::from_value(wire).expect("message deserializes");
    assert_eq!(round, message);
}

#[test]
fn assistant_content_blocks_discriminate_by_type() {
    let message = seed_assistant();
    let wire = serde_json::to_value(&message).expect("message serializes");
    let content = wire["content"].as_array().expect("content array");
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[2]["type"], "toolCall");
    assert_eq!(content[0]["thinkingSignature"], "sig");
    let round: Vec<AssistantBlock> =
        serde_json::from_value(wire["content"].clone()).expect("blocks deserialize");
    assert_eq!(round.len(), 3);
    match &round[2] {
        AssistantBlock::ToolCall(call) => assert_eq!(call.id, "call_1"),
        other => unreachable!("third block is a tool call, found {other:?}"),
    }
}

#[test]
fn user_message_accepts_plain_string_and_block_forms() {
    let text = UserMessage {
        content: UserContent::Text("hello".to_string()),
        timestamp: 7,
    };
    let wire = serde_json::to_value(&text).expect("user message serializes");
    assert_eq!(wire["content"], "hello");
    assert_eq!(wire["timestamp"], 7);
    let round: UserMessage = serde_json::from_value(wire).expect("user message");
    assert_eq!(round, text);

    let blocks = UserMessage {
        content: UserContent::Blocks(vec![
            UserBlock::Text(TextContent {
                text: "look".to_string(),
                text_signature: None,
            }),
            UserBlock::Image(ImageContent {
                data: "aGVsbG8=".to_string(),
                mime_type: "image/png".to_string(),
            }),
        ]),
        timestamp: 8,
    };
    let wire = serde_json::to_value(&blocks).expect("blocks serialize");
    assert_eq!(wire["content"][1]["type"], "image");
    assert_eq!(wire["content"][1]["mimeType"], "image/png");
    let round: UserMessage = serde_json::from_value(wire).expect("blocks round-trip");
    assert_eq!(round, blocks);
}

#[test]
fn tool_result_message_carries_tool_result_role_and_error_name() {
    let result = pi_ai::types::ToolResultMessage {
        tool_call_id: "call-1".to_string(),
        tool_name: "echo".to_string(),
        content: vec![pi_ai::types::ToolResultBlock::Text(TextContent {
            text: "done".to_string(),
            text_signature: None,
        })],
        details: Some(serde_json::json!({"rows": 3})),
        usage: None,
        added_tool_names: Some(vec!["later_tool".to_string()]),
        is_error: false,
        timestamp: 9,
    };
    let wire = serde_json::to_value(&result).expect("tool result serializes");
    assert_eq!(wire["toolCallId"], "call-1");
    let envelope =
        serde_json::to_value(Message::ToolResult(result.clone())).expect("message serializes");
    assert_eq!(envelope["role"], "toolResult");
    assert_eq!(wire["toolName"], "echo");
    assert_eq!(wire["isError"], false);
    assert_eq!(wire["addedToolNames"], serde_json::json!(["later_tool"]));
    let round: pi_ai::types::ToolResultMessage =
        serde_json::from_value(wire).expect("tool result round-trips");
    assert_eq!(round, result);
}

#[test]
fn message_role_tags_route_every_variant() {
    let context = Context {
        system_prompt: Some("be terse".to_string()),
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Text("hi".to_string()),
                timestamp: 1,
            }),
            Message::Assistant(seed_assistant()),
            Message::ToolResult(pi_ai::types::ToolResultMessage {
                tool_call_id: "call-1".to_string(),
                tool_name: "echo".to_string(),
                content: Vec::new(),
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: true,
                timestamp: 2,
            }),
        ],
        tools: Some(vec![Tool {
            name: "echo".to_string(),
            description: "Echoes the message back".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            constrained_sampling: None,
        }]),
    };
    let wire = serde_json::to_value(&context).expect("context serializes");
    let roles: Vec<&str> = wire["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .map(|message| message["role"].as_str().expect("role string"))
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "toolResult"]);
    assert_eq!(wire["tools"][0]["parameters"]["type"], "object");
    let round: Context = serde_json::from_value(wire).expect("context round-trips");
    assert_eq!(round.messages.len(), 3);
    assert!(round.tools.is_some());
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one branch per wire tag: the case list mirrors upstream's event protocol one to one"
)]
fn assistant_message_event_variants_keep_their_wire_tags() {
    let mut message = seed_assistant();
    message.stop_reason = StopReason::Pending;
    let events = vec![
        AssistantMessageEvent::Start {
            partial: message.clone(),
        },
        AssistantMessageEvent::TextStart {
            content_index: 1,
            partial: message.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index: 1,
            delta: "he".to_string(),
            partial: message.clone(),
        },
        AssistantMessageEvent::TextEnd {
            content_index: 1,
            content: "hello".to_string(),
            partial: message.clone(),
        },
        AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial: message.clone(),
        },
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta: "po".to_string(),
            partial: message.clone(),
        },
        AssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content: "ponder".to_string(),
            partial: message.clone(),
        },
        AssistantMessageEvent::ToolcallStart {
            content_index: 2,
            partial: message.clone(),
        },
        AssistantMessageEvent::ToolcallDelta {
            content_index: 2,
            delta: "{}".to_string(),
            partial: message.clone(),
        },
        AssistantMessageEvent::ToolcallEnd {
            content_index: 2,
            tool_call: pi_ai::types::ToolCall {
                id: "call-1".to_string(),
                name: "echo".to_string(),
                arguments: serde_json::Map::new(),
                thought_signature: None,
                namespace: None,
            },
            partial: message.clone(),
        },
        AssistantMessageEvent::Done {
            reason: StopReason::ToolUse,
            message: message.clone(),
        },
        AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error: message,
        },
    ];
    let mut done_seen = false;
    let mut error_seen = false;
    for event in events {
        let wire = serde_json::to_value(&event).expect("event serializes");
        let round: AssistantMessageEvent =
            serde_json::from_value(wire.clone()).expect("event round-trips");
        assert_eq!(round, event);
        match &event {
            AssistantMessageEvent::Start { partial } => {
                assert_eq!(wire["type"], "start");
                assert_eq!(wire["partial"]["model"], partial.model);
            }
            AssistantMessageEvent::TextStart { content_index, .. } => {
                assert_eq!(wire["type"], "text_start");
                assert_eq!(wire["contentIndex"], *content_index);
            }
            AssistantMessageEvent::TextDelta { delta, .. } => {
                assert_eq!(wire["type"], "text_delta");
                assert_eq!(wire["delta"], *delta);
            }
            AssistantMessageEvent::TextEnd { content, .. } => {
                assert_eq!(wire["type"], "text_end");
                assert_eq!(wire["content"], *content);
            }
            AssistantMessageEvent::ThinkingStart { .. } => {
                assert_eq!(wire["type"], "thinking_start");
            }
            AssistantMessageEvent::ThinkingDelta { .. } => {
                assert_eq!(wire["type"], "thinking_delta");
            }
            AssistantMessageEvent::ThinkingEnd { .. } => {
                assert_eq!(wire["type"], "thinking_end");
            }
            AssistantMessageEvent::ToolcallStart { .. } => {
                assert_eq!(wire["type"], "toolcall_start");
            }
            AssistantMessageEvent::ToolcallDelta { .. } => {
                assert_eq!(wire["type"], "toolcall_delta");
            }
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => {
                assert_eq!(wire["type"], "toolcall_end");
                assert_eq!(wire["toolCall"]["id"], tool_call.id);
            }
            AssistantMessageEvent::Done { .. } => {
                assert_eq!(wire["type"], "done");
                assert_eq!(wire["reason"], "toolUse");
                done_seen = true;
            }
            AssistantMessageEvent::Error { .. } => {
                assert_eq!(wire["type"], "error");
                assert_eq!(wire["reason"], "aborted");
                error_seen = true;
            }
        }
    }
    assert!(done_seen);
    assert!(error_seen);
}

#[test]
fn deferred_handle_keeps_the_camel_case_wire_fields() {
    let handle = pi_ai::types::DeferredHandle {
        provider: "openai-codex".to_string(),
        model_id: "gpt-5.5".to_string(),
        api: "openai-codex-responses".to_string(),
        id: "resp_123".to_string(),
        expires_at: Some(1_710_000_100_000),
        poll_after_ms: Some(2_000),
        data: Some(serde_json::json!({"row": 1})),
    };
    let wire = serde_json::to_value(&handle).expect("handle serializes");
    for name in [
        "provider",
        "modelId",
        "api",
        "id",
        "expiresAt",
        "pollAfterMs",
        "data",
    ] {
        assert!(wire.get(name).is_some(), "missing handle field {name}");
    }
    let round: pi_ai::types::DeferredHandle =
        serde_json::from_value(wire).expect("handle round-trips");
    assert_eq!(round, handle);
}

#[test]
fn deferred_request_takes_the_bool_and_window_forms() {
    let enabled: DeferredRequest =
        serde_json::from_value(serde_json::json!(true)).expect("bool form");
    assert_eq!(enabled, DeferredRequest::Enabled(true));
    let windowed: DeferredRequest =
        serde_json::from_value(serde_json::json!({"window": "1h"})).expect("windowed");
    assert_eq!(
        window_window(&windowed),
        Some(DeferredWindow::H1),
        "the wire's 1h window parses"
    );
    let empty: DeferredRequest =
        serde_json::from_value(serde_json::json!({})).expect("empty object");
    assert_eq!(window_window(&empty), None);
    let wire = serde_json::to_value(&DeferredRequest::Windowed {
        window: Some(DeferredWindow::H24),
    })
    .expect("window serializes");
    assert_eq!(wire, serde_json::json!({"window": "24h"}));
}

const fn window_window(request: &DeferredRequest) -> Option<DeferredWindow> {
    match request {
        DeferredRequest::Windowed { window } => *window,
        DeferredRequest::Enabled(_) => None,
    }
}

#[test]
fn chat_template_kwarg_value_covers_every_wire_form() {
    let template = ChatTemplateKwargValue::Template(pi_ai::types::ChatTemplateVar {
        var: ThinkingTemplateVar::ThinkingEffort,
        omit_when_off: Some(true),
    });
    let wire = serde_json::to_value(&template).expect("template serializes");
    assert_eq!(
        wire,
        serde_json::json!({"$var": "thinking.effort", "omitWhenOff": true})
    );
    let round: ChatTemplateKwargValue = serde_json::from_value(wire).expect("template round-trips");
    assert_eq!(round, template);

    for (wire, expected) in [
        (
            serde_json::json!("low"),
            ChatTemplateKwargValue::Str("low".to_string()),
        ),
        (
            serde_json::json!(12),
            ChatTemplateKwargValue::Number(serde_json::json!(12).as_number().expect("int").clone()),
        ),
        (
            serde_json::json!(1.5),
            ChatTemplateKwargValue::Number(
                serde_json::json!(1.5).as_number().expect("float").clone(),
            ),
        ),
        (serde_json::json!(true), ChatTemplateKwargValue::Bool(true)),
        (serde_json::json!(null), ChatTemplateKwargValue::Null),
    ] {
        let parsed: ChatTemplateKwargValue =
            serde_json::from_value(wire.clone()).expect("scalar parses");
        assert_eq!(parsed, expected, "wire form {wire}");
        let re = serde_json::to_value(&expected).expect("scalar serializes");
        assert_eq!(re, wire);
    }
}

#[test]
fn tool_constrained_sampling_takes_the_config_and_false_forms() {
    let json_schema: ConstrainedSamplingSetting = serde_json::from_value(serde_json::json!({
        "type": "json_schema",
        "strict": "prefer"
    }))
    .expect("json_schema config");
    match &json_schema {
        ConstrainedSamplingSetting::Config(
            pi_ai::types::ConstrainedSamplingConfig::JsonSchema { strict },
        ) => {
            assert_eq!(*strict, pi_ai::types::Strictness::Prefer);
        }
        other => unreachable!("expected a config, found {other:?}"),
    }
    let grammar: ConstrainedSamplingSetting = serde_json::from_value(serde_json::json!({
        "type": "grammar",
        "variants": {"openai_lark": "grammar.lark"}
    }))
    .expect("grammar config");
    match &grammar {
        ConstrainedSamplingSetting::Config(pi_ai::types::ConstrainedSamplingConfig::Grammar {
            variants,
        }) => {
            assert_eq!(
                variants.get(&GrammarFormat::OpenaiLark).map(String::as_str),
                Some("grammar.lark")
            );
        }
        other => unreachable!("expected a grammar config, found {other:?}"),
    }
    let disabled: ConstrainedSamplingSetting =
        serde_json::from_value(serde_json::json!(false)).expect("false disables");
    assert_eq!(disabled, ConstrainedSamplingSetting::Disabled(false));
}

#[test]
fn model_compat_and_routing_round_trip_the_open_router_shapes() {
    let model = Model {
        id: "gpt-5.5".to_string(),
        name: "GPT-5.5".to_string(),
        api: KnownApi::OpenaiCompletions.into(),
        provider: KnownProvider::Openai.into(),
        base_url: "https://api.openai.com/v1".to_string(),
        reasoning: true,
        thinking_level_map: Some(
            [
                (
                    pi_ai::types::ModelThinkingLevel::High,
                    Some("high".to_string()),
                ),
                (pi_ai::types::ModelThinkingLevel::Xhigh, None),
            ]
            .into_iter()
            .collect(),
        ),
        input: vec![Modality::Text, Modality::Image],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 1.25,
                output: 10.0,
                cache_read: 0.125,
                cache_write: 0.0,
            },
            tiers: Some(vec![pi_ai::types::ModelCostTier {
                rates: pi_ai::types::ModelCostRates {
                    input: 1.0,
                    output: 8.0,
                    cache_read: 0.1,
                    cache_write: 0.0,
                },
                input_tokens_above: 128_000,
            }]),
        },
        context_window: 400_000,
        max_tokens: 64_000,
        sampling_params: Some(BTreeMap::new()),
        headers: None,
        compat: Some(ModelCompat {
            open_router_routing: Some(pi_ai::types::OpenRouterRouting {
                sort: Some(pi_ai::types::SortPreference::Name("price".to_string())),
                max_price: Some(pi_ai::types::MaxPrice {
                    prompt: Some(pi_ai::types::PriceValue::Num(1.0)),
                    completion: Some(pi_ai::types::PriceValue::Str("2".to_string())),
                    ..pi_ai::types::MaxPrice::default()
                }),
                preferred_min_throughput: Some(pi_ai::types::PercentilePreference::Percentiles(
                    pi_ai::types::PercentileCutoffs {
                        p50: Some(40.0),
                        p99: Some(10.0),
                        ..pi_ai::types::PercentileCutoffs::default()
                    },
                )),
                ..pi_ai::types::OpenRouterRouting::default()
            }),
            session_affinity_format: Some(pi_ai::types::SessionAffinityFormat::Openrouter),
            ..ModelCompat::default()
        }),
    };
    let wire = serde_json::to_value(&model).expect("model serializes");
    for name in [
        "id",
        "name",
        "api",
        "provider",
        "baseUrl",
        "reasoning",
        "thinkingLevelMap",
        "input",
        "cost",
        "contextWindow",
        "maxTokens",
        "compat",
    ] {
        assert!(wire.get(name).is_some(), "missing model field {name}");
    }
    let compat = &wire["compat"];
    assert_eq!(
        compat["openRouterRouting"]["sort"],
        serde_json::json!("price")
    );
    assert_eq!(
        compat["openRouterRouting"]["max_price"]["prompt"],
        serde_json::json!(1.0)
    );
    assert_eq!(
        compat["openRouterRouting"]["preferred_min_throughput"]["p50"],
        serde_json::json!(40.0)
    );
    let round: Model = serde_json::from_value(wire).expect("model round-trips");
    assert_eq!(round, model);
}

#[test]
fn model_thinking_level_map_null_marks_unsupported() {
    let wire = serde_json::json!({
        "id": "m",
        "name": "M",
        "api": "openai-completions",
        "provider": "openai",
        "baseUrl": "https://x.invalid/v1",
        "reasoning": false,
        "thinkingLevelMap": {"high": "high", "xhigh": null},
        "input": ["text"],
        "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0},
        "contextWindow": 1000,
        "maxTokens": 100
    });
    let model: Model = serde_json::from_value(wire).expect("model with null level parses");
    let map = model.thinking_level_map.as_ref().expect("map present");
    assert_eq!(
        map.get(&pi_ai::types::ModelThinkingLevel::High),
        Some(&Some("high".to_string()))
    );
    assert_eq!(
        map.get(&pi_ai::types::ModelThinkingLevel::Xhigh),
        Some(&None)
    );
    let re = serde_json::to_value(&model).expect("model serializes");
    assert_eq!(re["thinkingLevelMap"]["xhigh"], serde_json::Value::Null);
}

#[test]
fn api_and_provider_ids_keep_open_string_behavior() {
    assert_eq!(
        KnownApi::try_from("openai-completions"),
        Ok(KnownApi::OpenaiCompletions)
    );
    assert!(KnownApi::try_from("not-an-api").is_err());

    let custom: pi_ai::types::Api = "my-gateway".into();
    assert!(!custom.is_known());
    assert_eq!(custom.as_known(), None);
    assert_eq!(custom.to_string(), "my-gateway");

    let known: pi_ai::types::Api = KnownApi::BedrockConverseStream.into();
    assert_eq!(known.as_known(), Some(KnownApi::BedrockConverseStream));
    assert!(known.is_known());
    assert_eq!(
        serde_json::to_value(&known).expect("api serializes"),
        serde_json::json!("bedrock-converse-stream")
    );

    let provider: pi_ai::types::ProviderId = KnownProvider::XiaomiTokenPlanAms.into();
    assert_eq!(provider.as_known(), Some(KnownProvider::XiaomiTokenPlanAms));
    assert_eq!(provider.to_string(), "xiaomi-token-plan-ams");

    let custom_provider: pi_ai::types::ProviderId = "my-relay".into();
    assert!(!custom_provider.is_known());
    assert_eq!(custom_provider.to_string(), "my-relay");

    let images_api: pi_ai::types::ImagesApi = pi_ai::types::KnownImagesApi::OpenrouterImages.into();
    assert_eq!(images_api.to_string(), "openrouter-images");
    assert_eq!(
        images_api_to_string(&"openrouter-images".into()),
        "openrouter-images"
    );
}

fn images_api_to_string(api: &pi_ai::types::ImagesApi) -> String {
    api.to_string()
}

#[test]
fn images_result_and_diagnostics_round_trip() {
    let diagnostic = AssistantMessageDiagnostic {
        kind: "provider.retry".to_string(),
        timestamp: 5,
        error: Some(DiagnosticErrorInfo {
            name: Some("RateLimitError".to_string()),
            message: "429".to_string(),
            stack: None,
            code: Some(serde_json::json!(429).as_number().expect("int").clone()),
        }),
        details: Some(BTreeMap::new()),
    };
    let wire = serde_json::to_value(&diagnostic).expect("diagnostic serializes");
    assert_eq!(wire["type"], "provider.retry");
    assert_eq!(wire["error"]["code"], serde_json::json!(429));
    let round: AssistantMessageDiagnostic =
        serde_json::from_value(wire).expect("diagnostic round-trips");
    assert_eq!(round, diagnostic);

    let images = pi_ai::types::AssistantImages {
        api: "openrouter-images".into(),
        provider: "openrouter".into(),
        model: "image-1".to_string(),
        output: vec![pi_ai::types::ImagesBlock::Text(TextContent {
            text: "rendered".to_string(),
            text_signature: None,
        })],
        response_id: None,
        usage: None,
        stop_reason: ImagesStopReason::Stop,
        error_message: None,
        timestamp: 9,
    };
    let wire = serde_json::to_value(&images).expect("images result serializes");
    assert_eq!(wire["stopReason"], "stop");
    assert_eq!(wire["output"][0]["type"], "text");
    let round: pi_ai::types::AssistantImages =
        serde_json::from_value(wire).expect("images result round-trips");
    assert_eq!(round, images);
}

#[test]
fn option_structs_carry_the_pure_data_fields() {
    // The options structs carry pure data at this layer; the transport fields
    // land with the HttpClient-seam child. This pins the pure fields.
    let options = pi_ai::types::StreamOptions {
        api_key: Some("key".to_string()),
        temperature: Some(0.7),
        max_tokens: Some(1_000),
        transport: Some(Transport::Auto),
        cache_retention: Some(CacheRetention::Long),
        session_id: Some("affinity".to_string()),
        websocket_connect_timeout_ms: Some(5_000),
        ..pi_ai::types::StreamOptions::default()
    };
    assert_eq!(options.timeout_ms, None);
    assert_eq!(options.max_retry_delay_ms, None);
    assert_eq!(options.transport, Some(Transport::Auto));
    assert_eq!(options.cache_retention, Some(CacheRetention::Long));

    let simple = pi_ai::types::SimpleStreamOptions {
        reasoning: Some(ThinkingLevel::Xhigh),
        tool_choice: Some(pi_ai::types::ToolChoice::None),
        deferred: Some(DeferredRequest::Windowed {
            window: Some(DeferredWindow::M15),
        }),
        thinking_budgets: Some(pi_ai::types::ThinkingBudgets {
            minimal: Some(512),
            low: Some(1_024),
            medium: Some(4_096),
            high: Some(16_384),
        }),
        ..pi_ai::types::SimpleStreamOptions::default()
    };
    assert_eq!(options.cache_retention, Some(CacheRetention::Long));
    assert_eq!(simple_reasoning(&simple), Some(ThinkingLevel::Xhigh));
}

const fn simple_reasoning(options: &pi_ai::types::SimpleStreamOptions) -> Option<ThinkingLevel> {
    options.reasoning
}
