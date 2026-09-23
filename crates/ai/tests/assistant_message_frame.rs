//! The assistant-message frame port, from `test/assistant-message-frame.test.ts`.
//! The `processResponsesStream` round-trip case rides with the
//! OpenAI-responses child, and the provider-scratch-field whitelist is
//! statically upheld by typed structs, so both are absent here.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use common::bare_assistant_message;
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, ProviderId, StopReason,
    TextContent, ThinkingContent, ToolCall,
};
use pi_ai::utils::assistant_message_frame::{
    AssistantMessageFrame, AssistantMessageFrameEncoder, reduce_assistant_message_frames,
};

fn seed() -> AssistantMessage {
    let mut message = bare_assistant_message();
    message.api = Api::from("test-api");
    message.provider = ProviderId::from("test-provider");
    message.model = String::from("test-model");
    message.timestamp = 1;
    message
}

/// The started-encoder fixture the frame suites drive, upstream's per-case
/// setup: the seed partial, optionally reshaped by `configure`, fed through
/// the `Start` event on a fresh encoder.
fn started_encoder(
    configure: impl FnOnce(&mut AssistantMessage),
) -> (AssistantMessage, AssistantMessageFrameEncoder) {
    let mut partial = seed();
    configure(&mut partial);
    let mut encoder = AssistantMessageFrameEncoder::default();
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    );
    (partial, encoder)
}

/// The started-frames fixture: the seed partial's `Start` frame collected,
/// the run continuing to append.
fn started_frames(
    configure: impl FnOnce(&mut AssistantMessage),
) -> (
    AssistantMessage,
    AssistantMessageFrameEncoder,
    Vec<AssistantMessageFrame>,
) {
    let mut partial = seed();
    configure(&mut partial);
    let mut encoder = AssistantMessageFrameEncoder::default();
    let frames = vec![frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    )];
    (partial, encoder, frames)
}

fn frame(
    encoder: &mut AssistantMessageFrameEncoder,
    event: AssistantMessageEvent,
) -> AssistantMessageFrame {
    encoder
        .encode(event)
        .expect("the event encodes")
        .expect("the event produces a frame")
}

fn frame_opt(
    encoder: &mut AssistantMessageFrameEncoder,
    event: AssistantMessageEvent,
) -> Option<AssistantMessageFrame> {
    encoder.encode(event).expect("the event encodes")
}

fn text(text: &str) -> TextContent {
    TextContent {
        text: text.to_owned(),
        text_signature: None,
    }
}

fn thinking(thinking_text: &str) -> ThinkingContent {
    ThinkingContent {
        thinking: thinking_text.to_owned(),
        thinking_signature: None,
        redacted: None,
    }
}

fn tool_call(name: &str, arguments: &serde_json::Value) -> ToolCall {
    ToolCall {
        id: String::from("call"),
        name: name.to_owned(),
        arguments: arguments.as_object().cloned().expect("an object"),
        thought_signature: None,
        namespace: None,
    }
}

#[test]
fn uses_authoritative_text_end_content_and_signature() {
    let (mut partial, mut encoder, mut frames) = started_frames(|_partial| {});
    partial.content.push(AssistantBlock::Text(text("Hello ")));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    partial.content[0] = AssistantBlock::Text(TextContent {
        text: String::from("Hello world"),
        text_signature: Some(String::from("sig-text")),
    });
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: String::from("incorrect"),
            partial: partial.clone(),
        },
    ));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: String::from("Hello world"),
            partial,
        },
    ));

    assert_eq!(
        frames.last(),
        Some(&AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::from("Hello world"),
            text_signature: Some(String::from("sig-text")),
        })
    );
    let reduced = reduce_assistant_message_frames(frames)
        .expect("the frames replay")
        .expect("a message");
    assert_eq!(
        reduced.content,
        vec![AssistantBlock::Text(TextContent {
            text: String::from("Hello world"),
            text_signature: Some(String::from("sig-text")),
        })]
    );
}

#[test]
fn preserves_provider_thinking_level_from_the_stream_start() {
    let mut partial = seed();
    partial.provider_thinking_level = Some(String::from("high"));
    let mut encoder = AssistantMessageFrameEncoder::default();
    let start = frame(&mut encoder, AssistantMessageEvent::Start { partial });

    let AssistantMessageFrame::Start { partial } = &start else {
        panic!("expected a start frame, got {start:?}")
    };
    assert_eq!(partial.provider_thinking_level, Some(String::from("high")));
    let reduced = reduce_assistant_message_frames(vec![start])
        .expect("the frames replay")
        .expect("a message");
    assert_eq!(reduced.provider_thinking_level, Some(String::from("high")));
}

#[test]
fn preserves_initial_and_final_thinking_metadata_including_redaction() {
    let (mut partial, mut encoder, mut frames) = started_frames(|_partial| {});
    partial
        .content
        .push(AssistantBlock::Thinking(ThinkingContent {
            thinking: String::from("[redacted]"),
            thinking_signature: Some(String::from("encrypted-start")),
            redacted: Some(true),
        }));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    partial.content[0] = AssistantBlock::Thinking(ThinkingContent {
        thinking: String::from("[redacted]"),
        thinking_signature: Some(String::from("encrypted-final")),
        redacted: Some(true),
    });
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content: String::from("[redacted]"),
            partial,
        },
    ));

    assert_eq!(
        frames.last(),
        Some(&AssistantMessageFrame::ThinkingEnd {
            content_index: 0,
            content: String::from("[redacted]"),
            thinking_signature: Some(String::from("encrypted-final")),
            redacted: Some(true),
        })
    );
    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content.first(),
        Some(&AssistantBlock::Thinking(ThinkingContent {
            thinking: String::from("[redacted]"),
            thinking_signature: Some(String::from("encrypted-final")),
            redacted: Some(true),
        }))
    );
}

#[test]
fn parses_unfinished_tool_json_once_and_uses_authoritative_completed_arguments() {
    let initial_frames = vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::ToolcallStart {
            content_index: 0,
            tool_call: tool_call("write", &json!({})),
        },
        AssistantMessageFrame::ToolcallDelta {
            content_index: 0,
            delta: String::from("{\"path\":\"READ"),
        },
    ];

    let reduced = reduce_assistant_message_frames(initial_frames.clone())
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content[0],
        AssistantBlock::ToolCall(tool_call("write", &json!({"path": "READ"})))
    );

    let mut complete_frames = initial_frames;
    complete_frames.push(AssistantMessageFrame::ToolcallDelta {
        content_index: 0,
        delta: String::from("ME.md\",\"lines\":[1,2]}"),
    });
    complete_frames.push(AssistantMessageFrame::ToolcallEnd {
        content_index: 0,
        id: String::from("final-id"),
        name: String::from("write_file"),
        arguments: json!({"path": "final.md", "lines": [3]})
            .as_object()
            .cloned()
            .expect("an object"),
        thought_signature: Some(String::from("thought")),
        namespace: Some(String::from("files")),
    });
    let mut final_tool = tool_call("write_file", &json!({"path": "final.md", "lines": [3]}));
    final_tool.id = String::from("final-id");
    final_tool.thought_signature = Some(String::from("thought"));
    final_tool.namespace = Some(String::from("files"));
    let reduced = reduce_assistant_message_frames(complete_frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(reduced.content[0], AssistantBlock::ToolCall(final_tool));
}

#[test]
fn reconciles_queued_text_events_against_one_advanced_live_partial_without_duplicate_content() {
    let mut partial = seed();
    let mut live_text = text("");
    partial
        .content
        .push(AssistantBlock::Text(live_text.clone()));
    // The events queue while the shared live partial advances; encoding
    // happens after the accumulation, so every event observes the final
    // partial, upstream's encode-time view of the shared object.
    let mut advanced = partial.clone();
    for delta in ["Hel", "lo", " ", "world"] {
        live_text.text.push_str(delta);
        if let Some(AssistantBlock::Text(text_block)) = advanced.content.last_mut() {
            text_block.text = live_text.text.clone();
        }
    }
    let mut events = vec![
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: advanced.clone(),
        },
    ];
    for delta in ["Hel", "lo", " ", "world"] {
        events.push(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: delta.to_owned(),
            partial: advanced.clone(),
        });
    }

    let mut encoder = AssistantMessageFrameEncoder::default();
    let frames: Vec<AssistantMessageFrame> = events
        .into_iter()
        .filter_map(|event| encoder.encode(event).expect("encodes"))
        .collect();

    let frame_types: Vec<&'static str> = frames.iter().map(frame_type_name).collect();
    assert_eq!(frame_types, ["start", "text_start"]);
    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content,
        vec![AssistantBlock::Text(text("Hello world"))]
    );
}

#[test]
fn trims_only_the_covered_prefix_when_a_start_snapshot_lands_inside_a_delta() {
    let (mut partial, mut encoder, mut frames) = started_frames(|_partial| {});
    partial.content.push(AssistantBlock::Text(text("Hel")));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    assert!(
        frame_opt(
            &mut encoder,
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: String::from("He"),
                partial: partial.clone(),
            }
        )
        .is_none()
    );
    let remainder = frame(
        &mut encoder,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: String::from("llo"),
            partial: partial.clone(),
        },
    );
    frames.push(remainder.clone());

    assert_eq!(
        remainder,
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: String::from("lo"),
        }
    );
    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(reduced.content, vec![AssistantBlock::Text(text("Hello"))]);
}

#[test]
fn checkpoints_queued_tool_json_without_replaying_covered_deltas() {
    let mut partial = seed();
    let mut tool = tool_call("write", &json!({}));
    partial.content.push(AssistantBlock::ToolCall(tool.clone()));
    // The deltas queue while the live partial advances to the final
    // arguments; the start and both deltas encode against the advanced
    // snapshot, upstream's encode-time view of the shared object.
    tool.arguments = json!({"path": "README.md"})
        .as_object()
        .cloned()
        .expect("an object");
    let mut advanced = partial.clone();
    if let Some(AssistantBlock::ToolCall(tool_block)) = advanced.content.last_mut() {
        tool_block.arguments = tool.arguments;
    }
    let events = vec![
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: advanced.clone(),
        },
        AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: String::from("{\"path\":\"READ"),
            partial: advanced.clone(),
        },
        AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: String::from("ME.md\"}"),
            partial: advanced.clone(),
        },
    ];

    let mut encoder = AssistantMessageFrameEncoder::default();
    let frames: Vec<AssistantMessageFrame> = events
        .into_iter()
        .filter_map(|event| encoder.encode(event).expect("encodes"))
        .collect();
    let frame_types: Vec<&'static str> = frames.iter().map(frame_type_name).collect();
    assert_eq!(
        frame_types,
        ["start", "toolcall_start", "toolcall_checkpoint"]
    );
    assert_eq!(
        frames.last(),
        Some(&AssistantMessageFrame::ToolcallCheckpoint {
            content_index: 0,
            json: String::from("{\"path\":\"README.md\"}"),
        })
    );
    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content,
        vec![AssistantBlock::ToolCall(tool_call(
            "write",
            &json!({"path": "README.md"})
        ))]
    );
}

#[test]
fn resumes_legacy_grammar_tool_json_from_initial_arguments() {
    let (mut partial, mut encoder, mut frames) = started_frames(|_partial| {});
    let mut tool = tool_call("bash", &json!({"input": "a"}));
    partial.content.push(AssistantBlock::ToolCall(tool.clone()));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    tool.arguments = json!({"input": "ab"})
        .as_object()
        .cloned()
        .expect("an object");
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: String::from("{\"input\":\"ab"),
            partial: partial.clone(),
        },
    ));
    tool.arguments = json!({"input": "abc"})
        .as_object()
        .cloned()
        .expect("an object");
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: String::from("c\"}"),
            partial: partial.clone(),
        },
    ));

    assert_eq![
        frames[2..],
        [
            AssistantMessageFrame::ToolcallCheckpoint {
                content_index: 0,
                json: String::from("{\"input\":\"ab"),
            },
            AssistantMessageFrame::ToolcallDelta {
                content_index: 0,
                delta: String::from("c\"}"),
            },
        ]
    ];
    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content,
        vec![AssistantBlock::ToolCall(tool_call(
            "bash",
            &json!({"input": "abc"})
        ))]
    );
}

#[test]
fn streams_tool_json_compactly_from_an_empty_argument_start() {
    let (mut partial, mut encoder, mut frames) = started_frames(|_partial| {});
    let mut tool = tool_call("bash", &json!({}));
    partial.content.push(AssistantBlock::ToolCall(tool.clone()));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    tool.arguments = json!({"command": "ls -la /tmp"})
        .as_object()
        .cloned()
        .expect("an object");
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: String::from("{\"command\":\"ls -la /tmp\"}"),
            partial: partial.clone(),
        },
    ));

    assert_eq!(
        frames.last(),
        Some(&AssistantMessageFrame::ToolcallDelta {
            content_index: 0,
            delta: String::from("{\"command\":\"ls -la /tmp\"}"),
        })
    );
    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content[0],
        AssistantBlock::ToolCall(tool_call("bash", &json!({"command": "ls -la /tmp"})))
    );
}

#[test]
fn accepts_a_pre_generation_error_but_rejects_success_or_updates_before_start() {
    let mut failed = seed();
    failed.stop_reason = StopReason::Error;
    failed.error_message = Some(String::from("setup failed"));
    let mut encoder = AssistantMessageFrameEncoder::default();
    assert!(
        encoder
            .encode(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: failed,
            })
            .expect("encodes")
            .is_none()
    );

    let mut completed = seed();
    completed.stop_reason = StopReason::Stop;
    let error = AssistantMessageFrameEncoder::default()
        .encode(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: completed,
        })
        .expect_err("done before start");
    assert!(error.contains("done event appears before start"), "{error}");
    let error = AssistantMessageFrameEncoder::default()
        .encode(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: String::from("x"),
            partial: seed(),
        })
        .expect_err("a delta before start");
    assert!(
        error.contains("text_delta event appears before start"),
        "{error}"
    );
}

#[test]
fn treats_end_signature_metadata_including_absence_as_authoritative() {
    let frames = vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: TextContent {
                text: String::new(),
                text_signature: Some(String::from("stale-text")),
            },
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::new(),
            text_signature: None,
        },
        AssistantMessageFrame::ThinkingStart {
            content_index: 1,
            content: ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(String::from("stale-thinking")),
                redacted: Some(true),
            },
        },
        AssistantMessageFrame::ThinkingEnd {
            content_index: 1,
            content: String::new(),
            thinking_signature: Some(String::new()),
            redacted: Some(false),
        },
        AssistantMessageFrame::ToolcallStart {
            content_index: 2,
            tool_call: ToolCall {
                id: String::from("call"),
                name: String::from("read"),
                arguments: serde_json::Map::new(),
                thought_signature: Some(String::from("stale-tool")),
                namespace: Some(String::from("stale-namespace")),
            },
        },
        AssistantMessageFrame::ToolcallEnd {
            content_index: 2,
            id: String::from("call"),
            name: String::from("read"),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        },
    ];

    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content,
        vec![
            AssistantBlock::Text(text("")),
            AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(String::new()),
                redacted: Some(false),
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: String::from("call"),
                name: String::from("read"),
                arguments: serde_json::Map::new(),
                thought_signature: None,
                namespace: None,
            }),
        ]
    );
}

#[test]
fn stores_authoritative_final_arguments_in_toolcall_end_frames() {
    let mut partial = seed();
    let mut tool = tool_call("read", &json!({"path": "README.md"}));
    tool.thought_signature = Some(String::from("thought"));
    tool.namespace = Some(String::from("files"));
    partial.content.push(AssistantBlock::ToolCall(tool.clone()));

    let mut encoder = AssistantMessageFrameEncoder::default();
    frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    );
    frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: partial.clone(),
        },
    );
    let end = frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallEnd {
            content_index: 0,
            tool_call: tool,
            partial: partial.clone(),
        },
    );
    assert_eq!(
        end,
        AssistantMessageFrame::ToolcallEnd {
            content_index: 0,
            id: String::from("call"),
            name: String::from("read"),
            arguments: json!({"path": "README.md"})
                .as_object()
                .cloned()
                .expect("an object"),
            thought_signature: Some(String::from("thought")),
            namespace: Some(String::from("files")),
        }
    );
}

#[test]
fn supports_interleaved_streams_by_content_index() {
    let frames = vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: text(""),
        },
        AssistantMessageFrame::ToolcallStart {
            content_index: 1,
            tool_call: tool_call("lookup", &json!({})),
        },
        AssistantMessageFrame::ThinkingStart {
            content_index: 2,
            content: thinking(""),
        },
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: String::from("answer"),
        },
        AssistantMessageFrame::ToolcallDelta {
            content_index: 1,
            delta: String::from("{\"query\":\"pi\"}"),
        },
        AssistantMessageFrame::ThinkingDelta {
            content_index: 2,
            delta: String::from("check"),
        },
        AssistantMessageFrame::ToolcallEnd {
            content_index: 1,
            id: String::from("call"),
            name: String::from("lookup"),
            arguments: json!({"query": "pi"})
                .as_object()
                .cloned()
                .expect("an object"),
            thought_signature: None,
            namespace: None,
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::from("answer"),
            text_signature: None,
        },
        AssistantMessageFrame::ThinkingEnd {
            content_index: 2,
            content: String::from("check"),
            thinking_signature: None,
            redacted: None,
        },
    ];

    let reduced = reduce_assistant_message_frames(frames)
        .expect("replays")
        .expect("a message");
    assert_eq!(
        reduced.content,
        vec![
            AssistantBlock::Text(text("answer")),
            AssistantBlock::ToolCall(tool_call("lookup", &json!({"query": "pi"}))),
            AssistantBlock::Thinking(thinking("check")),
        ]
    );
}

#[test]
fn snapshots_mutable_event_data_and_keeps_reduction_pure() {
    let mut partial = seed();
    let mut encoder = AssistantMessageFrameEncoder::default();
    let start = frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    );
    partial.usage.cost.total = 99.0;

    partial.content.push(AssistantBlock::ToolCall(tool_call(
        "run",
        &json!({"nested": {"value": "original"}}),
    )));
    let tool_start = frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: partial.clone(),
        },
    );
    // Rust move semantics replace the shared-mutable-partial hazard: the
    // frame owns its snapshot, and the reduced output is a fresh clone.
    let reduced = reduce_assistant_message_frames(vec![start, tool_start.clone()])
        .expect("replays")
        .expect("a message");
    assert![
        reduced.usage.cost.total == 0.0,
        "the start snapshot froze the usage at its seed value"
    ];
    assert_eq!(
        reduced.content[0],
        AssistantBlock::ToolCall(tool_call("run", &json!({"nested": {"value": "original"}})))
    );

    if let AssistantMessageFrame::ToolcallStart { tool_call, .. } = &tool_start {
        assert_eq!(
            tool_call.arguments.get("nested"),
            Some(&json!({"value": "original"}))
        );
    }
}

#[test]
fn omits_terminal_events_because_settlement_is_separate() {
    let mut message = seed();
    let mut completed = AssistantMessageFrameEncoder::default();
    completed
        .encode(AssistantMessageEvent::Start {
            partial: message.clone(),
        })
        .expect("encodes")
        .expect("a start frame");
    message.stop_reason = StopReason::Stop;
    assert!(
        completed
            .encode(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: message.clone(),
            })
            .expect("encodes")
            .is_none()
    );
    message.stop_reason = StopReason::Error;
    message.error_message = Some(String::from("failed"));
    assert!(
        AssistantMessageFrameEncoder::default()
            .encode(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: message,
            })
            .expect("encodes")
            .is_none()
    );
}

#[test]
fn returns_none_when_there_is_no_start_frame() {
    assert!(
        reduce_assistant_message_frames(Vec::<AssistantMessageFrame>::new())
            .expect("replays")
            .is_none()
    );
    assert!(
        reduce_assistant_message_frames(vec![AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: String::from("x"),
        }])
        .expect("replays")
        .is_none()
    );
}

#[test]
fn rejects_frames_before_start_wrong_block_kinds_duplicate_ends_and_index_gaps() {
    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: String::from("x"),
        },
        AssistantMessageFrame::Start { partial: seed() },
    ])
    .expect_err("a frame before start");
    assert!(error.contains("before the start frame"), "{error}");

    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::ToolcallStart {
            content_index: 0,
            tool_call: tool_call("run", &json!({})),
        },
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: String::from("wrong"),
        },
    ])
    .expect_err("a wrong-kind delta");
    assert!(error.contains("expected text block"), "{error}");

    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: text(""),
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::new(),
            text_signature: None,
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::new(),
            text_signature: None,
        },
    ])
    .expect_err("a duplicate end");
    assert!(error.contains("follows the end"), "{error}");

    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::TextStart {
            content_index: 1,
            content: text(""),
        },
    ])
    .expect_err("an index gap");
    assert!(error.contains("would leave a gap"), "{error}");
}

#[test]
fn rejects_conversion_events_whose_content_index_points_to_the_wrong_block_kind() {
    let mut partial = seed();
    let mut encoder = AssistantMessageFrameEncoder::default();
    encoder
        .encode(AssistantMessageEvent::Start {
            partial: partial.clone(),
        })
        .expect("encodes");
    partial.content.push(AssistantBlock::Thinking(thinking("")));
    let error = encoder
        .encode(AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        })
        .expect_err("wrong block kind");
    assert!(
        error.contains("text_start event points to thinking block"),
        "{error}"
    );
}

use serde_json::json;

const fn frame_type_name(frame: &AssistantMessageFrame) -> &'static str {
    match frame {
        AssistantMessageFrame::Start { .. } => "start",
        AssistantMessageFrame::TextStart { .. } => "text_start",
        AssistantMessageFrame::TextDelta { .. } => "text_delta",
        AssistantMessageFrame::TextEnd { .. } => "text_end",
        AssistantMessageFrame::ThinkingStart { .. } => "thinking_start",
        AssistantMessageFrame::ThinkingDelta { .. } => "thinking_delta",
        AssistantMessageFrame::ThinkingEnd { .. } => "thinking_end",
        AssistantMessageFrame::ToolcallStart { .. } => "toolcall_start",
        AssistantMessageFrame::ToolcallCheckpoint { .. } => "toolcall_checkpoint",
        AssistantMessageFrame::ToolcallDelta { .. } => "toolcall_delta",
        AssistantMessageFrame::ToolcallEnd { .. } => "toolcall_end",
    }
}

// --- Rust-native additions: encoder/reducer edge branches on top of the
// ported suites ---

#[test]
fn the_encoder_debugs_with_its_started_block_keys() {
    let encoder = AssistantMessageFrameEncoder::default();
    let debug = format!("{encoder:?}");
    assert![debug.starts_with("AssistantMessageFrameEncoder"), "{debug}"];
    assert![debug.contains("blocks: []"), "{debug}"];

    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::Text(text("hi")));
    });
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial,
        },
    );
    let debug = format!("{encoder:?}");
    assert![debug.contains("blocks: [0]"), "{debug}"];
}

#[test]
fn rejects_events_after_a_terminal_event_and_a_second_start() {
    let mut encoder = AssistantMessageFrameEncoder::default();
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::Start { partial: seed() },
    );
    let done = frame_opt(
        &mut encoder,
        AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: seed(),
        },
    );
    assert![done.is_none(), "the terminal event produces no frame"];
    let error = encoder
        .encode(AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: seed(),
        })
        .expect_err("no event follows a terminal event");
    assert![error.contains("follows a terminal event"), "{error}"];

    let mut encoder = AssistantMessageFrameEncoder::default();
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::Start { partial: seed() },
    );
    let error = encoder
        .encode(AssistantMessageEvent::Start { partial: seed() })
        .expect_err("a second start is rejected");
    assert_eq![
        error,
        "Assistant message stream contains more than one start event"
    ];
}

#[test]
fn rejects_start_events_pointing_at_other_block_kinds_or_missing_blocks() {
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::Text(text("hi")));
    });

    let error = encoder
        .encode(AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial: partial.clone(),
        })
        .expect_err("thinking_start at a text block");
    assert![
        error.contains("thinking_start event points to text block at index 0"),
        "{error}"
    ];
    let error = encoder
        .encode(AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: partial.clone(),
        })
        .expect_err("toolcall_start at a text block");
    assert![
        error.contains("toolcall_start event points to text block at index 0"),
        "{error}"
    ];
    let error = encoder
        .encode(AssistantMessageEvent::TextStart {
            content_index: 3,
            partial,
        })
        .expect_err("the index points at no block");
    assert![
        error.contains("text_start event has no content block at index 3"),
        "{error}"
    ];
}

#[test]
fn rejects_end_events_pointing_at_other_block_kinds() {
    // text_end at a thinking block
    let (partial, mut encoder) = started_encoder(|partial| {
        partial
            .content
            .push(AssistantBlock::Thinking(thinking("h")));
    });
    let error = encoder
        .encode(AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: String::from("h"),
            partial,
        })
        .expect_err("text_end at a thinking block");
    assert![
        error.contains("text_end event points to thinking block at index 0"),
        "{error}"
    ];

    // thinking_end at a text block
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::Text(text("h")));
    });
    let error = encoder
        .encode(AssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content: String::from("h"),
            partial,
        })
        .expect_err("thinking_end at a text block");
    assert![
        error.contains("thinking_end event points to text block at index 0"),
        "{error}"
    ];

    // toolcall_end at a text block
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::Text(text("h")));
    });
    let error = encoder
        .encode(AssistantMessageEvent::ToolcallEnd {
            content_index: 0,
            tool_call: tool_call("read", &serde_json::json!({})),
            partial,
        })
        .expect_err("toolcall_end at a text block");
    assert![
        error.contains("toolcall_end event points to text block at index 0"),
        "{error}"
    ];
}

#[test]
fn streams_thinking_deltas_and_ends_with_their_metadata() {
    let (mut partial, mut encoder, mut frames) = started_frames(|_partial| {});
    partial.content.push(AssistantBlock::Thinking(thinking("")));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta: String::from("abc"),
            partial: partial.clone(),
        },
    ));
    partial.content[0] = AssistantBlock::Thinking(ThinkingContent {
        thinking: String::from("abcd"),
        thinking_signature: Some(String::from("sig")),
        redacted: Some(true),
    });
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content: String::from("abcd"),
            partial,
        },
    ));

    assert_eq![
        frames.last(),
        Some(&AssistantMessageFrame::ThinkingEnd {
            content_index: 0,
            content: String::from("abcd"),
            thinking_signature: Some(String::from("sig")),
            redacted: Some(true),
        })
    ];
    let reduced = reduce_assistant_message_frames(frames)
        .expect("the frames replay")
        .expect("a message");
    assert_eq!(
        reduced.content,
        vec![AssistantBlock::Thinking(ThinkingContent {
            thinking: String::from("abcd"),
            thinking_signature: Some(String::from("sig")),
            redacted: Some(true),
        })]
    );
}

#[test]
fn an_empty_toolcall_delta_after_the_snapshot_produces_no_frame() {
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::ToolCall(tool_call(
            "read",
            &serde_json::json!({}),
        )));
    });
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial,
        },
    );
    // Empty arguments at start are already caught up: an empty delta
    // produces no frame, a real one streams as-is.
    assert_eq![
        frame_opt(
            &mut encoder,
            AssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: String::new(),
                partial: seed(),
            },
        ),
        None
    ];
    assert_eq![
        frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: String::from("{\"path\": \"a\""),
                partial: seed(),
            },
        ),
        AssistantMessageFrame::ToolcallDelta {
            content_index: 0,
            delta: String::from("{\"path\": \"a\""),
        }
    ];
}

#[test]
fn legacy_tool_json_catches_up_through_array_and_scalar_prefix_checks() {
    // The snapshot carries an array; the delta extends it and the prefix
    // check walks the array elements.
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::ToolCall(tool_call(
            "read",
            &serde_json::json!({"list": [1]}),
        )));
    });
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial,
        },
    );
    assert_eq![
        frame_opt(
            &mut encoder,
            AssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: String::from("{\"list\": [1,2"),
                partial: seed(),
            },
        ),
        Some(AssistantMessageFrame::ToolcallCheckpoint {
            content_index: 0,
            json: String::from("{\"list\": [1,2"),
        })
    ];

    // Scalar snapshot entries compare through the equality arm.
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::ToolCall(tool_call(
            "read",
            &serde_json::json!({"n": 1}),
        )));
    });
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial,
        },
    );
    assert_eq![
        frame_opt(
            &mut encoder,
            AssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: String::from("{\"n\":1,\"extra\":true"),
                partial: seed(),
            },
        ),
        Some(AssistantMessageFrame::ToolcallCheckpoint {
            content_index: 0,
            json: String::from("{\"n\":1,\"extra\":true"),
        })
    ];
}

#[test]
fn rejects_deltas_before_their_block_started_or_of_the_wrong_kind() {
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::Text(text("hi")));
        partial
            .content
            .push(AssistantBlock::Thinking(thinking("ho")));
    });

    let error = encoder
        .encode(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: String::from("x"),
            partial: partial.clone(),
        })
        .expect_err("the text block never started");
    assert![error.contains("text block 0 has not started"), "{error}"];

    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
    );
    let _ = frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingStart {
            content_index: 1,
            partial: partial.clone(),
        },
    );

    let error = encoder
        .encode(AssistantMessageEvent::TextDelta {
            content_index: 1,
            delta: String::from("x"),
            partial: partial.clone(),
        })
        .expect_err("index 1 is a thinking block");
    assert![error.contains("block 1 is thinking, not text"), "{error}"];

    // Starting the same block twice fails the duplicate check.
    let error = encoder
        .encode(AssistantMessageEvent::TextStart {
            content_index: 0,
            partial,
        })
        .expect_err("the text block already started");
    assert![error.contains("block 0 starts more than once"), "{error}"];
}

#[test]
fn reduces_reject_malformed_frame_sequences() {
    // A second start frame fails the sequence.
    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::Start { partial: seed() },
    ])
    .expect_err("two start frames");
    assert![error.contains("more than one start frame"), "{error}"];

    // A delta at an index with no block fails.
    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::TextDelta {
            content_index: 5,
            delta: String::from("x"),
        },
    ])
    .expect_err("the index points at nothing");
    assert![
        error.contains("text_delta frame has no started block at index 5"),
        "{error}"
    ];

    // A start frame whose partial already carries content leaves the block
    // without replay state: the end frame finds the block but no state.
    let mut seeded = seed();
    seeded.content.push(AssistantBlock::Text(text("existing")));
    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::Start { partial: seeded },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::from("x"),
            text_signature: None,
        },
    ])
    .expect_err("the block was never started through frames");
    assert![
        error.contains("text_end frame has no started block at index 0"),
        "{error}"
    ];

    // Starting an existing index reports the collision.
    let error = reduce_assistant_message_frames(vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: text("a"),
        },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: text("b"),
        },
    ])
    .expect_err("the block exists already");
    assert![error.contains("already exists"), "{error}"];
}

#[test]
fn a_checkpoint_frame_replays_the_parsed_arguments() {
    let frames = vec![
        AssistantMessageFrame::Start { partial: seed() },
        AssistantMessageFrame::ToolcallStart {
            content_index: 0,
            tool_call: tool_call("read", &serde_json::json!({})),
        },
        AssistantMessageFrame::ToolcallCheckpoint {
            content_index: 0,
            json: String::from("{\"path\": \"a\""),
        },
    ];
    let reduced = reduce_assistant_message_frames(frames)
        .expect("the frames replay")
        .expect("a message");
    match &reduced.content[0] {
        AssistantBlock::ToolCall(call) => {
            assert_eq![
                &call.arguments,
                serde_json::json!({"path": "a"})
                    .as_object()
                    .expect("an object")
            ];
        }
        other => panic!("expected a tool call block, got {other:?}"),
    }
}

#[test]
fn rejects_start_events_pointing_at_toolcall_blocks() {
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::ToolCall(tool_call(
            "read",
            &serde_json::json!({}),
        )));
    });

    let error = encoder
        .encode(AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        })
        .expect_err("text_start at a toolcall block");
    assert![
        error.contains("text_start event points to toolCall block at index 0"),
        "{error}"
    ];

    let error = encoder
        .encode(AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial,
        })
        .expect_err("thinking_start at a toolcall block");
    assert![
        error.contains("thinking_start event points to toolCall block at index 0"),
        "{error}"
    ];
}

#[test]
fn rejects_a_text_end_pointing_at_a_toolcall_block() {
    let (partial, mut encoder) = started_encoder(|partial| {
        partial.content.push(AssistantBlock::ToolCall(tool_call(
            "read",
            &serde_json::json!({}),
        )));
    });
    let error = encoder
        .encode(AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: String::from("hi"),
            partial,
        })
        .expect_err("text_end at a toolcall block");
    assert![
        error.contains("text_end event points to toolCall block at index 0"),
        "{error}"
    ];
}
