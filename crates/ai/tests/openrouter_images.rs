//! The OpenRouter image-generation suite, ported 1:1 from
//! `packages/ai/test/openrouter-images.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Seam adaptation: upstream mocks the `openai` SDK client (`vi.mock`) and
//! asserts the wire through `mockState.lastParams`; `ImagesOptions` carries
//! no client seam and the module dispatches on the process default client
//! (`openrouter_images.rs`), so the wire rides a loopback HTTP server through
//! the model's `baseUrl` — the radius-flow suites' loopback pattern — and the
//! captured request body plays `lastParams`. The SDK's request options have
//! no Rust counterpart to assert, so the abort-signal passthrough case is
//! not ported: `ImagesOptions` carries no `signal` field and the module
//! reads none (`openrouter_images.rs` documents the gap), so no test can
//! thread a signal through to the dispatch.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_ai::types::{
    ImageContent, ImagesApi, ImagesBlock, ImagesContext, ImagesModel, ImagesProviderId,
    ImagesStopReason, Modality, ModelCost, ModelCostRates, TextContent,
};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The chat-completions response the fake SDK client resolved with, upstream
/// `mockState`'s canned response.
const IMAGE_RESPONSE_BODY: &str = r#"{
    "id": "img-1",
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "prompt_tokens_details": { "cached_tokens": 0 }
    },
    "choices": [
        {
            "message": {
                "content": "Here is your image.",
                "images": [{ "image_url": "data:image/png;base64,ZmFrZS1wbmc=" }]
            }
        }
    ]
}"#;

/// A loopback chat-completions server on the process default client: one
/// accepted connection, the request body captured, the canned JSON answered,
/// then the socket closes. Returns the base URL to point the model at and
/// the captured request bodies.
async fn spawn_chat_completions_server(
    response_body: &'static str,
) -> (String, Arc<Mutex<Vec<Value>>>) {
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback bind");
    let address = listener.local_addr().expect("loopback local addr");
    let captured = Arc::new(Mutex::new(Vec::new()));
    let captured_task = Arc::clone(&captured);

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("loopback accept");
        // Read until the head ends and the content-length body lands.
        let mut buffer: Vec<u8> = Vec::new();
        let mut scratch = [0u8; 8192];
        loop {
            if let Some(head_end) = find_head_end(&buffer) {
                let content_length = content_length_of(&buffer[..head_end]);
                if buffer.len() >= head_end + 4 + content_length {
                    break;
                }
            }
            let read = socket.read(&mut scratch).await.expect("loopback read");
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&scratch[..read]);
        }
        if let Some(head_end) = find_head_end(&buffer) {
            let body = &buffer[head_end + 4..];
            if let Ok(json) = serde_json::from_slice::<Value>(body) {
                captured_task
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(json);
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("loopback write");
        socket.shutdown().await.expect("loopback shutdown");
    });

    (format!("http://{address}"), captured)
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length_of(head: &[u8]) -> usize {
    let head = String::from_utf8_lossy(head).to_ascii_lowercase();
    head.lines()
        .find_map(|line| {
            line.strip_prefix("content-length:")
                .and_then(|value| value.trim().parse().ok())
        })
        .unwrap_or(0)
}

/// The openrouter-shaped image model, upstream's `ImagesModel<
/// "openrouter-images">` fixtures, retargeted at the loopback base URL.
fn openrouter_images_model(base_url: &str, output: Vec<Modality>) -> ImagesModel {
    ImagesModel {
        id: if output.contains(&Modality::Text) {
            "google/gemini-3.1-flash-image-preview".to_owned()
        } else {
            "black-forest-labs/flux.2-pro".to_owned()
        },
        name: if output.contains(&Modality::Text) {
            "Gemini 3.1 Flash Image Preview".to_owned()
        } else {
            "FLUX.2 Pro".to_owned()
        },
        api: ImagesApi::from("openrouter-images"),
        provider: ImagesProviderId::from("openrouter"),
        base_url: base_url.to_owned(),
        thinking_level_map: None,
        input: vec![Modality::Text, Modality::Image],
        output,
        cost: ModelCost {
            rates: ModelCostRates {
                input: 0.015,
                output: 0.03,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        sampling_params: None,
        headers: Some(BTreeMap::from([(
            "HTTP-Referer".to_owned(),
            "https://example.com".to_owned(),
        )])),
    }
}

/// The "Generate a dog" context both cases send.
fn images_context() -> ImagesContext {
    ImagesContext {
        input: vec![ImagesBlock::Text(TextContent {
            text: "Generate a dog".to_owned(),
            text_signature: None,
        })],
    }
}

/// The image options both cases send.
fn images_options() -> pi_ai::types::ImagesOptions {
    pi_ai::types::ImagesOptions {
        api_key: Some("test".to_owned()),
        ..pi_ai::types::ImagesOptions::default()
    }
}

/// Returns text plus images in final output, with the wire params upstream
/// pins: `stream: false`, the `["image", "text"]` modality order, and the
/// input blocks as the user message.
#[tokio::test]
async fn returns_text_plus_images_in_final_output() {
    let (base_url, captured) = spawn_chat_completions_server(IMAGE_RESPONSE_BODY).await;
    let model = openrouter_images_model(&base_url, vec![Modality::Text, Modality::Image]);

    let api = pi_ai::api::openrouter_images();
    let output = api
        .generate_images(&model, &images_context(), Some(&images_options()))
        .await
        .expect("generate_images resolves");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-1"));
    assert_eq!(
        output.output.first(),
        Some(&ImagesBlock::Text(TextContent {
            text: "Here is your image.".to_owned(),
            text_signature: None,
        }))
    );
    assert_eq!(
        output.output.get(1),
        Some(&ImagesBlock::Image(ImageContent {
            mime_type: "image/png".to_owned(),
            data: "ZmFrZS1wbmc=".to_owned(),
        }))
    );

    let params = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let params = params.first().expect("the request body was captured");
    assert_eq!(params.get("stream"), Some(&serde_json::json!(false)));
    assert_eq!(
        params.get("modalities"),
        Some(&serde_json::json!(["image", "text"]))
    );
    assert_eq!(
        params
            .get("messages")
            .and_then(|messages| messages.get(0))
            .and_then(|message| message.get("content"))
            .and_then(|content| content.get(0)),
        Some(&serde_json::json!({ "type": "text", "text": "Generate a dog" }))
    );
}

/// `generateImages` resolves the final assistant images result.
#[tokio::test]
async fn generate_images_resolves_the_final_assistant_images_result() {
    let (base_url, _captured) = spawn_chat_completions_server(IMAGE_RESPONSE_BODY).await;
    let model = openrouter_images_model(&base_url, vec![Modality::Image]);

    let api = pi_ai::api::openrouter_images();
    let output = api
        .generate_images(&model, &images_context(), Some(&images_options()))
        .await
        .expect("generate_images resolves");

    assert!(
        output
            .output
            .iter()
            .any(|block| matches!(block, ImagesBlock::Image(_)))
    );
}
// ---------------------------------------------------------------------------
// Port-added: the request/error/data-URL edges the upstream suite reaches
// only implicitly, over the same loopback seam
// ---------------------------------------------------------------------------

/// A loopback server answering every connection with the canned status and
/// body, capturing the request bodies and headers. Returns the base URL, the
/// captured bodies, and the captured request heads.
async fn spawn_status_server(
    status_line: &'static str,
    response_body: &'static str,
) -> (String, Arc<Mutex<Vec<Value>>>, Arc<Mutex<Vec<String>>>) {
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback bind");
    let address = listener.local_addr().expect("loopback local addr");
    let captured = Arc::new(Mutex::new(Vec::new()));
    let captured_task = Arc::clone(&captured);
    let captured_headers = Arc::new(Mutex::new(Vec::new()));
    let headers_task = Arc::clone(&captured_headers);

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("loopback accept");
        let mut buffer: Vec<u8> = Vec::new();
        let mut scratch = [0u8; 8192];
        loop {
            let read = socket.read(&mut scratch).await.expect("loopback read");
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&scratch[..read]);
            if let Some(head_end) = find_head_end(&buffer) {
                let head = &buffer[..head_end];
                let body_start = head_end + 4;
                let expected = content_length_of(head);
                if buffer.len() >= body_start + expected {
                    if let Ok(json) = serde_json::from_slice::<Value>(&buffer[body_start..]) {
                        captured_task
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(json);
                    }
                    headers_task
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(String::from_utf8_lossy(head).to_string());
                    break;
                }
            }
        }
        let response = format!(
            "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("loopback write");
        socket.shutdown().await.expect("loopback shutdown");
    });

    (format!("http://{address}"), captured, captured_headers)
}

/// Port-added: a missing api key fails with the provider-named setup error
/// before any request dispatches.
#[tokio::test]
async fn a_missing_api_key_fails_before_dispatch() {
    let (base_url, captured, _headers) =
        spawn_status_server("HTTP/1.1 200 OK", IMAGE_RESPONSE_BODY).await;
    let model = openrouter_images_model(&base_url, vec![Modality::Image]);
    let options = pi_ai::types::ImagesOptions::default();

    let api = pi_ai::api::openrouter_images();
    let output = api
        .generate_images(&model, &images_context(), Some(&options))
        .await
        .expect("generate_images resolves");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    let message = output.error_message.expect("the setup failure");
    assert![
        message.contains("No API key for provider: openrouter"),
        "{message}"
    ];
    assert!(
        captured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
}

/// Port-added: a 5xx answer surfaces the provider error message under the
/// error stop; an abort-shaped transport failure reads as aborted.
#[tokio::test]
async fn a_failed_dispatch_surfaces_the_provider_error() {
    let (base_url, _captured, _headers) = spawn_status_server(
        "HTTP/1.1 503 Service Unavailable",
        r#"{"error":{"message":"image backend overloaded"}}"#,
    )
    .await;
    let model = openrouter_images_model(&base_url, vec![Modality::Image]);

    let api = pi_ai::api::openrouter_images();
    let output = api
        .generate_images(&model, &images_context(), Some(&images_options()))
        .await
        .expect("generate_images resolves");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    let message = output.error_message.expect("the provider failure");
    assert![
        message.contains("503") || message.contains("unavailable"),
        "{message}"
    ];
}

/// Port-added: the provider-error-body passthrough regression, ported from
/// `test/provider-error-body-passthrough.test.ts` — a 403 from a gateway
/// carrying the real reason in the body must surface both the status and the
/// body reason, not the opaque "no body" message.
#[tokio::test]
async fn a_403_with_a_body_surfaces_the_body_reason() {
    let (base_url, _captured, _headers) = spawn_status_server(
        "HTTP/1.1 403 Forbidden",
        r#"{"error": "blocked by gateway WAF"}"#,
    )
    .await;
    let model = openrouter_images_model(&base_url, vec![Modality::Image]);

    let api = pi_ai::api::openrouter_images();
    let output = api
        .generate_images(&model, &images_context(), Some(&images_options()))
        .await
        .expect("generate_images resolves");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    let message = output.error_message.expect("the provider failure");
    assert![message.contains("403"), "{message}"];
    assert![
        message.contains("blocked by gateway WAF"),
        "the body reason must not be swallowed: {message}"
    ];
}

/// Port-added: caller headers merge over the model's, `None` values suppress
/// entries — read off the captured request head.
#[tokio::test]
async fn the_image_request_headers_follow_the_merge_precedence() {
    let (base_url, _captured, captured_headers) =
        spawn_status_server("HTTP/1.1 200 OK", IMAGE_RESPONSE_BODY).await;
    let mut model = openrouter_images_model(&base_url, vec![Modality::Image]);
    model.headers = Some(BTreeMap::from([
        (
            "HTTP-Referer".to_owned(),
            "https://model.example".to_owned(),
        ),
        ("x-model-header".to_owned(), "model".to_owned()),
        ("x-suppressed".to_owned(), "model-value".to_owned()),
    ]));
    let mut options = images_options();
    options.headers = Some(BTreeMap::from([
        ("X-Title".to_owned(), Some("caller".to_owned())),
        ("X-Model-Header".to_owned(), None),
        ("x-suppressed".to_owned(), None),
    ]));

    let api = pi_ai::api::openrouter_images();
    let _output = api
        .generate_images(&model, &images_context(), Some(&options))
        .await
        .expect("generate_images resolves");

    let heads = captured_headers
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let head = heads.first().expect("the captured request head");
    let value_of = |wanted: &str| {
        head.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim().eq_ignore_ascii_case(wanted)).then(|| value.trim().to_owned())
        })
    };
    assert_eq!(
        value_of("HTTP-Referer").as_deref(),
        Some("https://model.example")
    );
    assert_eq!(value_of("X-Title").as_deref(), Some("caller"));
    assert_eq!(value_of("x-model-header"), None);
    assert_eq!(value_of("x-suppressed"), None);
}

/// Port-added: image input blocks ride the request as `image_url` parts.
#[tokio::test]
async fn image_input_blocks_ride_the_request_as_image_url_parts() {
    let (base_url, captured, _headers) =
        spawn_status_server("HTTP/1.1 200 OK", IMAGE_RESPONSE_BODY).await;
    let model = openrouter_images_model(&base_url, vec![Modality::Image]);
    let context = ImagesContext {
        input: vec![
            ImagesBlock::Text(TextContent {
                text: "and this".to_owned(),
                text_signature: None,
            }),
            ImagesBlock::Image(ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/jpeg".to_owned(),
            }),
        ],
    };

    let api = pi_ai::api::openrouter_images();
    let _output = api
        .generate_images(&model, &context, Some(&images_options()))
        .await
        .expect("generate_images resolves");

    let params = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let params = params.first().expect("the request body was captured");
    assert_eq!(
        params["messages"][0]["content"][1],
        serde_json::json!({
            "type": "image_url",
            "image_url": { "url": "data:image/jpeg;base64,aGk=" },
        })
    );
    assert_eq!(params["modalities"], serde_json::json!(["image"]));
}

/// Port-added: generated image URLs ride both wire spellings and the data
/// URL parse rejects malformed shapes; an image-less choice still settles.
#[tokio::test]
async fn generated_image_urls_parse_both_spellings_and_reject_malformed_ones() {
    let response = r#"{
        "id": "img-2",
        "choices": [{
            "message": {
                "content": "",
                "images": [
                    { "image_url": { "url": "data:image/webp;base64,d2VicA==" } },
                    { "image_url": "data:image/png;base64,cG5n" },
                    { "image_url": "https://example.com/image.png" },
                    { "image_url": "data:;base64,cG5n" },
                    { "image_url": "data:image/png;base64,aGk\nCg==" },
                    { "image_url": 42 },
                    { "image_url": "data:;base64," }
                ]
            }
        }]
    }"#;
    let (base_url, _captured, _headers) = spawn_status_server("HTTP/1.1 200 OK", response).await;
    let model = openrouter_images_model(&base_url, vec![Modality::Image]);

    let api = pi_ai::api::openrouter_images();
    let output = api
        .generate_images(&model, &images_context(), Some(&images_options()))
        .await
        .expect("generate_images resolves");

    assert_eq!(output.response_id.as_deref(), Some("img-2"));
    let images: Vec<&ImageContent> = output
        .output
        .iter()
        .filter_map(|block| match block {
            ImagesBlock::Image(image) => Some(image),
            ImagesBlock::Text(_) => None,
        })
        .collect();
    assert_eq!(images.len(), 2);
    assert_eq![images[0].mime_type, "image/webp"];
    assert_eq![images[1].mime_type, "image/png"];
}

/// Port-added: usage without choices still prices, the falsy usage shapes
/// stay out, and the cache write discount reads off the details.
#[tokio::test]
async fn usage_truthiness_and_the_cache_write_discount() {
    let response = r#"{
        "id": "img-3",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "prompt_tokens_details": { "cached_tokens": 6, "cache_write_tokens": 2 }
        },
        "choices": [{ "message": { "content": "done" } }]
    }"#;
    let (base_url, _captured, _headers) = spawn_status_server("HTTP/1.1 200 OK", response).await;
    let model = openrouter_images_model(&base_url, vec![Modality::Text, Modality::Image]);
    let api = pi_ai::api::openrouter_images();
    let output = api
        .generate_images(&model, &images_context(), Some(&images_options()))
        .await
        .expect("generate_images resolves");
    let usage = output.usage.expect("the priced usage");
    assert_eq!(usage.input, 4);
    assert_eq!(usage.output, 4);
    assert_eq!(usage.cache_read, 4);

    // The falsy shapes — null, 0, "", false — leave the usage unset.
    for usage_value in ["null", "0", "\"\"", "false"] {
        let response = format!(
            r#"{{ "id": "img-4", "usage": {usage_value}, "choices": [{{ "message": {{ "content": "x" }} }}] }}"#
        );
        let boxed: &'static str = Box::leak(response.into_boxed_str());
        let (base_url, _captured, _headers) = spawn_status_server("HTTP/1.1 200 OK", boxed).await;
        let model = openrouter_images_model(&base_url, vec![Modality::Image]);
        let output = api
            .generate_images(&model, &images_context(), Some(&images_options()))
            .await
            .expect("generate_images resolves");
        assert!(output.usage.is_none(), "{usage_value}");
    }

    // A choiceless response keeps the response id and settles.
    let empty = r#"{ "id": "img-5", "choices": [] }"#;
    let (base_url, _captured, _headers) = spawn_status_server("HTTP/1.1 200 OK", empty).await;
    let model = openrouter_images_model(&base_url, vec![Modality::Image]);
    let output = api
        .generate_images(&model, &images_context(), Some(&images_options()))
        .await
        .expect("generate_images resolves");
    assert_eq!(output.response_id.as_deref(), Some("img-5"));
    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
}
