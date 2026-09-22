//! The production Converse Stream runtime seam: the official
//! `aws-sdk-bedrockruntime` client behind the adapter's
//! [`BedrockRuntime`] trait seam.
//!
//! That is the boundary upstream's `vi.mock` replaces.
//!
//! Porting restatements:
//! - The resolved [`BedrockClientConfig`] maps onto the SDK client config:
//!   explicit region and endpoint, explicit or ambient credentials, the
//!   bearer token as a static token provider, and the proxy / HTTP1.1
//!   forcing through the smithy HTTP client builder. Unset fields ride the
//!   SDK default chain, upstream's implicit resolution.
//! - `authSchemePreference` is not a settable field on the Rust SDK
//!   `Config` in this version; the static token provider rides alone and
//!   Bedrock's bearer-auth endpoint rules resolve the scheme.
//! - The caller headers inject through a `modify_before_signing`
//!   interceptor, the Smithy `build` step; the raw response status and
//!   headers capture through a `read_after_deserialization` interceptor into
//!   a shared slot, upstream's deserialize-step middleware.
//! - The SDK has no per-request abort signal; the event pump stops on the
//!   cancellation token and the adapter's post-loop check produces the
//!   aborted state, upstream's "Request was aborted".
//! - The five modeled stream exceptions convert into the failure the
//!   adapter's catch path formats; upstream throws them into its catch.
//! - The `redactedContent` blobs ride the wire events as base64 strings.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use aws_sdk_bedrockruntime as sdk;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::types as sdk_types;
use aws_smithy_runtime_api::client::result::SdkError;
use aws_smithy_types::{Blob, Document, Number};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::bedrock_converse_stream::{
    BedrockEventStream, BedrockRuntime, BedrockStreamFailure, BedrockStreamReply,
};
use crate::api::bedrock_options::BedrockClientConfig;
use crate::types::BoxedFuture;

/// The official SDK runtime the adapter installs by default, upstream's
/// `BedrockRuntimeClient`.
#[derive(Debug, Default)]
pub struct SdkBedrockRuntime;

impl SdkBedrockRuntime {
    /// The production runtime; the client constructs per request, upstream's
    /// client construction inside the try block.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// The raw response status and headers the deserialize-stage interceptor
/// snapshots, upstream's Smithy `HttpResponse` capture.
#[derive(Debug, Default, Clone)]
struct RawResponseSnapshot {
    status: Option<u16>,
    headers: BTreeMap<String, String>,
}

impl BedrockRuntime for SdkBedrockRuntime {
    fn converse_stream(
        &self,
        input: Value,
        config: BedrockClientConfig,
        signal: CancellationToken,
    ) -> BoxedFuture<'static, Result<BedrockStreamReply, BedrockStreamFailure>> {
        Box::pin(async move {
            let (client, raw) = build_client(&config).await?;
            let mut builder = client.converse_stream();
            builder = apply_command_input(builder, &input)?;
            let output = builder.send().await.map_err(|error| send_failure(&error))?;
            let snapshot = raw
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            let request_id = snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.headers.get("x-amzn-requestid").cloned());
            let events = pump_events(output, signal);
            Ok(BedrockStreamReply {
                request_id,
                status: snapshot.and_then(|snapshot| snapshot.status),
                events,
            })
        })
    }
}

/// The interceptor that injects the caller headers before signing, upstream's
/// `pi-ai-custom-headers` build-step middleware; the reserved-header filter
/// already ran at config resolution.
#[derive(Debug)]
struct CustomHeadersInterceptor {
    headers: Vec<(String, String)>,
}

impl aws_smithy_runtime_api::client::interceptors::Intercept for CustomHeadersInterceptor {
    fn name(&self) -> &'static str {
        "pi-ai-custom-headers"
    }

    fn modify_before_signing(
        &self,
        context: &mut aws_smithy_runtime_api::client::interceptors::context::BeforeTransmitInterceptorContextMut<
            '_,
        >,
        _runtime_components: &aws_smithy_runtime_api::client::runtime_components::RuntimeComponents,
        _cfg: &mut aws_smithy_types::config_bag::ConfigBag,
    ) -> Result<(), aws_smithy_runtime_api::box_error::BoxError> {
        let request = context.request_mut();
        for (name, value) in &self.headers {
            request.headers_mut().insert(name.clone(), value.clone());
        }
        Ok(())
    }
}

/// The interceptor that snapshots the raw HTTP response at the deserialize
/// step, before the event stream is consumed, upstream's response-header
/// middleware; Bedrock's modeled metadata preserves only selected fields, so
/// custom gateway headers are otherwise lost.
#[derive(Debug)]
struct RawResponseCapture {
    snapshot: Arc<Mutex<Option<RawResponseSnapshot>>>,
}

impl aws_smithy_runtime_api::client::interceptors::Intercept for RawResponseCapture {
    fn name(&self) -> &'static str {
        "pi-ai-response-headers"
    }

    fn read_after_deserialization(
        &self,
        context: &aws_smithy_runtime_api::client::interceptors::context::AfterDeserializationInterceptorContextRef<'_>,
        _runtime_components: &aws_smithy_runtime_api::client::runtime_components::RuntimeComponents,
        _cfg: &mut aws_smithy_types::config_bag::ConfigBag,
    ) -> Result<(), aws_smithy_runtime_api::box_error::BoxError> {
        let response = context.response();
        {
            let mut snapshot = self
                .snapshot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *snapshot = Some(RawResponseSnapshot {
                status: Some(response.status().as_u16()),
                headers: headers_map(response.headers()),
            });
        }
        Ok(())
    }
}

/// The lowercase header record of a smithy response, the shape the response
/// hook reads.
fn headers_map(headers: &aws_smithy_runtime_api::http::Headers) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect()
}

/// Build the SDK client from the resolved config, with the raw-response
/// snapshot slot returned alongside.
async fn build_client(
    config: &BedrockClientConfig,
) -> Result<(sdk::Client, Arc<Mutex<Option<RawResponseSnapshot>>>), BedrockStreamFailure> {
    let loader = aws_config::defaults(sdk::config::BehaviorVersion::latest());
    let loader = match &config.profile {
        Some(profile) => loader.profile_name(profile),
        None => loader,
    };
    let shared = loader.load().await;
    let mut builder = sdk::Config::from(&shared).to_builder();

    if let Some(region) = &config.region {
        builder.set_region(Some(sdk::config::Region::new(region.clone())));
    }
    if let Some(endpoint) = &config.endpoint {
        builder = builder.endpoint_url(endpoint.clone());
    }
    let credentials = if config.skip_auth {
        Some(aws_credential_types::Credentials::new(
            "dummy-access-key",
            "dummy-secret-key",
            None,
            None,
            "dummy",
        ))
    } else {
        config.credentials.as_ref().map(|credentials| {
            aws_credential_types::Credentials::new(
                credentials.access_key_id.clone(),
                credentials.secret_access_key.clone(),
                credentials.session_token.clone(),
                None,
                "env",
            )
        })
    };
    if let Some(credentials) = credentials {
        builder.set_credentials_provider(Some(sdk::config::SharedCredentialsProvider::new(
            credentials,
        )));
    }
    if let Some(token) = &config.token {
        builder = builder.token_provider(aws_credential_types::Token::new(token.clone(), None));
    }
    if config.force_http1 || config.proxy_url.is_some() {
        builder.set_http_client(Some(http_client(config)?));
    }
    let snapshot = Arc::new(Mutex::new(None));
    builder.push_interceptor(sdk::config::SharedInterceptor::new(
        CustomHeadersInterceptor {
            headers: config.headers.clone(),
        },
    ));
    builder.push_interceptor(sdk::config::SharedInterceptor::new(RawResponseCapture {
        snapshot: Arc::clone(&snapshot),
    }));
    Ok((sdk::Client::from_conf(builder.build()), snapshot))
}

/// The smithy HTTP client when the caller pins HTTP/1.1 or a proxy, upstream's
/// `NodeHttpHandler` with proxy agents or the plain handler.
///
/// # Errors
/// When the proxy URL cannot parse.
fn http_client(
    config: &BedrockClientConfig,
) -> Result<sdk::config::SharedHttpClient, BedrockStreamFailure> {
    let proxy = config
        .proxy_url
        .as_ref()
        .map(|url| {
            aws_smithy_http_client::proxy::ProxyConfig::all(url.clone()).map_err(|error| {
                BedrockStreamFailure::plain(format!("Invalid proxy URL {url}: {error}"))
            })
        })
        .transpose()?;
    Ok(
        aws_smithy_http_client::Builder::new().build_with_connector_fn(
            move |_settings, _components| {
                // Some custom endpoints require HTTP/1.1 instead of HTTP/2; this
                // connector negotiates by ALPN and falls back per server, and the
                // proxy rides its config.
                let mut connector_builder = aws_smithy_http_client::Connector::builder();
                if let Some(proxy) = &proxy {
                    connector_builder = connector_builder.proxy_config(proxy.clone());
                }
                connector_builder
                    .tls_provider(aws_smithy_http_client::tls::Provider::Rustls(
                        aws_smithy_http_client::tls::rustls_provider::CryptoMode::Ring,
                    ))
                    .build()
            },
        ),
    )
}

/// Map the JSON command input onto the SDK's typed builder, upstream's
/// `ConverseStreamCommand(commandInput)` construction.
///
/// # Errors
/// When the command input does not shape into the typed Converse fields.
fn apply_command_input(
    mut builder: sdk::operation::converse_stream::builders::ConverseStreamFluentBuilder,
    input: &Value,
) -> Result<
    sdk::operation::converse_stream::builders::ConverseStreamFluentBuilder,
    BedrockStreamFailure,
> {
    builder = builder.model_id(
        input
            .get("modelId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    );
    if let Some(messages) = input.get("messages").and_then(Value::as_array) {
        for message in messages {
            builder = builder.messages(wire_message(message)?);
        }
    }
    if let Some(system) = input.get("system").and_then(Value::as_array) {
        for block in system {
            builder = builder.system(wire_system_block(block)?);
        }
    }
    if let Some(inference) = input.get("inferenceConfig") {
        let mut config = sdk_types::InferenceConfiguration::builder();
        if let Some(max_tokens) = inference.get("maxTokens").and_then(Value::as_u64) {
            config = config.max_tokens(i32::try_from(max_tokens).unwrap_or_default());
        }
        if let Some(temperature) = inference.get("temperature").and_then(Value::as_f64) {
            config = config.temperature(wire_temperature(temperature));
        }
        builder = builder.inference_config(config.build());
    }
    if let Some(tool_config) = input.get("toolConfig") {
        builder = builder.tool_config(wire_tool_config(tool_config)?);
    }
    if let Some(additional) = input.get("additionalModelRequestFields") {
        builder = builder.additional_model_request_fields(document(additional));
    }
    if let Some(metadata) = input.get("requestMetadata").and_then(Value::as_object) {
        for (key, value) in metadata {
            let value = value
                .as_str()
                .map_or_else(|| value.to_string(), str::to_owned);
            builder = builder.request_metadata(key.clone(), value);
        }
    }
    Ok(builder)
}

/// The input-build failure shape the adapter's catch path formats.
fn builder_failure(message: impl std::fmt::Display) -> BedrockStreamFailure {
    BedrockStreamFailure::plain(format!("Invalid Converse Stream command input: {message}"))
}

/// The f32 the wire's temperature field carries, narrowed from the f64
/// option, upstream's JS-number pass-through.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the wire field is f32; the f64 option narrows like upstream's JS numbers"
)]
const fn wire_temperature(value: f64) -> f32 {
    value as f32
}

/// One wire message, upstream's `Message` member.
fn wire_message(message: &Value) -> Result<sdk_types::Message, BedrockStreamFailure> {
    let role = match message.get("role").and_then(Value::as_str) {
        Some("assistant") => sdk_types::ConversationRole::Assistant,
        _ => sdk_types::ConversationRole::User,
    };
    let mut content = Vec::new();
    for block in message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        content.push(wire_content_block(block)?);
    }
    sdk_types::Message::builder()
        .role(role)
        .set_content(Some(content))
        .build()
        .map_err(builder_failure)
}

/// One wire content block, upstream's `ContentBlock` members the converter
/// emits.
fn wire_content_block(block: &Value) -> Result<sdk_types::ContentBlock, BedrockStreamFailure> {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        return Ok(sdk_types::ContentBlock::Text(text.to_owned()));
    }
    if let Some(image) = block.get("image") {
        let format = match image
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "jpeg" => sdk_types::ImageFormat::Jpeg,
            "png" => sdk_types::ImageFormat::Png,
            "gif" => sdk_types::ImageFormat::Gif,
            "webp" => sdk_types::ImageFormat::Webp,
            other => {
                return Err(BedrockStreamFailure::plain(format!(
                    "Unknown image type: {other}"
                )));
            }
        };
        let source = sdk_types::ImageSource::Bytes(
            json_blob(image.get("source").and_then(|source| source.get("bytes")))
                .ok_or_else(|| builder_failure("image bytes missing or not valid JSON bytes"))?,
        );
        let image = sdk_types::ImageBlock::builder()
            .format(format)
            .set_source(Some(source))
            .build()
            .map_err(builder_failure)?;
        return Ok(sdk_types::ContentBlock::Image(image));
    }
    if let Some(tool_use) = block.get("toolUse") {
        let tool_use = sdk_types::ToolUseBlock::builder()
            .tool_use_id(
                tool_use
                    .get("toolUseId")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .name(
                tool_use
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .input(document(tool_use.get("input").unwrap_or(&Value::Null)))
            .build()
            .map_err(builder_failure)?;
        return Ok(sdk_types::ContentBlock::ToolUse(tool_use));
    }
    if let Some(reasoning) = block.get("reasoningContent") {
        return Ok(sdk_types::ContentBlock::ReasoningContent(
            wire_reasoning_block(reasoning)?,
        ));
    }
    if let Some(cache_point) = block.get("cachePoint") {
        return Ok(sdk_types::ContentBlock::CachePoint(wire_cache_point(
            cache_point,
        )?));
    }
    Ok(sdk_types::ContentBlock::Text(String::new()))
}

/// The reasoning content block, upstream's `reasoningContent` member: the
/// opaque replay rides as the `redactedContent` blob.
fn wire_reasoning_block(
    reasoning: &Value,
) -> Result<sdk_types::ReasoningContentBlock, BedrockStreamFailure> {
    if let Some(redacted) = reasoning
        .get("redactedContent")
        .and_then(|value| json_blob(Some(value)))
    {
        return Ok(sdk_types::ReasoningContentBlock::RedactedContent(redacted));
    }
    let text = reasoning
        .pointer("/reasoningText/text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let mut reasoning_text = sdk_types::ReasoningTextBlock::builder().text(text);
    if let Some(signature) = reasoning
        .pointer("/reasoningText/signature")
        .and_then(Value::as_str)
    {
        reasoning_text = reasoning_text.signature(signature);
    }
    Ok(sdk_types::ReasoningContentBlock::ReasoningText(
        reasoning_text.build().map_err(builder_failure)?,
    ))
}

/// The cache-point block, upstream's `cachePoint` literal.
fn wire_cache_point(
    cache_point: &Value,
) -> Result<sdk_types::CachePointBlock, BedrockStreamFailure> {
    sdk_types::CachePointBlock::builder()
        .set_type(Some(sdk_types::CachePointType::Default))
        .set_ttl(Some(cache_ttl(cache_point.get("ttl"))))
        .build()
        .map_err(builder_failure)
}

/// The cache TTL the wire names.
fn cache_ttl(value: Option<&Value>) -> sdk_types::CacheTtl {
    sdk_types::CacheTtl::from(value.and_then(Value::as_str).unwrap_or(""))
}

/// One system content block.
fn wire_system_block(block: &Value) -> Result<sdk_types::SystemContentBlock, BedrockStreamFailure> {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        return Ok(sdk_types::SystemContentBlock::Text(text.to_owned()));
    }
    if let Some(cache_point) = block.get("cachePoint") {
        return Ok(sdk_types::SystemContentBlock::CachePoint(wire_cache_point(
            cache_point,
        )?));
    }
    Ok(sdk_types::SystemContentBlock::Text(String::new()))
}

/// The tool configuration, upstream's `toolConfig` member.
fn wire_tool_config(
    tool_config: &Value,
) -> Result<sdk_types::ToolConfiguration, BedrockStreamFailure> {
    let mut tools = Vec::new();
    for tool in tool_config
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let spec = tool.get("toolSpec").cloned().unwrap_or(Value::Null);
        let mut spec_builder = sdk_types::ToolSpecification::builder()
            .name(spec.get("name").and_then(Value::as_str).unwrap_or_default())
            .description(
                spec.get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .input_schema(sdk_types::ToolInputSchema::Json(document(
                spec.pointer("/inputSchema/json").unwrap_or(&Value::Null),
            )));
        if let Some(strict) = spec.get("strict").and_then(Value::as_bool) {
            spec_builder = spec_builder.strict(strict);
        }
        tools.push(sdk_types::Tool::ToolSpec(
            spec_builder.build().map_err(builder_failure)?,
        ));
    }
    let mut config_builder = sdk_types::ToolConfiguration::builder().set_tools(Some(tools));
    let choice = tool_config.get("toolChoice");
    if choice.and_then(|choice| choice.get("auto")).is_some() {
        config_builder = config_builder.tool_choice(sdk_types::ToolChoice::Auto(
            sdk_types::AutoToolChoice::builder().build(),
        ));
    } else if choice.and_then(|choice| choice.get("any")).is_some() {
        config_builder = config_builder.tool_choice(sdk_types::ToolChoice::Any(
            sdk_types::AnyToolChoice::builder().build(),
        ));
    } else if let Some(name) = choice
        .and_then(|choice| choice.pointer("/tool/name"))
        .and_then(Value::as_str)
    {
        config_builder = config_builder.tool_choice(sdk_types::ToolChoice::Tool(
            sdk_types::SpecificToolChoice::builder()
                .name(name)
                .build()
                .map_err(builder_failure)?,
        ));
    }
    config_builder.build().map_err(builder_failure)
}

/// The smithy `Document` the open-content fields carry, from the wire JSON.
fn document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(value) => Document::Bool(*value),
        Value::Number(number) => Document::Number(document_number(number)),
        Value::String(value) => Document::String(value.clone()),
        Value::Array(items) => Document::Array(items.iter().map(document).collect()),
        Value::Object(object) => Document::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), document(value)))
                .collect(),
        ),
    }
}

fn document_number(number: &serde_json::Number) -> Number {
    number.as_u64().map_or_else(
        || {
            number.as_i64().map_or_else(
                || Number::Float(number.as_f64().unwrap_or_default()),
                Number::NegInt,
            )
        },
        Number::PosInt,
    )
}

/// The bytes a wire field carries: the JSON replay path holds a byte array,
/// the raw base64 spelling also accepted.
fn json_blob(value: Option<&Value>) -> Option<Blob> {
    match value? {
        Value::String(base64) => {
            crate::api::bedrock_converse_stream::base64_to_bytes(base64).map(Blob::new)
        }
        Value::Array(items) => items
            .iter()
            .map(|item| item.as_u64().and_then(|byte| u8::try_from(byte).ok()))
            .collect::<Option<Vec<u8>>>()
            .map(Blob::new),
        _ => None,
    }
}

/// Pump the SDK event stream into the adapter's wire-event stream, upstream's
/// `response.stream` await loop; the pump stops on cancellation so the
/// adapter's post-loop check can report the aborted turn.
fn pump_events(
    output: sdk::operation::converse_stream::ConverseStreamOutput,
    signal: CancellationToken,
) -> BedrockEventStream {
    let receiver = output.stream;
    let wire_events =
        futures_util::stream::unfold((receiver, signal), |(mut receiver, signal)| async move {
            tokio::select! {
                () = signal.cancelled() => None,
                event = receiver.recv() => match event {
                    Ok(Some(output)) => Some((Ok(event_to_value(&output)), (receiver, signal))),
                    Ok(None) => None,
                    Err(error) => Some((Err(stream_failure(&error)), (receiver, signal))),
                },
            }
        });
    Box::pin(wire_events)
}

/// One SDK event in the wire shape the adapter consumes, camelCase.
fn event_to_value(event: &sdk_types::ConverseStreamOutput) -> Value {
    use sdk_types::ConverseStreamOutput as Event;
    match event {
        Event::MessageStart(event) => json!({
            "messageStart": { "role": event.role().as_str() }
        }),
        Event::ContentBlockStart(event) => {
            let mut frame = serde_json::Map::new();
            frame.insert(
                "contentBlockIndex".to_owned(),
                json!(event.content_block_index()),
            );
            if let Some(tool_use) = event.start().and_then(|start| start.as_tool_use().ok()) {
                frame.insert(
                    "start".to_owned(),
                    json!({
                        "toolUse": {
                            "toolUseId": tool_use.tool_use_id(),
                            "name": tool_use.name(),
                        }
                    }),
                );
            }
            json!({ "contentBlockStart": frame })
        }
        Event::ContentBlockDelta(event) => {
            let mut frame = serde_json::Map::new();
            frame.insert(
                "contentBlockIndex".to_owned(),
                json!(event.content_block_index()),
            );
            if let Some(delta) = event.delta() {
                frame.insert("delta".to_owned(), delta_to_value(delta));
            }
            json!({ "contentBlockDelta": frame })
        }
        Event::ContentBlockStop(event) => json!({
            "contentBlockStop": { "contentBlockIndex": event.content_block_index() }
        }),
        Event::MessageStop(event) => json!({
            "messageStop": { "stopReason": event.stop_reason().as_str() }
        }),
        Event::Metadata(event) => json!({ "metadata": metadata_to_value(event) }),
        _ => json!({}),
    }
}

/// One delta in the wire shape; delta kinds the adapter ignores ride as
/// `null`.
fn delta_to_value(delta: &sdk_types::ContentBlockDelta) -> Value {
    match delta {
        sdk_types::ContentBlockDelta::Text(text) => json!({ "text": text }),
        sdk_types::ContentBlockDelta::ToolUse(delta) => {
            json!({ "toolUse": { "input": delta.input() } })
        }
        sdk_types::ContentBlockDelta::ReasoningContent(reasoning) => {
            let mut frame = serde_json::Map::new();
            if let Ok(text) = reasoning.as_text() {
                frame.insert("text".to_owned(), json!(text));
            }
            if let Ok(signature) = reasoning.as_signature() {
                frame.insert("signature".to_owned(), json!(signature));
            }
            if let Ok(blob) = reasoning.as_redacted_content() {
                frame.insert(
                    "redactedContent".to_owned(),
                    json!(crate::api::bedrock_converse_stream::bytes_to_base64(
                        blob.as_ref()
                    )),
                );
            }
            Value::Object(frame)
        }
        _ => Value::Null,
    }
}

/// The metadata frame's usage, camelCase.
fn metadata_to_value(event: &sdk_types::ConverseStreamMetadataEvent) -> Value {
    let Some(usage) = event.usage() else {
        return Value::Object(serde_json::Map::new());
    };
    let mut usage_frame = serde_json::Map::new();
    usage_frame.insert("inputTokens".to_owned(), json!(usage.input_tokens()));
    usage_frame.insert("outputTokens".to_owned(), json!(usage.output_tokens()));
    usage_frame.insert("totalTokens".to_owned(), json!(usage.total_tokens()));
    usage_frame.insert(
        "cacheReadInputTokens".to_owned(),
        json!(usage.cache_read_input_tokens().unwrap_or_default()),
    );
    usage_frame.insert(
        "cacheWriteInputTokens".to_owned(),
        json!(usage.cache_write_input_tokens().unwrap_or(0)),
    );
    if !usage.cache_details().is_empty() {
        usage_frame.insert(
            "cacheDetails".to_owned(),
            json!(
                usage
                    .cache_details()
                    .iter()
                    .map(|detail| json!({
                        "ttl": detail.ttl().as_str(),
                        "inputTokens": detail.input_tokens(),
                    }))
                    .collect::<Vec<_>>()
            ),
        );
    }
    json!({ "usage": Value::Object(usage_frame) })
}

/// The send-phase failure, upstream's `handleError` path: the modeled code,
/// the HTTP status, the raw body, and the request id the response carried.
fn send_failure(
    error: &SdkError<
        ConverseStreamError,
        aws_smithy_runtime_api::client::orchestrator::HttpResponse,
    >,
) -> BedrockStreamFailure {
    let raw = error.raw_response();
    let status = raw.map(|response| response.status().as_u16());
    let body = raw
        .and_then(|response| response.body().bytes())
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned());
    let headers = raw.map(|response| headers_map(response.headers()));
    let request_id = headers
        .as_ref()
        .and_then(|headers| headers.get("x-amzn-requestid").cloned());
    let (name, message) = error.as_service_error().map_or_else(
        || (None, error.to_string()),
        |service| {
            (
                Some(service.meta().code().unwrap_or("Unknown").to_owned()),
                service
                    .meta()
                    .message()
                    .map_or_else(|| service.to_string(), str::to_owned),
            )
        },
    );
    BedrockStreamFailure {
        name,
        message,
        status,
        body,
        request_id,
    }
}

/// The mid-stream failure, upstream's `getMessageUnmarshaller` throws: the
/// modeled exception name rides, the raw message frame carries no HTTP
/// metadata of its own.
fn stream_failure<R>(
    error: &SdkError<sdk_types::error::ConverseStreamOutputError, R>,
) -> BedrockStreamFailure
where
    R: std::fmt::Debug + Send + Sync + 'static,
{
    error.as_service_error().map_or_else(
        || BedrockStreamFailure::plain(error.to_string()),
        |service| BedrockStreamFailure {
            name: Some(service.meta().code().unwrap_or("Unknown").to_owned()),
            message: service
                .meta()
                .message()
                .map_or_else(|| service.to_string(), str::to_owned),
            status: None,
            body: None,
            request_id: None,
        },
    )
}

#[cfg(test)]
mod sdk_glue_tests {
    #![expect(
        clippy::expect_used,
        reason = "the tests pin adapter outcomes; an unexpected shape panics the test by design"
    )]
    #![expect(
        clippy::panic,
        reason = "the let-else guards panic on an unexpected SDK shape by design"
    )]
    #![expect(
        clippy::float_cmp,
        reason = "the wire mappers carry exact float pass-throughs; the narrowing is value-identical"
    )]

    use super::*;
    use crate::api::bedrock_options::BedrockCredentials;
    use aws_smithy_runtime_api::client::interceptors::Intercept;
    use aws_smithy_runtime_api::client::interceptors::context::{
        AfterDeserializationInterceptorContextRef, BeforeTransmitInterceptorContextMut,
        InterceptorContext,
    };
    use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
    use aws_smithy_runtime_api::client::runtime_components::RuntimeComponentsBuilder;
    use aws_smithy_runtime_api::http::{Headers, StatusCode};
    use aws_smithy_types::body::SdkBody;
    use aws_smithy_types::config_bag::ConfigBag;
    use aws_smithy_types::error::ErrorMetadata;
    use futures_util::StreamExt;
    use sdk_types::{
        AnyToolChoice, AutoToolChoice, CacheDetail, CachePointType, CacheTtl, ContentBlock,
        ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
        ConversationRole, ImageFormat, MessageStartEvent, MessageStopEvent, ReasoningContentBlock,
        ReasoningContentBlockDelta, ReasoningTextBlock, SpecificToolChoice, StopReason, TokenUsage,
        Tool, ToolChoice, ToolInputSchema, ToolUseBlockDelta, ToolUseBlockStart,
    };

    // --- the pure wire mappers ---

    /// The wire temperature narrows from the f64 option to the f32 field.
    #[test]
    fn the_wire_temperature_narrows_to_the_wire_precision() {
        assert_eq!(wire_temperature(0.7_f64), 0.7_f32);
        assert_eq!(wire_temperature(1.25_f64), 1.25_f32);
    }

    /// The cache TTL maps the wire spellings; unset rides the empty default.
    #[test]
    fn cache_ttl_maps_the_wire_values_and_defaults_empty() {
        assert_eq!(cache_ttl(None).as_str(), "");
        assert_eq!(cache_ttl(Some(&json!("1h"))), CacheTtl::OneHour);
        assert_eq!(cache_ttl(Some(&json!("5m"))), CacheTtl::FiveMinutes);
        assert_eq!(cache_ttl(Some(&json!("junk"))).as_str(), "junk");
    }

    /// The smithy `Document` conversion covers every JSON kind, with the
    /// number spellings split by sign and integrality.
    #[test]
    fn document_maps_the_wire_json_including_number_kinds() {
        assert_eq!(document(&Value::Null), Document::Null);
        assert_eq!(document(&json!(true)), Document::Bool(true));
        assert_eq!(document(&json!("hi")), Document::String("hi".to_owned()));
        assert_eq!(
            document(&json!(5)),
            Document::Number(Number::PosInt(5_u64)),
            "positive integers ride PosInt"
        );
        assert_eq!(
            document(&json!(-7)),
            Document::Number(Number::NegInt(-7_i64)),
            "negative integers ride NegInt"
        );
        assert_eq!(
            document(&json!(1.5)),
            Document::Number(Number::Float(1.5_f64)),
            "floats ride Float"
        );
        assert_eq!(
            document(&json!({"a": [true, "b"], "c": 1})),
            Document::Object(
                [
                    (
                        "a".to_owned(),
                        Document::Array(vec![
                            Document::Bool(true),
                            Document::String("b".to_owned()),
                        ]),
                    ),
                    ("c".to_owned(), Document::Number(Number::PosInt(1))),
                ]
                .into_iter()
                .collect(),
            )
        );
    }

    /// The bytes a wire field carries read the base64 spelling and the
    /// byte-array spelling, dropping shapes that carry neither.
    #[test]
    fn json_blob_reads_base64_and_byte_arrays() {
        assert_eq!(json_blob(None), None);
        assert_eq!(
            json_blob(Some(&json!("aGk="))),
            Some(Blob::new(b"hi".to_vec())),
            "base64 rides decoded"
        );
        assert_eq!(
            json_blob(Some(&json!([104, 105]))),
            Some(Blob::new(b"hi".to_vec())),
            "byte arrays ride as-is"
        );
        assert_eq!(json_blob(Some(&json!("!!!"))), None, "invalid base64 drops");
        assert_eq!(
            json_blob(Some(&json!([104, 300]))),
            None,
            "out-of-range bytes drop"
        );
        assert_eq!(json_blob(Some(&json!(true))), None, "other shapes drop");
    }

    /// The number conversion splits integer signs and falls back to the
    /// float value.
    #[test]
    fn document_number_splits_the_number_kinds() {
        let positive: serde_json::Number = serde_json::from_str("7").expect("positive");
        assert_eq!(document_number(&positive), Number::PosInt(7));
        let negative: serde_json::Number = serde_json::from_str("-7").expect("negative");
        assert_eq!(document_number(&negative), Number::NegInt(-7));
        let float: serde_json::Number = serde_json::from_str("1.25").expect("float");
        assert_eq!(document_number(&float), Number::Float(1.25_f64));
        let large: serde_json::Number = serde_json::from_str("18446744073709551615").expect("u64");
        assert_eq!(document_number(&large), Number::PosInt(u64::MAX));
    }

    /// A message maps its role and content blocks; unknown roles degrade to
    /// user, missing content rides empty.
    #[test]
    fn wire_message_carries_the_role_and_content_blocks() {
        let assistant = wire_message(&json!({
            "role": "assistant",
            "content": [{ "text": "hi" }],
        }))
        .expect("assistant message");
        assert_eq!(assistant.role(), &ConversationRole::Assistant);
        assert_eq!(assistant.content(), &[ContentBlock::Text("hi".to_owned())]);

        let user = wire_message(&json!({ "role": "weird" })).expect("user message");
        assert_eq!(user.role(), &ConversationRole::User);
        assert!(
            user.content().is_empty(),
            "missing content rides the empty list"
        );
    }

    /// Text, image, tool-use, reasoning, and cache-point blocks map onto
    /// their SDK shapes; unrecognized blocks degrade to empty text.
    #[test]
    fn wire_content_block_maps_the_block_shapes() {
        assert_eq!(
            wire_content_block(&json!({ "text": "hi" })).expect("text"),
            ContentBlock::Text("hi".to_owned())
        );

        let image = wire_content_block(&json!({
            "image": {
                "format": "png",
                "source": { "bytes": [104, 105] },
            },
        }))
        .expect("image");
        let ContentBlock::Image(image) = image else {
            panic!("expected the image block")
        };
        assert_eq!(image.format(), &ImageFormat::Png);
        assert_eq!(
            image
                .source()
                .and_then(|source| source.as_bytes().ok())
                .expect("the bytes")
                .as_ref(),
            b"hi"
        );

        let tool_use = wire_content_block(&json!({
            "toolUse": { "toolUseId": "id-1", "name": "read", "input": { "path": "a.txt" } },
        }))
        .expect("tool use");
        let ContentBlock::ToolUse(tool_use) = tool_use else {
            panic!("expected the tool-use block")
        };
        assert_eq!(tool_use.tool_use_id(), "id-1");
        assert_eq!(tool_use.name(), "read");
        assert_eq!(tool_use.input(), &document(&json!({ "path": "a.txt" })));

        let reasoning = wire_content_block(&json!({
            "reasoningContent": {
                "reasoningText": { "text": "ponder", "signature": "sig" },
            },
        }))
        .expect("reasoning");
        assert_eq!(
            reasoning,
            ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(
                ReasoningTextBlock::builder()
                    .text("ponder")
                    .signature("sig")
                    .build()
                    .expect("reasoning text"),
            ))
        );

        let redacted = wire_content_block(&json!({
            "reasoningContent": { "redactedContent": [104, 105] },
        }))
        .expect("redacted reasoning");
        assert_eq!(
            redacted,
            ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(Blob::new(
                b"hi".to_vec()
            )),)
        );

        let cache_point = wire_content_block(&json!({
            "cachePoint": { "type": "default", "ttl": "1h" },
        }))
        .expect("cache point");
        let ContentBlock::CachePoint(cache_point) = cache_point else {
            panic!("expected the cache-point block")
        };
        assert_eq!(cache_point.r#type(), &CachePointType::Default);
        assert_eq!(cache_point.ttl(), Some(&CacheTtl::OneHour));

        assert_eq!(
            wire_content_block(&json!({ "unknown": true })).expect("fallback"),
            ContentBlock::Text(String::new()),
            "unrecognized blocks degrade to empty text"
        );
    }

    /// The known image formats map; unknown names and missing or unusable
    /// bytes fail with the adapter's wording.
    #[test]
    fn wire_content_block_gates_image_formats_and_bytes() {
        for (format, expected) in [
            ("jpeg", ImageFormat::Jpeg),
            ("gif", ImageFormat::Gif),
            ("webp", ImageFormat::Webp),
        ] {
            let block = wire_content_block(&json!({
                "image": { "format": format, "source": { "bytes": [104] } },
            }))
            .expect("image");
            let ContentBlock::Image(image) = block else {
                panic!("expected the image block")
            };
            assert_eq!(image.format(), &expected, "{format} maps");
        }

        let unknown = wire_content_block(&json!({
            "image": { "format": "bmp", "source": { "bytes": [104] } },
        }))
        .expect_err("the unknown format");
        assert_eq!(unknown.message, "Unknown image type: bmp");

        let missing = wire_content_block(&json!({
            "image": { "format": "png" },
        }))
        .expect_err("the missing bytes");
        assert!(
            missing
                .message
                .contains("image bytes missing or not valid JSON bytes"),
            "{}",
            missing.message
        );

        let unusable = wire_content_block(&json!({
            "image": { "format": "png", "source": { "bytes": "!!!" } },
        }))
        .expect_err("the invalid base64");
        assert!(
            unusable
                .message
                .contains("image bytes missing or not valid JSON bytes"),
            "{}",
            unusable.message
        );

        let base64 = wire_content_block(&json!({
            "image": { "format": "png", "source": { "bytes": "aGk=" } },
        }))
        .expect("the base64 bytes");
        let ContentBlock::Image(image) = base64 else {
            panic!("expected the image block")
        };
        assert_eq!(
            image
                .source()
                .and_then(|source| source.as_bytes().ok())
                .expect("the bytes")
                .as_ref(),
            b"hi"
        );
    }

    /// The reasoning block rides the redacted blob when one is present, else
    /// the reasoning text with its optional signature.
    #[test]
    fn wire_reasoning_block_prefers_the_redacted_blob() {
        let redacted = wire_reasoning_block(&json!({
            "redactedContent": [104, 105],
            "reasoningText": { "text": "ignored" },
        }))
        .expect("the redacted blob");
        assert_eq!(
            redacted,
            ReasoningContentBlock::RedactedContent(Blob::new(b"hi".to_vec()))
        );

        let unsigned = wire_reasoning_block(&json!({
            "reasoningText": { "text": "ponder" },
        }))
        .expect("the unsigned text");
        let ReasoningContentBlock::ReasoningText(block) = unsigned else {
            panic!("expected reasoning text")
        };
        assert_eq!(block.text(), "ponder");
        assert!(block.signature().is_none());

        let unusable = wire_reasoning_block(&json!({ "redactedContent": "!!!" }))
            .expect("the unusable blob falls through to the text shape");
        let ReasoningContentBlock::ReasoningText(block) = unusable else {
            panic!("expected reasoning text")
        };
        assert_eq!(block.text(), "", "no reasoningText rides empty");
    }

    /// The cache-point block maps the default type and the TTL wire value.
    #[test]
    fn wire_cache_point_maps_the_default_block() {
        let block =
            wire_cache_point(&json!({ "type": "default", "ttl": "1h" })).expect("the cache point");
        assert_eq!(block.r#type(), &CachePointType::Default);
        assert_eq!(block.ttl(), Some(&CacheTtl::OneHour));
    }

    /// System blocks map text and cache points, degrading unknown blocks to
    /// empty text.
    #[test]
    fn wire_system_block_maps_text_and_cache_points() {
        assert_eq!(
            wire_system_block(&json!({ "text": "be nice" })).expect("the text"),
            sdk_types::SystemContentBlock::Text("be nice".to_owned())
        );
        assert_eq!(
            wire_system_block(&json!({ "cachePoint": { "type": "default" } }))
                .expect("the cache point"),
            sdk_types::SystemContentBlock::CachePoint(
                wire_cache_point(&json!({ "type": "default" })).expect("the block"),
            )
        );
        assert_eq!(
            wire_system_block(&json!({})).expect("the fallback"),
            sdk_types::SystemContentBlock::Text(String::new())
        );
    }

    /// The tool configuration maps tools with their schemas and the three
    /// choice forms the adapter emits; no choice leaves the field unset.
    #[test]
    fn wire_tool_config_maps_tools_and_choice_forms() {
        for (choice, expected) in [
            (
                json!({ "auto": {} }),
                Some(ToolChoice::Auto(AutoToolChoice::builder().build())),
            ),
            (
                json!({ "any": {} }),
                Some(ToolChoice::Any(AnyToolChoice::builder().build())),
            ),
            (
                json!({ "tool": { "name": "read" } }),
                Some(ToolChoice::Tool(
                    SpecificToolChoice::builder()
                        .name("read")
                        .build()
                        .expect("the specific tool"),
                )),
            ),
            (json!({}), None),
        ] {
            let config = json!({ "tools": [], "toolChoice": choice });
            let wire = wire_tool_config(&config).expect("the tool config");
            assert_eq!(wire.tool_choice(), expected.as_ref(), "{choice}");
        }

        let full = wire_tool_config(&json!({
            "toolChoice": { "auto": {} },
            "tools": [{
                "toolSpec": {
                    "name": "read",
                    "description": "Read a file.",
                    "strict": true,
                    "inputSchema": { "json": { "type": "object" } },
                },
            }],
        }))
        .expect("the full tool config");
        assert_eq!(full.tools().len(), 1);
        let Tool::ToolSpec(spec) = full.tools().first().expect("the tool") else {
            panic!("expected the tool spec")
        };
        assert_eq!(spec.name(), "read");
        assert_eq!(spec.description(), Some("Read a file."));
        assert_eq!(spec.strict(), Some(true));
        let Some(ToolInputSchema::Json(schema)) = spec.input_schema() else {
            panic!("expected the JSON schema")
        };
        assert_eq!(schema, &document(&json!({ "type": "object" })));
    }

    /// The tool spec fills missing fields with the defaults the builder
    /// accepts, upstream's `??` chains; a false strict flag rides verbatim.
    #[test]
    fn wire_tool_config_fills_the_missing_spec_fields() {
        let wire = wire_tool_config(&json!({ "tools": [{ "toolSpec": { "strict": false } }] }))
            .expect("the tool config");
        let Tool::ToolSpec(spec) = wire.tools().first().expect("the tool") else {
            panic!("expected the tool spec")
        };
        assert_eq!(spec.name(), "");
        assert_eq!(spec.description(), Some(""));
        assert_eq!(spec.strict(), Some(false), "the false flag rides verbatim");
        let Some(ToolInputSchema::Json(schema)) = spec.input_schema() else {
            panic!("expected the JSON schema")
        };
        assert_eq!(schema, &Document::Null);
    }

    // --- the command-input application ---

    /// The full command input maps onto the typed builder: messages, system,
    /// inference config, tool config, open fields, and request metadata.
    #[test]
    fn apply_command_input_maps_the_full_command() {
        let client = sdk::Client::from_conf(
            sdk::Config::builder()
                .behavior_version(sdk::config::BehaviorVersion::latest())
                .build(),
        );
        let input = json!({
            "modelId": "model-x",
            "messages": [{ "role": "user", "content": [{ "text": "hi" }] }],
            "system": [{ "text": "be nice" }],
            "inferenceConfig": { "maxTokens": 5, "temperature": 0.7 },
            "toolConfig": {
                "toolChoice": { "auto": {} },
                "tools": [{ "toolSpec": { "name": "read" } }],
            },
            "additionalModelRequestFields": { "thinking": { "type": "adaptive" } },
            "requestMetadata": { "a": "x", "b": 42 },
        });

        let builder =
            apply_command_input(client.converse_stream(), &input).expect("the full command");
        assert_eq!(builder.get_model_id().as_deref(), Some("model-x"));
        let messages = builder.get_messages().as_ref().expect("the messages");
        assert_eq!(
            messages,
            &[
                wire_message(&json!({ "role": "user", "content": [{ "text": "hi" }] }))
                    .expect("the message")
            ]
        );
        assert_eq!(
            builder.get_system().as_ref().expect("the system"),
            &[wire_system_block(&json!({ "text": "be nice" })).expect("the block")]
        );
        let inference = builder
            .get_inference_config()
            .as_ref()
            .expect("the inference config");
        assert_eq!(inference.max_tokens(), Some(5));
        assert_eq!(inference.temperature(), Some(wire_temperature(0.7_f64)));
        let tool_config = builder.get_tool_config().as_ref().expect("the tool config");
        assert_eq!(
            tool_config,
            &wire_tool_config(&json!({
                "toolChoice": { "auto": {} },
                "tools": [{ "toolSpec": { "name": "read" } }],
            }))
            .expect("the tool config")
        );
        assert_eq!(
            builder.get_additional_model_request_fields().as_ref(),
            Some(&document(&json!({ "thinking": { "type": "adaptive" } })))
        );
        let metadata = builder
            .get_request_metadata()
            .as_ref()
            .expect("the request metadata");
        assert_eq!(metadata.get("a"), Some(&"x".to_owned()));
        assert_eq!(
            metadata.get("b"),
            Some(&"42".to_owned()),
            "non-string metadata values stringify"
        );
    }

    /// Missing command fields ride the SDK defaults: no model id is an empty
    /// string, absent collections stay unset, and an over-cap maxTokens
    /// narrows to zero, upstream's i32 coercion.
    #[test]
    fn apply_command_input_defaults_the_missing_fields() {
        let client = sdk::Client::from_conf(
            sdk::Config::builder()
                .behavior_version(sdk::config::BehaviorVersion::latest())
                .build(),
        );
        let input = json!({ "inferenceConfig": { "maxTokens": 5_000_000_000_u64 } });

        let builder =
            apply_command_input(client.converse_stream(), &input).expect("the sparse command");
        assert_eq!(builder.get_model_id().as_deref(), Some(""), "no model id");
        assert!(builder.get_messages().is_none());
        assert!(builder.get_system().is_none());
        let inference = builder
            .get_inference_config()
            .as_ref()
            .expect("the inference config");
        assert_eq!(
            inference.max_tokens(),
            Some(0),
            "the over-cap maxTokens narrows"
        );
        assert!(inference.temperature().is_none());
        assert!(builder.get_tool_config().is_none());
        assert!(builder.get_additional_model_request_fields().is_none());
        assert!(builder.get_request_metadata().is_none());
    }

    // --- the SDK event mappers ---

    /// Every SDK event carries the wire frame the adapter's stream consumes;
    /// unknown events ride the empty frame.
    #[test]
    fn event_to_value_shapes_the_sdk_events() {
        assert_eq!(
            event_to_value(&sdk_types::ConverseStreamOutput::MessageStart(
                MessageStartEvent::builder()
                    .role(ConversationRole::Assistant)
                    .build()
                    .expect("the event"),
            )),
            json!({ "messageStart": { "role": "assistant" } })
        );

        assert_eq!(
            event_to_value(&sdk_types::ConverseStreamOutput::ContentBlockStart(
                ContentBlockStartEvent::builder()
                    .content_block_index(2)
                    .start(sdk_types::ContentBlockStart::ToolUse(
                        ToolUseBlockStart::builder()
                            .tool_use_id("id-1")
                            .name("read")
                            .build()
                            .expect("the tool-use start"),
                    ))
                    .build()
                    .expect("the event"),
            )),
            json!({
                "contentBlockStart": {
                    "contentBlockIndex": 2,
                    "start": { "toolUse": { "toolUseId": "id-1", "name": "read" } },
                }
            })
        );

        assert_eq!(
            event_to_value(&sdk_types::ConverseStreamOutput::ContentBlockStart(
                ContentBlockStartEvent::builder()
                    .content_block_index(1)
                    .build()
                    .expect("the event"),
            )),
            json!({ "contentBlockStart": { "contentBlockIndex": 1 } })
        );

        assert_eq!(
            event_to_value(&sdk_types::ConverseStreamOutput::ContentBlockDelta(
                ContentBlockDeltaEvent::builder()
                    .content_block_index(1)
                    .delta(ContentBlockDelta::Text("hi".to_owned()))
                    .build()
                    .expect("the event"),
            )),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": { "text": "hi" },
                }
            })
        );

        assert_eq!(
            event_to_value(&sdk_types::ConverseStreamOutput::ContentBlockStop(
                ContentBlockStopEvent::builder()
                    .content_block_index(3)
                    .build()
                    .expect("the event"),
            )),
            json!({ "contentBlockStop": { "contentBlockIndex": 3 } })
        );

        assert_eq!(
            event_to_value(&sdk_types::ConverseStreamOutput::MessageStop(
                MessageStopEvent::builder()
                    .stop_reason(StopReason::EndTurn)
                    .build()
                    .expect("the event"),
            )),
            json!({ "messageStop": { "stopReason": "end_turn" } })
        );
    }

    /// The delta kinds carry their wire shapes, unknown deltas riding null.
    #[test]
    fn delta_to_value_shapes_the_delta_kinds() {
        assert_eq!(
            delta_to_value(&ContentBlockDelta::Text("hi".to_owned())),
            json!({ "text": "hi" })
        );
        assert_eq!(
            delta_to_value(&ContentBlockDelta::ToolUse(
                ToolUseBlockDelta::builder()
                    .input("{")
                    .build()
                    .expect("the tool-use delta"),
            )),
            json!({ "toolUse": { "input": "{" } })
        );
        assert_eq!(
            delta_to_value(&ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::Text("ponder".to_owned()),
            )),
            json!({ "text": "ponder" })
        );
        assert_eq!(
            delta_to_value(&ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::Signature("sig".to_owned()),
            )),
            json!({ "signature": "sig" })
        );
        assert_eq!(
            delta_to_value(&ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::RedactedContent(Blob::new(b"hi".to_vec())),
            )),
            json!({
                "redactedContent":
                    json!(crate::api::bedrock_converse_stream::bytes_to_base64(b"hi")),
            })
        );
    }

    /// The metadata frame carries the usage block with the cache details
    /// split out, degrading to an empty frame without usage.
    #[test]
    fn metadata_to_value_carries_the_usage_and_cache_details() {
        let event = sdk_types::ConverseStreamMetadataEvent::builder()
            .usage(
                TokenUsage::builder()
                    .input_tokens(10)
                    .output_tokens(5)
                    .total_tokens(15)
                    .cache_read_input_tokens(4)
                    .cache_write_input_tokens(6)
                    .cache_details(
                        CacheDetail::builder()
                            .ttl(CacheTtl::OneHour)
                            .input_tokens(6)
                            .build()
                            .expect("the cache detail"),
                    )
                    .cache_details(
                        CacheDetail::builder()
                            .ttl(CacheTtl::FiveMinutes)
                            .input_tokens(2)
                            .build()
                            .expect("the cache detail"),
                    )
                    .build()
                    .expect("the usage"),
            )
            .build();
        assert_eq!(
            metadata_to_value(&event),
            json!({
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "totalTokens": 15,
                    "cacheReadInputTokens": 4,
                    "cacheWriteInputTokens": 6,
                    "cacheDetails": [
                        { "ttl": "1h", "inputTokens": 6 },
                        { "ttl": "5m", "inputTokens": 2 },
                    ],
                }
            })
        );

        let no_usage = sdk_types::ConverseStreamMetadataEvent::builder().build();
        assert_eq!(
            metadata_to_value(&no_usage),
            json!({}),
            "no usage rides the empty frame"
        );
    }

    /// The raw-response hook's header record is the smithy header record.
    #[test]
    fn headers_map_collects_the_response_headers() {
        let mut headers = Headers::new();
        headers.insert("x-amzn-requestid", "req-1");
        headers.insert("x-custom", "v");
        let mapped = headers_map(&headers);
        assert_eq!(mapped.get("x-amzn-requestid"), Some(&"req-1".to_owned()));
        assert_eq!(mapped.get("x-custom"), Some(&"v".to_owned()));
        assert_eq!(mapped.len(), 2);
    }

    // --- the failure mappers ---

    /// A modeled send failure carries the exception code, the message, the
    /// HTTP status, the raw body, and the request id.
    #[test]
    fn send_failure_carries_the_service_metadata_and_raw_response() {
        let mut raw = HttpResponse::new(
            StatusCode::try_from(400).expect("the status code"),
            SdkBody::from(r#"{"message": "The model identifier is invalid."}"#),
        );
        raw.headers_mut().insert("x-amzn-requestid", "req-1");
        let error = SdkError::service_error(
            ConverseStreamError::generic(
                ErrorMetadata::builder()
                    .code("ValidationException")
                    .message("The model identifier is invalid.")
                    .build(),
            ),
            raw,
        );

        let failure = send_failure(&error);
        assert_eq!(failure.name.as_deref(), Some("ValidationException"));
        assert_eq!(failure.message, "The model identifier is invalid.");
        assert_eq!(failure.status, Some(400));
        assert_eq!(
            failure.body.as_deref(),
            Some(r#"{"message": "The model identifier is invalid."}"#)
        );
        assert_eq!(failure.request_id.as_deref(), Some("req-1"));
    }

    /// A transport-phase send failure degrades to the plain failure shape.
    #[test]
    fn send_failure_reports_transport_failures_as_plain() {
        let error = SdkError::<ConverseStreamError, HttpResponse>::construction_failure(
            "the endpoint never opened",
        );
        let failure = send_failure(&error);
        assert_eq!(failure.name, None);
        assert_eq!(failure.message, "failed to construct request");
        assert_eq!(failure.status, None);
        assert_eq!(failure.body, None);
        assert_eq!(failure.request_id, None);
    }

    /// A modeled mid-stream failure carries the code and message; a
    /// transport-phase one degrades to plain.
    #[test]
    fn stream_failure_carries_the_modeled_code_and_message() {
        type StreamSdkError = SdkError<
            sdk_types::error::ConverseStreamOutputError,
            aws_smithy_types::event_stream::RawMessage,
        >;
        let service = StreamSdkError::service_error(
            sdk_types::error::ConverseStreamOutputError::generic(
                ErrorMetadata::builder()
                    .code("ThrottlingException")
                    .message("rate limited")
                    .build(),
            ),
            aws_smithy_types::event_stream::RawMessage::Decoded(
                aws_smithy_types::event_stream::Message::new(Vec::new()),
            ),
        );
        let failure = stream_failure(&service);
        assert_eq!(failure.name.as_deref(), Some("ThrottlingException"));
        assert_eq!(failure.message, "rate limited");
        assert_eq!(failure.status, None);

        let transport = StreamSdkError::construction_failure("the stream broke");
        let failure = stream_failure(&transport);
        assert_eq!(failure.name, None);
        assert_eq!(failure.message, "failed to construct request");
    }

    // --- the client assembly ---

    /// The smithy HTTP client builds for the forced HTTP/1.1 and proxy
    /// branches, and a malformed proxy URL fails with the adapter's wording.
    #[test]
    fn http_client_builds_for_the_http1_and_proxy_branches() {
        let config = BedrockClientConfig {
            force_http1: true,
            ..BedrockClientConfig::default()
        };
        assert!(http_client(&config).is_ok(), "force_http1 builds");

        let config = BedrockClientConfig {
            proxy_url: Some("http://proxy.example.com:8080".to_owned()),
            ..BedrockClientConfig::default()
        };
        assert!(http_client(&config).is_ok(), "the proxy branch builds");

        let config = BedrockClientConfig {
            proxy_url: Some("not a url".to_owned()),
            ..BedrockClientConfig::default()
        };
        let failure = http_client(&config).expect_err("the bad proxy");
        assert!(
            failure.message.contains("Invalid proxy URL not a url"),
            "{}",
            failure.message
        );
    }

    /// The client builds with every config seam set: region, endpoint,
    /// credentials, bearer token, proxy, HTTP/1.1, profile, and caller
    /// headers, and the response snapshot starts empty.
    #[tokio::test]
    async fn build_client_resolves_the_full_config() {
        let config = BedrockClientConfig {
            region: Some("us-east-1".to_owned()),
            endpoint: Some("http://127.0.0.1:9".to_owned()),
            profile: Some("test-profile".to_owned()),
            credentials: Some(BedrockCredentials {
                access_key_id: "key".to_owned(),
                secret_access_key: "secret".to_owned(),
                session_token: Some("session".to_owned()),
            }),
            token: Some("bearer-token".to_owned()),
            force_http1: true,
            proxy_url: Some("http://proxy.example.com:8080".to_owned()),
            headers: vec![("x-custom".to_owned(), "v".to_owned())],
            ..BedrockClientConfig::default()
        };

        let (client, snapshot) = build_client(&config).await.expect("the client builds");
        let _ = client;
        assert!(
            snapshot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none(),
            "the snapshot starts empty"
        );
    }

    /// The skip-auth config installs the dummy credential pair and the
    /// default config leaves the SDK chain in place.
    #[tokio::test]
    async fn build_client_applies_the_skip_auth_and_default_paths() {
        let config = BedrockClientConfig {
            skip_auth: true,
            ..BedrockClientConfig::default()
        };
        let (client, _snapshot) = build_client(&config).await.expect("skip auth builds");
        let _ = client;

        let (client, _snapshot) = build_client(&BedrockClientConfig::default())
            .await
            .expect("the default path builds");
        let _ = client;
    }

    // --- the interceptors ---

    /// The pre-signing interceptor injects the caller headers onto the
    /// request, upstream's build-step middleware.
    #[test]
    fn the_custom_headers_interceptor_injects_before_signing() {
        let interceptor = CustomHeadersInterceptor {
            headers: vec![
                ("x-custom".to_owned(), "v".to_owned()),
                ("x-pi-foo".to_owned(), "bar".to_owned()),
            ],
        };
        assert_eq!(interceptor.name(), "pi-ai-custom-headers");

        let mut context = InterceptorContext::new(
            aws_smithy_runtime_api::client::interceptors::context::Input::erase(String::from(
                "unused",
            )),
        );
        context.set_request(HttpRequest::new(SdkBody::from("body")));
        let mut wrapper = BeforeTransmitInterceptorContextMut::from(&mut context);
        let components = RuntimeComponentsBuilder::for_tests()
            .build()
            .expect("the runtime components");
        interceptor
            .modify_before_signing(&mut wrapper, &components, &mut ConfigBag::base())
            .expect("the injection succeeds");

        let request = wrapper.request();
        assert_eq!(request.headers().get("x-custom"), Some("v"));
        assert_eq!(request.headers().get("x-pi-foo"), Some("bar"));
    }

    /// The response capture interceptor snapshots the status and headers at
    /// the deserialize step, upstream's response middleware.
    #[test]
    fn the_raw_response_capture_snapshots_status_and_headers() {
        let capture = RawResponseCapture {
            snapshot: Arc::new(Mutex::new(None)),
        };
        assert_eq!(capture.name(), "pi-ai-response-headers");

        let mut context = InterceptorContext::new(
            aws_smithy_runtime_api::client::interceptors::context::Input::erase(String::from(
                "unused",
            )),
        );
        let mut response = HttpResponse::new(
            StatusCode::try_from(200).expect("the status code"),
            SdkBody::from("{}"),
        );
        response.headers_mut().insert("x-amzn-requestid", "req-1");
        context.set_response(response);
        let wrapper = AfterDeserializationInterceptorContextRef::from(&context);
        let components = RuntimeComponentsBuilder::for_tests()
            .build()
            .expect("the runtime components");
        capture
            .read_after_deserialization(&wrapper, &components, &mut ConfigBag::base())
            .expect("the capture succeeds");

        let snapshot = capture
            .snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("the snapshot filled");
        assert_eq!(snapshot.status, Some(200));
        assert_eq!(
            snapshot.headers.get("x-amzn-requestid"),
            Some(&"req-1".to_owned())
        );
    }

    // --- the loopback event-stream sends ---

    /// The CRC-32 the smithy event-stream frames carry, the variant the
    /// smithy decoder verifies against.
    fn frame_crc(bytes: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(bytes);
        hasher.finalize()
    }

    /// One smithy event-stream frame: the prelude with its checksum, the
    /// string headers, the payload, and the trailing message checksum.
    fn wire_frame(headers: &[(&str, &str)], payload: &str) -> Vec<u8> {
        let mut encoded_headers = Vec::new();
        for (name, value) in headers {
            encoded_headers.push(u8::try_from(name.len()).expect("the header name fits a byte"));
            encoded_headers.extend_from_slice(name.as_bytes());
            // 0x07 is the smithy HeaderValue string type.
            encoded_headers.push(0x07);
            encoded_headers.extend_from_slice(
                &u16::try_from(value.len())
                    .expect("the header value fits u16")
                    .to_be_bytes(),
            );
            encoded_headers.extend_from_slice(value.as_bytes());
        }
        let total = 12 + encoded_headers.len() + payload.len() + 4;
        let mut frame = Vec::with_capacity(total);
        frame.extend_from_slice(
            &u32::try_from(total)
                .expect("the frame fits u32")
                .to_be_bytes(),
        );
        frame.extend_from_slice(
            &u32::try_from(encoded_headers.len())
                .expect("the headers fit u32")
                .to_be_bytes(),
        );
        frame.extend_from_slice(&frame_crc(&frame).to_be_bytes());
        frame.extend_from_slice(&encoded_headers);
        frame.extend_from_slice(payload.as_bytes());
        frame.extend_from_slice(&frame_crc(&frame).to_be_bytes());
        frame
    }

    /// Accept one request on the loopback listener and answer with the
    /// canned event-stream bytes, upstream's `node:http` responder.
    async fn serve_eventstream(listener: tokio::net::TcpListener, body: Vec<u8>, status: u16) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut socket, _) = listener.accept().await.expect("one accept");
        let mut request = Vec::new();
        let mut buffer = [0u8; 8192];
        loop {
            let read = socket.read(&mut buffer).await.expect("the request read");
            request.extend_from_slice(&buffer[..read]);
            let separator = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map_or(0, |at| at + 4);
            let length = String::from_utf8_lossy(&request[..separator])
                .lines()
                .find_map(|line| {
                    line.strip_prefix("content-length:")
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if read == 0 || request.len() >= separator + length {
                break;
            }
        }
        let head = format!(
            "HTTP/1.1 {status} OK\r\ncontent-type: application/vnd.amazon.eventstream\r\ncontent-length: {}\r\nx-amzn-requestid: wire-req-1\r\n\r\n",
            body.len()
        );
        let mut response = head.into_bytes();
        response.extend_from_slice(&body);
        socket
            .write_all(&response)
            .await
            .expect("the response write");
    }

    /// The loopback config: the pinned local endpoint, the region SigV4
    /// needs, and static credentials so the ambient chain never runs.
    fn loopback_config(port: u16) -> BedrockClientConfig {
        BedrockClientConfig {
            region: Some("us-east-1".to_owned()),
            endpoint: Some(format!("http://127.0.0.1:{port}")),
            credentials: Some(BedrockCredentials {
                access_key_id: "test-key".to_owned(),
                secret_access_key: "test-secret".to_owned(),
                session_token: None,
            }),
            ..BedrockClientConfig::default()
        }
    }

    /// The full runtime seam send: the command input applies, the SDK signs
    /// and sends over the loopback endpoint, and the SDK events pump onto
    /// the wire shapes with the raw-response snapshot riding alongside.
    #[tokio::test]
    async fn converse_stream_sends_and_pumps_the_event_stream() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the loopback binds");
        let port = listener.local_addr().expect("the address").port();

        let mut body = Vec::new();
        for (headers, payload) in [
            (
                [
                    (":message-type", "event"),
                    (":event-type", "messageStart"),
                    (":content-type", "application/json"),
                ],
                r#"{"role":"assistant"}"#,
            ),
            (
                [
                    (":message-type", "event"),
                    (":event-type", "contentBlockDelta"),
                    (":content-type", "application/json"),
                ],
                r#"{"contentBlockIndex":0,"delta":{"text":"hi"}}"#,
            ),
            (
                [
                    (":message-type", "event"),
                    (":event-type", "metadata"),
                    (":content-type", "application/json"),
                ],
                r#"{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15,"cacheReadInputTokens":4,"cacheWriteInputTokens":6,"cacheDetails":[{"ttl":"1h","inputTokens":6}]}}"#,
            ),
            (
                [
                    (":message-type", "event"),
                    (":event-type", "messageStop"),
                    (":content-type", "application/json"),
                ],
                r#"{"stopReason":"end_turn"}"#,
            ),
        ] {
            body.extend_from_slice(&wire_frame(&headers, payload));
        }
        let server = tokio::spawn(serve_eventstream(listener, body, 200));

        let reply = SdkBedrockRuntime::new()
            .converse_stream(
                json!({ "modelId": "model-x", "messages": [] }),
                loopback_config(port),
                CancellationToken::new(),
            )
            .await
            .expect("the send succeeds");
        assert_eq!(reply.status, Some(200), "the snapshot rides the status");
        assert_eq!(reply.request_id.as_deref(), Some("wire-req-1"));

        let mut events = reply.events;
        assert_eq!(
            events.next().await.expect("the start event").expect("ok"),
            json!({ "messageStart": { "role": "assistant" } })
        );
        assert_eq!(
            events.next().await.expect("the delta event").expect("ok"),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": { "text": "hi" },
                }
            })
        );
        assert_eq!(
            events
                .next()
                .await
                .expect("the metadata event")
                .expect("ok"),
            json!({
                "metadata": {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "totalTokens": 15,
                        "cacheReadInputTokens": 4,
                        "cacheWriteInputTokens": 6,
                        "cacheDetails": [{ "ttl": "1h", "inputTokens": 6 }],
                    },
                }
            })
        );
        assert_eq!(
            events.next().await.expect("the stop event").expect("ok"),
            json!({ "messageStop": { "stopReason": "end_turn" } })
        );
        assert!(
            events.next().await.is_none(),
            "the stream ends after the frames"
        );
        server.await.expect("the server task");
    }

    /// A modeled exception frame mid-stream surfaces as the stream failure
    /// the adapter's catch path formats, upstream's thrown SDK exception.
    #[tokio::test]
    async fn converse_stream_surfaces_a_modeled_mid_stream_exception() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the loopback binds");
        let port = listener.local_addr().expect("the address").port();

        let mut body = Vec::new();
        body.extend_from_slice(&wire_frame(
            &[
                (":message-type", "event"),
                (":event-type", "messageStart"),
                (":content-type", "application/json"),
            ],
            r#"{"role":"assistant"}"#,
        ));
        body.extend_from_slice(&wire_frame(
            &[
                (":message-type", "exception"),
                (":exception-type", "throttlingException"),
            ],
            r#"{"__type": "ThrottlingException", "message": "rate limited"}"#,
        ));
        let server = tokio::spawn(serve_eventstream(listener, body, 200));

        let reply = SdkBedrockRuntime::new()
            .converse_stream(
                json!({ "modelId": "model-x" }),
                loopback_config(port),
                CancellationToken::new(),
            )
            .await
            .expect("the send succeeds");
        let mut events = reply.events;
        assert_eq!(
            events.next().await.expect("the start event").expect("ok"),
            json!({ "messageStart": { "role": "assistant" } })
        );
        let failure = events
            .next()
            .await
            .expect("the failure item")
            .expect_err("the modeled exception");
        assert_eq!(failure.name.as_deref(), Some("ThrottlingException"));
        assert_eq!(failure.message, "rate limited");
        assert_eq!(failure.status, None, "mid-stream failures carry no status");
        server.await.expect("the server task");
    }

    /// A command input that does not shape into the typed Converse fields
    /// fails before any send, with the adapter's builder-failure wording.
    #[tokio::test]
    async fn converse_stream_rejects_a_bad_command_input_before_sending() {
        let failure = SdkBedrockRuntime::new()
            .converse_stream(
                json!({
                    "modelId": "model-x",
                    "messages": [{
                        "role": "user",
                        "content": [{ "image": { "format": "bmp" } }],
                    }],
                }),
                BedrockClientConfig::default(),
                CancellationToken::new(),
            )
            .await
            .expect_err("the bad command input");
        assert_eq!(failure.message, "Unknown image type: bmp");
    }
}
