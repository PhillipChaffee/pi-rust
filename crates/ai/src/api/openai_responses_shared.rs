//! Shared machinery of the OpenAI Responses wire APIs, ported from
//! `packages/ai/src/api/openai-responses-shared.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream shares this module between `openai-responses.ts` and
//! `azure-openai-responses.ts`; both Rust modules import from it the same way.
//!
//! Porting restatements:
//!
//! - The wire's `ResponseStreamEvent` union is read straight off the JSON:
//!   each `data:` frame of the seam SSE stream is one event object whose
//!   `type` field picks the branch, the way the sibling completions module
//!   reads its chunks. The `data: [DONE]` sentinel ends the stream.
//! - `sanitizeSurrogates` disappears statically: a Rust [`String`] cannot
//!   hold the unpaired surrogates it stripped, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire.
//! - The scratch fields upstream carries on streamed block objects
//!   (`partialJson`, `customInput`) live beside the blocks in the stream
//!   state and never persist into the message.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::{
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta, get_grammar_tool_input,
    get_json_schema_tool_parameters, resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
};
use crate::api::transform_messages::{ToolCallIdNormalizer, transform_messages};
use crate::http::client::{HttpError, HttpResponse};
use crate::http::sse::SseStream;
use crate::models::calculate_cost;
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, ImageContent, Message,
    Modality, Model, StopReason, TextContent, ThinkingContent, Tool, ToolCall, ToolResultBlock,
    ToolResultMessage, Usage, UserBlock, UserContent,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::hash::short_hash;
use crate::utils::json_parse::parse_streaming_json;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// The `phase` values the text signature carries, upstream's
/// `TextSignatureV1["phase"]`.
const TEXT_SIGNATURE_PHASES: [&str; 2] = ["commentary", "final_answer"];

/// Serialize the v1 text signature, upstream's `encodeTextSignatureV1`: the
/// object keeps the wire's field order (`v`, `id`, `phase`).
fn encode_text_signature_v1(id: &str, phase: Option<&str>) -> String {
    let mut payload = Map::new();
    payload.insert("v".to_owned(), json!(1));
    payload.insert("id".to_owned(), json!(id));
    if let Some(phase) = phase {
        payload.insert("phase".to_owned(), json!(phase));
    }
    Value::Object(payload).to_string()
}

/// A parsed text signature, upstream's `parseTextSignature` return shape: the
/// replayed message id and the phase when the signature carried one.
struct ParsedTextSignature {
    id: String,
    phase: Option<String>,
}

/// Parse a text block's signature, upstream's `parseTextSignature`: a
/// `{"v":1,"id":...}` object carries the id and its valid phase, anything
/// else — including a malformed object — is the legacy plain-string form.
fn parse_text_signature(signature: Option<&str>) -> Option<ParsedTextSignature> {
    let signature = signature?;
    if let Ok(parsed) = serde_json::from_str::<Value>(signature)
        && parsed.get("v") == Some(&json!(1))
        && let Some(id) = parsed.get("id").and_then(Value::as_str)
    {
        let phase = parsed
            .get("phase")
            .and_then(Value::as_str)
            .filter(|phase| TEXT_SIGNATURE_PHASES.contains(phase))
            .map(str::to_owned);
        return Some(ParsedTextSignature {
            id: id.to_owned(),
            phase,
        });
    }
    Some(ParsedTextSignature {
        id: signature.to_owned(),
        phase: None,
    })
}

/// JavaScript truthiness over a wire value, upstream's bare `if (value)`
/// checks: objects and arrays pass; `null`, `false`, `0`, and empty strings
/// fail.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|sample| sample != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// The tool result's `output` value: a plain string when the model cannot see
/// images or none arrived, else `input_text`/`input_image` parts, upstream's
/// `convertToolResultOutput`.
fn convert_tool_result_output(model: &Model, content: &[ToolResultBlock]) -> Value {
    let text_result = content
        .iter()
        .filter_map(|block| match block {
            ToolResultBlock::Text(text) => Some(text.text.as_str()),
            ToolResultBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<&ImageContent> = content
        .iter()
        .filter_map(|block| match block {
            ToolResultBlock::Image(image) => Some(image),
            ToolResultBlock::Text(_) => None,
        })
        .collect();
    let has_text = !text_result.is_empty();

    if images.is_empty() || !model.input.contains(&Modality::Image) {
        return Value::String(if has_text {
            text_result
        } else if !images.is_empty() {
            "(see attached image)".to_owned()
        } else {
            "(no tool output)".to_owned()
        });
    }

    let mut output: Vec<Value> = Vec::new();
    if has_text {
        output.push(json!({ "type": "input_text", "text": text_result }));
    }
    for image in images {
        output.push(json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
        }));
    }
    Value::Array(output)
}

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

/// A service-tier resolver hook, upstream's `resolveServiceTier`: the
/// response's `service_tier` (absent or `null` arrives as `None`) and the
/// request's, returning the tier pricing applies.
pub type ResolveServiceTier =
    Arc<dyn Fn(Option<&str>, Option<&str>) -> Option<String> + Send + Sync>;

/// A service-tier pricing hook, upstream's `applyServiceTierPricing`.
pub type ApplyServiceTierPricing = Arc<dyn Fn(&mut Usage, Option<&str>) + Send + Sync>;

/// The cost multiplier a service tier prices at, upstream's
/// `getServiceTierCostMultiplier`: flex halves the cost, priority doubles it
/// (`2.5` on the `"gpt-5.5"` model), anything else is full price.
#[must_use]
pub fn get_service_tier_cost_multiplier(model_id: &str, service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") if model_id == "gpt-5.5" => 2.5,
        Some("priority") => 2.0,
        _ => 1.0,
    }
}

/// Scale the usage cost by the request's service tier, upstream's
/// `applyServiceTierPricing`: the multiplier applies to every cost component
/// and the total recomputes from the parts.
pub fn apply_service_tier_pricing(usage: &mut Usage, service_tier: Option<&str>, model: &Model) {
    let multiplier = get_service_tier_cost_multiplier(&model.id, service_tier);
    #[expect(
        clippy::float_cmp,
        reason = "the multiplier gate mirrors upstream's `multiplier === 1` exact comparison; the literals compare exactly"
    )]
    if multiplier == 1.0 {
        return;
    }

    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

/// The service-tier pricing hook a stream passes into the shared processor,
/// upstream's `applyServiceTierPricing` option closure.
#[must_use]
pub fn pricing_hook(model: Model) -> ApplyServiceTierPricing {
    Arc::new(move |usage: &mut Usage, service_tier: Option<&str>| {
        apply_service_tier_pricing(usage, service_tier, &model);
    })
}

/// The stream-processing options shared by the OpenAI Responses wire APIs,
/// upstream's `OpenAIResponsesStreamOptions`.
#[derive(Clone, Default)]
pub struct OpenAiResponsesStreamOptions {
    /// The request's `service_tier` (`"auto"`, `"default"`, `"flex"`, or
    /// `"priority"`); the pricing hook's fallback when the terminal response
    /// omits one.
    pub service_tier: Option<String>,
    /// The grammar tool-input property per tool name, the map custom-tool
    /// calls read their raw input through.
    pub grammar_tool_input_properties: Option<BTreeMap<String, String>>,
    /// Overrides which tier pricing applies; when absent the response's tier
    /// falls back to the request's, upstream's `??`.
    pub resolve_service_tier: Option<ResolveServiceTier>,
    /// Applies provider-priced service-tier multipliers to the final usage.
    pub apply_service_tier_pricing: Option<ApplyServiceTierPricing>,
}

impl std::fmt::Debug for OpenAiResponsesStreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiResponsesStreamOptions")
            .field("service_tier", &self.service_tier)
            .field(
                "grammar_tool_input_properties",
                &self.grammar_tool_input_properties,
            )
            .finish_non_exhaustive()
    }
}

/// The wire form deferred tools re-enter through, upstream's
/// `deferredToolsMode?: "additional-tools" | "tool-search"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponsesDeferredToolsMode {
    /// The `additional_tools` input item.
    AdditionalTools,
    /// The `tool_search_call`/`tool_search_output` item pair.
    ToolSearch,
}

/// The per-conversion options, upstream's `ConvertResponsesMessagesOptions`.
#[derive(Clone, Debug, Default)]
pub struct ConvertResponsesMessagesOptions {
    /// Whether the context's system prompt opens the input. Default: `true`.
    pub include_system_prompt: Option<bool>,
    /// The grammar tool-input property per tool name, the map custom-tool
    /// calls and outputs are keyed through.
    pub grammar_tool_input_properties: Option<BTreeMap<String, String>>,
    /// The deferred tool definitions a tool result's `addedToolNames` load,
    /// keyed by name.
    pub deferred_tools: Option<BTreeMap<String, Tool>>,
    /// The wire form deferred tools re-enter through.
    pub deferred_tools_mode: Option<ResponsesDeferredToolsMode>,
    /// The tool-definition conversion options the deferred tools ride
    /// through.
    pub tool_options: Option<ConvertResponsesToolsOptions>,
}

/// The tool-definition conversion options, upstream's
/// `ConvertResponsesToolsOptions`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConvertResponsesToolsOptions {
    /// The default `strict` value of a tool entry, the wire's
    /// `boolean | null`.
    pub strict: Option<Option<bool>>,
    /// Whether the provider accepts the `strict` field in tool definitions.
    /// Default: `true`.
    pub supports_strict_mode: Option<bool>,
    /// Whether OpenAI custom grammar tools bind. Default: `false`.
    pub supports_openai_grammar_tools: Option<bool>,
    /// Whether tool definitions carry `defer_loading: true`.
    pub defer_loading: Option<bool>,
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Convert the conversation to the Responses `input` item list, upstream's
/// `convertResponsesMessages`.
///
/// # Errors
/// A replayed thinking signature that does not parse, a custom tool call
/// whose arguments do not carry the string input its grammar streams
/// through, and the tool-definition conversion failures a deferred-tools
/// item hits.
#[expect(
    clippy::too_many_lines,
    reason = "one branch per message role and block kind, each a wire shape; splitting them would scatter the replay rules"
)]
pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &BTreeSet<String>,
    options: Option<&ConvertResponsesMessagesOptions>,
) -> Result<Vec<Value>, String> {
    let mut messages: Vec<Value> = Vec::new();
    let mut loaded_tool_names: BTreeSet<String> = BTreeSet::new();

    // OpenAI's id fields cap at 64 characters; the cap counts JS `.length`
    // UTF-16 code units, which ASCII ids make equivalent to characters.
    let normalize_id_part = |part: &str| -> String {
        let sanitized: String = part
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                    character
                } else {
                    '_'
                }
            })
            .collect();
        let normalized: String = if sanitized.chars().count() > 64 {
            sanitized.chars().take(64).collect()
        } else {
            sanitized
        };
        normalized.trim_end_matches('_').to_owned()
    };

    let build_foreign_responses_item_id = |item_id: &str| -> String {
        let normalized = format!("fc_{}", short_hash(item_id));
        if normalized.chars().count() > 64 {
            normalized.chars().take(64).collect()
        } else {
            normalized
        }
    };

    let normalize_tool_call_id: &ToolCallIdNormalizer<'_> =
        &|id: &str, _target_model: &Model, source: &AssistantMessage| {
            if !allowed_tool_call_providers.contains(model.provider.0.as_str()) {
                return normalize_id_part(id);
            }
            if !id.contains('|') {
                return normalize_id_part(id);
            }
            let mut parts = id.split('|');
            let call_id = parts.next().unwrap_or_default();
            let item_id = parts.next().unwrap_or_default();
            let normalized_call_id = normalize_id_part(call_id);
            #[allow(
                clippy::suspicious_operation_groupings,
                reason = "the foreign-tool-call check mirrors upstream's field-for-field comparison"
            )]
            let is_foreign_tool_call = source.provider != model.provider || source.api != model.api;
            let mut normalized_item_id = if is_foreign_tool_call {
                build_foreign_responses_item_id(item_id)
            } else {
                normalize_id_part(item_id)
            };
            // OpenAI Responses API requires item id to start with "fc"
            if !normalized_item_id.starts_with("fc_") {
                normalized_item_id = normalize_id_part(&format!("fc_{normalized_item_id}"));
            }
            format!("{normalized_call_id}|{normalized_item_id}")
        };

    let transformed_messages = transform_messages(
        context.messages.clone(),
        model,
        Some(normalize_tool_call_id),
    );

    let include_system_prompt = options
        .and_then(|options| options.include_system_prompt)
        .unwrap_or(true);
    if include_system_prompt
        && let Some(system_prompt) = context
            .system_prompt
            .as_deref()
            .filter(|prompt| !prompt.is_empty())
    {
        let supports_developer_role = model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_developer_role);
        // `compat?.supportsDeveloperRole !== false`: an unset field admits the
        // developer role.
        let role = if model.reasoning && supports_developer_role != Some(false) {
            "developer"
        } else {
            "system"
        };
        messages.push(json!({ "role": role, "content": system_prompt }));
    }

    let mut msg_index: usize = 0;
    for msg in &transformed_messages {
        match msg {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    messages.push(json!({
                        "role": "user",
                        "content": [{ "type": "input_text", "text": text }],
                    }));
                }
                UserContent::Blocks(blocks) => {
                    let parts: Vec<Value> = blocks
                        .iter()
                        .map(|block| match block {
                            UserBlock::Text(text) => {
                                json!({ "type": "input_text", "text": text.text })
                            }
                            UserBlock::Image(image) => json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!(
                                    "data:{};base64,{}",
                                    image.mime_type, image.data
                                ),
                            }),
                        })
                        .collect();
                    if parts.is_empty() {
                        msg_index += 1;
                        continue;
                    }
                    messages.push(json!({ "role": "user", "content": parts }));
                }
            },
            Message::Assistant(assistant) => {
                let output = convert_assistant_message(model, assistant, options, msg_index)?;
                if !output.is_empty() {
                    messages.extend(output);
                }
            }
            Message::ToolResult(result) => {
                convert_tool_result_message(
                    model,
                    result,
                    options,
                    &mut loaded_tool_names,
                    &mut messages,
                )?;
            }
        }
        msg_index += 1;
    }

    Ok(messages)
}

/// One assistant history message's input items, upstream's `msg.role ===
/// "assistant"` branch. Signed thinking replays as its serialized reasoning
/// item, text blocks replay with their message ids, and tool calls replay as
/// `function_call`/`custom_tool_call` items whose item ids keep the pairing
/// validation quiet.
///
/// # Errors
/// A thinking signature that does not parse, and the grammar tool-input
/// rejection when a custom tool call's arguments do not carry the string
/// input its grammar streams through.
#[expect(
    clippy::too_many_lines,
    reason = "the thinking / text / tool-call branches are one wire shape per block kind"
)]
fn convert_assistant_message(
    model: &Model,
    assistant: &AssistantMessage,
    options: Option<&ConvertResponsesMessagesOptions>,
    msg_index: usize,
) -> Result<Vec<Value>, String> {
    let mut output: Vec<Value> = Vec::new();
    #[allow(
        clippy::suspicious_operation_groupings,
        reason = "the same-provider check mirrors upstream's field-for-field comparison"
    )]
    let is_same_provider_and_api =
        assistant.provider == model.provider && assistant.api == model.api;
    #[allow(
        clippy::suspicious_operation_groupings,
        reason = "the same-model and different-model pair mirrors upstream's field-for-field comparison"
    )]
    let is_same_model = is_same_provider_and_api && assistant.model == model.id;
    let is_different_model = is_same_provider_and_api && assistant.model != model.id;
    let mut text_block_index: usize = 0;

    for block in &assistant.content {
        match block {
            AssistantBlock::Thinking(thinking) => {
                if let Some(signature) = thinking
                    .thinking_signature
                    .as_deref()
                    .filter(|signature| !signature.is_empty())
                {
                    let reasoning_item: Value =
                        serde_json::from_str(signature).map_err(|error| error.to_string())?;
                    output.push(reasoning_item);
                }
            }
            AssistantBlock::Text(text) => {
                let parsed_signature = parse_text_signature(text.text_signature.as_deref());
                // OpenAI requires id to be max 64 characters
                let fallback_message_id = if text_block_index == 0 {
                    format!("msg_pi_{msg_index}")
                } else {
                    format!("msg_pi_{msg_index}_{text_block_index}")
                };
                text_block_index += 1;
                let msg_id = parsed_signature
                    .as_ref()
                    .map(|signature| signature.id.clone())
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| fallback_message_id.clone());
                let msg_id = if msg_id.chars().count() > 64 {
                    format!("msg_{}", short_hash(&msg_id))
                } else {
                    msg_id
                };
                let mut message = Map::new();
                message.insert("type".to_owned(), json!("message"));
                message.insert("role".to_owned(), json!("assistant"));
                message.insert(
                    "content".to_owned(),
                    json!([{ "type": "output_text", "text": text.text, "annotations": [] }]),
                );
                message.insert("status".to_owned(), json!("completed"));
                message.insert("id".to_owned(), json!(msg_id));
                if let Some(phase) = parsed_signature.and_then(|signature| signature.phase) {
                    message.insert("phase".to_owned(), json!(phase));
                }
                output.push(Value::Object(message));
            }
            AssistantBlock::ToolCall(tool_call) => {
                let mut id_parts = tool_call.id.split('|');
                let call_id = id_parts.next().unwrap_or_default();
                let item_id_raw = id_parts.next();
                let custom_input_property = options
                    .and_then(|options| options.grammar_tool_input_properties.as_ref())
                    .and_then(|properties| properties.get(&tool_call.name));
                let mut item_id: Option<String> = item_id_raw.map(str::to_owned);

                let starts_with_fc = item_id
                    .as_deref()
                    .is_some_and(|item_id| item_id.starts_with("fc_"));
                // For different-model messages, set id to undefined to avoid
                // pairing validation. OpenAI tracks which fc_xxx IDs were
                // paired with rs_xxx reasoning items. By omitting the id, we
                // avoid triggering that validation (like cross-provider does).
                // When replaying custom-tool calls as a function_call, also
                // drop non-fc_* ids such as ctc_* custom-tool ids because
                // function_call item ids must be fc_*.
                if (is_different_model && starts_with_fc)
                    || (custom_input_property.is_none() && !starts_with_fc)
                {
                    item_id = None;
                }

                let can_replay_namespace = is_same_model
                    || options
                        .and_then(|options| options.deferred_tools.as_ref())
                        .is_some_and(|deferred| deferred.contains_key(&tool_call.name));

                let item = if let Some(property) = custom_input_property {
                    let input =
                        get_grammar_tool_input(&tool_call.name, &tool_call.arguments, property)?;
                    let mut entry = Map::new();
                    entry.insert("type".to_owned(), json!("custom_tool_call"));
                    if let Some(item_id) = &item_id {
                        entry.insert("id".to_owned(), json!(item_id));
                    }
                    entry.insert("call_id".to_owned(), json!(call_id));
                    entry.insert("name".to_owned(), json!(tool_call.name));
                    entry.insert("input".to_owned(), json!(input));
                    if can_replay_namespace && let Some(namespace) = &tool_call.namespace {
                        entry.insert("namespace".to_owned(), json!(namespace));
                    }
                    Value::Object(entry)
                } else {
                    let mut entry = Map::new();
                    entry.insert("type".to_owned(), json!("function_call"));
                    if let Some(item_id) = &item_id {
                        entry.insert("id".to_owned(), json!(item_id));
                    }
                    entry.insert("call_id".to_owned(), json!(call_id));
                    entry.insert("name".to_owned(), json!(tool_call.name));
                    entry.insert(
                        "arguments".to_owned(),
                        json!(serde_json::to_string(&tool_call.arguments).unwrap_or_default()),
                    );
                    if can_replay_namespace && let Some(namespace) = &tool_call.namespace {
                        entry.insert("namespace".to_owned(), json!(namespace));
                    }
                    Value::Object(entry)
                };
                output.push(item);
            }
        }
    }
    Ok(output)
}

/// One tool-result run's input items, upstream's `msg.role === "toolResult"`
/// branch: the call output (`function_call_output` or, when the tool streams
/// grammar input, `custom_tool_call_output`), then the deferred tools the
/// result loads through the `additional_tools` item or the
/// `tool_search_call`/`tool_search_output` pair.
///
/// # Errors
/// The tool-definition conversion failures a deferred-tools item hits.
fn convert_tool_result_message(
    model: &Model,
    result: &ToolResultMessage,
    options: Option<&ConvertResponsesMessagesOptions>,
    loaded_tool_names: &mut BTreeSet<String>,
    messages: &mut Vec<Value>,
) -> Result<(), String> {
    let call_id = result.tool_call_id.split('|').next().unwrap_or_default();
    let output = convert_tool_result_output(model, &result.content);
    let is_custom = options
        .and_then(|options| options.grammar_tool_input_properties.as_ref())
        .is_some_and(|properties| properties.contains_key(&result.tool_name));
    if is_custom {
        messages.push(json!({
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": output,
        }));
    } else {
        messages.push(json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }));
    }

    let mut deferred_tools: Vec<Tool> = Vec::new();
    for name in result.added_tool_names.iter().flatten() {
        let Some(tool) = options
            .and_then(|options| options.deferred_tools.as_ref())
            .and_then(|deferred| deferred.get(name))
        else {
            continue;
        };
        if loaded_tool_names.contains(name) {
            continue;
        }
        loaded_tool_names.insert(name.clone());
        deferred_tools.push(tool.clone());
    }
    let deferred_tools_mode = options.and_then(|options| options.deferred_tools_mode);
    if !deferred_tools.is_empty()
        && deferred_tools_mode == Some(ResponsesDeferredToolsMode::AdditionalTools)
    {
        messages.push(json!({
            "type": "additional_tools",
            "role": "developer",
            "tools": convert_responses_tools(
                &deferred_tools,
                options.and_then(|options| options.tool_options).as_ref(),
            )?,
        }));
    } else if !deferred_tools.is_empty()
        && deferred_tools_mode == Some(ResponsesDeferredToolsMode::ToolSearch)
    {
        let names: Vec<&str> = deferred_tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        let search_call_id = format!(
            "pi_tool_load_{}",
            short_hash(&format!("{}:{}", result.tool_call_id, names.join(",")))
        );
        messages.push(json!({
            "type": "tool_search_call",
            "call_id": search_call_id,
            "execution": "client",
            "status": "completed",
            "arguments": { "query": names.join(" "), "limit": names.len() },
        }));
        let mut tool_options = options
            .and_then(|options| options.tool_options)
            .unwrap_or_default();
        tool_options.defer_loading = Some(true);
        messages.push(json!({
            "type": "tool_search_output",
            "call_id": search_call_id,
            "execution": "client",
            "status": "completed",
            "tools": convert_responses_tools(&deferred_tools, Some(&tool_options))?,
        }));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool conversion
// ---------------------------------------------------------------------------

/// The request's tool definitions, upstream's `convertResponsesTools`.
///
/// OpenAI Responses tools carry `name` directly (not nested under a
/// `function` object), grammar tools ride the `custom` shape, and `strict`
/// is present only when the provider accepts it.
///
/// # Errors
/// The grammar constrained-sampling rejection for a tool that opted in
/// without a usable variant or schema, and the strict-conversion rejection
/// for a tool that requires strict JSON-schema sampling.
pub fn convert_responses_tools(
    tools: &[Tool],
    options: Option<&ConvertResponsesToolsOptions>,
) -> Result<Vec<Value>, String> {
    // `strict === undefined` means the default `false`; the wire's `null`
    // rides through to the tool entry.
    let default_strict: Option<bool> = options
        .and_then(|options| options.strict)
        .unwrap_or(Some(false));
    let supports_strict_mode = options
        .and_then(|options| options.supports_strict_mode)
        .unwrap_or(true);
    let supports_openai_grammar_tools = options
        .and_then(|options| options.supports_openai_grammar_tools)
        .unwrap_or(false);
    let defer_loading = options
        .and_then(|options| options.defer_loading)
        .unwrap_or(false);

    let mut converted: Vec<Value> = Vec::new();
    for tool in tools {
        let grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?;
        if let Some(grammar) = grammar {
            let mut entry = Map::new();
            entry.insert("type".to_owned(), json!("custom"));
            entry.insert("name".to_owned(), json!(tool.name));
            entry.insert("description".to_owned(), json!(tool.description));
            entry.insert(
                "format".to_owned(),
                json!({
                    "type": "grammar",
                    "syntax": grammar.format,
                    "definition": grammar.definition,
                }),
            );
            if defer_loading {
                entry.insert("defer_loading".to_owned(), json!(true));
            }
            converted.push(Value::Object(entry));
            continue;
        }

        let constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)?;
        let strict = constrained_strict.or(default_strict);
        let parameters =
            get_json_schema_tool_parameters(tool, (strict == Some(true)).then_some(true))
                .map_err(|error| error.0)?;
        let mut function = Map::new();
        function.insert("type".to_owned(), json!("function"));
        function.insert("name".to_owned(), json!(tool.name));
        function.insert("description".to_owned(), json!(tool.description));
        function.insert("parameters".to_owned(), parameters);
        if defer_loading {
            function.insert("defer_loading".to_owned(), json!(true));
        }
        if supports_strict_mode {
            function.insert("strict".to_owned(), json!(strict));
        }
        converted.push(Value::Object(function));
    }
    Ok(converted)
}

// ---------------------------------------------------------------------------
// Stream processing
// ---------------------------------------------------------------------------

/// The streaming scratch of one tool-call block, upstream's `partialJson` /
/// `customInput` fields on the block object. The scratch never persists into
/// the message; the `output_item.done` handling drops it when the block
/// closes.
#[derive(Debug, Default)]
struct ToolCallScratch {
    /// The streamed arguments JSON a function call accumulates, upstream's
    /// `partialJson`. `None` once the item closed.
    partial_json: Option<String>,
    /// The grammar buffer a custom tool call's raw input streams through,
    /// upstream's `customInput`.
    custom_input: Option<CustomInputState>,
}

/// The grammar buffer a custom tool call's raw input streams through,
/// upstream's `customInput`.
#[derive(Debug)]
struct CustomInputState {
    /// The arguments property the grammar's input feeds.
    property: String,
    /// The incremental `{"property":"...","}` wrapper state.
    json_buffer: GrammarToolInputJsonBuffer,
}

/// The output slot an `output_index` maps to, upstream's `ResponsesOutputSlot`:
/// the open block's position in the message content, discriminated by kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputSlot {
    /// A `reasoning` item streaming into a thinking block.
    Thinking(usize),
    /// A `message` item streaming into a text block.
    Text(usize),
    /// A `function_call`/`custom_tool_call` item streaming into a tool-call
    /// block.
    ToolCall(usize),
}

/// The loop-carried state of a Responses event stream: the open output
/// slots keyed by the wire's `output_index`, the reasoning blocks whose
/// signatures a terminal response may backfill keyed by the wire's item id,
/// the tool-call scratch keyed by content position, and whether a terminal
/// event arrived.
#[derive(Debug, Default)]
struct ResponsesStreamState {
    slots: HashMap<u64, OutputSlot>,
    reasoning_blocks_by_id: HashMap<String, usize>,
    scratch: HashMap<usize, ToolCallScratch>,
    saw_terminal_response_event: bool,
}

impl ResponsesStreamState {
    /// The block position of the open slot at `output_index`, when it
    /// streams the requested kind.
    fn slot_of(&self, output_index: u64, kind: OutputSlotKind) -> Option<usize> {
        let index = match (self.slots.get(&output_index), kind) {
            (Some(OutputSlot::Thinking(index)), OutputSlotKind::Thinking)
            | (Some(OutputSlot::Text(index)), OutputSlotKind::Text)
            | (Some(OutputSlot::ToolCall(index)), OutputSlotKind::ToolCall) => *index,
            _ => return None,
        };
        Some(index)
    }
}

/// The slot discriminators [`ResponsesStreamState::slot_of`] filters by.
#[derive(Clone, Copy, Debug)]
enum OutputSlotKind {
    Thinking,
    Text,
    ToolCall,
}

/// The raw input a custom tool call has accumulated, upstream's
/// `getCustomToolCallInput`: the arguments value under the property, `""`
/// when absent or not a string.
fn get_custom_tool_call_input(block: &ToolCall, scratch: &ToolCallScratch) -> String {
    let Some(custom) = scratch.custom_input.as_ref() else {
        return String::new();
    };
    block
        .arguments
        .get(&custom.property)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// Restage the arguments object around the next input, upstream's
/// `appendCustomToolCallInput`: the grammar buffer extends the streamed JSON
/// fragment and the arguments carry the raw input under the property.
///
/// # Errors
/// The grammar buffer's rejection when the input changes non-monotonically
/// or after the property closed.
fn append_custom_tool_call_input(
    block: &mut ToolCall,
    scratch: &mut ToolCallScratch,
    next_input: &str,
    close: bool,
) -> Result<Option<String>, String> {
    let Some(custom) = scratch.custom_input.as_mut() else {
        return Ok(None);
    };
    let delta = append_grammar_tool_input_json_delta(
        &mut custom.json_buffer,
        &custom.property,
        next_input,
        close,
    )?;
    block.arguments = Map::from_iter([(custom.property.clone(), json!(next_input))]);
    Ok(delta)
}

/// Apply the message's phase to the stop reason, upstream's
/// `applyMessagePhaseStopReason`: a `final_answer` message marks the turn
/// finished even before the terminal event.
fn apply_message_phase_stop_reason(item: &Value, output: &mut AssistantMessage) {
    if item.get("type").and_then(Value::as_str) == Some("message")
        && item.get("phase").and_then(Value::as_str) == Some("final_answer")
    {
        output.stop_reason = StopReason::Stop;
    }
}

/// Push a tool-call delta when the scratch produced one, upstream's
/// `pushToolCallDelta`.
fn push_tool_call_delta(
    events: &AssistantMessageEventStream,
    output: &AssistantMessage,
    content_index: usize,
    delta: Option<String>,
) {
    if let Some(delta) = delta {
        events.push(AssistantMessageEvent::ToolcallDelta {
            content_index: content_index as u64,
            delta,
            partial: output.clone(),
        });
    }
}

/// Open (or reopen) the output slot a `response.output_item.added` or
/// `response.output_item.done` item streams into, upstream's `createSlot`:
/// thinking, text, and tool-call items claim a content position and push
/// their `*_start` event; other item types claim nothing.
///
/// Function calls stream their arguments through the scratch's
/// [`ToolCallScratch::partial_json`]; custom tool calls stream raw input
/// through the grammar buffer under the property the grammar map names,
/// defaulting to `"input"`.
#[expect(
    clippy::too_many_lines,
    reason = "one branch per streamable output item type, each a wire shape"
)]
fn create_slot(
    output_index: u64,
    item: &Value,
    output: &mut AssistantMessage,
    state: &mut ResponsesStreamState,
    events: &AssistantMessageEventStream,
    options: Option<&OpenAiResponsesStreamOptions>,
) -> Option<OutputSlot> {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
    match item_type {
        "reasoning" => {
            output
                .content
                .push(AssistantBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                    redacted: None,
                }));
            let content_index = output.content.len() - 1;
            let slot = OutputSlot::Thinking(content_index);
            state.slots.insert(output_index, slot);
            events.push(AssistantMessageEvent::ThinkingStart {
                content_index: content_index as u64,
                partial: output.clone(),
            });
            Some(slot)
        }
        "message" => {
            apply_message_phase_stop_reason(item, output);
            output.content.push(AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            }));
            let content_index = output.content.len() - 1;
            let slot = OutputSlot::Text(content_index);
            state.slots.insert(output_index, slot);
            events.push(AssistantMessageEvent::TextStart {
                content_index: content_index as u64,
                partial: output.clone(),
            });
            Some(slot)
        }
        "function_call" => {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default();
            let namespace = item
                .get("namespace")
                .and_then(Value::as_str)
                .map(str::to_owned);
            output.content.push(AssistantBlock::ToolCall(ToolCall {
                id: format!("{call_id}|{item_id}"),
                name: item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments: Map::new(),
                thought_signature: None,
                namespace,
            }));
            let content_index = output.content.len() - 1;
            let slot = OutputSlot::ToolCall(content_index);
            state.slots.insert(output_index, slot);
            state.scratch.insert(
                content_index,
                ToolCallScratch {
                    partial_json: Some(
                        item.get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    custom_input: None,
                },
            );
            events.push(AssistantMessageEvent::ToolcallStart {
                content_index: content_index as u64,
                partial: output.clone(),
            });
            Some(slot)
        }
        "custom_tool_call" => {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let input_property = options
                .and_then(|options| options.grammar_tool_input_properties.as_ref())
                .and_then(|properties| properties.get(&name))
                .map_or_else(|| "input".to_owned(), Clone::clone);
            let input = item
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let mut arguments = Map::new();
            arguments.insert(input_property.clone(), json!(input));
            let namespace = item
                .get("namespace")
                .and_then(Value::as_str)
                .map(str::to_owned);
            output.content.push(AssistantBlock::ToolCall(ToolCall {
                id: format!(
                    "{}|{}",
                    item.get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    item.get("id").and_then(Value::as_str).unwrap_or_default()
                ),
                name,
                arguments,
                thought_signature: None,
                namespace,
            }));
            let content_index = output.content.len() - 1;
            let slot = OutputSlot::ToolCall(content_index);
            state.slots.insert(output_index, slot);
            state.scratch.insert(
                content_index,
                ToolCallScratch {
                    partial_json: None,
                    custom_input: Some(CustomInputState {
                        property: input_property,
                        json_buffer: GrammarToolInputJsonBuffer::default(),
                    }),
                },
            );
            events.push(AssistantMessageEvent::ToolcallStart {
                content_index: content_index as u64,
                partial: output.clone(),
            });
            Some(slot)
        }
        _ => None,
    }
}

/// Process the Responses event stream into the accumulator and the event
/// stream, upstream's `processResponsesStream`.
///
/// The response lifecycle events drive the output-slot state machine, and
/// every stream must reach a terminal
/// `response.completed`/`response.incomplete`/`response.failed`.
///
/// # Errors
/// A frame that does not parse, a wire `error` event, a `response.failed`
/// terminal, the grammar-input rejections of a custom tool call, a reasoning
/// item that does not serialize, an unhandled response status, and a stream
/// that ends before any terminal response event.
pub async fn process_responses_stream(
    response: HttpResponse,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
    model: &Model,
    options: Option<&OpenAiResponsesStreamOptions>,
) -> Result<(), String> {
    let mut state = ResponsesStreamState::default();
    let mut sse = SseStream::new(response.body);
    while let Some(sse_event) = sse.next().await.map_err(|error| match error {
        HttpError::Aborted => "Request was aborted".to_owned(),
        other => other.to_string(),
    })? {
        if sse_event.data.trim() == "[DONE]" {
            continue;
        }
        let event = serde_json::from_str::<Value>(&sse_event.data).map_err(|error| {
            parse_event_error_message(&error.to_string(), &sse_event.data, &sse_event.raw)
        })?;
        if !event.is_object() {
            continue;
        }
        apply_event(&event, output, &mut state, model, options, events)?;
    }
    if !state.saw_terminal_response_event {
        return Err("OpenAI Responses stream ended before a terminal response event".to_owned());
    }
    Ok(())
}

/// The parse-failure message of one SSE frame, the seam-side port of the
/// pinned SDK's stream decode rejection.
fn parse_event_error_message(detail: &str, data: &str, raw: &[String]) -> String {
    format!(
        "Could not parse OpenAI Responses SSE event: {detail}; data={data}; raw={}",
        raw.join("\\n")
    )
}

/// Apply one stream event to the accumulator, upstream's loop body: the
/// slot state machine over the `response.*` event family, the terminal
/// events, and the in-stream error forms.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per wire event type; splitting them would scatter the state machine the tests replay"
)]
fn apply_event(
    event: &Value,
    output: &mut AssistantMessage,
    state: &mut ResponsesStreamState,
    model: &Model,
    options: Option<&OpenAiResponsesStreamOptions>,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let output_index = event
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    match event_type {
        "response.created" => {
            if let Some(id) = event
                .get("response")
                .and_then(|response| response.get("id"))
                .and_then(Value::as_str)
            {
                output.response_id = Some(id.to_owned());
            }
        }
        "response.output_item.added" => {
            let item = event.get("item").unwrap_or(&Value::Null);
            create_slot(output_index, item, output, state, events, options);
        }
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            let Some(content_index) = state.slot_of(output_index, OutputSlotKind::Thinking) else {
                return Ok(());
            };
            let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                return Ok(());
            };
            if let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(content_index) {
                block.thinking.push_str(delta);
            }
            events.push(AssistantMessageEvent::ThinkingDelta {
                content_index: content_index as u64,
                delta: delta.to_owned(),
                partial: output.clone(),
            });
        }
        "response.reasoning_summary_part.done" => {
            let Some(content_index) = state.slot_of(output_index, OutputSlotKind::Thinking) else {
                return Ok(());
            };
            if let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(content_index) {
                block.thinking.push_str("\n\n");
            }
            events.push(AssistantMessageEvent::ThinkingDelta {
                content_index: content_index as u64,
                delta: "\n\n".to_owned(),
                partial: output.clone(),
            });
        }
        "response.output_text.delta" | "response.refusal.delta" => {
            let Some(content_index) = state.slot_of(output_index, OutputSlotKind::Text) else {
                return Ok(());
            };
            let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                return Ok(());
            };
            if let Some(AssistantBlock::Text(block)) = output.content.get_mut(content_index) {
                block.text.push_str(delta);
            }
            events.push(AssistantMessageEvent::TextDelta {
                content_index: content_index as u64,
                delta: delta.to_owned(),
                partial: output.clone(),
            });
        }
        "response.function_call_arguments.delta" => {
            let Some(content_index) = state.slot_of(output_index, OutputSlotKind::ToolCall) else {
                return Ok(());
            };
            let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                return Ok(());
            };
            let Some(scratch) = state.scratch.get_mut(&content_index) else {
                return Ok(());
            };
            let Some(partial_json) = scratch.partial_json.as_mut() else {
                return Ok(());
            };
            partial_json.push_str(delta);
            if let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index) {
                block.arguments = parse_streaming_json(Some(partial_json))
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
            }
            events.push(AssistantMessageEvent::ToolcallDelta {
                content_index: content_index as u64,
                delta: delta.to_owned(),
                partial: output.clone(),
            });
        }
        "response.function_call_arguments.done" => {
            let Some(content_index) = state.slot_of(output_index, OutputSlotKind::ToolCall) else {
                return Ok(());
            };
            let Some(arguments) = event.get("arguments").and_then(Value::as_str) else {
                return Ok(());
            };
            let Some(scratch) = state.scratch.get_mut(&content_index) else {
                return Ok(());
            };
            let Some(partial_json) = scratch.partial_json.as_mut() else {
                return Ok(());
            };
            let previous_partial_json = std::mem::take(partial_json);
            arguments.clone_into(partial_json);
            if let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index) {
                block.arguments = parse_streaming_json(Some(arguments))
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
            }
            if arguments.starts_with(&previous_partial_json) {
                let delta = &arguments[previous_partial_json.len()..];
                if !delta.is_empty() {
                    events.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: content_index as u64,
                        delta: delta.to_owned(),
                        partial: output.clone(),
                    });
                }
            }
        }
        "response.custom_tool_call_input.delta" => {
            let Some(content_index) = state.slot_of(output_index, OutputSlotKind::ToolCall) else {
                return Ok(());
            };
            let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                return Ok(());
            };
            if state
                .scratch
                .get(&content_index)
                .is_none_or(|scratch| scratch.custom_input.is_none())
            {
                return Ok(());
            }
            let appended = {
                let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                let Some(scratch) = state.scratch.get_mut(&content_index) else {
                    return Ok(());
                };
                let current = get_custom_tool_call_input(block, scratch);
                let next_input = format!("{current}{delta}");
                append_custom_tool_call_input(block, scratch, &next_input, false)?
            };
            push_tool_call_delta(events, output, content_index, appended);
        }
        "response.custom_tool_call_input.done" => {
            let Some(content_index) = state.slot_of(output_index, OutputSlotKind::ToolCall) else {
                return Ok(());
            };
            if state
                .scratch
                .get(&content_index)
                .is_none_or(|scratch| scratch.custom_input.is_none())
            {
                return Ok(());
            }
            let next_input = {
                let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                let Some(scratch) = state.scratch.get_mut(&content_index) else {
                    return Ok(());
                };
                let current = get_custom_tool_call_input(block, scratch);
                // `event.input ?? current`: the wire's `null` and an absent
                // field both fall back to the streamed input.
                event
                    .get("input")
                    .and_then(Value::as_str)
                    .map_or_else(|| current.clone(), str::to_owned)
            };
            let (appended, tool_call) = {
                let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                let Some(scratch) = state.scratch.get_mut(&content_index) else {
                    return Ok(());
                };
                let appended = append_custom_tool_call_input(block, scratch, &next_input, true)?;
                (appended, block.clone())
            };
            push_tool_call_delta(events, output, content_index, appended);
            state.scratch.remove(&content_index);
            events.push(AssistantMessageEvent::ToolcallEnd {
                content_index: content_index as u64,
                tool_call,
                partial: output.clone(),
            });
            state.slots.remove(&output_index);
        }
        "response.output_item.done" => {
            let item = event.get("item").unwrap_or(&Value::Null);
            apply_message_phase_stop_reason(item, output);
            let slot = state
                .slots
                .get(&output_index)
                .copied()
                .or_else(|| create_slot(output_index, item, output, state, events, options));
            finish_output_item(item, output_index, slot, output, state, events)?;
        }
        "response.completed" | "response.incomplete" => {
            state.saw_terminal_response_event = true;
            let response = event.get("response").unwrap_or(&Value::Null);
            finalize_response(response, output, model, options, state)?;
        }
        "error" => {
            let code = event
                .get("code")
                .map_or_else(|| "undefined".to_owned(), wire_string);
            let message = event
                .get("message")
                .map_or_else(|| "undefined".to_owned(), wire_string);
            return Err(format!("Error Code {code}: {message}"));
        }
        "response.failed" => {
            state.saw_terminal_response_event = true;
            if let Some(status) = response_field(event, "status").and_then(Value::as_str) {
                output.raw_stop_reason = Some(status.to_owned());
            }
            let error = response_field(event, "error").filter(|error| is_truthy(error));
            let details = response_field(event, "incomplete_details");
            return Err(error.map_or_else(
                || {
                    details
                        .and_then(|details| details.get("reason"))
                        .and_then(Value::as_str)
                        .filter(|reason| !reason.is_empty())
                        .map_or_else(
                            || "Unknown error (no error details in response)".to_owned(),
                            |reason| format!("incomplete: {reason}"),
                        )
                },
                |error| {
                    let code = error
                        .get("code")
                        .and_then(Value::as_str)
                        .filter(|code| !code.is_empty())
                        .unwrap_or("unknown");
                    let message = error
                        .get("message")
                        .and_then(Value::as_str)
                        .filter(|message| !message.is_empty())
                        .unwrap_or("no message");
                    format!("{code}: {message}")
                },
            ));
        }
        _ => {}
    }
    Ok(())
}

/// Render a wire field the way the template literals upstream composes errors
/// with do: absent is `undefined`, `null` stays `null`, strings verbatim.
fn wire_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "null".to_owned(),
        other => other.to_string(),
    }
}

/// The `response` object of a `response.failed` event, empty when absent.
fn response_field<'a>(event: &'a Value, field: &str) -> Option<&'a Value> {
    event
        .get("response")
        .and_then(|response| response.get(field))
}

/// Close the open slot one `response.output_item.done` item belongs to,
/// upstream's `response.output_item.done` branch: authoritative text,
/// thinking signature, parsed tool-call arguments, and the `*_end` events.
#[expect(
    clippy::too_many_lines,
    reason = "one branch per terminal item type, each a wire shape; splitting them would scatter the close contract"
)]
fn finish_output_item(
    item: &Value,
    output_index: u64,
    slot: Option<OutputSlot>,
    output: &mut AssistantMessage,
    state: &mut ResponsesStreamState,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
    match (item_type, slot) {
        ("reasoning", Some(OutputSlot::Thinking(content_index))) => {
            // The wire's summary parts join with blank lines, as do the
            // reasoning-text parts; the streamed text only survives when the
            // terminal item carries neither.
            let summary_text = joined_part_text(item.get("summary"));
            let content_text = joined_part_text(item.get("content"));
            let thinking = {
                let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                if !summary_text.is_empty() {
                    block.thinking = summary_text;
                } else if !content_text.is_empty() {
                    block.thinking = content_text;
                }
                block.thinking_signature = Some(serde_json::to_string(item).unwrap_or_default());
                block.thinking.clone()
            };
            if let Some(id) = item.get("id").and_then(Value::as_str) {
                state
                    .reasoning_blocks_by_id
                    .insert(id.to_owned(), content_index);
            }
            events.push(AssistantMessageEvent::ThinkingEnd {
                content_index: content_index as u64,
                content: thinking,
                partial: output.clone(),
            });
            state.slots.remove(&output_index);
        }
        ("message", Some(OutputSlot::Text(content_index))) => {
            let text = item
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .map(|part| {
                            if part.get("type").and_then(Value::as_str) == Some("output_text") {
                                part.get("text").and_then(Value::as_str).unwrap_or_default()
                            } else {
                                part.get("refusal")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            let phase = item.get("phase").and_then(Value::as_str);
            let signature = encode_text_signature_v1(
                item.get("id").and_then(Value::as_str).unwrap_or_default(),
                phase,
            );
            let content = {
                let Some(AssistantBlock::Text(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                block.text = text;
                block.text_signature = Some(signature);
                block.text.clone()
            };
            events.push(AssistantMessageEvent::TextEnd {
                content_index: content_index as u64,
                content,
                partial: output.clone(),
            });
            state.slots.remove(&output_index);
        }
        ("function_call", Some(OutputSlot::ToolCall(content_index))) => {
            let partial_json = state
                .scratch
                .get(&content_index)
                .and_then(|scratch| scratch.partial_json.clone());
            let Some(partial_json) = partial_json else {
                return Ok(());
            };
            // `item.arguments || slot.block.partialJson || "{}"`: the first
            // non-empty string wins.
            let source = item
                .get("arguments")
                .and_then(Value::as_str)
                .filter(|arguments| !arguments.is_empty())
                .or_else(|| (!partial_json.is_empty()).then_some(partial_json.as_str()))
                .unwrap_or("{}");
            let namespace = item
                .get("namespace")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let tool_call = {
                let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                block.arguments = parse_streaming_json(Some(source))
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                if let Some(namespace) = namespace {
                    block.namespace = Some(namespace);
                }
                // Finalize in place and strip the scratch buffer so replay
                // only carries parsed arguments.
                block.clone()
            };
            state.scratch.remove(&content_index);
            events.push(AssistantMessageEvent::ToolcallEnd {
                content_index: content_index as u64,
                tool_call,
                partial: output.clone(),
            });
            state.slots.remove(&output_index);
        }
        ("custom_tool_call", Some(OutputSlot::ToolCall(content_index))) => {
            if state
                .scratch
                .get(&content_index)
                .is_none_or(|scratch| scratch.custom_input.is_none())
            {
                return Ok(());
            }
            let next_input = {
                let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                let Some(scratch) = state.scratch.get_mut(&content_index) else {
                    return Ok(());
                };
                let current = get_custom_tool_call_input(block, scratch);
                // `item.input ?? current`: the wire's `null` falls back to the
                // streamed input, an empty string does not.
                item.get("input")
                    .and_then(Value::as_str)
                    .map_or_else(|| current.clone(), str::to_owned)
            };
            let (appended, tool_call) = {
                let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index)
                else {
                    return Ok(());
                };
                let Some(scratch) = state.scratch.get_mut(&content_index) else {
                    return Ok(());
                };
                let appended = append_custom_tool_call_input(block, scratch, &next_input, true)?;
                let namespace = item
                    .get("namespace")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let Some(namespace) = namespace {
                    block.namespace = Some(namespace);
                }
                (appended, block.clone())
            };
            push_tool_call_delta(events, output, content_index, appended);
            state.scratch.remove(&content_index);
            events.push(AssistantMessageEvent::ToolcallEnd {
                content_index: content_index as u64,
                tool_call,
                partial: output.clone(),
            });
            state.slots.remove(&output_index);
        }
        _ => {}
    }
    Ok(())
}

/// The joined `text` fields of a reasoning item's summary/content part array,
/// upstream's `summary?.map((s) => s.text).join("\n\n") || ""`.
fn joined_part_text(parts: Option<&Value>) -> String {
    parts
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default()
}

/// Settle the terminal `response.completed`/`response.incomplete` response,
/// upstream's `finalizeResponse`: reasoning-signature backfill, response
/// identity, provider-reported usage, service-tier pricing, and the status
/// mapping.
///
/// # Errors
/// The unhandled-status rejection [`map_stop_reason`] raises.
fn finalize_response(
    response: &Value,
    output: &mut AssistantMessage,
    model: &Model,
    options: Option<&OpenAiResponsesStreamOptions>,
    state: &ResponsesStreamState,
) -> Result<(), String> {
    let response_output = response
        .get("output")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    backfill_reasoning_signatures(response_output, output, &state.reasoning_blocks_by_id)?;
    if let Some(id) = response
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        output.response_id = Some(id.to_owned());
    }
    if let Some(usage) = response.get("usage").filter(|usage| is_truthy(usage)) {
        let input_details = usage.get("input_tokens_details");
        let cached_tokens = input_details
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let cache_write_tokens = input_details
            .and_then(|details| details.get("cache_write_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let input_tokens = usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output_tokens = usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        // OpenAI includes cached and cache-write tokens in input_tokens, so
        // subtract both.
        let mut parsed_usage = Usage {
            input: input_tokens
                .saturating_sub(cached_tokens)
                .saturating_sub(cache_write_tokens),
            output: output_tokens,
            cache_read: cached_tokens,
            cache_write: cache_write_tokens,
            reasoning: Some(
                usage
                    .get("output_tokens_details")
                    .and_then(|details| details.get("reasoning_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            ),
            total_tokens: usage
                .get("total_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            ..Usage::default()
        };
        calculate_cost(model, &mut parsed_usage);
        output.usage = parsed_usage;
    }
    if let Some(options) = options
        && let Some(pricing) = &options.apply_service_tier_pricing
    {
        let response_tier = response.get("service_tier").and_then(Value::as_str);
        let request_tier = options.service_tier.as_deref();
        let service_tier = options.resolve_service_tier.as_ref().map_or_else(
            || {
                response_tier
                    .map(str::to_owned)
                    .or_else(|| options.service_tier.clone())
            },
            |resolve| resolve(response_tier, request_tier),
        );
        pricing(&mut output.usage, service_tier.as_deref());
    }
    // Map status to stop reason. For incomplete responses, retain the
    // provider's specific reason so max-output truncation and content
    // filtering stay distinct.
    let status = response.get("status").and_then(Value::as_str);
    let incomplete_reason = response
        .get("incomplete_details")
        .and_then(|details| details.get("reason"))
        .and_then(Value::as_str);
    output.raw_stop_reason = incomplete_reason.map_or_else(
        || status.map(str::to_owned),
        |reason| Some(format!("{}.{}", status.unwrap_or("undefined"), reason)),
    );
    let (stop_reason, error_message) = map_stop_reason(status, incomplete_reason)?;
    output.stop_reason = stop_reason;
    output.error_message = error_message;
    if output
        .content
        .iter()
        .any(|block| matches!(block, AssistantBlock::ToolCall(_)))
        && output.stop_reason == StopReason::Stop
    {
        output.stop_reason = StopReason::ToolUse;
    }
    Ok(())
}

/// Azure OpenAI can omit `reasoning.encrypted_content` from
/// `response.output_item.done` and provide it only in
/// `response.completed.response.output`. Backfill the persisted reasoning
/// signature from the terminal response to keep `store: false` multi-turn
/// replay stateless. See <https://github.com/earendil-works/pi/issues/6409>.
///
/// # Errors
/// A persisted reasoning signature that does not parse.
fn backfill_reasoning_signatures(
    response_output: &[Value],
    output: &mut AssistantMessage,
    reasoning_blocks_by_id: &HashMap<String, usize>,
) -> Result<(), String> {
    for item in response_output {
        if item.get("type").and_then(Value::as_str) != Some("reasoning") {
            continue;
        }
        let Some(encrypted_content) = item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        else {
            continue;
        };
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(&content_index) = reasoning_blocks_by_id.get(id) else {
            continue;
        };
        let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(content_index) else {
            continue;
        };
        let Some(signature) = block
            .thinking_signature
            .as_deref()
            .filter(|signature| !signature.is_empty())
        else {
            continue;
        };
        let mut stored: Value =
            serde_json::from_str(signature).map_err(|error| error.to_string())?;
        if stored.get("encrypted_content").is_some_and(is_truthy) {
            continue;
        }
        if let Some(object) = stored.as_object_mut() {
            // Map::insert keeps an existing key's position, matching the
            // spread's in-place overwrite.
            object.insert("encrypted_content".to_owned(), json!(encrypted_content));
        }
        block.thinking_signature = Some(stored.to_string());
    }
    Ok(())
}

/// The stop reason a terminal response status maps to, upstream's
/// `mapStopReason`; `incomplete` keeps the provider's specific reason so
/// max-output truncation and content filtering stay distinct.
///
/// # Errors
/// The unhandled-status rejection upstream's exhaustive `never` branch
/// raises.
#[expect(
    clippy::match_same_arms,
    reason = "the arms mirror upstream's status switch, where each status is a distinct wire state"
)]
fn map_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
) -> Result<(StopReason, Option<String>), String> {
    let Some(status) = status else {
        return Ok((StopReason::Stop, None));
    };
    match status {
        "completed" => Ok((StopReason::Stop, None)),
        "incomplete" if incomplete_reason == Some("max_output_tokens") => {
            Ok((StopReason::Length, None))
        }
        "incomplete" => Ok((
            StopReason::Error,
            Some(incomplete_reason.map_or_else(
                || "Response incomplete without a provider reason".to_owned(),
                |reason| format!("Response incomplete: {reason}"),
            )),
        )),
        "failed" | "cancelled" => Ok((StopReason::Error, None)),
        // These two are wonky ...
        "in_progress" | "queued" => Ok((StopReason::Stop, None)),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}
