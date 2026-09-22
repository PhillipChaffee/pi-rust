//! The Google Generative AI (Gemini) wire API, ported from
//! `packages/ai/src/api/google-generative-ai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//! - The `@google/genai` SDK surface collapses to one raw request: upstream's
//!   only SDK call is `client.models.generateContentStream(params)`, which is
//!   `POST {baseUrl}/models/{model}:streamGenerateContent?alt=sse` with
//!   `x-goog-api-key` auth and a JSON body of the SDK-style params flattened
//!   (`config` becomes top-level `generationConfig`; `systemInstruction`,
//!   `tools`, and `toolConfig` hoist to the body root). `onPayload` sees the
//!   SDK-style params, the hook's replacement included.
//! - The SDK's default headers drop with it: `x-goog-api-client` (SDK
//!   telemetry) is not reproduced, and `User-Agent` is pi's own value, which
//!   upstream overrides onto the SDK default anyway. `x-goog-api-key` keeps
//!   the SDK's only-when-absent rule, so a caller header can override it.
//! - The SDK's env fallbacks (`GOOGLE_API_KEY`/`GEMINI_API_KEY`,
//!   `GOOGLE_GENAI_USE_VERTEXAI`) never fire upstream because pi always passes
//!   an explicit key; the port requires the key outright.
//! - Upstream's `options.fetch` guard ("Custom fetch is not supported") has
//!   no counterpart: the transport seam is the supported injection point.
//! - Surrogate sanitization is statically upheld: Rust strings are valid
//!   UTF-8, so `sanitizeSurrogates` has no work.
//! - The streamed part index scratch field and its catch-path cleanup
//!   vanish; Rust owns blocks in the accumulator directly.
//! - The SDK's pre-stream error probe inspected the first decoded network
//!   chunk whole; the port probes the first SSE `data:` event, the same
//!   failure surface on every frame shape the API emits.
//! - A `usageMetadata` reporting more cached tokens than prompt tokens
//!   floors `usage.input` at zero (upstream's arithmetic can go negative;
//!   `Usage.input` is unsigned here).

use serde_json::{Value, json};

/// The adapter-facing options, upstream's `GoogleOptions`: the shared Google
/// options shape under the adapter's historical name.
pub use crate::api::google_shared::GoogleOptions as GoogleStreamOptions;
use crate::api::google_shared::{
    GoogleThinkingControl, ResolvedGoogleThinkingLevel, base_google_request_headers,
    build_google_params, consume_google_stream, dispatch_google_stream, finish_google_stream,
    is_gemini3_flash_model, is_gemini3_pro_model, resolve_google_thinking_level,
};
use crate::api::simple_options::build_base_options;
use crate::api::wire_common::{
    initial_output, missing_api_key_message, push_header_if_absent, setup_error_stream,
    spawn_adapter_stream,
};
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, Model, ModelThinkingLevel,
    SimpleStreamOptions, ThinkingBudgets,
};
use crate::utils::event_stream::AssistantMessageEventStream;

/// The model id the wire names when no custom base URL is set: the SDK's
/// default host plus its default `v1beta` version segment.
const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// The Google Generative AI streams, upstream's `googleGenerativeAIApi()`.
#[derive(Debug, Default)]
pub struct GoogleStreams;

crate::api::wire_common::forward_provider_streams!(GoogleStreams, GoogleStreamOptions);

/// Stream an assistant response, upstream's `stream` export.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&GoogleStreamOptions>,
) -> AssistantMessageEventStream {
    spawn_adapter_stream(
        model,
        context,
        options.cloned(),
        |model, _| initial_output(model),
        |model, context, options, output, events| {
            Box::pin(run_stream(model, context, options, output, events))
        },
    )
}

/// Stream a simple assistant response, upstream's `streamSimple` export.
///
/// The pi reasoning level resolves to a provider-native thinking level or a
/// token budget; no level disables thinking outright.
///
/// Provider-native levels ride Gemini 3 / Gemma 4; token budgets ride
/// Gemini 2.x.
///
/// Upstream throws synchronously when the key is missing; the port encodes
/// that failure as a settled error stream per the stream contract.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let Some(api_key) = options
        .as_ref()
        .and_then(|options| options.api_key.as_deref())
    else {
        return setup_error_stream(model, &missing_api_key_message(&model.provider));
    };

    // The base options carry sampling_params upstream-side, but the Google
    // options shape has no samplingParams field: the merge is dropped.
    let base = build_base_options(model, context, options, Some(api_key));
    let tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            crate::types::ToolChoice::Auto => "auto",
            crate::types::ToolChoice::None => "none",
        });
    let mut base_options = GoogleStreamOptions::from(base);
    base_options.tool_choice = tool_choice.map(str::to_owned);

    let Some(reasoning) = options.and_then(|options| options.reasoning) else {
        return stream(
            model,
            context,
            Some(&GoogleStreamOptions {
                thinking: Some(GoogleThinkingControl {
                    enabled: false,
                    ..GoogleThinkingControl::default()
                }),
                ..base_options
            }),
        );
    };

    let clamped = crate::models::clamp_thinking_level(model, ModelThinkingLevel::from(reasoning));
    let resolved = match resolve_google_thinking_level(model, clamped) {
        Ok(level) => level,
        Err(message) => return setup_error_stream(model, &message),
    };

    if is_gemini3_pro_model(model) || is_gemini3_flash_model(model) || is_gemma4_model(model) {
        return stream(
            model,
            context,
            Some(&GoogleStreamOptions {
                thinking: Some(GoogleThinkingControl {
                    enabled: true,
                    level: Some(get_thinking_level(resolved, model).to_owned()),
                    budget_tokens: None,
                }),
                ..base_options
            }),
        );
    }

    stream(
        model,
        context,
        Some(&GoogleStreamOptions {
            thinking: Some(GoogleThinkingControl {
                enabled: true,
                budget_tokens: Some(get_google_budget(
                    model,
                    resolved,
                    options.and_then(|options| options.thinking_budgets.as_ref()),
                )),
                level: None,
            }),
            ..base_options
        }),
    )
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &GoogleStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    if options.api_key.as_deref().unwrap_or_default().is_empty() {
        return Err(missing_api_key_message(&model.provider));
    }

    let mut params = build_params(model, context, options)?;
    if let Some(hook) = &options.transport_options.on_payload {
        params = hook
            .call(params.clone(), model.clone())
            .await
            .unwrap_or(params);
    }

    let response = dispatch_google_stream(
        model,
        options,
        stream_generate_url(model),
        build_request_headers(model, options),
        &params,
    )
    .await?;
    events.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    consume_google_stream(model, response, output, events).await?;
    finish_google_stream(
        &options.transport_options,
        output,
        events,
        "Google stream ended without a finish reason",
    )
}

/// The request URL, the SDK's URL construction: a custom `model.baseUrl` is
/// used verbatim (it already includes the version path; the SDK's empty
/// `apiVersion` skips the segment), otherwise the default host plus
/// `v1beta`.
fn stream_generate_url(model: &Model) -> String {
    let base = if model.base_url.trim().is_empty() {
        DEFAULT_GEMINI_BASE_URL.to_owned()
    } else {
        model.base_url.trim_end_matches('/').to_owned()
    };
    format!("{base}/models/{}:streamGenerateContent?alt=sse", model.id)
}

/// Assemble the request headers, upstream's `createClient` header merge: the
/// shared Google defaults, with the credential header appended last and only
/// when the caller did not already set it.
fn build_request_headers(model: &Model, options: &GoogleStreamOptions) -> Vec<(String, String)> {
    let mut headers = base_google_request_headers(model, options.headers.as_ref());
    push_header_if_absent(
        &mut headers,
        "x-goog-api-key",
        options.api_key.clone().unwrap_or_default(),
    );
    headers
}

/// Build the SDK-style params, upstream's `buildParams`: `{ model, contents,
/// config }` with the camelCase config fields. The wire flattening happens in
/// [`to_wire_body`]; `onPayload` sees this shape.
///
/// # Errors
/// When a tool requires strict sampling that cannot be resolved, or when the
/// request is already aborted.
fn build_params(
    model: &Model,
    context: &Context,
    options: &GoogleStreamOptions,
) -> Result<Value, String> {
    build_google_params(model, context, options, get_disabled_thinking_config)
}

/// The disabled-thinking config, upstream's `getDisabledThinkingConfig`.
///
/// Google docs: Gemini 3.1 Pro cannot disable thinking, and Gemini 3 Flash /
/// Flash-Lite do not support full thinking-off either. For Gemini 3 models,
/// the lowest supported `thinkingLevel` rides without `includeThoughts` so
/// hidden thinking stays invisible to pi. Gemini 2.x disables via
/// `thinkingBudget: 0`.
fn get_disabled_thinking_config(model: &Model) -> Value {
    if is_gemini3_pro_model(model) {
        json!({ "thinkingLevel": "LOW" })
    } else if is_gemini3_flash_model(model) || is_gemma4_model(model) {
        json!({ "thinkingLevel": "MINIMAL" })
    } else {
        json!({ "thinkingBudget": 0 })
    }
}

/// `/gemma-?4/` over the lowercased id.
fn is_gemma4_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    id.contains("gemma-4") || id.contains("gemma4")
}

/// Map a resolved pi level to the provider-native thinking level,
/// upstream's `getThinkingLevel`.
#[must_use]
pub fn get_thinking_level(effort: ResolvedGoogleThinkingLevel, model: &Model) -> &'static str {
    if is_gemini3_pro_model(model) {
        return match effort {
            ResolvedGoogleThinkingLevel::Minimal | ResolvedGoogleThinkingLevel::Low => "LOW",
            ResolvedGoogleThinkingLevel::Medium | ResolvedGoogleThinkingLevel::High => "HIGH",
        };
    }
    if is_gemma4_model(model) {
        return match effort {
            ResolvedGoogleThinkingLevel::Minimal | ResolvedGoogleThinkingLevel::Low => "MINIMAL",
            ResolvedGoogleThinkingLevel::Medium | ResolvedGoogleThinkingLevel::High => "HIGH",
        };
    }
    match effort {
        ResolvedGoogleThinkingLevel::Minimal => "MINIMAL",
        ResolvedGoogleThinkingLevel::Low => "LOW",
        ResolvedGoogleThinkingLevel::Medium => "MEDIUM",
        ResolvedGoogleThinkingLevel::High => "HIGH",
    }
}

/// The catalog's default thinking budgets, upstream's `getGoogleBudget`;
/// `-1` is the dynamic budget where the catalog has no entry.
#[must_use]
pub fn get_google_budget(
    model: &Model,
    level: ResolvedGoogleThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> i64 {
    if let Some(budgets) = custom_budgets {
        let custom = match level {
            ResolvedGoogleThinkingLevel::Minimal => budgets.minimal,
            ResolvedGoogleThinkingLevel::Low => budgets.low,
            ResolvedGoogleThinkingLevel::Medium => budgets.medium,
            ResolvedGoogleThinkingLevel::High => budgets.high,
        };
        if let Some(budget) = custom {
            return i64::try_from(budget).unwrap_or(-1);
        }
    }

    let (minimal, low, medium, high) = if model.id.contains("2.5-pro") {
        (128, 2048, 8192, 32768)
    } else if model.id.contains("2.5-flash-lite") {
        (512, 2048, 8192, 24576)
    } else if model.id.contains("2.5-flash") {
        (128, 2048, 8192, 24576)
    } else {
        return -1;
    };
    i64::from(match level {
        ResolvedGoogleThinkingLevel::Minimal => minimal,
        ResolvedGoogleThinkingLevel::Low => low,
        ResolvedGoogleThinkingLevel::Medium => medium,
        ResolvedGoogleThinkingLevel::High => high,
    })
}
