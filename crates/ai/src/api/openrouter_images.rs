//! The OpenRouter image-generation wire API, ported from
//! `packages/ai/src/api/openrouter-images.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//!
//! - The `openai` SDK client collapses onto the [`crate::http::HttpClient`]
//!   seam. The module assembles the wire request the pinned SDK sent — POST
//!   `{baseUrl}/chat/completions`, `Authorization: Bearer {apiKey}`, and the
//!   model's and caller's headers merged as the SDK's `defaultHeaders` — and
//!   executes it. [`ImagesOptions`] carries no client seam, so the request
//!   runs on the process default client.
//! - Upstream's `options.fetch`, `options.signal`, `options.onPayload`, and
//!   `options.onResponse` reads have no [`ImagesOptions`] field yet, so the
//!   hooks never fire and the caller's abort signal cannot cancel the
//!   request. The catch block's `options?.signal?.aborted` check maps onto
//!   the seam's abort-shaped [`ProviderRequestError::is_abort`], the only way
//!   an abort failure reaches the catch block through
//!   [`retry_provider_request`].
//! - The SDK call's `maxRetries: 0` and the surrounding `retryProviderRequest`
//!   collapse onto [`retry_provider_request`] with the caller's `maxRetries`
//!   and `maxRetryDelayMs`. A non-2xx response folds its body into the
//!   SDK-shaped error message the way `APIError.makeMessage` did, and the
//!   parsed body is probed by the [`crate::utils::error_body`] normalizer,
//!   upstream's catch block reading `error.status` and `error.error`.
//! - `sanitizeSurrogates` disappears statically: a Rust [`String`] cannot
//!   hold the unpaired surrogates it removed, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire.
//! - Upstream's `generateImages` never rejects: the catch block resolves with
//!   an error- or abort-shaped [`AssistantImages`], so this port always
//!   returns `Ok` and the trait's `Err` arm stays unused here.

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use serde_json::{Value, json};

use crate::api::request_seam::{
    execute_checked_response, provider_error_from_http, read_body_text,
};
use crate::auth::resolve::now_ms;
use crate::http::client::{HttpMethod, HttpRequest, HttpResponse};
use crate::types::{
    AssistantImages, ImageContent, ImagesBlock, ImagesContext, ImagesModel, ImagesOptions,
    ImagesStopReason, Modality, ProviderHeaders, TextContent, Usage, UsageCost,
};
use crate::utils::abort::operation_signal;
use crate::utils::error_body::{
    ErrorBody, SdkError, format_provider_error, normalize_provider_error,
};
use crate::utils::provider_retry::{
    ProviderRequestError, ProviderRetryOptions, retry_provider_request,
};

/// Generate images over the OpenRouter image-generation wire, upstream's
/// `generateImages`: one chat-completions request whose user message carries
/// the input blocks, resolved into text and `data:`-URL image blocks. Every
/// failure resolves to the error- or abort-shaped output like the catch block
/// it ports.
async fn generate_images(
    model: &ImagesModel,
    context: &ImagesContext,
    options: Option<&ImagesOptions>,
) -> AssistantImages {
    let mut output = initial_output(model);
    let parsed_error_body = Mutex::new(None);
    if let Err(error) = run_request(model, context, options, &mut output, &parsed_error_body).await
    {
        // The catch block: the abort-shaped seam failure stops as `aborted`,
        // everything else as `error`.
        let parsed_body = parsed_error_body
            .lock()
            .ok()
            .and_then(|mut slot| std::mem::take(&mut *slot));
        let sdk_error = SdkError {
            message: error.message.clone(),
            status: error.status,
            error: parsed_body.map(ErrorBody::Parsed),
            ..SdkError::default()
        };
        output.stop_reason = if error.is_abort() {
            ImagesStopReason::Aborted
        } else {
            ImagesStopReason::Error
        };
        output.error_message = Some(format_provider_error(
            &normalize_provider_error(sdk_error),
            None,
        ));
    }
    output
}

/// The fresh result a generation starts from, upstream's `output`: no blocks,
/// no usage, `stopReason: "stop"`.
fn initial_output(model: &ImagesModel) -> AssistantImages {
    AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: Vec::new(),
        response_id: None,
        usage: None,
        stop_reason: ImagesStopReason::Stop,
        error_message: None,
        timestamp: now_ms(),
    }
}

/// Build and dispatch the generation request and settle the output blocks,
/// upstream's try block.
///
/// # Errors
/// The missing-key rejection, the retried request's failure after its retry
/// budget is spent, and a response body that does not parse.
async fn run_request(
    model: &ImagesModel,
    context: &ImagesContext,
    options: Option<&ImagesOptions>,
    output: &mut AssistantImages,
    parsed_error_body: &Mutex<Option<Value>>,
) -> Result<(), ProviderRequestError> {
    let Some(api_key) = options.and_then(|options| options.api_key.as_deref()) else {
        let provider: &str = &model.provider;
        return Err(ProviderRequestError::new(
            None,
            None,
            format!("No API key for provider: {provider}"),
        ));
    };
    let payload = build_params(model, context);
    let response =
        dispatch_generate_request(model, options, api_key, payload, parsed_error_body).await?;
    let body_text = read_body_text(response.body)
        .await
        .map_err(provider_error_from_http)?;
    let image_response: Value = serde_json::from_str(&body_text)
        .map_err(|error| ProviderRequestError::new(None, None, error.to_string()))?;

    output.response_id = image_response
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(usage) = image_response.get("usage").filter(|usage| is_truthy(usage)) {
        output.usage = Some(parse_usage(usage, model));
    }

    let Some(choice) = image_response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .filter(|choice| is_truthy(choice))
    else {
        return Ok(());
    };
    let message = choice.get("message");
    if let Some(text) = message
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        output.output.push(ImagesBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        }));
    }
    for image in message
        .and_then(|message| message.get("images"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some((mime_type, data)) = image
            .get("image_url")
            .and_then(parse_generated_image_url)
            .and_then(parse_data_url)
        else {
            continue;
        };
        output.output.push(ImagesBlock::Image(ImageContent {
            data: data.to_owned(),
            mime_type: mime_type.to_owned(),
        }));
    }
    Ok(())
}

/// Merge the model's and caller's headers, upstream's
/// `providerHeadersToRecord({ ...model.headers, ...optionsHeaders })` feeding
/// the SDK's `defaultHeaders`: the caller's value overrides by name
/// case-insensitively and a `None` value suppresses the header.
fn build_request_headers(
    model: &ImagesModel,
    options_headers: Option<&ProviderHeaders>,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = Vec::new();
    if let Some(model_headers) = &model.headers {
        for (name, value) in model_headers {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
            headers.push((name.clone(), value.clone()));
        }
    }
    if let Some(options_headers) = options_headers {
        for (name, value) in options_headers {
            match value {
                Some(value) => {
                    headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
                    headers.push((name.clone(), value.clone()));
                }
                None => headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name)),
            }
        }
    }
    headers
}

/// The chat-completions endpoint, the path the pinned SDK appends to
/// `baseUrl`.
fn chat_completions_url(model: &ImagesModel) -> String {
    format!("{}/chat/completions", model.base_url.trim_end_matches('/'))
}

/// Build the request payload, upstream's `buildParams`: one user message
/// whose content parts map the input blocks, `stream: false`, and the wire's
/// `modalities` — `["image", "text"]` when the model outputs text, `["image"]`
/// otherwise.
fn build_params(model: &ImagesModel, context: &ImagesContext) -> Value {
    let content_parts: Vec<Value> = context
        .input
        .iter()
        .map(|block| match block {
            ImagesBlock::Text(text) => json!({ "type": "text", "text": text.text }),
            ImagesBlock::Image(image) => json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
            }),
        })
        .collect();
    let modalities: Vec<Value> = if model.output.contains(&Modality::Text) {
        vec![json!("image"), json!("text")]
    } else {
        vec![json!("image")]
    };
    json!({
        "model": model.id,
        "messages": [{ "role": "user", "content": content_parts }],
        "stream": false,
        "modalities": modalities,
    })
}

/// The URL one generated image's `image_url` entry carries, upstream's
/// `typeof image.image_url === "string" ? image.image_url : image.image_url?.url`.
fn parse_generated_image_url(image_url: &Value) -> Option<&str> {
    match image_url {
        Value::String(url) => Some(url),
        Value::Object(url) => url.get("url").and_then(Value::as_str),
        _ => None,
    }
}

/// The data-URL parse of upstream's
/// `imageUrl.match(/^data:([^;]+);base64,(.+)$/)` into the MIME type and the
/// base64 payload. Anchored like the regex: the MIME half admits no
/// semicolon like `[^;]+`, and the data half excludes the line terminators
/// the regex's `.` excludes (`\n`, `\r`, U+2028, U+2029).
fn parse_data_url(image_url: &str) -> Option<(&str, &str)> {
    let rest = image_url.strip_prefix("data:")?;
    let separator = rest.find(";base64,")?;
    let mime_type = &rest[..separator];
    if mime_type.is_empty() || mime_type.contains(';') {
        return None;
    }
    let data = &rest[separator + ";base64,".len()..];
    if data.is_empty()
        || data
            .chars()
            .any(|character| matches!(character, '\n' | '\r' | '\u{2028}' | '\u{2029}'))
    {
        return None;
    }
    Some((mime_type, data))
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

/// The wire usage fields an image response reports, priced into the result's
/// [`Usage`], upstream's `parseUsage`. `prompt_tokens_details.cached_tokens`
/// is the cache read, discounted by `cache_write_tokens` when the wire
/// carries one; `prompt_tokens` and `completion_tokens` are the raw counts.
/// The base rates price the run — image models carry no tiers and no 1h
/// split, so [`crate::models::calculate_cost`] does not apply. The wire's
/// `|| 0` defaults become zero reads of the JSON fields.
#[expect(
    clippy::cast_precision_loss,
    reason = "token counts are far below the f64 exact-integer range"
)]
fn parse_usage(raw_usage: &Value, model: &ImagesModel) -> Usage {
    let prompt_tokens = raw_usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let prompt_tokens_details = raw_usage.get("prompt_tokens_details");
    let reported_cached_tokens = prompt_tokens_details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write_tokens = prompt_tokens_details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_tokens = if cache_write_tokens > 0 {
        reported_cached_tokens.saturating_sub(cache_write_tokens)
    } else {
        reported_cached_tokens
    };
    let input = prompt_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    let output_tokens = raw_usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let rates = model.cost.rates;
    let mut cost = UsageCost {
        input: rates.input / 1_000_000.0 * input as f64,
        output: rates.output / 1_000_000.0 * output_tokens as f64,
        cache_read: rates.cache_read / 1_000_000.0 * cache_read_tokens as f64,
        cache_write: rates.cache_write / 1_000_000.0 * cache_write_tokens as f64,
        total: 0.0,
    };
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
    Usage {
        input,
        output: output_tokens,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input
            .saturating_add(output_tokens)
            .saturating_add(cache_read_tokens)
            .saturating_add(cache_write_tokens),
        cost,
    }
}

/// Execute the generation request with retries, the seam-side port of
/// upstream's `createClient` + `retryProviderRequest(create(...).withResponse())`.
/// A non-2xx response folds its body into the SDK-shaped error message and
/// keeps the parsed form for the catch block's body probe.
///
/// # Errors
/// The request's failure once the retry budget is spent, and the abort-shaped
/// error when the request is cancelled mid-flight.
async fn dispatch_generate_request(
    model: &ImagesModel,
    options: Option<&ImagesOptions>,
    api_key: &str,
    payload: Value,
    parsed_error_body: &Mutex<Option<Value>>,
) -> Result<HttpResponse, ProviderRequestError> {
    let mut headers = vec![("Authorization".to_owned(), format!("Bearer {api_key}"))];
    headers.extend(build_request_headers(
        model,
        options.and_then(|options| options.headers.as_ref()),
    ));

    let http_client = crate::http::default_http_client();
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: chat_completions_url(model),
        headers,
        body: Some(Bytes::from(payload.to_string())),
        timeout_ms: options.and_then(|options| options.timeout_ms),
        signal: operation_signal(None),
    };
    let retry_options = ProviderRetryOptions {
        max_retries: options.and_then(|options| options.max_retries).unwrap_or(0),
        max_retry_delay_ms: options.and_then(|options| options.max_retry_delay_ms),
        signal: None,
        random: None,
    };
    retry_provider_request(
        || {
            let client = Arc::clone(&http_client);
            let request = request.clone();
            async move { execute_checked_response(client, request, Some(parsed_error_body)).await }
        },
        &retry_options,
    )
    .await
}

/// The OpenRouter image-generation [`crate::types::ProviderImages`], upstream's
/// module-level `generateImages` export behind the registered
/// `openrouter-images` images provider.
#[derive(Debug, Default)]
pub struct OpenRouterImages;

impl crate::types::ProviderImages for OpenRouterImages {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> crate::types::BoxedFuture<'a, Result<AssistantImages, ProviderRequestError>> {
        Box::pin(async move { Ok(generate_images(model, context, options).await) })
    }
}
