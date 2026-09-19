//! Context-overflow detection, ported from
//! `packages/ai/src/utils/overflow.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! These patterns match the error messages providers return when the input
//! exceeds the model's context window, with per-provider coverage recorded on
//! each pattern: Anthropic, OpenAI and its compatible proxies, Google, xAI,
//! Groq, OpenRouter, Together AI, llama.cpp, LM Studio, GitHub Copilot,
//! MiniMax, Kimi For Coding, DS4, Cerebras, Mistral, DashScope/Qwen, Ollama,
//! and the silent overflows of z.ai (usage exceeds the window) and Xiaomi MiMo
//! (a length stop with zero output after the input filled the window).

use std::sync::LazyLock;

use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

/// The overflow patterns, in upstream order. Each entry names the provider it
/// covers in the source list below; see the upstream module doc for the
/// matching example messages.
const OVERFLOW_PATTERN_SOURCES: &[&str] = &[
    r"(?i)prompt is too long",                    // Anthropic token overflow
    r"(?i)request_too_large",                     // Anthropic request byte-size overflow (HTTP 413)
    r"(?i)input is too long for requested model", // Amazon Bedrock
    r"(?i)exceeds the context window",            // OpenAI (Completions & Responses API)
    r"(?i)exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))", // OpenAI-compatible proxies (LiteLLM)
    r"(?i)input token count.*exceeds the maximum", // Google (Gemini)
    r"(?i)maximum prompt length is \d+",           // xAI (Grok)
    r"(?i)reduce the length of the messages",      // Groq
    r"(?i)maximum context length is \d+ tokens",   // OpenRouter (most backends)
    r"(?i)exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", // OpenRouter/Poolside
    r"(?i)input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)", // Together AI
    r"(?i)exceeds the limit of \d+",           // GitHub Copilot
    r"(?i)exceeds the available context size", // llama.cpp server
    r"(?i)greater than the context length",    // LM Studio
    r"(?i)context window exceeds limit",       // MiniMax
    r"(?i)exceeded model token limit",         // Kimi For Coding
    r"(?i)too large for model with \d+ maximum context length", // Mistral
    r"(?i)prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?", // DS4 server
    r"(?i)model_context_window_exceeded", // z.ai non-standard finish_reason surfaced as error text
    r"(?i)prompt too long; exceeded (?:max )?context length", // Ollama explicit overflow error
    r"(?i)range of input length should be", // DashScope / Qwen Token Plan
    r"(?i)context[_ ]length[_ ]exceeded", // Generic fallback
    r"(?i)too many tokens",               // Generic fallback
    r"(?i)token limit exceeded",          // Generic fallback
    r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)", // Cerebras: 400/413 with no body
];

/// Patterns that indicate non-overflow errors (rate limiting, throttling,
/// server errors); a message matching any of these is never overflow, even
/// when it also matches an overflow pattern. Bedrock formats throttling as
/// "`ThrottlingException`: Too many tokens, please wait before trying again.",
/// which would otherwise match the generic token pattern above.
const NON_OVERFLOW_PATTERN_SOURCES: &[&str] = &[
    r"(?i)^(Throttling error|Service unavailable):", // AWS Bedrock non-overflow errors (human-readable prefixes from formatBedrockError)
    r"(?i)rate limit",                               // Generic rate limiting
    r"(?i)too many requests",                        // Generic HTTP 429 style
];

static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    OVERFLOW_PATTERN_SOURCES
        .iter()
        .filter_map(|source| Regex::new(source).ok())
        .collect()
});

static NON_OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    NON_OVERFLOW_PATTERN_SOURCES
        .iter()
        .filter_map(|source| Regex::new(source).ok())
        .collect()
});

/// The overflow patterns, for testing purposes: one source string per
/// pattern, in match order.
#[must_use]
pub const fn overflow_pattern_sources() -> &'static [&'static str] {
    OVERFLOW_PATTERN_SOURCES
}

/// Check if an assistant message represents a context overflow error.
///
/// Three cases, in upstream order:
///
/// 1. Error-based overflow: most providers return `stopReason: "error"` with
///    a specific error-message pattern.
/// 2. Silent overflow (z.ai style): the provider accepts the request and
///    returns successfully; detected when usage input exceeds the context
///    window.
/// 3. Length-stop overflow (Xiaomi MiMo style): the server truncates the
///    input to fit the context window and returns `stopReason: "length"` with
///    zero output.
///
/// Detection is reliable for providers that return detectable error
/// messages; silent truncation (z.ai sometimes, Ollama deployments) cannot be
/// detected here without the context window or cannot be detected at all.
/// Custom providers added through settings may need their own pattern —
/// send an oversized request, read the error message, and extend
/// the pattern source list.
#[must_use]
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    // Case 1: check error message patterns, skipping known non-overflow
    // messages (throttling, rate limits).
    if message.stop_reason == StopReason::Error
        && let Some(error_message) = &message.error_message
    {
        let is_non_overflow = NON_OVERFLOW_PATTERNS
            .iter()
            .any(|p| p.is_match(error_message));
        if !is_non_overflow && OVERFLOW_PATTERNS.iter().any(|p| p.is_match(error_message)) {
            return true;
        }
    }

    // A zero window is no window, upstream's falsy check.
    let context_window = context_window.filter(|window| *window > 0);

    // Case 2: silent overflow — the request succeeded but the input exceeds
    // the context.
    if let Some(window) = context_window {
        if message.stop_reason == StopReason::Stop {
            let input_tokens = message.usage.input + message.usage.cache_read;
            if input_tokens > window {
                return true;
            }
        }

        // Case 3: length-stop overflow — the input filled the window and no
        // output was generated.
        if message.stop_reason == StopReason::Length && message.usage.output == 0 {
            let input_tokens = message.usage.input + message.usage.cache_read;
            #[expect(
                clippy::cast_precision_loss,
                reason = "the 0.99-window comparison reproduces the JS number math that decides a silent overflow"
            )]
            if input_tokens as f64 >= window as f64 * 0.99 {
                return true;
            }
        }
    }

    false
}

/// Check whether a length stop ended below the caller or model's intended output limit.
///
/// Such responses may be caused by context pressure or provider-side truncation, so callers can
/// make one bounded compact-and-retry attempt. `desired_max_output` must be the original limit
/// before any context-based clamping.
#[must_use]
pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u64) -> bool {
    message.stop_reason == StopReason::Length
        && desired_max_output > 0
        && message.usage.output < desired_max_output
}
