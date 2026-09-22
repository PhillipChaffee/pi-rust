//! Simple-request option shaping shared by the wire APIs, ported from
//! `packages/ai/src/api/simple-options.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::types::{
    Context, Model, SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel,
};
use crate::utils::estimate::estimate_context_tokens;

/// Tokens always subtracted from the context window when fitting
/// `maxTokens`, upstream's `CONTEXT_SAFETY_TOKENS`.
pub const CONTEXT_SAFETY_TOKENS: u64 = 4096;
/// The floor `maxTokens` clamps to, upstream's `MIN_MAX_TOKENS`.
pub const MIN_MAX_TOKENS: u64 = 1;
/// Tokens always left for the answer when a thinking budget shares the
/// response ceiling, upstream's `MIN_ANSWER_TOKENS`.
pub const MIN_ANSWER_TOKENS: u64 = 1024;

/// The default per-level thinking budgets, upstream's
/// `DEFAULT_THINKING_BUDGETS`.
#[must_use]
pub const fn default_thinking_budgets() -> ThinkingBudgets {
    ThinkingBudgets {
        minimal: Some(1024),
        low: Some(2048),
        medium: Some(8192),
        high: Some(16384),
    }
}

/// Cap `maxTokens` to the room the context window leaves after the
/// estimated prompt and the safety margin, upstream's
/// `clampMaxTokensToContext`. A model without a context window skips the fit.
#[must_use]
pub fn clamp_max_tokens_to_context(model: &Model, context: &Context, max_tokens: u64) -> u64 {
    if model.context_window == 0 {
        return MIN_MAX_TOKENS.max(max_tokens);
    }
    let used = estimate_context_tokens(context)
        .tokens
        .saturating_add(CONTEXT_SAFETY_TOKENS);
    let available = model.context_window.saturating_sub(used);
    max_tokens.min(MIN_MAX_TOKENS.max(available))
}

/// Assemble the base [`StreamOptions`] a simple request maps to, upstream's
/// `buildBaseOptions`. Model sampling params merge under the request's.
#[must_use]
pub fn build_base_options(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
    api_key: Option<&str>,
) -> StreamOptions {
    let Some(options) = options else {
        return StreamOptions {
            max_tokens: Some(clamp_max_tokens_to_context(
                model,
                context,
                model.max_tokens,
            )),
            ..StreamOptions::default()
        };
    };
    let sampling_params = match (&model.sampling_params, &options.sampling_params) {
        (None, None) => None,
        (model_params, request_params) => {
            let mut merged = model_params.clone().unwrap_or_default();
            if let Some(request_params) = request_params {
                for (key, value) in request_params {
                    merged.insert(key.clone(), value.clone());
                }
            }
            Some(merged)
        }
    };
    StreamOptions {
        transport_options: options.transport_options.clone(),
        api_key: api_key.map_or_else(
            || options.api_key.clone(),
            |api_key| Some(api_key.to_owned()),
        ),
        telemetry_context: options.telemetry_context.clone(),
        env: options.env.clone(),
        headers: options.headers.clone(),
        timeout_ms: options.timeout_ms,
        max_retries: options.max_retries,
        max_retry_delay_ms: options.max_retry_delay_ms,
        temperature: options.temperature,
        sampling_params,
        max_tokens: Some(clamp_max_tokens_to_context(
            model,
            context,
            options.max_tokens.unwrap_or(model.max_tokens),
        )),
        transport: options.transport,
        cache_retention: options.cache_retention,
        session_id: options.session_id.clone(),
        websocket_connect_timeout_ms: options.websocket_connect_timeout_ms,
        metadata: options.metadata.clone(),
    }
}

/// The reasoning level a budget-based provider acts on, upstream's
/// `clampReasoning`: `xhigh` and `max` clamp to `high`.
#[must_use]
pub const fn clamp_reasoning(effort: Option<ThinkingLevel>) -> Option<ThinkingLevel> {
    match effort {
        Some(ThinkingLevel::Xhigh | ThinkingLevel::Max) => Some(ThinkingLevel::High),
        other => other,
    }
}

/// The thinking budget a level spends, upstream's
/// `thinkingBudgetForLevel`: the defaults overridden by `custom_budgets`.
///
/// # Panics
/// Never in practice: `clamp_reasoning` of a present level is always one of
/// the four budgeted levels, and the defaults fill all four.
#[must_use]
pub fn thinking_budget_for_level(
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> u64 {
    let defaults = default_thinking_budgets();
    let budgets = custom_budgets.map_or(defaults, |custom| ThinkingBudgets {
        minimal: custom.minimal.or(defaults.minimal),
        low: custom.low.or(defaults.low),
        medium: custom.medium.or(defaults.medium),
        high: custom.high.or(defaults.high),
    });
    let level = clamp_reasoning(Some(reasoning_level)).unwrap_or(ThinkingLevel::High);
    match level {
        ThinkingLevel::Minimal => budgets.minimal,
        ThinkingLevel::Low => budgets.low,
        ThinkingLevel::Medium => budgets.medium,
        ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => budgets.high,
    }
    .unwrap_or(0)
}

/// Cap a thinking budget so at least [`MIN_ANSWER_TOKENS`] remain under a
/// shared response ceiling, upstream's `clampThinkingBudgetToAnswerRoom`.
#[must_use]
pub fn clamp_thinking_budget_to_answer_room(thinking_budget: u64, ceiling: u64) -> u64 {
    thinking_budget.min(ceiling.saturating_sub(MIN_ANSWER_TOKENS))
}

/// Fit a thinking budget under the response cap, upstream's
/// `adjustMaxTokensForThinking`. `None` `base_max_tokens` means no explicit
/// caller cap: use the model cap and fit thinking inside it.
#[must_use]
pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: Option<u64>,
    model_max_tokens: u64,
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> (u64, u64) {
    let mut thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets);
    let max_tokens = base_max_tokens.map_or(model_max_tokens, |base| {
        base.saturating_add(thinking_budget).min(model_max_tokens)
    });
    if max_tokens <= thinking_budget {
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens);
    }
    (max_tokens, thinking_budget)
}
