//! Env-gated context-overflow probes across providers, ported from the
//! upstream `packages/ai/test/context-overflow.test.ts` suite at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! One `#[tokio::test]` per upstream `it` block, gated by its describe's
//! guard the way upstream's `it.skipIf` gates: a probe without its provider
//! credential — env key or resolved OAuth token — or without its local
//! (`tests/common/live.rs`) carries the `generateOverflowContent`,
//! `testContextOverflow`, and `logResult` restatements; each probe sends a
//! prompt exceeding the model's context window by 10k estimated tokens and
//! pins the same contract upstream does: the provider reports the overflow,
//! and `isContextOverflow` agrees.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]
#![expect(
    clippy::print_stdout,
    reason = "the suite logs each probe outcome like upstream's logResult console output"
)]

use std::process::{Command, Stdio};
use std::time::Duration;

use pi_ai::types::{Api, AssistantMessage, Modality, Model, ModelCost, ProviderId, StopReason};
use pi_ai::utils::overflow::is_context_overflow;
use regex::Regex;

mod common;
use common::live;

/// The upstream `expect(errorMessage).toMatch(/…/i)` restatement: the
/// pattern restated with the `/i` flag inline.
fn pattern(source: &str) -> Regex {
    Regex::new(&format!("(?i){source}")).expect("the upstream pattern compiles")
}

/// The wire name of a stop reason, upstream's `result.stopReason` log field.
const fn stop_reason_name(stop_reason: StopReason) -> &'static str {
    match stop_reason {
        StopReason::Pending => "pending",
        StopReason::Stop => "stop",
        StopReason::Length => "length",
        StopReason::ToolUse => "toolUse",
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
        StopReason::Deferred => "deferred",
    }
}

/// The model ids upstream prefers on Cerebras, in order, upstream's
/// `preferredCerebrasModelIds`.
const PREFERRED_CEREBRAS_MODEL_IDS: [&str; 3] = ["gpt-oss-120b", "zai-glm-4.7", "llama3.1-8b"];

/// Upstream's `logResult`: one line per probe-outcome field, usage as the
/// wire's JSON.
fn log_result(model: &Model, response: &AssistantMessage, has_usage_data: bool) {
    let usage = serde_json::to_string(&response.usage).expect("usage serializes");
    println!(
        "\n{} / {}:\n  contextWindow: {}\n  stopReason: {}\n  errorMessage: {}\n  usage: \
         {usage}\n  hasUsageData: {has_usage_data}",
        model.provider.0,
        model.id,
        model.context_window,
        stop_reason_name(response.stop_reason),
        response.error_message.as_deref().unwrap_or("none"),
    );
}

/// The oversized probe one block sends, upstream's `testContextOverflow` plus
/// its `logResult` call right after.
async fn probe_overflow(model: &Model, api_key: &str) -> AssistantMessage {
    let (response, has_usage_data) = live::complete_overflow(model, api_key).await;
    log_result(model, &response, has_usage_data);
    response
}

/// Upstream's `expect(result.stopReason).toBe("error")`, the error text in
/// the failure output.
fn assert_error_stop(response: &AssistantMessage) {
    assert_eq!(
        response.stop_reason,
        StopReason::Error,
        "errorMessage: {:?}",
        response.error_message
    );
}

/// Upstream's `expect(result.errorMessage).toMatch(/…/i)`: no message or a
/// non-matching one fails the block.
fn assert_error_message(response: &AssistantMessage, pattern: &Regex) {
    let message = response.error_message.as_deref().unwrap_or("");
    assert!(pattern.is_match(message), "errorMessage: {message}");
}

/// Upstream's `expect(isContextOverflow(result.response, model.contextWindow))
/// .toBe(true)`, the suite's closing assertion in every block.
fn assert_overflow(response: &AssistantMessage, context_window: u64) {
    assert!(
        is_context_overflow(response, Some(context_window)),
        "isContextOverflow returned false; errorMessage: {:?}",
        response.error_message
    );
}

/// The Xiaomi MiMo probe body, shared by upstream's four Xiaomi describes:
/// the server silently truncates oversized input to fill the context window,
/// then stops with `length` and zero output — overflow by the classifier's
/// third case, not by an error message.
async fn xiaomi_length_stop_overflow(api_key: &str, provider: &str) {
    let model = live::model(provider, "mimo-v2.5-pro");
    let response = probe_overflow(&model, api_key).await;
    assert_eq!(
        response.stop_reason,
        StopReason::Length,
        "errorMessage: {:?}",
        response.error_message
    );
    assert_eq!(
        response.usage.output, 0,
        "output: {}",
        response.usage.output
    );
    assert_overflow(&response, model.context_window);
}

/// The Qwen probe body, shared by upstream's three Qwen Token Plan describes:
/// an error stop naming the input length, and the classifier's agreement.
async fn qwen_input_length_overflow(api_key: &str, provider: &str, id: &str) {
    let model = live::model(provider, id);
    let response = probe_overflow(&model, api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"input length"));
    assert_overflow(&response, model.context_window);
}

/// Whether local-server probes may run, upstream's `!process.env.
/// PI_NO_LOCAL_LLM` module gate: unset or empty enables them.
fn local_llm_enabled() -> bool {
    !std::env::var("PI_NO_LOCAL_LLM").is_ok_and(|value| !value.is_empty())
}

/// Whether the `ollama` binary is installed, upstream's `which ollama`.
fn ollama_installed() -> bool {
    Command::new("which")
        .arg("ollama")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Whether an LM Studio server answers on its default port, upstream's
/// `curl -s --max-time 1 http://localhost:1234/v1/models`.
fn lm_studio_running() -> bool {
    Command::new("curl")
        .args(["-s", "--max-time", "1", "http://localhost:1234/v1/models"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Whether a llama.cpp server runs and still exposes `/v1/completions`,
/// upstream's health probe plus the POST probe whose status must not be
/// 404, 405, or the connection-failure `000`.
fn llama_cpp_running() -> bool {
    let health = Command::new("curl")
        .args(["-s", "--max-time", "1", "http://localhost:8081/health"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !health {
        return false;
    }
    let probe = Command::new("curl")
        .args([
            "-s",
            "--max-time",
            "1",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-X",
            "POST",
            "http://localhost:8081/v1/completions",
            "-H",
            "content-type: application/json",
            "-d",
            r#"{"model":"local-model","prompt":"ping","max_tokens":1}"#,
        ])
        .output();
    let Some(probe) = probe.ok().filter(|output| output.status.success()) else {
        return false;
    };
    let probe_stdout = String::from_utf8_lossy(&probe.stdout);
    let status = probe_stdout.trim();
    status != "404" && status != "405" && status != "000"
}

/// Wait for the spawned server's TCP listener, upstream's poll loop over
/// `fetch("http://localhost:11434/api/tags")`: connect every 500ms after an
/// initial second, bail out at the 55s budget.
async fn wait_for_ollama() -> bool {
    tokio::time::sleep(Duration::from_secs(1)).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(55);
    loop {
        if tokio::net::TcpStream::connect("127.0.0.1:11434")
            .await
            .is_ok()
        {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// The spawned `ollama serve` child, killed when the probe's scope ends —
/// panic or early return alike — upstream's `afterAll` kill.
struct ServeChild(std::process::Child);

impl Drop for ServeChild {
    fn drop(&mut self) {
        // A failed kill means the server already exited (port taken by a
        // running instance); the reap below is what matters either way.
        let _killed = self.0.kill();
        let _reaped = self.0.wait();
    }
}

/// The hand-built local model a local-server probe drives, upstream's inline
/// `Model<"openai-completions">` objects over the zero-cost shape.
fn local_model(
    id: &str,
    name: &str,
    provider: &str,
    base_url: &str,
    context_window: u64,
    max_tokens: u64,
    reasoning: bool,
) -> Model {
    Model {
        id: id.to_owned(),
        name: name.to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId::from(provider),
        base_url: base_url.to_owned(),
        reasoning,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window,
        max_tokens,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Upstream `Anthropic (API Key)` > it `claude-haiku-4-5 - should detect
/// overflow via isContextOverflow`, gated on `ANTHROPIC_API_KEY`.
#[tokio::test]
async fn anthropic_api_key_claude_haiku_4_5_detects_overflow() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let model = live::model("anthropic", "claude-haiku-4-5");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"prompt is too long"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `Anthropic (OAuth)` > it `claude-sonnet-4 - should detect
/// overflow via isContextOverflow`, gated on `ANTHROPIC_OAUTH_TOKEN`.
#[tokio::test]
async fn anthropic_oauth_claude_sonnet_4_6_detects_overflow() {
    let Some(api_key) = live::env_key("ANTHROPIC_OAUTH_TOKEN") else {
        return;
    };
    let model = live::model("anthropic", "claude-sonnet-4-6");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"prompt is too long"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `GitHub Copilot (OAuth)` > it `Google model - should detect
/// overflow via isContextOverflow`, gated on the resolved Copilot token:
/// the first catalog id starting with `gemini-` probes, absent is a bug.
#[tokio::test]
async fn copilot_google_model_detects_overflow() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let model = live::models("github-copilot")
        .into_iter()
        .find(|candidate| candidate.id.starts_with("gemini-"))
        .expect("No Google models available through GitHub Copilot");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"exceeds the limit of \d+"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `GitHub Copilot (OAuth)` > it `claude-sonnet-4 - should detect
/// overflow via isContextOverflow`, gated on the resolved Copilot token.
#[tokio::test]
async fn copilot_claude_sonnet_4_6_detects_overflow() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let model = live::model("github-copilot", "claude-sonnet-4.6");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(
        &response,
        &pattern(r"exceeds the limit of \d+|input is too long"),
    );
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenAI Completions` > it `gpt-4o-mini - should detect overflow
/// via isContextOverflow`, gated on `OPENAI_API_KEY`: the catalog model is
/// retargeted at the openai-completions wire, upstream's `{ ...getModel(...),
/// api: "openai-completions" }` spread.
#[tokio::test]
async fn openai_completions_gpt_4o_mini_detects_overflow() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let model = Model {
        api: Api::from("openai-completions"),
        ..live::model("openai", "gpt-4o-mini")
    };
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"maximum context length"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenAI Responses` > it `gpt-4o - should detect overflow via
/// isContextOverflow`, gated on `OPENAI_API_KEY`.
#[tokio::test]
async fn openai_responses_gpt_4o_detects_overflow() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let model = live::model("openai", "gpt-4o");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"exceeds the context window"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `Azure OpenAI Responses` > it `gpt-4o-mini - should detect
/// overflow via isContextOverflow`, gated on the Azure credential guard, the
/// `AZURE_OPENAI_API_KEY` value as the request key.
#[tokio::test]
async fn azure_openai_responses_gpt_4o_mini_detects_overflow() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let Some(api_key) = live::env_key("AZURE_OPENAI_API_KEY") else {
        return;
    };
    let model = live::model("azure-openai-responses", "gpt-4o-mini");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"context|maximum"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `Google` > it `gemini-2.5-flash - should detect overflow via
/// isContextOverflow`, gated on `GEMINI_API_KEY`.
#[tokio::test]
async fn google_gemini_2_5_flash_detects_overflow() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let model = live::model("google", "gemini-2.5-flash");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(
        &response,
        &pattern(r"input token count.*exceeds the maximum"),
    );
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenAI Codex (OAuth)` > it `gpt-5.5 - should detect overflow via
/// isContextOverflow`, gated on the resolved Codex token: no message pattern,
/// the classifier's verdict alone.
#[tokio::test]
async fn openai_codex_gpt_5_5_detects_overflow() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let model = live::model("openai-codex", "gpt-5.5");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `Amazon Bedrock` > it `claude-sonnet-4-5 - should detect overflow
/// via isContextOverflow`, gated on the Bedrock credential guard, the empty
/// key the SigV4 signer replaces.
#[tokio::test]
async fn amazon_bedrock_claude_sonnet_4_5_detects_overflow() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let model = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let response = probe_overflow(&model, "").await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `xAI` > it `grok-4.3 - should detect overflow via
/// isContextOverflow`, gated on `XAI_API_KEY`.
#[tokio::test]
async fn xai_grok_4_3_detects_overflow() {
    let Some(api_key) = live::env_key("XAI_API_KEY") else {
        return;
    };
    let model = live::model("xai", "grok-4.3");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"maximum prompt length is \d+"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `Groq` > it `llama-3.3-70b-versatile - should detect overflow via
/// isContextOverflow`, gated on `GROQ_API_KEY`.
#[tokio::test]
async fn groq_llama_3_3_70b_versatile_detects_overflow() {
    let Some(api_key) = live::env_key("GROQ_API_KEY") else {
        return;
    };
    let model = live::model("groq", "llama-3.3-70b-versatile");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"reduce the length of the messages"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `Cerebras` > it `available model - should detect overflow via
/// isContextOverflow`, gated on `CEREBRAS_API_KEY`: the preferred ids in
/// order, then the catalog's first model, none is a bug. The 400/413/429
/// status code arrives with no body.
#[tokio::test]
async fn cerebras_available_model_detects_overflow() {
    let Some(api_key) = live::env_key("CEREBRAS_API_KEY") else {
        return;
    };
    let cerebras_models = live::models("cerebras");
    let model = cerebras_models
        .iter()
        .find(|candidate| PREFERRED_CEREBRAS_MODEL_IDS.contains(&candidate.id.as_str()))
        .cloned()
        .or_else(|| cerebras_models.into_iter().next())
        .expect("No Cerebras models available");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"4(00|13|29).*\(no body\)"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `Hugging Face` > it `Kimi-K2.5 - should detect overflow via
/// isContextOverflow`, gated on `HF_TOKEN`.
#[tokio::test]
async fn huggingface_kimi_k2_5_detects_overflow() {
    let Some(api_key) = live::env_key("HF_TOKEN") else {
        return;
    };
    let model = live::model("huggingface", "moonshotai/Kimi-K2.5");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `Together AI` > it `Kimi-K2.6 - should detect overflow via
/// isContextOverflow`, gated on `TOGETHER_API_KEY`.
#[tokio::test]
async fn together_kimi_k2_6_detects_overflow() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let model = live::model("together", "moonshotai/Kimi-K2.6");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `z.ai` > it `glm-5.2 - should detect overflow via
/// isContextOverflow when z.ai reports it`, gated on `ZAI_API_KEY`: the
/// three-way branch — overflow error text pins the classifier, a
/// non-overflow error and a stop without oversized usage print and skip.
#[tokio::test]
async fn zai_glm_5_2_reports_overflow_when_it_reports_it() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let model = live::model("zai", "glm-5.2");
    let response = probe_overflow(&model, &api_key).await;
    if response.stop_reason == StopReason::Error {
        if response
            .error_message
            .as_deref()
            .is_some_and(|message| pattern(r"model_context_window_exceeded").is_match(message))
        {
            assert_overflow(&response, model.context_window);
        } else {
            println!(
                "  z.ai returned non-overflow error (possibly rate limited), skipping overflow detection"
            );
        }
    } else if response.stop_reason == StopReason::Stop {
        // Upstream's condition also gates on `hasUsageData`, which
        // `usage.input > contextWindow` subsumes: zero input never clears the
        // window.
        if response.usage.input > model.context_window {
            assert_overflow(&response, model.context_window);
        } else {
            println!(
                "  z.ai returned stop without overflow usage data, skipping overflow detection"
            );
        }
    }
}

/// Upstream `Mistral` > it `devstral-medium-latest - should detect overflow
/// via isContextOverflow`, gated on `MISTRAL_API_KEY`.
#[tokio::test]
async fn mistral_devstral_medium_latest_detects_overflow() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let model = live::model("mistral", "devstral-medium-latest");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(
        &response,
        &pattern(r"too large for model with \d+ maximum context length"),
    );
    assert_overflow(&response, model.context_window);
}

/// Upstream `MiniMax` > it `MiniMax-M2.7 - should detect overflow via
/// isContextOverflow`, gated on `MINIMAX_API_KEY`.
#[tokio::test]
async fn minimax_m2_7_detects_overflow() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let model = live::model("minimax", "MiniMax-M2.7");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `Xiaomi MiMo (API billing)` > it `mimo-v2.5-pro - should detect
/// overflow via isContextOverflow`, gated on `XIAOMI_API_KEY`.
#[tokio::test]
async fn xiaomi_mimo_v2_5_pro_detects_overflow() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    xiaomi_length_stop_overflow(&api_key, "xiaomi").await;
}

/// Upstream `Xiaomi MiMo Token Plan (CN)` > it `mimo-v2.5-pro - should detect
/// overflow via isContextOverflow`, gated on `XIAOMI_TOKEN_PLAN_CN_API_KEY`.
#[tokio::test]
async fn xiaomi_token_plan_cn_mimo_v2_5_pro_detects_overflow() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    xiaomi_length_stop_overflow(&api_key, "xiaomi-token-plan-cn").await;
}

/// Upstream `Xiaomi MiMo Token Plan (AMS)` > it `mimo-v2.5-pro - should
/// detect overflow via isContextOverflow`, gated on
/// `XIAOMI_TOKEN_PLAN_AMS_API_KEY`.
#[tokio::test]
async fn xiaomi_token_plan_ams_mimo_v2_5_pro_detects_overflow() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    xiaomi_length_stop_overflow(&api_key, "xiaomi-token-plan-ams").await;
}

/// Upstream `Xiaomi MiMo Token Plan (SGP)` > it `mimo-v2.5-pro - should
/// detect overflow via isContextOverflow`, gated on
/// `XIAOMI_TOKEN_PLAN_SGP_API_KEY`.
#[tokio::test]
async fn xiaomi_token_plan_sgp_mimo_v2_5_pro_detects_overflow() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    xiaomi_length_stop_overflow(&api_key, "xiaomi-token-plan-sgp").await;
}

/// Upstream `Qwen Token Plan` > it `qwen3.7-max - should detect overflow via
/// isContextOverflow`, gated on `QWEN_TOKEN_PLAN_API_KEY`.
#[tokio::test]
async fn qwen_token_plan_qwen_3_7_max_detects_overflow() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    qwen_input_length_overflow(&api_key, "qwen-token-plan", "qwen3.7-max").await;
}

/// Upstream `Qwen Token Plan Individual` > it `qwen3.8-max - should detect
/// overflow via isContextOverflow`, gated on the same `QWEN_TOKEN_PLAN_API_KEY`.
#[tokio::test]
async fn qwen_token_plan_individual_qwen_3_8_max_detects_overflow() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    qwen_input_length_overflow(&api_key, "qwen-token-plan-individual", "qwen3.8-max").await;
}

/// Upstream `Qwen Token Plan (CN)` > it `qwen3.7-max - should detect overflow
/// via isContextOverflow`, gated on `QWEN_TOKEN_PLAN_CN_API_KEY`.
#[tokio::test]
async fn qwen_token_plan_cn_qwen_3_7_max_detects_overflow() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    qwen_input_length_overflow(&api_key, "qwen-token-plan-cn", "qwen3.7-max").await;
}

/// Upstream `Kimi For Coding` > it `kimi-for-coding - should detect overflow
/// via isContextOverflow`, gated on `KIMI_API_KEY`.
#[tokio::test]
async fn kimi_for_coding_detects_overflow() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let model = live::model("kimi-coding", "kimi-for-coding");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `Vercel AI Gateway` > it `google/gemini-2.5-flash via AI Gateway - should detect overflow via isContextOverflow`, gated on `AI_GATEWAY_API_KEY`.
#[tokio::test]
async fn vercel_ai_gateway_gemini_2_5_flash_detects_overflow() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let model = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenRouter` > it `anthropic/claude-sonnet-4 via OpenRouter -
/// should detect overflow via isContextOverflow`, gated on
/// `OPENROUTER_API_KEY`.
#[tokio::test]
async fn openrouter_anthropic_claude_sonnet_4_detects_overflow() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let model = live::model("openrouter", "anthropic/claude-sonnet-4");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"maximum context length is \d+ tokens"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenRouter` > it `deepseek/deepseek-v3.2 via OpenRouter - should
/// detect overflow via isContextOverflow`, gated on `OPENROUTER_API_KEY`.
#[tokio::test]
async fn openrouter_deepseek_v3_2_detects_overflow() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let model = live::model("openrouter", "deepseek/deepseek-v3.2");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"maximum context length is \d+ tokens"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenRouter` > it `mistralai/mistral-large-2512 via OpenRouter -
/// should detect overflow via isContextOverflow`, gated on
/// `OPENROUTER_API_KEY`.
#[tokio::test]
async fn openrouter_mistral_large_2512_detects_overflow() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let model = live::model("openrouter", "mistralai/mistral-large-2512");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"maximum context length is \d+ tokens"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenRouter` > it `google/gemini-2.5-flash via OpenRouter -
/// should detect overflow via isContextOverflow`, gated on
/// `OPENROUTER_API_KEY`.
#[tokio::test]
async fn openrouter_google_gemini_2_5_flash_detects_overflow() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let model = live::model("openrouter", "google/gemini-2.5-flash");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"maximum context length is \d+ tokens"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `OpenRouter` > it `meta-llama/llama-4-scout via OpenRouter -
/// should detect overflow via isContextOverflow`, gated on
/// `OPENROUTER_API_KEY`.
#[tokio::test]
async fn openrouter_meta_llama_llama_4_scout_detects_overflow() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let model = live::model("openrouter", "meta-llama/llama-4-scout");
    let response = probe_overflow(&model, &api_key).await;
    assert_error_stop(&response);
    assert_error_message(&response, &pattern(r"maximum context length is \d+ tokens"));
    assert_overflow(&response, model.context_window);
}

/// Upstream `Ollama (local)` > it `gpt-oss:20b - should detect overflow via
/// isContextOverflow (ollama silently truncates)`, gated on
/// `PI_NO_LOCAL_LLM` unset and `ollama` installed: the model is pulled when
/// missing, the server spawned, and the probe either silently truncates
/// (stop with usage, printed and accepted) or reports an error the
/// classifier must flag.
#[tokio::test]
async fn ollama_gpt_oss_20b_silently_truncates_or_reports_overflow() {
    if !local_llm_enabled() || !ollama_installed() {
        return;
    }

    let listed = Command::new("ollama").arg("list").output();
    let have_model = listed.is_ok_and(|output| {
        output.status.success() && String::from_utf8_lossy(&output.stdout).contains("gpt-oss:20b")
    });
    if !have_model {
        println!("Pulling gpt-oss:20b model for Ollama overflow tests...");
        let pulled = Command::new("ollama")
            .arg("pull")
            .arg("gpt-oss:20b")
            .status()
            .is_ok_and(|status| status.success());
        if !pulled {
            println!("Failed to pull gpt-oss:20b model, tests will be skipped");
            return;
        }
    }

    let serve = ServeChild(
        Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("ollama serve spawns"),
    );

    if !wait_for_ollama().await {
        return;
    }

    let model = local_model(
        "gpt-oss:20b",
        "Ollama GPT-OSS 20B",
        "ollama",
        "http://localhost:11434/v1",
        128_000,
        16_000,
        true,
    );
    let response = probe_overflow(&model, "ollama").await;
    if response.stop_reason == StopReason::Stop && response.usage.input > 0 {
        println!(
            "  Ollama silently truncated input to {} tokens",
            response.usage.input
        );
    } else if response.stop_reason == StopReason::Error {
        assert_overflow(&response, model.context_window);
    }

    drop(serve);
}

/// Upstream `LM Studio (local)` > it `should detect overflow via
/// isContextOverflow`, gated on `PI_NO_LOCAL_LLM` unset and a running
/// server.
#[tokio::test]
async fn lm_studio_detects_overflow() {
    if !local_llm_enabled() || !lm_studio_running() {
        return;
    }
    let model = local_model(
        "local-model",
        "LM Studio Local Model",
        "lm-studio",
        "http://localhost:1234/v1",
        8_192,
        2_048,
        false,
    );
    let response = probe_overflow(&model, "lm-studio").await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}

/// Upstream `llama.cpp (local)` > it `should detect overflow via
/// isContextOverflow`, gated on `PI_NO_LOCAL_LLM` unset, a healthy server,
/// and the completions probe: the small window matches the server's
/// `--ctx-size`.
#[tokio::test]
async fn llama_cpp_detects_overflow() {
    if !local_llm_enabled() || !llama_cpp_running() {
        return;
    }
    let model = local_model(
        "local-model",
        "llama.cpp Local Model",
        "llama.cpp",
        "http://localhost:8081/v1",
        4_096,
        2_048,
        false,
    );
    let response = probe_overflow(&model, "llama.cpp").await;
    assert_error_stop(&response);
    assert_overflow(&response, model.context_window);
}
