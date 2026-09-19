//! Exhaustive wire-vocabulary tests: every id and closed-set enum round-trips
//! through `Display`, `TryFrom<&str>`, and serde with the exact wire string.
//! These cover the spelling tables the upstream `types.ts` unions define.

#![allow(
    clippy::expect_used,
    reason = "fixture-heavy wire assertions use expect's failure message as the assertion text; a panic names the broken shape"
)]

use pi_ai::types::{
    Api, CacheControlFormat, CacheRetention, DataCollection, DeferredToolsMode, DeferredWindow,
    GrammarFormat, ImagesApi, ImagesProviderId, KnownApi, KnownImagesApi, KnownImagesProvider,
    KnownProvider, MaxTokensField, Modality, PriceValue, ProviderId, SessionAffinityFormat,
    SortMetric, Strictness, ThinkingFormat, ThinkingTokenBudgetField, ToolChoice, Transport,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Pins one variant to its wire spelling through serde.
fn assert_wire<T>(wire: &str, variant: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let serialized = serde_json::to_string(variant).expect("variant serializes");
    assert_eq!(
        serialized,
        format!("\"{wire}\""),
        "serde spelling of {variant:?}"
    );
    let round: T = serde_json::from_str(&serialized).expect("wire string parses");
    assert_eq!(round, *variant, "serde round-trip of {variant:?}");
}

/// Pins the id types' `Display` to the wire spelling alongside serde.' `Display` to the wire spelling alongside serde.
fn assert_wire_display<T>(wire: &str, variant: &T)
where
    T: Serialize + DeserializeOwned + std::fmt::Display + PartialEq + std::fmt::Debug,
{
    assert_eq!(variant.to_string(), wire, "Display spelling of {variant:?}");
    let serialized = serde_json::to_string(variant).expect("variant serializes");
    assert_eq!(
        serialized,
        format!("\"{wire}\""),
        "serde spelling of {variant:?}"
    );
    let round: T = serde_json::from_str(&serialized).expect("wire string parses");
    assert_eq!(round, *variant, "serde round-trip of {variant:?}");
}

#[test]
fn known_api_wire_spellings_are_exhaustive() {
    let spellings = [
        (KnownApi::OpenaiCompletions, "openai-completions"),
        (KnownApi::MistralConversations, "mistral-conversations"),
        (KnownApi::OpenaiResponses, "openai-responses"),
        (KnownApi::AzureOpenaiResponses, "azure-openai-responses"),
        (KnownApi::OpenaiCodexResponses, "openai-codex-responses"),
        (KnownApi::AnthropicMessages, "anthropic-messages"),
        (KnownApi::BedrockConverseStream, "bedrock-converse-stream"),
        (KnownApi::GoogleGenerativeAi, "google-generative-ai"),
        (KnownApi::GoogleVertex, "google-vertex"),
        (KnownApi::PiMessages, "pi-messages"),
    ];
    for (variant, wire) in spellings {
        assert_wire_display(wire, &variant);
        assert_eq!(KnownApi::try_from(wire), Ok(variant), "TryFrom spelling");
    }
    assert!(KnownApi::try_from("other").is_err());
}

#[test]
fn known_provider_wire_spellings_are_exhaustive() {
    let spellings = [
        (KnownProvider::AmazonBedrock, "amazon-bedrock"),
        (KnownProvider::AntLing, "ant-ling"),
        (KnownProvider::Anthropic, "anthropic"),
        (KnownProvider::Google, "google"),
        (KnownProvider::GoogleVertex, "google-vertex"),
        (KnownProvider::Openai, "openai"),
        (
            KnownProvider::AzureOpenaiResponses,
            "azure-openai-responses",
        ),
        (KnownProvider::OpenaiCodex, "openai-codex"),
        (KnownProvider::Radius, "radius"),
        (KnownProvider::Nvidia, "nvidia"),
        (KnownProvider::Deepseek, "deepseek"),
        (KnownProvider::GithubCopilot, "github-copilot"),
        (KnownProvider::Xai, "xai"),
        (KnownProvider::Groq, "groq"),
        (KnownProvider::Cerebras, "cerebras"),
        (KnownProvider::Openrouter, "openrouter"),
        (KnownProvider::VercelAiGateway, "vercel-ai-gateway"),
        (KnownProvider::Zai, "zai"),
        (KnownProvider::ZaiCodingCn, "zai-coding-cn"),
        (KnownProvider::Mistral, "mistral"),
        (KnownProvider::Minimax, "minimax"),
        (KnownProvider::MinimaxCn, "minimax-cn"),
        (KnownProvider::Moonshotai, "moonshotai"),
        (KnownProvider::MoonshotaiCn, "moonshotai-cn"),
        (KnownProvider::Huggingface, "huggingface"),
        (KnownProvider::Fireworks, "fireworks"),
        (KnownProvider::Together, "together"),
        (KnownProvider::Baseten, "baseten"),
        (KnownProvider::Opencode, "opencode"),
        (KnownProvider::OpencodeGo, "opencode-go"),
        (KnownProvider::KimiCoding, "kimi-coding"),
        (KnownProvider::CloudflareWorkersAi, "cloudflare-workers-ai"),
        (KnownProvider::CloudflareAiGateway, "cloudflare-ai-gateway"),
        (KnownProvider::QwenTokenPlan, "qwen-token-plan"),
        (KnownProvider::QwenTokenPlanCn, "qwen-token-plan-cn"),
        (
            KnownProvider::QwenTokenPlanIndividual,
            "qwen-token-plan-individual",
        ),
        (KnownProvider::Xiaomi, "xiaomi"),
        (KnownProvider::XiaomiTokenPlanCn, "xiaomi-token-plan-cn"),
        (KnownProvider::XiaomiTokenPlanAms, "xiaomi-token-plan-ams"),
        (KnownProvider::XiaomiTokenPlanSgp, "xiaomi-token-plan-sgp"),
    ];
    for (variant, wire) in spellings {
        assert_wire_display(wire, &variant);
        assert_eq!(
            KnownProvider::try_from(wire),
            Ok(variant),
            "TryFrom spelling"
        );
    }
    assert!(KnownProvider::try_from("friend-host").is_err());
}

#[test]
fn image_id_spellings_are_exhaustive() {
    assert_wire_display("openrouter-images", &KnownImagesApi::OpenrouterImages);
    assert_wire_display("openrouter", &KnownImagesProvider::Openrouter);

    let api: ImagesApi = KnownImagesApi::OpenrouterImages.into();
    assert_eq!(api.to_string(), "openrouter-images");
    let custom_api: ImagesApi = String::from("friend-images").into();
    assert_eq!(custom_api.to_string(), "friend-images");

    let provider: ProviderId = KnownProvider::Anthropic.into();
    assert_eq!(provider.to_string(), "anthropic");
    let custom_provider: ProviderId = String::from("friend-relay").into();
    assert_eq!(custom_provider.to_string(), "friend-relay");
    // The newtypes deref to their string ids.
    assert_eq!(&*custom_provider, "friend-relay");
    let api_from_string: Api = String::from("deref-target").into();
    assert_eq!(&*api_from_string, "deref-target");

    let images_provider: ImagesProviderId = KnownImagesProvider::Openrouter.into();
    assert_eq!(images_provider.to_string(), "openrouter");
    let custom_images_provider: ImagesProviderId = String::from("friend-images-relay").into();
    assert_eq!(custom_images_provider.to_string(), "friend-images-relay");
}

#[test]
fn request_vocabulary_wire_spellings_are_exhaustive() {
    assert_wire("auto", &ToolChoice::Auto);
    assert_wire("none", &ToolChoice::None);

    for (variant, wire) in [
        (pi_ai::types::ThinkingLevel::Minimal, "minimal"),
        (pi_ai::types::ThinkingLevel::Low, "low"),
        (pi_ai::types::ThinkingLevel::Medium, "medium"),
        (pi_ai::types::ThinkingLevel::High, "high"),
        (pi_ai::types::ThinkingLevel::Xhigh, "xhigh"),
        (pi_ai::types::ThinkingLevel::Max, "max"),
    ] {
        assert_wire(wire, &variant);
    }

    for (variant, wire) in [
        (pi_ai::types::ModelThinkingLevel::Off, "off"),
        (pi_ai::types::ModelThinkingLevel::Minimal, "minimal"),
        (pi_ai::types::ModelThinkingLevel::Low, "low"),
        (pi_ai::types::ModelThinkingLevel::Medium, "medium"),
        (pi_ai::types::ModelThinkingLevel::High, "high"),
        (pi_ai::types::ModelThinkingLevel::Xhigh, "xhigh"),
        (pi_ai::types::ModelThinkingLevel::Max, "max"),
    ] {
        assert_wire(wire, &variant);
    }

    assert_wire("none", &CacheRetention::None);
    assert_wire("short", &CacheRetention::Short);
    assert_wire("long", &CacheRetention::Long);

    assert_wire("sse", &Transport::Sse);
    assert_wire("websocket", &Transport::Websocket);
    assert_wire("websocket-cached", &Transport::WebsocketCached);
    assert_wire("auto", &Transport::Auto);

    assert_wire("openai", &SessionAffinityFormat::Openai);
    assert_wire("openai-nosession", &SessionAffinityFormat::OpenaiNosession);
    assert_wire("openrouter", &SessionAffinityFormat::Openrouter);

    assert_wire(
        "max_completion_tokens",
        &MaxTokensField::MaxCompletionTokens,
    );
    assert_wire("max_tokens", &MaxTokensField::MaxTokens);

    for (variant, wire) in [
        (
            ThinkingTokenBudgetField::ThinkingTokenBudget,
            "thinking_token_budget",
        ),
        (ThinkingTokenBudgetField::ThinkingBudget, "thinking_budget"),
        (
            ThinkingTokenBudgetField::ThinkingBudgetTokens,
            "thinking_budget_tokens",
        ),
    ] {
        assert_wire(wire, &variant);
    }

    for (variant, wire) in [
        (ThinkingFormat::Openai, "openai"),
        (ThinkingFormat::Openrouter, "openrouter"),
        (ThinkingFormat::Deepseek, "deepseek"),
        (ThinkingFormat::Together, "together"),
        (ThinkingFormat::Baseten, "baseten"),
        (ThinkingFormat::Zai, "zai"),
        (ThinkingFormat::Qwen, "qwen"),
        (ThinkingFormat::ChatTemplate, "chat-template"),
        (ThinkingFormat::QwenChatTemplate, "qwen-chat-template"),
        (ThinkingFormat::StringThinking, "string-thinking"),
        (ThinkingFormat::AntLing, "ant-ling"),
    ] {
        assert_wire(wire, &variant);
    }

    assert_wire("anthropic", &CacheControlFormat::Anthropic);
    assert_wire("kimi", &DeferredToolsMode::Kimi);

    assert_wire("prefer", &Strictness::Prefer);
    assert_wire("require", &Strictness::Require);

    assert_wire("deny", &DataCollection::Deny);
    assert_wire("allow", &DataCollection::Allow);

    assert_wire("price", &SortMetric::Price);
    assert_wire("throughput", &SortMetric::Throughput);
    assert_wire("latency", &SortMetric::Latency);

    assert_wire("openai_lark", &GrammarFormat::OpenaiLark);
    assert_wire("openai_regex", &GrammarFormat::OpenaiRegex);

    assert_wire("15m", &DeferredWindow::M15);
    assert_wire("1h", &DeferredWindow::H1);
    assert_wire("24h", &DeferredWindow::H24);

    assert_wire("text", &Modality::Text);
    assert_wire("image", &Modality::Image);
}

#[test]
fn stop_reasons_cover_every_wire_value() {
    for (variant, wire) in [
        (pi_ai::types::StopReason::Pending, "pending"),
        (pi_ai::types::StopReason::Stop, "stop"),
        (pi_ai::types::StopReason::Length, "length"),
        (pi_ai::types::StopReason::ToolUse, "toolUse"),
        (pi_ai::types::StopReason::Error, "error"),
        (pi_ai::types::StopReason::Aborted, "aborted"),
        (pi_ai::types::StopReason::Deferred, "deferred"),
    ] {
        assert_wire(wire, &variant);
    }

    assert_wire("stop", &pi_ai::types::ImagesStopReason::Stop);
    assert_wire("error", &pi_ai::types::ImagesStopReason::Error);
    assert_wire("aborted", &pi_ai::types::ImagesStopReason::Aborted);
}

#[test]
fn open_router_price_and_sort_variants_round_trip() {
    let num: PriceValue = serde_json::from_value(serde_json::json!(3.5)).expect("number price");
    assert_eq!(num, PriceValue::Num(3.5));
    let text: PriceValue = serde_json::from_value(serde_json::json!("2")).expect("string price");
    assert_eq!(text, PriceValue::Str("2".to_string()));
    assert_eq!(
        serde_json::to_value(PriceValue::Num(2.0)).expect("number serializes"),
        serde_json::json!(2.0)
    );
    assert_eq!(
        serde_json::to_value(PriceValue::Str("2".to_string())).expect("string serializes"),
        serde_json::json!("2")
    );

    let deny: DataCollection =
        serde_json::from_value(serde_json::json!("deny")).expect("deny parses");
    assert_eq!(deny, DataCollection::Deny);
    let allow: DataCollection =
        serde_json::from_value(serde_json::json!("allow")).expect("allow parses");
    assert_eq!(allow, DataCollection::Allow);

    let detailed: pi_ai::types::SortPreference = serde_json::from_value(serde_json::json!({
        "by": "latency",
        "partition": null
    }))
    .expect("detailed sort");
    match &detailed {
        pi_ai::types::SortPreference::Detailed { by, partition } => {
            assert_eq!(*by, Some(SortMetric::Latency));
            assert_eq!(*partition, Some(None), "the wire's null partition is none");
        }
        pi_ai::types::SortPreference::Name(_) => {
            unreachable!("expected a detailed sort, found the name form")
        }
    }
    let named: pi_ai::types::SortPreference =
        serde_json::from_value(serde_json::json!("price")).expect("named sort");
    assert_eq!(
        named,
        pi_ai::types::SortPreference::Name("price".to_string())
    );
}

#[test]
fn id_newtypes_round_trip_through_transparent_serde() {
    let api = Api::from("openai-completions");
    assert_eq!(
        serde_json::to_string(&api).expect("api serializes"),
        "\"openai-completions\""
    );
    let parsed: Api = serde_json::from_str("\"my-gateway\"").expect("api parses");
    assert_eq!(parsed.to_string(), "my-gateway");

    let provider = ProviderId::from("zai");
    assert_eq!(
        serde_json::to_string(&provider).expect("provider serializes"),
        "\"zai\""
    );
    let images_api: ImagesApi = "openrouter-images".into();
    assert_eq!(
        serde_json::to_string(&images_api).expect("images api serializes"),
        "\"openrouter-images\""
    );
    let images_provider: ImagesProviderId = "openrouter".into();
    assert_eq!(
        serde_json::to_string(&images_provider).expect("images provider serializes"),
        "\"openrouter\""
    );
}
