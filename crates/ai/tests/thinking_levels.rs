//! The `getSupportedThinkingLevels` catalog assertions, ported from
//! `test/supports-xhigh.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. Every case reads the committed
//! generated catalog; the levels a model exposes are data, not behavior.

mod common;

use pi_ai::models::get_supported_thinking_levels;
use pi_ai::types::{Model, ModelThinkingLevel};

/// The supported levels of one catalog model, in the order
/// [`get_supported_thinking_levels`] emits them.
fn levels(provider: &str, id: &str) -> Vec<ModelThinkingLevel> {
    let model: Model = common::builtin_model(provider, id);
    get_supported_thinking_levels(&model)
}

#[test]
fn includes_max_but_not_xhigh_for_anthropic_opus_4_6() {
    let levels = levels("anthropic", "claude-opus-4-6");
    assert!(levels.contains(&ModelThinkingLevel::Max));
    assert!(!levels.contains(&ModelThinkingLevel::Xhigh));
}

#[test]
fn includes_xhigh_and_max_for_anthropic_opus_4_8() {
    let levels = levels("anthropic", "claude-opus-4-8");
    assert!(levels.contains(&ModelThinkingLevel::Xhigh));
    assert!(levels.contains(&ModelThinkingLevel::Max));
}

#[test]
fn includes_xhigh_and_max_for_anthropic_opus_5() {
    let levels = levels("anthropic", "claude-opus-5");
    assert!(levels.contains(&ModelThinkingLevel::Xhigh));
    assert!(levels.contains(&ModelThinkingLevel::Max));
}

#[test]
fn includes_max_but_not_xhigh_for_anthropic_sonnet_4_6() {
    let levels = levels("anthropic", "claude-sonnet-4-6");
    assert!(levels.contains(&ModelThinkingLevel::Max));
    assert!(!levels.contains(&ModelThinkingLevel::Xhigh));
}

#[test]
fn includes_xhigh_and_max_for_anthropic_sonnet_5() {
    let levels = levels("anthropic", "claude-sonnet-5");
    assert!(levels.contains(&ModelThinkingLevel::Xhigh));
    assert!(levels.contains(&ModelThinkingLevel::Max));
}

#[test]
fn includes_xhigh_and_max_but_not_off_for_anthropic_claude_fable_5() {
    let levels = levels("anthropic", "claude-fable-5");
    assert!(levels.contains(&ModelThinkingLevel::Xhigh));
    assert!(levels.contains(&ModelThinkingLevel::Max));
    assert!(!levels.contains(&ModelThinkingLevel::Off));
}

#[test]
fn does_not_include_xhigh_or_max_for_claude_sonnet_4_5() {
    let levels = levels("anthropic", "claude-sonnet-4-5");
    assert!(!levels.contains(&ModelThinkingLevel::Xhigh));
    assert!(!levels.contains(&ModelThinkingLevel::Max));
}

#[test]
fn includes_xhigh_for_openai_codex_gpt_5_models() {
    for id in [
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-6-astra",
    ] {
        assert!(
            levels("openai-codex", id).contains(&ModelThinkingLevel::Xhigh),
            "openai-codex {id} should expose xhigh"
        );
    }
}

#[test]
fn includes_xhigh_and_max_for_openai_gpt_5_6_models() {
    for id in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] {
        assert_eq!(
            levels("openai", id),
            vec![
                ModelThinkingLevel::Off,
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
                ModelThinkingLevel::Max,
            ]
        );
    }
}

#[test]
fn includes_only_medium_high_xhigh_for_openai_gpt_5_5_pro() {
    assert_eq!(
        levels("openai", "gpt-5.5-pro"),
        vec![
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Xhigh,
        ]
    );
}

#[test]
fn includes_only_medium_high_xhigh_for_openrouter_gpt_5_5_pro() {
    assert_eq!(
        levels("openrouter", "openai/gpt-5.5-pro"),
        vec![
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Xhigh,
        ]
    );
}

#[test]
fn includes_low_high_max_plus_off_for_deepseek_v4_1_flash_on_the_deepseek_provider() {
    assert_eq!(
        levels("deepseek", "deepseek-flash"),
        vec![
            ModelThinkingLevel::Off,
            ModelThinkingLevel::Low,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Max,
        ]
    );
}

#[test]
fn includes_low_high_max_plus_off_for_deepseek_v4_flash_on_opencode_go() {
    assert_eq!(
        levels("opencode-go", "deepseek-v4-flash"),
        vec![
            ModelThinkingLevel::Off,
            ModelThinkingLevel::Low,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Max,
        ]
    );
}

#[test]
fn includes_only_high_plus_off_for_opencode_go_kimi_k2_6() {
    assert_eq!(
        levels("opencode-go", "kimi-k2.6"),
        vec![ModelThinkingLevel::Off, ModelThinkingLevel::High]
    );
}

#[test]
fn excludes_thinking_off_for_moonshot_kimi_k2_7_code_models() {
    for provider in ["moonshotai", "moonshotai-cn"] {
        assert_eq!(
            levels(provider, "kimi-k2.7-code"),
            vec![
                ModelThinkingLevel::Minimal,
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
            ]
        );
    }
}

#[test]
fn uses_the_verified_effort_options_for_moonshot_kimi_k3() {
    for provider in ["moonshotai", "moonshotai-cn"] {
        assert_eq!(
            levels(provider, "kimi-k3"),
            vec![
                ModelThinkingLevel::Low,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Max,
            ]
        );
    }
}

#[test]
fn includes_only_low_high_max_for_kimi_coding_k3() {
    assert_eq!(
        levels("kimi-coding", "k3"),
        vec![
            ModelThinkingLevel::Low,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Max,
        ]
    );
}

#[test]
fn includes_only_high_for_opencode_grok_build() {
    assert_eq!(
        levels("opencode", "grok-build-0.1"),
        vec![ModelThinkingLevel::High]
    );
}

#[test]
fn includes_only_high_xhigh_plus_off_for_deepseek_v4_flash_on_openrouter() {
    assert_eq!(
        levels("openrouter", "deepseek/deepseek-v4-flash"),
        vec![
            ModelThinkingLevel::Off,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Xhigh,
        ]
    );
}

#[test]
fn includes_max_but_not_xhigh_for_openrouter_opus_4_6() {
    let levels = levels("openrouter", "anthropic/claude-opus-4.6");
    assert!(levels.contains(&ModelThinkingLevel::Max));
    assert!(!levels.contains(&ModelThinkingLevel::Xhigh));
}

#[test]
fn includes_xhigh_and_max_for_bedrock_claude_opus_5() {
    let levels = levels("amazon-bedrock", "global.anthropic.claude-opus-5");
    assert!(levels.contains(&ModelThinkingLevel::Xhigh));
    assert!(levels.contains(&ModelThinkingLevel::Max));
}

#[test]
fn includes_xhigh_but_not_off_or_max_for_xai_grok_4_6() {
    assert_eq!(
        levels("xai", "grok-4.6"),
        vec![
            ModelThinkingLevel::Low,
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Xhigh,
        ]
    );
}

#[test]
fn includes_xhigh_and_max_but_not_off_for_bedrock_claude_fable_5() {
    let levels = levels("amazon-bedrock", "global.anthropic.claude-fable-5");
    assert!(levels.contains(&ModelThinkingLevel::Xhigh));
    assert!(levels.contains(&ModelThinkingLevel::Max));
    assert!(!levels.contains(&ModelThinkingLevel::Off));
}
