//! The compaction model-override suite, upstream's
//! `packages/coding-agent/test/settings-manager-compaction.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated 1:1. Regression
//! coverage for upstream #8133.
//!
//! The `it.each([Number.NaN, Infinity, -Infinity])` non-finite runtime rows
//! of both `describe.each` matrices are not portable: serde_json has no
//! NaN/Infinity representation, so `applyOverrides` cannot carry them. The
//! invalid-value branch those rows exercise is the same branch the expressible
//! rows below pin. Every other row ports, the `describe.each`/`it.each`
//! matrices expanded into the workspace's helper-loop form.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use pi_coding_agent::settings_manager::{
    CompactionSettings, InMemorySettingsStorage, ModelKey, SettingsManager,
    SettingsManagerCreateOptions, SettingsScope,
};
use serde_json::{Value, json};

const DEFAULTS: CompactionSettings = CompactionSettings {
    enabled: true,
    reserve_tokens: 16_384,
    keep_recent_tokens: 20_000,
};

fn model() -> ModelKey {
    ModelKey {
        provider: "provider".to_string(),
        id: "family/model".to_string(),
    }
}

fn in_memory(settings: &Value) -> SettingsManager<InMemorySettingsStorage> {
    SettingsManager::in_memory(
        &settings.as_object().expect("settings object").clone(),
        SettingsManagerCreateOptions::default(),
    )
}

fn seeded_manager(compaction: &Value) -> SettingsManager<InMemorySettingsStorage> {
    let storage = InMemorySettingsStorage::default();
    storage.seed(
        SettingsScope::Global,
        json!({"compaction": compaction}).to_string(),
    );
    SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default())
}

/// The invalid values upstream's `it.each` rows feed, with the `String(value)`
/// spelling each error message carries: upstream's `String([])` is the empty
/// string and `String({})` is `[object Object]`.
fn invalid_token_values() -> Vec<(Value, &'static str)> {
    vec![
        (json!(null), "null"),
        (json!(-1), "-1"),
        (json!(1.5), "1.5"),
        (json!("400_000"), "400_000"),
        (json!(true), "true"),
        (json!({}), "[object Object]"),
        (json!([]), ""),
        (json!(9_007_199_254_740_992i64), "9007199254740992"),
    ]
}

/// The malformed model entries upstream's `reports malformed model entries`
/// `it.each` feeds, with the `String(entry)` spellings.
fn malformed_entries() -> Vec<(Value, &'static str)> {
    vec![
        (json!(null), "null"),
        (json!(false), "false"),
        (json!(42), "42"),
        (json!("invalid"), "invalid"),
        (json!([]), ""),
    ]
}

#[test]
fn uses_defaults_without_compaction_settings() {
    let manager = in_memory(&json!({}));
    assert_eq!(
        manager.get_compaction_settings(None).expect("valid"),
        DEFAULTS
    );
    assert_eq!(
        manager
            .get_compaction_settings(Some(&model()))
            .expect("valid"),
        DEFAULTS
    );
}

#[test]
fn resolves_each_field_independently_and_keeps_individual_getters_consistent() {
    let mut manager = in_memory(&json!({
        "compaction": {
            "reserveTokens": 8_192,
            "keepRecentTokens": 10_000,
            "modelOverrides": {"provider/family/model": {"reserveTokens": 400_000}},
        },
    }));
    let model = model();
    assert_eq!(
        manager
            .get_compaction_settings(Some(&model))
            .expect("valid"),
        CompactionSettings {
            enabled: true,
            reserve_tokens: 400_000,
            keep_recent_tokens: 10_000
        }
    );
    assert_eq!(
        manager
            .get_compaction_reserve_tokens(Some(&model))
            .expect("valid"),
        400_000
    );
    assert_eq!(
        manager
            .get_compaction_keep_recent_tokens(Some(&model))
            .expect("valid"),
        10_000
    );
    assert_eq!(
        manager.get_compaction_settings(None).expect("valid"),
        CompactionSettings {
            enabled: true,
            reserve_tokens: 8_192,
            keep_recent_tokens: 10_000
        }
    );

    manager.apply_overrides(
        &json!({
            "compaction": {"modelOverrides": {"provider/family/model": {"keepRecentTokens": 30_000}}},
        })
        .as_object()
        .expect("object")
        .clone(),
    );
    assert_eq!(
        manager
            .get_compaction_keep_recent_tokens(Some(&model))
            .expect("valid"),
        30_000
    );
    assert_eq!(
        manager
            .get_compaction_reserve_tokens(Some(&model))
            .expect("valid"),
        400_000
    );
}

#[test]
fn falls_back_to_built_in_defaults_for_missing_fields() {
    let manager = in_memory(&json!({
        "compaction": {"modelOverrides": {"provider/family/model": {"keepRecentTokens": 1_024}}},
    }));
    assert_eq!(
        manager
            .get_compaction_settings(Some(&model()))
            .expect("valid"),
        CompactionSettings {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 1_024
        }
    );
}

#[test]
fn matches_exact_provider_model_ids_including_ids_containing_slashes() {
    let manager = in_memory(&json!({
        "compaction": {
            "modelOverrides": {
                "provider/family/model": {"reserveTokens": 400_000},
                "provider/*": {"reserveTokens": 1},
                "family/model": {"reserveTokens": 2},
            },
        },
    }));
    assert_eq!(
        manager
            .get_compaction_reserve_tokens(Some(&model()))
            .expect("valid"),
        400_000
    );
    let others = [
        ModelKey {
            provider: "other".to_string(),
            id: "family/model".to_string(),
        },
        ModelKey {
            provider: "provider".to_string(),
            id: "other".to_string(),
        },
        ModelKey {
            provider: "provider".to_string(),
            id: "family/Model".to_string(),
        },
    ];
    for other in &others {
        assert_eq!(
            manager.get_compaction_settings(Some(other)).expect("valid"),
            DEFAULTS,
            "{other:?}"
        );
    }
}

#[test]
fn merges_project_model_overrides_per_field_before_resolving_fallbacks() {
    let storage = InMemorySettingsStorage::default();
    storage.seed(
        SettingsScope::Global,
        json!({
            "compaction": {
                "reserveTokens": 8_192,
                "modelOverrides": {
                    "provider/family/model": {"reserveTokens": 400_000, "keepRecentTokens": 30_000},
                    "provider/other": {"keepRecentTokens": 4_096},
                },
            },
        })
        .to_string(),
    );
    storage.seed(
        SettingsScope::Project,
        json!({
            "compaction": {
                "reserveTokens": 1_024,
                "modelOverrides": {"provider/family/model": {"keepRecentTokens": 2_000}},
            },
        })
        .to_string(),
    );
    let mut manager =
        SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default());
    let model = model();
    assert_eq!(
        manager
            .get_compaction_settings(Some(&model))
            .expect("valid"),
        CompactionSettings {
            enabled: true,
            reserve_tokens: 400_000,
            keep_recent_tokens: 2_000
        }
    );
    assert_eq!(
        manager
            .get_compaction_settings(Some(&ModelKey {
                provider: "provider".to_string(),
                id: "other".to_string(),
            }))
            .expect("valid"),
        CompactionSettings {
            enabled: true,
            reserve_tokens: 1_024,
            keep_recent_tokens: 4_096
        }
    );
    manager.reload();
    assert_eq!(
        manager
            .get_compaction_keep_recent_tokens(Some(&model))
            .expect("valid"),
        2_000
    );
    manager.set_project_trusted(false);
    assert_eq!(
        manager
            .get_compaction_keep_recent_tokens(Some(&model))
            .expect("valid"),
        30_000
    );
}

#[tokio::test]
async fn keeps_enabled_global_and_preserves_overrides_when_saving_the_toggle() {
    let storage = InMemorySettingsStorage::default();
    storage.seed(
        SettingsScope::Global,
        json!({
            "compaction": {
                "modelOverrides": {"provider/family/model": {"enabled": false, "reserveTokens": 400_000}},
            },
        })
        .to_string(),
    );
    let mut manager =
        SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default());
    assert!(
        manager
            .get_compaction_settings(Some(&model()))
            .expect("valid")
            .enabled
    );
    manager.set_compaction_enabled(false);
    manager.flush().await;
    manager.reload();
    assert_eq!(
        manager
            .get_compaction_settings(Some(&model()))
            .expect("valid"),
        CompactionSettings {
            enabled: false,
            reserve_tokens: 400_000,
            keep_recent_tokens: 20_000
        }
    );
}

// === describe.each("model override %s") =====================================

#[test]
fn model_override_reports_invalid_token_values() {
    for field in ["reserveTokens", "keepRecentTokens"] {
        for (value, spelling) in invalid_token_values() {
            let manager = seeded_manager(&json!({
                "modelOverrides": {"provider/family/model": {(field): value}},
            }));
            let error = format!(
                "Invalid compaction.modelOverrides[\"provider/family/model\"].{field} setting: \
                 {spelling}. Expected a non-negative safe integer."
            );
            assert_eq!(
                manager.get_compaction_settings(Some(&model())),
                Err(error.clone()),
                "{field}: {spelling}"
            );
            assert_eq!(
                manager
                    .get_compaction_settings(None)
                    .expect("ordinary scope untouched"),
                DEFAULTS,
                "{field}: {spelling}"
            );
            assert_eq!(
                manager
                    .get_compaction_settings(Some(&ModelKey {
                        provider: "other".to_string(),
                        id: "family/model".to_string(),
                    }))
                    .expect("other model untouched"),
                DEFAULTS,
                "{field}: {spelling}"
            );
        }
    }
}

// The non-finite runtime rows (NaN, Infinity, -Infinity) of this matrix are
// not portable — see the file header.

// === describe.each("ordinary compaction.%s") ================================

#[test]
fn ordinary_compaction_reports_invalid_values_even_when_a_valid_model_override_exists() {
    for field in ["reserveTokens", "keepRecentTokens"] {
        for (value, spelling) in invalid_token_values() {
            let manager = seeded_manager(&json!({
                (field): value,
                "modelOverrides": {"provider/family/model": {(field): 4_096}},
            }));
            let error = format!(
                "Invalid compaction.{field} setting: {spelling}. \
                 Expected a non-negative safe integer."
            );
            assert_eq!(
                manager.get_compaction_settings(None),
                Err(error.clone()),
                "{field}: {spelling}"
            );
            assert_eq!(
                manager.get_compaction_settings(Some(&model())),
                Err(error),
                "{field}: {spelling}"
            );
        }
    }
}

// The non-finite runtime rows (NaN, Infinity, -Infinity) of this matrix are
// not portable — see the file header.

#[test]
fn reports_malformed_model_entries() {
    for (entry, spelling) in malformed_entries() {
        let manager = seeded_manager(&json!({
            "modelOverrides": {"provider/family/model": entry},
        }));
        let error = format!(
            "Invalid compaction.modelOverrides[\"provider/family/model\"] setting: {spelling}. \
             Expected an object."
        );
        assert_eq!(
            manager.get_compaction_settings(Some(&model())),
            Err(error),
            "{spelling}"
        );
    }
}

#[test]
fn accepts_zero_in_ordinary_settings_and_model_overrides() {
    let mut manager = in_memory(&json!({
        "compaction": {"reserveTokens": 0, "keepRecentTokens": 0},
    }));
    assert_eq!(
        manager
            .get_compaction_settings(Some(&model()))
            .expect("valid"),
        CompactionSettings {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 0
        }
    );
    manager.apply_overrides(
        &json!({
            "compaction": {
                "reserveTokens": 1000,
                "keepRecentTokens": 1000,
                "modelOverrides": {
                    "provider/family/model": {"reserveTokens": 0, "keepRecentTokens": 0},
                },
            },
        })
        .as_object()
        .expect("object")
        .clone(),
    );
    assert_eq!(
        manager
            .get_compaction_settings(Some(&model()))
            .expect("valid"),
        CompactionSettings {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 0
        }
    );
}
