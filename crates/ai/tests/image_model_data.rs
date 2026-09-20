//! The OpenRouter image-model parsing suite, ported from
//! `packages/ai/test/image-model-data.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_ai::image_models::parse_openrouter_image_models;

fn valid_image_model() -> serde_json::Value {
    serde_json::json!({
        "id": "example/image-model",
        "name": "Example Image Model",
        "architecture": {
            "input_modalities": ["text", "image"],
            "output_modalities": ["image"],
        },
        "pricing": {
            "prompt": "0.000001",
            "completion": "0.000002",
        },
    })
}

#[test]
fn rejects_a_missing_or_empty_strict_catalog() {
    for payload in [
        serde_json::json!({}),
        serde_json::json!({ "data": [] }),
        serde_json::json!({ "data": "invalid" }),
    ] {
        let error =
            parse_openrouter_image_models(&payload, true).expect_err("strict catalog fails");
        assert_eq!(
            error.to_string(),
            "OpenRouter API returned a missing or empty image model list"
        );
    }
}

#[test]
fn rejects_a_strict_catalog_with_no_usable_image_models() {
    let payload = serde_json::json!({
        "data": [{
            "id": "example/image-model",
            "name": "Example Image Model",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
            },
        }],
    });
    let error = parse_openrouter_image_models(&payload, true).expect_err("no usable models fails");
    assert_eq!(
        error.to_string(),
        "OpenRouter API returned no usable image models"
    );
}

#[test]
fn parses_a_non_empty_image_model_catalog() {
    let payload = serde_json::json!({ "data": [valid_image_model()] });
    let models = parse_openrouter_image_models(&payload, true).expect("parses");
    assert_eq!(models.len(), 1);
    let model = &models[0];
    assert_eq!(model.id, "example/image-model");
    assert_eq!(
        model.input,
        vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image]
    );
    assert_eq!(model.output, vec![pi_ai::types::Modality::Image]);
}
