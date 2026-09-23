//! The image-generation dispatch suites, ported from
//! `packages/ai/test/images.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The E2E block gates on `OPENROUTER_API_KEY` exactly like upstream's
//! `describe.skipIf`: a probe without its provider credential returns early,
//! and the registry behavior assertions always run. The `red-circle.png`
//! fixture is the suite's only fixture, carried from upstream's `test/data`.
//!
//! The registry boundary tests go past upstream's file: the image-API
//! registry's mismatch wrapper, the builtin seed's no-clobber rule, and the
//! source-id unregistration have no upstream suite of their own.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected shape panics the test by design"
)]
use std::sync::Arc;

use pi_ai::image_models::get_image_model;
use pi_ai::images::generate_images;
use pi_ai::images_api_registry::{
    ImagesApiProvider, get_images_api_provider, register_images_api_provider,
    unregister_images_api_providers,
};
use pi_ai::providers::images::generate_images_open_router;
use pi_ai::types::{
    AssistantImages, BoxedFuture, ImageContent, ImagesBlock, ImagesContext, ImagesModel,
    ImagesOptions, ImagesStopReason, Modality, ProviderImages, TextContent,
};
use pi_ai::utils::provider_retry::ProviderRequestError;

/// The process-wide serialization for every test touching the
/// `openrouter-images` registration: the registry is process-wide and the
/// tests in this file run on parallel threads, so the seed probe and the
/// live block serialize the way upstream's module-global registry serializes
/// its suite.
static OPENROUTER_IMAGES_GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The single E2E model upstream's live block runs against.
const OPENROUTER_IMAGE_MODEL: (&str, &str) = ("openrouter", "google/gemini-2.5-flash-image");

fn openrouter_image_model() -> ImagesModel {
    get_image_model(OPENROUTER_IMAGE_MODEL.0, OPENROUTER_IMAGE_MODEL.1)
        .expect("the generated catalog pins google/gemini-2.5-flash-image")
}

fn text_input(text: &str) -> ImagesContext {
    ImagesContext {
        input: vec![ImagesBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
    }
}

fn basic_image_generation(model: &ImagesModel) -> BoxedFuture<'_, ()> {
    let model = model.clone();
    Box::pin(async move {
        let request_context =
            text_input("Generate a simple red circle on a plain white background. No text.");

        let response = generate_images(&model, &request_context, None)
            .await
            .expect("the live probe dispatches");

        assert!(
            response.stop_reason == ImagesStopReason::Stop,
            "Error: {}",
            response.error_message.as_deref().unwrap_or_default()
        );
        assert!(response.error_message.is_none());
        assert!(
            response
                .output
                .iter()
                .any(|item| matches!(item, ImagesBlock::Image(..)))
        );
        assert!(response.timestamp > 0);
    })
}

fn handle_text_and_image_output(model: &ImagesModel) -> BoxedFuture<'_, ()> {
    let model = model.clone();
    Box::pin(async move {
        if !model.output.contains(&Modality::Text) {
            // Skipping text+image output — the model doesn't support text
            // output.
            return;
        }

        let request_context =
            text_input("Generate a red circle and include a brief description of the image.");

        let response = generate_images(&model, &request_context, None)
            .await
            .expect("the live probe dispatches");

        assert!(
            response.stop_reason == ImagesStopReason::Stop,
            "Error: {}",
            response.error_message.as_deref().unwrap_or_default()
        );
        assert!(
            response
                .output
                .iter()
                .any(|item| matches!(item, ImagesBlock::Image(..)))
        );
        assert!(response.output.iter().any(|item| {
            matches!(item, ImagesBlock::Text(text) if !text.text.trim().is_empty())
        }));
    })
}

/// The standard base64 alphabet the fixture rides in, upstream's
/// `imageBuffer.toString("base64")`.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        encoded.push(ALPHABET[(n >> 18) as usize & 63] as char);
        encoded.push(ALPHABET[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            encoded.push(ALPHABET[(n >> 6) as usize & 63] as char);
        } else {
            encoded.push('=');
        }
        if chunk.len() > 2 {
            encoded.push(ALPHABET[n as usize & 63] as char);
        } else {
            encoded.push('=');
        }
    }
    encoded
}

fn red_circle_fixture() -> ImageContent {
    let image_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("red-circle.png");
    let image_bytes = std::fs::read(&image_path).expect("the red-circle fixture reads");
    ImageContent {
        data: base64_encode(&image_bytes),
        mime_type: "image/png".to_owned(),
    }
}

fn handle_image_input(model: &ImagesModel) -> BoxedFuture<'_, ()> {
    let model = model.clone();
    Box::pin(async move {
        if !model.input.contains(&Modality::Image) {
            // Skipping image input — the model doesn't support image input.
            return;
        }

        let fixture = red_circle_fixture();
        let request_context = ImagesContext {
            input: vec![
                ImagesBlock::Text(TextContent {
                    text: "Create a variation of this image with a blue background.".to_owned(),
                    text_signature: None,
                }),
                ImagesBlock::Image(fixture),
            ],
        };

        let response = generate_images(&model, &request_context, None)
            .await
            .expect("the live probe dispatches");

        assert!(
            response.stop_reason == ImagesStopReason::Stop,
            "Error: {}",
            response.error_message.as_deref().unwrap_or_default()
        );
        assert!(
            response
                .output
                .iter()
                .any(|item| matches!(item, ImagesBlock::Image(..)))
        );
    })
}

/// The live probes, upstream's `Images E2E Tests` describe block.
#[tokio::test]
async fn live_images_e2e() {
    let Ok(key) = std::env::var("OPENROUTER_API_KEY") else {
        // Without the provider credential the probes skip, upstream's
        // `describe.skipIf`.
        return;
    };
    let _ = key;
    let _guard = OPENROUTER_IMAGES_GUARD.lock().await;
    let model = openrouter_image_model();

    basic_image_generation(&model).await;
    handle_text_and_image_output(&model).await;
    handle_image_input(&model).await;
}

/// A stub image API recording dispatch and reporting a sentinel result, the
/// registry boundary tests' fixture.
fn stub_api(marker: &'static str) -> Arc<dyn ProviderImages> {
    struct StubApi(&'static str);
    impl ProviderImages for StubApi {
        fn generate_images<'a>(
            &'a self,
            model: &'a ImagesModel,
            _context: &'a ImagesContext,
            _options: Option<&'a ImagesOptions>,
        ) -> BoxedFuture<'a, Result<AssistantImages, ProviderRequestError>> {
            Box::pin(async move {
                Ok(AssistantImages {
                    api: model.api.clone(),
                    provider: model.provider.clone(),
                    model: model.id.clone(),
                    response_id: Some(self.0.to_owned()),
                    output: Vec::new(),
                    usage: None,
                    stop_reason: ImagesStopReason::Stop,
                    error_message: None,
                    timestamp: 1,
                })
            })
        }
    }
    Arc::new(StubApi(marker))
}

#[tokio::test]
async fn fails_when_no_api_provider_is_registered_for_the_model_api() {
    let model = openrouter_image_model();
    let mut unknown = model.clone();
    unknown.api = pi_ai::types::ImagesApi::from("no-such-images-api");

    let error = generate_images(&unknown, &text_input("x"), None)
        .await
        .expect_err("no provider is registered for the unknown api");
    assert_eq!(
        error.message,
        "No API provider registered for api: no-such-images-api"
    );
}

/// The builtin seed lifecycle: an explicit registration made before or after
/// the seed wins, and once the seed has run the dispatch no longer re-registers.
#[tokio::test]
async fn the_dispatch_seeds_the_builtins_once_and_never_clobbers_an_explicit_registration() {
    let _guard = OPENROUTER_IMAGES_GUARD.lock().await;
    register_images_api_provider(
        ImagesApiProvider {
            api: "openrouter-images".to_owned(),
            generate_images: stub_api("explicit-stub"),
        },
        Some("seed-probe"),
    );

    let model = openrouter_image_model();
    let response = generate_images(&model, &text_input("x"), None)
        .await
        .expect("the explicit registration dispatches");
    assert_eq!(response.response_id.as_deref(), Some("explicit-stub"));

    unregister_images_api_providers("seed-probe");
    let error = generate_images(&model, &text_input("x"), None)
        .await
        .expect_err("the seed has run, so the dispatch does not re-register");
    assert_eq!(
        error.message,
        "No API provider registered for api: openrouter-images"
    );
}

#[tokio::test]
async fn the_mismatch_check_fails_a_direct_registry_call_with_another_api_model() {
    register_images_api_provider(
        ImagesApiProvider {
            api: "mismatch-test-api".to_owned(),
            generate_images: stub_api("mismatch-stub"),
        },
        None,
    );
    let provider =
        get_images_api_provider("mismatch-test-api").expect("the registration dispatches");
    let mut model = openrouter_image_model();
    model.api = pi_ai::types::ImagesApi::from("other-api");

    let error = provider
        .generate_images(&model, &text_input("x"), None)
        .await
        .expect_err("the mismatch check fails the call");
    assert_eq!(
        error.message,
        "Mismatched api: other-api expected mismatch-test-api"
    );
}

#[tokio::test]
async fn per_source_unregistration_sweeps_only_its_registrations() {
    register_images_api_provider(
        ImagesApiProvider {
            api: "sweep-test-api".to_owned(),
            generate_images: stub_api("sweep-stub"),
        },
        Some("sweep-source"),
    );
    assert!(get_images_api_provider("sweep-test-api").is_some());

    unregister_images_api_providers("sweep-source");
    assert!(get_images_api_provider("sweep-test-api").is_none());
}

#[tokio::test]
async fn the_open_router_wrapper_reports_delegation_failures_in_band() {
    // The real lazy wrapper under an isolated api id, so the probe never
    // touches the live suite's registration space.
    register_images_api_provider(
        ImagesApiProvider {
            api: "openrouter-images-lazy-probe".to_owned(),
            generate_images: generate_images_open_router(),
        },
        None,
    );
    let mut model = openrouter_image_model();
    model.api = pi_ai::types::ImagesApi::from("openrouter-images-lazy-probe");

    let response = generate_images(&model, &text_input("x"), None)
        .await
        .expect("the wrapper always resolves");
    assert_eq!(response.stop_reason, ImagesStopReason::Error);
    assert!(
        response
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("API key"),
        "the delegated failure surfaced in band: {}",
        response.error_message.as_deref().unwrap_or_default()
    );
}

#[tokio::test]
async fn registry_entries_debug_with_their_api_id() {
    let provider = ImagesApiProvider {
        api: "debug-probe-api".to_owned(),
        generate_images: stub_api("debug-stub"),
    };
    let debug = format!("{provider:?}");
    assert!(debug.contains("debug-probe-api"), "{debug}");
}
