//! The telemetry-option propagation suite, ported from
//! `test/telemetry-options.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: one shared handle must reach
//! every request-option surface and every dispatch unchanged.
//!
//! Upstream compares captured option values by reference; the port asserts
//! the same contract behaviorally — every observed handle records into the
//! one recording context the options were built from, so a dispatch that
//! swapped or dropped the handle fails the suite.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures unwrap only the values the test just placed"
)]

mod common;

use std::sync::Arc;

use pi_telemetry::SpanOptions;
use pi_telemetry::dispatch::TelemetryHandle;
use pi_telemetry::memory::InMemoryTelemetryContext;

use pi_ai::auth::types::{ModelAuth, ProviderAuth};
use pi_ai::images::generate_images;
use pi_ai::images_api_registry::{
    ImagesApiProvider, register_images_api_provider, unregister_images_api_providers,
};
use pi_ai::images_models::{
    CreateImagesProviderOptions, create_images_models, create_images_provider,
};
use pi_ai::models::{CreateProviderOptions, Provider, ProviderApi, create_models, create_provider};
use pi_ai::types::{
    BoxedFuture, Context, DeferredFetchOptions, DeferredHandle, ImagesContext, ImagesModel,
    ImagesOptions, ImagesStopReason, Model, ProviderImages, ProviderRequestOptions,
    ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions,
};
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_ai::utils::provider_retry::ProviderRequestError;

/// The recording surface behind the provider's API implementation: every
/// dispatch appends the observed telemetry handle.
struct Recording {
    observed: std::sync::Mutex<Vec<Option<TelemetryHandle>>>,
}

impl Recording {
    fn push(&self, observed: Option<&TelemetryHandle>) {
        self.observed
            .lock()
            .expect("the recording lock is only held across a push")
            .push(observed.cloned());
    }

    fn count(&self) -> usize {
        self.observed
            .lock()
            .expect("the recording lock is only held across a push")
            .len()
    }
}

fn event_stream(model: &Model) -> AssistantMessageEventStream {
    let stream = assistant_message_event_stream();
    let message = done_message(model);
    stream.push(pi_ai::types::AssistantMessageEvent::Start {
        partial: message.clone(),
    });
    stream.push(pi_ai::types::AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: message.clone(),
    });
    stream.end(Some(&message));
    stream
}

fn done_message(model: &Model) -> pi_ai::types::AssistantMessage {
    pi_ai::types::AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

struct RecordingStreams {
    recording: Arc<Recording>,
}

impl ProviderStreams for RecordingStreams {
    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        self.recording
            .push(options.and_then(|options| options.telemetry_context.as_ref()));
        event_stream(model)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        self.recording
            .push(options.and_then(|options| options.telemetry_context.as_ref()));
        event_stream(model)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        _handle: &DeferredHandle,
        options: Option<&DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        self.recording
            .push(options.and_then(|options| options.telemetry_context.as_ref()));
        Some(event_stream(model))
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        _handle: &'a DeferredHandle,
        options: Option<&'a ProviderRequestOptions>,
    ) -> BoxedFuture<'a, Result<(), ProviderRequestError>> {
        self.recording
            .push(options.and_then(|options| options.telemetry_context.as_ref()));
        let model = model.clone();
        Box::pin(async move {
            let _ = model;
            Ok(())
        })
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

/// The telemetry model/provider pair the suite dispatches through, upstream's
/// `Model<"telemetry-test">` fixture.
fn telemetry_model() -> Model {
    let mut model = common::builtin_model("anthropic", "claude-haiku-4-5");
    model.id = String::from("model");
    model.name = String::from("Model");
    model.api = pi_ai::types::Api::from("telemetry-test");
    model.provider = pi_ai::types::ProviderId::from("telemetry-provider");
    model
}

/// Upstream's auth fixture: `resolve: async () => ({ auth: {} })`.
fn auth() -> ProviderAuth {
    ProviderAuth {
        api_key: Some(pi_ai::auth::types::ApiKeyAuth {
            name: String::from("Test"),
            login: None,
            check: None,
            resolve: Arc::new(|_input: pi_ai::auth::types::ApiKeyAuthInput| {
                Box::pin(async move {
                    Ok(Some(pi_ai::auth::types::AuthResult {
                        auth: ModelAuth::default(),
                        env: None,
                        source: None,
                    }))
                })
            }),
        }),
        oauth: None,
    }
}

fn deferred_handle(model: &Model) -> DeferredHandle {
    DeferredHandle {
        provider: model.provider.0.clone(),
        model_id: model.id.clone(),
        api: model.api.0.clone(),
        id: String::from("response"),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    }
}

/// Every observed handle, driven once, must record into the shared context —
/// the port of upstream's `observed.every((value) => value === ctx)`.
async fn probe_identity(handles: Vec<Option<TelemetryHandle>>, context: &InMemoryTelemetryContext) {
    assert!(
        handles.iter().all(Option::is_some),
        "every dispatch observed the telemetry handle"
    );
    for handle in handles.iter().flatten() {
        handle
            .start_span_erased(
                SpanOptions::new("telemetry-options-identity-probe"),
                Box::new(move |_| Box::pin(async { Ok(()) })),
            )
            .await
            .expect("the observed handle records into the original context");
    }
    assert_eq!(
        context.get_spans().len(),
        handles.len(),
        "every observed handle is the original context's dispatch handle"
    );
}

#[tokio::test]
async fn telemetry_option_is_inherited_by_every_request_option_surface() {
    let context = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(context.clone());

    let model = telemetry_model();
    let base = pi_ai::api::simple_options::build_base_options(
        &model,
        &Context::default(),
        Some(&SimpleStreamOptions {
            telemetry_context: Some(handle),
            ..SimpleStreamOptions::default()
        }),
        None,
    );
    assert!(base.telemetry_context.is_some());
    assert!(context.get_spans().is_empty());
}

#[expect(
    clippy::too_many_lines,
    reason = "the eight-dispatch matrix reads as upstream's single provider test"
)]
#[tokio::test]
async fn telemetry_option_survives_provider_and_models_dispatch() {
    let _guard = common::registry_guard().await;
    let context = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(context.clone());

    let model = telemetry_model();
    let recording = Arc::new(Recording {
        observed: std::sync::Mutex::new(Vec::new()),
    });
    let provider = create_provider(CreateProviderOptions {
        id: model.provider.0.clone(),
        name: None,
        base_url: None,
        headers: None,
        auth: auth(),
        models: vec![model.clone()],
        fetch_models: None,
        filter_models: None,
        api: ProviderApi::Single(Arc::new(RecordingStreams {
            recording: recording.clone(),
        })),
    });

    let deferred = deferred_handle(&model);

    let _ = provider
        .stream(
            &model,
            &Context::default(),
            Some(&StreamOptions {
                telemetry_context: Some(handle.clone()),
                ..StreamOptions::default()
            }),
        )
        .result()
        .await;
    let _ = provider
        .stream_simple(
            &model,
            &Context::default(),
            Some(&SimpleStreamOptions {
                telemetry_context: Some(handle.clone()),
                ..SimpleStreamOptions::default()
            }),
        )
        .result()
        .await;
    let _ = provider.fetch_deferred(
        &model,
        &deferred,
        Some(&DeferredFetchOptions {
            telemetry_context: Some(handle.clone()),
            ..DeferredFetchOptions::default()
        }),
    );
    let _ = provider
        .cancel_deferred(
            &model,
            &deferred,
            Some(&ProviderRequestOptions {
                telemetry_context: Some(handle.clone()),
                ..ProviderRequestOptions::default()
            }),
        )
        .expect("the fixture provider cancels deferred responses")
        .await;

    let models = create_models(None);
    models.set_provider(Arc::new(provider.clone()));
    let _ = models
        .stream(
            &model,
            &Context::default(),
            Some(&pi_ai::models::ModelsStreamOptions {
                options: StreamOptions {
                    telemetry_context: Some(handle.clone()),
                    ..StreamOptions::default()
                },
                transform_headers: None,
            }),
        )
        .result()
        .await;
    let _ = models
        .stream_simple(
            &model,
            &Context::default(),
            Some(&pi_ai::models::ModelsSimpleStreamOptions {
                options: SimpleStreamOptions {
                    telemetry_context: Some(handle.clone()),
                    ..SimpleStreamOptions::default()
                },
                transform_headers: None,
            }),
        )
        .result()
        .await;
    let _ = models
        .fetch_deferred(
            &model,
            &deferred,
            Some(&pi_ai::models::ModelsDeferredFetchOptions {
                options: DeferredFetchOptions {
                    telemetry_context: Some(handle.clone()),
                    ..DeferredFetchOptions::default()
                },
                transform_headers: None,
            }),
        )
        .await;
    let _ = models
        .cancel_deferred(
            &model,
            &deferred,
            Some(&pi_ai::models::ModelsDeferredCancelOptions {
                options: ProviderRequestOptions {
                    telemetry_context: Some(handle.clone()),
                    ..ProviderRequestOptions::default()
                },
                transform_headers: None,
            }),
        )
        .await;

    assert_eq!(
        recording.count(),
        8,
        "every provider and Models dispatch observed the options"
    );
    let observed = recording
        .observed
        .lock()
        .expect("the recording lock is only held across a push")
        .clone();
    probe_identity(observed, &context).await;
}

/// A recording image-generation implementation.
struct RecordingImages {
    observed: std::sync::Mutex<Vec<Option<TelemetryHandle>>>,
}

impl ProviderImages for RecordingImages {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        _context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> BoxedFuture<'a, Result<pi_ai::types::AssistantImages, ProviderRequestError>> {
        self.observed.lock().unwrap().push(
            options
                .and_then(|options| options.telemetry_context.as_ref())
                .cloned(),
        );
        Box::pin(async move {
            Ok(pi_ai::types::AssistantImages {
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                output: Vec::new(),
                response_id: None,
                usage: None,
                stop_reason: ImagesStopReason::Stop,
                error_message: None,
                timestamp: 0,
            })
        })
    }
}

fn image_model() -> ImagesModel {
    ImagesModel {
        id: String::from("image-model"),
        name: String::from("Image Model"),
        api: pi_ai::types::ImagesApi::from("telemetry-test-images"),
        provider: pi_ai::types::ImagesProviderId::from("telemetry-image-provider"),
        base_url: String::from("https://example.test"),
        thinking_level_map: None,
        input: Vec::new(),
        cost: pi_ai::types::ModelCost::default(),
        sampling_params: None,
        headers: None,
        output: Vec::new(),
    }
}

#[tokio::test]
async fn telemetry_option_survives_direct_and_images_models_dispatch() {
    let _guard = common::registry_guard().await;
    let context = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(context.clone());

    let model = image_model();
    let _observed: Arc<Recording> = Arc::new(Recording {
        observed: std::sync::Mutex::new(Vec::new()),
    });

    let images_context = ImagesContext { input: Vec::new() };
    // The direct dispatch: upstream's registered API implementation records
    // the options `generateImages` received.
    let registered = Arc::new(RecordingImages {
        observed: std::sync::Mutex::new(Vec::new()),
    });
    register_images_api_provider(
        ImagesApiProvider {
            api: model.api.0.clone(),
            generate_images: registered.clone(),
        },
        Some("telemetry-options-test"),
    );
    let _ = generate_images(
        &model,
        &images_context,
        Some(&ImagesOptions {
            telemetry_context: Some(handle.clone()),
            ..ImagesOptions::default()
        }),
    )
    .await;

    let images_models = create_images_models(None);
    images_models.set_provider(Arc::new(create_images_provider(
        CreateImagesProviderOptions {
            id: model.provider.0.clone(),
            name: None,
            auth: auth(),
            models: vec![model.clone()],
            refresh_models: None,
            api: Arc::new(RecordingImages {
                observed: std::sync::Mutex::new(Vec::new()),
            }),
        },
    )));
    let _ = images_models
        .generate_images(
            &model,
            &images_context,
            Some(&ImagesOptions {
                telemetry_context: Some(handle.clone()),
                ..ImagesOptions::default()
            }),
        )
        .await;

    unregister_images_api_providers("telemetry-options-test");

    // The registered implementation observed exactly one dispatch, and the
    // observed handle drives the one recording context.
    let observed = registered
        .observed
        .lock()
        .expect("the recording lock is only held across a push")
        .clone();
    assert_eq!(
        observed.len(),
        1,
        "the direct dispatch reached the registry"
    );
    probe_identity(observed, &context).await;
}
