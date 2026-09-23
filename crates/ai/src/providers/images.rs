//! The builtin image-API registrations, ported from
//! `packages/ai/src/providers/images/register-builtins.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream registers the OpenRouter images provider as a module-load side
//! effect with a lazy dynamic import of the `openrouter-images` API module
//! behind it; Rust modules have no import side effects and the API module is
//! a constructor call, so the registration is an explicit, idempotent seed —
//! [`crate::images::generate_images`] runs it before its first lookup, and
//! extensions call it or register their own providers explicitly. Seeding
//! never clobbers an explicit registration, the no-clobber rule compat's
//! builtin registration records: a test or extension may register an
//! override for a builtin api id before the seed first runs.

use std::sync::{Arc, OnceLock};

use crate::types::{
    AssistantImages, BoxedFuture, ImagesContext, ImagesModel, ImagesOptions, ProviderImages,
};
use crate::utils::provider_retry::ProviderRequestError;

/// The OpenRouter image-generation provider, upstream's
/// `generateImagesOpenRouter`: loads the API implementation once and
/// delegates, converting every failure into an error result, the way the
/// lazy-load catch does.
#[derive(Default)]
struct OpenRouterImagesLazy {
    delegate: OnceLock<Arc<dyn ProviderImages>>,
}

impl ProviderImages for OpenRouterImagesLazy {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> BoxedFuture<'a, Result<AssistantImages, ProviderRequestError>> {
        Box::pin(async move {
            let delegate = self.delegate.get_or_init(crate::api::openrouter_images);
            match delegate.generate_images(model, context, options).await {
                Ok(images) => Ok(images),
                Err(error) => Ok(create_lazy_load_error_images(model, &error.to_string())),
            }
        })
    }
}

/// The error result a failed load or delegation reports, upstream's
/// `createLazyLoadErrorImages`.
fn create_lazy_load_error_images(model: &ImagesModel, message: &str) -> AssistantImages {
    crate::images_models::error_images(model, message)
}

/// The OpenRouter images-API implementation, upstream's
/// `generateImagesOpenRouter`.
#[must_use]
pub fn generate_images_open_router() -> Arc<dyn ProviderImages> {
    Arc::new(OpenRouterImagesLazy::default())
}

/// The builtin image-API registrations, upstream's
/// `registerBuiltInImagesApiProviders`. Idempotent: an api with an existing
/// registration keeps it.
pub fn register_built_in_images_api_providers() {
    static SEEDED: OnceLock<()> = OnceLock::new();
    SEEDED.get_or_init(|| {
        if crate::images_api_registry::get_images_api_provider("openrouter-images").is_none() {
            crate::images_api_registry::register_images_api_provider(
                crate::images_api_registry::ImagesApiProvider {
                    api: "openrouter-images".to_owned(),
                    generate_images: generate_images_open_router(),
                },
                None,
            );
        }
    });
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected shape panics the test by design"
)]
mod lazy_wrapper_tests {
    use std::sync::Arc;

    use super::*;
    use crate::types::{ImagesApi, ImagesProviderId, ImagesStopReason, Modality, ModelCost};
    use crate::utils::provider_retry::ProviderRequestError;

    /// A delegate that always fails, the module-load failure upstream's
    /// catch block covers.
    struct FailingDelegate;

    impl ProviderImages for FailingDelegate {
        fn generate_images<'a>(
            &'a self,
            _model: &'a ImagesModel,
            _context: &'a ImagesContext,
            _options: Option<&'a ImagesOptions>,
        ) -> BoxedFuture<'a, Result<AssistantImages, ProviderRequestError>> {
            Box::pin(async move { Err(ProviderRequestError::new(None, None, "delegate failed")) })
        }
    }

    fn fixture_model() -> ImagesModel {
        ImagesModel {
            id: "fixture".to_owned(),
            name: "fixture".to_owned(),
            api: ImagesApi::from("openrouter-images"),
            provider: ImagesProviderId::from("openrouter"),
            base_url: "http://localhost:0".to_owned(),
            thinking_level_map: None,
            input: vec![Modality::Text],
            cost: ModelCost::default(),
            sampling_params: None,
            headers: None,
            output: vec![Modality::Image],
        }
    }

    #[tokio::test]
    async fn a_failing_delegate_resolves_to_the_error_result() {
        let wrapper = OpenRouterImagesLazy::default();
        wrapper.delegate.get_or_init(|| Arc::new(FailingDelegate));
        let images = wrapper
            .generate_images(&fixture_model(), &ImagesContext::default(), None)
            .await
            .expect("the wrapper always resolves");
        assert_eq!(images.stop_reason, ImagesStopReason::Error);
        assert_eq!(images.error_message.as_deref(), Some("delegate failed"));
    }
}
