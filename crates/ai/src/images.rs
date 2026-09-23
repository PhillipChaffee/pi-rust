//! The image-generation dispatch, ported from `packages/ai/src/images.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! One entry point: resolve the registered image-API implementation for the
//! model's `api` and delegate. Upstream seeds the builtin registrations as
//! an import side effect; here the seed runs once before the first lookup
//! and never clobbers an explicit registration (see
//! [`crate::providers::images`]).

use crate::types::{AssistantImages, BoxedFuture, ImagesContext, ImagesModel, ImagesOptions};
use crate::utils::provider_retry::ProviderRequestError;

/// Generate images for the input context through the registered image-API
/// implementation, upstream's `generateImages`.
///
/// Fails when no API provider is registered for the model's api — the port
/// of the throw.
#[must_use]
pub fn generate_images<'a>(
    model: &'a ImagesModel,
    context: &'a ImagesContext,
    options: Option<&'a ImagesOptions>,
) -> BoxedFuture<'a, Result<AssistantImages, ProviderRequestError>> {
    crate::providers::images::register_built_in_images_api_providers();
    let Some(provider) = crate::images_api_registry::get_images_api_provider(&model.api.0) else {
        let api = model.api.0.clone();
        return Box::pin(async move {
            Err(ProviderRequestError::new(
                None,
                None,
                format!("No API provider registered for api: {api}"),
            ))
        });
    };
    Box::pin(async move { provider.generate_images(model, context, options).await })
}
