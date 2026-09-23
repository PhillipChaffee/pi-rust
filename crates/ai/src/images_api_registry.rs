//! The image-API registry, ported from
//! `packages/ai/src/images-api-registry.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! A process-wide map from image-API id to the generation implementation,
//! keyed by the model's `api`. [`crate::images::generate_images`] resolves
//! through it; extensions and tests register their own image APIs here.
//!
//! Porting restatement: upstream's `ImagesApiProvider<TApi, TOptions>` and
//! the `ImagesFunction` value shape collapse into one
//! `Arc<dyn ProviderImages>` field, the crate's dispatch idiom; the
//! registered implementation is wrapped so a call with a mismatched
//! `model.api` fails instead of dispatching into untyped code.

use std::sync::{Arc, LazyLock, RwLock};

use crate::types::{
    AssistantImages, BoxedFuture, ImagesContext, ImagesModel, ImagesOptions, ProviderImages,
};
use crate::utils::provider_retry::ProviderRequestError;

/// One registered image-API implementation, upstream's `ImagesApiProvider`.
#[derive(Clone)]
pub struct ImagesApiProvider {
    /// The image-API id the implementation serves, upstream's `api`.
    pub api: String,
    /// The generation implementation, upstream's `generateImages`.
    pub generate_images: Arc<dyn ProviderImages>,
}

impl std::fmt::Debug for ImagesApiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImagesApiProvider")
            .field("api", &self.api)
            .finish_non_exhaustive()
    }
}

/// The registered entry: the mismatch-checked wrapper plus the source id the
/// registration was made under, upstream's `RegisteredImagesApiProvider`.
type RegisteredImagesApiProvider = (Arc<dyn ProviderImages>, Option<String>);

static IMAGES_API_PROVIDER_REGISTRY: LazyLock<
    RwLock<std::collections::BTreeMap<String, RegisteredImagesApiProvider>>,
> = LazyLock::new(|| RwLock::new(std::collections::BTreeMap::new()));

/// The generation dispatch the registry stores, upstream's
/// `wrapGenerateImages`: a mismatched model api fails before the
/// implementation runs, so a registration never sees another API's models.
struct MismatchCheckedImages {
    api: String,
    inner: Arc<dyn ProviderImages>,
}

impl ProviderImages for MismatchCheckedImages {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> BoxedFuture<'a, Result<AssistantImages, ProviderRequestError>> {
        Box::pin(async move {
            if model.api.0 != self.api {
                return Err(ProviderRequestError::new(
                    None,
                    None,
                    format!("Mismatched api: {} expected {}", model.api.0, self.api),
                ));
            }
            self.inner.generate_images(model, context, options).await
        })
    }
}

/// Register an image-API implementation, replacing any prior registration
/// for its api, upstream's `registerImagesApiProvider`.
///
/// `source_id` tags the registration for later
/// [`unregister_images_api_providers`] sweeps.
pub fn register_images_api_provider(provider: ImagesApiProvider, source_id: Option<&str>) {
    IMAGES_API_PROVIDER_REGISTRY
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            provider.api.clone(),
            (
                Arc::new(MismatchCheckedImages {
                    api: provider.api.clone(),
                    inner: provider.generate_images,
                }),
                source_id.map(ToOwned::to_owned),
            ),
        );
}

/// The registered implementation for one image-API id, with the mismatch
/// check upstream's `getImagesApiProvider` return carries.
#[must_use]
pub fn get_images_api_provider(api: &str) -> Option<Arc<dyn ProviderImages>> {
    IMAGES_API_PROVIDER_REGISTRY
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(api)
        .map(|(provider, _)| Arc::clone(provider))
}

/// Remove every registration made under `source_id`, upstream's per-source
/// unregistration pattern (compat's `unregisterApiProviders` shape).
pub fn unregister_images_api_providers(source_id: &str) {
    IMAGES_API_PROVIDER_REGISTRY
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|_, (_, registered_source)| registered_source.as_deref() != Some(source_id));
}
