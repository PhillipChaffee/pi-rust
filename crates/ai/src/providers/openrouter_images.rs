//! The OpenRouter image-generation provider factory, ported from
//! `packages/ai/src/providers/openrouter-images.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::{env_api_key_auth, lazy_oauth};
use crate::auth::oauth::load_openrouter_oauth;
use crate::auth::types::ProviderAuth;
use crate::image_models::get_image_models;
use crate::images_models::{
    CreateImagesProviderOptions, ImagesProviderImpl, create_images_provider,
};

/// The OpenRouter image-generation provider, upstream's
/// `openrouterImagesProvider()`.
#[must_use]
pub fn openrouter_images_provider() -> ImagesProviderImpl {
    create_images_provider(CreateImagesProviderOptions {
        id: "openrouter".to_owned(),
        name: Some("OpenRouter".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "OpenRouter API key",
                &["OPENROUTER_API_KEY"],
            )),
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: "OpenRouter OAuth".to_owned(),
                is_subscription: None,
                login_label: Some("Sign in with OpenRouter".to_owned()),
                load: Arc::new(|| load_openrouter_oauth()),
            })),
        },
        models: get_image_models("openrouter"),
        api: crate::api::not_ported_images("openrouter-images"),
        refresh_models: None,
    })
}
