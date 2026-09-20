//! The Cloudflare stream wrapper, ported from
//! `packages/ai/src/providers/cloudflare-stream.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::types::{
    Context, Model, ProviderEnv, ProviderStreams, SimpleStreamOptions, StreamOptions,
};
use crate::utils::event_stream::AssistantMessageEventStream;

const CLOUDFLARE_ACCOUNT_ID: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_GATEWAY_ID: &str = "CLOUDFLARE_GATEWAY_ID";

/// Materialize Cloudflare account/gateway endpoint placeholders from the
/// resolved provider env, upstream's `resolveCloudflareModel`. A placeholder
/// without a value stays literal.
#[must_use]
pub fn resolve_cloudflare_model(model: &Model, env: Option<&ProviderEnv>) -> Model {
    let Some(env) = env else {
        return model.clone();
    };
    let account_placeholder = format!("{{{CLOUDFLARE_ACCOUNT_ID}}}");
    let gateway_placeholder = format!("{{{CLOUDFLARE_GATEWAY_ID}}}");
    let base_url = model
        .base_url
        .replace(
            &account_placeholder,
            env.get(CLOUDFLARE_ACCOUNT_ID)
                .map_or_else(|| account_placeholder.clone(), ToString::to_string)
                .as_str(),
        )
        .replace(
            &gateway_placeholder,
            env.get(CLOUDFLARE_GATEWAY_ID)
                .map_or_else(|| gateway_placeholder.clone(), ToString::to_string)
                .as_str(),
        );
    if base_url == model.base_url {
        return model.clone();
    }
    let mut resolved = model.clone();
    resolved.base_url = base_url;
    resolved
}

/// Wrap an API implementation so Cloudflare account/gateway endpoint
/// placeholders materialize from the resolved provider env before dispatch,
/// upstream's `cloudflareStreams`.
#[must_use]
pub fn cloudflare_streams(streams: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
    Arc::new(CloudflareStreams(streams))
}

/// The wrapped streams of [`cloudflare_streams`].
struct CloudflareStreams(Arc<dyn ProviderStreams>);

impl ProviderStreams for CloudflareStreams {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let resolved =
            resolve_cloudflare_model(model, options.and_then(|options| options.env.as_ref()));
        self.0.stream(&resolved, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let resolved =
            resolve_cloudflare_model(model, options.and_then(|options| options.env.as_ref()));
        self.0.stream_simple(&resolved, context, options)
    }
}
