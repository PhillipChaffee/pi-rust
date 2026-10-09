//! Provider attribution headers, upstream `src/core/provider-attribution.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::BTreeMap;

use pi_ai::types::{Model, ProviderHeaders};
use url::Url;

use crate::config::EnvLookup;
use crate::settings_manager::{SettingsManager, SettingsStorage};
use crate::telemetry::is_install_telemetry_enabled;

const OPENROUTER_HOST: &str = "openrouter.ai";
const NVIDIA_NIM_HOST: &str = "integrate.api.nvidia.com";
const CLOUDFLARE_API_HOST: &str = "api.cloudflare.com";
const CLOUDFLARE_AI_GATEWAY_HOST: &str = "gateway.ai.cloudflare.com";
const OPENCODE_HOST: &str = "opencode.ai";

/// The base URL's hostname, upstream's `new URL(baseUrl).hostname`; an
/// unparseable URL matches nothing.
fn matches_host(base_url: &str, expected_host: &str) -> bool {
    Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| host == expected_host)
}

fn is_openrouter_model(model: &Model) -> bool {
    model.provider.0 == "openrouter" || model.base_url.contains(OPENROUTER_HOST)
}

fn is_nvidia_nim_model(model: &Model) -> bool {
    model.provider.0 == "nvidia" || matches_host(&model.base_url, NVIDIA_NIM_HOST)
}

fn is_cloudflare_model(model: &Model) -> bool {
    model.provider.0 == "cloudflare-workers-ai"
        || model.provider.0 == "cloudflare-ai-gateway"
        || matches_host(&model.base_url, CLOUDFLARE_API_HOST)
        || matches_host(&model.base_url, CLOUDFLARE_AI_GATEWAY_HOST)
}

fn get_default_attribution_headers<S: SettingsStorage>(
    model: &Model,
    settings_manager: &SettingsManager<S>,
    env: &EnvLookup,
) -> Option<ProviderHeaders> {
    if !is_install_telemetry_enabled(settings_manager, env) {
        return None;
    }

    if is_openrouter_model(model) {
        return Some(BTreeMap::from([
            ("HTTP-Referer".to_owned(), Some("https://pi.dev".to_owned())),
            ("X-OpenRouter-Title".to_owned(), Some("pi".to_owned())),
            (
                "X-OpenRouter-Categories".to_owned(),
                Some("cli-agent".to_owned()),
            ),
        ]));
    }

    if is_nvidia_nim_model(model) {
        return Some(BTreeMap::from([(
            "X-BILLING-INVOKE-ORIGIN".to_owned(),
            Some("Pi".to_owned()),
        )]));
    }

    if is_cloudflare_model(model) {
        return Some(BTreeMap::from([(
            "User-Agent".to_owned(),
            Some("pi-coding-agent".to_owned()),
        )]));
    }

    None
}

fn get_session_headers(model: &Model, session_id: Option<&str>) -> Option<ProviderHeaders> {
    let session_id = session_id?;
    if model.provider.0 != "opencode"
        && model.provider.0 != "opencode-go"
        && !matches_host(&model.base_url, OPENCODE_HOST)
    {
        return None;
    }
    Some(BTreeMap::from([
        ("x-opencode-session".to_owned(), Some(session_id.to_owned())),
        ("x-opencode-client".to_owned(), Some("pi".to_owned())),
    ]))
}

/// Merge the session and attribution headers with caller-supplied sources,
/// upstream's `mergeProviderAttributionHeaders`. Later sources win; the
/// result is `None` when nothing merged.
#[must_use]
pub fn merge_provider_attribution_headers<S: SettingsStorage>(
    model: &Model,
    settings_manager: &SettingsManager<S>,
    env: &EnvLookup,
    session_id: Option<&str>,
    header_sources: &[&ProviderHeaders],
) -> Option<ProviderHeaders> {
    let mut merged: ProviderHeaders = BTreeMap::new();
    if let Some(session_headers) = get_session_headers(model, session_id) {
        merged.extend(session_headers);
    }
    if let Some(attribution) = get_default_attribution_headers(model, settings_manager, env) {
        merged.extend(attribution);
    }

    for headers in header_sources {
        for (key, value) in *headers {
            merged.insert(key.clone(), value.clone());
        }
    }

    (!merged.is_empty()).then_some(merged)
}
