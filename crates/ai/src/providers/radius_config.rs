//! The Radius gateway config plumbing, ported from
//! `packages/ai/src/providers/radius-config.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::auth::types::OAuthCredentials;
use crate::types::{BoxedFuture, KnownApi, Model, ProviderId};

/// The default Radius gateway, upstream's `DEFAULT_RADIUS_GATEWAY`.
pub const DEFAULT_RADIUS_GATEWAY: &str = "https://radius.pi.dev";

/// One gateway model entry, upstream's `RadiusGatewayModel`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RadiusGatewayModel {
    /// The model id.
    pub id: String,
    /// The display name.
    pub name: String,
    /// Whether the model supports reasoning.
    pub reasoning: bool,
    /// Maps pi thinking levels to provider/model-specific values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<crate::types::ThinkingLevelMap>,
    /// The input modalities.
    pub input: Vec<crate::types::Modality>,
    /// The pricing.
    pub cost: crate::types::ModelCost,
    /// The model's context window in tokens.
    pub context_window: u64,
    /// The maximum output tokens.
    pub max_tokens: u64,
}

/// The gateway config payload, upstream's `RadiusGatewayConfig`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RadiusGatewayConfig {
    /// The base URL requests go through.
    pub base_url: String,
    /// The catalog models.
    pub models: Vec<RadiusGatewayModel>,
}

/// Validate and decode one gateway model entry, upstream's
/// `isRadiusGatewayModel`.
fn is_radius_gateway_model(value: &serde_json::Value) -> bool {
    serde_json::from_value::<RadiusGatewayModel>(value.clone()).is_ok()
}

/// Sanitize a decoded gateway config, upstream's
/// `sanitizeRadiusGatewayConfig`.
#[must_use]
pub fn sanitize_radius_gateway_config(value: &serde_json::Value) -> Option<RadiusGatewayConfig> {
    let object = value.as_object()?;
    let base_url = object.get("baseUrl").and_then(serde_json::Value::as_str)?;
    let models = object.get("models").and_then(serde_json::Value::as_array)?;
    Some(RadiusGatewayConfig {
        base_url: base_url.to_owned(),
        models: models
            .iter()
            .filter(|entry| is_radius_gateway_model(entry))
            .filter_map(|entry| serde_json::from_value(entry.clone()).ok())
            .collect(),
    })
}

/// Normalize a gateway URL, upstream's `normalizeRadiusGatewayUrl`: add the
/// `https://` scheme when missing and strip trailing slashes.
#[must_use]
pub fn normalize_radius_gateway_url(value: &str) -> String {
    let with_scheme = if value.starts_with("http://") || value.starts_with("https://") {
        value.to_owned()
    } else {
        format!("https://{value}")
    };
    with_scheme.trim_end_matches('/').to_owned()
}

/// The gateway config stored with the OAuth credential, upstream's
/// `RadiusOAuthCredential.gatewayConfig`.
#[must_use]
pub fn get_radius_credential_config(
    credential: Option<&OAuthCredentials>,
) -> Option<RadiusGatewayConfig> {
    credential
        .and_then(|credential| credential.extra.get("gatewayConfig"))
        .and_then(sanitize_radius_gateway_config)
}

/// Build the models of a gateway config, upstream's
/// `getRadiusModelsFromConfig`.
#[must_use]
pub fn get_radius_models_from_config(
    provider_id: &str,
    config: &RadiusGatewayConfig,
) -> Vec<Model> {
    config
        .models
        .iter()
        .map(|model| Model {
            id: model.id.clone(),
            name: model.name.clone(),
            api: crate::types::Api::from(KnownApi::PiMessages),
            provider: ProviderId::from(provider_id),
            base_url: config.base_url.clone(),
            reasoning: model.reasoning,
            thinking_level_map: model.thinking_level_map.clone(),
            input: model.input.clone(),
            cost: model.cost.clone(),
            context_window: model.context_window,
            max_tokens: model.max_tokens,
            sampling_params: None,
            headers: None,
            compat: None,
        })
        .collect()
}

/// The models a credential's gateway config provides, upstream's
/// `getRadiusModels`.
#[must_use]
pub fn get_radius_models(provider_id: &str, credential: Option<&OAuthCredentials>) -> Vec<Model> {
    get_radius_credential_config(credential).map_or_else(Vec::new, |config| {
        get_radius_models_from_config(provider_id, &config)
    })
}

/// Truncate a response body for error messages, upstream's
/// `truncateHttpBody`.
fn truncate_http_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() > 512 {
        let truncated: String = trimmed.chars().take(512).collect();
        return format!("{truncated}\u{2026}");
    }
    trimmed.to_owned()
}

/// Load the gateway config over the network, upstream's
/// `loadRadiusGatewayConfig`.
///
/// # Errors
/// A transport failure, a non-2xx response, or an invalid config body.
#[must_use]
pub fn load_radius_gateway_config<'a>(
    gateway: &'a str,
    api_key: Option<&'a str>,
    signal: &'a CancellationToken,
) -> BoxedFuture<'a, Result<RadiusGatewayConfig, GatewayConfigError>> {
    let url = format!("{gateway}/v1/config");
    let mut headers = vec![("accept".to_owned(), "application/json".to_owned())];
    if let Some(api_key) = api_key {
        headers.push(("authorization".to_owned(), format!("Bearer {api_key}")));
    }
    Box::pin(async move {
        let request = crate::http::client::HttpRequest {
            method: crate::http::client::HttpMethod::Get,
            url,
            headers,
            body: None,
            timeout_ms: None,
            signal: signal.clone(),
        };
        let client = crate::http::default_http_client();
        let mut response = client
            .execute(request)
            .await
            .map_err(|error| GatewayConfigError::transport(&error))?;
        let status = response.status;
        let body = read_body(&mut response.body)
            .await
            .map_err(|error| GatewayConfigError::transport(&error))?;
        if !(200..300).contains(&status) {
            return Err(GatewayConfigError {
                message: format!(
                    "Could not load Radius config from {gateway}: {status}: {}",
                    truncate_http_body(&body)
                ),
            });
        }
        let decoded: serde_json::Value =
            serde_json::from_str(&body).map_err(|error| GatewayConfigError {
                message: format!("Invalid Radius config from {gateway}: {error}"),
            })?;
        sanitize_radius_gateway_config(&decoded).ok_or_else(|| GatewayConfigError {
            message: format!("Invalid Radius config from {gateway}"),
        })
    })
}

/// The failure loading a gateway config reports.
#[derive(Debug)]
pub struct GatewayConfigError {
    /// The failure description.
    pub message: String,
}

impl std::fmt::Display for GatewayConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GatewayConfigError {}

impl GatewayConfigError {
    fn transport(error: &crate::http::client::HttpError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

/// Drain the response body into a string.
async fn read_body(
    body: &mut crate::http::client::HttpByteStream,
) -> Result<String, crate::http::client::HttpError> {
    let mut buffer = Vec::new();
    while let Some(chunk) = body.next_chunk().await? {
        buffer.extend_from_slice(&chunk);
    }
    String::from_utf8(buffer)
        .map_err(|error| crate::http::client::HttpError::Transport(error.to_string()))
}
