//! The env API-key registry, ported from `packages/ai/src/env-api-keys.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::types::ProviderEnv;
use crate::utils::provider_env::get_provider_env_value;

/// The Anthropic auth-token env var, upstream's `ANTHROPIC_AUTH_TOKEN_ENV`.
pub const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
/// The Anthropic OAuth-token env var, upstream's `ANTHROPIC_OAUTH_TOKEN_ENV`.
pub const ANTHROPIC_OAUTH_TOKEN_ENV: &str = "ANTHROPIC_OAUTH_TOKEN";
/// The Anthropic API-key env var, upstream's `ANTHROPIC_API_KEY_ENV`.
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// The API-key env vars of a provider, upstream's `getApiKeyEnvVars`.
///
/// `ANTHROPIC_AUTH_TOKEN` participates in env discovery/status, but
/// [`get_env_api_key`] skips it because requests must pass it as
/// `Authorization: Bearer`.
#[must_use]
fn get_api_key_env_vars(provider: &str) -> Option<Vec<&'static str>> {
    match provider {
        "github-copilot" => Some(vec!["COPILOT_GITHUB_TOKEN"]),
        "anthropic" => Some(vec![
            ANTHROPIC_AUTH_TOKEN_ENV,
            ANTHROPIC_OAUTH_TOKEN_ENV,
            ANTHROPIC_API_KEY_ENV,
        ]),
        "ant-ling" => Some(vec!["ANT_LING_API_KEY"]),
        "qwen-token-plan" | "qwen-token-plan-individual" => Some(vec!["QWEN_TOKEN_PLAN_API_KEY"]),
        "qwen-token-plan-cn" => Some(vec!["QWEN_TOKEN_PLAN_CN_API_KEY"]),
        "openai" => Some(vec!["OPENAI_API_KEY"]),
        "azure-openai-responses" => Some(vec!["AZURE_OPENAI_API_KEY"]),
        "nvidia" => Some(vec!["NVIDIA_API_KEY"]),
        "deepseek" => Some(vec!["DEEPSEEK_API_KEY"]),
        "google" => Some(vec!["GEMINI_API_KEY"]),
        "google-vertex" => Some(vec!["GOOGLE_CLOUD_API_KEY"]),
        "groq" => Some(vec!["GROQ_API_KEY"]),
        "cerebras" => Some(vec!["CEREBRAS_API_KEY"]),
        "xai" => Some(vec!["XAI_API_KEY"]),
        "radius" => Some(vec!["RADIUS_API_KEY"]),
        "openrouter" => Some(vec!["OPENROUTER_API_KEY"]),
        "vercel-ai-gateway" => Some(vec!["AI_GATEWAY_API_KEY"]),
        "zai" => Some(vec!["ZAI_API_KEY"]),
        "zai-coding-cn" => Some(vec!["ZAI_CODING_CN_API_KEY"]),
        "mistral" => Some(vec!["MISTRAL_API_KEY"]),
        "minimax" => Some(vec!["MINIMAX_API_KEY"]),
        "minimax-cn" => Some(vec!["MINIMAX_CN_API_KEY"]),
        "moonshotai" | "moonshotai-cn" => Some(vec!["MOONSHOT_API_KEY"]),
        "huggingface" => Some(vec!["HF_TOKEN"]),
        "fireworks" => Some(vec!["FIREWORKS_API_KEY"]),
        "together" => Some(vec!["TOGETHER_API_KEY"]),
        "baseten" => Some(vec!["BASETEN_API_KEY"]),
        "opencode" | "opencode-go" => Some(vec!["OPENCODE_API_KEY"]),
        "kimi-coding" => Some(vec!["KIMI_API_KEY"]),
        "cloudflare-workers-ai" | "cloudflare-ai-gateway" => Some(vec!["CLOUDFLARE_API_KEY"]),
        "xiaomi" => Some(vec!["XIAOMI_API_KEY"]),
        "xiaomi-token-plan-cn" => Some(vec!["XIAOMI_TOKEN_PLAN_CN_API_KEY"]),
        "xiaomi-token-plan-ams" => Some(vec!["XIAOMI_TOKEN_PLAN_AMS_API_KEY"]),
        "xiaomi-token-plan-sgp" => Some(vec!["XIAOMI_TOKEN_PLAN_SGP_API_KEY"]),
        _ => None,
    }
}

/// Whether Application Default Credentials exist for Vertex, upstream's
/// `hasVertexAdcCredentials`: the explicit `GOOGLE_APPLICATION_CREDENTIALS`
/// path, else the default ADC path.
fn has_vertex_adc_credentials(env: Option<&ProviderEnv>) -> bool {
    if let Some(explicit) = env
        .and_then(|env| env.get("GOOGLE_APPLICATION_CREDENTIALS"))
        .filter(|value| !value.is_empty())
    {
        return std::fs::metadata(explicit).is_ok();
    }
    let gac_path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env);
    if let Some(path) = gac_path {
        return std::fs::metadata(path).is_ok();
    }
    let default_path =
        home_dir().map(|home| home.join(".config/gcloud/application_default_credentials.json"));
    default_path.is_some_and(|path| std::fs::metadata(path).is_ok())
}

/// The user's home directory from the process environment.
fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Find configured environment variables that can provide an API key for a
/// provider, upstream's `findEnvKeys`.
///
/// This only reports actual API key variables. It intentionally excludes
/// ambient credential sources such as AWS profiles, AWS IAM credentials, and
/// Google Application Default Credentials.
#[must_use]
pub fn find_env_keys(provider: &str, env: Option<&ProviderEnv>) -> Option<Vec<String>> {
    let env_vars = get_api_key_env_vars(provider)?;
    let found: Vec<String> = env_vars
        .into_iter()
        .filter(|env_var| get_provider_env_value(env_var, env).is_some())
        .map(ToOwned::to_owned)
        .collect();
    (!found.is_empty()).then_some(found)
}

/// Get the API key for a provider from its known environment variables, e.g.
/// `OPENAI_API_KEY`, upstream's `getEnvApiKey`.
///
/// Will not return API keys for providers that require OAuth tokens.
/// Vertex supports either an explicit API key or Application Default
/// Credentials (via `gcloud auth application-default login`); Amazon Bedrock
/// supports AWS profiles, IAM keys, bearer tokens, ECS task roles, and web
/// identity tokens.
#[must_use]
pub fn get_env_api_key(provider: &str, env: Option<&ProviderEnv>) -> Option<String> {
    if let Some(env_keys) = find_env_keys(provider, env) {
        let api_key_env = if provider == "anthropic" {
            env_keys
                .iter()
                .find(|key| key.as_str() != ANTHROPIC_AUTH_TOKEN_ENV)
        } else {
            env_keys.first()
        };
        if let Some(api_key_env) = api_key_env {
            return get_provider_env_value(api_key_env, env);
        }
    }

    if provider == "google-vertex" {
        let has_credentials = has_vertex_adc_credentials(env);
        let has_project = get_provider_env_value("GOOGLE_CLOUD_PROJECT", env).is_some()
            || get_provider_env_value("GCLOUD_PROJECT", env).is_some();
        let has_location = get_provider_env_value("GOOGLE_CLOUD_LOCATION", env).is_some();
        if has_credentials && has_project && has_location {
            return Some("<authenticated>".to_owned());
        }
    }

    if provider == "amazon-bedrock" {
        // Amazon Bedrock supports multiple credential sources:
        // 1. AWS_PROFILE - named profile from ~/.aws/credentials
        // 2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY - standard IAM keys
        // 3. AWS_BEARER_TOKEN_BEDROCK - Bedrock bearer token
        // 4. AWS_CONTAINER_CREDENTIALS_RELATIVE_URI - ECS task roles
        // 5. AWS_CONTAINER_CREDENTIALS_FULL_URI - ECS task roles (full URI)
        // 6. AWS_WEB_IDENTITY_TOKEN_FILE - IRSA (IAM Roles for Service Accounts)
        if get_provider_env_value("AWS_PROFILE", env).is_some()
            || (get_provider_env_value("AWS_ACCESS_KEY_ID", env).is_some()
                && get_provider_env_value("AWS_SECRET_ACCESS_KEY", env).is_some())
            || get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env).is_some()
            || get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env).is_some()
            || get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env).is_some()
            || get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env).is_some()
        {
            return Some("<authenticated>".to_owned());
        }
    }

    None
}
