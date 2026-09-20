//! Live-credential guards for the wire-API suites, ported from
//! `packages/ai/test/azure-utils.ts`, `packages/ai/test/bedrock-utils.ts`,
//! and `packages/ai/test/cloudflare-utils.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream reads `process.env` directly; the port takes the environment as
//! an injectable `&dyn Fn(&str) -> Option<String>` so the suites drive the
//! same guards hermetically. "Set" mirrors the falsy-string semantics
//! upstream's `!!process.env.X` checks lean on: a value must be present and
//! non-empty. These are the guards the Azure, Bedrock, and Cloudflare
//! wire-API children reuse to skip their live-credential cases.

use std::collections::BTreeMap;

/// `!!process.env.X`: present and non-empty, the falsy-string semantics the
/// upstream guards check with.
fn configured(env: &dyn Fn(&str) -> Option<String>, name: &str) -> bool {
    env(name).is_some_and(|value| !value.is_empty())
}

/// Whether the Azure OpenAI live credentials are configured (upstream's
/// `hasAzureOpenAICredentials`): the api key plus a base URL or a resource
/// name.
#[must_use]
pub fn has_azure_openai_credentials(env: &dyn Fn(&str) -> Option<String>) -> bool {
    let has_key = configured(env, "AZURE_OPENAI_API_KEY");
    let has_endpoint =
        configured(env, "AZURE_OPENAI_BASE_URL") || configured(env, "AZURE_OPENAI_RESOURCE_NAME");
    has_key && has_endpoint
}

/// The `modelId=deployment` pairs of `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`,
/// upstream's `parseDeploymentNameMap`: comma-separated entries, each split
/// at its first `=` with both halves trimmed; empty halves and entries
/// without a separator are skipped. Upstream's `split("=", 2)` truncates a
/// value at the second `=`; the port keeps the remainder, splitting only on
/// the first.
fn parse_deployment_name_map(value: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for entry in value.split(',') {
        let trimmed = entry.trim();
        let Some((model_id, deployment_name)) = trimmed.split_once('=') else {
            continue;
        };
        let (model_id, deployment_name) = (model_id.trim(), deployment_name.trim());
        if model_id.is_empty() || deployment_name.is_empty() {
            continue;
        }
        map.insert(model_id.to_owned(), deployment_name.to_owned());
    }
    map
}

/// The Azure deployment name mapped for `model_id` from
/// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`, upstream's `resolveAzureDeploymentName`;
/// `None` when the variable is unset or the model has no entry.
#[must_use]
pub fn resolve_azure_deployment_name(
    model_id: &str,
    env: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    let map_value = env("AZURE_OPENAI_DEPLOYMENT_NAME_MAP")?;
    parse_deployment_name_map(&map_value)
        .get(model_id)
        .cloned()
}

/// Check if any valid AWS credentials are configured for Bedrock. Returns
/// true if any of the following are set:
/// - `AWS_PROFILE` (named profile from ~/.aws/credentials)
/// - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (IAM keys)
/// - `AWS_BEARER_TOKEN_BEDROCK` (Bedrock API key)
#[must_use]
pub fn has_bedrock_credentials(env: &dyn Fn(&str) -> Option<String>) -> bool {
    configured(env, "AWS_PROFILE")
        || (configured(env, "AWS_ACCESS_KEY_ID") && configured(env, "AWS_SECRET_ACCESS_KEY"))
        || configured(env, "AWS_BEARER_TOKEN_BEDROCK")
}

/// Whether the Cloudflare Workers AI live credentials are configured: the
/// api key plus the account id.
#[must_use]
pub fn has_cloudflare_workers_ai_credentials(env: &dyn Fn(&str) -> Option<String>) -> bool {
    configured(env, "CLOUDFLARE_API_KEY") && configured(env, "CLOUDFLARE_ACCOUNT_ID")
}

/// Whether the Cloudflare AI Gateway live credentials are configured: the
/// Workers AI pair (api key and account id) plus the gateway id.
#[must_use]
pub fn has_cloudflare_ai_gateway_credentials(env: &dyn Fn(&str) -> Option<String>) -> bool {
    configured(env, "CLOUDFLARE_API_KEY")
        && configured(env, "CLOUDFLARE_ACCOUNT_ID")
        && configured(env, "CLOUDFLARE_GATEWAY_ID")
}