//! The provider factory belt the per-provider factory files share, ported
//! from upstream's `packages/ai/src/providers/*.ts` factories at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's factory files repeat one `createProvider` shape per provider;
//! only the id, display name, base URL, credential prose, environment keys,
//! and wire-API seam vary. The macro keeps that variation at the call site
//! and the plumbing single-sourced.

/// Generate an env-key provider factory, upstream's per-provider
/// `xxxProvider()`: a `createProvider` whose credential resolves the named
/// environment keys and whose models come from the builtin catalog.
macro_rules! env_key_provider {
    (
        $(#[$doc:meta])*
        $fn_name:ident, $id:expr, $name:expr, $base_url:expr,
        $credential:expr, [$($env_key:expr),+ $(,)?], $api:expr $(,)?
    ) => {
        $(#[$doc])*
        #[must_use]
        pub fn $fn_name() -> std::sync::Arc<dyn crate::models::Provider> {
            let base_url: Option<&str> = $base_url;
            std::sync::Arc::new(crate::models::create_provider(
                crate::models::CreateProviderOptions {
                    id: $id.to_owned(),
                    name: Some($name.to_owned()),
                    base_url: base_url.map(str::to_owned),
                    auth: crate::auth::types::ProviderAuth {
                        api_key: Some(crate::auth::helpers::env_api_key_auth(
                            $credential,
                            &[$($env_key),+],
                        )),
                        oauth: None,
                    },
                    models: crate::providers::catalog::get_builtin_models($id),
                    api: $api,
                    headers: None,
                    fetch_models: None,
                    filter_models: None,
                },
            ))
        }
    };
}

pub(crate) use env_key_provider;
