//! Request auth for compaction and branch summarization.

use pi_ai::types::ProviderEnv;
use pi_ai::types::ProviderHeaders;

use crate::settings_manager::SettingsStorage;

pub(super) struct RequestAuth {
    pub(super) model: pi_ai::types::Model,
    pub(super) api_key: Option<String>,
    pub(super) headers: Option<ProviderHeaders>,
    pub(super) env: Option<ProviderEnv>,
}

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    pub(super) async fn request_auth(
        &self,
        model: &pi_ai::types::Model,
        required: bool,
    ) -> Option<RequestAuth> {
        let result = self.model_runtime.get_auth_for_model(model, None).await;
        let Ok(Some(result)) = result else {
            return if required {
                None
            } else {
                Some(empty_auth(model))
            };
        };
        if result.auth.api_key.is_none() && result.auth.headers.is_none() {
            return if required {
                None
            } else {
                Some(empty_auth(model))
            };
        }
        let request_model = result.auth.base_url.as_ref().map_or_else(
            || model.clone(),
            |base_url| pi_ai::types::Model {
                base_url: base_url.clone(),
                ..model.clone()
            },
        );
        Some(RequestAuth {
            model: request_model,
            api_key: result.auth.api_key,
            headers: result.auth.headers,
            env: result.env,
        })
    }
}

pub(super) fn empty_auth(model: &pi_ai::types::Model) -> RequestAuth {
    RequestAuth {
        model: model.clone(),
        api_key: None,
        headers: None,
        env: None,
    }
}
