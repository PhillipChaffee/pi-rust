//! A scripted login-interaction double for the OAuth flow suites, the port
//! of the hand-rolled `prompt`/`notify` closures upstream builds in
//! `test/xai-oauth.test.ts` and `test/radius-oauth.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: `prompt` resolves with the
//! scripted reply (or rejects as aborted, the throwing prompt), `notify`
//! records every event, and a device-code event can cancel the flow's signal
//! the way upstream's `onDeviceCode: () => controller.abort()` callback does.

use std::sync::{Arc, Mutex};

use pi_ai::auth::types::{AuthEvent, AuthInteraction, AuthPrompt, ProviderAuthInteraction};
use pi_ai::utils::abort::AbortError;
use tokio_util::sync::CancellationToken;

/// The scripted login interaction: one fixed `prompt` reply and an event
/// sink the suites read during and after a flow.
#[derive(Clone, Debug)]
pub struct ScriptedAuthInteraction {
    /// What `prompt` resolves with, or its rejection.
    prompt_reply: Result<String, AbortError>,
    /// Every notified event, oldest first.
    events: Arc<Mutex<Vec<AuthEvent>>>,
    /// The signal to cancel when a device-code event arrives, the port of
    /// upstream's aborting `onDeviceCode` callback.
    abort_on_device_code: Option<CancellationToken>,
}

impl ScriptedAuthInteraction {
    /// An interaction whose `prompt` resolves with `reply`, upstream's
    /// `prompt: async () => loginMethod`.
    #[must_use]
    pub fn answering(reply: impl Into<String>) -> Self {
        Self::with_prompt_reply(Ok(reply.into()))
    }

    /// An interaction whose `prompt` rejects as aborted, upstream's
    /// `prompt: () => { throw new Error(message); }`.
    #[must_use]
    pub fn rejecting_prompt() -> Self {
        Self::with_prompt_reply(Err(AbortError))
    }

    /// Cancel `signal` when the flow reports its device code, the port of
    /// upstream's `onDeviceCode: () => controller.abort()` callback.
    #[must_use]
    pub fn aborting_on_device_code(mut self, signal: CancellationToken) -> Self {
        self.abort_on_device_code = Some(signal);
        self
    }

    fn with_prompt_reply(reply: Result<String, AbortError>) -> Self {
        Self {
            prompt_reply: reply,
            events: Arc::new(Mutex::new(Vec::new())),
            abort_on_device_code: None,
        }
    }

    /// The notified events so far, oldest first.
    #[must_use]
    pub fn events(&self) -> Vec<AuthEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The scripted double as the merged core's [`AuthInteraction`]: the
    /// prompt closure resolves with the scripted reply, the notify closure
    /// records and can abort.
    #[must_use]
    pub fn interaction(&self) -> AuthInteraction {
        let prompt: pi_ai::auth::types::PromptFn = {
            let scripted = self.clone();
            Arc::new(move |_prompt: AuthPrompt| {
                let reply = scripted.prompt_reply.clone();
                Box::pin(async move { reply })
            })
        };
        let notify = {
            let scripted = self.clone();
            Arc::new(move |event: AuthEvent| {
                if let AuthEvent::DeviceCode { .. } = &event
                    && let Some(signal) = &scripted.abort_on_device_code
                {
                    signal.cancel();
                }
                scripted
                    .events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(event);
            })
        };
        AuthInteraction {
            signal: None,
            prompt,
            notify,
        }
    }
}

/// Wrap the scripted double as the interaction argument the flows take, the
/// port of upstream's `{ signal, prompt, notify }` object.
#[must_use]
pub fn provider_interaction(
    scripted: &ScriptedAuthInteraction,
    signal: CancellationToken,
) -> ProviderAuthInteraction {
    ProviderAuthInteraction::from_interaction(scripted.interaction(), signal)
}
