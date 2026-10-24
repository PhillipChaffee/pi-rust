//! Thinking-level management.

use pi_agent_core::types::ThinkingLevel;
use pi_ai::models::clamp_thinking_level;
use pi_ai::models::get_supported_thinking_levels;
use pi_ai::types::ModelThinkingLevel;

use super::lock;
use super::types::ModelMutationOptions;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Set the thinking level, clamped to the model's supported levels.
    pub fn set_thinking_level(&self, level: ThinkingLevel, options: ModelMutationOptions) {
        let available = self.available_thinking_levels();
        let effective = if available.contains(&level) {
            level
        } else {
            self.clamp_thinking_level(level)
        };
        let previous = self.thinking_level();
        self.agent.set_thinking_level(effective);
        if options.persist {
            lock(&self.settings_manager).set_default_thinking_level(level);
        }
        if effective != previous {
            let _ = lock(&self.session_manager)
                .append_thinking_level_change(&thinking_level_str(effective));
            self.emit(super::types::AgentSessionEvent::ThinkingLevelChanged { level: effective });
        }
    }

    /// Cycle to the next thinking level.
    pub fn cycle_thinking_level(&self, options: ModelMutationOptions) -> Option<ThinkingLevel> {
        if !self.supports_thinking() {
            return None;
        }
        let levels = self.available_thinking_levels();
        let index = levels
            .iter()
            .position(|level| *level == self.thinking_level())
            .unwrap_or(0);
        let next = levels[(index + 1) % levels.len()];
        self.set_thinking_level(next, options);
        Some(next)
    }

    /// The available thinking levels for the current model.
    #[must_use]
    pub fn available_thinking_levels(&self) -> Vec<ThinkingLevel> {
        let Some(model) = self.model() else {
            return crate::defaults::THINKING_LEVEL_OPTIONS.to_vec();
        };
        get_supported_thinking_levels(&model)
            .into_iter()
            .map(from_model_thinking_level)
            .collect()
    }

    /// Whether the current model supports thinking.
    #[must_use]
    pub fn supports_thinking(&self) -> bool {
        self.model().is_some_and(|model| model.reasoning)
    }

    pub(crate) fn thinking_level_for_model_switch(
        &self,
        target: Option<&pi_ai::types::Model>,
        explicit: Option<ThinkingLevel>,
    ) -> ThinkingLevel {
        if let Some(explicit) = explicit {
            return explicit;
        }
        if let Some(model) = target
            && let Some(level) =
                lock(&self.settings_manager).get_model_thinking_level(&model.provider.0, &model.id)
        {
            return level;
        }
        lock(&self.settings_manager)
            .get_default_thinking_level()
            .unwrap_or_else(|| {
                let current = self.thinking_level();
                if current == ThinkingLevel::Off {
                    crate::defaults::DEFAULT_THINKING_LEVEL
                } else {
                    current
                }
            })
    }

    fn clamp_thinking_level(&self, level: ThinkingLevel) -> ThinkingLevel {
        self.model().map_or(ThinkingLevel::Off, |model| {
            from_model_thinking_level(clamp_thinking_level(&model, to_model_thinking_level(level)))
        })
    }
}

const fn to_model_thinking_level(level: ThinkingLevel) -> ModelThinkingLevel {
    match level {
        ThinkingLevel::Off => ModelThinkingLevel::Off,
        ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
        ThinkingLevel::Low => ModelThinkingLevel::Low,
        ThinkingLevel::Medium => ModelThinkingLevel::Medium,
        ThinkingLevel::High => ModelThinkingLevel::High,
        ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
        ThinkingLevel::Max => ModelThinkingLevel::Max,
    }
}

/// Clamp a thinking level to the model's supported levels, the SDK factory's
/// clamp before the session exists.
#[must_use]
#[expect(
    clippy::redundant_pub_crate,
    reason = "the private submodule's item is re-exported at crate scope"
)]
pub(crate) fn clamp_thinking_level_for_model(
    model: &pi_ai::types::Model,
    level: ThinkingLevel,
) -> ThinkingLevel {
    from_model_thinking_level(clamp_thinking_level(model, to_model_thinking_level(level)))
}

const fn from_model_thinking_level(level: ModelThinkingLevel) -> ThinkingLevel {
    match level {
        ModelThinkingLevel::Off => ThinkingLevel::Off,
        ModelThinkingLevel::Minimal => ThinkingLevel::Minimal,
        ModelThinkingLevel::Low => ThinkingLevel::Low,
        ModelThinkingLevel::Medium => ThinkingLevel::Medium,
        ModelThinkingLevel::High => ThinkingLevel::High,
        ModelThinkingLevel::Xhigh => ThinkingLevel::Xhigh,
        ModelThinkingLevel::Max => ThinkingLevel::Max,
    }
}

fn thinking_level_str(level: ThinkingLevel) -> String {
    serde_json::to_value(level)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "off".to_owned())
}
