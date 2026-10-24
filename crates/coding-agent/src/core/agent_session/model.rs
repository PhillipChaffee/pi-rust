//! Model selection and cycling.

use pi_ai::models::models_are_equal;
use pi_ai::types::Model;

use super::lock;
use super::types::CycleDirection;
use super::types::ModelCycleResult;
use super::types::ModelMutationOptions;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Set the model directly, validating configured auth.
    ///
    /// # Errors
    /// When no auth is configured for the provider.
    pub async fn set_model(
        &self,
        model: Model,
        options: ModelMutationOptions,
    ) -> Result<(), String> {
        if self
            .model_runtime
            .check_auth(&model.provider.0, None)
            .await
            .map_or(true, |check| check.is_none())
        {
            return Err(format!("No API key for {}/{}", model.provider.0, model.id));
        }
        let thinking_level = self.thinking_level_for_model_switch(Some(&model), None);
        self.agent.set_model(model.clone());
        let _ = lock(&self.session_manager).append_model_change(&model.provider.0, &model.id);
        if options.persist {
            lock(&self.settings_manager)
                .set_default_model_and_provider(&model.provider.0, &model.id);
            self.add_persisted_default_to_non_empty_scope(&model);
        }
        self.set_thinking_level(thinking_level, ModelMutationOptions::default());
        Ok(())
    }

    /// Cycle to the next or previous model.
    pub fn cycle_model(
        &self,
        direction: CycleDirection,
        options: ModelMutationOptions,
    ) -> Option<ModelCycleResult> {
        if lock(&self.scoped_models).is_empty() {
            self.cycle_available_model(direction, options)
        } else {
            self.cycle_scoped_model(direction, options)
        }
    }

    fn cycle_scoped_model(
        &self,
        direction: CycleDirection,
        options: ModelMutationOptions,
    ) -> Option<ModelCycleResult> {
        let available: std::collections::BTreeSet<String> = self
            .model_runtime
            .get_available_snapshot()
            .into_iter()
            .map(|model| format!("{}\0{}", model.provider.0, model.id))
            .collect();
        let scoped: Vec<_> = lock(&self.scoped_models)
            .iter()
            .filter(|entry| {
                available.contains(&format!("{}\0{}", entry.model.provider.0, entry.model.id))
            })
            .cloned()
            .collect();
        if scoped.len() <= 1 {
            return None;
        }
        let current = self.model();
        let mut index = scoped
            .iter()
            .position(|entry| models_are_equal(Some(&entry.model), current.as_ref()))
            .unwrap_or(0);
        index = step_index(index, scoped.len(), direction);
        let next = &scoped[index];
        let thinking_level =
            self.thinking_level_for_model_switch(Some(&next.model), next.thinking_level);
        self.agent.set_model(next.model.clone());
        let _ =
            lock(&self.session_manager).append_model_change(&next.model.provider.0, &next.model.id);
        if options.persist {
            lock(&self.settings_manager)
                .set_default_model_and_provider(&next.model.provider.0, &next.model.id);
            self.add_persisted_default_to_non_empty_scope(&next.model);
        }
        self.set_thinking_level(thinking_level, ModelMutationOptions::default());
        Some(ModelCycleResult {
            model: next.model.clone(),
            thinking_level: self.thinking_level(),
            is_scoped: true,
        })
    }

    fn cycle_available_model(
        &self,
        direction: CycleDirection,
        options: ModelMutationOptions,
    ) -> Option<ModelCycleResult> {
        let models = self.model_runtime.get_available_snapshot();
        if models.len() <= 1 {
            return None;
        }
        let current = self.model();
        let mut index = models
            .iter()
            .position(|model| models_are_equal(Some(model), current.as_ref()))
            .unwrap_or(0);
        index = step_index(index, models.len(), direction);
        let next = models[index].clone();
        let thinking_level = self.thinking_level_for_model_switch(Some(&next), None);
        self.agent.set_model(next.clone());
        let _ = lock(&self.session_manager).append_model_change(&next.provider.0, &next.id);
        if options.persist {
            lock(&self.settings_manager).set_default_model_and_provider(&next.provider.0, &next.id);
            self.add_persisted_default_to_non_empty_scope(&next);
        }
        self.set_thinking_level(thinking_level, ModelMutationOptions::default());
        Some(ModelCycleResult {
            model: next,
            thinking_level: self.thinking_level(),
            is_scoped: false,
        })
    }

    fn add_persisted_default_to_non_empty_scope(&self, model: &Model) {
        let mut scoped = lock(&self.scoped_models);
        if scoped.is_empty()
            || scoped
                .iter()
                .any(|entry| models_are_equal(Some(&entry.model), Some(model)))
        {
            return;
        }
        scoped.push(crate::model_resolver::ScopedModel {
            model: model.clone(),
            thinking_level: None,
        });
        drop(scoped);
        let enabled = lock(&self.settings_manager).get_enabled_models();
        let Some(mut enabled) = enabled.filter(|models| !models.is_empty()) else {
            return;
        };
        let reference = format!("{}/{}", model.provider.0, model.id);
        if enabled
            .iter()
            .any(|pattern| pattern.eq_ignore_ascii_case(&reference))
        {
            return;
        }
        enabled.push(reference);
        lock(&self.settings_manager).set_enabled_models(Some(&enabled));
    }
}

const fn step_index(index: usize, len: usize, direction: CycleDirection) -> usize {
    match direction {
        CycleDirection::Forward => (index + 1) % len,
        CycleDirection::Backward => (index + len - 1) % len,
    }
}
