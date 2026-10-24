//! Queueing user input and expanding skill commands.

use pi_ai::types::ImageContent;

use super::helpers::user_message;
use super::lock;
use super::types::AgentSessionEvent;
use crate::extensions::reasons::StreamingBehavior;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Queue a steering message while the agent runs.
    ///
    /// # Errors
    /// Slice 1 refuses nothing here; the extension-command refusal lands
    /// with the extension slice.
    pub async fn steer(
        &self,
        text: String,
        images: Option<Vec<ImageContent>>,
        source: Option<crate::extensions::reasons::InputSource>,
    ) -> Result<(), String> {
        self.queue_user_input(text, images, source, StreamingBehavior::Steer)
            .await
    }

    /// Queue a follow-up message for after the run settles.
    ///
    /// # Errors
    /// When the text names an extension command, which must run through
    /// `prompt()`.
    pub async fn follow_up(
        &self,
        text: String,
        images: Option<Vec<ImageContent>>,
        source: Option<crate::extensions::reasons::InputSource>,
    ) -> Result<(), String> {
        self.queue_user_input(text, images, source, StreamingBehavior::FollowUp)
            .await
    }

    async fn queue_user_input(
        &self,
        text: String,
        images: Option<Vec<ImageContent>>,
        source: Option<crate::extensions::reasons::InputSource>,
        behavior: StreamingBehavior,
    ) -> Result<(), String> {
        if text.starts_with('/') {
            self.throw_if_extension_command(&text)?;
        }
        let Some((text, images)) = self
            .run_input_handlers(
                text,
                images,
                source.unwrap_or_default(),
                if self.is_streaming() {
                    Some(behavior)
                } else {
                    None
                },
            )
            .await
        else {
            return Ok(());
        };
        let text = self.expand_skill_command(&text);
        let text = crate::prompt_templates::expand_prompt_template(&text, &self.prompt_templates());
        if behavior == StreamingBehavior::Steer {
            self.queue_steer(text, images);
        } else {
            self.queue_follow_up(text, images);
        }
        Ok(())
    }

    pub(crate) fn queue_steer(&self, text: String, images: Option<Vec<ImageContent>>) {
        lock(&self.steering_messages).push(text.clone());
        self.emit_queue_update();
        self.agent.steer(user_message(text, images));
    }

    pub(crate) fn queue_follow_up(&self, text: String, images: Option<Vec<ImageContent>>) {
        lock(&self.follow_up_messages).push(text.clone());
        self.emit(AgentSessionEvent::QueueUpdate {
            steering: lock(&self.steering_messages).clone(),
            follow_up: lock(&self.follow_up_messages).clone(),
        });
        self.agent.follow_up(user_message(text, images));
    }

    pub(crate) fn expand_skill_command(&self, text: &str) -> String {
        let Some(rest) = text.strip_prefix("/skill:") else {
            return text.to_owned();
        };
        let (skill_name, args) = rest
            .split_once(' ')
            .map_or((rest, ""), |(name, args)| (name, args.trim()));
        let skills = self.with_loader(|loader| loader.get_skills().skills);
        let Some(skill) = skills.iter().find(|skill| skill.name == skill_name) else {
            return text.to_owned();
        };
        std::fs::read_to_string(&skill.file_path).map_or_else(
            |_| text.to_owned(),
            |content| {
                let body = crate::utils::frontmatter::strip_frontmatter(&content)
                    .unwrap_or_else(|_| content.clone())
                    .trim()
                    .to_owned();
                let block = format!(
                    "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
                    skill.name, skill.file_path, skill.base_dir, body
                );
                if args.is_empty() {
                    block
                } else {
                    format!("{block}\n\n{args}")
                }
            },
        )
    }
}

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    pub(crate) fn sync_queue_modes_from_settings(&self) {
        let settings = lock(&self.settings_manager);
        self.agent
            .set_steering_mode(agent_queue_mode(settings.get_steering_mode()));
        self.agent
            .set_follow_up_mode(agent_queue_mode(settings.get_follow_up_mode()));
    }

    /// The current steering queue mode.
    #[must_use]
    pub fn steering_mode(&self) -> pi_agent_core::types::QueueMode {
        self.agent.steering_mode()
    }

    /// The current follow-up queue mode.
    #[must_use]
    pub fn follow_up_mode(&self) -> pi_agent_core::types::QueueMode {
        self.agent.follow_up_mode()
    }

    /// Set the steering mode and persist it.
    pub fn set_steering_mode(&self, mode: pi_agent_core::types::QueueMode) {
        self.agent.set_steering_mode(mode);
        lock(&self.settings_manager).set_steering_mode(settings_queue_mode(mode));
    }

    /// Set the follow-up mode and persist it.
    pub fn set_follow_up_mode(&self, mode: pi_agent_core::types::QueueMode) {
        self.agent.set_follow_up_mode(mode);
        lock(&self.settings_manager).set_follow_up_mode(settings_queue_mode(mode));
    }
}

const fn agent_queue_mode(
    mode: crate::settings_manager::QueueMode,
) -> pi_agent_core::types::QueueMode {
    match mode {
        crate::settings_manager::QueueMode::All => pi_agent_core::types::QueueMode::All,
        crate::settings_manager::QueueMode::OneAtATime => {
            pi_agent_core::types::QueueMode::OneAtATime
        }
    }
}

const fn settings_queue_mode(
    mode: pi_agent_core::types::QueueMode,
) -> crate::settings_manager::QueueMode {
    match mode {
        pi_agent_core::types::QueueMode::All => crate::settings_manager::QueueMode::All,
        pi_agent_core::types::QueueMode::OneAtATime => {
            crate::settings_manager::QueueMode::OneAtATime
        }
    }
}
