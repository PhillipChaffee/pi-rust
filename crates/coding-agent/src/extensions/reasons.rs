//! Small shared vocabularies for the extension events, upstream's
//! `src/core/extensions/types.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use serde_json::Value;

/// The run mode a session binds, upstream's `ExtensionMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExtensionMode {
    /// The interactive TUI.
    Tui,
    /// The RPC mode.
    Rpc,
    /// The JSON-event mode.
    Json,
    /// The print mode.
    #[default]
    Print,
}

/// Where user input came from, upstream's `InputSource`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputSource {
    /// The interactive editor.
    #[default]
    Interactive,
    /// An RPC client.
    Rpc,
    /// An extension (`sendUserMessage`).
    Extension,
}

/// One model cycled through with Ctrl+P, upstream's `ScopedModel`.
pub use crate::model_resolver::ScopedModel;

/// Why a session started, upstream's `SessionStartEvent.reason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionStartReason {
    /// First startup.
    Startup,
    /// An existing session resumed.
    Resume,
    /// A fresh session replaced the current one.
    New,
    /// A fork replaced the current one.
    Fork,
    /// Extensions, skills, prompts, themes reloaded in place.
    Reload,
}

/// Why the current session is shutting down, upstream's `SessionShutdownEvent.reason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionShutdownReason {
    /// Replaced by a new session.
    New,
    /// Replaced by a resumed session.
    Resume,
    /// Replaced by a fork.
    Fork,
    /// The app is quitting.
    Quit,
}

/// Why compaction runs, upstream's `"manual" | "threshold" | "overflow"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionReason {
    /// The `/compact` command or a caller.
    Manual,
    /// Context usage crossed the configured threshold.
    Threshold,
    /// A context-overflow error or recoverable length stop.
    Overflow,
}

/// How a model selection moved, upstream's `ModelSelectEvent.source`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSelectSource {
    /// Directly set.
    Set,
    /// Cycled with Ctrl+P.
    Cycle,
    /// Restored from session data.
    Restore,
}

/// Whether a fork positions before or at the selected entry, upstream's
/// `session_before_fork` `position`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForkPosition {
    /// The entry before the selected user message.
    Before,
    /// At the selected entry.
    At,
}

/// How a queued message joins a streaming run, upstream's `"steer" | "followUp"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamingBehavior {
    /// Interrupt after the current tool batch.
    Steer,
    /// Wait for the run to finish.
    FollowUp,
}

/// The tree-navigation preparation the before-tree event carries, upstream's
/// `TreePreparation`.
#[derive(Clone, Debug)]
pub struct TreePreparation {
    /// The entry the navigation targets.
    pub target_id: String,
    /// The leaf the session leaves.
    pub old_leaf_id: Option<String>,
    /// The common ancestor of the old leaf and the target.
    pub common_ancestor_id: Option<String>,
    /// The entries the branch summary would cover.
    pub entries_to_summarize: Vec<crate::session_manager::SessionEntry>,
    /// Whether the user asked for a summary.
    pub user_wants_summary: bool,
    /// The caller's custom instructions.
    pub custom_instructions: Option<String>,
    /// Whether custom instructions replace the default prompt.
    pub replace_instructions: Option<bool>,
    /// The label to attach to the branch summary entry.
    pub label: Option<String>,
}

/// The custom-message payload extensions enqueue, upstream's
/// `Pick<CustomMessage, "customType" | "content" | "display" | "details">`.
#[derive(Clone, Debug)]
pub struct CustomMessagePayload {
    /// The custom type name.
    pub custom_type: String,
    /// The content blocks.
    pub content: Vec<pi_ai::types::UserBlock>,
    /// Whether the message displays in the transcript.
    pub display: bool,
    /// Arbitrary details.
    pub details: Option<Value>,
}

/// The user text block constructor, upstream's `{ type: "text", text }`.
#[must_use]
pub fn text_block(text: impl Into<String>) -> pi_ai::types::UserBlock {
    pi_ai::types::UserBlock::Text(pi_ai::types::TextContent {
        text: text.into(),
        text_signature: None,
    })
}

/// The user image block constructor, upstream's `{ type: "image", ... }`.
#[must_use]
pub const fn image_block(image: pi_ai::types::ImageContent) -> pi_ai::types::UserBlock {
    pi_ai::types::UserBlock::Image(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    use pi_ai::types::ImageContent;

    #[test]
    fn extension_mode_defaults_to_print() {
        assert_eq!(ExtensionMode::default(), ExtensionMode::Print);
        assert_eq!(InputSource::default(), InputSource::Interactive);
        for mode in [
            ExtensionMode::Tui,
            ExtensionMode::Rpc,
            ExtensionMode::Json,
            ExtensionMode::Print,
        ] {
            assert!(!format!("{mode:?}").is_empty());
        }
        for source in [
            InputSource::Interactive,
            InputSource::Rpc,
            InputSource::Extension,
        ] {
            assert!(!format!("{source:?}").is_empty());
        }
    }

    #[test]
    fn every_reason_enum_debugs() {
        for reason in [
            SessionStartReason::Startup,
            SessionStartReason::Resume,
            SessionStartReason::New,
            SessionStartReason::Fork,
            SessionStartReason::Reload,
        ] {
            assert!(!format!("{reason:?}").is_empty());
        }
        for reason in [
            SessionShutdownReason::New,
            SessionShutdownReason::Resume,
            SessionShutdownReason::Fork,
            SessionShutdownReason::Quit,
        ] {
            assert!(!format!("{reason:?}").is_empty());
        }
        for reason in [
            CompactionReason::Manual,
            CompactionReason::Threshold,
            CompactionReason::Overflow,
        ] {
            assert!(!format!("{reason:?}").is_empty());
        }
        for source in [
            ModelSelectSource::Set,
            ModelSelectSource::Cycle,
            ModelSelectSource::Restore,
        ] {
            assert!(!format!("{source:?}").is_empty());
        }
        for position in [ForkPosition::Before, ForkPosition::At] {
            assert!(!format!("{position:?}").is_empty());
        }
        for behavior in [StreamingBehavior::Steer, StreamingBehavior::FollowUp] {
            assert!(!format!("{behavior:?}").is_empty());
        }
    }

    #[test]
    fn payloads_debug_and_block_constructors_build() {
        let preparation = TreePreparation {
            target_id: "e-1".to_string(),
            old_leaf_id: Some("e-0".to_string()),
            common_ancestor_id: None,
            entries_to_summarize: Vec::new(),
            user_wants_summary: true,
            custom_instructions: Some("ci".to_string()),
            replace_instructions: Some(false),
            label: Some("l".to_string()),
        };
        let payload = CustomMessagePayload {
            custom_type: "artifact".to_string(),
            content: vec![
                text_block("hello"),
                image_block(ImageContent {
                    data: "aGk=".to_string(),
                    mime_type: "image/png".to_string(),
                }),
            ],
            display: true,
            details: None,
        };
        assert!(!format!("{preparation:?}").is_empty());
        assert!(!format!("{payload:?}").is_empty());
        assert!(matches!(
            text_block("hello"),
            pi_ai::types::UserBlock::Text(_)
        ));
        assert!(matches!(
            image_block(ImageContent {
                data: "aGk=".to_string(),
                mime_type: "image/png".to_string(),
            }),
            pi_ai::types::UserBlock::Image(_)
        ));
    }
}
