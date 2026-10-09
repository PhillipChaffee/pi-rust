//! Shared fixtures for the compaction/usage suites (#123), restating the
//! entry builders upstream's `compaction.test.ts` keeps module-local at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    Api, AssistantBlock, KnownApi, Message, ProviderId, StopReason, TextContent, Usage, UsageCost,
    UserContent, UserMessage,
};

use pi_coding_agent::session_manager::entries::{
    CompactionEntry, CustomMessageEntry, MessageEntry, ModelChangeEntry, SessionEntry,
    SessionEntryBase, ThinkingLevelChangeEntry,
};

/// The mock usage upstream's `createMockUsage` builds.
#[must_use]
pub const fn mock_usage(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output + cache_read + cache_write,
        cost: UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

/// A user message, upstream's `createUserMessage`.
#[must_use]
pub fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: 0,
    }))
}

/// An assistant message, upstream's `createAssistantMessage`: anthropic /
/// `claude-sonnet-4-5`, one text block, the given usage (default
/// `createMockUsage(100, 50)`).
#[must_use]
pub fn assistant_message(text: &str, usage: Option<Usage>) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(pi_ai::types::AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        model: "claude-sonnet-4-5".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage.unwrap_or_else(|| mock_usage(100, 50, 0, 0)),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }))
}

/// The entry-chain builder upstream's module-level `entryCounter`/`lastId`
/// pair keeps; one per test replaces the `beforeEach` reset.
pub struct EntryChain {
    counter: usize,
    last_id: Option<String>,
}

impl EntryChain {
    /// A fresh chain, upstream's `resetEntryCounter()`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counter: 0,
            last_id: None,
        }
    }

    fn next_id(&mut self) -> String {
        let id = format!("test-id-{}", self.counter);
        self.counter += 1;
        id
    }

    /// A message entry linked to the chain, upstream's `createMessageEntry`.
    #[must_use]
    pub fn message_entry(&mut self, message: AgentMessage) -> SessionEntry {
        let id = self.next_id();
        let entry = SessionEntry::Message(MessageEntry {
            base: SessionEntryBase {
                id: Some(id.clone()),
                parent_id: self.last_id.clone(),
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            message: Some(message),
            extras: serde_json::Map::new(),
        });
        self.last_id = Some(id);
        entry
    }

    /// A compaction entry linked to the chain, upstream's
    /// `createCompactionEntry` (`tokensBefore` 10000).
    #[must_use]
    pub fn compaction_entry(&mut self, summary: &str, first_kept_entry_id: &str) -> SessionEntry {
        let id = self.next_id();
        let entry = SessionEntry::Compaction(CompactionEntry {
            base: SessionEntryBase {
                id: Some(id.clone()),
                parent_id: self.last_id.clone(),
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            summary: summary.to_owned(),
            first_kept_entry_id: Some(first_kept_entry_id.to_owned()),
            tokens_before: 10000,
            details: None,
            usage: None,
            from_hook: None,
            extras: serde_json::Map::new(),
        });
        self.last_id = Some(id);
        entry
    }

    /// A model-change entry linked to the chain, upstream's
    /// `createModelChangeEntry`.
    #[must_use]
    pub fn model_change_entry(&mut self, provider: &str, model_id: &str) -> SessionEntry {
        let id = self.next_id();
        let entry = SessionEntry::ModelChange(ModelChangeEntry {
            base: SessionEntryBase {
                id: Some(id.clone()),
                parent_id: self.last_id.clone(),
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            provider: provider.to_owned(),
            model_id: model_id.to_owned(),
            extras: serde_json::Map::new(),
        });
        self.last_id = Some(id);
        entry
    }

    /// A thinking-level entry linked to the chain, upstream's
    /// `createThinkingLevelEntry`.
    #[must_use]
    pub fn thinking_level_entry(&mut self, thinking_level: &str) -> SessionEntry {
        let id = self.next_id();
        let entry = SessionEntry::ThinkingLevelChange(ThinkingLevelChangeEntry {
            base: SessionEntryBase {
                id: Some(id.clone()),
                parent_id: self.last_id.clone(),
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            thinking_level: thinking_level.to_owned(),
            extras: serde_json::Map::new(),
        });
        self.last_id = Some(id);
        entry
    }

    /// A custom-message entry linked to the chain, upstream's
    /// `createCustomMessageEntry`.
    #[must_use]
    pub fn custom_message_entry(&mut self, content: &str) -> SessionEntry {
        let id = self.next_id();
        let entry = SessionEntry::CustomMessage(CustomMessageEntry {
            base: SessionEntryBase {
                id: Some(id.clone()),
                parent_id: self.last_id.clone(),
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            custom_type: "test".to_owned(),
            content: Some(UserContent::Text(content.to_owned())),
            display: true,
            details: None,
            extras: serde_json::Map::new(),
        });
        self.last_id = Some(id);
        entry
    }
}

impl Default for EntryChain {
    fn default() -> Self {
        Self::new()
    }
}

/// The text one agent message contributes to the transcript, upstream's
/// `extractText`.
#[must_use]
pub fn extract_text(messages: &[AgentMessage]) -> String {
    messages
        .iter()
        .map(|message| match message {
            AgentMessage::Standard(Message::User(user)) => match &user.content {
                UserContent::Text(text) => text.clone(),
                UserContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        pi_ai::types::UserBlock::Text(content) => Some(content.text.as_str()),
                        pi_ai::types::UserBlock::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            },
            AgentMessage::Standard(Message::Assistant(assistant)) => assistant
                .content
                .iter()
                .filter_map(|block| match block {
                    AssistantBlock::Text(content) => Some(content.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
            AgentMessage::Standard(Message::ToolResult(result)) => result
                .content
                .iter()
                .filter_map(|block| match block {
                    pi_ai::types::ToolResultBlock::Text(content) => Some(content.text.as_str()),
                    pi_ai::types::ToolResultBlock::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
            AgentMessage::Custom(custom) => match custom.role.as_str() {
                "branchSummary" | "compactionSummary" => custom
                    .field("summary")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                "custom" => match custom.field("content") {
                    Some(serde_json::Value::String(text)) => text.clone(),
                    _ => String::new(),
                },
                "bashExecution" => {
                    let command = custom
                        .field("command")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default();
                    let output = custom
                        .field("output")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default();
                    format!("{command}\n{output}")
                }
                _ => String::new(),
            },
        })
        .collect::<Vec<_>>()
        .join("\n")
}
