//! Shared fixtures for the compaction suites, mirroring the shapes
//! upstream's `test/harness/compaction.test.ts` builds (the id counter,
//! the usage factory, the message and entry factories, and the faux
//! models).

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use pi_agent_core::harness::session::types::{
    BranchSummaryEntryBody, CompactionEntryBody, CustomEntryBody, Entry, MessageEntry,
};
use pi_agent_core::types::AgentMessage;
use pi_ai::models::{Models, create_models};
use pi_ai::providers::faux::{
    FauxAssistantMessageOptions, FauxModelDefinition, FauxProviderHandle,
    RegisterFauxProviderOptions, faux_assistant_message,
};
use pi_ai::types::{
    Api, Message, ProviderId, TextContent, Usage, UsageCost, UserBlock, UserContent, UserMessage,
};

/// The timestamp every fixture entry and message carries; nothing asserts
/// it, so a fixed value keeps the fixtures deterministic.
pub const NOW: i64 = 1_700_000_000_000;

/// The id counter, upstream's `nextId`/`createId` pair; ids only need
/// uniqueness, and the counter never resets across tests in one binary.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// The faux-provider counter, upstream's `fauxCount`: each faux provider
/// gets a unique id so coexisting fakes route correctly.
static FAUX_COUNT: AtomicU64 = AtomicU64::new(0);

/// The entry id, upstream's `createId`.
pub fn create_id() -> String {
    format!("entry-{}", NEXT_ID.fetch_add(1, Ordering::SeqCst))
}

/// The usage fixture, upstream's `createMockUsage`.
pub fn create_mock_usage(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output + cache_read + cache_write,
        cost: UsageCost::default(),
    }
}

/// The user message fixture, upstream's `createUserMessage`.
pub fn create_user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })]),
        timestamp: NOW,
    }))
}

/// The assistant message fixture, upstream's `createAssistantMessage`.
pub fn create_assistant_message(text: &str, usage: Usage) -> pi_ai::types::AssistantMessage {
    pi_ai::types::AssistantMessage {
        api: Api::from("anthropic-messages"),
        provider: ProviderId::from("anthropic"),
        model: "claude-sonnet-4-5".to_owned(),
        usage,
        ..faux_assistant_message(text, FauxAssistantMessageOptions::default())
    }
}

/// The message entry fixture, upstream's `createMessageEntry`; the
/// sequence shares the id counter like upstream's `seq: nextId`.
pub fn create_message_entry(message: AgentMessage, parent_id: Option<String>) -> Entry {
    Entry::Message {
        id: create_id(),
        parent_id,
        seq: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        timestamp: NOW,
        body: Box::new(MessageEntry {
            message,
            terminate: None,
        }),
    }
}

/// The compaction entry fixture, upstream's `createCompactionEntry` with
/// its optional `retainedTail` and inline `details` arguments factored
/// into one builder.
pub fn create_compaction_entry_with(
    summary: &str,
    parent_id: Option<String>,
    retained_tail: Vec<AgentMessage>,
    details: Option<serde_json::Value>,
) -> Entry {
    Entry::Compaction {
        id: create_id(),
        parent_id,
        seq: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        timestamp: NOW,
        body: CompactionEntryBody {
            summary: summary.to_owned(),
            retained_tail,
            tokens_before: 1234,
            details,
            usage: None,
            from_hook: false,
        },
    }
}

/// The compaction entry fixture, upstream's `createCompactionEntry`.
pub fn create_compaction_entry(summary: &str, parent_id: Option<String>) -> Entry {
    create_compaction_entry_with(summary, parent_id, Vec::new(), None)
}

/// The branch-summary entry fixture with file-operation details, the
/// preparation scenarios' seeding fixture.
pub fn create_branch_summary_entry_with_details(
    parent_id: Option<String>,
    from_id: &str,
    summary: &str,
    details: serde_json::Value,
) -> Entry {
    match create_branch_summary_entry(parent_id, from_id, summary) {
        Entry::BranchSummary {
            id,
            parent_id,
            seq,
            timestamp,
            body,
        } => Entry::BranchSummary {
            id,
            parent_id,
            seq,
            timestamp,
            body: BranchSummaryEntryBody {
                details: Some(details),
                ..body
            },
        },
        _ => unreachable!("create_branch_summary_entry builds a branch-summary entry"),
    }
}

/// The custom entry fixture, upstream's `createCustomEntry`.
pub fn create_custom_entry(custom_type: &str, parent_id: Option<String>) -> Entry {
    Entry::Custom {
        id: create_id(),
        parent_id,
        seq: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        timestamp: NOW,
        body: CustomEntryBody {
            custom_type: custom_type.to_owned(),
            data: None,
        },
    }
}

/// The branch-summary entry fixture, upstream's inline `BranchSummaryEntry`
/// literal in the cut-point edge cases.
pub fn create_branch_summary_entry(
    parent_id: Option<String>,
    from_id: &str,
    summary: &str,
) -> Entry {
    Entry::BranchSummary {
        id: create_id(),
        parent_id,
        seq: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        timestamp: NOW,
        body: BranchSummaryEntryBody {
            from_id: Some(from_id.to_owned()),
            summary: summary.to_owned(),
            details: None,
            usage: None,
            from_hook: false,
        },
    }
}

/// The provider collection the suites run against, upstream's shared
/// `createModels()`; each test builds its own collection where upstream
/// shared one module-level instance.
pub fn test_models() -> Models {
    create_models(None)
}

/// The faux model fixture, upstream's `createFauxModel`: one model, named
/// by its reasoning support, on a uniquely identified faux provider.
pub fn create_faux_model(models: &Models, reasoning: bool, max_tokens: u64) -> FauxProviderHandle {
    let handle = pi_ai::providers::faux::faux_provider(RegisterFauxProviderOptions {
        provider: Some(format!(
            "faux-{}",
            FAUX_COUNT.fetch_add(1, Ordering::SeqCst)
        )),
        models: vec![FauxModelDefinition {
            id: if reasoning {
                "reasoning-model".to_owned()
            } else {
                "non-reasoning-model".to_owned()
            },
            reasoning: Some(reasoning),
            context_window: Some(200_000),
            max_tokens: Some(max_tokens),
            ..FauxModelDefinition::default()
        }],
        ..RegisterFauxProviderOptions::default()
    });
    models.set_provider(Arc::new(handle.provider.clone()));
    handle
}

/// The options the suites' factories record, upstream's
/// `seenOptions` array of the request options each call saw.
pub type SeenOptions = Arc<Mutex<Vec<pi_ai::types::SimpleStreamOptions>>>;

/// The entry's stable id, the `entry.id` reads the tests make.
pub fn entry_id(entry: &Entry) -> String {
    entry.id().to_owned()
}

/// The role string of one agent message, upstream's `.role` reads in the
/// role-list assertions.
pub fn message_role(message: &AgentMessage) -> &str {
    match message {
        AgentMessage::Standard(Message::User(_)) => "user",
        AgentMessage::Standard(Message::Assistant(_)) => "assistant",
        AgentMessage::Standard(Message::ToolResult(_)) => "toolResult",
        AgentMessage::Custom(custom) => &custom.role,
    }
}
