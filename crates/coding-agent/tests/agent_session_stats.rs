//! `test/agent-session-stats.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    clippy::cast_precision_loss,
    reason = "the tests pin outcomes; the context percent restates upstream's token/window arithmetic"
)]

mod common;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Message;
use pi_ai::types::ToolResultBlock;
use pi_ai::types::ToolResultMessage;
use pi_ai::types::Usage;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::session_manager::FileEntry;
use pi_coding_agent::session_manager::SessionEntry;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::usage_totals::UsageCostBreakdownEntry;
use pi_coding_agent::usage_totals::get_usage_cost_breakdown;

use common::agent_session::anthropic_model;
use common::agent_session::assistant_with_usage;
use common::agent_session::build_session;
use common::agent_session::usage;
use common::agent_session::usage_with_cost;
use common::agent_session::user_message;

fn create_usage(total_tokens: u64) -> Usage {
    usage(total_tokens, 0)
}

fn tool_result_message(usage: Usage) -> AgentMessage {
    AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
        tool_call_id: "tool-call-1".to_owned(),
        tool_name: "test_tool".to_owned(),
        content: vec![ToolResultBlock::Text(pi_ai::types::TextContent {
            text: "tool result".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: Some(usage),
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    }))
}

fn cost_usage() -> Usage {
    Usage {
        input: 10,
        output: 20,
        cache_read: 30,
        cache_write: 40,
        total_tokens: 100,
        cost: pi_ai::types::UsageCost {
            input: 0.1,
            output: 0.2,
            cache_read: 0.3,
            cache_write: 0.4,
            total: 1.0,
        },
        ..Usage::default()
    }
}

async fn create_session(
    manager: SessionManager,
) -> std::sync::Arc<AgentSession<InMemorySettingsStorage>> {
    let messages = manager.build_session_context().messages;
    build_session(
        "/tmp/pi-session-stats",
        Settings::new(),
        manager,
        messages,
        None,
    )
    .await
}

fn entries(manager: &SessionManager) -> Vec<SessionEntry> {
    manager
        .entries()
        .into_iter()
        .filter_map(|entry| match entry {
            FileEntry::Entry(entry) => Some(entry.clone()),
            _ => None,
        })
        .collect()
}

fn assert_cost_totals(session: &AgentSession<InMemorySettingsStorage>) {
    let stats = session.session_stats();
    assert_eq!(
        (
            stats.tokens.input,
            stats.tokens.output,
            stats.tokens.cache_read,
            stats.tokens.cache_write,
            stats.tokens.total
        ),
        (10, 20, 30, 40, 100)
    );
    assert!((stats.cost - 1.0).abs() < f64::EPSILON);
}

fn compacted_history(root: &str) -> SessionManager {
    let mut manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    manager
        .append_message(user_message("first", 1))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            assistant_with_usage("response1", create_usage(180_000), 2),
        )))
        .expect("append");
    let kept_user_id = manager
        .append_message(user_message("second", 3))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            assistant_with_usage("response2", create_usage(195_000), 4),
        )))
        .expect("append");
    manager
        .append_compaction("summary", &kept_user_id, 195_000, None, None, None)
        .expect("compaction");
    manager
        .append_message(user_message("third", 5))
        .expect("append");
    manager
}

#[tokio::test]
async fn exposes_the_current_context_usage_alongside_token_totals() {
    let mut manager =
        SessionManager::in_memory(Some("/tmp/pi-session-stats"), None, None).expect("session");
    manager
        .append_message(user_message("hello", 1))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            assistant_with_usage("hi", create_usage(200), 2),
        )))
        .expect("append");
    let session = create_session(manager).await;

    let stats = session.session_stats();
    assert_eq!(stats.context_usage, session.context_usage());
    let context_usage = stats.context_usage.expect("context usage");
    assert_eq!(context_usage.tokens, Some(200));
    assert_eq!(
        context_usage.context_window,
        anthropic_model().context_window
    );
    assert_eq!(
        context_usage.percent,
        Some((200.0 / anthropic_model().context_window as f64) * 100.0)
    );
}

#[tokio::test]
async fn reports_unknown_current_context_usage_immediately_after_compaction() {
    let manager = compacted_history("/tmp/pi-session-stats");
    let session = create_session(manager).await;

    let stats = session.session_stats();
    assert_eq!(stats.tokens.input, 375_000);
    let context_usage = stats.context_usage.expect("context usage");
    assert_eq!(context_usage.tokens, None);
    assert_eq!(context_usage.percent, None);
}

#[tokio::test]
async fn uses_post_compaction_usage_for_current_context_instead_of_stale_kept_usage() {
    let mut manager = compacted_history("/tmp/pi-session-stats");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            assistant_with_usage("response3", create_usage(25_000), 6),
        )))
        .expect("append");
    let session = create_session(manager).await;

    let stats = session.session_stats();
    assert_eq!(stats.tokens.input, 400_000);
    let context_usage = stats.context_usage.expect("context usage");
    assert_eq!(context_usage.tokens, Some(25_000));
    assert_eq!(
        context_usage.percent,
        Some((25_000.0 / anthropic_model().context_window as f64) * 100.0)
    );
}

#[tokio::test]
async fn includes_branch_summary_usage_in_session_totals() {
    let mut manager =
        SessionManager::in_memory(Some("/tmp/pi-session-stats"), None, None).expect("session");
    manager
        .branch_with_summary(None, "summary", None, None, Some(cost_usage()))
        .expect("branch summary");
    let session = create_session(manager).await;

    assert_cost_totals(&session);
}

#[tokio::test]
async fn includes_compaction_usage_in_session_totals() {
    let mut manager =
        SessionManager::in_memory(Some("/tmp/pi-session-stats"), None, None).expect("session");
    let first_kept_entry_id = manager
        .append_message(user_message("hello", 1))
        .expect("append");
    manager
        .append_compaction(
            "summary",
            &first_kept_entry_id,
            100,
            None,
            None,
            Some(cost_usage()),
        )
        .expect("compaction");
    let session = create_session(manager).await;

    assert_cost_totals(&session);
}

#[tokio::test]
async fn includes_tool_result_usage_in_session_totals() {
    let mut manager =
        SessionManager::in_memory(Some("/tmp/pi-session-stats"), None, None).expect("session");
    manager
        .append_message(tool_result_message(cost_usage()))
        .expect("append");
    let session = create_session(manager).await;

    assert_cost_totals(&session);
}

#[test]
fn groups_tool_and_summary_usage_separately_from_model_attributed_usage() {
    let mut manager =
        SessionManager::in_memory(Some("/tmp/pi-session-stats"), None, None).expect("session");
    let root_id = manager
        .append_message(user_message("hello", 1))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            assistant_with_usage("response", usage_with_cost(100, 0.5), 2),
        )))
        .expect("append");
    manager
        .append_message(tool_result_message(usage_with_cost(100, 1.0)))
        .expect("append");
    manager
        .append_compaction(
            "summary",
            &root_id,
            100,
            None,
            None,
            Some(usage_with_cost(100, 2.0)),
        )
        .expect("compaction");
    manager
        .branch_with_summary(
            None,
            "branch summary",
            None,
            None,
            Some(usage_with_cost(100, 3.0)),
        )
        .expect("branch summary");

    assert_eq!(
        get_usage_cost_breakdown(&entries(&manager)),
        vec![
            UsageCostBreakdownEntry {
                key: "Tools/summaries".to_owned(),
                cost: 6.0,
                tokens: 300,
            },
            UsageCostBreakdownEntry {
                key: format!("{}/{}", anthropic_model().provider.0, anthropic_model().id),
                cost: 0.5,
                tokens: 100,
            },
        ]
    );
}

#[tokio::test]
async fn ignores_zero_usage_messages_when_checking_for_post_compaction_context_usage() {
    let mut manager = compacted_history("/tmp/pi-session-stats");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            assistant_with_usage("response3", create_usage(25_000), 6),
        )))
        .expect("append");
    manager
        .append_message(user_message("continue", 7))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            assistant_with_usage("partial", create_usage(0), 8),
        )))
        .expect("append");
    let session = create_session(manager).await;

    let stats = session.session_stats();
    let context_usage = stats.context_usage.expect("context usage");
    let tokens = context_usage.tokens.expect("known tokens");
    assert!(tokens > 25_000);
}
