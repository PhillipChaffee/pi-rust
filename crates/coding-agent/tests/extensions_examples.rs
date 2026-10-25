//! The example-extension suites, upstream's five example-as-test files at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: `plan-mode-extension`,
//! `git-merge-and-resolve-extension`, `input-transform-streaming-example`,
//! `trigger-compact-extension`, and `compaction-extensions-example`.
//!
//! Each example's TypeScript body restates as an inline Rust factory loaded
//! through the `tests/common/extensions` stand-in. The port ships no
//! `examples/` tree, so the body lives beside its assertions. Restatements
//! recorded at the call sites: upstream's `pi.exec` fake is an injected
//! closure (the production exec surface rides its shell consumer), plan-mode's
//! UI is the `TestUi` double downcast from `ctx.ui()` (the typed UI rides
//! #132), `pi.registerShortcut` is absent (keybindings ride #132), and
//! custom-compaction's model completion is an injected closure (upstream's
//! `modelRegistry.complete` mock).
//!
//! The plan-mode handlers upstream registers but its suite never drives
//! (`tool_call`, `context`, `before_agent_start`, `turn_end`, `session_start`)
//! are not restated: the suite's subjects are the active-tool swap, the
//! `agent_end` plan detection, and the refine/execute follow-ups.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::too_many_lines,
    reason = "the example bodies mirror upstream's single-file examples"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin event shapes; a mismatch panics by design"
)]
#![expect(
    clippy::significant_drop_tightening,
    reason = "the assertions read captured state through guards that span the assertion block"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "the context-usage percent is a UI-facing estimate, matching upstream's division"
)]

mod common;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Message, UserContent};
use pi_coding_agent::extensions::api::{ExtensionApi, ExtensionFactory};
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::reasons::{CustomMessagePayload, text_block};
use pi_coding_agent::extensions::reasons::{InputSource, StreamingBehavior};
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::results::InputResult;
use pi_coding_agent::extensions::runner::{DeliverAs, SendMessageOptions, SendUserMessageOptions};
use pi_coding_agent::extensions::runtime::{
    CommandHandler, ContextActions, FlagType, FlagValue, HandlerContext,
};
use regex::Regex;
use serde_json::json;

use common::extensions_examples::{ActionLog, ExampleHarness, TestUi, command, harness, test_ui};

fn context_actions() -> ContextActions {
    ContextActions::default()
}

async fn example_registry() -> pi_coding_agent::model_registry::ModelRegistry {
    common::model_layer::create_in_memory_model_registry(common::model_layer::empty_auth_storage())
        .await
}

fn assistant_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(
        common::agent_session::assistant_message(text, 0),
    ))
}

// =============================================================================
// plan-mode example
// =============================================================================

const PLAN_MODE_TOOLS: [&str; 6] = ["read", "bash", "grep", "find", "ls", "questionnaire"];
const NORMAL_MODE_TOOLS: [&str; 4] = ["read", "bash", "edit", "write"];

#[derive(Clone, Debug, PartialEq, Eq)]
struct TodoItem {
    step: usize,
    text: String,
    completed: bool,
}

#[derive(Default)]
struct PlanState {
    enabled: Mutex<bool>,
    execution: Mutex<bool>,
    todos: Mutex<Vec<TodoItem>>,
    tools_before_plan_mode: Mutex<Option<Vec<String>>>,
}

fn suite_ui(ctx: &HandlerContext) -> Arc<TestUi> {
    test_ui(
        &ctx.ui()
            .expect("the runtime is active")
            .expect("the suite binds a UI"),
    )
}

fn unique(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}

/// Upstream's `getPlanModeTools`: the active tools minus the write pair,
/// then the plan set appended.
fn plan_mode_tools(active: &[String]) -> Vec<String> {
    unique(
        active
            .iter()
            .filter(|name| !matches!(name.as_str(), "edit" | "write"))
            .cloned()
            .chain(PLAN_MODE_TOOLS.iter().map(|name| (*name).to_string())),
    )
}

/// Upstream's `getNormalModeTools`: the normal set, then every active tool
/// plan mode does not manage.
fn normal_mode_tools(active: &[String]) -> Vec<String> {
    let managed: HashSet<&str> = PLAN_MODE_TOOLS
        .iter()
        .chain(NORMAL_MODE_TOOLS.iter())
        .copied()
        .collect();
    unique(
        NORMAL_MODE_TOOLS
            .iter()
            .map(|name| (*name).to_string())
            .chain(
                active
                    .iter()
                    .filter(|name| !managed.contains(name.as_str()))
                    .cloned(),
            ),
    )
}

/// Upstream's `cleanStepText`.
fn clean_step_text(text: &str) -> String {
    let bold = Regex::new(r"\*{1,2}([^*]+)\*{1,2}").expect("valid regex");
    let code = Regex::new(r"`([^`]+)`").expect("valid regex");
    let verb = Regex::new(
        r"(?i)^(Use|Run|Execute|Create|Write|Read|Check|Verify|Update|Modify|Add|Remove|Delete|Install)\s+(the\s+)?",
    )
    .expect("valid regex");
    let spaces = Regex::new(r"\s+").expect("valid regex");
    let mut cleaned = bold.replace_all(text, "$1").into_owned();
    cleaned = code.replace_all(&cleaned, "$1").into_owned();
    cleaned = verb.replace(&cleaned, "").into_owned();
    cleaned = spaces.replace_all(&cleaned, " ").trim().to_string();
    if let Some(first) = cleaned.chars().next() {
        let rest: String = cleaned.chars().skip(1).collect();
        cleaned = format!("{}{rest}", first.to_uppercase());
    }
    if cleaned.chars().count() > 50 {
        cleaned = format!("{}...", cleaned.chars().take(47).collect::<String>());
    }
    cleaned
}

/// Upstream's `extractTodoItems`.
fn extract_todo_items(message: &str) -> Vec<TodoItem> {
    let header = Regex::new(r"(?i)\*{0,2}Plan:\*{0,2}\s*\n").expect("valid regex");
    let Some(header_match) = header.find(message) else {
        return Vec::new();
    };
    let plan_section = &message[header_match.end()..];
    let numbered = Regex::new(r"(?m)^\s*(\d+)[.)]\s+\*{0,2}([^*\n]+)").expect("valid regex");
    let trailing = Regex::new(r"\*{1,2}$").expect("valid regex");
    let mut items = Vec::new();
    for captured in numbered.captures_iter(plan_section) {
        let text = trailing
            .replace(captured.get(2).expect("the text group").as_str().trim(), "")
            .trim()
            .to_string();
        if text.chars().count() <= 5
            || text.starts_with('`')
            || text.starts_with('/')
            || text.starts_with('-')
        {
            continue;
        }
        let cleaned = clean_step_text(&text);
        if cleaned.chars().count() > 3 {
            items.push(TodoItem {
                step: items.len() + 1,
                text: cleaned,
                completed: false,
            });
        }
    }
    items
}

/// Upstream's `updateStatus`.
fn update_status(ctx: &HandlerContext, state: &PlanState) {
    let ui = suite_ui(ctx);
    let execution = *state
        .execution
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let todos = state
        .todos
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let enabled = *state
        .enabled
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if execution && !todos.is_empty() {
        let completed = todos.iter().filter(|item| item.completed).count();
        let status = ui.fg("accent", &format!("📋 {completed}/{}", todos.len()));
        ui.set_status("plan-mode", Some(&status));
        let lines = todos
            .iter()
            .map(|item| {
                if item.completed {
                    format!(
                        "{}{}",
                        ui.fg("success", "☑ "),
                        ui.fg("muted", &ui.strikethrough(&item.text))
                    )
                } else {
                    format!("{}{}", ui.fg("muted", "☐ "), item.text)
                }
            })
            .collect();
        ui.set_widget("plan-todos", Some(lines));
    } else if enabled {
        let status = ui.fg("warning", "⏸ plan");
        ui.set_status("plan-mode", Some(&status));
        ui.set_widget("plan-todos", None);
    } else {
        ui.set_status("plan-mode", None);
        ui.set_widget("plan-todos", None);
    }
}

/// Upstream's `persistState`.
fn persist_state(api: &ExtensionApi, state: &PlanState) -> Result<(), String> {
    let enabled = *state
        .enabled
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let execution = *state
        .execution
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let todos = state
        .todos
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let before = state
        .tools_before_plan_mode
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    api.append_entry(
        "plan-mode".to_string(),
        Some(json!({
            "enabled": enabled,
            "todos": todos.iter().map(|item| json!({
                "step": item.step,
                "text": item.text,
                "completed": item.completed,
            })).collect::<Vec<_>>(),
            "executing": execution,
            "toolsBeforePlanMode": before,
        })),
    )
}

/// Upstream's `togglePlanMode`.
fn toggle_plan_mode(
    api: &ExtensionApi,
    ctx: &HandlerContext,
    state: &PlanState,
) -> Result<(), String> {
    let enabled = {
        let mut enabled = state
            .enabled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *enabled = !*enabled;
        *enabled
    };
    *state
        .execution
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
    state
        .todos
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    let ui = suite_ui(ctx);
    if enabled {
        let before = {
            let mut before = state
                .tools_before_plan_mode
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if before.is_none() {
                *before = Some(api.get_active_tools()?);
            }
            before.clone().unwrap_or_default()
        };
        api.set_active_tools(plan_mode_tools(&before))?;
        ui.notify(
            "Plan mode enabled. Built-in write tools disabled.",
            Some("info"),
        );
    } else {
        let before = state
            .tools_before_plan_mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let tools = before
            .unwrap_or_else(|| normal_mode_tools(&api.get_active_tools().unwrap_or_default()));
        api.set_active_tools(tools)?;
        ui.notify("Plan mode disabled. Full access restored.", Some("info"));
    }
    update_status(ctx, state);
    persist_state(api, state)
}

/// The plan-mode example body: the flag, the `plan`/`todos` commands, and
/// the `agent_end` completion prompt.
fn plan_mode_factory() -> ExtensionFactory {
    Arc::new(|api: Arc<ExtensionApi>| {
        Box::pin(async move {
            let state = Arc::new(PlanState::default());
            api.register_flag(
                "plan".to_string(),
                Some("Start in plan mode (read-only exploration)".to_string()),
                FlagType::Boolean,
                Some(FlagValue::Bool(false)),
            )?;
            let toggle: CommandHandler = {
                let api = Arc::clone(&api);
                let state = Arc::clone(&state);
                Arc::new(move |_args: String, ctx: HandlerContext| {
                    let api = Arc::clone(&api);
                    let state = Arc::clone(&state);
                    Box::pin(async move {
                        toggle_plan_mode(&api, &ctx, &state).expect("the plan toggle succeeds");
                    })
                })
            };
            api.register_command(
                "plan".to_string(),
                Some("Toggle plan mode (read-only exploration)".to_string()),
                toggle,
            )?;
            let todos_command: CommandHandler = {
                let state = Arc::clone(&state);
                Arc::new(move |_args: String, ctx: HandlerContext| {
                    let state = Arc::clone(&state);
                    Box::pin(async move {
                        let todos = state
                            .todos
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone();
                        let ui = suite_ui(&ctx);
                        if todos.is_empty() {
                            ui.notify("No todos. Create a plan first with /plan", Some("info"));
                            return;
                        }
                        let list = todos
                            .iter()
                            .enumerate()
                            .map(|(index, item)| {
                                format!(
                                    "{}. {} {}",
                                    index + 1,
                                    if item.completed { "✓" } else { "○" },
                                    item.text
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        ui.notify(&format!("Plan Progress:\n{list}"), Some("info"));
                    })
                })
            };
            api.register_command(
                "todos".to_string(),
                Some("Show current plan todo list".to_string()),
                todos_command,
            )?;
            let handler_api = Arc::clone(&api);
            api.on(
                "agent_end",
                Arc::new(move |event: ExtensionEvent, ctx: HandlerContext| {
                    let api = Arc::clone(&handler_api);
                    let state = Arc::clone(&state);
                    Box::pin(async move {
                        let ExtensionEvent::AgentEnd { messages } = event else {
                            return HandlerResult::None;
                        };
                        let execution = *state
                            .execution
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        let todos_empty = state
                            .todos
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .is_empty();
                        if execution && !todos_empty {
                            let all_done = state
                                .todos
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .iter()
                                .all(|item| item.completed);
                            if all_done {
                                let list = state
                                    .todos
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .iter()
                                    .map(|item| format!("~~{}~~", item.text))
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                api.send_message(
                                    CustomMessagePayload {
                                        custom_type: "plan-complete".to_string(),
                                        content: vec![text_block(format!(
                                            "**Plan Complete!** ✓\n\n{list}"
                                        ))],
                                        display: true,
                                        details: None,
                                    },
                                    Some(SendMessageOptions {
                                        trigger_turn: Some(false),
                                        deliver_as: None,
                                    }),
                                )
                                .expect("the completion message sends");
                                *state
                                    .execution
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
                                state
                                    .todos
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .clear();
                                update_status(&ctx, &state);
                                persist_state(&api, &state).expect("the cleared state persists");
                            }
                            return HandlerResult::None;
                        }
                        let enabled = *state
                            .enabled
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if !enabled || !ctx.has_ui().unwrap_or(false) {
                            return HandlerResult::None;
                        }
                        let last = messages.iter().rev().find_map(|message| match message {
                            AgentMessage::Standard(Message::Assistant(assistant)) => Some(
                                assistant
                                    .content
                                    .iter()
                                    .filter_map(|block| match block {
                                        pi_ai::types::AssistantBlock::Text(text) => {
                                            Some(text.text.clone())
                                        }
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            ),
                            _ => None,
                        });
                        if let Some(text) = last {
                            let extracted = extract_todo_items(&text);
                            if !extracted.is_empty() {
                                *state
                                    .todos
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) = extracted;
                            }
                        }
                        let todos = state
                            .todos
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone();
                        if todos.is_empty() {
                            return HandlerResult::None;
                        }
                        persist_state(&api, &state).expect("the plan state persists");
                        let todo_list_text = todos
                            .iter()
                            .enumerate()
                            .map(|(index, item)| format!("{}. ☐ {}", index + 1, item.text))
                            .collect::<Vec<_>>()
                            .join("\n");
                        let plan_todo_list = CustomMessagePayload {
                            custom_type: "plan-todo-list".to_string(),
                            content: vec![text_block(format!(
                                "**Plan Steps ({}):**\n\n{todo_list_text}",
                                todos.len()
                            ))],
                            display: true,
                            details: None,
                        };
                        let ui = suite_ui(&ctx);
                        let choice = ui.select(
                            "Plan mode - what next?",
                            &[
                                "Execute the plan (track progress)",
                                "Stay in plan mode",
                                "Refine the plan",
                            ],
                        );
                        if let Some(choice) = choice {
                            if choice.starts_with("Execute") {
                                let first = todos[0].text.clone();
                                *state
                                    .enabled
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
                                *state
                                    .execution
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
                                let before = state
                                    .tools_before_plan_mode
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .take();
                                let tools = before.unwrap_or_else(|| {
                                    normal_mode_tools(&api.get_active_tools().unwrap_or_default())
                                });
                                api.set_active_tools(tools)
                                    .expect("the normal tools restore");
                                update_status(&ctx, &state);
                                persist_state(&api, &state).expect("the execution state persists");
                                let remaining = todos
                                    .iter()
                                    .map(|item| format!("{}. {}", item.step, item.text))
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                let exec_message = format!(
                                    "Execute the plan.\n\nRemaining steps:\n{remaining}\n\nStart \
                                     with: {first}\nAfter completing a step, include a [DONE:n] \
                                     tag in your response."
                                );
                                api.send_message(
                                    plan_todo_list,
                                    Some(SendMessageOptions {
                                        trigger_turn: None,
                                        deliver_as: Some(DeliverAs::FollowUp),
                                    }),
                                )
                                .expect("the plan list sends");
                                api.send_message(
                                    CustomMessagePayload {
                                        custom_type: "plan-mode-execute".to_string(),
                                        content: vec![text_block(exec_message)],
                                        display: true,
                                        details: None,
                                    },
                                    Some(SendMessageOptions {
                                        trigger_turn: Some(true),
                                        deliver_as: Some(DeliverAs::FollowUp),
                                    }),
                                )
                                .expect("the execute message sends");
                            } else if choice == "Refine the plan"
                                && let Some(refinement) = ui.editor("Refine the plan:", "")
                            {
                                let trimmed = refinement.trim();
                                if !trimmed.is_empty() {
                                    api.send_message(
                                        plan_todo_list,
                                        Some(SendMessageOptions {
                                            trigger_turn: None,
                                            deliver_as: Some(DeliverAs::FollowUp),
                                        }),
                                    )
                                    .expect("the plan list sends");
                                    api.send_user_message(
                                        UserContent::Text(trimmed.to_string()),
                                        Some(SendUserMessageOptions {
                                            deliver_as: Some(DeliverAs::FollowUp),
                                            ..SendUserMessageOptions::default()
                                        }),
                                    )
                                    .expect("the refinement sends");
                                }
                            }
                        }
                        HandlerResult::None
                    })
                }),
            )?;
            Ok(())
        })
    })
}

async fn plan_harness(
    active_tools: &[&str],
    select: Option<&str>,
    editor: Option<&str>,
) -> ExampleHarness {
    let ui = Arc::new(TestUi::default());
    if let Some(choice) = select {
        ui.select_with(Some(choice));
    }
    if let Some(text) = editor {
        ui.edit_with(text);
    }
    let log = Arc::new(ActionLog::default());
    log.set_active_tools(active_tools);
    harness(
        "plan-mode",
        "/tmp/pi-example-plan-mode",
        plan_mode_factory(),
        ui,
        log,
        context_actions(),
        example_registry().await,
    )
    .await
}

async fn run_plan_command(harness: &ExampleHarness, name: &str) {
    let handler = command(&harness.runner, name).handler;
    (handler)(String::new(), harness.runner.context()).await;
}

async fn trigger_agent_end(harness: &ExampleHarness, text: &str) {
    harness
        .runner
        .emit(ExtensionEvent::AgentEnd {
            messages: vec![assistant_message(text)],
        })
        .await;
}

#[tokio::test]
async fn plan_mode_preserves_custom_active_tools_while_toggling() {
    let example = plan_harness(&["read", "bash", "edit", "write", "echo_tool"], None, None).await;
    run_plan_command(&example, "plan").await;
    assert_eq!(
        example.log.active_tools(),
        [
            "read",
            "bash",
            "echo_tool",
            "grep",
            "find",
            "ls",
            "questionnaire"
        ]
    );
    run_plan_command(&example, "plan").await;
    assert_eq!(
        example.log.active_tools(),
        ["read", "bash", "edit", "write", "echo_tool"]
    );
}

#[tokio::test]
async fn plan_mode_does_not_prompt_without_a_plan() {
    let example = plan_harness(&["read", "bash", "edit", "write"], None, None).await;
    run_plan_command(&example, "plan").await;
    trigger_agent_end(
        &example,
        "This file defines the command-line argument parser.",
    )
    .await;
    assert_eq!(example.ui.select_calls(), 0);
    assert!(example.log.messages.lock().expect("messages").is_empty());
}

#[tokio::test]
async fn plan_mode_queues_refinement_as_a_follow_up_user_message() {
    let example = plan_harness(
        &["read", "bash", "edit", "write"],
        Some("Refine the plan"),
        Some("Add a regression test."),
    )
    .await;
    run_plan_command(&example, "plan").await;
    trigger_agent_end(
        &example,
        "Plan:\n1. Inspect the current implementation\n2. Add a regression test",
    )
    .await;
    let user_messages = example.log.user_messages.lock().expect("user messages");
    assert_eq!(user_messages.len(), 1);
    assert_eq!(
        user_messages[0].content,
        UserContent::Text("Add a regression test.".to_string())
    );
    assert_eq!(
        user_messages[0].options.expect("options").deliver_as,
        Some(DeliverAs::FollowUp)
    );
}

#[tokio::test]
async fn plan_mode_queues_execution_as_a_follow_up_custom_message() {
    let example = plan_harness(
        &["read", "bash", "edit", "write", "echo_tool"],
        Some("Execute the plan (track progress)"),
        None,
    )
    .await;
    run_plan_command(&example, "plan").await;
    trigger_agent_end(
        &example,
        "Plan:\n1. Inspect the current implementation\n2. Add a regression test",
    )
    .await;
    assert_eq!(
        example.log.active_tools(),
        ["read", "bash", "edit", "write", "echo_tool"]
    );
    let messages = example.log.messages.lock().expect("messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].message.custom_type, "plan-mode-execute");
    let options = messages[1].options.expect("options");
    assert_eq!(options.trigger_turn, Some(true));
    assert_eq!(options.deliver_as, Some(DeliverAs::FollowUp));
}

#[tokio::test]
async fn plan_mode_todos_command_reports_the_extracted_plan() {
    let example = plan_harness(&["read", "bash", "edit", "write"], None, None).await;
    run_plan_command(&example, "todos").await;
    assert_eq!(
        example
            .ui
            .notifications
            .lock()
            .expect("notifications")
            .last()
            .expect("a notification")
            .0,
        "No todos. Create a plan first with /plan"
    );

    example.ui.select_with(Some("Stay in plan mode"));
    run_plan_command(&example, "plan").await;
    trigger_agent_end(
        &example,
        "Plan:\n1. Inspect the current implementation\n2. Add a regression test",
    )
    .await;
    run_plan_command(&example, "todos").await;
    let notifications = example.ui.notifications.lock().expect("notifications");
    assert_eq!(
        notifications.last().expect("a notification").0,
        "Plan Progress:\n1. ○ Inspect the current implementation\n2. ○ A regression test."
    );
    assert!(example.log.messages.lock().expect("messages").is_empty());
}

// =============================================================================
// git-merge-and-resolve example
// =============================================================================

type ExecFn =
    Arc<dyn Fn(&str, &[&str]) -> pi_ai::types::BoxedFuture<'static, ExecResult> + Send + Sync>;

use pi_coding_agent::core::exec::ExecResult;

const fn ok_result() -> ExecResult {
    ExecResult {
        stdout: String::new(),
        stderr: String::new(),
        code: 0,
        killed: false,
    }
}

fn fail_result() -> ExecResult {
    ExecResult {
        stdout: String::new(),
        stderr: "error".to_string(),
        code: 1,
        killed: false,
    }
}

/// Upstream's `vi.fn` exec double: results keyed by the joined command.
#[derive(Default)]
struct FakeExec {
    calls: Mutex<Vec<(String, Vec<String>)>>,
    results: Mutex<std::collections::HashMap<String, ExecResult>>,
}

impl FakeExec {
    fn with(results: &[(&str, ExecResult)]) -> Arc<Self> {
        let fake = Arc::new(Self::default());
        let mut map = fake.results.lock().expect("results");
        for (key, result) in results {
            map.insert((*key).to_string(), result.clone());
        }
        drop(map);
        fake
    }

    fn exec(self: &Arc<Self>) -> ExecFn {
        let fake = Arc::clone(self);
        Arc::new(move |command: &str, args: &[&str]| {
            let fake = Arc::clone(&fake);
            let key = std::iter::once(command.to_string())
                .chain(args.iter().map(|arg| (*arg).to_string()))
                .collect::<Vec<_>>()
                .join(" ");
            let called_args = args.iter().map(|arg| (*arg).to_string()).collect();
            let called_command = command.to_string();
            Box::pin(async move {
                fake.calls
                    .lock()
                    .expect("calls")
                    .push((called_command, called_args));
                fake.results
                    .lock()
                    .expect("results")
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(fail_result)
            })
        })
    }

    fn called_with(&self, command: &str, args: &[&str]) -> bool {
        self.calls
            .lock()
            .expect("calls")
            .iter()
            .any(|(name, called)| {
                name == command
                    && called.len() == args.len()
                    && called.iter().zip(args).all(|(left, right)| left == right)
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConflictBlock {
    file: String,
    start_line: usize,
    separator_line: usize,
    end_line: usize,
}

/// Upstream's `findConflicts`: parse the conflict markers from every
/// unmerged path.
async fn find_conflicts(exec: &ExecFn, cwd: &str) -> Vec<ConflictBlock> {
    let result = (exec)("git", &["diff", "--name-only", "--diff-filter=U"]).await;
    if result.code != 0 || result.stdout.trim().is_empty() {
        return Vec::new();
    }
    let mut blocks = Vec::new();
    for file in result.stdout.trim().lines() {
        let Ok(content) =
            std::fs::read_to_string(std::path::Path::join(std::path::Path::new(cwd), file))
        else {
            continue;
        };
        let mut block_start = None;
        let mut separator_line = None;
        for (index, line) in content.lines().enumerate() {
            let line_no = index + 1;
            if line.starts_with("<<<<<<<") {
                block_start = Some(line_no);
                separator_line = None;
            } else if line.starts_with("=======") && block_start.is_some() {
                separator_line = Some(line_no);
            } else if line.starts_with(">>>>>>>")
                && let (Some(start), Some(separator)) = (block_start, separator_line)
            {
                blocks.push(ConflictBlock {
                    file: file.to_string(),
                    start_line: start,
                    separator_line: separator,
                    end_line: line_no,
                });
                block_start = None;
                separator_line = None;
            }
        }
    }
    blocks
}

/// Upstream's `formatRange`.
fn format_range(start: usize, end: usize) -> String {
    match start.cmp(&end) {
        std::cmp::Ordering::Greater => "empty".to_string(),
        std::cmp::Ordering::Equal => start.to_string(),
        std::cmp::Ordering::Less => format!("{start}-{end}"),
    }
}

/// Upstream's `formatConflicts`.
fn format_conflicts(reference: &str, blocks: &[ConflictBlock]) -> String {
    let mut lines = vec![format!("Merged {reference} with conflicts:"), String::new()];
    for block in blocks {
        let ours = format_range(block.start_line + 1, block.separator_line.saturating_sub(1));
        let theirs = format_range(block.separator_line + 1, block.end_line.saturating_sub(1));
        lines.push(format!(
            "  {}:{}-{} (ours {ours}, theirs {theirs})",
            block.file, block.start_line, block.end_line
        ));
    }
    lines.push(String::new());
    lines.push("Resolve these conflicts.".to_string());
    lines.join("\n")
}

/// The merge-and-resolve example body.
fn git_merge_factory(exec: ExecFn) -> ExtensionFactory {
    Arc::new(move |api: Arc<ExtensionApi>| {
        let exec = Arc::clone(&exec);
        Box::pin(async move {
            let handler_api = Arc::clone(&api);
            api.on(
                "agent_end",
                Arc::new(move |event: ExtensionEvent, ctx: HandlerContext| {
                    let exec = Arc::clone(&exec);
                    let api = Arc::clone(&handler_api);
                    Box::pin(async move {
                        if !matches!(event, ExtensionEvent::AgentEnd { .. }) {
                            return HandlerResult::None;
                        }
                        let cwd = ctx.cwd().expect("the runtime is active");
                        let rev_parse = (exec)("git", &["rev-parse", "--git-dir"]).await;
                        if rev_parse.code != 0 {
                            return HandlerResult::None;
                        }
                        let mut reference = "MERGE_HEAD".to_string();
                        let merge_head = (exec)("git", &["rev-parse", "MERGE_HEAD"]).await;
                        if merge_head.code != 0 {
                            let status = (exec)("git", &["status", "--porcelain"]).await;
                            if !status.stdout.trim().is_empty() {
                                return HandlerResult::None;
                            }
                            let upstream = (exec)(
                                "git",
                                &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                            )
                            .await;
                            if upstream.code != 0 {
                                return HandlerResult::None;
                            }
                            reference = upstream.stdout.trim().to_string();
                            let remote =
                                reference.split('/').next().unwrap_or_default().to_string();
                            suite_ui(&ctx).notify(
                                &format!(
                                    "git-merge-and-resolve: fetching {remote}, merging {reference}"
                                ),
                                Some("info"),
                            );
                            let fetch = (exec)("git", &["fetch", &remote]).await;
                            if fetch.code != 0 {
                                suite_ui(&ctx).notify(
                                    &format!(
                                        "git-merge-and-resolve: fetch failed: {}",
                                        fetch.stderr.trim()
                                    ),
                                    Some("warning"),
                                );
                                return HandlerResult::None;
                            }
                            let merge = (exec)("git", &["merge", "--no-ff", &reference]).await;
                            if merge.code == 0 {
                                return HandlerResult::None;
                            }
                        }
                        let conflicts = find_conflicts(&exec, &cwd).await;
                        if conflicts.is_empty() {
                            return HandlerResult::None;
                        }
                        api.send_user_message(
                            UserContent::Text(format_conflicts(&reference, &conflicts)),
                            Some(SendUserMessageOptions {
                                deliver_as: Some(DeliverAs::FollowUp),
                                ..SendUserMessageOptions::default()
                            }),
                        )
                        .expect("the conflict report sends");
                        HandlerResult::None
                    })
                }),
            )?;
            Ok(())
        })
    })
}

async fn git_harness(cwd: &str, results: &[(&str, ExecResult)]) -> (ExampleHarness, Arc<FakeExec>) {
    let fake = FakeExec::with(results);
    let example = harness(
        "git-merge-and-resolve",
        cwd,
        git_merge_factory(fake.exec()),
        Arc::new(TestUi::default()),
        Arc::new(ActionLog::default()),
        context_actions(),
        example_registry().await,
    )
    .await;
    (example, fake)
}

/// Upstream's `withUpstream`: a clean repo tracking `origin/main`.
fn with_upstream(results: &mut Vec<(&'static str, ExecResult)>) {
    results.push(("git rev-parse --git-dir", ok_result()));
    results.push(("git rev-parse MERGE_HEAD", fail_result()));
    results.push(("git status --porcelain", ok_result()));
    results.push((
        "git rev-parse --abbrev-ref --symbolic-full-name @{u}",
        ExecResult {
            stdout: "origin/main\n".to_string(),
            ..ok_result()
        },
    ));
    results.push(("git fetch origin", ok_result()));
}

async fn trigger_merge(example: &ExampleHarness) {
    example
        .runner
        .emit(ExtensionEvent::AgentEnd {
            messages: Vec::new(),
        })
        .await;
}

#[tokio::test]
async fn git_merge_skips_when_not_a_git_repository() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    let (example, fake) = git_harness(&cwd, &[("git rev-parse --git-dir", fail_result())]).await;
    trigger_merge(&example).await;
    assert_eq!(fake.calls.lock().expect("calls").len(), 1);
    assert!(
        example
            .log
            .user_messages
            .lock()
            .expect("messages")
            .is_empty()
    );
}

#[tokio::test]
async fn git_merge_skips_when_no_upstream_is_configured() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    let (example, _fake) = git_harness(
        &cwd,
        &[
            ("git rev-parse --git-dir", ok_result()),
            (
                "git rev-parse --abbrev-ref --symbolic-full-name @{u}",
                fail_result(),
            ),
        ],
    )
    .await;
    trigger_merge(&example).await;
    assert!(
        example
            .log
            .user_messages
            .lock()
            .expect("messages")
            .is_empty()
    );
}

#[tokio::test]
async fn git_merge_resends_conflicts_from_an_unfinished_merge() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    std::fs::write(
        root.path().join("file.ts"),
        [
            "<<<<<<< HEAD",
            "ours",
            "=======",
            "theirs",
            ">>>>>>> origin/main",
        ]
        .join("\n"),
    )
    .expect("conflict file");
    let (example, fake) = git_harness(
        &cwd,
        &[
            ("git rev-parse --git-dir", ok_result()),
            ("git rev-parse MERGE_HEAD", ok_result()),
            (
                "git diff --name-only --diff-filter=U",
                ExecResult {
                    stdout: "file.ts\n".to_string(),
                    ..ok_result()
                },
            ),
        ],
    )
    .await;
    trigger_merge(&example).await;
    assert!(!fake.called_with("git", &["fetch", "origin"]));
    let messages = example.log.user_messages.lock().expect("messages");
    assert_eq!(messages.len(), 1);
    let UserContent::Text(text) = &messages[0].content else {
        panic!("the report is text");
    };
    assert!(text.contains("file.ts:1-5"), "{text}");
}

#[tokio::test]
async fn git_merge_skips_a_dirty_working_tree() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    let (example, fake) = git_harness(
        &cwd,
        &[
            ("git rev-parse --git-dir", ok_result()),
            ("git rev-parse MERGE_HEAD", fail_result()),
            (
                "git status --porcelain",
                ExecResult {
                    stdout: " M src/index.ts\n".to_string(),
                    ..ok_result()
                },
            ),
        ],
    )
    .await;
    trigger_merge(&example).await;
    assert!(!fake.called_with("git", &["fetch", "origin"]));
    assert!(
        example
            .log
            .user_messages
            .lock()
            .expect("messages")
            .is_empty()
    );
}

#[tokio::test]
async fn git_merge_skips_when_fetch_fails() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    let mut results = Vec::new();
    with_upstream(&mut results);
    results.push(("git fetch origin", fail_result()));
    let (example, _fake) = git_harness(&cwd, &results).await;
    trigger_merge(&example).await;
    assert!(
        example
            .log
            .user_messages
            .lock()
            .expect("messages")
            .is_empty()
    );
}

#[tokio::test]
async fn git_merge_skips_a_clean_merge() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    let mut results = Vec::new();
    with_upstream(&mut results);
    results.push(("git merge --no-ff origin/main", ok_result()));
    let (example, _fake) = git_harness(&cwd, &results).await;
    trigger_merge(&example).await;
    assert!(
        example
            .log
            .user_messages
            .lock()
            .expect("messages")
            .is_empty()
    );
}

#[tokio::test]
async fn git_merge_reports_each_conflict_block_as_a_follow_up() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    let content = [
        "line 1",
        "<<<<<<< HEAD",
        "our change",
        "=======",
        "their change",
        ">>>>>>> origin/main",
        "line 7",
        "<<<<<<< HEAD",
        "second conflict",
        "=======",
        "their second",
        ">>>>>>> origin/main",
    ]
    .join("\n");
    std::fs::create_dir_all(root.path().join("src")).expect("src dir");
    std::fs::write(root.path().join("src/index.ts"), content).expect("conflict file");
    let mut results = Vec::new();
    with_upstream(&mut results);
    results.push(("git merge --no-ff origin/main", fail_result()));
    results.push((
        "git diff --name-only --diff-filter=U",
        ExecResult {
            stdout: "src/index.ts\n".to_string(),
            ..ok_result()
        },
    ));
    let (example, _fake) = git_harness(&cwd, &results).await;
    trigger_merge(&example).await;
    let messages = example.log.user_messages.lock().expect("messages");
    assert_eq!(messages.len(), 1);
    let UserContent::Text(text) = &messages[0].content else {
        panic!("the report is text");
    };
    assert!(
        text.contains("src/index.ts:2-6 (ours 3, theirs 5)"),
        "{text}"
    );
    assert!(
        text.contains("src/index.ts:8-12 (ours 9, theirs 11)"),
        "{text}"
    );
    assert_eq!(
        messages[0].options.expect("options").deliver_as,
        Some(DeliverAs::FollowUp)
    );
}

#[tokio::test]
async fn git_merge_handles_empty_ours_or_theirs_sections() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    std::fs::write(
        root.path().join("empty-ours.ts"),
        [
            "<<<<<<< HEAD",
            "=======",
            "only theirs",
            ">>>>>>> origin/main",
        ]
        .join("\n"),
    )
    .expect("conflict file");
    let mut results = Vec::new();
    with_upstream(&mut results);
    results.push(("git merge --no-ff origin/main", fail_result()));
    results.push((
        "git diff --name-only --diff-filter=U",
        ExecResult {
            stdout: "empty-ours.ts\n".to_string(),
            ..ok_result()
        },
    ));
    let (example, _fake) = git_harness(&cwd, &results).await;
    trigger_merge(&example).await;
    let messages = example.log.user_messages.lock().expect("messages");
    assert_eq!(messages.len(), 1);
    let UserContent::Text(text) = &messages[0].content else {
        panic!("the report is text");
    };
    assert!(
        text.contains("empty-ours.ts:1-4 (ours empty, theirs 3)"),
        "{text}"
    );
}

#[tokio::test]
async fn git_merge_skips_the_report_when_no_markers_are_found() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().display().to_string();
    let mut results = Vec::new();
    with_upstream(&mut results);
    results.push(("git merge --no-ff origin/main", fail_result()));
    results.push((
        "git diff --name-only --diff-filter=U",
        ExecResult {
            stdout: String::new(),
            ..ok_result()
        },
    ));
    let (example, _fake) = git_harness(&cwd, &results).await;
    trigger_merge(&example).await;
    assert!(
        example
            .log
            .user_messages
            .lock()
            .expect("messages")
            .is_empty()
    );
}

// =============================================================================
// input-transform-streaming example
// =============================================================================

/// Upstream's `TRIGGER = /\b(changes?|diff|modified)\b/i`.
fn mentions_changes(text: &str) -> bool {
    Regex::new(r"(?i)\b(changes?|diff|modified)\b")
        .expect("valid regex")
        .is_match(text)
}

/// The streaming input gate example body.
fn input_transform_factory(exec: ExecFn) -> ExtensionFactory {
    Arc::new(move |api: Arc<ExtensionApi>| {
        let exec = Arc::clone(&exec);
        Box::pin(async move {
            api.on(
                "input",
                Arc::new(move |event: ExtensionEvent, _ctx: HandlerContext| {
                    let exec = Arc::clone(&exec);
                    Box::pin(async move {
                        let ExtensionEvent::Input {
                            text,
                            source: _,
                            streaming_behavior,
                            ..
                        } = event
                        else {
                            return HandlerResult::None;
                        };
                        if streaming_behavior == Some(StreamingBehavior::Steer) {
                            return HandlerResult::Input(InputResult::Continue);
                        }
                        if !mentions_changes(&text) {
                            return HandlerResult::Input(InputResult::Continue);
                        }
                        let result = (exec)("git", &["diff", "--stat"]).await;
                        if result.code != 0 || result.stdout.trim().is_empty() {
                            return HandlerResult::Input(InputResult::Continue);
                        }
                        HandlerResult::Input(InputResult::Transform {
                            text: format!(
                                "{}\n\nCurrent uncommitted changes:\n```\n{}\n```",
                                text,
                                result.stdout.trim()
                            ),
                            images: None,
                        })
                    })
                }),
            )?;
            Ok(())
        })
    })
}

async fn input_harness(result: ExecResult) -> (ExampleHarness, Arc<FakeExec>) {
    let fake = FakeExec::with(&[("git diff --stat", result)]);
    let example = harness(
        "input-transform-streaming",
        "/tmp/pi-example-input-transform",
        input_transform_factory(fake.exec()),
        Arc::new(TestUi::default()),
        Arc::new(ActionLog::default()),
        context_actions(),
        example_registry().await,
    )
    .await;
    (example, fake)
}

async fn emit_input(
    example: &ExampleHarness,
    text: &str,
    behavior: Option<StreamingBehavior>,
) -> InputResult {
    example
        .runner
        .emit_input(text.to_string(), None, InputSource::Interactive, behavior)
        .await
}

#[tokio::test]
async fn input_transform_skips_exec_during_steering() {
    let (example, fake) = input_harness(ExecResult {
        stdout: " src/index.ts | 5 ++---\n 1 file changed, 2 insertions(+), 3 deletions(-)"
            .to_string(),
        ..ok_result()
    })
    .await;
    let result = emit_input(
        &example,
        "what changes did I make?",
        Some(StreamingBehavior::Steer),
    )
    .await;
    assert!(matches!(result, InputResult::Continue));
    assert!(fake.calls.lock().expect("calls").is_empty());
}

#[tokio::test]
async fn input_transform_transforms_when_idle_and_the_text_matches() {
    let (example, fake) = input_harness(ExecResult {
        stdout: " src/index.ts | 5 ++---\n 1 file changed, 2 insertions(+), 3 deletions(-)"
            .to_string(),
        ..ok_result()
    })
    .await;
    let result = emit_input(&example, "review my changes", None).await;
    assert!(fake.called_with("git", &["diff", "--stat"]));
    let InputResult::Transform { text, .. } = result else {
        panic!("the text transforms");
    };
    assert!(text.contains("review my changes"), "{text}");
    assert!(text.contains("src/index.ts"), "{text}");
}

#[tokio::test]
async fn input_transform_transforms_when_queued_as_a_follow_up() {
    let (example, fake) = input_harness(ExecResult {
        stdout: " src/index.ts | 5 ++---".to_string(),
        ..ok_result()
    })
    .await;
    let result = emit_input(
        &example,
        "show me the diff",
        Some(StreamingBehavior::FollowUp),
    )
    .await;
    assert!(!fake.calls.lock().expect("calls").is_empty());
    assert!(matches!(result, InputResult::Transform { .. }));
}

#[tokio::test]
async fn input_transform_continues_when_the_text_does_not_match() {
    let (example, fake) = input_harness(ExecResult {
        stdout: " src/index.ts | 5 ++---".to_string(),
        ..ok_result()
    })
    .await;
    let result = emit_input(&example, "explain this function", None).await;
    assert!(matches!(result, InputResult::Continue));
    assert!(fake.calls.lock().expect("calls").is_empty());
}

#[tokio::test]
async fn input_transform_continues_when_git_diff_is_empty() {
    let (example, _fake) = input_harness(ok_result()).await;
    let result = emit_input(&example, "any changes?", None).await;
    assert!(matches!(result, InputResult::Continue));
}

#[tokio::test]
async fn input_transform_continues_when_git_fails() {
    let (example, _fake) = input_harness(ExecResult {
        stderr: "not a git repo".to_string(),
        code: 128,
        ..ok_result()
    })
    .await;
    let result = emit_input(&example, "show modified files", None).await;
    assert!(matches!(result, InputResult::Continue));
}

// =============================================================================
// trigger-compact example
// =============================================================================

const COMPACT_THRESHOLD_TOKENS: u64 = 100_000;

/// The auto-compact example body.
fn trigger_compact_factory() -> ExtensionFactory {
    Arc::new(|api: Arc<ExtensionApi>| {
        Box::pin(async move {
            let previous_tokens = Arc::new(Mutex::new(None::<Option<u64>>));
            let trigger_compaction = |ctx: &HandlerContext, custom_instructions: Option<String>| {
                if ctx.has_ui().unwrap_or(false) {
                    suite_ui(ctx).notify("Compaction started", Some("info"));
                }
                ctx.compact(custom_instructions)
                    .expect("the compaction starts");
            };
            let handler_previous = Arc::clone(&previous_tokens);
            api.on(
                "turn_end",
                Arc::new(move |_event: ExtensionEvent, ctx: HandlerContext| {
                    let previous_tokens = Arc::clone(&handler_previous);
                    Box::pin(async move {
                        let usage = ctx.get_context_usage().expect("the runtime is active");
                        let Some(current_tokens) = usage.and_then(|usage| usage.tokens) else {
                            return HandlerResult::None;
                        };
                        let previous = *previous_tokens
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        let crossed_threshold = match previous {
                            Some(Some(previous)) => previous <= COMPACT_THRESHOLD_TOKENS,
                            _ => false,
                        };
                        *previous_tokens
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(Some(current_tokens));
                        if !crossed_threshold || current_tokens <= COMPACT_THRESHOLD_TOKENS {
                            return HandlerResult::None;
                        }
                        trigger_compaction(&ctx, None);
                        HandlerResult::None
                    })
                }),
            )?;
            api.register_command(
                "trigger-compact".to_string(),
                Some("Trigger compaction immediately".to_string()),
                Arc::new(move |args: String, ctx: HandlerContext| {
                    Box::pin(async move {
                        let instructions = args.trim();
                        trigger_compaction(
                            &ctx,
                            (!instructions.is_empty()).then(|| instructions.to_string()),
                        );
                    })
                }),
            )?;
            Ok(())
        })
    })
}

async fn compact_harness(
    tokens: Arc<Mutex<Option<u64>>>,
    compact_calls: Arc<Mutex<Vec<Option<String>>>>,
) -> ExampleHarness {
    let usage_tokens = Arc::clone(&tokens);
    let compact = Arc::clone(&compact_calls);
    harness(
        "trigger-compact",
        "/tmp/pi-example-trigger-compact",
        trigger_compact_factory(),
        Arc::new(TestUi::default()),
        Arc::new(ActionLog::default()),
        ContextActions {
            get_context_usage: Some(Arc::new(move || {
                usage_tokens
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .map(
                        |tokens| pi_coding_agent::extensions::runtime::ContextUsage {
                            tokens: Some(tokens),
                            context_window: 200_000,
                            percent: Some(tokens as f64 / 2000.0),
                        },
                    )
            })),
            compact: Some(Arc::new(move |instructions| {
                compact
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(instructions);
            })),
            ..ContextActions::default()
        },
        example_registry().await,
    )
    .await
}

fn set_tokens(tokens: &Mutex<Option<u64>>, value: u64) {
    *tokens
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(value);
}

async fn trigger_turn_end(example: &ExampleHarness) {
    example
        .runner
        .emit(ExtensionEvent::TurnEnd {
            turn_index: 0,
            message: assistant_message(""),
            tool_results: Vec::new(),
        })
        .await;
}

#[tokio::test]
async fn trigger_compact_only_auto_compacts_past_the_threshold() {
    let tokens = Arc::new(Mutex::new(None));
    let compact_calls = Arc::new(Mutex::new(Vec::new()));
    let example = compact_harness(Arc::clone(&tokens), Arc::clone(&compact_calls)).await;

    set_tokens(&tokens, 110_000);
    trigger_turn_end(&example).await;
    set_tokens(&tokens, 120_000);
    trigger_turn_end(&example).await;
    set_tokens(&tokens, 95_000);
    trigger_turn_end(&example).await;
    assert!(compact_calls.lock().expect("calls").is_empty());

    set_tokens(&tokens, 105_000);
    trigger_turn_end(&example).await;
    assert_eq!(compact_calls.lock().expect("calls").len(), 1);
}

#[tokio::test]
async fn trigger_compact_command_passes_its_instructions() {
    let tokens = Arc::new(Mutex::new(None));
    let compact_calls = Arc::new(Mutex::new(Vec::new()));
    let example = compact_harness(Arc::clone(&tokens), Arc::clone(&compact_calls)).await;
    let handler = command(&example.runner, "trigger-compact").handler;
    (handler)(
        "  keep the api notes  ".to_string(),
        example.runner.context(),
    )
    .await;
    assert_eq!(
        compact_calls.lock().expect("calls").clone(),
        [Some("keep the api notes".to_string())]
    );
}

// =============================================================================
// custom-compaction (documentation example)
// =============================================================================

type CompleteFn = Arc<
    dyn Fn(
            pi_ai::types::Model,
            pi_ai::types::Context,
            u64,
            Option<pi_coding_agent::extensions::events::AbortSignal>,
            Option<String>,
        ) -> pi_ai::types::BoxedFuture<'static, pi_ai::types::AssistantMessage>
        + Send
        + Sync,
>;

/// The custom-compaction example body: summarize the whole context through
/// the injected completion, upstream's `modelRegistry.complete` mock.
fn custom_compaction_factory(complete: CompleteFn) -> ExtensionFactory {
    Arc::new(move |api: Arc<ExtensionApi>| {
        let complete = Arc::clone(&complete);
        Box::pin(async move {
            api.on(
                "session_before_compact",
                Arc::new(move |event: ExtensionEvent, ctx: HandlerContext| {
                    let complete = Arc::clone(&complete);
                    Box::pin(async move {
                        let ExtensionEvent::SessionBeforeCompact {
                            preparation,
                            signal,
                            ..
                        } = event
                        else {
                            return HandlerResult::None;
                        };
                        let ui = suite_ui(&ctx);
                        ui.notify("Custom compaction extension triggered", Some("info"));
                        let registry = ctx.model_registry().expect("the runtime is active");
                        let Some(model) = registry.find("google", "gemini-2.5-flash") else {
                            ui.notify(
                                "Could not find Gemini Flash model, using default compaction",
                                Some("warning"),
                            );
                            return HandlerResult::None;
                        };
                        let mut all_messages = preparation.messages_to_summarize;
                        all_messages.extend(preparation.turn_prefix_messages);
                        ui.notify(
                            &format!(
                                "Custom compaction: summarizing {} messages ({} tokens) with {}...",
                                all_messages.len(),
                                preparation.tokens_before,
                                model.id
                            ),
                            Some("info"),
                        );
                        let conversation_text = pi_coding_agent::compaction::serialize_conversation(
                            &pi_coding_agent::messages::convert_to_llm(&all_messages),
                        );
                        let previous_context = preparation
                            .previous_summary
                            .as_ref()
                            .map(|summary| {
                                format!("\n\nPrevious session summary for context:\n{summary}")
                            })
                            .unwrap_or_default();
                        let prompt = format!(
                            "You are a conversation summarizer. Create a comprehensive summary \
                             of this conversation that captures:{previous_context}\n\n1. The main \
                             goals and objectives discussed\n2. Key decisions made and their \
                             rationale\n3. Important code changes, file modifications, or \
                             technical details\n4. Current state of any ongoing work\n5. Any \
                             blockers, issues, or open questions\n6. Next steps that were planned \
                             or suggested\n\nBe thorough but concise. The summary will replace the \
                             ENTIRE conversation history, so include all information needed to \
                             continue the work effectively.\n\nFormat the summary as structured \
                             markdown with clear sections.\n\n<conversation>\n{conversation_text}\n\
                             </conversation>"
                        );
                        let context = pi_ai::types::Context {
                            messages: vec![Message::User(pi_ai::types::UserMessage {
                                content: UserContent::Blocks(vec![text_block(prompt)]),
                                timestamp: 1,
                            })],
                            ..pi_ai::types::Context::default()
                        };
                        let session_id = pi_ai::utils::uuid::uuidv7(None).ok();
                        let response = (complete)(model, context, 8192, signal, session_id).await;
                        let summary = response
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                pi_ai::types::AssistantBlock::Text(text) => Some(text.text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if summary.trim().is_empty() {
                            ui.notify(
                                "Compaction summary was empty, using default compaction",
                                Some("warning"),
                            );
                            return HandlerResult::None;
                        }
                        HandlerResult::SessionBefore(
                            pi_coding_agent::extensions::results::SessionBeforeResult::Compact {
                                cancel: false,
                                compaction: Some(pi_coding_agent::compaction::CompactionResult {
                                    summary,
                                    first_kept_entry_id: preparation.first_kept_entry_id,
                                    tokens_before: preparation.tokens_before,
                                    estimated_tokens_after: None,
                                    usage: response.usage,
                                    details: None,
                                }),
                            },
                        )
                    })
                }),
            )?;
            Ok(())
        })
    })
}

struct CompleteCall {
    model: pi_ai::types::Model,
    context: pi_ai::types::Context,
    max_tokens: u64,
    signal: Option<pi_coding_agent::extensions::events::AbortSignal>,
    session_id: Option<String>,
}

async fn compaction_harness() -> (ExampleHarness, Arc<Mutex<Vec<CompleteCall>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let complete: CompleteFn = {
        let calls = Arc::clone(&calls);
        Arc::new(move |model, context, max_tokens, signal, session_id| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.lock().expect("calls").push(CompleteCall {
                    model,
                    context,
                    max_tokens,
                    signal,
                    session_id,
                });
                common::agent_session::assistant_with_usage(
                    "custom provider summary",
                    common::agent_session::usage(1, 2),
                    0,
                )
            })
        })
    };
    let example = harness(
        "custom-compaction",
        "/tmp/pi-example-custom-compaction",
        custom_compaction_factory(complete),
        Arc::new(TestUi::default()),
        Arc::new(ActionLog::default()),
        context_actions(),
        example_registry().await,
    )
    .await;
    (example, calls)
}

fn compaction_preparation() -> pi_coding_agent::compaction::CompactionPreparation {
    pi_coding_agent::compaction::CompactionPreparation {
        first_kept_entry_id: "entry-1".to_string(),
        messages_to_summarize: vec![common::agent_session::user_message(
            "please remember this",
            0,
        )],
        turn_prefix_messages: Vec::new(),
        is_split_turn: false,
        tokens_before: 42,
        previous_summary: None,
        file_ops: pi_coding_agent::compaction::FileOperations::default(),
        settings: pi_coding_agent::settings_manager::CompactionSettings {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 0,
        },
    }
}

#[tokio::test]
async fn custom_compaction_uses_the_documented_event_shape() {
    let preparation = compaction_preparation();
    let event = ExtensionEvent::SessionBeforeCompact {
        preparation,
        branch_entries: Vec::new(),
        custom_instructions: None,
        reason: pi_coding_agent::extensions::reasons::CompactionReason::Manual,
        will_retry: false,
        signal: None,
    };
    let ExtensionEvent::SessionBeforeCompact {
        preparation: carried,
        branch_entries,
        ..
    } = event
    else {
        panic!("the event is session_before_compact");
    };
    assert_eq!(carried.messages_to_summarize.len(), 1);
    assert!(carried.turn_prefix_messages.is_empty());
    assert!(!carried.is_split_turn);
    assert_eq!(carried.tokens_before, 42);
    assert_eq!(carried.first_kept_entry_id, "entry-1");
    assert!(branch_entries.is_empty());

    let entry = pi_coding_agent::session_manager::CompactionEntry {
        base: pi_coding_agent::session_manager::SessionEntryBase {
            id: Some("compaction-1".to_string()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        summary: "summary".to_string(),
        first_kept_entry_id: Some("entry-1".to_string()),
        tokens_before: 42,
        details: None,
        usage: None,
        from_hook: Some(true),
        extras: serde_json::Map::new(),
    };
    let compact = ExtensionEvent::SessionCompact {
        compaction_entry: entry,
        from_extension: true,
        reason: pi_coding_agent::extensions::reasons::CompactionReason::Manual,
        will_retry: false,
    };
    let ExtensionEvent::SessionCompact {
        compaction_entry,
        from_extension,
        ..
    } = compact
    else {
        panic!("the event is session_compact");
    };
    assert_eq!(compaction_entry.summary, "summary");
    assert_eq!(compaction_entry.tokens_before, 42);
    assert!(from_extension);
}

#[tokio::test]
async fn custom_compaction_dispatches_through_the_injected_complete() {
    let (example, calls) = compaction_harness().await;
    let result = example
        .runner
        .emit(ExtensionEvent::SessionBeforeCompact {
            preparation: compaction_preparation(),
            branch_entries: Vec::new(),
            custom_instructions: None,
            reason: pi_coding_agent::extensions::reasons::CompactionReason::Manual,
            will_retry: false,
            signal: None,
        })
        .await;

    let calls = calls.lock().expect("calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].model.id, "gemini-2.5-flash");
    assert_eq!(calls[0].max_tokens, 8192);
    assert!(calls[0].signal.is_none());
    assert!(calls[0].session_id.is_some());
    let Message::User(user) = &calls[0].context.messages[0] else {
        panic!("the summary request is a user message");
    };
    let UserContent::Blocks(blocks) = &user.content else {
        panic!("the summary request carries content blocks");
    };
    let pi_ai::types::UserBlock::Text(prompt) = &blocks[0] else {
        panic!("the prompt is text");
    };
    assert!(prompt.text.contains("[User]: please remember this"));

    let pi_coding_agent::extensions::results::SessionBeforeResult::Compact {
        cancel,
        compaction: Some(compaction),
    } = result
    else {
        panic!("the extension supplies a compaction");
    };
    assert!(!cancel);
    assert_eq!(compaction.summary, "custom provider summary");
    assert_eq!(compaction.first_kept_entry_id, "entry-1");
    assert_eq!(compaction.tokens_before, 42);
    assert_eq!(compaction.usage.input, 1);
    assert_eq!(compaction.usage.output, 2);
    let notifications = example.ui.notifications.lock().expect("notifications");
    assert!(
        notifications
            .iter()
            .any(|(message, _)| message == "Custom compaction extension triggered")
    );
}
