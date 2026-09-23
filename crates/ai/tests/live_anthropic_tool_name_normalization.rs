//! Env-gated Anthropic OAuth tool-name normalization E2E suite, ported from
//! the upstream `anthropic-tool-name-normalization.test.ts` probes — the
//! `Anthropic OAuth tool name normalization` describe — at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! With Claude Code OAuth, tool names must match CC's canonical casing
//! outbound and return to the caller's original casing inbound — a
//! case-insensitive round-trip, not a rename map: `todowrite` → `TodoWrite`
//! → `todowrite` round-trips, while `find` is a different tool from CC's
//! `Glob` and must pass through unchanged.
//!
//! Like upstream's `describe.skipIf(!oauthToken)`, a probe without a
//! resolvable anthropic OAuth token returns early; upstream resolves the
//! token once at module load, the port resolves per test with the same
//! skipIf semantics.

// binary and carries its own expects only for the lints its own code knows
// it trips; these cover the rest so the suite's gate runs clean without
// editing the shared file. Stale entries fail the build on their own.

use pi_ai::types::{AssistantMessage, AssistantMessageEvent, Context, Model, StopReason, Tool};
use serde_json::json;

mod common;
use common::live;

/// The model the probes drive, upstream's `getModel("anthropic",
/// "claude-sonnet-4-6")`.
fn model() -> Model {
    live::model("anthropic", "claude-sonnet-4-6")
}

/// The single-string tool fixture, upstream's `Type.Object` schemas.
fn tool(name: &str, description: &str, property: &str, property_description: &str) -> Tool {
    let mut properties = serde_json::Map::new();
    properties.insert(
        property.to_owned(),
        json!({
            "type": "string",
            "description": property_description,
        }),
    );
    Tool {
        name: name.to_owned(),
        description: description.to_owned(),
        parameters: json!({ "type": "object", "properties": properties }),
        constrained_sampling: None,
    }
}

/// The shared probe run, upstream's four identical stream loops: send the
/// tool prompt over the compat surface with the OAuth token, take the tool
/// name the last `toolcall_end` carries, and settle.
async fn tool_call_name(
    model: &Model,
    context: &Context,
    oauth_token: &str,
) -> (Option<String>, AssistantMessage) {
    let options = live::LiveOptions::key(oauth_token.to_owned());
    let stream = live::stream(model, context, &options);
    let mut name: Option<String> = None;
    while let Some(event) = stream.next().await {
        if let AssistantMessageEvent::ToolcallEnd { tool_call, .. } = event {
            name = Some(tool_call.name);
        }
    }
    let response = stream.result().await;
    (name, response)
}

/// Upstream it "should normalize user-defined tool matching CC name
/// (`todowrite` -> `TodoWrite` -> `todowrite`)": the lowercase user tool
/// round-trips through CC's canonical casing and comes back as the original
/// `todowrite`.
#[tokio::test]
async fn should_normalize_user_defined_tool_matching_cc_name() {
    let Some(oauth_token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let context = Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the todowrite tool when asked to add todos."
                .to_owned(),
        ),
        messages: vec![live::user_message(
            "Add a todo: buy milk. Use the todowrite tool.",
        )],
        tools: Some(vec![tool(
            "todowrite",
            "Write a todo item",
            "task",
            "The task to add",
        )]),
    };

    let (tool_call_name, response) = tool_call_name(&model(), &context, &oauth_token).await;

    assert_eq!(
        response.stop_reason,
        StopReason::ToolUse,
        "Error: {:?}",
        response.error_message
    );
    assert_eq!(
        tool_call_name.as_deref(),
        Some("todowrite"),
        "the original casing round-trips, not CC's TodoWrite"
    );
}

/// Upstream it "should handle pi's built-in tools (read, write, edit, bash)":
/// the lowercase `read` tool returns as `read`, not CC's `Read`.
#[tokio::test]
async fn should_handle_pis_builtin_tools() {
    let Some(oauth_token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let context = Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the read tool to read files.".to_owned(),
        ),
        messages: vec![live::user_message(
            "Read the file /tmp/test.txt using the read tool.",
        )],
        tools: Some(vec![tool("read", "Read a file", "path", "File path")]),
    };

    let (tool_call_name, response) = tool_call_name(&model(), &context, &oauth_token).await;

    assert_eq!(
        response.stop_reason,
        StopReason::ToolUse,
        "Error: {:?}",
        response.error_message
    );
    assert_eq!(tool_call_name.as_deref(), Some("read"));
}

/// Upstream it "should NOT map find to Glob - find is not a CC tool name":
/// `find` and CC's `Glob` are different tools, so the broken find→Glob
/// mapping would send `Glob` outbound and echo `Glob` back for a tool the
/// context does not carry; the correct round-trip leaves `find` unchanged.
#[tokio::test]
async fn should_not_map_find_to_glob() {
    let Some(oauth_token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let context = Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the find tool to search for files.".to_owned(),
        ),
        messages: vec![live::user_message(
            "Find all .ts files using the find tool.",
        )],
        tools: Some(vec![tool(
            "find",
            "Find files by pattern",
            "pattern",
            "Glob pattern",
        )]),
    };

    let (tool_call_name, response) = tool_call_name(&model(), &context, &oauth_token).await;

    assert_eq!(
        response.stop_reason,
        StopReason::ToolUse,
        "Error: {:?}",
        response.error_message
    );
    assert_eq!(tool_call_name.as_deref(), Some("find"));
}

/// Upstream it "should handle custom tools that don't match any CC tool
/// names": a completely custom tool passes through unchanged.
#[tokio::test]
async fn should_handle_custom_tools_that_dont_match_any_cc_tool_names() {
    let Some(oauth_token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let context = Context {
        system_prompt: Some(
            "You are a helpful assistant. Use my_custom_tool when asked.".to_owned(),
        ),
        messages: vec![live::user_message("Use my_custom_tool with input 'hello'.")],
        tools: Some(vec![tool(
            "my_custom_tool",
            "A custom tool",
            "input",
            "Input value",
        )]),
    };

    let (tool_call_name, response) = tool_call_name(&model(), &context, &oauth_token).await;

    assert_eq!(
        response.stop_reason,
        StopReason::ToolUse,
        "Error: {:?}",
        response.error_message
    );
    assert_eq!(tool_call_name.as_deref(), Some("my_custom_tool"));
}
