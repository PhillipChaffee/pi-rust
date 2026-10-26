//! `test/rpc-prompt-response-semantics.test.ts` and
//! `test/suite/regressions/5868-rpc-unknown-command-id.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements recorded here:
//! - Upstream's preflight-rejection case fails on a missing API key; the port
//!   resolves auth inside the stream and reports it as an assistant error, so
//!   the reachable preflight rejection is a prompt sent mid-run without a
//!   streaming behavior.
//! - Upstream drives `runRpcMode` over a cast runtime host and a mocked
//!   output guard; the port drives a real in-memory runtime and captures the
//!   raw writer.
//! - The stdout-cleanliness mode half is pinned by every captured line
//!   parsing as JSON; the entry's startup chatter rides #185.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use common::rpc::RpcHarness;
use serde_json::Value;
use serde_json::json;

fn prompt_responses<'a>(lines: &'a [Value], id: &str) -> Vec<&'a Value> {
    lines
        .iter()
        .filter(|line| {
            line["id"] == id && line["type"] == "response" && line["command"] == "prompt"
        })
        .collect()
}

#[tokio::test]
async fn emits_one_success_response_when_prompt_preflight_succeeds() {
    let harness = RpcHarness::start(0).await;

    harness.send(json!({ "id": "b2", "type": "prompt", "message": "Hello" }));
    let lines = harness
        .wait_for(|lines| prompt_responses(lines, "b2").len() == 1)
        .await;

    let response = prompt_responses(&lines, "b2")[0];
    assert_eq!(response["success"], true);
    assert!(
        response.get("data").is_none(),
        "the success arm carries no data"
    );
    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn emits_one_failure_response_when_prompt_preflight_rejects() {
    let harness = RpcHarness::start(500).await;

    harness.send(json!({ "id": "b1-start", "type": "prompt", "message": "Start" }));
    let _ = harness
        .wait_for(|lines| prompt_responses(lines, "b1-start").len() == 1)
        .await;
    let _ = harness
        .wait_for(|lines| lines.iter().any(|line| line["type"] == "agent_start"))
        .await;

    harness.send(json!({ "id": "b1", "type": "prompt", "message": "Hello" }));
    let lines = harness
        .wait_for(|lines| prompt_responses(lines, "b1").len() == 1)
        .await;

    let responses = prompt_responses(&lines, "b1");
    assert_eq!(responses.len(), 1, "exactly one response per prompt id");
    let response = responses[0];
    assert_eq!(response["success"], false);
    assert!(
        response["error"]
            .as_str()
            .is_some_and(|error| error.contains("already processing")),
        "the preflight rejection reports its message: {response}"
    );
    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn emits_one_success_response_when_prompt_is_queued_during_streaming() {
    let harness = RpcHarness::start(100).await;

    harness.send(json!({ "id": "b3-start", "type": "prompt", "message": "Start" }));
    let _ = harness
        .wait_for(|lines| prompt_responses(lines, "b3-start").len() == 1)
        .await;
    let _ = harness
        .wait_for(|lines| lines.iter().any(|line| line["type"] == "agent_start"))
        .await;

    harness.send(json!({
        "id": "b3",
        "type": "prompt",
        "message": "Queue this",
        "streamingBehavior": "followUp"
    }));
    let lines = harness
        .wait_for(|lines| prompt_responses(lines, "b3").len() == 1)
        .await;

    assert_eq!(prompt_responses(&lines, "b3")[0]["success"], true);
    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn returns_and_clears_queued_steering_and_follow_up_messages() {
    let harness = RpcHarness::start(500).await;

    harness.send(json!({ "id": "clear-start", "type": "prompt", "message": "Start" }));
    let _ = harness
        .wait_for(|lines| prompt_responses(lines, "clear-start").len() == 1)
        .await;
    let _ = harness
        .wait_for(|lines| lines.iter().any(|line| line["type"] == "agent_start"))
        .await;

    harness.send(json!({
        "id": "clear-steering",
        "type": "prompt",
        "message": "Change direction",
        "streamingBehavior": "steer"
    }));
    let _ = harness
        .wait_for(|lines| prompt_responses(lines, "clear-steering").len() == 1)
        .await;

    harness.send(json!({
        "id": "clear-follow-up",
        "type": "prompt",
        "message": "Summarize when finished",
        "streamingBehavior": "followUp"
    }));
    let _ = harness
        .wait_for(|lines| prompt_responses(lines, "clear-follow-up").len() == 1)
        .await;

    harness.send(json!({ "id": "clear", "type": "clear_queue" }));
    let lines = harness
        .wait_for(|lines| lines.iter().any(|line| line["id"] == "clear"))
        .await;
    let cleared = lines
        .iter()
        .find(|line| line["id"] == "clear")
        .expect("the clear response landed");

    assert_eq!(
        *cleared,
        json!({
            "id": "clear",
            "type": "response",
            "command": "clear_queue",
            "success": true,
            "data": {
                "steering": ["Change direction"],
                "followUp": ["Summarize when finished"]
            }
        })
    );

    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    let agent_starts = harness
        .parsed()
        .iter()
        .filter(|line| line["type"] == "agent_start")
        .count();
    assert_eq!(agent_starts, 1, "the cleared queue starts no second run");
    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn preserves_the_request_id_on_unknown_command_errors() {
    let harness = RpcHarness::start(0).await;

    harness.send(json!({ "id": "test", "type": "foobar" }));
    let lines = harness
        .wait_for(|lines| lines.iter().any(|line| line["id"] == "test"))
        .await;
    let response = lines
        .iter()
        .find(|line| line["id"] == "test")
        .expect("the error response landed");

    assert_eq!(
        *response,
        json!({
            "id": "test",
            "type": "response",
            "command": "foobar",
            "success": false,
            "error": "Unknown command: foobar"
        })
    );
    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn reports_parse_failures_on_the_parse_command() {
    let harness = RpcHarness::start(0).await;

    harness.send_raw("not json");
    let lines = harness
        .wait_for(|lines| lines.iter().any(|line| line["command"] == "parse"))
        .await;
    let response = lines
        .iter()
        .find(|line| line["command"] == "parse")
        .expect("the parse error landed");

    assert_eq!(response["success"], false);
    assert!(
        response["error"]
            .as_str()
            .is_some_and(|error| error.starts_with("Failed to parse command: ")),
        "the parse message carries the parser detail: {response}"
    );
    assert_eq!(harness.shutdown().await, 0);
}
