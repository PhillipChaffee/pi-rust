//! The RPC command surface over a real in-memory runtime, upstream's
//! `src/modes/rpc/rpc-mode.ts` dispatch at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's `rpc.test.ts` covers this surface through the CLI with live
//! credentials; the port's binary entry rides #185, so the same commands run
//! in process here against the faux provider.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use common::rpc::RpcHarness;
use serde_json::Value;
use serde_json::json;

/// Send a command and return its response line.
async fn call(harness: &RpcHarness, command: Value) -> Value {
    let id = command["id"]
        .as_str()
        .expect("the test commands carry an id")
        .to_owned();
    harness.send(command);
    let lines = harness
        .wait_for(|lines| {
            lines
                .iter()
                .any(|line| line["type"] == "response" && line["id"] == id.as_str())
        })
        .await;
    lines
        .iter()
        .find(|line| line["type"] == "response" && line["id"] == id.as_str())
        .expect("the response landed")
        .clone()
}

#[tokio::test]
async fn state_and_model_commands_round_trip() {
    let harness = RpcHarness::start(0).await;

    let state = call(&harness, json!({ "id": "s1", "type": "get_state" })).await;
    assert_eq!(state["success"], true);
    assert!(
        state["data"]["thinkingLevel"].is_string(),
        "the session reports a thinking level: {state}"
    );
    assert_eq!(state["data"]["messageCount"], 0);
    assert_eq!(state["data"]["model"]["id"], "faux-1");
    assert!(
        state["data"].get("sessionFile").is_none(),
        "an in-memory session omits the file"
    );

    let models = call(
        &harness,
        json!({ "id": "m1", "type": "get_available_models" }),
    )
    .await;
    assert_eq!(
        models["data"]["models"].as_array().expect("models").len(),
        2
    );

    let selected = call(
        &harness,
        json!({ "id": "m2", "type": "set_model", "provider": "faux", "modelId": "faux-2" }),
    )
    .await;
    assert_eq!(selected["data"]["id"], "faux-2");

    let missing = call(
        &harness,
        json!({ "id": "m3", "type": "set_model", "provider": "nope", "modelId": "nope" }),
    )
    .await;
    assert_eq!(missing["success"], false);
    assert_eq!(missing["error"], "Model not found: nope/nope");

    let cycled = call(&harness, json!({ "id": "m4", "type": "cycle_model" })).await;
    assert_eq!(cycled["success"], true);
    assert!(cycled["data"]["model"]["id"].is_string());
    assert_eq!(cycled["data"]["isScoped"], false);

    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn thinking_and_queue_mode_commands_apply_and_read_back() {
    let harness = RpcHarness::start(0).await;

    let set = call(
        &harness,
        json!({ "id": "t1", "type": "set_thinking_level", "level": "high" }),
    )
    .await;
    assert_eq!(set["success"], true);
    assert!(set.get("data").is_none(), "the setter carries no data");

    let state = call(&harness, json!({ "id": "t2", "type": "get_state" })).await;
    assert_eq!(state["data"]["thinkingLevel"], "high");

    let levels = call(
        &harness,
        json!({ "id": "t3", "type": "get_available_thinking_levels" }),
    )
    .await;
    assert!(levels["data"]["levels"].is_array());

    let cycled = call(
        &harness,
        json!({ "id": "t4", "type": "cycle_thinking_level" }),
    )
    .await;
    assert_eq!(cycled["success"], true);

    let _ = call(
        &harness,
        json!({ "id": "q1", "type": "set_steering_mode", "mode": "one-at-a-time" }),
    )
    .await;
    let _ = call(
        &harness,
        json!({ "id": "q2", "type": "set_follow_up_mode", "mode": "all" }),
    )
    .await;
    let state = call(&harness, json!({ "id": "q3", "type": "get_state" })).await;
    assert_eq!(state["data"]["steeringMode"], "one-at-a-time");
    assert_eq!(state["data"]["followUpMode"], "all");

    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn toggle_and_abort_commands_answer() {
    let harness = RpcHarness::start(0).await;

    let _ = call(
        &harness,
        json!({ "id": "c1", "type": "set_auto_compaction", "enabled": false }),
    )
    .await;
    let _ = call(
        &harness,
        json!({ "id": "c2", "type": "set_auto_retry", "enabled": true }),
    )
    .await;
    assert_eq!(
        call(&harness, json!({ "id": "c3", "type": "abort_retry" })).await["success"],
        true
    );
    assert_eq!(
        call(&harness, json!({ "id": "c4", "type": "abort" })).await["success"],
        true
    );
    let state = call(&harness, json!({ "id": "c5", "type": "get_state" })).await;
    assert_eq!(state["data"]["autoCompactionEnabled"], false);

    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn session_inspection_commands_report_the_transcript() {
    let harness = RpcHarness::start(0).await;
    harness.send(json!({ "id": "p1", "type": "prompt", "message": "Hello" }));
    let _ = harness
        .wait_for(|lines| lines.iter().any(|line| line["type"] == "agent_settled"))
        .await;

    let stats = call(&harness, json!({ "id": "i1", "type": "get_session_stats" })).await;
    assert!(stats["data"]["sessionId"].is_string());
    assert!(stats["data"]["userMessages"].as_u64().expect("count") >= 1);
    assert!(stats["data"]["assistantMessages"].as_u64().expect("count") >= 1);

    let entries = call(&harness, json!({ "id": "i2", "type": "get_entries" })).await;
    let entry_ids: Vec<&str> = entries["data"]["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|entry| entry["id"].as_str().expect("ids"))
        .collect();
    assert!(entry_ids.len() >= 2);
    assert_eq!(
        entries["data"]["leafId"].as_str(),
        entry_ids.last().copied()
    );

    let since = call(
        &harness,
        json!({ "id": "i3", "type": "get_entries", "since": entry_ids[0] }),
    )
    .await;
    assert_eq!(
        since["data"]["entries"].as_array().expect("entries").len(),
        entry_ids.len() - 1
    );

    let missing = call(
        &harness,
        json!({ "id": "i4", "type": "get_entries", "since": "nope" }),
    )
    .await;
    assert_eq!(missing["success"], false);
    assert_eq!(missing["error"], "Entry not found: nope");

    let tree = call(&harness, json!({ "id": "i5", "type": "get_tree" })).await;
    assert_eq!(tree["data"]["tree"].as_array().expect("tree").len(), 1);
    assert_eq!(tree["data"]["leafId"].as_str(), entry_ids.last().copied());

    let messages = call(&harness, json!({ "id": "i6", "type": "get_messages" })).await;
    assert!(
        messages["data"]["messages"]
            .as_array()
            .expect("messages")
            .len()
            >= 2
    );

    let text = call(
        &harness,
        json!({ "id": "i7", "type": "get_last_assistant_text" }),
    )
    .await;
    assert_eq!(text["data"]["text"], "ok");

    let forks = call(&harness, json!({ "id": "i8", "type": "get_fork_messages" })).await;
    assert_eq!(forks["data"]["messages"][0]["text"], "Hello");

    let commands = call(&harness, json!({ "id": "i9", "type": "get_commands" })).await;
    assert!(commands["data"]["commands"].is_array());

    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn session_mutation_commands_replace_the_session() {
    let harness = RpcHarness::start(0).await;
    harness.send(json!({ "id": "p1", "type": "prompt", "message": "Hello" }));
    let _ = harness
        .wait_for(|lines| lines.iter().any(|line| line["type"] == "agent_settled"))
        .await;

    let named = call(
        &harness,
        json!({ "id": "n1", "type": "set_session_name", "name": "my-test-session" }),
    )
    .await;
    assert_eq!(named["success"], true);
    let state = call(&harness, json!({ "id": "n2", "type": "get_state" })).await;
    assert_eq!(state["data"]["sessionName"], "my-test-session");

    let empty = call(
        &harness,
        json!({ "id": "n3", "type": "set_session_name", "name": "   " }),
    )
    .await;
    assert_eq!(empty["success"], false);
    assert_eq!(empty["error"], "Session name cannot be empty");

    let forks = call(&harness, json!({ "id": "f1", "type": "get_fork_messages" })).await;
    let entry_id = forks["data"]["messages"][0]["entryId"]
        .as_str()
        .expect("the forking entry")
        .to_owned();
    let forked = call(
        &harness,
        json!({ "id": "f2", "type": "fork", "entryId": entry_id }),
    )
    .await;
    assert_eq!(forked["success"], true);
    assert_eq!(forked["data"]["cancelled"], false);
    assert_eq!(forked["data"]["text"], "Hello");

    let cloned = call(&harness, json!({ "id": "f3", "type": "clone" })).await;
    assert_eq!(cloned["success"], true);
    assert_eq!(cloned["data"]["cancelled"], false);

    let fresh = call(&harness, json!({ "id": "f4", "type": "new_session" })).await;
    assert_eq!(fresh["success"], true);
    assert_eq!(fresh["data"]["cancelled"], false);
    let state = call(&harness, json!({ "id": "f5", "type": "get_state" })).await;
    assert_eq!(state["data"]["messageCount"], 0);

    let no_leaf = call(&harness, json!({ "id": "f6", "type": "clone" })).await;
    assert_eq!(
        no_leaf["success"], true,
        "the fresh session's model-change entry is its leaf"
    );

    let switched = call(
        &harness,
        json!({ "id": "f7", "type": "switch_session", "sessionPath": "/nope/missing.jsonl" }),
    )
    .await;
    assert_eq!(switched["success"], false);

    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn bash_export_and_compaction_commands_run() {
    let harness = RpcHarness::start_persisted(0).await;

    let bash = call(
        &harness,
        json!({ "id": "b1", "type": "bash", "command": "echo hello" }),
    )
    .await;
    assert_eq!(bash["success"], true);
    assert_eq!(bash["data"]["exitCode"], 0);
    assert!(
        bash["data"]["output"]
            .as_str()
            .expect("output")
            .contains("hello")
    );

    assert_eq!(
        call(&harness, json!({ "id": "b2", "type": "abort_bash" })).await["success"],
        true
    );

    harness.send(json!({ "id": "p1", "type": "prompt", "message": "Hello" }));
    let _ = harness
        .wait_for(|lines| lines.iter().any(|line| line["type"] == "agent_settled"))
        .await;

    let exported = call(&harness, json!({ "id": "e1", "type": "export_html" })).await;
    assert_eq!(exported["success"], true);
    let path = exported["data"]["path"].as_str().expect("path");
    assert!(
        std::path::Path::new(path)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("html")),
        "the export writes HTML: {path}"
    );
    assert!(std::path::Path::new(path).exists());

    let compact = call(
        &harness,
        json!({ "id": "c1", "type": "compact", "customInstructions": "short" }),
    )
    .await;
    assert_eq!(compact["success"], false, "a tiny session cannot compact");
    assert_eq!(compact["error"], "Nothing to compact (session too small)");

    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn steer_and_follow_up_commands_queue_messages() {
    let release = std::sync::Arc::new(tokio::sync::Notify::new());
    let harness = RpcHarness::start_gated(std::sync::Arc::clone(&release)).await;
    harness.send(json!({ "id": "p1", "type": "prompt", "message": "Start" }));
    let _ = harness
        .wait_for(|lines| {
            lines.iter().any(|line| {
                line["type"] == "message_start" && line["message"]["role"] == "assistant"
            })
        })
        .await;

    let steered = call(
        &harness,
        json!({ "id": "q1", "type": "steer", "message": "Change direction" }),
    )
    .await;
    assert_eq!(steered["success"], true);
    let followed = call(
        &harness,
        json!({ "id": "q2", "type": "follow_up", "message": "Summarize when finished" }),
    )
    .await;
    assert_eq!(followed["success"], true);

    let cleared = call(&harness, json!({ "id": "q3", "type": "clear_queue" })).await;
    assert_eq!(
        cleared["data"],
        json!({
            "steering": ["Change direction"],
            "followUp": ["Summarize when finished"]
        })
    );
    release.notify_waiters();
    assert_eq!(harness.shutdown().await, 0);
}

#[tokio::test]
async fn a_malformed_command_reports_the_parse_detail() {
    let harness = RpcHarness::start(0).await;

    harness.send(json!({ "id": "bad", "type": "set_model" }));
    let lines = harness
        .wait_for(|lines| lines.iter().any(|line| line["id"] == "bad"))
        .await;
    let response = lines
        .iter()
        .find(|line| line["id"] == "bad")
        .expect("the error response landed");

    assert_eq!(response["success"], false);
    assert_eq!(response["command"], "set_model");
    assert!(
        response["error"]
            .as_str()
            .is_some_and(|error| error.contains("missing field")),
        "the deserialization detail rides the error: {response}"
    );
    assert_eq!(harness.shutdown().await, 0);
}
