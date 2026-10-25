//! `test/rpc-client-clear-queue.test.ts`, `test/rpc-client-clone.test.ts`,
//! and `test/rpc-client-process-exit.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream fakes the private `send`/`getData` pair; the port drives a real
//! child process through the wire, so each case also pins the framing.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

use std::os::unix::fs::PermissionsExt;
use std::time::Duration;

use pi_coding_agent::modes::rpc::rpc_client::RpcClient;
use pi_coding_agent::modes::rpc::rpc_client::RpcClientOptions;
use serde_json::json;
use tempfile::TempDir;

/// Write an executable shell script and return its path.
fn child_script(contents: &str) -> (TempDir, String) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("child.sh");
    std::fs::write(&path, contents).expect("script");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    (dir, path.display().to_string())
}

fn client_for(path: &str) -> RpcClient {
    RpcClient::new(RpcClientOptions {
        cli_path: Some(path.to_owned()),
        ..RpcClientOptions::default()
    })
}

#[tokio::test]
async fn clear_queue_sends_the_command_and_returns_its_data() {
    let (_dir, path) = child_script(
        r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *clear_queue*)
      printf '%s\n' '{"type":"response","id":"req_1","command":"clear_queue","success":true,"data":{"steering":["Change direction"],"followUp":["Summarize when finished"]}}'
      ;;
  esac
done
"#,
    );
    let mut client = client_for(&path);
    client.start().await.expect("start");

    let result = client.clear_queue().await.expect("clear_queue");

    assert_eq!(
        result,
        json!({ "steering": ["Change direction"], "followUp": ["Summarize when finished"] })
    );
    client.stop().await;
}

#[tokio::test]
async fn clone_sends_the_command_and_returns_its_data() {
    let (_dir, path) = child_script(
        r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *clone*)
      printf '%s\n' '{"type":"response","id":"req_1","command":"clone","success":true,"data":{"cancelled":false}}'
      ;;
  esac
done
"#,
    );
    let mut client = client_for(&path);
    client.start().await.expect("start");

    let result = client.clone().await.expect("clone");

    assert_eq!(result, json!({ "cancelled": false }));
    client.stop().await;
}

#[tokio::test]
async fn an_in_flight_request_rejects_when_the_child_process_exits() {
    let (_dir, path) = child_script(
        "#!/bin/sh
read -r line
exit 43
",
    );
    let mut client = client_for(&path);
    client.start().await.expect("start");

    let error = client
        .get_commands()
        .await
        .expect_err("the request rejects");

    assert!(
        error.contains("Agent process exited (code=43 signal=null)"),
        "the exit message carries the code and stderr: {error}"
    );
    client.stop().await;
}

#[tokio::test]
async fn events_and_responses_share_the_wire_without_colliding() {
    let (_dir, path) = child_script(
        r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *get_state*)
      printf '%s\n' '{"type":"agent_start"}'
      printf '%s\n' '{"type":"response","id":"req_1","command":"get_state","success":true,"data":{"sessionId":"s1","thinkingLevel":"off","isStreaming":false,"isCompacting":false,"steeringMode":"all","followUpMode":"all","autoCompactionEnabled":true,"messageCount":0,"pendingMessageCount":0}}'
      ;;
  esac
done
"#,
    );
    let mut client = client_for(&path);
    client.start().await.expect("start");
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&seen);
    let unsubscribe = client.on_event(std::sync::Arc::new(move |event| {
        sink.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
    }));

    let state = client.get_state().await.expect("get_state");

    assert_eq!(state["sessionId"], "s1");
    let events = seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["type"], "agent_start");
    unsubscribe();
    client.stop().await;
}

#[tokio::test]
async fn an_error_response_rejects_with_its_message() {
    let (_dir, path) = child_script(
        r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *get_tree*)
      printf '%s\n' '{"type":"response","id":"req_1","command":"get_tree","success":false,"error":"boom"}'
      ;;
  esac
done
"#,
    );
    let mut client = client_for(&path);
    client.start().await.expect("start");

    let error = client.get_tree().await.expect_err("the error arm rejects");

    assert_eq!(error, "boom");
    client.stop().await;
}

#[tokio::test]
async fn stop_terminates_the_child() {
    let (_dir, path) = child_script("#!/bin/sh\nwhile true; do sleep 1; done\n");
    let mut client = client_for(&path);
    client.start().await.expect("start");

    client.stop().await;

    assert!(
        client.get_state().await.is_err(),
        "a stopped client refuses commands"
    );
}

#[tokio::test]
async fn every_command_method_round_trips() {
    let (_dir, path) = child_script(
        "#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p')
  printf '{\"type\":\"response\",\"id\":\"%s\",\"command\":\"x\",\"success\":true,\"data\":{}}\\n' \"$id\"
done
",
    );
    let mut client = client_for(&path);
    client.start().await.expect("start");

    client.prompt("m", None).await.expect("prompt");
    client.steer("m", None).await.expect("steer");
    client.follow_up("m", None).await.expect("follow_up");
    client.abort().await.expect("abort");
    client.clear_queue().await.expect("clear_queue");
    client.new_session(None).await.expect("new_session");
    client.get_state().await.expect("get_state");
    client.set_model("p", "m").await.expect("set_model");
    client.cycle_model().await.expect("cycle_model");
    client
        .get_available_models()
        .await
        .expect("get_available_models");
    client
        .set_thinking_level("high")
        .await
        .expect("set_thinking_level");
    client
        .cycle_thinking_level()
        .await
        .expect("cycle_thinking_level");
    client
        .get_available_thinking_levels()
        .await
        .expect("get_available_thinking_levels");
    client
        .set_steering_mode("all")
        .await
        .expect("set_steering_mode");
    client
        .set_follow_up_mode("all")
        .await
        .expect("set_follow_up_mode");
    client.compact(None).await.expect("compact");
    client
        .set_auto_compaction(true)
        .await
        .expect("set_auto_compaction");
    client.set_auto_retry(true).await.expect("set_auto_retry");
    client.abort_retry().await.expect("abort_retry");
    client.bash("echo hi").await.expect("bash");
    client.abort_bash().await.expect("abort_bash");
    client.get_session_stats().await.expect("get_session_stats");
    client.export_html(None).await.expect("export_html");
    client
        .switch_session("/s.jsonl")
        .await
        .expect("switch_session");
    client.fork("e1").await.expect("fork");
    client.get_fork_messages().await.expect("get_fork_messages");
    client.get_entries(None).await.expect("get_entries");
    client.get_tree().await.expect("get_tree");
    client
        .get_last_assistant_text()
        .await
        .expect("get_last_assistant_text");
    client
        .set_session_name("n")
        .await
        .expect("set_session_name");
    client.get_messages().await.expect("get_messages");
    client.get_commands().await.expect("get_commands");
    client.stop().await;
}

#[tokio::test]
async fn the_idle_helpers_resolve_on_agent_settled() {
    let script = "#!/bin/sh\nwhile true; do printf '%s\\n' '{\"type\":\"agent_settled\"}'; sleep 0.1; done\n";
    let (_dir, path) = child_script(script);
    let mut client = client_for(&path);
    client.start().await.expect("start");

    client
        .wait_for_idle(Duration::from_secs(1))
        .await
        .expect("wait_for_idle");
    client.stop().await;

    let (_dir, path) = child_script(script);
    let mut client = client_for(&path);
    client.start().await.expect("start");
    let events = client
        .collect_events(Duration::from_secs(1))
        .await
        .expect("collect_events");
    assert!(
        events.iter().any(|event| event["type"] == "agent_settled"),
        "the collection settles on the event: {events:?}"
    );
    assert_eq!(events.last().expect("events")["type"], "agent_settled");
    client.stop().await;
}

#[tokio::test]
async fn prompt_and_wait_collects_until_settled() {
    let (_dir, path) = child_script(
        "#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p')
  printf '{\"type\":\"response\",\"id\":\"%s\",\"command\":\"prompt\",\"success\":true}\\n' \"$id\"
  printf '%s\\n' '{\"type\":\"agent_settled\"}'
done
",
    );
    let mut client = client_for(&path);
    client.start().await.expect("start");

    let events = client
        .prompt_and_wait("hello", None, Duration::from_secs(1))
        .await
        .expect("prompt_and_wait");

    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["type"], "agent_settled");
    client.stop().await;
}
