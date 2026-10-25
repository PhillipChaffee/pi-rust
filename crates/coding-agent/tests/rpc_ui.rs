//! The RPC mode's extension UI surface, upstream's `createExtensionUIContext`
//! inside `rpc-mode.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use common::rpc::CapturedStdout;
use common::rpc::RpcHarness;
use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::context::with_cancel;
use pi_coding_agent::modes::rpc::rpc_mode::DialogOptions;
use pi_coding_agent::modes::rpc::rpc_mode::RpcUiContext;
use pi_coding_agent::modes::rpc::rpc_types::RpcExtensionUiResponse;
use pi_coding_agent::modes::rpc::rpc_types::RpcNotifyType;
use pi_coding_agent::modes::rpc::rpc_types::RpcWidgetPlacement;
use serde_json::Value;
use serde_json::json;

/// Poll the capture for one request with `method`.
async fn wait_for_request(captured: &CapturedStdout, method: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(request) = captured
            .parsed()
            .into_iter()
            .find(|line| line["type"] == "extension_ui_request" && line["method"] == method)
        {
            return request;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {method}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn reply(
    id: &str,
    value: Option<&str>,
    confirmed: Option<bool>,
    cancelled: bool,
) -> RpcExtensionUiResponse {
    RpcExtensionUiResponse {
        id: id.to_owned(),
        value: value.map(str::to_owned),
        confirmed,
        cancelled: cancelled.then_some(true),
    }
}

#[tokio::test]
async fn select_emits_the_request_and_resolves_the_reply() {
    let captured = CapturedStdout::new();
    let ui = RpcUiContext::new();
    let select = tokio::spawn({
        let ui = Arc::clone(&ui);
        async move {
            ui.select(
                "Pick".to_owned(),
                vec!["a".to_owned(), "b".to_owned()],
                None,
            )
            .await
        }
    });

    let request = wait_for_request(&captured, "select").await;
    assert_eq!(request["title"], "Pick");
    assert_eq!(request["options"], json!(["a", "b"]));
    assert!(request.get("timeout").is_none());

    let id = request["id"].as_str().expect("the request id");
    ui.handle_response(reply(id, Some("b"), None, false));

    assert_eq!(select.await.expect("select task"), Some("b".to_owned()));
}

#[tokio::test]
async fn cancel_and_absent_values_read_as_the_defaults() {
    let captured = CapturedStdout::new();
    let ui = RpcUiContext::new();

    let select = tokio::spawn({
        let ui = Arc::clone(&ui);
        async move { ui.select("Pick".to_owned(), Vec::new(), None).await }
    });
    let request = wait_for_request(&captured, "select").await;
    let id = request["id"].as_str().expect("id").to_owned();
    ui.handle_response(reply(&id, None, None, true));
    assert_eq!(select.await.expect("select"), None);

    let confirm = tokio::spawn({
        let ui = Arc::clone(&ui);
        async move {
            ui.confirm("Keep".to_owned(), "Sure?".to_owned(), None)
                .await
        }
    });
    let request = wait_for_request(&captured, "confirm").await;
    let id = request["id"].as_str().expect("id").to_owned();
    ui.handle_response(reply(&id, None, Some(true), false));
    assert!(confirm.await.expect("confirm"));

    let input = tokio::spawn({
        let ui = Arc::clone(&ui);
        async move {
            ui.input("Name".to_owned(), Some("type".to_owned()), None)
                .await
        }
    });
    let request = wait_for_request(&captured, "input").await;
    assert_eq!(request["placeholder"], "type");
    let id = request["id"].as_str().expect("id").to_owned();
    ui.handle_response(reply(&id, None, None, true));
    assert_eq!(input.await.expect("input"), None);

    let editor = tokio::spawn({
        let ui = Arc::clone(&ui);
        async move {
            ui.editor("Edit".to_owned(), Some("prefill".to_owned()))
                .await
        }
    });
    let request = wait_for_request(&captured, "editor").await;
    assert_eq!(request["prefill"], "prefill");
    let id = request["id"].as_str().expect("id").to_owned();
    ui.handle_response(reply(&id, Some("text"), None, false));
    assert_eq!(editor.await.expect("editor"), Some("text".to_owned()));
}

#[tokio::test]
async fn a_timeout_and_a_pre_aborted_signal_resolve_the_default() {
    let captured = CapturedStdout::new();
    let ui = RpcUiContext::new();

    let timed_out = ui
        .select(
            "Pick".to_owned(),
            Vec::new(),
            Some(DialogOptions {
                signal: None,
                timeout: Some(10),
            }),
        )
        .await;
    assert_eq!(timed_out, None);

    let (_ctx, controller) = with_cancel(&background_context());
    controller.abort_without_reason();
    let signal = controller.signal().clone();
    let aborted = ui
        .confirm(
            "Keep".to_owned(),
            "Sure?".to_owned(),
            Some(DialogOptions {
                signal: Some(signal),
                timeout: None,
            }),
        )
        .await;
    assert!(!aborted);
    assert!(
        captured
            .parsed()
            .iter()
            .all(|line| line["method"] != "confirm"),
        "an aborted dialog emits nothing"
    );
}

#[tokio::test]
async fn fire_and_forget_ui_frames_match_the_wire() {
    let captured = CapturedStdout::new();
    let ui = RpcUiContext::new();

    ui.notify("hi", Some(RpcNotifyType::Warning));
    let notify = wait_for_request(&captured, "notify").await;
    assert_eq!(notify["message"], "hi");
    assert_eq!(notify["notifyType"], "warning");

    ui.set_status("k", None);
    let status = wait_for_request(&captured, "setStatus").await;
    assert_eq!(status["statusKey"], "k");
    assert_eq!(status["statusText"], Value::Null);

    ui.set_widget(
        "w",
        Some(vec!["line".to_owned()]),
        Some(RpcWidgetPlacement::BelowEditor),
    );
    let widget = wait_for_request(&captured, "setWidget").await;
    assert_eq!(widget["widgetLines"], json!(["line"]));
    assert_eq!(widget["widgetPlacement"], "belowEditor");

    ui.set_title("title");
    assert_eq!(
        wait_for_request(&captured, "setTitle").await["title"],
        "title"
    );

    ui.set_editor_text("text");
    assert_eq!(
        wait_for_request(&captured, "set_editor_text").await["text"],
        "text"
    );

    ui.paste_to_editor("pasted");
    assert_eq!(
        wait_for_request(&captured, "set_editor_text").await["text"],
        "pasted"
    );
}

#[tokio::test]
async fn the_mode_ignores_ui_replies_with_no_pending_request() {
    let harness = RpcHarness::start(0).await;

    harness.send(json!({
        "type": "extension_ui_response",
        "id": "unknown",
        "value": "x"
    }));

    let state = {
        harness.send(json!({ "id": "s1", "type": "get_state" }));
        harness
            .wait_for(|lines| lines.iter().any(|line| line["id"] == "s1"))
            .await
    };
    assert_eq!(state.len(), 1, "the stray reply emitted nothing");
    assert_eq!(harness.shutdown().await, 0);
}
