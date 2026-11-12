//! `test/interactive-mode-tree-navigation.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements:
//! - Upstream drives the selector's `onSelect` and mocks
//!   `showExtensionSelector`; the port's selector callback sends the same
//!   navigation command, so the tests call `navigate_tree`, the handler that
//!   command runs, and answer the real extension selector by keyboard.
//! - `session.isStreaming`/`isCompacting` are real session state: a gated
//!   faux stream holds the run open, and a real `compact` call holds the
//!   compaction open.
//! - The `showError`/`showStatusIndicator`/`clearStatusIndicator` spies become
//!   the transcript's error row and the active indicator kind.
//! - The `defaultEditor.onEscape` identity check uses the port's
//!   `AppEditor::on_escape` getter and `Rc::ptr_eq`.

#![expect(
    clippy::expect_used,
    reason = "the fixture session is pinned; a rejected append is a test bug"
)]

mod common;

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use common::interactive::{Harness, HarnessOptions};
use pi_coding_agent::session_manager::SessionManager;

const BUSY: &str = "Wait for the current compaction or tree navigation to finish before navigating the session tree.";

/// A frame with `first` (user) -> `reply` (assistant), plus the target id.
async fn tree_harness(options: HarnessOptions) -> (Harness, String) {
    let target: Arc<std::sync::Mutex<Option<String>>> = Arc::new(std::sync::Mutex::new(None));
    let target_seed = Arc::clone(&target);
    let mut options = options;
    // A small context window and a long first message make the real
    // compaction path find something to summarize.
    options.model.context_window = 1_000;
    options.model.max_tokens = 100;
    options.settings.insert(
        "compaction".to_owned(),
        serde_json::json!({ "reserveTokens": 10, "keepRecentTokens": 1 }),
    );
    options.seed_manager = Some(Box::new(move |manager: &mut SessionManager| {
        let id = manager
            .append_message(common::agent_session::user_message(
                &"first ".repeat(600),
                1,
            ))
            .expect("user");
        *target_seed.lock().expect("target") = Some(id);
        manager
            .append_message(pi_agent_core::types::AgentMessage::Standard(
                pi_ai::types::Message::Assistant(common::agent_session::assistant_message(
                    "reply", 2,
                )),
            ))
            .expect("assistant");
        manager
            .append_message(common::agent_session::user_message("second", 3))
            .expect("second user");
        manager
            .append_message(pi_agent_core::types::AgentMessage::Standard(
                pi_ai::types::Message::Assistant(common::agent_session::assistant_message(
                    "second reply",
                    4,
                )),
            ))
            .expect("second assistant");
    }));
    let harness = Harness::new(options).await;
    harness.frame.tui().start();
    harness
        .frame
        .editor()
        .set_on_escape(Some(std::rc::Rc::new(|| {})));
    let target = target.lock().expect("target").clone().expect("target id");
    (harness, target)
}

/// Answer the active extension selector: `index` down-arrows then enter.
fn answer_selector(harness: &Harness, index: usize) {
    let selector = harness
        .frame
        .extension_selector()
        .expect("the summary selector is mounted");
    for _ in 0..index {
        selector.handle_input("\x1b[B");
    }
    selector.handle_input("\r");
}

/// Poll `future` until it completes or `millis` elapses.
async fn wait_for<F: Future>(future: std::pin::Pin<&mut F>, millis: u64) {
    tokio::select! {
        () = async { let _ = future.await; } => {}
        () = tokio::time::sleep(Duration::from_millis(millis)) => {}
    }
}

#[tokio::test]
async fn preserves_operation_ui_when_choosing_summarize_while_busy() {
    for choice in ["Summarize", "No summary"] {
        let release = Arc::new(tokio::sync::Notify::new());
        let (harness, target) = tree_harness(HarnessOptions {
            stream: Some(gated_stream(Arc::clone(&release))),
            ..HarnessOptions::default()
        })
        .await;
        let leaf = harness.frame.session().leaf_id();
        let escape = harness.frame.editor().on_escape();
        let navigate = harness.frame.navigate_tree(&target);
        tokio::pin!(navigate);
        wait_for(navigate.as_mut(), 30).await;
        answer_selector(&harness, usize::from(choice == "Summarize"));
        // Compaction starts while the dialog closes, the race the recheck
        // guards.
        let session = harness.frame.session();
        let compaction = tokio::spawn({
            let session = Arc::clone(&session);
            async move { session.compact(None).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        wait_for(navigate.as_mut(), 50).await;

        let text = harness.chat_text();
        assert!(text.contains(BUSY), "{choice}: {text}");
        assert_ne!(
            harness.frame.active_status_kind(),
            Some(
                pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind::BranchSummary
            )
        );
        assert_eq!(harness.frame.session().leaf_id(), leaf);
        assert!(text.contains(BUSY), "{choice}: {text}");
        // The operation's own compaction handler is still installed.
        assert!(
            harness.frame.editor().on_escape().is_some(),
            "{choice}: the compaction escape handler is present"
        );
        session.abort_compaction();
        release.notify_waiters();
        let _ = compaction.await;
        harness.frame.drain_events();
        if let Some(escape) = escape {
            assert!(std::rc::Rc::ptr_eq(
                &escape,
                &harness.frame.editor().on_escape().expect("escape handler")
            ));
        }
    }
}

#[tokio::test]
async fn allows_navigation_when_compaction_finishes_while_the_dialog_is_open() {
    let release = Arc::new(tokio::sync::Notify::new());
    let (harness, target) = tree_harness(HarnessOptions {
        stream: Some(gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    let old_leaf = harness.frame.session().leaf_id();
    let session = harness.frame.session();
    let compaction = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.compact(None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(session.is_compacting());

    let navigate = harness.frame.navigate_tree(&target);
    tokio::pin!(navigate);
    wait_for(navigate.as_mut(), 30).await;
    // The compaction finishes while the dialog is open.
    let release_after = Arc::clone(&release);
    tokio::spawn(async move {
        for _ in 0..20 {
            release_after.notify_waiters();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(!session.is_compacting());
    answer_selector(&harness, 0);
    wait_for(navigate.as_mut(), 100).await;

    let text = harness.chat_text();
    assert!(!text.contains(BUSY), "{text}");
    assert!(text.contains("Navigated to selected point"), "{text}");
    assert_ne!(harness.frame.session().leaf_id(), old_leaf);
    session.abort_compaction();
    let _ = compaction.await;
    release.notify_waiters();
}

#[tokio::test]
async fn still_aborts_an_active_response_before_navigating() {
    let release = Arc::new(tokio::sync::Notify::new());
    let (harness, target) = tree_harness(HarnessOptions {
        stream: Some(gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    let session = harness.frame.session();
    let prompt = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.prompt("first".to_owned(), None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    session
        .steer("queued".to_owned(), None, None)
        .await
        .expect("steer");

    let navigate = harness.frame.navigate_tree(&target);
    tokio::pin!(navigate);
    wait_for(navigate.as_mut(), 30).await;
    answer_selector(&harness, 0);
    let release_after = Arc::clone(&release);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        release_after.notify_one();
    });
    wait_for(navigate.as_mut(), 300).await;

    assert!(!session.is_streaming());
    assert_eq!(harness.frame.editor().get_text(), "queued");
    assert!(harness.chat_text().contains("Navigated to selected point"));
    assert!(!harness.chat_text().contains(BUSY));
    release.notify_one();
    let _ = prompt.await;
}

#[tokio::test]
async fn rechecks_availability_after_the_response_abort_settles() {
    let release = Arc::new(tokio::sync::Notify::new());
    let (harness, target) = tree_harness(HarnessOptions {
        stream: Some(gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    let session = harness.frame.session();
    let escape = harness.frame.editor().on_escape();
    let prompt = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.prompt("first".to_owned(), None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    // The response settles before the dialog, the state the recheck sees
    // after the abort branch has run.
    release.notify_one();
    let _ = prompt.await;
    let leaf = session.leaf_id();

    // A compaction is active once the response has settled.
    let compaction = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.compact(None).await }
    });
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(session.is_compacting());
    let navigate = harness.frame.navigate_tree(&target);
    tokio::pin!(navigate);
    wait_for(navigate.as_mut(), 30).await;
    assert!(session.is_compacting(), "the compaction is still active");
    answer_selector(&harness, 1);
    wait_for(navigate.as_mut(), 100).await;

    let text = harness.chat_text();
    assert!(text.contains(BUSY), "{text}");
    assert_ne!(
        harness.frame.active_status_kind(),
        Some(
            pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind::BranchSummary
        )
    );
    assert_eq!(harness.frame.session().leaf_id(), leaf);
    session.abort_compaction();
    release.notify_waiters();
    let _ = compaction.await;
    harness.frame.drain_events();
    assert!(std::rc::Rc::ptr_eq(
        &escape.expect("escape handler"),
        &harness.frame.editor().on_escape().expect("escape handler")
    ));
}

/// A stream that opens immediately and settles only when `release` notifies.
fn gated_stream(release: Arc<tokio::sync::Notify>) -> pi_agent_core::types::StreamFn {
    use pi_ai::types::{AssistantMessageEvent, StopReason};
    Arc::new(move |model, _context, _options| {
        let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
        let mut message = common::sdk_fixture::done_message();
        message.provider = model.provider.clone();
        message.model.clone_from(&model.id);
        stream.push(AssistantMessageEvent::Start {
            partial: pi_ai::types::AssistantMessage {
                content: Vec::new(),
                ..message.clone()
            },
        });
        let settle = stream.clone();
        let release = Arc::clone(&release);
        tokio::spawn(async move {
            release.notified().await;
            settle.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            });
        });
        stream
    })
}
