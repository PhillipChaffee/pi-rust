//! `test/suite/regressions/6363-agent-settled-event.test.ts`, the retry and
//! follow-up blocks, at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_ai::types::UserContent;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::runner::DeliverAs;
use pi_coding_agent::extensions::runner::SendUserMessageOptions;
use pi_coding_agent::settings_manager::Settings;
use serde_json::json;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::agent_session::error_message;
use common::extension_regressions::ApiSlot;
use common::extension_regressions::manager;
use common::extension_regressions::scripted;
use common::extension_regressions::user_texts;

/// #6363: one `agent_settled` after the automatic retry finishes.
#[tokio::test]
async fn agent_settled_emits_once_after_retry() {
    let root = "/tmp/pi-ext-reg-settled-retry";
    let extension_events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&extension_events);
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let recorded = Arc::clone(&recorded);
            Box::pin(async move {
                let end_recorded = Arc::clone(&recorded);
                api.on(
                    "agent_end",
                    Arc::new(move |_event: ExtensionEvent, _ctx| {
                        end_recorded
                            .lock()
                            .expect("lock")
                            .push("agent_end".to_owned());
                        Box::pin(async { HandlerResult::None })
                    }),
                )
                .expect("agent_end handler");
                api.on(
                    "agent_settled",
                    Arc::new(move |_event: ExtensionEvent, ctx| {
                        let idle = ctx.is_idle().unwrap_or(false);
                        recorded
                            .lock()
                            .expect("lock")
                            .push(format!("agent_settled:{idle}"));
                        Box::pin(async { HandlerResult::None })
                    }),
                )
                .expect("agent_settled handler");
                Ok(())
            })
        }),
    };
    let public_ends: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let public_settled = Arc::new(AtomicUsize::new(0));
    let mut setup = SessionSetup::new(root, manager(root));
    setup.settings = Settings::from_iter([(
        "retry".to_owned(),
        json!({ "enabled": true, "maxRetries": 3, "baseDelayMs": 1 }),
    )]);
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    setup.stream = Some(scripted(vec![
        error_message("overloaded_error", 0),
        assistant_message("recovered", 0),
    ]));
    let session = build(setup).await;
    authorize(&session).await;
    let end_sink = Arc::clone(&public_ends);
    let settled_sink = Arc::clone(&public_settled);
    let _unsubscribe = session.subscribe(Arc::new(move |event| match event {
        AgentSessionEvent::AgentEnd { will_retry, .. } => {
            end_sink.lock().expect("lock").push(will_retry);
        }
        AgentSessionEvent::AgentSettled => {
            settled_sink.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    }));

    session
        .prompt("test".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(public_ends.lock().expect("lock").clone(), vec![true, false]);
    assert_eq!(public_settled.load(Ordering::SeqCst), 1);
    assert_eq!(
        extension_events.lock().expect("lock").clone(),
        [
            "agent_end".to_owned(),
            "agent_end".to_owned(),
            "agent_settled:true".to_owned(),
        ]
    );
}

/// #6363: settlement waits for `agent_end` follow-ups.
#[tokio::test]
async fn agent_settled_waits_for_agent_end_follow_ups() {
    let root = "/tmp/pi-ext-reg-settled-followup";
    let api_slot: ApiSlot = Arc::new(Mutex::new(None));
    let settled_states: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let queued = Arc::new(AtomicBool::new(false));
    let slot = Arc::clone(&api_slot);
    let states = Arc::clone(&settled_states);
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&slot);
            let states = Arc::clone(&states);
            let queued = Arc::clone(&queued);
            let follow_api = Arc::clone(&api);
            Box::pin(async move {
                api.on(
                    "agent_end",
                    Arc::new(move |_event: ExtensionEvent, _ctx| {
                        let follow_api = Arc::clone(&follow_api);
                        if !queued.swap(true, Ordering::SeqCst) {
                            follow_api
                                .send_user_message(
                                    UserContent::Text("status follow-up".to_owned()),
                                    Some(SendUserMessageOptions {
                                        deliver_as: Some(DeliverAs::FollowUp),
                                        expand_prompt_templates: None,
                                    }),
                                )
                                .expect("follow-up sends");
                        }
                        Box::pin(async {
                            tokio::task::yield_now().await;
                            HandlerResult::None
                        })
                    }),
                )
                .expect("agent_end handler");
                api.on(
                    "agent_settled",
                    Arc::new(move |_event: ExtensionEvent, ctx| {
                        states
                            .lock()
                            .expect("lock")
                            .push(ctx.is_idle().unwrap_or(false));
                        Box::pin(async { HandlerResult::None })
                    }),
                )
                .expect("agent_settled handler");
                *slot.lock().expect("lock") = Some(api);
                Ok(())
            })
        }),
    };
    let public_ends = Arc::new(AtomicUsize::new(0));
    let public_settled = Arc::new(AtomicUsize::new(0));
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    setup.stream = Some(scripted(vec![
        assistant_message("first", 0),
        assistant_message("second", 0),
    ]));
    let session = build(setup).await;
    authorize(&session).await;
    let ends = Arc::clone(&public_ends);
    let settled = Arc::clone(&public_settled);
    let _unsubscribe = session.subscribe(Arc::new(move |event| match event {
        AgentSessionEvent::AgentEnd { .. } => {
            ends.fetch_add(1, Ordering::SeqCst);
        }
        AgentSessionEvent::AgentSettled => {
            settled.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    }));

    session
        .prompt("hello".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(
        user_texts(&session),
        ["hello".to_owned(), "status follow-up".to_owned()]
    );
    assert_eq!(public_ends.load(Ordering::SeqCst), 2);
    assert_eq!(public_settled.load(Ordering::SeqCst), 1);
    assert_eq!(settled_states.lock().expect("lock").clone(), vec![true]);
}
