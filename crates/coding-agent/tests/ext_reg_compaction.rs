//! `test/suite/regressions/5217-compaction-reason.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use pi_ai::types::Usage;
use pi_coding_agent::compaction::CompactionResult;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::reasons::CompactionReason;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::results::SessionBeforeResult;
use pi_coding_agent::settings_manager::Settings;
use serde_json::json;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::Session;
use common::extension_regressions::manager;
use common::extension_regressions::scripted;

#[derive(Clone, Debug, PartialEq)]
struct Recorded {
    event: &'static str,
    reason: CompactionReason,
    will_retry: bool,
}

fn recording_extension(recorded: Arc<Mutex<Vec<Recorded>>>) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let recorded = Arc::clone(&recorded);
            Box::pin(async move {
                let compact_recorded = Arc::clone(&recorded);
                api.on(
                    "session_before_compact",
                    Arc::new(move |event: ExtensionEvent, _ctx| {
                        let ExtensionEvent::SessionBeforeCompact {
                            preparation,
                            reason,
                            will_retry,
                            ..
                        } = event
                        else {
                            return Box::pin(async { HandlerResult::None });
                        };
                        compact_recorded.lock().expect("lock").push(Recorded {
                            event: "session_before_compact",
                            reason,
                            will_retry,
                        });
                        Box::pin(async move {
                            HandlerResult::SessionBefore(SessionBeforeResult::Compact {
                                cancel: false,
                                compaction: Some(CompactionResult {
                                    summary: "summary from extension".to_owned(),
                                    first_kept_entry_id: preparation.first_kept_entry_id,
                                    tokens_before: preparation.tokens_before,
                                    estimated_tokens_after: None,
                                    usage: Usage::default(),
                                    details: Some(json!({})),
                                }),
                            })
                        })
                    }),
                )
                .expect("session_before_compact handler");
                api.on(
                    "session_compact",
                    Arc::new(move |event: ExtensionEvent, _ctx| {
                        if let ExtensionEvent::SessionCompact {
                            reason, will_retry, ..
                        } = event
                        {
                            recorded.lock().expect("lock").push(Recorded {
                                event: "session_compact",
                                reason,
                                will_retry,
                            });
                        }
                        Box::pin(async { HandlerResult::None })
                    }),
                )
                .expect("session_compact handler");
                Ok(())
            })
        }),
    }
}

async fn compaction_harness(root: &str, recorded: Arc<Mutex<Vec<Recorded>>>) -> Session {
    let mut setup = SessionSetup::new(root, manager(root));
    setup.settings =
        Settings::from_iter([("compaction".to_owned(), json!({"keepRecentTokens": 1}))]);
    setup.extensions = vec![recording_extension(recorded)];
    setup.bind_extensions = true;
    setup.stream = Some(scripted(vec![
        assistant_message("one", 0),
        assistant_message("two", 0),
    ]));
    let session = build(setup).await;
    authorize(&session).await;
    session
        .prompt("first".to_owned(), None)
        .await
        .expect("first");
    session
        .prompt("second".to_owned(), None)
        .await
        .expect("second");
    session
}

fn expected(reason: CompactionReason, will_retry: bool) -> Vec<Recorded> {
    vec![
        Recorded {
            event: "session_before_compact",
            reason,
            will_retry,
        },
        Recorded {
            event: "session_compact",
            reason,
            will_retry,
        },
    ]
}

/// #5217: `compact()` reports the manual reason.
#[tokio::test]
async fn compaction_events_report_manual_reason() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let session = compaction_harness("/tmp/pi-ext-reg-compact-manual", Arc::clone(&recorded)).await;

    session.compact(None).await.expect("compact");

    assert_eq!(
        recorded.lock().expect("lock").clone(),
        expected(CompactionReason::Manual, false)
    );
}

/// #5217: automatic compaction reports the threshold reason.
#[tokio::test]
async fn compaction_events_report_threshold_reason() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let session =
        compaction_harness("/tmp/pi-ext-reg-compact-threshold", Arc::clone(&recorded)).await;

    session
        .run_auto_compaction(CompactionReason::Threshold, false)
        .await;

    assert_eq!(
        recorded.lock().expect("lock").clone(),
        expected(CompactionReason::Threshold, false)
    );
}

/// #5217: overflow recovery reports its reason and retry flag.
#[tokio::test]
async fn compaction_events_report_overflow_reason() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let session =
        compaction_harness("/tmp/pi-ext-reg-compact-overflow", Arc::clone(&recorded)).await;

    session
        .run_auto_compaction(CompactionReason::Overflow, true)
        .await;

    assert_eq!(
        recorded.lock().expect("lock").clone(),
        expected(CompactionReason::Overflow, true)
    );
}
