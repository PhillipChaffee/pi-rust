//! The session runtime's replacement flows and lifecycle events, upstream's
//! `test/suite/agent-session-runtime.test.ts`,
//! `test/suite/regressions/2860-replaced-session-context.test.ts`, and
//! `test/agent-session-runtime-events.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use pi_coding_agent::core::agent_session::ExtensionBindings;
use pi_coding_agent::core::agent_session_runtime::AgentSessionRuntimeError;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeOptions;
use pi_coding_agent::core::agent_session_runtime::ForkOptions;
use pi_coding_agent::core::agent_session_runtime::NewSessionRequest;
use pi_coding_agent::core::agent_session_runtime::SwitchSessionOptions;
use pi_coding_agent::core::agent_session_runtime::create_agent_session_runtime;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::reasons::ForkPosition;
use pi_coding_agent::extensions::reasons::SessionShutdownReason;
use pi_coding_agent::extensions::reasons::SessionStartReason;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::results::SessionBeforeResult;
use pi_coding_agent::extensions::runtime::FlagType;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use tempfile::TempDir;

#[derive(Default)]
struct Cancels {
    switch: AtomicBool,
    fork: AtomicBool,
}

/// The event payloads the runtime-lifecycle suite asserts on.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Recorded {
    Start {
        reason: SessionStartReason,
        previous: Option<String>,
    },
    Shutdown {
        reason: SessionShutdownReason,
        target: Option<String>,
    },
    BeforeSwitch {
        reason: SessionShutdownReason,
        target: Option<String>,
    },
    BeforeFork {
        entry_id: String,
        position: ForkPosition,
    },
}

#[expect(
    clippy::too_many_lines,
    reason = "the harness registers every lifecycle channel in one factory, upstream's inline extension"
)]
fn event_factory(
    log: Arc<Mutex<Vec<String>>>,
    cancels: Arc<Cancels>,
    detail: Arc<Mutex<Vec<Recorded>>>,
) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let log = Arc::clone(&log);
            let cancels = Arc::clone(&cancels);
            let detail = Arc::clone(&detail);
            Box::pin(async move {
                let switch_log = Arc::clone(&log);
                let switch_cancels = Arc::clone(&cancels);
                let switch_detail = Arc::clone(&detail);
                api.on(
                    "session_before_switch",
                    Arc::new(move |event, _ctx| {
                        switch_log
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push("before_switch".to_owned());
                        if let ExtensionEvent::SessionBeforeSwitch {
                            reason,
                            target_session_file,
                        } = event
                        {
                            switch_detail
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push(Recorded::BeforeSwitch {
                                    reason,
                                    target: target_session_file,
                                });
                        }
                        let cancel = switch_cancels.switch.load(Ordering::SeqCst);
                        Box::pin(async move {
                            HandlerResult::SessionBefore(SessionBeforeResult::Switch { cancel })
                        })
                    }),
                )
                .expect("switch handler");
                let fork_log = Arc::clone(&log);
                let fork_cancels = Arc::clone(&cancels);
                let fork_detail = Arc::clone(&detail);
                api.on(
                    "session_before_fork",
                    Arc::new(move |event, _ctx| {
                        fork_log
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push("before_fork".to_owned());
                        if let ExtensionEvent::SessionBeforeFork { entry_id, position } = event {
                            fork_detail
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push(Recorded::BeforeFork { entry_id, position });
                        }
                        let cancel = fork_cancels.fork.load(Ordering::SeqCst);
                        Box::pin(async move {
                            HandlerResult::SessionBefore(SessionBeforeResult::Fork {
                                cancel,
                                skip_conversation_restore: false,
                            })
                        })
                    }),
                )
                .expect("fork handler");
                for channel in ["session_start", "session_shutdown"] {
                    let label = channel.to_owned();
                    let log = Arc::clone(&log);
                    let detail = Arc::clone(&detail);
                    api.on(
                        channel,
                        Arc::new(move |event: ExtensionEvent, _ctx| {
                            let name = match &event {
                                ExtensionEvent::SessionStart { .. } => "start",
                                ExtensionEvent::SessionShutdown { .. } => "shutdown",
                                _ => "other",
                            };
                            log.lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push(name.to_owned());
                            let _ = &label;
                            match event {
                                ExtensionEvent::SessionStart {
                                    reason,
                                    previous_session_file,
                                } => detail.lock().unwrap_or_else(PoisonError::into_inner).push(
                                    Recorded::Start {
                                        reason,
                                        previous: previous_session_file,
                                    },
                                ),
                                ExtensionEvent::SessionShutdown {
                                    reason,
                                    target_session_file,
                                } => detail.lock().unwrap_or_else(PoisonError::into_inner).push(
                                    Recorded::Shutdown {
                                        reason,
                                        target: target_session_file,
                                    },
                                ),
                                _ => {}
                            }
                            Box::pin(async { HandlerResult::None })
                        }),
                    )
                    .expect("event handler");
                }
                api.register_flag("demo_flag".to_owned(), None, FlagType::Boolean, None)
                    .expect("flag");
                Ok(())
            })
        }),
    }
}

struct Harness {
    runtime:
        pi_coding_agent::core::agent_session_runtime::AgentSessionRuntime<InMemorySettingsStorage>,
    log: Arc<Mutex<Vec<String>>>,
    detail: Arc<Mutex<Vec<Recorded>>>,
    cancels: Arc<Cancels>,
    _temp: TempDir,
    cwd: String,
}

async fn harness_with(build: impl FnOnce(&str) -> SessionManager) -> Harness {
    let fixture = common::sdk_fixture::Fixture::new();
    let cwd = fixture.cwd.clone();
    let agent_dir = fixture.agent_dir.clone();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let model = common::sdk_fixture::faux_model();
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let detail: Arc<Mutex<Vec<Recorded>>> = Arc::new(Mutex::new(Vec::new()));
    let cancels = Arc::new(Cancels::default());
    let log_for_factory = Arc::clone(&log);
    let detail_for_factory = Arc::clone(&detail);
    let cancels_for_factory = Arc::clone(&cancels);
    let factory =
        common::agent_session::runtime_factory(common::agent_session::RuntimeFactoryOptions {
            runtime: registry.runtime().clone(),
            model,
            extensions: Arc::new(move || {
                vec![event_factory(
                    Arc::clone(&log_for_factory),
                    Arc::clone(&cancels_for_factory),
                    Arc::clone(&detail_for_factory),
                )]
            }),
            no_extensions: true,
            tools: None,
        });
    let runtime = create_agent_session_runtime(
        factory,
        CreateAgentSessionRuntimeOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            session_manager: build(&cwd),
            session_start_event: None,
            project_trust_context: None,
        },
    )
    .await
    .expect("runtime");
    runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    log.lock().unwrap_or_else(PoisonError::into_inner).clear();
    detail
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
    Harness {
        runtime,
        log,
        detail,
        cancels,
        _temp: fixture.temp,
        cwd,
    }
}

async fn harness(session_manager: SessionManager) -> Harness {
    harness_with(move |_cwd| session_manager).await
}

async fn persisted_harness() -> Harness {
    harness_with(|cwd| SessionManager::create(cwd, None, None).expect("persisted manager")).await
}

fn entries_log(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    log.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

fn in_memory_manager(cwd: &str) -> SessionManager {
    SessionManager::in_memory(Some(cwd), None, None).expect("session manager")
}

fn persisted_manager(cwd: &str) -> SessionManager {
    SessionManager::create(cwd, None, None).expect("persisted manager")
}

#[tokio::test]
async fn new_session_replaces_the_session_and_emits_events() {
    let mut harness = persisted_harness().await;
    let original = harness.runtime.session().session_id();
    let outcome = harness
        .runtime
        .new_session(NewSessionRequest::default())
        .await
        .expect("new session");
    assert!(!outcome.cancelled);
    assert_ne!(original, harness.runtime.session().session_id());
    harness
        .runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    assert_eq!(
        entries_log(&harness.log),
        vec![
            "before_switch".to_owned(),
            "shutdown".to_owned(),
            "start".to_owned()
        ]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn new_session_honors_before_switch_cancellation() {
    let mut harness = persisted_harness().await;
    let original = harness.runtime.session().session_id();
    let original_file = harness.runtime.session().session_file();
    harness.cancels.switch.store(true, Ordering::SeqCst);
    harness
        .detail
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
    let outcome = harness
        .runtime
        .new_session(NewSessionRequest::default())
        .await
        .expect("new session");
    assert!(outcome.cancelled);
    assert_eq!(original, harness.runtime.session().session_id());
    assert_eq!(harness.runtime.session().session_file(), original_file);
    assert_eq!(entries_log(&harness.log), vec!["before_switch".to_owned()]);
    assert_eq!(
        harness
            .detail
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone(),
        vec![Recorded::BeforeSwitch {
            reason: SessionShutdownReason::New,
            target: None,
        }]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn new_session_runs_setup_before_loading_messages() {
    let mut harness = persisted_harness().await;
    let setup: pi_coding_agent::core::agent_session_runtime::NewSessionSetup =
        Box::new(|mut manager: SessionManager| {
            Box::pin(async move {
                let _ = manager.append_session_info("renamed");
                manager
            })
        });
    let outcome = harness
        .runtime
        .new_session(NewSessionRequest {
            parent_session: None,
            setup: Some(setup),
            with_session: None,
        })
        .await
        .expect("new session");
    assert!(!outcome.cancelled);
    assert_eq!(
        harness.runtime.session().session_name().as_deref(),
        Some("renamed")
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn fork_before_returns_the_user_text_and_at_does_not() {
    let cwd = "/tmp/pi-runtime-fork";
    let mut manager = in_memory_manager(cwd);
    let entry_id = manager
        .append_message(common::sdk_fixture::user_message("hello there"))
        .expect("message");
    let mut first = harness(manager).await;
    let outcome = first
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::Before),
                with_session: None,
            },
        )
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.selected_text.as_deref(), Some("hello there"));
    assert_eq!(
        entries_log(&first.log),
        vec!["before_fork".to_owned(), "shutdown".to_owned()]
    );
    first.runtime.dispose().await;

    let mut manager = in_memory_manager(cwd);
    let entry_id = manager
        .append_message(common::sdk_fixture::user_message("hello there"))
        .expect("message");
    let mut second = harness(manager).await;
    let outcome = second
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::At),
                with_session: None,
            },
        )
        .await
        .expect("fork");
    assert_eq!(outcome.selected_text, None);
    second.runtime.dispose().await;
}

#[tokio::test]
async fn fork_rejects_an_invalid_entry_and_cancellation() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-fork-invalid")).await;
    let error = harness
        .runtime
        .fork("missing-entry", ForkOptions::default())
        .await
        .expect_err("invalid entry");
    assert!(error.to_string().contains("Invalid entry ID for forking"));
    harness.cancels.fork.store(true, Ordering::SeqCst);
    let outcome = harness
        .runtime
        .fork("missing-entry", ForkOptions::default())
        .await
        .expect("cancelled");
    assert!(outcome.cancelled);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn switch_session_reopens_the_target() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-switch")).await;
    let mut target = persisted_manager(&harness.cwd);
    target
        .append_message(common::sdk_fixture::user_message("other"))
        .expect("target message");
    target
        .append_message(common::sdk_fixture::assistant_message("ok"))
        .expect("target flush");
    let target_id = target.session_id().to_owned();
    let target_file = target.session_file().expect("target file").to_owned();
    let outcome = harness
        .runtime
        .switch_session(
            &target_file,
            SwitchSessionOptions {
                cwd_override: Some(harness.cwd.clone()),
                ..SwitchSessionOptions::default()
            },
        )
        .await
        .expect("switch");
    assert!(!outcome.cancelled);
    assert_eq!(harness.runtime.session().session_id(), target_id);
    harness
        .runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    assert_eq!(
        entries_log(&harness.log),
        vec![
            "before_switch".to_owned(),
            "shutdown".to_owned(),
            "start".to_owned()
        ]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn import_copies_the_file_and_reports_missing_sources() {
    let mut harness = persisted_harness().await;
    let missing = harness
        .runtime
        .import_from_jsonl("/tmp/definitely-not-a-session.jsonl", None)
        .await
        .expect_err("missing file");
    assert!(matches!(
        missing,
        AgentSessionRuntimeError::ImportFileNotFound(_)
    ));

    let mut source = persisted_manager(&harness.cwd);
    source
        .append_message(common::sdk_fixture::user_message("imported"))
        .expect("source message");
    source
        .append_message(common::sdk_fixture::assistant_message("ok"))
        .expect("source flush");
    let source_file = source.session_file().expect("source file").to_owned();
    let outcome = harness
        .runtime
        .import_from_jsonl(&source_file, None)
        .await
        .expect("import");
    assert!(!outcome.cancelled);
    let imported = harness
        .runtime
        .session()
        .session_file()
        .expect("imported file");
    assert!(imported.ends_with(&format!(
        "/{}",
        pi_coding_agent::utils::paths::basename_posix(&source_file)
    )));
    let again = harness
        .runtime
        .import_from_jsonl(&imported, None)
        .await
        .expect("already-stored import");
    assert!(!again.cancelled);
    assert_eq!(harness.runtime.services().cwd, harness.cwd);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn replacement_callbacks_run_in_order() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-callbacks")).await;
    let log = Arc::clone(&harness.log);
    harness
        .runtime
        .set_rebind_session(Some(Arc::new(move |_session| {
            let log = Arc::clone(&log);
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push("rebind".to_owned());
            })
        })));
    let log = Arc::clone(&harness.log);
    harness
        .runtime
        .new_session(NewSessionRequest {
            parent_session: None,
            setup: None,
            with_session: Some(Box::new(move |_ctx| {
                let log = Arc::clone(&log);
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push("with".to_owned());
                })
            })),
        })
        .await
        .expect("new session");
    assert_eq!(
        entries_log(&harness.log),
        vec![
            "before_switch".to_owned(),
            "shutdown".to_owned(),
            "rebind".to_owned(),
            "with".to_owned()
        ]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn dispose_emits_the_quit_shutdown() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-dispose")).await;
    harness.runtime.dispose().await;
    assert_eq!(entries_log(&harness.log), vec!["shutdown".to_owned()]);
}

#[tokio::test]
async fn persisted_fork_at_an_entry_creates_a_branched_session() {
    let entry_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&entry_slot);
    let mut harness = harness_with(move |cwd| {
        let mut manager = SessionManager::create(cwd, None, None).expect("manager");
        let id = manager
            .append_message(common::sdk_fixture::user_message("hello"))
            .expect("user message");
        manager
            .append_message(common::sdk_fixture::assistant_message("ok"))
            .expect("assistant message");
        *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(id);
        manager
    })
    .await;
    let entry_id = entry_slot
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
        .expect("entry id");
    let original_file = harness.runtime.session().session_file();
    let outcome = harness
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::At),
                with_session: None,
            },
        )
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    let forked_file = harness.runtime.session().session_file();
    assert!(forked_file.is_some());
    assert_ne!(forked_file, original_file);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn persisted_fork_reports_an_unflushed_session() {
    let entry_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&entry_slot);
    let mut harness = harness_with(move |cwd| {
        let mut manager = SessionManager::create(cwd, None, None).expect("manager");
        let id = manager
            .append_message(common::sdk_fixture::user_message("hello"))
            .expect("user message");
        *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(id);
        manager
    })
    .await;
    let entry_id = entry_slot
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
        .expect("entry id");
    let error = harness
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::At),
                with_session: None,
            },
        )
        .await
        .expect_err("unflushed session");
    assert!(
        error
            .to_string()
            .contains("This session has not been saved yet"),
        "{error}"
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn import_renames_a_colliding_destination() {
    let mut harness = persisted_harness().await;
    let directory = TempDir::new().expect("tempdir");
    let source_file = directory.path().join("dup.jsonl");
    let source_manager = SessionManager::create(
        &directory.path().display().to_string(),
        Some(&directory.path().display().to_string()),
        None,
    )
    .expect("source manager");
    drop(source_manager);
    std::fs::write(
        &source_file,
        "{\"type\":\"session\",\"version\":3,\"id\":\"x\"}\n",
    )
    .expect("write");
    let first = harness
        .runtime
        .import_from_jsonl(&source_file.display().to_string(), None)
        .await
        .expect("first import");
    assert!(!first.cancelled);
    let second = harness
        .runtime
        .import_from_jsonl(&source_file.display().to_string(), None)
        .await
        .expect("second import");
    assert!(!second.cancelled);
    let imported = harness
        .runtime
        .session()
        .session_file()
        .expect("imported file");
    assert!(imported.ends_with("dup-1.jsonl"), "{imported}");
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn before_session_invalidate_and_trust_factory_run() {
    let mut harness = persisted_harness().await;
    let invalidated = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&invalidated);
    harness
        .runtime
        .set_before_session_invalidate(Some(Box::new(move || {
            flag.store(true, Ordering::SeqCst);
        })));
    let mut target = persisted_manager(&harness.cwd);
    target
        .append_message(common::sdk_fixture::assistant_message("ok"))
        .expect("target flush");
    let target_file = target.session_file().expect("target file").to_owned();
    let trusted = Arc::new(AtomicBool::new(false));
    let trust_flag = Arc::clone(&trusted);
    harness
        .runtime
        .switch_session(
            &target_file,
            SwitchSessionOptions {
                cwd_override: None,
                with_session: None,
                project_trust_context_factory: Some(Arc::new(move |_cwd| {
                    trust_flag.store(true, Ordering::SeqCst);
                    None
                })),
            },
        )
        .await
        .expect("switch");
    assert!(invalidated.load(Ordering::SeqCst));
    assert!(trusted.load(Ordering::SeqCst));
    assert_eq!(harness.runtime.cwd(), harness.cwd);
    assert!(harness.runtime.model_fallback_message().is_none());
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn persisted_fork_before_a_root_message_starts_a_new_session() {
    let entry_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&entry_slot);
    let mut harness = harness_with(move |cwd| {
        let mut manager = SessionManager::create(cwd, None, None).expect("manager");
        let id = manager
            .append_message(common::sdk_fixture::user_message("hello"))
            .expect("user message");
        manager
            .append_message(common::sdk_fixture::assistant_message("ok"))
            .expect("assistant message");
        *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(id);
        manager
    })
    .await;
    let entry_id = entry_slot
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
        .expect("entry id");
    let original_file = harness.runtime.session().session_file();
    let outcome = harness
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::Before),
                with_session: None,
            },
        )
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.selected_text.as_deref(), Some("hello"));
    assert!(harness.runtime.session().session_file().is_some());
    assert_ne!(harness.runtime.session().session_file(), original_file);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn switch_reports_a_missing_stored_cwd() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-missing-cwd")).await;
    let directory = TempDir::new().expect("tempdir");
    let file = directory.path().join("stale.jsonl");
    let mut manager = SessionManager::create(
        "/tmp/pi-runtime-gone",
        Some(&directory.path().display().to_string()),
        None,
    )
    .expect("manager");
    manager
        .append_message(common::sdk_fixture::assistant_message("ok"))
        .expect("flush");
    std::fs::copy(manager.session_file().expect("file"), &file).expect("copy");
    let error = harness
        .runtime
        .switch_session(&file.display().to_string(), SwitchSessionOptions::default())
        .await
        .expect_err("missing cwd");
    assert!(matches!(
        error,
        AgentSessionRuntimeError::MissingSessionCwd(_)
    ));
    assert!(error.to_string().contains("does not exist"));
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn new_session_carries_the_parent_session() {
    let mut harness = persisted_harness().await;
    let parent = harness
        .runtime
        .session()
        .session_file()
        .expect("parent file");
    let outcome = harness
        .runtime
        .new_session(NewSessionRequest {
            parent_session: Some(parent),
            setup: None,
            with_session: None,
        })
        .await
        .expect("new session");
    assert!(!outcome.cancelled);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn fork_rejects_a_non_user_entry_before_position() {
    let mut manager = in_memory_manager("/tmp/pi-runtime-fork-assistant");
    let entry_id = manager
        .append_message(common::sdk_fixture::assistant_message("ok"))
        .expect("assistant message");
    let mut harness = harness(manager).await;
    let error = harness
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::Before),
                with_session: None,
            },
        )
        .await
        .expect_err("assistant entry");
    assert!(error.to_string().contains("Invalid entry ID for forking"));
    harness.runtime.dispose().await;
}

fn detail_log(harness: &Harness) -> Vec<Recorded> {
    harness
        .detail
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

fn clear_detail(harness: &Harness) {
    harness
        .detail
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
}

#[tokio::test]
async fn emits_the_runtime_lifecycle_events_for_new_and_resume() {
    let mut harness = persisted_harness().await;
    assert_eq!(
        detail_log(&harness),
        vec![Recorded::Start {
            reason: SessionStartReason::Startup,
            previous: None,
        }]
    );
    clear_detail(&harness);

    harness
        .runtime
        .session()
        .prompt("hello".to_owned(), None)
        .await
        .expect("prompt");
    let original_file = harness
        .runtime
        .session()
        .session_file()
        .expect("session file");

    let outcome = harness
        .runtime
        .new_session(NewSessionRequest::default())
        .await
        .expect("new session");
    assert!(!outcome.cancelled);
    harness
        .runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    let second_file = harness
        .runtime
        .session()
        .session_file()
        .expect("second session file");
    assert_eq!(
        detail_log(&harness),
        vec![
            Recorded::BeforeSwitch {
                reason: SessionShutdownReason::New,
                target: None,
            },
            Recorded::Shutdown {
                reason: SessionShutdownReason::New,
                target: Some(second_file.clone()),
            },
            Recorded::Start {
                reason: SessionStartReason::New,
                previous: Some(original_file.clone()),
            },
        ]
    );
    clear_detail(&harness);

    let outcome = harness
        .runtime
        .switch_session(
            &original_file,
            SwitchSessionOptions {
                cwd_override: Some(harness.cwd.clone()),
                ..SwitchSessionOptions::default()
            },
        )
        .await
        .expect("switch");
    assert!(!outcome.cancelled);
    harness
        .runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    assert_eq!(
        detail_log(&harness),
        vec![
            Recorded::BeforeSwitch {
                reason: SessionShutdownReason::Resume,
                target: Some(original_file.clone()),
            },
            Recorded::Shutdown {
                reason: SessionShutdownReason::Resume,
                target: Some(original_file),
            },
            Recorded::Start {
                reason: SessionStartReason::Resume,
                previous: Some(second_file),
            },
        ]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn runs_before_session_invalidate_after_shutdown_and_before_rebind() {
    let mut harness = persisted_harness().await;
    let phases: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let old_session = Arc::clone(harness.runtime.session());
    let cwd = harness.cwd.clone();
    let before_phases = Arc::clone(&phases);
    let before_session = Arc::clone(&old_session);
    let before_cwd = cwd.clone();
    harness
        .runtime
        .set_before_session_invalidate(Some(Box::new(move || {
            before_phases
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push("beforeSessionInvalidate".to_owned());
            assert_eq!(
                before_session
                    .extension_runner()
                    .context()
                    .cwd()
                    .expect("live context"),
                before_cwd
            );
        })));
    let rebind_phases = Arc::clone(&phases);
    harness.runtime.set_rebind_session(Some(Arc::new(
        move |_session: &Arc<
            pi_coding_agent::core::agent_session::AgentSession<InMemorySettingsStorage>,
        >| {
            let rebind_phases = Arc::clone(&rebind_phases);
            Box::pin(async move {
                rebind_phases
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push("rebindSession".to_owned());
            })
        },
    )));

    harness
        .runtime
        .new_session(NewSessionRequest::default())
        .await
        .expect("new session");

    assert_eq!(
        phases
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone(),
        vec![
            "beforeSessionInvalidate".to_owned(),
            "rebindSession".to_owned()
        ]
    );
    let error = old_session
        .extension_runner()
        .context()
        .cwd()
        .expect_err("the old context is stale");
    assert_eq!(
        error,
        "This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload()."
    );
    harness.runtime.set_before_session_invalidate(None);
    harness.runtime.set_rebind_session(None);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn emits_session_before_fork_and_honors_cancellation() {
    let mut harness = persisted_harness().await;
    harness
        .runtime
        .session()
        .prompt("hello".to_owned(), None)
        .await
        .expect("prompt");
    let entry_id = harness
        .runtime
        .session()
        .user_messages_for_forking()
        .first()
        .map(|(id, _)| id.clone())
        .expect("user message");
    let previous_file = harness.runtime.session().session_file();
    clear_detail(&harness);

    let outcome = harness
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::Before),
                with_session: None,
            },
        )
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.selected_text.as_deref(), Some("hello"));
    harness
        .runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    assert_eq!(
        detail_log(&harness),
        vec![
            Recorded::BeforeFork {
                entry_id: entry_id.clone(),
                position: ForkPosition::Before,
            },
            Recorded::Shutdown {
                reason: SessionShutdownReason::Fork,
                target: harness.runtime.session().session_file(),
            },
            Recorded::Start {
                reason: SessionStartReason::Fork,
                previous: previous_file,
            },
        ]
    );
    clear_detail(&harness);

    harness.cancels.fork.store(true, Ordering::SeqCst);
    let outcome = harness
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::Before),
                with_session: None,
            },
        )
        .await
        .expect("cancelled fork");
    assert!(outcome.cancelled);
    assert_eq!(
        detail_log(&harness),
        vec![Recorded::BeforeFork {
            entry_id: entry_id.clone(),
            position: ForkPosition::Before,
        }]
    );
    clear_detail(&harness);

    let outcome = harness
        .runtime
        .fork(
            "missing-entry",
            ForkOptions {
                position: Some(ForkPosition::At),
                with_session: None,
            },
        )
        .await
        .expect("cancelled fork");
    assert!(outcome.cancelled);
    assert_eq!(
        detail_log(&harness),
        vec![Recorded::BeforeFork {
            entry_id: "missing-entry".to_owned(),
            position: ForkPosition::At,
        }]
    );
    harness.runtime.dispose().await;
}
