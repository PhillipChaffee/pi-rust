//! `test/suite/regressions/3686-session-name-event.test.ts` and
//! `5996-session-name-newlines.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;

use common::agent_session::SessionSetup;
use common::agent_session::build;
use common::extension_regressions::ApiSlot;
use common::extension_regressions::capture_api_extension;
use common::extension_regressions::manager;

type NameLog = Arc<Mutex<Vec<Option<String>>>>;
type Listener = Arc<dyn Fn(AgentSessionEvent) + Send + Sync>;

fn record_names() -> (NameLog, Listener) {
    let names: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&names);
    let listener: Arc<dyn Fn(AgentSessionEvent) + Send + Sync> = Arc::new(move |event| {
        if let AgentSessionEvent::SessionInfoChanged { name } = event {
            sink.lock().expect("lock").push(name);
        }
    });
    (names, listener)
}

/// #3686: a session name change emits to listeners.
#[tokio::test]
async fn session_name_change_emits_to_listeners() {
    let root = "/tmp/pi-ext-reg-name-event";
    let session = build(SessionSetup::new(root, manager(root))).await;
    let (names, listener) = record_names();
    let _unsubscribe = session.subscribe(listener);

    session.set_session_name("hello world");

    assert_eq!(session.session_name().as_deref(), Some("hello world"));
    assert_eq!(
        names.lock().expect("lock").clone(),
        vec![Some("hello world".to_owned())]
    );
}

/// #3686: a name change made through `pi.setSessionName` emits too.
#[tokio::test]
async fn extension_session_name_change_emits_to_listeners() {
    let root = "/tmp/pi-ext-reg-name-api";
    let api_slot: ApiSlot = Arc::new(Mutex::new(None));
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![capture_api_extension(Arc::clone(&api_slot))];
    setup.bind_extensions = true;
    let session = build(setup).await;
    let (names, listener) = record_names();
    let _unsubscribe = session.subscribe(listener);

    api_slot
        .lock()
        .expect("lock")
        .as_ref()
        .expect("api")
        .set_session_name("from extension".to_owned())
        .expect("name sets");
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    assert_eq!(session.session_name().as_deref(), Some("from extension"));
    assert_eq!(
        names.lock().expect("lock").clone(),
        vec![Some("from extension".to_owned())]
    );
}

/// #3686: extensions hear `session_info_changed` from both callers.
#[tokio::test]
async fn session_info_changed_reaches_extensions() {
    let root = "/tmp/pi-ext-reg-name-extension";
    let api_slot: ApiSlot = Arc::new(Mutex::new(None));
    let events: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let slot = Arc::clone(&api_slot);
    let sink = Arc::clone(&events);
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&slot);
            let sink = Arc::clone(&sink);
            Box::pin(async move {
                api.on(
                    "session_info_changed",
                    Arc::new(move |event: ExtensionEvent, _ctx| {
                        if let ExtensionEvent::SessionInfoChanged { name } = event {
                            sink.lock().expect("lock").push(name);
                        }
                        Box::pin(async { HandlerResult::None })
                    }),
                )
                .expect("session_info_changed handler");
                *slot.lock().expect("lock") = Some(api);
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    let session = build(setup).await;

    api_slot
        .lock()
        .expect("lock")
        .as_ref()
        .expect("api")
        .set_session_name("first".to_owned())
        .expect("name sets");
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    session.set_session_name("second");
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    assert_eq!(
        events.lock().expect("lock").clone(),
        vec![Some("first".to_owned()), Some("second".to_owned())]
    );
}

/// #5996: session names filter newlines.
#[tokio::test]
async fn session_names_filter_newlines() {
    let root = "/tmp/pi-ext-reg-name-newlines";
    let session = build(SessionSetup::new(root, manager(root))).await;
    let (names, listener) = record_names();
    let _unsubscribe = session.subscribe(listener);

    session.set_session_name("hello\nworld\r\nagain");

    assert_eq!(session.session_name().as_deref(), Some("hello world again"));
    assert_eq!(
        names.lock().expect("lock").clone(),
        vec![Some("hello world again".to_owned())]
    );
}

/// #5996: extension-set names filter newlines too.
#[tokio::test]
async fn extension_session_names_filter_newlines() {
    let root = "/tmp/pi-ext-reg-name-extension-newlines";
    let api_slot: ApiSlot = Arc::new(Mutex::new(None));
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![capture_api_extension(Arc::clone(&api_slot))];
    setup.bind_extensions = true;
    let session = build(setup).await;

    api_slot
        .lock()
        .expect("lock")
        .as_ref()
        .expect("api")
        .set_session_name("from\nextension".to_owned())
        .expect("name sets");
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    assert_eq!(session.session_name().as_deref(), Some("from extension"));
}
