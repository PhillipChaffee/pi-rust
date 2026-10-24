//! `test/agent-session-branching.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream gates the suite on `API_KEY`; the port keeps that gate as
//! `#[ignore]` so a credentialed shell opts in with `--ignored`.

#![expect(
    clippy::expect_used,
    clippy::print_stderr,
    reason = "the tests pin outcomes and log their credential skip the way upstream does"
)]

mod common;

use std::sync::Arc;

use pi_coding_agent::core::agent_session::ExtensionBindings;
use pi_coding_agent::core::agent_session_runtime::AgentSessionRuntime;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeOptions;
use pi_coding_agent::core::agent_session_runtime::ForkOptions;
use pi_coding_agent::core::agent_session_runtime::create_agent_session_runtime;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use tempfile::TempDir;

async fn create_runtime(
    root: &str,
    in_memory: bool,
) -> (AgentSessionRuntime<InMemorySettingsStorage>, TempDir) {
    let temp = TempDir::new().expect("tempdir");
    let token = common::agent_session::live_token().expect("credential");
    let agent_dir = format!("{root}/agent");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    registry
        .runtime()
        .set_runtime_api_key("anthropic", token, None)
        .await
        .expect("runtime key");
    let model = common::agent_session::anthropic_model();
    let manager = if in_memory {
        SessionManager::in_memory(Some(root), None, None).expect("in-memory manager")
    } else {
        SessionManager::create(root, Some(&format!("{root}/sessions")), None).expect("manager")
    };
    let factory =
        common::agent_session::runtime_factory(common::agent_session::RuntimeFactoryOptions {
            runtime: registry.runtime().clone(),
            model,
            extensions: Arc::new(Vec::new),
            no_extensions: true,
            tools: Some(vec![
                "read".to_owned(),
                "bash".to_owned(),
                "edit".to_owned(),
                "write".to_owned(),
            ]),
        });
    let runtime = create_agent_session_runtime(
        factory,
        CreateAgentSessionRuntimeOptions {
            cwd: root.to_owned(),
            agent_dir,
            session_manager: manager,
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
    (runtime, temp)
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn allows_forking_from_a_single_message() {
    let Some(_token) = common::agent_session::live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-branching-single";
    let (mut runtime, _temp) = create_runtime(root, false).await;
    runtime
        .session()
        .prompt("Say hello".to_owned(), None)
        .await
        .expect("prompt");
    runtime.session().agent().wait_for_idle().await;

    let user_messages = runtime.session().user_messages_for_forking();
    assert_eq!(user_messages.len(), 1);
    assert_eq!(user_messages[0].1, "Say hello");

    let outcome = runtime
        .fork(&user_messages[0].0, ForkOptions::default())
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.selected_text.as_deref(), Some("Say hello"));
    assert!(runtime.session().agent().state().messages.is_empty());
    assert!(runtime.session().session_file().is_some());
    runtime.dispose().await;
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn supports_in_memory_forking_in_no_session_mode() {
    let Some(_token) = common::agent_session::live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-branching-memory";
    let (mut runtime, _temp) = create_runtime(root, true).await;
    assert!(runtime.session().session_file().is_none());

    runtime
        .session()
        .prompt("Say hi".to_owned(), None)
        .await
        .expect("prompt");
    runtime.session().agent().wait_for_idle().await;
    let user_messages = runtime.session().user_messages_for_forking();
    assert_eq!(user_messages.len(), 1);
    assert!(!runtime.session().agent().state().messages.is_empty());

    let outcome = runtime
        .fork(&user_messages[0].0, ForkOptions::default())
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.selected_text.as_deref(), Some("Say hi"));
    assert!(runtime.session().agent().state().messages.is_empty());
    assert!(runtime.session().session_file().is_none());
    runtime.dispose().await;
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC credentials are set, upstream's describe.skipIf"]
async fn forks_from_the_middle_of_a_conversation() {
    let Some(_token) = common::agent_session::live_token() else {
        eprintln!("skipping: no ANTHROPIC credential");
        return;
    };
    let root = "/tmp/pi-session-branching-middle";
    let (mut runtime, _temp) = create_runtime(root, false).await;
    for text in ["Say one", "Say two", "Say three"] {
        runtime
            .session()
            .prompt(text.to_owned(), None)
            .await
            .expect("prompt");
        runtime.session().agent().wait_for_idle().await;
    }
    let user_messages = runtime.session().user_messages_for_forking();
    assert_eq!(user_messages.len(), 3);

    let outcome = runtime
        .fork(&user_messages[1].0, ForkOptions::default())
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.selected_text.as_deref(), Some("Say two"));
    let messages = runtime.session().agent().state().messages.clone();
    assert_eq!(messages.len(), 2);
    assert!(matches!(
        messages[0],
        pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::User(_))
    ));
    assert!(matches!(
        messages[1],
        pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(_))
    ));
    runtime.dispose().await;
}
