//! `test/suite/regressions/5109-exclude-tools.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use common::agent_session::SessionSetup;
use common::agent_session::build;
use common::extension_regressions::manager;
use common::extension_regressions::session_start_tool_extension;
use common::extension_regressions::sorted_tool_names;
use common::extension_regressions::tool_definition;

fn filter_extension() -> pi_coding_agent::extensions::api::InlineExtension {
    session_start_tool_extension(vec![
        tool_definition("ask_question", "Ask a question", Some("Ask a question")),
        tool_definition(
            "dynamic_tool",
            "Dynamic test tool",
            Some("Run dynamic test behavior"),
        ),
    ])
}

/// #5109: exclusions filter built-in and extension tools.
#[tokio::test]
async fn exclude_tools_filters_builtin_and_extension_tools() {
    let root = "/tmp/pi-ext-reg-exclude";
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![filter_extension()];
    setup.excluded_tool_names = Some(vec!["read".to_owned(), "ask_question".to_owned()]);
    setup.bind_extensions = true;
    let session = build(setup).await;

    let all = sorted_tool_names(&session);
    assert!(!all.contains(&"read".to_owned()));
    assert!(!all.contains(&"ask_question".to_owned()));
    assert!(all.contains(&"bash".to_owned()));
    assert!(all.contains(&"dynamic_tool".to_owned()));
    let mut active = session.active_tool_names();
    active.sort();
    assert_eq!(
        active,
        [
            "bash".to_owned(),
            "dynamic_tool".to_owned(),
            "edit".to_owned(),
            "write".to_owned(),
        ]
    );
    assert!(!session.system_prompt().contains("- read:"));
    assert!(!session.system_prompt().contains("ask_question"));
    assert!(
        session
            .system_prompt()
            .contains("- dynamic_tool: Run dynamic test behavior")
    );
}

/// #5109: exclusions override the allowlist.
#[tokio::test]
async fn excluded_tools_override_the_allowlist() {
    let root = "/tmp/pi-ext-reg-exclude-override";
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![filter_extension()];
    setup.allowed_tool_names = Some(vec![
        "read".to_owned(),
        "bash".to_owned(),
        "ask_question".to_owned(),
    ]);
    setup.excluded_tool_names = Some(vec!["read".to_owned(), "ask_question".to_owned()]);
    setup.initial_active_tool_names = Some(vec![
        "read".to_owned(),
        "bash".to_owned(),
        "ask_question".to_owned(),
    ]);
    setup.bind_extensions = true;
    let session = build(setup).await;

    assert_eq!(sorted_tool_names(&session), ["bash".to_owned()]);
    assert_eq!(session.active_tool_names(), ["bash".to_owned()]);
    assert!(session.system_prompt().contains("- bash:"));
    assert!(!session.system_prompt().contains("- read:"));
    assert!(!session.system_prompt().contains("ask_question"));
}
