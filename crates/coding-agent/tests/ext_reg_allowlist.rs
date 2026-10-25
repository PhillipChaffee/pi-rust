//! `test/suite/regressions/2835-tools-allowlist-filters-extension-tools.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use common::extension_regressions::sdk_session;
use common::extension_regressions::session_start_tool_extension;
use common::extension_regressions::sorted_tool_names;
use common::extension_regressions::tool_definition;

/// #2835: an allowlist filters extension tools along with built-ins.
#[tokio::test]
async fn allowlist_filters_extension_tools() {
    let session = sdk_session(
        "/tmp/pi-ext-reg-allowlist",
        session_start_tool_extension(vec![tool_definition(
            "dynamic_tool",
            "Tool registered from session_start",
            Some("Run dynamic test behavior"),
        )]),
        Some(vec!["read".to_owned(), "dynamic_tool".to_owned()]),
        None,
        None,
    )
    .await;

    assert_eq!(
        sorted_tool_names(&session),
        ["dynamic_tool".to_owned(), "read".to_owned()]
    );
    let mut active = session.active_tool_names();
    active.sort();
    assert_eq!(active, ["dynamic_tool".to_owned(), "read".to_owned()]);
    assert!(
        session
            .system_prompt()
            .contains("- read: Read file contents")
    );
    assert!(
        session
            .system_prompt()
            .contains("- dynamic_tool: Run dynamic test behavior")
    );
    assert!(!session.system_prompt().contains("- bash:"));
    assert!(!session.system_prompt().contains("- edit:"));
}

/// #2835: an empty allowlist disables every tool, extension tools included.
#[tokio::test]
async fn empty_allowlist_disables_every_tool() {
    let session = sdk_session(
        "/tmp/pi-ext-reg-allowlist-empty",
        session_start_tool_extension(vec![tool_definition(
            "dynamic_tool",
            "Tool registered from session_start",
            Some("Run dynamic test behavior"),
        )]),
        Some(Vec::new()),
        None,
        None,
    )
    .await;

    assert!(session.get_all_tools().is_empty());
    assert!(session.active_tool_names().is_empty());
    assert!(session.system_prompt().contains("Available tools:\n(none)"));
    assert!(!session.system_prompt().contains("dynamic_tool"));
}
