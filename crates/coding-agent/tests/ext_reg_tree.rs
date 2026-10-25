//! `test/suite/regressions/3688-tree-cancel-compacting.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Message;
use pi_coding_agent::core::agent_session::NavigateTreeOptions;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::results::SessionBeforeResult;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::build;
use common::extension_regressions::manager;

/// #3688: a cancelled tree navigation clears the branch-summary state.
#[tokio::test]
async fn cancelled_tree_navigation_clears_compaction_state() {
    let root = "/tmp/pi-ext-reg-tree-cancel";
    let mut seed = manager(root);
    let target = seed
        .append_message(common::agent_session::user_message("first", 0))
        .expect("first");
    seed.append_message(AgentMessage::Standard(Message::Assistant(
        assistant_message("reply", 0),
    )))
    .expect("reply");
    let current = seed
        .append_message(common::agent_session::user_message("second", 0))
        .expect("second");
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.on(
                    "session_before_tree",
                    Arc::new(|_event: ExtensionEvent, _ctx| {
                        Box::pin(async {
                            HandlerResult::SessionBefore(SessionBeforeResult::Tree {
                                cancel: true,
                                summary: None,
                                custom_instructions: None,
                                replace_instructions: None,
                                label: None,
                            })
                        })
                    }),
                )
                .expect("session_before_tree handler");
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, seed);
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    let session = build(setup).await;

    let result = session
        .navigate_tree(&target, NavigateTreeOptions::default())
        .await
        .expect("navigate");

    assert!(result.cancelled);
    assert!(!session.is_compacting());
    assert_eq!(session.leaf_id().as_deref(), Some(current.as_str()));
}
