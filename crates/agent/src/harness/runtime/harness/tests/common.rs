//! The harness-container suite's fixtures, ported from upstream
//! `test/harness/runtime/harness.test.ts`'s module fixture block at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the harness options over a
//! fresh faux provider (`harnessOptions`), the storage-backed session over
//! a plain memory backend (`createSession`), the harness creation
//! (`createHarness`), the `faux`/`faux-1`/`low`/`read` configuration
//! (`configured`), and the session ledger the suite closes in its
//! teardown (`afterEach`). The accept and watch suites reuse these
//! fixtures over their own storage decorators.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_ai::models::create_models;
use pi_ai::providers::faux::{FauxProviderHandle, RegisterFauxProviderOptions, faux_provider};

use crate::harness::agent_harness::AgentHarnessOptions;
use crate::harness::context::background_context;
use crate::harness::runtime::harness::{Harness, create_agent_harness};
use crate::harness::runtime::test_support::runtime_session_metadata;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::{LaneConfiguration, ModelIdentity, Session};
use crate::types::ThinkingLevel;

/// The ledger the suites close in their teardown, upstream's module-level
/// `sessions: Session[]` array.
pub(super) type SessionLedger = Vec<Arc<StorageBackedSession>>;

/// Builds the harness options over one session, upstream's
/// `harnessOptions(session)`: a fresh faux provider registered into a
/// fresh models catalog, the `medium` thinking level, and the
/// `read`/`bash` active tool names. The faux handle rides along for the
/// response-scripting tests.
#[must_use]
/// The agent-harness options' defaults the fixtures share: every override
/// `None` over the given session, models, model identity, and the
/// thinking/active-tools seeds.
pub(super) fn default_agent_options(
    session: Arc<StorageBackedSession>,
    models: Arc<pi_ai::models::Models>,
    model: pi_ai::types::Model,
    thinking_level: Option<ThinkingLevel>,
    active_tool_names: Option<Vec<String>>,
) -> AgentHarnessOptions {
    AgentHarnessOptions {
        session,
        models,
        model,
        thinking_level,
        active_tool_names,
        tools: None,
        tool_context: None,
        system_prompt: None,
        resources: None,
        stream_options: None,
        retry: None,
        compaction: None,
        steering_mode: None,
        follow_up_mode: None,
        tool_execution: None,
        to_provider_messages: None,
        entry_projectors: None,
    }
}

pub(super) fn harness_options(
    session: Arc<StorageBackedSession>,
) -> (AgentHarnessOptions, FauxProviderHandle) {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let options = default_agent_options(
        session,
        models,
        faux.first_model(),
        Some(ThinkingLevel::Medium),
        Some(vec!["read".to_owned(), "bash".to_owned()]),
    );
    (options, faux)
}

/// Opens one session over a plain memory backend and records it in the
/// ledger, upstream's `createSession(id = `session-${sessions.length}`)`.
#[must_use]
pub(super) fn create_session(sessions: &mut SessionLedger) -> Arc<StorageBackedSession> {
    create_session_with_id(sessions, &format!("session-{}", sessions.len()))
}

/// The id-named session variant, upstream's `createSession("shared-session")`.
#[must_use]
pub(super) fn create_session_with_id(
    sessions: &mut SessionLedger,
    id: &str,
) -> Arc<StorageBackedSession> {
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(id.to_owned()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        StorageBackedSessionOptions::default(),
    ));
    sessions.push(Arc::clone(&session));
    session
}

/// Attaches the runtime over a fresh session, upstream's `createHarness`:
/// the session defaults to a ledgered one, and the created harness is the
/// concrete runtime container (`created.harness instanceof Harness` is a
/// type identity in Rust).
///
/// # Panics
/// The creation's failure.
pub(super) async fn create_harness(sessions: &mut SessionLedger) -> Harness {
    let session = create_session(sessions);
    let (options, _faux) = harness_options(session);
    create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates")
        .harness
}

/// The lane configuration the suite pins, upstream's `configured`
/// constant: `faux`/`faux-1`, `low`, and the `read` active tool name.
#[must_use]
pub(super) fn configured() -> LaneConfiguration {
    LaneConfiguration {
        model: ModelIdentity {
            provider: "faux".to_owned(),
            model_id: "faux-1".to_owned(),
        },
        thinking_level: ThinkingLevel::Low,
        active_tool_names: vec!["read".to_owned()],
    }
}

/// Closes the ledgered sessions and clears the ledger, upstream's
/// `afterEach` teardown.
///
/// # Panics
/// A session close's failure.
pub(super) async fn close_sessions(sessions: &mut SessionLedger) {
    for session in sessions.drain(..) {
        session
            .close(&background_context())
            .await
            .expect("the session closes");
    }
}
