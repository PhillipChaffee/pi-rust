//! Shared fixtures for the session-layer suites, mirroring the shapes
//! upstream's `test/harness/*.test.ts` build (the injected clock, the
//! metadata, the message factories, and the deferred gate).

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]
#![expect(
    clippy::expect_used,
    reason = "the fixtures pin shapes; an unexpected result panics the test by design"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use pi_agent_core::harness::session::memory::{
    MemorySessionRepo, MemorySessionRepoOptions, MemoryStorage,
};
use pi_agent_core::harness::session::session::StorageBackedSession;
use pi_agent_core::harness::session::types::{
    CommitResult, Session, SessionError, SessionMetadata, Storage,
};
use pi_agent_core::harness::session::values::Write;
use pi_agent_core::types::AgentMessage;

/// The clock value the suites pin, upstream's `NOW`.
pub const NOW: i64 = 1_700_000_000_000;

/// The entry id the suites pin, upstream's `ENTRY_ID`.
pub const ENTRY_ID: &str = "00000000-0000-7000-8000-000000000001";

/// The metadata the suites pin, upstream's `metadata` fixture.
pub fn metadata() -> SessionMetadata {
    SessionMetadata {
        id: "session".to_owned(),
        created_at: NOW,
        storage_version: 1,
        cwd: Some("/workspace".to_owned()),
        parent_session_id: None,
        legacy_parent_session_path: None,
    }
}

/// The clock factory upstream's `now: () => timestamp++` restates: each call
/// returns the current value and then advances it.
pub fn ticking_clock() -> pi_agent_core::harness::session::memory::NowFn {
    let counter = Arc::new(AtomicI64::new(NOW));
    Arc::new(move || counter.fetch_add(1, Ordering::AcqRel))
}

/// The fixed clock factory upstream's `now: () => NOW` restates.
pub fn fixed_clock(value: i64) -> pi_agent_core::harness::session::memory::NowFn {
    Arc::new(move || value)
}

/// The user message the suites build, upstream's
/// `{ role: "user", content: [{ type: "text", text }], timestamp: NOW }`.
pub fn user_message(text: &str) -> AgentMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": [{ "type": "text", "text": text }],
        "timestamp": NOW,
    }))
    .expect("user message wire")
}

/// The assistant message the suites build with a pinned stop reason,
/// upstream's `assistantMessage(stopReason)`.
pub fn assistant_message(stop_reason: &str, text: &str) -> AgentMessage {
    let content = if stop_reason == "toolUse" {
        serde_json::json!([{ "type": "toolCall", "id": "call", "name": "read", "arguments": {} }])
    } else {
        serde_json::json!([{ "type": "text", "text": text }])
    };
    let mut wire = serde_json::json!({
        "role": "assistant",
        "content": content,
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
        },
        "stopReason": stop_reason,
        "timestamp": NOW,
    });
    if stop_reason == "deferred" {
        wire["deferred"] = serde_json::json!({
            "provider": "anthropic",
            "modelId": "claude-sonnet-4-5",
            "api": "anthropic-messages",
            "id": "job",
        });
    }
    serde_json::from_value(wire).expect("assistant message wire")
}

/// The zero usage row the suites compare, upstream's inline `usage` fixture.
pub fn zero_usage() -> pi_ai::types::Usage {
    pi_ai::types::Usage::default()
}

/// The one-shot release gate upstream's `deferred()` restates.
pub fn deferred() -> (
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    tokio::sync::oneshot::channel()
}

/// The repo factory most suites share, upstream's
/// `new MemorySessionRepo({ now: () => NOW })`.
pub fn memory_repo() -> MemorySessionRepo {
    MemorySessionRepo::new(MemorySessionRepoOptions {
        now: Some(fixed_clock(NOW)),
    })
}

/// The commit-through-mutator helper the suites share, upstream's
/// `commitSession(session, transaction)`.
pub async fn commit_session(
    session: &dyn Session,
    transaction: Vec<Write>,
) -> Result<CommitResult, SessionError> {
    use pi_agent_core::harness::session::session::StorageBackedSession;
    let context = pi_agent_core::harness::context::background_context();
    let boxed = session
        .mutate(
            StorageBackedSession::commit_writes_callback(transaction),
            &context,
        )
        .await?;
    boxed.downcast::<CommitResult>().map_or_else(
        |_| {
            Err(SessionError::Message(
                "expected the commit result back from the mutator".to_owned(),
            ))
        },
        |result| Ok(*result),
    )
}

/// The commit hook's type, the overridden `commit` body the owner supplies.
pub type CommitHook = Box<
    dyn Fn(
            Arc<MemoryStorage>,
            Vec<Write>,
            pi_agent_core::harness::context::Context,
        ) -> pi_ai::types::BoxedFuture<'static, Result<CommitResult, SessionError>>
        + Send
        + Sync,
>;

/// The delegating storage the controlled delegates compose, upstream's
/// `extends MemoryStorage` overrides: every read forwards to the memory
/// backend and the commit runs the owner's hook.
pub struct HookedStorage {
    /// The memory backend the reads forward to.
    pub base: Arc<MemoryStorage>,
    /// The commit hook the owner drives, upstream's overridden `commit`;
    /// it owns the backend handle and context clone, so the returned
    /// future is `'static`.
    pub commit_hook: CommitHook,
}

impl Storage for HookedStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<'_, Result<CommitResult, SessionError>> {
        let hook = &self.commit_hook;
        let base = self.base.clone();
        let context = context.clone();
        Box::pin(hook(base, writes, context))
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<
            std::collections::BTreeMap<String, pi_agent_core::harness::session::types::Entry>,
            SessionError,
        >,
    > {
        self.base.get_entries(ids, context)
    }

    fn get_value(
        &self,
        address: &pi_agent_core::harness::session::values::ValueAddress,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<Option<pi_agent_core::harness::session::values::StoredValue>, SessionError>,
    > {
        self.base.get_value(address, context)
    }

    fn scan_values(
        &self,
        prefix: &pi_agent_core::harness::session::values::ValueAddress,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<Vec<pi_agent_core::harness::session::values::StoredValue>, SessionError>,
    > {
        self.base.scan_values(prefix, context)
    }

    fn read_list(
        &self,
        address: &pi_agent_core::harness::session::values::ListAddress,
        options: Option<pi_agent_core::harness::session::values::ListReadOptions>,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<Vec<pi_agent_core::harness::session::values::ListElement>, SessionError>,
    > {
        self.base.read_list(address, options, context)
    }

    fn scan_branch(
        &self,
        query: &pi_agent_core::harness::session::types::StorageBranchScan,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<Vec<pi_agent_core::harness::session::types::Entry>, SessionError>,
    > {
        self.base.scan_branch(query, context)
    }

    fn scan_branch_structure(
        &self,
        query: &pi_agent_core::harness::session::types::StorageBranchScan,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<Vec<pi_agent_core::harness::session::types::EntryStructure>, SessionError>,
    > {
        self.base.scan_branch_structure(query, context)
    }

    fn scan_entries(
        &self,
        query: &pi_agent_core::harness::session::types::EntryScan,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<Vec<pi_agent_core::harness::session::types::Entry>, SessionError>,
    > {
        self.base.scan_entries(query, context)
    }

    fn scan_usage(
        &self,
        query: &pi_agent_core::harness::session::types::UsageScan,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<Vec<pi_agent_core::harness::session::types::UsageRow>, SessionError>,
    > {
        self.base.scan_usage(query, context)
    }

    fn get_stats(
        &self,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<
        '_,
        Result<pi_agent_core::harness::session::types::SessionStats, SessionError>,
    > {
        self.base.get_stats(context)
    }

    fn close(
        &self,
        context: &pi_agent_core::harness::context::Context,
    ) -> pi_ai::types::BoxedFuture<'_, Result<(), SessionError>> {
        self.base.close(context)
    }
}

/// The memory backend the session suites wrap, with the fixed NOW clock.
pub fn memory_storage() -> Arc<MemoryStorage> {
    Arc::new(MemoryStorage::new(
        pi_agent_core::harness::session::memory::MemoryStorageOptions {
            now: Some(fixed_clock(NOW)),
        },
    ))
}

/// The queued mutation probe upstream's `session.mutate(() => { queuedStarted
/// = true })`: the spawn drives the future, the flag records whether the
/// callback body started.
pub fn queued_mutate_probe(
    session: Arc<StorageBackedSession>,
    started: Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        session
            .mutate(
                Box::new(
                    move |_mutator: &dyn pi_agent_core::harness::session::types::SessionMutator,
                          _context|
                          -> pi_ai::types::BoxedFuture<
                        '_,
                        Result<Box<dyn std::any::Any + Send>, SessionError>,
                    > {
                        started.store(true, Ordering::Release);
                        let done: Box<dyn std::any::Any + Send> = Box::new(());
                        Box::pin(std::future::ready(Ok(done)))
                    },
                ),
                &pi_agent_core::harness::context::background_context(),
            )
            .await
            .expect("queued mutate");
    })
}

/// The storage-backed session the suites construct over the shared memory
/// backend, with the fixture metadata.
pub fn storage_backed_session(storage: Arc<dyn Storage>) -> StorageBackedSession {
    StorageBackedSession::new(
        metadata(),
        storage,
        pi_agent_core::harness::session::session::StorageBackedSessionOptions::default(),
    )
}

/// The second-commit-attempt assert, upstream's
/// `rejects.toThrow("commit already attempted")`.
///
/// # Panics
/// When the second attempt does not reject with the consumed-guard error.
pub async fn assert_second_attempt_rejected(
    mutator: &dyn pi_agent_core::harness::session::types::SessionMutator,
    context: &pi_agent_core::harness::context::Context,
) {
    let rejected = mutator.commit(Vec::new(), context).await;
    assert!(
        rejected
            .err()
            .expect("second attempt")
            .to_string()
            .contains("commit already attempted"),
    );
}
