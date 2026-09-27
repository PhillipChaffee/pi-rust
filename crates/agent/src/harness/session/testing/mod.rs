//! The storage decorators of the conformance surface, ported from upstream
//! `src/harness/session/testing/`: the forwarding base, the commit-attempt
//! recorder, and the deterministic commit gate.

pub mod conformance;
mod gating_storage;
mod instrumented_storage;
mod storage_decorator;
mod types;

pub use conformance::session_repo::{
    create_session_repo_conformance, create_session_repo_fork_behavior_conformance,
    create_session_repo_fork_conformance, create_session_repo_fork_coordination_conformance,
    create_session_repo_fork_destination_reservation_conformance,
    create_session_repo_fork_source_snapshot_conformance,
    create_session_repo_lifecycle_conformance, create_session_repo_message_conformance,
    create_session_repo_ownership_conformance, create_session_repo_streaming_fork_conformance,
};
pub use conformance::storage::create_storage_conformance;
pub use gating_storage::GatingStorage;
pub use instrumented_storage::InstrumentedStorage;
pub use storage_decorator::StorageDecorator;
pub use types::{ConformanceCase, RepoFixture, StorageFixture};
