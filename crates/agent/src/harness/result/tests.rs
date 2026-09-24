//! The error taxonomy's display and tag surface, upstream's
//! `matchError`/`is`/`message` table exercised through the one enum.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use crate::harness::result::{HarnessClosed, HarnessError, HarnessFault};
use crate::harness::session::types::OperationKind;

/// Every variant carries its upstream `_tag` and a `message` field.
#[test]
fn every_variant_carries_its_tag_and_message() {
    let errors = vec![
        HarnessError::LaneBusy {
            lane: "main".to_owned(),
            operation_id: "run".to_owned(),
            operation_kind: OperationKind::Run,
            message: "busy".to_owned(),
        },
        HarnessError::Closed {
            message: "closed".to_owned(),
        },
        HarnessError::UnknownSkill {
            name: "skill".to_owned(),
            message: "unknown".to_owned(),
        },
        HarnessError::NoActiveOperation {
            lane: "main".to_owned(),
            message: "no operation".to_owned(),
        },
    ];
    for error in errors {
        assert!(!error.tag().is_empty());
        assert_eq!(error.to_string(), error.message());
    }
    let _ = OperationKind::Compaction;
}

/// A fault chains its typed source; the closed marker carries the fixed
/// upstream message.
#[test]
fn faults_chain_their_source() {
    let fault = HarnessFault::new("fault", Box::new(std::io::Error::other("cause")));
    assert_eq!(fault.to_string(), "fault");
    let source = std::error::Error::source(&fault);
    assert!(source.is_some());
    let closed = HarnessClosed;
    assert_eq!(closed.to_string(), "AgentHarness was closed while the operation was active");
}