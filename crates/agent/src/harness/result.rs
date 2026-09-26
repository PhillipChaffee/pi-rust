//! The harness error taxonomy, ported from upstream `src/harness/result.ts`.
//!
//! Upstream's hand-rolled `Result`/`ok`/`err` restates as
//! [`std::result::Result`] (see the [`super::types`] module docs) and its
//! `matchError` dispatcher restates as a `match` — exhaustive over
//! [`HarnessError`]'s variants, which is the guarantee upstream's
//! `ErrorMatchers` table buys.
//!
//! The thirteen `TaggedError` classes collapse into one enum whose
//! serialized form keeps upstream's wire shape: `{ "_tag": "LaneBusy",
//! "message": ..., ...props }`, the `toJSON` the session layer persists.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::harness::session::types::OperationKind;

/// The tagged harness operation errors, upstream's `TaggedError` family in
/// `result.ts` (`LaneBusy`, `Closed`, `InvalidNavigation`, and friends).
///
/// Each variant carries the tagged class's fields plus its `message`; the
/// serialized tag rides as `"_tag"` with camelCase fields, matching the
/// `toJSON` upstream builds. `is(value)` checks restate as `matches!`
/// over the variant, and the factory's dynamic property assignment
/// restates as the variant's fields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all_fields = "camelCase")]
pub enum HarnessError {
    /// The lane already has an active operation, upstream's `LaneBusy`.
    LaneBusy {
        /// The lane that rejected the operation.
        lane: String,
        /// The operation id the lane is busy with.
        operation_id: String,
        /// The kind of operation the lane is busy with.
        operation_kind: OperationKind,
        /// The failure message.
        message: String,
    },
    /// The operation no longer matches the lane's current invocation,
    /// upstream's `OperationMismatch`.
    OperationMismatch {
        /// The lane that reported the mismatch.
        lane: String,
        /// The operation id the caller expected.
        expected_operation_id: String,
        /// The lane's current operation id, when one is active.
        current_operation_id: Option<String>,
        /// The last settled operation id.
        last_operation_id: Option<String>,
        /// The human-readable failure message.
        message: String,
    },
    /// No run invocation is active on the lane, upstream's `NoActiveRun`.
    NoActiveRun {
        /// The lane that has no active run.
        lane: String,
        /// The human-readable failure message.
        message: String,
    },
    /// No operation of any kind is active on the lane, upstream's
    /// `NoActiveOperation`.
    NoActiveOperation {
        /// The lane that has no active operation.
        lane: String,
        /// The human-readable failure message.
        message: String,
    },
    /// No suspended run exists to resume, upstream's `NothingToResume`.
    NothingToResume {
        /// The lane that has nothing to resume.
        lane: String,
        /// The human-readable failure message.
        message: String,
    },
    /// No compactable history exists, upstream's `NothingToCompact`.
    NothingToCompact {
        /// The lane that has nothing to compact.
        lane: String,
        /// The human-readable failure message.
        message: String,
    },
    /// The queued message was rejected as malformed, upstream's
    /// `InvalidMessage`.
    InvalidMessage {
        /// The lane that rejected the message.
        lane: String,
        /// Why the message was rejected.
        reason: String,
        /// The human-readable failure message.
        message: String,
    },
    /// The navigation target or options were rejected, upstream's
    /// `InvalidNavigation`.
    InvalidNavigation {
        /// The lane that rejected the navigation.
        lane: String,
        /// Why the navigation was rejected.
        reason: String,
        /// The human-readable failure message.
        message: String,
    },
    /// The named skill does not exist, upstream's `UnknownSkill`.
    UnknownSkill {
        /// The skill name that did not resolve.
        name: String,
        /// The human-readable failure message.
        message: String,
    },
    /// The named prompt template does not exist, upstream's
    /// `UnknownTemplate`.
    UnknownTemplate {
        /// The template name that did not resolve.
        name: String,
        /// The human-readable failure message.
        message: String,
    },
    /// The navigation target id does not resolve, upstream's
    /// `UnknownTarget`.
    UnknownTarget {
        /// The target id that did not resolve.
        target_id: String,
        /// The human-readable failure message.
        message: String,
    },
    /// The lane name or acquisition options were rejected, upstream's
    /// `InvalidLane`.
    InvalidLane {
        /// The lane name that was rejected.
        lane: String,
        /// Why the lane was rejected.
        reason: String,
        /// The human-readable failure message.
        message: String,
    },
    /// The harness is closed, upstream's `Closed`.
    Closed {
        /// The human-readable failure message.
        message: String,
    },
}

impl HarnessError {
    /// The upstream `_tag` value for the variant.
    #[must_use]
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::LaneBusy { .. } => "LaneBusy",
            Self::OperationMismatch { .. } => "OperationMismatch",
            Self::NoActiveRun { .. } => "NoActiveRun",
            Self::NoActiveOperation { .. } => "NoActiveOperation",
            Self::NothingToResume { .. } => "NothingToResume",
            Self::NothingToCompact { .. } => "NothingToCompact",
            Self::InvalidMessage { .. } => "InvalidMessage",
            Self::InvalidNavigation { .. } => "InvalidNavigation",
            Self::UnknownSkill { .. } => "UnknownSkill",
            Self::UnknownTemplate { .. } => "UnknownTemplate",
            Self::UnknownTarget { .. } => "UnknownTarget",
            Self::InvalidLane { .. } => "InvalidLane",
            Self::Closed { .. } => "Closed",
        }
    }

    /// The variant's human-readable `message` field.
    ///
    /// # Panics
    /// Never; every variant carries a `message` field.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::LaneBusy { message, .. }
            | Self::OperationMismatch { message, .. }
            | Self::NoActiveRun { message, .. }
            | Self::NothingToResume { message, .. }
            | Self::NothingToCompact { message, .. }
            | Self::InvalidMessage { message, .. }
            | Self::InvalidNavigation { message, .. }
            | Self::UnknownSkill { message, .. }
            | Self::UnknownTemplate { message, .. }
            | Self::UnknownTarget { message, .. }
            | Self::InvalidLane { message, .. }
            | Self::Closed { message }
            | Self::NoActiveOperation { message, .. } => message,
        }
    }
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for HarnessError {}

/// An unexpected harness-internal failure with its cause, upstream's
/// `HarnessFault`.
///
/// Upstream carries `cause: unknown`; the port carries the typed source
/// box.
#[derive(Debug)]
pub struct HarnessFault {
    /// The human-readable failure message.
    pub message: String,
    /// The underlying failure.
    pub cause: Box<dyn std::error::Error + Send + Sync>,
}

impl HarnessFault {
    /// Builds a fault from its message and cause.
    #[must_use]
    pub fn new(
        message: impl Into<String>,
        cause: Box<dyn std::error::Error + Send + Sync>,
    ) -> Self {
        Self {
            message: message.into(),
            cause,
        }
    }
}

impl fmt::Display for HarnessFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HarnessFault {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}

/// The harness closed while an operation was active, upstream's
/// `HarnessClosed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessClosed;

impl fmt::Display for HarnessClosed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AgentHarness was closed while the operation was active")
    }
}

impl std::error::Error for HarnessClosed {}

#[cfg(test)]
mod tests;
