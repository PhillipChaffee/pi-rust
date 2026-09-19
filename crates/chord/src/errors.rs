//! Error model of the runtime.
//!
//! The eight `RemoteServiceErrorCode` variants ported from upstream
//! `src/services/errors.ts` with the ordered `REMOTE_SERVICE_ERROR_CODES`
//! const and its parse surface, plus the crate-level `ChordError` that
//! aggregates several failures into one (upstream's `AggregateError` role)
//! when disposal or activation collects multiple failures.

use std::fmt;

/// The error code the wire carries for one remote-service failure, ported
/// from upstream's `RemoteServiceErrorCode` union.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteServiceErrorCode {
    /// `service_not_allowed`
    ServiceNotAllowed,
    /// `service_not_found`
    ServiceNotFound,
    /// `service_mode_mismatch`
    ServiceModeMismatch,
    /// `service_member_not_found`
    ServiceMemberNotFound,
    /// `service_member_mismatch`
    ServiceMemberMismatch,
    /// `service_instance_not_found`
    ServiceInstanceNotFound,
    /// `service_stale_instance`
    ServiceStaleInstance,
    /// `service_invalid_value`
    ServiceInvalidValue,
}

impl RemoteServiceErrorCode {
    /// The identifier the `$chord.service` wire carries, in the upstream
    /// `REMOTE_SERVICE_ERROR_CODES` order.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ServiceNotAllowed => "service_not_allowed",
            Self::ServiceNotFound => "service_not_found",
            Self::ServiceModeMismatch => "service_mode_mismatch",
            Self::ServiceMemberNotFound => "service_member_not_found",
            Self::ServiceMemberMismatch => "service_member_mismatch",
            Self::ServiceInstanceNotFound => "service_instance_not_found",
            Self::ServiceStaleInstance => "service_stale_instance",
            Self::ServiceInvalidValue => "service_invalid_value",
        }
    }

    /// Parses the wire spelling, the `isRemoteServiceErrorCode` check.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let code = match value {
            "service_not_allowed" => Self::ServiceNotAllowed,
            "service_not_found" => Self::ServiceNotFound,
            "service_mode_mismatch" => Self::ServiceModeMismatch,
            "service_member_not_found" => Self::ServiceMemberNotFound,
            "service_member_mismatch" => Self::ServiceMemberMismatch,
            "service_instance_not_found" => Self::ServiceInstanceNotFound,
            "service_stale_instance" => Self::ServiceStaleInstance,
            "service_invalid_value" => Self::ServiceInvalidValue,
            _ => return None,
        };
        Some(code)
    }
}

impl fmt::Display for RemoteServiceErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A remote-boundary rejection with its wire code, upstream's
/// `RemoteServiceError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteServiceError {
    /// The wire's name for what failed.
    pub code: RemoteServiceErrorCode,
    /// The human-readable message, matching upstream's `Error.message`.
    pub message: String,
}

impl RemoteServiceError {
    /// Builds an error carrying `code` and `message`.
    #[must_use]
    pub fn new(code: RemoteServiceErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for RemoteServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RemoteServiceError {}

/// Any failure the runtime raises or a member throws: a plain message
/// (upstream's `Error`), a [`RemoteServiceError`], or an aggregation of
/// several (upstream's `AggregateError`).
#[derive(Debug, Clone)]
pub enum ChordError {
    /// A plain error message.
    Message(String),
    /// A remote-service failure with its wire code.
    Remote(RemoteServiceError),
    /// Several failures collected by a disposal or activation fan-out, with
    /// the message that wraps them.
    Aggregate(Vec<Self>, String),
}

impl ChordError {
    /// A plain message, the shape most rejections carry.
    #[must_use]
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl fmt::Display for ChordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => f.write_str(message),
            Self::Remote(error) => f.write_str(&error.message),
            Self::Aggregate(errors, message) => {
                write!(f, "{message} ({} errors:", errors.len())?;
                for (index, error) in errors.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, " {error}")?;
                }
                f.write_str(")")
            }
        }
    }
}

impl std::error::Error for ChordError {}

impl From<RemoteServiceError> for ChordError {
    fn from(error: RemoteServiceError) -> Self {
        Self::Remote(error)
    }
}

impl From<String> for ChordError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

impl From<&str> for ChordError {
    fn from(message: &str) -> Self {
        Self::Message(message.to_string())
    }
}

/// Collects one-or-many failures into the shape upstream throws: the single
/// failure itself, or an aggregate when several accumulated.
#[must_use]
pub fn collect_errors(errors: Vec<ChordError>, message: impl Into<String>) -> Option<ChordError> {
    match errors.len() {
        0 => None,
        1 => Some(errors.into_iter().next().unwrap_or(ChordError::Message(String::new()))),
        _ => Some(ChordError::Aggregate(errors, message.into())),
    }
}
impl From<crate::delta::DeltaError> for ChordError {
    fn from(error: crate::delta::DeltaError) -> Self {
        Self::Message(error.to_string())
    }
}
