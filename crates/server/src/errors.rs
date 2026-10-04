//! The routed-error taxonomy, ported from upstream `src/errors.ts` plus the
//! arbitrary `Error` instances the host seams carry.
//!
//! Upstream's server classifies whatever a host raises with `instanceof`
//! checks (`ServerError`, chord's `RemoteServiceError`, protocol's
//! `ProtocolValidationError`) and answers everything else with the opaque
//! internal-error response after reporting it. The port carries the same
//! discrimination as the owned [`Failure`] enum: the first three arms cross
//! the wire with their code and message, every other arm answers
//! `internal_error` and is reported to the [`Server`](crate::Server)'s error
//! observer. Identity-carrying opaque errors ride an `Rc`, so a listener or
//! host failure rethrown verbatim (upstream's `rejects.toBe(failure)`) stays
//! observable to the caller.

use std::fmt;
use std::rc::Rc;

use pi_chord::errors::{RemoteServiceError, RemoteServiceErrorCode};
use pi_protocol::ProtocolValidationError;

/// The error code a host or lifecycle failure can safely cross the protocol
/// boundary with.
///
/// Upstream's `ServerOperationErrorCode` union: the eight chord
/// remote-service codes plus the five server-routed codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerOperationErrorCode {
    /// A chord remote-service code, upstream's `RemoteServiceErrorCode` arm.
    Remote(RemoteServiceErrorCode),
    /// `wrong_server`
    WrongServer,
    /// `session_not_found`
    SessionNotFound,
    /// `session_ambiguous`
    SessionAmbiguous,
    /// `session_not_attached`
    SessionNotAttached,
    /// `server_draining`
    ServerDraining,
}

impl fmt::Display for ServerOperationErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Remote(code) => formatter.write_str(code.as_str()),
            Self::WrongServer => formatter.write_str("wrong_server"),
            Self::SessionNotFound => formatter.write_str("session_not_found"),
            Self::SessionAmbiguous => formatter.write_str("session_ambiguous"),
            Self::SessionNotAttached => formatter.write_str("session_not_attached"),
            Self::ServerDraining => formatter.write_str("server_draining"),
        }
    }
}

/// The message every unclassified failure answers the wire with, upstream's
/// `INTERNAL_SERVER_ERROR_MESSAGE`.
pub const INTERNAL_SERVER_ERROR_MESSAGE: &str = "Internal server error";

/// A host or lifecycle error that can safely cross the protocol boundary,
/// upstream's `ServerError` with its five subclasses restated as
/// constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerError {
    /// The wire's name for what failed.
    pub code: ServerOperationErrorCode,
    /// The human-readable message.
    pub message: String,
}

impl ServerError {
    /// Builds an error carrying `code` and `message`, upstream's
    /// `new ServerError(code, message)`.
    #[must_use]
    pub fn new(code: ServerOperationErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// `WrongServerError`: the request was addressed to another server.
    #[must_use]
    pub fn wrong_server() -> Self {
        Self::new(
            ServerOperationErrorCode::WrongServer,
            "Request was addressed to another server",
        )
    }

    /// `SessionNotFoundError`, with upstream's default message when none is
    /// supplied.
    #[must_use]
    pub fn session_not_found(message: impl Into<String>) -> Self {
        Self::new(ServerOperationErrorCode::SessionNotFound, message)
    }

    /// `SessionAmbiguousError`: the id matches more than one session.
    #[must_use]
    pub fn session_ambiguous() -> Self {
        Self::new(
            ServerOperationErrorCode::SessionAmbiguous,
            "Session ID matches more than one session",
        )
    }

    /// `SessionNotAttachedError`: the session is not attached to this
    /// client.
    #[must_use]
    pub fn session_not_attached() -> Self {
        Self::new(
            ServerOperationErrorCode::SessionNotAttached,
            "Session is not attached to this client",
        )
    }

    /// `ServerDrainingError`: the server is draining.
    #[must_use]
    pub fn server_draining() -> Self {
        Self::new(
            ServerOperationErrorCode::ServerDraining,
            "Server is draining",
        )
    }
}

impl fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ServerError {}

/// Any failure the crate routes: a bounded [`ServerError`], chord's
/// [`RemoteServiceError`], protocol's [`ProtocolValidationError`], an
/// aggregate of several, or an identity-carrying opaque error.
#[derive(Debug, Clone)]
pub enum Failure {
    /// A bounded routing error, upstream's `ServerError`.
    Server(ServerError),
    /// A chord remote-service failure, upstream's `RemoteServiceError`.
    Remote(RemoteServiceError),
    /// A protocol validation failure, upstream's `ProtocolValidationError`.
    Validation(ProtocolValidationError),
    /// Several failures collected by one shutdown or release fan-out,
    /// upstream's `AggregateError`.
    Aggregate {
        /// The message that wraps the collected failures.
        message: String,
        /// The collected failures.
        errors: Vec<Self>,
    },
    /// The drain-time cleanup aggregate, upstream's `SessionCleanupError
    /// extends AggregateError`: distinct from [`Failure::Aggregate`] because
    /// the router's close collects exactly these.
    Cleanup {
        /// The message that wraps the collected failures.
        message: String,
        /// The collected failures.
        errors: Vec<Self>,
    },
    /// Any other defect; the `Rc` preserves identity for verbatim rethrows,
    /// upstream's arbitrary `Error` instances.
    Other(Rc<dyn std::error::Error>),
}

impl Failure {
    /// An opaque failure carrying only a message, upstream's `new
    /// Error(message)` throws.
    #[must_use]
    pub fn message(message: impl Into<String>) -> Self {
        Self::Other(Rc::new(MessageError(message.into())))
    }

    /// An opaque failure wrapping a caller-owned error; the `Rc` identity
    /// survives every rethrow, upstream's `rejects.toBe(failure)` contract.
    #[must_use]
    pub fn other(error: Rc<dyn std::error::Error>) -> Self {
        Self::Other(error)
    }

    /// The code and message this failure crosses the wire with, upstream's
    /// `toProtocolError` classification without its reporting side effect:
    /// bounded errors carry their own code, a validation failure restates to
    /// `invalid_request`, everything else answers the opaque internal error.
    pub(crate) fn wire_error(&self) -> pi_protocol::ProtocolError {
        match self {
            Self::Server(error) => pi_protocol::ProtocolError {
                code: error.code.to_string(),
                message: error.message.clone(),
            },
            Self::Remote(error) => pi_protocol::ProtocolError {
                code: error.code.as_str().to_string(),
                message: error.message.clone(),
            },
            Self::Validation(error) => pi_protocol::ProtocolError {
                code: "invalid_request".to_string(),
                message: error.message().to_string(),
            },
            Self::Aggregate { .. } | Self::Cleanup { .. } | Self::Other(_) => {
                pi_protocol::ProtocolError {
                    code: "internal_error".to_string(),
                    message: INTERNAL_SERVER_ERROR_MESSAGE.to_string(),
                }
            }
        }
    }

    /// Whether this is the drain-time cleanup aggregate, upstream's
    /// `instanceof SessionCleanupError` check the router's close filters on.
    pub(crate) const fn is_session_cleanup(&self) -> bool {
        matches!(self, Self::Cleanup { .. })
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Server(error) => write!(formatter, "{error}"),
            Self::Remote(error) => write!(formatter, "{error}"),
            Self::Validation(error) => write!(formatter, "{}", error.message()),
            Self::Aggregate { message, .. } | Self::Cleanup { message, .. } => {
                formatter.write_str(message)
            }
            Self::Other(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Other(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct MessageError(String);

impl fmt::Display for MessageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MessageError {}
