//! The client error taxonomy, ported from upstream `src/errors.ts`.
//!
//! Upstream ships one `Error` subclass per failure kind plus normalization
//! helpers over arbitrary throwables (`toError`, `toDisconnectedError`) and
//! a cause-walking errno probe (`isErrorCode`). The port restates the kinds
//! as one enum: every failure crossing the client is already a
//! [`ClientError`], so the `unknown` → `Error` normalization is the identity
//! and drops out, while the disconnected wrapper and the errno walk carry
//! over unchanged.

use std::fmt;

use pi_protocol::ProtocolError;

/// A failure the client surfaces, the union of upstream's error classes.
///
/// - [`Server`](Self::Server) restates upstream's `ServerError`: the wire's
///   `ProtocolError{code, message}` a failed response carried.
/// - [`Disconnected`](Self::Disconnected) restates `DisconnectedError`,
///   including its optional `cause`.
/// - [`Disposed`](Self::Disposed) restates `ClientDisposedError`.
/// - [`Protocol`](Self::Protocol) restates the re-raised
///   `ProtocolValidationError`s the client surfaces for invalid envelopes
///   and payloads.
/// - [`Other`](Self::Other) restates upstream's generic `Error`: transport
///   I/O, discovery filesystem walks, and abort reasons. `code` carries the
///   errno name the cause reported (`ENOENT`, `ECONNREFUSED`, ...) when one
///   exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientError {
    /// A typed failure the server reported on a failed response.
    Server(ProtocolError),
    /// The connection is unusable. `message` mirrors upstream's
    /// `DisconnectedError.message`; `cause` the wrapped failure.
    Disconnected {
        /// Why the connection is unusable.
        message: String,
        /// The underlying failure, when one exists.
        cause: Option<Box<Self>>,
    },
    /// The client was disposed.
    Disposed,
    /// A protocol or validation failure surfaced through the client.
    Protocol(pi_protocol::ProtocolValidationError),
    /// Any other failure: transport I/O, discovery filesystem walks, abort
    /// reasons.
    Other {
        /// The human-readable message, upstream's `Error.message`.
        message: String,
        /// The errno name the cause reported, when present.
        code: Option<String>,
    },
}

impl ClientError {
    /// The default disconnected failure, upstream's
    /// `new DisconnectedError()`.
    #[must_use]
    pub fn disconnected() -> Self {
        Self::Disconnected {
            message: "Client is disconnected".to_string(),
            cause: None,
        }
    }

    /// A disconnected failure with a custom message, upstream's
    /// `new DisconnectedError(message)`.
    #[must_use]
    pub fn disconnected_with(message: impl Into<String>) -> Self {
        Self::Disconnected {
            message: message.into(),
            cause: None,
        }
    }

    /// A generic failure without an errno code, upstream's
    /// `new Error(message)`.
    #[must_use]
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
            code: None,
        }
    }

    /// A generic failure carrying an errno name, upstream's errors whose
    /// `code` property holds one (`ENOENT`, `ENOTDIR`, ...).
    #[must_use]
    pub fn other_with_code(message: impl Into<String>, code: Option<String>) -> Self {
        Self::Other {
            message: message.into(),
            code,
        }
    }

    /// Maps one I/O failure onto the taxonomy, upstream's `Error` instances
    /// carrying a `code` property for errno failures.
    #[must_use]
    pub fn from_io(error: &std::io::Error) -> Self {
        Self::Other {
            message: error.to_string(),
            code: unix_error_name(error),
        }
    }
}

impl fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Server(error) => write!(formatter, "{}", error.message),
            Self::Disconnected { message, .. } | Self::Other { message, .. } => {
                write!(formatter, "{message}")
            }
            Self::Disposed => write!(formatter, "Client is disposed"),
            Self::Protocol(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for ClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Disconnected {
                cause: Some(cause), ..
            } => Some(cause.as_ref()),
            _ => None,
        }
    }
}

/// Wraps any failure as the disconnected kind, upstream's
/// `toDisconnectedError`: an already-disconnected failure passes through,
/// everything else wraps as the cause with the cause's message.
#[must_use]
pub fn to_disconnected_error(error: &ClientError) -> ClientError {
    if matches!(error, ClientError::Disconnected { .. }) {
        return error.clone();
    }
    ClientError::Disconnected {
        message: error.to_string(),
        cause: Some(Box::new(error.clone())),
    }
}

/// Whether the failure or one of its causes reports `code`, upstream's
/// `isErrorCode` cause walk.
#[must_use]
pub fn is_error_code(error: &ClientError, code: &str) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if let ClientError::Other {
            code: Some(actual), ..
        } = error
            && actual == code
        {
            return true;
        }
        current = std::error::Error::source(error)
            .and_then(|source| source.downcast_ref::<ClientError>());
    }
    false
}

/// The errno name an I/O failure carries, the `code` property Node puts on
/// filesystem and socket errors. Only the errors the suite asserts on are
/// named; everything else maps to [`None`].
fn unix_error_name(error: &std::io::Error) -> Option<String> {
    let errno = error.raw_os_error()?;
    let name = match errno {
        errno::ENOENT => "ENOENT",
        errno::ENOTDIR => "ENOTDIR",
        errno::ECONNREFUSED => "ECONNREFUSED",
        errno::ECONNRESET => "ECONNRESET",
        errno::EPIPE => "EPIPE",
        errno::ETIMEDOUT => "ETIMEDOUT",
        _ => return None,
    };
    Some(name.to_string())
}

/// The platform errno values for the names [`unix_error_name`] maps. macOS
/// and Linux number connection failures differently; both tables are
/// needed because CI runs on Linux while development runs on macOS.
mod errno {
    #[cfg(target_os = "macos")]
    pub(super) const ENOENT: i32 = 2;
    #[cfg(target_os = "macos")]
    pub(super) const ENOTDIR: i32 = 20;
    #[cfg(target_os = "macos")]
    pub(super) const ECONNREFUSED: i32 = 61;
    #[cfg(target_os = "macos")]
    pub(super) const ECONNRESET: i32 = 54;
    #[cfg(target_os = "macos")]
    pub(super) const EPIPE: i32 = 32;
    #[cfg(target_os = "macos")]
    pub(super) const ETIMEDOUT: i32 = 60;

    #[cfg(target_os = "linux")]
    pub(super) const ENOENT: i32 = 2;
    #[cfg(target_os = "linux")]
    pub(super) const ENOTDIR: i32 = 20;
    #[cfg(target_os = "linux")]
    pub(super) const ECONNREFUSED: i32 = 111;
    #[cfg(target_os = "linux")]
    pub(super) const ECONNRESET: i32 = 104;
    #[cfg(target_os = "linux")]
    pub(super) const EPIPE: i32 = 32;
    #[cfg(target_os = "linux")]
    pub(super) const ETIMEDOUT: i32 = 110;
}
