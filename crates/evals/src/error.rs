//! The crate-wide error type.
//!
//! Upstream throws `TypeError` (plan validation, observation validation) or
//! plain `Error` (reader failures) with a message the tests assert on; the
//! port keeps one message-carrying error type for all of them.

/// An error carrying the upstream failure message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalError(String);

impl EvalError {
    /// Builds an error whose [`std::fmt::Display`] output is `message`.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The upstream failure message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EvalError {}

impl From<std::io::Error> for EvalError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}
