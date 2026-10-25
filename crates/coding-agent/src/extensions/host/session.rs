//! The live extension session over an injected duplex link ([ADR 0010]).
//!
//! [`HostSession::connect`] runs the handshake and registration pass, binds
//! the child's declarations into an in-process [`Extension`] whose handlers
//! proxy back over the link, and leaves the session ready for the runner's
//! dispatch. One call scope runs at a time: a dispatch sends its call frame
//! and reads until the matching result, answering context and UI requests
//! and forwarding tool updates and provider stream events on the way.
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::sync::Arc;

use tokio::io::{AsyncRead, AsyncWrite};

use crate::extensions::api::PendingProviderRegistration;
use crate::extensions::runtime::{Extension, ExtensionError, HandlerContext};
use crate::source_info::SourceInfo;

use super::context::UiCallHandler;
use super::{ExtensionIdentity, WireError};

mod call;
mod connection;
mod dispatch;
#[cfg(test)]
mod tests;

use connection::{SessionCore, handshake, read_registrations};

/// The dispatch-error listener, upstream's `ExtensionErrorListener`.
pub(super) type ErrorListener = Arc<dyn Fn(&ExtensionError) + Send + Sync>;

/// The ingredients the host session binds at connect time.
pub struct HostConnectOptions {
    /// The path the extension was requested by.
    pub path: String,
    /// The resolved load path.
    pub resolved_path: String,
    /// The provenance the loader stamps on the registrations.
    pub source_info: SourceInfo,
    /// The context the session answers context calls from.
    pub context: HandlerContext,
    /// The UI pass-through, absent when no UI is bound.
    pub ui: Option<Arc<dyn UiCallHandler>>,
    /// The listener a crash or handler failure reports to.
    pub error_listener: Option<ErrorListener>,
}

impl std::fmt::Debug for HostConnectOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostConnectOptions")
            .field("path", &self.path)
            .field("resolved_path", &self.resolved_path)
            .field("source_info", &self.source_info)
            .field("context", &self.context)
            .field("has_ui", &self.ui.is_some())
            .field("has_error_listener", &self.error_listener.is_some())
            .finish()
    }
}

/// A connected extension: the in-process record, its identity, and the
/// provider registrations the bind pass queues.
pub struct HostSession<T> {
    core: Arc<SessionCore<T>>,
    identity: ExtensionIdentity,
    extension: Arc<Extension>,
    providers: Vec<PendingProviderRegistration>,
}

impl<T> std::fmt::Debug for HostSession<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostSession")
            .field("identity", &self.identity)
            .field("extension", &self.extension)
            .field("providers", &self.providers.len())
            .finish_non_exhaustive()
    }
}

impl<T> HostSession<T>
where
    T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    /// Handshake, read the declarations, and bind the extension record.
    ///
    /// # Errors
    ///
    /// Returns [`WireError`] when the handshake is refused, a registration
    /// frame is malformed or unexpected, or the link ends first.
    pub async fn connect(io: T, options: HostConnectOptions) -> Result<Self, WireError> {
        let core = Arc::new(SessionCore::new(io, &options)?);
        let identity = handshake(&core).await?;
        let registrations = read_registrations(&core).await?;
        let extension = dispatch::binding::build_extension(&core, &options, &registrations);
        let providers = dispatch::binding::build_providers(&core, &options, &registrations);
        Ok(Self {
            core,
            identity,
            extension,
            providers,
        })
    }

    /// The bound in-process record the runner dispatches through.
    #[must_use]
    pub fn extension(&self) -> Arc<Extension> {
        Arc::clone(&self.extension)
    }

    /// The identity the child announced.
    #[must_use]
    pub const fn identity(&self) -> &ExtensionIdentity {
        &self.identity
    }

    /// The provider registrations the bind pass queues, each carrying the
    /// stream closure that proxies `call_provider` back over the link.
    #[must_use]
    pub fn take_provider_registrations(&mut self) -> Vec<PendingProviderRegistration> {
        std::mem::take(&mut self.providers)
    }

    /// Whether the link is dead after a crash or protocol failure.
    #[must_use]
    pub fn is_dead(&self) -> bool {
        self.core.is_dead()
    }

    /// Close the write half of the link.
    ///
    /// # Errors
    ///
    /// Returns [`WireError`] when the transport fails to close.
    pub async fn close(&self) -> Result<(), WireError> {
        self.core.close().await
    }
}
