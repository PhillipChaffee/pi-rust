//! The in-process transport, ported from upstream
//! `src/services/loopback.ts`.
//!
//! Connects a provider to a binding without changing remote service
//! semantics: invocations delegate straight to the provider, and
//! subscriptions carry the provider's snapshot with its activate/close
//! gates.

use std::rc::Rc;

use crate::context::Context;
use crate::future::{LocalBoxFuture, boxed};
use crate::services::provider::RemoteServiceProvider;
use crate::types::{
    JsonValue, RemoteServiceTransport, ServiceMode, ServiceProviderListener, ServiceSubscription,
};

/// Connects a provider to a binding without changing remote service
/// semantics, upstream's `createLoopbackServiceTransport`.
#[must_use]
pub fn create_loopback_service_transport(
    provider: &RemoteServiceProvider,
) -> Rc<dyn RemoteServiceTransport> {
    Rc::new(LoopbackTransport {
        provider: provider.clone(),
    })
}

struct LoopbackTransport {
    provider: RemoteServiceProvider,
}

impl RemoteServiceTransport for LoopbackTransport {
    fn invoke(
        &self,
        call: crate::types::ServiceCall,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, crate::errors::ChordError>> {
        self.provider.invoke(call, context)
    }

    fn subscribe(
        &self,
        service_id: String,
        mode: ServiceMode,
        listener: ServiceProviderListener,
        _context: Context,
    ) -> LocalBoxFuture<Result<ServiceSubscription, crate::errors::ChordError>> {
        let subscription = self.provider.subscribe(&service_id, mode, listener);
        boxed(async move {
            let subscription = subscription?;
            let snapshot = subscription.snapshot.clone();
            let activate = subscription.activate;
            let close = subscription.close;
            Ok(ServiceSubscription {
                snapshot,
                activate,
                close,
            })
        })
    }
}
