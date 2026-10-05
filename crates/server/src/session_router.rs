//! The routed-Session attachment machine, ported from upstream
//! `src/session-router.ts`.
//!
//! Upstream serializes each client's routing operations behind a promise
//! chain (`runForClient`), tracks in-flight service calls per attachment,
//! and latches releases so a release started concurrently with a disconnect
//! runs once. The port restates the chain as one driver task per client
//! running queued operations FIFO, tracks calls with latches, and latches
//! releases the same way; client identity is the presentation's
//! [`ClientToken`]. A panicking host future aborts the operation that called
//! into it — upstream's promise chain would reject instead — so host code
//! reports failures through `Result`, and the driver task isolates the panic
//! so later operations still run.
//!
//! # Module-level lints
#![allow(
    clippy::redundant_pub_crate,
    reason = "SessionRouter is the crate-wide router; the server core is its only consumer"
)]

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use pi_agent_core::harness::context::{Context, background_context};
use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::types::{JsonValue, ServiceCall};

use crate::connection::ClientToken;
use crate::errors::{Failure, ServerError};
use crate::latch::CloseLatch as RouterCloseLatch;
use crate::latch::Latch;
use crate::server::ServerCore;
use crate::types::{
    HasSessionId, RoutedSessionAttachment, RoutedSessionHandle, ServerHost, ServicePublisher, ready,
};

/// The result one tracked service call settles with, upstream's
/// `Promise<JsonValue | undefined>`.
type TrackedResult = Result<Option<JsonValue>, Failure>;

/// The result acquiring one lease settles with, upstream's
/// `Promise<RoutedSessionAttachment>`.
type AcquireResult = Result<Rc<dyn RoutedSessionAttachment>, Failure>;

/// One open's outcome, upstream's `Promise<HostedSession>`.
type OpenResult = Result<Rc<HostedSession>, Failure>;

/// One in-flight open's shared slot: the join latch plus whether the open
/// failed, the join rule the interest note reads.
struct Opening {
    latch: Latch<OpenResult>,
    failed: Cell<bool>,
}

/// One hosted Session's shared routing state, upstream's `HostedSession`.
struct HostedSession {
    /// The durable session id, upstream's `id`.
    id: String,
    /// The host-provided handle, upstream's `handle`.
    handle: Rc<dyn RoutedSessionHandle>,
    /// The live attachments keyed by client identity, upstream's
    /// `attachments`; weak because each attachment roots its session.
    attachments: RefCell<HashMap<usize, Weak<ClientAttachment>>>,
}

/// One client's attachment to one hosted Session, upstream's
/// `ClientAttachment`.
struct ClientAttachment {
    /// The attachment identity the client routes on, upstream's `id`.
    id: String,
    /// The owning presentation's identity, upstream's `client`.
    client: ClientToken,
    /// The hosted Session the attachment routes into, upstream's `session`.
    session: Rc<HostedSession>,
    /// The in-flight service calls, upstream's `operations`.
    operations: RefCell<Vec<Rc<Latch<TrackedResult>>>>,
    /// The in-flight acquisition, upstream's `acquiring`.
    acquiring: RefCell<Option<Rc<Latch<AcquireResult>>>>,
    /// The acquired lease, upstream's `lease`.
    lease: RefCell<Option<Rc<dyn RoutedSessionAttachment>>>,
    /// The one-shot release, upstream's `releasing`.
    releasing: RefCell<Option<RouterCloseLatch>>,
}

/// One queued routing operation; the driver runs `run` and removes `settle`
/// from its pending list when the task completes.
struct RouterJob {
    run: LocalBoxFuture<()>,
    settle: Rc<Latch<()>>,
}

/// One client's serialized-operation driver, upstream's `clientOperations`
/// chain entry.
struct ClientDriver {
    /// The FIFO queue the driver drains.
    queue: tokio::sync::mpsc::UnboundedSender<RouterJob>,
    /// The settle latches of every queued operation, upstream's chain tail
    /// the close awaits.
    pending: Rc<RefCell<Vec<Rc<Latch<()>>>>>,
}

/// The routing bindings the server fixes, upstream's `SessionRouterOptions`;
/// the core back-reference resolves `isClosing`, attachment publication, and
/// error reporting against the owning server.
struct RouterInner<H: ServerHost> {
    host: Rc<H>,
    server_id: pi_protocol::ServerId,
    core: Weak<ServerCore<H>>,
    hosted_sessions: RefCell<HashMap<String, Rc<HostedSession>>>,
    opening_sessions: RefCell<HashMap<String, Rc<Opening>>>,
    attachments_by_client: RefCell<HashMap<usize, Rc<ClientAttachment>>>,
    disconnected_clients: RefCell<HashSet<usize>>,
    client_drivers: RefCell<HashMap<usize, Rc<ClientDriver>>>,
    close_latch: RefCell<Option<RouterCloseLatch>>,
    /// The session each client's in-flight request targets, recorded at
    /// dispatch so a request that arrives while its session's open is in
    /// flight joins that open, upstream's synchronous acquire-check.
    session_interest: RefCell<HashMap<usize, String>>,
}

/// Routes presentation requests to host-hosted Sessions, upstream's
/// `SessionRouter<TMetadata>`.
pub(crate) struct SessionRouter<H: ServerHost> {
    inner: Rc<RouterInner<H>>,
}

impl<H: ServerHost + 'static> SessionRouter<H> {
    pub(crate) fn new(
        host: Rc<H>,
        server_id: pi_protocol::ServerId,
        core: Weak<ServerCore<H>>,
    ) -> Self {
        Self {
            inner: Rc::new(RouterInner {
                host,
                server_id,
                core,
                hosted_sessions: RefCell::new(HashMap::new()),
                opening_sessions: RefCell::new(HashMap::new()),
                attachments_by_client: RefCell::new(HashMap::new()),
                disconnected_clients: RefCell::new(HashSet::new()),
                client_drivers: RefCell::new(HashMap::new()),
                close_latch: RefCell::new(None),
                session_interest: RefCell::new(HashMap::new()),
            }),
        }
    }

    /// Records the session a just-dispatched request targets, upstream's
    /// request arriving while an open is in flight.
    ///
    /// Only an open still in flight counts: a request whose attach call runs
    /// after the open settled is a fresh attempt, not a joiner.
    pub(crate) fn note_session_interest(&self, client: &ClientToken, session_id: &str) {
        let unsettled = self
            .inner
            .opening_sessions
            .borrow()
            .get(session_id)
            .is_some_and(|opening| !opening.latch.is_settled());
        if !unsettled {
            return;
        }
        self.inner
            .session_interest
            .borrow_mut()
            .insert(client.key(), session_id.to_string());
    }

    /// Clears the record once the request's processing ends.
    pub(crate) fn clear_session_interest(&self, client: &ClientToken) {
        self.inner
            .session_interest
            .borrow_mut()
            .remove(&client.key());
    }

    /// Route one service call to the client's attached Session, upstream's
    /// `executeServiceCall`; the call starts inside the client's chain and
    /// the returned future awaits the tracked call itself.
    pub(crate) fn execute_service_call(
        &self,
        client: &ClientToken,
        target: pi_protocol::RpcTarget,
        call: ServiceCall,
        publish: ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<TrackedResult> {
        let inner = Rc::clone(&self.inner);
        let token = client.clone();
        boxed(async move {
            let tracked = inner
                .run_for_client(&token, {
                    let inner = Rc::clone(&inner);
                    let token = token.clone();
                    boxed(std::future::ready(
                        inner.start_service_call(&token, &target, call, publish, context),
                    ))
                })
                .await?;
            tracked.wait().await
        })
    }

    /// Attach the presentation to one hosted Session, upstream's
    /// `attachClient`.
    pub(crate) fn attach_client(
        &self,
        client: &ClientToken,
        session_id: String,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        let inner = Rc::clone(&self.inner);
        if inner.is_closing() {
            return ready(Err(Failure::Server(ServerError::server_draining())));
        }
        let token = client.clone();
        boxed(async move {
            inner
                .run_for_client(
                    &token,
                    boxed({
                        let inner = Rc::clone(&inner);
                        let token = token.clone();
                        async move { inner.attach_client_now(&token, &session_id, &context).await }
                    }),
                )
                .await
        })
    }

    /// Detach the presentation from its Session, upstream's `detachClient`.
    pub(crate) fn detach_client(
        &self,
        client: &ClientToken,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        let inner = Rc::clone(&self.inner);
        let token = client.clone();
        let operation = release_if_held(Rc::clone(&inner), token.clone(), context, true);
        boxed(async move { inner.run_for_client(&token, operation).await })
    }

    /// Release routed attachments and the handle before the application
    /// deletes durable metadata, upstream's `removeSession`.
    pub(crate) fn remove_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        let inner = Rc::clone(&self.inner);
        if inner.is_closing() {
            return ready(Err(Failure::Server(ServerError::server_draining())));
        }
        let session_id = session_id.to_string();
        boxed(async move {
            let hosted = inner.hosted_sessions.borrow().get(&session_id).cloned();
            let Some(hosted) = hosted else {
                return Ok(());
            };
            let mut errors: Vec<Failure> = Vec::new();
            let attachments: Vec<Rc<ClientAttachment>> = hosted
                .attachments
                .borrow()
                .values()
                .filter_map(Weak::upgrade)
                .collect();
            let mut handles = Vec::with_capacity(attachments.len());
            for attachment in attachments {
                let inner = Rc::clone(&inner);
                let context = context.clone();
                handles.push(tokio::task::spawn_local(async move {
                    inner.release_attachment(&attachment, &context, true).await
                }));
            }
            for handle in handles {
                match handle.await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => errors.push(error),
                    Err(join) => errors.push(Failure::message(join.to_string())),
                }
            }
            if let Err(error) = hosted.handle.close(context.clone()).await {
                errors.push(error);
            }
            let mut hosted_sessions = inner.hosted_sessions.borrow_mut();
            if hosted_sessions
                .get(&session_id)
                .is_some_and(|current| Rc::ptr_eq(current, &hosted))
            {
                hosted_sessions.remove(&session_id);
            }
            drop(hosted_sessions);
            match errors.len() {
                0 => Ok(()),
                1 => Err(errors.swap_remove(0)),
                _ => Err(Failure::Aggregate {
                    message: format!("Failed to close Session {session_id}"),
                    errors,
                }),
            }
        })
    }

    /// Release the client's attachment without publishing the detach,
    /// upstream's `disconnect`.
    pub(crate) fn disconnect(
        &self,
        client: &ClientToken,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        let inner = Rc::clone(&self.inner);
        let token = client.clone();
        let operation = release_if_held(Rc::clone(&inner), token.clone(), context, false);
        boxed(async move {
            inner.disconnected_clients.borrow_mut().insert(token.key());
            let result = inner.run_for_client(&token, operation).await;
            inner.disconnected_clients.borrow_mut().remove(&token.key());
            result
        })
    }

    /// Close every routed attachment and handle, upstream's `close`.
    pub(crate) fn close(&self, context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        let inner = Rc::clone(&self.inner);
        if let Some(existing) = inner.close_latch.borrow().clone() {
            return boxed(async move { existing.wait().await });
        }
        let latch = Rc::new(Latch::new());
        *inner.close_latch.borrow_mut() = Some(Rc::clone(&latch));
        let spawn_latch = Rc::clone(&latch);
        tokio::task::spawn_local(async move {
            let result = inner.close_internal(&context).await;
            spawn_latch.settle(result);
        });
        boxed(async move { latch.wait().await })
    }
}

impl<H: ServerHost + 'static> RouterInner<H> {
    fn is_closing(&self) -> bool {
        self.core.upgrade().is_some_and(|core| core.closing.get())
    }

    fn report_error(&self, error: &Failure) {
        if let Some(core) = self.core.upgrade() {
            core.report_error(error);
        }
    }

    fn server_draining() -> Failure {
        Failure::Server(ServerError::server_draining())
    }

    async fn close_internal(self: &Rc<Self>, context: &Context) -> Result<(), Failure> {
        // Both snapshots happen before any wait, upstream's synchronous
        // `[...this.clientOperations.values()]` /
        // `[...this.openingSessions.values()]` at close start.
        let pending: Vec<Rc<Latch<()>>> = self
            .client_drivers
            .borrow()
            .values()
            .flat_map(|driver| driver.pending.borrow().clone())
            .collect();
        let openings: Vec<Rc<Opening>> = self.opening_sessions.borrow().values().cloned().collect();
        // The queued operations never fail, upstream's always-fulfilled
        // chain tails; the close only waits for them to settle.
        for latch in pending {
            latch.wait().await;
        }
        let mut close_errors: Vec<Failure> = Vec::new();
        for opening in openings {
            if let Err(error) = opening.latch.wait().await {
                self.report_error(&error);
                if error.is_session_cleanup() {
                    close_errors.push(error);
                }
            }
        }
        let pairs: Vec<Rc<ClientAttachment>> = self
            .hosted_sessions
            .borrow()
            .values()
            .flat_map(|session| {
                session
                    .attachments
                    .borrow()
                    .values()
                    .filter_map(Weak::upgrade)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut handles = Vec::with_capacity(pairs.len());
        for attachment in pairs {
            let inner = Rc::clone(self);
            let context = context.clone();
            handles.push(tokio::task::spawn_local(async move {
                inner.release_attachment(&attachment, &context, true).await
            }));
        }
        for handle in handles {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => close_errors.push(error),
                Err(join) => close_errors.push(Failure::message(join.to_string())),
            }
        }
        let hosted: Vec<Rc<HostedSession>> =
            self.hosted_sessions.borrow().values().cloned().collect();
        let mut close_handles = Vec::with_capacity(hosted.len());
        for session in hosted {
            let context = context.clone();
            close_handles.push(tokio::task::spawn_local(async move {
                (session.clone(), session.handle.close(context).await)
            }));
        }
        for handle in close_handles {
            match handle.await {
                Ok((session, Ok(()))) => {
                    let mut hosted_sessions = self.hosted_sessions.borrow_mut();
                    if hosted_sessions
                        .get(&session.id)
                        .is_some_and(|current| Rc::ptr_eq(current, &session))
                    {
                        hosted_sessions.remove(&session.id);
                    }
                }
                Ok((_, Err(error))) => {
                    self.report_error(&error);
                    close_errors.push(error);
                }
                Err(join) => close_errors.push(Failure::message(join.to_string())),
            }
        }
        self.attachments_by_client.borrow_mut().clear();
        self.client_drivers.borrow_mut().clear();
        self.opening_sessions.borrow_mut().clear();
        if close_errors.is_empty() {
            return Ok(());
        }
        Err(Failure::Aggregate {
            message: "Failed to close routed Sessions".to_string(),
            errors: close_errors,
        })
    }

    /// Chains one operation behind the client's settled operations, upstream's
    /// `runForClient`: FIFO per client, a failed operation never poisons the
    /// next.
    fn run_for_client<T: Clone + 'static>(
        self: &Rc<Self>,
        client: &ClientToken,
        operation: LocalBoxFuture<Result<T, Failure>>,
    ) -> LocalBoxFuture<Result<T, Failure>> {
        let key = client.key();
        let existing = self.client_drivers.borrow().get(&key).cloned();
        let driver = existing.unwrap_or_else(|| self.create_client_driver(key));
        let settle: Rc<Latch<()>> = Rc::new(Latch::new());
        driver.pending.borrow_mut().push(Rc::clone(&settle));
        let result: Rc<RefCell<Option<Result<T, Failure>>>> = Rc::new(RefCell::new(None));
        let _ = driver.queue.send(RouterJob {
            settle: Rc::clone(&settle),
            run: boxed({
                let result = Rc::clone(&result);
                async move {
                    let settled = operation.await;
                    *result.borrow_mut() = Some(settled);
                }
            }),
        });
        let result_for_wait = Rc::clone(&result);
        boxed(async move {
            settle.wait().await;
            result_for_wait
                .borrow_mut()
                .take()
                .unwrap_or_else(|| Err(Failure::message("client operation chain was discarded")))
        })
    }

    /// Creates one client's driver task, the `runForClient` chain restated:
    /// jobs run FIFO, a settled job never poisons the next, and the map
    /// entry disappears when the queue drains.
    fn create_client_driver(&self, key: usize) -> Rc<ClientDriver> {
        let (queue, mut receiver) = tokio::sync::mpsc::unbounded_channel::<RouterJob>();
        let pending: Rc<RefCell<Vec<Rc<Latch<()>>>>> = Rc::new(RefCell::new(Vec::new()));
        let core = self.core.clone();
        let loop_pending = Rc::clone(&pending);
        tokio::task::spawn_local(async move {
            while let Some(job) = receiver.recv().await {
                // The per-job task isolates a panicking operation so the
                // chain keeps running; the panicking operation's own caller
                // never settles.
                let completed = tokio::task::spawn_local(job.run).await.is_ok();
                if completed {
                    job.settle.settle(());
                    loop_pending
                        .borrow_mut()
                        .retain(|latch| !Rc::ptr_eq(latch, &job.settle));
                }
                if let Some(core) = core.upgrade()
                    && loop_pending.borrow().is_empty()
                {
                    // The driver's sender lives in the map entry; dropping
                    // the entry ends this recv loop once the queue drains.
                    core.sessions.inner.client_drivers.borrow_mut().remove(&key);
                }
            }
        });
        let driver = Rc::new(ClientDriver {
            queue,
            pending: Rc::clone(&pending),
        });
        self.client_drivers
            .borrow_mut()
            .insert(key, Rc::clone(&driver));
        driver
    }

    async fn attach_client_now(
        self: &Rc<Self>,
        client: &ClientToken,
        session_id: &str,
        context: &Context,
    ) -> Result<(), Failure> {
        if self.is_closing() || self.disconnected_clients.borrow().contains(&client.key()) {
            return Err(Self::server_draining());
        }
        let current = self
            .attachments_by_client
            .borrow()
            .get(&client.key())
            .cloned();
        if current
            .as_ref()
            .is_some_and(|current| current.session.id == session_id)
        {
            return Ok(());
        }
        let hosted = self.acquire(client, session_id, context).await?;
        if self.is_closing() || self.disconnected_clients.borrow().contains(&client.key()) {
            return Err(Self::server_draining());
        }
        if let Some(current) = current {
            self.release_attachment_owned(&current, context, false)
                .await?;
        }
        let attachment = Rc::new(ClientAttachment {
            id: uuid::Uuid::new_v4().to_string(),
            client: client.clone(),
            session: Rc::clone(&hosted),
            operations: RefCell::new(Vec::new()),
            acquiring: RefCell::new(None),
            lease: RefCell::new(None),
            releasing: RefCell::new(None),
        });
        hosted
            .attachments
            .borrow_mut()
            .insert(client.key(), Rc::downgrade(&attachment));
        let acquiring = Rc::new(Latch::new());
        *attachment.acquiring.borrow_mut() = Some(Rc::clone(&acquiring));
        let acquiring_future = hosted.handle.attach_client(context.clone());
        {
            let latch = Rc::clone(&acquiring);
            tokio::task::spawn_local(async move {
                latch.settle(acquiring_future.await);
            });
        }
        let lease = match acquiring.wait().await {
            Ok(lease) => lease,
            Err(error) => {
                hosted.attachments.borrow_mut().remove(&client.key());
                return Err(error);
            }
        };
        *attachment.lease.borrow_mut() = Some(lease);
        let still_hosted = self
            .hosted_sessions
            .borrow()
            .get(&hosted.id)
            .is_some_and(|current| Rc::ptr_eq(current, &hosted));
        let still_attached = hosted
            .attachments
            .borrow()
            .get(&client.key())
            .and_then(Weak::upgrade)
            .is_some_and(|current| Rc::ptr_eq(&current, &attachment));
        if !still_hosted
            || !still_attached
            || self.disconnected_clients.borrow().contains(&client.key())
            || self.is_closing()
        {
            self.release_attachment_owned(&attachment, context, true)
                .await?;
            return Err(Self::server_draining());
        }
        self.attachments_by_client
            .borrow_mut()
            .insert(client.key(), Rc::clone(&attachment));
        let target = pi_protocol::SessionTarget {
            server_id: self.server_id.clone(),
            session_id: session_id.to_string(),
            attachment_id: attachment.id.clone(),
        };
        if let Some(core) = self.core.upgrade() {
            core.publish_attachment(client, Some(target), context).await;
        }
        Ok(())
    }

    fn start_service_call(
        self: &Rc<Self>,
        client: &ClientToken,
        target: &pi_protocol::RpcTarget,
        call: ServiceCall,
        publish: ServicePublisher,
        context: Context,
    ) -> Result<Rc<Latch<TrackedResult>>, Failure> {
        let attachment = self.require_attachment(client, target)?;
        let lease = attachment
            .lease
            .borrow()
            .clone()
            .ok_or_else(|| Failure::message("Session attachment has no acquired lease"))?;
        let invocation = lease.invoke_service(call, publish, context);
        let tracked = Rc::new(Latch::new());
        attachment.operations.borrow_mut().push(Rc::clone(&tracked));
        let watcher = Rc::clone(&attachment);
        let settled = Rc::clone(&tracked);
        tokio::task::spawn_local(async move {
            let result = invocation.await;
            settled.settle(result);
            watcher
                .operations
                .borrow_mut()
                .retain(|operation| !Rc::ptr_eq(operation, &settled));
        });
        Ok(tracked)
    }

    fn require_attachment(
        &self,
        client: &ClientToken,
        target: &pi_protocol::RpcTarget,
    ) -> Result<Rc<ClientAttachment>, Failure> {
        if self.is_closing() || self.disconnected_clients.borrow().contains(&client.key()) {
            return Err(Self::server_draining());
        }
        let pi_protocol::RpcTarget::Session(session) = target else {
            return Err(Failure::Server(ServerError::session_not_attached()));
        };
        let Some(attachment) = self
            .attachments_by_client
            .borrow()
            .get(&client.key())
            .cloned()
        else {
            return Err(Failure::Server(ServerError::session_not_attached()));
        };
        if attachment.session.id != session.session_id || attachment.id != session.attachment_id {
            return Err(Failure::Server(ServerError::session_not_attached()));
        }
        Ok(attachment)
    }

    /// Releases one attachment exactly once, upstream's `releaseAttachment`.
    fn release_attachment(
        self: &Rc<Self>,
        attachment: &Rc<ClientAttachment>,
        context: &Context,
        publish: bool,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        self.release_attachment_owned(attachment, context, publish)
    }

    /// The same release from an owned reference, the shape the spawned body
    /// and the awaiting callers share.
    fn release_attachment_owned(
        self: &Rc<Self>,
        attachment: &Rc<ClientAttachment>,
        context: &Context,
        publish: bool,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        if let Some(existing) = attachment.releasing.borrow().clone() {
            return boxed(async move { existing.wait().await });
        }
        let latch = Rc::new(Latch::new());
        *attachment.releasing.borrow_mut() = Some(Rc::clone(&latch));
        let inner = Rc::clone(self);
        let attachment = Rc::clone(attachment);
        let context = context.clone();
        let spawn_latch = Rc::clone(&latch);
        tokio::task::spawn_local(async move {
            let result = inner
                .release_attachment_body(&attachment, &context, publish)
                .await;
            spawn_latch.settle(result);
        });
        boxed(async move { latch.wait().await })
    }

    async fn release_attachment_body(
        &self,
        attachment: &Rc<ClientAttachment>,
        context: &Context,
        publish: bool,
    ) -> Result<(), Failure> {
        let result: Result<(), Failure> = async {
            let mut errors: Vec<Failure> = Vec::new();
            let operations: Vec<Rc<Latch<TrackedResult>>> = attachment.operations.borrow().clone();
            for operation in operations {
                let _ = operation.wait().await;
            }
            let settled_lease = attachment.lease.borrow_mut().take();
            let lease = if let Some(lease) = settled_lease {
                Some(lease)
            } else {
                let acquiring = attachment.acquiring.borrow().clone();
                match acquiring {
                    Some(acquiring) => match acquiring.wait().await {
                        Ok(lease) => Some(lease),
                        Err(error) => {
                            errors.push(error);
                            None
                        }
                    },
                    None => None,
                }
            };
            if let Some(lease) = lease
                && let Err(error) = lease.release(context.clone()).await
            {
                errors.push(error);
            }
            match errors.len() {
                0 => Ok(()),
                1 => Err(errors.swap_remove(0)),
                _ => Err(Failure::Aggregate {
                    message: "Failed to release Session attachment".to_string(),
                    errors,
                }),
            }
        }
        .await;
        self.clear_attachment(attachment, context, publish).await;
        result
    }

    async fn clear_attachment(
        &self,
        attachment: &Rc<ClientAttachment>,
        context: &Context,
        publish: bool,
    ) {
        attachment
            .session
            .attachments
            .borrow_mut()
            .remove(&attachment.client.key());
        let owning = self
            .attachments_by_client
            .borrow()
            .get(&attachment.client.key())
            .is_some_and(|current| Rc::ptr_eq(current, attachment));
        if owning {
            self.attachments_by_client
                .borrow_mut()
                .remove(&attachment.client.key());
            if publish && let Some(core) = self.core.upgrade() {
                core.publish_attachment(&attachment.client, None, context)
                    .await;
            }
        }
    }

    /// Opens or joins the in-flight open of one hosted Session, upstream's
    /// `acquire`.
    ///
    /// A request dispatched while its session's open was in flight joins
    /// that open even if it settled in between — upstream's synchronous
    /// acquire-check reads `openingSessions` before the settled open's
    /// `finally` retires the entry; the port records that fact at dispatch
    /// and honors it here. Anything else finding a settled open retires the
    /// entry and opens fresh, the later-attach retry.
    async fn acquire(
        self: &Rc<Self>,
        client: &ClientToken,
        session_id: &str,
        context: &Context,
    ) -> OpenResult {
        if let Some(existing) = self.hosted_sessions.borrow().get(session_id).cloned() {
            return Ok(existing);
        }
        let opening = self.opening_sessions.borrow().get(session_id).cloned();
        if let Some(opening) = opening {
            let interested = self
                .session_interest
                .borrow()
                .get(&client.key())
                .is_some_and(|noted| noted == session_id);
            if !opening.latch.is_settled() || (interested && opening.failed.get()) {
                return opening.latch.wait().await;
            }
            let mut opening_sessions = self.opening_sessions.borrow_mut();
            if opening_sessions
                .get(session_id)
                .is_some_and(|current| Rc::ptr_eq(current, &opening))
            {
                opening_sessions.remove(session_id);
            }
        }
        let opening = Rc::new(Opening {
            latch: Latch::new(),
            failed: Cell::new(false),
        });
        self.opening_sessions
            .borrow_mut()
            .insert(session_id.to_string(), Rc::clone(&opening));
        let inner = Rc::clone(self);
        let session_id = session_id.to_string();
        let context = context.clone();
        let settle = opening.latch.clone();
        let failed = Rc::clone(&opening);
        tokio::task::spawn_local(async move {
            let result = inner.open(&session_id, &context).await;
            if result.is_err() {
                failed.failed.set(true);
            }
            // One scheduling turn before the settle: the join window
            // upstream's synchronous acquire-check gives its concurrent
            // attachers, so a request dispatched mid-open notes its
            // interest against an unsettled open.
            tokio::task::yield_now().await;
            settle.settle(result);
        });
        opening.latch.wait().await
    }

    async fn open(self: &Rc<Self>, session_id: &str, context: &Context) -> OpenResult {
        let metadata = self
            .host
            .resolve_session(session_id, context.clone())
            .await?;
        let handle = self
            .host
            .open_session(Rc::clone(&metadata), context.clone())
            .await?;
        if self.is_closing() {
            return match handle.close(context.clone()).await {
                Ok(()) => Err(Self::server_draining()),
                Err(error) => {
                    self.report_error(&error);
                    Err(Failure::Cleanup {
                        message: "Failed to close routed Session acquired while draining"
                            .to_string(),
                        errors: vec![Self::server_draining(), error],
                    })
                }
            };
        }
        let hosted = Rc::new(HostedSession {
            id: metadata.session_id().to_string(),
            handle: Rc::clone(&handle),
            attachments: RefCell::new(HashMap::new()),
        });
        self.hosted_sessions
            .borrow_mut()
            .insert(hosted.id.clone(), Rc::clone(&hosted));
        if let Some(terminated) = handle.terminated() {
            let inner = Rc::clone(self);
            let watched = Rc::clone(&hosted);
            tokio::task::spawn_local(async move {
                let error = terminated.await;
                inner.invalidate(&watched, error);
            });
        }
        Ok(hosted)
    }

    /// Drops a terminated Session's routing state and releases its
    /// attachments, upstream's `invalidate`.
    fn invalidate(self: &Rc<Self>, hosted: &Rc<HostedSession>, error: Option<Failure>) {
        let owning = self
            .hosted_sessions
            .borrow()
            .get(&hosted.id)
            .is_some_and(|current| Rc::ptr_eq(current, hosted));
        if !owning {
            return;
        }
        self.hosted_sessions.borrow_mut().remove(&hosted.id);
        let attachments: Vec<Rc<ClientAttachment>> = hosted
            .attachments
            .borrow()
            .values()
            .filter_map(Weak::upgrade)
            .collect();
        for attachment in attachments {
            let inner = Rc::clone(self);
            tokio::task::spawn_local(async move {
                if let Err(release) = inner
                    .release_attachment(&attachment, &background_context(), true)
                    .await
                {
                    inner.report_error(&release);
                }
            });
        }
        if let Some(error) = error {
            self.report_error(&error);
        }
    }
}

/// The chain body `detach_client` and `disconnect` share: release the
/// client's attachment when it holds one, upstream's two call sites of
/// `releaseAttachment` behind `runForClient`.
#[allow(
    clippy::needless_pass_by_value,
    reason = "the router handle, token, and context join the spawned chain body"
)]
fn release_if_held<H: ServerHost + 'static>(
    inner: Rc<RouterInner<H>>,
    token: ClientToken,
    context: Context,
    publish: bool,
) -> LocalBoxFuture<Result<(), Failure>> {
    boxed({
        async move {
            let attachment = inner
                .attachments_by_client
                .borrow()
                .get(&token.key())
                .cloned();
            if let Some(attachment) = attachment {
                inner
                    .release_attachment(&attachment, &context, publish)
                    .await?;
            }
            Ok(())
        }
    })
}
