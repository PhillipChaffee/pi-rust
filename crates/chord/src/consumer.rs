//! Remote service consumers, ported from upstream
//! `src/services/consumer.ts` and the binding side of `handle.ts`.
//!
//! A [`RemoteFacade`] is one service's consumer-side object: lazy member
//! slots backed by the transport, state replicas hydrated from snapshots,
//! and kind pinning that rejects a member used as two different kinds.
//! [`RemoteServiceBinding`] keeps facades stable across provider
//! replacement and rebind, replays readiness through revision guards, and
//! routes keyed observation through the instance directory. Stored
//! start-up and transition futures settle once and leave a settled copy in
//! their cell, the single-owner restatement of reusable promises.

use std::cell::{Cell, RefCell};
use std::panic::AssertUnwindSafe;
use std::rc::Rc;

use crate::context::{Context, background_context};
use crate::errors::{ChordError, RemoteServiceError, RemoteServiceErrorCode, collect_errors};
use crate::future::{LocalBoxFuture, boxed, join_all, ready_with};
use crate::handle::{
    AssertAccess, ErrorReporter, RemoteMethod, ServiceSlot, ServiceTarget, allow_access,
};
use crate::services::instances::InstanceDirectory;
use crate::services::state::{
    ReplicatedStateReplica, ValueListener, panic_message, service_delivery_context,
};
use crate::types::{
    JsonValue, KeyedServiceHandler, RemoteServiceTransport, Service, ServiceCall,
    ServiceInstanceAddress, ServiceInstanceSnapshot, ServiceMemberKind, ServiceMode,
    ServiceProviderListener, ServiceProviderUpdate, ServiceSubscription, Unsubscribe,
};

/// The transport handle a binding consumes, held by shared reference the
/// single-threaded runtime keeps alive.
pub type SharedTransport = Rc<dyn RemoteServiceTransport>;

/// A stored start/transition future, the reusable-promise cell.
pub(crate) type StoredStart = Rc<RefCell<Option<LocalBoxFuture<Result<(), ChordError>>>>>;

fn remote_error(code: RemoteServiceErrorCode, message: impl Into<String>) -> ChordError {
    ChordError::Remote(RemoteServiceError::new(code, message))
}

/// The options a binding is built with, upstream's
/// `RemoteServiceBindingOptions`.
pub struct RemoteServiceBindingOptions {
    /// The services this binding may touch.
    pub services: Vec<Service>,
    /// The transport every invocation and subscription rides.
    pub transport: SharedTransport,
    /// Whether subscriptions start immediately; `false` defers them until
    /// [`rebind`](RemoteServiceBinding::rebind).
    pub bound: bool,
    /// The reporter for failures outside a caller's control.
    pub on_error: ErrorReporter,
    /// The gate every handle access passes.
    pub assert_access: Option<AssertAccess>,
}

impl std::fmt::Debug for RemoteServiceBindingOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteServiceBindingOptions")
            .field("services", &self.services)
            .field("bound", &self.bound)
            .finish_non_exhaustive()
    }
}

/// Builds the binding from its options, upstream's
/// `createRemoteServiceBinding`.
///
/// # Errors
/// [`ChordError`] when the service list has duplicate IDs.
pub fn create_remote_service_binding(
    options: RemoteServiceBindingOptions,
) -> Result<RemoteServiceBinding, ChordError> {
    let mut allowlist = std::collections::HashSet::new();
    for service in &options.services {
        if !allowlist.insert(service.id.clone()) {
            return Err(ChordError::Message(
                "Remote service binding has duplicate service IDs".to_string(),
            ));
        }
    }
    Ok(RemoteServiceBinding(Rc::new(BindingCore {
        transport: options.transport,
        allowlist,
        report_error: options.on_error,
        modes: RefCell::new(Vec::new()),
        assert_access: options.assert_access.unwrap_or_else(allow_access),
        singletons: RefCell::new(Vec::new()),
        keyed: RefCell::new(Vec::new()),
        // A shared cell: the facades' is-active closures read this flag
        // live, so a rebind flips every handle at once, upstream's live
        // `binding.bound` read.
        bound: Rc::new(Cell::new(options.bound)),
        readiness_revision: Cell::new(0),
        binding_transition: Rc::new(RefCell::new(None)),
        disposed: Rc::new(Cell::new(false)),
    })))
}

/// Takes a stored start/transition future, awaits it once, and leaves a
/// settled copy in its place: every later await sees the same result, the
/// reusable-promise contract.
/// Drives a stored start once, the eager start upstream's microtask
/// scheduling performs: an in-process transport's subscription settles in
/// one poll, and a future needing more turns stays stored for
/// [`RemoteServiceBinding::ready`].
fn drive_start_now(cell: &StoredStart) {
    let taken = cell.borrow_mut().take();
    if let Some(future) = taken {
        match crate::future::drive_once(future) {
            Ok(result) => *cell.borrow_mut() = Some(boxed(ready_with(result))),
            Err(still_pending) => *cell.borrow_mut() = Some(still_pending),
        }
    }
}

fn settle_stored(cell: &StoredStart) -> LocalBoxFuture<Result<(), ChordError>> {
    let cell = cell.clone();
    let existing = cell.borrow_mut().take();
    existing.map_or_else(
        || boxed(ready_with(Ok(()))),
        |future| {
            boxed(async move {
                let result = future.await;
                *cell.borrow_mut() = Some(boxed(ready_with(result.clone())));
                result
            })
        },
    )
}

/// The per-member consumer slot, upstream's `MemberSlot`: a replicated
/// replica for state members, a transport invocation for methods, and a
/// kind pin that rejects a member used as two different kinds.
pub struct MemberSlot {
    service_id: String,
    member: String,
    invoke: RemoteMethod,
    state: ReplicatedStateReplica,
    is_active: Rc<dyn Fn() -> bool>,
    assert_access: AssertAccess,
    kind: Cell<Option<ServiceMemberKind>>,
    expected_kind: Cell<Option<ServiceMemberKind>>,
}

impl std::fmt::Debug for MemberSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemberSlot")
            .field("service_id", &self.service_id)
            .field("member", &self.member)
            .finish_non_exhaustive()
    }
}

impl MemberSlot {
    fn new(
        service_id: &str,
        member: &str,
        invoke: RemoteMethod,
        is_active: Rc<dyn Fn() -> bool>,
        assert_access: AssertAccess,
        report_error: ErrorReporter,
    ) -> Self {
        Self {
            service_id: service_id.to_string(),
            member: member.to_string(),
            invoke,
            state: ReplicatedStateReplica::new(report_error),
            is_active,
            assert_access,
            kind: Cell::new(None),
            expected_kind: Cell::new(None),
        }
    }

    fn set_description(&self, kind: ServiceMemberKind) -> Result<(), ChordError> {
        if let Some(existing) = self.kind.get()
            && existing != kind
        {
            return Err(ChordError::Message(format!(
                "Remote service member {}.{} changed kind",
                self.service_id, self.member
            )));
        }
        self.kind.set(Some(kind));
        if let Some(expected) = self.expected_kind.get()
            && expected != kind
        {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceMemberMismatch,
                format!(
                    "Remote service member {}.{} is {}, not {}",
                    self.service_id,
                    self.member,
                    kind_str(kind),
                    kind_str(expected)
                ),
            ));
        }
        Ok(())
    }

    fn expect(&self, kind: ServiceMemberKind) -> Result<(), ChordError> {
        if let Some(expected) = self.expected_kind.get()
            && expected != kind
        {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceMemberMismatch,
                format!(
                    "Remote service member {}.{} was used as two different kinds",
                    self.service_id, self.member
                ),
            ));
        }
        self.expected_kind.set(Some(kind));
        if let Some(actual) = self.kind.get()
            && actual != kind
        {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceMemberMismatch,
                format!(
                    "Remote service member {}.{} is {}, not {}",
                    self.service_id,
                    self.member,
                    kind_str(actual),
                    kind_str(kind)
                ),
            ));
        }
        Ok(())
    }

    fn hydrate(
        &self,
        sequence: u64,
        ops: &[crate::delta::Op],
        context: &Context,
    ) -> Result<(), ChordError> {
        self.set_description(ServiceMemberKind::State)?;
        self.state.hydrate(sequence, ops, context)
    }

    fn update(
        &self,
        sequence: u64,
        ops: &[crate::delta::Op],
        context: &Context,
    ) -> Result<(), ChordError> {
        self.set_description(ServiceMemberKind::State)?;
        self.state.update(sequence, ops, context)
    }

    fn clear(&self) {
        self.state.clear();
    }

    /// The current state value, upstream's `.value` getter with its access
    /// and kind checks.
    ///
    /// # Errors
    /// [`ChordError`] when the access gate rejects or the member is not
    /// state.
    pub fn state_value(&self) -> Result<Option<JsonValue>, ChordError> {
        (self.assert_access.clone())()?;
        self.expect(ServiceMemberKind::State)?;
        Ok(self.state.value())
    }

    /// Subscribes to state revisions, upstream's `.subscribe` with its
    /// access and kind checks.
    ///
    /// # Errors
    /// [`ChordError`] when the access gate rejects or the member is not
    /// state.
    pub fn state_subscribe(&self, listener: ValueListener) -> Result<Unsubscribe, ChordError> {
        (self.assert_access.clone())()?;
        self.expect(ServiceMemberKind::State)?;
        Ok(self.state.subscribe(listener))
    }

    /// Invokes the method member, upstream's proxy apply: the access gate,
    /// the kind pin, the active check, then the transport.
    pub(crate) fn call(
        self: &Rc<Self>,
        args: Vec<JsonValue>,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>> {
        let slot = self.clone();
        boxed(async move {
            (slot.assert_access.clone())()?;
            slot.expect(ServiceMemberKind::Method)?;
            if !(slot.is_active)() {
                return Err(remote_error(
                    RemoteServiceErrorCode::ServiceStaleInstance,
                    format!("Remote service {} binding is closed", slot.service_id),
                ));
            }
            (slot.invoke)(args, context).await
        })
    }
}

const fn kind_str(kind: ServiceMemberKind) -> &'static str {
    match kind {
        ServiceMemberKind::Method => "method",
        ServiceMemberKind::State => "state",
    }
}

struct FacadeCore {
    service_id: String,
    address: Option<ServiceInstanceAddress>,
    transport: Rc<dyn RemoteServiceTransport>,
    slots: RefCell<Vec<(String, Rc<MemberSlot>)>>,
    descriptions: RefCell<Vec<(String, ServiceMemberKind)>>,
    is_active: Rc<dyn Fn() -> bool>,
    assert_access: AssertAccess,
    report_error: ErrorReporter,
}

/// The consumer facade for one service: lazily created member slots over
/// the transport, stable across provider replacement because the facade
/// object never changes.
#[derive(Clone)]
pub struct RemoteFacade(Rc<FacadeCore>);

impl std::fmt::Debug for RemoteFacade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteFacade")
            .field("service_id", &self.0.service_id)
            .field("address", &self.0.address)
            .finish()
    }
}

impl RemoteFacade {
    fn new(
        service_id: &str,
        address: Option<ServiceInstanceAddress>,
        transport: Rc<dyn RemoteServiceTransport>,
        is_active: Rc<dyn Fn() -> bool>,
        assert_access: AssertAccess,
        report_error: ErrorReporter,
    ) -> Self {
        Self(Rc::new(FacadeCore {
            service_id: service_id.to_string(),
            address,
            transport,
            slots: RefCell::new(Vec::new()),
            descriptions: RefCell::new(Vec::new()),
            is_active,
            assert_access,
            report_error,
        }))
    }

    /// The facade identity upstream's proxy identity restates: two facades
    /// are the same when they share one core.
    #[must_use]
    pub fn same_facade(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    fn find_slot(&self, member: &str) -> Option<Rc<MemberSlot>> {
        self.0
            .slots
            .borrow()
            .iter()
            .find(|(name, _)| name == member)
            .map(|(_, slot)| slot.clone())
    }

    fn slot(self: &Rc<Self>, member: &str) -> Rc<MemberSlot> {
        if let Some(slot) = self.find_slot(member) {
            return slot;
        }
        let transport = self.0.transport.clone();
        let service_id = self.0.service_id.clone();
        let address = self.0.address.clone();
        let member_name = member.to_string();
        let invoke: RemoteMethod = Rc::new(move |args: Vec<JsonValue>, context: Context| {
            transport.invoke(
                ServiceCall {
                    service_id: service_id.clone(),
                    instance: address.clone(),
                    member: member_name.clone(),
                    args,
                },
                context,
            )
        });
        let is_active = self.0.is_active.clone();
        let assert_access = self.0.assert_access.clone();
        let report_error = self.0.report_error.clone();
        let slot = Rc::new(MemberSlot::new(
            &self.0.service_id,
            member,
            invoke,
            is_active,
            assert_access,
            report_error,
        ));
        self.0
            .slots
            .borrow_mut()
            .push((member.to_string(), slot.clone()));
        if let Some((_, kind)) = self
            .0
            .descriptions
            .borrow()
            .iter()
            .find(|(name, _)| name == member)
        {
            let _ = slot.set_description(*kind);
        }
        slot
    }

    /// The member slot for `member`, applying any known description kind.
    pub(crate) fn member_slot(self: &Rc<Self>, member: &str) -> Rc<MemberSlot> {
        self.slot(member)
    }

    /// Invokes one method member through its slot.
    pub(crate) fn call_member(
        self: &Rc<Self>,
        member: &str,
        args: Vec<JsonValue>,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>> {
        let slot = self.slot(member);
        slot.call(args, context)
    }

    fn install(
        self: &Rc<Self>,
        snapshot: &ServiceInstanceSnapshot,
        context: &Context,
    ) -> Result<(), ChordError> {
        if !same_address(snapshot.instance.as_ref(), self.0.address.as_ref()) {
            return Err(ChordError::Message(
                "Remote service snapshot has the wrong address".to_string(),
            ));
        }
        validate_members(&snapshot.members)?;
        for (name, _) in self.0.slots.borrow().iter() {
            if !snapshot.members.iter().any(|member| member.name() == name) {
                return Err(remote_error(
                    RemoteServiceErrorCode::ServiceMemberNotFound,
                    format!(
                        "Unknown remote service member {}.{}",
                        self.0.service_id, name
                    ),
                ));
            }
        }
        *self.0.descriptions.borrow_mut() = snapshot
            .members
            .iter()
            .map(|member| (member.name().to_string(), member.kind()))
            .collect();
        for member in &snapshot.members {
            match member {
                crate::types::ServiceMemberSnapshot::State {
                    name,
                    sequence,
                    ops,
                } => {
                    self.slot(name).hydrate(*sequence, ops, context)?;
                }
                crate::types::ServiceMemberSnapshot::Method { name } => {
                    if let Some(slot) = self.find_slot(name) {
                        slot.set_description(ServiceMemberKind::Method)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn update(
        self: &Rc<Self>,
        member: &str,
        sequence: u64,
        ops: &[crate::delta::Op],
        context: &Context,
    ) -> Result<(), ChordError> {
        let described = self
            .0
            .descriptions
            .borrow()
            .iter()
            .find(|(name, _)| name == member)
            .map(|(_, kind)| *kind);
        if described != Some(ServiceMemberKind::State) {
            return Err(ChordError::Message(format!(
                "Remote service update targets non-state member {}.{}",
                self.0.service_id, member
            )));
        }
        self.slot(member).update(sequence, ops, context)
    }

    fn clear(self: &Rc<Self>) {
        for (_, slot) in self.0.slots.borrow().iter() {
            slot.clear();
        }
    }
}

fn validate_members(members: &[crate::types::ServiceMemberSnapshot]) -> Result<(), ChordError> {
    for (index, member) in members.iter().enumerate() {
        if member.name().is_empty()
            || members[..index]
                .iter()
                .any(|other| other.name() == member.name())
        {
            return Err(ChordError::Message(
                "Remote service has invalid member descriptions".to_string(),
            ));
        }
    }
    Ok(())
}

fn same_address(
    left: Option<&ServiceInstanceAddress>,
    right: Option<&ServiceInstanceAddress>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => left.key == right.key && left.generation == right.generation,
        _ => false,
    }
}

struct SingletonBindingState {
    slot: ServiceSlot,
    facade: Rc<RemoteFacade>,
    view: RefCell<Option<crate::handle::ServiceView>>,
    subscription: RefCell<Option<ServiceSubscription>>,
    starting: StoredStart,
    slot_view_access: AssertAccess,
    active: Rc<Cell<bool>>,
    revision: Cell<u64>,
}

impl SingletonBindingState {
    fn view(&self) -> crate::handle::ServiceView {
        self.view
            .borrow_mut()
            .get_or_insert_with(|| self.slot.view(self.slot_view_access.clone()))
            .clone()
    }
}

struct KeyedBindingState {
    service_id: String,
    instances: InstanceDirectory,
    subscription: RefCell<Option<ServiceSubscription>>,
    starting: StoredStart,
    closed: Cell<bool>,
    bound: Cell<bool>,
    revision: Cell<u64>,
}

struct BindingCore {
    transport: Rc<dyn RemoteServiceTransport>,
    allowlist: std::collections::HashSet<String>,
    report_error: ErrorReporter,
    modes: RefCell<Vec<(String, ServiceMode)>>,
    assert_access: AssertAccess,
    singletons: RefCell<Vec<(String, Rc<SingletonBindingState>)>>,
    keyed: RefCell<Vec<(String, Rc<KeyedBindingState>)>>,
    bound: Rc<Cell<bool>>,
    readiness_revision: Cell<u64>,
    binding_transition: StoredStart,
    disposed: Rc<Cell<bool>>,
}

/// The consumer side of the remote boundary: acquires singleton facades,
/// observes keyed instances, and keeps every handle stable across provider
/// replacement and rebind.
#[derive(Clone)]
pub struct RemoteServiceBinding(Rc<BindingCore>);

impl std::fmt::Debug for RemoteServiceBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteServiceBinding")
            .field("bound", &self.0.bound.get())
            .finish()
    }
}

impl RemoteServiceBinding {
    /// Acquires one singleton service's facade. The facade and its slot are
    /// created once per service and stay stable across replacement and
    /// rebind.
    ///
    /// # Errors
    /// [`ChordError`] when the binding is disposed, the service is local,
    /// not allowlisted, or already used as keyed.
    pub fn use_service(&self, service: &Service) -> Result<crate::handle::ServiceView, ChordError> {
        Self::assert_remotable(service)?;
        self.assert_available(&service.id, ServiceMode::Singleton)?;
        if let Some((_, binding)) = self
            .0
            .singletons
            .borrow()
            .iter()
            .find(|(id, _)| id == &service.id)
        {
            return Ok(binding.view());
        }
        let core = self.0.clone();
        let active = Rc::new(Cell::new(true));
        let disposed = core.disposed.clone();
        let bound = core.bound.clone();
        let is_active: Rc<dyn Fn() -> bool> = {
            let active = active.clone();
            Rc::new(move || active.get() && !disposed.get() && bound.get())
        };
        let assert_handle: AssertAccess = {
            let core = core.clone();
            Rc::new(move || core.assert_handle_access())
        };
        let facade = Rc::new(RemoteFacade::new(
            &service.id,
            None,
            core.transport.clone(),
            is_active,
            assert_handle.clone(),
            core.report_error.clone(),
        ));
        let slot = ServiceSlot::new(&service.id);
        slot.bind(ServiceTarget::Facade(facade.clone()));
        let binding = Rc::new(SingletonBindingState {
            slot,
            facade,
            view: RefCell::new(None),
            subscription: RefCell::new(None),
            starting: Rc::new(RefCell::new(None)),
            slot_view_access: assert_handle.clone(),
            active,
            revision: Cell::new(0),
        });
        self.0
            .singletons
            .borrow_mut()
            .push((service.id.clone(), binding.clone()));
        self.0
            .readiness_revision
            .set(self.0.readiness_revision.get() + 1);
        if self.0.bound.get() {
            let revision = binding.revision.get();
            let starting =
                self.start_singleton_wrapped(service.id.clone(), binding.clone(), revision);
            *binding.starting.borrow_mut() = Some(starting);
            drive_start_now(&binding.starting);
        }
        Ok(binding.view())
    }

    /// Observes every live instance of one keyed service.
    ///
    /// # Errors
    /// [`ChordError`] when the binding is disposed, the service is local,
    /// not allowlisted, or already used as singleton.
    pub fn observe(
        &self,
        service: &Service,
        handler: crate::types::KeyedViewHandler,
    ) -> Result<Unsubscribe, ChordError> {
        Self::assert_remotable(service)?;
        self.assert_available(&service.id, ServiceMode::Keyed)?;
        let core = self.0.clone();
        let existing = self
            .0
            .keyed
            .borrow()
            .iter()
            .find(|(id, _)| id == &service.id)
            .map(|(_, binding)| binding.clone());
        let binding = existing.unwrap_or_else(|| {
            let binding = Rc::new(KeyedBindingState {
                service_id: service.id.clone(),
                instances: InstanceDirectory::new(
                    false,
                    Rc::new({
                        let report = core.report_error.clone();
                        move |error| report(error)
                    }),
                ),
                subscription: RefCell::new(None),
                starting: Rc::new(RefCell::new(None)),
                closed: Cell::new(false),
                bound: Cell::new(core.bound.get()),
                revision: Cell::new(0),
            });
            core.keyed
                .borrow_mut()
                .push((service.id.clone(), binding.clone()));
            core.readiness_revision
                .set(core.readiness_revision.get() + 1);
            binding
        });
        let stopped = Rc::new(Cell::new(false));
        let observe_handler: KeyedServiceHandler = {
            let service_id = Rc::new(service.id.clone());
            let access: AssertAccess = {
                let core = core.clone();
                Rc::new(move || core.assert_handle_access())
            };
            let stopped = stopped.clone();
            Box::new(move |target, context| {
                let slot = ServiceSlot::new(service_id.as_str());
                slot.bind(target);
                let assert: AssertAccess = {
                    let access = access.clone();
                    let stopped = stopped.clone();
                    let service_id = service_id.clone();
                    let context = context.clone();
                    Rc::new(move || {
                        (access.clone())()?;
                        if stopped.get()
                            || context
                                .abort_signal()
                                .is_some_and(|signal| signal.aborted())
                        {
                            return Err(remote_error(
                                RemoteServiceErrorCode::ServiceStaleInstance,
                                format!("Remote service {service_id} observation is closed"),
                            ));
                        }
                        Ok(())
                    })
                };
                let view = slot.view(assert);
                (handler)(view, context);
            })
        };
        let stop = binding.instances.observe(observe_handler)?;
        if self.0.bound.get() && binding.starting.borrow().is_none() {
            let revision = binding.revision.get();
            let starting = self.start_keyed_wrapped(binding.clone(), revision);
            *binding.starting.borrow_mut() = Some(starting);
            drive_start_now(&binding.starting);
        }
        Ok(Box::new(move || {
            if stopped.get() {
                return;
            }
            stopped.set(true);
            stop();
            if binding.instances.observer_count() == 0 {
                binding.on_empty(&core);
            }
        }))
    }

    /// Waits until every currently acquired service has installed its
    /// initial snapshot, retrying while readiness revisions keep changing.
    #[must_use]
    pub fn ready(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        let core = self.0.clone();
        boxed(async move {
            if core.disposed.get() {
                return Err(ChordError::Message(
                    "Remote service binding is disposed".to_string(),
                ));
            }
            loop {
                let revision = core.readiness_revision.get();
                let mut starts: Vec<LocalBoxFuture<Result<(), ChordError>>> = Vec::new();
                if core.binding_transition.borrow().is_some() {
                    starts.push(settle_stored(&core.binding_transition));
                }
                for (_, binding) in core.singletons.borrow().iter() {
                    if binding.starting.borrow().is_some() {
                        starts.push(settle_stored(&binding.starting));
                    }
                }
                for (_, binding) in core.keyed.borrow().iter() {
                    starts.push(settle_stored(&binding.starting));
                }
                let outcome = crate::context::await_with_context(join_all(starts), &context).await;
                match outcome {
                    Err(reason) => return Err(ChordError::Message(reason.to_string())),
                    Ok(results) => {
                        for result in results {
                            result?;
                        }
                    }
                }
                if core.disposed.get() {
                    return Err(ChordError::Message(
                        "Remote service binding is disposed".to_string(),
                    ));
                }
                if revision == core.readiness_revision.get() {
                    return Ok(());
                }
            }
        })
    }

    /// Rebinds every acquired service: subscriptions close and restart, or
    /// close without restart when `bound` is `false`. Collected transition
    /// failures aggregate.
    #[must_use]
    pub fn rebind(&self, bound: bool, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        let core = self.0.clone();
        boxed(async move {
            if core.disposed.get() {
                return Err(ChordError::Message(
                    "Remote service binding is disposed".to_string(),
                ));
            }
            core.bound.set(bound);
            core.readiness_revision
                .set(core.readiness_revision.get() + 1);
            let mut transitions: Vec<LocalBoxFuture<Result<(), ChordError>>> = Vec::new();
            let singletons: Vec<(String, Rc<SingletonBindingState>)> =
                core.singletons.borrow().clone();
            for (service_id, binding) in singletons {
                binding.revision.set(binding.revision.get() + 1);
                binding.facade.clear();
                let subscription = binding.subscription.borrow_mut().take();
                let revision = binding.revision.get();
                let transition = {
                    let core = core.clone();
                    let binding = binding.clone();
                    let context = context.clone();
                    boxed(async move {
                        if let Some(subscription) = subscription {
                            (subscription.close)(Some(context.clone())).await?;
                        }
                        if bound {
                            Self::start_singleton(&core, &service_id, &binding, revision).await?;
                        }
                        Ok(())
                    })
                };
                *binding.starting.borrow_mut() = Some(transition);
                transitions.push(settle_stored(&binding.starting));
            }
            let keyed: Vec<Rc<KeyedBindingState>> = core
                .keyed
                .borrow()
                .iter()
                .map(|(_, binding)| binding.clone())
                .collect();
            for binding in keyed {
                let core = core.clone();
                let context = context.clone();
                transitions.push(boxed(
                    async move { binding.rebind(&core, bound, context).await },
                ));
            }
            let completion: LocalBoxFuture<Result<(), ChordError>> = boxed(async move {
                let results = join_all(transitions).await;
                let errors: Vec<ChordError> = results.into_iter().filter_map(Result::err).collect();
                collect_errors(errors, "Failed to rebind services").map_or(Ok(()), Err)
            });
            *core.binding_transition.borrow_mut() = Some(completion);
            settle_stored(&core.binding_transition).await
        })
    }

    /// Tears down every subscription and facade this binding owns.
    #[must_use]
    pub fn dispose(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        let core = self.0.clone();
        boxed(async move {
            if core.disposed.get() {
                return Ok(());
            }
            core.disposed.set(true);
            let mut closes: Vec<LocalBoxFuture<Result<(), ChordError>>> = Vec::new();
            let singletons: Vec<Rc<SingletonBindingState>> = core
                .singletons
                .borrow()
                .iter()
                .map(|(_, b)| b.clone())
                .collect();
            for binding in singletons {
                binding.active.set(false);
                binding.facade.clear();
                // Upstream swallows start failures on the disposal path
                // (`binding.starting.catch(() => {})`); readiness reports
                // them, disposal collects only close failures.
                if binding.starting.borrow().is_some() {
                    let starting = settle_stored(&binding.starting);
                    closes.push(boxed(async move {
                        let _ = starting.await;
                        Ok(())
                    }));
                }
                let subscription = binding.subscription.borrow_mut().take();
                if let Some(subscription) = subscription {
                    closes.push((subscription.close)(Some(context.clone())));
                }
            }
            let keyed: Vec<Rc<KeyedBindingState>> =
                core.keyed.borrow().iter().map(|(_, b)| b.clone()).collect();
            for binding in keyed {
                let core = core.clone();
                let context = context.clone();
                closes.push(boxed(async move { binding.close(&core, context).await }));
            }
            core.singletons.borrow_mut().clear();
            core.keyed.borrow_mut().clear();
            let results = join_all(closes).await;
            let errors: Vec<ChordError> = results.into_iter().filter_map(Result::err).collect();
            collect_errors(errors, "Failed to dispose services").map_or(Ok(()), Err)
        })
    }

    fn start_singleton_wrapped(
        &self,
        service_id: String,
        binding: Rc<SingletonBindingState>,
        revision: u64,
    ) -> LocalBoxFuture<Result<(), ChordError>> {
        let core = self.0.clone();
        boxed(async move {
            let result = Self::start_singleton(&core, &service_id, &binding, revision).await;
            if let Err(error) = &result
                && binding.active.get()
                && binding.revision.get() == revision
                && !core.disposed.get()
                && core.bound.get()
            {
                (core.report_error)(error);
            }
            result
        })
    }

    fn start_keyed_wrapped(
        &self,
        binding: Rc<KeyedBindingState>,
        revision: u64,
    ) -> LocalBoxFuture<Result<(), ChordError>> {
        let core = self.0.clone();
        boxed(async move {
            let result = binding.start(&core, revision).await;
            if let Err(error) = &result
                && !binding.closed.get()
                && binding.revision.get() == revision
                && binding.bound.get()
            {
                (core.report_error)(error);
            }
            result
        })
    }

    fn assert_remotable(service: &Service) -> Result<(), ChordError> {
        if service.local {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceNotAllowed,
                format!("Service {} is process-local", service.id),
            ));
        }
        Ok(())
    }

    fn assert_available(&self, service_id: &str, mode: ServiceMode) -> Result<(), ChordError> {
        if self.0.disposed.get() {
            return Err(ChordError::Message(
                "Remote service binding is disposed".to_string(),
            ));
        }
        if !self.0.allowlist.contains(service_id) {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceNotAllowed,
                format!("Remote service {service_id} is not allowlisted"),
            ));
        }
        if let Some(existing) = self
            .0
            .modes
            .borrow()
            .iter()
            .find(|(id, _)| id == service_id)
            .map(|(_, mode)| *mode)
        {
            if existing != mode {
                return Err(remote_error(
                    RemoteServiceErrorCode::ServiceModeMismatch,
                    format!(
                        "Remote service {service_id} is already used as {}",
                        existing.as_str()
                    ),
                ));
            }
            return Ok(());
        }
        self.0
            .modes
            .borrow_mut()
            .push((service_id.to_string(), mode));
        Ok(())
    }
}

impl BindingCore {
    fn assert_handle_access(&self) -> Result<(), ChordError> {
        if self.disposed.get() {
            return Err(ChordError::Message(
                "Remote service binding is disposed".to_string(),
            ));
        }
        (self.assert_access.clone())()
    }
}

impl RemoteServiceBinding {
    /// Starts one singleton's subscription and installs its initial
    /// snapshot, upstream's `#startSingleton`.
    ///
    /// # Panics
    /// Only if the subscription stored above disappears between the store
    /// and this read, which nothing between can make happen; a panic here is
    /// a port bug, not a runtime condition.
    async fn start_singleton(
        core: &Rc<BindingCore>,
        service_id: &str,
        binding: &Rc<SingletonBindingState>,
        revision: u64,
    ) -> Result<(), ChordError> {
        let listener: ServiceProviderListener = {
            let core_report = core.report_error.clone();
            let binding = binding.clone();
            Rc::new(move |update: &ServiceProviderUpdate, context: &Context| {
                if !binding.active.get() || binding.revision.get() != revision {
                    return;
                }
                if let Err(panic_catch) = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    let facade = binding.facade.clone();
                    let result: Result<(), ChordError> = match update {
                        ServiceProviderUpdate::Unavailable => {
                            facade.clear();
                            Ok(())
                        }
                        ServiceProviderUpdate::Replaced { snapshot } => {
                            if snapshot.instance.is_some() {
                                Err(ChordError::Message(
                                    "Singleton replacement has an instance address".to_string(),
                                ))
                            } else {
                                facade.install(snapshot, context)
                            }
                        }
                        ServiceProviderUpdate::State {
                            instance: None,
                            member,
                            sequence,
                            ops,
                        } => facade.update(member, *sequence, ops, context),
                        ServiceProviderUpdate::State { .. }
                        | ServiceProviderUpdate::Spawned { .. }
                        | ServiceProviderUpdate::Closed { .. } => Ok(()),
                    };
                    if let Err(error) = result {
                        (core_report)(&error);
                    }
                })) {
                    (core_report)(&ChordError::Message(panic_message(&*panic_catch)));
                }
            })
        };
        let subscription = core
            .transport
            .subscribe(
                service_id.to_string(),
                ServiceMode::Singleton,
                listener,
                background_context(),
            )
            .await?;
        // The await boundary upstream's resolved promise still inserts: the
        // subscribe call issues eagerly, and the installation continuation
        // runs when the awaiting task polls again.
        crate::future::yield_once().await;
        if !binding.active.get()
            || core.disposed.get()
            || !core.bound.get()
            || binding.revision.get() != revision
        {
            (subscription.close)(Some(background_context())).await?;
            return Ok(());
        }
        let snapshot = subscription.snapshot.clone();
        *binding.subscription.borrow_mut() = Some(subscription);
        if snapshot.mode != ServiceMode::Singleton
            || snapshot.service_id != service_id
            || snapshot.instances.len() != 1
        {
            return Err(ChordError::Message(format!(
                "Remote service {service_id} returned an invalid singleton snapshot"
            )));
        }
        binding
            .facade
            .install(&snapshot.instances[0], &service_delivery_context())?;
        #[allow(
            clippy::expect_used,
            reason = "the subscription was stored above and nothing between can remove it"
        )]
        {
            let stored = binding
                .subscription
                .borrow_mut()
                .take()
                .expect("subscription stored above");
            (stored.activate)()?;
            *binding.subscription.borrow_mut() = Some(stored);
        }
        Ok(())
    }
}

impl KeyedBindingState {
    fn on_empty(self: &Rc<Self>, core: &Rc<BindingCore>) {
        if !core
            .keyed
            .borrow()
            .iter()
            .any(|(_, binding)| Rc::ptr_eq(binding, self))
        {
            return;
        }
        core.keyed
            .borrow_mut()
            .retain(|(id, binding)| !(id == &self.service_id && Rc::ptr_eq(binding, self)));
        core.readiness_revision
            .set(core.readiness_revision.get() + 1);
        let this = self.clone();
        let close_core = core.clone();
        let close = boxed(async move { this.close(&close_core, background_context()).await });
        match crate::future::settle_now(close) {
            Some(Ok(())) | None => {}
            Some(Err(error)) => (core.report_error)(&error),
        }
    }

    /// Starts one keyed service's subscription, spawns every instance in the
    /// initial snapshot, and marks the directory ready.
    ///
    /// # Panics
    /// Only if the subscription stored above disappears between the store
    /// and this read, which nothing between can make happen; a panic here is
    /// a port bug, not a runtime condition.
    async fn start(
        self: &Rc<Self>,
        core: &Rc<BindingCore>,
        revision: u64,
    ) -> Result<(), ChordError> {
        let listener = keyed_listener(core, self, revision);
        let subscription = core
            .transport
            .subscribe(
                self.service_id.clone(),
                ServiceMode::Keyed,
                listener,
                background_context(),
            )
            .await?;
        if self.closed.get() || !self.bound.get() || self.revision.get() != revision {
            (subscription.close)(Some(background_context())).await?;
            return Ok(());
        }
        let snapshot = subscription.snapshot.clone();
        *self.subscription.borrow_mut() = Some(subscription);
        if snapshot.mode != ServiceMode::Keyed || snapshot.service_id != self.service_id {
            return Err(ChordError::Message(format!(
                "Remote service {} returned the wrong keyed snapshot",
                self.service_id
            )));
        }
        for instance in &snapshot.instances {
            self.spawn(core, instance, &service_delivery_context())?;
        }
        #[allow(
            clippy::expect_used,
            reason = "the subscription was stored above and nothing between can remove it"
        )]
        {
            let stored = self
                .subscription
                .borrow_mut()
                .take()
                .expect("subscription stored above");
            (stored.activate)()?;
            *self.subscription.borrow_mut() = Some(stored);
        }
        self.instances.ready()?;
        Ok(())
    }

    fn spawn(
        &self,
        core: &Rc<BindingCore>,
        snapshot: &ServiceInstanceSnapshot,
        context: &Context,
    ) -> Result<(), ChordError> {
        let Some(address) = &snapshot.instance else {
            return Err(ChordError::Message(
                "Keyed service instance snapshot has no address".to_string(),
            ));
        };
        let active = Rc::new(Cell::new(true));
        let closed = Rc::new(Cell::new(false));
        let is_active: Rc<dyn Fn() -> bool> = {
            let active = active.clone();
            Rc::new(move || active.get() && !closed.get())
        };
        let assert_access: AssertAccess = {
            let core = core.clone();
            Rc::new(move || core.assert_handle_access())
        };
        let facade = Rc::new(RemoteFacade::new(
            &self.service_id,
            Some(address.clone()),
            core.transport.clone(),
            is_active,
            assert_access,
            core.report_error.clone(),
        ));
        facade.install(snapshot, context)?;
        let entry = Rc::new(crate::services::instances::InstanceDirectoryEntry {
            key: address.key.clone(),
            generation: address.generation,
            service: ServiceTarget::Facade(facade.clone()),
            deactivate: Rc::new(move || {
                active.set(false);
                facade.clear();
            }),
        });
        self.instances.replace(entry)?;
        Ok(())
    }

    fn update(
        &self,
        core: &Rc<BindingCore>,
        update: &ServiceProviderUpdate,
        context: &Context,
    ) -> Result<(), ChordError> {
        if self.closed.get() {
            return Ok(());
        }
        match update {
            ServiceProviderUpdate::Unavailable | ServiceProviderUpdate::Replaced { .. } => {
                Err(ChordError::Message(
                    "Keyed service received a singleton lifecycle update".to_string(),
                ))
            }
            ServiceProviderUpdate::Spawned { instance } => self.spawn(core, instance, context),
            ServiceProviderUpdate::Closed { instance } => {
                let live = self.instances.get(&instance.key);
                if let Some(entry) = live
                    && entry.generation == instance.generation
                {
                    self.instances.remove(&entry);
                }
                Ok(())
            }
            ServiceProviderUpdate::State {
                instance: Some(address),
                member,
                sequence,
                ops,
            } => {
                let live = self.instances.get(&address.key);
                if let Some(entry) = live {
                    if entry.generation != address.generation {
                        return Ok(());
                    }
                    if let ServiceTarget::Facade(facade) = &entry.service {
                        facade.update(member, *sequence, ops, context)?;
                    }
                }
                Ok(())
            }
            ServiceProviderUpdate::State { instance: None, .. } => Err(ChordError::Message(
                "Keyed state update has no instance address".to_string(),
            )),
        }
    }

    async fn rebind(
        self: &Rc<Self>,
        core: &Rc<BindingCore>,
        bound: bool,
        context: Context,
    ) -> Result<(), ChordError> {
        if self.closed.get() {
            return Ok(());
        }
        self.bound.set(bound);
        self.revision.set(self.revision.get() + 1);
        let revision = self.revision.get();
        self.reset(core, &context, false).await?;
        if self.closed.get() || self.revision.get() != revision || self.bound.get() != bound {
            return Ok(());
        }
        if bound && self.instances.observer_count() > 0 {
            let this = self.clone();
            let core = core.clone();
            let starting = boxed(async move { this.start(&core, revision).await });
            *self.starting.borrow_mut() = Some(starting);
            settle_stored(&self.starting).await?;
        }
        Ok(())
    }

    async fn reset(
        &self,
        _core: &Rc<BindingCore>,
        context: &Context,
        wait_for_starting: bool,
    ) -> Result<(), ChordError> {
        self.instances.reset();
        let starting = self.starting.borrow_mut().take();
        let subscription = self.subscription.borrow_mut().take();
        if wait_for_starting && let Some(starting) = starting {
            let _ = starting.await;
        }
        if let Some(subscription) = subscription {
            (subscription.close)(Some(context.clone())).await?;
        }
        Ok(())
    }

    async fn close(
        self: &Rc<Self>,
        core: &Rc<BindingCore>,
        context: Context,
    ) -> Result<(), ChordError> {
        if self.closed.get() {
            return Ok(());
        }
        self.closed.set(true);
        self.revision.set(self.revision.get() + 1);
        self.reset(core, &context, true).await?;
        self.instances.dispose();
        Ok(())
    }
}

fn keyed_listener(
    core: &Rc<BindingCore>,
    binding: &Rc<KeyedBindingState>,
    revision: u64,
) -> ServiceProviderListener {
    let core = core.clone();
    let binding = binding.clone();
    Rc::new(move |update: &ServiceProviderUpdate, context: &Context| {
        if binding.revision.get() != revision {
            return;
        }
        let result =
            std::panic::catch_unwind(AssertUnwindSafe(|| binding.update(&core, update, context)));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => (core.report_error)(&error),
            Err(caught) => (core.report_error)(&ChordError::Message(panic_message(&*caught))),
        }
    })
}
impl crate::types::RemoteServices for RemoteServiceBinding {
    fn use_service(&self, service: &Service) -> Result<crate::handle::ServiceView, ChordError> {
        Self::use_service(self, service)
    }

    fn observe(
        &self,
        service: &Service,
        handler: crate::types::KeyedViewHandler,
    ) -> Result<Unsubscribe, ChordError> {
        Self::observe(self, service, handler)
    }

    fn ready(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        Self::ready(self, context)
    }

    fn dispose(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        Self::dispose(self, context)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        reason = "the slot fixtures settle results the case's own assertions pin"
    )]
    use super::*;
    use crate::handle::no_error_reporter;
    use crate::services::loopback::create_loopback_service_transport;

    fn transport() -> Rc<dyn RemoteServiceTransport> {
        let provider = crate::services::provider::RemoteServiceProvider::new(Vec::new())
            .expect("an empty catalogue has no duplicates");
        create_loopback_service_transport(&provider)
    }

    fn slot() -> MemberSlot {
        MemberSlot::new(
            "test.slot",
            "member",
            Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(ready_with(Ok(None)))),
            Rc::new(|| true),
            allow_access(),
            no_error_reporter(),
        )
    }

    #[test]
    fn spells_the_slot_and_facade_debug_surfaces() {
        assert!(format!("{:?}", slot()).starts_with("MemberSlot { service_id: \"test.slot\""));
        let options = RemoteServiceBindingOptions {
            services: Vec::new(),
            transport: transport(),
            bound: true,
            on_error: no_error_reporter(),
            assert_access: None,
        };
        assert!(format!("{options:?}").starts_with("RemoteServiceBindingOptions {"));
        let is_active: Rc<dyn Fn() -> bool> = Rc::new(|| true);
        let facade = RemoteFacade::new(
            "test.facade",
            None,
            transport(),
            is_active,
            allow_access(),
            no_error_reporter(),
        );
        assert!(format!("{facade:?}").contains("test.facade"));
        assert!(slot().state_value().is_ok());
        assert!(facade.same_facade(&facade));
    }

    #[test]
    fn slots_pin_member_kinds() {
        // A member described twice with different kinds rejects.
        let described = slot();
        described
            .set_description(ServiceMemberKind::State)
            .expect("the first kind lands");
        let error = described
            .set_description(ServiceMemberKind::Method)
            .expect_err("a kind change rejects");
        assert!(error.to_string().contains("changed kind"));

        // An expectation the description contradicts rejects.
        let expected = slot();
        expected
            .expect(ServiceMemberKind::Method)
            .expect("the first use pins");
        let error = expected
            .set_description(ServiceMemberKind::State)
            .expect_err("the description contradicts the expectation");
        assert!(error.to_string().contains("is state, not method"));

        // A member used as two different kinds rejects.
        let used = slot();
        used.expect(ServiceMemberKind::State)
            .expect("the state use pins");
        let error = used
            .expect(ServiceMemberKind::Method)
            .expect_err("the second use rejects");
        assert!(error.to_string().contains("used as two different kinds"));

        // A described method member rejects state reads.
        let method = slot();
        method
            .set_description(ServiceMemberKind::Method)
            .expect("the kind pins");
        let error = method
            .state_value()
            .expect_err("a method member rejects state reads");
        assert!(error.to_string().contains("is method, not state"));
    }
}
