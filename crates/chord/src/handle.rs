//! Service handles: the implementation member model and the guarded views
//! consumers see.
//!
//! Ported from upstream `src/services/handle.ts` with the classification
//! vocabulary upstream derives by reflecting over implementation objects.
//! JavaScript reads member sets off a live object; owned Rust data has no
//! reflection, so an implementation declares its members explicitly in a
//! [`ServiceImplementation`] and every consumer reaches them through a
//! [`ServiceView`] whose target is resolved at call time — a bound target
//! can be swapped underneath a retained view, which is what keeps handles
//! stable across provider replacement and facet reloads. The upstream
//! deep-proxy chain (a member resolving to another member-view of a nested
//! object) restates as direct access inside method bodies and [`ServiceMember::Value`]
//! members, because Rust methods own their data instead of walking
//! property chains.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::context::Context;
use crate::errors::ChordError;
use crate::future::{LocalBoxFuture, boxed};
use crate::services::state::MutableReplicatedState;
use crate::types::{JsonValue, ServiceMemberKind, Unsubscribe};

/// The gate every handle access passes before touching a service target.
/// Upstream threads an `assertAccess: () => void` closure that throws;
/// failures carry as [`ChordError`] here.
pub type AssertAccess = Rc<dyn Fn() -> Result<(), ChordError>>;

/// A gate that always passes, the default for hosts and bindings.
#[must_use]
pub fn allow_access() -> AssertAccess {
    Rc::new(|| Ok(()))
}

/// Reports asynchronous failures outside a caller's control, upstream's
/// `onError`.
pub type ErrorReporter = Rc<dyn Fn(&ChordError)>;

/// A reporter that drops everything, the default.
#[must_use]
pub fn no_error_reporter() -> ErrorReporter {
    Rc::new(|_| {})
}

/// One resource cleanup, upstream's `() => void | Promise<void>` disposal:
/// the future arm keeps asynchronous teardown possible, the result carries
/// the failure a disposal fan-out collects.
pub type Disposal = Box<dyn FnOnce() -> LocalBoxFuture<Result<(), ChordError>>>;

/// Builds a [`Disposal`] from a synchronous body.
#[must_use]
pub fn sync_disposal(disposal: impl FnOnce() -> Result<(), ChordError> + 'static) -> Disposal {
    Box::new(move || boxed(async move { disposal() }))
}

/// A remote method member: arguments in wire order, then the invocation
/// context.
///
/// Upstream spells the return a `Promise` of JSON or void; the boxed future
/// is that promise, driven by the caller's runtime. Shared ownership
/// ([`Rc`]) lets providers clone a member into the future that awaits it.
pub type RemoteMethod =
    Rc<dyn Fn(Vec<JsonValue>, Context) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>>>;

/// Builds a [`RemoteMethod`] from a synchronous body, for methods that
/// produce their result without an await point.
#[must_use]
pub fn sync_method(
    method: impl Fn(Vec<JsonValue>, &Context) -> Result<Option<JsonValue>, ChordError> + 'static,
) -> RemoteMethod {
    let method = Rc::new(method);
    Rc::new(move |args, context| {
        let method = method.clone();
        boxed(async move { method(args, &context) })
    })
}

/// One member of a [`ServiceImplementation`].
#[derive(Clone)]
pub enum ServiceMember {
    /// An invocable method.
    Method(RemoteMethod),
    /// Replicated state the provider publishes and consumers subscribe to.
    State(MutableReplicatedState),
    /// Arbitrary owned data, process-local only: remote implementations
    /// reject it because no wire form carries it.
    Value(Rc<dyn Any>),
}

impl std::fmt::Debug for ServiceMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Method(_) => f.write_str("Method(_)"),
            Self::State(state) => write!(f, "State({state:?})"),
            Self::Value(_) => f.write_str("Value(_)"),
        }
    }
}

/// A service implementation as an explicit member registry, the owned-data
/// restatement of the object upstream classifies by reflection.
///
/// Member order is sorted by name, matching the classification order
/// upstream's snapshots and shape checks observe.
#[derive(Clone, Debug, Default)]
pub struct ServiceImplementation {
    members: std::collections::BTreeMap<String, ServiceMember>,
}

impl ServiceImplementation {
    /// An implementation with no members; provide sites reject it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a method member, replacing any previous one under `name`.
    pub fn method(&mut self, name: impl Into<String>, method: RemoteMethod) -> &mut Self {
        self.members
            .insert(name.into(), ServiceMember::Method(method));
        self
    }

    /// Adds a replicated-state member.
    pub fn state(&mut self, name: impl Into<String>, state: MutableReplicatedState) -> &mut Self {
        self.members
            .insert(name.into(), ServiceMember::State(state));
        self
    }

    /// Adds an owned data member; only process-local implementations accept
    /// these, and remote validation rejects them as not exposable.
    pub fn value(&mut self, name: impl Into<String>, data: Rc<dyn Any>) -> &mut Self {
        self.members.insert(name.into(), ServiceMember::Value(data));
        self
    }

    /// The members in registration (sorted-name) order.
    pub fn members(&self) -> impl Iterator<Item = (&String, &ServiceMember)> {
        self.members.iter()
    }

    /// The member under `name`.
    #[must_use]
    pub fn member(&self, name: &str) -> Option<&ServiceMember> {
        self.members.get(name)
    }

    /// Whether any member is declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The member kinds by name, the shape a singleton replacement must
    /// preserve.
    #[must_use]
    pub fn member_shape(&self) -> Vec<(String, ServiceMemberKind)> {
        self.members
            .iter()
            .map(|(name, member)| {
                let kind = match member {
                    ServiceMember::Method(_) => ServiceMemberKind::Method,
                    ServiceMember::State(_) | ServiceMember::Value(_) => ServiceMemberKind::State,
                };
                (name.clone(), kind)
            })
            .collect()
    }
}

/// Validates an implementation as remotely exposable: every member is a
/// method or replicated state, and at least one member exists.
///
/// # Errors
/// [`ChordError`] when a member is a value (not remotely exposable) or the
/// implementation has no members.
pub fn validate_remote_implementation(
    service_id: &str,
    implementation: &ServiceImplementation,
) -> Result<(), ChordError> {
    if implementation.is_empty() {
        return Err(ChordError::Message(format!(
            "Remote service {service_id} has no members"
        )));
    }
    for (name, member) in &implementation.members {
        if matches!(member, ServiceMember::Value(_)) {
            return Err(ChordError::Message(format!(
                "Remote service member {service_id}.{name} is not remotely exposable"
            )));
        }
    }
    Ok(())
}

/// The target a [`ServiceSlot`] resolves member access against.
///
/// Targets are a local implementation registry, a consumer facade over a
/// transport, or a view another source already guards (the layering
/// upstream's proxy chain restates).
#[derive(Clone, Debug)]
pub enum ServiceTarget {
    /// A host-provided implementation.
    Local(Rc<ServiceImplementation>),
    /// A consumer facade; defined in [`crate::consumer`].
    Facade(Rc<crate::consumer::RemoteFacade>),
    /// A view another layer produced; every accessor delegates to it,
    /// keeping that layer's access gate in the chain.
    View(ServiceView),
}

struct SlotCore {
    service_id: String,
    target: RefCell<Option<ServiceTarget>>,
}

/// Host-owned mutable target with consumer-owned guarded views.
///
/// Upstream's `ServiceSlot` binds the current target and hands out proxies
/// that resolve every property at access time; the port hands out
/// [`ServiceView`]s whose accessors resolve the bound target per call, so
/// a rebind swaps the target under a retained view. Upstream's
/// `wrapObjects` flag (whether object members become call-through proxies)
/// restates to nothing: members are addressed by name, never by property
/// chain.
#[derive(Clone)]
pub struct ServiceSlot(Rc<SlotCore>);

impl std::fmt::Debug for ServiceSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceSlot")
            .field("service_id", &self.0.service_id)
            .field("bound", &self.0.target.borrow().is_some())
            .finish()
    }
}

impl ServiceSlot {
    /// An unbound slot for `service_id`.
    #[must_use]
    pub fn new(service_id: impl Into<String>) -> Self {
        Self(Rc::new(SlotCore {
            service_id: service_id.into(),
            target: RefCell::new(None),
        }))
    }

    /// Binds `target`; every later access resolves through it.
    pub fn bind(&self, target: ServiceTarget) {
        *self.0.target.borrow_mut() = Some(target);
    }

    /// Clears the binding; accessors report the service as disconnected.
    pub fn unbind(&self) {
        *self.0.target.borrow_mut() = None;
    }

    /// Whether a target is currently bound.
    #[must_use]
    pub fn is_bound(&self) -> bool {
        self.0.target.borrow().is_some()
    }

    /// A guarded view over this slot; each call mints a fresh handle, so
    /// callers that need stable identity cache the view.
    #[must_use]
    pub fn view(&self, assert_access: AssertAccess) -> ServiceView {
        ServiceView {
            slot: self.clone(),
            assert_access,
            identity: Rc::new(()),
        }
    }
}

/// A consumer-owned view over one [`ServiceSlot`]: every accessor first
/// passes the view's access gate, then resolves the slot's current target.
#[derive(Clone)]
pub struct ServiceView {
    slot: ServiceSlot,
    assert_access: AssertAccess,
    /// Per-view identity, minted at view creation: two views over one slot
    /// are still distinct handles, the proxy-object identity upstream's
    /// `Object.is` assertions observe across facets.
    identity: Rc<()>,
}

impl std::fmt::Debug for ServiceView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceView")
            .field("service_id", &self.service_id())
            .finish()
    }
}

impl ServiceView {
    /// Whether two views are the same handle, the restatement of
    /// upstream's proxy identity (`Object.is` on handles). Hosts and
    /// bindings cache one view per service per facet, which is what makes
    /// identity stable across replacement, exactly as upstream's proxy
    /// caches do.
    #[must_use]
    pub fn same_handle(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.identity, &other.identity)
    }

    /// The service ID this view addresses.
    #[must_use]
    pub fn service_id(&self) -> &str {
        &self.slot.0.service_id
    }

    fn target(&self) -> Result<ServiceTarget, ChordError> {
        self.slot.0.target.borrow().clone().ok_or_else(|| {
            ChordError::Message(format!(
                "Service {} is disconnected",
                self.slot.0.service_id
            ))
        })
    }

    /// Invokes one method member.
    ///
    /// # Errors
    /// [`ChordError`] when the access gate or the member dispatch rejects,
    /// or the member is not a method.
    #[must_use]
    pub fn call(
        &self,
        member: &str,
        args: Vec<JsonValue>,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>> {
        let access = self.assert_access.clone();
        let slot = self.slot.clone();
        let member = member.to_string();
        boxed(async move {
            (access)()?;
            let target = resolve_target(&slot)?;
            match target {
                ServiceTarget::Local(implementation) => {
                    invoke_local_member(&member, &implementation, args, context).await
                }
                ServiceTarget::Facade(facade) => facade.call_member(&member, args, context).await,
                ServiceTarget::View(view) => view.call(&member, args, context).await,
            }
        })
    }

    /// Resolves one state member, either the source state a local target
    /// carries or the replica a facade keeps.
    ///
    /// # Errors
    /// [`ChordError`] when the access gate or member dispatch rejects.
    pub fn state(&self, member: &str) -> Result<StateMemberView, ChordError> {
        (self.assert_access.clone())()?;
        match self.target()? {
            ServiceTarget::Local(implementation) => match implementation.member(member) {
                Some(ServiceMember::State(state)) => Ok(StateMemberView {
                    access: self.assert_access.clone(),
                    inner: StateMemberInner::Source(state.clone()),
                }),
                Some(_) => Err(ChordError::Message(
                    "Service member is not replicated state".to_string(),
                )),
                None => Err(ChordError::Message(format!(
                    "Service {} is disconnected",
                    self.slot.0.service_id
                ))),
            },
            ServiceTarget::Facade(facade) => {
                let slot = facade.member_slot(member);
                Ok(StateMemberView {
                    access: self.assert_access.clone(),
                    inner: StateMemberInner::Replica(slot),
                })
            }
            ServiceTarget::View(view) => {
                // The outer gate stays on the member view: every later read
                // passes the facet's lifecycle gate and then the inner view's
                // own gate, upstream's facet-then-binding gate chain on a
                // retained state member.
                Ok(StateMemberView {
                    access: self.assert_access.clone(),
                    inner: view.state(member)?.inner,
                })
            }
        }
    }

    /// Reads one [`ServiceMember::Value`] member of a local target, guarded
    /// by the access gate. The caller narrows the payload.
    ///
    /// # Errors
    /// [`ChordError`] when the access gate or the member dispatch rejects.
    pub fn with_value<R>(
        &self,
        member: &str,
        read: impl FnOnce(&dyn Any) -> R,
    ) -> Result<R, ChordError> {
        (self.assert_access.clone())()?;
        match self.target()? {
            ServiceTarget::Local(implementation) => match implementation.member(member) {
                Some(ServiceMember::Value(data)) => Ok(read(data.as_ref())),
                Some(_) => Err(ChordError::Message(
                    "Service member is not a value".to_string(),
                )),
                None => Err(ChordError::Message(format!(
                    "Service {} is disconnected",
                    self.slot.0.service_id
                ))),
            },
            ServiceTarget::Facade(_) => Err(ChordError::Message(
                "Remote service members are not values".to_string(),
            )),
            ServiceTarget::View(view) => view.with_value(member, read),
        }
    }
}

/// One replicated-state member as a consumer sees it.
///
/// The target is a source state when local, a replica when the target is a
/// facade. Every read re-passes the view's access gate, the per-property
/// proxy trap.
pub struct StateMemberView {
    access: AssertAccess,
    inner: StateMemberInner,
}

impl std::fmt::Debug for StateMemberView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateMemberView").finish_non_exhaustive()
    }
}

enum StateMemberInner {
    Source(MutableReplicatedState),
    Replica(Rc<crate::consumer::MemberSlot>),
}

impl StateMemberView {
    fn check(&self) -> Result<(), ChordError> {
        (self.access)()
    }

    /// The current value, or [`None`] until hydration.
    ///
    /// # Errors
    /// [`ChordError`] when the access gate rejects or the member is not
    /// state.
    pub fn value(&self) -> Result<Option<JsonValue>, ChordError> {
        self.check()?;
        match &self.inner {
            StateMemberInner::Source(state) => Ok(Some(state.value())),
            StateMemberInner::Replica(slot) => slot.state_value(),
        }
    }

    /// Subscribes to revisions; the first delivery hydrates the listener
    /// with the current value.
    ///
    /// # Errors
    /// [`ChordError`] when the access gate rejects or the member is not
    /// state.
    pub fn subscribe(
        &self,
        listener: crate::services::state::ValueListener,
    ) -> Result<Unsubscribe, ChordError> {
        self.check()?;
        match &self.inner {
            StateMemberInner::Source(state) => state.subscribe(listener),
            StateMemberInner::Replica(slot) => slot.state_subscribe(listener),
        }
    }
}

fn resolve_target(slot: &ServiceSlot) -> Result<ServiceTarget, ChordError> {
    slot.0.target.borrow().clone().ok_or_else(|| {
        ChordError::Message(format!("Service {} is disconnected", slot.0.service_id))
    })
}

fn invoke_local_member(
    member: &str,
    implementation: &ServiceImplementation,
    args: Vec<JsonValue>,
    context: Context,
) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>> {
    match implementation.member(member) {
        Some(ServiceMember::Method(method)) => method(args, context),
        _ => boxed(std::future::ready(Err(ChordError::Message(
            "Service member is not callable".to_string(),
        )))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::background_context;

    fn implementation_with_state_and_value() -> Rc<ServiceImplementation> {
        let state = MutableReplicatedState::new(JsonValue::Null);
        let mut implementation = ServiceImplementation::new();
        implementation.method(
            "read",
            sync_method(|_args: Vec<JsonValue>, _context: &Context| Ok(None)),
        );
        implementation.state("state", state);
        implementation.value("value", Rc::new(7u32));
        Rc::new(implementation)
    }

    #[test]
    fn spells_member_kinds_and_slot_surfaces() {
        let implementation = implementation_with_state_and_value();
        assert_eq!(
            format!("{:?}", implementation.member("read")),
            "Some(Method(_))"
        );
        assert!(
            format!("{:?}", implementation.member("state"))
                .starts_with("Some(State(MutableReplicatedState")
        );
        assert_eq!(
            format!("{:?}", implementation.member("value")),
            "Some(Value(_))"
        );
        let slot = ServiceSlot::new("test.member-kinds");
        assert!(!slot.is_bound());
        assert!(format!("{slot:?}").contains("test.member-kinds"));
        assert!(format!("{:?}", slot.view(allow_access())).contains("test.member-kinds"));
    }

    #[test]
    fn validates_remote_implementations_by_shape() {
        let empty = ServiceImplementation::new();
        let Err(error) = validate_remote_implementation("test.remote", &empty) else {
            unreachable!("an empty implementation rejects")
        };
        assert!(error.to_string().contains("has no members"));
    }

    #[test]
    fn local_views_surface_state_members_and_rejections() {
        let slot = ServiceSlot::new("test.local-surface");
        slot.bind(ServiceTarget::Local(implementation_with_state_and_value()));
        let view = slot.view(allow_access());
        let Ok(state) = view.state("state") else {
            unreachable!("the state member binds")
        };
        assert!(format!("{state:?}").starts_with("StateMemberView"));
        let Ok(value) = state.value() else {
            unreachable!("the state value reads")
        };
        assert_eq!(value, Some(JsonValue::Null));
        let Err(error) = view.state("read") else {
            unreachable!("a method member is not state")
        };
        assert!(error.to_string().contains("not replicated state"));
        let Err(error) = view.state("absent") else {
            unreachable!("an absent member disconnects")
        };
        assert!(error.to_string().contains("disconnected"));
        // A value member reads through its data; a non-value member rejects.
        let answer: u32 = view
            .with_value("value", |data: &dyn Any| {
                data.downcast_ref::<u32>().copied().unwrap_or_default()
            })
            .unwrap_or_default();
        assert_eq!(answer, 7);
        let Err(read_error): Result<u32, ChordError> =
            view.with_value("read", |_never: &dyn Any| unreachable!())
        else {
            unreachable!("a method member is not a value")
        };
        assert!(read_error.to_string().contains("not a value"));
        let Err(absent_error): Result<u32, ChordError> =
            view.with_value("absent", |_never: &dyn Any| unreachable!())
        else {
            unreachable!("an absent member disconnects")
        };
        assert!(absent_error.to_string().contains("disconnected"));
        // A value member subscribes to its source state through the same
        // gate.
        let listener: crate::services::state::ValueListener = Rc::new(|_, _, _| ());
        assert!(state.subscribe(listener).is_ok());
        slot.unbind();
        let Err(error): Result<u32, ChordError> =
            view.with_value("value", |_never: &dyn Any| unreachable!())
        else {
            unreachable!("an unbound slot disconnects")
        };
        assert!(error.to_string().contains("disconnected"));
    }

    #[test]
    fn disconnected_views_reject_invocations() {
        let slot = ServiceSlot::new("test.disconnected");
        let view = slot.view(allow_access());
        let Some(Err(error)) =
            crate::future::drive_once(view.call("read", vec![], background_context())).ok()
        else {
            unreachable!("a disconnected call settles immediately")
        };
        assert!(error.to_string().contains("disconnected"));
    }

    #[test]
    fn non_callable_members_reject_invocations() {
        let slot = ServiceSlot::new("test.not-callable");
        slot.bind(ServiceTarget::Local(implementation_with_state_and_value()));
        let view = slot.view(allow_access());
        let Some(Err(error)) =
            crate::future::drive_once(view.call("state", vec![], background_context())).ok()
        else {
            unreachable!("a local rejection settles immediately")
        };
        assert!(error.to_string().contains("not callable"));
    }
}
