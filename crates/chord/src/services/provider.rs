//! The remote service provider, ported from upstream
//! `src/services/provider.ts`.
//!
//! The provider hosts one implementation per registered service (a singleton
//! or keyed instances), resolves `$chord.service` invocations against live
//! members, and fans state publications out to subscribers with the
//! event-loop ordering upstream pins: inactive subscribers buffer updates,
//! and activation replays the buffer in order before any collected listener
//! failure surfaces. Implementations arrive as explicit member registries
//! ([`crate::handle::ServiceImplementation`]) — the owned-data restatement
//! of classifying a live object by reflection.

use std::cell::{Cell, RefCell};
use std::panic::AssertUnwindSafe;
use std::rc::Rc;

use crate::context::Context;
use crate::delta::Op;
use crate::errors::{ChordError, RemoteServiceError, RemoteServiceErrorCode, collect_errors};
use crate::future::{LocalBoxFuture, boxed};
use crate::handle::{RemoteMethod, ServiceImplementation, ServiceMember, validate_remote_implementation};
use crate::services::state::service_delivery_context;
use crate::types::{
    JsonValue, Service, ServiceCall, ServiceCatalogueEntry, ServiceInstanceAddress, ServiceMemberKind,
    ServiceMode, ServiceProviderListener, ServiceProviderUpdate, ServiceSubscription, Unsubscribe,
};
use crate::services::wire::{
    ServiceControlCall, catalogue_to_json, decode_service_control_call, snapshot_to_json,
};

/// The definition one provider registers a service under.
#[derive(Debug, Clone)]
pub struct ServiceProviderDefinition {
    /// The service being published.
    pub service: Service,
    /// The mode it is provided in.
    pub mode: ServiceMode,
}

/// The subscription close handle, upstream's `() => Promise<void>` as the
/// boxed synchronous close the port hands back.
pub(crate) type SubscriptionClose = Box<dyn Fn(Option<Context>) -> LocalBoxFuture<Result<(), ChordError>>>;

/// A singleton definition, upstream's bare `{ id }` catalogue entry.
#[must_use]
pub const fn singleton_definition(service: Service) -> ServiceProviderDefinition {
    ServiceProviderDefinition {
        service,
        mode: ServiceMode::Singleton,
    }
}

struct ProviderInstance {
    address: Option<ServiceInstanceAddress>,
    implementation: Rc<ServiceImplementation>,
    remove_member_listeners: RefCell<Vec<Unsubscribe>>,
    active: Cell<bool>,
}

struct Subscriber {
    listener: ServiceProviderListener,
    buffer: RefCell<Vec<(ServiceProviderUpdate, Context)>>,
    active: Cell<bool>,
    terminated: Cell<bool>,
    closed: Cell<bool>,
}

struct Registration {
    service_id: String,
    mode: ServiceMode,
    singleton: Option<Rc<ProviderInstance>>,
    singleton_shape: Option<Vec<(String, ServiceMemberKind)>>,
    /// Live keyed instances in spawn order, upstream's `Map` insertion
    /// order.
    instances: Vec<(String, Rc<ProviderInstance>)>,
    generations: std::collections::HashMap<String, u64>,
    subscribers: Vec<Rc<Subscriber>>,
}

struct ProviderCore {
    catalogue: Vec<ServiceCatalogueEntry>,
    registrations: RefCell<Vec<(String, Rc<RefCell<Registration>>)>>,
    disposed: Cell<bool>,
}

/// Hosts one implementation per registered remote service and owns that
/// service's instances and subscribers.
#[derive(Clone)]
pub struct RemoteServiceProvider(Rc<ProviderCore>);

impl std::fmt::Debug for RemoteServiceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteServiceProvider")
            .field("catalogue", &self.0.catalogue)
            .finish()
    }
}

impl RemoteServiceProvider {
    /// Builds a provider for `definitions`.
    ///
    /// # Errors
    /// [`ChordError`] when a local service is listed or the catalogue has
    /// duplicate IDs, upstream's constructor `TypeError`s.
    pub fn new(definitions: Vec<ServiceProviderDefinition>) -> Result<Self, ChordError> {
        for definition in &definitions {
            if definition.service.local {
                return Err(ChordError::Message(format!(
                    "Local service {} cannot be published remotely",
                    definition.service.id
                )));
            }
        }
        let ids: Vec<&str> = definitions.iter().map(|definition| definition.service.id.as_str()).collect();
        let unique = ids.iter().collect::<std::collections::BTreeSet<_>>().len();
        if unique != ids.len() {
            return Err(ChordError::Message(
                "Remote service catalogue contains duplicate IDs".to_string(),
            ));
        }
        let catalogue = definitions
            .iter()
            .map(|definition| ServiceCatalogueEntry {
                service_id: definition.service.id.clone(),
                mode: definition.mode,
            })
            .collect();
        let registrations = definitions
            .into_iter()
            .map(|definition| {
                (
                    definition.service.id.clone(),
                    Rc::new(RefCell::new(Registration {
                        service_id: definition.service.id,
                        mode: definition.mode,
                        singleton: None,
                        singleton_shape: None,
                        instances: Vec::new(),
                        generations: std::collections::HashMap::new(),
                        subscribers: Vec::new(),
                    })),
                )
            })
            .collect();
        Ok(Self(Rc::new(ProviderCore {
            catalogue,
            registrations: RefCell::new(registrations),
            disposed: Cell::new(false),
        })))
    }

    /// The catalogue this provider publishes, frozen at construction.
    #[must_use]
    pub fn catalogue(&self) -> &[ServiceCatalogueEntry] {
        &self.0.catalogue
    }

    /// Provides one singleton implementation.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service is local or
    /// not allowlisted, already provided, or the implementation is not
    /// remotely exposable.
    pub fn provide(&self, service: &Service, implementation: ServiceImplementation) -> Result<(), ChordError> {
        self.assert_active()?;
        assert_remotable(service)?;
        self.assert_allowed(&service.id)?;
        let registration = self.registration(&service.id, ServiceMode::Singleton)?;
        if registration.borrow().singleton.is_some() {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceModeMismatch,
                format!("Remote service {} already has a provider", service.id),
            ));
        }
        validate_remote_implementation(&service.id, &implementation)?;
        let implementation = Rc::new(implementation);
        let shape = implementation.member_shape();
        assert_singleton_shape(&registration.borrow(), &shape)?;
        let instance = create_instance(&registration, implementation, None);
        registration.borrow_mut().singleton = Some(instance);
        Ok(())
    }

    /// Disconnects one singleton while preserving active subscriptions and
    /// remote facades.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service is local or
    /// not allowlisted, or the mode mismatches.
    pub fn withdraw(&self, service: &Service) -> Result<(), ChordError> {
        self.assert_active()?;
        assert_remotable(service)?;
        self.assert_allowed(&service.id)?;
        let registration = self.registration(&service.id, ServiceMode::Singleton)?;
        let previous = registration.borrow().singleton.clone();
        let Some(previous) = previous else {
            return Ok(());
        };
        {
            let mut registration = registration.borrow_mut();
            previous.active.set(false);
            for remove in previous.remove_member_listeners.borrow().iter() {
                remove();
            }
            registration.singleton = None;
        }
        emit(&registration, &ServiceProviderUpdate::Unavailable, None)
    }

    /// Validates one singleton replacement without changing the active
    /// provider.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service is local or
    /// not allowlisted, or the replacement breaks the member shape.
    pub fn validate_replacement(
        &self,
        service: &Service,
        implementation: &ServiceImplementation,
    ) -> Result<(), ChordError> {
        self.assert_active()?;
        assert_remotable(service)?;
        self.assert_allowed(&service.id)?;
        let registration = self.registration(&service.id, ServiceMode::Singleton)?;
        validate_remote_implementation(&registration.borrow().service_id, implementation)?;
        assert_singleton_shape(&registration.borrow(), &implementation.member_shape())
    }

    /// Replaces one singleton without making its stable remote facade
    /// unavailable.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service is local or
    /// not allowlisted, or the replacement breaks the member shape.
    pub fn replace(&self, service: &Service, implementation: ServiceImplementation) -> Result<(), ChordError> {
        self.assert_active()?;
        assert_remotable(service)?;
        self.assert_allowed(&service.id)?;
        let registration = self.registration(&service.id, ServiceMode::Singleton)?;
        validate_remote_implementation(&service.id, &implementation)?;
        let implementation = Rc::new(implementation);
        let shape = implementation.member_shape();
        assert_singleton_shape(&registration.borrow(), &shape)?;
        let replacement = create_instance(&registration, implementation, None);
        {
            let mut registration = registration.borrow_mut();
            if let Some(previous) = registration.singleton.take() {
                previous.active.set(false);
                for remove in previous.remove_member_listeners.borrow().iter() {
                    remove();
                }
            }
            registration.singleton = Some(replacement.clone());
            registration.singleton_shape = Some(shape);
        }
        emit(
            &registration,
            &ServiceProviderUpdate::Replaced {
                snapshot: snapshot_instance(&replacement),
            },
            None,
        )
    }

    /// The local implementation of one singleton, for host-side access.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service is local or
    /// not allowlisted, or no singleton is provided.
    pub fn use_service(&self, service: &Service) -> Result<Rc<ServiceImplementation>, ChordError> {
        self.assert_active()?;
        assert_remotable(service)?;
        self.assert_allowed(&service.id)?;
        let registration = self.find_registration(&service.id)?;
        let registration = registration.borrow();
        if registration.mode != ServiceMode::Singleton {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceNotFound,
                format!("Remote service {} has no local provider", service.id),
            ));
        }
        let Some(instance) = registration.singleton.as_ref() else {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceNotFound,
                format!("Remote service {} has no local provider", service.id),
            ));
        };
        Ok(instance.implementation.clone())
    }

    /// Spawns one keyed instance and returns the closure that closes it.
    /// Closing is idempotent, and closing a since-replaced key is a no-op.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service is local or
    /// not allowlisted, the key is empty or live, or the mode mismatches.
    ///
    /// # Panics
    /// Never for instances [`spawn`](Self::spawn) created: the close closure
    /// reads back the address spawn attached; a panic would be a port bug,
    /// not a runtime condition.
    pub fn spawn(
        &self,
        service: &Service,
        key: &str,
        implementation: ServiceImplementation,
    ) -> Result<impl Fn() -> Result<(), ChordError> + 'static, ChordError> {
        self.assert_active()?;
        assert_remotable(service)?;
        self.assert_allowed(&service.id)?;
        if key.is_empty() {
            return Err(ChordError::Message(
                "Remote service instance key must not be empty".to_string(),
            ));
        }
        let registration = self.registration(&service.id, ServiceMode::Keyed)?;
        if registration.borrow().instances.iter().any(|(live, _)| live == key) {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceModeMismatch,
                format!("Remote service {} already has a live instance with key {key}", service.id),
            ));
        }
        let generation = registration.borrow().generations.get(key).copied().unwrap_or(0) + 1;
        registration.borrow_mut().generations.insert(key.to_string(), generation);
        let address = ServiceInstanceAddress {
            key: key.to_string(),
            generation,
        };
        validate_remote_implementation(&service.id, &implementation)?;
        let implementation = Rc::new(implementation);
        let instance = create_instance(&registration, implementation, Some(address));
        registration
            .borrow_mut()
            .instances
            .push((key.to_string(), instance.clone()));
        emit(
            &registration,
            &ServiceProviderUpdate::Spawned {
                instance: snapshot_instance(&instance),
            },
            None,
        )?;
        let close_registration = registration;
        let close_instance = instance;
        Ok(move || {
            let mut registration = close_registration.borrow_mut();
            if !registration
                .instances
                .iter()
                .any(|(_, current)| Rc::ptr_eq(current, &close_instance))
            {
                return Ok(());
            }
            close_instance.active.set(false);
            for remove in close_instance.remove_member_listeners.borrow().iter() {
                remove();
            }
            registration.instances.retain(|(_, current)| !Rc::ptr_eq(current, &close_instance));
            #[allow(
                clippy::expect_used,
                reason = "keyed instances always carry the address spawn attached; this closure only closes instances spawn created"
            )]
            let address = close_instance.address.clone().expect("keyed instances carry addresses");
            drop(registration);
            emit(&close_registration, &ServiceProviderUpdate::Closed { instance: address }, None)
        })
    }

    /// Invokes one remote method against the addressed instance.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service or member
    /// is unknown, the mode or member mismatches, or the member fails.
    #[must_use]
    pub fn invoke(&self, call: ServiceCall, context: Context) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>> {
        let resolved = self.resolve_method(call);
        boxed(async move {
            let (method, args) = resolved?;
            method(args, context).await
        })
    }

    /// Subscribes to one service's updates. The subscription starts
    /// inactive: buffered updates replay on
    /// [`ServiceSubscription::activate`], and pending publications flush
    /// before the subscription is registered, upstream's `#publishPending`.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is disposed, the service is not
    /// allowlisted or has no singleton provider, the mode mismatches, or the
    /// pending publication fails.
    pub fn subscribe(
        &self,
        service_id: &str,
        mode: ServiceMode,
        listener: ServiceProviderListener,
    ) -> Result<ServiceSubscription, ChordError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        let registration = self.registration(service_id, mode)?;
        if registration.borrow().mode == ServiceMode::Singleton && registration.borrow().singleton.is_none() {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceNotFound,
                format!("Remote service {service_id} has no provider"),
            ));
        }
        publish_pending(&registration)?;
        let subscriber = Rc::new(Subscriber {
            listener,
            buffer: RefCell::new(Vec::new()),
            active: Cell::new(false),
            terminated: Cell::new(false),
            closed: Cell::new(false),
        });
        registration.borrow_mut().subscribers.push(subscriber.clone());
        let snapshot = snapshot(&registration.borrow());
        let activate_subscriber = subscriber.clone();
        let activate: Box<dyn Fn() -> Result<(), ChordError>> = Box::new(move || {
            if activate_subscriber.closed.get() || activate_subscriber.active.get() {
                return Ok(());
            }
            activate_subscriber.active.set(true);
            let entries: Vec<(ServiceProviderUpdate, Context)> =
                activate_subscriber.buffer.borrow_mut().drain(..).collect();
            let mut errors = Vec::new();
            for (update, context) in entries {
                if let Err(error) = call_listener(&activate_subscriber.listener, &update, &context) {
                    errors.push(error);
                }
            }
            if activate_subscriber.terminated.get() {
                activate_subscriber.closed.set(true);
            }
            collect_errors(errors, "Failed to activate remote service subscription").map_or(Ok(()), Err)
        });
        let close_subscriber = subscriber;
        let close_registration = registration;
        let close: SubscriptionClose = Box::new(move |_| {
            if close_subscriber.closed.get() {
                return boxed(std::future::ready(Ok(())));
            }
            close_subscriber.closed.set(true);
            close_subscriber.buffer.borrow_mut().clear();
            close_registration
                .borrow_mut()
                .subscribers
                .retain(|current| !Rc::ptr_eq(current, &close_subscriber));
            boxed(std::future::ready(Ok(())))
        });
        Ok(ServiceSubscription {
            snapshot,
            activate,
            close,
        })
    }

    /// Tears every registration down: singletons become unavailable, keyed
    /// instances close, active subscribers close, and pending subscribers
    /// terminate. Collected publication failures aggregate.
    ///
    /// # Errors
    /// [`ChordError`] aggregating the failures collected while emitting.
    ///
    /// # Panics
    /// Never for instances [`spawn`](Self::spawn) created: dispose reads
    /// back the address spawn attached; a panic would be a port bug, not a
    /// runtime condition.
    pub fn dispose(&self) -> Result<(), ChordError> {
        if self.0.disposed.get() {
            return Ok(());
        }
        self.0.disposed.set(true);
        let mut errors = Vec::new();
        let registrations: Vec<Rc<RefCell<Registration>>> =
            self.0.registrations.borrow().iter().map(|(_, r)| r.clone()).collect();
        for registration in registrations {
            let singleton = registration.borrow().singleton.clone();
            if let Some(singleton) = singleton {
                {
                    let mut registration = registration.borrow_mut();
                    singleton.active.set(false);
                    for remove in singleton.remove_member_listeners.borrow().iter() {
                        remove();
                    }
                    registration.singleton = None;
                }
                if let Err(error) = emit(&registration, &ServiceProviderUpdate::Unavailable, None) {
                    errors.push(error);
                }
            }
            let instances: Vec<Rc<ProviderInstance>> =
                registration.borrow().instances.iter().map(|(_, i)| i.clone()).collect();
            for instance in instances {
                {
                    let mut registration = registration.borrow_mut();
                    instance.active.set(false);
                    for remove in instance.remove_member_listeners.borrow().iter() {
                        remove();
                    }
                    registration
                        .instances
                        .retain(|(_, current)| !Rc::ptr_eq(current, &instance));
                }
                #[allow(
                    clippy::expect_used,
                    reason = "keyed instances always carry the address spawn attached; dispose only closes instances spawn created"
                )]
                let address = instance.address.clone().expect("keyed instances carry addresses");
                if let Err(error) = emit(&registration, &ServiceProviderUpdate::Closed { instance: address }, None) {
                    errors.push(error);
                }
            }
            let subscribers: Vec<Rc<Subscriber>> = registration.borrow().subscribers.clone();
            for subscriber in subscribers {
                if subscriber.active.get() {
                    subscriber.closed.set(true);
                    subscriber.buffer.borrow_mut().clear();
                } else {
                    subscriber.terminated.set(true);
                }
            }
            registration.borrow_mut().subscribers.clear();
        }
        self.0.registrations.borrow_mut().clear();
        collect_errors(errors, "Failed to dispose remote service provider").map_or(Ok(()), Err)
    }

    fn find_registration(&self, service_id: &str) -> Result<Rc<RefCell<Registration>>, ChordError> {
        self.0
            .registrations
            .borrow()
            .iter()
            .find(|(id, _)| id == service_id)
            .map(|(_, registration)| registration.clone())
            .ok_or_else(|| {
                remote_error(
                    RemoteServiceErrorCode::ServiceNotFound,
                    format!("Unknown remote service {service_id}"),
                )
            })
    }

    fn registration(&self, service_id: &str, mode: ServiceMode) -> Result<Rc<RefCell<Registration>>, ChordError> {
        let registration = self.find_registration(service_id)?;
        if registration.borrow().mode != mode {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceModeMismatch,
                format!(
                    "Remote service {} is {}, not {}",
                    service_id,
                    registration.borrow().mode.as_str(),
                    mode.as_str()
                ),
            ));
        }
        Ok(registration)
    }

    fn assert_allowed(&self, service_id: &str) -> Result<(), ChordError> {
        if !self.0.registrations.borrow().iter().any(|(id, _)| id == service_id) {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceNotAllowed,
                format!("Remote service {service_id} is not allowlisted"),
            ));
        }
        Ok(())
    }

    fn assert_active(&self) -> Result<(), ChordError> {
        if self.0.disposed.get() {
            return Err(ChordError::Message("Remote service provider is disposed".to_string()));
        }
        Ok(())
    }

    fn resolve_method(
        &self,
        call: ServiceCall,
    ) -> Result<(RemoteMethod, Vec<JsonValue>), ChordError> {
        self.assert_active()?;
        self.assert_allowed(&call.service_id)?;
        let registration = self.find_registration(&call.service_id)?;
        let registration = registration.borrow();
        let instance = resolve_instance(&registration, call.instance.as_ref())?;
        let member = instance.implementation.member(&call.member).ok_or_else(|| {
            remote_error(
                RemoteServiceErrorCode::ServiceMemberNotFound,
                format!("Unknown remote service member {}.{}", call.service_id, call.member),
            )
        })?;
        let ServiceMember::Method(method) = member else {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceMemberMismatch,
                format!(
                    "Remote service member {}.{} is not a method",
                    call.service_id, call.member
                ),
            ));
        };
        Ok((method.clone(), call.args))
    }
}

/// Resolves the live instance one call addresses, keyed instances by
/// address and singletons by absence of one.
///
/// # Errors
/// [`ChordError`] when the mode mismatches the address, the singleton is
/// missing, the instance is unknown, or the generation is stale.
fn resolve_instance(
    registration: &Registration,
    address: Option<&ServiceInstanceAddress>,
) -> Result<Rc<ProviderInstance>, ChordError> {
    if registration.mode == ServiceMode::Singleton {
        if address.is_some() {
            return Err(remote_error(
                RemoteServiceErrorCode::ServiceModeMismatch,
                format!("Remote service {} is singleton", registration.service_id),
            ));
        }
        return registration.singleton.clone().ok_or_else(|| {
            remote_error(
                RemoteServiceErrorCode::ServiceNotFound,
                format!("Remote service {} has no provider", registration.service_id),
            )
        });
    }
    let Some(address) = address else {
        return Err(remote_error(
            RemoteServiceErrorCode::ServiceModeMismatch,
            format!("Remote service {} is keyed", registration.service_id),
        ));
    };
    let instance = registration
        .instances
        .iter()
        .find(|(key, _)| key == &address.key)
        .map(|(_, instance)| instance.clone())
        .ok_or_else(|| {
            remote_error(
                RemoteServiceErrorCode::ServiceInstanceNotFound,
                format!(
                    "Remote service {} has no instance {}",
                    registration.service_id, address.key
                ),
            )
        })?;
    if instance.address.as_ref().map(|stored| stored.generation) != Some(address.generation) {
        return Err(remote_error(
            RemoteServiceErrorCode::ServiceStaleInstance,
            format!(
                "Remote service {} instance {} is stale",
                registration.service_id, address.key
            ),
        ));
    }
    Ok(instance)
}

/// Flushes the tracked surface of every provided state before a
/// subscription registers, upstream's `#publishPending`.
///
/// # Errors
/// [`ChordError`] when one of the states fails to publish.
fn publish_pending(registration: &Rc<RefCell<Registration>>) -> Result<(), ChordError> {
    let context = service_delivery_context();
    let instances: Vec<Rc<ProviderInstance>> = match registration.borrow().mode {
        ServiceMode::Singleton => registration.borrow().singleton.clone().into_iter().collect(),
        ServiceMode::Keyed => registration
            .borrow()
            .instances
            .iter()
            .map(|(_, instance)| instance.clone())
            .collect(),
    };
    for instance in instances {
        for (_, member) in instance.implementation.members() {
            if let ServiceMember::State(state) = member {
                state.publish(&context)?;
            }
        }
    }
    Ok(())
}

/// Builds the snapshot a subscription starts from, keyed instances sorted
/// by key.
fn snapshot(registration: &Registration) -> crate::types::ServiceSubscriptionSnapshot {
    let instances = match registration.mode {
        ServiceMode::Singleton => registration
            .singleton
            .as_ref()
            .map(|singleton| vec![snapshot_instance(singleton)])
            .unwrap_or_default(),
        ServiceMode::Keyed => {
            let mut instances: Vec<Rc<ProviderInstance>> =
                registration.instances.iter().map(|(_, i)| i.clone()).collect();
            instances.sort_by_key(|instance| {
                instance
                    .address
                    .as_ref()
                    .map(|address| address.key.clone())
                    .unwrap_or_default()
            });
            instances.iter().map(|instance| snapshot_instance(instance)).collect()
        }
    };
    crate::types::ServiceSubscriptionSnapshot {
        service_id: registration.service_id.clone(),
        mode: registration.mode,
        instances,
    }
}

/// Builds one instance's member snapshot, upstream's `snapshotInstance`.
fn snapshot_instance(instance: &ProviderInstance) -> crate::types::ServiceInstanceSnapshot {
    let members = instance
        .implementation
        .members()
        .map(|(name, member)| match member {
            ServiceMember::Method(_) => crate::types::ServiceMemberSnapshot::Method { name: name.clone() },
            ServiceMember::State(state) => crate::types::ServiceMemberSnapshot::State {
                name: name.clone(),
                sequence: state.sequence(),
                ops: vec![Op::Replace(state.value())],
            },
            ServiceMember::Value(_) => unreachable!(),
        })
        .collect();
    crate::types::ServiceInstanceSnapshot {
        instance: instance.address.clone(),
        members,
    }
}

/// Fans one update out to a registration's subscribers: buffering inactive
/// ones and collecting delivery failures, upstream's `#deliver`.
///
/// # Errors
/// [`ChordError`] aggregating the listener failures, or the first listener
/// panic carried as a [`ChordError`].
fn emit(
    registration: &Rc<RefCell<Registration>>,
    update: &ServiceProviderUpdate,
    context: Option<&Context>,
) -> Result<(), ChordError> {
    let subscribers: Vec<Rc<Subscriber>> = {
        let registration = registration.borrow();
        registration.subscribers.clone()
    };
    if subscribers.is_empty() {
        return Ok(());
    }
    let delivery_context = context.map_or_else(service_delivery_context, Clone::clone);
    let mut errors = Vec::new();
    for subscriber in subscribers {
        if subscriber.closed.get() {
            continue;
        }
        if !subscriber.active.get() {
            subscriber
                .buffer
                .borrow_mut()
                .push((update.clone(), delivery_context.clone()));
            continue;
        }
        if let Err(error) = call_listener(&subscriber.listener, update, &delivery_context) {
            errors.push(error);
        }
    }
    collect_errors(
        errors,
        format!("Failed to publish remote service {service_id} update", service_id = registration.borrow().service_id),
    )
    .map_or(Ok(()), Err)
}

/// Validates that a service may be published remotely.
///
/// # Errors
/// [`ChordError`] when the service is process-local.
fn assert_remotable(service: &Service) -> Result<(), ChordError> {
    if service.local {
        return Err(remote_error(
            RemoteServiceErrorCode::ServiceNotAllowed,
            format!("Service {} is process-local", service.id),
        ));
    }
    Ok(())
}

/// Builds one instance and subscribes its state members to the
/// registration's fan-out, upstream's `#createInstance`.
fn create_instance(
    registration: &Rc<RefCell<Registration>>,
    implementation: Rc<ServiceImplementation>,
    address: Option<ServiceInstanceAddress>,
) -> Rc<ProviderInstance> {
    let instance = Rc::new(ProviderInstance {
        address,
        implementation,
        remove_member_listeners: RefCell::new(Vec::new()),
        active: Cell::new(true),
    });
    let members: Vec<(String, ServiceMember)> = instance
        .implementation
        .members()
        .map(|(name, member)| {
            let member = match member {
                ServiceMember::Method(method) => ServiceMember::Method(method.clone()),
                ServiceMember::State(state) => ServiceMember::State(state.clone()),
                ServiceMember::Value(value) => ServiceMember::Value(value.clone()),
            };
            (name.clone(), member)
        })
        .collect();
    for (name, member) in members {
        let ServiceMember::State(state) = member else {
            continue;
        };
        let listener = {
            let registration = registration.clone();
            let instance = instance.clone();
            let address = instance.address.clone();
            move |ops: &[Op], sequence: u64, context: &Context| -> Result<(), ChordError> {
                if !instance.active.get() {
                    return Ok(());
                }
                emit(
                    &registration,
                    &ServiceProviderUpdate::State {
                        instance: address.clone(),
                        member: name.clone(),
                        sequence,
                        ops: ops.to_vec(),
                    },
                    Some(context),
                )
            }
        };
        let unsubscribe = state.source_subscribe(Rc::new(listener));
        instance.remove_member_listeners.borrow_mut().push(unsubscribe);
    }
    instance
}

/// Runs one listener, upstream's try/catch around every delivery: a panic
/// carries as a [`ChordError`] so a fan-out keeps collecting.
fn call_listener(
    listener: &ServiceProviderListener,
    update: &ServiceProviderUpdate,
    context: &Context,
) -> Result<(), ChordError> {
    std::panic::catch_unwind(AssertUnwindSafe(|| listener(update, context)))
        .map_err(|panic| ChordError::Message(crate::services::state::panic_message(&*panic)))
}

fn remote_error(code: RemoteServiceErrorCode, message: impl Into<String>) -> ChordError {
    ChordError::Remote(RemoteServiceError::new(code, message))
}

fn assert_singleton_shape(
    registration: &Registration,
    replacement: &[(String, ServiceMemberKind)],
) -> Result<(), ChordError> {
    match &registration.singleton_shape {
        None => Ok(()),
        Some(current) if current.as_slice() == replacement => Ok(()),
        Some(_) => Err(remote_error(
            RemoteServiceErrorCode::ServiceMemberMismatch,
            format!(
                "Remote service {} replacement must preserve its member shape",
                registration.service_id
            ),
        )),
    }
}

/// Hosts one provider for one remote consumer and owns that consumer's
/// subscriptions, upstream's `RemoteServiceEndpoint`.
///
/// The publisher an endpoint hands subscription updates to, upstream's
/// `ServiceUpdatePublisher` with the promise arm dropped: publishing is
/// synchronous, and a failure propagates like upstream's sync throw.
#[derive(Clone)]
pub struct RemoteServiceEndpoint {
    provider: RemoteServiceProvider,
    subscriptions: Rc<RefCell<Vec<(String, ServiceSubscription)>>>,
    disposed: Rc<Cell<bool>>,
}

impl std::fmt::Debug for RemoteServiceEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteServiceEndpoint").finish_non_exhaustive()
    }
}

impl RemoteServiceEndpoint {
    /// Invokes one call, decoding `$chord.service` control calls first.
    ///
    /// # Errors
    /// [`ChordError`] when the endpoint is disposed, a control call is
    /// malformed, the subscription identity is duplicate or unknown, or the
    /// underlying invocation fails.
    pub fn invoke(
        &self,
        call: ServiceCall,
        publish: crate::types::ServiceUpdatePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>> {
        if self.disposed.get() {
            return boxed(std::future::ready(Err(ChordError::Message(
                "Remote service endpoint is disposed".to_string(),
            ))));
        }
        let control = decode_service_control_call(&call);
        match control {
            Some(ServiceControlCall::Catalogue) => {
                boxed(std::future::ready(Ok(Some(catalogue_to_json(self.provider.catalogue())))))
            }
            Some(ServiceControlCall::Subscribe {
                subscription_id,
                service_id,
                mode,
            }) => {
                if self
                    .subscriptions
                    .borrow()
                    .iter()
                    .any(|(id, _)| *id == subscription_id)
                {
                    return boxed(std::future::ready(Err(ChordError::Message(
                        "Service subscription ID is already active".to_string(),
                    ))));
                }
                let listener: ServiceProviderListener = {
                    let subscription_id = subscription_id.clone();
                    Rc::new(move |update: &ServiceProviderUpdate, update_context: &Context| {
                        publish(&subscription_id, update, update_context);
                    })
                };
                let subscription = match self.provider.subscribe(&service_id, mode, listener) {
                    Ok(subscription) => subscription,
                    Err(error) => return boxed(std::future::ready(Err(error))),
                };
                if let Err(error) = (subscription.activate)() {
                    return boxed(std::future::ready(Err(error)));
                }
                let snapshot = snapshot_to_json(&subscription.snapshot);
                self.subscriptions
                    .borrow_mut()
                    .push((subscription_id, subscription));
                boxed(std::future::ready(Ok(Some(snapshot))))
            }
            Some(ServiceControlCall::Unsubscribe { subscription_id }) => {
                let found = {
                    let mut subscriptions = self.subscriptions.borrow_mut();
                    subscriptions
                        .iter()
                        .position(|(id, _)| *id == subscription_id)
                        .map(|at| subscriptions.remove(at).1)
                };
                found.map_or_else(
                    || {
                        boxed(std::future::ready(Err(ChordError::Message(
                            "Service subscription was not found".to_string(),
                        ))))
                    },
                    |subscription| {
                        boxed(async move {
                            (subscription.close)(None).await?;
                            Ok(None)
                        })
                    },
                )
            }
            None => self.provider.invoke(call, context),
        }
    }

    /// Closes every subscription this endpoint opened; repeat calls are
    /// no-ops.
    pub fn dispose(&self) {
        if self.disposed.get() {
            return;
        }
        self.disposed.set(true);
        let subscriptions: Vec<ServiceSubscription> =
            self.subscriptions.borrow_mut().drain(..).map(|(_, s)| s).collect();
        for subscription in subscriptions {
            drop((subscription.close)(None));
        }
    }
}

/// Creates the endpoint view over one provider, the host side of the
/// `$chord.service` grammar.
#[must_use]
pub fn create_remote_service_endpoint(provider: &RemoteServiceProvider) -> RemoteServiceEndpoint {
    RemoteServiceEndpoint {
        provider: provider.clone(),
        subscriptions: Rc::new(RefCell::new(Vec::new())),
        disposed: Rc::new(Cell::new(false)),
    }
}

/// Validates one implementation as remotely exposable, the export upstream
/// spells `validateRemoteServiceImplementation`.
///
/// # Errors
/// [`ChordError`] when the implementation carries non-exposable members.
pub fn validate_remote_service_implementation(
    service_id: &str,
    implementation: &ServiceImplementation,
) -> Result<(), ChordError> {
    validate_remote_implementation(service_id, implementation)
}