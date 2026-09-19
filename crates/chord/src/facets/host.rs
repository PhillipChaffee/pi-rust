//! The facet kernel, ported from upstream `src/facets/host.ts`.
//!
//! The host validates the facet graph (missing dependencies, cycles,
//! duplicate providers), binds stable service handles, activates providers
//! before consumers, and disposes in reverse order. Reloads stage a fresh
//! generation, validate its shape against the live one, and cut over only
//! after every candidate activates. Service access is gated per facet
//! lifecycle, upstream's lifecycle-state machine; keyed observations start
//! only when the observing facet activates.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::consumer::create_remote_service_binding;
use crate::consumer::{RemoteServiceBinding, RemoteServiceBindingOptions};
use crate::context::background_context;
use crate::errors::{ChordError, collect_errors};
use crate::future::{LocalBoxFuture, boxed, join_all};
use crate::handle::{
    AssertAccess, Disposal, ErrorReporter, ServiceImplementation, ServiceSlot, ServiceTarget,
    ServiceView, allow_access, sync_disposal,
};
use crate::services::instances::{InstanceDirectory, InstanceDirectoryEntry};
use crate::services::provider::RemoteServiceProvider;
use crate::services::state::MutableReplicatedState;
use crate::types::{
    FacetDef, KeyedServiceHandler, RemoteServiceSource, RemoteServices, Service,
    ServiceCatalogueEntry, ServiceMode, Unsubscribe,
};

/// One declared service reference on a facet's requirement or provision
/// list.
#[derive(Debug, Clone)]
struct FacetServiceReference {
    service_id: String,
    mode: ServiceMode,
}

#[derive(Clone)]
enum Provision {
    Singleton {
        service: Service,
        implementation: Rc<ServiceImplementation>,
    },
    Keyed {
        service: Service,
        spawner: Rc<StagedServiceSpawner>,
    },
}

impl Provision {
    const fn service(&self) -> &Service {
        match self {
            Self::Singleton { service, .. } | Self::Keyed { service, .. } => service,
        }
    }
}

struct RuntimeRecord {
    facet_id: String,
    requires: RefCell<Vec<FacetServiceReference>>,
    provides: RefCell<Vec<FacetServiceReference>>,
    provisions: RefCell<Vec<Provision>>,
    singleton_views: RefCell<HashMap<String, ServiceView>>,
    lifecycle: Rc<FacetLifecycle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecycleState {
    SettingUp,
    Prepared,
    Active,
    Disposing,
    Dead,
}

struct FacetLifecycle {
    id: String,
    state: Cell<LifecycleState>,
    service_access: Cell<bool>,
    effects: RefCell<Vec<Disposal>>,
    observations: RefCell<Vec<Observation>>,
    activate_callbacks: RefCell<Vec<ActivationCallback>>,
}

/// One deferred keyed-observation start, upstream's observation thunk whose
/// result is the disposal that stops it.
type Observation = Box<dyn FnOnce() -> Result<Disposal, ChordError>>;

/// One activation hook, upstream's `() => void | Promise<void>` with the
/// failure the await propagates carried as the result.
pub type ActivationCallback = Box<dyn Fn() -> LocalBoxFuture<Result<(), ChordError>>>;

impl FacetLifecycle {
    fn assert_active(&self, operation: &str) -> Result<(), ChordError> {
        if self.state.get() != LifecycleState::Active {
            return Err(ChordError::Message(format!(
                "Facet {} can {operation} only while active",
                self.id
            )));
        }
        Ok(())
    }

    fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            state: Cell::new(LifecycleState::SettingUp),
            service_access: Cell::new(false),
            effects: RefCell::new(Vec::new()),
            observations: RefCell::new(Vec::new()),
            activate_callbacks: RefCell::new(Vec::new()),
        }
    }

    fn assert_setting_up(&self, operation: &str) -> Result<(), ChordError> {
        if self.state.get() != LifecycleState::SettingUp {
            return Err(ChordError::Message(format!(
                "Facet {} can {operation} only during setup",
                self.id
            )));
        }
        Ok(())
    }

    fn assert_running(&self, operation: &str) -> Result<(), ChordError> {
        let state = self.state.get();
        if state != LifecycleState::SettingUp && state != LifecycleState::Active {
            return Err(ChordError::Message(format!(
                "Facet {} cannot {operation} while {}",
                self.id,
                state_str(state)
            )));
        }
        Ok(())
    }

    fn assert_service_access(&self) -> Result<(), ChordError> {
        if !self.service_access.get() {
            return Err(ChordError::Message(format!(
                "Facet {} service handles cannot be used while {}",
                self.id,
                state_str(self.state.get())
            )));
        }
        Ok(())
    }

    fn revoke(&self) {
        self.service_access.set(false);
    }

    fn own(&self, disposal: Disposal) -> Result<(), ChordError> {
        self.assert_running("own resources")?;
        self.effects.borrow_mut().push(disposal);
        Ok(())
    }

    fn observe(&self, start: Observation) -> Result<(), ChordError> {
        self.assert_setting_up("observe services")?;
        self.observations.borrow_mut().push(start);
        Ok(())
    }

    fn on_activate(&self, callback: ActivationCallback) -> Result<(), ChordError> {
        self.assert_setting_up("register activation callbacks")?;
        self.activate_callbacks.borrow_mut().push(callback);
        Ok(())
    }

    fn prepared(&self) -> Result<(), ChordError> {
        self.assert_setting_up("finish setup")?;
        self.state.set(LifecycleState::Prepared);
        Ok(())
    }

    async fn activate(&self) -> Result<(), ChordError> {
        if self.state.get() != LifecycleState::Prepared {
            return Err(ChordError::Message(format!(
                "Facet {} is not prepared",
                self.id
            )));
        }
        self.state.set(LifecycleState::Active);
        self.service_access.set(true);
        let starts: Vec<Observation> = self.observations.borrow_mut().drain(..).collect();
        for start in starts {
            self.effects.borrow_mut().push(start()?);
        }
        let callbacks: Vec<ActivationCallback> =
            self.activate_callbacks.borrow_mut().drain(..).collect();
        for callback in callbacks {
            callback().await?;
        }
        Ok(())
    }

    async fn dispose(&self) -> Result<(), ChordError> {
        if self.state.get() == LifecycleState::Dead {
            return Ok(());
        }
        self.state.set(LifecycleState::Disposing);
        let mut errors = Vec::new();
        let effects: Vec<Disposal> = self.effects.borrow_mut().drain(..).rev().collect();
        for effect in effects {
            if let Err(error) = effect().await {
                errors.push(error);
            }
        }
        self.observations.borrow_mut().clear();
        self.activate_callbacks.borrow_mut().clear();
        self.service_access.set(false);
        self.state.set(LifecycleState::Dead);
        collect_errors(errors, format!("Failed to dispose facet {}", self.id)).map_or(Ok(()), Err)
    }
}

const fn state_str(state: LifecycleState) -> &'static str {
    match state {
        LifecycleState::SettingUp => "setting_up",
        LifecycleState::Prepared => "prepared",
        LifecycleState::Active => "active",
        LifecycleState::Disposing => "disposing",
        LifecycleState::Dead => "dead",
    }
}

/// The environment one facet's setup receives, upstream's
/// `FacetEnvironment`. Requirements and provisions recorded here shape the
/// facet graph the kernel validates before any handle is usable.
pub struct FacetEnvironment {
    runtime: Rc<RuntimeRecord>,
    slots: Rc<HostServiceSlots>,
}

impl std::fmt::Debug for FacetEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetEnvironment").finish_non_exhaustive()
    }
}

impl FacetEnvironment {
    /// Declares and installs this facet's singleton implementation of a
    /// service.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is no longer setting up.
    pub fn provide(
        &mut self,
        service: &Service,
        implementation: ServiceImplementation,
    ) -> Result<(), ChordError> {
        self.runtime
            .lifecycle
            .assert_setting_up("provide services")?;
        self.runtime
            .provides
            .borrow_mut()
            .push(FacetServiceReference {
                service_id: service.id.clone(),
                mode: ServiceMode::Singleton,
            });
        self.runtime
            .provisions
            .borrow_mut()
            .push(Provision::Singleton {
                service: service.clone(),
                implementation: Rc::new(implementation),
            });
        Ok(())
    }

    /// Declares ownership of a multi-instance service and returns its
    /// deferred spawning capability.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is no longer setting up.
    pub fn provide_many(
        &mut self,
        service: &Service,
    ) -> Result<Rc<StagedServiceSpawner>, ChordError> {
        self.runtime
            .lifecycle
            .assert_setting_up("provide service instances")?;
        self.runtime
            .provides
            .borrow_mut()
            .push(FacetServiceReference {
                service_id: service.id.clone(),
                mode: ServiceMode::Keyed,
            });
        let spawner = Rc::new(StagedServiceSpawner {
            lifecycle: self.runtime.lifecycle.clone(),
            service: service.clone(),
            instances: Rc::new(RefCell::new(Vec::new())),
            installer: Rc::new(RefCell::new(None)),
            connected: Cell::new(false),
        });
        self.runtime.provisions.borrow_mut().push(Provision::Keyed {
            service: service.clone(),
            spawner: spawner.clone(),
        });
        Ok(spawner)
    }

    /// Declares a hard dependency on one singleton service and returns its
    /// stable handle.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is no longer setting up.
    pub fn use_service(&mut self, service: &Service) -> Result<ServiceView, ChordError> {
        self.runtime
            .lifecycle
            .assert_setting_up("acquire services")?;
        self.runtime
            .requires
            .borrow_mut()
            .push(FacetServiceReference {
                service_id: service.id.clone(),
                mode: ServiceMode::Singleton,
            });
        if let Some(view) = self.runtime.singleton_views.borrow().get(&service.id) {
            return Ok(view.clone());
        }
        let lifecycle = self.runtime.lifecycle.clone();
        let slot = self.slots.get_singleton(service);
        let access: AssertAccess = Rc::new(move || lifecycle.assert_service_access());
        let view = slot.view(access);
        self.runtime
            .singleton_views
            .borrow_mut()
            .insert(service.id.clone(), view.clone());
        Ok(view)
    }

    /// Declares a hard dependency on a keyed service and observes each live
    /// instance when the facet activates.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is no longer setting up.
    pub fn observe_service(
        &mut self,
        service: &Service,
        handler: crate::types::KeyedViewHandler,
    ) -> Result<(), ChordError> {
        self.runtime
            .lifecycle
            .assert_setting_up("observe services")?;
        self.runtime
            .requires
            .borrow_mut()
            .push(FacetServiceReference {
                service_id: service.id.clone(),
                mode: ServiceMode::Keyed,
            });
        let slots = self.slots.clone();
        let lifecycle = self.runtime.lifecycle.clone();
        let service = service.clone();
        let observe: Observation = Box::new(move || {
            let stopped = Rc::new(Cell::new(false));
            let base_access: AssertAccess = {
                let lifecycle = lifecycle.clone();
                Rc::new(move || lifecycle.assert_service_access())
            };
            let stop = slots.observe(&service, base_access, handler)?;
            Ok(sync_disposal(move || {
                if stopped.get() {
                    return Ok(());
                }
                stopped.set(true);
                drop(stop());
                Ok(())
            }))
        });
        self.runtime.lifecycle.observe(observe)
    }

    /// Creates initialized mutable state suitable for exposing through a
    /// service implementation.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is neither setting up nor active.
    pub fn replicated_state(
        &mut self,
        initial: crate::types::JsonValue,
    ) -> Result<MutableReplicatedState, ChordError> {
        self.runtime
            .lifecycle
            .assert_running("create replicated state")?;
        Ok(MutableReplicatedState::new(initial))
    }

    /// Gives the facet ownership of a resource cleanup function.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is neither setting up nor active.
    pub fn own(&mut self, disposal: Disposal) -> Result<(), ChordError> {
        self.runtime.lifecycle.own(disposal)
    }

    /// Registers asynchronous initialization after dependencies are bound
    /// and ready.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is no longer setting up.
    pub fn on_activate(&mut self, callback: ActivationCallback) -> Result<(), ChordError> {
        self.runtime.lifecycle.on_activate(callback)
    }

    /// Registers final facet teardown.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is no longer setting up.
    pub fn on_deactivate(&mut self, callback: ActivationCallback) -> Result<(), ChordError> {
        self.runtime
            .lifecycle
            .own(Box::new(move || boxed(async move { callback().await })))
    }
}

/// The spawn capability `provide_many` returns, upstream's
/// `StagedServiceSpawner`: instances spawn during activation and connect to
/// their provider once the host assembles it.
pub struct StagedServiceSpawner {
    lifecycle: Rc<FacetLifecycle>,
    service: Service,
    instances: Rc<RefCell<Vec<StagedInstance>>>,
    installer: Rc<RefCell<Option<Installer>>>,
    connected: Cell<bool>,
}

type CloseInstance = Rc<dyn Fn() -> Result<(), ChordError>>;

/// The provider's per-instance install hook, upstream's installer closure:
/// key and implementation in, the close path out.
type Installer = Box<dyn Fn(&str, Rc<ServiceImplementation>) -> Result<CloseInstance, ChordError>>;

/// A staged reload generation, or the staged records to dispose when
/// staging fails.
type StagedGeneration = Result<Vec<Rc<RuntimeRecord>>, (ChordError, Vec<Rc<RuntimeRecord>>)>;

impl std::fmt::Debug for StagedServiceSpawner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedServiceSpawner")
            .field("service", &self.service)
            .field("connected", &self.connected.get())
            .finish_non_exhaustive()
    }
}

impl StagedServiceSpawner {
    /// Connects the installer; staged instances install immediately and
    /// later spawns install on arrival.
    ///
    /// # Errors
    /// [`ChordError`] when the spawner is already connected.
    pub fn connect(
        &self,
        installer: impl Fn(&str, Rc<ServiceImplementation>) -> Result<CloseInstance, ChordError>
        + 'static,
    ) -> Result<(), ChordError> {
        if self.connected.get() {
            return Err(ChordError::Message(
                "Facet service provider is already connected".to_string(),
            ));
        }
        self.connected.set(true);
        let installer = Rc::new(installer);
        let boxed_installer: Installer = {
            let installer = installer.clone();
            Box::new(move |key, implementation| installer(key, implementation))
        };
        *self.installer.borrow_mut() = Some(boxed_installer);
        let staged: Vec<(String, Rc<ServiceImplementation>)> = self
            .instances
            .borrow()
            .iter()
            .map(|instance| (instance.key.clone(), instance.implementation.clone()))
            .collect();
        for (key, implementation) in staged {
            let release = installer(&key, implementation)?;
            if let Some(instance) = self
                .instances
                .borrow_mut()
                .iter_mut()
                .find(|instance| instance.key == key)
            {
                *instance.release.borrow_mut() = Some(release);
            }
        }
        Ok(())
    }

    /// Spawns one keyed instance; only callable while the facet is active.
    /// The returned closure closes the instance, and the facet lifecycle
    /// owns the same close as a disposal.
    ///
    /// # Errors
    /// [`ChordError`] when the facet is not active, the key is empty or
    /// live, or the implementation is not exposable for a remote service.
    pub fn spawn(
        &self,
        key: &str,
        implementation: ServiceImplementation,
    ) -> Result<CloseInstance, ChordError> {
        self.lifecycle.assert_active("spawn service instances")?;
        if key.is_empty() {
            return Err(ChordError::Message(
                "Facet service instance key must not be empty".to_string(),
            ));
        }
        if !self.service.local {
            crate::handle::validate_remote_implementation(&self.service.id, &implementation)?;
        }
        if self
            .instances
            .borrow()
            .iter()
            .any(|instance| instance.key == key)
        {
            return Err(ChordError::Message(format!(
                "Facet service already has a live instance with key {key}"
            )));
        }
        let implementation = Rc::new(implementation);
        let release = {
            let installer = self.installer.borrow();
            match installer.as_ref() {
                Some(installer) => Some(installer(key, implementation.clone())?),
                None => None,
            }
        };
        self.instances.borrow_mut().push(StagedInstance {
            key: key.to_string(),
            implementation,
            release: RefCell::new(release),
        });
        let close: CloseInstance = {
            let instances = Rc::clone(&self.instances);
            let key = key.to_string();
            Rc::new(move || {
                let mut instances = instances.borrow_mut();
                let Some(at) = instances.iter().position(|instance| instance.key == key) else {
                    return Ok(());
                };
                let release = instances.remove(at).release;
                drop(instances);
                if let Some(release) = release.borrow_mut().take() {
                    release()?;
                }
                Ok(())
            })
        };
        let close_handle = close.clone();
        self.lifecycle.own(sync_disposal(move || close_handle()))?;
        Ok(close)
    }
}

struct StagedInstance {
    key: String,
    implementation: Rc<ServiceImplementation>,
    release: RefCell<Option<CloseInstance>>,
}

/// The keyed-instance source the host binds observation to, upstream's
/// `KeyedServiceSource`.
pub trait KeyedSource {
    /// Observes every live instance of one keyed service.
    ///
    /// # Errors
    /// [`ChordError`] when the source is disconnected or disposed.
    fn observe(
        &self,
        service: &Service,
        handler: KeyedServiceHandler,
    ) -> Result<Unsubscribe, ChordError>;
}

/// The registry of process-local keyed services, upstream's
/// `LocalKeyedServiceRegistry`.
pub struct LocalKeyedServiceRegistry {
    registrations: RefCell<Vec<(String, Rc<RefCell<LocalKeyedRegistration>>)>>,
    disposed: Cell<bool>,
}

impl std::fmt::Debug for LocalKeyedServiceRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalKeyedServiceRegistry")
            .field("disposed", &self.disposed.get())
            .finish_non_exhaustive()
    }
}

struct LocalKeyedRegistration {
    generations: HashMap<String, u64>,
    directory: InstanceDirectory,
}

impl LocalKeyedServiceRegistry {
    fn new(services: Vec<Service>, on_error: &ErrorReporter) -> Self {
        Self {
            registrations: RefCell::new(
                services
                    .into_iter()
                    .map(|service| {
                        (
                            service.id,
                            Rc::new(RefCell::new(LocalKeyedRegistration {
                                generations: HashMap::new(),
                                directory: InstanceDirectory::new(true, on_error.clone()),
                            })),
                        )
                    })
                    .collect(),
            ),
            disposed: Cell::new(false),
        }
    }

    /// Spawns one keyed instance and returns the closure that removes it.
    ///
    /// # Errors
    /// [`ChordError`] when the registry is disposed, the key is empty or
    /// live, or the service is unregistered.
    pub fn spawn(
        &self,
        service: &Service,
        key: &str,
        implementation: ServiceImplementation,
    ) -> Result<impl Fn() + 'static, ChordError> {
        self.assert_active()?;
        if key.is_empty() {
            return Err(ChordError::Message(
                "Local service instance key must not be empty".to_string(),
            ));
        }
        let registration = self.registration(&service.id)?;
        if registration.borrow().directory.get(key).is_some() {
            return Err(ChordError::Message(format!(
                "Local service {} already has a live instance with key {key}",
                service.id
            )));
        }
        let generation = registration
            .borrow()
            .generations
            .get(key)
            .copied()
            .unwrap_or(0)
            + 1;
        registration
            .borrow_mut()
            .generations
            .insert(key.to_string(), generation);
        let entry = Rc::new(InstanceDirectoryEntry {
            key: key.to_string(),
            generation,
            service: ServiceTarget::Local(Rc::new(implementation)),
            deactivate: Rc::new(|| {}),
        });
        registration.borrow().directory.insert(entry.clone())?;
        Ok(move || {
            registration.borrow().directory.remove(&entry);
        })
    }

    fn registration(
        &self,
        service_id: &str,
    ) -> Result<Rc<RefCell<LocalKeyedRegistration>>, ChordError> {
        self.registrations
            .borrow()
            .iter()
            .find(|(id, _)| id == service_id)
            .map(|(_, registration)| registration.clone())
            .ok_or_else(|| {
                ChordError::Message(format!(
                    "Local keyed service {service_id} is not registered"
                ))
            })
    }

    fn assert_active(&self) -> Result<(), ChordError> {
        if self.disposed.get() {
            return Err(ChordError::Message(
                "Local keyed service registry is disposed".to_string(),
            ));
        }
        Ok(())
    }

    fn dispose(&self) {
        if self.disposed.get() {
            return;
        }
        self.disposed.set(true);
        for (_, registration) in self.registrations.borrow().iter() {
            registration.borrow().directory.dispose();
        }
        self.registrations.borrow_mut().clear();
    }
}

impl KeyedSource for LocalKeyedServiceRegistry {
    fn observe(
        &self,
        service: &Service,
        handler: KeyedServiceHandler,
    ) -> Result<Unsubscribe, ChordError> {
        self.assert_active()?;
        let registration = self.registration(&service.id)?;
        registration.borrow().directory.observe(handler)
    }
}

/// Host-owned singleton slots and keyed sources, upstream's
/// `HostServiceSlots`.
pub struct HostServiceSlots {
    singletons: RefCell<HashMap<String, (ServiceSlot, Rc<AssertAccess>)>>,
    keyed_sources: RefCell<HashMap<String, Rc<dyn KeyedSource>>>,
}

impl std::fmt::Debug for HostServiceSlots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostServiceSlots").finish_non_exhaustive()
    }
}

impl HostServiceSlots {
    fn new() -> Self {
        Self {
            singletons: RefCell::new(HashMap::new()),
            keyed_sources: RefCell::new(HashMap::new()),
        }
    }

    fn get_singleton(&self, service: &Service) -> ServiceSlot {
        let mut singletons = self.singletons.borrow_mut();
        singletons
            .entry(service.id.clone())
            .or_insert_with(|| (ServiceSlot::new(&service.id), Rc::new(allow_access())))
            .0
            .clone()
    }

    fn has_singleton(&self, service_id: &str) -> bool {
        self.singletons.borrow().contains_key(service_id)
    }

    fn bind_singleton(&self, service_id: &str, target: ServiceTarget) {
        if let Some((slot, _)) = self.singletons.borrow().get(service_id) {
            slot.bind(target);
        }
    }

    fn bind_keyed(&self, service_id: &str, source: Rc<dyn KeyedSource>) {
        self.keyed_sources
            .borrow_mut()
            .insert(service_id.to_string(), source);
    }

    fn observe(
        &self,
        service: &Service,
        assert_access: AssertAccess,
        handler: crate::types::KeyedViewHandler,
    ) -> Result<Disposal, ChordError> {
        let source = self
            .keyed_sources
            .borrow()
            .get(&service.id)
            .cloned()
            .ok_or_else(|| {
                ChordError::Message(format!("Service {} is disconnected", service.id))
            })?;
        let stopped = Rc::new(Cell::new(false));
        let wrapped: KeyedServiceHandler = {
            let service_id = Rc::new(service.id.clone());
            let stopped = stopped.clone();
            Box::new(move |target, context| {
                let slot = ServiceSlot::new(service_id.as_str());
                slot.bind(target);
                let assert: AssertAccess = {
                    let access = assert_access.clone();
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
                            return Err(ChordError::Message(format!(
                                "Keyed service {service_id} observation is closed"
                            )));
                        }
                        Ok(())
                    })
                };
                let view = slot.view(assert);
                (handler)(view, context);
            })
        };
        let stop = source.observe(service, wrapped)?;
        Ok(Box::new(move || {
            if stopped.get() {
                return sync_disposal(|| Ok(()))();
            }
            stopped.set(true);
            stop();
            sync_disposal(|| Ok(()))()
        }))
    }

    fn dispose(&self) {
        for (slot, _) in self.singletons.borrow().values() {
            slot.unbind();
        }
        self.singletons.borrow_mut().clear();
        self.keyed_sources.borrow_mut().clear();
    }
}

// The kernel lands in this module's second half; see FacetKernel below.

/// The phase one facet generation is in, upstream's `GenerationPhase`. The
/// phase gates when service targets may be used and when a host may reload
/// or die.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationPhase {
    /// Facet setups have not run yet.
    Setup,
    /// The graph is validated and providers are assembling.
    Assembling,
    /// Source bindings are installing their initial snapshots.
    Connecting,
    /// Facets are activating in topological order.
    Activating,
    /// The generation is live; reload and dispose are legal.
    Active,
    /// A replacement generation is staging and cutting over.
    Reloading,
    /// Termination has started; only cleanup runs.
    Disposing,
    /// The host is terminated.
    Dead,
}

/// The spelling a phase carries in host error messages, upstream's string
/// interpolation over `GenerationPhase`.
#[must_use]
pub const fn phase_str(phase: GenerationPhase) -> &'static str {
    match phase {
        GenerationPhase::Setup => "setup",
        GenerationPhase::Assembling => "assembling",
        GenerationPhase::Connecting => "connecting",
        GenerationPhase::Activating => "activating",
        GenerationPhase::Active => "active",
        GenerationPhase::Reloading => "reloading",
        GenerationPhase::Disposing => "disposing",
        GenerationPhase::Dead => "dead",
    }
}

/// The options one kernel generation starts from, upstream's
/// `FacetOptions`.
pub struct FacetKernelOptions {
    /// The facets of the initial generation.
    pub facets: Vec<FacetDef>,
    /// The external service sources requirements may bind to.
    pub service_sources: Vec<Rc<dyn RemoteServiceSource>>,
    /// The reporter for failures outside a caller's control.
    pub on_error: ErrorReporter,
}

impl std::fmt::Debug for FacetKernelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetKernelOptions")
            .field("facets", &self.facets)
            .finish_non_exhaustive()
    }
}

/// Private lifecycle and dependency kernel behind the atomic host entry
/// point, upstream's `FacetKernel`.
pub struct FacetKernel {
    core: Rc<KernelCore>,
}

impl std::fmt::Debug for FacetKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetKernel").finish_non_exhaustive()
    }
}

/// One external service a source offers to the host.
struct ExternalService {
    service: Service,
    mode: ServiceMode,
    source_index: usize,
}

struct KernelCore {
    initial_facets: RefCell<Vec<FacetDef>>,
    service_sources: Vec<Rc<dyn RemoteServiceSource>>,
    on_error: ErrorReporter,
    slots: Rc<HostServiceSlots>,
    facets: RefCell<Vec<(String, Rc<RuntimeRecord>)>>,
    activation_order: RefCell<Vec<String>>,
    source_bindings: RefCell<Vec<(usize, Rc<dyn RemoteServices>)>>,
    provider: RefCell<Option<RemoteServiceProvider>>,
    internal_services: RefCell<Option<Rc<RemoteServiceBinding>>>,
    local_keyed_services: RefCell<Option<Rc<LocalKeyedServiceRegistry>>>,
    phase: Cell<GenerationPhase>,
}

impl FacetKernel {
    /// Builds the kernel for one generation's options.
    ///
    /// # Errors
    /// [`ChordError`] when a facet ID is empty or the IDs are not unique
    /// within the generation.
    pub fn new(options: FacetKernelOptions) -> Result<Self, ChordError> {
        let mut ids = std::collections::HashSet::new();
        for facet in &options.facets {
            if facet.id.is_empty() {
                return Err(ChordError::Message(
                    "Facet ID must not be empty".to_string(),
                ));
            }
            if !ids.insert(facet.id.clone()) {
                return Err(ChordError::Message(
                    "Facet IDs must be unique within a generation".to_string(),
                ));
            }
        }
        Ok(Self {
            core: Rc::new(KernelCore {
                initial_facets: RefCell::new(options.facets),
                service_sources: options.service_sources,
                on_error: options.on_error,
                slots: Rc::new(HostServiceSlots::new()),
                facets: RefCell::new(Vec::new()),
                activation_order: RefCell::new(Vec::new()),
                source_bindings: RefCell::new(Vec::new()),
                provider: RefCell::new(None),
                internal_services: RefCell::new(None),
                local_keyed_services: RefCell::new(None),
                phase: Cell::new(GenerationPhase::Setup),
            }),
        })
    }

    fn create_runtime(_core: &Rc<KernelCore>, facet_id: &str) -> Rc<RuntimeRecord> {
        Rc::new(RuntimeRecord {
            facet_id: facet_id.to_string(),
            requires: RefCell::new(Vec::new()),
            provides: RefCell::new(Vec::new()),
            provisions: RefCell::new(Vec::new()),
            singleton_views: RefCell::new(HashMap::new()),
            lifecycle: Rc::new(FacetLifecycle::new(facet_id)),
        })
    }

    fn setup_facet(
        core: &Rc<KernelCore>,
        facet: &FacetDef,
        record: &Rc<RuntimeRecord>,
    ) -> Result<(), ChordError> {
        let mut environment = FacetEnvironment {
            runtime: record.clone(),
            slots: core.slots.clone(),
        };
        (facet.setup)(&mut environment);
        record.lifecycle.prepared()
    }

    /// The assembled provider, available between assembling and death.
    ///
    /// # Errors
    /// [`ChordError`] when the provider is not assembled.
    pub fn provider(&self) -> Result<RemoteServiceProvider, ChordError> {
        self.core.provider.borrow().clone().ok_or_else(|| {
            ChordError::Message("Facet service provider is not assembled".to_string())
        })
    }
}

impl FacetKernel {
    /// Activates the initial generation: setups run, the facet graph is
    /// validated, providers assemble, source bindings install their initial
    /// snapshots, and facets activate in dependency order. Any failure
    /// terminates the host and aggregates the cleanup failures into the
    /// report.
    ///
    /// # Errors
    /// [`ChordError`] when any activation stage fails; cleanup failures
    /// aggregate into the report.
    pub async fn activate(&self) -> Result<(), ChordError> {
        match self.activate_inner().await {
            Ok(()) => Ok(()),
            Err(error) => {
                let cleanup_errors = self.terminate(&[]).await;
                match cleanup_errors.len() {
                    0 => Err(error),
                    _ => Err(ChordError::Aggregate(
                        std::iter::once(error).chain(cleanup_errors).collect(),
                        "Facet generation startup and cleanup failed".to_string(),
                    )),
                }
            }
        }
    }

    async fn activate_inner(&self) -> Result<(), ChordError> {
        let facets: Vec<FacetDef> = self.core.initial_facets.borrow_mut().drain(..).collect();
        for facet in facets {
            let record = Self::create_runtime(&self.core, &facet.id);
            self.core
                .facets
                .borrow_mut()
                .push((facet.id.clone(), record.clone()));
            Self::setup_facet(&self.core, &facet, &record)?;
        }
        self.core.phase.set(GenerationPhase::Assembling);
        let external = self.resolve_external_services().await?;
        let order = validate_facets(&self.core.facets.borrow(), &external)?;
        *self.core.activation_order.borrow_mut() = order;
        self.assemble_providers()?;
        self.bind_services(&external)?;

        self.core.phase.set(GenerationPhase::Connecting);
        let mut readys: Vec<LocalBoxFuture<Result<(), ChordError>>> = Vec::new();
        for (_, services) in self.core.source_bindings.borrow().iter() {
            readys.push(services.ready(background_context()));
        }
        if let Some(internal) = self.core.internal_services.borrow().clone() {
            readys.push(internal.ready(background_context()));
        }
        for result in join_all(readys).await {
            result?;
        }

        self.core.phase.set(GenerationPhase::Activating);
        let order = self.core.activation_order.borrow().clone();
        for id in order {
            let record = self
                .core
                .facets
                .borrow()
                .iter()
                .find(|(facet_id, _)| *facet_id == id)
                .map(|(_, record)| record.clone());
            if let Some(record) = record {
                record.lifecycle.activate().await?;
            }
        }
        self.core.phase.set(GenerationPhase::Active);
        Ok(())
    }

    /// Activates and replaces facets with matching IDs without disconnecting
    /// consumer service handles.
    ///
    /// The replacement stages and validates before cutover; a failure before
    /// cutover keeps the old generation live, and a failure after cutover
    /// aborts the host.
    ///
    /// # Errors
    /// [`ChordError`] when a facet is unknown, staging or activation fails,
    /// or cutover aborts the host; cleanup failures aggregate into the
    /// report.
    /// Swaps the staged candidates into the facet map and returns the
    /// records they replace, upstream's cutover prelude.
    fn install_candidates(&self, candidate_order: &[Rc<RuntimeRecord>]) -> Vec<Rc<RuntimeRecord>> {
        let previous: Vec<Rc<RuntimeRecord>> = candidate_order
            .iter()
            .filter_map(|candidate| {
                self.core
                    .facets
                    .borrow()
                    .iter()
                    .find(|(id, _)| id == &candidate.facet_id)
                    .map(|(_, record)| record.clone())
            })
            .collect();
        for candidate in candidate_order {
            let mut facets = self.core.facets.borrow_mut();
            if let Some(slot) = facets.iter_mut().find(|(id, _)| id == &candidate.facet_id) {
                slot.1 = candidate.clone();
            }
        }
        previous
    }

    /// Validates a reload's preconditions, upstream's reload prelude.
    fn check_reload_requirements(&self, facets: &[FacetDef]) -> Result<(), ChordError> {
        let mut ids = std::collections::HashSet::new();
        for facet in facets {
            if facet.id.is_empty() {
                return Err(ChordError::Message(
                    "Facet ID must not be empty".to_string(),
                ));
            }
            if !ids.insert(facet.id.clone()) {
                return Err(ChordError::Message(
                    "Reloaded facet IDs must be unique".to_string(),
                ));
            }
        }
        for facet in facets {
            if !self
                .core
                .facets
                .borrow()
                .iter()
                .any(|(id, _)| id == &facet.id)
            {
                return Err(ChordError::Message(format!(
                    "Facet {} is not active",
                    facet.id
                )));
            }
        }
        Ok(())
    }

    /// Activates and replaces facets with matching IDs without disconnecting
    /// consumer service handles.
    ///
    /// # Errors
    /// [`ChordError`] when the host is not active, a reloaded facet is
    /// unknown, shapes change, or any stage of the reload fails; cleanup
    /// failures aggregate into the report.
    pub async fn reload(&self, facets: Vec<FacetDef>) -> Result<(), ChordError> {
        if self.core.phase.get() != GenerationPhase::Active {
            return Err(ChordError::Message(format!(
                "Facet host cannot reload while {}",
                phase_str(self.core.phase.get())
            )));
        }
        self.check_reload_requirements(&facets)?;
        self.core.phase.set(GenerationPhase::Reloading);

        let staged = match self.stage_generation(facets) {
            Ok(staged) => staged,
            Err((error, staged)) => {
                return self
                    .dispose_failed_stage(error, staged, "Facet reload setup and cleanup failed")
                    .await;
            }
        };

        let replacements: HashMap<String, Rc<RuntimeRecord>> = staged
            .iter()
            .map(|record| (record.facet_id.clone(), record.clone()))
            .collect();
        let candidate_order: Vec<Rc<RuntimeRecord>> = self
            .core
            .activation_order
            .borrow()
            .iter()
            .filter_map(|id| replacements.get(id).cloned())
            .collect();

        let activation_result = self.activate_candidates(&candidate_order).await;
        if let Err(error) = activation_result {
            let cleanup_errors =
                dispose_records(&candidate_order.iter().rev().cloned().collect::<Vec<_>>()).await;
            if cleanup_errors.is_empty() {
                self.core.phase.set(GenerationPhase::Active);
                return Err(error);
            }
            let abort_errors = self.abort(&[]).await;
            return Err(ChordError::Aggregate(
                std::iter::once(error)
                    .chain(cleanup_errors)
                    .chain(abort_errors)
                    .collect(),
                "Facet reload activation and cleanup failed".to_string(),
            ));
        }

        let previous = self.install_candidates(&candidate_order);
        match self.cut_over(&candidate_order).await {
            Ok(()) => {}
            Err(error) => {
                let abort_errors = self.abort(&previous).await;
                return Err(ChordError::Aggregate(
                    std::iter::once(error).chain(abort_errors).collect(),
                    "Facet reload failed after cutover".to_string(),
                ));
            }
        }
        self.core.phase.set(GenerationPhase::Active);
        Ok(())
    }

    /// Tears the host down: every lifecycle disposes in reverse activation
    /// order, then the keyed registry, source bindings, slots, and provider.
    ///
    /// # Errors
    /// [`ChordError`] aggregating the failures collected along the way.
    pub async fn dispose(&self) -> Result<(), ChordError> {
        if self.core.phase.get() == GenerationPhase::Dead {
            return Ok(());
        }
        if self.core.phase.get() != GenerationPhase::Active {
            return Err(ChordError::Message(format!(
                "Facet host cannot be disposed while {}",
                phase_str(self.core.phase.get())
            )));
        }
        let errors = self.terminate(&[]).await;
        collect_errors(errors, "Failed to dispose facet generation").map_or(Ok(()), Err)
    }

    /// Disposes a failed reload stage in reverse order and folds the cleanup
    /// failures into the reported error.
    async fn dispose_failed_stage(
        &self,
        error: ChordError,
        staged: Vec<Rc<RuntimeRecord>>,
        label: &str,
    ) -> Result<(), ChordError> {
        let mut reversed = staged;
        reversed.reverse();
        let cleanup_errors = dispose_records(&reversed).await;
        if cleanup_errors.is_empty() {
            self.core.phase.set(GenerationPhase::Active);
            return Err(error);
        }
        let abort_errors = self.abort(&[]).await;
        Err(ChordError::Aggregate(
            std::iter::once(error)
                .chain(cleanup_errors)
                .chain(abort_errors)
                .collect(),
            label.to_string(),
        ))
    }

    /// Activates the staged candidates in dependency order and re-validates
    /// their provisions, upstream's reload activation phase.
    async fn activate_candidates(
        &self,
        candidate_order: &[Rc<RuntimeRecord>],
    ) -> Result<(), ChordError> {
        for candidate in candidate_order {
            candidate.lifecycle.activate().await?;
        }
        for candidate in candidate_order {
            self.validate_replacement_provisions(candidate)?;
        }
        Ok(())
    }

    fn stage_generation(&self, facets: Vec<FacetDef>) -> StagedGeneration {
        let mut staged: Vec<Rc<RuntimeRecord>> = Vec::new();
        for facet in facets {
            let record = Self::create_runtime(&self.core, &facet.id);
            staged.push(record.clone());
            if let Err(error) = Self::setup_facet(&self.core, &facet, &record) {
                return Err((error, staged));
            }
            #[allow(
                clippy::expect_used,
                reason = "reload checks every facet is active before staging"
            )]
            let previous = self
                .core
                .facets
                .borrow()
                .iter()
                .find(|(id, _)| id == &record.facet_id)
                .map(|(_, record)| record.clone())
                .expect("reload checks the facet is active before staging");
            if !same_facet_shape(&previous, &record) {
                return Err((
                    ChordError::Message(format!(
                        "Reloaded facet {} must preserve its service requirements and provisions",
                        record.facet_id
                    )),
                    staged,
                ));
            }
            if let Err(error) = self.validate_replacement_provisions(&record) {
                return Err((error, staged));
            }
        }
        Ok(staged)
    }

    async fn cut_over(&self, candidate_order: &[Rc<RuntimeRecord>]) -> Result<(), ChordError> {
        for candidate in candidate_order {
            for provision in candidate.provisions.borrow().iter() {
                let Provision::Singleton {
                    service,
                    implementation,
                } = provision
                else {
                    continue;
                };
                if service.local {
                    self.core
                        .slots
                        .bind_singleton(&service.id, ServiceTarget::Local(implementation.clone()));
                } else {
                    self.provider()?
                        .replace(service, implementation.as_ref().clone())?;
                }
            }
        }
        let previous: Vec<Rc<RuntimeRecord>> = candidate_order
            .iter()
            .filter_map(|candidate| {
                self.core
                    .facets
                    .borrow()
                    .iter()
                    .find(|(id, _)| id == &candidate.facet_id)
                    .map(|(_, record)| record.clone())
            })
            .collect();
        let retirement_errors = dispose_records(&previous).await;
        if let Some(error) = collect_errors(retirement_errors, "Failed to retire replaced facets") {
            return Err(error);
        }
        for candidate in candidate_order {
            for provision in candidate.provisions.borrow().iter() {
                let Provision::Keyed { service, spawner } = provision else {
                    continue;
                };
                if service.local {
                    let registry =
                        self.core
                            .local_keyed_services
                            .borrow()
                            .clone()
                            .ok_or_else(|| {
                                ChordError::Message(
                                    "Facet keyed services are not assembled".to_string(),
                                )
                            })?;
                    spawner.connect({
                        let registry = registry.clone();
                        let service = service.clone();
                        move |key, implementation| {
                            let close =
                                registry.spawn(&service, key, implementation.as_ref().clone())?;
                            Ok(Rc::new(move || {
                                close();
                                Ok(())
                            }))
                        }
                    })?;
                } else {
                    let provider = self.provider()?;
                    spawner.connect({
                        let provider = provider.clone();
                        let service = service.clone();
                        move |key, implementation| {
                            let close = provider.spawn(&service, key, (*implementation).clone())?;
                            Ok(Rc::new(close))
                        }
                    })?;
                }
            }
        }
        Ok(())
    }

    fn validate_replacement_provisions(
        &self,
        record: &Rc<RuntimeRecord>,
    ) -> Result<(), ChordError> {
        for provision in record.provisions.borrow().iter() {
            let Provision::Singleton {
                service,
                implementation,
            } = provision
            else {
                continue;
            };
            if service.local {
                continue;
            }
            self.provider()?
                .validate_replacement(service, implementation)?;
        }
        Ok(())
    }

    async fn resolve_external_services(
        &self,
    ) -> Result<HashMap<String, ExternalService>, ChordError> {
        let sources = self.core.service_sources.clone();
        let offered: Vec<(String, ServiceMode, usize)> = collect_offerings(&sources).await?;
        let local: std::collections::HashSet<String> = self
            .core
            .facets
            .borrow()
            .iter()
            .flat_map(|(_, record)| {
                record
                    .provides
                    .borrow()
                    .iter()
                    .map(|reference| reference.service_id.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut external: HashMap<String, ExternalService> = HashMap::new();
        for (_, record) in self.core.facets.borrow().iter() {
            for requirement in record.requires.borrow().iter() {
                if local.contains(&requirement.service_id)
                    || external.contains_key(&requirement.service_id)
                {
                    continue;
                }
                let found = offered
                    .iter()
                    .enumerate()
                    .find(|(_, (service_id, _, _))| *service_id == requirement.service_id)
                    .map(|(_, (service_id, mode, index))| (service_id.clone(), *mode, *index));
                let found = requirement_source(&self.core, requirement, found)?;
                if let Some((service_id, mode, source_index)) = found {
                    external.insert(
                        service_id.clone(),
                        ExternalService {
                            service: Service {
                                id: service_id,
                                local: false,
                            },
                            mode,
                            source_index,
                        },
                    );
                }
            }
        }

        let mut service_ids_by_source: HashMap<usize, Vec<Service>> = HashMap::new();
        for (service_id, external) in &external {
            let services = service_ids_by_source
                .entry(external.source_index)
                .or_default();
            services.push(Service {
                id: service_id.clone(),
                local: false,
            });
        }
        let mut opened: Vec<(usize, Rc<dyn RemoteServices>)> = Vec::new();
        for (source_index, services) in service_ids_by_source {
            let source = self.core.service_sources[source_index].clone();
            let binding = source.open(crate::types::RemoteServiceSourceOpenOptions {
                services,
                assert_access: {
                    let core = self.core.clone();
                    Rc::new(move || assert_service_target_access(&core))
                },
                on_error: self.core.on_error.clone(),
            });
            self.core
                .source_bindings
                .borrow_mut()
                .push((source_index, binding.clone()));
            opened.push((source_index, binding));
        }
        let _ = opened;
        Ok(external)
    }

    fn assemble_providers(&self) -> Result<(), ChordError> {
        let provisions = self.provisions();
        let remote_provisions: Vec<&Provision> = provisions
            .iter()
            .filter(|provision| !provision.service().local)
            .collect();
        let definitions: Vec<crate::services::provider::ServiceProviderDefinition> =
            remote_provisions
                .iter()
                .map(
                    |provision| crate::services::provider::ServiceProviderDefinition {
                        service: provision.service().clone(),
                        mode: match provision {
                            Provision::Singleton { .. } => ServiceMode::Singleton,
                            Provision::Keyed { .. } => ServiceMode::Keyed,
                        },
                    },
                )
                .collect();
        let provider = RemoteServiceProvider::new(definitions)?;
        let transport = crate::services::loopback::create_loopback_service_transport(&provider);
        let internal_services = create_remote_service_binding(RemoteServiceBindingOptions {
            services: remote_provisions
                .iter()
                .map(|provision| provision.service().clone())
                .collect(),
            transport,
            bound: true,
            on_error: self.core.on_error.clone(),
            assert_access: None,
        })
        .map(Rc::new)?;
        let local_keyed_services = Rc::new(LocalKeyedServiceRegistry::new(
            provisions
                .iter()
                .filter_map(|provision| match provision {
                    Provision::Keyed { service, .. } if service.local => Some(service.clone()),
                    _ => None,
                })
                .collect(),
            &self.core.on_error,
        ));
        for provision in &provisions {
            match provision {
                Provision::Singleton {
                    service,
                    implementation,
                } => {
                    if !service.local {
                        provider.provide(service, implementation.as_ref().clone())?;
                    }
                }
                Provision::Keyed { service, spawner } => {
                    if service.local {
                        let registry =
                            self.core
                                .local_keyed_services
                                .borrow()
                                .clone()
                                .ok_or_else(|| {
                                    ChordError::Message(
                                        "Facet keyed services are not assembled".to_string(),
                                    )
                                })?;
                        spawner.connect({
                            let registry = registry.clone();
                            let service = service.clone();
                            move |key, implementation| {
                                let close =
                                    registry.spawn(&service, key, (*implementation).clone())?;
                                Ok(Rc::new(move || {
                                    close();
                                    Ok(())
                                }))
                            }
                        })?;
                    } else {
                        let provider = provider.clone();
                        let service = service.clone();
                        spawner.connect(move |key, implementation| {
                            let close = provider.spawn(&service, key, (*implementation).clone())?;
                            Ok(Rc::new(close))
                        })?;
                    }
                }
            }
        }
        *self.core.provider.borrow_mut() = Some(provider);
        *self.core.internal_services.borrow_mut() = Some(internal_services);
        *self.core.local_keyed_services.borrow_mut() = Some(local_keyed_services);
        Ok(())
    }

    fn bind_services(&self, external: &HashMap<String, ExternalService>) -> Result<(), ChordError> {
        for provision in self.provisions() {
            match provision {
                Provision::Singleton {
                    service,
                    implementation,
                } => {
                    if !self.core.slots.has_singleton(&service.id) {
                        continue;
                    }
                    let target = if service.local {
                        ServiceTarget::Local(implementation.clone())
                    } else {
                        let internal =
                            self.core
                                .internal_services
                                .borrow()
                                .clone()
                                .ok_or_else(|| {
                                    ChordError::Message(
                                        "Facet remote services are not assembled".to_string(),
                                    )
                                })?;
                        ServiceTarget::View(internal.use_service(&service)?)
                    };
                    self.core.slots.bind_singleton(&service.id, target);
                }
                Provision::Keyed { service, .. } => {
                    let source: Rc<dyn KeyedSource> =
                        if service.local {
                            let registry =
                                self.core.local_keyed_services.borrow().clone().ok_or_else(
                                    || {
                                        ChordError::Message(
                                            "Facet keyed services are not assembled".to_string(),
                                        )
                                    },
                                )?;
                            Rc::new(RegistryKeyedSource(registry))
                        } else {
                            let internal = self
                                .core
                                .internal_services
                                .borrow()
                                .clone()
                                .ok_or_else(|| {
                                    ChordError::Message(
                                        "Facet remote services are not assembled".to_string(),
                                    )
                                })?;
                            Rc::new(SourceKeyedSource(internal))
                        };
                    self.core.slots.bind_keyed(&service.id, source);
                }
            }
        }
        for (service_id, external) in external {
            let services = self
                .core
                .source_bindings
                .borrow()
                .iter()
                .find(|(index, _)| *index == external.source_index)
                .map(|(_, services)| services.clone())
                .ok_or_else(|| {
                    ChordError::Message(format!("Service source for {service_id} is not open"))
                })?;
            if external.mode == ServiceMode::Singleton {
                self.core.slots.bind_singleton(
                    service_id,
                    ServiceTarget::View(services.use_service(&external.service)?),
                );
            } else {
                let source = SourceKeyedSource(services.clone());
                self.core.slots.bind_keyed(service_id, Rc::new(source));
            }
        }
        Ok(())
    }

    fn provisions(&self) -> Vec<Provision> {
        self.core
            .facets
            .borrow()
            .iter()
            .flat_map(|(_, record)| record.provisions.borrow().clone())
            .collect()
    }

    async fn abort(&self, extra_records: &[Rc<RuntimeRecord>]) -> Vec<ChordError> {
        for (_, record) in self.core.facets.borrow().iter() {
            record.lifecycle.revoke();
        }
        for record in extra_records {
            record.lifecycle.revoke();
        }
        self.terminate(extra_records).await
    }

    async fn terminate(&self, extra_records: &[Rc<RuntimeRecord>]) -> Vec<ChordError> {
        self.core.phase.set(GenerationPhase::Disposing);
        let mut errors = self.dispose_lifecycles().await;
        let mut extras: Vec<Rc<RuntimeRecord>> = extra_records.to_vec();
        extras.reverse();
        errors.extend(dispose_records(&extras).await);
        if let Some(registry) = self.core.local_keyed_services.borrow_mut().take() {
            registry.dispose();
        }
        let bindings: Vec<Rc<dyn RemoteServices>> = {
            let mut bindings = self
                .core
                .source_bindings
                .borrow_mut()
                .drain(..)
                .map(|(_, services)| services)
                .collect::<Vec<_>>();
            if let Some(internal) = self.core.internal_services.borrow_mut().take() {
                bindings.push(internal);
            }
            bindings
        };
        for services in bindings {
            if let Err(error) = services.dispose(background_context()).await {
                errors.push(error);
            }
        }
        self.core.slots.dispose();
        if let Some(provider) = self.core.provider.borrow_mut().take()
            && let Err(error) = provider.dispose()
        {
            errors.push(error);
        }
        self.core.phase.set(GenerationPhase::Dead);
        errors
    }

    async fn dispose_lifecycles(&self) -> Vec<ChordError> {
        let order: Vec<String> = if self.core.activation_order.borrow().is_empty() {
            self.core
                .facets
                .borrow()
                .iter()
                .map(|(id, _)| id.clone())
                .rev()
                .collect()
        } else {
            self.core
                .activation_order
                .borrow()
                .iter()
                .rev()
                .cloned()
                .collect()
        };
        let mut errors = Vec::new();
        for id in order {
            let record = self
                .core
                .facets
                .borrow()
                .iter()
                .find(|(facet_id, _)| *facet_id == id)
                .map(|(_, record)| record.clone());
            let Some(record) = record else {
                continue;
            };
            self.core
                .facets
                .borrow_mut()
                .retain(|(facet_id, _)| facet_id != &id);
            let errors_at = record.lifecycle.dispose().await;
            if let Err(error) = errors_at {
                errors.push(error);
            }
        }
        errors
    }
}

async fn dispose_records(records: &[Rc<RuntimeRecord>]) -> Vec<ChordError> {
    let mut errors = Vec::new();
    for record in records {
        if let Err(error) = record.lifecycle.dispose().await {
            errors.push(error);
        }
    }
    errors
}

fn same_facet_shape(left: &RuntimeRecord, right: &RuntimeRecord) -> bool {
    same_references(&left.requires.borrow(), &right.requires.borrow())
        && same_references(&left.provides.borrow(), &right.provides.borrow())
}

fn same_references(left: &[FacetServiceReference], right: &[FacetServiceReference]) -> bool {
    left.len() == right.len()
        && left.iter().all(|reference| {
            right.iter().any(|other| {
                other.service_id == reference.service_id && other.mode == reference.mode
            })
        })
}

/// The keyed-source adapter over the local registry: its directory hands
/// raw implementation targets straight through.
struct RegistryKeyedSource(Rc<LocalKeyedServiceRegistry>);

impl KeyedSource for RegistryKeyedSource {
    fn observe(
        &self,
        service: &Service,
        handler: KeyedServiceHandler,
    ) -> Result<Unsubscribe, ChordError> {
        LocalKeyedServiceRegistry::observe(&self.0, service, handler)
    }
}

/// The keyed-source adapter over any [`RemoteServices`] surface: the view
/// its observe hands back rides to the caller as a
/// [`ServiceTarget::View`], keeping the source's access gate in the chain.
struct SourceKeyedSource(Rc<dyn RemoteServices>);

impl KeyedSource for SourceKeyedSource {
    fn observe(
        &self,
        service: &Service,
        handler: KeyedServiceHandler,
    ) -> Result<Unsubscribe, ChordError> {
        let handler = Rc::new(handler);
        let wrapped: crate::types::KeyedViewHandler =
            Rc::new(move |view, context| handler(ServiceTarget::View(view), context));
        self.0.observe(service, wrapped)
    }
}

/// Joins every source's catalogue into the offerings list, upstream's
/// catalogue `Promise.all`. Indexes track the source each entry came from.
async fn collect_offerings(
    sources: &[Rc<dyn RemoteServiceSource>],
) -> Result<Vec<(String, ServiceMode, usize)>, ChordError> {
    let mut indexed: Vec<(usize, Rc<dyn RemoteServiceSource>)> = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        indexed.push((index, source.clone()));
    }
    let catalogues: Vec<Result<(usize, Vec<ServiceCatalogueEntry>), ChordError>> = join_all(
        indexed
            .into_iter()
            .map(|(index, source)| {
                boxed(async move {
                    let entries = source.catalogue(background_context()).await?;
                    Ok((index, entries))
                })
            })
            .collect(),
    )
    .await;
    let mut offered: Vec<(String, ServiceMode, usize)> = Vec::new();
    for result in catalogues {
        let (index, entries) = result?;
        for entry in entries {
            if offered.iter().any(|(id, _, _)| *id == entry.service_id) {
                return Err(ChordError::Message(format!(
                    "Facet host service {} is offered by more than one source",
                    entry.service_id
                )));
            }
            offered.push((entry.service_id, entry.mode, index));
        }
    }
    Ok(offered)
}

fn assert_service_target_access(core: &KernelCore) -> Result<(), ChordError> {
    if !matches!(
        core.phase.get(),
        GenerationPhase::Activating
            | GenerationPhase::Active
            | GenerationPhase::Reloading
            | GenerationPhase::Disposing
    ) {
        return Err(ChordError::Message(format!(
            "Facet service targets cannot be used during {}",
            phase_str(core.phase.get())
        )));
    }
    Ok(())
}

/// Resolves one requirement's provider: the source that offers it, or the
/// single source that defers availability, or an error when several sources
/// would defer the same service.
fn requirement_source(
    core: &KernelCore,
    requirement: &FacetServiceReference,
    found: Option<(String, ServiceMode, usize)>,
) -> Result<Option<(String, ServiceMode, usize)>, ChordError> {
    if let Some(found) = found {
        return Ok(Some(found));
    }
    let deferred: Vec<usize> = core
        .service_sources
        .iter()
        .enumerate()
        .filter(|(_, source)| source.accepts_unavailable_services())
        .map(|(index, _)| index)
        .collect();
    match deferred.len() {
        0 => Ok(None),
        1 => Ok(Some((
            requirement.service_id.clone(),
            requirement.mode,
            deferred[0],
        ))),
        _ => Err(ChordError::Message(format!(
            "Facet host service {} has more than one deferred source",
            requirement.service_id
        ))),
    }
}

fn validate_facets(
    records: &[(String, Rc<RuntimeRecord>)],
    external: &HashMap<String, ExternalService>,
) -> Result<Vec<String>, ChordError> {
    let mut providers: Vec<(String, Option<String>, Option<ServiceMode>)> = external
        .iter()
        .map(|(service_id, external)| (service_id.clone(), None, Some(external.mode)))
        .collect();
    for (facet_id, record) in records {
        for provision in record.provides.borrow().iter() {
            let existing = providers
                .iter()
                .find(|(service_id, _, _)| *service_id == provision.service_id)
                .cloned();
            if let Some((service_id, provider_facet, provider_mode)) = existing {
                if let Some(mode) = provider_mode
                    && mode != provision.mode
                {
                    return Err(ChordError::Message(format!(
                        "Service {service_id} is provided as both singleton and keyed"
                    )));
                }
                if provider_facet.is_none() {
                    return Err(ChordError::Message(format!(
                        "Service {service_id} is provided by both the host and {facet_id}"
                    )));
                }
                return Err(ChordError::Message(format!(
                    "Service {service_id} is provided by both {provider_facet:?} and {facet_id}"
                )));
            }
            providers.push((
                provision.service_id.clone(),
                Some(facet_id.clone()),
                Some(provision.mode),
            ));
        }
    }

    let mut dependencies: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
    let mut dependents: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
    for (facet_id, _) in records {
        dependencies.insert(facet_id.clone(), std::collections::HashSet::new());
        dependents.insert(facet_id.clone(), std::collections::HashSet::new());
    }
    for (facet_id, record) in records {
        for requirement in record.requires.borrow().iter() {
            let provider = providers
                .iter()
                .find(|(service_id, _, _)| *service_id == requirement.service_id)
                .ok_or_else(|| {
                    ChordError::Message(format!(
                        "Facet {} requires local/{}/{}, but no facet provides it",
                        facet_id, requirement.service_id, requirement.mode
                    ))
                })?;
            if let Some(mode) = provider.2
                && mode != requirement.mode
            {
                return Err(ChordError::Message(format!(
                    "Facet {} requires {} as {}, but {} provides it as {}",
                    facet_id,
                    requirement.service_id,
                    requirement.mode,
                    provider.1.as_deref().unwrap_or("the host"),
                    mode
                )));
            }
            let Some(provider_facet) = &provider.1 else {
                continue;
            };
            if provider_facet == facet_id {
                continue;
            }
            #[allow(
                clippy::expect_used,
                reason = "every facet id was seeded into both maps before this loop"
            )]
            {
                dependencies
                    .get_mut(facet_id)
                    .expect("seeded")
                    .insert(provider_facet.clone());
                dependents
                    .get_mut(provider_facet)
                    .expect("registered when staged")
                    .insert(facet_id.clone());
            }
        }
    }
    topological_order(records, &dependencies, &dependents)
}

fn topological_order(
    records: &[(String, Rc<RuntimeRecord>)],
    dependencies: &HashMap<String, std::collections::HashSet<String>>,
    dependents: &HashMap<String, std::collections::HashSet<String>>,
) -> Result<Vec<String>, ChordError> {
    let mut remaining: HashMap<&str, usize> = records
        .iter()
        .map(|(facet_id, _)| (facet_id.as_str(), dependencies[facet_id.as_str()].len()))
        .collect();
    let mut ready: Vec<&str> = records
        .iter()
        .map(|(facet_id, _)| facet_id.as_str())
        .filter(|id| remaining[id] == 0)
        .collect();
    let mut order: Vec<String> = Vec::new();
    while let Some(id) = (!ready.is_empty()).then(|| {
        let id = ready[0];
        ready.remove(0);
        id
    }) {
        order.push(id.to_string());
        if let Some(dependent_ids) = dependents.get(id) {
            for dependent in dependent_ids {
                let count = remaining[dependent.as_str()] - 1;
                remaining.insert(dependent.as_str(), count);
                if count == 0 {
                    ready.push(dependent.as_str());
                }
            }
        }
    }
    if order.len() != records.len() {
        let names: Vec<String> = records
            .iter()
            .map(|(facet_id, _)| facet_id.clone())
            .filter(|id| remaining[id.as_str()] > 0)
            .collect();
        return Err(ChordError::Message(format!(
            "Facet dependency cycle: {}",
            names.join(", ")
        )));
    }
    Ok(order)
}
