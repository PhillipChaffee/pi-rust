//! Core data model of the runtime, ported from upstream `src/types.ts`.
//!
//! The owned [`JsonValue`] tree, the service vocabulary (`ServiceType`
//! singleton and keyed variants, `ServiceCall`, `ServiceProviderUpdate`,
//! `ServiceCatalogueEntry`, `ServiceInstanceAddress`), the `Service` and
//! `Context` contracts, and the facet-host plus remote-transport surfaces.
//! Contracts TypeScript enforces only at compile time (`RemoteServiceContract`,
//! `JsonRepresentation`) restate here as trait bounds over owned wire-safe
//! data.

use std::fmt;
use std::rc::Rc;

/// A finite JSON number.
///
/// Upstream spells this half of `JsonValue` as the JavaScript `number`; the
/// newtype keeps the constructor surface the place where non-finite values
/// die, so a [`JsonValue`] tree can never carry `NaN` or an infinity and
/// every consumer may assume `Number.isFinite` semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsonNumber(f64);

impl JsonNumber {
    /// The finite value, or [`None`] for `NaN` and the infinities.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// The value as a plain float. Every returned value is finite.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl From<i64> for JsonNumber {
    fn from(value: i64) -> Self {
        #[allow(
            clippy::cast_precision_loss,
            reason = "JavaScript numbers are f64, so the wire loses precision past 2^53 for any producer; the constructor mirrors that"
        )]
        Self(value as f64)
    }
}

impl From<u64> for JsonNumber {
    fn from(value: u64) -> Self {
        #[allow(
            clippy::cast_precision_loss,
            reason = "JavaScript numbers are f64, so the wire loses precision past 2^53 for any producer; the constructor mirrors that"
        )]
        Self(value as f64)
    }
}

/// A JSON object preserving insertion order.
///
/// JavaScript object semantics are the contract the port mirrors: assigning
/// to an existing key updates it in place, a fresh key appends at the end,
/// and equality ignores order because delta replication preserves JSON
/// values, not insertion order.
#[derive(Debug, Clone, Default)]
pub struct JsonObject {
    entries: Vec<(String, JsonValue)>,
}

impl JsonObject {
    /// An empty object.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an object from `(key, value)` pairs in iteration order.
    #[must_use]
    pub fn from_entries(entries: Vec<(String, JsonValue)>) -> Self {
        let mut object = Self::default();
        for (key, value) in entries {
            object.set(key, value);
        }
        object
    }

    /// The value stored under `key`, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.entries
            .iter()
            .find(|(stored, _)| stored == key)
            .map(|(_, value)| value)
    }

    /// The value stored under `key`, mutable, if present.
    pub fn as_value_mut(&mut self, key: &str) -> Option<&mut JsonValue> {
        self.entries
            .iter_mut()
            .find(|(stored, _)| stored == key)
            .map(|(_, value)| value)
    }

    /// Assigns `key`: an existing entry updates in place, a fresh one appends.
    pub fn set(&mut self, key: impl Into<String>, value: JsonValue) {
        let key = key.into();
        match self.entries.iter_mut().find(|(stored, _)| *stored == key) {
            Some((_, stored)) => *stored = value,
            None => self.entries.push((key, value)),
        }
    }

    /// Removes `key`, returning its value if it was present.
    pub fn remove(&mut self, key: &str) -> Option<JsonValue> {
        let at = self.entries.iter().position(|(stored, _)| stored == key)?;
        Some(self.entries.remove(at).1)
    }

    /// Whether `key` is stored.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(stored, _)| stored == key)
    }

    /// The keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(key, _)| key.as_str())
    }

    /// The number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the object holds no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The `(key, value)` entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &JsonValue)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }
}

/// Strict JSON: finite numbers, plain objects, no cycles.
///
/// Owned data restates the guarantees `isJsonValue` checks for at runtime in
/// TypeScript: a [`JsonNumber`] cannot be non-finite, a [`JsonObject`] cannot
/// carry an exotic prototype, and a tree owned by value cannot cycle, so
/// [`crate::json::is_json_value`] only has depth left to police.
#[derive(Debug, Clone)]
#[allow(
    clippy::enum_variant_names,
    reason = "the variant names mirror the upstream JsonValue union, which spells JsonValue"
)]
pub enum JsonValue {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A finite number.
    Number(JsonNumber),
    /// A string.
    Str(String),
    /// An ordered array; holes cannot be represented.
    Array(Vec<Self>),
    /// An object with string keys in insertion order.
    Object(JsonObject),
}

impl JsonValue {
    /// A finite number, or [`None`] when `value` is `NaN` or infinite.
    #[must_use]
    pub fn number(value: f64) -> Option<Self> {
        Some(Self::Number(JsonNumber::new(value)?))
    }

    /// A string value.
    #[must_use]
    pub fn string(value: impl Into<String>) -> Self {
        Self::Str(value.into())
    }

    /// The string content, or [`None`] when this is not a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    /// The number as a plain float, or [`None`] when this is not a number.
    #[must_use]
    pub const fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(number) => Some(number.get()),
            _ => None,
        }
    }

    /// Whether the value is a container: an array or an object.
    ///
    /// Upstream spells this `isObj`; every walk, diff, and resolve branches on
    /// it.
    #[must_use]
    pub const fn is_container(&self) -> bool {
        matches!(self, Self::Array(_) | Self::Object(_))
    }

    /// The array elements, or [`None`] when this is not an array.
    #[must_use]
    pub const fn as_array(&self) -> Option<&Vec<Self>> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The object entries, or [`None`] when this is not an object.
    #[must_use]
    pub const fn as_object(&self) -> Option<&JsonObject> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// The object as a mutable entry list, or [`None`] when this is not an
    /// object.
    pub const fn as_object_mut(&mut self) -> Option<&mut JsonObject> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// The array as a mutable element list, or [`None`] when this is not an
    /// array.
    pub const fn as_array_mut(&mut self) -> Option<&mut Vec<Self>> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Compact JSON text, the form `JSON.stringify` produces for the tree.
    ///
    /// Object keys serialize in insertion order, matching upstream's
    /// `JSON.stringify`.
    ///
    /// # Panics
    /// Never: every [`JsonNumber`] is finite by construction.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Number(value) => {
                let text = value.get().to_string();
                out.push_str(&text);
            }
            Self::Str(text) => write_json_string(text, out),
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write_json(out);
                }
                out.push(']');
            }
            Self::Object(object) => {
                out.push('{');
                for (index, (key, value)) in object.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_json_string(key, out);
                    out.push(':');
                    value.write_json(out);
                }
                out.push('}');
            }
        }
    }
}

/// Writes `text` as a JSON string literal, escaping quotes, backslashes, and
/// the control characters `JSON.stringify` escapes.
fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            character if (character as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", character as u32);
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

/// Equality of JSON values as data: arrays element-wise in order, objects by
/// key sets.
///
/// JavaScript's structural comparison and vitest's `toEqual` ignore object
/// insertion order — delta replication preserves JSON values, not order — so
/// the derived order-sensitive comparison would fail ports of upstream
/// assertions that hold. Strings stay compared exactly; numbers keep float
/// equality, where `-0.0 == 0.0` matches `===`.
impl PartialEq for JsonValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Number(left), Self::Number(right)) => left == right,
            (Self::Str(left), Self::Str(right)) => left == right,
            (Self::Array(left), Self::Array(right)) => left == right,
            (Self::Object(left), Self::Object(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(key, value)| {
                        right
                            .get(key)
                            .is_some_and(|other_value| value == other_value)
                    })
            }
            _ => false,
        }
    }
}

/// Equality is an equivalence relation: numbers are finite (no `NaN`), so
/// reflexivity, symmetry, and transitivity all hold, and object comparison is
/// key-set equality, which is an equivalence on maps.
impl Eq for JsonValue {}

impl Eq for JsonNumber {}

/// The mode one service is provided and addressed in, the wire strings
/// `singleton` and `keyed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceMode {
    /// One implementation shared by every consumer.
    Singleton,
    /// Instances addressed by key, each with a monotonically rising
    /// generation.
    Keyed,
}

impl ServiceMode {
    /// The wire spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Singleton => "singleton",
            Self::Keyed => "keyed",
        }
    }

    /// Parses the wire spelling.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "singleton" => Some(Self::Singleton),
            "keyed" => Some(Self::Keyed),
            _ => None,
        }
    }
}

impl fmt::Display for ServiceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stable identity for one service contract, upstream's `Service<T>`.
///
/// The compile-time `SERVICE_TYPE` brand is dropped: runtime identity is
/// the ID alone, and the member registry carries what the type parameter
/// described.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Service {
    /// The service ID, `$chord.`-prefixed IDs reserved.
    pub id: String,
    /// Process-local services accept unrestricted contracts and are never
    /// published remotely.
    pub local: bool,
}

/// One catalogue line: which service a provider offers and in which mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceCatalogueEntry {
    /// The service ID.
    pub service_id: String,
    /// The mode the provider registered it in.
    pub mode: ServiceMode,
}

/// Where one keyed instance lives: the key plus the generation that fences
/// stale references across respawns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceInstanceAddress {
    /// The instance key.
    pub key: String,
    /// Rising per key; a reference with a lower generation is stale.
    pub generation: u64,
}

/// Whether a member is invocable or replicated state, the wire `kind`
/// spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceMemberKind {
    /// A method.
    Method,
    /// Replicated state.
    State,
}

/// One member's description in a snapshot: a method, or state with its
/// sequence and the operation batch that carries the current value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceMemberSnapshot {
    /// A method member.
    Method {
        /// The member name.
        name: String,
    },
    /// Replicated state.
    State {
        /// The member name.
        name: String,
        /// The provider's publication sequence for this state.
        sequence: u64,
        /// The operations from the empty value to the current one; a base
        /// batch (`is_base`) for sequence 0.
        ops: Vec<crate::delta::Op>,
    },
}

impl ServiceMemberSnapshot {
    /// The member name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Method { name } | Self::State { name, .. } => name,
        }
    }

    /// The member kind.
    #[must_use]
    pub const fn kind(&self) -> ServiceMemberKind {
        match self {
            Self::Method { .. } => ServiceMemberKind::Method,
            Self::State { .. } => ServiceMemberKind::State,
        }
    }
}

/// One live instance's members, optionally addressed when keyed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceInstanceSnapshot {
    /// The address, present for keyed instances.
    pub instance: Option<ServiceInstanceAddress>,
    /// The members in registration order.
    pub members: Vec<ServiceMemberSnapshot>,
}

/// The initial snapshot of one subscription: which service, its mode, and
/// every live instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceSubscriptionSnapshot {
    /// The subscribed service.
    pub service_id: String,
    /// The mode the subscription asked for.
    pub mode: ServiceMode,
    /// The live instances at subscription time.
    pub instances: Vec<ServiceInstanceSnapshot>,
}

/// One update a provider pushes to its subscribers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceProviderUpdate {
    /// State changed on one member of one instance.
    State {
        /// The instance address, absent for singletons.
        instance: Option<ServiceInstanceAddress>,
        /// The state member that published.
        member: String,
        /// The provider's sequence for this publication; consumers fence on
        /// exactly `previous + 1`.
        sequence: u64,
        /// The flushed operations.
        ops: Vec<crate::delta::Op>,
    },
    /// The singleton became unavailable.
    Unavailable,
    /// The singleton was replaced; the snapshot carries its fresh state.
    Replaced {
        /// The replacement's snapshot.
        snapshot: ServiceInstanceSnapshot,
    },
    /// A keyed instance came up.
    Spawned {
        /// The new instance's snapshot.
        instance: ServiceInstanceSnapshot,
    },
    /// A keyed instance closed.
    Closed {
        /// The address that closed.
        instance: ServiceInstanceAddress,
    },
}

/// One remote method invocation: which instance, which member, borrowed
/// arguments. Upstream validates values at the parse boundary only and
/// never clones them.
#[derive(Debug, Clone)]
pub struct ServiceCall {
    /// The service to invoke on.
    pub service_id: String,
    /// The keyed instance, absent for singletons.
    pub instance: Option<ServiceInstanceAddress>,
    /// The method member.
    pub member: String,
    /// The borrowed immutable arguments.
    pub args: Vec<JsonValue>,
}

/// Whether a replicated-state delivery hydrates or updates, the wire
/// `kind` spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicatedStateDeliveryKind {
    /// The complete initial value.
    Hydrate,
    /// An incremental revision.
    Update,
}

impl ReplicatedStateDeliveryKind {
    /// The wire spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hydrate => "hydrate",
            Self::Update => "update",
        }
    }
}

impl fmt::Display for ReplicatedStateDeliveryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a replicated-state listener receives: the delivery kind and the
/// sequence it corresponds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicatedStateDelivery {
    /// Hydrate or update.
    pub kind: ReplicatedStateDeliveryKind,
    /// The sequence this value reflects.
    pub sequence: u64,
}

/// An unsubscribe handle; calling it removes one listener, and repeat calls
/// are no-ops (upstream's returned `() => void`).
pub type Unsubscribe = Box<dyn Fn()>;

/// The close path of one live [`ServiceSubscription`].
pub(crate) type SubscriptionClose = Box<
    dyn Fn(
        Option<crate::context::Context>,
    ) -> crate::future::LocalBoxFuture<Result<(), crate::errors::ChordError>>,
>;

/// A keyed-service observer, upstream's `(service, context) => void |
/// Promise<void>` with the promise arm dropped.
///
/// The directory starts the handler synchronously with the raw bound
/// target; the wrapping caller builds the guarded view over it, and
/// cancellation is cooperative through the context, so an async
/// continuation would have nothing to drive it.
pub type KeyedServiceHandler = Box<dyn Fn(crate::handle::ServiceTarget, crate::context::Context)>;

/// The view-level keyed observer: the wrapping layer converts the raw
/// target into a guarded view before invoking it.
pub type KeyedViewHandler = Rc<dyn Fn(crate::handle::ServiceView, crate::context::Context)>;

/// One live subscription a transport hands back: the initial snapshot, the
/// activation gate, and the close path.
pub struct ServiceSubscription {
    /// The snapshot taken when the provider accepted the subscription.
    pub snapshot: ServiceSubscriptionSnapshot,
    /// Starts buffered-update delivery; repeat calls are no-ops. A failure
    /// surfaces from the collected listener reports, upstream's throw.
    pub activate: Box<dyn Fn() -> Result<(), crate::errors::ChordError>>,
    /// Closes the subscription, dropping buffered updates. Never rejects
    /// for in-process transports; remote transports may.
    pub close: SubscriptionClose,
}

impl fmt::Debug for ServiceSubscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServiceSubscription")
            .field("snapshot", &self.snapshot)
            .finish_non_exhaustive()
    }
}

/// The pluggable wire boundary a remote service binding consumes,
/// upstream's `RemoteServiceTransport`.
///
/// Implementations choose transport, framing, routing, and envelope
/// encoding; values crossing the boundary stay strict JSON, and adapters
/// own serialization and any isolation copies.
pub trait RemoteServiceTransport {
    /// Invokes one remote method.
    fn invoke(
        &self,
        call: ServiceCall,
        context: crate::context::Context,
    ) -> crate::future::LocalBoxFuture<Result<Option<JsonValue>, crate::errors::ChordError>>;

    /// Opens a subscription to one service.
    fn subscribe(
        &self,
        service_id: String,
        mode: ServiceMode,
        listener: ServiceProviderListener,
        context: crate::context::Context,
    ) -> crate::future::LocalBoxFuture<Result<ServiceSubscription, crate::errors::ChordError>>;
}

/// A provider-update listener, upstream's `(update, context) => void`.
pub type ServiceProviderListener = Rc<dyn Fn(&ServiceProviderUpdate, &crate::context::Context)>;

/// The publisher one remote endpoint hands subscription updates to,
/// upstream's `ServiceUpdatePublisher` with the promise arm dropped.
///
/// Publishing is synchronous, and a failure propagates like upstream's
/// sync throw before the `Promise.resolve` wrapper.
pub type ServiceUpdatePublisher =
    Rc<dyn Fn(&str, &ServiceProviderUpdate, &crate::context::Context)>;

/// The service surface a facet or binding consumes, upstream's
/// `RemoteServices`. Handles returned here stay stable across provider
/// replacement and rebind.
pub trait RemoteServices {
    /// Acquire one singleton service's handle.
    ///
    /// # Errors
    /// The crate's error model; upstream throws.
    fn use_service(
        &self,
        service: &Service,
    ) -> Result<crate::handle::ServiceView, crate::errors::ChordError>;

    /// Observe every live instance of one keyed service.
    ///
    /// # Errors
    /// The crate's error model; upstream throws.
    fn observe(
        &self,
        service: &Service,
        handler: KeyedViewHandler,
    ) -> Result<Unsubscribe, crate::errors::ChordError>;

    /// Wait until every currently acquired service has installed its initial
    /// snapshot.
    fn ready(
        &self,
        context: crate::context::Context,
    ) -> crate::future::LocalBoxFuture<Result<(), crate::errors::ChordError>>;

    /// Tear down every subscription and facade this binding owns.
    fn dispose(
        &self,
        context: crate::context::Context,
    ) -> crate::future::LocalBoxFuture<Result<(), crate::errors::ChordError>>;
}

/// Whether a source may provisionally own absent requirements while it is
/// unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteServiceSourceOptions {
    /// The `acceptsUnavailableServices` flag.
    pub accepts_unavailable_services: bool,
}

/// One external service source the facet host can bind requirements to,
/// upstream's `RemoteServiceSource`.
pub trait RemoteServiceSource {
    /// Whether this currently unavailable source may provisionally own
    /// absent requirements.
    fn accepts_unavailable_services(&self) -> bool;

    /// The catalogue this source currently offers.
    fn catalogue(
        &self,
        context: crate::context::Context,
    ) -> crate::future::LocalBoxFuture<Result<Vec<ServiceCatalogueEntry>, crate::errors::ChordError>>;

    /// Opens the services interface for the listed service IDs.
    fn open(&self, options: RemoteServiceSourceOpenOptions) -> Rc<dyn RemoteServices>;
}

/// The options [`RemoteServiceSource::open`] receives.
pub struct RemoteServiceSourceOpenOptions {
    /// The service IDs this source was selected for.
    pub services: Vec<Service>,
    /// The host's access gate; handles call it before every use.
    pub assert_access: crate::handle::AssertAccess,
    /// The host's error reporter.
    pub on_error: crate::handle::ErrorReporter,
}

impl fmt::Debug for RemoteServiceSourceOpenOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteServiceSourceOpenOptions")
            .field("services", &self.services)
            .finish_non_exhaustive()
    }
}

/// One facet: a synchronous setup that declares requirements and
/// provisions through the environment.
///
/// The setup is synchronous by construction — the closure receives
/// `&mut FacetEnvironment` and returns nothing, which restates upstream's
/// "setup must be synchronous" contract.
pub struct FacetDef {
    /// The facet ID, unique within one generation.
    pub id: String,
    /// The setup body, shared so loaders can hand the same facet out
    /// repeatedly.
    pub setup: Rc<dyn Fn(&mut crate::facets::host::FacetEnvironment)>,
}

impl fmt::Debug for FacetDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FacetDef")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Clone for FacetDef {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            setup: self.setup.clone(),
        }
    }
}

/// What a loader produced: the facets plus their teardown, upstream's
/// `LoadedFacets`.
pub struct LoadedFacets {
    /// The loaded facets, in loader order.
    pub facets: Vec<FacetDef>,
    /// Disposes the loaded generation; repeat calls are no-ops.
    pub dispose: crate::handle::Disposal,
}

impl fmt::Debug for LoadedFacets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadedFacets")
            .field(
                "facets",
                &self
                    .facets
                    .iter()
                    .map(|facet| facet.id.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

/// Produces one generation of facets, upstream's `FacetLoader`.
pub trait FacetLoader {
    /// Loads the facets.
    fn load(
        &self,
    ) -> crate::future::LocalBoxFuture<Result<LoadedFacets, crate::errors::ChordError>>;
}
