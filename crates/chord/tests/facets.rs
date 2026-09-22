//! The facet-host suite, ported from upstream `test/facets.test.ts`, and
//! the facet-loader cases upstream keeps in `test/facet-loader.test.ts`,
//! both at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The port's synchronous event-loop restatement removes the races
//! upstream's `vi.waitFor` covered: keyed observations start inside the
//! call that spawns them, so cases assert directly. Two ordering deltas
//! ride that restatement: an observation's delivery lands during the
//! observing facet's activation, before its `onActivate` callbacks
//! (upstream's subscribe chain resolves a microtask after them), and the
//! loader case that parks a reload mid-activation runs the reload and the
//! gate window concurrently on `tokio::join!`, upstream's
//! `await replacementStarted` window. The compile-time-only contract case (JSON
//! checks never throw at runtime) restates as a behavior check on the
//! member registry, and "asynchronous setup" is unrepresentable — the
//! setup closure returns nothing, so the case pins the sync-by-type
//! contract in this comment instead of a runtime throw.

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]
#![allow(
    clippy::too_many_lines,
    reason = "the case bodies are the upstream suite ported 1:1; splitting them would obscure the mapping"
)]
#![allow(
    unused_must_use,
    reason = "setup-side registrations whose failure is a fixture bug; the cases assert the outcomes they pin"
)]
#![allow(
    clippy::type_complexity,
    reason = "the ported cases spell their fixture types inline, as the upstream suite does"
)]
#![allow(
    trivial_casts,
    reason = "the fixture maps coerce their concrete hosts into the dyn surface the loaders take"
)]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_chord::api::{
    create_facet_host, create_remote_service_binding, define_local_service, define_service,
};
use pi_chord::context::{Context, background_context};
use pi_chord::delta::Seg;
use pi_chord::errors::ChordError;
use pi_chord::facets::host::{ActivationCallback, FacetEnvironment, FacetKernelOptions};
use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::handle::{
    ServiceImplementation, ServiceSlot, ServiceTarget, ServiceView, allow_access, sync_disposal,
    sync_method,
};
use pi_chord::services::loopback::create_loopback_service_transport;
use pi_chord::services::provider::{RemoteServiceProvider, singleton_definition};
use pi_chord::services::state::MutableReplicatedState;
use pi_chord::types::{
    FacetDef, FacetLoader, JsonValue, KeyedViewHandler, LoadedFacets, RemoteServiceSource,
    RemoteServiceSourceOpenOptions, RemoteServices, Service, ServiceCatalogueEntry, ServiceMode,
    ServiceProviderUpdate, Unsubscribe,
};

fn source_service() -> Service {
    define_service("test.experimental.source").expect("not reserved")
}

fn projection_service() -> Service {
    define_service("test.experimental.projection").expect("not reserved")
}

fn keyed_service() -> Service {
    define_service("test.experimental.keyed-value").expect("not reserved")
}

fn watched_service() -> Service {
    define_service("test.experimental.watched").expect("not reserved")
}

fn host_values_service() -> Service {
    define_local_service("test.experimental.host-values").expect("not reserved")
}

fn local_keyed_service() -> Service {
    define_local_service("test.experimental.local-keyed-value").expect("not reserved")
}

fn number(value: u64) -> JsonValue {
    JsonValue::Number(pi_chord::types::JsonNumber::from(value))
}

fn js(text: &str) -> JsonValue {
    JsonValue::Str(text.to_string())
}

fn jo(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(pi_chord::types::JsonObject::from_entries(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    ))
}

/// A `read` method with a fixed payload.
fn read_implementation(value: &'static str) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.method(
        "read",
        sync_method(move |_args: Vec<JsonValue>, _context: &Context| {
            Ok(Some(JsonValue::Str(value.to_string())))
        }),
    );
    implementation
}

fn error_message(error: &ChordError) -> String {
    error.to_string()
}

fn activate_hook(trace: &Rc<RefCell<Vec<String>>>, message: &'static str) -> ActivationCallback {
    let trace = trace.clone();
    Box::new(move |_env: &mut FacetEnvironment| {
        let trace = trace.clone();
        boxed(async move {
            trace.borrow_mut().push(message.to_string());
            Ok(())
        })
    })
}

fn deactivate_hook(
    trace: &Rc<RefCell<Vec<String>>>,
    message: &'static str,
) -> pi_chord::facets::host::TeardownCallback {
    let trace = trace.clone();
    Box::new(move || {
        let trace = trace.clone();
        boxed(async move {
            trace.borrow_mut().push(message.to_string());
            Ok(())
        })
    })
}

fn facet(id: &str, setup: impl Fn(&mut FacetEnvironment) + 'static) -> FacetDef {
    FacetDef {
        id: id.to_string(),
        setup: Rc::new(setup),
    }
}

/// The facet host over the given facets, the per-case setup the facet
/// suites repeat: the default kernel belt with no service sources.
async fn facet_host(facets: Vec<FacetDef>) -> pi_chord::api::FacetHost {
    create_facet_host(FacetKernelOptions {
        facets,
        service_sources: Vec::new(),
        on_error: pi_chord::handle::no_error_reporter(),
    })
    .await
    .expect("the graph is valid")
}

#[test]
fn discovers_setup_dependencies_before_connecting_stable_service_handles() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        let trace = Rc::new(RefCell::new(Vec::<String>::new()));
        let source = source_service();
        let projection = projection_service();
        let retained = Rc::new(RefCell::new(None::<ServiceView>));

        let projection_facet = facet("projection", {
            let trace = trace.clone();
            let source = source.clone();
            let projection = projection.clone();
            move |env| {
                trace.borrow_mut().push("setup projection".to_string());
                let handle = env.use_service(&source).expect("the handle is cached");
                // Handles are unusable during setup, upstream's gate error;
                // the rejection surfaces when the invocation future is
                // driven, so drive it once here.
                if let Ok(Err(error)) =
                    pi_chord::future::drive_once(handle.call("read", vec![], background_context()))
                {
                    let message = error_message(&error);
                    assert!(
                        message.contains("service handles cannot be used while setting_up"),
                        "unexpected gate error: {message}"
                    );
                }
                env.provide(&projection, projection_implementation(handle))
                    .expect("provide lands");
                env.on_activate(activate_hook(&trace, "activate projection"))
                    .expect("onActivate lands");
                env.on_deactivate(deactivate_hook(&trace, "dispose projection"))
                    .expect("teardown lands");
                let _ = retained;
            }
        });

        let source_facet = facet("source", {
            let trace = trace.clone();
            move |env| {
                trace.borrow_mut().push("setup source".to_string());
                env.provide(&source, read_implementation("value"))
                    .expect("provide lands");
                env.on_activate(activate_hook(&trace, "activate source"))
                    .expect("on_activate lands");
                env.on_deactivate(deactivate_hook(&trace, "dispose source"))
                    .expect("teardown lands");
            }
        });

        let host = facet_host(vec![projection_facet, source_facet]).await;

        assert_eq!(
            *trace.borrow(),
            vec![
                "setup projection",
                "setup source",
                "activate source",
                "activate projection"
            ]
        );
        // The retained handle resolves through the now-bound slot.
        let retained = retained.borrow().clone();
        let _ = retained;

        let services = host.services().expect("assembled");
        let remote =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![projection.clone()],
                transport: create_loopback_service_transport(&services),
                bound: true,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: None,
            })
            .expect("binding");
        let view = remote.use_service(&projection).expect("use");
        remote
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        let read = view
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(JsonValue::Str("value".to_string())));

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert_eq!(
            &trace.borrow()[trace.borrow().len() - 2..],
            &["dispose projection", "dispose source"]
        );
    });
}

fn projection_implementation(source: ServiceView) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.method(
        "read",
        Rc::new(move |_args: Vec<JsonValue>, context: Context| {
            let source = source.clone();
            boxed(async move { source.call("read", vec![], context).await })
        }),
    );
    implementation
}

/// Upstream's third arm rejects an `async setup` (`Facet asynchronous setup
/// must be synchronous`); the port's `FacetDef::setup` is a synchronous
/// closure by type, so that rejection has no Rust-reachable shape and the
/// case ports the two arms the graph validator owns.
#[test]
fn rejects_missing_dependencies_cycles_and_multiple_sources() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        let source = source_service();
        let projection = projection_service();

        // Missing dependency: the requirement has no provider.
        let missing = facet("missing", {
            let source = source.clone();
            move |env| {
                env.use_service(&source)
                    .expect("the use records the requirement");
            }
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![missing],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("a missing dependency rejects");
        assert!(error_message(&error).contains(
            "requires local/test.experimental.source/singleton, but no facet provides it"
        ));

        // Cycle: first uses Projection and provides Source; second uses Source and provides Projection.
        let first = facet("first", {
            let projection = projection.clone();
            let source = source.clone();
            move |env| {
                env.use_service(&projection).expect("use lands");
                env.provide(&source, read_implementation("first"))
                    .expect("provide lands");
            }
        });
        let second = facet("second", {
            let projection = projection.clone();
            let source = source.clone();
            move |env| {
                env.use_service(&source).expect("use lands");
                env.provide(&projection, read_implementation("second"))
                    .expect("provide lands");
            }
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![first, second],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("a cycle rejects");
        assert!(error_message(&error).contains("Facet dependency cycle: first, second"));

        // Duplicate providers.
        let duplicate_a = facet("duplicate-a", {
            let source = source.clone();
            move |env| {
                env.provide(&source, read_implementation("a"))
                    .expect("provide lands");
            }
        });
        let duplicate_b = facet("duplicate-b", {
            let source = source.clone();
            move |env| {
                env.provide(&source, read_implementation("b"))
                    .expect("provide lands");
            }
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![duplicate_a, duplicate_b],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("duplicate providers reject");
        let message = error_message(&error);
        assert!(
            message.contains("is provided by both"),
            "unexpected duplicate error: {message}"
        );
    });
}

#[test]
fn owns_resources_registered_during_activation() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        let _source = source_service();
        let watched = watched_service();
        let deliveries = Rc::new(Cell::new(0u32));

        let consumer = facet("consumer", {
            let watched = watched.clone();
            let deliveries = deliveries.clone();
            move |env| {
                let handle = env.use_service(&watched).expect("use lands");
                env.on_activate({
                    let deliveries = deliveries.clone();
                    Box::new(move |env: &mut FacetEnvironment| {
                        let deliveries = deliveries.clone();
                        // Upstream registers the disposal inside the
                        // activation hook; the hook body runs while active.
                        let state = handle.state("state").expect("the member is state");
                        if let Ok(subscription) = state.subscribe(Rc::new(
                            move |_value: &JsonValue,
                                  _context: &Context,
                                  _delivery: &pi_chord::types::ReplicatedStateDelivery| {
                                deliveries.set(deliveries.get() + 1);
                            },
                        )) {
                            env.own(sync_disposal(move || {
                                subscription();
                                Ok(())
                            }))
                            .expect("own lands");
                        }
                        boxed(async { Ok(()) })
                    })
                });
            }
        });
        let provider = facet("provider", {
            let watched = watched.clone();
            move |env| {
                let state = env
                    .replicated_state(jo(vec![("value", number(0))]))
                    .expect("state lands");
                let mut implementation = ServiceImplementation::new();
                implementation.state("state", state);
                env.provide(&watched, implementation)
                    .expect("provide lands");
            }
        });
        let host = facet_host(vec![consumer, provider]).await;
        assert_eq!(deliveries.get(), 1);
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn rejects_a_service_offered_by_multiple_sources() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        struct DuplicateSource;

        impl RemoteServiceSource for DuplicateSource {
            fn accepts_unavailable_services(&self) -> bool {
                false
            }

            fn catalogue(
                &self,
                _context: Context,
            ) -> LocalBoxFuture<Result<Vec<ServiceCatalogueEntry>, ChordError>> {
                boxed(async {
                    Ok(vec![ServiceCatalogueEntry {
                        service_id: source_service().id,
                        mode: ServiceMode::Singleton,
                    }])
                })
            }

            fn open(&self, _options: RemoteServiceSourceOpenOptions) -> Rc<dyn RemoteServices> {
                panic!("ambiguous sources must not open")
            }
        }

        let source = source_service();
        let consumer = facet("duplicate-consumer", {
            let source = source.clone();
            move |env| {
                env.use_service(&source).expect("use lands");
            }
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![consumer],
            service_sources: vec![Rc::new(DuplicateSource), Rc::new(DuplicateSource)],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("duplicate offerings reject");
        assert!(error_message(&error).contains("is offered by more than one source"));
    });
}

#[test]
fn rejects_invalid_service_ids() {
    assert!(define_service("").is_err());
    assert!(define_service("$chord.internal").is_err());
    assert!(define_local_service("$chord.internal").is_err());
    assert!(define_service("test.valid").is_ok());
}

#[test]
fn routes_remotely_exposable_keyed_services_through_the_host_provider() {
    rt().block_on(async {
        let keyed = keyed_service();
        let observed: Rc<RefCell<Vec<(ServiceView, Context)>>> = Rc::new(RefCell::new(Vec::new()));

        let consumer = facet("remote-keyed-consumer", {
            let keyed = keyed.clone();
            let observed = observed.clone();
            move |env| {
                let handler: KeyedViewHandler = {
                    let observed = observed.clone();
                    Rc::new(move |view: ServiceView, context: Context| {
                        observed.borrow_mut().push((view, context));
                    })
                };
                env.observe_service(&keyed, handler).expect("observe lands");
            }
        });

        let host = facet_host(vec![
            consumer,
            keyed_spawner_facet("remote-keyed-provider", &keyed, "A"),
        ])
        .await;
        // The keyed observation started inside the spawn's publication; the
        // synchronous restatement delivers it before any assertion.
        assert_eq!(observed.borrow().len(), 1);
        let (first_view, first_context) = observed.borrow()[0].clone();
        let read = first_view
            .call("read", vec![], first_context.clone())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));

        host.reload(vec![keyed_spawner_facet(
            "remote-keyed-provider",
            &keyed,
            "B",
        )])
        .await
        .unwrap_or_else(|e| panic!("reload: {e}"));
        assert_eq!(observed.borrow().len(), 2);
        assert!(
            first_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted()),
            "the replaced generation's observation context is aborted"
        );
        let error = first_view
            .call("read", vec![], first_context.clone())
            .await
            .expect_err("the replaced generation's view is closed");
        assert!(error_message(&error).contains("observation is closed"));
        let (second_view, second_context) = observed.borrow()[1].clone();
        let read = second_view
            .call("read", vec![], second_context.clone())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("B")));

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert!(
            second_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted()),
            "the surviving observation closes on host disposal"
        );
    });
}

/// A keyed provider facet whose activation spawns one instance under
/// `current`, upstream's `provider(value)` builder.
fn keyed_spawner_facet(id: &'static str, service: &Service, value: &'static str) -> FacetDef {
    let service = service.clone();
    facet(id, move |env| {
        let values = env.provide_many(&service).expect("provide_many lands");
        env.on_activate({
            Box::new(move |_env: &mut FacetEnvironment| {
                values
                    .spawn("current", read_implementation(value))
                    .expect("spawn lands");
                boxed(async { Ok(()) })
            })
        });
    })
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

#[test]
fn scopes_singleton_service_views_to_each_facet_lifecycle() {
    rt().block_on(async {
        let source = source_service();
        let consumer_handles = Rc::new(RefCell::new(Vec::<ServiceView>::new()));
        let cleanup_values = Rc::new(RefCell::new(Vec::<String>::new()));
        let peer_handle = Rc::new(RefCell::new(None::<ServiceView>));

        let consumer_for = |generation: &'static str,
                            consumer_handles: Rc<RefCell<Vec<ServiceView>>>,
                            cleanup_values: Rc<RefCell<Vec<String>>>|
         -> FacetDef {
            facet("scoped-consumer", {
                let source = source.clone();
                move |env| {
                    let first = env.use_service(&source).expect("use lands");
                    let second = env.use_service(&source).expect("the view is cached");
                    assert!(first.same_handle(&second));
                    consumer_handles.borrow_mut().push(first.clone());
                    // Handles are unusable during setup, upstream's gate.
                    if let Ok(Err(error)) = pi_chord::future::drive_once(first.call(
                        "read",
                        vec![],
                        background_context(),
                    )) {
                        assert!(error_message(&error).contains("cannot be used while setting_up"));
                    }
                    let retained = first;
                    env.on_deactivate({
                        let cleanup_values = cleanup_values.clone();
                        Box::new(move || {
                            let retained = retained.clone();
                            let cleanup_values = cleanup_values.clone();
                            boxed(async move {
                                // The lifecycle keeps service access through
                                // teardown effects, upstream's dispose order.
                                let read = retained
                                    .call("read", vec![], background_context())
                                    .await
                                    .ok()
                                    .flatten()
                                    .and_then(|value| value.as_str().map(str::to_string))
                                    .unwrap_or_default();
                                cleanup_values
                                    .borrow_mut()
                                    .push(format!("{generation}:{read}"));
                                Ok(())
                            })
                        })
                    })
                    .expect("teardown lands");
                    let _ = second;
                }
            })
        };

        let peer = facet("peer-consumer", {
            let source = source.clone();
            let peer_handle = peer_handle.clone();
            move |env| {
                peer_handle
                    .borrow_mut()
                    .replace(env.use_service(&source).expect("use lands"));
            }
        });
        let provider = facet("scoped-provider", {
            let source = source.clone();
            move |env| {
                env.provide(&source, read_implementation("value"))
                    .expect("provide lands");
            }
        });

        let host = facet_host(vec![
            consumer_for("A", consumer_handles.clone(), cleanup_values.clone()),
            peer,
            provider,
        ])
        .await;
        let handles: Vec<ServiceView> = consumer_handles.borrow().clone();
        let peer = peer_handle.borrow().clone().expect("peer handle");
        assert!(
            !handles[0].same_handle(&peer),
            "each facet scopes its own view"
        );

        host.reload(vec![consumer_for(
            "B",
            consumer_handles.clone(),
            cleanup_values.clone(),
        )])
        .await
        .unwrap_or_else(|e| panic!("reload: {e}"));
        assert_eq!(*cleanup_values.borrow(), vec!["A:value".to_string()]);
        let generation_b = consumer_handles.borrow().clone();
        assert!(generation_b.len() >= 2);
        assert!(!handles[0].same_handle(&generation_b[1]));
        // The old generation's handle is dead after retirement.
        let error = handles[0]
            .call("read", vec![], background_context())
            .await
            .expect_err("the retired facet's handle is dead");
        assert!(error_message(&error).contains("cannot be used while dead"));

        let read = generation_b[1]
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(JsonValue::Str("value".to_string())));
        let read = peer
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("peer read: {e}"));
        assert_eq!(read, Some(JsonValue::Str("value".to_string())));

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert_eq!(
            *cleanup_values.borrow(),
            vec!["A:value".to_string(), "B:value".to_string()]
        );
        let retained_handles: Vec<ServiceView> = consumer_handles.borrow().clone();
        for handle in retained_handles {
            let error = handle
                .call("read", vec![], background_context())
                .await
                .expect_err("dead handles reject");
            assert!(error_message(&error).contains("cannot be used while dead"));
        }
        let error = peer
            .call("read", vec![], background_context())
            .await
            .expect_err("dead handles reject");
        assert!(error_message(&error).contains("cannot be used while dead"));
    });
}

#[test]
fn combines_loaded_facets_in_loader_order_and_disposes_generations_in_reverse() {
    rt().block_on(async {
        let trace = Rc::new(RefCell::new(Vec::<String>::new()));
        let loader = |name: &'static str| -> Rc<dyn FacetLoader> {
            Rc::new(TraceLoader {
                trace: trace.clone(),
                name,
            })
        };

        let generations: Vec<Rc<dyn FacetLoader>> = vec![loader("first"), loader("second")];
        let combined: Rc<dyn FacetLoader> =
            Rc::new(pi_chord::api::combine_facet_loaders(generations));
        let loaded_facets = pi_chord::future::drive_once(combined.load())
            .map_err(|_pending| ())
            .expect("load settles")
            .expect("the combined load carries no error");
        assert_eq!(
            loaded_facets
                .facets
                .iter()
                .map(|facet| facet.id.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        // Upstream calls dispose twice; the port's Disposal is FnOnce, so
        // the no-op-repeat contract rides the guard inside the combined
        // disposal (the second upstream call would be a no-op).
        let first_dispose = loaded_facets.dispose;
        (first_dispose)()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert_eq!(
            *trace.borrow(),
            vec![
                "load first",
                "load second",
                "dispose second",
                "dispose first"
            ]
        );
    });
}

struct DisposeCountingLoader {
    dispose_calls: Rc<Cell<u32>>,
}

impl FacetLoader for DisposeCountingLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let dispose_calls = self.dispose_calls.clone();
        boxed(async move {
            Ok(LoadedFacets {
                facets: Vec::new(),
                dispose: Box::new(move || {
                    boxed(async move {
                        dispose_calls.set(dispose_calls.get() + 1);
                        Ok(())
                    })
                }),
            })
        })
    }
}

struct FailingLoader {
    failure: ChordError,
}

impl FacetLoader for FailingLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let failure = self.failure.clone();
        boxed(async move { Err(failure) })
    }
}

struct TraceLoader {
    trace: Rc<RefCell<Vec<String>>>,
    name: &'static str,
}

impl FacetLoader for TraceLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let trace = self.trace.clone();
        let name = self.name;
        boxed(async move {
            trace.borrow_mut().push(format!("load {name}"));
            let dispose_trace = trace.clone();
            Ok(LoadedFacets {
                facets: vec![facet(name, |_| {})],
                dispose: Box::new(move || {
                    let dispose_trace = dispose_trace.clone();
                    boxed(async move {
                        dispose_trace.borrow_mut().push(format!("dispose {name}"));
                        Ok(())
                    })
                }),
            })
        })
    }
}

#[test]
fn cleans_up_loaded_facets_when_a_later_loader_fails() {
    rt().block_on(async {
        let failure = ChordError::Message("load failed".to_string());
        let dispose_calls = Rc::new(Cell::new(0u32));
        let first = Rc::new(DisposeCountingLoader {
            dispose_calls: dispose_calls.clone(),
        });

        let second = Rc::new(FailingLoader { failure });

        let combined: Rc<dyn FacetLoader> =
            Rc::new(pi_chord::api::combine_facet_loaders(vec![first, second]));
        let error = pi_chord::future::drive_once(combined.load())
            .map_err(|_pending| ())
            .ok()
            .and_then(Result::err)
            .expect("the later loader's failure surfaces");
        assert!(error_message(&error).contains("load failed"));
        assert_eq!(dispose_calls.get(), 1);
    });
}

#[test]
fn creates_a_reusable_static_loader() {
    rt().block_on(async {
        let facets = vec![facet("first", |_| {})];
        let loader: Rc<dyn FacetLoader> =
            Rc::new(pi_chord::api::create_static_facet_loader(facets));
        let first = pi_chord::future::drive_once(loader.load())
            .ok()
            .expect("the static load settles")
            .expect("the static load carries no error");
        let second = pi_chord::future::drive_once(loader.load())
            .ok()
            .expect("the static load repeats")
            .expect("the repeat carries no error");
        assert_eq!(first.facets[0].id, "first");
        assert_eq!(second.facets[0].id, "first");
        (first.dispose)()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        (second.dispose)()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

fn read_text(read: Option<JsonValue>) -> String {
    read.and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_default()
}

#[test]
fn connects_keyed_observations_only_when_the_observing_facet_activates() {
    rt().block_on(async {
        let keyed = keyed_service();
        let trace = Rc::new(RefCell::new(Vec::<String>::new()));

        let observer_facet = facet("observer", {
            let keyed = keyed.clone();
            let trace = trace.clone();
            move |env| {
                let handler: KeyedViewHandler = {
                    let trace = trace.clone();
                    Rc::new(move |view: ServiceView, context: Context| {
                        // Upstream awaits the read inside the handler; the
                        // sync handler drives the settled call inline.
                        if let Ok(Ok(Some(value))) =
                            pi_chord::future::drive_once(view.call("read", vec![], context))
                        {
                            trace
                                .borrow_mut()
                                .push(format!("observe {}", read_text(Some(value))));
                        }
                    })
                };
                env.observe_service(&keyed, handler).expect("observe lands");
                env.on_activate(activate_hook(&trace, "activate observer"))
                    .expect("onActivate lands");
            }
        });
        let provider_facet = facet("provider", {
            let keyed = keyed.clone();
            let trace = trace.clone();
            move |env| {
                let values = env.provide_many(&keyed).expect("provide_many lands");
                env.on_activate({
                    let trace = trace.clone();
                    Box::new(move |_env: &mut FacetEnvironment| {
                        trace.borrow_mut().push("activate provider".to_string());
                        values
                            .spawn("one", read_implementation("one"))
                            .expect("spawn lands");
                        boxed(async { Ok(()) })
                    })
                });
            }
        });

        let host = facet_host(vec![observer_facet, provider_facet]).await;
        // Upstream's trace pins the gate: the observation connects when the
        // observing facet activates, never at setup or at the provider's
        // spawn. The port's eager keyed start delivers inside that
        // activation, before the onActivate callbacks — upstream's subscribe
        // chain resolves after them — so the port's trace swaps the last
        // two entries.
        assert_eq!(
            *trace.borrow(),
            vec!["activate provider", "observe one", "activate observer"]
        );

        let services = host.services().expect("assembled");
        let remote =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![keyed.clone()],
                transport: create_loopback_service_transport(&services),
                bound: true,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: None,
            })
            .expect("binding");
        let remote_values = Rc::new(RefCell::new(Vec::<String>::new()));
        remote
            .observe(&keyed, {
                let remote_values = remote_values.clone();
                Rc::new(move |view: ServiceView, context: Context| {
                    if let Ok(Ok(Some(value))) =
                        pi_chord::future::drive_once(view.call("read", vec![], context))
                    {
                        remote_values.borrow_mut().push(read_text(Some(value)));
                    }
                })
            })
            .expect("observe lands");
        remote
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(*remote_values.borrow(), vec!["one".to_string()]);

        remote
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn terminates_the_host_when_keyed_replacement_publication_fails() {
    rt().block_on(async {
        let keyed = keyed_service();

        let host = facet_host(vec![keyed_spawner_facet(
            "failing-keyed-provider",
            &keyed,
            "A",
        )])
        .await;
        let services = host.services().expect("assembled");
        let subscription = services
            .subscribe(
                keyed.id.as_str(),
                ServiceMode::Keyed,
                Rc::new(|update: &ServiceProviderUpdate, _context: &Context| {
                    if matches!(update, ServiceProviderUpdate::Spawned { .. }) {
                        panic!("spawn publication failed");
                    }
                }),
            )
            .expect("subscribe lands");
        (subscription.activate)().expect("activate lands");

        let error = host
            .reload(vec![keyed_spawner_facet(
                "failing-keyed-provider",
                &keyed,
                "B",
            )])
            .await
            .expect_err("the publication failure terminates the host");
        assert!(error_message(&error).contains("Facet reload failed after cutover"));
        let error = host
            .reload(vec![])
            .await
            .expect_err("the dead host rejects reloads");
        assert!(error_message(&error).contains("cannot reload while dead"));
        let error = services
            .use_service(&keyed)
            .expect_err("the disposed provider rejects use");
        assert!(error_message(&error).contains("Remote service provider is disposed"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn terminates_the_host_when_keyed_retirement_publication_fails() {
    rt().block_on(async {
        let keyed = keyed_service();

        let host = facet_host(vec![keyed_spawner_facet(
            "failing-keyed-retirement-provider",
            &keyed,
            "A",
        )])
        .await;
        let services = host.services().expect("assembled");
        let subscription = services
            .subscribe(
                keyed.id.as_str(),
                ServiceMode::Keyed,
                Rc::new(|update: &ServiceProviderUpdate, _context: &Context| {
                    if matches!(update, ServiceProviderUpdate::Closed { .. }) {
                        panic!("close publication failed");
                    }
                }),
            )
            .expect("subscribe lands");
        (subscription.activate)().expect("activate lands");

        let error = host
            .reload(vec![keyed_spawner_facet(
                "failing-keyed-retirement-provider",
                &keyed,
                "B",
            )])
            .await
            .expect_err("the publication failure terminates the host");
        assert!(error_message(&error).contains("Facet reload failed after cutover"));
        let error = host
            .reload(vec![])
            .await
            .expect_err("the dead host rejects reloads");
        assert!(error_message(&error).contains("cannot reload while dead"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

/// A process-local keyed implementation: a `read` method plus the
/// `metadata` map as a local value member, upstream's object literal.
fn local_keyed_implementation(value: &'static str) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.method(
        "read",
        sync_method(move |_args: Vec<JsonValue>, _context: &Context| Ok(Some(js(value)))),
    );
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("value".to_string(), value.to_string());
    implementation.value("metadata", Rc::new(metadata));
    implementation
}

#[test]
fn keeps_unrestricted_local_keyed_services_process_local_across_provider_reloads() {
    rt().block_on(async {
        let local_keyed = local_keyed_service();
        let observed: Rc<RefCell<Vec<(ServiceView, Context)>>> = Rc::new(RefCell::new(Vec::new()));

        let consumer = facet("local-keyed-consumer", {
            let local_keyed = local_keyed.clone();
            let observed = observed.clone();
            move |env| {
                let handler: KeyedViewHandler = {
                    let observed = observed.clone();
                    Rc::new(move |view: ServiceView, context: Context| {
                        observed.borrow_mut().push((view, context));
                    })
                };
                env.observe_service(&local_keyed, handler)
                    .expect("observe lands");
            }
        });
        let provider_for = |value: &'static str| -> FacetDef {
            facet("local-keyed-provider", {
                let local_keyed = local_keyed.clone();
                move |env| {
                    let values = env.provide_many(&local_keyed).expect("provide_many lands");
                    env.on_activate({
                        Box::new(move |_env: &mut FacetEnvironment| {
                            values
                                .spawn("current", local_keyed_implementation(value))
                                .expect("spawn lands");
                            boxed(async { Ok(()) })
                        })
                    });
                }
            })
        };

        let host = facet_host(vec![consumer, provider_for("A")]).await;
        assert_eq!(observed.borrow().len(), 1);
        let (first_view, first_context) = observed.borrow()[0].clone();
        let read = first_view
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));
        let metadata = metadata_value(&first_view);
        assert_eq!(metadata.as_deref(), Some("A"));

        let services = host.services().expect("assembled");
        assert!(
            !services
                .catalogue()
                .iter()
                .any(|entry| entry.service_id == local_keyed.id),
            "the process-local keyed service stays out of the catalogue"
        );
        let error = services
            .use_service(&local_keyed)
            .expect_err("process-local services reject use");
        assert!(error_message(&error).contains("process-local"));

        host.reload(vec![provider_for("B")])
            .await
            .unwrap_or_else(|e| panic!("reload: {e}"));
        assert_eq!(observed.borrow().len(), 2);
        assert!(
            first_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted()),
            "the replaced generation's observation context is aborted"
        );
        let error = first_view
            .call("read", vec![], background_context())
            .await
            .expect_err("the closed observation rejects");
        assert!(error_message(&error).contains("observation is closed"));
        let (second_view, second_context) = observed.borrow()[1].clone();
        let read = second_view
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("B")));
        let metadata = metadata_value(&second_view);
        assert_eq!(metadata.as_deref(), Some("B"));

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert!(
            second_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted()),
            "the surviving observation closes on host disposal"
        );
    });
}

fn metadata_value(view: &ServiceView) -> Option<String> {
    view.with_value("metadata", |data| {
        data.downcast_ref::<std::collections::HashMap<String, String>>()
            .expect("the metadata map")
            .get("value")
            .cloned()
    })
    .expect("the value member is reachable")
}

#[test]
fn keeps_remotely_exposable_local_state_replicas_stable_across_provider_reloads() {
    rt().block_on(async {
        let watched = watched_service();
        let sources: Rc<RefCell<Vec<MutableReplicatedState>>> = Rc::new(RefCell::new(Vec::new()));
        let revisions = Rc::new(RefCell::new(Vec::<JsonValue>::new()));
        let retained_handle = Rc::new(RefCell::new(None::<ServiceView>));

        let consumer = facet("state-consumer", {
            let watched = watched.clone();
            let revisions = revisions.clone();
            let retained_handle = retained_handle.clone();
            move |env| {
                let handle = env.use_service(&watched).expect("use lands");
                *retained_handle.borrow_mut() = Some(handle.clone());
                env.on_activate({
                    let revisions = revisions.clone();
                    Box::new(move |env: &mut FacetEnvironment| {
                        let state = handle.state("state").expect("the member is state");
                        if let Ok(unsubscribe) = state.subscribe({
                            let revisions = revisions.clone();
                            Rc::new(
                                move |value: &JsonValue,
                                      _context: &Context,
                                      _delivery: &pi_chord::types::ReplicatedStateDelivery| {
                                    if let JsonValue::Object(object) = value
                                        && let Some(field) = object.get("value")
                                    {
                                        revisions.borrow_mut().push(field.clone());
                                    }
                                },
                            )
                        }) {
                            env.own(sync_disposal(move || {
                                unsubscribe();
                                Ok(())
                            }))
                            .expect("own lands");
                        }
                        boxed(async { Ok(()) })
                    })
                });
            }
        });
        let provider_for = |value: u64| -> FacetDef {
            facet("state-provider", {
                let watched = watched.clone();
                let sources = sources.clone();
                move |env| {
                    let state = env
                        .replicated_state(jo(vec![("value", number(value))]))
                        .expect("state lands");
                    sources.borrow_mut().push(state.clone());
                    let mut implementation = ServiceImplementation::new();
                    implementation.state("state", state);
                    env.provide(&watched, implementation)
                        .expect("provide lands");
                }
            })
        };

        let host = facet_host(vec![consumer, provider_for(1)]).await;
        let retained = retained_handle
            .borrow()
            .clone()
            .expect("the consumer's handle")
            .state("state")
            .expect("the member is state");
        let value = retained.value().unwrap_or_else(|e| panic!("state: {e}"));
        assert_eq!(value, Some(jo(vec![("value", number(1))])));
        assert_eq!(*revisions.borrow(), vec![number(1)]);

        host.reload(vec![provider_for(2)])
            .await
            .unwrap_or_else(|e| panic!("reload: {e}"));
        // The replica rides the facade's per-name member slot, and the
        // facade never changes across a reload, so the retained view is the
        // same handle upstream's `Object.is` pins; it hydrates the
        // replacement's state.
        let value = retained.value().unwrap_or_else(|e| panic!("state: {e}"));
        assert_eq!(value, Some(jo(vec![("value", number(2))])));
        assert_eq!(*revisions.borrow(), vec![number(1), number(2)]);

        // The retired source still publishes, upstream's direct state write;
        // its deactivated provider instance drops the batch.
        sources.borrow()[0].mutate(|tracker| {
            tracker
                .set(&[Seg::Key("value".to_string())], number(3))
                .expect("the write lands");
        });
        sources.borrow()[0]
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        let value = retained.value().unwrap_or_else(|e| panic!("state: {e}"));
        assert_eq!(value, Some(jo(vec![("value", number(2))])));
        assert_eq!(revisions.borrow().len(), 2);

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let error = retained
            .value()
            .expect_err("the dead facet's replica rejects");
        assert!(error_message(&error).contains("cannot be used while dead"));
    });
}

#[test]
fn provides_arbitrary_host_services_through_the_facet_graph() {
    rt().block_on(async {
        let host_values = host_values_service();
        let consumer = facet("host-service-consumer", {
            let host_values = host_values.clone();
            move |env| {
                let handle = env.use_service(&host_values).expect("use lands");
                // Upstream touches the implementation's `use` property during
                // setup; the value-member read hits the lifecycle gate.
                let error = handle
                    .with_value("use", |_data: &dyn std::any::Any| ())
                    .expect_err("setup-time access is gated");
                assert!(error_message(&error).contains("cannot be used while setting_up"));
                env.on_activate(Box::new(move |_env: &mut FacetEnvironment| {
                    // The handle is the port's view over the slot, not
                    // the implementation value itself, upstream's proxy
                    // identity check.
                    assert_eq!(read_string_member(&handle, "name"), "session");
                    assert_eq!(read_string_member(&handle, "use"), "host value");
                    boxed(async { Ok(()) })
                }));
            }
        });
        let provider = facet("host-service-provider", {
            let host_values = host_values.clone();
            move |env| {
                let mut implementation = ServiceImplementation::new();
                implementation.value("name", Rc::new("session".to_string()));
                implementation.value("use", Rc::new("host value".to_string()));
                env.provide(&host_values, implementation)
                    .expect("provide lands");
            }
        });

        let host = facet_host(vec![consumer, provider]).await;
        let services = host.services().expect("assembled");
        let error = services
            .use_service(&host_values)
            .expect_err("process-local services reject use");
        assert!(
            error_message(&error)
                .contains("Service test.experimental.host-values is process-local")
        );
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

fn read_string_member(handle: &ServiceView, member: &str) -> String {
    handle
        .with_value(member, |data| {
            data.downcast_ref::<String>()
                .expect("the value member carries a string")
                .clone()
        })
        .unwrap_or_default()
}

/// The pre-built namespace a source hands back on every open, carrying the
/// ready override upstream's `Object.assign(namespace, { ready })` makes:
/// the binding is unbound, so readiness rebinds first.
struct RebindingNamespace {
    binding: pi_chord::consumer::RemoteServiceBinding,
}

impl RemoteServices for RebindingNamespace {
    fn use_service(&self, service: &Service) -> Result<ServiceView, ChordError> {
        self.binding.use_service(service)
    }

    fn observe(
        &self,
        service: &Service,
        handler: KeyedViewHandler,
    ) -> Result<Unsubscribe, ChordError> {
        self.binding.observe(service, handler)
    }

    fn ready(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        let binding = self.binding.clone();
        boxed(async move {
            binding.rebind(true, context.clone()).await?;
            binding.ready(context).await
        })
    }

    fn dispose(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        self.binding.dispose(context)
    }
}

/// The service source over one pre-built namespace, upstream's mapped
/// `{ namespace, provider }` source objects.
struct NamespaceSource {
    provider: RemoteServiceProvider,
    namespace: pi_chord::consumer::RemoteServiceBinding,
}

impl RemoteServiceSource for NamespaceSource {
    fn accepts_unavailable_services(&self) -> bool {
        false
    }

    fn catalogue(
        &self,
        _context: Context,
    ) -> LocalBoxFuture<Result<Vec<ServiceCatalogueEntry>, ChordError>> {
        let entries = self.provider.catalogue().to_vec();
        boxed(std::future::ready(Ok(entries)))
    }

    fn open(&self, _options: RemoteServiceSourceOpenOptions) -> Rc<dyn RemoteServices> {
        Rc::new(RebindingNamespace {
            binding: self.namespace.clone(),
        })
    }
}

#[test]
fn combines_connected_services_and_facet_provided_services_in_one_host() {
    rt().block_on(async {
        let left = define_service("test.experimental.left-value").expect("not reserved");
        let right = define_service("test.experimental.right-value").expect("not reserved");
        let combined = define_service("test.experimental.combined-value").expect("not reserved");

        let left_provider =
            RemoteServiceProvider::new(vec![singleton_definition(left.clone())]).expect("provider");
        left_provider
            .provide(&left, read_implementation("left"))
            .expect("provide lands");
        let right_provider = RemoteServiceProvider::new(vec![singleton_definition(right.clone())])
            .expect("provider");
        right_provider
            .provide(&right, read_implementation("right"))
            .expect("provide lands");

        let left_namespace =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![left.clone()],
                transport: create_loopback_service_transport(&left_provider),
                bound: false,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: None,
            })
            .expect("binding");
        let right_namespace =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![right.clone()],
                transport: create_loopback_service_transport(&right_provider),
                bound: false,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: None,
            })
            .expect("binding");

        let combined_facet = facet("combined", {
            let left = left.clone();
            let right = right.clone();
            let combined = combined.clone();
            move |env| {
                let left_handle = env.use_service(&left).expect("use lands");
                let right_handle = env.use_service(&right).expect("use lands");
                let mut implementation = ServiceImplementation::new();
                implementation.method(
                    "read",
                    Rc::new(move |_args: Vec<JsonValue>, context: Context| {
                        let left_handle = left_handle.clone();
                        let right_handle = right_handle.clone();
                        boxed(async move {
                            let left_read =
                                left_handle.call("read", vec![], context.clone()).await?;
                            let right_read = right_handle.call("read", vec![], context).await?;
                            Ok(Some(js(&format!(
                                "{} {}",
                                read_text(left_read),
                                read_text(right_read)
                            ))))
                        })
                    }),
                );
                env.provide(&combined, implementation)
                    .expect("provide lands");
            }
        });

        let host = create_facet_host(FacetKernelOptions {
            facets: vec![combined_facet],
            service_sources: vec![
                Rc::new(NamespaceSource {
                    provider: left_provider.clone(),
                    namespace: left_namespace.clone(),
                }),
                Rc::new(NamespaceSource {
                    provider: right_provider.clone(),
                    namespace: right_namespace.clone(),
                }),
            ],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the host assembles");

        let services = host.services().expect("assembled");
        let implementation = services.use_service(&combined).expect("use lands");
        let slot = ServiceSlot::new(combined.id.as_str());
        slot.bind(ServiceTarget::Local(implementation.clone()));
        let view = slot.view(allow_access());
        let read = view
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("left right")));

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        left_namespace
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        right_namespace
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        left_provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        right_provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

/// The per-generation namespace one source open builds: readiness rebinds
/// the unbound binding first, and disposal counts the generations that
/// tear down, upstream's open-time wrapper.
struct GenerationNamespace {
    binding: pi_chord::consumer::RemoteServiceBinding,
    disposed: Rc<Cell<u32>>,
}

impl RemoteServices for GenerationNamespace {
    fn use_service(&self, service: &Service) -> Result<ServiceView, ChordError> {
        self.binding.use_service(service)
    }

    fn observe(
        &self,
        service: &Service,
        handler: KeyedViewHandler,
    ) -> Result<Unsubscribe, ChordError> {
        self.binding.observe(service, handler)
    }

    fn ready(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        let binding = self.binding.clone();
        boxed(async move {
            binding.rebind(true, context.clone()).await?;
            binding.ready(context).await
        })
    }

    fn dispose(&self, context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        let binding = self.binding.clone();
        let disposed = self.disposed.clone();
        boxed(async move {
            disposed.set(disposed.get() + 1);
            binding.dispose(context).await
        })
    }
}

/// The catalogue-following source, upstream's `source` object whose open
/// builds one binding per host generation over the current provider.
struct GenerationSource {
    current: Rc<RefCell<RemoteServiceProvider>>,
    opened: Rc<Cell<u32>>,
    disposed: Rc<Cell<u32>>,
}

impl RemoteServiceSource for GenerationSource {
    fn accepts_unavailable_services(&self) -> bool {
        false
    }

    fn catalogue(
        &self,
        _context: Context,
    ) -> LocalBoxFuture<Result<Vec<ServiceCatalogueEntry>, ChordError>> {
        let entries = self.current.borrow().catalogue().to_vec();
        boxed(std::future::ready(Ok(entries)))
    }

    fn open(&self, options: RemoteServiceSourceOpenOptions) -> Rc<dyn RemoteServices> {
        self.opened.set(self.opened.get() + 1);
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: options.services,
                transport: create_loopback_service_transport(&self.current.borrow()),
                bound: false,
                on_error: options.on_error,
                assert_access: Some(options.assert_access),
            })
            .expect("the source opens its binding");
        Rc::new(GenerationNamespace {
            binding,
            disposed: self.disposed.clone(),
        })
    }
}

#[test]
fn reopens_source_bindings_from_changed_catalogues_for_a_replacement_generation() {
    rt().block_on(async {
        let left = define_service("test.experimental.left-value").expect("not reserved");
        let right = define_service("test.experimental.right-value").expect("not reserved");

        let left_provider =
            RemoteServiceProvider::new(vec![singleton_definition(left.clone())]).expect("provider");
        left_provider
            .provide(&left, read_implementation("left"))
            .expect("provide lands");
        let right_provider = RemoteServiceProvider::new(vec![singleton_definition(right.clone())])
            .expect("provider");
        right_provider
            .provide(&right, read_implementation("right"))
            .expect("provide lands");

        let current = Rc::new(RefCell::new(left_provider.clone()));
        let opened = Rc::new(Cell::new(0u32));
        let disposed = Rc::new(Cell::new(0u32));
        let source = Rc::new(GenerationSource {
            current: current.clone(),
            opened: opened.clone(),
            disposed: disposed.clone(),
        });

        let values = Rc::new(RefCell::new(Vec::<String>::new()));
        let consumer_for = |facet_id: &'static str, service: &Service| -> FacetDef {
            let service = service.clone();
            let values = values.clone();
            facet(facet_id, move |env| {
                let handle = env.use_service(&service).expect("use lands");
                env.on_activate({
                    let values = values.clone();
                    Box::new(move |_env: &mut FacetEnvironment| {
                        let handle = handle.clone();
                        let values = values.clone();
                        boxed(async move {
                            let read = handle
                                .call("read", vec![], background_context())
                                .await
                                .unwrap_or_else(|e| panic!("read: {e}"));
                            values.borrow_mut().push(read_text(read));
                            Ok(())
                        })
                    })
                });
            })
        };

        let first = create_facet_host(FacetKernelOptions {
            facets: vec![consumer_for("left-consumer", &left)],
            service_sources: vec![source.clone()],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the first generation assembles");
        first
            .dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        *current.borrow_mut() = right_provider.clone();
        let second = create_facet_host(FacetKernelOptions {
            facets: vec![consumer_for("right-consumer", &right)],
            service_sources: vec![source.clone()],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the second generation assembles");
        second
            .dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        assert_eq!(
            *values.borrow(),
            vec!["left".to_string(), "right".to_string()]
        );
        assert_eq!(opened.get(), 2);
        assert_eq!(disposed.get(), 2);
        left_provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        right_provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

/// The generation loader upstream's `loader` object spells: each load
/// produces one provider facet generation named A then B, and the B
/// generation parks mid-activation until the case opens the gate.
struct GenerationLoader {
    trace: Rc<RefCell<Vec<String>>>,
    local_service: Service,
    remote_service: Service,
    generation: Rc<Cell<u32>>,
    gate: Rc<
        RefCell<
            Option<(
                tokio::sync::oneshot::Sender<()>,
                tokio::sync::oneshot::Receiver<()>,
            )>,
        >,
    >,
}

impl FacetLoader for GenerationLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let trace = self.trace.clone();
        let local = self.local_service.clone();
        let remote = self.remote_service.clone();
        let generation = self.generation.get();
        self.generation.set(generation + 1);
        let name = if generation == 0 { "A" } else { "B" };
        let gate = self.gate.clone();
        boxed(async move {
            trace.borrow_mut().push(format!("load {name}"));
            let provider = facet("provider", {
                let trace = trace.clone();
                let local = local.clone();
                let remote = remote.clone();
                let gate = gate.clone();
                move |env| {
                    trace.borrow_mut().push(format!("setup provider {name}"));
                    let implementation = read_implementation(name);
                    env.provide(&local, implementation.clone())
                        .expect("provide lands");
                    env.provide(&remote, implementation).expect("provide lands");
                    env.on_activate({
                        let trace = trace.clone();
                        let gate = gate.clone();
                        Box::new(move |_env: &mut FacetEnvironment| {
                            let trace = trace.clone();
                            let gate = gate.clone();
                            boxed(async move {
                                trace.borrow_mut().push(format!("activate provider {name}"));
                                if name == "B" {
                                    let gate = gate.borrow_mut().take();
                                    if let Some((started, can_continue)) = gate {
                                        let _ = started.send(());
                                        let _ = can_continue.await;
                                    }
                                }
                                Ok(())
                            })
                        })
                    });
                    env.on_deactivate({
                        let trace = trace.clone();
                        Box::new(move || {
                            let trace = trace.clone();
                            boxed(async move {
                                trace
                                    .borrow_mut()
                                    .push(format!("deactivate provider {name}"));
                                Ok(())
                            })
                        })
                    })
                    .expect("teardown lands");
                }
            });
            Ok(LoadedFacets {
                facets: vec![provider],
                dispose: {
                    let trace = trace.clone();
                    Box::new(move || {
                        let trace = trace.clone();
                        boxed(async move {
                            trace.borrow_mut().push(format!("unload {name}"));
                            Ok(())
                        })
                    })
                },
            })
        })
    }
}

#[test]
fn keeps_local_and_rpc_service_handles_stable_when_their_provider_facet_reloads() {
    rt().block_on(async {
        let local_service =
            define_local_service("test.experimental.local-generation-value").expect("not reserved");
        let remote_service =
            define_service("test.experimental.remote-generation-value").expect("not reserved");
        let trace = Rc::new(RefCell::new(Vec::<String>::new()));
        let local_handle = Rc::new(RefCell::new(None::<ServiceView>));

        let consumer = facet("consumer", {
            let local_service = local_service.clone();
            let trace = trace.clone();
            let local_handle = local_handle.clone();
            move |env| {
                trace.borrow_mut().push("setup consumer".to_string());
                let handle = env.use_service(&local_service).expect("use lands");
                *local_handle.borrow_mut() = Some(handle.clone());
                env.on_activate({
                    let trace = trace.clone();
                    Box::new(move |_env: &mut FacetEnvironment| {
                        let handle = handle.clone();
                        let trace = trace.clone();
                        boxed(async move {
                            let read = handle
                                .call("read", vec![], background_context())
                                .await
                                .unwrap_or_else(|e| panic!("read: {e}"));
                            trace
                                .borrow_mut()
                                .push(format!("activate consumer:{}", read_text(read)));
                            Ok(())
                        })
                    })
                });
                env.on_deactivate({
                    let trace = trace.clone();
                    Box::new(move || {
                        let trace = trace.clone();
                        boxed(async move {
                            trace.borrow_mut().push("deactivate consumer".to_string());
                            Ok(())
                        })
                    })
                })
                .expect("teardown lands");
            }
        });

        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let (open_tx, open_rx) = tokio::sync::oneshot::channel::<()>();
        let generation_loader = Rc::new(GenerationLoader {
            trace: trace.clone(),
            local_service: local_service.clone(),
            remote_service: remote_service.clone(),
            generation: Rc::new(Cell::new(0u32)),
            gate: Rc::new(RefCell::new(Some((started_tx, open_rx)))),
        });

        let loaded_a = generation_loader.load().await.expect("load lands");
        let mut initial = vec![consumer];
        initial.extend(loaded_a.facets);
        let host = create_facet_host(FacetKernelOptions {
            facets: initial,
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the graph is valid");
        let original_local = local_handle.borrow().clone().expect("handle");
        let services = host.services().expect("assembled");
        let remote_services =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![remote_service.clone()],
                transport: create_loopback_service_transport(&services),
                bound: true,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: None,
            })
            .expect("binding");
        let original_remote = remote_services
            .use_service(&remote_service)
            .expect("use lands");
        remote_services
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        let read = original_local
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));
        let read = original_remote
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));

        // A replacement that drops the remote provision fails its shape
        // check before cutover, and the old generation keeps serving.
        let invalid = facet("provider", {
            let local = local_service.clone();
            move |env| {
                env.provide(&local, read_implementation("invalid"))
                    .expect("provide lands");
            }
        });
        let error = host
            .reload(vec![invalid])
            .await
            .expect_err("the shape change rejects");
        assert!(error_message(&error).contains(
            "Reloaded facet provider must preserve its service requirements and provisions"
        ));
        let read = original_local
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));
        let read = original_remote
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));

        let loaded_b = generation_loader.load().await.expect("load lands");
        let gate_window = async {
            started_rx.await.expect("the replacement starts");
            // The parked replacement keeps the old generation live, upstream's
            // window between `replacementStarted` and the gate opening.
            let read = original_local
                .call("read", vec![], background_context())
                .await
                .unwrap_or_else(|e| panic!("read: {e}"));
            assert_eq!(read, Some(js("A")));
            let read = original_remote
                .call("read", vec![], background_context())
                .await
                .unwrap_or_else(|e| panic!("read: {e}"));
            assert_eq!(read, Some(js("A")));
            let _ = open_tx.send(());
        };
        let (reload_result, ()) = tokio::join!(host.reload(loaded_b.facets), gate_window);
        reload_result.unwrap_or_else(|e| panic!("reload: {e}"));
        (loaded_a.dispose)()
            .await
            .unwrap_or_else(|e| panic!("unload: {e}"));

        assert!(
            local_handle
                .borrow()
                .as_ref()
                .expect("handle")
                .same_handle(&original_local),
            "the local handle survives the reload"
        );
        assert!(
            remote_services
                .use_service(&remote_service)
                .expect("use lands")
                .same_handle(&original_remote),
            "the RPC handle survives the reload"
        );
        let read = original_local
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("B")));
        let read = original_remote
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("B")));

        remote_services
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        (loaded_b.dispose)()
            .await
            .unwrap_or_else(|e| panic!("unload: {e}"));
        assert_eq!(
            *trace.borrow(),
            vec![
                "load A",
                "setup consumer",
                "setup provider A",
                "activate provider A",
                "activate consumer:A",
                "load B",
                "setup provider B",
                "activate provider B",
                "deactivate provider A",
                "unload A",
                "deactivate consumer",
                "deactivate provider B",
                "unload B"
            ]
        );
    });
}

#[test]
fn rejects_remote_singleton_member_shape_changes_before_reload_cutover() {
    rt().block_on(async {
        let remote =
            define_service("test.experimental.remote-generation-value").expect("not reserved");
        let provider_disposed = Rc::new(Cell::new(false));
        let retained_handle = Rc::new(RefCell::new(None::<ServiceView>));

        let consumer = facet("shape-consumer", {
            let remote = remote.clone();
            let retained_handle = retained_handle.clone();
            move |env| {
                let handle = env.use_service(&remote).expect("use lands");
                *retained_handle.borrow_mut() = Some(handle);
            }
        });
        let provider = facet("shape-provider", {
            let remote = remote.clone();
            let provider_disposed = provider_disposed.clone();
            move |env| {
                env.provide(&remote, read_implementation("A"))
                    .expect("provide lands");
                env.on_deactivate({
                    let provider_disposed = provider_disposed.clone();
                    Box::new(move || {
                        provider_disposed.set(true);
                        boxed(async { Ok(()) })
                    })
                })
                .expect("teardown lands");
            }
        });
        let host = facet_host(vec![consumer, provider]).await;

        let replacement = facet("shape-provider", {
            let remote = remote.clone();
            move |env| {
                let mut implementation = ServiceImplementation::new();
                implementation.method(
                    "renamed",
                    sync_method(|_args: Vec<JsonValue>, _context: &Context| Ok(Some(js("B")))),
                );
                env.provide(&remote, implementation).expect("provide lands");
            }
        });
        let error = host
            .reload(vec![replacement])
            .await
            .expect_err("the member shape change rejects");
        assert!(error_message(&error).contains("replacement must preserve its member shape"));
        assert!(!provider_disposed.get(), "cutover never ran");
        let retained = retained_handle.borrow().clone().expect("handle");
        let read = retained
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert!(
            provider_disposed.get(),
            "the live provider retires on disposal"
        );
    });
}

#[test]
fn terminates_the_host_when_old_cleanup_fails_after_cutover() {
    rt().block_on(async {
        let remote =
            define_service("test.experimental.remote-generation-value").expect("not reserved");
        let provider_for = |name: &'static str, fail_cleanup: bool| -> FacetDef {
            facet("cleanup-provider", {
                let remote = remote.clone();
                move |env| {
                    env.provide(&remote, read_implementation(name))
                        .expect("provide lands");
                    if fail_cleanup {
                        env.on_deactivate(Box::new(|| {
                            boxed(async { Err(ChordError::Message("cleanup failed".to_string())) })
                        }))
                        .expect("teardown lands");
                    }
                }
            })
        };

        let host = facet_host(vec![provider_for("A", true)]).await;

        let error = host
            .reload(vec![provider_for("B", false)])
            .await
            .expect_err("the cleanup failure terminates the host");
        assert!(error_message(&error).contains("Facet reload failed after cutover"));
        let error = host
            .reload(vec![])
            .await
            .expect_err("the dead host rejects reloads");
        assert!(error_message(&error).contains("cannot reload while dead"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn cleans_failed_candidate_activation_in_reverse_dependency_order() {
    rt().block_on(async {
        let remote =
            define_service("test.experimental.remote-generation-value").expect("not reserved");
        let trace = Rc::new(RefCell::new(Vec::<String>::new()));

        let provider_for = |name: &'static str| -> FacetDef {
            facet("ordered-provider", {
                let remote = remote.clone();
                let trace = trace.clone();
                move |env| {
                    env.provide(&remote, read_implementation(name))
                        .expect("provide lands");
                    env.on_activate({
                        let trace = trace.clone();
                        Box::new(move |_env: &mut FacetEnvironment| {
                            let trace = trace.clone();
                            boxed(async move {
                                trace.borrow_mut().push(format!("activate provider {name}"));
                                Ok(())
                            })
                        })
                    })
                    .expect("onActivate lands");
                    env.on_deactivate({
                        let trace = trace.clone();
                        Box::new(move || {
                            let trace = trace.clone();
                            boxed(async move {
                                trace
                                    .borrow_mut()
                                    .push(format!("deactivate provider {name}"));
                                Ok(())
                            })
                        })
                    })
                    .expect("teardown lands");
                }
            })
        };
        let consumer_for = |name: &'static str, fail: bool| -> FacetDef {
            facet("ordered-consumer", {
                let remote = remote.clone();
                let trace = trace.clone();
                move |env| {
                    env.use_service(&remote).expect("use lands");
                    env.on_activate({
                        let trace = trace.clone();
                        Box::new(move |_env: &mut FacetEnvironment| {
                            let trace = trace.clone();
                            boxed(async move {
                                trace.borrow_mut().push(format!("activate consumer {name}"));
                                if fail {
                                    return Err(ChordError::Message(
                                        "consumer activation failed".to_string(),
                                    ));
                                }
                                Ok(())
                            })
                        })
                    })
                    .expect("onActivate lands");
                    env.on_deactivate({
                        let trace = trace.clone();
                        Box::new(move || {
                            let trace = trace.clone();
                            boxed(async move {
                                trace
                                    .borrow_mut()
                                    .push(format!("deactivate consumer {name}"));
                                Ok(())
                            })
                        })
                    })
                    .expect("teardown lands");
                }
            })
        };

        let host = facet_host(vec![consumer_for("A", false), provider_for("A")]).await;
        trace.borrow_mut().clear();

        let error = host
            .reload(vec![consumer_for("B", true), provider_for("B")])
            .await
            .expect_err("the activation failure rejects");
        assert!(error_message(&error).contains("consumer activation failed"));
        assert_eq!(
            *trace.borrow(),
            vec![
                "activate provider B",
                "activate consumer B",
                "deactivate consumer B",
                "deactivate provider B"
            ]
        );
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn terminates_the_host_when_replacement_publication_fails_after_cutover() {
    rt().block_on(async {
        let remote =
            define_service("test.experimental.remote-generation-value").expect("not reserved");
        let provider_for = |name: &'static str| -> FacetDef {
            facet("publication-provider", {
                let remote = remote.clone();
                move |env| {
                    env.provide(&remote, read_implementation(name))
                        .expect("provide lands");
                }
            })
        };

        let host = facet_host(vec![provider_for("A")]).await;
        let services = host.services().expect("assembled");
        let subscription = services
            .subscribe(
                remote.id.as_str(),
                ServiceMode::Singleton,
                Rc::new(|_update: &ServiceProviderUpdate, _context: &Context| {
                    panic!("publication failed");
                }),
            )
            .expect("subscribe lands");
        (subscription.activate)().expect("activate lands");

        let error = host
            .reload(vec![provider_for("B")])
            .await
            .expect_err("the publication failure terminates the host");
        assert!(error_message(&error).contains("Facet reload failed after cutover"));
        let error = host
            .reload(vec![])
            .await
            .expect_err("the dead host rejects reloads");
        assert!(error_message(&error).contains("cannot reload while dead"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn keeps_the_old_generation_active_when_replacement_activation_fails_before_cutover() {
    rt().block_on(async {
        let remote =
            define_service("test.experimental.remote-generation-value").expect("not reserved");
        let trace = Rc::new(RefCell::new(Vec::<String>::new()));
        let retained_handle = Rc::new(RefCell::new(None::<ServiceView>));

        let consumer = facet("terminal-consumer", {
            let remote = remote.clone();
            let retained_handle = retained_handle.clone();
            let trace = trace.clone();
            move |env| {
                let handle = env.use_service(&remote).expect("use lands");
                *retained_handle.borrow_mut() = Some(handle);
                env.on_deactivate({
                    let trace = trace.clone();
                    Box::new(move || {
                        let trace = trace.clone();
                        boxed(async move {
                            trace.borrow_mut().push("deactivate consumer".to_string());
                            Ok(())
                        })
                    })
                })
                .expect("teardown lands");
            }
        });
        let provider_for = |name: &'static str, fail: bool| -> FacetDef {
            facet("terminal-provider", {
                let remote = remote.clone();
                let trace = trace.clone();
                move |env| {
                    env.provide(&remote, read_implementation(name))
                        .expect("provide lands");
                    env.on_activate({
                        let trace = trace.clone();
                        Box::new(move |_env: &mut FacetEnvironment| {
                            let trace = trace.clone();
                            boxed(async move {
                                trace.borrow_mut().push(format!("activate {name}"));
                                if fail {
                                    return Err(ChordError::Message(
                                        "replacement activation failed".to_string(),
                                    ));
                                }
                                Ok(())
                            })
                        })
                    })
                    .expect("onActivate lands");
                    env.on_deactivate({
                        let trace = trace.clone();
                        Box::new(move || {
                            let trace = trace.clone();
                            boxed(async move {
                                trace.borrow_mut().push(format!("deactivate {name}"));
                                Ok(())
                            })
                        })
                    })
                    .expect("teardown lands");
                }
            })
        };

        let host = facet_host(vec![consumer, provider_for("A", false)]).await;
        let retained = retained_handle.borrow().clone().expect("handle");
        let read = retained
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));

        let error = host
            .reload(vec![provider_for("B", true)])
            .await
            .expect_err("the activation failure rejects");
        assert!(error_message(&error).contains("replacement activation failed"));
        assert_eq!(
            *trace.borrow(),
            vec!["activate A", "activate B", "deactivate B"]
        );
        let read = retained
            .call("read", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(read, Some(js("A")));
        host.reload(vec![])
            .await
            .unwrap_or_else(|e| panic!("reload: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert_eq!(
            *trace.borrow(),
            vec![
                "activate A",
                "activate B",
                "deactivate B",
                "deactivate consumer",
                "deactivate A"
            ]
        );
    });
}

#[test]
fn fences_generation_construction_and_reloads() {
    let rt = rt();
    rt.block_on(async {
        let source = source_service();
        let noop = |env: &mut FacetEnvironment| {
            let _ = env;
        };

        // Facet IDs must be unique and non-empty within a generation.
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![facet("twin", noop), facet("twin", noop)],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("duplicate facet IDs reject");
        assert!(error_message(&error).contains("must be unique within a generation"));
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![facet("", noop)],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("an empty facet ID rejects");
        assert!(error_message(&error).contains("Facet ID must not be empty"));

        // A live host rejects reloads that break those rules or touch unknown
        // facets.
        let provider = facet("provider", {
            let source = source.clone();
            move |env| {
                env.provide(&source, read_implementation("value"))
                    .expect("provide lands");
            }
        });
        let consumer = facet("consumer", {
            let source = source.clone();
            move |env| {
                let _handle = env.use_service(&source).expect("use lands");
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![provider, consumer],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("host: {e}"));
        assert!(format!("{host:?}").starts_with("FacetHost"));
        let options = FacetKernelOptions {
            facets: Vec::new(),
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        };
        assert!(format!("{options:?}").contains("facets"));

        let error = host
            .reload(vec![facet("unknown", noop)])
            .await
            .expect_err("an unknown facet rejects the reload");
        assert!(error_message(&error).contains("is not active"));
        let error = host
            .reload(vec![facet("twin", noop), facet("twin", noop)])
            .await
            .expect_err("a duplicate reload rejects");
        assert!(error_message(&error).contains("must be unique"));
        let error = host
            .reload(vec![facet("", noop)])
            .await
            .expect_err("an empty reload ID rejects");
        assert!(error_message(&error).contains("must not be empty"));

        // Disposal settles once; a dead host accepts the second disposal and
        // rejects further service access.
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("second dispose: {e}"));
        let error = host
            .services()
            .expect_err("a disposed host has no provider");
        assert!(error_message(&error).contains("not assembled"));
    });
}

#[test]
fn keyed_spawners_fence_their_instances() {
    let rt = rt();
    rt.block_on(async {
        let dialogs = local_keyed_service();

        // An empty instance key rejects at activation.
        let empty = facet("empty-key", {
            let dialogs = dialogs.clone();
            move |env| {
                let values = env.provide_many(&dialogs).expect("provide_many lands");
                env.on_activate(Box::new(move |_env: &mut FacetEnvironment| {
                    let error = values.spawn("", read_implementation("value"));
                    let error = error.err().expect("an empty key rejects");
                    assert!(error_message(&error).contains("key must not be empty"));
                    boxed(async { Ok(()) })
                }));
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![empty],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("host: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        // A repeated live key rejects.
        let duplicate = facet("duplicate-key", {
            let dialogs = dialogs.clone();
            move |env| {
                let values = env.provide_many(&dialogs).expect("provide_many lands");
                env.on_activate(Box::new(move |_env: &mut FacetEnvironment| {
                    values
                        .spawn("current", read_implementation("value"))
                        .expect("spawn lands");
                    let error = values
                        .spawn("current", read_implementation("value"))
                        .err()
                        .expect("a repeated live key rejects");
                    assert!(
                        error_message(&error).contains("already has a live instance"),
                        "{error}"
                    );
                    boxed(async { Ok(()) })
                }));
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![duplicate],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("host: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn startup_failures_aggregate_with_cleanup_failures() {
    let rt = rt();
    rt.block_on(async {
        // A facet whose activation fails and whose teardown fails as well
        // aggregates both, upstream's aggregate over cleanup errors.
        let failing = facet("failing", |env| {
            env.on_activate(Box::new(|_env: &mut FacetEnvironment| {
                boxed(async { Err(ChordError::Message("activation failed".to_string())) })
            }));
            env.on_deactivate(Box::new(|| {
                boxed(async { Err(ChordError::Message("cleanup failed".to_string())) })
            }));
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![failing],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("a failing activation rejects the host");
        let message = error_message(&error);
        assert!(
            message.contains("startup and cleanup failed"),
            "the failures aggregate: {message}"
        );
    });
}

#[test]
fn facets_fence_their_setup_and_spawner_lifecycles() {
    rt().block_on(async {
        let singleton = define_service("test.fence1.svc").unwrap_or_else(|e| panic!("define: {e}"));
        let keyed = define_service("test.fence1.keyed").unwrap_or_else(|e| panic!("define: {e}"));
        let local_keyed = define_local_service("test.fence1.local-keyed")
            .unwrap_or_else(|e| panic!("define: {e}"));
        let _setup_read_error: Rc<RefCell<Option<ChordError>>> = Rc::new(RefCell::new(None));
        let spawner_cell: Rc<RefCell<Option<Rc<pi_chord::facets::host::StagedServiceSpawner>>>> =
            Rc::new(RefCell::new(None));
        let local_spawner_cell: Rc<
            RefCell<Option<Rc<pi_chord::facets::host::StagedServiceSpawner>>>,
        > = Rc::new(RefCell::new(None));
        let setup_error = Rc::new(RefCell::new(None::<ChordError>));
        let activation_error = Rc::new(RefCell::new(None::<ChordError>));
        let view_cell: Rc<RefCell<Option<ServiceView>>> = Rc::new(RefCell::new(None));

        let fencing = facet("fencing-facet", {
            let singleton = singleton.clone();
            let keyed = keyed.clone();
            let local_keyed = local_keyed.clone();
            let setup_read_error = setup_error.clone();
            let spawner_cell = spawner_cell.clone();
            let local_spawner_cell = local_spawner_cell.clone();
            let activation_error = activation_error.clone();
            let view_cell = view_cell.clone();
            move |env| {
                // The environment and spawner spell debug surfaces.
                assert!(format!("{env:?}").contains("FacetEnvironment"));
                // A view captured during setup rejects reads until the facet
                // activates.
                let view = env
                    .use_service(&singleton)
                    .unwrap_or_else(|e| panic!("use: {e}"));
                if let Err(error) = view.state("state") {
                    *setup_read_error.borrow_mut() = Some(error);
                } else {
                    panic!("a setup-time read rejects");
                }
                *view_cell.borrow_mut() = Some(view);
                let spawner = env.provide_many(&keyed).expect("provide_many lands");
                assert!(format!("{spawner:?}").contains("test.fence1.keyed"));
                *spawner_cell.borrow_mut() = Some(spawner);
                let local_spawner = env
                    .provide_many(&local_keyed)
                    .expect("local provide_many lands");
                *local_spawner_cell.borrow_mut() = Some(local_spawner);
                let mut stateful = ServiceImplementation::new();
                stateful.state("state", MutableReplicatedState::new(js("A")));
                stateful.method(
                    "read",
                    sync_method(|_args: Vec<JsonValue>, _context: &Context| Ok(Some(js("A")))),
                );
                env.provide(&singleton, stateful).expect("provide lands");
                let late_service = singleton.clone();
                env.on_activate({
                    let env_error = activation_error.clone();
                    Box::new(move |env: &mut FacetEnvironment| {
                        // Providing during activation is fenced.
                        if let Err(error) = env.provide(&late_service, read_implementation("late"))
                        {
                            *env_error.borrow_mut() = Some(error);
                        }
                        boxed(async { Ok(()) })
                    })
                });
            }
        });
        let host = facet_host(vec![fencing]).await;

        let error = setup_error
            .borrow()
            .clone()
            .expect("the setup-time read rejected");
        assert!(
            error_message(&error).contains("cannot be used while setting_up"),
            "setup-time read: {error}"
        );
        let error = activation_error
            .borrow()
            .clone()
            .expect("the activation fence fires");
        assert!(
            error_message(&error).contains("can provide services only during setup"),
            "activation provide: {error}"
        );

        // The captured view works after activation.
        let view = view_cell.borrow().clone().expect("the view is captured");
        assert!(view.state("state").is_ok());

        // The host connected the spawner, so connecting again rejects; the
        // local spawner's registry fences empty and duplicate keys, and
        // closing twice settles twice.
        let spawner = spawner_cell.borrow().clone().expect("spawner captured");
        let error = spawner
            .connect(|_key, _implementation| {
                let close: Rc<dyn Fn() -> Result<(), ChordError>> = Rc::new(|| Ok(()));
                Ok(close)
            })
            .expect_err("the spawner is already connected");
        assert!(error_message(&error).contains("already connected"));

        let local_spawner = local_spawner_cell
            .borrow()
            .clone()
            .expect("local spawner captured");
        let close = local_spawner
            .spawn("dialog", read_implementation("L"))
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        close().unwrap_or_else(|e| panic!("close: {e}"));
        close().unwrap_or_else(|e| panic!("close again: {e}"));
        let error = local_spawner
            .spawn("", read_implementation("L"))
            .err()
            .expect("an empty key rejects");
        assert!(
            error_message(&error).contains("key must not be empty"),
            "empty key: {error}"
        );
        let first = local_spawner
            .spawn("dup", read_implementation("L"))
            .unwrap_or_else(|e| panic!("spawn dup: {e}"));
        let error = local_spawner
            .spawn("dup", read_implementation("L"))
            .err()
            .expect("a duplicate key rejects");
        assert!(
            error_message(&error).contains("already has a live instance"),
            "duplicate key: {error}"
        );
        first().unwrap_or_else(|e| panic!("close dup: {e}"));

        // After disposal the spawner's lifecycle gate rejects spawns.
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let error = local_spawner
            .spawn("late", read_implementation("L"))
            .err()
            .expect("a disposed facet rejects spawns");
        assert!(
            error_message(&error).contains("can spawn service instances only while active"),
            "late spawn: {error}"
        );
    });
}

#[test]
fn facet_loaders_aggregate_and_replay_their_failures() {
    rt().block_on(async {
        // A failure on the first loader propagates plainly.
        let plain_loader =
            pi_chord::api::combine_facet_loaders(vec![Rc::new(MessageFailingLoader {
                message: "first loader failed",
            })]);
        let error = plain_loader
            .load()
            .await
            .expect_err("the combined loader fails");
        assert!(error_message(&error).contains("first loader failed"));

        // A failing disposal from an earlier loader folds into the
        // aggregate.
        let aggregate_loader = pi_chord::api::combine_facet_loaders(vec![
            Rc::new(OkFailingDisposalLoader),
            Rc::new(MessageFailingLoader {
                message: "second loader failed",
            }),
        ]);
        let error = aggregate_loader
            .load()
            .await
            .expect_err("the aggregate forms");
        assert!(
            error_message(&error).contains("Facet loading and cleanup failed")
                && error_message(&error).contains("loader disposal failed"),
            "aggregate: {error}"
        );

        // The loaded generation's disposal collects failures and settles
        // once: two failing disposals aggregate.
        let disposal_loader = pi_chord::api::combine_facet_loaders(vec![
            Rc::new(LoadedWithFailingDisposal),
            Rc::new(LoadedWithFailingDisposal),
        ]);
        let loaded = disposal_loader
            .load()
            .await
            .unwrap_or_else(|e| panic!("load: {e}"));
        let error = (loaded.dispose)()
            .await
            .expect_err("the failing disposals aggregate");
        assert!(
            error_message(&error).contains("Failed to dispose loaded facets"),
            "disposal: {error}"
        );

        // The default reporter and identity facet constructors answer.
        let _ = pi_chord::api::default_on_error();
        let facet_def = facet("identity", |_env: &mut FacetEnvironment| ());
        assert_eq!(
            pi_chord::api::define_facet(facet_def.clone()).id,
            facet_def.id
        );
    });
}

#[test]
fn facet_hosts_gate_phase_and_release_external_sources() {
    rt().block_on(async {
        let external =
            define_service("test.phase.external").unwrap_or_else(|e| panic!("define: {e}"));

        // A kernel that never activated rejects disposal with its phase.
        let kernel = pi_chord::facets::host::FacetKernel::new(FacetKernelOptions {
            facets: Vec::new(),
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .unwrap_or_else(|e| panic!("kernel: {e}"));
        assert!(format!("{kernel:?}").contains("FacetKernel"));
        let error = kernel
            .dispose()
            .await
            .expect_err("an unactivated host rejects disposal");
        assert!(
            error_message(&error).contains("cannot be disposed while setup"),
            "phase gate: {error}"
        );

        // An external source binds requirements; a retained handle rejects
        // reads after the host dies, through the host's phase gate.
        let source_views: Rc<RefCell<Vec<ServiceView>>> = Rc::new(RefCell::new(Vec::new()));
        let source = ExternalSource {
            accepts_unavailable: false,
            fail_dispose: false,
            offered: vec![ServiceCatalogueEntry {
                service_id: external.id.clone(),
                mode: ServiceMode::Singleton,
            }],
            views: source_views.clone(),
        };
        let consumer = facet("external-consumer", {
            let external = external.clone();
            move |env| {
                env.use_service(&external)
                    .unwrap_or_else(|e| panic!("use: {e}"));
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![consumer],
            service_sources: vec![Rc::new(source)],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the external binding activates");
        let view = source_views.borrow()[0].clone();
        assert!(view.state("state").is_err());
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let error = view
            .state("state")
            .expect_err("the disposed host's phase gate rejects");
        assert!(
            error_message(&error).contains("cannot be used during dead"),
            "dead-phase read: {error}"
        );

        // A source whose binding fails disposal folds the failure into the
        // host's report.
        let deferred =
            define_service("test.phase.deferred").unwrap_or_else(|e| panic!("define: {e}"));
        let source = ExternalSource {
            accepts_unavailable: true,
            fail_dispose: true,
            offered: Vec::new(),
            views: Rc::new(RefCell::new(Vec::new())),
        };
        let consumer = facet("deferred-consumer", {
            let deferred = deferred.clone();
            move |env| {
                env.use_service(&deferred)
                    .unwrap_or_else(|e| panic!("use: {e}"));
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![consumer],
            service_sources: vec![Rc::new(source)],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("the deferred binding activates: {e}"));
        let error = host
            .dispose()
            .await
            .expect_err("the failing source disposal aggregates");
        assert!(
            error_message(&error).contains("external source disposal failed"),
            "source disposal: {error}"
        );

        // Two deferred sources for one requirement reject the graph.
        let consumer = facet("orphan-consumer", {
            let deferred = deferred.clone();
            move |env| {
                env.use_service(&deferred)
                    .unwrap_or_else(|e| panic!("use: {e}"));
            }
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![consumer],
            service_sources: vec![
                Rc::new(ExternalSource {
                    accepts_unavailable: true,
                    fail_dispose: false,
                    offered: Vec::new(),
                    views: Rc::new(RefCell::new(Vec::new())),
                }),
                Rc::new(ExternalSource {
                    accepts_unavailable: true,
                    fail_dispose: false,
                    offered: Vec::new(),
                    views: Rc::new(RefCell::new(Vec::new())),
                }),
            ],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("two deferred sources reject");
        assert!(
            error_message(&error).contains("more than one deferred source"),
            "deferred fence: {error}"
        );
    });
}

/// An external service source whose catalogue and disposal the case
/// controls. Its services interface serves local-backed views whose reads
/// ride the host's access gate; the views it hands out are retained in
/// `views` so a case can read one after the host dies.
struct ExternalSource {
    accepts_unavailable: bool,
    fail_dispose: bool,
    offered: Vec<ServiceCatalogueEntry>,
    views: Rc<RefCell<Vec<ServiceView>>>,
}

impl RemoteServiceSource for ExternalSource {
    fn accepts_unavailable_services(&self) -> bool {
        self.accepts_unavailable
    }

    fn catalogue(
        &self,
        _context: Context,
    ) -> LocalBoxFuture<Result<Vec<ServiceCatalogueEntry>, ChordError>> {
        boxed(std::future::ready(Ok(self.offered.clone())))
    }

    fn open(&self, options: RemoteServiceSourceOpenOptions) -> Rc<dyn RemoteServices> {
        Rc::new(ExternalServices {
            assert: options.assert_access,
            fail_dispose: self.fail_dispose,
            views: self.views.clone(),
        })
    }
}

struct ExternalServices {
    assert: pi_chord::handle::AssertAccess,
    fail_dispose: bool,
    views: Rc<RefCell<Vec<ServiceView>>>,
}

impl RemoteServices for ExternalServices {
    fn use_service(&self, service: &Service) -> Result<ServiceView, ChordError> {
        let slot = ServiceSlot::new(&service.id);
        slot.bind(ServiceTarget::Local(Rc::new(read_implementation("A"))));
        let view = slot.view(self.assert.clone());
        self.views.borrow_mut().push(view.clone());
        Ok(view)
    }

    fn observe(
        &self,
        _service: &Service,
        _handler: KeyedViewHandler,
    ) -> Result<Unsubscribe, ChordError> {
        Ok(Box::new(|| ()))
    }

    fn ready(&self, _context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        boxed(std::future::ready(Ok(())))
    }

    fn dispose(&self, _context: Context) -> LocalBoxFuture<Result<(), ChordError>> {
        if self.fail_dispose {
            boxed(std::future::ready(Err(ChordError::Message(
                "external source disposal failed".to_string(),
            ))))
        } else {
            boxed(std::future::ready(Ok(())))
        }
    }
}

#[test]
fn facet_graphs_fence_duplicate_provisions_and_requirement_mismatches() {
    rt().block_on(async {
        let mixed = define_service("test.graph.mixed").unwrap_or_else(|e| panic!("define: {e}"));

        // One service provided as both singleton and keyed rejects.
        let host_facet = facet("graph-singleton", {
            let mixed = mixed.clone();
            move |env| {
                env.provide(&mixed, read_implementation("A"))
                    .expect("provide lands");
            }
        });
        let keyed_facet = facet("mixed-keyed", {
            let mixed = mixed.clone();
            move |env| {
                env.provide_many(&mixed).expect("provide_many lands");
            }
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![host_facet, keyed_facet],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("the mixed provision rejects");
        assert!(
            error_message(&error).contains("provided as both singleton and keyed"),
            "mixed: {error}"
        );

        // A requirement whose mode mismatches the provision rejects.
        // (An external entry cannot double-provide: requirements resolve to
        // the external surface only for services no facet provides, so the
        // host-and-facet provision arm stays unreachable by construction.)
        let keyed_only =
            define_service("test.graph.keyed-only").unwrap_or_else(|e| panic!("define: {e}"));
        let consumer = facet("graph-consumer-2", {
            let keyed_only = keyed_only.clone();
            move |env| {
                env.use_service(&keyed_only)
                    .unwrap_or_else(|e| panic!("use: {e}"));
            }
        });
        let provider = facet("graph-provider-2", {
            let keyed_only = keyed_only.clone();
            move |env| {
                env.provide_many(&keyed_only).expect("provide_many lands");
            }
        });
        let error = create_facet_host(FacetKernelOptions {
            facets: vec![consumer, provider],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect_err("the mode mismatch rejects");
        assert!(
            error_message(&error).contains("but"),
            "mode mismatch: {error}"
        );

        // A facet requiring its own provision activates.
        let self_served =
            define_service("test.graph.self").unwrap_or_else(|e| panic!("define: {e}"));
        let self_facet = facet("self-serving", {
            let self_served = self_served.clone();
            move |env| {
                env.provide(&self_served, read_implementation("A"))
                    .expect("provide lands");
                env.use_service(&self_served)
                    .unwrap_or_else(|e| panic!("use: {e}"));
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![self_facet],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("self-requirement activates: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

/// A loader whose loaded generation's teardown fails, the fixture the
/// disposal-aggregation case rides on.
struct OkFailingDisposalLoader;

impl FacetLoader for OkFailingDisposalLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let failing: pi_chord::handle::Disposal = Box::new(|| {
            boxed(async { Err(ChordError::Message("loader disposal failed".to_string())) })
        });
        boxed(std::future::ready(Ok(LoadedFacets {
            facets: Vec::new(),
            dispose: failing,
        })))
    }
}

/// A loader that fails its load with the message the case pins.
struct MessageFailingLoader {
    message: &'static str,
}

impl FacetLoader for MessageFailingLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        boxed(std::future::ready(Err(ChordError::Message(
            self.message.to_string(),
        ))))
    }
}

/// A loader whose loaded generation's teardown fails, the fixture the
/// disposal-aggregation case rides on.
struct LoadedWithFailingDisposal;

impl FacetLoader for LoadedWithFailingDisposal {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let failing: pi_chord::handle::Disposal = Box::new(|| {
            boxed(async { Err(ChordError::Message("loaded disposal failed".to_string())) })
        });
        boxed(std::future::ready(Ok(LoadedFacets {
            facets: Vec::new(),
            dispose: failing,
        })))
    }
}

#[test]
fn facet_reloads_aggregate_stage_and_activation_cleanup_failures() {
    rt().block_on(async {
        let svc = define_service("test.reload.svc").unwrap_or_else(|e| panic!("define: {e}"));

        // A staged reload whose setup registered a failing teardown folds
        // the cleanup failure into the report.
        let failing_teardown = facet("reloadable", {
            let svc = svc.clone();
            move |env| {
                env.provide(&svc, read_implementation("A"))
                    .expect("provide lands");
                env.on_deactivate(Box::new(|| {
                    boxed(async { Err(ChordError::Message("teardown failed".to_string())) })
                }));
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![failing_teardown.clone()],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("host: {e}"));

        // A staged replacement whose implementation is invalid fails
        // staging; its teardown failure aggregates.
        // A staged reload whose implementation is not remotely
        // exposable fails staging; its teardown failure aggregates.
        let invalid = facet("reloadable", {
            let svc = svc.clone();
            move |env| {
                let mut non_exposable = ServiceImplementation::new();
                non_exposable.method(
                    "read",
                    sync_method(|_args: Vec<JsonValue>, _context: &Context| Ok(Some(js("A")))),
                );
                non_exposable.value("payload", Rc::new(()));
                env.provide(&svc, non_exposable).expect("provide lands");
                env.on_deactivate(Box::new(|| {
                    boxed(async { Err(ChordError::Message("staged teardown failed".to_string())) })
                }));
            }
        });
        let error = host
            .reload(vec![invalid])
            .await
            .expect_err("the invalid stage rejects");
        assert!(
            error_message(&error).contains("Facet reload setup and cleanup failed")
                && error_message(&error).contains("staged teardown failed"),
            "stage aggregate: {error}"
        );

        // A failed cleanup aborts the host, so the activation-failure
        // aggregate needs a fresh host.
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![failing_teardown],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("second host: {e}"));

        // An activation failure with a failing teardown aggregates the
        // abort.
        let failing_activation = facet("reloadable", {
            let svc = svc.clone();
            move |env| {
                env.provide(&svc, read_implementation("B"))
                    .expect("provide lands");
                env.on_activate(Box::new(|_env: &mut FacetEnvironment| {
                    boxed(async { Err(ChordError::Message("activation failed".to_string())) })
                }));
                env.on_deactivate(Box::new(|| {
                    boxed(async {
                        Err(ChordError::Message(
                            "activation teardown failed".to_string(),
                        ))
                    })
                }));
            }
        });
        let error = host
            .reload(vec![failing_activation])
            .await
            .expect_err("the failed activation aggregates");
        assert!(
            error_message(&error).contains("Facet reload activation and cleanup failed")
                && error_message(&error).contains("activation failed")
                && error_message(&error).contains("teardown failed"),
            "activation aggregate: {error}"
        );

        // The surviving generation still serves reads, and the host
        // disposes cleanly.
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn facet_hosts_bind_external_keyed_sources() {
    rt().block_on(async {
        let external =
            define_service("test.extkey.dialogs").unwrap_or_else(|e| panic!("define: {e}"));
        let source = ExternalSource {
            accepts_unavailable: false,
            fail_dispose: false,
            offered: vec![ServiceCatalogueEntry {
                service_id: external.id.clone(),
                mode: ServiceMode::Keyed,
            }],
            views: Rc::new(RefCell::new(Vec::new())),
        };
        let consumer = facet("external-keyed-consumer", {
            let external = external.clone();
            move |env| {
                env.observe_service(
                    &external,
                    Rc::new(|_view: ServiceView, _context: Context| ()),
                )
                .expect("observe lands");
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![consumer],
            service_sources: vec![Rc::new(source)],
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .unwrap_or_else(|e| panic!("the external keyed binding activates: {e}"));
        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}
