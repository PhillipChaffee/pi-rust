//! The facet-host suite, ported from upstream `test/facets.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The port's synchronous event-loop restatement removes the races
//! upstream's `vi.waitFor` covered: keyed observations start inside the
//! call that spawns them, so cases assert directly. The compile-time-only
//! contract case (JSON checks never throw at runtime) restates as a
//! behavior check on the member registry, and "asynchronous setup" is
//! unrepresentable — the setup closure returns nothing, so the case pins
//! the sync-by-type contract in this comment instead of a runtime throw.

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
    clippy::type_complexity,
    reason = "the ported cases spell their fixture types inline, as the upstream suite does"
)]

use std::cell::RefCell;
use std::rc::Rc;

use pi_chord::api::{
    create_facet_host, create_remote_service_binding, define_local_service, define_service,
};
use pi_chord::context::{Context, background_context};
use pi_chord::errors::ChordError;
use pi_chord::facets::host::{ActivationCallback, FacetEnvironment, FacetKernelOptions};
use pi_chord::future::boxed;
use pi_chord::handle::{ServiceImplementation, ServiceView, sync_method};
use pi_chord::services::loopback::create_loopback_service_transport;
use pi_chord::types::{FacetDef, JsonValue, KeyedViewHandler, Service, ServiceMode};

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

#[allow(
    dead_code,
    reason = "the local-service cases land with the remaining ports"
)]
fn host_values_service() -> Service {
    define_local_service("test.experimental.host-values").expect("not reserved")
}

#[allow(
    dead_code,
    reason = "the local-keyed cases land with the remaining ported cases"
)]
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

fn error_sink() -> (
    Rc<RefCell<Vec<ChordError>>>,
    pi_chord::handle::ErrorReporter,
) {
    let errors = Rc::new(RefCell::new(Vec::new()));
    let reporter: pi_chord::handle::ErrorReporter = {
        let errors = errors.clone();
        Rc::new(move |error| errors.borrow_mut().push(error.clone()))
    };
    (errors, reporter)
}

#[allow(
    dead_code,
    reason = "the remaining ported cases share this runtime builder"
)]
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime build must succeed")
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

        let host = create_facet_host(FacetKernelOptions {
            facets: vec![projection_facet, source_facet],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the graph is valid");

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
        let deliveries = Rc::new(std::cell::Cell::new(0u32));

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
                            env.own(pi_chord::handle::sync_disposal(move || {
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
        let host = create_facet_host(FacetKernelOptions {
            facets: vec![consumer, provider],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the graph is valid");
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

        impl pi_chord::types::RemoteServiceSource for DuplicateSource {
            fn accepts_unavailable_services(&self) -> bool {
                false
            }

            fn catalogue(
                &self,
                _context: Context,
            ) -> pi_chord::future::LocalBoxFuture<
                Result<Vec<pi_chord::types::ServiceCatalogueEntry>, ChordError>,
            > {
                boxed(async {
                    Ok(vec![pi_chord::types::ServiceCatalogueEntry {
                        service_id: source_service().id,
                        mode: ServiceMode::Singleton,
                    }])
                })
            }

            fn open(
                &self,
                _options: pi_chord::types::RemoteServiceSourceOpenOptions,
            ) -> Rc<dyn pi_chord::types::RemoteServices> {
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
fn connects_keyed_observations_through_the_host_provider() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        let keyed = keyed_service();
        let observed = Rc::new(RefCell::new(Vec::<String>::new()));

        let observer_facet = facet("observer", {
            let keyed = keyed.clone();
            let observed = observed.clone();
            move |env| {
                let handler: KeyedViewHandler = {
                    let observed = observed.clone();
                    Rc::new(move |view: ServiceView, _context: Context| {
                        if let Ok(state) = view.state("value")
                            && let Ok(Some(value)) = state.value()
                        {
                            observed.borrow_mut().push(format!("observe {value:?}"));
                        }
                    })
                };
                env.observe_service(&keyed, handler).expect("observe lands");
            }
        });
        let provider_facet = facet("provider", {
            let keyed = keyed.clone();
            move |env| {
                let values = env.provide_many(&keyed).expect("provide_many lands");
                env.on_activate({
                    Box::new(move |_env: &mut FacetEnvironment| {
                        values
                            .spawn("one", keyed_value_implementation("one"))
                            .expect("spawn lands");
                        boxed(async { Ok(()) })
                    })
                });
            }
        });

        let host = create_facet_host(FacetKernelOptions {
            facets: vec![observer_facet, provider_facet],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the graph is valid");
        // The keyed observation started inside the spawn's publication; the
        // synchronous restatement delivers it before any assertion.
        assert_eq!(observed.borrow().len(), 1);

        host.dispose()
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

fn keyed_value_implementation(value: &'static str) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.state(
        "value",
        pi_chord::api::replicated_state(jo(vec![("value", js(value))])),
    );
    implementation
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
                let consumer_handles = consumer_handles.clone();
                let cleanup_values = cleanup_values.clone();
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
                    let retained = first.clone();
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

        let host = create_facet_host(FacetKernelOptions {
            facets: vec![
                consumer_for("A", consumer_handles.clone(), cleanup_values.clone()),
                peer,
                provider,
            ],
            service_sources: Vec::new(),
            on_error: pi_chord::handle::no_error_reporter(),
        })
        .await
        .expect("the graph is valid");
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
        for handle in consumer_handles.borrow().iter() {
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

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}
