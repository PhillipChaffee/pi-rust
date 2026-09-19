//! The remote services suite, ported from upstream `test/services.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The port's synchronous event-loop restatement removes the races
//! upstream's `vi.waitFor` covered: subscription starts, observer starts,
//! and buffered replay all settle inside the call that schedules them, so
//! every case asserts directly. Proxy-identity assertions
//! (`Object.is` on handles) ride
//! [`pi_chord::handle::ServiceView::same_handle`], and the compile-time
//! JSON contract check becomes the member registry's remote validation,
//! which rejects value members when an implementation is provided.

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
#![allow(
    trivial_casts,
    reason = "the fixture dispatch maps spell the dyn transport the binding options take"
)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pi_chord::api::{
    create_remote_service_binding, define_local_service, define_service, replicated_state,
};
use pi_chord::context::{Context, background_context};
use pi_chord::delta::{Op, Seg};
use pi_chord::errors::{ChordError, RemoteServiceErrorCode};
use pi_chord::handle::{ServiceImplementation, ServiceView};
use pi_chord::services::loopback::create_loopback_service_transport;
use pi_chord::services::provider::{
    RemoteServiceProvider, create_remote_service_endpoint, singleton_definition,
    validate_remote_service_implementation,
};
use pi_chord::services::state::MutableReplicatedState;
use pi_chord::types::{
    JsonValue, RemoteServiceTransport, Service, ServiceCall, ServiceInstanceAddress,
    ServiceInstanceSnapshot, ServiceMemberSnapshot, ServiceMode, ServiceProviderListener,
    ServiceProviderUpdate, ServiceSubscription, ServiceSubscriptionSnapshot,
};

fn key(text: &str) -> Seg {
    Seg::Key(text.to_string())
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

fn number(value: u64) -> JsonValue {
    JsonValue::Number(pi_chord::types::JsonNumber::from(value))
}

fn revision_of(value: &JsonValue) -> u64 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "fixture revisions are non-negative small integers"
    )]
    let revision = value
        .as_object()
        .and_then(|object| object.get("revision"))
        .and_then(JsonValue::as_number)
        .map_or_else(
            || panic!("fixture carries a revision"),
            |revision| revision as u64,
        );
    revision
}

fn error_message(error: &ChordError) -> String {
    error.to_string()
}

const fn error_code(error: &ChordError) -> Option<RemoteServiceErrorCode> {
    match error {
        ChordError::Remote(remote) => Some(remote.code),
        _ => None,
    }
}

/// A reporter collecting the failures a binding reports.
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

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime build must succeed")
}

fn provider_for(services: &[&Service]) -> RemoteServiceProvider {
    RemoteServiceProvider::new(
        services
            .iter()
            .map(|service| singleton_definition((*service).clone()))
            .collect(),
    )
    .expect("the catalogue has no duplicates")
}

fn expect_err<T, E: std::fmt::Debug>(result: Result<T, E>) -> E {
    match result {
        Ok(_) => panic!("the case's operation rejects"),
        Err(error) => error,
    }
}

/// The state-member accessor a view exposes for `models.state` reads and
/// subscriptions.
fn state_of(view: &ServiceView, member: &str) -> pi_chord::handle::StateMemberView {
    view.state(member).expect("the member is state")
}

#[test]
fn marks_services_remotable_by_default_and_reserves_chord_service_ids() {
    let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
    let local = define_local_service("test.local").unwrap_or_else(|e| panic!("define: {e}"));
    assert!(!models.local);
    assert!(local.local);
    let error = define_service("$chord.internal").expect_err("reserved prefix rejects");
    assert!(error_message(&error).contains("Service IDs beginning with $chord. are reserved"));
    let error = RemoteServiceProvider::new(vec![singleton_definition(local)])
        .expect_err("a local service cannot join a remote catalogue");
    assert!(error_message(&error).contains("cannot be published remotely"));
}

/// Upstream checks the JSON-only contract at compile time through
/// `RemoteServiceContract<T>`; the owned-data port enforces it when an
/// implementation is provided, so the runtime half of the case pins that a
/// remote implementation rejects a value member while a local service may
/// carry arbitrary data.
#[test]
fn rejects_remote_members_that_cannot_cross_the_wire() {
    let service =
        define_service("test.non-json-argument").unwrap_or_else(|e| panic!("define: {e}"));
    let provider = provider_for(&[&service]);
    let mut implementation = ServiceImplementation::new();
    implementation.value("payload", Rc::new(()));
    let error = provider
        .provide(&service, implementation)
        .expect_err("a value member is not remotely exposable");
    assert!(error_message(&error).contains("is not remotely exposable"));

    let local =
        define_local_service("test.local-non-json").unwrap_or_else(|e| panic!("define: {e}"));
    assert!(local.local);
}

#[test]
fn tracks_mutable_source_state_while_publishing_immutable_revisions() {
    let initial = jo(vec![("selected", JsonValue::Null), ("revision", number(0))]);
    let state = replicated_state(initial.clone());
    let delivered = Rc::new(RefCell::new(Vec::<(JsonValue, String)>::new()));
    let listener = {
        let delivered = delivered.clone();
        Rc::new(
            move |value: &JsonValue,
                  _context: &Context,
                  delivery: &pi_chord::types::ReplicatedStateDelivery| {
                delivered
                    .borrow_mut()
                    .push((value.clone(), delivery.kind.to_string()));
            },
        )
    };
    let unsubscribe = state
        .subscribe(listener)
        .unwrap_or_else(|e| panic!("subscribe: {e}"));
    assert_eq!(state.value(), initial);
    assert_eq!(delivered.borrow().len(), 1);
    let hydrated = delivered.borrow()[0].0.clone();
    assert_eq!(hydrated, initial);

    state.mutate(|tracker| {
        tracker
            .set(
                &[key("selected")],
                jo(vec![("provider", js("test")), ("modelId", js("one"))]),
            )
            .expect("the write lands");
        tracker
            .set(&[key("revision")], number(1))
            .expect("the write lands");
    });
    state
        .publish(&background_context())
        .unwrap_or_else(|e| panic!("publish: {e}"));
    assert_eq!(
        state.value(),
        jo(vec![
            (
                "selected",
                jo(vec![("provider", js("test")), ("modelId", js("one"))])
            ),
            ("revision", number(1)),
        ])
    );
    assert_eq!(hydrated, initial);
    let kinds: Vec<String> = delivered
        .borrow()
        .iter()
        .map(|(_, kind)| kind.clone())
        .collect();
    assert_eq!(kinds, vec!["hydrate", "update"]);
    unsubscribe();
}

#[test]
fn flushes_pending_mutations_before_hydrating_a_new_state_subscriber() {
    let initial = jo(vec![(
        "entries",
        JsonValue::Array(vec![jo(vec![("id", js("one"))])]),
    )]);
    let state = replicated_state(initial);
    let first = Rc::new(RefCell::new(Vec::<JsonValue>::new()));
    let _unsubscribe_first = state
        .subscribe({
            let first = first.clone();
            Rc::new(
                move |value: &JsonValue,
                      _context: &Context,
                      _delivery: &pi_chord::types::ReplicatedStateDelivery| {
                    first.borrow_mut().push(value.clone());
                },
            )
        })
        .unwrap_or_else(|e| panic!("subscribe: {e}"));

    state.mutate(|tracker| {
        tracker
            .set(
                &[key("entries"), Seg::Index(1)],
                jo(vec![("id", js("two"))]),
            )
            .expect("the write lands");
    });

    let second = Rc::new(RefCell::new(Vec::<JsonValue>::new()));
    let _unsubscribe_second = state
        .subscribe({
            let second = second.clone();
            Rc::new(
                move |value: &JsonValue,
                      _context: &Context,
                      _delivery: &pi_chord::types::ReplicatedStateDelivery| {
                    second.borrow_mut().push(value.clone());
                },
            )
        })
        .unwrap_or_else(|e| panic!("subscribe: {e}"));

    assert_eq!(
        *first.borrow(),
        vec![
            jo(vec![(
                "entries",
                JsonValue::Array(vec![jo(vec![("id", js("one"))])])
            )]),
            jo(vec![(
                "entries",
                JsonValue::Array(vec![
                    jo(vec![("id", js("one"))]),
                    jo(vec![("id", js("two"))])
                ])
            )]),
        ]
    );
    assert_eq!(
        *second.borrow(),
        vec![jo(vec![(
            "entries",
            JsonValue::Array(vec![
                jo(vec![("id", js("one"))]),
                jo(vec![("id", js("two"))])
            ])
        )])]
    );
}

/// Upstream's `does not defensively clone method arguments or results`
/// asserts `Object.is` identity on the argument the implementation receives
/// and the result the caller receives; owned data has no identity to spell,
/// so the port pins the loopback handing both sides their values unmodified.
#[test]
fn passes_method_arguments_and_results_through_the_loopback_unchanged() {
    let rt = runtime();
    rt.block_on(async {
        let echo = define_service("test.echo").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&echo]);
        let received = Rc::new(RefCell::new(None));
        let mut implementation = ServiceImplementation::new();
        implementation.method("echo", {
            let received = received.clone();
            Rc::new(move |args: Vec<JsonValue>, _context: Context| {
                let received = received.clone();
                let response = jo(vec![("value", js("response"))]);
                boxed(async move {
                    *received.borrow_mut() = args.first().cloned();
                    Ok(Some(response))
                })
            })
        });
        provider
            .provide(&echo, implementation)
            .unwrap_or_else(|e| panic!("provide: {e}"));

        let binding = binding_for(vec![echo.clone()], &provider, None);
        let echo_view = binding
            .use_service(&echo)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));

        let request = jo(vec![("value", js("request"))]);
        let response = echo_view
            .call("echo", vec![request.clone()], background_context())
            .await
            .unwrap_or_else(|e| panic!("echo: {e}"));
        assert_eq!(response, Some(jo(vec![("value", js("response"))])));
        assert_eq!(*received.borrow(), Some(request));

        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

/// Builds the Models implementation: a `state` member plus a `select`
/// method that records the model, bumps the revision, publishes with the
/// invocation context, and captures the published value — the port of the
/// upstream object literal.
fn models_implementation(
    state: &MutableReplicatedState,
    published: &Rc<RefCell<Option<JsonValue>>>,
) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.state("state", state.clone());
    let state_for_select = state.clone();
    let published = published.clone();
    implementation.method(
        "select",
        Rc::new(move |args: Vec<JsonValue>, context: Context| {
            let state = state_for_select.clone();
            let published = published.clone();
            let model = args.first().cloned().unwrap_or(JsonValue::Null);
            boxed(async move {
                state.mutate(|tracker| {
                    tracker
                        .set(&[key("selected")], model)
                        .expect("the write lands");
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "fixture revisions are non-negative small integers"
                    )]
                    let revision = tracker
                        .get(&[key("revision")])
                        .and_then(JsonValue::as_number)
                        .map(|revision| revision as u64)
                        .unwrap_or_default();
                    tracker
                        .set(&[key("revision")], number(revision + 1))
                        .expect("the write lands");
                });
                state.publish(&context).expect("the publication lands");
                *published.borrow_mut() = Some(state.value());
                Ok(None)
            })
        }),
    );
    implementation
}

fn boxed<F: Future<Output = T> + 'static, T>(future: F) -> Pin<Box<dyn Future<Output = T>>> {
    Box::pin(future)
}

use std::future::Future;
use std::pin::Pin;

#[test]
fn provides_and_consumes_one_singleton_with_replicated_state() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        assert_eq!(
            provider.catalogue(),
            &[pi_chord::types::ServiceCatalogueEntry {
                service_id: models.id.clone(),
                mode: ServiceMode::Singleton,
            }]
        );
        let initial = jo(vec![("selected", JsonValue::Null), ("revision", number(0))]);
        let state = replicated_state(initial.clone());
        let published = Rc::new(RefCell::new(None));
        provider
            .provide(&models, models_implementation(&state, &published))
            .unwrap_or_else(|e| panic!("provide: {e}"));

        let (errors, on_error) = error_sink();
        let binding = binding_for(vec![models.clone()], &provider, Some(on_error));
        let first = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let second = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        assert!(first.same_handle(&second));
        assert!(
            state_of(&first, "state")
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .is_none()
        );
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(
            state_of(&first, "state")
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .expect("hydrated after ready"),
            initial
        );

        let updates = Rc::new(RefCell::new(Vec::<JsonValue>::new()));
        let listener = {
            let updates = updates.clone();
            Rc::new(
                move |value: &JsonValue,
                      _context: &Context,
                      _delivery: &pi_chord::types::ReplicatedStateDelivery| {
                    updates.borrow_mut().push(value.clone());
                },
            )
        };
        let second_state = state_of(&second_view_of(&binding, &models), "state");
        let _unsubscribe = second_state
            .subscribe(listener)
            .unwrap_or_else(|e| panic!("subscribe: {e}"));

        first
            .call(
                "select",
                vec![jo(vec![("provider", js("test")), ("modelId", js("one"))])],
                background_context(),
            )
            .await
            .unwrap_or_else(|e| panic!("select: {e}"));
        let published_value = published.borrow().clone().expect("select published");
        assert_eq!(
            state_of(&first, "state").value().expect("read"),
            Some(published_value.clone())
        );
        assert_eq!(
            state_of(&first, "state").value().expect("read"),
            Some(jo(vec![
                (
                    "selected",
                    jo(vec![("provider", js("test")), ("modelId", js("one"))])
                ),
                ("revision", number(1)),
            ]))
        );
        assert_eq!(
            *updates.borrow(),
            vec![
                jo(vec![("selected", JsonValue::Null), ("revision", number(0))]),
                jo(vec![
                    (
                        "selected",
                        jo(vec![("provider", js("test")), ("modelId", js("one"))])
                    ),
                    ("revision", number(1))
                ]),
            ]
        );
        assert!(errors.borrow().is_empty());

        let late_binding = binding_for(vec![models.clone()], &provider, None);
        let late_models = late_binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        late_binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("late ready: {e}"));
        let late_revision = state_of(&late_models, "state")
            .value()
            .unwrap_or_else(|e| panic!("value: {e}"))
            .map(|value| revision_of(&value));
        assert_eq!(late_revision, Some(1));

        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        late_binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

fn second_view_of(
    binding: &pi_chord::consumer::RemoteServiceBinding,
    service: &Service,
) -> ServiceView {
    binding.use_service(service).expect("the facade is cached")
}

fn binding_for(
    services: Vec<Service>,
    provider: &RemoteServiceProvider,
    on_error: Option<pi_chord::handle::ErrorReporter>,
) -> pi_chord::consumer::RemoteServiceBinding {
    create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
        services,
        transport: create_loopback_service_transport(provider),
        bound: true,
        on_error: on_error.unwrap_or_else(pi_chord::handle::no_error_reporter),
        assert_access: None,
    })
    .expect("the binding allowlist has no duplicates")
}

#[test]
fn publishes_compact_tracked_operations_through_the_remote_provider() {
    let rt = runtime();
    rt.block_on(async {
        let timeline = define_service("test.timeline").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&timeline]);
        let initial = jo(vec![
            (
                "entries",
                JsonValue::Array(vec![jo(vec![("id", js("one"))])]),
            ),
            ("retained", jo(vec![("value", number(1))])),
        ]);
        let source = replicated_state(initial.clone());
        let mut implementation = ServiceImplementation::new();
        implementation.state("state", source.clone());
        provider
            .provide(&timeline, implementation)
            .unwrap_or_else(|e| panic!("provide: {e}"));

        let updates = Rc::new(RefCell::new(Vec::<ServiceProviderUpdate>::new()));
        let listener = {
            let updates = updates.clone();
            Rc::new(move |update: &ServiceProviderUpdate, _context: &Context| {
                updates.borrow_mut().push(update.clone());
            })
        };
        let raw = provider
            .subscribe(timeline.id.as_str(), ServiceMode::Singleton, listener)
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        let member_snapshot = raw.snapshot.instances[0].members[0].clone();
        let Some(ServiceMemberSnapshot::State { sequence, ops, .. }) = Some(&member_snapshot)
        else {
            panic!("state member");
        };
        assert_eq!(*sequence, 0);
        assert_eq!(ops, &vec![Op::Replace(initial.clone())]);
        (raw.activate)().unwrap_or_else(|e| panic!("activate: {e}"));

        let binding = binding_for(vec![timeline.clone()], &provider, None);
        let timeline_view = binding
            .use_service(&timeline)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        let previous = state_of(&timeline_view, "state")
            .value()
            .expect("read")
            .expect("hydrated");
        let next = jo(vec![
            (
                "entries",
                JsonValue::Array(vec![
                    jo(vec![("id", js("one"))]),
                    jo(vec![("id", js("two"))]),
                ]),
            ),
            ("retained", jo(vec![("value", number(1))])),
        ]);
        source.mutate(|tracker| {
            tracker
                .push(&[key("entries")], vec![jo(vec![("id", js("two"))])])
                .expect("the append lands");
        });
        source
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));

        let state_updates: Vec<ServiceProviderUpdate> = updates
            .borrow()
            .iter()
            .filter(|update| matches!(update, ServiceProviderUpdate::State { .. }))
            .cloned()
            .collect();
        let Some(ServiceProviderUpdate::State {
            member,
            sequence,
            ops,
            ..
        }) = state_updates.iter().find(|update| {
            let ServiceProviderUpdate::State {
                sequence: found, ..
            } = update
            else {
                panic!("state update");
            };
            *found == 1
        })
        else {
            panic!("a sequence-1 update exists: {state_updates:?}");
        };
        assert_eq!(member, "state");
        assert_eq!(*sequence, 1);
        assert_eq!(
            ops,
            &vec![Op::Splice {
                path: vec![key("entries")],
                index: 1,
                remove: 0,
                items: vec![jo(vec![("id", js("two"))])],
            }]
        );
        assert_eq!(previous, initial);
        assert_eq!(
            state_of(&timeline_view, "state")
                .value()
                .expect("read")
                .expect("hydrated"),
            next
        );

        source.mutate(|tracker| {
            tracker
                .push(&[key("entries")], vec![jo(vec![("id", js("three"))])])
                .expect("the append lands");
        });
        let late = provider
            .subscribe(
                timeline.id.as_str(),
                ServiceMode::Singleton,
                Rc::new(|_update: &ServiceProviderUpdate, _context: &Context| {}),
            )
            .unwrap_or_else(|e| panic!("late subscribe: {e}"));
        let Some(ServiceMemberSnapshot::State {
            sequence: late_sequence,
            ops: late_ops,
            ..
        }) = Some(&late.snapshot.instances[0].members[0])
        else {
            panic!("state member");
        };
        assert_eq!(*late_sequence, 2);
        assert_eq!(
            *late_ops,
            vec![Op::Replace(jo(vec![
                (
                    "entries",
                    JsonValue::Array(vec![
                        jo(vec![("id", js("one"))]),
                        jo(vec![("id", js("two"))]),
                        jo(vec![("id", js("three"))])
                    ]),
                ),
                ("retained", jo(vec![("value", number(1))])),
            ]))]
        );
        (late.close)(None)
            .await
            .unwrap_or_else(|e| panic!("close: {e}"));
        (raw.close)(None)
            .await
            .unwrap_or_else(|e| panic!("close: {e}"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn keeps_singleton_facades_stable_when_their_provider_is_replaced() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![
                    ("selected", JsonValue::Null),
                    ("revision", number(1)),
                ]))),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let binding = binding_for(vec![models.clone()], &provider, None);
        let models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let state = state_of(&models_view, "state");
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(
            revision_of(
                &state
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            1
        );

        provider
            .withdraw(&models)
            .unwrap_or_else(|e| panic!("withdraw: {e}"));
        assert!(
            state
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .is_none()
        );
        let error = models_view
            .call("select", vec![model_ref()], background_context())
            .await
            .expect_err("a withdrawn singleton has no provider");
        assert_eq!(
            error_code(&error),
            Some(RemoteServiceErrorCode::ServiceNotFound)
        );

        let replacement_calls = Rc::new(std::cell::Cell::new(0u32));
        let replacement = replacement_implementation(&replacement_calls);
        provider
            .replace(&models, replacement)
            .unwrap_or_else(|e| panic!("replace: {e}"));

        let again = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        assert!(again.same_handle(&models_view));
        assert_eq!(
            revision_of(
                &state
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            2
        );
        models_view
            .call("select", vec![model_ref()], background_context())
            .await
            .unwrap_or_else(|e| panic!("replacement select: {e}"));
        assert_eq!(replacement_calls.get(), 1);
        let error = provider
            .replace(&models, method_only_implementation())
            .expect_err("a member-shape change rejects");
        assert!(error_message(&error).contains("replacement must preserve its member shape"));
        let error = provider
            .replace(&models, two_state_implementation())
            .expect_err("a member-shape change rejects");
        assert!(error_message(&error).contains("replacement must preserve its member shape"));
        assert_eq!(
            revision_of(
                &state
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            2
        );

        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

fn model_ref() -> JsonValue {
    jo(vec![
        ("provider", js("test")),
        ("modelId", js("replacement")),
    ])
}

fn implementation_with_state_and_noop_select(
    state: MutableReplicatedState,
) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.state("state", state);
    implementation.method(
        "select",
        Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(async { Ok(None) })),
    );
    implementation
}

fn replacement_implementation(calls: &Rc<std::cell::Cell<u32>>) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.state(
        "state",
        replicated_state(jo(vec![
            ("selected", JsonValue::Null),
            ("revision", number(2)),
        ])),
    );
    let calls = calls.clone();
    implementation.method(
        "select",
        Rc::new(move |_args: Vec<JsonValue>, _context: Context| {
            let calls = calls.clone();
            boxed(async move {
                calls.set(calls.get() + 1);
                Ok(None)
            })
        }),
    );
    implementation
}

fn method_only_implementation() -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.method(
        "select",
        Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(async { Ok(None) })),
    );
    implementation
}

fn two_state_implementation() -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.method(
        "state",
        Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(async { Ok(None) })),
    );
    implementation.method(
        "select",
        Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(async { Ok(None) })),
    );
    implementation
}

#[test]
fn delivers_active_subscriber_updates_before_reporting_listener_failures() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![
                    ("selected", JsonValue::Null),
                    ("revision", number(1)),
                ]))),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let delivered = Rc::new(std::cell::Cell::new(0u32));
        let failing = provider
            .subscribe(
                models.id.as_str(),
                ServiceMode::Singleton,
                Rc::new(|_update: &ServiceProviderUpdate, _context: &Context| {
                    panic!("listener failed");
                }),
            )
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        let succeeding = provider
            .subscribe(models.id.as_str(), ServiceMode::Singleton, {
                let delivered = delivered.clone();
                Rc::new(move |_update: &ServiceProviderUpdate, _context: &Context| {
                    delivered.set(delivered.get() + 1);
                })
            })
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        (failing.activate)().unwrap_or_else(|e| panic!("activate failing: {e}"));
        (succeeding.activate)().unwrap_or_else(|e| panic!("activate succeeding: {e}"));

        let error = provider
            .replace(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![
                    ("selected", JsonValue::Null),
                    ("revision", number(2)),
                ]))),
            )
            .expect_err("the failing listener surfaces");
        assert!(error_message(&error).contains("listener failed"));
        assert_eq!(delivered.get(), 1);

        (failing.close)(None)
            .await
            .unwrap_or_else(|e| panic!("close: {e}"));
        (succeeding.close)(None)
            .await
            .unwrap_or_else(|e| panic!("close: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn replays_every_buffered_update_before_reporting_listener_failures() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        let state = replicated_state(jo(vec![
            ("selected", JsonValue::Null),
            ("revision", number(0)),
        ]));
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(state.clone()),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let delivered = Rc::new(std::cell::Cell::new(0u32));
        let subscription = provider
            .subscribe(models.id.as_str(), ServiceMode::Singleton, {
                let delivered = delivered.clone();
                Rc::new(move |_update: &ServiceProviderUpdate, _context: &Context| {
                    delivered.set(delivered.get() + 1);
                    panic!("listener failed");
                })
            })
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        state.mutate(|tracker| {
            tracker
                .set(&[key("revision")], number(1))
                .expect("the write lands");
        });
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        state.mutate(|tracker| {
            tracker
                .set(&[key("revision")], number(2))
                .expect("the write lands");
        });
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));

        let error = (subscription.activate)().expect_err("activation collects the failure");
        assert!(error_message(&error).contains("Failed to activate remote service subscription"));
        assert_eq!(delivered.get(), 2);
        (subscription.close)(None)
            .await
            .unwrap_or_else(|e| panic!("close: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn clears_retained_facades_when_providers_and_bindings_are_disposed() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![
                    ("selected", JsonValue::Null),
                    ("revision", number(1)),
                ]))),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let binding = binding_for(vec![models.clone()], &provider, None);
        let models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let state = state_of(&models_view, "state");
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(
            revision_of(
                &state
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            1
        );

        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
        assert!(
            state
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .is_none()
        );
        let error = models_view
            .call("select", vec![model_ref()], background_context())
            .await
            .expect_err("a disposed provider rejects");
        assert!(error_message(&error).contains("Remote service provider is disposed"));

        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let error = state
            .value()
            .expect_err("a disposed binding's gate rejects");
        assert!(error_message(&error).contains("Remote service binding is disposed"));
    });
}

#[test]
fn applies_provider_disposal_buffered_while_subscriptions_are_starting() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let dialogs =
            define_service("test.question-dialog").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            singleton_definition(models.clone()),
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![
                    ("selected", JsonValue::Null),
                    ("revision", number(1)),
                ]))),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let spawned = provider
            .spawn(
                &dialogs,
                "pending",
                keyed_dialogs_implementation("Pending?"),
            )
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        let binding = binding_for(vec![models.clone(), dialogs.clone()], &provider, None);
        let models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let _observed = Rc::new(RefCell::new(Vec::<()>::new()));
        let stop = binding
            .observe(&dialogs, {
                let observed = Rc::new(std::cell::Cell::new(0u32));
                Rc::new(move |_view: ServiceView, _context: Context| {
                    observed.set(observed.get() + 1);
                })
            })
            .unwrap_or_else(|e| panic!("observe: {e}"));

        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        let state = state_of(&models_view, "state");
        assert!(
            state
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .is_none()
        );
        stop();
        let _ = spawned;
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

fn keyed_dialogs_implementation(question: &str) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.state(
        "request",
        replicated_state(jo(vec![("question", js(question))])),
    );
    implementation.method(
        "submit",
        Rc::new(|_args: Vec<JsonValue>, _context: Context| {
            boxed(async { Ok(Some(jo(vec![("accepted", JsonValue::Bool(true))]))) })
        }),
    );
    implementation
}

#[test]
fn keeps_deferred_service_handles_inaccessible_until_host_activation() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![
                    ("selected", JsonValue::Null),
                    ("revision", number(0)),
                ]))),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let active = Rc::new(std::cell::Cell::new(false));
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![models.clone()],
                transport: create_loopback_service_transport(&provider),
                bound: false,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: Some({
                    let active = active.clone();
                    Rc::new(move || {
                        if active.get() {
                            Ok(())
                        } else {
                            Err(ChordError::Message(
                                "Service handles are not active".to_string(),
                            ))
                        }
                    })
                }),
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));

        let error = expect_err(models_view.state("state"));
        assert!(error_message(&error).contains("Service handles are not active"));
        let no_op_listener: pi_chord::services::state::ValueListener = Rc::new(
            |_value: &JsonValue,
             _context: &Context,
             _delivery: &pi_chord::types::ReplicatedStateDelivery| {},
        );
        let error = match models_view.state("state") {
            Ok(state) => match state.subscribe(no_op_listener) {
                Ok(_) => panic!("the gate rejects the subscription"),
                Err(error) => error,
            },
            Err(error) => error,
        };
        assert!(error_message(&error).contains("Service handles are not active"));
        let error = models_view
            .call("select", vec![model_ref()], background_context())
            .await
            .expect_err("the gate rejects");
        assert!(error_message(&error).contains("Service handles are not active"));

        binding
            .rebind(true, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind: {e}"));
        active.set(true);
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(
            revision_of(
                &models_view
                    .state("state")
                    .unwrap_or_else(|error| panic!("state after activation: {error}"))
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated"),
            ),
            0
        );

        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn rejects_namespace_readiness_when_initial_hydration_fails() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let (errors, on_error) = error_sink();
        let failure = ChordError::Message("initial hydration failed".to_string());
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(FailingTransport {
            failure: failure.clone(),
        });
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![models.clone()],
                transport,
                bound: true,
                on_error,
                assert_access: None,
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let _models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));

        let error = binding
            .ready(background_context())
            .await
            .expect_err("readiness rejects");
        assert!(error_message(&error).contains("initial hydration failed"));
        assert!(
            state_of(
                &binding
                    .use_service(&models)
                    .unwrap_or_else(|e| panic!("use: {e}")),
                "state"
            )
            .value()
            .unwrap_or_else(|e| panic!("value: {e}"))
            .is_none()
        );
        assert_eq!(errors.borrow().len(), 1);
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

struct FailingTransport {
    failure: ChordError,
}

impl RemoteServiceTransport for FailingTransport {
    fn invoke(
        &self,
        _call: ServiceCall,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<Option<JsonValue>, ChordError>>>> {
        boxed(async { Err(ChordError::Message("unexpected invocation".to_string())) })
    }

    fn subscribe(
        &self,
        _service_id: String,
        _mode: ServiceMode,
        _listener: ServiceProviderListener,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<ServiceSubscription, ChordError>>>> {
        let failure = self.failure.clone();
        boxed(async { Err(failure) })
    }
}

#[test]
fn buffers_state_updates_that_race_subscription_hydration() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        let state = replicated_state(jo(vec![
            ("selected", JsonValue::Null),
            ("revision", number(0)),
        ]));
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(state.clone()),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        // The custom transport publishes between the provider's subscribe
        // and the subscription's return — the race the case pins; the
        // buffered subscriber replays it on activation.
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(RacingTransport {
            provider: provider.clone(),
            state: state.clone(),
        });
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![models.clone()],
                transport,
                bound: true,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: None,
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let revisions = Rc::new(RefCell::new(Vec::<u64>::new()));
        let listener = {
            let revisions = revisions.clone();
            Rc::new(
                move |value: &JsonValue,
                      _context: &Context,
                      _delivery: &pi_chord::types::ReplicatedStateDelivery| {
                    revisions.borrow_mut().push(revision_of(value));
                },
            )
        };
        let _unsubscribe = state_of(&models_view, "state")
            .subscribe(listener)
            .unwrap_or_else(|e| panic!("subscribe: {e}"));

        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(*revisions.borrow(), vec![0, 1]);
        assert_eq!(
            revision_of(
                &state_of(&models_view, "state")
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            1
        );
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

struct RacingTransport {
    provider: RemoteServiceProvider,
    state: MutableReplicatedState,
}

impl RemoteServiceTransport for RacingTransport {
    fn invoke(
        &self,
        call: ServiceCall,
        context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<Option<JsonValue>, ChordError>>>> {
        self.provider.invoke(call, context)
    }

    fn subscribe(
        &self,
        service_id: String,
        mode: ServiceMode,
        listener: ServiceProviderListener,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<ServiceSubscription, ChordError>>>> {
        let subscription = self.provider.subscribe(&service_id, mode, listener);
        self.state.mutate(|tracker| {
            tracker
                .set(&[key("revision")], number(1))
                .expect("the write lands");
        });
        self.state
            .publish(&background_context())
            .expect("the publication lands");
        let subscription = subscription;
        boxed(async move {
            let subscription = subscription?;
            let snapshot = subscription.snapshot.clone();
            Ok(ServiceSubscription {
                snapshot,
                activate: subscription.activate,
                close: subscription.close,
            })
        })
    }
}

#[test]
fn clears_replicated_state_after_a_duplicate_operation_sequence() {
    sequence_gap_case(0);
}

#[test]
fn clears_replicated_state_after_a_gap_operation_sequence() {
    sequence_gap_case(2);
}

fn sequence_gap_case(sequence: u64) {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let (errors, on_error) = error_sink();
        let send_update = Rc::new(RefCell::new(None::<ServiceProviderListener>));
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(ManualTransport {
            service_id: models.id.clone(),
            send_update: send_update.clone(),
        });
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![models.clone()],
                transport,
                bound: true,
                on_error,
                assert_access: None,
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(
            revision_of(
                &state_of(&models_view, "state")
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            0
        );

        let bad_update = ServiceProviderUpdate::State {
            instance: None,
            member: "state".to_string(),
            sequence,
            ops: vec![Op::Replace(jo(vec![
                ("selected", JsonValue::Null),
                ("revision", number(sequence)),
            ]))],
        };
        let listener = send_update
            .borrow()
            .clone()
            .expect("subscribe registered the listener");
        listener(&bad_update, &background_context());
        assert!(
            state_of(&models_view, "state")
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .is_none()
        );
        assert_eq!(errors.borrow().len(), 1);
        assert!(error_message(&errors.borrow()[0]).contains("sequence has a gap"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

/// A transport whose snapshot is hand-built and whose update listener is
/// exposed to the case, upstream's `sendUpdate` capture.
struct ManualTransport {
    service_id: String,
    send_update: Rc<RefCell<Option<ServiceProviderListener>>>,
}

impl RemoteServiceTransport for ManualTransport {
    fn invoke(
        &self,
        _call: ServiceCall,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<Option<JsonValue>, ChordError>>>> {
        boxed(async { Err(ChordError::Message("unexpected invocation".to_string())) })
    }

    fn subscribe(
        &self,
        _service_id: String,
        _mode: ServiceMode,
        listener: ServiceProviderListener,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<ServiceSubscription, ChordError>>>> {
        *self.send_update.borrow_mut() = Some(listener.clone());
        let snapshot = ServiceSubscriptionSnapshot {
            service_id: self.service_id.clone(),
            mode: ServiceMode::Singleton,
            instances: vec![ServiceInstanceSnapshot {
                instance: None,
                members: vec![
                    ServiceMemberSnapshot::Method {
                        name: "select".to_string(),
                    },
                    ServiceMemberSnapshot::State {
                        name: "state".to_string(),
                        sequence: 0,
                        ops: vec![Op::Replace(jo(vec![
                            ("selected", JsonValue::Null),
                            ("revision", number(0)),
                        ]))],
                    },
                ],
            }],
        };
        boxed(async move {
            Ok(ServiceSubscription {
                snapshot,
                activate: Box::new(|| Ok(())),
                close: boxed_no_close(),
            })
        })
    }
}

fn boxed_no_close()
-> Box<dyn Fn(Option<Context>) -> Pin<Box<dyn Future<Output = Result<(), ChordError>>>>> {
    Box::new(|_context| boxed(async { Ok(()) }))
}

#[test]
fn hydrates_cold_replicated_state_replicas_and_replaces_them_across_rebinds() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        let state = replicated_state(jo(vec![
            ("selected", JsonValue::Null),
            ("revision", number(0)),
        ]));
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(state.clone()),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let binding = binding_deferred(vec![models.clone()], &provider, None);
        let models_view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let revisions = Rc::new(RefCell::new(Vec::<u64>::new()));
        let listener = {
            let revisions = revisions.clone();
            Rc::new(
                move |value: &JsonValue,
                      _context: &Context,
                      _delivery: &pi_chord::types::ReplicatedStateDelivery| {
                    revisions.borrow_mut().push(revision_of(value));
                },
            )
        };
        let _unsubscribe = state_of(&models_view, "state")
            .subscribe(listener)
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        assert!(
            state_of(&models_view, "state")
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .is_none()
        );
        assert!(revisions.borrow().is_empty());

        state.mutate(|tracker| {
            tracker
                .set(&[key("revision")], number(1))
                .expect("the write lands");
        });
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        binding
            .rebind(true, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind: {e}"));
        assert_eq!(
            revision_of(
                &state_of(&models_view, "state")
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            1
        );
        assert_eq!(*revisions.borrow(), vec![1]);
        state.mutate(|tracker| {
            tracker
                .set(&[key("revision")], number(2))
                .expect("the write lands");
        });
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        assert_eq!(*revisions.borrow(), vec![1, 2]);

        binding
            .rebind(false, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind off: {e}"));
        assert!(
            state_of(&models_view, "state")
                .value()
                .unwrap_or_else(|e| panic!("value: {e}"))
                .is_none()
        );
        state.mutate(|tracker| {
            tracker
                .set(&[key("revision")], number(3))
                .expect("the write lands");
        });
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        assert_eq!(*revisions.borrow(), vec![1, 2]);
        binding
            .rebind(true, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind on: {e}"));
        assert_eq!(
            revision_of(
                &state_of(&models_view, "state")
                    .value()
                    .unwrap_or_else(|e| panic!("value: {e}"))
                    .expect("hydrated")
            ),
            3
        );
        assert_eq!(*revisions.borrow(), vec![1, 2, 3]);

        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

fn binding_deferred(
    services: Vec<Service>,
    provider: &RemoteServiceProvider,
    on_error: Option<pi_chord::handle::ErrorReporter>,
) -> pi_chord::consumer::RemoteServiceBinding {
    create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
        services,
        transport: create_loopback_service_transport(provider),
        bound: false,
        on_error: on_error.unwrap_or_else(pi_chord::handle::no_error_reporter),
        assert_access: None,
    })
    .expect("the binding allowlist has no duplicates")
}

#[test]
fn rejects_mode_mixing_and_unsupported_members() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.models").unwrap_or_else(|e| panic!("define: {e}"));
        let dialogs =
            define_service("test.question-dialog").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            singleton_definition(models.clone()),
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![
                    ("selected", JsonValue::Null),
                    ("revision", number(0)),
                ]))),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let error = expect_err(provider.spawn(&models, "wrong", method_only_implementation()));
        assert!(error_message(&error).contains("singleton"));

        let mut non_exposable = ServiceImplementation::new();
        non_exposable.value("request", Rc::new(()));
        let error = expect_err(provider.spawn(&dialogs, "invalid", non_exposable));
        assert!(error_message(&error).contains("not remotely exposable"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn hydrates_keyed_state_before_observe_handlers_and_fences_reused_keys() {
    let rt = runtime();
    rt.block_on(async {
        let dialogs =
            define_service("test.question-dialog").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        assert_eq!(
            provider.catalogue(),
            &[pi_chord::types::ServiceCatalogueEntry {
                service_id: dialogs.id.clone(),
                mode: ServiceMode::Keyed,
            }]
        );
        let (errors, on_error) = error_sink();
        let binding = binding_for(vec![dialogs.clone()], &provider, Some(on_error));
        let observed = Rc::new(RefCell::new(
            Vec::<(Option<JsonValue>, ServiceView, Context)>::new(),
        ));
        let stop = binding
            .observe(&dialogs, {
                let observed = observed.clone();
                Rc::new(move |view: ServiceView, context: Context| {
                    let question = view
                        .state("request")
                        .ok()
                        .and_then(|state| state.value().ok())
                        .flatten();
                    observed.borrow_mut().push((question, view, context));
                })
            })
            .unwrap_or_else(|e| panic!("observe: {e}"));

        let first_request = replicated_state(jo(vec![("question", js("First?"))]));
        let first_submit = Rc::new(std::cell::Cell::new(0u32));
        let close_first = provider
            .spawn(
                &dialogs,
                "invocation-1",
                keyed_dialogs_with(&first_request, &first_submit),
            )
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        // The spawn emission buffered while the keyed start was mid-boundary;
        // awaiting readiness flushes it, upstream's vi.waitFor.
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(observed.borrow().len(), 1);
        let (question, first_service, first_context) = observed.borrow()[0].clone();
        assert_eq!(question, Some(jo(vec![("question", js("First?"))])));

        first_request.mutate(|tracker| {
            tracker
                .set(&[key("question")], js("Updated?"))
                .expect("the write lands");
        });
        first_request
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        let (_question, live_first_service, _) = observed.borrow()[0].clone();
        let question = live_first_service
            .state("request")
            .and_then(|state| state.value())
            .unwrap_or_else(|e| panic!("state read: {e}"));
        assert_eq!(question, Some(jo(vec![("question", js("Updated?"))])));
        let accepted = first_service
            .call("submit", vec![js("yes")], background_context())
            .await
            .unwrap_or_else(|e| panic!("submit: {e}"));
        assert_eq!(
            accepted,
            Some(jo(vec![("accepted", JsonValue::Bool(true))]))
        );
        assert_eq!(first_submit.get(), 1);

        let _retained_view = first_service.clone();
        close_first().unwrap_or_else(|e| panic!("close first: {e}"));
        assert!(
            first_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
        );
        let error = retained_view_state(&first_service).expect_err("the observation is closed");
        assert!(error_message(&error).contains("observation is closed"));
        let error = first_service
            .call("submit", vec![js("late")], background_context())
            .await
            .expect_err("the observation is closed");
        assert!(error_message(&error).contains("observation is closed"));

        let second_request = replicated_state(jo(vec![("question", js("Again?"))]));
        let close_second = provider
            .spawn(
                &dialogs,
                "invocation-1",
                keyed_dialogs_with(&second_request, &Rc::new(std::cell::Cell::new(0u32))),
            )
            .unwrap_or_else(|e| panic!("spawn second: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready second: {e}"));
        assert_eq!(observed.borrow().len(), 2);
        assert!(errors.borrow().is_empty());
        let (question, _second_service, second_context) = observed.borrow()[1].clone();
        assert_eq!(question, Some(jo(vec![("question", js("Again?"))])));

        stop();
        close_second().unwrap_or_else(|e| panic!("close second: {e}"));
        assert!(
            second_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
        );
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

fn retained_view_state(view: &ServiceView) -> Result<Option<JsonValue>, ChordError> {
    view.state("request").and_then(|state| state.value())
}

fn keyed_dialogs_with(
    state: &MutableReplicatedState,
    submit_calls: &Rc<std::cell::Cell<u32>>,
) -> ServiceImplementation {
    let mut implementation = ServiceImplementation::new();
    implementation.state("request", state.clone());
    let submit_calls = submit_calls.clone();
    implementation.method(
        "submit",
        Rc::new(move |_args: Vec<JsonValue>, _context: Context| {
            let calls = submit_calls.clone();
            boxed(async move {
                calls.set(calls.get() + 1);
                Ok(Some(jo(vec![("accepted", JsonValue::Bool(true))])))
            })
        }),
    );
    implementation
}

#[test]
fn providers_fence_their_registrations() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.fence.models").unwrap_or_else(|e| panic!("define: {e}"));
        let absent = define_service("test.fence.absent").unwrap_or_else(|e| panic!("define: {e}"));
        let dialogs =
            define_service("test.fence.dialogs").unwrap_or_else(|e| panic!("define: {e}"));

        // The catalogue rejects duplicate IDs and spells its debug surface.
        let error = RemoteServiceProvider::new(vec![
            singleton_definition(models.clone()),
            singleton_definition(models.clone()),
        ])
        .expect_err("a duplicate catalogue rejects");
        assert!(error_message(&error).contains("duplicate IDs"));

        let provider = provider_for(&[&models, &absent]);
        assert!(format!("{provider:?}").contains("test.fence.models"));

        // Providing the same singleton twice mismatches the mode.
        provider
            .provide(&models, method_only_implementation())
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let error = provider
            .provide(&models, method_only_implementation())
            .expect_err("a second provider rejects");
        assert!(
            error_code(&error)
                .is_some_and(|code| code == RemoteServiceErrorCode::ServiceModeMismatch)
        );

        // Withdrawing an unprovided singleton is a no-op.
        provider
            .withdraw(&absent)
            .unwrap_or_else(|e| panic!("withdraw: {e}"));

        // The implementation lookup and subscriptions of an unprovided
        // singleton reject.
        let error = provider
            .use_service(&absent)
            .expect_err("an unprovided singleton rejects");
        assert!(
            error_code(&error).is_some_and(|code| code == RemoteServiceErrorCode::ServiceNotFound)
        );
        let listener: ServiceProviderListener = Rc::new(|_update, _context| ());
        let error = provider
            .subscribe(&absent.id, ServiceMode::Singleton, listener)
            .expect_err("an unprovided singleton rejects");
        assert!(
            error_code(&error).is_some_and(|code| code == RemoteServiceErrorCode::ServiceNotFound)
        );

        // Spawn guards: an empty key and a repeated live key.
        let error = provider
            .spawn(&models, "", method_only_implementation())
            .err()
            .expect("an empty key rejects");
        assert!(error.to_string().contains("key must not be empty"));

        // Disposal settles once; later use rejects.
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        let provider = RemoteServiceProvider::new(vec![
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("keyed provider: {e}"));
        let _close = provider
            .spawn(&dialogs, "dialog", method_only_implementation())
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        let error = provider
            .spawn(&dialogs, "dialog", method_only_implementation())
            .err()
            .expect("a repeated live key rejects");
        assert!(
            error_code(&error)
                .is_some_and(|code| code == RemoteServiceErrorCode::ServiceModeMismatch)
        );
        // A keyed registration rejects singleton-shaped lookups.
        let error = provider
            .use_service(&dialogs)
            .expect_err("a keyed registration rejects singleton lookups");
        assert!(
            error_code(&error).is_some_and(|code| code == RemoteServiceErrorCode::ServiceNotFound)
        );
    });
}

#[test]
fn bindings_fence_their_services_and_survive_disposal() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.bind.models").unwrap_or_else(|e| panic!("define: {e}"));
        let foreign = define_service("test.bind.foreign").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(&models, method_only_implementation())
            .unwrap_or_else(|e| panic!("provide: {e}"));

        // Duplicate service IDs reject at construction.
        let error =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![models.clone(), models.clone()],
                transport: create_loopback_service_transport(&provider),
                bound: true,
                on_error: pi_chord::handle::no_error_reporter(),
                assert_access: None,
            })
            .expect_err("duplicate service IDs reject");
        assert!(error_message(&error).contains("duplicate service IDs"));

        let binding = binding_for(vec![models.clone()], &provider, None);
        assert!(format!("{binding:?}").contains("bound: true"));

        // A foreign service rejects the allowlist.
        let error = binding
            .use_service(&foreign)
            .expect_err("a foreign service rejects");
        assert!(
            error_code(&error)
                .is_some_and(|code| code == RemoteServiceErrorCode::ServiceNotAllowed)
        );

        // A service used as one mode rejects the other.
        let handler: pi_chord::types::KeyedViewHandler = Rc::new(|_view, _context| ());
        let _unsubscribe = binding
            .observe(&models, handler)
            .unwrap_or_else(|e| panic!("observe: {e}"));
        let error = binding
            .use_service(&models)
            .expect_err("a keyed registration rejects singleton use");
        assert!(
            error_code(&error)
                .is_some_and(|code| code == RemoteServiceErrorCode::ServiceModeMismatch)
        );

        // Disposal settles once and rejects later readiness and rebinding.
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let error = binding
            .ready(background_context())
            .await
            .expect_err("readiness after disposal rejects");
        assert!(error_message(&error).contains("disposed"));
        let error = binding
            .rebind(true, background_context())
            .await
            .expect_err("rebinding after disposal rejects");
        assert!(error_message(&error).contains("disposed"));
    });
}

#[test]
fn closed_bindings_reject_member_use() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.closed.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(&models, method_only_implementation())
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let binding = binding_for(vec![models.clone()], &provider, None);
        let view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .rebind(false, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind: {e}"));
        let error = view
            .call("select", vec![], background_context())
            .await
            .expect_err("a closed binding rejects calls");
        assert!(error_message(&error).contains("binding is closed"));
        binding
            .rebind(true, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind: {e}"));
        let answer = view
            .call("select", vec![], background_context())
            .await
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert!(answer.is_none());
    });
}

struct SnapshotTransport {
    snapshot: Rc<RefCell<ServiceSubscriptionSnapshot>>,
    updates: Rc<RefCell<Vec<ServiceProviderUpdate>>>,
}

impl RemoteServiceTransport for SnapshotTransport {
    fn invoke(
        &self,
        _call: ServiceCall,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<Option<JsonValue>, ChordError>>>> {
        boxed(async { Err(ChordError::Message("unexpected invocation".to_string())) })
    }

    fn subscribe(
        &self,
        _service_id: String,
        _mode: ServiceMode,
        listener: ServiceProviderListener,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<ServiceSubscription, ChordError>>>> {
        let snapshot = self.snapshot.borrow().clone();
        let updates = self.updates.clone();
        let listener = listener.clone();
        boxed(async move {
            Ok(ServiceSubscription {
                snapshot: snapshot.clone(),
                activate: Box::new(move || {
                    for update in updates.borrow().clone() {
                        listener(&update, &background_context());
                    }
                    Ok(())
                }),
                close: Box::new(|_context| Box::pin(std::future::ready(Ok(())))),
            })
        })
    }
}

#[test]
fn bindings_reject_snapshots_that_break_their_contract() {
    let rt = runtime();
    rt.block_on(async {
        let models =
            define_service("test.snapshot.models").unwrap_or_else(|e| panic!("define: {e}"));
        let _keyed =
            define_service("test.snapshot.keyed").unwrap_or_else(|e| panic!("define: {e}"));

        // A singleton subscription whose snapshot is not singleton-shaped
        // rejects hydration.
        let wrong_mode = Rc::new(RefCell::new(ServiceSubscriptionSnapshot {
            service_id: "test.snapshot.models".to_string(),
            mode: ServiceMode::Keyed,
            instances: Vec::new(),
        }));
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(SnapshotTransport {
            snapshot: wrong_mode.clone(),
            updates: Rc::new(RefCell::new(Vec::new())),
        });
        let (errors, on_error) = error_sink();
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![models.clone()],
                transport,
                bound: true,
                on_error,
                assert_access: None,
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let _view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let error = binding
            .ready(background_context())
            .await
            .expect_err("readiness rejects the snapshot");
        assert!(error_message(&error).contains("invalid singleton snapshot"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let _ = (errors, wrong_mode.clone());
    });
}

#[test]
fn keyed_bindings_reject_snapshot_and_update_contract_breaks() {
    let rt = runtime();
    rt.block_on(async {
        let dialogs = define_service("test.snap.dialogs").unwrap_or_else(|e| panic!("define: {e}"));

        // A keyed subscription whose snapshot is not keyed-shaped reports the
        // break and rejects readiness.
        let wrong_mode = ServiceSubscriptionSnapshot {
            service_id: "test.snap.dialogs".to_string(),
            mode: ServiceMode::Singleton,
            instances: Vec::new(),
        };
        let (errors, on_error) = error_sink();
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(SnapshotTransport {
            snapshot: Rc::new(RefCell::new(wrong_mode)),
            updates: Rc::new(RefCell::new(Vec::new())),
        });
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![dialogs.clone()],
                transport,
                bound: true,
                on_error,
                assert_access: None,
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let handler: pi_chord::types::KeyedViewHandler = Rc::new(|_view, _context| ());
        let _subscription = binding
            .observe(&dialogs, handler)
            .unwrap_or_else(|e| panic!("observe: {e}"));
        let error = binding
            .ready(background_context())
            .await
            .expect_err("readiness rejects the keyed snapshot");
        assert!(error_message(&error).contains("wrong keyed snapshot"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert_eq!(errors.borrow().len(), 1);

        // A spawned instance without an address rejects.
        let (errors, on_error) = error_sink();
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(SnapshotTransport {
            snapshot: Rc::new(RefCell::new(ServiceSubscriptionSnapshot {
                service_id: "test.snap.dialogs".to_string(),
                mode: ServiceMode::Keyed,
                instances: vec![ServiceInstanceSnapshot {
                    instance: None,
                    members: Vec::new(),
                }],
            })),
            updates: Rc::new(RefCell::new(Vec::new())),
        });
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![dialogs.clone()],
                transport,
                bound: true,
                on_error,
                assert_access: None,
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let handler: pi_chord::types::KeyedViewHandler = Rc::new(|_view, _context| ());
        let _subscription = binding
            .observe(&dialogs, handler)
            .unwrap_or_else(|e| panic!("observe: {e}"));
        let error = binding
            .ready(background_context())
            .await
            .expect_err("a spawned snapshot without an address rejects");
        assert!(error_message(&error).contains("no address"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert_eq!(errors.borrow().len(), 1);

        // Lifecycle updates that do not fit the keyed contract report.
        let (errors, on_error) = error_sink();
        let address = ServiceInstanceAddress {
            key: "dialog".to_string(),
            generation: 1,
        };
        let updates = vec![
            ServiceProviderUpdate::Unavailable,
            ServiceProviderUpdate::State {
                instance: None,
                member: "state".to_string(),
                sequence: 1,
                ops: Vec::new(),
            },
        ];
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(SnapshotTransport {
            snapshot: Rc::new(RefCell::new(ServiceSubscriptionSnapshot {
                service_id: "test.snap.dialogs".to_string(),
                mode: ServiceMode::Keyed,
                instances: Vec::new(),
            })),
            updates: Rc::new(RefCell::new(updates)),
        });
        let binding =
            create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
                services: vec![dialogs.clone()],
                transport,
                bound: true,
                on_error,
                assert_access: None,
            })
            .unwrap_or_else(|e| panic!("binding: {e}"));
        let handler: pi_chord::types::KeyedViewHandler = Rc::new(|_view, _context| ());
        let _subscription = binding
            .observe(&dialogs, handler)
            .unwrap_or_else(|e| panic!("observe: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("readiness: {e}"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let reported = errors.borrow().clone();
        assert!(
            reported
                .iter()
                .any(|error| error.to_string().contains("singleton lifecycle update")),
            "the singleton-shaped update is reported: {reported:?}"
        );
        assert!(
            reported
                .iter()
                .any(|error| error.to_string().contains("no instance address")),
            "the address-less state update is reported: {reported:?}"
        );
        let _ = address;
    });
}

/// A transport whose snapshot, replay queue, listener, start delay, and
/// resolve-time hook the case controls, upstream's hand-rolled fixtures.
struct HookTransport {
    snapshot: Rc<RefCell<ServiceSubscriptionSnapshot>>,
    updates: Rc<RefCell<Vec<ServiceProviderUpdate>>>,
    listener: Rc<RefCell<Option<ServiceProviderListener>>>,
    on_resolve: Rc<RefCell<Option<Box<dyn Fn()>>>>,
    delay: usize,
}

impl HookTransport {
    fn singleton_snapshot(
        service_id: &str,
        members: Vec<ServiceMemberSnapshot>,
    ) -> Rc<RefCell<ServiceSubscriptionSnapshot>> {
        Rc::new(RefCell::new(ServiceSubscriptionSnapshot {
            service_id: service_id.to_string(),
            mode: ServiceMode::Singleton,
            instances: vec![ServiceInstanceSnapshot {
                instance: None,
                members,
            }],
        }))
    }

    fn keyed_snapshot(
        service_id: &str,
        address: &ServiceInstanceAddress,
        members: Vec<ServiceMemberSnapshot>,
    ) -> Rc<RefCell<ServiceSubscriptionSnapshot>> {
        Rc::new(RefCell::new(ServiceSubscriptionSnapshot {
            service_id: service_id.to_string(),
            mode: ServiceMode::Keyed,
            instances: vec![ServiceInstanceSnapshot {
                instance: Some(address.clone()),
                members,
            }],
        }))
    }
}

impl RemoteServiceTransport for HookTransport {
    fn invoke(
        &self,
        _call: ServiceCall,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<Option<JsonValue>, ChordError>>>> {
        boxed(async { Err(ChordError::Message("unexpected invocation".to_string())) })
    }

    fn subscribe(
        &self,
        _service_id: String,
        _mode: ServiceMode,
        listener: ServiceProviderListener,
        _context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<ServiceSubscription, ChordError>>>> {
        *self.listener.borrow_mut() = Some(listener.clone());
        let snapshot = self.snapshot.borrow().clone();
        let updates = self.updates.clone();
        let listener = listener.clone();
        let on_resolve = self.on_resolve.clone();
        let delay = self.delay;
        boxed(async move {
            for _ in 0..delay {
                tokio::task::yield_now().await;
            }
            if let Some(hook) = on_resolve.borrow_mut().take() {
                hook();
            }
            Ok(ServiceSubscription {
                snapshot,
                activate: Box::new(move || {
                    for update in updates.borrow().clone() {
                        listener(&update, &background_context());
                    }
                    Ok(())
                }),
                close: Box::new(|_| boxed(std::future::ready(Ok(())))),
            })
        })
    }
}

/// Routes every transport operation to the transport registered for the
/// service, the fixture a binding over several transports rides on.
struct DispatcherTransport {
    by_id: HashMap<String, Rc<dyn RemoteServiceTransport>>,
}

impl RemoteServiceTransport for DispatcherTransport {
    fn invoke(
        &self,
        call: ServiceCall,
        context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<Option<JsonValue>, ChordError>>>> {
        self.by_id[&call.service_id].invoke(call, context)
    }

    fn subscribe(
        &self,
        service_id: String,
        mode: ServiceMode,
        listener: ServiceProviderListener,
        context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<ServiceSubscription, ChordError>>>> {
        self.by_id[&service_id].subscribe(service_id, mode, listener, context)
    }
}

fn state_member(name: &str, sequence: u64, value: JsonValue) -> ServiceMemberSnapshot {
    ServiceMemberSnapshot::State {
        name: name.to_string(),
        sequence,
        ops: vec![Op::Replace(value)],
    }
}

fn method_member(name: &str) -> ServiceMemberSnapshot {
    ServiceMemberSnapshot::Method {
        name: name.to_string(),
    }
}

fn hook_binding(
    services: Vec<Service>,
    transport: Rc<dyn RemoteServiceTransport>,
    on_error: Option<pi_chord::handle::ErrorReporter>,
) -> pi_chord::consumer::RemoteServiceBinding {
    create_remote_service_binding(pi_chord::consumer::RemoteServiceBindingOptions {
        services,
        transport,
        bound: true,
        on_error: on_error.unwrap_or_else(pi_chord::handle::no_error_reporter),
        assert_access: None,
    })
    .unwrap_or_else(|e| panic!("binding: {e}"))
}

#[test]
fn bindings_fence_singleton_snapshot_contracts_at_install() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.hook.models").unwrap_or_else(|e| panic!("define: {e}"));

        // A snapshot whose single instance carries an address rejects the
        // install: singleton facades have no address.
        let (errors, on_error) = error_sink();
        let snapshot = HookTransport::singleton_snapshot("test.hook.models", vec![]);
        snapshot.borrow_mut().instances[0].instance = Some(ServiceInstanceAddress {
            key: "wrong".to_string(),
            generation: 1,
        });
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
            snapshot,
            updates: Rc::new(RefCell::new(Vec::new())),
            listener: Rc::new(RefCell::new(None)),
            on_resolve: Rc::new(RefCell::new(None)),
            delay: 0,
        });
        let binding = hook_binding(vec![models.clone()], transport, Some(on_error));
        let _view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let error = binding
            .ready(background_context())
            .await
            .expect_err("the wrong-address snapshot rejects");
        assert!(error_message(&error).contains("wrong address"));
        assert!(
            errors
                .borrow()
                .iter()
                .any(|error| error.to_string().contains("wrong address")),
            "the break is reported: {:?}",
            errors.borrow()
        );
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        // A snapshot with empty or duplicate member names rejects.
        for members in [
            vec![
                state_member("state", 0, JsonValue::Null),
                state_member("state", 0, JsonValue::Null),
            ],
            vec![method_member("")],
        ] {
            let (errors, on_error) = error_sink();
            let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
                snapshot: HookTransport::singleton_snapshot("test.hook.models", members),
                updates: Rc::new(RefCell::new(Vec::new())),
                listener: Rc::new(RefCell::new(None)),
                on_resolve: Rc::new(RefCell::new(None)),
                delay: 0,
            });
            let binding = hook_binding(vec![models.clone()], transport, Some(on_error));
            let _view = binding
                .use_service(&models)
                .unwrap_or_else(|e| panic!("use: {e}"));
            let error = binding
                .ready(background_context())
                .await
                .expect_err("the member-shape break rejects");
            assert!(error_message(&error).contains("invalid member descriptions"));
            assert_eq!(errors.borrow().len(), 1);
            binding
                .dispose(background_context())
                .await
                .unwrap_or_else(|e| panic!("dispose: {e}"));
        }
    });
}

#[test]
fn singleton_listeners_report_and_swallow_update_contract_breaks() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.hook.models").unwrap_or_else(|e| panic!("define: {e}"));
        let initial = jo(vec![("selected", JsonValue::Null), ("revision", number(0))]);

        // A replacement carrying an instance address is reported.
        let (errors, on_error) = error_sink();
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
            snapshot: HookTransport::singleton_snapshot(
                "test.hook.models",
                vec![
                    state_member("state", 0, initial.clone()),
                    method_member("select"),
                ],
            ),
            updates: Rc::new(RefCell::new(vec![ServiceProviderUpdate::Replaced {
                snapshot: ServiceInstanceSnapshot {
                    instance: Some(ServiceInstanceAddress {
                        key: "wrong".to_string(),
                        generation: 1,
                    }),
                    members: vec![state_member("state", 0, initial.clone())],
                },
            }])),
            listener: Rc::new(RefCell::new(None)),
            on_resolve: Rc::new(RefCell::new(None)),
            delay: 0,
        });
        let binding = hook_binding(vec![models.clone()], transport, Some(on_error));
        let _view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert!(
            errors.borrow().iter().any(|error| error
                .to_string()
                .contains("Singleton replacement has an instance address")),
            "the replacement break is reported: {:?}",
            errors.borrow()
        );
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        // A replacement dropping a member the facade already holds reports
        // the unknown member; a replacement flipping the member kind reports
        // the kind change.
        for (replacement, expected) in [
            (
                ServiceInstanceSnapshot {
                    instance: None,
                    members: vec![method_member("select")],
                },
                "Unknown remote service member",
            ),
            (
                ServiceInstanceSnapshot {
                    instance: None,
                    members: vec![method_member("state"), method_member("select")],
                },
                "changed kind",
            ),
        ] {
            let (errors, on_error) = error_sink();
            let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
                snapshot: HookTransport::singleton_snapshot(
                    "test.hook.models",
                    vec![
                        state_member("state", 0, initial.clone()),
                        method_member("select"),
                    ],
                ),
                updates: Rc::new(RefCell::new(vec![ServiceProviderUpdate::Replaced {
                    snapshot: replacement,
                }])),
                listener: Rc::new(RefCell::new(None)),
                on_resolve: Rc::new(RefCell::new(None)),
                delay: 0,
            });
            let binding = hook_binding(vec![models.clone()], transport, Some(on_error));
            let _view = binding
                .use_service(&models)
                .unwrap_or_else(|e| panic!("use: {e}"));
            binding
                .ready(background_context())
                .await
                .unwrap_or_else(|e| panic!("ready: {e}"));
            assert!(
                errors
                    .borrow()
                    .iter()
                    .any(|error| error.to_string().contains(expected)),
                "the {expected} break is reported: {:?}",
                errors.borrow()
            );
            binding
                .dispose(background_context())
                .await
                .unwrap_or_else(|e| panic!("dispose: {e}"));
        }

        // State updates for non-state members are reported: one for a member
        // described as a method, one for an undescribed member.
        for member in ["select", "ghost"] {
            let (errors, on_error) = error_sink();
            let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
                snapshot: HookTransport::singleton_snapshot(
                    "test.hook.models",
                    vec![
                        state_member("state", 0, initial.clone()),
                        method_member("select"),
                    ],
                ),
                updates: Rc::new(RefCell::new(vec![ServiceProviderUpdate::State {
                    instance: None,
                    member: member.to_string(),
                    sequence: 1,
                    ops: vec![Op::Replace(initial.clone())],
                }])),
                listener: Rc::new(RefCell::new(None)),
                on_resolve: Rc::new(RefCell::new(None)),
                delay: 0,
            });
            let binding = hook_binding(vec![models.clone()], transport, Some(on_error));
            let _view = binding
                .use_service(&models)
                .unwrap_or_else(|e| panic!("use: {e}"));
            binding
                .ready(background_context())
                .await
                .unwrap_or_else(|e| panic!("ready: {e}"));
            assert!(
                errors
                    .borrow()
                    .iter()
                    .any(|error| error.to_string().contains("targets non-state member")),
                "the {member} update is reported: {:?}",
                errors.borrow()
            );
            binding
                .dispose(background_context())
                .await
                .unwrap_or_else(|e| panic!("dispose: {e}"));
        }

        // Updates the closed-off listener sees are swallowed: a revision
        // mismatch after rebind, and keyed-shaped updates a singleton
        // listener cannot act on.
        let (errors, on_error) = error_sink();
        let snapshot = HookTransport::singleton_snapshot(
            "test.hook.models",
            vec![
                state_member("state", 0, initial.clone()),
                method_member("select"),
            ],
        );
        let _updates: Rc<RefCell<Vec<ServiceProviderUpdate>>> = Rc::new(RefCell::new(Vec::new()));
        let listener_cell: Rc<RefCell<Option<ServiceProviderListener>>> =
            Rc::new(RefCell::new(None));
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
            snapshot: snapshot.clone(),
            updates: Rc::new(RefCell::new(Vec::new())),
            listener: listener_cell.clone(),
            on_resolve: Rc::new(RefCell::new(None)),
            delay: 0,
        });
        let binding = hook_binding(vec![models.clone()], transport, Some(on_error));
        let _view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        binding
            .rebind(false, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind: {e}"));
        let listener = listener_cell.borrow().clone().expect("listener captured");
        listener(
            &ServiceProviderUpdate::State {
                instance: None,
                member: "state".to_string(),
                sequence: 1,
                ops: vec![Op::Replace(initial.clone())],
            },
            &background_context(),
        );
        for update in [
            ServiceProviderUpdate::Unavailable,
            ServiceProviderUpdate::Spawned {
                instance: ServiceInstanceSnapshot {
                    instance: Some(ServiceInstanceAddress {
                        key: "k".to_string(),
                        generation: 1,
                    }),
                    members: Vec::new(),
                },
            },
            ServiceProviderUpdate::Closed {
                instance: ServiceInstanceAddress {
                    key: "k".to_string(),
                    generation: 1,
                },
            },
            ServiceProviderUpdate::State {
                instance: Some(ServiceInstanceAddress {
                    key: "k".to_string(),
                    generation: 1,
                }),
                member: "state".to_string(),
                sequence: 1,
                ops: Vec::new(),
            },
        ] {
            listener(&update, &background_context());
        }
        assert!(
            errors.borrow().is_empty(),
            "stale and keyed-shaped updates stay silent: {:?}",
            errors.borrow()
        );
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn readiness_and_disposal_settle_through_slow_subscribe_boundaries() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.slow.models").unwrap_or_else(|e| panic!("define: {e}"));
        let initial = jo(vec![("selected", JsonValue::Null), ("revision", number(0))]);
        let slow_models_transport = || -> Rc<dyn RemoteServiceTransport> {
            Rc::new(HookTransport {
                snapshot: HookTransport::singleton_snapshot(
                    "test.slow.models",
                    vec![state_member("state", 0, initial.clone())],
                ),
                updates: Rc::new(RefCell::new(Vec::new())),
                listener: Rc::new(RefCell::new(None)),
                on_resolve: Rc::new(RefCell::new(None)),
                delay: 1,
            })
        };

        // Disposal collects a start that is still stored: the settle runs
        // inside the disposal, and no close failure surfaces.
        let binding = hook_binding(vec![models.clone()], slow_models_transport(), None);
        let _view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        let second = define_service("test.slow.second").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            singleton_definition(models.clone()),
            singleton_definition(second.clone()),
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(initial.clone())),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        provider
            .provide(&second, method_only_implementation())
            .unwrap_or_else(|e| panic!("provide second: {e}"));
        let second_transport = create_loopback_service_transport(&provider);

        // A disposal issued from inside the subscribe boundary marks the
        // binding disposed before the start resumes; the stale start closes
        // its subscription and readiness reports the disposal. The second
        // dispatcher acquires a second service mid-boundary instead, so the
        // readiness loop settles a second generation; both share the
        // fixture below, only their hooks differ.
        let mid_boundary_transport = |hook: Option<Box<dyn Fn()>>| {
            let by_id: HashMap<String, Rc<dyn RemoteServiceTransport>> = HashMap::from([
                (
                    models.id.clone(),
                    Rc::new(HookTransport {
                        snapshot: HookTransport::singleton_snapshot(
                            "test.slow.models",
                            vec![state_member("state", 0, initial.clone())],
                        ),
                        updates: Rc::new(RefCell::new(Vec::new())),
                        listener: Rc::new(RefCell::new(None)),
                        on_resolve: Rc::new(RefCell::new(hook)),
                        delay: 1,
                    }) as Rc<dyn RemoteServiceTransport>,
                ),
                (second.id.clone(), second_transport.clone()),
            ]);
            Rc::new(DispatcherTransport { by_id })
        };
        let late_binding: Rc<RefCell<Option<pi_chord::consumer::RemoteServiceBinding>>> =
            Rc::new(RefCell::new(None));
        let transport = mid_boundary_transport(Some({
            let late = late_binding.clone();
            Box::new(move || {
                let binding = late.borrow().clone().expect("binding registered");
                let disposal = binding.dispose(background_context());
                assert!(
                    pi_chord::future::settle_now(disposal)
                        .expect("the disposal settles inside the boundary")
                        .is_ok()
                );
            })
        }));
        let binding = hook_binding(vec![models.clone()], transport, None);
        *late_binding.borrow_mut() = Some(binding.clone());
        let _view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        let error = binding
            .ready(background_context())
            .await
            .expect_err("the mid-boundary disposal surfaces through readiness");
        assert!(error_message(&error).contains("binding is disposed"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));

        // A second service acquired mid-boundary bumps the readiness
        // revision, so the readiness loop settles a second generation.
        let late_binding: Rc<RefCell<Option<pi_chord::consumer::RemoteServiceBinding>>> =
            Rc::new(RefCell::new(None));
        let transport = mid_boundary_transport(Some({
            let late = late_binding.clone();
            let second = second.clone();
            Box::new(move || {
                let binding = late.borrow().clone().expect("binding registered");
                binding
                    .use_service(&second)
                    .unwrap_or_else(|e| panic!("use second: {e}"));
            })
        }));
        let binding = hook_binding(vec![models.clone(), second.clone()], transport, None);
        *late_binding.borrow_mut() = Some(binding.clone());
        let _view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn keyed_listeners_route_updates_and_fence_their_contract() {
    let rt = runtime();
    rt.block_on(async {
        let dialogs = define_service("test.hook.dialogs").unwrap_or_else(|e| panic!("define: {e}"));
        let address = ServiceInstanceAddress {
            key: "dialog".to_string(),
            generation: 1,
        };
        let initial = jo(vec![("question", js("First?"))]);
        let updates: Rc<RefCell<Vec<ServiceProviderUpdate>>> = Rc::new(RefCell::new(Vec::new()));
        let listener_cell: Rc<RefCell<Option<ServiceProviderListener>>> =
            Rc::new(RefCell::new(None));
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
            snapshot: HookTransport::keyed_snapshot(
                "test.hook.dialogs",
                &address,
                vec![
                    state_member("request", 0, initial.clone()),
                    method_member("submit"),
                ],
            ),
            updates: updates.clone(),
            listener: listener_cell.clone(),
            on_resolve: Rc::new(RefCell::new(None)),
            delay: 0,
        });
        let (errors, on_error) = error_sink();
        let binding = hook_binding(vec![dialogs.clone()], transport, Some(on_error));
        let views_sink: Rc<RefCell<Vec<ServiceView>>> = Rc::new(RefCell::new(Vec::new()));
        let _views = views_sink.clone();
        let observed = Rc::new(std::cell::Cell::new(0u32));
        let stop = binding
            .observe(&dialogs, {
                let observed = observed.clone();
                let views = views_sink.clone();
                Rc::new(move |view: ServiceView, _context: Context| {
                    observed.set(observed.get() + 1);
                    views.borrow_mut().push(view);
                })
            })
            .unwrap_or_else(|e| panic!("observe: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(observed.get(), 1);

        // A state update for the live generation routes to the facade; a
        // stale generation is swallowed.
        let listener = listener_cell.borrow().clone().expect("listener captured");
        listener(
            &ServiceProviderUpdate::State {
                instance: Some(address.clone()),
                member: "request".to_string(),
                sequence: 1,
                ops: vec![Op::Replace(jo(vec![("question", js("Updated?"))]))],
            },
            &background_context(),
        );
        listener(
            &ServiceProviderUpdate::State {
                instance: Some(ServiceInstanceAddress {
                    key: "dialog".to_string(),
                    generation: 2,
                }),
                member: "request".to_string(),
                sequence: 1,
                ops: Vec::new(),
            },
            &background_context(),
        );
        assert!(
            errors.borrow().is_empty(),
            "routed and stale updates stay silent: {:?}",
            errors.borrow()
        );
        let question = views_sink.borrow()[0]
            .state("request")
            .and_then(|state| state.value())
            .unwrap_or_else(|e| panic!("state read: {e}"));
        assert_eq!(question, Some(jo(vec![("question", js("Updated?"))])));

        // Rebinding keeps the observation: the keyed transition closes and
        // restarts the subscription, and the observation still routes.
        binding
            .rebind(false, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind off: {e}"));
        binding
            .rebind(true, background_context())
            .await
            .unwrap_or_else(|e| panic!("rebind on: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        assert_eq!(observed.get(), 2);

        // Unsubscribing twice is a no-op, and the last unsubscribe releases
        // the keyed binding: a fresh observe starts a fresh subscription.
        stop();
        stop();
        let _stop = binding
            .observe(&dialogs, {
                Rc::new(move |_view: ServiceView, _context: Context| ())
            })
            .unwrap_or_else(|e| panic!("observe again: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready again: {e}"));
        assert_eq!(observed.get(), 2);

        // A disposed binding's listener stays silent on later updates.
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        listener(
            &ServiceProviderUpdate::State {
                instance: Some(address.clone()),
                member: "request".to_string(),
                sequence: 2,
                ops: Vec::new(),
            },
            &background_context(),
        );
        assert!(
            errors.borrow().is_empty(),
            "the disposed listener stays silent: {:?}",
            errors.borrow()
        );
    });
}

#[test]
fn providers_fence_member_and_address_resolution() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.fence2.models").unwrap_or_else(|e| panic!("define: {e}"));
        let foreign =
            define_service("test.fence2.foreign").unwrap_or_else(|e| panic!("define: {e}"));
        let dialogs =
            define_service("test.fence2.dialogs").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            singleton_definition(models.clone()),
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        let state = replicated_state(jo(vec![("selected", JsonValue::Null)]));
        let mut implementation = ServiceImplementation::new();
        implementation.state("state", state);
        implementation.method(
            "select",
            Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(async { Ok(None) })),
        );
        provider
            .provide(&models, implementation)
            .unwrap_or_else(|e| panic!("provide: {e}"));

        let call_of = |member: &str, instance: Option<ServiceInstanceAddress>| ServiceCall {
            service_id: models.id.clone(),
            instance,
            member: member.to_string(),
            args: Vec::new(),
        };
        let error = pi_chord::future::drive_once(
            provider.invoke(call_of("absent", None), background_context()),
        )
        .ok()
        .and_then(Result::err)
        .expect("an unknown member rejects");
        assert!(
            error.to_string().contains("Unknown remote service member"),
            "unknown member: {error}"
        );
        let error = pi_chord::future::drive_once(
            provider.invoke(call_of("state", None), background_context()),
        )
        .ok()
        .and_then(Result::err)
        .expect("a state member is not callable");
        assert!(
            error.to_string().contains("is not a method"),
            "state member: {error}"
        );
        let error = pi_chord::future::drive_once(provider.invoke(
            call_of(
                "select",
                Some(ServiceInstanceAddress {
                    key: "k".to_string(),
                    generation: 1,
                }),
            ),
            background_context(),
        ))
        .ok()
        .and_then(Result::err)
        .expect("a singleton rejects an instance address");
        assert!(
            error.to_string().contains("is singleton"),
            "singleton: {error}"
        );

        let close = provider
            .spawn(&dialogs, "dialog", method_only_implementation())
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        let call_of_keyed = |member: &str, instance: Option<ServiceInstanceAddress>| ServiceCall {
            service_id: dialogs.id.clone(),
            instance,
            member: member.to_string(),
            args: Vec::new(),
        };
        let error = pi_chord::future::drive_once(
            provider.invoke(call_of_keyed("submit", None), background_context()),
        )
        .ok()
        .and_then(Result::err)
        .expect("a keyed service rejects an address-less call");
        assert!(error.to_string().contains("is keyed"), "keyed: {error}");
        let error = pi_chord::future::drive_once(provider.invoke(
            call_of_keyed(
                "submit",
                Some(ServiceInstanceAddress {
                    key: "absent".to_string(),
                    generation: 1,
                }),
            ),
            background_context(),
        ))
        .ok()
        .and_then(Result::err)
        .expect("an unknown keyed instance rejects");
        assert!(
            error.to_string().contains("no instance"),
            "unknown instance: {error}"
        );
        let error = pi_chord::future::drive_once(provider.invoke(
            ServiceCall {
                service_id: foreign.id,
                instance: None,
                member: "submit".to_string(),
                args: Vec::new(),
            },
            background_context(),
        ))
        .ok()
        .and_then(Result::err)
        .expect("a foreign service rejects");
        assert!(
            error.to_string().contains("is not allowlisted"),
            "foreign: {error}"
        );
        let _ = close;
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn providers_fence_stale_generations_and_subscription_gates() {
    let rt = runtime();
    rt.block_on(async {
        let dialogs =
            define_service("test.fence3.dialogs").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        let close_first = provider
            .spawn(&dialogs, "dialog", method_only_implementation())
            .unwrap_or_else(|e| panic!("spawn: {e}"));

        // The snapshot carries the live generation; closing and respawning
        // the key bumps it, so the retained address is stale.
        let listener: ServiceProviderListener =
            Rc::new(|_update: &ServiceProviderUpdate, _context: &Context| ());
        let subscription = provider
            .subscribe(&dialogs.id, ServiceMode::Keyed, listener)
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        let stale_address = subscription.snapshot.instances[0]
            .instance
            .clone()
            .expect("the keyed instance carries an address");
        // Activating twice is a no-op the second time.
        (subscription.activate)().unwrap_or_else(|e| panic!("activate: {e}"));
        (subscription.activate)().unwrap_or_else(|e| panic!("activate again: {e}"));
        (subscription.close)(None)
            .await
            .unwrap_or_else(|e| panic!("close: {e}"));
        close_first().unwrap_or_else(|e| panic!("close instance: {e}"));
        let _close_second = provider
            .spawn(&dialogs, "dialog", method_only_implementation())
            .unwrap_or_else(|e| panic!("respawn: {e}"));
        let error = pi_chord::future::drive_once(provider.invoke(
            ServiceCall {
                service_id: dialogs.id.clone(),
                instance: Some(stale_address),
                member: "submit".to_string(),
                args: Vec::new(),
            },
            background_context(),
        ))
        .ok()
        .and_then(Result::err)
        .expect("a stale generation rejects");
        assert!(error.to_string().contains("is stale"), "stale: {error}");

        // A closed subscription is skipped by later publications, and
        // disposal reports a Closed delivery that fails.
        let (errors, on_error) = error_sink();
        let state = replicated_state(jo(vec![("selected", JsonValue::Null)]));
        let mut implementation = ServiceImplementation::new();
        implementation.state("state", state.clone());
        let _close = provider
            .spawn(&dialogs, "later", implementation)
            .unwrap_or_else(|e| panic!("spawn later: {e}"));
        let subscriber = provider
            .subscribe(&dialogs.id, ServiceMode::Keyed, {
                Rc::new(|update: &ServiceProviderUpdate, _context: &Context| {
                    if matches!(update, ServiceProviderUpdate::Closed { .. }) {
                        panic!("listener failed on close");
                    }
                })
            })
            .unwrap_or_else(|e| panic!("subscribe second: {e}"));
        (subscriber.activate)().unwrap_or_else(|e| panic!("activate second: {e}"));
        let quiet = provider
            .subscribe(&dialogs.id, ServiceMode::Keyed, {
                Rc::new(|_update: &ServiceProviderUpdate, _context: &Context| {
                    panic!("a closed subscription still delivers");
                })
            })
            .unwrap_or_else(|e| panic!("subscribe third: {e}"));
        (quiet.close)(None)
            .await
            .unwrap_or_else(|e| panic!("close third: {e}"));
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        let error = provider
            .dispose()
            .expect_err("the Closed delivery failure aggregates");
        assert!(error.to_string().contains("listener failed on close"));
        let _ = (errors, on_error, close_first);
    });
}

#[test]
fn remote_service_endpoints_guard_their_grammar() {
    let rt = runtime();
    rt.block_on(async {
        let models =
            define_service("test.endpoint.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(&models, method_only_implementation())
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let endpoint = create_remote_service_endpoint(&provider);
        assert!(format!("{endpoint:?}").contains("RemoteServiceEndpoint"));

        let publish: pi_chord::types::ServiceUpdatePublisher = Rc::new(
            |_subscription_id: &str, _update: &ServiceProviderUpdate, _context: &Context| (),
        );

        // A plain invocation routes through the provider.
        let answer = endpoint
            .invoke(
                ServiceCall {
                    service_id: models.id.clone(),
                    instance: None,
                    member: "select".to_string(),
                    args: Vec::new(),
                },
                publish.clone(),
                background_context(),
            )
            .await
            .unwrap_or_else(|e| panic!("invoke: {e}"));
        assert!(answer.is_none());

        // The subscribe control call opens and unsubscriptions close; a
        // duplicate ID and an unknown service reject; a missing
        // subscription rejects.
        let snapshot = endpoint
            .invoke(
                pi_chord::services::wire::create_service_subscribe_call(
                    "sub-1",
                    &models.id,
                    ServiceMode::Singleton,
                ),
                publish.clone(),
                background_context(),
            )
            .await
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        assert!(snapshot.is_some());
        let error = endpoint
            .invoke(
                pi_chord::services::wire::create_service_subscribe_call(
                    "sub-1",
                    &models.id,
                    ServiceMode::Singleton,
                ),
                publish.clone(),
                background_context(),
            )
            .await
            .expect_err("a duplicate subscription ID rejects");
        assert!(error_message(&error).contains("already active"));
        let error = endpoint
            .invoke(
                pi_chord::services::wire::create_service_subscribe_call(
                    "sub-2",
                    "test.endpoint.absent",
                    ServiceMode::Singleton,
                ),
                publish.clone(),
                background_context(),
            )
            .await
            .expect_err("an unknown service rejects");
        assert!(error_message(&error).contains("not allowlisted"));
        let error = endpoint
            .invoke(
                pi_chord::services::wire::create_service_unsubscribe_call("sub-9"),
                publish.clone(),
                background_context(),
            )
            .await
            .expect_err("an unknown subscription rejects");
        assert!(error_message(&error).contains("was not found"));
        endpoint
            .invoke(
                pi_chord::services::wire::create_service_unsubscribe_call("sub-1"),
                publish.clone(),
                background_context(),
            )
            .await
            .unwrap_or_else(|e| panic!("unsubscribe: {e}"));

        // Disposal settles once; later invocations reject.
        endpoint.dispose();
        endpoint.dispose();
        let error = endpoint
            .invoke(
                pi_chord::services::wire::create_service_catalogue_call(),
                publish.clone(),
                background_context(),
            )
            .await
            .expect_err("a disposed endpoint rejects");
        assert!(error_message(&error).contains("endpoint is disposed"));

        // The implementation validator surfaces its rejections.
        let error =
            validate_remote_service_implementation(&models.id, &ServiceImplementation::new())
                .expect_err("an empty implementation rejects");
        assert!(error_message(&error).contains("has no members"));
        let mut non_exposable = ServiceImplementation::new();
        non_exposable.method(
            "read",
            Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(async { Ok(None) })),
        );
        non_exposable.value("payload", Rc::new(()));
        let error = validate_remote_service_implementation(&models.id, &non_exposable)
            .expect_err("a value member is not exposable");
        assert!(error_message(&error).contains("not remotely exposable"));
        let mut valid = ServiceImplementation::new();
        valid.method(
            "read",
            Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(async { Ok(None) })),
        );
        assert!(validate_remote_service_implementation(&models.id, &valid).is_ok());
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn deferred_bindings_settle_empty_starts_and_route_through_the_trait() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.trait.models").unwrap_or_else(|e| panic!("define: {e}"));
        let keyed = define_service("test.trait.keyed").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            singleton_definition(models.clone()),
            pi_chord::services::provider::ServiceProviderDefinition {
                service: keyed.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        provider
            .provide(
                &models,
                implementation_with_state_and_noop_select(replicated_state(jo(vec![(
                    "selected",
                    JsonValue::Null,
                )]))),
            )
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let close = provider
            .spawn(&keyed, "dialog", method_only_implementation())
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        let _ = close;

        // A deferred binding observes keyed services without starting them;
        // readiness settles the empty stored start.
        let binding = binding_deferred(vec![models.clone(), keyed.clone()], &provider, None);
        let observation_stop = binding
            .observe(&keyed, Rc::new(|_view: ServiceView, _context: Context| ()))
            .unwrap_or_else(|e| panic!("observe: {e}"));
        let _ = &observation_stop;
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));

        // A local service rejects the binding, and a disposed binding
        // rejects acquisition.
        let local =
            define_local_service("test.trait.local").unwrap_or_else(|e| panic!("define: {e}"));
        let error = binding
            .use_service(&local)
            .expect_err("a local service rejects");
        assert!(error.to_string().contains("process-local"));

        // Every surface routes through the trait object.
        let view = use_via_trait(&binding, &models).unwrap_or_else(|e| panic!("trait use: {e}"));
        let ready = use_via_trait_ready(&binding).await;
        ready.unwrap_or_else(|e| panic!("trait ready: {e}"));
        assert!(view.same_handle(&view));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        let error = binding
            .use_service(&models)
            .expect_err("a disposed binding rejects acquisition");
        assert!(error.to_string().contains("disposed"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

fn use_via_trait(
    services: &dyn pi_chord::types::RemoteServices,
    service: &Service,
) -> Result<ServiceView, ChordError> {
    services.use_service(service)
}

async fn use_via_trait_ready(
    binding: &pi_chord::consumer::RemoteServiceBinding,
) -> Result<(), ChordError> {
    pi_chord::types::RemoteServices::ready(binding, background_context()).await
}

#[test]
fn keyed_observations_release_their_binding_when_the_last_observer_stops() {
    let rt = runtime();
    rt.block_on(async {
        let dialogs =
            define_service("test.release.dialogs").unwrap_or_else(|e| panic!("define: {e}"));
        let address = ServiceInstanceAddress {
            key: "dialog".to_string(),
            generation: 1,
        };
        let listener_cell: Rc<RefCell<Option<ServiceProviderListener>>> =
            Rc::new(RefCell::new(None));
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
            snapshot: HookTransport::keyed_snapshot(
                "test.release.dialogs",
                &address,
                vec![state_member("request", 0, jo(vec![("question", js("Q?"))]))],
            ),
            updates: Rc::new(RefCell::new(Vec::new())),
            listener: listener_cell.clone(),
            on_resolve: Rc::new(RefCell::new(None)),
            delay: 0,
        });
        let (errors, on_error) = error_sink();
        let binding = hook_binding(vec![dialogs.clone()], transport, Some(on_error));
        let first_observation_stop = binding
            .observe(
                &dialogs,
                Rc::new(|_view: ServiceView, _context: Context| ()),
            )
            .unwrap_or_else(|e| panic!("observe: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));

        // The last unsubscribe closes the keyed subscription and releases
        // the binding; the release settles quietly.
        let release_stop = binding
            .observe(
                &dialogs,
                Rc::new(|_view: ServiceView, _context: Context| ()),
            )
            .unwrap_or_else(|e| panic!("observe first: {e}"));
        let close_stop = binding
            .observe(
                &dialogs,
                Rc::new(|_view: ServiceView, _context: Context| ()),
            )
            .unwrap_or_else(|e| panic!("observe second: {e}"));
        close_stop();
        release_stop();
        first_observation_stop();
        assert!(
            errors.borrow().is_empty(),
            "the release stays quiet: {:?}",
            errors.borrow()
        );

        // The closed subscription's listener stays silent afterwards.
        let listener = listener_cell.borrow().clone().expect("listener captured");
        listener(
            &ServiceProviderUpdate::State {
                instance: Some(address),
                member: "request".to_string(),
                sequence: 1,
                ops: Vec::new(),
            },
            &background_context(),
        );
        assert!(
            errors.borrow().is_empty(),
            "the released binding stays silent: {:?}",
            errors.borrow()
        );
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

#[test]
fn keyed_disposal_during_a_pending_start_closes_the_subscription() {
    let rt = runtime();
    rt.block_on(async {
        let dialogs =
            define_service("test.slowkey.dialogs").unwrap_or_else(|e| panic!("define: {e}"));
        let address = ServiceInstanceAddress {
            key: "dialog".to_string(),
            generation: 1,
        };
        let transport: Rc<dyn RemoteServiceTransport> = Rc::new(HookTransport {
            snapshot: HookTransport::keyed_snapshot(
                "test.slowkey.dialogs",
                &address,
                vec![state_member("request", 0, jo(vec![("question", js("Q?"))]))],
            ),
            updates: Rc::new(RefCell::new(Vec::new())),
            listener: Rc::new(RefCell::new(None)),
            on_resolve: Rc::new(RefCell::new(None)),
            delay: 1,
        });
        let (errors, on_error) = error_sink();
        let binding = hook_binding(vec![dialogs.clone()], transport, Some(on_error));
        let _stop = binding
            .observe(
                &dialogs,
                Rc::new(|_view: ServiceView, _context: Context| ()),
            )
            .unwrap_or_else(|e| panic!("observe: {e}"));
        // The keyed start parked at the transport boundary; disposal waits
        // for it, and the stale start closes its own subscription.
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        assert!(
            errors.borrow().is_empty(),
            "the disposal stays quiet: {:?}",
            errors.borrow()
        );
    });
}

#[test]
fn remote_views_reject_value_reads_and_delegate_through_view_targets() {
    let rt = runtime();
    rt.block_on(async {
        let models = define_service("test.value.models").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = provider_for(&[&models]);
        provider
            .provide(&models, method_only_implementation())
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let binding = binding_for(vec![models.clone()], &provider, None);
        let view = binding
            .use_service(&models)
            .unwrap_or_else(|e| panic!("use: {e}"));
        binding
            .ready(background_context())
            .await
            .unwrap_or_else(|e| panic!("ready: {e}"));
        let error: Result<u32, ChordError> = view
            .with_value("select", |_never: &dyn std::any::Any| {
                unreachable!("the read rejects before the closure")
            });
        let error = error.expect_err("a remote facade member is not a value");
        assert!(error_message(&error).contains("Remote service members are not values"));
        binding
            .dispose(background_context())
            .await
            .unwrap_or_else(|e| panic!("dispose: {e}"));
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
    });
}

#[test]
fn spawn_closures_settle_after_disposal() {
    let rt = runtime();
    rt.block_on(async {
        let dialogs =
            define_service("test.close.dialogs").unwrap_or_else(|e| panic!("define: {e}"));
        let provider = RemoteServiceProvider::new(vec![
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        let close = provider
            .spawn(&dialogs, "dialog", method_only_implementation())
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        // Disposal removes the instance; the retained close is a no-op.
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("provider dispose: {e}"));
        close().unwrap_or_else(|e| panic!("close after dispose: {e}"));
    });
}
