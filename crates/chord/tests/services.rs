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

use std::cell::RefCell;
use std::rc::Rc;

use pi_chord::api::{
    create_remote_service_binding, define_local_service, define_service, replicated_state,
};
use pi_chord::context::{Context, background_context};
use pi_chord::delta::{Op, Seg};
use pi_chord::errors::{ChordError, RemoteServiceErrorCode};
use pi_chord::handle::{ServiceImplementation, ServiceView};
use pi_chord::services::loopback::create_loopback_service_transport;
use pi_chord::services::provider::{RemoteServiceProvider, singleton_definition};
use pi_chord::services::state::MutableReplicatedState;
use pi_chord::types::{
    JsonValue, RemoteServiceTransport, Service, ServiceCall, ServiceMemberSnapshot, ServiceMode,
    ServiceProviderListener, ServiceProviderUpdate, ServiceSubscription,
    ServiceSubscriptionSnapshot,
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
            instances: vec![pi_chord::types::ServiceInstanceSnapshot {
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
