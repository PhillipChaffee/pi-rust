//! The `$chord.service` wire protocol suite, ported from upstream
//! `test/service-wire.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The control-call encoding, the strict parse surface for calls,
//! catalogues, snapshots, and provider updates, the per-subscription state
//! codecs' dictionary isolation, and the endpoint's subscribe/cleanup
//! lifecycle. Cases upstream spells as object literals build the owned
//! [`JsonValue`] trees here.

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

use std::cell::RefCell;
use std::rc::Rc;

use pi_chord::context::{Context, background_context};
use pi_chord::delta::{Op, PathRef, Seg, WireOp};
use pi_chord::handle::ServiceImplementation;
use pi_chord::services::provider::{RemoteServiceProvider, create_remote_service_endpoint};
use pi_chord::services::state::MutableReplicatedState;
use pi_chord::services::state_codec::{create_service_state_decoder, create_service_state_encoder};
use pi_chord::services::wire::{
    ServiceControlCall, WireServiceInstanceSnapshot, WireServiceMemberSnapshot,
    WireServiceSubscriptionSnapshot, catalogue_to_json, create_service_catalogue_call,
    create_service_subscribe_call, create_service_unsubscribe_call, decode_service_control_call,
    object, parse_service_call, parse_service_catalogue, parse_service_provider_update,
    parse_service_subscription_snapshot, parse_wire_service_provider_update,
    parse_wire_service_subscription_snapshot, snapshot_to_json,
};
use pi_chord::types::{
    JsonValue, ServiceCatalogueEntry, ServiceInstanceAddress, ServiceInstanceSnapshot,
    ServiceMemberSnapshot, ServiceMode, ServiceProviderUpdate, ServiceSubscriptionSnapshot,
};

fn key(text: &str) -> Seg {
    Seg::Key(text.to_string())
}

fn js(text: &str) -> JsonValue {
    JsonValue::Str(text.to_string())
}

fn jo(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    object(entries)
}

fn number(value: u64) -> JsonValue {
    JsonValue::Number(pi_chord::types::JsonNumber::from(value))
}

#[test]
fn encodes_control_calls_and_validates_service_values() {
    assert_eq!(
        decode_service_control_call(&create_service_catalogue_call()),
        Some(ServiceControlCall::Catalogue)
    );
    let entries = parse_catalogue(&catalogue_to_json(&[
        catalogue_entry("pi.models", ServiceMode::Singleton),
        catalogue_entry("pi.dialogs", ServiceMode::Keyed),
    ]));
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].service_id, "pi.models");
    assert_eq!(entries[1].mode, ServiceMode::Keyed);
    let subscribe =
        create_service_subscribe_call("subscription-1", "pi.models", ServiceMode::Singleton);
    assert_eq!(
        decode_service_control_call(&subscribe),
        Some(ServiceControlCall::Subscribe {
            subscription_id: "subscription-1".to_string(),
            service_id: "pi.models".to_string(),
            mode: ServiceMode::Singleton,
        })
    );
    assert_eq!(
        decode_service_control_call(&create_service_unsubscribe_call("subscription-1")),
        Some(ServiceControlCall::Unsubscribe {
            subscription_id: "subscription-1".to_string(),
        })
    );
    let call = jo(vec![
        ("serviceId", js("pi.question-dialog")),
        (
            "instance",
            jo(vec![("key", js("invocation-1")), ("generation", number(2))]),
        ),
        ("member", js("submit")),
        (
            "args",
            JsonValue::Array(vec![jo(vec![
                ("outcome", js("selected")),
                ("index", number(0)),
            ])]),
        ),
    ]);
    let parsed = expect_ok(parse_service_call(&call));
    assert_eq!(parsed.member, "submit");
    assert_eq!(
        parsed.instance,
        Some(ServiceInstanceAddress {
            key: "invocation-1".to_string(),
            generation: 2,
        })
    );
}

#[test]
fn rejects_malformed_service_values() {
    let extra = jo(vec![
        ("serviceId", js("pi.models")),
        ("member", js("list")),
        ("args", JsonValue::Array(vec![])),
        ("extra", JsonValue::Bool(true)),
    ]);
    let error = expect_err(parse_service_call(&extra));
    assert!(error.to_string().contains("Invalid service call"));

    let bad_mode = JsonValue::Array(vec![jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("unknown")),
    ])]);
    let error = expect_err(parse_service_catalogue(&bad_mode));
    assert!(error.to_string().contains("Invalid service catalogue"));

    let zero_sequence = jo(vec![
        ("type", js("state")),
        ("member", js("state")),
        ("sequence", number(0)),
        ("ops", JsonValue::Array(vec![])),
    ]);
    let error = expect_err(parse_service_provider_update(&zero_sequence));
    assert!(error.to_string().contains("Invalid service state update"));

    let malformed_op = jo(vec![
        ("type", js("state")),
        ("member", js("state")),
        ("sequence", number(1)),
        (
            "ops",
            JsonValue::Array(vec![jo(vec![("bogus", JsonValue::Null)])]),
        ),
    ]);
    assert!(parse_wire_service_provider_update(&malformed_op).is_err());
}

#[test]
fn validates_decoded_and_wire_snapshots_and_updates() {
    let snapshot = singleton_snapshot(jo(vec![("revision", number(1))]));
    let json = snapshot_to_json(&snapshot);
    let reparsed = parse_snapshot(&json);
    assert_eq!(reparsed.service_id, snapshot.service_id);
    assert_eq!(reparsed.instances.len(), snapshot.instances.len());

    let mut encoder = encoder();
    let wire_snapshot = encoder
        .encode_snapshot(&snapshot)
        .unwrap_or_else(|error| panic!("snapshot encodes: {error}"));
    let parsed_wire = parse_wire_snapshot(&wire_snapshot_to_json(&wire_snapshot));
    assert_eq!(parsed_wire.service_id, snapshot.service_id);
    assert_eq!(parsed_wire.instances.len(), snapshot.instances.len());

    let update = set_update("state", 1, 1);
    let wire_update = encoder
        .encode_update(&update)
        .unwrap_or_else(|error| panic!("update encodes: {error}"));
    assert!(parse_wire_service_provider_update(&wire_update_to_json(&wire_update)).is_ok());
    assert!(parse_service_provider_update(&update_to_json(&update)).is_ok());
}

#[test]
fn keeps_one_operation_codec_pair_for_one_subscription_state() {
    let mut encoder = encoder();
    let snapshot = singleton_snapshot(jo(vec![("revision", number(0))]));
    let mut batch_decoder = decoder();
    let decoded = batch_decoder
        .decode_snapshot(
            &encoder
                .encode_snapshot(&snapshot)
                .unwrap_or_else(|e| panic!("snapshot: {e}")),
        )
        .unwrap_or_else(|e| panic!("decode snapshot: {e}"));
    assert_eq!(decoded.service_id, snapshot.service_id);

    let first = set_update("state", 1, 1);
    let second = set_update("state", 2, 2);
    let first_wire = encoder
        .encode_update(&first)
        .unwrap_or_else(|e| panic!("first: {e}"));
    let second_wire = encoder
        .encode_update(&second)
        .unwrap_or_else(|e| panic!("second: {e}"));
    assert_ops(
        &first_wire,
        &[WireOp::Set {
            path: PathRef::Inline(vec![key("revision")]),
            value: number(1),
        }],
    );
    assert_ops(
        &second_wire,
        &[
            WireOp::Define {
                id: 0,
                path: vec![key("revision")],
            },
            WireOp::Set {
                path: PathRef::Id(0),
                value: number(2),
            },
        ],
    );
    assert_eq!(
        batch_decoder
            .decode_update(&first_wire)
            .unwrap_or_else(|e| panic!("df: {e}")),
        first
    );
    assert_eq!(
        batch_decoder
            .decode_update(&second_wire)
            .unwrap_or_else(|e| panic!("ds: {e}")),
        second
    );
}

#[test]
fn isolates_operation_dictionaries_between_states_and_subscriptions() {
    let snapshot = snapshot_with_two_states();
    let mut first_encoder = encoder();
    let mut second_encoder = encoder();
    let mut first_decoder = decoder();
    let _ = first_decoder
        .decode_snapshot(
            &first_encoder
                .encode_snapshot(&snapshot)
                .unwrap_or_else(|e| panic!("first: {e}")),
        )
        .unwrap_or_else(|e| panic!("first decode: {e}"));
    let mut second_decoder = decoder();
    let _ = second_decoder
        .decode_snapshot(
            &second_encoder
                .encode_snapshot(&snapshot)
                .unwrap_or_else(|e| panic!("second: {e}")),
        )
        .unwrap_or_else(|e| panic!("decode second: {e}"));

    let update = |member: &str, sequence: u64, revision: u64| ServiceProviderUpdate::State {
        instance: None,
        member: member.to_string(),
        sequence,
        ops: vec![Op::Set {
            path: vec![key("revision")],
            value: number(revision),
        }],
    };
    let first_left = first_encoder
        .encode_update(&update("left", 1, 1))
        .unwrap_or_else(|e| panic!("fl: {e}"));
    let first_right = first_encoder
        .encode_update(&update("right", 1, 1))
        .unwrap_or_else(|e| panic!("fr: {e}"));
    let second_left = first_encoder
        .encode_update(&update("left", 2, 2))
        .unwrap_or_else(|e| panic!("sl: {e}"));
    let second_right = first_encoder
        .encode_update(&update("right", 2, 2))
        .unwrap_or_else(|e| panic!("sr: {e}"));
    for wire in [&first_left, &first_right] {
        assert_sets(wire, false, &number(1));
    }
    for wire in [&second_left, &second_right] {
        assert_sets(wire, true, &number(2));
    }
    assert_eq!(
        first_decoder
            .decode_update(&first_left)
            .unwrap_or_else(|e| panic!("dfl: {e}")),
        update("left", 1, 1)
    );
    assert_eq!(
        first_decoder
            .decode_update(&first_right)
            .unwrap_or_else(|e| panic!("dfr: {e}")),
        update("right", 1, 1)
    );
    assert_eq!(
        first_decoder
            .decode_update(&second_left)
            .unwrap_or_else(|e| panic!("dsl: {e}")),
        update("left", 2, 2)
    );
    assert_eq!(
        first_decoder
            .decode_update(&second_right)
            .unwrap_or_else(|e| panic!("dsr: {e}")),
        update("right", 2, 2)
    );

    let independent_left = second_encoder
        .encode_update(&update("left", 1, 1))
        .unwrap_or_else(|e| panic!("il: {e}"));
    assert_eq!(
        second_decoder
            .decode_update(&independent_left)
            .unwrap_or_else(|e| panic!("dil: {e}")),
        update("left", 1, 1)
    );

    let left_base = ServiceProviderUpdate::State {
        instance: None,
        member: "left".to_string(),
        sequence: 3,
        ops: vec![Op::Replace(jo(vec![("revision", number(3))]))],
    };
    let encoded_base = first_encoder
        .encode_update(&left_base)
        .unwrap_or_else(|e| panic!("lb: {e}"));
    assert_eq!(
        first_decoder
            .decode_update(&encoded_base)
            .unwrap_or_else(|e| panic!("dlb: {e}")),
        left_base
    );
    let third_right = first_encoder
        .encode_update(&update("right", 3, 3))
        .unwrap_or_else(|e| panic!("tr: {e}"));
    assert_ops(
        &third_right,
        &[WireOp::Set {
            path: PathRef::Id(0),
            value: number(3),
        }],
    );
}

fn assert_sets(
    wire: &pi_chord::services::wire::WireServiceProviderUpdate,
    interned: bool,
    value: &JsonValue,
) {
    let pi_chord::services::wire::WireServiceProviderUpdate::State { ops, .. } = wire else {
        panic!("state update");
    };
    if interned {
        assert_eq!(
            ops[0],
            WireOp::Define {
                id: 0,
                path: vec![key("revision")],
            }
        );
        assert_eq!(
            ops[1],
            WireOp::Set {
                path: PathRef::Id(0),
                value: value.clone(),
            }
        );
    } else {
        assert_eq!(
            ops[0],
            WireOp::Set {
                path: PathRef::Inline(vec![key("revision")]),
                value: value.clone(),
            }
        );
    }
}

fn snapshot_with_two_states() -> ServiceSubscriptionSnapshot {
    ServiceSubscriptionSnapshot {
        service_id: "pi.states".to_string(),
        mode: ServiceMode::Singleton,
        instances: vec![ServiceInstanceSnapshot {
            instance: None,
            members: vec![
                ServiceMemberSnapshot::State {
                    name: "left".to_string(),
                    sequence: 0,
                    ops: vec![Op::Replace(jo(vec![("revision", number(0))]))],
                },
                ServiceMemberSnapshot::State {
                    name: "right".to_string(),
                    sequence: 0,
                    ops: vec![Op::Replace(jo(vec![("revision", number(0))]))],
                },
            ],
        }],
    }
}

#[test]
fn creates_and_removes_keyed_instance_codecs_with_their_lifecycle() {
    let mut enc = encoder();
    let mut dec = decoder();
    let empty = ServiceSubscriptionSnapshot {
        service_id: "pi.dialogs".to_string(),
        mode: ServiceMode::Keyed,
        instances: vec![],
    };
    let decoded = dec
        .decode_snapshot(
            &enc.encode_snapshot(&empty)
                .unwrap_or_else(|e| panic!("encode: {e}")),
        )
        .unwrap_or_else(|e| panic!("decode: {e}"));
    assert_eq!(decoded.instances.len(), 0);
    let address = ServiceInstanceAddress {
        key: "dialog-1".to_string(),
        generation: 1,
    };
    let spawned = ServiceProviderUpdate::Spawned {
        instance: ServiceInstanceSnapshot {
            instance: Some(address.clone()),
            members: vec![ServiceMemberSnapshot::State {
                name: "request".to_string(),
                sequence: 0,
                ops: vec![Op::Replace(jo(vec![("value", number(0))]))],
            }],
        },
    };
    let encoded_spawn = enc
        .encode_update(&spawned)
        .unwrap_or_else(|e| panic!("spawn: {e}"));
    assert_eq!(
        dec.decode_update(&encoded_spawn)
            .unwrap_or_else(|e| panic!("spawn decode: {e}")),
        spawned
    );
    let update = keyed_update(&address, 1, 1);
    assert_eq!(
        dec.decode_update(
            &enc.encode_update(&update)
                .unwrap_or_else(|e| panic!("update: {e}"))
        )
        .unwrap_or_else(|e| panic!("update decode: {e}")),
        update
    );
    let closed = ServiceProviderUpdate::Closed {
        instance: address.clone(),
    };
    assert_eq!(
        dec.decode_update(
            &enc.encode_update(&closed)
                .unwrap_or_else(|e| panic!("close: {e}"))
        )
        .unwrap_or_else(|e| panic!("close decode: {e}")),
        closed
    );
    let error = expect_err(enc.encode_update(&keyed_update(&address, 2, 2)));
    assert!(error.to_string().contains("Unknown service state"));
}

fn keyed_update(
    address: &ServiceInstanceAddress,
    sequence: u64,
    value: u64,
) -> ServiceProviderUpdate {
    ServiceProviderUpdate::State {
        instance: Some(address.clone()),
        member: "request".to_string(),
        sequence,
        ops: vec![Op::Set {
            path: vec![key("value")],
            value: number(value),
        }],
    }
}

#[test]
fn remote_service_endpoints_publish_and_clean_up_provider_subscriptions() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|error| panic!("runtime: {error}"));
    runtime.block_on(async {
        let service =
            pi_chord::api::define_service("test.counter").unwrap_or_else(|e| panic!("define: {e}"));
        let provider =
            RemoteServiceProvider::new(vec![pi_chord::services::provider::singleton_definition(
                service.clone(),
            )])
            .unwrap_or_else(|e| panic!("provider: {e}"));
        let state = MutableReplicatedState::new(jo(vec![("value", number(0))]));
        let mut implementation = ServiceImplementation::new();
        implementation.state("state", state.clone());
        provider
            .provide(&service, implementation)
            .unwrap_or_else(|e| panic!("provide: {e}"));
        let endpoint = create_remote_service_endpoint(&provider);
        let updates = Rc::new(RefCell::new(Vec::<ServiceProviderUpdate>::new()));
        let publisher: pi_chord::types::ServiceUpdatePublisher = {
            let updates = updates.clone();
            Rc::new(
                move |_id: &str, update: &ServiceProviderUpdate, _context: &Context| {
                    updates.borrow_mut().push(update.clone());
                },
            )
        };

        let catalogue = endpoint
            .invoke(
                create_service_catalogue_call(),
                publisher.clone(),
                background_context(),
            )
            .await
            .unwrap_or_else(|e| panic!("catalogue: {e}"));
        assert_eq!(
            catalogue,
            Some(catalogue_to_json(&[catalogue_entry(
                service.id.as_str(),
                ServiceMode::Singleton
            )]))
        );
        let snapshot = endpoint
            .invoke(
                create_service_subscribe_call(
                    "subscription-1",
                    service.id.as_str(),
                    ServiceMode::Singleton,
                ),
                publisher.clone(),
                background_context(),
            )
            .await
            .unwrap_or_else(|e| panic!("subscribe: {e}"));
        let Some(snapshot) = snapshot else {
            panic!("subscribe returns the snapshot");
        };
        let reparsed = parse_snapshot(&snapshot);
        assert_eq!(reparsed.service_id, service.id);
        assert_eq!(reparsed.mode, ServiceMode::Singleton);

        state.mutate(|tracker| {
            tracker
                .set(&[key("value")], number(1))
                .unwrap_or_else(|e| panic!("set: {e}"));
        });
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        assert_eq!(updates.borrow().len(), 1);
        endpoint.dispose();
        state.mutate(|tracker| {
            tracker
                .set(&[key("value")], number(2))
                .unwrap_or_else(|e| panic!("set: {e}"));
        });
        state
            .publish(&background_context())
            .unwrap_or_else(|e| panic!("publish: {e}"));
        assert_eq!(updates.borrow().len(), 1);
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}

fn set_update(member: &str, sequence: u64, revision: u64) -> ServiceProviderUpdate {
    ServiceProviderUpdate::State {
        instance: None,
        member: member.to_string(),
        sequence,
        ops: vec![Op::Set {
            path: vec![key("revision")],
            value: number(revision),
        }],
    }
}

fn catalogue_entry(service_id: &str, mode: ServiceMode) -> ServiceCatalogueEntry {
    ServiceCatalogueEntry {
        service_id: service_id.to_string(),
        mode,
    }
}

fn parse_catalogue(value: &JsonValue) -> Vec<ServiceCatalogueEntry> {
    parse_service_catalogue(value)
        .map_err(|error| std::format!("{error}"))
        .unwrap_or_default()
}

fn parse_snapshot(value: &JsonValue) -> ServiceSubscriptionSnapshot {
    parse_service_subscription_snapshot(value)
        .unwrap_or_else(|error| panic!("snapshot parses: {error}"))
}

fn parse_wire_snapshot(value: &JsonValue) -> WireServiceSubscriptionSnapshot {
    parse_wire_service_subscription_snapshot(value)
        .unwrap_or_else(|error| panic!("wire snapshot parses: {error}"))
}

fn encoder() -> pi_chord::services::state_codec::ServiceStateEncoder {
    create_service_state_encoder()
}

fn decoder() -> pi_chord::services::state_codec::ServiceStateDecoder {
    create_service_state_decoder()
}

fn singleton_snapshot(value: JsonValue) -> ServiceSubscriptionSnapshot {
    ServiceSubscriptionSnapshot {
        service_id: "pi.models".to_string(),
        mode: ServiceMode::Singleton,
        instances: vec![ServiceInstanceSnapshot {
            instance: None,
            members: vec![ServiceMemberSnapshot::State {
                name: "state".to_string(),
                sequence: 0,
                ops: vec![Op::Replace(value)],
            }],
        }],
    }
}

/// Settles a result the case expects to carry a value.
#[must_use]
fn expect_ok<T, E>(result: Result<T, E>) -> T
where
    T: std::fmt::Debug,
    E: std::fmt::Display + std::fmt::Debug,
{
    #[allow(
        clippy::expect_used,
        reason = "test assertions surface the unexpected failure at the failing case only"
    )]
    result.expect("the case's parse succeeds")
}

/// Settles a result the case expects to carry an error.
#[must_use]
fn expect_err<T, E>(result: Result<T, E>) -> E
where
    T: std::fmt::Debug,
    E: std::fmt::Debug,
{
    #[allow(
        clippy::expect_used,
        reason = "test assertions surface the unexpected success at the failing case only"
    )]
    result.expect_err("the case's parse rejects")
}

fn assert_ops(wire: &pi_chord::services::wire::WireServiceProviderUpdate, expected: &[WireOp]) {
    let pi_chord::services::wire::WireServiceProviderUpdate::State { ops, .. } = wire else {
        panic!("state update");
    };
    assert_eq!(ops, expected);
}
fn wire_snapshot_to_json(snapshot: &WireServiceSubscriptionSnapshot) -> JsonValue {
    object(vec![
        ("serviceId", js(snapshot.service_id.as_str())),
        ("mode", js(snapshot.mode.as_str())),
        (
            "instances",
            JsonValue::Array(
                snapshot
                    .instances
                    .iter()
                    .map(|instance| {
                        let mut fields = Vec::new();
                        if let Some(address) = &instance.instance {
                            fields.push((
                                "instance",
                                jo(vec![
                                    ("key", js(address.key.as_str())),
                                    ("generation", number(address.generation)),
                                ]),
                            ));
                        }
                        fields.push((
                            "members",
                            JsonValue::Array(
                                instance
                                    .members
                                    .iter()
                                    .map(|member| match member {
                                        WireServiceMemberSnapshot::Method { name } => jo(vec![
                                            ("name", js(name.as_str())),
                                            ("kind", js("method")),
                                        ]),
                                        WireServiceMemberSnapshot::State {
                                            name,
                                            sequence,
                                            ops,
                                        } => jo(vec![
                                            ("name", js(name.as_str())),
                                            ("kind", js("state")),
                                            ("sequence", number(*sequence)),
                                            (
                                                "ops",
                                                JsonValue::Array(
                                                    ops.iter().map(WireOp::to_json).collect(),
                                                ),
                                            ),
                                        ]),
                                    })
                                    .collect(),
                            ),
                        ));
                        object(fields)
                    })
                    .collect(),
            ),
        ),
    ])
}

fn wire_update_to_json(update: &pi_chord::services::wire::WireServiceProviderUpdate) -> JsonValue {
    use pi_chord::services::wire::WireServiceProviderUpdate as Wire;
    match update {
        Wire::State {
            instance,
            member,
            sequence,
            ops,
        } => {
            let mut fields = vec![("type", js("state")), ("member", js(member.as_str()))];
            if let Some(address) = instance {
                fields.push((
                    "instance",
                    jo(vec![
                        ("key", js(address.key.as_str())),
                        ("generation", number(address.generation)),
                    ]),
                ));
            }
            fields.push(("sequence", number(*sequence)));
            fields.push((
                "ops",
                JsonValue::Array(ops.iter().map(WireOp::to_json).collect()),
            ));
            object(fields)
        }
        Wire::Unavailable => jo(vec![("type", js("unavailable"))]),
        Wire::Replaced { snapshot } => jo(vec![
            ("type", js("replaced")),
            ("snapshot", wire_instance_to_json(snapshot)),
        ]),
        Wire::Spawned { instance } => jo(vec![
            ("type", js("spawned")),
            ("instance", wire_instance_to_json(instance)),
        ]),
        Wire::Closed { instance } => jo(vec![
            ("type", js("closed")),
            (
                "instance",
                jo(vec![
                    ("key", js(instance.key.as_str())),
                    ("generation", number(instance.generation)),
                ]),
            ),
        ]),
    }
}

fn wire_instance_to_json(instance: &WireServiceInstanceSnapshot) -> JsonValue {
    let mut fields = Vec::new();
    if let Some(address) = &instance.instance {
        fields.push((
            "instance",
            jo(vec![
                ("key", js(address.key.as_str())),
                ("generation", number(address.generation)),
            ]),
        ));
    }
    fields.push((
        "members",
        JsonValue::Array(
            instance
                .members
                .iter()
                .map(|member| match member {
                    WireServiceMemberSnapshot::Method { name } => {
                        jo(vec![("name", js(name.as_str())), ("kind", js("method"))])
                    }
                    WireServiceMemberSnapshot::State {
                        name,
                        sequence,
                        ops,
                    } => jo(vec![
                        ("name", js(name.as_str())),
                        ("kind", js("state")),
                        ("sequence", number(*sequence)),
                        (
                            "ops",
                            JsonValue::Array(ops.iter().map(WireOp::to_json).collect()),
                        ),
                    ]),
                })
                .collect(),
        ),
    ));
    object(fields)
}

fn update_to_json(update: &ServiceProviderUpdate) -> JsonValue {
    match update {
        ServiceProviderUpdate::State {
            instance,
            member,
            sequence,
            ops,
        } => {
            let mut fields = vec![("type", js("state")), ("member", js(member.as_str()))];
            if let Some(address) = instance {
                fields.push((
                    "instance",
                    jo(vec![
                        ("key", js(address.key.as_str())),
                        ("generation", number(address.generation)),
                    ]),
                ));
            }
            fields.push(("sequence", number(*sequence)));
            fields.push((
                "ops",
                JsonValue::Array(ops.iter().map(Op::to_json).collect()),
            ));
            object(fields)
        }
        ServiceProviderUpdate::Unavailable => jo(vec![("type", js("unavailable"))]),
        ServiceProviderUpdate::Replaced { snapshot } => jo(vec![
            ("type", js("replaced")),
            ("snapshot", decoded_instance_to_json(snapshot)),
        ]),
        ServiceProviderUpdate::Spawned { instance } => jo(vec![
            ("type", js("spawned")),
            ("instance", decoded_instance_to_json(instance)),
        ]),
        ServiceProviderUpdate::Closed { instance } => jo(vec![
            ("type", js("closed")),
            (
                "instance",
                jo(vec![
                    ("key", js(instance.key.as_str())),
                    ("generation", number(instance.generation)),
                ]),
            ),
        ]),
    }
}

fn decoded_instance_to_json(instance: &ServiceInstanceSnapshot) -> JsonValue {
    let mut fields = Vec::new();
    if let Some(address) = &instance.instance {
        fields.push((
            "instance",
            jo(vec![
                ("key", js(address.key.as_str())),
                ("generation", number(address.generation)),
            ]),
        ));
    }
    fields.push((
        "members",
        JsonValue::Array(
            instance
                .members
                .iter()
                .map(|member| match member {
                    ServiceMemberSnapshot::Method { name } => {
                        jo(vec![("name", js(name.as_str())), ("kind", js("method"))])
                    }
                    ServiceMemberSnapshot::State {
                        name,
                        sequence,
                        ops,
                    } => jo(vec![
                        ("name", js(name.as_str())),
                        ("kind", js("state")),
                        ("sequence", number(*sequence)),
                        (
                            "ops",
                            JsonValue::Array(ops.iter().map(Op::to_json).collect()),
                        ),
                    ]),
                })
                .collect(),
        ),
    ));
    object(fields)
}

#[test]
fn encodes_and_decodes_lifecycle_updates_over_the_state_codecs() {
    // Unavailable round-trips and resets the codec registry.
    let wire = expect_ok(encoder().encode_update(&ServiceProviderUpdate::Unavailable));
    assert!(matches!(
        wire,
        pi_chord::services::wire::WireServiceProviderUpdate::Unavailable
    ));
    let decoded = expect_ok(decoder().decode_update(&wire));
    assert!(matches!(decoded, ServiceProviderUpdate::Unavailable));

    // Replaced snapshots round-trip method and state members.
    let replaced = ServiceProviderUpdate::Replaced {
        snapshot: ServiceInstanceSnapshot {
            instance: None,
            members: vec![
                ServiceMemberSnapshot::Method {
                    name: "read".to_string(),
                },
                ServiceMemberSnapshot::State {
                    name: "state".to_string(),
                    sequence: 2,
                    ops: vec![Op::Replace(js("next"))],
                },
            ],
        },
    };
    let wire = expect_ok(encoder().encode_update(&replaced));
    let decoded = expect_ok(decoder().decode_update(&wire));
    let ServiceProviderUpdate::Replaced { snapshot } = decoded else {
        panic!("a replaced update decodes");
    };
    assert_eq!(snapshot.members.len(), 2);

    // The codec surfaces carry debug spellings.
    assert!(format!("{:?}", encoder()).starts_with("ServiceStateEncoder"));
    assert!(format!("{:?}", decoder()).starts_with("ServiceStateDecoder"));

    // Two snapshots naming the same state member reject at registration.
    let duplicated = ServiceSubscriptionSnapshot {
        service_id: "pi.models".to_string(),
        mode: ServiceMode::Singleton,
        instances: vec![ServiceInstanceSnapshot {
            instance: None,
            members: vec![
                ServiceMemberSnapshot::State {
                    name: "state".to_string(),
                    sequence: 0,
                    ops: vec![Op::Replace(js("one"))],
                },
                ServiceMemberSnapshot::State {
                    name: "state".to_string(),
                    sequence: 0,
                    ops: vec![Op::Replace(js("one"))],
                },
            ],
        }],
    };
    let error = expect_err(encoder().encode_snapshot(&duplicated));
    assert!(error.to_string().contains("Duplicate service state"));

    // State updates decode against the codec registration a snapshot built.
    let address = ServiceInstanceAddress {
        key: "instance".to_string(),
        generation: 0,
    };
    let subscription = ServiceSubscriptionSnapshot {
        service_id: "pi.models".to_string(),
        mode: ServiceMode::Singleton,
        instances: vec![ServiceInstanceSnapshot {
            instance: Some(address.clone()),
            members: vec![ServiceMemberSnapshot::State {
                name: "state".to_string(),
                sequence: 0,
                ops: vec![Op::Replace(js("one"))],
            }],
        }],
    };
    let mut codec_encoder = encoder();
    let snapshot_wire = expect_ok(codec_encoder.encode_snapshot(&subscription));
    let mut receiving = decoder();
    let _ = expect_ok(receiving.decode_snapshot(&snapshot_wire));
    let state = ServiceProviderUpdate::State {
        instance: Some(address.clone()),
        member: "state".to_string(),
        sequence: 1,
        ops: vec![Op::Replace(js("two"))],
    };
    let state_wire = expect_ok(codec_encoder.encode_update(&state));
    let decoded = expect_ok(receiving.decode_update(&state_wire));
    let ServiceProviderUpdate::State { ops, .. } = decoded else {
        panic!("a state update decodes");
    };
    assert_eq!(ops, vec![Op::Replace(js("two"))]);

    // A state update the subscription never saw rejects on both sides.
    let error = expect_err(codec_encoder.encode_update(&ServiceProviderUpdate::State {
        instance: Some(address.clone()),
        member: "unknown".to_string(),
        sequence: 1,
        ops: Vec::new(),
    }));
    assert!(error.to_string().contains("Unknown service state"));
    let closed = ServiceProviderUpdate::Closed { instance: address };
    let _ = expect_ok(codec_encoder.encode_update(&closed));
}

#[test]
fn spells_the_codec_dictionary_isolation_on_the_method_members() {
    // Method members carry no dictionary state; snapshots with both kinds
    // round-trip through encode_snapshot / decode_snapshot.
    let snapshot = ServiceSubscriptionSnapshot {
        service_id: "pi.models".to_string(),
        mode: ServiceMode::Keyed,
        instances: vec![ServiceInstanceSnapshot {
            instance: Some(ServiceInstanceAddress {
                key: "dialog".to_string(),
                generation: 1,
            }),
            members: vec![ServiceMemberSnapshot::Method {
                name: "send".to_string(),
            }],
        }],
    };
    let wire = expect_ok(encoder().encode_snapshot(&snapshot));
    let decoded = expect_ok(decoder().decode_snapshot(&wire));
    assert_eq!(decoded.service_id, "pi.models");
    assert_eq!(decoded.mode, ServiceMode::Keyed);
    assert_eq!(decoded.instances.len(), 1);
    assert!(matches!(
        decoded.instances[0].members[0],
        ServiceMemberSnapshot::Method { .. }
    ));
}

#[test]
fn parses_every_update_arm_and_guards_control_calls() {
    let address = ServiceInstanceAddress {
        key: "dialog".to_string(),
        generation: 1,
    };
    let instance = ServiceInstanceSnapshot {
        instance: Some(address.clone()),
        members: vec![ServiceMemberSnapshot::Method {
            name: "send".to_string(),
        }],
    };
    let updates = [
        ServiceProviderUpdate::Unavailable,
        ServiceProviderUpdate::Replaced {
            snapshot: ServiceInstanceSnapshot {
                instance: None,
                members: Vec::new(),
            },
        },
        ServiceProviderUpdate::Spawned { instance },
        ServiceProviderUpdate::Closed {
            instance: address.clone(),
        },
    ];
    for update in updates {
        let json = update_to_json(&update);
        let parsed = parse_service_provider_update(&json)
            .unwrap_or_else(|error| panic!("update parses: {error}"));
        assert_eq!(
            update_to_json(&parsed).to_json_string(),
            json.to_json_string()
        );
    }

    // The control call guards reject foreign services, instances, unknown
    // members, and mismatched arguments.
    let call = pi_chord::types::ServiceCall {
        service_id: "other".to_string(),
        member: "catalogue".to_string(),
        args: Vec::new(),
        instance: None,
    };
    assert!(decode_service_control_call(&call).is_none());
    let call = pi_chord::types::ServiceCall {
        service_id: pi_chord::services::wire::SERVICE_CONTROL_ID.to_string(),
        member: "catalogue".to_string(),
        args: Vec::new(),
        instance: Some(address),
    };
    assert!(decode_service_control_call(&call).is_none());
    let call = pi_chord::types::ServiceCall {
        service_id: pi_chord::services::wire::SERVICE_CONTROL_ID.to_string(),
        member: "unknown".to_string(),
        args: Vec::new(),
        instance: None,
    };
    assert!(decode_service_control_call(&call).is_none());
    let call = pi_chord::types::ServiceCall {
        service_id: pi_chord::services::wire::SERVICE_CONTROL_ID.to_string(),
        member: "catalogue".to_string(),
        args: vec![js("spurious")],
        instance: None,
    };
    assert!(decode_service_control_call(&call).is_none());

    // Catalogue rejections: a non-array, an entry with unknown keys, and a
    // duplicate service ID.
    let error = expect_err(parse_service_catalogue(&js("not a list")));
    assert!(error.to_string().contains("Invalid service catalogue"));
    let duplicate = JsonValue::Array(vec![
        jo(vec![
            ("serviceId", js("pi.models")),
            ("mode", js("singleton")),
        ]),
        jo(vec![("serviceId", js("pi.models")), ("mode", js("keyed"))]),
    ]);
    let error = expect_err(parse_service_catalogue(&duplicate));
    assert!(error.to_string().contains("Invalid service catalogue"));
    let bad_entry = JsonValue::Array(vec![jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        ("extra", JsonValue::Bool(true)),
    ])]);
    let error = expect_err(parse_service_catalogue(&bad_entry));
    assert!(error.to_string().contains("Invalid service catalogue"));

    // A service call with non-array args rejects.
    let bad_args = jo(vec![
        ("serviceId", js("pi.models")),
        ("member", js("list")),
        ("args", js("not a list")),
    ]);
    let error = expect_err(parse_service_call(&bad_args));
    assert!(error.to_string().contains("Invalid service call"));
}

#[test]
fn parses_every_update_arm_in_both_forms_and_fences_snapshot_shapes() {
    let address = ServiceInstanceAddress {
        key: "dialog".to_string(),
        generation: 1,
    };

    // Every decoded update kind round-trips, with state and method members
    // riding along.
    let decoded_updates = [
        ServiceProviderUpdate::Unavailable,
        ServiceProviderUpdate::Replaced {
            snapshot: ServiceInstanceSnapshot {
                instance: None,
                members: vec![
                    ServiceMemberSnapshot::Method {
                        name: "send".to_string(),
                    },
                    ServiceMemberSnapshot::State {
                        name: "state".to_string(),
                        sequence: 2,
                        ops: vec![Op::Replace(js("value"))],
                    },
                ],
            },
        },
        ServiceProviderUpdate::Spawned {
            instance: ServiceInstanceSnapshot {
                instance: Some(address.clone()),
                members: vec![ServiceMemberSnapshot::State {
                    name: "request".to_string(),
                    sequence: 3,
                    ops: vec![Op::Replace(js("First?"))],
                }],
            },
        },
        ServiceProviderUpdate::Closed {
            instance: address.clone(),
        },
        ServiceProviderUpdate::State {
            instance: Some(address.clone()),
            member: "request".to_string(),
            sequence: 1,
            ops: vec![Op::Replace(js("Updated?"))],
        },
    ];
    for update in &decoded_updates {
        let json = update_to_json(update);
        let parsed = expect_ok(parse_service_provider_update(&json));
        assert_eq!(
            update_to_json(&parsed).to_json_string(),
            json.to_json_string()
        );
    }

    // The wire form round-trips the same way through its own parser.
    let wire_updates = [
        pi_chord::services::wire::WireServiceProviderUpdate::Unavailable,
        pi_chord::services::wire::WireServiceProviderUpdate::Replaced {
            snapshot: WireServiceInstanceSnapshot {
                instance: None,
                members: vec![
                    WireServiceMemberSnapshot::Method {
                        name: "send".to_string(),
                    },
                    WireServiceMemberSnapshot::State {
                        name: "state".to_string(),
                        sequence: 2,
                        ops: vec![WireOp::Replace(js("value"))],
                    },
                ],
            },
        },
        pi_chord::services::wire::WireServiceProviderUpdate::Spawned {
            instance: WireServiceInstanceSnapshot {
                instance: Some(address.clone()),
                members: vec![WireServiceMemberSnapshot::State {
                    name: "request".to_string(),
                    sequence: 1,
                    ops: vec![WireOp::Replace(js("First?"))],
                }],
            },
        },
        pi_chord::services::wire::WireServiceProviderUpdate::Closed {
            instance: address.clone(),
        },
        pi_chord::services::wire::WireServiceProviderUpdate::State {
            instance: Some(address),
            member: "request".to_string(),
            sequence: 1,
            ops: vec![WireOp::Replace(js("Updated?"))],
        },
    ];
    for update in &wire_updates {
        let json = wire_update_to_json(update);
        let parsed = expect_ok(parse_wire_service_provider_update(&json));
        assert_eq!(
            wire_update_to_json(&parsed).to_json_string(),
            json.to_json_string()
        );
    }

    // Subscription snapshots with method members parse in both forms.
    let decoded_snapshot = ServiceSubscriptionSnapshot {
        service_id: "pi.models".to_string(),
        mode: ServiceMode::Singleton,
        instances: vec![ServiceInstanceSnapshot {
            instance: None,
            members: vec![ServiceMemberSnapshot::Method {
                name: "select".to_string(),
            }],
        }],
    };
    let parsed = parse_snapshot(&snapshot_to_json(&decoded_snapshot));
    assert!(matches!(
        parsed.instances[0].members[0],
        ServiceMemberSnapshot::Method { .. }
    ));
    let wire_snapshot = WireServiceSubscriptionSnapshot {
        service_id: "pi.models".to_string(),
        mode: ServiceMode::Singleton,
        instances: vec![WireServiceInstanceSnapshot {
            instance: None,
            members: vec![WireServiceMemberSnapshot::Method {
                name: "select".to_string(),
            }],
        }],
    };
    let parsed = parse_wire_snapshot(&wire_snapshot_to_json(&wire_snapshot));
    assert!(matches!(
        parsed.instances[0].members[0],
        WireServiceMemberSnapshot::Method { .. }
    ));

    // A service call without an instance parses; the address stays absent.
    let bare_call = jo(vec![
        ("serviceId", js("pi.models")),
        ("member", js("select")),
        ("args", JsonValue::Array(vec![])),
    ]);
    let parsed = expect_ok(parse_service_call(&bare_call));
    assert_eq!(parsed.instance, None);

    // Snapshot and instance shape violations reject.
    let with_unknown_key = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        ("instances", JsonValue::Array(vec![])),
        ("extra", JsonValue::Bool(true)),
    ]);
    let error = expect_err(parse_service_subscription_snapshot(&with_unknown_key));
    assert!(
        error
            .to_string()
            .contains("Invalid service subscription snapshot")
    );

    let instances_not_array = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        ("instances", js("not a list")),
    ]);
    assert!(parse_service_subscription_snapshot(&instances_not_array).is_err());

    let instance_not_record = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        ("instances", JsonValue::Array(vec![js("not an object")])),
    ]);
    assert!(parse_service_subscription_snapshot(&instance_not_record).is_err());

    let members_not_array = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        (
            "instances",
            JsonValue::Array(vec![jo(vec![("members", js("not a list"))])]),
        ),
    ]);
    assert!(parse_service_subscription_snapshot(&instance_not_record).is_err());
    assert!(parse_service_subscription_snapshot(&members_not_array).is_err());

    let missing_required = jo(vec![
        ("serviceId", js("pi.models")),
        ("instances", JsonValue::Array(vec![])),
    ]);
    assert!(parse_service_subscription_snapshot(&missing_required).is_err());

    let bad_state_name = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        (
            "instances",
            JsonValue::Array(vec![jo(vec![(
                "members",
                JsonValue::Array(vec![jo(vec![
                    ("name", number(7)),
                    ("kind", js("state")),
                    ("sequence", number(0)),
                    ("ops", JsonValue::Array(vec![])),
                ])]),
            )])]),
        ),
    ]);
    assert!(parse_service_subscription_snapshot(&bad_state_name).is_err());

    let bad_ops = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        (
            "instances",
            JsonValue::Array(vec![jo(vec![(
                "members",
                JsonValue::Array(vec![jo(vec![
                    ("name", js("state")),
                    ("kind", js("state")),
                    ("sequence", number(0)),
                    ("ops", js("not a list")),
                ])]),
            )])]),
        ),
    ]);
    assert!(parse_service_subscription_snapshot(&bad_ops).is_err());

    let unknown_kind = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("singleton")),
        (
            "instances",
            JsonValue::Array(vec![jo(vec![(
                "members",
                JsonValue::Array(vec![jo(vec![("name", js("x")), ("kind", js("weird"))])]),
            )])]),
        ),
    ]);
    assert!(parse_service_subscription_snapshot(&unknown_kind).is_err());

    let address_extra = jo(vec![
        ("key", js("dialog")),
        ("generation", number(1)),
        ("extra", JsonValue::Bool(true)),
    ]);
    let bad_address = jo(vec![
        ("serviceId", js("pi.models")),
        ("mode", js("keyed")),
        (
            "instances",
            JsonValue::Array(vec![jo(vec![
                ("instance", address_extra),
                ("members", JsonValue::Array(vec![])),
            ])]),
        ),
    ]);
    assert!(parse_service_subscription_snapshot(&bad_address).is_err());

    // Provider update shape violations reject: a non-string member, a
    // non-array op list, and a state update with a malformed address.
    let bad_member = jo(vec![
        ("type", js("state")),
        ("member", number(7)),
        ("sequence", number(1)),
        ("ops", JsonValue::Array(vec![])),
    ]);
    let error = expect_err(parse_service_provider_update(&bad_member));
    assert!(error.to_string().contains("Invalid service state update"));
    let bad_update_ops = jo(vec![
        ("type", js("state")),
        ("member", js("state")),
        ("sequence", number(1)),
        ("ops", js("not a list")),
    ]);
    assert!(parse_service_provider_update(&bad_update_ops).is_err());
    let unknown_update = jo(vec![("type", js("weird"))]);
    let error = expect_err(parse_service_provider_update(&unknown_update));
    assert!(
        error
            .to_string()
            .contains("Invalid service provider update")
    );
    let bad_closed = jo(vec![
        ("type", js("closed")),
        ("instance", js("not an address")),
    ]);
    assert!(parse_service_provider_update(&bad_closed).is_err());
}

#[test]
fn endpoint_keyed_subscriptions_render_addressed_snapshots() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        let dialogs = pi_chord::api::define_service("test.endpoint.keyed").expect("not reserved");
        let provider = RemoteServiceProvider::new(vec![
            pi_chord::services::provider::ServiceProviderDefinition {
                service: dialogs.clone(),
                mode: ServiceMode::Keyed,
            },
        ])
        .unwrap_or_else(|e| panic!("provider: {e}"));
        let _close = provider
            .spawn(&dialogs, "dialog", {
                let mut implementation = ServiceImplementation::new();
                implementation.method(
                    "read",
                    Rc::new(|_args: Vec<JsonValue>, _context: Context| {
                        Box::pin(std::future::ready(Ok(None)))
                    }),
                );
                implementation
            })
            .unwrap_or_else(|e| panic!("spawn: {e}"));
        let endpoint = create_remote_service_endpoint(&provider);
        let publish: pi_chord::types::ServiceUpdatePublisher = Rc::new(|_id, _update, _context| ());
        let snapshot = endpoint
            .invoke(
                create_service_subscribe_call("keyed-1", &dialogs.id, ServiceMode::Keyed),
                publish,
                background_context(),
            )
            .await
            .unwrap_or_else(|e| panic!("subscribe: {e}"))
            .expect("the snapshot renders");
        // The keyed instance renders its address.
        let rendered = snapshot.to_json_string();
        assert!(
            rendered.contains("\"instance\""),
            "keyed snapshot: {rendered}"
        );
        assert!(
            rendered.contains("\"generation\""),
            "keyed snapshot: {rendered}"
        );
        let parsed = parse_snapshot(&snapshot);
        assert_eq!(
            parsed.instances[0].instance,
            Some(ServiceInstanceAddress {
                key: "dialog".to_string(),
                generation: 1,
            })
        );
        let wire_parsed = parse_wire_snapshot(&snapshot);
        assert!(wire_parsed.instances[0].instance.is_some());
        endpoint.dispose();
        provider
            .dispose()
            .unwrap_or_else(|e| panic!("dispose: {e}"));
    });
}
