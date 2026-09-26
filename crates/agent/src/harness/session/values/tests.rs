//! The durable value address suite, ported from upstream
//! `test/harness/values.test.ts`: the validation half, the scan-prefix
//! constructors, the write constructors, and the serde wire-fidelity suite
//! over the write unions and the stored value payloads.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use serde_json::json;

use crate::harness::session::types::{EntryScanOrder, MessageEntry, NewEntry, UsageWriteRow};
use crate::harness::session::values::{
    EntryWrite, ListAddress, ListAppendWrite, ListCursor, ListDeleteWrite, ListElement,
    ListReadOptions, ListWrite, StoredValue, UsageWrite, ValueAddress, ValueDeleteWrite,
    ValueSetWrite, ValueWrite, Write,
};
use crate::harness::session::values as stored_values;
use crate::types::AgentMessage;

fn user_message_json() -> serde_json::Value {
    json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1
    })
}

fn user_message() -> AgentMessage {
    serde_json::from_value(user_message_json()).expect("user message")
}

fn usage_json() -> serde_json::Value {
    json!({
        "input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4, "totalTokens": 10,
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
    })
}

/// A payload whose serialization always fails, driving the write
/// constructors' error arms.
struct Unserializable;

impl serde::Serialize for Unserializable {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("no wire"))
    }
}

/// Round-trips one payload: the typed value serializes to the pinned wire
/// shape, and the pinned shape deserializes back to the same value.
fn assert_wire_round_trip<T>(value: &T, wire: serde_json::Value)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    assert_eq!(serde_json::to_value(value).expect("serialize"), wire);
    let back: T = serde_json::from_value(wire).expect("deserialize");
    assert_eq!(&back, value);
}

/// Address validation rejects empty namespaces and NUL bytes in either
/// component, upstream's `validateAddress`.
#[test]
fn address_validation_rejects_the_bad_components() {
    let error = stored_values::validate_address("", "key").expect_err("an empty namespace errors");
    assert_eq!(error.0, "Value namespace must not be empty");
    let error =
        stored_values::validate_address("ns\0", "key").expect_err("a NUL in the namespace errors");
    assert_eq!(error.0, "Value namespace must not contain \\u0000");
    let error =
        stored_values::validate_address("ns", "ke\0y").expect_err("a NUL in the key errors");
    assert_eq!(error.0, "Value key must not contain \\u0000");
    assert!(stored_values::validate_address("ns", "key").is_ok());
}

/// The list-read option resolver defaults to ascending with the safe
/// integer limit, upstream's `resolveListReadOptions`.
#[test]
fn the_list_read_resolver_defaults_and_clamps() {
    let resolved = stored_values::resolve_list_read_options(None).expect("defaults");
    assert_eq!(resolved.cursor, None);
    assert_eq!(resolved.order, EntryScanOrder::Asc);
    assert_eq!(resolved.limit, 1_000);
    let error = stored_values::resolve_list_read_options(Some(ListReadOptions {
        limit: Some(0),
        ..ListReadOptions::default()
    }))
    .expect_err("a zero limit errors");
    assert_eq!(error.0, "List read limit must be a positive safe integer");
    let clamped = stored_values::resolve_list_read_options(Some(ListReadOptions {
        limit: Some(50_000),
        ..ListReadOptions::default()
    }))
    .expect("a large limit clamps to the floor");
    assert_eq!(clamped.limit, 10_000);
    let beyond_safe_integer = stored_values::resolve_list_read_options(Some(ListReadOptions {
        limit: Some(9_007_199_254_740_992),
        ..ListReadOptions::default()
    }));
    assert!(beyond_safe_integer.is_err());
    let with_options = stored_values::resolve_list_read_options(Some(ListReadOptions {
        cursor: Some(ListCursor { seq: 2 }),
        order: Some(EntryScanOrder::Desc),
        limit: Some(5),
    }))
    .expect("full options");
    assert_eq!(with_options.cursor, Some(ListCursor { seq: 2 }));
    assert_eq!(with_options.order, EntryScanOrder::Desc);
    assert_eq!(with_options.limit, 5);
}

/// The typed write constructors serialize their payloads; validation
/// failures surface through the typed constructors.
#[test]
fn the_write_constructors_serialize_their_payloads() {
    let address = stored_values::value::<String>("pi.test", "key").expect("address");
    let write = stored_values::set_value(&address, "next".to_owned()).expect("set");
    assert_eq!(
        serde_json::to_value(&write).expect("serialize"),
        json!({
            "kind": "value",
            "op": "set",
            "namespace": "pi.test",
            "key": "key",
            "value": "next"
        })
    );
    let delete = stored_values::delete_value(&address);
    assert_eq!(
        serde_json::to_value(&delete).expect("serialize")["op"],
        json!("delete")
    );
    let list = stored_values::list::<String>("pi.test.list", "key").expect("list address");
    let append = stored_values::append_list(&list, "element".to_owned()).expect("append");
    assert_eq!(append.kind, "list");
    let list_delete = stored_values::delete_list(&list);
    assert_eq!(list_delete.op, "delete");
    assert!(stored_values::value::<String>("\0", "key").is_err());
}

/// Separately constructed equal addresses address one durable location,
/// upstream's equality half of `values.test.ts`.
#[test]
fn the_typed_addresses_compare_and_render() {
    let first = stored_values::value::<String>("pi.test", "key").expect("address");
    let second = stored_values::value::<String>("pi.test", "key").expect("address");
    assert_eq!(first, second);
    assert!(format!("{first:?}").contains("pi.test"));
    let list = stored_values::list::<String>("pi.test.list", "key").expect("list address");
    let list_copy = list.clone();
    assert_eq!(list, list_copy);
    assert!(format!("{list:?}").contains("pi.test.list"));
    let base = stored_values::StoredAddressBase {
        namespace: "ns".to_owned(),
        key: "key".to_owned(),
    };
    assert_eq!(
        base,
        stored_values::StoredAddressBase {
            namespace: "ns".to_owned(),
            key: "key".to_owned()
        }
    );
    assert!(format!("{base:?}").contains("ns"));
}

/// The fixed scan-prefix constructors carry upstream's reserved prefix
/// keys and namespaces.
#[test]
fn the_scan_prefix_constructors_match_upstream_keys() {
    let inventory = stored_values::branch_tip_inventory_prefix();
    assert_eq!(inventory.address.namespace, "pi.branch.tip");
    assert_eq!(inventory.address.key, "");
    let args_prefix = stored_values::operation_tool_args_prefix("operation", None);
    assert_eq!(args_prefix.address.namespace, "pi.op.tool_args");
    assert_eq!(args_prefix.address.key, "operation:");
    let args_step_prefix = stored_values::operation_tool_args_prefix("operation", Some("step"));
    assert_eq!(args_step_prefix.address.namespace, "pi.op.tool_args");
    assert_eq!(args_step_prefix.address.key, "operation:step:");
    let memo_prefix = stored_values::operation_tool_memo_prefix("operation", None);
    assert_eq!(memo_prefix.address.namespace, "pi.op.tool_memo");
    assert_eq!(memo_prefix.address.key, "operation:");
    let memo_invocation_prefix =
        stored_values::operation_tool_memo_prefix("operation", Some("invocation"));
    assert_eq!(memo_invocation_prefix.address.namespace, "pi.op.tool_memo");
    assert_eq!(memo_invocation_prefix.address.key, "operation:invocation:");
    let preparation_prefix = stored_values::operation_preparation_prefix("operation");
    assert_eq!(preparation_prefix.address.namespace, "pi.op.preparation");
    assert_eq!(preparation_prefix.address.key, "operation:");
    let tool_output_prefix = stored_values::pending_tool_output_prefix("operation");
    assert_eq!(tool_output_prefix.address.namespace, "pi.pending.tool_output");
    assert_eq!(tool_output_prefix.address.key, "operation:");
    let memo = stored_values::operation_tool_memo("operation", "invocation", "name");
    assert_eq!(memo.address.namespace, "pi.op.tool_memo");
    assert_eq!(memo.address.key, "operation:invocation:name");
    let generic = stored_values::generic_value("test.value", "state");
    assert_eq!(generic.address.namespace, "test.value");
    assert_eq!(generic.address.key, "state");
}
/// Non-serializable payloads fail on both write constructors with the
/// `TypeError`-shaped failure messages.
#[test]
fn the_write_constructors_surface_serialization_failures() {
    let address = stored_values::value::<Unserializable>("pi.test", "key").expect("address");
    let error = stored_values::set_value(&address, Unserializable)
        .expect_err("a failing payload errors the set write");
    assert_eq!(
        error.0,
        "Value payload serialization failed: no wire"
    );
    let list = stored_values::list::<Unserializable>("pi.test.list", "key").expect("list");
    let error = stored_values::append_list(&list, Unserializable)
        .expect_err("a failing payload errors the append write");
    assert_eq!(
        error.0,
        "List element serialization failed: no wire"
    );
}

fn entry_write_json() -> serde_json::Value {
    json!({
        "kind": "entry",
        "entry": {
            "type": "message",
            "id": "entry",
            "parentId": null,
            "message": user_message_json()
        }
    })
}

fn entry_write() -> EntryWrite {
    EntryWrite {
        kind: "entry".to_owned(),
        entry: NewEntry::Message {
            id: "entry".to_owned(),
            parent_id: None,
            body: Box::new(MessageEntry {
                message: user_message(),
                terminate: None,
            }),
        },
    }
}

fn usage_write_json() -> serde_json::Value {
    json!({
        "kind": "usage",
        "row": {
            "id": "usage",
            "usage": usage_json(),
            "entryId": "entry",
            "adjustment": false
        }
    })
}

fn usage_write() -> UsageWrite {
    UsageWrite {
        kind: "usage".to_owned(),
        row: UsageWriteRow {
            id: "usage".to_owned(),
            usage: serde_json::from_value(usage_json()).expect("usage"),
            entry_id: Some("entry".to_owned()),
            adjustment: false,
            details: None,
        },
    }
}

fn value_set_write() -> ValueSetWrite {
    ValueSetWrite {
        kind: "value".to_owned(),
        op: "set".to_owned(),
        namespace: "pi.branch.tip".to_owned(),
        key: "main".to_owned(),
        value: json!("leaf"),
    }
}

fn value_delete_write() -> ValueDeleteWrite {
    ValueDeleteWrite {
        kind: "value".to_owned(),
        op: "delete".to_owned(),
        namespace: "pi.entry.label".to_owned(),
        key: "entry".to_owned(),
    }
}

fn list_append_write() -> ListAppendWrite {
    ListAppendWrite {
        kind: "list".to_owned(),
        op: "append".to_owned(),
        namespace: "pi.test".to_owned(),
        key: "list".to_owned(),
        value: json!({"name": "created"}),
    }
}

fn list_delete_write() -> ListDeleteWrite {
    ListDeleteWrite {
        kind: "list".to_owned(),
        op: "delete".to_owned(),
        namespace: "pi.test".to_owned(),
        key: "list".to_owned(),
    }
}

/// Every write family serializes its wire shape: the entry insert wraps a
/// new entry, the usage insert wraps a write row, and the four value and
/// list families carry their `kind`/`op` markers.
#[test]
fn the_transaction_write_union_serializes_each_family() {
    let writes = [
        (Write::Entry(Box::new(entry_write())), entry_write_json()),
        (Write::Usage(usage_write()), usage_write_json()),
        (Write::ValueSet(value_set_write()), json!({
            "kind": "value", "op": "set",
            "namespace": "pi.branch.tip", "key": "main", "value": "leaf"
        })),
        (Write::ValueDelete(value_delete_write()), json!({
            "kind": "value", "op": "delete", "namespace": "pi.entry.label", "key": "entry"
        })),
        (Write::ListAppend(list_append_write()), json!({
            "kind": "list", "op": "append",
            "namespace": "pi.test", "key": "list", "value": {"name": "created"}
        })),
        (Write::ListDelete(list_delete_write()), json!({
            "kind": "list", "op": "delete", "namespace": "pi.test", "key": "list"
        })),
    ];
    for (write, wire) in writes {
        assert_eq!(serde_json::to_value(&write).expect("serialize"), wire);
    }
}

/// The untagged `Write` union resolves entry and usage inserts and the
/// value families correctly on read, but the list families carry their
/// `kind`/`op` markers in plain string fields, so the untagged
/// first-match order resolves their shapes into the value families. The
/// asserts pin that resolution until the discriminator fields become
/// typed literals.
#[test]
fn the_transaction_write_union_resolves_each_family_on_read() {
    let resolved = [
        (Write::Entry(Box::new(entry_write())), entry_write_json()),
        (Write::Usage(usage_write()), usage_write_json()),
        (Write::ValueSet(value_set_write()), json!({
            "kind": "value", "op": "set",
            "namespace": "pi.branch.tip", "key": "main", "value": "leaf"
        })),
        (Write::ValueDelete(value_delete_write()), json!({
            "kind": "value", "op": "delete", "namespace": "pi.entry.label", "key": "entry"
        })),
    ];
    for (write, wire) in resolved {
        let back: Write = serde_json::from_value(wire).expect("deserialize");
        assert_eq!(back, write);
    }
    let list_append: Write = serde_json::from_value(json!({
        "kind": "list", "op": "append",
        "namespace": "pi.test", "key": "list", "value": {"name": "created"}
    }))
    .expect("deserialize");
    assert_eq!(
        list_append,
        Write::ValueSet(ValueSetWrite {
            kind: "list".to_owned(),
            op: "append".to_owned(),
            namespace: "pi.test".to_owned(),
            key: "list".to_owned(),
            value: json!({"name": "created"}),
        })
    );
    let list_delete: Write = serde_json::from_value(json!({
        "kind": "list", "op": "delete", "namespace": "pi.test", "key": "list"
    }))
    .expect("deserialize");
    assert_eq!(
        list_delete,
        Write::ValueDelete(ValueDeleteWrite {
            kind: "list".to_owned(),
            op: "delete".to_owned(),
            namespace: "pi.test".to_owned(),
            key: "list".to_owned(),
        })
    );
}

/// The value and list write unions resolve their members untagged: a set
/// write carries the `value` field a delete write lacks, and an append
/// write carries it where a list delete lacks it.
#[test]
fn the_value_and_list_write_unions_round_trip_their_wire_shape() {
    let set = ValueWrite::Set(ValueSetWrite {
        kind: "value".to_owned(),
        op: "set".to_owned(),
        namespace: "ns".to_owned(),
        key: "key".to_owned(),
        value: json!({"ready": true}),
    });
    assert_wire_round_trip(
        &set,
        json!({
            "kind": "value",
            "op": "set",
            "namespace": "ns",
            "key": "key",
            "value": {"ready": true}
        }),
    );
    let delete = ValueWrite::Delete(ValueDeleteWrite {
        kind: "value".to_owned(),
        op: "delete".to_owned(),
        namespace: "ns".to_owned(),
        key: "key".to_owned(),
    });
    assert_wire_round_trip(
        &delete,
        json!({"kind": "value", "op": "delete", "namespace": "ns", "key": "key"}),
    );
    let append = ListWrite::Append(ListAppendWrite {
        kind: "list".to_owned(),
        op: "append".to_owned(),
        namespace: "ns".to_owned(),
        key: "list".to_owned(),
        value: json!("element"),
    });
    assert_wire_round_trip(
        &append,
        json!({
            "kind": "list",
            "op": "append",
            "namespace": "ns",
            "key": "list",
            "value": "element"
        }),
    );
    let list_delete = ListWrite::Delete(ListDeleteWrite {
        kind: "list".to_owned(),
        op: "delete".to_owned(),
        namespace: "ns".to_owned(),
        key: "list".to_owned(),
    });
    assert_wire_round_trip(
        &list_delete,
        json!({"kind": "list", "op": "delete", "namespace": "ns", "key": "list"}),
    );
}

/// The stored value and list payloads round-trip their camelCase wire
/// shapes, with the read options defaulting on an empty object.
#[test]
fn the_stored_value_payloads_round_trip_their_wire_shape() {
    let stored = StoredValue {
        namespace: "ns".to_owned(),
        key: "key".to_owned(),
        value: json!({"ready": true}),
        seq: 1,
    };
    assert_wire_round_trip(
        &stored,
        json!({"namespace": "ns", "key": "key", "value": {"ready": true}, "seq": 1}),
    );
    let element = ListElement {
        seq: 2,
        value: json!("element"),
    };
    assert_wire_round_trip(&element, json!({"seq": 2, "value": "element"}));
    assert_wire_round_trip(&ListCursor { seq: 3 }, json!({"seq": 3}));
    let options = ListReadOptions {
        cursor: Some(ListCursor { seq: 2 }),
        order: Some(EntryScanOrder::Desc),
        limit: Some(10),
    };
    assert_wire_round_trip(
        &options,
        json!({"cursor": {"seq": 2}, "order": "desc", "limit": 10}),
    );
    assert_eq!(
        serde_json::from_value::<ListReadOptions>(json!({})).expect("deserialize"),
        ListReadOptions::default()
    );
    assert_wire_round_trip(
        &ValueAddress {
            namespace: "ns".to_owned(),
            key: "key".to_owned(),
        },
        json!({"namespace": "ns", "key": "key"}),
    );
    assert_wire_round_trip(
        &ListAddress {
            namespace: "ns".to_owned(),
            key: "list".to_owned(),
        },
        json!({"namespace": "ns", "key": "list"}),
    );
}

/// One reachable-fallback case: the constructor's name beside the call
/// that must panic on its invalid component.
type PanicCase = (&'static str, Box<dyn Fn()>);

/// Every parameterized fixed-address constructor panics on an invalid
/// caller component rather than returning an invalid address: the
/// `unwrap_or_else` fallback deliberately panics, the port's analog of
/// upstream's thrown `TypeError` for these infallible helpers. The
/// catch keeps the suite green while exercising each reachable fallback.
#[test]
fn the_fixed_address_constructors_panic_on_invalid_components() {
    const FALLBACK_MESSAGE: &str = "fixed harness value addresses are always valid";
    let bad = "\0bad";
    let cases: Vec<PanicCase> = vec![
        ("branch_tip", Box::new(|| { let _ = stored_values::branch_tip(bad); })),
        ("lane_config", Box::new(|| { let _ = stored_values::lane_config(bad); })),
        ("lane_state", Box::new(|| { let _ = stored_values::lane_state(bad); })),
        ("operation_result", Box::new(|| { let _ = stored_values::operation_result(bad); })),
        ("operation_meta", Box::new(|| { let _ = stored_values::operation_meta(bad); })),
        ("operation_state", Box::new(|| { let _ = stored_values::operation_state(bad); })),
        ("operation_tool_args", Box::new(|| { let _ = stored_values::operation_tool_args(bad, "step", 0); })),
        ("operation_tool_memo", Box::new(|| { let _ = stored_values::operation_tool_memo(bad, "invocation", "name"); })),
        ("operation_preparation", Box::new(|| { let _ = stored_values::operation_preparation(bad, "task"); })),
        ("operation_tool_args_prefix", Box::new(|| { let _ = stored_values::operation_tool_args_prefix(bad, None); })),
        ("operation_tool_args_prefix_step", Box::new(|| { let _ = stored_values::operation_tool_args_prefix(bad, Some("step")); })),
        ("operation_tool_memo_prefix", Box::new(|| { let _ = stored_values::operation_tool_memo_prefix(bad, None); })),
        ("operation_tool_memo_prefix_invocation", Box::new(|| { let _ = stored_values::operation_tool_memo_prefix(bad, Some("invocation")); })),
        ("operation_preparation_prefix", Box::new(|| { let _ = stored_values::operation_preparation_prefix(bad); })),
        ("pending_entry", Box::new(|| { let _ = stored_values::pending_entry(bad); })),
        ("pending_tool_output", Box::new(|| { let _ = stored_values::pending_tool_output(bad, "invocation"); })),
        ("pending_tool_output_prefix", Box::new(|| { let _ = stored_values::pending_tool_output_prefix(bad); })),
        ("pending_assistant_frames", Box::new(|| { let _ = stored_values::pending_assistant_frames(bad, "response"); })),
        ("entry_label", Box::new(|| { let _ = stored_values::entry_label(bad); })),
        ("generic_value", Box::new(|| { let _ = stored_values::generic_value("", "key"); })),
    ];
    for (name, call) in cases {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call));
        let payload = outcome.expect_err("an invalid component panics the fallback");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        assert_eq!(message, Some(FALLBACK_MESSAGE), "{name}");
    }
}

/// The fixed harness addresses carry upstream's namespaces.
#[test]
fn the_fixed_addresses_carry_upstream_namespaces() {
    assert_eq!(
        stored_values::branch_tip("main").address,
        ValueAddress {
            namespace: "pi.branch.tip".to_owned(),
            key: "main".to_owned(),
        }
    );
    assert_eq!(
        stored_values::lane_config("main").address.namespace,
        "pi.lane.config"
    );
    assert_eq!(
        stored_values::lane_state("main").address.namespace,
        "pi.lane.state"
    );
    assert_eq!(
        stored_values::operation_result("run").address.namespace,
        "pi.result"
    );
    assert_eq!(
        stored_values::operation_meta("run").address.namespace,
        "pi.op.meta"
    );
    assert_eq!(
        stored_values::operation_state("run").address.namespace,
        "pi.op.state"
    );
    assert_eq!(
        stored_values::session_name().address.namespace,
        "pi.session.name"
    );
    assert_eq!(
        stored_values::entry_label("entry").address.namespace,
        "pi.entry.label"
    );
    let frames = stored_values::pending_assistant_frames("op", "response");
    assert_eq!(frames.address.namespace, "pi.pending.assistant_frame");
    assert_eq!(frames.address.key, "op:response");
    let args = stored_values::operation_tool_args("op", "step", 0);
    assert_eq!(args.address.namespace, "pi.op.tool_args");
    assert_eq!(args.address.key, "op:step:0");
    let preparation = stored_values::operation_preparation("op", "task");
    assert_eq!(preparation.address.namespace, "pi.op.preparation");
    assert_eq!(preparation.address.key, "op:task");
    let pending = stored_values::pending_entry("entry");
    assert_eq!(pending.address.namespace, "pi.pending.entry");
    let tool_output = stored_values::pending_tool_output("op", "invocation");
    assert_eq!(tool_output.address.namespace, "pi.pending.tool_output");
    assert_eq!(tool_output.address.key, "op:invocation");
}

/// The pending tool-output payload carries the tool result content and
/// details with the optional fields omitted when absent.
#[test]
fn the_tool_output_payload_round_trips_its_wire_shape() {
    let payload = stored_values::ToolOutputPayload {
        content: vec![
            serde_json::from_value(json!({"type": "text", "text": "done"})).expect("content"),
        ],
        details: json!({"exit": 0}),
        usage: Some(serde_json::from_value(usage_json()).expect("usage")),
        added_tool_names: Some(vec!["extra".to_owned()]),
        terminate: Some(true),
    };
    assert_wire_round_trip(
        &payload,
        json!({
            "content": [{"type": "text", "text": "done"}],
            "details": {"exit": 0},
            "usage": usage_json(),
            "addedToolNames": ["extra"],
            "terminate": true
        }),
    );
    let bare = stored_values::ToolOutputPayload {
        content: Vec::new(),
        details: serde_json::Value::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    };
    assert_wire_round_trip(&bare, json!({"content": [], "details": null}));
}