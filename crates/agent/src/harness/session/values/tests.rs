//! The durable value address suite, ported from upstream
//! `test/harness/values.test.ts`'s validation half (the typed sugar and
//! the address constructors exercise through the session-layer child).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use serde_json::json;

use crate::harness::session::values as stored_values;

/// Address validation rejects empty namespaces and NUL bytes in either
/// component, upstream's `validateAddress`.
#[test]
fn address_validation_rejects_the_bad_components() {
    let error = stored_values::validate_address("", "key").expect_err("an empty namespace errors");
    assert_eq!(error.0, "Value namespace must not be empty");
    let error =
        stored_values::validate_address("ns\0", "key").expect_err("a NUL in the namespace errors");
    assert_eq!(error.0, "Value namespace must not contain \\u0000");
    let error = stored_values::validate_address("ns", "ke\0y")
        .expect_err("a NUL in the key errors");
    assert_eq!(error.0, "Value key must not contain \\u0000");
    assert!(stored_values::validate_address("ns", "key").is_ok());
}

/// The list-read option resolver defaults to ascending with the safe
/// integer limit, upstream's `resolveListReadOptions`.
#[test]
fn the_list_read_resolver_defaults_and_clamps() {
    let resolved = stored_values::resolve_list_read_options(None).expect("defaults");
    assert_eq!(resolved.cursor, None);
    assert_eq!(
        resolved.order,
        crate::harness::session::types::EntryScanOrder::Asc
    );
    assert_eq!(resolved.limit, 1_000);
    let error = stored_values::resolve_list_read_options(Some(
        stored_values::ListReadOptions {
            limit: Some(0),
            ..stored_values::ListReadOptions::default()
        },
    ))
    .expect_err("a zero limit errors");
    assert_eq!(error.0, "List read limit must be a positive safe integer");
    let clamped = stored_values::resolve_list_read_options(Some(
        stored_values::ListReadOptions {
            limit: Some(50_000),
            ..stored_values::ListReadOptions::default()
        },
    ))
    .expect("a large limit clamps to the floor");
    assert_eq!(clamped.limit, 10_000);
    let beyond_safe_integer = stored_values::resolve_list_read_options(Some(
        stored_values::ListReadOptions {
            limit: Some(9_007_199_254_740_992),
            ..stored_values::ListReadOptions::default()
        },
    ));
    assert!(beyond_safe_integer.is_err());
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

/// The fixed harness addresses carry upstream's namespaces.
#[test]
fn the_fixed_addresses_carry_upstream_namespaces() {
    assert_eq!(
        stored_values::branch_tip("main").address,
        stored_values::ValueAddress {
            namespace: "pi.branch.tip".to_owned(),
            key: "main".to_owned(),
        }
    );
    assert_eq!(stored_values::lane_config("main").address.namespace, "pi.lane.config");
    assert_eq!(stored_values::lane_state("main").address.namespace, "pi.lane.state");
    assert_eq!(stored_values::operation_result("run").address.namespace, "pi.result");
    assert_eq!(stored_values::operation_meta("run").address.namespace, "pi.op.meta");
    assert_eq!(stored_values::operation_state("run").address.namespace, "pi.op.state");
    assert_eq!(stored_values::session_name().address.namespace, "pi.session.name");
    assert_eq!(stored_values::entry_label("entry").address.namespace, "pi.entry.label");
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