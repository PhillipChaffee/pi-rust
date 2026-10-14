use super::*;
use crate::delta::WireOp;
use crate::test_support::{applied, idx, ja, jn, jo, js, key, ok};

#[test]
fn does_not_mutate_a_replacement_payload_targeted_by_a_later_operation() {
    let replacement = jo(vec![("nested", jo(vec![("value", jn(1.0))]))]);
    let ops = vec![
        Op::Replace(replacement),
        Op::Set {
            path: vec![key("nested"), key("value")],
            value: jn(2.0),
        },
    ];
    let next = ok(apply_immutable(None, &ops));
    // The borrowed ops still carry the untouched payload.
    assert_eq!(
        ops[0],
        Op::Replace(jo(vec![("nested", jo(vec![("value", jn(1.0))]))]))
    );
    assert_eq!(
        next.as_object()
            .and_then(|object| object.get("nested"))
            .and_then(JsonValue::as_object)
            .and_then(|nested| nested.get("value"))
            .and_then(JsonValue::as_number),
        Some(2.0)
    );
}

// Upstream adopts the `r` payload by reference, so two in-process
// consumers applying one batch alias each other — an ownership rule.
// Rust move semantics carry the same rule at compile time: the batch is
// consumed, and two replicas require the caller to clone at the fan-out
// point. The case ports as: each consumer that clones the batch owns an
// independent replica.
#[test]
fn adopts_an_r_payload_rather_than_copying_it() {
    let mut t = crate::delta::Tracker::new(jo(vec![("n", jn(0.0))]));
    let batch = t.flush();
    // `apply` moved the payloads out of each batch copy; the two replicas
    // are distinct owned values, each owned outright by its consumer.
    let a = applied(None, batch.clone());
    let b = applied(None, batch);
    assert_eq!(a, b);
}

#[test]
fn does_not_alias_the_producer() {
    let mut t = crate::delta::Tracker::new(jo(vec![("x", jn(0.0))]));
    let replica = applied(None, t.flush());
    ok(t.set(&[key("x")], jn(999.0)));
    let _ = t.flush();
    assert_eq!(
        replica
            .as_object()
            .and_then(|object| object.get("x"))
            .and_then(JsonValue::as_number),
        Some(0.0)
    );
}

#[test]
fn adopts_assigned_values_without_cloning_them() {
    let mut t = crate::delta::Tracker::new(jo(Vec::new()));
    let replica = applied(None, t.flush());
    ok(t.set(&[key("item")], jo(vec![("value", jn(0.0))])));
    ok(t.set(&[key("item"), key("value")], jn(1.0)));
    assert_eq!(applied(Some(replica), t.flush()), t.value().clone());
}

#[test]
fn does_not_alias_pushed_values_with_an_in_process_consumer() {
    let mut t = crate::delta::Tracker::new(jo(vec![("xs", ja(Vec::new()))]));
    let replica = applied(None, t.flush());
    ok(t.push(&[key("xs")], vec![jo(vec![("value", jn(1.0))])]));
    let next = applied(Some(replica), t.flush());
    assert_eq!(
        next.as_object()
            .and_then(|object| object.get("xs"))
            .and_then(JsonValue::as_array)
            .and_then(|items| items.first())
            .and_then(JsonValue::as_object)
            .and_then(|first| first.get("value"))
            .and_then(JsonValue::as_number),
        Some(1.0)
    );
    assert_eq!(
        t.value()
            .as_object()
            .and_then(|object| object.get("xs"))
            .and_then(JsonValue::as_array)
            .and_then(|items| items.first())
            .and_then(JsonValue::as_object)
            .and_then(|first| first.get("value"))
            .and_then(JsonValue::as_number),
        Some(1.0)
    );
}

#[test]
fn folds_a_whole_stream_without_a_base_batch_branch() {
    let mut t = crate::delta::Tracker::new(jo(vec![("x", jn(0.0)), ("l", ja(Vec::new()))]));
    let mut enc = crate::delta::encoder();
    let mut dec = crate::delta::decoder();
    let mut replica: Option<JsonValue> = None;
    let mut send =
        |enc: &mut crate::delta::Encoder, dec: &mut crate::delta::Decoder, ops: Vec<Op>| {
            let decoded = ok(dec.decode(&enc.encode(&ops)));
            replica = Some(applied(replica.take(), decoded));
        };
    ok(t.set(&[key("x")], jn(100.0)));
    ok(t.push(&[key("l")], vec![js("xyz")]));
    send(&mut enc, &mut dec, t.flush());
    ok(t.set(&[key("x")], jn(101.0)));
    send(&mut enc, &mut dec, t.flush());
    assert_eq!(replica, Some(t.value().clone()));
}

#[test]
fn writes_an_existing_index() {
    assert_eq!(
        applied(
            Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))])),
            vec![Op::Set {
                path: vec![key("xs"), idx(1)],
                value: jn(9.0),
            }],
        ),
        jo(vec![("xs", ja(vec![jn(1.0), jn(9.0), jn(3.0)]))])
    );
}

#[test]
fn appends_one_past_the_end() {
    assert_eq!(
        applied(
            Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))])),
            vec![Op::Set {
                path: vec![key("xs"), idx(3)],
                value: jn(9.0),
            }],
        ),
        jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0), jn(9.0)]))])
    );
}

#[test]
fn rejects_a_gap() {
    assert!(
        apply(
            Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))])),
            vec![Op::Set {
                path: vec![key("xs"), idx(5)],
                value: jn(9.0),
            }],
        )
        .is_err()
    );
}

#[test]
fn rejects_a_huge_index() {
    assert!(
        apply(
            Some(jo(vec![("xs", ja(Vec::new()))])),
            vec![Op::Set {
                path: vec![key("xs"), idx(4_294_967_290)],
                value: jn(1.0),
            }],
        )
        .is_err()
    );
}

#[test]
fn rejects_string_spelled_array_indices_at_the_consumer() {
    assert!(
        apply(
            Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))])),
            vec![Op::Set {
                path: vec![key("xs"), key("7")],
                value: jn(9.0),
            }],
        )
        .is_err()
    );
    assert!(
        apply(
            Some(jo(vec![("xs", ja(vec![js("a")]))])),
            vec![Op::Append {
                path: vec![key("xs"), key("0")],
                suffix: "b".to_string(),
            }],
        )
        .is_err()
    );
}

#[test]
fn allows_explicit_growth_with_nulls() {
    assert_eq!(
        applied(
            Some(jo(vec![("xs", ja(vec![jn(1.0)]))])),
            vec![Op::Splice {
                path: vec![key("xs")],
                index: 1,
                remove: 0,
                items: vec![JsonValue::Null, JsonValue::Null, jn(9.0)],
            }],
        ),
        jo(vec![(
            "xs",
            ja(vec![jn(1.0), JsonValue::Null, JsonValue::Null, jn(9.0)])
        )])
    );
}

#[test]
fn rejects_deleting_one_past_an_array_s_end() {
    assert!(
        apply(
            Some(jo(vec![("xs", ja(vec![jn(1.0)]))])),
            vec![Op::Delete {
                path: vec![key("xs"), idx(1)],
            }],
        )
        .is_err()
    );
}

#[test]
fn applies_large_splice_payloads_without_spreading_them_at_once() {
    let items = vec![JsonValue::Null; 300_000];
    let length = items.len();
    let result = applied(
        Some(jo(vec![("xs", ja(Vec::new()))])),
        vec![Op::Splice {
            path: vec![key("xs")],
            index: 0,
            remove: 0,
            items,
        }],
    );
    assert_eq!(
        result
            .as_object()
            .and_then(|object| object.get("xs"))
            .and_then(JsonValue::as_array)
            .map(Vec::len),
        Some(length)
    );
}

#[test]
fn rejects_an_unknown_verb_rather_than_skipping_it() {
    assert!(Op::from_json(&ja(vec![js("ZZZ"), ja(vec![js("a")]), jn(9.0),])).is_err());
}

#[test]
fn rejects_non_array_splice_items() {
    assert!(
        Op::from_json(&ja(vec![
            js("p"),
            ja(vec![js("xs")]),
            jn(0.0),
            jn(0.0),
            js("not-an-array"),
        ]))
        .is_err()
    );
}

#[test]
fn rejects_a_string_path() {
    assert!(Op::from_json(&ja(vec![js("s"), js("a"), jn(9.0)])).is_err());
}

#[test]
fn rejects_a_non_tuple_op() {
    assert!(Op::from_json(&jo(vec![("op", js("s"))])).is_err());
    assert!(Op::from_json(&JsonValue::Null).is_err());
}

#[test]
fn rejects_append_to_a_missing_or_non_string_value() {
    let missing = ok(Op::from_json(&ja(vec![
        js("a"),
        ja(vec![js("missing")]),
        js("x"),
    ])));
    assert!(apply(Some(jo(vec![("a", jn(1.0))])), vec![missing]).is_err());
    let non_string = Op::Append {
        path: vec![key("a")],
        suffix: "x".to_string(),
    };
    assert!(apply(Some(jo(vec![("a", jn(1.0))])), vec![non_string]).is_err());
}

#[test]
fn rejects_negative_truncation() {
    assert!(Op::from_json(&ja(vec![js("t"), ja(vec![js("a")]), jn(-1.0)])).is_err());
    assert!(WireOp::from_json(&ja(vec![js("t"), ja(vec![js("a")]), jn(-1.0)])).is_err());
}

#[test]
fn clamps_a_splice_remove_past_the_end_like_array_prototype_splice_does() {
    assert_eq!(
        applied(
            Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0)]))])),
            vec![Op::Splice {
                path: vec![key("xs")],
                index: 0,
                remove: 1_000_000_000,
                items: Vec::new(),
            }],
        ),
        jo(vec![("xs", ja(Vec::new()))])
    );
}

/// Settles a result the case expects to reject.
fn rejection_or_panic(result: Result<JsonValue, DeltaError>) -> DeltaError {
    match result {
        Err(error) => error,
        Ok(_) => unreachable!("the case's mutation rejects"),
    }
}

#[test]
fn rejects_apply_edge_shapes() {
    // Deleting a missing array index rejects.
    let base = jo(vec![("xs", ja(vec![jn(1.0), jn(2.0)]))]);
    let error = rejection_or_panic(apply(
        Some(base.clone()),
        vec![Op::Delete {
            path: vec![key("xs"), idx(5)],
        }],
    ));
    assert!(error.to_string().contains("unsafe path segment"), "{error}");

    // Splicing outside an array rejects.
    let error = rejection_or_panic(apply(
        Some(jo(vec![("nested", jo(vec![]))])),
        vec![Op::Splice {
            path: vec![key("nested"), idx(0)],
            index: 0,
            remove: 0,
            items: Vec::new(),
        }],
    ));
    assert!(error.to_string().contains("unresolvable path"), "{error}");

    // Writing through a key into an array rejects as unsafe.
    let error = rejection_or_panic(apply(
        Some(base),
        vec![Op::Set {
            path: vec![key("xs"), key("value")],
            value: jn(1.0),
        }],
    ));
    assert!(error.to_string().contains("unsafe path segment"));

    // Truncating across a UTF-8 boundary rejects.
    let error = rejection_or_panic(apply(
        Some(jo(vec![("t", js("héllo"))])),
        vec![Op::Truncate {
            path: vec![key("t")],
            count: 2,
        }],
    ));
    assert!(
        error
            .to_string()
            .contains("splits a UTF-8 character boundary")
    );

    // The whole-root replacement is in place.
    let applied_root = ok(apply(None, vec![Op::Replace(jn(1.0))]));
    assert_eq!(applied_root, jn(1.0));
}
