use super::*;
use crate::test_support::{ja, jb, jn, jo, js};

#[test]
fn parses_decoded_ops_and_rejects_wire_forms() {
    // Validating Op against the wire grammar would be laxer than the
    // type: a two-element `["s", value]` would pass, and apply would read
    // the value as a path. Each vocabulary gets the validator that
    // matches it.
    for op in [
        ja(vec![js("r"), jo(vec![("a", jn(1.0))])]),
        ja(vec![js("s"), ja(vec![js("a")]), jn(1.0)]),
        ja(vec![js("d"), ja(vec![js("a")])]),
        ja(vec![js("a"), ja(vec![js("a")]), js("x")]),
        ja(vec![js("t"), ja(vec![js("a")]), jn(2.0)]),
        ja(vec![
            js("p"),
            ja(vec![js("a")]),
            jn(0.0),
            jn(0.0),
            ja(Vec::new()),
        ]),
    ] {
        assert!(Op::from_json(&op).is_ok());
    }
    for wire_only in [
        ja(vec![js("s"), jn(1.0)]),
        ja(vec![js("d")]),
        ja(vec![js("a"), js("x")]),
        ja(vec![js("t"), jn(2.0)]),
        ja(vec![js("p"), jn(0.0), jn(0.0), ja(Vec::new())]),
        ja(vec![js("#"), jn(0.0), ja(vec![js("a")])]),
        ja(vec![js("s"), jn(0.0), jn(1.0)]),
    ] {
        assert!(Op::from_json(&wire_only).is_err());
        assert!(WireOp::from_json(&wire_only).is_ok());
    }
}

// Upstream validates only the tuple shape, never the payload's contents;
// owned data restates this: a payload is JsonValue by construction, so a
// recursive inspection cannot exist.
#[test]
fn does_not_recursively_inspect_operation_payloads() {
    assert!(Op::from_json(&ja(vec![js("s"), ja(vec![js("value")]), jo(Vec::new())])).is_ok());
    assert!(WireOp::from_json(&ja(vec![js("r"), jo(Vec::new())])).is_ok());
}

fn rejection(op: &JsonValue) -> String {
    match Op::from_json(op) {
        Ok(_) => unreachable!("expected a rejection for {op:?}"),
        Err(error) => error.to_string(),
    }
}

fn wire_rejection(op: &JsonValue) -> String {
    match WireOp::from_json(op) {
        Ok(_) => unreachable!("expected a rejection for {op:?}"),
        Err(error) => error.to_string(),
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "each rejection branch is one wire-vocabulary pin; splitting would obscure the mapping"
)]
fn rejects_malformed_op_tuples() {
    assert_eq!(rejection(&js("r")), "op is not a tuple");
    assert_eq!(rejection(&ja(Vec::new())), "op is not a tuple");
    assert_eq!(rejection(&ja(vec![jn(1.0)])), "op is not a tuple");
    assert_eq!(rejection(&ja(vec![js("r")])), "r arity");
    assert_eq!(rejection(&ja(vec![js("r"), jn(1.0), jn(2.0)])), "r arity");
    assert_eq!(rejection(&ja(vec![js("s"), ja(vec![js("a")])])), "s arity");
    assert_eq!(
        rejection(&ja(vec![js("s"), ja(vec![js("a")]), jn(1.0), jn(2.0)])),
        "s arity"
    );
    assert_eq!(
        rejection(&ja(vec![js("s"), js("a"), jn(1.0)])),
        "path is not an array"
    );
    assert_eq!(
        rejection(&ja(vec![js("s"), ja(vec![]), jn(1.0)])),
        "path is empty"
    );
    assert_eq!(
        rejection(&ja(vec![js("s"), ja(vec![jn(1.5)]), jn(1.0)])),
        "path segment is not a non-negative integer"
    );
    assert_eq!(
        rejection(&ja(vec![js("s"), ja(vec![jn(-1.0)]), jn(1.0)])),
        "path segment is not a non-negative integer"
    );
    assert_eq!(
        rejection(&ja(vec![js("s"), ja(vec![jb(true)]), jn(1.0)])),
        "path segment is not a string or number"
    );
    assert_eq!(
        rejection(&ja(vec![js("s"), ja(vec![js("__proto__")]), jn(1.0)])),
        "unsafe path segment: __proto__"
    );
    assert_eq!(rejection(&ja(vec![js("d"), ja(vec![])])), "path is empty");
    assert_eq!(
        rejection(&ja(vec![js("d"), ja(vec![js("a")]), jn(1.0)])),
        "d arity"
    );
    assert_eq!(rejection(&ja(vec![js("d")])), "d arity");
    assert_eq!(
        rejection(&ja(vec![js("a"), ja(vec![js("a")]), jn(1.0)])),
        "a shape"
    );
    assert_eq!(rejection(&ja(vec![js("a"), ja(vec![js("a")])])), "a shape");
    assert_eq!(
        rejection(&ja(vec![js("t"), ja(vec![js("a")]), js("x")])),
        "t count is not a number"
    );
    assert_eq!(
        rejection(&ja(vec![js("t"), ja(vec![js("a")]), jn(1.5)])),
        "t count is not a non-negative integer"
    );
    assert_eq!(
        rejection(&ja(vec![js("t"), ja(vec![js("a")]), jn(-1.0)])),
        "t count is not a non-negative integer"
    );
    assert_eq!(
        rejection(&ja(vec![js("p"), ja(vec![js("a")]), jn(0.0), jn(0.0)])),
        "p arity"
    );
    assert_eq!(
        rejection(&ja(vec![
            js("p"),
            ja(vec![js("a")]),
            jn(0.0),
            jn(0.0),
            ja(Vec::new()),
            jn(0.0)
        ])),
        "p arity"
    );
    assert_eq!(
        rejection(&ja(vec![
            js("p"),
            ja(vec![js("a")]),
            jn(0.0),
            jn(0.0),
            js("items")
        ])),
        "p items"
    );
    assert_eq!(
        rejection(&ja(vec![
            js("p"),
            ja(vec![js("a")]),
            js("x"),
            jn(0.0),
            ja(Vec::new())
        ])),
        "p index is not a number"
    );
    assert_eq!(
        rejection(&ja(vec![
            js("p"),
            ja(vec![js("a")]),
            jn(0.0),
            js("x"),
            ja(Vec::new())
        ])),
        "p remove is not a number"
    );
    assert_eq!(rejection(&ja(vec![js("x"), jn(1.0)])), "unknown op verb: x");
}

#[test]
fn rejects_malformed_wire_op_tuples() {
    assert_eq!(
        wire_rejection(&ja(vec![js("s"), ja(vec![js("a")]), jn(1.0), jn(2.0)])),
        "s arity"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("s"), jn(1.5), jn(1.0)])),
        "bad path id"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("s"), jn(4_294_967_296.0), jn(1.0)])),
        "bad path id"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("s"), js("a"), jn(1.0)])),
        "path is not an array"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("d"), ja(vec![js("a")]), jn(1.0)])),
        "d arity"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("a"), ja(vec![js("a")]), jn(1.0)])),
        "a value"
    );
    assert_eq!(wire_rejection(&ja(vec![js("a")])), "a arity");
    assert_eq!(
        wire_rejection(&ja(vec![js("a"), js("x"), js("y")])),
        "path is not an array"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("t"), ja(vec![js("a")]), jn(2.0), jn(3.0)])),
        "t arity"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("t"), js("x")])),
        "t count is not a number"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("p"), jn(0.0), jn(0.0)])),
        "p arity"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("p"), jn(0.0), jn(0.0), js("items")])),
        "p items"
    );
    assert_eq!(
        wire_rejection(&ja(vec![
            js("p"),
            jn(0.0),
            jn(0.0),
            ja(Vec::new()),
            jn(0.0)
        ])),
        "p remove is not a number"
    );
    assert_eq!(
        wire_rejection(&ja(vec![
            js("p"),
            ja(vec![js("a")]),
            jn(0.0),
            jn(0.0),
            ja(Vec::new()),
            jn(0.0)
        ])),
        "p arity"
    );
    assert_eq!(wire_rejection(&ja(vec![js("#")])), "# shape");
    assert_eq!(
        wire_rejection(&ja(vec![js("#"), jn(0.0), ja(vec![js("a")]), jn(1.0)])),
        "# shape"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("#"), js("zero"), ja(vec![js("a")])])),
        "# shape"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("#"), jn(1.5), ja(vec![js("a")])])),
        "# shape"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("#"), jn(-1.0), ja(vec![js("a")])])),
        "# shape"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("#"), jn(4_294_967_296.0), ja(vec![js("a")])])),
        "# shape"
    );
    assert_eq!(
        wire_rejection(&ja(vec![js("#"), jn(0.0), ja(vec![js("__proto__")])])),
        "unsafe path segment: __proto__"
    );
    assert_eq!(wire_rejection(&ja(vec![js("z")])), "unknown op verb: z");
}

#[test]
fn wire_ops_accept_every_short_and_id_form() {
    for op in [
        ja(vec![js("s"), jn(1.0)]),
        ja(vec![js("s"), jn(0.0), jn(1.0)]),
        ja(vec![js("s"), ja(vec![js("a")]), jn(1.0)]),
        ja(vec![js("d")]),
        ja(vec![js("d"), jn(3.0)]),
        ja(vec![js("d"), ja(vec![js("a")])]),
        ja(vec![js("a"), js("x")]),
        ja(vec![js("a"), ja(vec![js("a")]), js("x")]),
        ja(vec![js("t"), jn(2.0)]),
        ja(vec![js("t"), jn(0.0), jn(2.0)]),
        ja(vec![js("t"), ja(vec![js("a")]), jn(2.0)]),
        ja(vec![js("p"), jn(0.0), jn(0.0), ja(Vec::new())]),
        ja(vec![js("p"), jn(0.0), jn(1.0), jn(0.0), ja(Vec::new())]),
        ja(vec![
            js("p"),
            ja(vec![js("a")]),
            jn(1.0),
            jn(0.0),
            ja(Vec::new()),
        ]),
        ja(vec![js("#"), jn(7.0), ja(vec![js("a")])]),
    ] {
        assert!(WireOp::from_json(&op).is_ok(), "{op:?}");
    }
}

#[test]
fn round_trips_every_op_through_its_wire_tuple() {
    let path = vec![Seg::Key("a".to_string()), Seg::Index(1)];
    for op in [
        Op::Replace(jn(1.0)),
        Op::Set {
            path: path.clone(),
            value: jn(1.0),
        },
        Op::Delete { path: path.clone() },
        Op::Append {
            path: path.clone(),
            suffix: "x".to_string(),
        },
        Op::Truncate {
            path: path.clone(),
            count: 2,
        },
        Op::Splice {
            path,
            index: 1,
            remove: 0,
            items: vec![jn(1.0)],
        },
    ] {
        assert!(Op::from_json(&op.to_json()).is_ok());
    }
    let id = PathRef::Id(3);
    for wire in [
        WireOp::Replace(jn(1.0)),
        WireOp::Set {
            path: id.clone(),
            value: jn(1.0),
        },
        WireOp::SetShort { value: jn(1.0) },
        WireOp::Delete { path: id.clone() },
        WireOp::DeleteShort,
        WireOp::Append {
            path: id.clone(),
            suffix: "x".to_string(),
        },
        WireOp::AppendShort {
            suffix: "x".to_string(),
        },
        WireOp::Truncate {
            path: id.clone(),
            count: 2,
        },
        WireOp::TruncateShort { count: 2 },
        WireOp::Splice {
            path: id,
            index: 1,
            remove: 0,
            items: vec![jn(1.0)],
        },
        WireOp::SpliceShort {
            index: 1,
            remove: 0,
            items: vec![jn(1.0)],
        },
        WireOp::Define {
            id: 7,
            path: vec![Seg::Key("a".to_string())],
        },
    ] {
        assert!(WireOp::from_json(&wire.to_json()).is_ok(), "{wire:?}");
    }
}

#[test]
fn classifies_replacements_and_batches() {
    assert!(is_replace(&Op::Replace(jn(1.0))));
    assert!(!is_replace(&Op::Delete { path: vec![] }));
    assert!(is_replace_wire(&WireOp::Replace(jn(1.0))));
    assert!(!is_replace_wire(&WireOp::DeleteShort));
    assert!(is_base(&[
        Op::Replace(jn(1.0)),
        Op::Delete { path: vec![] }
    ]));
    assert!(!is_base(&[Op::Delete { path: vec![] }]));
    assert!(!is_base(&[]));
    assert!(is_base_wire(&[
        WireOp::Replace(jn(1.0)),
        WireOp::DeleteShort
    ]));
    assert!(!is_base_wire(&[WireOp::DeleteShort]));
    assert!(!is_base_wire(&[]));
}

#[test]
fn spells_path_references_and_errors_as_json_text() {
    assert_eq!(
        path_ref_to_json_text(&PathRef::Inline(vec![
            Seg::Key("a".to_string()),
            Seg::Index(2)
        ])),
        r#"["a",2]"#
    );
    assert_eq!(path_ref_to_json_text(&PathRef::Id(9)), "9");
    // DeltaError's three arms spell their inner errors through.
    let unresolvable = DeltaError::from(PathError {
        path: PathRef::Id(4),
    });
    assert_eq!(unresolvable.to_string(), "unresolvable path: 4");
    let unsafe_path = DeltaError::from(UnsafePathError {
        segment: Seg::Key("__proto__".to_string()),
    });
    assert_eq!(unsafe_path.to_string(), "unsafe path segment: __proto__");
    let shape = DeltaError::from(OpShapeError {
        message: "t arity".to_string(),
    });
    assert_eq!(shape.to_string(), "t arity");
}
