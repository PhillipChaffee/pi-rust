use super::*;
use crate::delta::Seg;
use crate::delta::op::is_base_wire;
use crate::test_support::{applied, ja, jn, jo, js, key, ok, some};
use crate::types::JsonValue;

// One pair per stream. The table spans a whole subscription or file, so a
// second consumer joining later needs its own encoder.
fn round_trip(batches: &[Vec<Op>]) -> Vec<Vec<Op>> {
    let mut enc = encoder();
    let mut dec = decoder();
    batches
        .iter()
        .map(|ops| ok(dec.decode(&enc.encode(ops))))
        .collect()
}

#[test]
fn round_trips_a_stream_exactly() {
    let mut t = crate::delta::Tracker::new(jo(vec![
        ("a", jo(vec![("deep", js(""))])),
        ("b", jo(vec![("deep", js(""))])),
    ]));
    let _ = t.flush();
    let mut batches: Vec<Vec<Op>> = Vec::new();
    for index in 0..6 {
        let a = t
            .get(&[key("a"), key("deep")])
            .and_then(JsonValue::as_str)
            .map_or(String::new(), str::to_owned);
        let b = t
            .get(&[key("b"), key("deep")])
            .and_then(JsonValue::as_str)
            .map_or(String::new(), str::to_owned);
        ok(t.set(&[key("a"), key("deep")], js(&format!("{a}x{index}"))));
        ok(t.set(&[key("b"), key("deep")], js(&format!("{b}y{index}"))));
        batches.push(t.flush());
    }
    assert_eq!(round_trip(&batches), batches);
}

#[test]
fn interns_on_second_use_not_first() {
    let mut enc = encoder();
    let path = vec![key("a"), key("deep")];
    let first = enc.encode(&[Op::Append {
        path: path.clone(),
        suffix: "1".to_string(),
    }]);
    let second = enc.encode(&[Op::Append {
        path: path.clone(),
        suffix: "2".to_string(),
    }]);
    assert_eq!(
        first,
        vec![WireOp::Append {
            path: PathRef::Inline(path.clone()),
            suffix: "1".to_string(),
        }]
    );
    assert_eq!(
        second,
        vec![
            WireOp::Define { id: 0, path },
            WireOp::Append {
                path: PathRef::Id(0),
                suffix: "2".to_string(),
            },
        ]
    );
}

#[test]
fn omits_the_path_when_it_repeats() {
    let mut enc = encoder();
    let path = vec![key("a")];
    assert_eq!(
        enc.encode(&[
            Op::Set {
                path: path.clone(),
                value: jn(1.0),
            },
            Op::Set {
                path: path.clone(),
                value: jn(2.0),
            },
        ]),
        vec![
            WireOp::Set {
                path: PathRef::Inline(path),
                value: jn(1.0),
            },
            WireOp::SetShort { value: jn(2.0) },
        ]
    );
}

#[test]
fn does_not_collide_paths_containing_null_characters() {
    let first = vec![Seg::Key("a\u{0}b".to_string())];
    let second = vec![Seg::Key("a".to_string()), Seg::Key("b".to_string())];
    let ops = vec![
        Op::Set {
            path: first,
            value: jn(1.0),
        },
        Op::Set {
            path: second,
            value: jn(2.0),
        },
    ];
    let mut enc = encoder();
    let mut dec = decoder();
    assert_eq!(ok(dec.decode(&enc.encode(&ops))), ops);
}

#[test]
fn rejects_a_short_form_without_a_previous_path() {
    assert!(
        decoder()
            .decode(&[WireOp::AppendShort {
                suffix: "x".to_string()
            }])
            .is_err()
    );
}

#[test]
fn clears_decoder_ids_on_a_base_batch() {
    let mut dec = decoder();
    ok(dec.decode(&[
        WireOp::Define {
            id: 0,
            path: vec![key("a")],
        },
        WireOp::Append {
            path: PathRef::Id(0),
            suffix: "1".to_string(),
        },
    ]));
    ok(dec.decode(&[WireOp::Replace(jo(vec![("a", js(""))]))]));
    assert!(
        dec.decode(&[WireOp::Append {
            path: PathRef::Id(0),
            suffix: "2".to_string(),
        }])
        .is_err()
    );
}

#[test]
fn resets_the_table_on_a_base_batch_so_recovery_is_self_contained() {
    let mut enc = encoder();
    let path = vec![key("a"), key("deep")];
    let _ = enc.encode(&[Op::Append {
        path: path.clone(),
        suffix: "1".to_string(),
    }]);
    let _ = enc.encode(&[Op::Append {
        path: path.clone(),
        suffix: "2".to_string(),
    }]);
    let base = enc.encode(&[Op::Replace(jo(vec![("a", jo(vec![("deep", js("x"))]))]))]);
    let after = enc.encode(&[Op::Append {
        path: path.clone(),
        suffix: "3".to_string(),
    }]);
    assert_eq!(
        base,
        vec![WireOp::Replace(jo(vec![(
            "a",
            jo(vec![("deep", js("x"))])
        )]))]
    );
    assert_eq!(
        after,
        vec![WireOp::Append {
            path: PathRef::Inline(path.clone()),
            suffix: "3".to_string(),
        }]
    );

    let mut dec = decoder();
    ok(dec.decode(&base));
    assert_eq!(
        ok(dec.decode(&after)),
        vec![Op::Append {
            path,
            suffix: "3".to_string(),
        }]
    );
}

#[test]
fn survives_recovery_from_the_last_base_batch() {
    let mut enc = encoder();
    let mut t = crate::delta::Tracker::new(jo(vec![
        ("a", jo(vec![("deep", js(""))])),
        ("b", jo(vec![("deep", js(""))])),
    ]));
    let _ = t.flush();
    let mut wire: Vec<Vec<WireOp>> = Vec::new();
    for index in 0..8 {
        let a = t
            .get(&[key("a"), key("deep")])
            .and_then(JsonValue::as_str)
            .map_or(String::new(), str::to_owned);
        let b = t
            .get(&[key("b"), key("deep")])
            .and_then(JsonValue::as_str)
            .map_or(String::new(), str::to_owned);
        ok(t.set(&[key("a"), key("deep")], js(&format!("{a}x{index}"))));
        ok(t.set(&[key("b"), key("deep")], js(&format!("{b}y{index}"))));
        if index == 5 {
            t.rebase();
        }
        wire.push(enc.encode(&t.flush()));
    }
    let last_base = some(wire.iter().rposition(|batch| is_base_wire(batch)));
    let mut dec = decoder();
    let mut replica: Option<JsonValue> = None;
    for batch in &wire[last_base..] {
        let ops = ok(dec.decode(batch));
        replica = Some(applied(replica, ops));
    }
    assert_eq!(replica, Some(t.value().clone()));
}

#[test]
fn round_trips_random_streams() {
    use crate::test_support::seeded_rng;
    use rand::Rng;

    let mut rng = seeded_rng(0xC0DE_F00D);
    for _ in 0..300 {
        let mut t = crate::delta::Tracker::new(jo(vec![
            ("a", jo(vec![("p", js("")), ("q", js(""))])),
            ("b", ja(Vec::new())),
            ("c", jn(0.0)),
        ]));
        let _ = t.flush();
        let mut batches: Vec<Vec<Op>> = Vec::new();
        for index in 0..8 {
            let roll = rng.random::<f64>();
            if roll < 0.3 {
                let text = t
                    .get(&[key("a"), key("p")])
                    .and_then(JsonValue::as_str)
                    .map_or(String::new(), str::to_owned);
                ok(t.set(&[key("a"), key("p")], js(&(text + "x"))));
            } else if roll < 0.5 {
                let text = t
                    .get(&[key("a"), key("q")])
                    .and_then(JsonValue::as_str)
                    .map_or(String::new(), str::to_owned);
                ok(t.set(&[key("a"), key("q")], js(&(text + "y"))));
            } else if roll < 0.65 {
                ok(t.push(&[key("b")], vec![jn(f64::from(index))]));
            } else if roll < 0.8 {
                ok(t.set(&[key("c")], jn(f64::from(index))));
            } else if roll < 0.9 {
                ok(t.delete(&[key("c")]));
            } else {
                t.rebase();
            }
            let ops = t.flush();
            if !ops.is_empty() {
                batches.push(ops);
            }
        }
        assert_eq!(round_trip(&batches), batches);
    }
}
