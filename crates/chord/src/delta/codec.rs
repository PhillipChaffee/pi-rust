//! The wire codec, ported from upstream `src/delta/index.ts`
//! (`encoder`/`decoder`).
//!
//! Path interning and arity omission live between the tracker and a boundary;
//! [`Op`] and [`apply()`][crate::delta::apply()] knows nothing about them.
//!
//! ONE PAIR PER INDEPENDENT STATE STREAM. Every decoder must observe exactly
//! the batches encoded by its matching encoder, beginning with that state's
//! base. Sharing a transport connection does not make separately hydrated
//! states one stream.

use std::collections::{HashMap, HashSet};

use crate::delta::op::{DeltaError, PathError, PathRef};
use crate::delta::ops::{Path, assert_safe_path};
use crate::delta::{Op, WireOp, path_to_json_text};

/// Interns on SECOND use. A definition costs more than the path it replaces,
/// so interning on first use loses on the many paths written exactly once.
#[derive(Debug, Default)]
pub struct Encoder {
    seen: HashSet<String>,
    ids: HashMap<String, u32>,
    next_id: u32,
}

/// Builds an encoder, upstream's `encoder()` factory.
#[must_use]
pub fn encoder() -> Encoder {
    Encoder::default()
}

impl Encoder {
    /// Compresses one batch of decoded ops into wire ops.
    ///
    /// Arity omission is scoped to a batch: letting it span batches would
    /// make a batch's first op depend on the previous batch's last one, so a
    /// reader that skips or reorders a batch decodes into the wrong path. Ids
    /// are the only cross-batch state, and the dictionary makes those
    /// explicit; a replacement batch is a recovery point that resets the
    /// table.
    #[must_use]
    pub fn encode(&mut self, ops: &[Op]) -> Vec<WireOp> {
        let mut previous: Option<String> = None;
        let mut out = Vec::new();
        for op in ops {
            if let Op::Replace(value) = op {
                out.push(WireOp::Replace(value.clone()));
                // A base batch is a RECOVERY POINT: a reader replays from
                // the last one with a fresh decoder, so everything after
                // it must be self-contained.
                self.seen.clear();
                self.ids.clear();
                self.next_id = 0;
                previous = None;
            } else {
                let Some(path) = op_path(op) else {
                    continue;
                };
                let key = path_to_json_text(path);
                if previous.as_deref() == Some(key.as_str()) {
                    out.push(short_form(op));
                    continue;
                }
                let path_ref = if let Some(id) = self.ids.get(&key) {
                    PathRef::Id(*id)
                } else if self.seen.contains(&key) {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.ids.insert(key.clone(), id);
                    out.push(WireOp::Define {
                        id,
                        path: path.clone(),
                    });
                    PathRef::Id(id)
                } else {
                    self.seen.insert(key.clone());
                    PathRef::Inline(path.clone())
                };
                out.push(full_form(op, path_ref));
                previous = Some(key);
            }
        }
        out
    }
}

/// Restores complete paths from interned ids and short forms.
#[derive(Debug, Default)]
pub struct Decoder {
    paths: HashMap<u32, Path>,
}

/// Builds a decoder, upstream's `decoder()` factory.
#[must_use]
pub fn decoder() -> Decoder {
    Decoder::default()
}

impl Decoder {
    /// Decodes one batch of wire ops back into decoded ops.
    ///
    /// # Errors
    /// [`DeltaError`] when a wire op is malformed, an id is unresolvable, a
    /// short form has no previous path, or a path is unsafe.
    pub fn decode(&mut self, wire: &[WireOp]) -> Result<Vec<Op>, DeltaError> {
        let mut previous: Option<Path> = None;
        let mut out = Vec::new();
        for op in wire {
            match op {
                WireOp::Define { id, path } => {
                    assert_safe_path(path)?;
                    self.paths.insert(*id, path.clone());
                }
                WireOp::Replace(value) => {
                    out.push(Op::Replace(value.clone()));
                    self.paths.clear();
                    previous = None;
                }
                _ => {
                    let short = matches!(
                        op,
                        WireOp::DeleteShort
                            | WireOp::SetShort { .. }
                            | WireOp::AppendShort { .. }
                            | WireOp::TruncateShort { .. }
                            | WireOp::SpliceShort { .. }
                    );
                    let path = if short {
                        previous.clone().ok_or(PathError {
                            path: PathRef::Inline(Vec::new()),
                        })?
                    } else {
                        let resolved = match wire_path_ref(op) {
                            PathRef::Inline(path) => {
                                assert_safe_path(path)?;
                                path.clone()
                            }
                            PathRef::Id(id) => self.paths.get(id).cloned().ok_or(PathError {
                                path: PathRef::Id(*id),
                            })?,
                        };
                        previous = Some(resolved.clone());
                        resolved
                    };
                    if !matches!(op, WireOp::Splice { .. } | WireOp::SpliceShort { .. })
                        && path.is_empty()
                    {
                        return Err(PathError {
                            path: PathRef::Inline(path),
                        }
                        .into());
                    }
                    out.push(decoded_form(op, &path));
                }
            }
        }
        Ok(out)
    }
}

/// The path an op addresses, absent for replacements.
const fn op_path(op: &Op) -> Option<&Path> {
    match op {
        Op::Replace(_) => None,
        Op::Set { path, .. }
        | Op::Delete { path }
        | Op::Append { path, .. }
        | Op::Truncate { path, .. }
        | Op::Splice { path, .. } => Some(path),
    }
}

fn wire_path_ref(op: &WireOp) -> &PathRef {
    match op {
        WireOp::Set { path, .. }
        | WireOp::Delete { path }
        | WireOp::Append { path, .. }
        | WireOp::Truncate { path, .. }
        | WireOp::Splice { path, .. } => path,
        _ => unreachable!("short forms do not carry a reference"),
    }
}

fn short_form(op: &Op) -> WireOp {
    match op {
        Op::Replace(value) => WireOp::Replace(value.clone()),
        Op::Set { value, .. } => WireOp::SetShort {
            value: value.clone(),
        },
        Op::Delete { .. } => WireOp::DeleteShort,
        Op::Append { suffix, .. } => WireOp::AppendShort {
            suffix: suffix.clone(),
        },
        Op::Truncate { count, .. } => WireOp::TruncateShort { count: *count },
        Op::Splice {
            index,
            remove,
            items,
            ..
        } => WireOp::SpliceShort {
            index: *index,
            remove: *remove,
            items: items.clone(),
        },
    }
}

fn full_form(op: &Op, path: PathRef) -> WireOp {
    match op {
        Op::Replace(value) => WireOp::Replace(value.clone()),
        Op::Set { value, .. } => WireOp::Set {
            path,
            value: value.clone(),
        },
        Op::Delete { .. } => WireOp::Delete { path },
        Op::Append { suffix, .. } => WireOp::Append {
            path,
            suffix: suffix.clone(),
        },
        Op::Truncate { count, .. } => WireOp::Truncate {
            path,
            count: *count,
        },
        Op::Splice {
            index,
            remove,
            items,
            ..
        } => WireOp::Splice {
            path,
            index: *index,
            remove: *remove,
            items: items.clone(),
        },
    }
}

fn decoded_form(op: &WireOp, path: &Path) -> Op {
    match op {
        WireOp::Replace(_) => unreachable!("handled by the decode loop"),
        WireOp::Set { value, .. } | WireOp::SetShort { value } => Op::Set {
            path: path.clone(),
            value: value.clone(),
        },
        WireOp::Delete { .. } | WireOp::DeleteShort => Op::Delete { path: path.clone() },
        WireOp::Append { suffix, .. } | WireOp::AppendShort { suffix } => Op::Append {
            path: path.clone(),
            suffix: suffix.clone(),
        },
        WireOp::Truncate { count, .. } | WireOp::TruncateShort { count } => Op::Truncate {
            path: path.clone(),
            count: *count,
        },
        WireOp::Splice {
            index,
            remove,
            items,
            ..
        }
        | WireOp::SpliceShort {
            index,
            remove,
            items,
        } => Op::Splice {
            path: path.clone(),
            index: *index,
            remove: *remove,
            items: items.clone(),
        },
        WireOp::Define { .. } => unreachable!("definitions do not decode to ops"),
    }
}

#[cfg(test)]
mod tests {
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
}
