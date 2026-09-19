//! The applier, ported from upstream `src/delta/index.ts` (`apply`,
//! `applyImmutable`, and the path resolution).
//!
//! Takes decoded ops — path ids and omitted paths are a wire concern, so a
//! caller whose ops came from a boundary runs `decode` first.

use crate::delta::Op;
use crate::delta::op::{DeltaError, OpShapeError, PathError, PathRef};
use crate::delta::ops::{Seg, UnsafePathError, assert_safe_path};
use crate::types::{JsonObject, JsonValue};

/// Applies ops to a plain mutable value.
///
/// Takes decoded ops. Returns the value because `r` replaces it outright.
/// The `r` payloads are adopted, not copied — in TypeScript that makes
/// fan-out alias replicas, an ownership rule; here the moved [`Op`] values
/// hand ownership to the caller, and the same rule reads: copy the batch at
/// the fan-out point, or let each consumer decode its own.
///
/// # Errors
/// [`DeltaError`] when an op is malformed, a path is unsafe or unresolvable,
/// or a write cannot be expressed on the value it meets.
pub fn apply(target: Option<JsonValue>, ops: Vec<Op>) -> Result<JsonValue, DeltaError> {
    let mut root = target;
    for op in ops {
        match op {
            Op::Replace(value) => root = Some(value),
            other => apply_op(&mut root, &other)?,
        }
    }
    Ok(root.unwrap_or(JsonValue::Null))
}

/// Applies decoded operations without mutating either input.
///
/// One whole-tree copy per op restates upstream's copy-on-spine, which
/// shares unchanged subtrees — owned data cannot share, so the port copies
/// the tree and mutates the copy.
///
/// # Errors
/// [`DeltaError`] when an op is malformed or cannot be applied.
pub fn apply_immutable(target: Option<JsonValue>, ops: &[Op]) -> Result<JsonValue, DeltaError> {
    let mut root = target;
    for op in ops {
        match op {
            Op::Replace(value) => root = Some(value.clone()),
            other => {
                let mut protected = Some(root.take().unwrap_or(JsonValue::Null).clone());
                apply_op(&mut protected, other)?;
                root = protected;
            }
        }
    }
    Ok(root.unwrap_or(JsonValue::Null))
}

/// Validates and applies one op against the root in place.
fn apply_op(root: &mut Option<JsonValue>, op: &Op) -> Result<(), DeltaError> {
    match op {
        Op::Replace(value) => {
            *root = Some(value.clone());
            Ok(())
        }
        Op::Splice {
            path,
            index,
            remove,
            items,
        } => apply_splice(root, path, *index, *remove, items),
        Op::Set { path, value } => apply_slot_write(root, path, |slot| match slot {
            Slot::ArrayIndex { items, index } => {
                if index == items.len() {
                    items.push(value.clone());
                } else {
                    items[index] = value.clone();
                }
                Ok(())
            }
            Slot::ObjectKey { object, key } => {
                object.set(key_value(&key), value.clone());
                Ok(())
            }
        }),
        Op::Delete { path } => apply_slot_write(root, path, |slot| match slot {
            Slot::ArrayIndex { items, index } => {
                if index >= items.len() {
                    return Err(path_error(path));
                }
                items.remove(index);
                Ok(())
            }
            Slot::ObjectKey { object, key } => {
                object.remove(&key_value(&key));
                Ok(())
            }
        }),
        Op::Append { path, suffix } => apply_string_write(root, path, |text| {
            let mut combined = text.clone();
            combined.push_str(suffix);
            Ok(combined)
        }),
        Op::Truncate { path, count } => apply_string_write(root, path, |text| {
            if *count < text.len() && !text.is_char_boundary(*count) {
                return Err(DeltaError::Shape(OpShapeError {
                    message: "t count splits a UTF-8 character boundary".to_string(),
                }));
            }
            Ok(if *count >= text.len() {
                String::new()
            } else {
                text[*count..].to_string()
            })
        }),
    }
}

/// The `p` write: splice the array at the addressed path.
fn apply_splice(
    root: &mut Option<JsonValue>,
    path: &[Seg],
    index: usize,
    remove: usize,
    items: &[JsonValue],
) -> Result<(), DeltaError> {
    assert_safe_path(path)?;
    let container = resolve_container(root.as_mut(), path)?;
    let Some(array) = container.as_array_mut() else {
        return Err(path_error(path));
    };
    let index = index.min(array.len());
    let remove = remove.min(array.len() - index);
    array.splice(index..index + remove, items.iter().cloned());
    Ok(())
}

/// The slot an `s`/`d` write addresses.
enum Slot<'a> {
    ArrayIndex {
        items: &'a mut Vec<JsonValue>,
        index: usize,
    },
    ObjectKey {
        object: &'a mut JsonObject,
        key: Seg,
    },
}

/// Resolves the op's parent, runs the array-parent key and range checks, and
/// hands the write the addressed slot.
fn apply_slot_write(
    root: &mut Option<JsonValue>,
    path: &[Seg],
    write: impl FnOnce(Slot<'_>) -> Result<(), DeltaError>,
) -> Result<(), DeltaError> {
    assert_safe_path(path)?;
    let (container_path, key) = split_last(path)?;
    let parent = resolve_container(root.as_mut(), container_path)?;
    if let Some(items) = parent.as_array_mut() {
        let Seg::Index(index) = key else {
            return Err(UnsafePathError {
                segment: key.clone(),
            }
            .into());
        };
        if *index > items.len() {
            return Err(UnsafePathError {
                segment: key.clone(),
            }
            .into());
        }
        let index = *index;
        return write(Slot::ArrayIndex { items, index });
    }
    let Some(object) = parent.as_object_mut() else {
        return Err(path_error(path));
    };
    write(Slot::ObjectKey {
        object,
        key: key.clone(),
    })
}

/// Resolves the op's parent and hands the write the addressed string slot,
/// for the `a` and `t` writes that read and rewrite one string value.
fn apply_string_write(
    root: &mut Option<JsonValue>,
    path: &[Seg],
    transform: impl FnOnce(&String) -> Result<String, DeltaError>,
) -> Result<(), DeltaError> {
    assert_safe_path(path)?;
    let (container_path, key) = split_last(path)?;
    let parent = resolve_container(root.as_mut(), container_path)?;
    let current = own_value(parent, key).ok_or_else(|| path_error(path))?;
    let JsonValue::Str(text) = current else {
        return Err(path_error(path));
    };
    let written = transform(text)?;
    write_back_string(parent, key, written);
    Ok(())
}

fn path_error(path: &[Seg]) -> DeltaError {
    DeltaError::Path(PathError {
        path: PathRef::Inline(path.to_vec()),
    })
}

/// Resolves the parent container of the op's last segment, or the addressed
/// container for a splice: every intermediate step must be a container and
/// every segment an own property.
fn resolve_container<'a>(
    root: Option<&'a mut JsonValue>,
    path: &[Seg],
) -> Result<&'a mut JsonValue, DeltaError> {
    let root = root.ok_or_else(|| path_error(path))?;
    let mut cursor = root;
    for segment in path {
        if matches!(cursor, JsonValue::Array(_)) && matches!(segment, Seg::Key(_)) {
            return Err(UnsafePathError {
                segment: segment.clone(),
            }
            .into());
        }
        let child = own_value_mut(cursor, segment).ok_or_else(|| path_error(path))?;
        if !child.is_container() {
            return Err(path_error(path));
        }
        cursor = child;
    }
    Ok(cursor)
}

fn split_last(path: &[Seg]) -> Result<(&[Seg], &Seg), DeltaError> {
    let (last, rest) = path.split_last().ok_or_else(|| {
        DeltaError::Shape(OpShapeError {
            message: "path is empty".to_string(),
        })
    })?;
    Ok((rest, last))
}

/// The string key a segment addresses: keys as sent, array indices as their
/// canonical decimal form for object parents.
fn key_value(segment: &Seg) -> String {
    match segment {
        Seg::Key(key) => key.clone(),
        Seg::Index(index) => index.to_string(),
    }
}

fn own_value<'a>(value: &'a JsonValue, segment: &Seg) -> Option<&'a JsonValue> {
    match (value, segment) {
        (JsonValue::Object(object), Seg::Key(key)) => object.get(key),
        (JsonValue::Object(object), Seg::Index(index)) => object.get(&index.to_string()),
        (JsonValue::Array(items), Seg::Index(index)) => items.get(*index),
        _ => None,
    }
}

fn own_value_mut<'a>(value: &'a mut JsonValue, segment: &Seg) -> Option<&'a mut JsonValue> {
    match (value, segment) {
        (JsonValue::Object(object), Seg::Key(key)) => object.as_value_mut(key),
        (JsonValue::Object(object), Seg::Index(index)) => object.as_value_mut(&index.to_string()),
        (JsonValue::Array(items), Seg::Index(index)) => items.get_mut(*index),
        _ => None,
    }
}

/// Writes a rewritten string back at the addressed slot, the array and
/// object paths of an `a`/`t` write.
fn write_back_string(parent: &mut JsonValue, segment: &Seg, text: String) {
    match (parent, segment) {
        (JsonValue::Array(items), Seg::Index(index)) => {
            if *index < items.len() {
                items[*index] = JsonValue::Str(text);
            }
        }
        (JsonValue::Object(object), Seg::Key(key)) => {
            object.set(key.clone(), JsonValue::Str(text));
        }
        (JsonValue::Object(object), Seg::Index(index)) => {
            object.set(index.to_string(), JsonValue::Str(text));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
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
}
