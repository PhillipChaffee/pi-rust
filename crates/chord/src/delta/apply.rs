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
