//! The value-level diff, ported from upstream `src/delta/index.ts`
//! (`diffValue` and friends).
//!
//! Flush calls these on dirty paths: compare the published baseline against
//! the current value and emit the operation tuples that transform one into
//! the other. String operations carry byte offsets, the port's segment
//! contract — an append carries the bytes to add, a front-truncate carries a
//! number of bytes to remove.

use crate::delta::Op;
use crate::delta::ops::{Path, RESERVED_SEGMENTS, Seg, overlap};
use crate::types::{JsonObject, JsonValue};

/// The value at a path, or [`None`] when the path holds nothing.
pub(crate) type MaybeJson<'a> = Option<&'a JsonValue>;

/// Emits a whole-value replacement, or a set when the path names a child.
pub(crate) fn emit_set(path: &[Seg], value: &JsonValue, out: &mut Vec<Op>) {
    if path.is_empty() {
        out.push(Op::Replace(value.clone()));
    } else {
        out.push(Op::Set {
            path: path.to_vec(),
            value: value.clone(),
        });
    }
}

/// Emits a delete; only ever called with the child segment already appended,
/// so the tracked root itself is never addressed.
pub(crate) fn emit_delete(path: &[Seg], out: &mut Vec<Op>) {
    out.push(Op::Delete {
        path: path.to_vec(),
    });
}

/// Longest-suffix diff of one string value, carrying an append when the new
/// value extends the old one and truncate-plus-append around a rolling
/// window.
pub(crate) fn diff_string(before: &str, after: &str, path: &[Seg], scan: usize, out: &mut Vec<Op>) {
    if before == after {
        return;
    }
    if path.is_empty() {
        emit_set(path, &JsonValue::Str(after.to_string()), out);
        return;
    }
    if after.len() > before.len() && &after.as_bytes()[..before.len()] == before.as_bytes() {
        out.push(Op::Append {
            path: path.to_vec(),
            suffix: after[before.len()..].to_string(),
        });
        return;
    }
    let shared = overlap(before, after, scan, 64, 8);
    if shared == 0 {
        out.push(Op::Set {
            path: path.to_vec(),
            value: JsonValue::Str(after.to_string()),
        });
        return;
    }
    let keep = before.chars().count() - shared;
    let cut = byte_index_of(before, keep);
    out.push(Op::Truncate {
        path: path.to_vec(),
        count: cut,
    });
    if after.chars().count() > shared {
        let at = byte_index_of(after, shared);
        out.push(Op::Append {
            path: path.to_vec(),
            suffix: after[at..].to_string(),
        });
    }
}

/// The byte offset of the character at `index`, or the end of the string.
fn byte_index_of(text: &str, index: usize) -> usize {
    text.char_indices()
        .nth(index)
        .map_or(text.len(), |(offset, _)| offset)
}

/// Diffs one value slot: a missing-before set, a missing-after delete, a
/// recursion into matching containers, or a replacement.
pub(crate) fn diff_value(
    before: MaybeJson<'_>,
    after: MaybeJson<'_>,
    path: &[Seg],
    scan: usize,
    out: &mut Vec<Op>,
) {
    match (before, after) {
        (None, Some(after)) => emit_set(path, after, out),
        (Some(_), None) => emit_delete(path, out),
        (None, None) => {}
        (Some(before), Some(after)) => {
            if !before.is_container() && !after.is_container() && before == after {
                return;
            }
            match (before, after) {
                (JsonValue::Str(before), JsonValue::Str(after)) => {
                    diff_string(before, after, path, scan, out);
                }
                (JsonValue::Array(before), JsonValue::Array(after)) => {
                    diff_array(before, after, path, scan, out);
                }
                (JsonValue::Object(before), JsonValue::Object(after)) => {
                    diff_object(before, after, path, scan, out);
                }
                _ => emit_set(path, after, out),
            }
        }
    }
}

fn diff_object(
    before: &JsonObject,
    after: &JsonObject,
    path: &[Seg],
    scan: usize,
    out: &mut Vec<Op>,
) {
    if before
        .keys()
        .chain(after.keys())
        .any(|key| RESERVED_SEGMENTS.contains(&key))
    {
        emit_set(path, &JsonValue::Object(after.clone()), out);
        return;
    }
    for (key, after_value) in after.iter() {
        let child_path = key_path(path, key);
        diff_value(before.get(key), Some(after_value), &child_path, scan, out);
    }
    for key in before.keys() {
        if !after.contains_key(key) {
            emit_delete(&key_path(path, key), out);
        }
    }
}

fn diff_array(
    before: &[JsonValue],
    after: &[JsonValue],
    path: &[Seg],
    scan: usize,
    out: &mut Vec<Op>,
) {
    if before.len() == after.len() {
        for (index, (before_item, after_item)) in before.iter().zip(after.iter()).enumerate() {
            let child_path = index_path(path, index);
            diff_value(Some(before_item), Some(after_item), &child_path, scan, out);
        }
        return;
    }
    let mut prefix = 0;
    while prefix < before.len() && prefix < after.len() && before[prefix] == after[prefix] {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < before.len() - prefix
        && suffix < after.len() - prefix
        && before[before.len() - 1 - suffix] == after[after.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let shorter = before.len().min(after.len());
    if prefix + suffix == shorter {
        let remove = before.len() - prefix - suffix;
        let items = after[prefix..after.len() - suffix].to_vec();
        if prefix == 0 && remove == before.len() {
            emit_set(path, &JsonValue::Array(after.to_vec()), out);
        } else {
            out.push(Op::Splice {
                path: path.to_vec(),
                index: prefix,
                remove,
                items,
            });
        }
        return;
    }
    for (index, (before_item, after_item)) in before.iter().zip(after.iter()).enumerate() {
        let child_path = index_path(path, index);
        diff_value(Some(before_item), Some(after_item), &child_path, scan, out);
    }
    if after.len() > before.len() {
        out.push(Op::Splice {
            path: path.to_vec(),
            index: before.len(),
            remove: 0,
            items: after[before.len()..].to_vec(),
        });
    } else if after.is_empty() {
        emit_set(path, &JsonValue::Array(Vec::new()), out);
    } else {
        out.push(Op::Splice {
            path: path.to_vec(),
            index: after.len(),
            remove: before.len() - after.len(),
            items: Vec::new(),
        });
    }
}

fn key_path(path: &[Seg], key: &str) -> Path {
    let mut child = path.to_vec();
    child.push(Seg::Key(key.to_string()));
    child
}

fn index_path(path: &[Seg], index: usize) -> Path {
    let mut child = path.to_vec();
    child.push(Seg::Index(index));
    child
}
