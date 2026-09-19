//! The op vocabularies and their validators, ported from upstream
//! `src/delta/index.ts`.
//!
//! [`Op`] is the decoded form — complete paths, in memory, on disk, and the
//! form [`crate::delta::apply`] takes. [`WireOp`] is what crosses a boundary,
//! adding path interning and arity omission between [`crate::delta::encoder`]
//! and [`crate::delta::decoder`] only.
//!
//! [`Op`] knows nothing about the path dictionary: interning, id references,
//! and omitted paths live in [`WireOp`] and exist only between encode and
//! decode. The wire tuples upstream spells as JSON arrays arrive here as
//! [`JsonValue`] trees, and each vocabulary gets the parser that matches it:
//! validating an [`Op`] against the wire grammar would be laxer than the
//! vocabulary, because a two-element `["s", value]` wire form would pass and
//! apply would then read the value as a path.

use crate::delta::ops::{Path, Seg, UnsafePathError, assert_safe_path};
use crate::types::JsonValue;

/// A path inline, or an id assigned by the encoder on second use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathRef {
    /// The complete path.
    Inline(Path),
    /// A reference to a path defined by a `#` op.
    Id(u32),
}

/// Verb, arity, and payload shape for a **decoded** op: paths inline, no `#`,
/// no short forms.
///
/// `r` is the ONLY op that replaces a whole value; `s`/`d`/`a`/`t` cannot
/// target the root, which [`crate::delta::apply`] enforces on the paths. `p`
/// may target the root, because a tracked value can itself be an array.
///
/// String operations carry byte offsets, the port's segment contract (see the
/// crate README): an append carries the bytes to add, a front-truncate carries
/// a number of bytes to remove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `["r", value]` — replace the whole value.
    Replace(JsonValue),
    /// `["s", path, value]` — set the value at a non-empty path.
    Set {
        /// The path to the value; never empty.
        path: Path,
        /// The replacement value, cloned at emission.
        value: JsonValue,
    },
    /// `["d", path]` — delete the value at the non-empty path.
    Delete {
        /// The path to the entry to remove; never empty.
        path: Path,
    },
    /// `["a", path, suffix]` — append bytes to the string at the path.
    Append {
        /// The path to the string; never empty.
        path: Path,
        /// The bytes to add at the end.
        suffix: String,
    },
    /// `["t", path, count]` — remove `count` bytes from the front of the
    /// string at the path.
    Truncate {
        /// The path to the string.
        path: Path,
        /// The number of bytes to remove; must not split a UTF-8 character
        /// boundary of the value it meets.
        count: usize,
    },
    /// `["p", path, index, remove, items]` — splice the array at the path;
    /// the path may be empty, addressing a tracked root that is itself an
    /// array.
    Splice {
        /// The path to the array; may be empty only for a root array.
        path: Path,
        /// The index the splice starts at.
        index: usize,
        /// The number of elements to remove.
        remove: usize,
        /// The items inserted at `index` after the removal.
        items: Vec<JsonValue>,
    },
}

/// What crosses a boundary. Adds two compressions and nothing else:
///
/// - `#` defines an id, emitted on a path's SECOND use
/// - an id reference replaces an inline path
/// - a shortened tuple reuses the previous op's path; arity disambiguates
///
/// `["r", value]` carries no path, so it encodes to itself — which is why
/// [`is_base`] works unchanged on either vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireOp {
    /// `["r", value]`.
    Replace(JsonValue),
    /// `["s", pathRef, value]`.
    Set {
        /// The path reference.
        path: PathRef,
        /// The replacement value.
        value: JsonValue,
    },
    /// `["s", value]` — the previous op's path.
    SetShort {
        /// The replacement value.
        value: JsonValue,
    },
    /// `["d", pathRef]`.
    Delete {
        /// The path reference.
        path: PathRef,
    },
    /// `["d"]` — the previous op's path.
    DeleteShort,
    /// `["a", pathRef, suffix]`.
    Append {
        /// The path reference.
        path: PathRef,
        /// The bytes to add at the end.
        suffix: String,
    },
    /// `["a", suffix]` — the previous op's path.
    AppendShort {
        /// The bytes to add at the end.
        suffix: String,
    },
    /// `["t", pathRef, count]`.
    Truncate {
        /// The path reference.
        path: PathRef,
        /// The number of bytes to remove from the front.
        count: usize,
    },
    /// `["t", count]` — the previous op's path.
    TruncateShort {
        /// The number of bytes to remove from the front.
        count: usize,
    },
    /// `["p", pathRef, index, remove, items]`.
    Splice {
        /// The path reference.
        path: PathRef,
        /// The index the splice starts at.
        index: usize,
        /// The number of elements to remove.
        remove: usize,
        /// The items inserted after the removal.
        items: Vec<JsonValue>,
    },
    /// `["p", index, remove, items]` — the previous op's path.
    SpliceShort {
        /// The index the splice starts at.
        index: usize,
        /// The number of elements to remove.
        remove: usize,
        /// The items inserted after the removal.
        items: Vec<JsonValue>,
    },
    /// `["#", id, path]` — defines an id for the path, from its second use on.
    Define {
        /// The id being defined.
        id: u32,
        /// The complete path the id stands for.
        path: Path,
    },
}

/// Whether the op replaces a whole value.
#[must_use]
pub const fn is_replace(op: &Op) -> bool {
    matches!(op, Op::Replace(_))
}

/// Whether the wire op replaces a whole value.
#[must_use]
pub const fn is_replace_wire(op: &WireOp) -> bool {
    matches!(op, WireOp::Replace(_))
}

/// Whether a batch begins with a replacement.
///
/// Flush guarantees `r` is at index 0 or absent, so this is exact rather than
/// a heuristic.
#[must_use]
pub const fn is_base(ops: &[Op]) -> bool {
    matches!(ops.first(), Some(Op::Replace(_)))
}

/// Whether a wire batch begins with a replacement.
#[must_use]
pub const fn is_base_wire(ops: &[WireOp]) -> bool {
    matches!(ops.first(), Some(WireOp::Replace(_)))
}

/// A path that resolves to nothing the op can address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathError {
    /// The unresolvable path, or the unresolvable path id.
    pub path: PathRef,
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unresolvable path: {}",
            path_ref_to_json_text(&self.path)
        )
    }
}

impl std::error::Error for PathError {}

/// A malformed op tuple: wrong arity, wrong verb, or a payload of the wrong
/// shape. Upstream spells these `TypeError`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpShapeError {
    /// The validation message, in upstream's words.
    pub message: String,
}

impl std::fmt::Display for OpShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OpShapeError {}

/// Every failure mode of op parsing and application: tuple-shape rejections,
/// unsafe path segments, and unresolvable paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaError {
    /// The op tuple is malformed.
    Shape(OpShapeError),
    /// A path segment would reach the prototype chain.
    UnsafePath(UnsafePathError),
    /// A path or id resolves to nothing the op can address.
    Path(PathError),
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(error) => write!(f, "{error}"),
            Self::UnsafePath(error) => write!(f, "{error}"),
            Self::Path(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DeltaError {}

impl From<UnsafePathError> for DeltaError {
    fn from(error: UnsafePathError) -> Self {
        Self::UnsafePath(error)
    }
}

impl From<PathError> for DeltaError {
    fn from(error: PathError) -> Self {
        Self::Path(error)
    }
}

impl From<OpShapeError> for DeltaError {
    fn from(error: OpShapeError) -> Self {
        Self::Shape(error)
    }
}

fn shape_error(message: impl Into<String>) -> DeltaError {
    DeltaError::Shape(OpShapeError {
        message: message.into(),
    })
}

/// The JSON text `JSON.stringify` produces for a path reference.
#[must_use]
pub fn path_ref_to_json_text(path: &PathRef) -> String {
    match path {
        PathRef::Inline(path) => path_to_json_text(path),
        PathRef::Id(id) => id.to_string(),
    }
}

/// The JSON text `JSON.stringify` produces for a path.
#[must_use]
pub fn path_to_json_text(path: &[Seg]) -> String {
    let mut out = String::from("[");
    for (index, segment) in path.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        match segment {
            Seg::Key(key) => {
                let value = JsonValue::Str(key.clone());
                out.push_str(&value.to_json_string());
            }
            Seg::Index(index) => out.push_str(&index.to_string()),
        }
    }
    out.push(']');
    out
}

/// Parses a path argument: an array whose segments are strings or
/// non-negative integers, none reserved.
fn path_arg(value: &JsonValue, non_empty: bool) -> Result<Path, DeltaError> {
    let Some(items) = value.as_array() else {
        return Err(shape_error("path is not an array"));
    };
    if non_empty && items.is_empty() {
        return Err(shape_error("path is empty"));
    }
    let mut path = Vec::with_capacity(items.len());
    for item in items {
        path.push(seg_arg(item)?);
    }
    assert_safe_path(&path)?;
    Ok(path)
}

/// Parses one path segment: a string key or a non-negative integer index.
fn seg_arg(value: &JsonValue) -> Result<Seg, DeltaError> {
    match value {
        JsonValue::Str(key) => Ok(Seg::Key(key.clone())),
        JsonValue::Number(number) => {
            let raw = number.get();
            if raw.fract() != 0.0 || raw < 0.0 {
                return Err(shape_error("path segment is not a non-negative integer"));
            }
            usize::try_from(f64_to_int(raw))
                .map(Seg::Index)
                .map_err(|_| shape_error("path segment is not a non-negative integer"))
        }
        _ => Err(shape_error("path segment is not a string or number")),
    }
}

/// Parses a wire path reference: an id or an inline path. A string is never a
/// path — unchecked, `"a".slice(0, -1)` resolves to the root and writes there,
/// a path that is not a path, accepted.
fn path_ref_arg(value: &JsonValue) -> Result<PathRef, DeltaError> {
    match value {
        JsonValue::Number(number) => {
            let raw = number.get();
            if raw.fract() != 0.0 || raw < 0.0 {
                return Err(shape_error("bad path id"));
            }
            Ok(PathRef::Id(
                u32::try_from(f64_to_int(raw)).map_err(|_| shape_error("bad path id"))?,
            ))
        }
        JsonValue::Array(_) => Ok(PathRef::Inline(path_arg(value, false)?)),
        _ => Err(shape_error("path is not an array")),
    }
}

/// A non-negative integer payload field spelled on the wire as a float.
fn wire_count(value: &JsonValue, what: &str) -> Result<usize, DeltaError> {
    let JsonValue::Number(number) = value else {
        return Err(shape_error(format!("{what} is not a number")));
    };
    let raw = number.get();
    if raw.fract() != 0.0 || raw < 0.0 {
        return Err(shape_error(format!("{what} is not a non-negative integer")));
    }
    usize::try_from(f64_to_int(raw))
        .map_err(|_| shape_error(format!("{what} is not a non-negative integer")))
}

/// The exact integer a wire float spells, as a [`u128`] that may exceed
/// every target width so the [`usize`] conversion can reject the excess.
///
/// # Panics
/// Never: the caller has checked `fract() == 0.0` and `raw >= 0.0` first, so
/// the cast cannot truncate or lose a sign.
const fn f64_to_int(raw: f64) -> u128 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the caller checked integrality and sign; the wide type makes the range rejection exact"
    )]
    {
        raw as u128
    }
}

impl Op {
    /// The wire tuple the op spells, as JSON. Paths serialize inline.
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let path_value = |path: &[Seg]| {
            JsonValue::Array(
                path.iter()
                    .map(|segment| match segment {
                        Seg::Key(key) => JsonValue::Str(key.clone()),
                        Seg::Index(index) => JsonValue::Number((*index as u64).into()),
                    })
                    .collect(),
            )
        };
        match self {
            Self::Replace(value) => {
                JsonValue::Array(vec![JsonValue::Str("r".into()), value.clone()])
            }
            Self::Set { path, value } => JsonValue::Array(vec![
                JsonValue::Str("s".into()),
                path_value(path),
                value.clone(),
            ]),
            Self::Delete { path } => {
                JsonValue::Array(vec![JsonValue::Str("d".into()), path_value(path)])
            }
            Self::Append { path, suffix } => JsonValue::Array(vec![
                JsonValue::Str("a".into()),
                path_value(path),
                JsonValue::Str(suffix.clone()),
            ]),
            Self::Truncate { path, count } => JsonValue::Array(vec![
                JsonValue::Str("t".into()),
                path_value(path),
                JsonValue::Number((*count as u64).into()),
            ]),
            Self::Splice {
                path,
                index,
                remove,
                items,
            } => JsonValue::Array(vec![
                JsonValue::Str("p".into()),
                path_value(path),
                JsonValue::Number((*index as u64).into()),
                JsonValue::Number((*remove as u64).into()),
                JsonValue::Array(items.clone()),
            ]),
        }
    }

    /// Parses and validates a decoded op tuple.
    ///
    /// Each vocabulary gets the validator that matches it; this one rejects
    /// the wire forms — a two-element `["s", value]`, a bare `["d"]`, an id
    /// reference — rather than reading a value as a path.
    ///
    /// # Errors
    /// [`DeltaError`] when the tuple is malformed or carries an unsafe path.
    pub fn from_json(value: &JsonValue) -> Result<Self, DeltaError> {
        let Some(op) = value.as_array() else {
            return Err(shape_error("op is not a tuple"));
        };
        if op.is_empty() {
            return Err(shape_error("op is not a tuple"));
        }
        let Some(verb) = op.first().and_then(JsonValue::as_verb) else {
            return Err(shape_error("op is not a tuple"));
        };
        match verb {
            "r" => {
                if op.len() != 2 {
                    return Err(shape_error("r arity"));
                }
                Ok(Self::Replace(op[1].clone()))
            }
            "s" => {
                if op.len() != 3 {
                    return Err(shape_error("s arity"));
                }
                Ok(Self::Set {
                    path: path_arg(&op[1], true)?,
                    value: op[2].clone(),
                })
            }
            "d" => {
                if op.len() != 2 {
                    return Err(shape_error("d arity"));
                }
                Ok(Self::Delete {
                    path: path_arg(&op[1], true)?,
                })
            }
            "a" => {
                if op.len() != 3 {
                    return Err(shape_error("a shape"));
                }
                let JsonValue::Str(suffix) = &op[2] else {
                    return Err(shape_error("a shape"));
                };
                Ok(Self::Append {
                    path: path_arg(&op[1], true)?,
                    suffix: suffix.clone(),
                })
            }
            "t" => {
                if op.len() != 3 {
                    return Err(shape_error("t shape"));
                }
                Ok(Self::Truncate {
                    path: path_arg(&op[1], true)?,
                    count: wire_count(&op[2], "t count")?,
                })
            }
            "p" => {
                if op.len() != 5 {
                    return Err(shape_error("p arity"));
                }
                Ok(Self::Splice {
                    path: path_arg(&op[1], false)?,
                    index: wire_count(&op[2], "p index")?,
                    remove: wire_count(&op[3], "p remove")?,
                    items: items_arg(&op[4])?,
                })
            }
            // Silently skipping an unknown verb is how a newer producer's op
            // vanishes.
            unknown => Err(shape_error(format!("unknown op verb: {unknown}"))),
        }
    }
}

impl WireOp {
    /// Parses and validates a wire op tuple: ids and short forms are legal
    /// here.
    ///
    /// # Errors
    /// [`DeltaError`] when the tuple is malformed or carries an unsafe path.
    pub fn from_json(value: &JsonValue) -> Result<Self, DeltaError> {
        let Some(op) = value.as_array() else {
            return Err(shape_error("op is not a tuple"));
        };
        if op.is_empty() {
            return Err(shape_error("op is not a tuple"));
        }
        let Some(verb) = op.first().and_then(JsonValue::as_verb) else {
            return Err(shape_error("op is not a tuple"));
        };
        match verb {
            "r" => {
                if op.len() != 2 {
                    return Err(shape_error("r arity"));
                }
                Ok(Self::Replace(op[1].clone()))
            }
            "s" => match op.len() {
                3 => Ok(Self::Set {
                    path: path_ref_arg(&op[1])?,
                    value: op[2].clone(),
                }),
                2 => Ok(Self::SetShort {
                    value: op[1].clone(),
                }),
                _ => Err(shape_error("s arity")),
            },
            "d" => match op.len() {
                2 => Ok(Self::Delete {
                    path: path_ref_arg(&op[1])?,
                }),
                1 => Ok(Self::DeleteShort),
                _ => Err(shape_error("d arity")),
            },
            "a" => match op.len() {
                3 => {
                    let path = path_ref_arg(&op[1])?;
                    let JsonValue::Str(suffix) = &op[2] else {
                        return Err(shape_error("a value"));
                    };
                    Ok(Self::Append {
                        path,
                        suffix: suffix.clone(),
                    })
                }
                2 => {
                    let JsonValue::Str(suffix) = &op[1] else {
                        return Err(shape_error("a value"));
                    };
                    Ok(Self::AppendShort {
                        suffix: suffix.clone(),
                    })
                }
                _ => Err(shape_error("a arity")),
            },
            "t" => match op.len() {
                3 => {
                    let path = path_ref_arg(&op[1])?;
                    Ok(Self::Truncate {
                        path,
                        count: wire_count(&op[2], "t count")?,
                    })
                }
                2 => Ok(Self::TruncateShort {
                    count: wire_count(&op[1], "t count")?,
                }),
                _ => Err(shape_error("t arity")),
            },
            "p" => match op.len() {
                5 => Ok(Self::Splice {
                    path: path_ref_arg(&op[1])?,
                    index: wire_count(&op[2], "p index")?,
                    remove: wire_count(&op[3], "p remove")?,
                    items: items_arg(&op[4])?,
                }),
                4 => Ok(Self::SpliceShort {
                    index: wire_count(&op[1], "p index")?,
                    remove: wire_count(&op[2], "p remove")?,
                    items: items_arg(&op[3])?,
                }),
                _ => Err(shape_error("p arity")),
            },
            "#" => {
                if op.len() != 3 {
                    return Err(shape_error("# shape"));
                }
                let JsonValue::Number(number) = &op[1] else {
                    return Err(shape_error("# shape"));
                };
                let raw = number.get();
                if raw.fract() != 0.0 || raw < 0.0 {
                    return Err(shape_error("# shape"));
                }
                let id = u32::try_from(f64_to_int(raw)).map_err(|_| shape_error("# shape"))?;
                let path = path_arg(&op[2], false)?;
                Ok(Self::Define { id, path })
            }
            // Silently skipping an unknown verb is how a newer producer's op
            // vanishes.
            unknown => Err(shape_error(format!("unknown op verb: {unknown}"))),
        }
    }

    /// The wire tuple text form.
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let path_ref_value = |path: &PathRef| match path {
            PathRef::Inline(path) => JsonValue::Array(
                path.iter()
                    .map(|segment| match segment {
                        Seg::Key(key) => JsonValue::Str(key.clone()),
                        Seg::Index(index) => JsonValue::Number((*index as u64).into()),
                    })
                    .collect(),
            ),
            PathRef::Id(id) => JsonValue::Number(u64::from(*id).into()),
        };
        match self {
            Self::Replace(value) => {
                JsonValue::Array(vec![JsonValue::Str("r".into()), value.clone()])
            }
            Self::Set { path, value } => JsonValue::Array(vec![
                JsonValue::Str("s".into()),
                path_ref_value(path),
                value.clone(),
            ]),
            Self::SetShort { value } => {
                JsonValue::Array(vec![JsonValue::Str("s".into()), value.clone()])
            }
            Self::Delete { path } => {
                JsonValue::Array(vec![JsonValue::Str("d".into()), path_ref_value(path)])
            }
            Self::DeleteShort => JsonValue::Array(vec![JsonValue::Str("d".into())]),
            Self::Append { path, suffix } => JsonValue::Array(vec![
                JsonValue::Str("a".into()),
                path_ref_value(path),
                JsonValue::Str(suffix.clone()),
            ]),
            Self::AppendShort { suffix } => JsonValue::Array(vec![
                JsonValue::Str("a".into()),
                JsonValue::Str(suffix.clone()),
            ]),
            Self::Truncate { path, count } => JsonValue::Array(vec![
                JsonValue::Str("t".into()),
                path_ref_value(path),
                JsonValue::Number((*count as u64).into()),
            ]),
            Self::TruncateShort { count } => JsonValue::Array(vec![
                JsonValue::Str("t".into()),
                JsonValue::Number((*count as u64).into()),
            ]),
            Self::Splice {
                path,
                index,
                remove,
                items,
            } => JsonValue::Array(vec![
                JsonValue::Str("p".into()),
                path_ref_value(path),
                JsonValue::Number((*index as u64).into()),
                JsonValue::Number((*remove as u64).into()),
                JsonValue::Array(items.clone()),
            ]),
            Self::SpliceShort {
                index,
                remove,
                items,
            } => JsonValue::Array(vec![
                JsonValue::Str("p".into()),
                JsonValue::Number((*index as u64).into()),
                JsonValue::Number((*remove as u64).into()),
                JsonValue::Array(items.clone()),
            ]),
            Self::Define { id, path } => JsonValue::Array(vec![
                JsonValue::Str("#".into()),
                JsonValue::Number(u64::from(*id).into()),
                JsonValue::Array(
                    path.iter()
                        .map(|segment| match segment {
                            Seg::Key(key) => JsonValue::Str(key.clone()),
                            Seg::Index(index) => JsonValue::Number((*index as u64).into()),
                        })
                        .collect(),
                ),
            ]),
        }
    }
}

fn items_arg(value: &JsonValue) -> Result<Vec<JsonValue>, DeltaError> {
    value
        .as_array()
        .cloned()
        .ok_or_else(|| shape_error("p items"))
}

impl JsonValue {
    fn as_verb(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }
}
