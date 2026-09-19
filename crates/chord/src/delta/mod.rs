//! The standalone delta primitive, ported from upstream `src/delta/index.ts`
//! and its README: flush-time change tracking over plain JSON, where a
//! change is an operation tuple — replace the whole value, set a property,
//! delete one, append to a string, truncate a string's front, or splice an
//! array. [`ops`] carries the path vocabulary those tuples address values
//! through, its safety rejection, and the overlap scan the string
//! operations are diffed with. Reserved path segments stay rejected
//! wherever a path is data, because an applier performs
//! `parent[key] = value` writes and ops echo untrusted input.

pub mod ops;

pub use ops::{Path, RESERVED_SEGMENTS, Seg, UnsafePathError, assert_safe_path, overlap};
