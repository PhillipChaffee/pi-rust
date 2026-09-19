//! The standalone delta primitive, ported from upstream
//! `src/delta/index.ts` and its README.
//!
//! Flush-time change tracking over plain JSON, where a change is an
//! operation tuple — replace the whole value, set a property, delete one,
//! append to a string, truncate a string's front, or splice an array.
//! [`ops`] carries the path vocabulary those tuples address values through,
//! its safety rejection, and the overlap scan the string operations are
//! diffed with. Reserved path segments stay rejected wherever a path is
//! data, because an applier performs `parent[key] = value` writes and ops
//! echo untrusted input.

use crate::types::JsonValue;

pub mod apply;
pub mod codec;
pub mod diff;
pub mod op;
pub mod ops;
pub mod tracker;

pub use apply::{apply, apply_immutable};
pub use codec::{Decoder, Encoder, decoder, encoder};
pub use op::{
    DeltaError, Op, OpShapeError, PathError, PathRef, WireOp, is_base, is_base_wire, is_replace,
    is_replace_wire, path_ref_to_json_text, path_to_json_text,
};
pub use ops::{Path, RESERVED_SEGMENTS, Seg, UnsafePathError, assert_safe_path, overlap};
pub use tracker::{Tracker, TrackerError, TrackerOptions};

/// Tracks `root`; the first `flush` is a base batch.
#[must_use]
pub fn track(root: JsonValue) -> Tracker {
    Tracker::new(root)
}

/// Tracks `root` with a bounded overlap scan.
#[must_use]
pub fn track_with_options(root: JsonValue, options: TrackerOptions) -> Tracker {
    Tracker::with_options(root, options)
}
