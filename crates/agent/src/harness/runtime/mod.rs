//! The harness runtime foundations, ported from upstream
//! `packages/agent/src/harness/runtime/` (map child "pi-agent-core: runtime
//! foundations - lanes, reducer, and progress").
//!
//! The command vocabulary, the drive pass, the lane implementation, the
//! snapshot reducer, the progress channels, and the transcript helpers.
//! Upstream's staging device (`SliceNotImplemented`) marks the seams owned
//! by later children: compaction and summarized-navigation preparation ride
//! the compaction child, skill and prompt-template formatting ride the
//! skills child, and the drive-pass procedure loop rides the drive child.

#[cfg(test)]
pub(crate) mod test_support;
pub mod lane;
pub mod progress;
pub mod reducer;
pub mod restore;
pub mod transcript;
pub mod types;