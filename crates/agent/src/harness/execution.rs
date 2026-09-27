//! The execution layer, ported from upstream `src/harness/execution/` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Two modules carry the drive-pass execution primitives: the tool-call
//! pipeline ([`tools`]) and the assistant stream runner ([`assistant`]).
//! The effect gate itself homes in [`super::gate`] — `hooks.ts` consumes
//! the procedure-facing view, so the harness-foundations child carried the
//! whole module and these modules consume it.

pub mod assistant;
pub mod tools;
