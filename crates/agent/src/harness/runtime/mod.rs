//! The harness runtime foundations, ported from upstream
//! `packages/agent/src/harness/runtime/` (map child "pi-agent-core: runtime
//! foundations - lanes, reducer, and progress").
//!
//! The command vocabulary, the drive pass, the lane implementation, the
//! snapshot reducer, the progress channels, and the transcript helpers.
//! The drive-pass procedure loop dispatches from [`drive`]: an installed
//! pass spawns it detached, and its structural serializers serve the
//! lane's compaction and summarized-navigation admissions. Skill and
//! prompt-template formatting serves the admissions through the skills
//! child.

#[cfg(test)]
mod boundary;
mod clock;
pub mod drive;
pub mod harness;
pub mod lane;
pub mod progress;
pub mod reducer;
pub mod restore;
#[cfg(test)]
pub(crate) mod test_support;
pub mod transcript;
pub mod types;

/// The attach entry point, upstream's `runtime/index.ts`
/// (`export { createAgentHarness } from "./harness.ts"`).
pub use harness::create_agent_harness;
