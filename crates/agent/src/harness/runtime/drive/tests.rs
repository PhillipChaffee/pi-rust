//! The drive-suite registry, ported 1:1 from the upstream drive test
//! cluster at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: one
//! submodule per upstream test file, plus `common` carrying the shared
//! fixtures.
//!
//! - `common` ← the per-file fixture blocks the drive suites share
//! - `structural` ← `drive-structural.test.ts` ("runtime structural drive")
//! - `generation` ← `drive-generation.test.ts` ("runtime generation
//!   checkpoint" / "runtime assistant generation")
//! - `response` ← the response-settlement boundary tests binding
//!   `drive/response.ts`'s branches the generation suite never reaches
//! - `tools` ← `drive-tools.test.ts` ("durable tool batch")
//! - `reconcile` ← `drive-reconcile.test.ts` ("runtime total drive" /
//!   "runtime cancellation reconciliation")
//! - `retry_deferred` ← `drive-retry-deferred.test.ts` ("runtime assistant
//!   retry wait" / "runtime deferred polling")
//! - `public` ← `drive-public.test.ts` ("runtime public drive")
//! - `terminal` ← `drive-terminal.test.ts` ("runtime terminal cleanup
//!   mechanics" / "runtime operation result records")
//! - `retry` ← `drive-retry.test.ts` ("runtime retry delay")
//! - `boundary` ← the spine's and procedures' uncovered-arm sweep: the
//!   no-progress invariant, the dispatch arms, and the planners' invariants
//!   upstream's suites don't reach, bound through the same procedures
//!
//! The suites exercise the crate-private drive procedures directly, so
//! they live src-side under `#[cfg(test)]` instead of the integration-test
//! binary.

mod boundary;
mod common;
mod generation;
mod public;
mod reconcile;
mod response;
mod retry;
mod retry_deferred;
mod structural;
mod terminal;
mod tools;
