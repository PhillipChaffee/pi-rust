//! The harness-container suite, ported 1:1 from the upstream harness
//! runtime test files at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//! - `common` ← the per-file fixture blocks (`harnessOptions`,
//!   `createSession`, `createHarness`, `configured`, the session-close
//!   teardown)
//! - `harness_cases` ← `harness.test.ts` ("runtime Harness lane
//!   management" / "runtime Harness global metadata")
//! - `accept` ← `accept.test.ts` ("runtime atomic run acceptance")
//! - `watch` ← `watch.test.ts` ("runtime lane watch")
//! - `boundary` ← the container's uncovered-arm sweep: the latch, catch,
//!   validation, and slice-stub arms upstream's suites don't reach, bound
//!   through the public surface that reaches them
//!
//! The suites drive the concrete [`Harness`] container and the runtime
//! [`Lane`] it builds, so they live src-side under `#[cfg(test)]`.

#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; a violated expectation panics the test by design"
)]

mod accept;
mod boundary;
mod common;
mod harness_cases;
mod watch;
