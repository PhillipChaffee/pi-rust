//! Pure experiment logic for pi behavioral evals, ported from
//! `packages/evals` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream evals is a private harness-and-runner package; only its pure
//! experiment logic is portable, and only that ports here
//! ([ADR 0005](https://github.com/PhillipChaffee/pi-rust/blob/main/docs/adr/0005-evals-re-scope.md)):
//!
//! - [`plan`] derives eval-case identities and the `(case, variant, model,
//!   run)` task plan with alternating variant order per repetition.
//! - [`report`] reads one run's observation out of a Vitest JSON report,
//!   pairs control/treatment observations, computes pass-rate lift and
//!   paired efficiency deltas, and formats the comparison report.
//! - [`report_reader`] is the inlined minimal slice of `@vitest-evals/core`
//!   (0.15.0) that `report::read_task_observation` pins: Vitest JSON report
//!   validation, eval/harness metadata extraction, and session-artifact
//!   persistence. The full framework contract (multi-run workspaces, run
//!   totals, traces/spans validation, glob and directory resolution) stays
//!   out of the slice — it re-enters with the Rust-native eval rig, not
//!   through this crate.
//!
//! The 32 Node-welded test cases (harness, configured-runtime, acme-server)
//! are not ported; their substance re-enters through the eval rig. The 21
//! portable test cases are the acceptance gate and live in the crate's
//! integration tests.
//!
//! Two upstream shapes are restated rather than copied:
//!
//! - Upstream `readTaskObservation` is async because it persists the session
//!   artifact through `node:fs/promises`; the port is sync on `std::fs`, and
//!   the caller supplies the artifact directory explicitly.
//! - Upstream report reading runs both readers concurrently and collapses
//!   any failure into an errored outcome; the port runs the single shared
//!   parse first and collapses any validation failure the same way.

#![forbid(unsafe_code)]

mod error;
pub mod plan;
pub mod report;
pub mod report_reader;

pub use error::EvalError;
