//! The harness output belt, ported from upstream `src/harness/utils/`.
//!
//! `truncate.ts`, `output-capture.ts`, `adaptive-publisher.ts`, and
//! `shell-output.ts` land with the harness-foundations child — the
//! nodejs execution environment and its suite import them. `usage.ts`
//! rides the built-in tools child (recorded there).

pub mod adaptive_publisher;
pub mod output_capture;
pub mod shell_output;
pub mod truncate;
