//! The harness runtime foundations, ported from upstream
//! `packages/agent/src/harness/` (map child "pi-agent-core: harness
//! foundations").
//!
//! The modules below carry the foundation slice the later harness children
//! build on: the chord-backed context seam, the error taxonomy, the
//! capability and option types, the session contract surface, the
//! compaction type shells, the telemetry schemas and typed starters, the
//! message helpers, the effect gate, the hook registry, the event bus, and
//! the agent-harness type surface. The nodejs execution environment, the
//! session and compaction implementations, and the runtime constructor
//! ride their own tickets.
pub mod agent_harness;
pub mod compaction;
pub mod config;
pub mod context;
pub mod env;
pub mod events;
pub mod gate;
pub mod hooks;
pub mod messages;
pub mod result;
pub mod session;
pub mod telemetry;
pub mod types;
pub mod utils;
