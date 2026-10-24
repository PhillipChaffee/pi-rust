//! Core runtime surfaces shared by the modes, upstream's `src/core/`
//! modules outside `tools/` (map child "pi-coding-agent: built-in coding
//! tools").

pub mod agent_session;
pub mod bash_executor;
pub mod event_bus;
pub mod exec;
pub mod output_guard;
pub mod slash_commands;
pub mod timings;
