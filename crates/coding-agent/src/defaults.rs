//! The defaults module, upstream's `src/core/defaults.ts`: the thinking
//! level prompts start at and the levels the prompt surface cycles through.
//!
//! `ThinkingLevel` comes from `pi-agent-core`, upstream's import from
//! `@earendil-works/pi-agent-core`.

use pi_agent_core::types::ThinkingLevel;

/// The thinking level prompts default to, upstream's
/// `DEFAULT_THINKING_LEVEL`.
pub const DEFAULT_THINKING_LEVEL: ThinkingLevel = ThinkingLevel::Medium;

/// The thinking levels the prompt surface cycles through, upstream's
/// `THINKING_LEVEL_OPTIONS`, in upstream order.
pub const THINKING_LEVEL_OPTIONS: [ThinkingLevel; 7] = [
    ThinkingLevel::Off,
    ThinkingLevel::Minimal,
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::Xhigh,
    ThinkingLevel::Max,
];
