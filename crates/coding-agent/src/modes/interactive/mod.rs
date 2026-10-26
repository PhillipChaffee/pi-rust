//! The interactive mode, upstream's
//! `packages/coding-agent/src/modes/interactive/`.
//!
//! The theme system and its component slice land here first (#187); the
//! message and tool renderers ride #188, the selectors #195/#198/#200/#202,
//! and the app frame with its remaining selectors rides #131.

pub mod components;
pub mod model_search;
pub mod theme;
