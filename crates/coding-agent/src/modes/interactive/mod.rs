//! The interactive mode, upstream's
//! `packages/coding-agent/src/modes/interactive/`.
//!
//! The theme system and its component slice landed first (#187); the
//! message and tool renderers ride #188, the selectors #195/#198/#200/#202,
//! the remaining components and CLI startup shell #213/#131, and the app
//! frame rides #131.

pub mod chat_viewport;
pub mod components;
pub mod external_editor;
pub mod interactive_mode;
pub mod model_catalog_refresh;
pub mod model_search;
pub mod session_share;
pub mod theme;
pub mod tui_renderer;
