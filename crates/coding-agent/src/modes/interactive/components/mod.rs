//! The interactive components, upstream's
//! `packages/coding-agent/src/modes/interactive/components/`.
//!
//! This module carries the theme-selector pair (#187), the message/tool
//! renderers (#188), the thinking/trust selectors (#195), the scoped-model
//! picker (#198), the ordinary model picker (#200), the session picker with
//! its search helper (#202), the OAuth provider selector (#204), and the
//! settings selector with its submenus (#206), and the session tree selector
//! (#208); the extension dialogs and the app frame's remaining selectors land
//! with #131.

pub mod assistant_message;
pub mod bash_execution;
pub mod custom_message;
pub mod diff;
pub mod dynamic_border;
pub mod keybinding_hints;
pub mod markdown_transform;
pub mod mermaid;
pub mod model_selector;
pub mod oauth_selector;
pub mod scoped_models_selector;
pub mod session_selector;
pub mod session_selector_search;
pub mod settings_selector;
pub mod settings_submenu;
pub mod theme_selector;
pub mod thinking_selector;
pub mod tool_execution;
pub mod tree_selector;
pub mod trust_selector;
pub mod user_message;
pub mod visual_truncate;
