//! The interactive components, upstream's
//! `packages/coding-agent/src/modes/interactive/components/`.
//!
//! This module carries the theme-selector pair (#187), the message/tool
//! renderers (#188), the thinking/trust selectors (#195), the scoped-model
//! picker (#198), the ordinary model picker (#200), the session picker with
//! its search helper (#202), the OAuth provider selector (#204), the
//! settings selector with its submenus (#206), the session tree selector
//! (#208), the user-message/show-images selectors (#210), the config and
//! extension selectors with their countdown timer (#213), and the remaining
//! startup, message, chrome, and dialog components (#131).

pub mod armin;
pub mod assistant_message;
pub mod bash_execution;
pub mod bordered_loader;
pub mod branch_summary_message;
pub mod compaction_summary_message;
pub mod config_selector;
pub mod countdown_timer;
pub mod custom_editor;
pub mod custom_entry;
pub mod custom_message;
pub mod daxnuts;
pub mod diff;
pub mod dynamic_border;
pub mod earendil_announcement;
pub mod extension_editor;
pub mod extension_input;
pub mod extension_selector;
pub mod first_time_setup;
pub mod footer;
pub mod keybinding_hints;
pub mod login_dialog;
pub mod markdown_transform;
pub mod mermaid;
pub mod model_selector;
pub mod oauth_selector;
pub mod scoped_models_selector;
pub mod session_selector;
pub mod session_selector_search;
pub mod settings_selector;
pub mod settings_submenu;
pub mod show_images_selector;
pub mod skill_invocation_message;
pub mod status_indicator;
pub mod theme_selector;
pub mod thinking_selector;
pub mod tool_execution;
pub mod tree_selector;
pub mod trust_selector;
pub mod user_message;
pub mod user_message_selector;
pub mod visual_truncate;
