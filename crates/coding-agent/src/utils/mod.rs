//! Utility belts shared inside the crate, upstream's `src/utils/` modules.
//!
//! Each module mirrors its upstream file. The ported slice lands with the
//! utils ticket: the abort/text/json/frontmatter/paths/sleep/shell/
//! child-process/clipboard/deprecation/fs-watch/git/html/image/mime/
//! open-browser/pi-user-agent/syntax-highlight/tool-result-images belt.
//! The WASM `photon` loader upstream carries (`photon.ts`) has no module —
//! the `image` crate compiles in — and the bun/Node machinery around it
//! (`image-resize-worker.ts`'s worker entry, the `readFileSync` patching)
//! rides the bun runtime upstream alone. The `windows-self-update`
//! machinery rides its own ticket; the `tools-manager` landed with the
//! tools slice and the `changelog`/`version-check` belt landed with the
//! package-manager slice.
//!
//! The environment seam mirrors the config foundation's: the `_with`
//! variants inject an [`crate::config::EnvLookup`] and the plain getters
//! read the real environment.

pub mod abort;
pub mod ansi;
pub mod changelog;
pub mod child_process;
pub mod clipboard;
pub mod clipboard_command;
pub mod clipboard_image;
pub mod deprecation;
pub mod exif_orientation;
pub mod frontmatter;
pub mod fs_watch;
pub mod git;
pub mod html;
pub mod image_convert;
pub mod image_process;
pub mod image_resize;
pub mod image_resize_core;
pub mod json;
pub mod management_http;
pub mod mime;
pub mod open_browser;
pub mod paths;
pub mod pi_user_agent;
pub mod shell;
pub mod sleep;
pub mod syntax_highlight;
pub mod text;
pub mod tool_result_images;
pub mod tools_manager;
pub mod version_check;

pub mod minimatch;

#[cfg(test)]
pub mod test_env;
