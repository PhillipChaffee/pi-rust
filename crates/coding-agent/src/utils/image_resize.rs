//! The off-thread resize entry, upstream's `src/utils/image-resize.ts`.
//!
//! Upstream runs photon in a Node worker thread so WASM decode/resize/
//! encode never blocks the TUI event loop, with an in-process fallback for
//! bun-compiled layouts; the port hands the same work to tokio's blocking
//! pool ([`tokio::task::spawn_blocking`]), which keeps the async executor
//! free the same way and has no worker-entrypoint failure mode to fall
//! back from. The `Bun compiled executables` and `import.meta.url` worker
//! resolution ride the bun runtime upstream alone.

use super::image_resize_core::resize_image_in_process;
pub use super::image_resize_core::{ImageResizeOptions, ResizedImage, format_dimension_note};

/// Resize an image to fit within the specified max dimensions and encoded
/// file size, off the async executor, upstream's `resizeImage`.
///
/// Returns `None` when the bytes do not decode or no candidate gets under
/// the byte cap.
pub async fn resize_image(
    input_bytes: Vec<u8>,
    mime_type: String,
    options: Option<ImageResizeOptions>,
) -> Option<ResizedImage> {
    tokio::task::spawn_blocking(move || {
        resize_image_in_process(&input_bytes, &mime_type, options.as_ref())
    })
    .await
    .ok()?
}
