//! The cell-dimension store and the image-line probe of
//! `packages/tui/src/terminal-image.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Only the pieces the TUI core touches land here ([#43](https://github.com/PhillipChaffee/pi-rust/issues/43)):
//! the process-global cell-dimension store the `CSI 6 ; h ; w t` response
//! feeds, and [`is_image_line`], the prefix probe the compositor and the
//! line-reset pass consult. The rest of the file — capability probes, image
//! rendering, the Kitty/Iterm2 protocol placements — is the image ticket's
//! scope ([#51](https://github.com/PhillipChaffee/pi-rust/issues/51)) and lands there; the TUI core
//! reads image support through an injected probe so this slice does not pull
//! the capability matrix in early.
//!
//! Restatement: upstream stores cell dimensions in a module global read
//! across TUI instances; the workspace forbids the `unsafe` a naked `static
//! mut` would need, so the store sits behind a `Mutex` with the same
//! process-wide visibility and the upstream 9x18 default.

use std::sync::Mutex;
use std::sync::PoisonError;

const KITTY_PREFIX: &str = "\x1b_G";
const ITERM2_PREFIX: &str = "\x1b]1337;File=";

/// Physical cell size in pixels, upstream `CellDimensions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellDimensions {
    /// Cell width in pixels.
    pub width_px: u32,
    /// Cell height in pixels.
    pub height_px: u32,
}

static CELL_DIMENSIONS: Mutex<CellDimensions> = Mutex::new(CellDimensions {
    width_px: 9,
    height_px: 18,
});

/// The process-global cell dimensions, upstream `getCellDimensions`.
///
/// # Panics
///
/// Never: a poisoned lock falls back to the pre-poison value, matching the
/// module-global semantics upstream reads.
#[must_use]
pub fn get_cell_dimensions() -> CellDimensions {
    *CELL_DIMENSIONS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Store the process-global cell dimensions, upstream `setCellDimensions`.
///
/// # Panics
///
/// Never: a poisoned lock falls back to writing through it.
pub fn set_cell_dimensions(dims: CellDimensions) {
    *CELL_DIMENSIONS
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = dims;
}

/// Whether a rendered line carries a terminal-image placement instead of
/// text, upstream `isImageLine`: the Kitty or Iterm2 prefix anywhere in the
/// line.
///
/// The compositor and the line-reset pass leave image lines untouched.
#[must_use]
pub fn is_image_line(line: &str) -> bool {
    line.contains(KITTY_PREFIX) || line.contains(ITERM2_PREFIX)
}
