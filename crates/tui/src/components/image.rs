//! Image component, ported from `packages/tui/src/components/image.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Renders an inline image through the terminal's Kitty or Iterm2 protocol
//! ([#51](https://github.com/PhillipChaffee/pi-rust/issues/51)), with a
//! themed one-line text fallback when the terminal cannot. The Kitty
//! placement carries `C=1` (no terminal-side cursor movement) and an
//! auto-allocated image id, and answers `rows` lines so the TUI accounts for
//! the image height; the Iterm2 placement rides a cursor-up prefix so the
//! TUI's cursor accounting stays inside the scroll area.

use std::cell::RefCell;

use crate::components::ColorFn;
use crate::terminal_image::{
    ImageDimensions, ImageProtocol, ImageRenderOptions, allocate_image_id, get_cell_dimensions,
    get_image_dimensions, image_fallback, render_image,
};
use crate::tui::Component;
use crate::utils::truncate_to_width;

/// The theme for the text fallback, upstream `ImageTheme`.
#[derive(Clone)]
pub struct ImageTheme {
    /// Upstream `fallbackColor`, the colorizer for the fallback line.
    pub fallback_color: ColorFn,
}

impl std::fmt::Debug for ImageTheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageTheme").finish_non_exhaustive()
    }
}

/// Options for [`Image`], upstream `ImageOptions`.
#[derive(Debug, Default, Clone)]
pub struct ImageOptions {
    /// Upstream `maxWidthCells`.
    pub max_width_cells: Option<usize>,
    /// Upstream `maxHeightCells`.
    pub max_height_cells: Option<usize>,
    /// Upstream `filename`, shown shortened in the fallback.
    pub filename: Option<String>,
    /// Upstream `imageId`: when set, this image reuses that Kitty id (for
    /// animations/updates) instead of allocating one.
    pub image_id: Option<u64>,
}

/// The inline-image component, upstream `components/image.ts`'s `Image`.
pub struct Image {
    base64_data: String,
    mime_type: String,
    dimensions: ImageDimensions,
    theme: ImageTheme,
    options: ImageOptions,
    /// The Kitty image id; `None` allocates one on the first Kitty render.
    /// A `RefCell` because [`Component::render`] takes `&self`.
    kitty_image_id: RefCell<Option<u64>>,
    /// The `(width, lines)` render cache, upstream `cachedLines`/`cachedWidth`.
    cache: RefCell<Option<(usize, Vec<String>)>>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("base64_data", &self.base64_data)
            .field("mime_type", &self.mime_type)
            .field("dimensions", &self.dimensions)
            .field("theme", &self.theme)
            .field("options", &self.options)
            .field("kitty_image_id", &self.kitty_image_id)
            .field("cache", &self.cache)
            .finish()
    }
}

impl Image {
    /// Upstream `new Image(base64Data, mimeType, theme, options)`: the image
    /// dimensions come from the payload parser, defaulting to 800x600 when
    /// the format is unknown.
    #[must_use]
    pub fn new(
        base64_data: &str,
        mime_type: &str,
        theme: ImageTheme,
        options: ImageOptions,
    ) -> Self {
        let dimensions = get_image_dimensions(base64_data, mime_type).unwrap_or(ImageDimensions {
            width_px: 800,
            height_px: 600,
        });
        Self::with_dimensions(base64_data, mime_type, theme, options, dimensions)
    }

    /// [`Image::new`] with explicit dimensions, upstream's constructor
    /// `dimensions` argument.
    #[must_use]
    pub fn with_dimensions(
        base64_data: &str,
        mime_type: &str,
        theme: ImageTheme,
        options: ImageOptions,
        dimensions: ImageDimensions,
    ) -> Self {
        let image_id = options.image_id;
        Self {
            base64_data: base64_data.to_string(),
            mime_type: mime_type.to_string(),
            dimensions,
            theme,
            options,
            kitty_image_id: RefCell::new(image_id),
            cache: RefCell::new(None),
        }
    }

    /// The Kitty image id in use, upstream `getImageId` — allocated on the
    /// first render when the terminal speaks Kitty and none was given.
    #[must_use]
    pub fn get_image_id(&self) -> Option<u64> {
        *self.kitty_image_id.borrow()
    }

    fn fallback_lines(&self, width: usize) -> Vec<String> {
        let fallback = image_fallback(
            &self.mime_type,
            Some(self.dimensions),
            self.options.filename.as_deref(),
        );
        let colored = (self.theme.fallback_color)(&fallback);
        vec![truncate_to_width(&colored, width, "...", false)]
    }
}

impl Component for Image {
    fn render(&self, width: usize) -> Vec<String> {
        if let Some((cached_width, cached_lines)) = self.cache.borrow().as_ref()
            && *cached_width == width
        {
            return cached_lines.clone();
        }

        let max_width = width
            .saturating_sub(2)
            .min(self.options.max_width_cells.unwrap_or(60))
            .max(1);
        let cell_dimensions = get_cell_dimensions();
        // The ceiling of a small cell-extent quotient is a small positive integer;
        // upstream computes the same conversion on JS Numbers.
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "cell widths are u16-sized integers and the ceiling is a small positive integer"
        )]
        let default_max_height = ((max_width as f64 * f64::from(cell_dimensions.width_px)
            / f64::from(cell_dimensions.height_px))
        .ceil() as usize)
            .max(1);
        let max_height = self.options.max_height_cells.unwrap_or(default_max_height);

        let caps = crate::terminal_image::get_capabilities();
        let lines: Vec<String> = match caps.images {
            Some(protocol) => {
                if protocol == ImageProtocol::Kitty && self.kitty_image_id.borrow().is_none() {
                    *self.kitty_image_id.borrow_mut() = Some(allocate_image_id());
                }
                let result = render_image(
                    &self.base64_data,
                    self.dimensions,
                    &ImageRenderOptions {
                        max_width_cells: Some(max_width),
                        max_height_cells: Some(max_height),
                        image_id: self.kitty_image_id.borrow().as_ref().copied(),
                        // Kitty: C=1 already prevents cursor movement, so the
                        // component never needs the terminal-side move.
                        move_cursor: Some(false),
                        ..ImageRenderOptions::default()
                    },
                );
                let Some(result) = result else {
                    return self.fallback_lines(width);
                };
                if let Some(image_id) = result.image_id {
                    *self.kitty_image_id.borrow_mut() = Some(image_id);
                }

                let mut lines: Vec<String> = Vec::new();
                match protocol {
                    ImageProtocol::Kitty => {
                        lines.push(result.sequence);
                        for _ in 1..result.rows {
                            lines.push(String::new());
                        }
                    }
                    ImageProtocol::Iterm2 => {
                        // First (rows-1) lines are empty and cleared before
                        // the image is drawn; the last line moves the cursor
                        // back up, draws, then lands where TUI cursor
                        // accounting expects it.
                        for _ in 1..result.rows {
                            lines.push(String::new());
                        }
                        let row_offset = result.rows - 1;
                        let move_up = if row_offset > 0 {
                            format!("\x1b[{row_offset}A")
                        } else {
                            String::new()
                        };
                        lines.push(format!("{move_up}{}", result.sequence));
                    }
                }
                lines
            }
            None => self.fallback_lines(width),
        };

        *self.cache.borrow_mut() = Some((width, lines.clone()));
        lines
    }

    fn invalidate(&self) {
        *self.cache.borrow_mut() = None;
    }
}
