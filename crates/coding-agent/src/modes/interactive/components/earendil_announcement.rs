//! The Earendil announcement, upstream
//! `src/modes/interactive/components/earendil-announcement.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The block frames the announcement text between accent rules and adds the
//! bundled `clankolas.png` when the asset is present. Restatement: upstream's
//! `getBundledInteractiveAssetPath` resolves `src`/`dist` layouts under the
//! package; the native binary keeps `modes/interactive/assets` beside the
//! executable, and a missing file degrades to the text-only announcement,
//! exactly upstream's `catch`.

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use base64::Engine as _;
use pi_tui::components::image::{Image, ImageOptions, ImageTheme};
use pi_tui::components::{ColorFn, Spacer, Text};
use pi_tui::tui::{Component, Container};

use super::dynamic_border::DynamicBorder;
use crate::config::get_package_dir;
use crate::modes::interactive::theme::theme::{bold, theme};

const BLOG_URL: &str = "https://mariozechner.at/posts/2026-04-08-ive-sold-out/";
const IMAGE_FILENAME: &str = "clankolas.png";

/// Upstream's `getBundledInteractiveAssetPath` for the native layout.
fn bundled_interactive_asset_path(name: &str) -> PathBuf {
    get_package_dir()
        .join("modes")
        .join("interactive")
        .join("assets")
        .join(name)
}

/// Read the bundled image once, upstream's `cachedImageBase64` and
/// `attemptedImageLoad`.
fn load_image_base64() -> Option<String> {
    static CACHED: LazyLock<Option<String>> = LazyLock::new(|| {
        let bytes = std::fs::read(bundled_interactive_asset_path(IMAGE_FILENAME)).ok()?;
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    });
    CACHED.clone()
}

/// The announcement shown when `pi has joined Earendil`.
pub struct EarendilAnnouncementComponent {
    container: Container,
}

impl std::fmt::Debug for EarendilAnnouncementComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EarendilAnnouncementComponent")
            .finish_non_exhaustive()
    }
}

impl Default for EarendilAnnouncementComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl EarendilAnnouncementComponent {
    /// Build the announcement from the bundled asset, upstream's constructor.
    #[must_use]
    pub fn new() -> Self {
        Self::with_image_base64(load_image_base64())
    }

    /// Build with an explicit image payload; `None` is the text-only form,
    /// which is also what a missing bundle produces. Test seam.
    #[must_use]
    pub fn with_image_base64(image_base64: Option<String>) -> Self {
        let accent: Rc<dyn Fn(&str) -> String> = Rc::new(|text: &str| theme().fg("accent", text));
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::new(Rc::clone(&accent))));
        container.add_child(Rc::new(Text::with_padding(
            bold(&theme().fg("accent", "pi has joined Earendil")),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("muted", "Read the blog post:"),
            1,
            0,
        )));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("mdLink", BLOG_URL),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));

        if let Some(image_base64) = image_base64 {
            let fallback: ColorFn = Arc::new(|text: &str| theme().fg("muted", text));
            container.add_child(Rc::new(Image::new(
                &image_base64,
                "image/png",
                ImageTheme {
                    fallback_color: fallback,
                },
                ImageOptions {
                    max_width_cells: Some(56),
                    filename: Some(IMAGE_FILENAME.to_string()),
                    ..ImageOptions::default()
                },
            )));
            container.add_child(Rc::new(Spacer::new()));
        }

        container.add_child(Rc::new(DynamicBorder::new(accent)));
        Self { container }
    }
}

impl Component for EarendilAnnouncementComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}
