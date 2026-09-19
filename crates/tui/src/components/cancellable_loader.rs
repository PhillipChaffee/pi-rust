//! CancellableLoader component, ported from
//! `packages/tui/src/components/cancellable-loader.ts` in earendil-works/pi
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#42).
//!
//! A loader that can be cancelled with Escape.
//!
//! Restatement: `CancellableLoader`'s `AbortController`/`AbortSignal` pair
//! becomes `tokio_util::sync::CancellationToken` per the stack decision (map
//! ticket "Decide the Rust stack"). `class CancellableLoader extends Loader`
//! becomes composition: this embeds a [`Loader`], reachable through
//! [`CancellableLoader::loader`] for the inherited surface.

use std::cell::RefCell;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::components::{ColorFn, Loader, LoaderIndicatorOptions};
use crate::keybindings::get_keybindings;
use crate::keys::KeyParser;
use crate::tui::Component;

/// The abort callback, upstream `onAbort`.
type OnAbort = Box<dyn FnMut()>;

/// Loader that can be cancelled with Escape. Wraps a [`Loader`] with a
/// cancellation token for cancelling async operations.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use pi_tui::components::CancellableLoader;
///
/// let loader = CancellableLoader::new(
///     Arc::new(|| {}),
///     Arc::new(|text: &str| text.to_string()),
///     Arc::new(|text: &str| text.to_string()),
///     "Working...",
///     None,
/// );
/// loader.set_on_abort(Some(Box::new(|| { /* done */ })));
/// let token = loader.cancellation_token();
/// do_work(token);
/// # fn do_work(token: tokio_util::sync::CancellationToken) {}
/// ```
pub struct CancellableLoader {
    loader: Loader,
    token: CancellationToken,
    /// Session-owned parser restatement: upstream's global manager paired
    /// `matches` with the module-global parser; the port's
    /// [`crate::keybindings::KeybindingsManager::matches`] takes the parser
    /// explicitly, and a loader not wired to a session owns a default one.
    parser: KeyParser,
    on_abort: RefCell<Option<OnAbort>>,
}

impl CancellableLoader {
    /// Upstream's inherited `new CancellableLoader(ui, spinnerColorFn,
    /// messageColorFn, message, indicator)`.
    #[must_use]
    pub fn new(
        render_request: Arc<dyn Fn() + Send + Sync>,
        spinner_color_fn: ColorFn,
        message_color_fn: ColorFn,
        message: impl Into<String>,
        indicator: Option<LoaderIndicatorOptions>,
    ) -> Self {
        Self {
            loader: Loader::new(
                render_request,
                spinner_color_fn,
                message_color_fn,
                message,
                indicator,
            ),
            token: CancellationToken::new(),
            parser: KeyParser::new(),
            on_abort: RefCell::new(None),
        }
    }

    /// Upstream `onAbort`: the callback invoked when the user presses
    /// Escape. `None` clears it.
    pub fn set_on_abort(&self, on_abort: Option<Box<dyn FnMut()>>) {
        *self.on_abort.borrow_mut() = on_abort;
    }

    /// Upstream `get signal`: the cancellation token aborted when the user
    /// presses Escape.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Upstream `get aborted`: whether the loader was aborted.
    #[must_use]
    pub fn is_aborted(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Upstream `dispose`.
    pub fn dispose(&self) {
        self.loader.stop();
    }

    /// The embedded [`Loader`], for the surface upstream inherited.
    #[must_use]
    pub const fn loader(&self) -> &Loader {
        &self.loader
    }
}

impl std::fmt::Debug for CancellableLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellableLoader").finish_non_exhaustive()
    }
}

impl Component for CancellableLoader {
    fn render(&self, width: usize) -> Vec<String> {
        self.loader.render(width)
    }

    /// Upstream `handleInput`: cancel on the `tui.select.cancel` binding.
    fn handle_input(&self, data: &str) {
        if get_keybindings().matches(&self.parser, data, "tui.select.cancel") {
            self.token.cancel();
            if let Some(on_abort) = self.on_abort.borrow_mut().as_mut() {
                on_abort();
            }
        }
    }

    fn invalidate(&self) {
        self.loader.invalidate();
    }
}
