//! The bordered extension loader, upstream
//! `src/modes/interactive/components/bordered-loader.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The block frames a loader between two theme-colored rules; the cancellable
//! form adds an escape hint and cancels its [`CancellationToken`] on the
//! `tui.select.cancel` binding, while the plain form exposes a token that is
//! never cancelled. Restatements: upstream's `TUI` handle narrows to the
//! [`RenderRequest`] closure the loader drives, its `AbortController` signal
//! becomes the cancellation token the workspace already uses, and
//! `class BorderedLoader extends Container` becomes an owned
//! [`Container`] with [`Component`] delegation.

use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::{CancellableLoader, ColorFn, Loader, Spacer, Text};
use pi_tui::tui::{Component, Container, RenderRequest};
use tokio_util::sync::CancellationToken;

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::key_hint;
use crate::modes::interactive::theme::theme::Theme;

/// The constructor's optional flags, upstream's `{ cancellable? }`.
#[derive(Clone, Copy, Debug)]
pub struct BorderedLoaderOptions {
    /// Whether Escape cancels the loader; defaults to `true`, upstream's
    /// `options?.cancellable ?? true`.
    pub cancellable: bool,
}

impl Default for BorderedLoaderOptions {
    fn default() -> Self {
        Self { cancellable: true }
    }
}

/// Erase a shared loader to the component trait object without a cast
/// expression, which the workspace's trivial-cast lint rejects.
fn shared_component<T: Component + 'static>(component: &Rc<T>) -> Rc<dyn Component> {
    let shared: Rc<T> = Rc::clone(component);
    let erased: Rc<dyn Component> = shared;
    erased
}

/// The embedded loader, cancellable or plain.
enum LoaderKind {
    Cancellable(Rc<CancellableLoader>),
    Plain {
        loader: Rc<Loader>,
        /// The token the plain form exposes, upstream's private
        /// `AbortController` that nothing ever aborts.
        signal: CancellationToken,
    },
}

/// Loader wrapped with borders for extension UI.
pub struct BorderedLoader {
    container: Container,
    loader: LoaderKind,
}

impl std::fmt::Debug for BorderedLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BorderedLoader")
            .field(
                "cancellable",
                &matches!(self.loader, LoaderKind::Cancellable(_)),
            )
            .finish_non_exhaustive()
    }
}

impl BorderedLoader {
    /// Build the block, upstream's constructor.
    #[must_use]
    pub fn new(
        render_request: RenderRequest,
        theme: &Arc<Theme>,
        message: &str,
        options: BorderedLoaderOptions,
    ) -> Self {
        let border_color: Rc<dyn Fn(&str) -> String> = {
            let theme = Arc::clone(theme);
            Rc::new(move |text| theme.fg("border", text))
        };
        let spinner_color: ColorFn = {
            let theme = Arc::clone(theme);
            Arc::new(move |text| theme.fg("accent", text))
        };
        let message_color: ColorFn = {
            let theme = Arc::clone(theme);
            Arc::new(move |text| theme.fg("muted", text))
        };

        let loader = if options.cancellable {
            LoaderKind::Cancellable(Rc::new(CancellableLoader::new(
                render_request,
                spinner_color,
                message_color,
                message,
                None,
            )))
        } else {
            LoaderKind::Plain {
                loader: Rc::new(Loader::new(
                    render_request,
                    spinner_color,
                    message_color,
                    message,
                    None,
                )),
                signal: CancellationToken::new(),
            }
        };

        let container = Container::default();
        let component = Self { container, loader };
        component
            .container
            .add_child(Rc::new(DynamicBorder::new(Rc::clone(&border_color))));
        let loader_child: Rc<dyn Component> = match &component.loader {
            LoaderKind::Cancellable(loader) => shared_component(loader),
            LoaderKind::Plain { loader, .. } => shared_component(loader),
        };
        component.container.add_child(loader_child);
        if options.cancellable {
            component.container.add_child(Rc::new(Spacer::new()));
            component.container.add_child(Rc::new(Text::with_padding(
                key_hint("tui.select.cancel", "cancel"),
                1,
                0,
            )));
        }
        component.container.add_child(Rc::new(Spacer::new()));
        component
            .container
            .add_child(Rc::new(DynamicBorder::new(border_color)));
        component
    }

    /// Upstream `get signal`: the token aborted when a cancellable loader is
    /// escaped, or the never-cancelled token of the plain form.
    #[must_use]
    pub fn signal(&self) -> CancellationToken {
        match &self.loader {
            LoaderKind::Cancellable(loader) => loader.cancellation_token(),
            LoaderKind::Plain { signal, .. } => signal.clone(),
        }
    }

    /// Upstream's `onAbort` setter; ignored by the plain form.
    pub fn set_on_abort(&self, on_abort: Option<Box<dyn FnMut()>>) {
        if let LoaderKind::Cancellable(loader) = &self.loader {
            loader.set_on_abort(on_abort);
        }
    }

    /// Stop the loader, upstream `dispose`.
    pub fn dispose(&self) {
        match &self.loader {
            LoaderKind::Cancellable(loader) => loader.dispose(),
            LoaderKind::Plain { loader, .. } => loader.stop(),
        }
    }
}

impl Component for BorderedLoader {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    /// Upstream forwards input only to the cancellable loader.
    fn handle_input(&self, data: &str) {
        if let LoaderKind::Cancellable(loader) = &self.loader {
            loader.handle_input(data);
        }
    }

    fn wants_input(&self) -> bool {
        matches!(self.loader, LoaderKind::Cancellable(_))
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}
