//! The custom session-entry block, upstream
//! `src/modes/interactive/components/custom-entry.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! An extension renderer draws its own entry; when it declines, the host
//! adds no row, and when it panics the block degrades to a red error box.
//! Restatement: upstream's `EntryRenderer` and `EntryRenderOptions` live in
//! `core/extensions/types.ts`, which the port has not carried yet, so this
//! module defines the same call shape locally; the host owns transcript
//! spacing and the renderer answers only its content.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::{Box, ColorFn, Text};
use pi_tui::tui::{Component, Container};

use crate::modes::interactive::theme::theme::{Theme, theme};
use crate::session_manager::entries::CustomEntry;

/// The expanded flag handed to an entry renderer, upstream's
/// `EntryRenderOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntryRenderOptions {
    /// Whether the entry view is expanded.
    pub expanded: bool,
}

/// A custom-entry renderer, upstream's `EntryRenderer`.
pub type EntryRenderer = Arc<
    dyn Fn(&CustomEntry, &EntryRenderOptions, &Theme) -> Option<Rc<dyn Component>> + Send + Sync,
>;

/// Component that renders a custom session entry from extensions.
pub struct CustomEntryComponent {
    entry: CustomEntry,
    renderer: EntryRenderer,
    custom_component: RefCell<Option<Rc<dyn Component>>>,
    expanded: Cell<bool>,
    container: Container,
}

impl std::fmt::Debug for CustomEntryComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomEntryComponent")
            .field("custom_type", &self.entry.custom_type)
            .field("expanded", &self.expanded.get())
            .field("has_content", &self.has_content())
            .finish_non_exhaustive()
    }
}

impl CustomEntryComponent {
    /// Build the block, upstream's constructor.
    #[must_use]
    pub fn new(entry: CustomEntry, renderer: EntryRenderer) -> Self {
        let component = Self {
            entry,
            renderer,
            custom_component: RefCell::new(None),
            expanded: Cell::new(false),
            container: Container::default(),
        };
        component.rebuild();
        component
    }

    /// Whether the renderer produced a child, upstream's `hasContent`.
    #[must_use]
    pub fn has_content(&self) -> bool {
        self.custom_component.borrow().is_some()
    }

    /// Set the expanded state and repaint when it changes.
    pub fn set_expanded(&self, expanded: bool) {
        if self.expanded.get() != expanded {
            self.expanded.set(expanded);
            self.rebuild();
        }
    }

    /// Whether the entry is expanded.
    #[must_use]
    pub const fn is_expanded(&self) -> bool {
        self.expanded.get()
    }

    /// Rebuild from the renderer, upstream's `rebuild`.
    fn rebuild(&self) {
        self.container.clear();
        *self.custom_component.borrow_mut() = None;

        let active = theme();
        let options = EntryRenderOptions {
            expanded: self.expanded.get(),
        };
        let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.renderer)(&self.entry, &options, &active)
        }));
        let component = match rendered {
            Ok(component) => component,
            Err(payload) => Some(error_box(&self.entry.custom_type, &panic_message(&payload))),
        };

        let Some(component) = component else {
            return;
        };
        *self.custom_component.borrow_mut() = Some(Rc::clone(&component));
        self.container
            .add_child(Rc::new(pi_tui::components::Spacer::new()));
        self.container.add_child(component);
    }
}

/// The fallback row a panicking renderer produces, upstream's `catch` box.
fn error_box(custom_type: &str, message: &str) -> Rc<dyn Component> {
    let active = theme();
    let background: ColorFn = {
        let active = Arc::clone(&active);
        Arc::new(move |text| active.bg("customMessageBg", text))
    };
    let box_component = Box::with_padding(1, 1, Some(background));
    box_component.add_child(Rc::new(Text::with_padding(
        active.fg(
            "error",
            &format!("[{custom_type}] renderer failed: {message}"),
        ),
        0,
        0,
    )));
    Rc::new(box_component)
}

/// The message inside a caught panic payload, upstream's `Error.message`
/// extraction.
fn panic_message(payload: &std::boxed::Box<dyn Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown error".to_string())
}

impl Component for CustomEntryComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_mouse(
        &self,
        event: &pi_tui::tui::TuiMouseEvent,
    ) -> Option<pi_tui::tui::TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }

    fn invalidate(&self) {
        self.rebuild();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}
