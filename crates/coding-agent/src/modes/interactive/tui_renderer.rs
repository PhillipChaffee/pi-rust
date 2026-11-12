//! The interactive TUI composition root and its stable handle, upstream
//! `src/modes/interactive/tui-renderer.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! [`create_interactive_tui`] selects the alternate-screen renderer for
//! `TuiMode::Fullscreen` and the main-screen renderer for
//! `TuiMode::Regular`, wiring the theme's search-match styles, the
//! jump-to-latest indicator with its configured shortcut, URL opening,
//! right-click paste, copy-on-select, and the native clipboard into the
//! alternate-screen renderer.
//!
//! Port restatements:
//!
//! - Upstream's two overloads return `TuiAltScreen`/`TuiMainScreen` because
//!   each renderer *is* the TUI. Here [`Tui::new`] consumes the renderer, so
//!   [`InteractiveTui`] keeps the base `Rc<Tui>` plus the renderer-local
//!   handle: the cloned [`TuiAltScreen`], or the `Rc<TuiMainScreen>`
//!   behind a local delegating [`TuiRenderer`]. The frame reaches the
//!   alternate-screen-only surface (`set_copy_on_select`, selection state)
//!   through [`InteractiveTui::viewport`].
//! - `createInteractiveTuiReference`'s dynamic `Proxy` has no Rust
//!   counterpart, so it becomes [`InteractiveTuiReference`]: an explicit
//!   handle holding the current `Rc<Tui>` and the force-free
//!   [`RenderRequest`] that targets it. The frame re-points the handle with
//!   [`InteractiveTuiReference::repoint`] after a renderer swap; delegated
//!   calls read the current handle every time, and a [`RenderRequest`]
//!   captured before the swap routes to the replacement through the shared
//!   forward slot (upstream's #7731 regression). The rest of the TUI
//!   surface is reachable through [`InteractiveTuiReference::with_current`];
//!   upstream's `get`/`set`/`has` reach every property, which a typed handle
//!   cannot enumerate.
//! - `copySelection`'s awaited `copyToClipboard` becomes the synchronous
//!   native-clipboard write the tui port's `copySelection` contract records:
//!   the renderer callback is sync, so there is no place to await the
//!   command/OSC 52 fallbacks, and a missing native clipboard answers the
//!   renderer's generic failure.
//! - [`TuiAltScreenConfig`] and [`TuiMainScreenConfig`] carry the environment
//!   and platform seams in their ported types; this module leaves them at
//!   their process defaults, matching upstream's direct `process.env` and
//!   `process.platform` reads.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex, PoisonError};

use pi_tui::alt_screen_search::NavigationButtonStyleFn;
use pi_tui::components::InputStyleFn;
use pi_tui::terminal::{ProcessTerminal, Terminal};
use pi_tui::tui::{Component, RenderRequest, Tui, TuiConfig, TuiMode, TuiRenderer, TuiStopOptions};
use pi_tui::tui_alt_screen::{
    CopySelectionCallback, CopySelectionResult, OpenUrlCallback, PasteCallback,
    ScrollToEndIndicatorCallback, TuiAltScreen, TuiAltScreenConfig,
};
use pi_tui::tui_main_screen::{TuiMainScreen, TuiMainScreenConfig};

use crate::modes::interactive::components::keybinding_hints::key_display_text;
use crate::modes::interactive::theme::theme::{bold, inverse, theme, underline};
use crate::utils::clipboard::get_native_clipboard;
use crate::utils::open_browser::open_browser;

/// The construction options, upstream `InteractiveTuiOptions`.
pub struct InteractiveTuiOptions {
    /// The renderer strategy, upstream's `tuiMode` union.
    pub tui_mode: TuiMode,
    /// Whether the hardware cursor shows when positioned.
    pub show_hardware_cursor: bool,
    /// The debug/crash log directory, upstream `logDirectory`.
    pub log_directory: String,
    /// The terminal to render through; the process terminal when absent.
    pub terminal: Option<Box<dyn Terminal>>,
    /// Handle a secondary-button paste request, upstream
    /// `onRightClickPaste`.
    pub on_right_click_paste: Option<PasteCallback>,
    /// Copy a completed selection automatically, upstream
    /// `fullscreenCopyOnSelect`.
    pub fullscreen_copy_on_select: Option<bool>,
}

impl std::fmt::Debug for InteractiveTuiOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveTuiOptions")
            .field("tui_mode", &self.tui_mode)
            .field("show_hardware_cursor", &self.show_hardware_cursor)
            .field("log_directory", &self.log_directory)
            .field("terminal", &self.terminal.is_some())
            .field("on_right_click_paste", &self.on_right_click_paste.is_some())
            .field("fullscreen_copy_on_select", &self.fullscreen_copy_on_select)
            .finish()
    }
}

/// The renderer-backed TUI a presentation mounts, upstream's
/// `TuiMainScreen | TuiAltScreen`.
pub struct InteractiveTui {
    tui: Rc<Tui>,
    main_screen: Option<Rc<TuiMainScreen>>,
    viewport: Option<TuiAltScreen>,
}

impl std::fmt::Debug for InteractiveTui {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveTui")
            .field("tui", &self.tui)
            .field("main_screen", &self.main_screen.is_some())
            .field("viewport", &self.viewport.is_some())
            .finish()
    }
}

impl InteractiveTui {
    /// The base TUI every presentation method runs on.
    #[must_use]
    pub fn tui(&self) -> Rc<Tui> {
        Rc::clone(&self.tui)
    }

    /// The main-screen renderer handle in regular mode.
    #[must_use]
    pub const fn main_screen(&self) -> Option<&Rc<TuiMainScreen>> {
        self.main_screen.as_ref()
    }

    /// The alternate-screen renderer handle in fullscreen mode.
    #[must_use]
    pub const fn viewport(&self) -> Option<&TuiAltScreen> {
        self.viewport.as_ref()
    }
}

/// A shared main-screen renderer, the cloneable handle [`Tui::new`] needs
/// while the frame keeps [`TuiMainScreen`] for render-state capture and
/// restore.
#[derive(Debug)]
struct SharedMainScreen(Rc<TuiMainScreen>);

impl TuiRenderer for SharedMainScreen {
    fn mode(&self) -> TuiMode {
        TuiRenderer::mode(self.0.as_ref())
    }

    fn do_render(&self, tui: &Tui) {
        TuiRenderer::do_render(self.0.as_ref(), tui);
    }

    fn reset_render_state(&self) {
        TuiRenderer::reset_render_state(self.0.as_ref());
    }

    fn before_terminal_stop(&self, tui: &Tui, options: &TuiStopOptions) {
        TuiRenderer::before_terminal_stop(self.0.as_ref(), tui, options);
    }
}

/// The alternate-screen search-match style: the theme's search background
/// and text color, underlined.
fn search_match_style() -> InputStyleFn {
    Rc::new(|text| {
        let active_theme = theme();
        underline(&active_theme.bg("searchMatchBg", &active_theme.fg("searchMatchText", text)))
    })
}

/// The current alternate-screen search-match style: bold and inverse over
/// the shared search-match coloring.
fn search_current_match_style() -> InputStyleFn {
    Rc::new(|text| {
        let active_theme = theme();
        bold(&inverse(&active_theme.bg(
            "searchMatchBg",
            &active_theme.fg("searchMatchText", text),
        )))
    })
}

/// The search navigation button style: underlined while hovered.
fn search_navigation_button_style() -> NavigationButtonStyleFn {
    Rc::new(|text, hovered| {
        if hovered {
            underline(text)
        } else {
            text.to_string()
        }
    })
}

/// The jump-to-latest indicator, upstream `scrollToEndIndicator`.
fn scroll_to_end_indicator() -> ScrollToEndIndicatorCallback {
    Rc::new(move || {
        let active_theme = theme();
        let shortcut = key_display_text("tui.altScreen.bottom");
        let label = if shortcut.is_empty() {
            " ↓ Jump to latest message ".to_string()
        } else {
            format!(" ↓ Jump to latest message · {shortcut} ")
        };
        active_theme.bg("selectedBg", &active_theme.fg("text", &label))
    })
}

/// Open an activated hyperlink in the platform browser, upstream
/// `openUrl: openBrowser`.
fn open_url_callback() -> OpenUrlCallback {
    Rc::new(open_browser)
}

/// Copy a selection through the native clipboard, upstream's
/// `copySelection` over `copyToClipboard`.
fn copy_selection_callback() -> CopySelectionCallback {
    Rc::new(|text| match get_native_clipboard() {
        Some(clipboard) if clipboard.set_text(text) => CopySelectionResult::Copied,
        Some(_) | None => CopySelectionResult::Failed,
    })
}

/// The composition root shared by coding-agent presentations, upstream
/// `createInteractiveTui`.
///
/// The fullscreen branch wires every alternate-screen callback; the regular
/// branch keeps the main-screen renderer bare, exactly as upstream's
/// `new TuiMainScreen(terminal, ...)` does.
#[must_use]
pub fn create_interactive_tui(options: InteractiveTuiOptions) -> InteractiveTui {
    let terminal = options
        .terminal
        .unwrap_or_else(|| Box::new(ProcessTerminal::new()));
    let show_hardware_cursor = Some(options.show_hardware_cursor);
    let log_directory = Some(PathBuf::from(&options.log_directory));

    match options.tui_mode {
        TuiMode::Fullscreen => {
            let viewport = TuiAltScreen::new(TuiAltScreenConfig {
                search_match_style: Some(search_match_style()),
                search_current_match_style: Some(search_current_match_style()),
                search_navigation_button_style: Some(search_navigation_button_style()),
                scroll_to_end_indicator: Some(scroll_to_end_indicator()),
                open_url: Some(open_url_callback()),
                on_right_click_paste: options.on_right_click_paste,
                copy_on_select: options.fullscreen_copy_on_select,
                copy_selection: Some(copy_selection_callback()),
                ..TuiAltScreenConfig::default()
            });
            let tui = Tui::new(TuiConfig {
                terminal: Some(terminal),
                renderer: Some(Box::new(viewport.clone())),
                show_hardware_cursor,
                log_directory,
                ..TuiConfig::default()
            });
            InteractiveTui {
                tui,
                main_screen: None,
                viewport: Some(viewport),
            }
        }
        TuiMode::Regular => {
            let main_screen = Rc::new(TuiMainScreen::new(TuiMainScreenConfig::default()));
            let tui = Tui::new(TuiConfig {
                terminal: Some(terminal),
                renderer: Some(Box::new(SharedMainScreen(Rc::clone(&main_screen)))),
                show_hardware_cursor,
                log_directory,
                ..TuiConfig::default()
            });
            InteractiveTui {
                tui,
                main_screen: Some(main_screen),
                viewport: None,
            }
        }
    }
}

/// The stable handle components render through while the interactive frame
/// replaces the active TUI, upstream `createInteractiveTuiReference`.
#[derive(Clone)]
pub struct InteractiveTuiReference {
    current: Rc<RefCell<Rc<Tui>>>,
    /// The force-free render request targeting the current TUI, replaced on
    /// every [`Self::repoint`] so a request minted earlier reaches the
    /// replacement.
    forward: Arc<Mutex<Option<RenderRequest>>>,
}

impl std::fmt::Debug for InteractiveTuiReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveTuiReference")
            .field("mode", &self.mode())
            .finish_non_exhaustive()
    }
}

impl InteractiveTuiReference {
    /// Create the handle over the first TUI.
    #[must_use]
    pub fn new(tui: Rc<Tui>) -> Self {
        let forward = Arc::new(Mutex::new(Some(tui.render_request())));
        Self {
            current: Rc::new(RefCell::new(tui)),
            forward,
        }
    }

    /// The current TUI, upstream the proxy's `get`/`has` target.
    #[must_use]
    pub fn current(&self) -> Rc<Tui> {
        Rc::clone(&self.current.borrow())
    }

    /// Re-point the handle at the replacement TUI, the frame's stand-in for
    /// updating the getter closure upstream's proxy reads.
    pub fn repoint(&self, tui: Rc<Tui>) {
        let forward = tui.render_request();
        *self.forward.lock().unwrap_or_else(PoisonError::into_inner) = Some(forward);
        *self.current.borrow_mut() = tui;
    }

    /// Run one call against the current TUI, upstream's property forwarding.
    pub fn with_current<R>(&self, call: impl FnOnce(&Tui) -> R) -> R {
        let tui = self.current();
        call(&tui)
    }

    /// Upstream `requestRender`.
    pub fn request_render(&self, force: bool) {
        self.current().request_render(force);
    }

    /// A force-free render request that routes through the handle, so a
    /// component holding it keeps repainting the active TUI after a swap.
    #[must_use]
    pub fn render_request(&self) -> RenderRequest {
        let forward = Arc::clone(&self.forward);
        Arc::new(move || {
            if let Some(request) = forward
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_ref()
            {
                request();
            }
        })
    }

    /// Upstream `setFocus`.
    pub fn set_focus(&self, component: Option<Rc<dyn Component>>) {
        self.current().set_focus(component);
    }

    /// Upstream `getFocusedComponent`.
    #[must_use]
    pub fn get_focused_component(&self) -> Option<Rc<dyn Component>> {
        self.current().get_focused_component()
    }

    /// Upstream `addChild`.
    pub fn add_child(&self, component: Rc<dyn Component>) {
        self.current().add_child(component);
    }

    /// Upstream `removeChild`.
    pub fn remove_child(&self, component: &Rc<dyn Component>) {
        self.current().remove_child(component);
    }

    /// Upstream `clear`.
    pub fn clear(&self) {
        self.current().clear();
    }

    /// Upstream `mode`.
    #[must_use]
    pub fn mode(&self) -> TuiMode {
        self.current().mode()
    }

    /// Upstream `isViewportTUI`.
    #[must_use]
    pub fn is_viewport_tui(&self) -> bool {
        self.current().is_viewport_tui()
    }

    /// Upstream `setLayoutRoot`.
    pub fn set_layout_root(&self, component: Option<Rc<dyn Component>>) {
        self.current().set_layout_root(component);
    }

    /// Upstream `setClearOnShrink`.
    pub fn set_clear_on_shrink(&self, enabled: bool) {
        self.current().set_clear_on_shrink(enabled);
    }

    /// Upstream `start`.
    pub fn start(&self) {
        self.current().start();
    }

    /// Upstream `stop`.
    pub fn stop(&self, options: TuiStopOptions) {
        self.current().stop(options);
    }
}

/// Create the stable handle components render through, upstream
/// `createInteractiveTuiReference`.
#[must_use]
pub fn create_interactive_tui_reference(tui: Rc<Tui>) -> InteractiveTuiReference {
    InteractiveTuiReference::new(tui)
}
