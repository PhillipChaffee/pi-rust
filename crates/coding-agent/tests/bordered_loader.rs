//! The bordered extension loader: cancellable and plain forms, the signal
//! contract, the escape path, and the dispose surface.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use pi_coding_agent::modes::interactive::components::bordered_loader::{
    BorderedLoader, BorderedLoaderOptions,
};
use pi_coding_agent::modes::interactive::components::keybinding_hints::key_text;
use pi_coding_agent::modes::interactive::theme::theme::{init_theme, theme};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::{Component, RenderRequest};

fn noop_request() -> RenderRequest {
    Arc::new(|| {})
}

fn text(loader: &BorderedLoader) -> String {
    strip_ansi(&loader.render(60).join("\n"))
}

#[test]
fn cancellable_form_frames_the_loader_and_shows_the_cancel_hint() {
    init_theme(Some("dark"), false);
    let active = theme();
    let loader = BorderedLoader::new(
        noop_request(),
        &active,
        "Loading things...",
        BorderedLoaderOptions::default(),
    );
    let rendered = text(&loader);
    assert!(rendered.contains("─"));
    assert!(rendered.contains("Loading things..."));
    assert!(rendered.contains(&format!("{} cancel", key_text("tui.select.cancel"))));
    loader.dispose();
}

#[test]
fn plain_form_omits_the_cancel_hint() {
    init_theme(Some("dark"), false);
    let active = theme();
    let loader = BorderedLoader::new(
        noop_request(),
        &active,
        "Loading things...",
        BorderedLoaderOptions { cancellable: false },
    );
    let rendered = text(&loader);
    assert!(rendered.contains("Loading things..."));
    assert!(!rendered.contains(&format!("{} cancel", key_text("tui.select.cancel"))));
    loader.dispose();
}

#[test]
fn escape_cancels_the_cancellable_signal_and_fires_on_abort() {
    init_theme(Some("dark"), false);
    let active = theme();
    let loader = BorderedLoader::new(
        noop_request(),
        &active,
        "Loading...",
        BorderedLoaderOptions::default(),
    );
    let signal = loader.signal();
    let aborted = Rc::new(Cell::new(false));
    let flag = Rc::clone(&aborted);
    loader.set_on_abort(Some(Box::new(move || flag.set(true))));

    assert!(!signal.is_cancelled());
    loader.handle_input("\x1b");
    assert!(signal.is_cancelled());
    assert!(aborted.get());
    loader.dispose();
}

#[test]
fn plain_form_ignores_escape_and_never_cancels() {
    init_theme(Some("dark"), false);
    let active = theme();
    let loader = BorderedLoader::new(
        noop_request(),
        &active,
        "Loading...",
        BorderedLoaderOptions { cancellable: false },
    );
    let signal = loader.signal();
    loader.set_on_abort(Some(Box::new(|| {})));

    loader.handle_input("\x1b");
    assert!(!signal.is_cancelled());
    assert!(!loader.wants_input());
    loader.dispose();
}

#[test]
fn cancellable_form_accepts_focus() {
    init_theme(Some("dark"), false);
    let active = theme();
    let loader = BorderedLoader::new(
        noop_request(),
        &active,
        "Loading...",
        BorderedLoaderOptions::default(),
    );
    assert!(loader.wants_input());
    assert_eq!(loader.children().len(), 6);
    loader.dispose();
}

#[test]
fn plain_form_has_four_children() {
    init_theme(Some("dark"), false);
    let active = theme();
    let loader = BorderedLoader::new(
        noop_request(),
        &active,
        "Loading...",
        BorderedLoaderOptions { cancellable: false },
    );
    assert_eq!(loader.children().len(), 4);
    loader.dispose();
}
