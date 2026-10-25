//! `trust-selector.test.ts` at pin `60e7e76`, ported 1:1, plus boundaries.

#[path = "common/selectors.rs"]
mod selectors;

use std::rc::Rc;

use pi_coding_agent::modes::interactive::components::trust_selector::{
    TrustSelection, TrustSelectorComponent, TrustSelectorOptions,
};
use pi_coding_agent::trust_manager::{ProjectTrustStoreEntry, ProjectTrustUpdate};
use pi_tui::tui::Component;

use selectors::{cancellation, click, output, recording, setup};

fn options(cwd: &str, saved: Option<(&str, bool)>, project_trusted: bool) -> TrustSelectorOptions {
    TrustSelectorOptions {
        cwd: cwd.to_string(),
        saved_decision: saved.map(|(path, decision)| ProjectTrustStoreEntry {
            path: path.to_string(),
            decision,
        }),
        project_trusted,
        on_select: Rc::new(|_| {}),
        on_cancel: Rc::new(|| {}),
    }
}

fn update(path: &str, decision: Option<bool>) -> ProjectTrustUpdate {
    ProjectTrustUpdate {
        path: path.to_string(),
        decision,
    }
}

#[test]
fn keeps_the_saved_trusted_decision_marked_while_browsing() {
    setup(&[]);
    let selector = TrustSelectorComponent::new(options("/project", Some(("/project", true)), true));
    let rendered = output(&selector, 120);
    assert!(rendered.contains("Saved decision: trusted (/project)"));
    assert!(rendered.contains("Current session: trusted"));
    assert!(rendered.contains("→ ✓ Trust"));
    selector.handle_input("\x1b[B");
    let rendered = output(&selector, 120);
    assert!(rendered.contains("✓ Trust"));
    assert!(rendered.contains("→   Trust parent folder (/)"));
    assert!(!rendered.contains("✓ Do not trust"));
}

#[test]
fn selects_a_trust_decision() {
    setup(&[]);
    let (selected, select) = recording();
    let selector = TrustSelectorComponent::new(TrustSelectorOptions {
        on_select: select,
        ..options("/project", None, false)
    });
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![TrustSelection {
            trusted: true,
            updates: vec![update("/project", Some(true))]
        }]
    );
}

#[test]
fn labels_saved_ancestor_decisions_as_inherited() {
    setup(&[]);
    let selector = TrustSelectorComponent::new(options(
        "/parent/project/nested",
        Some(("/parent", true)),
        true,
    ));
    assert!(output(&selector, 120).contains("Saved decision: trusted (inherited from /parent)"));
}

#[test]
fn adds_a_trust_parent_option() {
    setup(&[]);
    let (selected, select) = recording();
    let selector = TrustSelectorComponent::new(TrustSelectorOptions {
        on_select: select,
        ..options("/parent/project", Some(("/parent", true)), true)
    });
    let rendered = output(&selector, 120);
    assert!(rendered.contains("Saved decision: trusted (inherited from /parent)"));
    assert!(rendered.contains("✓ Trust parent folder (/parent)"));
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![TrustSelection {
            trusted: true,
            updates: vec![
                update("/parent", Some(true)),
                update("/parent/project", None)
            ]
        }]
    );
}

#[test]
fn saved_untrusted_decision_is_independent_of_the_session_and_browsing() {
    setup(&[]);
    let (selected, select) = recording();
    let (cancelled, cancel) = cancellation();
    let selector = TrustSelectorComponent::new(TrustSelectorOptions {
        on_select: select,
        on_cancel: cancel,
        ..options("/project", Some(("/project", false)), true)
    });
    let rendered = output(&selector, 120);
    assert!(rendered.contains("Saved decision: untrusted (/project)"));
    assert!(rendered.contains("Current session: trusted"));
    assert!(rendered.contains("→ ✓ Do not trust"));
    for _ in 0..5 {
        selector.handle_input("j");
    }
    assert!(output(&selector, 120).contains("→ ✓ Do not trust"));
    selector.handle_input("\r");
    assert_eq!(
        *selected.borrow(),
        vec![TrustSelection {
            trusted: false,
            updates: vec![update("/project", Some(false))]
        }]
    );
    for _ in 0..5 {
        selector.handle_input("k");
    }
    assert!(output(&selector, 120).contains("→   Trust"));
    assert!(output(&selector, 120).contains("✓ Do not trust"));
    selector.handle_input("ignored");
    selector.handle_input("\x1b");
    assert_eq!(cancelled.get(), 1);
    assert_eq!(selected.borrow().len(), 1);
}

#[test]
fn inherited_untrusted_decisions_have_no_false_current_marker() {
    setup(&[]);
    let selector =
        TrustSelectorComponent::new(options("/parent/project", Some(("/parent", false)), false));
    let rendered = output(&selector, 120);
    assert!(rendered.contains("Saved decision: untrusted (inherited from /parent)"));
    assert!(rendered.contains("Current session: untrusted"));
    assert!(!rendered.contains('✓'));
    assert!(rendered.contains("→   Trust"));
}

#[test]
fn root_has_no_parent_and_uses_remapped_navigation_confirm_and_cancel() {
    setup(&[
        ("tui.select.up", &["ctrl+p"]),
        ("tui.select.down", &["ctrl+n"]),
        ("tui.select.confirm", &["ctrl+y"]),
        ("tui.select.cancel", &["ctrl+x"]),
    ]);
    let (selected, select) = recording();
    let (cancelled, cancel) = cancellation();
    let selector = TrustSelectorComponent::new(TrustSelectorOptions {
        on_select: select,
        on_cancel: cancel,
        ..options("/", None, false)
    });
    let rendered = output(&selector, 120);
    assert!(!rendered.contains("parent folder"));
    assert!(rendered.contains("Saved decision: none"));
    assert!(rendered.contains("ctrl+y save"));
    assert!(rendered.contains("ctrl+x cancel"));
    for key in ["\x1b[B", "\r", "\x1b"] {
        selector.handle_input(key);
    }
    assert!(selected.borrow().is_empty());
    assert_eq!(cancelled.get(), 0);
    selector.handle_input("\x0e");
    selector.handle_input("\x19");
    selector.handle_input("\x10");
    selector.handle_input("\n");
    selector.handle_input("\x18");
    assert_eq!(
        *selected.borrow(),
        vec![
            TrustSelection {
                trusted: false,
                updates: vec![update("/", Some(false))]
            },
            TrustSelection {
                trusted: true,
                updates: vec![update("/", Some(true))]
            }
        ]
    );
    assert_eq!(cancelled.get(), 1);
}

#[test]
fn delegates_container_surface_without_mouse_changing_trust() {
    setup(&[]);
    let (selected, select) = recording();
    let selector = TrustSelectorComponent::new(TrustSelectorOptions {
        on_select: select,
        ..options("/project", None, false)
    });
    assert!(selector.wants_input());
    assert!(format!("{selector:?}").contains("TrustSelectorComponent"));
    assert!(format!("{:?}", options("/project", None, false)).contains("TrustSelectorOptions"));
    assert!(!selector.children().is_empty());
    assert!(selector.as_focusable().is_none());
    let rendered = selector.render(80);
    assert!(selector.handle_mouse(&click(0, 20)).is_none());
    assert!(selected.borrow().is_empty());
    selector.invalidate();
    assert_eq!(selector.render(80), rendered);
}
