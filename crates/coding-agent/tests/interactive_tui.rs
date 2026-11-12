//! `test/interactive-tui.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! The first `createInteractiveTui` case (renderer selection) lives in
//! `tests/tui_renderer.rs`; this file ports the remaining cases and keeps the
//! alternate-screen write assertion the earlier suite did not make.
//!
//! Restatements:
//! - Upstream's `VirtualTerminal.getViewport` becomes assertions over the raw
//!   terminal write stream; the label text is contiguous in the styled write.
//! - The clipboard mock becomes the port's native-clipboard override.
//! - `requestRender` call-count assertions are dropped where the port's
//!   renderer writes through the demand flag rather than a spy; the user
//!   visible confirmation (`Copied!` / status row) is asserted instead.

#![expect(
    clippy::expect_used,
    reason = "the fixture components pin the expected focus and writes"
)]

mod common;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use common::interactive::{ClipboardGuard, Harness, HarnessOptions, RecordingTerminal, TestEditor};
use pi_coding_agent::core::keybindings;
use pi_coding_agent::modes::interactive::components::status_indicator::{
    StatusIndicator, StatusIndicatorKind,
};
use pi_coding_agent::modes::interactive::interactive_mode::{
    AppEditor, EditorFactory, InteractiveMode,
};
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::modes::interactive::tui_renderer::{
    InteractiveTuiOptions, create_interactive_tui,
};
use pi_coding_agent::settings_manager::Settings;
use pi_tui::components::ColorFn;
use pi_tui::components::Text;
use pi_tui::components::scroll_view::{FollowMode, ScrollView, ScrollViewOptions};
use pi_tui::components::{StackChild, StackEntryOptions, VStack};
use pi_tui::keybindings::{KeybindingsConfig, KeybindingsManager, set_keybindings};
use pi_tui::tui::{Component, Tui, TuiMode, TuiStopOptions};

fn fullscreen_terminal(width: u16, height: u16) -> RecordingTerminal {
    RecordingTerminal::new(width, height)
}

#[test]
fn selects_the_alternate_screen_renderer_only_when_requested() {
    init_theme(Some("dark"), false);
    let regular_terminal = fullscreen_terminal(80, 24);
    let regular = create_interactive_tui(InteractiveTuiOptions {
        tui_mode: TuiMode::Regular,
        show_hardware_cursor: false,
        log_directory: "/tmp".to_string(),
        terminal: Some(Box::new(regular_terminal.clone())),
        on_right_click_paste: None,
        fullscreen_copy_on_select: Some(false),
    });
    regular.tui().start();
    regular.tui().render_now(true);
    assert_eq!(regular.tui().mode(), TuiMode::Regular);
    assert!(!regular.tui().is_viewport_tui());
    assert!(!regular_terminal.writes().contains("\x1b[?1049h"));
    regular.tui().stop(TuiStopOptions::default());

    let alt_terminal = fullscreen_terminal(80, 24);
    let alt = create_interactive_tui(InteractiveTuiOptions {
        tui_mode: TuiMode::Fullscreen,
        show_hardware_cursor: false,
        log_directory: "/tmp".to_string(),
        terminal: Some(Box::new(alt_terminal.clone())),
        on_right_click_paste: None,
        fullscreen_copy_on_select: Some(false),
    });
    alt.tui().start();
    alt.tui().render_now(true);
    assert_eq!(alt.tui().mode(), TuiMode::Fullscreen);
    assert!(alt.tui().is_viewport_tui());
    assert!(alt_terminal.writes().contains("\x1b[?1049h"));
    alt.tui().stop(TuiStopOptions::default());
}

#[test]
fn shows_the_configured_jump_to_bottom_shortcut_while_scrolled_up() {
    init_theme(Some("dark"), false);
    set_keybindings(KeybindingsManager::with_user_bindings(
        keybindings::keybindings(),
        KeybindingsConfig::new().bind("tui.altScreen.bottom", ["ctrl+j"]),
    ));
    let terminal = fullscreen_terminal(50, 4);
    let ui = create_interactive_tui(InteractiveTuiOptions {
        tui_mode: TuiMode::Fullscreen,
        show_hardware_cursor: false,
        log_directory: "/tmp".to_string(),
        terminal: Some(Box::new(terminal.clone())),
        on_right_click_paste: None,
        fullscreen_copy_on_select: Some(false),
    });
    let transcript = ScrollView::new(
        Rc::new(Text::with_padding(
            (1..=8)
                .map(|index| format!("line {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
            0,
            0,
        )),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    ui.tui().set_layout_root(Some(VStack::new(
        vec![StackChild::entry(
            {
                let child: Rc<dyn Component> = transcript.clone();
                child
            },
            StackEntryOptions {
                basis: Some(pi_tui::layout_node::Basis::Cells(0)),
                grow: Some(1),
                min_size: Some(1),
                ..StackEntryOptions::default()
            },
        )],
        pi_tui::components::StackOptions::default(),
    )));
    ui.tui().start();
    ui.tui().render_now(true);
    let _ = transcript.scroll_by(-5);
    ui.tui().render_now(true);

    let writes = terminal.writes();
    assert!(writes.contains("↓ Jump to latest message"), "{writes}");
    assert!(writes.contains("Ctrl+J"), "{writes}");
    ui.tui().stop(TuiStopOptions::default());
}

/// A component whose invalidation records the TUI's mode at that moment.
struct ModeRecorder {
    tui: RefCell<Option<Rc<Tui>>>,
    modes: RefCell<Vec<TuiMode>>,
}

impl Component for ModeRecorder {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["content".to_string()]
    }

    fn invalidate(&self) {
        if let Some(tui) = self.tui.borrow().as_ref() {
            self.modes.borrow_mut().push(tui.mode());
        }
    }
}

#[tokio::test]
async fn replaces_the_renderer_and_restores_the_previous_screen_for_resume_hint_exits() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let terminal = harness.terminal.clone();
    let component = Rc::new(ModeRecorder {
        tui: RefCell::new(None),
        modes: RefCell::new(Vec::new()),
    });
    *component.tui.borrow_mut() = Some(harness.frame.tui());
    let child: Rc<dyn Component> = component.clone();
    harness.frame.tui().add_child(Rc::clone(&child));
    harness.frame.tui().set_focus(Some(Rc::clone(&child)));
    harness.frame.tui().start();

    assert!(
        harness
            .frame
            .switch_tui_mode(TuiMode::Fullscreen, false, true)
    );

    assert_eq!(harness.frame.tui().mode(), TuiMode::Fullscreen);
    assert!(
        harness
            .frame
            .tui()
            .children()
            .iter()
            .any(|mounted| Rc::ptr_eq(mounted, &child))
    );
    assert!(Rc::ptr_eq(
        &harness
            .frame
            .tui()
            .get_focused_component()
            .expect("focus survives"),
        &child
    ));
    assert_eq!(component.modes.borrow().as_slice(), [TuiMode::Fullscreen]);
    assert_eq!((terminal.start_count(), terminal.stop_count()), (2, 1));

    harness
        .frame
        .stop(pi_coding_agent::settings_manager::FullscreenExitOutput::ResumeHint);

    assert_eq!(harness.frame.tui().mode(), TuiMode::Fullscreen);
    assert_eq!((terminal.start_count(), terminal.stop_count()), (2, 2));
}

/// A focusable component that records the input it receives.
struct InputRecorder {
    inputs: RefCell<Vec<String>>,
}

impl Component for InputRecorder {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["target".to_string()]
    }

    fn handle_input(&self, data: &str) {
        self.inputs.borrow_mut().push(data.to_owned());
    }

    fn invalidate(&self) {}
}

#[tokio::test]
async fn feeds_clipboard_text_to_the_focused_component_as_a_bracketed_paste() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let clipboard = ClipboardGuard::new(Some("clipboard text"));
    let target = Rc::new(InputRecorder {
        inputs: RefCell::new(Vec::new()),
    });
    let target_child: Rc<dyn Component> = target.clone();
    harness.frame.tui().set_focus(Some(target_child));

    harness.frame.handle_right_click_paste().await;

    assert_eq!(
        target.inputs.borrow().as_slice(),
        ["\x1b[200~clipboard text\x1b[201~"]
    );
    assert_eq!(clipboard.reads(), 1);
}

fn copy_settings(copy_on_select: bool, mode: TuiMode) -> HarnessOptions {
    let mut settings = Settings::new();
    settings.insert(
        "fullscreenCopyOnSelect".to_owned(),
        serde_json::Value::Bool(copy_on_select),
    );
    HarnessOptions {
        settings,
        tui_mode: mode,
        terminal_size: (40, 4),
        ..HarnessOptions::default()
    }
}

fn seed_assistant_message() -> Vec<pi_agent_core::types::AgentMessage> {
    vec![
        common::agent_session::user_message("prompt", 1),
        pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(
            common::agent_session::assistant_message("assistant response", 2),
        )),
    ]
}

/// Mount a four-line transcript and select `alpha\nbeta` with the mouse.
fn select_first_two_lines(harness: &Harness, terminal: &RecordingTerminal) {
    let transcript: Rc<dyn Component> =
        Rc::new(Text::with_padding("alpha\nbeta\ngamma\ndelta", 0, 0));
    harness.frame.tui().set_layout_root(Some(VStack::new(
        vec![StackChild::entry(
            transcript,
            StackEntryOptions {
                grow: Some(1),
                ..StackEntryOptions::default()
            },
        )],
        pi_tui::components::StackOptions::default(),
    )));
    harness.frame.tui().start();
    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    harness.frame.tui().render_now(true);
}

#[tokio::test]
async fn copies_an_active_fullscreen_selection_when_copy_on_select_is_disabled() {
    let clipboard = ClipboardGuard::new(None);
    let harness = Harness::new(copy_settings(false, TuiMode::Fullscreen)).await;
    select_first_two_lines(&harness, &harness.terminal);
    clipboard.clear();

    harness.frame.handle_copy_command(true, true).await;
    harness.frame.tui().render_now(true);

    assert_eq!(clipboard.copied(), vec!["alpha\nbeta".to_owned()]);
    assert!(harness.terminal.writes().contains("Copied!"));
    assert!(!harness.chat_text().contains("Copied last agent message"));
}

#[tokio::test]
async fn copies_the_last_assistant_message_with_an_active_fullscreen_selection_when_copy_on_select_is_enabled()
 {
    let clipboard = ClipboardGuard::new(None);
    let harness = Harness::new(HarnessOptions {
        messages: seed_assistant_message(),
        ..copy_settings(true, TuiMode::Fullscreen)
    })
    .await;
    select_first_two_lines(&harness, &harness.terminal);
    // Copy-on-select may hand its write to the clipboard worker; let it
    // settle before clearing so the explicit copy is the only recorded write.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    clipboard.clear();

    harness.frame.handle_copy_command(true, true).await;

    let mut copied = Vec::new();
    for _ in 0..100 {
        copied = clipboard.copied();
        if !copied.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(copied, vec!["assistant response".to_owned()]);
    assert!(harness.terminal.writes().contains("Copied!"));
    assert!(!harness.chat_text().contains("Copied last agent message"));
}

#[tokio::test]
async fn flashes_copied_for_the_copy_shortcut_in_fullscreen_mode() {
    let clipboard = ClipboardGuard::new(None);
    let harness = Harness::new(HarnessOptions {
        messages: seed_assistant_message(),
        ..copy_settings(true, TuiMode::Fullscreen)
    })
    .await;
    harness.frame.tui().start();

    harness.frame.handle_copy_command(true, true).await;
    harness.frame.tui().render_now(true);

    assert_eq!(clipboard.copied(), vec!["assistant response".to_owned()]);
    assert!(harness.terminal.writes().contains("Copied!"));
    assert!(!harness.chat_text().contains("Copied last agent message"));
}

#[tokio::test]
async fn keeps_the_status_line_confirmation_for_the_copy_shortcut_in_regular_mode() {
    let clipboard = ClipboardGuard::new(None);
    let harness = Harness::new(HarnessOptions {
        messages: seed_assistant_message(),
        ..copy_settings(true, TuiMode::Regular)
    })
    .await;

    harness.frame.handle_copy_command(true, true).await;

    assert_eq!(clipboard.copied(), vec!["assistant response".to_owned()]);
    assert!(
        harness
            .chat_text()
            .contains("Copied last agent message to clipboard")
    );
}

fn indicator(
    frame: &InteractiveMode<pi_coding_agent::settings_manager::InMemorySettingsStorage>,
    kind: StatusIndicatorKind,
) -> Rc<StatusIndicator> {
    let color: ColorFn = Arc::new(|text: &str| text.to_string());
    Rc::new(StatusIndicator::new(
        kind,
        frame.tui().render_request(),
        Arc::clone(&color),
        color,
        "Status".to_string(),
        None,
    ))
}

fn custom_editor_factory(editor: Rc<TestEditor>) -> EditorFactory {
    Rc::new(move |_tui, _theme, _keybindings| {
        let editor: Rc<dyn AppEditor> = editor.clone();
        editor
    })
}

#[tokio::test]
async fn routes_every_status_through_the_editor_opt_in() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let indicators = [
        StatusIndicatorKind::Working,
        StatusIndicatorKind::Compaction,
        StatusIndicatorKind::BranchSummary,
        StatusIndicatorKind::Retry,
    ];
    for kind in indicators {
        let indicator = indicator(&harness.frame, kind);
        harness.frame.show_status_indicator(indicator);
        assert_eq!(harness.frame.active_status_kind(), Some(kind));
        assert_eq!(
            harness.frame.status_container().children.borrow().len(),
            0,
            "the default editor embeds the {kind:?} indicator"
        );
    }

    let editor = Rc::new(TestEditor::new("CUSTOM", false));
    harness
        .frame
        .set_custom_editor_component(Some(custom_editor_factory(Rc::clone(&editor))));
    for kind in indicators {
        let indicator = indicator(&harness.frame, kind);
        harness.frame.show_status_indicator(indicator);
        assert_eq!(harness.frame.active_status_kind(), Some(kind));
        assert_eq!(
            harness.frame.status_container().children.borrow().len(),
            1,
            "a custom editor without the opt-in uses the standalone row"
        );
        assert!(
            editor.indicator_writes().is_empty(),
            "the non-opted-in editor receives no indicator writes"
        );
    }
}

#[tokio::test]
async fn does_not_reserve_separate_status_height_for_an_embedded_indicator() {
    let harness = Harness::new(HarnessOptions::default()).await;
    for kind in [
        StatusIndicatorKind::Working,
        StatusIndicatorKind::Compaction,
        StatusIndicatorKind::BranchSummary,
        StatusIndicatorKind::Retry,
    ] {
        harness
            .frame
            .show_status_indicator(indicator(&harness.frame, kind));
        harness.frame.clear_status_indicator(Some(kind));
        assert_eq!(harness.frame.active_status_kind(), None);
        assert_eq!(
            harness.frame.status_container().children.borrow().len(),
            0,
            "the embedded {kind:?} indicator leaves no idle row"
        );
    }
}

#[tokio::test]
async fn uses_the_standalone_row_for_a_custom_editor_that_has_not_opted_in() {
    for (mode, expected_children) in [(TuiMode::Regular, 1), (TuiMode::Fullscreen, 0)] {
        let mut settings = Settings::new();
        settings.insert(
            "terminal".to_owned(),
            serde_json::json!({ "clearOnShrink": true }),
        );
        let harness = Harness::new(HarnessOptions {
            settings,
            tui_mode: mode,
            ..HarnessOptions::default()
        })
        .await;
        let editor = Rc::new(TestEditor::new("CUSTOM", false));
        harness
            .frame
            .set_custom_editor_component(Some(custom_editor_factory(Rc::clone(&editor))));
        harness
            .frame
            .show_status_indicator(indicator(&harness.frame, StatusIndicatorKind::Working));

        harness.frame.clear_status_indicator(None);

        assert!(
            editor.indicator_writes().is_empty(),
            "the non-opted-in editor receives no indicator writes"
        );
        assert_eq!(
            harness.frame.status_container().children.borrow().len(),
            expected_children,
            "{mode:?}: the idle row is added only in regular mode with clear-on-shrink"
        );
    }
}
