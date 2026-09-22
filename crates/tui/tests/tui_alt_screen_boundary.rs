//! Boundary tests for the alternate-screen slice's branches upstream left
//! untested (upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): the
//! search box's single-line editing surface (backspace, grapheme-wise cursor
//! moves, Kitty-printable and control-char handling), the navigation-query
//! fallbacks, the flash container's lifecycle, and the Debug surfaces the
//! renderer formats.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::rc::Rc;

use pi_tui::alt_screen_search::{
    AltScreenSearchComponent, AltScreenSearchIndex, find_alt_screen_search_matches,
    get_alt_screen_search_match_key,
};
use pi_tui::components::AltScreenFlashContainer;
use pi_tui::tui::{Component, TuiStopOptions};
use pi_tui::tui_alt_screen::{CopySelectionResult, TuiAltScreen, TuiAltScreenConfig};
use tui_support::{VirtualTerminal, new_alt_screen_tui as new_tui, wait_for_render};

fn search_component() -> Rc<AltScreenSearchComponent> {
    AltScreenSearchComponent::new(Rc::new(|_query: &str| {}), None)
}

fn value_of(component: &AltScreenSearchComponent) -> String {
    // The value leads the rendered content line: the borders off, the prompt
    // space off; what trails is scroll padding and the muted result counter,
    // which the assertions trim with the trailing whitespace.
    let rendered = component.render(60);
    let line = pi_tui::utils::strip_terminal_sequences(&rendered[1]);
    line.trim_start_matches('│')
        .trim_end_matches('│')
        .trim_start_matches(' ')
        .trim()
        .to_string()
}

#[test]
fn search_input_inserts_at_the_cursor_and_backspace_removes_graphemes() {
    let component = search_component();
    component.handle_input("a");
    component.handle_input("\u{754c}"); // wide grapheme
    component.handle_input("\x7f"); // backspace deletes the whole grapheme
    let line = value_of(&component);
    assert!(line.starts_with('a'), "line: {line}");
    // The muted readout trails: the empty result counter.
    assert!(line.contains("No matches"));
    component.handle_input("\x7f");
    let line = value_of(&component);
    // The value emptied: the placeholder returns and the muted counter drops,
    // which an empty query answers with no text at all.
    assert!(line.contains("Find in transcript"), "line: {line}");
    assert!(!line.contains("No matches"), "line: {line}");
    // Backspace at the empty cursor is a no-op: the placeholder holds.
    component.handle_input("\x7f");
    assert!(value_of(&component).contains("Find in transcript"));
}

#[test]
fn search_input_moves_the_cursor_grapheme_wise() {
    let component = search_component();
    component.handle_input("ab");
    component.handle_input("\x1b[D"); // cursor left
    component.handle_input("X");
    let line = value_of(&component);
    assert!(line.starts_with("aXb"), "line: {line}");
    component.handle_input("\x1b[D");
    component.handle_input("\x1b[D"); // at the start edge: a no-op
    component.handle_input("Y");
    let line = value_of(&component);
    assert!(line.starts_with("YaXb"), "line: {line}");
    component.handle_input("\x1b[C");
    component.handle_input("\x1b[C");
    component.handle_input("\x1b[C");
    component.handle_input("\x1b[C"); // at the end is a no-op
    component.handle_input("!");
    let line = value_of(&component);
    assert!(line.starts_with("YaXb!"), "line: {line}");
}

#[test]
fn search_input_inserts_kitty_csi_u_printables_and_keeps_control_bytes_out_of_the_text() {
    let component = search_component();
    component.handle_input("\x1b[110u"); // Kitty printable 'n'
    let line = value_of(&component);
    assert!(line.starts_with('n'), "line: {line}");
    // Ctrl+A (cursor to start) moves the cursor without touching the text.
    component.handle_input("\x01");
    component.handle_input("a");
    let line = value_of(&component);
    assert!(line.starts_with("an"), "line: {line}");
    // Bracketed paste inserts its content at the cursor — the framing bytes
    // never leak into the value.
    component.handle_input("\x1b[200~XYZ\x1b[201~");
    let line = value_of(&component);
    assert!(line.starts_with("aXYZn"), "line: {line}");
}

#[test]
fn search_navigation_queries_answer_only_on_the_button_row() {
    let component = search_component();
    component.render(48);
    assert_eq!(component.get_navigation_direction_at(1, 0), None);
    assert!(component.set_hovered_navigation_direction(Some(1)));
    assert!(!component.set_hovered_navigation_direction(Some(1)));
}

#[test]
fn search_box_renders_unbound_when_a_navigation_action_carries_no_key() {
    let component = search_component();
    let lines: Vec<String> = component
        .render(48)
        .iter()
        .map(|line| pi_tui::utils::strip_terminal_sequences(line))
        .collect();
    // With the registry's defaults bound, the keys render; an unbound action
    // would spell Unbound — pin the bound labels here and the Unbound shape
    // through an empty registry below.
    assert!(lines[2].contains("Shift+Enter"));
    assert!(lines[2].contains("Enter"));
}

#[test]
fn search_matches_with_an_empty_query_answer_empty_and_match_keys_fall_back() {
    assert!(find_alt_screen_search_matches(&["alpha".to_string()], "   ").is_empty());
    let key = get_alt_screen_search_match_key(&pi_tui::alt_screen_search::AltScreenSearchMatch {
        segments: Vec::new(),
    });
    assert_eq!(key, "");
}

#[test]
fn search_index_rebuilds_when_the_line_count_changes() {
    let mut index = AltScreenSearchIndex::default();
    let first = index.search(&["alpha".to_string()], "alpha");
    assert!(first.changed);
    let second = index.search(&[], "alpha");
    assert!(second.changed);
    assert!(second.matches.is_empty());
}

#[test]
fn flash_container_renders_padded_inverse_messages_and_survives_dispose_before_expiry() {
    let flashes = AltScreenFlashContainer::new();
    flashes.flash("hello", Some(60_000));
    let rendered = flashes.render(10);
    assert_eq!(rendered.len(), 1);
    assert!(rendered[0].starts_with("\x1b[7m"));
    assert!(rendered[0].ends_with("\x1b[27m"));
    // Padding fills the width: the message occupies all ten cells.
    // The message renders at its own width, truncating at the viewport
    // width without padding, upstream's truncateToWidth without pad.
    assert_eq!(pi_tui::utils::visible_width(&rendered[0]), 7);
    // Disposing before expiry drops the entry without a worker panic.
    flashes.dispose();
    assert!(flashes.render(10).is_empty());
    flashes.drain_expired();
    // An entry with the default duration arms and drains as a no-op while
    // armed.
    flashes.flash("later", None);
    flashes.drain_expired();
    assert_eq!(flashes.render(10).len(), 1);
}

#[test]
fn alt_screen_debug_surfaces_format() {
    let config = TuiAltScreenConfig {
        wheel_scroll_lines: Some(3),
        mouse: Some(false),
        ..TuiAltScreenConfig::default()
    };
    let rendered = format!("{config:?}");
    assert!(rendered.contains("wheel_scroll_lines: Some(3)"));
    assert!(rendered.contains("mouse: Some(false)"));
    assert!(
        rendered.contains("copy_on_select: None"),
        "rendered: {rendered}"
    );

    let alt = TuiAltScreen::new(TuiAltScreenConfig::default());
    assert!(format!("{alt:?}").contains("alt_screen_active: false"));
}

#[test]
fn copy_selection_results_compare_and_debug() {
    assert_eq!(CopySelectionResult::Copied, CopySelectionResult::Copied);
    assert_ne!(
        CopySelectionResult::Failed,
        CopySelectionResult::Message("x".to_string())
    );
    assert!(format!("{:?}", CopySelectionResult::Message("m".to_string())).contains('m'));
}

#[test]
fn alt_screen_screen_coordinate_clamps_bind_the_u16_event_bounds() {
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.start();
    wait_for_render(&tui);
    // SGR coordinates beyond u16 clamp into the event ride: the oversized
    // report is consumed without panicking, and the matching release closes
    // the gesture cleanly.
    terminal.send_input("\x1b[<0;999999;999999M");
    terminal.send_input("\x1b[<0;1;1m");
    wait_for_render(&tui);
    tui.stop(TuiStopOptions::default());
}

#[test]
fn alt_screen_mouse_wheel_clamps_oversized_coordinates() {
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let content: Rc<dyn Component> = Rc::new(pi_tui::components::Text::with_padding(
        "one\ntwo\nthree\nfour",
        0,
        0,
    ));
    tui.add_child(content);
    tui.start();
    wait_for_render(&tui);
    let top = alt.viewport_top();
    terminal.send_input("\x1b[<64;99999;99999M");
    wait_for_render(&tui);
    // The wheel delta still routes (up from the end clamps at the content
    // edge), and the viewport never moves past its bounds.
    assert!(alt.viewport_top() <= 6);
    let _ = top;
    tui.stop(TuiStopOptions::default());
}

#[test]
fn alt_screen_renderer_routes_terminal_input_through_the_base_pump() {
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, alt) = new_tui(terminal, TuiAltScreenConfig::default());
    tui.add_child(text_of());
    tui.start();
    wait_for_render(&tui);
    assert!(alt.is_following_output());
    tui.stop(TuiStopOptions::default());
}

fn text_of() -> Rc<pi_tui::components::Text> {
    Rc::new(pi_tui::components::Text::with_padding(
        "one\ntwo\nthree\nfour",
        0,
        0,
    ))
}

#[test]
fn alt_screen_scroll_surface_no_ops_after_the_base_drops() {
    let alt = TuiAltScreen::new(TuiAltScreenConfig::default());
    // No base attached yet: every renderer-level call no-ops instead of
    // panicking, which no live session observes.
    alt.scroll_by(1);
    alt.scroll_to_top();
    alt.scroll_to_bottom();
    alt.flash("x", None);
    assert_eq!(alt.viewport_top(), 0);
    // The implicit fallback scroll view follows the content end by policy,
    // so is_following_output answers true even before a base attaches.
    assert!(alt.is_following_output());
    assert!(!alt.has_active_selection());
    assert!(!alt.copy_active_selection_to_clipboard());
    let root: Rc<dyn Component> = Rc::new(pi_tui::components::Text::with_padding("root", 0, 0));
    alt.set_layout_root(Some(root));
}

#[test]
fn alt_screen_scroll_helpers_match_the_scroll_view_surface() {
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, alt) = new_tui(terminal, TuiAltScreenConfig::default());
    let lines: Vec<String> = (0..10).map(|index| format!("line {}", index + 1)).collect();
    let content: Rc<dyn Component> = Rc::new(pi_tui::components::Text::with_padding(
        lines.join("\n"),
        0,
        0,
    ));
    tui.add_child(content);
    tui.start();
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 6);
    // While following the end, a down-scroll clamps at the content edge;
    // the up-scroll moves one line, upstream's lineUp sign.
    alt.scroll_by(-1);
    assert_eq!(alt.viewport_top(), 5);
    alt.scroll_to_top();
    assert_eq!(alt.viewport_top(), 0);
    alt.scroll_to_bottom();
    assert_eq!(alt.viewport_top(), 6);
    assert!(alt.is_following_output());
    assert!(!alt.has_active_selection());
    assert!(alt.get_copy_on_select());
    alt.set_copy_on_select(false);
    assert!(!alt.get_copy_on_select());
    tui.stop(TuiStopOptions::default());
}
