//! `layout.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): the constrained layout
//! engine — stack allocation, nested minimums, gap omission, scroll-view
//! clipping and follow-end, scrollbar geometry and styling, and the
//! per-frame geometry rebuild.
//!
//! Restatement: the "paints only clipped rows from very large scroll
//! content" fixture builds its million-row content as a real `Vec` at a
//! tenth of the upstream sparse-array size — the paint loop's clip check
//! is what the test pins, and a sparse JS array has no Rust counterpart.

#![expect(
    clippy::expect_used,
    clippy::too_many_lines,
    reason = "test fixtures fail loudly when the engine misbehaves; expecting keeps the failure modes readable, and the long scrollbar suite mirrors its upstream test block for block"
)]

use std::cell::Cell;
use std::rc::Rc;

use pi_tui::components::{
    FollowMode, HStack, ScrollView, ScrollViewOptions, ScrollViewScrollbar, StackChild,
    StackEntryOptions, Text, VStack,
};
use pi_tui::layout::{
    LayoutRect, get_layout_boxes_at, get_scroll_view_box, get_scroll_views_at, render_layout_frame,
};
use pi_tui::layout_node::Basis;
use pi_tui::terminal_image::{
    EncodeKittyOptions, KittyImageMetadata, encode_kitty, register_kitty_image_metadata,
};
use pi_tui::tui::{Component, RenderRequest};
use pi_tui::utils::strip_terminal_sequences;

/// Upstream's `() => {}` render request.
fn noop_render_request() -> RenderRequest {
    std::sync::Arc::new(|| {})
}

fn visible_lines(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| strip_terminal_sequences(line).trim_end().to_string())
        .collect()
}

/// Coerce a concrete component into the dyn handle, upstream's structural
/// typing.
fn comp(component: Rc<impl Component + 'static>) -> Rc<dyn Component> {
    component
}

/// The test's `Text("...", 0, 0)`: no padding.
fn text(content: &str) -> Rc<Text> {
    Rc::new(Text::with_padding(content, 0, 0))
}

#[test]
fn allocates_vertical_grow_space_deterministically() {
    let frame = render_layout_frame(
        &comp(VStack::new(
            vec![
                StackChild::entry(
                    text("top"),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(1)),
                        shrink: Some(0),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::entry(
                    text("body"),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(0)),
                        grow: Some(1),
                        ..StackEntryOptions::default()
                    },
                ),
            ],
            pi_tui::components::StackOptions::default(),
        )),
        10,
        4,
        &noop_render_request(),
    );

    let heights: Vec<i64> = frame
        .root
        .children
        .iter()
        .map(|child| child.rect.height)
        .collect();
    assert_eq!(heights, [1, 3]);
    assert_eq!(visible_lines(&frame.lines), ["top", "body", "", ""]);
}

#[test]
fn does_not_render_fixed_basis_scroll_content_during_stack_measurement() {
    let render_count = Rc::new(Cell::new(0));
    let counter = Rc::clone(&render_count);
    let transcript = ScrollView::new(
        Rc::new(CountingContent {
            render_count: counter,
            lines: vec!["one".to_string(), "two".to_string(), "three".to_string()],
        }),
        ScrollViewOptions::default(),
    );
    let root = VStack::new(
        vec![
            StackChild::entry(
                comp(Rc::clone(&transcript)),
                StackEntryOptions {
                    basis: Some(Basis::Cells(0)),
                    grow: Some(1),
                    ..StackEntryOptions::default()
                },
            ),
            StackChild::component(text("dock")),
        ],
        pi_tui::components::StackOptions::default(),
    );
    let _ = render_layout_frame(&comp(Rc::clone(&root)), 10, 3, &noop_render_request());

    assert_eq!(render_count.get(), 1);
}

#[test]
fn paints_only_clipped_rows_from_very_large_scroll_content() {
    // Upstream builds a sparse billion-row array; the Rust fixture renders
    // a tall-but-materializable content whose visible window sits at the
    // tail — the pinned behavior is the paint loop's clip check.
    let line_count = 1_000_000;
    let mut lines: Vec<String> = vec![String::new(); line_count];
    lines[line_count - 4] = "before".to_string();
    lines[line_count - 3] = "visible 1".to_string();
    lines[line_count - 2] = "visible 2".to_string();
    lines[line_count - 1] = "visible 3".to_string();
    let transcript = ScrollView::new(
        Rc::new(CountingContent {
            render_count: Rc::new(Cell::new(0)),
            lines,
        }),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            ..ScrollViewOptions::default()
        },
    );

    let frame = render_layout_frame(&comp(Rc::clone(&transcript)), 10, 3, &noop_render_request());
    assert_eq!(
        visible_lines(&frame.lines),
        ["visible 1", "visible 2", "visible 3"]
    );
}

#[test]
fn shrinks_entries_to_their_minimum_sizes() {
    let frame = render_layout_frame(
        &comp(VStack::new(
            vec![
                StackChild::entry(
                    text("a1\na2\na3"),
                    StackEntryOptions {
                        min_size: Some(1),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::entry(
                    text("b1\nb2\nb3"),
                    StackEntryOptions {
                        shrink: Some(0),
                        ..StackEntryOptions::default()
                    },
                ),
            ],
            pi_tui::components::StackOptions::default(),
        )),
        10,
        4,
        &noop_render_request(),
    );

    let heights: Vec<i64> = frame
        .root
        .children
        .iter()
        .map(|child| child.rect.height)
        .collect();
    assert_eq!(heights, [1, 3]);
    assert_eq!(visible_lines(&frame.lines), ["a1", "b1", "b2", "b3"]);
}

#[test]
fn includes_nested_minimum_sizes_in_intrinsic_stack_measurement() {
    let dock = VStack::new(
        vec![
            StackChild::component(text("top1\ntop2\ntop3")),
            StackChild::entry(
                text("selector"),
                StackEntryOptions {
                    min_size: Some(3),
                    ..StackEntryOptions::default()
                },
            ),
            StackChild::component(text("below")),
            StackChild::entry(
                text("footer"),
                StackEntryOptions {
                    min_size: Some(1),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        pi_tui::components::StackOptions::default(),
    );
    let frame = render_layout_frame(
        &comp(VStack::new(
            vec![
                StackChild::entry(
                    text("body"),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(0)),
                        grow: Some(1),
                        min_size: Some(1),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::entry(
                    comp(Rc::clone(&dock)),
                    StackEntryOptions {
                        basis: Some(Basis::Auto),
                        min_size: Some(1),
                        ..StackEntryOptions::default()
                    },
                ),
            ],
            pi_tui::components::StackOptions::default(),
        )),
        10,
        9,
        &noop_render_request(),
    );

    assert_eq!(
        visible_lines(&frame.lines),
        [
            "body", "top1", "top2", "top3", "selector", "", "", "below", "footer"
        ]
    );
}

#[test]
fn omits_gaps_around_invisible_entries() {
    let stack = VStack::new(
        vec![
            StackChild::component(text("one")),
            StackChild::entry(
                text("hidden"),
                StackEntryOptions {
                    visible: Some(Rc::new(|_| false)),
                    ..StackEntryOptions::default()
                },
            ),
            StackChild::component(text("two")),
        ],
        pi_tui::components::StackOptions {
            gap: Some(1),
            ..pi_tui::components::StackOptions::default()
        },
    );
    let rendered: Vec<String> = stack
        .render(10)
        .iter()
        .map(|line| line.trim_end().to_string())
        .collect();
    assert_eq!(rendered, ["one", "", "two"]);
}

#[test]
fn crops_kitty_images_at_a_scroll_view_s_lower_boundary() {
    let image_id = 124;
    let image_line = encode_kitty(
        "AAAA",
        EncodeKittyOptions {
            columns: Some(2),
            rows: Some(3),
            image_id: Some(image_id),
            move_cursor: Some(false),
        },
    );
    register_kitty_image_metadata(KittyImageMetadata {
        image_id,
        columns: 2,
        rows: 3,
        width_px: 100,
        height_px: 100,
    });
    let transcript = ScrollView::new(
        Rc::new(CountingContent {
            render_count: Rc::new(Cell::new(0)),
            lines: vec![
                "one".to_string(),
                "two".to_string(),
                image_line,
                String::new(),
                String::new(),
            ],
        }),
        ScrollViewOptions::default(),
    );
    let frame = render_layout_frame(
        &comp(VStack::new(
            vec![
                StackChild::entry(
                    comp(Rc::clone(&transcript)),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(0)),
                        grow: Some(1),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::component(text("dock")),
            ],
            pi_tui::components::StackOptions::default(),
        )),
        20,
        4,
        &noop_render_request(),
    );

    assert!(
        frame.lines[2].contains("y=0,h=34,r=1"),
        "expected the cropped placement, got {:?}",
        frame.lines[2]
    );
}

#[test]
fn composes_horizontal_children_at_allocated_widths() {
    let frame = render_layout_frame(
        &comp(HStack::new(
            vec![
                StackChild::entry(
                    text("left"),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(6)),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::entry(
                    text("right"),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(6)),
                        ..StackEntryOptions::default()
                    },
                ),
            ],
            pi_tui::components::StackOptions::default(),
        )),
        12,
        1,
        &noop_render_request(),
    );
    assert_eq!(visible_lines(&frame.lines), ["left  right"]);
}

#[test]

fn does_not_paint_zero_width_horizontal_children() {
    let frame = render_layout_frame(
        &comp(HStack::new(
            vec![
                StackChild::entry(
                    text("hidden"),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(0)),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::entry(
                    text("shown"),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(0)),
                        grow: Some(1),
                        ..StackEntryOptions::default()
                    },
                ),
            ],
            pi_tui::components::StackOptions::default(),
        )),
        5,
        1,
        &noop_render_request(),
    );
    assert_eq!(visible_lines(&frame.lines), ["shown"]);
}

#[test]
fn tracks_follow_end_state_and_returns_unused_scroll_delta() {
    let scroll_view = ScrollView::new(
        text("1\n2\n3\n4\n5\n6"),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    let _ = render_layout_frame(
        &comp(Rc::clone(&scroll_view)),
        10,
        3,
        &noop_render_request(),
    );

    assert_eq!(scroll_view.scroll_top(), 3);
    assert!(scroll_view.is_following_end());

    assert_eq!(scroll_view.scroll_by(-2), 0);
    assert_eq!(scroll_view.scroll_top(), 1);
    assert!(!scroll_view.is_following_end());
    assert_eq!(scroll_view.scroll_by(-3), -2);
    assert_eq!(scroll_view.scroll_top(), 0);
    assert_eq!(scroll_view.scroll_by(10), 7);
    assert_eq!(scroll_view.scroll_top(), 3);
    assert!(scroll_view.is_following_end());
}

#[test]
fn renders_a_proportional_glyph_scrollbar_with_an_expanded_active_thumb() {
    let source_lines: Vec<String> = [
        "abcd界", "abcde2", "abcde3", "abcde4", "abcde5", "abcde6", "abcde7", "abcde8",
    ]
    .iter()
    .map(|line| (*line).to_string())
    .collect();
    let content_background = "\x1b[42m";
    let track_color = "\x1b[38;5;2m";
    let thumb_color = "\x1b[38;5;1m";
    let track_color_for: pi_tui::components::ColorFn = {
        let track_color = track_color.to_string();
        std::sync::Arc::new(move |text: &str| format!("{track_color}{text}\x1b[39m"))
    };
    let thumb_color_for: pi_tui::components::ColorFn = {
        let thumb_color = thumb_color.to_string();
        std::sync::Arc::new(move |text: &str| format!("{thumb_color}{text}\x1b[39m"))
    };
    let content = Text::with_background(
        source_lines.join("\n"),
        0,
        0,
        Some({
            let content_background = content_background.to_string();
            std::sync::Arc::new(move |text: &str| format!("{content_background}{text}\x1b[49m"))
        }),
    );
    let scroll_view = ScrollView::new(
        comp(Rc::new(content)),
        ScrollViewOptions {
            scrollbar: Some(ScrollViewScrollbar::Auto),
            scrollbar_track_style: Some(track_color_for.clone()),
            scrollbar_thumb_style: Some(thumb_color_for.clone()),
            scrollbar_hide_delay_ms: Some(10),
            ..ScrollViewOptions::default()
        },
    );
    let render =
        || render_layout_frame(&comp(Rc::clone(&scroll_view)), 6, 4, &noop_render_request()).lines;
    let visible = |lines: &[String]| -> Vec<String> {
        lines
            .iter()
            .map(|line| strip_terminal_sequences(line))
            .collect()
    };

    let lines = render();
    assert_eq!(visible(&lines), source_lines[..4]);

    let _ = scroll_view.scroll_by(2);
    let lines = render();
    assert_eq!(visible(&lines), ["abcde│", "abcde┃", "abcde┃", "abcde│"]);
    let has_track: Vec<bool> = lines
        .iter()
        .map(|line| line.contains(track_color))
        .collect();
    assert_eq!(has_track, [true, false, false, true]);
    let has_thumb: Vec<bool> = lines
        .iter()
        .map(|line| line.contains(thumb_color))
        .collect();
    assert_eq!(has_thumb, [false, true, true, false]);

    scroll_view.set_scrollbar_active(true);
    let lines = render();
    assert_eq!(visible(&lines), ["abcde│", "abcde█", "abcde█", "abcde│"]);
    let has_thumb: Vec<bool> = lines
        .iter()
        .map(|line| line.contains(thumb_color))
        .collect();
    assert_eq!(has_thumb, [false, true, true, false]);
    assert!(
        lines[1]
            .rfind(content_background)
            .expect("content background")
            < lines[1].rfind(thumb_color).expect("thumb color"),
        "the thumb style lands after the content background"
    );

    scroll_view.set_scrollbar_active(false);
    std::thread::sleep(std::time::Duration::from_millis(60));
    let lines = render();
    assert_eq!(visible(&lines), source_lines[2..6]);

    scroll_view.scroll_to_end();
    let lines = render();
    assert_eq!(visible(&lines), ["abcde│", "abcde│", "abcde┃", "abcde┃"]);

    scroll_view.scroll_to_start();
    let lines = render();
    assert_eq!(visible(&lines)[0], "abcd ┃");

    let followed_content = text(&source_lines.join("\n"));
    let followed = ScrollView::new(
        comp(Rc::clone(&followed_content)),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            scrollbar: Some(ScrollViewScrollbar::Auto),
            scrollbar_track_style: Some(track_color_for.clone()),
            scrollbar_thumb_style: Some(thumb_color_for.clone()),
            ..ScrollViewOptions::default()
        },
    );
    let _ = render_layout_frame(&comp(Rc::clone(&followed)), 6, 4, &noop_render_request());

    assert_eq!(followed.scroll_top(), 4);
    followed_content.set_text(&format!("{}\nabcde9", source_lines.join("\n")));
    let growth_frame =
        render_layout_frame(&comp(Rc::clone(&followed)), 6, 4, &noop_render_request());
    assert_eq!(followed.scroll_top(), 5);
    assert!(
        growth_frame
            .lines
            .iter()
            .all(|line| !strip_terminal_sequences(line).contains(['│', '┃'])),
        "a growing followed view hides the transient scrollbar"
    );

    let automatic = ScrollView::new(
        comp(text("1\n2")),
        ScrollViewOptions {
            scrollbar: Some(ScrollViewScrollbar::Auto),
            scrollbar_thumb_style: Some(thumb_color_for.clone()),
            ..ScrollViewOptions::default()
        },
    );
    let _ = render_layout_frame(&comp(Rc::clone(&automatic)), 6, 4, &noop_render_request());
    let _ = automatic.scroll_by(1);
    assert!(
        render_layout_frame(&comp(Rc::clone(&automatic)), 6, 4, &noop_render_request())
            .lines
            .iter()
            .all(|line| !strip_terminal_sequences(line).contains(['│', '┃'])),
        "a fitting auto view never shows the scrollbar"
    );

    let always_fitting = ScrollView::new(
        comp(text("1\n2")),
        ScrollViewOptions {
            scrollbar: Some(ScrollViewScrollbar::Always),
            scrollbar_thumb_style: Some(thumb_color_for.clone()),
            ..ScrollViewOptions::default()
        },
    );
    let always_fitting_frame = render_layout_frame(
        &comp(Rc::clone(&always_fitting)),
        6,
        4,
        &noop_render_request(),
    );
    assert_eq!(always_fitting_frame.root.children[0].rect.width, 5);
    assert!(
        visible(&always_fitting_frame.lines)
            .iter()
            .all(|line| line.ends_with('┃')),
        "an always-reserved fitting view paints the full thumb"
    );

    let content = Text::with_background(
        source_lines.join("\n"),
        0,
        0,
        Some({
            let content_background = content_background.to_string();
            std::sync::Arc::new(move |text: &str| format!("{content_background}{text}\x1b[49m"))
        }),
    );
    let always_overflowing = ScrollView::new(
        comp(Rc::new(content)),
        ScrollViewOptions {
            scrollbar: Some(ScrollViewScrollbar::Always),
            scrollbar_track_style: Some(track_color_for.clone()),
            scrollbar_thumb_style: Some(thumb_color_for.clone()),
            ..ScrollViewOptions::default()
        },
    );
    let always_overflowing_frame = render_layout_frame(
        &comp(Rc::clone(&always_overflowing)),
        6,
        4,
        &noop_render_request(),
    );
    assert_eq!(always_overflowing_frame.root.children[0].rect.width, 5);
    assert_eq!(
        visible(&always_overflowing_frame.lines)
            .iter()
            .filter(|line| line.ends_with('┃'))
            .count(),
        2
    );
    assert_eq!(
        visible(&always_overflowing_frame.lines)
            .iter()
            .filter(|line| line.ends_with('│'))
            .count(),
        2
    );
    for line in &always_overflowing_frame.lines {
        let scrollbar_style_index = line
            .rfind(track_color)
            .unwrap_or(0)
            .max(line.rfind(thumb_color).unwrap_or(0));
        let reserved_column_reset_index = line[..scrollbar_style_index]
            .rfind("\x1b[0m\x1b]8;;\x07")
            .expect("the reserved column resets before the scrollbar style");
        assert!(
            reserved_column_reset_index
                > line.rfind(content_background).expect("content background"),
            "the reset sits after the content background"
        );
    }

    let thumb_height_for = |content_height: usize| {
        let sized = ScrollView::new(
            text(&vec!["x"; content_height].join("\n")),
            ScrollViewOptions {
                scrollbar: Some(ScrollViewScrollbar::Auto),
                scrollbar_thumb_style: Some(thumb_color_for.clone()),
                ..ScrollViewOptions::default()
            },
        );
        let _ = render_layout_frame(&comp(Rc::clone(&sized)), 6, 20, &noop_render_request());
        let _ = sized.scroll_by(1);
        render_layout_frame(&comp(Rc::clone(&sized)), 6, 20, &noop_render_request())
            .lines
            .iter()
            .filter(|line| strip_terminal_sequences(line).ends_with('┃'))
            .count()
    };
    assert_eq!(thumb_height_for(21), 19);
    assert_eq!(thumb_height_for(40), 10);
    assert_eq!(thumb_height_for(100), 4);
    assert_eq!(thumb_height_for(400), 2);
}

#[test]
fn preserves_only_the_underlying_background_beneath_overlay_scrollbar_glyphs() {
    let background = "\x1b[42m";
    let content_lines: Vec<String> = (0..8)
        .map(|_| format!("{background}{}│\x1b[39m\x1b[49m", "x".repeat(5)))
        .collect();
    let content = CountingContent {
        render_count: Rc::new(Cell::new(0)),
        lines: content_lines,
    };
    let scroll_view = ScrollView::new(
        Rc::new(content),
        ScrollViewOptions {
            scrollbar: Some(ScrollViewScrollbar::Auto),
            scrollbar_track_style: Some(std::sync::Arc::new(ToString::to_string)),
            scrollbar_thumb_style: Some(std::sync::Arc::new(ToString::to_string)),
            ..ScrollViewOptions::default()
        },
    );
    let _ = render_layout_frame(&comp(Rc::clone(&scroll_view)), 6, 4, &noop_render_request());
    let _ = scroll_view.scroll_by(1);
    let frame = render_layout_frame(&comp(Rc::clone(&scroll_view)), 6, 4, &noop_render_request());

    let visible: Vec<String> = frame
        .lines
        .iter()
        .map(|line| strip_terminal_sequences(line))
        .collect();
    assert_eq!(visible, ["xxxxx│", "xxxxx┃", "xxxxx┃", "xxxxx│"]);
    for line in &frame.lines {
        assert!(line.contains(background));
        assert!(
            !line.contains("\x1b[31m"),
            "the border foreground never leaks"
        );
        assert!(
            line.contains("\x1b[0m\x1b]8;;\x07\x1b[42m"),
            "the background is re-emitted"
        );
    }
}

#[test]
fn updates_reserved_scrollbar_layout_at_runtime() {
    let scroll_view = ScrollView::new(
        text("123456"),
        ScrollViewOptions {
            scrollbar: Some(ScrollViewScrollbar::Always),
            ..ScrollViewOptions::default()
        },
    );
    let render = || {
        render_layout_frame(
            &comp(HStack::new(
                vec![StackChild::component(comp(Rc::clone(&scroll_view)))],
                pi_tui::components::StackOptions {
                    align: Some(pi_tui::layout_node::StackAlign::Start),
                    ..pi_tui::components::StackOptions::default()
                },
            )),
            6,
            2,
            &noop_render_request(),
        )
    };
    let always = render();
    assert_eq!(visible_lines(&always.lines), ["12345┃", "6    ┃"]);
    assert_eq!(always.root.children[0].rect.width, 6);
    assert_eq!(always.root.children[0].children[0].rect.width, 5);

    scroll_view.set_scrollbar(ScrollViewScrollbar::Hidden);
    assert_eq!(render().root.children[0].children[0].rect.width, 6);
    assert!(!scroll_view.is_scrollbar_visible());
}

#[test]
fn measures_nested_scroll_content_from_constrained_child_geometry() {
    let inner = ScrollView::new(text("1\n2\n3\n4\n5\n6"), ScrollViewOptions::default());
    let outer = ScrollView::new(
        comp(VStack::new(
            vec![
                StackChild::entry(
                    comp(Rc::clone(&inner)),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(2)),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::component(text("tail")),
            ],
            pi_tui::components::StackOptions::default(),
        )),
        ScrollViewOptions::default(),
    );
    let _ = render_layout_frame(&comp(Rc::clone(&outer)), 10, 2, &noop_render_request());

    assert_eq!(inner.viewport_height(), 2);
    assert_eq!(outer.scroll_by(10), 9);
    assert_eq!(outer.scroll_top(), 1);
}

#[test]
fn rebuilds_geometry_after_content_changes() {
    let content = text("one");
    let root = VStack::new(
        vec![StackChild::component(comp(Rc::clone(&content)))],
        pi_tui::components::StackOptions::default(),
    );
    let first = render_layout_frame(&comp(Rc::clone(&root)), 10, 4, &noop_render_request());
    content.set_text("one\ntwo\nthree");
    let second = render_layout_frame(&comp(Rc::clone(&root)), 10, 4, &noop_render_request());

    assert_eq!(first.root.children[0].lines.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        second.root.children[0].lines.as_ref().map(Vec::len),
        Some(3)
    );
}

// === fixtures ===

/// A fixed-content leaf with a render counter, restating the suite's
/// inline object components and the `render`-counting transcript.
struct CountingContent {
    render_count: Rc<Cell<usize>>,
    lines: Vec<String>,
}

impl Component for CountingContent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.render_count.set(self.render_count.get() + 1);
        self.lines.clone()
    }

    fn invalidate(&self) {}
}

/// Keep the frame-consumer surface exercised beside the suites.
#[test]
fn frame_consumers_surface_the_box_tree() {
    let scroll_view = ScrollView::new(text("1\n2\n3\n4\n5\n6"), ScrollViewOptions::default());
    let frame = render_layout_frame(
        &comp(Rc::clone(&scroll_view)),
        10,
        3,
        &noop_render_request(),
    );
    // Hit path: the deepest box first.
    let boxes = get_layout_boxes_at(&frame, 0, 0);
    assert!(!boxes.is_empty());
    // Scroll view lookup by state handle.
    let primary = frame
        .primary_scroll_view
        .as_ref()
        .expect("the sole scroll view is primary");
    let box_ = get_scroll_view_box(&frame, primary).expect("the scroll box exists");
    assert_eq!(box_.rect.width, 10);
    assert_eq!(get_scroll_views_at(&frame, 0, 0).len(), 1);
    assert!(get_scroll_views_at(&frame, 50, 50).is_empty());
    // The scrollbar geometry surface answers None for a hidden scrollbar.
    assert!(pi_tui::layout::get_scrollbar_geometry(box_, false).is_none());
    let _ = LayoutRect::default();
}
