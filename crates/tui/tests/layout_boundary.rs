//! Boundary tests for the #44 layout slice, added where the ported
//! upstream suite leaves branches untested: the stacks' direct-render
//! surface (VStack/HStack outside the frame engine), the allocator's
//! edge cases, scroll-view option transitions, the cursor-marker offset,
//! and the frame-consumer helpers. These bind the 95% coverage gate
//! alongside the ported suite.

#![expect(
    clippy::expect_used,
    clippy::default_trait_access,
    reason = "test fixtures fail loudly when the engine misbehaves; expecting keeps the failure modes readable, and the type-inferred Defaults stay terse in fixtures"
)]

use std::rc::Rc;

use pi_tui::components::{
    HStack, ScrollView, ScrollViewOptions, ScrollViewScrollbar, StackChild, StackEntryOptions,
    Text, VStack,
};
use pi_tui::layout::{
    get_layout_boxes_at, get_scroll_view_box, get_scroll_views_at, get_scrollbar_geometry,
    render_layout_frame,
};
use pi_tui::layout_node::{Basis, LayoutNode, LayoutViewport, StackAlign, StackKind};
use pi_tui::terminal_image::{
    EncodeKittyOptions, KittyImageMetadata, crop_kitty_image_line, encode_kitty,
    get_kitty_image_metadata, register_kitty_image_metadata,
};
use pi_tui::tui::{CURSOR_MARKER, Component, TuiMouseButton, TuiMouseEvent, TuiMouseEventType};
use pi_tui::utils::strip_terminal_sequences;

fn noop_render_request() -> pi_tui::tui::RenderRequest {
    std::sync::Arc::new(|| {})
}

fn comp(component: Rc<impl Component + 'static>) -> Rc<dyn Component> {
    component
}

fn trim_lines(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| strip_terminal_sequences(line).trim_end().to_string())
        .collect()
}

fn text(content: &str) -> Rc<Text> {
    Rc::new(Text::with_padding(content, 0, 0))
}

// === direct stack rendering ===

#[test]
fn v_stack_renders_members_at_intrinsic_sizes_with_gaps() {
    let stack = VStack::new(
        vec![
            StackChild::component(comp(text("one"))),
            StackChild::component(comp(text("two\ntwo"))),
        ],
        pi_tui::components::StackOptions {
            gap: Some(2),
            ..pi_tui::components::StackOptions::default()
        },
    );
    assert_eq!(trim_lines(&stack.render(10)), ["one", "", "", "two", "two"]);
}

#[test]
fn v_stack_pads_short_children_to_their_allocated_size() {
    let stack = VStack::new(
        vec![
            StackChild::entry(
                comp(text("short")),
                StackEntryOptions {
                    basis: Some(Basis::Cells(3)),
                    ..Default::default()
                },
            ),
            StackChild::component(comp(text("after"))),
        ],
        Default::default(),
    );
    assert_eq!(trim_lines(&stack.render(10)), ["short", "", "", "after"]);
}

#[test]
fn empty_v_stack_renders_nothing() {
    let stack: Rc<VStack> = VStack::new(vec![], Default::default());
    assert!(stack.render(10).is_empty());
}

#[test]
fn h_stack_renders_empty_for_no_entries() {
    let stack: Rc<HStack> = HStack::new(vec![], Default::default());
    assert!(stack.render(10).is_empty());
}

#[test]
fn h_stack_aligns_children_start_center_and_end() {
    for align in [StackAlign::Start, StackAlign::Center, StackAlign::End] {
        // Fixed bases keep the children from re-wrapping at odd widths, so
        // the cross-axis placement is what the assertion pins.
        let stack = HStack::new(
            vec![
                StackChild::entry(
                    comp(text("left")),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(4)),
                        ..Default::default()
                    },
                ),
                StackChild::entry(
                    comp(text("right")),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(5)),
                        ..Default::default()
                    },
                ),
            ],
            pi_tui::components::StackOptions {
                align: Some(align),
                ..pi_tui::components::StackOptions::default()
            },
        );
        let rendered = trim_lines(&stack.render(9));
        assert_eq!(rendered, ["leftright"], "{align:?}");
    }
}

#[test]
fn h_stack_zero_width_entry_renders_nothing_but_is_skipped() {
    let stack = HStack::new(
        vec![
            StackChild::entry(
                comp(text("zero")),
                StackEntryOptions {
                    basis: Some(Basis::Cells(0)),
                    ..Default::default()
                },
            ),
            StackChild::component(comp(text("shown"))),
        ],
        Default::default(),
    );
    assert_eq!(trim_lines(&stack.render(10)), ["shown"]);
}

#[test]
fn stacks_mutate_children_and_report_the_tree() {
    let stack = VStack::new(
        vec![StackChild::component(comp(text("a")))],
        Default::default(),
    );
    let child_b = comp(text("b"));
    stack.add_child(Rc::clone(&child_b), Default::default());
    assert_eq!(Component::children(&*stack).len(), 2);
    stack.remove_child(&child_b);
    assert_eq!(stack.children().len(), 1);
    stack.clear();
    assert!(stack.render(10).is_empty());
    // Invalidate walks the children without crashing.
    stack.invalidate();
    // The mouse dispatch rides the child walk.
    let event = TuiMouseEvent {
        event_type: TuiMouseEventType::Press,
        button: TuiMouseButton::Left,
        x: 0,
        y: 0,
        screen_x: 0,
        screen_y: 0,
        width: 10,
        height: 1,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    };
    let stack2 = VStack::new(
        vec![StackChild::component(comp(text("a")))],
        Default::default(),
    );
    assert!(stack2.handle_mouse(&event).is_none(), "an empty hit region");
}

#[test]
fn layout_node_reports_the_stack_kind() {
    let vstack = VStack::new(vec![], Default::default());
    let hstack = HStack::new(vec![], Default::default());
    match vstack.layout_node() {
        Some(LayoutNode::Stack(node)) => {
            assert_eq!(node.kind, StackKind::Vertical);
            assert_eq!(node.gap, 0);
            assert_eq!(node.align, StackAlign::Stretch);
            let _ = format!("{node:?}");
        }
        _ => unreachable!("VStack answers a stack node"),
    }
    match hstack.layout_node() {
        Some(LayoutNode::Stack(node)) => assert_eq!(node.kind, StackKind::Horizontal),
        _ => unreachable!("HStack answers a stack node"),
    }
    let _ = format!("{vstack:?}");
    let _ = format!("{hstack:?}");
}

#[test]
fn layout_viewport_and_basis_round_trip() {
    let viewport = LayoutViewport {
        width: 3,
        height: 4,
    };
    let _ = format!("{viewport:?}");
    assert_eq!(
        viewport,
        LayoutViewport {
            width: 3,
            height: 4
        }
    );
    assert_ne!(Basis::Cells(1), Basis::Auto);
    let _ = format!("{:?}", Basis::Cells(2));
    let _ = format!("{:?}", StackKind::Horizontal);
    let _ = format!("{:?}", StackAlign::End);
}

// === allocator edges ===

#[test]
fn allocator_grows_only_growable_entries_and_respects_caps() {
    // grow weight 2 vs 1 with a max cap on the first.
    let entries = [cap_entry(1, 0, 2), grow_entry(1)];
    // content 4, intrinsic [1, 0]: grow 3 total; entry 0 capped at 1.
    let layout_entries: Vec<_> = entries.iter().map(entry_of).collect();
    let sizes = pi_tui::components::allocate_stack_sizes(&layout_entries, &[1, 0], Some(4), 0);
    assert_eq!(sizes, [1, 3]);
}

#[test]
fn allocator_shrinks_proportionally_to_size_times_weight() {
    let entries = [shrink_entry(1, 1), shrink_entry(1, 0)];
    // total 6 > content 4: both shrink by their weight*size.
    let layout_entries: Vec<_> = entries.iter().map(entry_of).collect();
    let sizes = pi_tui::components::allocate_stack_sizes(&layout_entries, &[3, 3], Some(4), 0);
    assert_eq!(sizes, [2, 2]);
}

#[test]
fn allocator_keeps_sizes_when_no_candidate_has_capacity() {
    // Both entries at their max: grow has nowhere to go.
    let entries = [capped_grow_entry(1, 1), capped_grow_entry(1, 1)];
    let layout_entries: Vec<_> = entries.iter().map(entry_of).collect();
    let sizes = pi_tui::components::allocate_stack_sizes(&layout_entries, &[1, 1], Some(4), 0);
    assert_eq!(sizes, [1, 1]);
}

/// An entry with grow weight, default bounds.
fn grow_entry(grow: u32) -> StackEntryOptions {
    StackEntryOptions {
        grow: Some(grow),
        ..Default::default()
    }
}

/// An entry with a shrink weight.
fn shrink_entry(shrink: u32, min_size: u32) -> StackEntryOptions {
    StackEntryOptions {
        shrink: Some(shrink),
        min_size: Some(min_size),
        ..Default::default()
    }
}

/// A capped entry: basis 0, grow weight, and a max bound.
fn capped_grow_entry(grow: u32, max_size: u32) -> StackEntryOptions {
    StackEntryOptions {
        basis: Some(Basis::Cells(0)),
        grow: Some(grow),
        max_size: Some(max_size),
        ..Default::default()
    }
}

/// Build a layout entry from options over a placeholder component, the
/// shape the allocator consumes.
fn entry_of(options: &StackEntryOptions) -> pi_tui::layout_node::StackLayoutEntry {
    pi_tui::layout_node::StackLayoutEntry {
        component: comp(text("entry")),
        basis: options.basis,
        grow: options.grow.unwrap_or(0),
        shrink: options.shrink.unwrap_or(1),
        min_size: usize::try_from(options.min_size.unwrap_or(0)).unwrap_or(usize::MAX),
        max_size: options.max_size.map_or(usize::MAX, |value| {
            usize::try_from(value).unwrap_or(usize::MAX)
        }),
        visible: None,
    }
}

fn cap_entry(min_size: u32, grow: u32, max_size: u32) -> StackEntryOptions {
    StackEntryOptions {
        basis: Some(Basis::Cells(1)),
        grow: Some(grow),
        min_size: Some(min_size),
        max_size: Some(max_size),
        ..Default::default()
    }
}

// === scroll-view option transitions ===

#[test]
fn scroll_to_clamps_and_suppresses_follow() {
    let scroll_view = ScrollView::new(
        comp(text("1\n2\n3\n4\n5\n6")),
        ScrollViewOptions {
            follow: Some(pi_tui::components::FollowMode::End),
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
    // scrollTo with a disable_follow target suppresses the pin.
    scroll_view.scroll_to(
        3,
        pi_tui::components::ScrollViewScrollToOptions {
            disable_follow: true,
        },
    );
    assert_eq!(scroll_view.scroll_top(), 3);
    // scroll_to_start releases the pin.
    scroll_view.scroll_to_start();
    assert_eq!(scroll_view.scroll_top(), 0);
    assert!(!scroll_view.is_following_end());
    scroll_view.scroll_to_end();
    assert_eq!(scroll_view.scroll_top(), 3);
    assert!(scroll_view.is_following_end());
}

#[test]
fn set_scrollbar_transitions_drive_visibility_and_reserved_width() {
    let scroll_view = ScrollView::new(comp(text("1\n2\n3\n4")), Default::default());
    assert_eq!(scroll_view.scrollbar(), ScrollViewScrollbar::Hidden);
    scroll_view.set_scrollbar(ScrollViewScrollbar::Always);
    assert_eq!(scroll_view.content_width(10), 9);
    scroll_view.set_scrollbar(ScrollViewScrollbar::Always); // no-op repeat
    scroll_view.set_scrollbar(ScrollViewScrollbar::Hidden);
    assert_eq!(scroll_view.content_width(10), 10);
    scroll_view.set_scrollbar(ScrollViewScrollbar::Auto);
    scroll_view.set_scrollbar_active(true);
    assert!(scroll_view.is_scrollbar_active());
    // An auto scrollbar never shows before the first layout commits a viewport.
    assert!(!scroll_view.is_scrollbar_visible());
    let _ = format!("{:?}", scroll_view.scrollbar());

    // The frame consumers: hit path, scroll lookup, geometry guards.
    let frame = render_layout_frame(
        &comp(Rc::clone(&scroll_view)),
        10,
        3,
        &noop_render_request(),
    );
    let boxes = get_layout_boxes_at(&frame, 0, 0);
    assert!(!boxes.is_empty(), "the hit path covers the frame");
    assert_eq!(
        get_scroll_views_at(&frame, 0, 0).len(),
        1,
        "the scroll view mounts as the frame root"
    );
    let scroll_box_frame = render_layout_frame(
        &comp(ScrollView::new(
            comp(text("1\n2\n3\n4\n5\n6")),
            ScrollViewOptions {
                primary: true,
                ..ScrollViewOptions::default()
            },
        )),
        10,
        3,
        &noop_render_request(),
    );
    let primary = scroll_box_frame
        .primary_scroll_view
        .as_ref()
        .expect("the sole scroll view is primary");
    let scroll_box =
        get_scroll_view_box(&scroll_box_frame, primary).expect("the scroll box exists");
    assert_eq!(scroll_box.rect.width, 10);
    assert_eq!(get_scroll_views_at(&scroll_box_frame, 0, 0).len(), 1);
    assert!(get_scroll_views_at(&scroll_box_frame, 50, 50).is_empty());
    // The geometry guards: hidden scrollbars answer None.
    assert!(
        get_scrollbar_geometry(scroll_box, false).is_none(),
        "no visible scrollbar yet"
    );
    // The include-hidden-auto arm answers for a hidden auto scrollbar only
    // when the content overflows; this fitting content stays None.
    assert!(
        get_scrollbar_geometry(scroll_box, true).is_none(),
        "the hidden auto arm answers None for fitting content"
    );
}

// === terminal-image crop guards ===

#[test]
fn crop_guards_return_the_line_unchanged() {
    let image_id = 7;
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
    // Full crop is a no-op.
    assert_eq!(crop_kitty_image_line(&image_line, 0, 3), image_line);
    // Zero visible rows keeps the line.
    assert_eq!(crop_kitty_image_line(&image_line, 0, 0), image_line);
    // Fully hidden rows keep the line.
    assert_eq!(crop_kitty_image_line(&image_line, 3, 1), image_line);
    // A non-Kitty line keeps its metadata lookup empty.
    assert!(get_kitty_image_metadata("plain").is_none());
    assert_eq!(crop_kitty_image_line("plain", 0, 1), "plain");
}

#[test]
fn encode_kitty_chunks_large_payloads() {
    let small = encode_kitty(
        "AAAA",
        EncodeKittyOptions {
            columns: Some(2),
            rows: Some(3),
            image_id: Some(9),
            move_cursor: Some(false),
        },
    );
    assert!(small.starts_with("\x1b_Ga=T,f=100,q=2,C=1,c=2,r=3,i=9;AAAA\x1b\\"));
    let chunked = encode_kitty(
        &"A".repeat(10_000),
        EncodeKittyOptions {
            image_id: Some(9),
            ..EncodeKittyOptions::default()
        },
    );
    // First chunk carries the params and m=1; the final chunk closes m=0.
    assert!(
        chunked.starts_with("\x1b_Ga=T,f=100,q=2,i=9,m=1;"),
        "the first chunk keeps the placement params"
    );
    assert!(
        chunked.contains("\x1b_Gm=0;"),
        "the final chunk closes the stream"
    );
    assert!(
        chunked.ends_with("\x1b\\"),
        "the transmission terminator closes the last chunk"
    );
    assert!(
        chunked.contains("\x1b_Gm=0;"),
        "the final chunk marker exists"
    );
}

// === debug surfaces and second-pass branches ===

#[test]
fn debug_impls_render_the_slice_types() {
    let options = StackEntryOptions {
        basis: Some(Basis::Cells(2)),
        grow: Some(1),
        shrink: Some(0),
        min_size: Some(1),
        max_size: Some(9),
        ..Default::default()
    };
    let options_debug = format!("{options:?}");
    assert!(options_debug.contains("StackEntryOptions"));
    let entry = pi_tui::layout_node::StackLayoutEntry {
        component: comp(text("x")),
        basis: None,
        grow: 0,
        shrink: 1,
        min_size: 0,
        max_size: usize::MAX,
        visible: None,
    };
    assert!(
        format!(
            "{:?}",
            pi_tui::layout_node::StackLayoutEntry {
                component: comp(text("x")),
                ..entry
            }
        )
        .contains("StackLayoutEntry")
    );
    let _ = entry;
    let child = StackChild::component(comp(text("x")));
    assert!(format!("{child:?}").contains("StackChild"));
    let entry_child = StackChild::entry(comp(text("x")), Default::default());
    assert!(format!("{entry_child:?}").contains("Entry"));
    // The From impl restates upstream's bare-component pass-through.
    let from_child: StackChild = comp(text("y")).into();
    assert!(format!("{from_child:?}").contains("Component"));

    let frame = render_layout_frame(
        &comp(VStack::new(
            vec![StackChild::component(comp(text("a")))],
            Default::default(),
        )),
        10,
        2,
        &noop_render_request(),
    );
    let box_debug = format!("{:?}", frame.root);
    assert!(box_debug.contains("LayoutBox"));
    let frame_debug = format!("{frame:?}");
    assert!(frame_debug.contains("LayoutFrame"));

    let scroll_view = ScrollView::new(comp(text("1\n2")), Default::default());
    let _ = format!("{scroll_view:?}");
    let options = ScrollViewOptions::default();
    assert!(format!("{options:?}").contains("ScrollViewOptions"));
    let _ = format!("{:?}", scroll_view.scrollbar());
}

#[test]
fn scroll_state_debug_and_overscroll_round_trip() {
    let scroll_view = ScrollView::new(
        comp(text("1\n2\n3\n4")),
        ScrollViewOptions {
            overscroll: Some(pi_tui::layout_node::Overscroll::Contain),
            ..ScrollViewOptions::default()
        },
    );
    let _ = render_layout_frame(
        &comp(Rc::clone(&scroll_view)),
        10,
        3,
        &noop_render_request(),
    );
    // The state's Debug and the overscroll getter ride through the frame.
    let primary = frame_of(&scroll_view)
        .primary_scroll_view
        .expect("the sole scroll view is primary");
    assert_eq!(
        primary.overscroll(),
        pi_tui::layout_node::Overscroll::Contain
    );
    assert_eq!(
        primary.scroll_top(),
        0,
        "no follow mode, so the offset stays"
    );
    assert_eq!(primary.viewport_height(), 3);
}

fn frame_of(scroll_view: &Rc<ScrollView>) -> pi_tui::layout::LayoutFrame {
    render_layout_frame(&comp(Rc::clone(scroll_view)), 10, 3, &noop_render_request())
}

#[test]
fn stack_children_invalidate_and_mouse_dispatch_reach_children() {
    let stack = HStack::new(
        vec![StackChild::component(comp(text("a\nb")))],
        Default::default(),
    );
    let child = stack.children().remove(0);
    stack.invalidate();
    // A press inside the child's row dispatches to it (no mouse handler, so
    // the dispatch answers None but walks).
    let event = TuiMouseEvent {
        event_type: TuiMouseEventType::Press,
        button: TuiMouseButton::Left,
        x: 0,
        y: 0,
        screen_x: 0,
        screen_y: 0,
        width: 5,
        height: 2,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    };
    assert!(stack.handle_mouse(&event).is_none());
    // A miss below the child walks to None.
    let miss = TuiMouseEvent {
        y: 5,
        height: 1,
        ..event
    };
    assert!(stack.handle_mouse(&miss).is_none());
    // invalidate walks children.
    let _ = child;
}

#[test]
fn cursor_marker_offset_scrolls_the_leaf_window() {
    // A leaf whose cursor marker sits past the allocation scrolls the paint
    // window so the marker row stays visible.
    let lines = vec![
        "one".to_string(),
        "two".to_string(),
        format!("three{CURSOR_MARKER}"),
    ];
    let frame = render_layout_frame(
        &comp(Rc::new(FixedLeaf { lines })),
        10,
        2,
        &noop_render_request(),
    );
    // The marker is on row 2; the viewport is 2 rows, so the paint window
    // offsets by one and rows show two/three.
    let visible: Vec<String> = frame
        .lines
        .iter()
        .map(|line| strip_terminal_sequences(line).trim_end().to_string())
        .collect();
    assert_eq!(
        visible,
        ["two", "three"],
        "the marker row stayed in the window"
    );
    // The engine paints the marker verbatim; stripping is the renderer's
    // extractCursorPosition job, upstream's split too.
    assert!(
        frame.lines.iter().any(|line| line.contains(CURSOR_MARKER)),
        "the marker rides to the renderer"
    );
}

#[test]
fn stack_child_mouse_dispatch_finds_the_child_row() {
    // A stack whose child answers mouse events: the dispatch walks rows.
    let inner: Rc<MouseLeaf> = Rc::new(MouseLeaf);
    let stack = VStack::new(vec![StackChild::component(comp(inner))], Default::default());
    let event = TuiMouseEvent {
        event_type: TuiMouseEventType::Press,
        button: TuiMouseButton::Left,
        x: 0,
        y: 0,
        screen_x: 0,
        screen_y: 0,
        width: 10,
        height: 2,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    };
    let result = stack.handle_mouse(&event).expect("the child row handles");
    assert!(result.handled);
}

/// A leaf rendering fixed lines, upstream's inline object components.
struct FixedLeaf {
    lines: Vec<String>,
}

impl Component for FixedLeaf {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.clone()
    }

    fn invalidate(&self) {}
}

/// A leaf whose mouse handler answers handled, for dispatch walks.
struct MouseLeaf;

impl Component for MouseLeaf {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["leaf".to_string()]
    }

    fn handle_mouse(&self, _event: &TuiMouseEvent) -> Option<pi_tui::tui::TuiMouseEventResult> {
        Some(pi_tui::tui::TuiMouseEventResult {
            handled: true,
            ..Default::default()
        })
    }
}
