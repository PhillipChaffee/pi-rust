//! Boundary tests for the #42 component slice, added where upstream's
//! behavioral suites leave branches untested; the box, spacer, and loader
//! components have no upstream behavioral suite at all (only the excluded
//! `chat-simple.ts` demo drives the loader), so their contracts bind here.
#![expect(
    clippy::expect_used,
    reason = "a failed dispatch or missing target in a fixture is a test-environment failure; expecting keeps the assertions readable"
)]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use pi_tui::components::{
    Box, CancellableLoader, ColorFn, Loader, LoaderIndicatorOptions, Spacer, Text, TruncatedText,
};
use pi_tui::tui::{
    Component, TuiMouseButton, TuiMouseDispatchTarget, TuiMouseEvent, TuiMouseEventResult,
    TuiMouseEventType, dispatch_mouse_event,
};

/// A component that renders a fixed number of lines and counts renders, so
/// tests can observe cache behavior.
struct FakeComponent {
    lines: u16,
    render_count: Cell<usize>,
    mouse_result: TuiMouseEventResult,
}

impl FakeComponent {
    fn new(lines: u16) -> Rc<Self> {
        Rc::new(Self {
            lines,
            render_count: Cell::new(0),
            mouse_result: TuiMouseEventResult::default(),
        })
    }

    fn styled(lines: u16, mouse_result: TuiMouseEventResult) -> Rc<Self> {
        Rc::new(Self {
            lines,
            render_count: Cell::new(0),
            mouse_result,
        })
    }
}

impl Component for FakeComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.render_count.set(self.render_count.get() + 1);
        (0..self.lines).map(|line| format!("line{line}")).collect()
    }

    fn handle_mouse(&self, _event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        Some(self.mouse_result.clone())
    }
}

/// Implements only `render`, so the trait's default method bodies execute.
struct BareComponent;

impl Component for BareComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["bare".to_string()]
    }
}

const fn press_at(x: u16, y: u16, width: u16, height: u16) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type: TuiMouseEventType::Press,
        button: TuiMouseButton::Left,
        x,
        y,
        screen_x: x,
        screen_y: y,
        width,
        height,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    }
}

fn identity_color_fn() -> ColorFn {
    Arc::new(|text: &str| text.to_string())
}

fn noop_render_request() -> Arc<dyn Fn() + Send + Sync> {
    Arc::new(|| {})
}

fn counting_render_request(count: &Arc<AtomicUsize>) -> Arc<dyn Fn() + Send + Sync> {
    let count = Arc::clone(count);
    Arc::new(move || {
        count.fetch_add(1, Ordering::SeqCst);
    })
}

/// Polls `check` until it passes or the deadline expires, returning the
/// final check.
fn wait_for(mut check: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if check() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    check()
}

mod box_component {
    use super::*;

    #[test]
    fn renders_children_with_horizontal_and_vertical_padding() {
        let container = Box::with_padding(1, 1, None);
        container.add_child(Rc::new(TruncatedText::with_padding("hi", 0, 0)));

        let lines = container.render(10);

        // 1 top pad + 1 content + 1 bottom pad
        assert_eq!(lines.len(), 3);
        // Padded to full width, content indented by the padding
        assert_eq!(pi_tui::utils::visible_width(&lines[1]), 10);
        assert!(lines[1].trim().contains("hi"));
        assert_eq!(lines[0], " ".repeat(10));
        assert_eq!(lines[2], " ".repeat(10));
    }

    #[test]
    fn applies_background_to_padded_lines() {
        let container = Box::with_padding(
            1,
            0,
            Some(Arc::new(|text: &str| format!("\x1b[41m{text}\x1b[0m"))),
        );
        container.add_child(Rc::new(TruncatedText::with_padding("hi", 0, 0)));

        let lines = container.render(8);
        assert!(lines[0].starts_with("\x1b[41m"));
        assert!(lines[0].ends_with("\x1b[0m"));
        assert_eq!(pi_tui::utils::visible_width(&lines[0]), 8);
    }

    #[test]
    fn renders_nothing_without_children() {
        let container = Box::new();
        assert!(container.render(20).is_empty());
    }

    #[test]
    fn caches_render_output_until_invalidated() {
        let container = Box::with_padding(0, 0, None);
        let child = FakeComponent::new(2);
        container.add_child(child.clone());

        let first = container.render(20);
        let second = container.render(20);
        assert_eq!(first, second);
        // Upstream's cache saves the background application, not child
        // rendering: children render on every call.
        assert_eq!(child.render_count.get(), 2);

        container.invalidate();
        container.render(20);
        assert_eq!(child.render_count.get(), 3);
    }

    #[test]
    fn detects_background_changes_by_sampling_without_invalidation() {
        let container = Box::with_padding(0, 0, None);
        container.add_child(FakeComponent::new(1));

        let before = container.render(20);
        container.set_bg_fn(Some(Arc::new(|text: &str| {
            format!("\x1b[44m{text}\x1b[0m")
        })));
        let after = container.render(20);

        assert_ne!(before, after);
        assert!(after[0].contains("\x1b[44m"));
    }

    #[test]
    fn removes_and_clears_children() {
        let container = Box::with_padding(0, 0, None);
        let child = FakeComponent::new(1);
        let first: Rc<dyn Component> = child;
        container.add_child(first.clone());
        container.remove_child(&first);
        assert!(container.render(10).is_empty());

        container.add_child(first);
        container.clear();
        assert!(container.render(10).is_empty());
    }

    #[test]
    fn dispatches_mouse_events_to_the_hit_child_with_local_coordinates() {
        let container = Box::with_padding(1, 1, None);
        let first = FakeComponent::styled(
            3,
            TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            },
        );
        let second: Rc<dyn Component> = FakeComponent::styled(
            2,
            TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            },
        );
        container.add_child(first);
        container.add_child(second.clone());
        container.render(10);

        // Row 4 falls in the second child: content_y = 4 - 1 = 3, the first
        // child occupies content rows 0..3, so the local y is 0.
        let event = press_at(2, 4, 10, 10);
        let result = container.handle_mouse(&event).expect("should dispatch");
        let target = result.target.expect("dispatch should retain a target");
        assert!(Rc::ptr_eq(&target.component, &second));
        assert_eq!(target.origin_x, 1);
        assert_eq!(target.origin_y, 4);
        assert_eq!(target.width, 8);
        assert_eq!(target.height, 2);
    }

    #[test]
    fn drops_mouse_events_outside_the_content_area() {
        let container = Box::with_padding(1, 1, None);
        container.add_child(FakeComponent::new(3));
        container.render(10);

        // In the top padding row.
        assert!(container.handle_mouse(&press_at(2, 0, 10, 10)).is_none());
        // Right of the content area.
        assert!(container.handle_mouse(&press_at(9, 2, 10, 10)).is_none());
        // Below the children.
        assert!(container.handle_mouse(&press_at(2, 6, 10, 10)).is_none());
    }

    #[test]
    fn remeasures_child_heights_when_the_content_width_changes() {
        let container = Box::with_padding(0, 0, None);
        let child = FakeComponent::new(2);
        container.add_child(child.clone());
        container.render(20);

        // A dispatch at a different width re-measures by rendering.
        container.handle_mouse(&press_at(0, 1, 30, 30));
        assert_eq!(child.render_count.get(), 2);
    }
}

mod text_component {
    use super::*;

    #[test]
    fn renders_wrapped_lines_with_margins() {
        let text = Text::new("hello world once more than fits one line");
        let lines = text.render(20);

        // 1 top pad + wrapped content + 1 bottom pad
        assert!(lines.len() > 3);
        assert!(lines[1].trim().contains("hello world"));
        for line in &lines {
            assert_eq!(pi_tui::utils::visible_width(line), 20);
        }
    }

    #[test]
    fn caches_render_output_per_text_and_width() {
        let text = Text::new("hello");
        let first = text.render(20);
        assert_eq!(text.render(20), first);

        text.set_text("world");
        assert!(text.render(20)[1].contains("world"));
    }

    #[test]
    fn renders_nothing_for_blank_text() {
        let text = Text::new("   ");
        assert!(text.render(20).is_empty());

        // The blank render is cached: rendering again stays empty.
        assert!(text.render(20).is_empty());
    }

    #[test]
    fn normalizes_tabs_to_three_spaces() {
        let text = Text::new("a\tb");
        let lines = text.render(40);
        assert!(lines[1].contains("a   b"));
    }

    #[test]
    fn shrinks_margins_when_the_width_is_tight() {
        // paddingX=4 cannot fit into a width of 5; it must reduce to
        // floor((width-1)/2) = 2.
        let text = Text::with_padding("word word word", 4, 0);
        let lines = text.render(5);
        assert!(!lines.is_empty());
        for line in &lines {
            assert_eq!(pi_tui::utils::visible_width(line), 5);
        }
    }

    #[test]
    fn applies_custom_background_to_content_and_padding() {
        let text = Text::with_background("hi", 1, 1, Some(background_styler()));
        let lines = text.render(10);

        assert_eq!(lines.len(), 3);
        for line in &lines {
            assert!(line.starts_with("\x1b[41m"));
            assert!(line.ends_with("\x1b[0m"));
            assert_eq!(pi_tui::utils::visible_width(line), 10);
        }
    }

    #[test]
    fn set_custom_bg_fn_replaces_the_background() {
        let text = Text::with_background("hi", 0, 0, None);
        let plain = text.render(10);
        assert!(!plain[0].contains("\x1b["));

        text.set_custom_bg_fn(Some(Arc::new(|line: &str| {
            format!("\x1b[44m{line}\x1b[0m")
        })));
        let styled = text.render(10);
        assert!(styled[0].contains("\x1b[44m"));
    }

    #[test]
    fn invalidate_clears_the_cache() {
        let text = Text::new("hello");
        let first = text.render(20);

        // Invalidation forces a recompute producing identical output.
        text.invalidate();
        assert_eq!(text.render(20), first);
    }

    fn background_styler() -> ColorFn {
        Arc::new(|text: &str| format!("\x1b[41m{text}\x1b[0m"))
    }
}

mod spacer_component {
    use super::*;

    #[test]
    fn renders_the_configured_number_of_blank_lines() {
        let spacer = Spacer::with_lines(3);
        assert_eq!(spacer.render(40), vec!["", "", ""]);

        spacer.set_lines(1);
        assert_eq!(spacer.render(40), vec![""]);
    }

    #[test]
    fn defaults_to_one_blank_line() {
        let spacer = Spacer::default();
        assert_eq!(spacer.render(40), vec![""]);
    }
}

mod dispatch {
    use super::*;

    #[test]
    fn synthesizes_a_target_for_a_plain_handled_result() {
        let component: Rc<dyn Component> = FakeComponent::styled(
            1,
            TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            },
        );

        let event = press_at(3, 4, 10, 20);
        let result = dispatch_mouse_event(&component, &event).expect("handled events dispatch");
        assert!(result.handled);
        let target = result.target.expect("dispatch attaches a target");
        assert!(Rc::ptr_eq(&target.component, &component));
        assert_eq!(target.origin_x, 0);
        assert_eq!(target.origin_y, 0);
        assert_eq!(target.width, 10);
        assert_eq!(target.height, 20);
        assert!(result.focus_target.is_none());
    }

    #[test]
    fn drops_results_that_handle_neither_capture_nor_focus() {
        let component: Rc<dyn Component> = FakeComponent::styled(1, TuiMouseEventResult::default());
        assert!(dispatch_mouse_event(&component, &press_at(0, 0, 10, 10)).is_none());
    }

    #[test]
    fn attaches_a_focus_target_when_focus_is_requested() {
        let component: Rc<dyn Component> = FakeComponent::styled(
            1,
            TuiMouseEventResult {
                focus: true,
                ..TuiMouseEventResult::default()
            },
        );

        let result = dispatch_mouse_event(&component, &press_at(3, 4, 10, 20)).expect("dispatches");
        assert!(result.handled);
        assert!(result.focus_target.is_some());
        let target = result.target.expect("dispatch attaches a target");
        assert_eq!(target.origin_x, 0);
        assert_eq!(target.origin_y, 0);
    }

    #[test]
    fn passes_an_already_dispatched_result_through_verbatim() {
        let inner: Rc<dyn Component> = FakeComponent::styled(
            1,
            TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            },
        );
        let component: Rc<dyn Component> = FakeComponent::styled(
            1,
            TuiMouseEventResult {
                handled: true,
                target: Some(TuiMouseDispatchTarget {
                    component: inner.clone(),
                    origin_x: 7,
                    origin_y: 8,
                    width: 3,
                    height: 4,
                }),
                ..TuiMouseEventResult::default()
            },
        );

        let result = dispatch_mouse_event(&component, &press_at(1, 1, 10, 10)).expect("dispatches");
        let target = result.target.expect("passthrough keeps the target");
        assert!(Rc::ptr_eq(&target.component, &inner));
        assert_eq!(target.origin_x, 7);
    }

    #[test]
    fn skips_components_without_a_mouse_handler() {
        let component: Rc<dyn Component> = Rc::new(BareComponent);
        assert!(dispatch_mouse_event(&component, &press_at(0, 0, 10, 10)).is_none());
    }

    #[test]
    fn exposes_the_trait_defaults_upstream_made_optional() {
        let component: Rc<dyn Component> = Rc::new(BareComponent);
        component.handle_input("x");
        assert!(!component.wants_key_release());
        assert!(component.handle_mouse(&press_at(0, 0, 10, 10)).is_none());
        component.invalidate();
        assert_eq!(component.render(10), vec!["bare".to_string()]);
    }
}

mod loader_component {
    use super::*;

    fn loader_with(frames: Vec<String>, interval_ms: u64) -> (Loader, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let loader = Loader::new(
            counting_render_request(&count),
            identity_color_fn(),
            identity_color_fn(),
            "Loading",
            Some(LoaderIndicatorOptions {
                frames: Some(frames),
                interval_ms: Some(interval_ms),
            }),
        );
        (loader, count)
    }

    #[test]
    fn renders_a_leading_blank_line_before_the_display() {
        let (loader, _count) = loader_with(vec!["a".to_string()], 0);
        let lines = loader.render(20);
        assert_eq!(lines[0], "");
        assert!(lines[1].contains("Loading"));
    }

    #[test]
    fn advances_frames_on_the_animation_thread() {
        let (loader, count) =
            loader_with(vec!["a".to_string(), "b".to_string(), "c".to_string()], 10);

        let initial = loader.render(20)[1].clone();
        let advanced = wait_for(|| loader.render(20)[1] != initial);
        assert!(advanced, "animation should advance the displayed frame");
        assert!(count.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn set_message_updates_the_display_immediately() {
        let (loader, _count) = loader_with(vec!["a".to_string()], 0);
        loader.set_message("Porting...");
        assert!(loader.render(20)[1].contains("Porting..."));
    }

    #[test]
    fn empty_frames_hide_the_indicator_and_skip_the_animation() {
        let (loader, count) = loader_with(vec![], 0);

        assert_eq!(loader.render(20)[1].trim(), "Loading");
        // No animation worker arms itself for a single empty frame.
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn zero_interval_selects_the_default_interval() {
        let count = Arc::new(AtomicUsize::new(0));
        let loader = Loader::new(
            counting_render_request(&count),
            identity_color_fn(),
            identity_color_fn(),
            "Loading",
            Some(LoaderIndicatorOptions {
                frames: Some(vec!["x".to_string(), "y".to_string()]),
                interval_ms: Some(0),
            }),
        );

        let initial = loader.render(20)[1].trim().to_string();
        assert_eq!(initial, "x Loading");
        assert!(wait_for(|| loader.render(20)[1] != initial));
    }

    #[test]
    fn stop_disarms_the_animation() {
        let (loader, count) = loader_with(vec!["a".to_string(), "b".to_string()], 10);

        let initial = loader.render(20)[1].clone();
        assert!(wait_for(|| loader.render(20)[1] != initial));
        loader.stop();

        // The stop is best-effort: one set already in flight may land, so
        // assert the render requests settle instead of asserting an exact
        // display. A still-running worker would tick every 10 ms.
        std::thread::sleep(Duration::from_millis(80));
        let settled = count.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(150));
        assert!(count.load(Ordering::SeqCst) <= settled + 1);
    }

    #[test]
    fn invalidate_refreshes_the_display_and_requests_a_render() {
        let (loader, count) = loader_with(vec!["a".to_string(), "b".to_string()], 1_000_000);
        let before = count.load(Ordering::SeqCst);

        loader.invalidate();
        assert!(count.load(Ordering::SeqCst) > before);
        assert!(loader.render(20)[1].contains("Loading"));
    }

    #[test]
    fn dropping_the_loader_stops_the_thread() {
        let count = Arc::new(AtomicUsize::new(0));
        let loader = Loader::new(
            counting_render_request(&count),
            identity_color_fn(),
            identity_color_fn(),
            "Loading",
            Some(LoaderIndicatorOptions {
                frames: Some(vec!["a".to_string(), "b".to_string()]),
                interval_ms: Some(10),
            }),
        );
        assert!(wait_for(|| count.load(Ordering::SeqCst) > 1));
        drop(loader);

        let before = count.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(60));
        assert!(count.load(Ordering::SeqCst) <= before + 2);
    }
}

mod boundary_extras {
    use super::*;

    #[test]
    fn debug_impls_render_without_panicking() {
        let container = Box::with_padding(2, 3, None);
        assert!(format!("{container:?}").contains("padding_x: 2"));

        let text = Text::new("hello");
        assert!(format!("{text:?}").contains("Text"));

        let loader = Loader::new(
            noop_render_request(),
            identity_color_fn(),
            identity_color_fn(),
            "Loading",
            None,
        );
        assert!(format!("{loader:?}").contains("Loader"));

        let cancellable = CancellableLoader::new(
            noop_render_request(),
            identity_color_fn(),
            identity_color_fn(),
            "Working...",
            None,
        );
        assert!(format!("{cancellable:?}").contains("CancellableLoader"));

        let result = TuiMouseEventResult {
            handled: true,
            render: Some(true),
            ..TuiMouseEventResult::default()
        };
        let rendered = format!("{result:?}");
        assert!(rendered.contains("handled: true"));
        assert!(rendered.contains("render: Some(true)"));

        let inner: Rc<dyn Component> = FakeComponent::new(1);
        let target = TuiMouseDispatchTarget {
            component: inner,
            origin_x: 1,
            origin_y: 2,
            width: 3,
            height: 4,
        };
        let rendered = format!("{target:?}");
        assert!(rendered.contains("origin_x: 1"));
        assert!(rendered.contains("height: 4"));
    }

    #[test]
    fn set_indicator_without_options_restores_the_default_spinner() {
        let count = Arc::new(AtomicUsize::new(0));
        let loader = Loader::new(
            counting_render_request(&count),
            identity_color_fn(),
            identity_color_fn(),
            "Loading",
            Some(LoaderIndicatorOptions {
                frames: Some(vec!["a".to_string()]),
                interval_ms: None,
            }),
        );

        // No options: the default braille spinner frames, styled through the
        // spinner color function, at the default interval.
        loader.set_indicator(None);
        let display = loader.render(20)[1].trim().to_string();
        assert!(display.starts_with("⠋ Loading"));
        assert!(wait_for(|| loader.render(20)[1] != display));
    }

    #[test]
    fn an_in_flight_set_exits_when_the_indicator_switches_to_a_single_frame() {
        let count = Arc::new(AtomicUsize::new(0));
        let loader = Loader::new(
            counting_render_request(&count),
            identity_color_fn(),
            identity_color_fn(),
            "Loading",
            Some(LoaderIndicatorOptions {
                frames: Some(vec!["a".to_string(), "b".to_string()]),
                interval_ms: Some(10),
            }),
        );
        assert!(wait_for(|| count.load(Ordering::SeqCst) > 1));

        // Swapping to a single frame stops arming new ticks: the in-flight
        // worker exits on its next wake instead of advancing, though one
        // set already in flight may land.
        loader.set_indicator(Some(LoaderIndicatorOptions {
            frames: Some(vec!["z".to_string()]),
            interval_ms: None,
        }));
        std::thread::sleep(Duration::from_millis(60));
        let settled = count.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(loader.render(20)[1].trim(), "z Loading");
        assert!(count.load(Ordering::SeqCst) <= settled + 1);
    }

    #[test]
    fn children_rendering_nothing_leave_the_box_empty() {
        let container = Box::with_padding(1, 1, None);
        container.add_child(FakeComponent::new(0));
        assert!(container.render(10).is_empty());
    }

    #[test]
    fn default_box_matches_the_upstream_defaults() {
        let container = Box::default();
        container.add_child(Rc::new(TruncatedText::new("hi")));
        let lines = container.render(10);
        assert_eq!(lines.len(), 3);
        assert_eq!(pi_tui::utils::visible_width(&lines[1]), 10);
    }
}

mod fuzzy_boundaries {
    use pi_tui::fuzzy::{fuzzy_filter, fuzzy_match};

    #[test]
    fn a_swapped_query_that_still_fails_returns_the_primary_miss() {
        #[expect(
            clippy::float_cmp,
            reason = "a failed match returns the literal 0.0; no arithmetic accumulates"
        )]
        fn expect_no_match(query: &str, text: &str) {
            let result = fuzzy_match(query, text);
            assert!(!result.matches);
            assert_eq!(result.score, 0.0);
        }

        expect_no_match("x52", "abc");
        expect_no_match("52x", "abc");
    }

    #[test]
    fn a_query_of_only_separators_returns_all_items() {
        let items = vec!["apple", "banana"];
        let result = fuzzy_filter(&items, "/", |x| (*x).to_string());
        assert_eq!(result, items.iter().collect::<Vec<_>>());
    }
}

mod cancellable_loader_component {
    use super::*;

    fn make_cancellable() -> (CancellableLoader, Rc<Cell<usize>>) {
        let aborts = Rc::new(Cell::new(0));
        let loader = CancellableLoader::new(
            noop_render_request(),
            identity_color_fn(),
            identity_color_fn(),
            "Working...",
            Some(LoaderIndicatorOptions {
                frames: Some(vec!["a".to_string()]),
                interval_ms: None,
            }),
        );
        let slot = aborts.clone();
        loader.set_on_abort(Some(std::boxed::Box::new(move || slot.set(slot.get() + 1))));
        (loader, aborts)
    }

    #[test]
    fn escape_cancels_the_token_and_fires_the_abort_callback() {
        let (loader, aborts) = make_cancellable();
        assert!(!loader.is_aborted());

        loader.handle_input("\x1b");
        assert!(loader.is_aborted());
        assert_eq!(aborts.get(), 1);
        assert!(loader.cancellation_token().is_cancelled());
    }

    #[test]
    fn other_input_does_not_cancel() {
        let (loader, aborts) = make_cancellable();
        loader.handle_input("x");
        assert!(!loader.is_aborted());
        assert_eq!(aborts.get(), 0);
    }

    #[test]
    fn render_and_invalidate_delegate_to_the_embedded_loader() {
        let (loader, _aborts) = make_cancellable();
        let lines = loader.render(20);
        assert_eq!(lines[0], "");
        assert!(lines[1].contains("Working..."));

        loader.invalidate();
        loader.loader().stop();
        loader.dispose();
        assert!(!loader.is_aborted());
    }

    #[test]
    fn cancel_shows_in_the_rendered_indicator_after_refresh() {
        let (loader, _aborts) = make_cancellable();
        loader.handle_input("\x1b");
        assert!(loader.is_aborted());
        assert!(loader.render(20)[1].contains("Working..."));
    }
}
