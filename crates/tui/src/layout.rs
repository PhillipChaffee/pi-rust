//! The constrained layout engine of `packages/tui/src/layout.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! ([#44](https://github.com/PhillipChaffee/pi-rust/issues/44)).
//!
//! It rebuilds geometry per frame with leaf-cache reuse, intersects clips,
//! translates scroll views, paints scrollbars, and produces the hit-testing
//! the alternate screen's mouse dispatch walks.
//!
//! The `tui-plan.md` decisions bind (map ticket "Survey the tui package"):
//! layout is alt-screen-only; per-frame geometry rebuild with leaf-cache
//! reuse — `invalidate()` stays single-owner on the component and the
//! engine's render cache lives one frame; layout internals stay private
//! (only the frame consumers are public).
//!
//! Restatements against upstream:
//!
//! - `LayoutBox.parent` is omitted: upstream sets it at every level and
//!   nothing in the package reads it.
//! - [`LayoutRect`] fields are `i64`: a scrolled content box translates
//!   above the viewport and its rect goes negative before `updateClips`
//!   clamps; widths and heights are clamped non-negative at construction,
//!   as upstream's `Math.max(0, ...)` does.
//! - The per-frame render cache keys on the component's allocation
//!   address beside the width, upstream's `Map<Component, Map<number,
//!   string[]>>` object identity; the cache lives one frame, so every
//!   component is alive for the whole walk.
//! - `getKittyImageMetadata`/`cropKittyImageLine`/`isImageLine` come from
//!   the terminal-image slice ([`crate::terminal_image`]); the placement
//!   extraction stays with #46.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use regex::Regex;

use crate::layout_node::{Basis, LayoutNode, ScrollStateHandle, StackAlign, StackKind};
use crate::terminal_image::{crop_kitty_image_line, get_kitty_image_metadata, is_image_line};
use crate::tui::{CURSOR_MARKER, Component, RenderRequest, composite_tui_line};
use crate::utils::{
    extract_ansi_code, get_active_background_ansi, get_grapheme_cell_range, slice_by_column,
    visible_width,
};

/// Upstream `OSC133_ZONE_PREFIX`: the semantic-prompt zone markers to
/// strip from painted lines.
fn osc133_zone_prefix() -> &'static Regex {
    static PATTERN: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        crate::utils::static_regex(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")
    });
    &PATTERN
}

/// A rectangle in frame coordinates, upstream `LayoutRect`. X and Y may be
/// negative for content translated above the viewport; widths and heights
/// are clamped non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayoutRect {
    /// Left edge.
    pub x: i64,
    /// Top edge.
    pub y: i64,
    /// Width in columns (never negative).
    pub width: i64,
    /// Height in rows (never negative).
    pub height: i64,
}

/// One laid-out component in the frame tree, upstream `LayoutBox`.
pub struct LayoutBox {
    /// The laid-out component.
    pub component: Rc<dyn Component>,
    /// The allocated rectangle.
    pub rect: LayoutRect,
    /// The visible intersection with the ancestor clips.
    pub clip: LayoutRect,
    /// Laid-out children.
    pub children: Vec<Self>,
    /// The leaf's rendered lines, upstream `lines?`.
    pub lines: Option<Vec<String>>,
    /// Row offset when a cursor marker forces the tail visible, upstream
    /// `lineOffset?`.
    pub line_offset: Option<usize>,
    /// The scroll state, for scroll-view boxes.
    pub scroll_view: Option<ScrollStateHandle>,
    /// The scroll content's full lines, upstream `scrollContentLines?`.
    pub scroll_content_lines: Option<Vec<String>>,
    /// Paint layer, upstream `layer` (0 in this engine; the hit-path sort
    /// reads it).
    pub layer: i64,
}

impl std::fmt::Debug for LayoutBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutBox")
            .field("rect", &self.rect)
            .field("clip", &self.clip)
            .field("children", &self.children.len())
            .field("has_lines", &self.lines.is_some())
            .field("has_scroll_view", &self.scroll_view.is_some())
            .finish_non_exhaustive()
    }
}

/// One laid-out frame, upstream `LayoutFrame`.
pub struct LayoutFrame {
    /// The laid-out tree.
    pub root: LayoutBox,
    /// Frame width in columns.
    pub width: usize,
    /// Frame height in rows.
    pub height: usize,
    /// The painted screen rows.
    pub lines: Vec<String>,
    /// The frame's designated scroll view, upstream
    /// `primaryScrollView?`.
    pub primary_scroll_view: Option<ScrollStateHandle>,
}

impl std::fmt::Debug for LayoutFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutFrame")
            .field("root", &self.root)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("lines", &self.lines)
            .finish_non_exhaustive()
    }
}

/// Where the scrollbar paints, upstream `ScrollbarGeometry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollbarGeometry {
    /// The column the glyphs paint at.
    pub column: usize,
    /// First row of the track.
    pub track_top: usize,
    /// Track length in rows.
    pub track_height: usize,
    /// First row of the thumb.
    pub thumb_top: usize,
    /// Thumb length in rows.
    pub thumb_height: usize,
    /// The largest reachable scroll offset.
    pub max_scroll_top: usize,
}

/// The per-frame render cache key: component identity beside the width,
/// upstream's `Map<Component, Map<number, string[]>>` object identity.
type RenderCacheKey = (usize, usize);

/// The per-frame measurement context, upstream `LayoutContext`.
struct LayoutContext<'a> {
    viewport: crate::layout_node::LayoutViewport,
    render_cache: RefCell<HashMap<RenderCacheKey, Rc<Vec<String>>>>,
    request_render: &'a RenderRequest,
    primary_scroll_view: RefCell<Option<ScrollStateHandle>>,
}

fn intersect(a: LayoutRect, b: LayoutRect) -> LayoutRect {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    LayoutRect {
        x,
        y,
        width: (right - x).max(0),
        height: (bottom - y).max(0),
    }
}

/// The one-render-per-component-per-width guarantee, upstream
/// `renderCached`: the frame's cache keyed by component identity and
/// width.
fn render_cached(
    context: &LayoutContext<'_>,
    component: &Rc<dyn Component>,
    width: usize,
) -> Rc<Vec<String>> {
    let safe_width = width.max(1);
    let key = (Rc::as_ptr(component).cast::<()>() as usize, safe_width);
    let mut cache = context.render_cache.borrow_mut();
    if let Some(lines) = cache.get(&key) {
        return Rc::clone(lines);
    }
    let lines = Rc::new(component.render(safe_width));
    cache.insert(key, Rc::clone(&lines));
    lines
}

fn measure_height(
    context: &LayoutContext<'_>,
    component: &Rc<dyn Component>,
    width: usize,
) -> usize {
    render_cached(context, component, width).len()
}

fn measure_width(
    context: &LayoutContext<'_>,
    component: &Rc<dyn Component>,
    width: usize,
) -> usize {
    render_cached(context, component, width)
        .iter()
        .map(|line| visible_width(line))
        .max()
        .unwrap_or(0)
}

fn translate_box(box_: &mut LayoutBox, delta_y: i64) {
    box_.rect.y += delta_y;
    for child in &mut box_.children {
        translate_box(child, delta_y);
    }
}

fn update_clips(box_: &mut LayoutBox, parent_clip: LayoutRect) {
    box_.clip = intersect(parent_clip, box_.rect);
    for child in &mut box_.children {
        update_clips(child, box_.clip);
    }
}

/// The cursor-marker offset a clipped leaf paints from, upstream's
/// `lineOffset` walk: when the marker falls below the allocation, scroll
/// the source window so the cursor line stays visible.
fn cursor_line_offset(lines: &[String], allocated_height: usize) -> usize {
    if lines.len() > allocated_height
        && allocated_height > 0
        && let Some(cursor_line) = lines.iter().position(|line| line.contains(CURSOR_MARKER))
        && cursor_line >= allocated_height
    {
        return cursor_line - allocated_height + 1;
    }
    0
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "layout coordinates ride terminal dimensions (u16) on both sides; the i64 rectangle keeps scrolled translations negative without wrapping, and the fn mirrors upstream's layoutComponent block for block"
)]
fn layout_component(
    context: &LayoutContext<'_>,
    component: &Rc<dyn Component>,
    x: i64,
    y: i64,
    width: usize,
    height: Option<usize>,
    clip: LayoutRect,
) -> LayoutBox {
    let safe_width = width.max(1);
    let Some(node) = component.layout_node() else {
        let lines = render_cached(context, component, safe_width);
        let allocated_height = height
            .map_or(lines.len() as i64, |height| height as i64)
            .max(0);
        let line_offset = cursor_line_offset(&lines, allocated_height.max(0) as usize);
        return LayoutBox {
            component: Rc::clone(component),
            rect: LayoutRect {
                x,
                y,
                width: safe_width as i64,
                height: allocated_height,
            },
            clip: intersect(
                clip,
                LayoutRect {
                    x,
                    y,
                    width: safe_width as i64,
                    height: allocated_height,
                },
            ),
            children: Vec::new(),
            lines: Some((*lines).clone()),
            line_offset: Some(line_offset),
            scroll_view: None,
            scroll_content_lines: None,
            layer: 0,
        };
    };

    match node {
        LayoutNode::Scroll(node) => {
            let previous_scroll_top = node.state.scroll_top();
            let content_width = node.state.content_width(safe_width);
            let mut child_box = layout_component(
                context,
                &node.component,
                x,
                y - previous_scroll_top as i64,
                content_width,
                None,
                clip,
            );
            let content_height = child_box.rect.height;
            let viewport_height = height.map_or(content_height, |height| height as i64).max(0);
            node.state.update_layout(
                content_height.max(0) as usize,
                viewport_height.max(0) as usize,
                context.request_render,
            );
            let scroll_top = node.state.scroll_top();
            translate_box(
                &mut child_box,
                previous_scroll_top as i64 - scroll_top as i64,
            );
            let scroll_view: ScrollStateHandle = Arc::clone(&node.state);
            if node.state.is_primary() || context.primary_scroll_view.borrow().is_none() {
                *context.primary_scroll_view.borrow_mut() = Some(Arc::clone(&scroll_view));
            }
            let rect = LayoutRect {
                x,
                y,
                width: safe_width as i64,
                height: viewport_height,
            };
            let child_clip = intersect(clip, rect);
            let mut child_box = child_box;
            update_clips(&mut child_box, child_clip);
            LayoutBox {
                component: Rc::clone(component),
                rect,
                clip: child_clip,
                children: vec![child_box],
                lines: None,
                line_offset: None,
                scroll_view: Some(scroll_view),
                scroll_content_lines: Some(
                    (*render_cached(context, &node.component, content_width)).clone(),
                ),
                layer: 0,
            }
        }
        LayoutNode::Stack(node) => {
            let entries =
                crate::components::stack::visible_stack_entries(&node.entries, &context.viewport);
            let gap_total = entries.len().saturating_sub(1) * node.gap;
            match node.kind {
                StackKind::Vertical => {
                    let intrinsic_heights: Vec<usize> = entries
                        .iter()
                        .map(|entry| match entry.basis {
                            Some(Basis::Cells(cells)) => {
                                usize::try_from(cells.max(0)).unwrap_or(usize::MAX)
                            }
                            _ => measure_height(context, &entry.component, safe_width),
                        })
                        .collect();
                    let sizes = crate::components::stack::allocate_stack_sizes(
                        &entries,
                        &intrinsic_heights,
                        height,
                        node.gap,
                    );
                    let natural_height: usize = sizes.iter().sum::<usize>() + gap_total;
                    let allocated_height = height
                        .map_or(natural_height as i64, |height| height as i64)
                        .max(0);
                    let rect = LayoutRect {
                        x,
                        y,
                        width: safe_width as i64,
                        height: allocated_height,
                    };
                    let mut stack_box = LayoutBox {
                        component: Rc::clone(component),
                        rect,
                        clip: intersect(clip, rect),
                        children: Vec::new(),
                        lines: None,
                        line_offset: None,
                        scroll_view: None,
                        scroll_content_lines: None,
                        layer: 0,
                    };
                    let mut child_y = y;
                    for (index, entry) in entries.iter().enumerate() {
                        let child = layout_component(
                            context,
                            &entry.component,
                            x,
                            child_y,
                            safe_width,
                            Some(sizes[index]),
                            stack_box.clip,
                        );
                        stack_box.children.push(child);
                        child_y += sizes[index] as i64 + node.gap as i64;
                    }
                    stack_box
                }
                StackKind::Horizontal => {
                    let intrinsic_widths: Vec<usize> = entries
                        .iter()
                        .map(|entry| match entry.basis {
                            Some(Basis::Cells(cells)) => {
                                usize::try_from(cells.max(0)).unwrap_or(usize::MAX)
                            }
                            _ => measure_width(context, &entry.component, safe_width),
                        })
                        .collect();
                    let widths = crate::components::stack::allocate_stack_sizes(
                        &entries,
                        &intrinsic_widths,
                        Some(safe_width),
                        node.gap,
                    );
                    let intrinsic_heights: Vec<usize> = entries
                        .iter()
                        .enumerate()
                        .map(|(index, entry)| {
                            measure_height(context, &entry.component, widths[index].max(1))
                        })
                        .collect();
                    let allocated_height = height
                        .unwrap_or_else(|| intrinsic_heights.iter().copied().max().unwrap_or(0))
                        as i64;
                    let rect = LayoutRect {
                        x,
                        y,
                        width: safe_width as i64,
                        height: allocated_height,
                    };
                    let mut stack_box = LayoutBox {
                        component: Rc::clone(component),
                        rect,
                        clip: intersect(clip, rect),
                        children: Vec::new(),
                        lines: None,
                        line_offset: None,
                        scroll_view: None,
                        scroll_content_lines: None,
                        layer: 0,
                    };
                    let mut child_x = x;
                    for (index, entry) in entries.iter().enumerate() {
                        let natural_child_height = intrinsic_heights[index];
                        let child_height = if node.align == StackAlign::Stretch {
                            allocated_height
                        } else {
                            allocated_height.min(natural_child_height as i64)
                        };
                        let mut child_y = y;
                        if node.align == StackAlign::Center {
                            child_y += (allocated_height - child_height) / 2;
                        } else if node.align == StackAlign::End {
                            child_y += allocated_height - child_height;
                        }
                        let child_width = widths[index];
                        if child_width == 0 {
                            stack_box.children.push(LayoutBox {
                                component: Rc::clone(&entry.component),
                                rect: LayoutRect {
                                    x: child_x,
                                    y: child_y,
                                    width: 0,
                                    height: child_height,
                                },
                                clip: LayoutRect {
                                    x: child_x,
                                    y: child_y,
                                    width: 0,
                                    height: 0,
                                },
                                children: Vec::new(),
                                lines: None,
                                line_offset: None,
                                scroll_view: None,
                                scroll_content_lines: None,
                                layer: 0,
                            });
                        } else {
                            let child = layout_component(
                                context,
                                &entry.component,
                                child_x,
                                child_y,
                                child_width,
                                Some(child_height.max(0) as usize),
                                stack_box.clip,
                            );
                            stack_box.children.push(child);
                        }
                        child_x += widths[index] as i64 + node.gap as i64;
                    }
                    stack_box
                }
            }
        }
    }
}
/// Replace one scrollbar cell in a painted row, upstream
/// `replaceScrollbarCell`: the glyph takes the target cell's spot, and the
/// underlying background is preserved through an explicit reset so the
/// border foreground never leaks.
fn replace_scrollbar_cell(
    line: &str,
    column: usize,
    total_width: usize,
    replacement: &str,
    preserve_target_background: bool,
) -> String {
    if is_image_line(line) {
        return line.to_string();
    }

    let grapheme_range = get_grapheme_cell_range(line, column);
    let start = grapheme_range.map_or(column, |range| range.start);
    let end = grapheme_range.map_or(column + 1, |range| range.end);
    let before = slice_by_column(line, 0, start, true);
    let target = slice_by_column(line, start, end - start, true);
    let after = slice_by_column(line, end, total_width.saturating_sub(end), true);

    let mut target_prefix = String::new();
    let mut target_index = 0;
    while target_index < target.len() {
        let Some(ansi) = extract_ansi_code(&target, target_index) else {
            break;
        };
        target_prefix.push_str(ansi.code);
        target_index += ansi.length;
    }
    let before_width = visible_width(&before);
    let before_padding = " ".repeat(start.saturating_sub(before_width));
    let cell_padding_before = " ".repeat(column.saturating_sub(start));
    let cell_padding_after = " ".repeat(end.saturating_sub(column + 1));
    let target_style = format!(
        "\x1b[0m\x1b]8;;\x07{}",
        if preserve_target_background {
            get_active_background_ansi(&target_prefix)
        } else {
            String::new()
        }
    );
    format!(
        "{before}{before_padding}{target_style}{cell_padding_before}{replacement}{cell_padding_after}{after}"
    )
}

/// Where the scrollbar glyphs paint for one scroll box, upstream
/// `getScrollbarGeometry`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "paint coordinates ride terminal dimensions (u16); the f64 proportional-thumb math rounds against a bounded track"
)]
pub fn get_scrollbar_geometry(
    box_: &LayoutBox,
    include_hidden_auto: bool,
) -> Option<ScrollbarGeometry> {
    let scroll_view = box_.scroll_view.as_ref()?;
    if box_.rect.width <= 0 || box_.rect.height <= 0 {
        return None;
    }

    let content_height = box_
        .children
        .first()
        .map_or(0, |child| child.rect.height.max(0) as usize);
    let track_height = box_.rect.height.max(0) as usize;
    let can_reveal_hidden_auto = include_hidden_auto
        && scroll_view.scrollbar() == crate::components::scroll_view::ScrollViewScrollbar::Auto
        && content_height > track_height;
    if !scroll_view.is_scrollbar_visible() && !can_reveal_hidden_auto {
        return None;
    }

    let min_thumb_height = track_height.min(2);
    let proportional = ((track_height as f64 * track_height as f64) / content_height.max(1) as f64)
        .round() as usize;
    let thumb_height = min_thumb_height.max(proportional.min(track_height));
    let max_scroll_top = content_height.saturating_sub(track_height);
    let max_thumb_top = track_height.saturating_sub(thumb_height);
    let thumb_offset = if max_scroll_top == 0 {
        0
    } else {
        ((scroll_view.scroll_top() as f64 / max_scroll_top as f64) * max_thumb_top as f64).round()
            as usize
    };
    let column = (box_.rect.x + box_.rect.width - 1) as usize;
    if (column as i64) < box_.clip.x || (column as i64) >= box_.clip.x + box_.clip.width {
        return None;
    }

    Some(ScrollbarGeometry {
        column,
        track_top: box_.rect.y.max(0) as usize,
        track_height,
        thumb_top: (box_.rect.y.max(0) as usize) + thumb_offset,
        thumb_height,
        max_scroll_top,
    })
}

/// Paint the scrollbar glyphs over one scroll box's rows, upstream
/// `paintScrollbar`.
#[expect(
    clippy::cast_possible_wrap,
    reason = "paint coordinates ride terminal dimensions (u16); the i64 clip keeps scrolled rows negative"
)]
fn paint_scrollbar(box_: &LayoutBox, screen: &mut [String], total_width: usize) {
    let Some(scroll_view) = box_.scroll_view.as_ref() else {
        return;
    };
    let Some(geometry) = get_scrollbar_geometry(box_, false) else {
        return;
    };

    for offset in 0..geometry.track_height {
        let row = geometry.track_top + offset;
        if (row as i64) < box_.clip.y
            || (row as i64) >= box_.clip.y + box_.clip.height
            || row >= screen.len()
        {
            continue;
        }
        let is_thumb =
            row >= geometry.thumb_top && row < geometry.thumb_top + geometry.thumb_height;
        let glyph = if is_thumb {
            if scroll_view.is_scrollbar_active() {
                "█"
            } else {
                "┃"
            }
        } else {
            "│"
        };
        let replacement = if is_thumb {
            scroll_view.scrollbar_thumb_style()(glyph)
        } else {
            scroll_view.scrollbar_track_style()(glyph)
        };
        screen[row] = replace_scrollbar_cell(
            screen.get(row).map_or("", String::as_str),
            geometry.column,
            total_width,
            &replacement,
            scroll_view.scrollbar() != crate::components::scroll_view::ScrollViewScrollbar::Always,
        );
    }
}

/// Paint one box and its children onto the screen rows, upstream
/// `paintBox`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "paint coordinates ride terminal dimensions (u16) on both sides; the i64 rectangle keeps scrolled translations negative without wrapping"
)]
fn paint_box(box_: &LayoutBox, screen: &mut [String], total_width: usize) {
    if let Some(lines) = &box_.lines {
        let offset = box_.line_offset.unwrap_or(0);
        let first_row = (box_.rect.y).max(box_.clip.y).max(0);
        let last_row = (box_.rect.y + box_.rect.height)
            .min(box_.clip.y + box_.clip.height)
            .min(screen.len() as i64);
        for row in first_row..last_row {
            let row = row as usize;
            let source_index = (offset as i64 + row as i64 - box_.rect.y) as usize;
            let Some(source_line) = lines.get(source_index) else {
                continue;
            };
            let mut line = osc133_zone_prefix().replace(source_line, "").into_owned();
            if let Some(metadata) = get_kitty_image_metadata(&line) {
                let clip_bottom = (screen.len() as i64).min(box_.clip.y + box_.clip.height);
                let visible_rows = (clip_bottom - row as i64).max(0) as usize;
                if visible_rows < metadata.rows {
                    line = crop_kitty_image_line(&line, 0, visible_rows);
                }
            }
            // Fast path: a full-width box painting onto an untouched row can
            // use the source line directly. Compositing would rebuild the row
            // string through ANSI/grapheme segmentation every frame; padding
            // is unnecessary because rows are written with erase-line and the
            // final width clamp still truncates over-wide lines.
            if box_.rect.x == 0
                && box_.rect.width >= total_width as i64
                && (is_image_line(&line) || screen[row].is_empty())
            {
                screen[row] = line;
            } else {
                screen[row] = composite_tui_line(
                    screen.get(row).map_or("", String::as_str),
                    &line,
                    box_.rect.x.max(0) as usize,
                    box_.rect.width.max(0) as usize,
                    total_width,
                );
            }
        }
    }
    for child in &box_.children {
        paint_box(child, screen, total_width);
    }

    if let (Some(scroll_view), Some(content_lines)) =
        (&box_.scroll_view, &box_.scroll_content_lines)
        && scroll_view.scroll_top() > 0
        && box_.rect.height > 0
    {
        let scroll_top = scroll_view.scroll_top();
        for image_row in (0..scroll_top).rev() {
            let image_line = content_lines.get(image_row).map_or("", String::as_str);
            if let Some(metadata) = get_kitty_image_metadata(image_line) {
                let hidden_rows = scroll_top - image_row;
                if hidden_rows < metadata.rows {
                    let visible_rows =
                        (box_.rect.height.max(0) as usize).min(metadata.rows - hidden_rows);
                    let cropped = crop_kitty_image_line(image_line, hidden_rows, visible_rows);
                    if box_.rect.x == 0 && box_.rect.width >= total_width as i64 {
                        screen[box_.rect.y.max(0) as usize] = cropped;
                    }
                }
                break;
            }
            if image_line.is_empty() {
                break;
            }
        }
    }

    paint_scrollbar(box_, screen, total_width);
}

/// Build one layout frame and paint it, upstream `renderLayoutFrame`.
#[must_use]
#[expect(
    clippy::cast_possible_wrap,
    reason = "the frame origin rides terminal dimensions (u16)"
)]
pub fn render_layout_frame(
    root: &Rc<dyn Component>,
    width: usize,
    height: usize,
    request_render: &RenderRequest,
) -> LayoutFrame {
    let safe_width = width.max(1);
    let safe_height = height.max(1);
    let context = LayoutContext {
        viewport: crate::layout_node::LayoutViewport {
            width: safe_width,
            height: safe_height,
        },
        render_cache: RefCell::new(HashMap::new()),
        request_render,
        primary_scroll_view: RefCell::new(None),
    };
    let root_box = layout_component(
        &context,
        root,
        0,
        0,
        safe_width,
        Some(safe_height),
        LayoutRect {
            x: 0,
            y: 0,
            width: safe_width as i64,
            height: safe_height as i64,
        },
    );
    let mut lines = vec![String::new(); safe_height];
    paint_box(&root_box, &mut lines, safe_width);
    LayoutFrame {
        root: root_box,
        width: safe_width,
        height: safe_height,
        lines,
        primary_scroll_view: context.primary_scroll_view.take(),
    }
}

const fn contains_point(rect: LayoutRect, x: i64, y: i64) -> bool {
    x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height
}

/// Return the visual hit path from the deepest component to the layout
/// root, upstream `getLayoutBoxesAt`.
#[must_use]
pub fn get_layout_boxes_at(frame: &LayoutFrame, x: i64, y: i64) -> Vec<&LayoutBox> {
    fn visit<'a>(
        box_: &'a LayoutBox,
        depth: i64,
        x: i64,
        y: i64,
        result: &mut Vec<(&'a LayoutBox, i64)>,
    ) {
        if !contains_point(box_.clip, x, y) {
            return;
        }
        result.push((box_, depth));
        for child in &box_.children {
            visit(child, depth + 1, x, y, result);
        }
    }
    let mut result: Vec<(&LayoutBox, i64)> = Vec::new();
    visit(&frame.root, 0, x, y, &mut result);
    result.sort_by_key(|(box_, depth)| (std::cmp::Reverse(box_.layer), std::cmp::Reverse(*depth)));
    result.into_iter().map(|(box_, _)| box_).collect()
}

/// Find the box a scroll state laid out into, upstream `getScrollViewBox`.
#[must_use]
pub fn get_scroll_view_box<'a>(
    frame: &'a LayoutFrame,
    scroll_view: &'a ScrollStateHandle,
) -> Option<&'a LayoutBox> {
    fn visit<'a>(box_: &'a LayoutBox, scroll_view: &ScrollStateHandle) -> Option<&'a LayoutBox> {
        if let Some(box_scroll_view) = &box_.scroll_view
            && Arc::ptr_eq(box_scroll_view, scroll_view)
        {
            return Some(box_);
        }
        box_.children
            .iter()
            .find_map(|child| visit(child, scroll_view))
    }
    visit(&frame.root, scroll_view)
}

/// The scroll views whose clip contains the point, deepest first, upstream
/// `getScrollViewsAt`.
#[must_use]
pub fn get_scroll_views_at(frame: &LayoutFrame, x: usize, y: usize) -> Vec<ScrollStateHandle> {
    fn visit(
        box_: &LayoutBox,
        depth: i64,
        x: i64,
        y: i64,
        result: &mut Vec<(ScrollStateHandle, i64)>,
    ) {
        if !contains_point(box_.clip, x, y) {
            return;
        }
        if let Some(scroll_view) = &box_.scroll_view
            && contains_point(box_.rect, x, y)
        {
            result.push((Arc::clone(scroll_view), depth));
        }
        for child in &box_.children {
            visit(child, depth + 1, x, y, result);
        }
    }
    let mut result: Vec<(ScrollStateHandle, i64)> = Vec::new();
    visit(
        &frame.root,
        0,
        i64::try_from(x).unwrap_or(i64::MAX),
        i64::try_from(y).unwrap_or(i64::MAX),
        &mut result,
    );
    result.sort_by_key(|(_, depth)| std::cmp::Reverse(*depth));
    result
        .into_iter()
        .map(|(scroll_view, _)| scroll_view)
        .collect()
}
