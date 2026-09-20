//! The stack machinery of `packages/tui/src/components/stack.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#44).
//!
//! It carries the shared entries bookkeeping behind the `VStack` and
//! `HStack` structs (the public types in [`super`]) and the size allocator.
//!
//! Restatements against upstream (survey flags in map ticket "Survey the
//! tui package"):
//!
//! - `abstract class Stack extends Container` becomes `StackCore`, the
//!   entries/gap/align bookkeeping both stacks compose; `VStack`/`HStack`
//!   are the public structs. The `layoutType` abstract property becomes a
//!   [`StackKind`] on the core, and the `LAYOUT_NODE` symbol becomes
//!   [`Component::layout_node`] answering [`crate::layout_node::LayoutNode::Stack`].
//! - Entry options normalize at `addChild` exactly as upstream's
//!   `normalizeSize` does: non-finite values fall back (`grow` 0, `shrink`
//!   1, `minSize` 0, `maxSize` `usize::MAX`), and every value floors at
//!   zero.
//! - `StackChild`'s `Component | StackEntry` union (duck-typed on
//!   `"render" in child`) becomes [`StackChild`], an explicit enum.

use std::cell::RefCell;
use std::rc::Rc;

use crate::layout_node::{
    Basis, EntryVisibility, LayoutViewport, StackAlign, StackKind, StackLayoutEntry,
};
use crate::tui::Component;

/// Per-child sizing options, upstream `StackEntryOptions`.
#[derive(Clone, Default)]
pub struct StackEntryOptions {
    /// Sizing basis, upstream `basis?: number | "auto"`; `None` is auto.
    pub basis: Option<Basis>,
    /// Grow weight, upstream `grow`; `None` keeps the 0 default.
    pub grow: Option<u32>,
    /// Shrink weight, upstream `shrink`; `None` keeps the 1 default.
    pub shrink: Option<u32>,
    /// Lower size bound, upstream `minSize`.
    pub min_size: Option<u32>,
    /// Upper size bound, upstream `maxSize`.
    pub max_size: Option<u32>,
    /// Visibility gate over the layout viewport, upstream `visible`.
    pub visible: Option<EntryVisibility>,
}

impl std::fmt::Debug for StackEntryOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackEntryOptions")
            .field("basis", &self.basis)
            .field("grow", &self.grow)
            .field("shrink", &self.shrink)
            .field("min_size", &self.min_size)
            .field("max_size", &self.max_size)
            .field("visible", &self.visible.is_some())
            .finish()
    }
}

/// A stack member with options, upstream `StackEntry`.
#[derive(Clone)]
pub struct StackEntry {
    /// The member component.
    pub component: Rc<dyn Component>,
    /// The member's sizing options.
    pub options: StackEntryOptions,
}

impl std::fmt::Debug for StackEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackEntry").finish_non_exhaustive()
    }
}

/// A stack child, upstream `StackChild = Component | StackEntry`.
#[derive(Clone)]
pub enum StackChild {
    /// A plain member, upstream passing a bare component.
    Component(Rc<dyn Component>),
    /// A member with sizing options, upstream `StackEntry`.
    Entry(StackEntry),
}

impl std::fmt::Debug for StackChild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Component(_) => f.write_str("StackChild::Component"),
            Self::Entry(_) => f.debug_struct("StackChild::Entry").finish_non_exhaustive(),
        }
    }
}

impl From<Rc<dyn Component>> for StackChild {
    fn from(component: Rc<dyn Component>) -> Self {
        Self::Component(component)
    }
}

impl StackChild {
    /// The entry variant, upstream passing `{ component, ...options }`.
    pub fn entry(component: Rc<dyn Component>, options: StackEntryOptions) -> Self {
        Self::Entry(StackEntry { component, options })
    }

    /// The plain-component variant, upstream passing a bare component.
    pub fn component(component: Rc<dyn Component>) -> Self {
        Self::Component(component)
    }
}

/// Stack constructor options, upstream `StackOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct StackOptions {
    /// Blank rows or columns between members, upstream `gap`.
    pub gap: Option<u32>,
    /// Cross-axis alignment, upstream `align` (default stretch).
    pub align: Option<StackAlign>,
}

/// Normalize a size option, upstream `normalizeSize`: `None` and non-finite
/// values fall back, everything else floors at zero.
fn normalize_size(value: Option<u32>, fallback: u32) -> u32 {
    value.unwrap_or(fallback)
}

/// The stack machinery the two stacks share, upstream's `protected`
/// fields of `abstract class Stack extends Container`: the container's
/// child list beside the entries the layout engine consumes.
pub(crate) struct StackCore {
    /// The stack axis, upstream's abstract `layoutType`.
    kind: StackKind,
    /// The child components, upstream's inherited `Container.children`.
    children: RefCell<Vec<Rc<dyn Component>>>,
    entries: RefCell<Vec<StackLayoutEntry>>,
    /// Blank rows or columns between members, normalized at construction.
    pub(crate) gap: usize,
    /// Cross-axis alignment.
    pub(crate) align: StackAlign,
}

impl StackCore {
    /// Upstream's `Stack` constructor over normalized options.
    pub(crate) fn new(kind: StackKind, options: StackOptions) -> Self {
        Self {
            kind,
            children: RefCell::new(Vec::new()),
            entries: RefCell::new(Vec::new()),
            gap: usize::try_from(options.gap.unwrap_or(0)).unwrap_or(usize::MAX),
            align: options.align.unwrap_or_default(),
        }
    }

    /// Upstream `Stack.addChild(component, options?)`: push the component
    /// and record its normalized entry.
    pub(crate) fn add_child(&self, component: Rc<dyn Component>, options: StackEntryOptions) {
        self.children.borrow_mut().push(Rc::clone(&component));
        self.entries.borrow_mut().push(StackLayoutEntry {
            component,
            basis: options.basis,
            grow: normalize_size(options.grow, 0),
            shrink: normalize_size(options.shrink, 1),
            min_size: usize::try_from(normalize_size(options.min_size, 0)).unwrap_or(usize::MAX),
            // An unset bound is unbounded, upstream's MAX_SAFE_INTEGER
            // normalization.
            max_size: options.max_size.map_or(usize::MAX, |value| {
                usize::try_from(value).unwrap_or(usize::MAX)
            }),
            visible: options.visible,
        });
    }

    /// Upstream `Stack.removeChild`.
    pub(crate) fn remove_child(&self, component: &Rc<dyn Component>) {
        self.children
            .borrow_mut()
            .retain(|child| !Rc::ptr_eq(child, component));
        self.entries
            .borrow_mut()
            .retain(|entry| !Rc::ptr_eq(&entry.component, component));
    }

    /// Upstream `Stack.clear`.
    pub(crate) fn clear(&self) {
        self.children.borrow_mut().clear();
        self.entries.borrow_mut().clear();
    }

    /// The raw child list, upstream's inherited `Container.children`.
    pub(crate) fn children(&self) -> Vec<Rc<dyn Component>> {
        self.children.borrow().clone()
    }

    /// Upstream `Container.invalidate` over the children.
    pub(crate) fn invalidate_children(&self) {
        for child in self.children.borrow().iter() {
            child.invalidate();
        }
    }

    /// Upstream `Container.handleMouse`: the stack's overridden render
    /// never commits the mouse-layout cache, so every dispatch re-measures
    /// fresh — upstream's `mouseLayout?.width === event.width` miss path.
    pub(crate) fn handle_mouse(
        &self,
        event: &crate::tui::TuiMouseEvent,
    ) -> Option<crate::tui::TuiMouseEventResult> {
        if event.y >= event.height {
            return None;
        }
        let mut child_y: usize = 0;
        for child in self.children.borrow().iter() {
            let child_height = child.render(usize::from(event.width)).len();
            if usize::from(event.y) >= child_y && usize::from(event.y) < child_y + child_height {
                let child_event = crate::tui::TuiMouseEvent {
                    y: event.y - u16::try_from(child_y).unwrap_or(u16::MAX),
                    height: u16::try_from(child_height).unwrap_or(u16::MAX),
                    ..event.clone()
                };
                return crate::tui::dispatch_mouse_event(child, &child_event);
            }
            child_y += child_height;
        }
        None
    }

    /// Upstream `[LAYOUT_NODE]()`: a snapshot of the entries with the
    /// stack's kind, gap, and align.
    pub(crate) fn layout_node(&self) -> crate::layout_node::StackLayoutNode {
        crate::layout_node::StackLayoutNode {
            kind: self.kind,
            entries: self.entries.borrow().clone(),
            gap: self.gap,
            align: self.align,
        }
    }

    /// The visible entries at a viewport, upstream `visibleStackEntries`.
    pub(crate) fn visible_entries(&self, viewport: &LayoutViewport) -> Vec<StackLayoutEntry> {
        visible_stack_entries(&self.entries.borrow(), viewport)
    }
}

/// Filter entries by their visibility gates, upstream
/// `visibleStackEntries`.
#[must_use]
pub fn visible_stack_entries(
    entries: &[StackLayoutEntry],
    viewport: &LayoutViewport,
) -> Vec<StackLayoutEntry> {
    entries
        .iter()
        .filter(|entry| {
            entry
                .visible
                .as_ref()
                .is_none_or(|visible| visible(viewport))
        })
        .cloned()
        .collect()
}

/// Clamp a computed size into the entry's min/max bounds, upstream
/// `clampSize`.
fn clamp_size(size: i64, entry: &StackLayoutEntry) -> usize {
    let min = entry.min_size;
    let max = entry.max_size.max(min);
    let size = usize::try_from(size.max(0)).unwrap_or(usize::MAX);
    size.clamp(min, max)
}

/// Distribute `amount` cells across the entries by grow or shrink weight,
/// upstream `distribute`. The loop terminates when no candidate has
/// capacity left, exactly as the upstream `distributed === 0` bail does.
fn distribute(
    sizes: &mut [usize],
    entries: &[StackLayoutEntry],
    amount: usize,
    mode: DistributeMode,
) {
    let mut remaining = amount;
    while remaining > 0 {
        let candidates: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(index, entry)| match mode {
                DistributeMode::Grow => entry.grow > 0 && sizes[*index] < entry.max_size,
                DistributeMode::Shrink => entry.shrink > 0 && sizes[*index] > entry.min_size,
            })
            .map(|(index, _)| index)
            .collect();
        if candidates.is_empty() {
            return;
        }

        let total_weight: u64 = candidates
            .iter()
            .map(|index| weight(&entries[*index], sizes[*index], mode))
            .sum();
        let mut distributed = 0;
        for index in &candidates {
            if remaining == 0 {
                break;
            }
            let entry = &entries[*index];
            let weight = weight(entry, sizes[*index], mode);
            let remaining_weight = u64::from(u32::try_from(remaining).unwrap_or(u32::MAX));
            let proposed = usize::try_from(remaining_weight * weight / total_weight)
                .unwrap_or(usize::MAX)
                .max(1);
            let capacity = match mode {
                DistributeMode::Grow => entry.max_size.saturating_sub(sizes[*index]),
                DistributeMode::Shrink => sizes[*index].saturating_sub(entry.min_size),
            };
            let delta = remaining.min(proposed).min(capacity);
            if delta == 0 {
                continue;
            }
            match mode {
                DistributeMode::Grow => sizes[*index] += delta,
                DistributeMode::Shrink => sizes[*index] -= delta,
            }
            remaining -= delta;
            distributed += delta;
        }
        if distributed == 0 {
            return;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DistributeMode {
    Grow,
    Shrink,
}

/// The distribution weight, upstream's mode-dependent expression: grow
/// weight, or shrink weight scaled by the current size.
fn weight(entry: &StackLayoutEntry, size: usize, mode: DistributeMode) -> u64 {
    match mode {
        DistributeMode::Grow => u64::from(entry.grow),
        DistributeMode::Shrink => {
            u64::from(entry.shrink) * u64::try_from(size.max(1)).unwrap_or(u64::MAX)
        }
    }
}

/// Allocate stack member sizes, upstream `allocateStackSizes`: clamp the
/// basis (or intrinsic measurement) into min/max, then grow or shrink into
/// the available size after the gaps.
#[must_use]
pub fn allocate_stack_sizes(
    entries: &[StackLayoutEntry],
    intrinsic_sizes: &[usize],
    available_size: Option<usize>,
    gap: usize,
) -> Vec<usize> {
    let mut sizes: Vec<usize> = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let intrinsic = intrinsic_sizes.get(index).copied().unwrap_or(0);
            let basis = match entry.basis {
                None | Some(Basis::Auto) => i64::try_from(intrinsic).unwrap_or(i64::MAX),
                Some(Basis::Cells(cells)) => cells,
            };
            clamp_size(basis, entry)
        })
        .collect();
    let Some(available_size) = available_size else {
        return sizes;
    };

    let content_size =
        available_size.saturating_sub(gap.saturating_mul(entries.len().saturating_sub(1)));
    let total: usize = sizes.iter().sum();
    if total < content_size {
        distribute(
            &mut sizes,
            entries,
            content_size - total,
            DistributeMode::Grow,
        );
    } else if total > content_size {
        distribute(
            &mut sizes,
            entries,
            total - content_size,
            DistributeMode::Shrink,
        );
    }
    sizes
}
