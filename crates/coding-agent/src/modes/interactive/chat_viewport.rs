//! The fullscreen chat viewport, upstream
//! `src/modes/interactive/chat-viewport.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! The transcript is the frame's primary scroll view — following the
//! content end, chaining overscroll outward, and painting the scrollbar
//! `"auto"` unless the caller overrides it. The dock beneath it stacks the
//! pending-message rows, the status row, the above-editor widgets, the
//! editor (which keeps three rows even under pressure), the below-editor
//! widgets, and the footer. The root splits the viewport between the
//! transcript (grow-one, min one row) and the auto-sized dock.
//!
//! Port restatement: upstream's absent optional fields are `None` here, and
//! its component references are `Rc<dyn Component>`; the absent branches
//! leave the widget out of the dock rather than sizing a placeholder.

use std::rc::Rc;

use pi_tui::components::{
    ColorFn, FollowMode, ScrollView, ScrollViewOptions, ScrollViewScrollbar, StackChild,
    StackEntryOptions, StackOptions, VStack,
};
use pi_tui::layout_node::{Basis, Overscroll};
use pi_tui::tui::Component;

/// The viewport constructor options, upstream `ChatViewportOptions`.
#[derive(Clone)]
pub struct ChatViewportOptions {
    /// The transcript document the primary scroll view wraps.
    pub document: Rc<dyn Component>,
    /// Rows for queued user input awaiting the model's turn, upstream
    /// `pendingMessages`.
    pub pending_messages: Rc<dyn Component>,
    /// The transient status row, upstream `status`.
    pub status: Rc<dyn Component>,
    /// The input editor, upstream `editor`; it keeps three rows.
    pub editor: Rc<dyn Component>,
    /// The footer beneath the editor, upstream `footer`.
    pub footer: Rc<dyn Component>,
    /// The widget stack above the editor, upstream `widgetsAbove`.
    pub widgets_above: Option<Rc<dyn Component>>,
    /// The widget stack below the editor, upstream `widgetsBelow`.
    pub widgets_below: Option<Rc<dyn Component>>,
    /// The scrollbar mode; upstream `scrollbar ?? "auto"`.
    pub scrollbar: Option<ScrollViewScrollbar>,
    /// The scrollbar track glyph style, upstream `scrollbarTrackStyle`.
    pub scrollbar_track_style: Option<ColorFn>,
    /// The scrollbar thumb glyph style, upstream `scrollbarThumbStyle`.
    pub scrollbar_thumb_style: Option<ColorFn>,
}

impl std::fmt::Debug for ChatViewportOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatViewportOptions")
            .field("scrollbar", &self.scrollbar)
            .field(
                "scrollbar_track_style",
                &self.scrollbar_track_style.is_some(),
            )
            .field(
                "scrollbar_thumb_style",
                &self.scrollbar_thumb_style.is_some(),
            )
            .field("widgets_above", &self.widgets_above.is_some())
            .field("widgets_below", &self.widgets_below.is_some())
            .finish_non_exhaustive()
    }
}

/// The created frame pieces, upstream `ChatViewport`.
#[derive(Clone)]
pub struct ChatViewport {
    /// The root stack mounted on the frame, upstream `root`.
    pub root: Rc<dyn Component>,
    /// The primary scroll view the frame drives, upstream `transcript`.
    pub transcript: Rc<ScrollView>,
}

impl std::fmt::Debug for ChatViewport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatViewport")
            .field("root", &Rc::as_ptr(&self.root))
            .field("transcript", &Rc::as_ptr(&self.transcript))
            .finish_non_exhaustive()
    }
}

/// A dock entry with the upstream sizing: `shrink: 1` and the given minimum
/// row count, everything else at its default.
fn dock_entry(component: Rc<dyn Component>, min_size: u32) -> StackChild {
    StackChild::entry(
        component,
        StackEntryOptions {
            shrink: Some(1),
            min_size: Some(min_size),
            ..StackEntryOptions::default()
        },
    )
}

/// Build the shared fullscreen transcript and fixed input-dock layout,
/// upstream `createChatViewport`.
#[must_use]
pub fn create_chat_viewport(options: ChatViewportOptions) -> ChatViewport {
    let transcript = ScrollView::new(
        options.document,
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            overscroll: Some(Overscroll::Chain),
            scrollbar: Some(options.scrollbar.unwrap_or(ScrollViewScrollbar::Auto)),
            scrollbar_track_style: options.scrollbar_track_style,
            scrollbar_thumb_style: options.scrollbar_thumb_style,
            ..ScrollViewOptions::default()
        },
    );

    let mut dock_children = vec![
        dock_entry(options.pending_messages, 0),
        dock_entry(options.status, 0),
    ];
    if let Some(widgets_above) = options.widgets_above {
        dock_children.push(dock_entry(widgets_above, 0));
    }
    dock_children.push(dock_entry(options.editor, 3));
    if let Some(widgets_below) = options.widgets_below {
        dock_children.push(dock_entry(widgets_below, 0));
    }
    dock_children.push(dock_entry(options.footer, 0));

    let dock: Rc<dyn Component> = VStack::new(dock_children, StackOptions::default());
    let transcript_component: Rc<dyn Component> = transcript.clone();
    let root = VStack::new(
        vec![
            StackChild::entry(
                transcript_component,
                StackEntryOptions {
                    basis: Some(Basis::Cells(0)),
                    grow: Some(1),
                    shrink: Some(1),
                    min_size: Some(1),
                    ..StackEntryOptions::default()
                },
            ),
            StackChild::entry(
                dock,
                StackEntryOptions {
                    basis: Some(Basis::Auto),
                    grow: Some(0),
                    shrink: Some(1),
                    min_size: Some(1),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    );

    ChatViewport { root, transcript }
}
