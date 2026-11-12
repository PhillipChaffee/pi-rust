//! The chat viewport at pin `60e7e76`: the upstream scrollbar-default case
//! plus boundary coverage for the dock composition and root split.

use std::rc::Rc;

use pi_coding_agent::modes::interactive::chat_viewport::{
    ChatViewportOptions, create_chat_viewport,
};
use pi_tui::components::{ScrollViewScrollbar, Text};
use pi_tui::tui::{Component, Container};

fn component() -> Rc<dyn Component> {
    Rc::new(Container::default())
}

fn options() -> ChatViewportOptions {
    ChatViewportOptions {
        document: component(),
        pending_messages: component(),
        status: component(),
        editor: component(),
        footer: component(),
        widgets_above: None,
        widgets_below: None,
        scrollbar: None,
        scrollbar_track_style: None,
        scrollbar_thumb_style: None,
    }
}

#[test]
fn defaults_the_transcript_scrollbar_to_auto_and_accepts_overrides() {
    let automatic = create_chat_viewport(options());
    let mut hidden_options = options();
    hidden_options.scrollbar = Some(ScrollViewScrollbar::Hidden);
    let hidden = create_chat_viewport(hidden_options);

    assert_eq!(automatic.transcript.scrollbar(), ScrollViewScrollbar::Auto);
    assert_eq!(hidden.transcript.scrollbar(), ScrollViewScrollbar::Hidden);
}

#[test]
fn transcript_is_the_follow_end_scroll_view() {
    let viewport = create_chat_viewport(options());

    assert!(viewport.transcript.is_following_end());
    assert_eq!(viewport.transcript.scrollbar(), ScrollViewScrollbar::Auto);
}

#[test]
fn root_splits_the_transcript_from_the_auto_sized_dock() {
    let viewport = create_chat_viewport(options());
    let root_children = viewport.root.children();
    assert_eq!(root_children.len(), 2, "transcript plus dock");
    let transcript_component: Rc<dyn Component> = viewport.transcript.clone();
    assert!(
        Rc::ptr_eq(&root_children[0], &transcript_component),
        "the first root child is the transcript"
    );

    let dock_children = root_children[1].children();
    assert_eq!(dock_children.len(), 4, "no optional widgets");
}

#[test]
fn dock_stacks_pending_status_widgets_editor_footer_in_order() {
    let pending_messages = component();
    let status = component();
    let widgets_above = component();
    let editor = component();
    let widgets_below = component();
    let footer = component();
    let mut options = options();
    options.pending_messages = Rc::clone(&pending_messages);
    options.status = Rc::clone(&status);
    options.widgets_above = Some(Rc::clone(&widgets_above));
    options.editor = Rc::clone(&editor);
    options.widgets_below = Some(Rc::clone(&widgets_below));
    options.footer = Rc::clone(&footer);

    let viewport = create_chat_viewport(options);
    let dock = viewport.root.children()[1].clone();
    let children = dock.children();
    let expected = [
        pending_messages,
        status,
        widgets_above,
        editor,
        widgets_below,
        footer,
    ];
    assert_eq!(children.len(), expected.len());
    for (index, expected) in expected.iter().enumerate() {
        assert!(
            Rc::ptr_eq(&children[index], expected),
            "dock child {index} keeps its upstream position"
        );
    }
}

#[test]
fn root_renders_the_transcript_above_the_dock() {
    let mut options = options();
    options.document = Rc::new(Text::with_padding("document", 0, 0));
    options.pending_messages = Rc::new(Text::with_padding("pending", 0, 0));
    options.status = Rc::new(Text::with_padding("status", 0, 0));
    options.editor = Rc::new(Text::with_padding("editor", 0, 0));
    options.footer = Rc::new(Text::with_padding("footer", 0, 0));

    let viewport = create_chat_viewport(options);
    let rendered = viewport.root.render(10);
    let visible: Vec<&str> = rendered
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect();
    assert_eq!(
        visible,
        ["document", "pending", "status", "editor", "footer"]
    );
}
