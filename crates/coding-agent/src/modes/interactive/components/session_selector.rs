//! Session picker and its header, upstream's `session-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#202).
//!
//! Async orchestration belongs to the host, matching the model-picker slice
//! (#200): the component takes the current-session snapshot, asks for the
//! all-session list through [`SessionSelectorOptions::on_request_all_sessions`],
//! and receives every completion through
//! [`SessionSelectorComponent::set_all_sessions`] or
//! [`SessionSelectorComponent::set_load_error`]. It still owns the loading
//! flag and the scope guard that drops a late completion after the user
//! toggled back to the current folder. Delete and rename ask the host to
//! refresh through [`SessionSelectorOptions::on_refresh_sessions`].
//!
//! Restatements: upstream awaited loaders and the rename callback; the host
//! now runs both, so status auto-hide rides an expiry checked at render
//! rather than a timer, and an in-flight load reports through the same
//! setters a finished one uses.

use std::cell::{Cell, RefCell};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable, TuiMouseEvent, TuiMouseEventResult};
use pi_tui::utils::{truncate_to_width, visible_width};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{key_hint, key_text};
use super::session_selector_search::{
    NameFilter, SortMode, filter_and_sort_sessions, has_session_name,
};
use crate::config::home_dir;
use crate::modes::interactive::theme::theme::{bold, theme};
use crate::session_manager::SessionInfo;
use crate::utils::paths::canonicalize_path;

/// Which session directory the picker lists, upstream's `SessionScope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SessionScope {
    /// Sessions started in the picker's working directory.
    #[default]
    Current,
    /// Sessions from every project directory.
    All,
}

/// How [`delete_session_file`] removed the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionDeleteMethod {
    /// The `trash` CLI accepted the path, or the file vanished under it.
    Trash,
    /// Permanent [`fs::remove_file`] fallback.
    Unlink,
}

/// The outcome of [`delete_session_file`], upstream's
/// `{ ok, method, error? }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionDeleteResult {
    /// Whether the session file is gone.
    pub ok: bool,
    /// The mechanism that ran (or the one that failed).
    pub method: SessionDeleteMethod,
    /// The unlink failure, suffixed with the first `trash` diagnostic.
    pub error: Option<String>,
}

impl SessionDeleteResult {
    const fn deleted(method: SessionDeleteMethod) -> Self {
        Self {
            ok: true,
            method,
            error: None,
        }
    }
}

/// Deletes a session file, trying `trash` first and falling back to a
/// permanent unlink.
///
/// A missing file counts as deleted because the `trash` failure made the
/// existence check fail, upstream's success arm.
#[must_use]
pub fn delete_session_file(session_path: &str) -> SessionDeleteResult {
    let mut command = Command::new("trash");
    if session_path.starts_with('-') {
        command.arg("--");
    }
    let output = command.arg(session_path).output();

    let mut trash_hint: Option<String> = None;
    let trash_ok = match output {
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stderr = stderr.trim();
            if !stderr.is_empty() {
                let first = stderr.lines().next().unwrap_or(stderr);
                let hint: String = first.chars().take(200).collect();
                trash_hint = Some(format!("trash: {hint}"));
            }
            output.status.success()
        }
        Err(error) => {
            trash_hint = Some(format!("trash: {error}"));
            false
        }
    };

    if trash_ok || !Path::new(session_path).exists() {
        return SessionDeleteResult::deleted(SessionDeleteMethod::Trash);
    }

    match fs::remove_file(session_path) {
        Ok(()) => SessionDeleteResult::deleted(SessionDeleteMethod::Unlink),
        Err(error) => {
            let message =
                trash_hint.map_or_else(|| error.to_string(), |hint| format!("{error} ({hint})"));
            SessionDeleteResult {
                ok: false,
                method: SessionDeleteMethod::Unlink,
                error: Some(message),
            }
        }
    }
}

fn shorten_path(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let home = home_dir();
    if !home.is_empty() && path.starts_with(&home) {
        return format!("~{}", &path[home.len()..]);
    }
    path.to_string()
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            i64::try_from(since.as_millis()).unwrap_or(i64::MAX)
        })
}

/// Upstream's `formatSessionDate`: a coarse age label, computed at render.
fn format_session_date(modified_millis: i64, now_millis: i64) -> String {
    let diff = now_millis.saturating_sub(modified_millis);
    let minutes = diff / 60_000;
    if minutes < 1 {
        return "now".to_string();
    }
    if minutes < 60 {
        return format!("{minutes}m");
    }
    let hours = diff / 3_600_000;
    if hours < 24 {
        return format!("{hours}h");
    }
    let days = diff / 86_400_000;
    if days < 7 {
        return format!("{days}d");
    }
    if days < 30 {
        return format!("{}w", days / 7);
    }
    if days < 365 {
        return format!("{}mo", days / 30);
    }
    format!("{}y", days / 365)
}

fn canonical(path: &str) -> String {
    if path.is_empty() {
        path.to_string()
    } else {
        canonicalize_path(path)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatusKind {
    Info,
    Error,
}

#[derive(Debug, Clone)]
struct StatusMessage {
    kind: StatusKind,
    message: String,
    expires_at: Option<Instant>,
}

struct SessionSelectorHeader {
    scope: Cell<SessionScope>,
    sort_mode: Cell<SortMode>,
    name_filter: Cell<NameFilter>,
    loading: Cell<bool>,
    load_progress: Cell<Option<(u64, u64)>>,
    show_path: Cell<bool>,
    show_rename_hint: Cell<bool>,
    confirming_delete_path: RefCell<Option<String>>,
    status: RefCell<Option<StatusMessage>>,
}

impl std::fmt::Debug for SessionSelectorHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionSelectorHeader")
            .field("scope", &self.scope)
            .field("sort_mode", &self.sort_mode)
            .field("name_filter", &self.name_filter)
            .field("loading", &self.loading)
            .finish_non_exhaustive()
    }
}

impl SessionSelectorHeader {
    const fn new(scope: SessionScope, sort_mode: SortMode, name_filter: NameFilter) -> Self {
        Self {
            scope: Cell::new(scope),
            sort_mode: Cell::new(sort_mode),
            name_filter: Cell::new(name_filter),
            loading: Cell::new(false),
            load_progress: Cell::new(None),
            show_path: Cell::new(false),
            show_rename_hint: Cell::new(false),
            confirming_delete_path: RefCell::new(None),
            status: RefCell::new(None),
        }
    }

    fn set_loading(&self, loading: bool) {
        self.loading.set(loading);
        // Progress belongs to the current load; every loading transition
        // clears it, upstream's `setLoading`.
        self.load_progress.set(None);
    }

    fn set_status_message(&self, kind: StatusKind, message: &str, auto_hide: Option<Duration>) {
        *self.status.borrow_mut() = Some(StatusMessage {
            kind,
            message: message.to_string(),
            expires_at: auto_hide.map(|duration| Instant::now() + duration),
        });
    }

    fn active_status(&self) -> Option<StatusMessage> {
        let status = self.status.borrow().clone()?;
        if status.expires_at.is_some_and(|at| Instant::now() >= at) {
            *self.status.borrow_mut() = None;
            return None;
        }
        Some(status)
    }

    fn clear_status(&self) {
        *self.status.borrow_mut() = None;
    }

    fn scope_line(&self) -> String {
        if self.loading.get() {
            let progress = self.load_progress.get().map_or_else(
                || "...".to_string(),
                |(loaded, total)| format!("{loaded}/{total}"),
            );
            return format!(
                "{} {}",
                theme().fg("muted", "○ Current Folder |"),
                theme().fg("accent", &format!("Loading {progress}"))
            );
        }
        match self.scope.get() {
            SessionScope::Current => format!(
                "{} {} {}",
                theme().fg("accent", "◉ Current Folder"),
                theme().fg("muted", "|"),
                theme().fg("muted", "○ All")
            ),
            SessionScope::All => format!(
                "{} {} {}",
                theme().fg("muted", "○ Current Folder"),
                theme().fg("muted", "|"),
                theme().fg("accent", "◉ All")
            ),
        }
    }
}

impl Component for SessionSelectorHeader {
    fn render(&self, width: usize) -> Vec<String> {
        let title = match self.scope.get() {
            SessionScope::Current => "Resume Session (Current Folder)",
            SessionScope::All => "Resume Session (All)",
        };
        let sort_label = match self.sort_mode.get() {
            SortMode::Threaded => "Threaded",
            SortMode::Recent => "Recent",
            SortMode::Relevance => "Fuzzy",
        };
        let name_label = match self.name_filter.get() {
            NameFilter::All => "All",
            NameFilter::Named => "Named",
        };
        let sort_text = theme().fg("muted", "Sort: ") + &theme().fg("accent", sort_label);
        let name_text = theme().fg("muted", "Name: ") + &theme().fg("accent", name_label);
        let scope_text = self.scope_line();

        let right = truncate_to_width(
            &format!("{scope_text}  {name_text}  {sort_text}"),
            width,
            "",
            false,
        );
        let available_left = width
            .saturating_sub(visible_width(&right))
            .saturating_sub(1);
        let left = truncate_to_width(&bold(title), available_left, "", false);
        let spacing = width.saturating_sub(visible_width(&left) + visible_width(&right));

        let (hint_line_1, hint_line_2) = if self.confirming_delete_path.borrow().is_some() {
            (
                theme().fg(
                    "error",
                    &truncate_to_width(
                        &format!(
                            "Delete session? {} · {}",
                            key_hint("tui.select.confirm", "confirm"),
                            key_hint("tui.select.cancel", "cancel")
                        ),
                        width,
                        "…",
                        false,
                    ),
                ),
                String::new(),
            )
        } else if let Some(status) = self.active_status() {
            let color = if status.kind == StatusKind::Error {
                "error"
            } else {
                "accent"
            };
            (
                theme().fg(
                    color,
                    &truncate_to_width(&status.message, width, "…", false),
                ),
                String::new(),
            )
        } else {
            let sep = theme().fg("muted", " · ");
            let hint_1 = key_hint("tui.input.tab", "scope")
                + &sep
                + &theme().fg("muted", "re:<pattern> regex · \"phrase\" exact");
            let path_state = if self.show_path.get() {
                "(on)"
            } else {
                "(off)"
            };
            let mut hint_parts = vec![
                key_hint("app.session.toggleSort", "sort"),
                key_hint("app.session.toggleNamedFilter", "named"),
                key_hint("app.session.delete", "delete"),
                key_hint("app.session.togglePath", &format!("path {path_state}")),
            ];
            if self.show_rename_hint.get() {
                hint_parts.push(key_hint("app.session.rename", "rename"));
            }
            (
                truncate_to_width(&hint_1, width, "…", false),
                truncate_to_width(&hint_parts.join(&sep), width, "…", false),
            )
        };

        vec![
            format!("{left}{}{right}", " ".repeat(spacing)),
            hint_line_1,
            hint_line_2,
        ]
    }
}

/// One session in the threaded view; children hold indexes into the forest's
/// node table, not nested boxes.
#[derive(Debug)]
struct SessionTreeNode {
    session: SessionInfo,
    children: Vec<usize>,
    latest_activity: i64,
}

/// A flattened row with the tree drawing metadata, upstream's
/// `FlatSessionNode`.
#[derive(Debug, Clone)]
struct FlatSessionNode {
    session: SessionInfo,
    depth: usize,
    is_last: bool,
    ancestor_continues: Vec<bool>,
}

/// The forest plus its node table, upstream's `SessionTreeNode[]` roots.
struct SessionTree {
    nodes: Vec<SessionTreeNode>,
    roots: Vec<usize>,
}

fn update_latest_activity(nodes: &mut [SessionTreeNode], index: usize) -> i64 {
    let children = nodes[index].children.clone();
    let mut latest = nodes[index].session.modified;
    for child in children {
        latest = latest.max(update_latest_activity(nodes, child));
    }
    nodes[index].latest_activity = latest;
    latest
}

fn sort_nodes(nodes: &mut [SessionTreeNode], indices: &mut [usize]) {
    indices.sort_by(|left, right| {
        nodes[*right]
            .latest_activity
            .cmp(&nodes[*left].latest_activity)
    });
    let ordered = indices.to_vec();
    for index in ordered {
        let mut children = std::mem::take(&mut nodes[index].children);
        sort_nodes(nodes, &mut children);
        nodes[index].children = children;
    }
}

/// Builds the parent/child forest from `parent_session_path`, canonicalizing
/// both sides so symlink aliases thread together. Roots and siblings sort by
/// latest activity anywhere in their subtree, descending.
fn build_session_tree(sessions: &[SessionInfo]) -> SessionTree {
    let mut nodes = Vec::with_capacity(sessions.len());
    let mut by_path = std::collections::HashMap::with_capacity(sessions.len());
    for session in sessions {
        by_path.insert(canonical(&session.path), nodes.len());
        nodes.push(SessionTreeNode {
            session: session.clone(),
            children: Vec::new(),
            latest_activity: session.modified,
        });
    }

    let mut roots = Vec::new();
    for session in sessions {
        let Some(index) = by_path.get(&canonical(&session.path)).copied() else {
            continue;
        };
        match session
            .parent_session_path
            .as_deref()
            .map(canonical)
            .as_ref()
            .and_then(|parent| by_path.get(parent))
        {
            Some(parent) => nodes[*parent].children.push(index),
            None => roots.push(index),
        }
    }

    let mut tree = SessionTree { nodes, roots };
    for root in tree.roots.clone() {
        update_latest_activity(&mut tree.nodes, root);
    }
    sort_nodes(&mut tree.nodes, &mut tree.roots);
    tree
}

fn flatten_session_tree(tree: &SessionTree) -> Vec<FlatSessionNode> {
    fn walk(
        nodes: &[SessionTreeNode],
        index: usize,
        depth: usize,
        ancestor_continues: &[bool],
        is_last: bool,
        out: &mut Vec<FlatSessionNode>,
    ) {
        let node = &nodes[index];
        out.push(FlatSessionNode {
            session: node.session.clone(),
            depth,
            is_last,
            ancestor_continues: ancestor_continues.to_vec(),
        });
        for (child_index, child) in node.children.iter().enumerate() {
            // Only non-root ancestors draw a continuation line.
            let continues = depth > 0 && !is_last;
            let mut next = ancestor_continues.to_vec();
            next.push(continues);
            walk(
                nodes,
                *child,
                depth + 1,
                &next,
                child_index + 1 == node.children.len(),
                out,
            );
        }
    }

    let mut out = Vec::new();
    for (index, root) in tree.roots.iter().enumerate() {
        walk(
            &tree.nodes,
            *root,
            0,
            &[],
            index + 1 == tree.roots.len(),
            &mut out,
        );
    }
    out
}

/// A callback receiving a session file path.
pub type SessionPathCallback = Rc<dyn Fn(&str)>;

/// A parameterless list action, upstream's `() => void` members.
pub type SessionListAction = Rc<dyn Fn()>;

/// A callback receiving the path awaiting delete confirmation, if any.
pub type DeleteConfirmationCallback = Rc<dyn Fn(Option<String>)>;

/// A callback receiving the new path-display visibility.
pub type TogglePathCallback = Rc<dyn Fn(bool)>;

/// A callback receiving an error message for the header status line.
pub type SessionErrorCallback = Rc<dyn Fn(&str)>;

/// A callback persisting a session's new name.
pub type RenameSessionCallback = Rc<dyn Fn(&str, &str)>;

/// The rows shown before scrolling, upstream's `maxVisible`.
const MAX_VISIBLE: usize = 10;

/// The keyboard-driven list, upstream's `SessionList`.
pub struct SessionList {
    all_sessions: RefCell<Vec<SessionInfo>>,
    filtered_sessions: RefCell<Vec<FlatSessionNode>>,
    selected_index: Cell<usize>,
    search_input: Rc<Input>,
    show_cwd: Cell<bool>,
    sort_mode: Cell<SortMode>,
    name_filter: Cell<NameFilter>,
    show_path: Cell<bool>,
    confirming_delete_path: RefCell<Option<String>>,
    current_session_canonical_path: Option<String>,
    focused: Cell<bool>,
    parser: KeyParser,
    /// Confirm and select the highlighted session.
    pub on_select: RefCell<Option<SessionPathCallback>>,
    /// Escape or the cancel binding.
    pub on_cancel: RefCell<Option<SessionListAction>>,
    /// Tab switches between the current folder and all sessions.
    pub on_toggle_scope: RefCell<Option<SessionListAction>>,
    /// Sort mode cycles threaded, recent, relevance.
    pub on_toggle_sort: RefCell<Option<SessionListAction>>,
    /// The named/all filter toggles.
    pub on_toggle_name_filter: RefCell<Option<SessionListAction>>,
    /// Path display toggles; receives the new visibility.
    pub on_toggle_path: RefCell<Option<TogglePathCallback>>,
    /// Delete confirmation appears, changes, or clears with this path.
    pub on_delete_confirmation_change: RefCell<Option<DeleteConfirmationCallback>>,
    /// Confirmed deletions are the host's filesystem action.
    pub on_delete_session: RefCell<Option<SessionPathCallback>>,
    /// Rename mode is requested for the highlighted session.
    pub on_rename_session: RefCell<Option<SessionPathCallback>>,
    /// Non-fatal list errors, such as deleting the active session.
    pub on_error: RefCell<Option<SessionErrorCallback>>,
}

impl std::fmt::Debug for SessionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionList")
            .field("all_sessions", &self.all_sessions.borrow().len())
            .field("selected_index", &self.selected_index)
            .field("sort_mode", &self.sort_mode)
            .finish_non_exhaustive()
    }
}

impl SessionList {
    fn new(
        sessions: Vec<SessionInfo>,
        show_cwd: bool,
        sort_mode: SortMode,
        name_filter: NameFilter,
        current_session_path: Option<&str>,
    ) -> Rc<Self> {
        let list = Rc::new(Self {
            all_sessions: RefCell::new(sessions),
            filtered_sessions: RefCell::new(Vec::new()),
            selected_index: Cell::new(0),
            search_input: Rc::new(Input::new()),
            show_cwd: Cell::new(show_cwd),
            sort_mode: Cell::new(sort_mode),
            name_filter: Cell::new(name_filter),
            show_path: Cell::new(false),
            confirming_delete_path: RefCell::new(None),
            current_session_canonical_path: current_session_path.map(canonical),
            focused: Cell::new(false),
            parser: KeyParser::new(),
            on_select: RefCell::new(None),
            on_cancel: RefCell::new(None),
            on_toggle_scope: RefCell::new(None),
            on_toggle_sort: RefCell::new(None),
            on_toggle_name_filter: RefCell::new(None),
            on_toggle_path: RefCell::new(None),
            on_delete_confirmation_change: RefCell::new(None),
            on_delete_session: RefCell::new(None),
            on_rename_session: RefCell::new(None),
            on_error: RefCell::new(None),
        });
        list.filter_sessions("");
        let weak = Rc::downgrade(&list);
        *list.search_input.on_submit.borrow_mut() = Some(Rc::new(move |_| {
            if let Some(list) = weak.upgrade() {
                list.submit_selected();
            }
        }));
        list
    }

    fn submit_selected(&self) {
        if let Some(path) = self.selected_session_path()
            && let Some(on_select) = self.on_select.borrow().as_ref()
        {
            on_select(&path);
        }
    }

    /// The highlighted session file, if the filtered list is non-empty.
    #[must_use]
    pub fn selected_session_path(&self) -> Option<String> {
        self.filtered_sessions
            .borrow()
            .get(self.selected_index.get())
            .map(|node| node.session.path.clone())
    }

    fn set_sort_mode(&self, sort_mode: SortMode) {
        self.sort_mode.set(sort_mode);
        self.filter_sessions(&self.search_input.get_value());
    }

    fn set_name_filter(&self, name_filter: NameFilter) {
        self.name_filter.set(name_filter);
        self.filter_sessions(&self.search_input.get_value());
    }

    fn set_sessions(&self, sessions: Vec<SessionInfo>, show_cwd: bool) {
        *self.all_sessions.borrow_mut() = sessions;
        self.show_cwd.set(show_cwd);
        self.filter_sessions(&self.search_input.get_value());
    }

    fn filter_sessions(&self, query: &str) {
        let all_sessions = self.all_sessions.borrow();
        let name_filtered: Vec<SessionInfo> = match self.name_filter.get() {
            NameFilter::All => all_sessions.clone(),
            NameFilter::Named => all_sessions
                .iter()
                .filter(|session| has_session_name(session))
                .cloned()
                .collect(),
        };
        drop(all_sessions);

        let filtered = if self.sort_mode.get() == SortMode::Threaded && query.trim().is_empty() {
            let tree = build_session_tree(&name_filtered);
            flatten_session_tree(&tree)
        } else {
            filter_and_sort_sessions(&name_filtered, query, self.sort_mode.get(), NameFilter::All)
                .into_iter()
                .map(|session| FlatSessionNode {
                    session,
                    depth: 0,
                    is_last: true,
                    ancestor_continues: Vec::new(),
                })
                .collect()
        };
        let len = filtered.len();
        *self.filtered_sessions.borrow_mut() = filtered;
        self.selected_index
            .set(self.selected_index.get().min(len.saturating_sub(1)));
    }

    fn set_confirming_delete_path(&self, path: Option<String>) {
        self.confirming_delete_path.borrow_mut().clone_from(&path);
        if let Some(callback) = self.on_delete_confirmation_change.borrow().as_ref() {
            callback(path);
        }
    }

    fn start_delete_confirmation_for_selected_session(&self) {
        let Some(selected) = self
            .filtered_sessions
            .borrow()
            .get(self.selected_index.get())
            .map(|node| node.session.path.clone())
        else {
            return;
        };
        if self.is_current_session_path(&selected) {
            if let Some(on_error) = self.on_error.borrow().as_ref() {
                on_error("Cannot delete the currently active session");
            }
            return;
        }
        self.set_confirming_delete_path(Some(selected));
    }

    fn is_current_session_path(&self, path: &str) -> bool {
        let Some(current) = self.current_session_canonical_path.as_deref() else {
            return false;
        };
        canonical(path) == current
    }

    fn build_tree_prefix(node: &FlatSessionNode) -> String {
        if node.depth == 0 {
            return String::new();
        }
        let mut prefix: String = node
            .ancestor_continues
            .iter()
            .map(|continues| if *continues { "│  " } else { "   " })
            .collect();
        prefix.push_str(if node.is_last { "└─ " } else { "├─ " });
        prefix
    }
}

impl Component for SessionList {
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's single render method branch for branch; splitting it would detach the row layout from the header it pairs with"
    )]
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = self.search_input.render(width);
        lines.push(String::new());

        let filtered = self.filtered_sessions.borrow();
        if filtered.is_empty() {
            let message = match self.name_filter.get() {
                NameFilter::Named => {
                    let toggle_key = key_text("app.session.toggleNamedFilter");
                    if self.show_cwd.get() {
                        format!("  No named sessions found. Press {toggle_key} to show all.")
                    } else {
                        format!(
                            "  No named sessions in current folder. Press {toggle_key} to show all, or Tab to view all."
                        )
                    }
                }
                NameFilter::All if self.show_cwd.get() => "  No sessions found".to_string(),
                NameFilter::All => {
                    "  No sessions in current folder. Press Tab to view all.".to_string()
                }
            };
            lines.push(theme().fg("muted", &truncate_to_width(&message, width, "…", false)));
            return lines;
        }

        let len = filtered.len();
        let selected_index = self.selected_index.get();
        let start = selected_index
            .saturating_sub(MAX_VISIBLE / 2)
            .min(len.saturating_sub(MAX_VISIBLE));
        let end = (start + MAX_VISIBLE).min(len);
        let now = now_millis();

        for (index, node) in filtered.iter().enumerate().take(end).skip(start) {
            let session = &node.session;
            let is_selected = index == selected_index;
            let is_confirming_delete = self
                .confirming_delete_path
                .borrow()
                .as_deref()
                .is_some_and(|path| path == session.path);
            let is_current = self.is_current_session_path(&session.path);
            let has_name = session.name.as_deref().is_some_and(|name| !name.is_empty());
            let display = session.name.as_deref().unwrap_or(&session.first_message);
            let normalized_message: String = display
                .chars()
                .map(|ch| {
                    if (ch as u32) < 0x20 || ch == '\u{7f}' {
                        ' '
                    } else {
                        ch
                    }
                })
                .collect::<String>()
                .trim()
                .to_string();

            let age = format_session_date(session.modified, now);
            let mut right_part = format!("{} {age}", session.message_count);
            if self.show_cwd.get() && !session.cwd.is_empty() {
                right_part = format!("{} {right_part}", shorten_path(&session.cwd));
            }
            if self.show_path.get() {
                right_part = format!("{} {right_part}", shorten_path(&session.path));
            }

            let prefix = Self::build_tree_prefix(node);
            let cursor = if is_selected {
                theme().fg("accent", "› ")
            } else {
                "  ".to_string()
            };
            let available_for_message = width
                .saturating_sub(2 + visible_width(&prefix) + visible_width(&right_part) + 2)
                .max(10);
            let truncated_message =
                truncate_to_width(&normalized_message, available_for_message, "…", false);

            let message_color = if is_confirming_delete {
                Some("error")
            } else if is_current {
                Some("accent")
            } else if has_name {
                Some("warning")
            } else {
                None
            };
            let mut styled_message = message_color.map_or_else(
                || truncated_message.clone(),
                |color| theme().fg(color, &truncated_message),
            );
            if is_selected {
                styled_message = bold(&styled_message);
            }

            let left_part = cursor + &theme().fg("dim", &prefix) + &styled_message;
            let spacing = width
                .saturating_sub(visible_width(&left_part) + visible_width(&right_part))
                .max(1);
            let styled_right = theme().fg(
                if is_confirming_delete { "error" } else { "dim" },
                &right_part,
            );
            let mut line = format!("{left_part}{}{styled_right}", " ".repeat(spacing));
            if is_selected {
                line = theme().bg("selectedBg", &line);
            }
            lines.push(truncate_to_width(&line, width, "", false));
        }

        if start > 0 || end < len {
            lines.push(theme().fg(
                "muted",
                &truncate_to_width(
                    &format!("  ({}/{len})", selected_index + 1),
                    width,
                    "",
                    false,
                ),
            ));
        }
        lines
    }

    fn handle_input(&self, data: &str) {
        let matches = |action: &str| get_keybindings().matches(&self.parser, data, action);

        if self.confirming_delete_path.borrow().is_some() {
            if matches("tui.select.confirm") {
                let path = self.confirming_delete_path.borrow_mut().take();
                if let Some(path) = path {
                    if let Some(callback) = self.on_delete_confirmation_change.borrow().as_ref() {
                        callback(None);
                    }
                    if let Some(on_delete_session) = self.on_delete_session.borrow().as_ref() {
                        on_delete_session(&path);
                    }
                }
            } else if matches("tui.select.cancel") {
                self.set_confirming_delete_path(None);
            }
            return;
        }

        if matches("tui.input.tab") {
            if let Some(callback) = self.on_toggle_scope.borrow().as_ref() {
                callback();
            }
            return;
        }
        if matches("app.session.toggleSort") {
            if let Some(callback) = self.on_toggle_sort.borrow().as_ref() {
                callback();
            }
            return;
        }
        if matches("app.session.toggleNamedFilter") {
            if let Some(callback) = self.on_toggle_name_filter.borrow().as_ref() {
                callback();
            }
            return;
        }
        if matches("app.session.togglePath") {
            let show_path = !self.show_path.get();
            self.show_path.set(show_path);
            if let Some(callback) = self.on_toggle_path.borrow().as_ref() {
                callback(show_path);
            }
            return;
        }
        if matches("app.session.delete") {
            self.start_delete_confirmation_for_selected_session();
            return;
        }
        if matches("app.session.rename") {
            if let Some(path) = self.selected_session_path()
                && let Some(callback) = self.on_rename_session.borrow().as_ref()
            {
                callback(&path);
            }
            return;
        }
        if matches("app.session.deleteNoninvasive") {
            if !self.search_input.get_value().is_empty() {
                self.search_input.handle_input(data);
                self.filter_sessions(&self.search_input.get_value());
                return;
            }
            self.start_delete_confirmation_for_selected_session();
            return;
        }

        if matches("tui.select.up") {
            self.selected_index
                .set(self.selected_index.get().saturating_sub(1));
        } else if matches("tui.select.down") {
            let len = self.filtered_sessions.borrow().len();
            self.selected_index
                .set((self.selected_index.get() + 1).min(len.saturating_sub(1)));
        } else if matches("tui.select.pageUp") {
            self.selected_index
                .set(self.selected_index.get().saturating_sub(MAX_VISIBLE));
        } else if matches("tui.select.pageDown") {
            let len = self.filtered_sessions.borrow().len();
            self.selected_index
                .set((self.selected_index.get() + MAX_VISIBLE).min(len.saturating_sub(1)));
        } else if matches("tui.select.confirm") {
            self.submit_selected();
        } else if matches("tui.select.cancel") {
            if let Some(callback) = self.on_cancel.borrow().as_ref() {
                callback();
            }
        } else {
            self.search_input.handle_input(data);
            self.filter_sessions(&self.search_input.get_value());
        }
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for SessionList {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.search_input.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

/// Everything the picker needs from its host. Data is a snapshot; callbacks
/// own persistence, loading, and redraw.
pub struct SessionSelectorOptions {
    /// Sessions from the current working directory, already loaded.
    pub current_sessions: Vec<SessionInfo>,
    /// The active session file; its row is marked and cannot be deleted.
    pub current_session_path: Option<String>,
    /// Confirm selects the path and closes.
    pub on_select: SessionPathCallback,
    /// Escape or the cancel binding closes without selecting.
    pub on_cancel: SessionListAction,
    /// The user toggled to all sessions and none are cached yet; the host
    /// loads them and answers with `set_all_sessions` or `set_load_error`.
    pub on_request_all_sessions: SessionListAction,
    /// A delete or rename landed; the host reloads the scopes it shows.
    pub on_refresh_sessions: SessionListAction,
    /// Redraw request after an internal state change.
    pub request_render: SessionListAction,
    /// Persists a new name; without it rename mode is unavailable.
    pub rename_session: Option<RenameSessionCallback>,
    /// Overrides the rename hint; defaults to whether renaming is wired.
    pub show_rename_hint: Option<bool>,
}

impl std::fmt::Debug for SessionSelectorOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionSelectorOptions")
            .field("current_sessions", &self.current_sessions.len())
            .field("current_session_path", &self.current_session_path)
            .field("rename_session", &self.rename_session.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    List,
    Rename,
}

/// The session picker at upstream pin `60e7e76`, host-driven per the module
/// docs. Construct through [`SessionSelectorComponent::new`], which returns
/// the shared handle the callbacks need.
pub struct SessionSelectorComponent {
    container: Container,
    header: Rc<SessionSelectorHeader>,
    session_list: Rc<SessionList>,
    rename_input: Rc<Input>,
    current_sessions: RefCell<Vec<SessionInfo>>,
    all_sessions: RefCell<Option<Vec<SessionInfo>>>,
    scope: Cell<SessionScope>,
    sort_mode: Cell<SortMode>,
    name_filter: Cell<NameFilter>,
    all_loading: Cell<bool>,
    mode: Cell<Mode>,
    rename_target: RefCell<Option<String>>,
    rename_session: Option<RenameSessionCallback>,
    on_select: SessionPathCallback,
    on_cancel: SessionListAction,
    on_request_all_sessions: SessionListAction,
    on_refresh_sessions: SessionListAction,
    request_render: SessionListAction,
    focused: Cell<bool>,
    parser: KeyParser,
}

impl std::fmt::Debug for SessionSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionSelectorComponent")
            .field("scope", &self.scope)
            .field("sort_mode", &self.sort_mode)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl SessionSelectorComponent {
    /// Builds the picker around the current-session snapshot and starts in
    /// the list mode with the current scope.
    #[must_use]
    pub fn new(options: SessionSelectorOptions) -> Rc<Self> {
        let sort_mode = SortMode::Threaded;
        let name_filter = NameFilter::All;
        let header = Rc::new(SessionSelectorHeader::new(
            SessionScope::Current,
            sort_mode,
            name_filter,
        ));
        header.show_rename_hint.set(
            options
                .show_rename_hint
                .unwrap_or_else(|| options.rename_session.is_some()),
        );
        let session_list = SessionList::new(
            options.current_sessions.clone(),
            false,
            sort_mode,
            name_filter,
            options.current_session_path.as_deref(),
        );
        let component = Rc::new(Self {
            container: Container::default(),
            header,
            session_list,
            rename_input: Rc::new(Input::new()),
            current_sessions: RefCell::new(options.current_sessions),
            all_sessions: RefCell::new(None),
            scope: Cell::new(SessionScope::Current),
            sort_mode: Cell::new(sort_mode),
            name_filter: Cell::new(name_filter),
            all_loading: Cell::new(false),
            mode: Cell::new(Mode::List),
            rename_target: RefCell::new(None),
            rename_session: options.rename_session,
            on_select: options.on_select,
            on_cancel: options.on_cancel,
            on_request_all_sessions: options.on_request_all_sessions,
            on_refresh_sessions: options.on_refresh_sessions,
            request_render: options.request_render,
            focused: Cell::new(false),
            parser: KeyParser::new(),
        });
        component.install_handlers();
        let list: Rc<dyn Component> = component.session_list.clone();
        component.build_base_layout(list, true);
        component
    }

    fn install_handlers(self: &Rc<Self>) {
        let weak = Rc::downgrade(self);
        let list = Rc::clone(&self.session_list);

        let parent = weak.clone();
        *list.on_select.borrow_mut() = Some(Rc::new(move |path: &str| {
            if let Some(component) = parent.upgrade() {
                component.header.clear_status();
                (component.on_select)(path);
            }
        }));
        let parent = weak.clone();
        *list.on_cancel.borrow_mut() = Some(Rc::new(move || {
            if let Some(component) = parent.upgrade() {
                component.header.clear_status();
                (component.on_cancel)();
            }
        }));
        let parent = weak.clone();
        *list.on_toggle_scope.borrow_mut() = Some(Rc::new(move || {
            if let Some(component) = parent.upgrade() {
                component.toggle_scope();
            }
        }));
        let parent = weak.clone();
        *list.on_toggle_sort.borrow_mut() = Some(Rc::new(move || {
            if let Some(component) = parent.upgrade() {
                component.toggle_sort_mode();
            }
        }));
        let parent = weak.clone();
        *list.on_toggle_name_filter.borrow_mut() = Some(Rc::new(move || {
            if let Some(component) = parent.upgrade() {
                component.toggle_name_filter();
            }
        }));
        let parent = weak.clone();
        *list.on_toggle_path.borrow_mut() = Some(Rc::new(move |show_path: bool| {
            if let Some(component) = parent.upgrade() {
                component.header.show_path.set(show_path);
                (component.request_render)();
            }
        }));
        let parent = weak.clone();
        *list.on_delete_confirmation_change.borrow_mut() =
            Some(Rc::new(move |path: Option<String>| {
                if let Some(component) = parent.upgrade() {
                    *component.header.confirming_delete_path.borrow_mut() = path;
                    (component.request_render)();
                }
            }));
        let parent = weak.clone();
        *list.on_delete_session.borrow_mut() = Some(Rc::new(move |path: &str| {
            if let Some(component) = parent.upgrade() {
                component.delete_session(path);
            }
        }));
        let parent = weak.clone();
        *list.on_error.borrow_mut() = Some(Rc::new(move |message: &str| {
            if let Some(component) = parent.upgrade() {
                component.header.set_status_message(
                    StatusKind::Error,
                    message,
                    Some(Duration::from_secs(3)),
                );
                (component.request_render)();
            }
        }));
        let parent = weak.clone();
        *list.on_rename_session.borrow_mut() = Some(Rc::new(move |path: &str| {
            if let Some(component) = parent.upgrade() {
                component.enter_rename_mode(path);
            }
        }));

        let parent = weak;
        *self.rename_input.on_submit.borrow_mut() = Some(Rc::new(move |value: &str| {
            if let Some(component) = parent.upgrade() {
                component.confirm_rename(value);
            }
        }));
    }
}

impl SessionSelectorComponent {
    /// Replaces the current-folder snapshot, typically after a host refresh.
    pub fn set_current_sessions(&self, sessions: Vec<SessionInfo>) {
        self.current_sessions.replace(sessions.clone());
        if self.scope.get() == SessionScope::Current {
            self.session_list.set_sessions(sessions, false);
            (self.request_render)();
        }
    }

    /// Completes an all-scope load. A late answer for a scope the user left
    /// only caches the list; it never switches the visible scope back.
    pub fn set_all_sessions(&self, sessions: Vec<SessionInfo>) {
        self.all_loading.set(false);
        let visible = self.scope.get() == SessionScope::All;
        let shown = sessions.clone();
        *self.all_sessions.borrow_mut() = Some(sessions);
        if visible {
            self.header.set_loading(false);
            self.session_list.set_sessions(shown, true);
            (self.request_render)();
        }
    }

    /// Reports a failed load for `scope`; the status line names the failure
    /// and a late answer for another scope is dropped. An initial
    /// current-folder failure also clears the list through
    /// [`SessionSelectorComponent::set_current_sessions`].
    pub fn set_load_error(&self, scope: SessionScope, message: &str) {
        if scope == SessionScope::All {
            self.all_loading.set(false);
        }
        if self.scope.get() != scope {
            return;
        }
        self.header.set_loading(false);
        self.header.set_status_message(
            StatusKind::Error,
            &format!("Failed to load sessions: {message}"),
            Some(Duration::from_secs(4)),
        );
        (self.request_render)();
    }

    /// Updates the header's in-flight progress counter; the header hides it
    /// unless a load is running.
    pub fn set_progress(&self, loaded: u64, total: u64) {
        self.header.load_progress.set(Some((loaded, total)));
        (self.request_render)();
    }

    fn sessions_for(&self, scope: SessionScope) -> Vec<SessionInfo> {
        match scope {
            SessionScope::Current => self.current_sessions.borrow().clone(),
            SessionScope::All => self.all_sessions.borrow().clone().unwrap_or_default(),
        }
    }

    fn toggle_scope(&self) {
        match self.scope.get() {
            SessionScope::Current => {
                self.scope.set(SessionScope::All);
                self.header.scope.set(SessionScope::All);
                if let Some(sessions) = self.all_sessions.borrow().clone() {
                    self.header.set_loading(false);
                    self.session_list.set_sessions(sessions, true);
                    (self.request_render)();
                    return;
                }
                if !self.all_loading.get() {
                    self.all_loading.set(true);
                    self.header.set_loading(true);
                    (self.request_render)();
                    (self.on_request_all_sessions)();
                }
            }
            SessionScope::All => {
                self.scope.set(SessionScope::Current);
                self.header.scope.set(SessionScope::Current);
                self.header.set_loading(false);
                self.session_list
                    .set_sessions(self.current_sessions.borrow().clone(), false);
                (self.request_render)();
            }
        }
    }

    fn toggle_sort_mode(&self) {
        let next = match self.sort_mode.get() {
            SortMode::Threaded => SortMode::Recent,
            SortMode::Recent => SortMode::Relevance,
            SortMode::Relevance => SortMode::Threaded,
        };
        self.sort_mode.set(next);
        self.header.sort_mode.set(next);
        self.session_list.set_sort_mode(next);
        (self.request_render)();
    }

    fn toggle_name_filter(&self) {
        let next = match self.name_filter.get() {
            NameFilter::All => NameFilter::Named,
            NameFilter::Named => NameFilter::All,
        };
        self.name_filter.set(next);
        self.header.name_filter.set(next);
        self.session_list.set_name_filter(next);
        (self.request_render)();
    }

    fn delete_session(&self, path: &str) {
        let result = delete_session_file(path);
        if result.ok {
            self.current_sessions
                .borrow_mut()
                .retain(|session| session.path != path);
            if let Some(all_sessions) = self.all_sessions.borrow_mut().as_mut() {
                all_sessions.retain(|session| session.path != path);
            }
            let scope = self.scope.get();
            self.session_list
                .set_sessions(self.sessions_for(scope), scope == SessionScope::All);
            let message = if result.method == SessionDeleteMethod::Trash {
                "Session moved to trash"
            } else {
                "Session deleted"
            };
            self.header
                .set_status_message(StatusKind::Info, message, Some(Duration::from_secs(2)));
            (self.on_refresh_sessions)();
        } else {
            let message = format!(
                "Failed to delete: {}",
                result.error.as_deref().unwrap_or("Unknown error")
            );
            self.header.set_status_message(
                StatusKind::Error,
                &message,
                Some(Duration::from_secs(3)),
            );
        }
        (self.request_render)();
    }

    fn enter_rename_mode(&self, path: &str) {
        if self.rename_session.is_none() {
            return;
        }
        if self.scope.get() == SessionScope::All && self.all_loading.get() {
            return;
        }
        let name = self
            .sessions_for(self.scope.get())
            .into_iter()
            .find(|session| session.path == path)
            .and_then(|session| session.name);
        self.mode.set(Mode::Rename);
        *self.rename_target.borrow_mut() = Some(path.to_string());
        self.rename_input.set_value(name.unwrap_or_default());
        self.rename_input.set_focused(true);

        let panel = Container::default();
        panel.add_child(Rc::new(Text::with_padding(bold("Rename Session"), 1, 0)));
        panel.add_child(Rc::new(Spacer::new()));
        let rename_input: Rc<dyn Component> = self.rename_input.clone();
        panel.add_child(rename_input);
        panel.add_child(Rc::new(Spacer::new()));
        panel.add_child(Rc::new(Text::with_padding(
            theme().fg(
                "muted",
                &format!(
                    "{} to save · {} to cancel",
                    key_text("tui.select.confirm"),
                    key_text("tui.select.cancel")
                ),
            ),
            1,
            0,
        )));
        self.build_base_layout(Rc::new(panel), false);
        (self.request_render)();
    }

    fn exit_rename_mode(&self) {
        self.mode.set(Mode::List);
        *self.rename_target.borrow_mut() = None;
        let list: Rc<dyn Component> = self.session_list.clone();
        self.build_base_layout(list, true);
        (self.request_render)();
    }

    fn confirm_rename(&self, value: &str) {
        let next = value.trim().to_string();
        if next.is_empty() {
            return;
        }
        let Some(target) = self.rename_target.borrow().clone() else {
            self.exit_rename_mode();
            return;
        };
        let Some(rename_session) = self.rename_session.as_ref() else {
            self.exit_rename_mode();
            return;
        };
        rename_session(&target, &next);
        (self.on_refresh_sessions)();
        self.exit_rename_mode();
    }

    fn build_base_layout(&self, content: Rc<dyn Component>, show_header: bool) {
        self.container.clear();
        self.container.add_child(Rc::new(Spacer::new()));
        self.container.add_child(Rc::new(DynamicBorder::themed()));
        self.container.add_child(Rc::new(Spacer::new()));
        if show_header {
            let header: Rc<dyn Component> = self.header.clone();
            self.container.add_child(header);
            self.container.add_child(Rc::new(Spacer::new()));
        }
        self.container.add_child(content);
        self.container.add_child(Rc::new(Spacer::new()));
        self.container.add_child(Rc::new(DynamicBorder::themed()));
    }

    /// The keyboard focus target inside the picker, upstream's
    /// `getSessionList`.
    #[must_use]
    pub fn get_session_list(&self) -> Rc<SessionList> {
        Rc::clone(&self.session_list)
    }
}

impl Component for SessionSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        if self.mode.get() == Mode::Rename {
            if get_keybindings().matches(&self.parser, data, "tui.select.cancel") {
                self.exit_rename_mode();
                return;
            }
            self.rename_input.handle_input(data);
            return;
        }
        self.session_list.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }
}

impl Focusable for SessionSelectorComponent {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.session_list.set_focused(focused);
        self.rename_input.set_focused(focused);
        if focused && self.mode.get() == Mode::Rename {
            self.rename_input.set_focused(true);
        }
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}
