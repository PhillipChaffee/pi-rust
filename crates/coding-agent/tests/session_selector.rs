//! The upstream rename and path/delete suites at pin `60e7e76`, plus
//! boundary coverage for the host-driven load setters, empty states, and
//! scrolling. The two deferred All-load cases port against
//! `set_all_sessions`/`on_request_all_sessions`.

#![expect(
    clippy::expect_used,
    reason = "temp paths and rendered rows fail the assertion when unexpected"
)]

mod common;
#[path = "common/selectors.rs"]
mod selectors;

use std::cell::{Cell, RefCell};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use pi_coding_agent::modes::interactive::components::session_selector::{
    RenameSessionCallback, SessionDeleteMethod, SessionScope, SessionSelectorComponent,
    SessionSelectorOptions, delete_session_file,
};
use pi_coding_agent::session_manager::SessionInfo;
use pi_tui::tui::{Component, Focusable};

use selectors::{output, setup};

const BASE: i64 = 1_767_225_600_000; // 2026-01-01T00:00:00Z
const DAY: i64 = 86_400_000;
const CTRL_R: &str = "\x1b[114;5u";
const CTRL_D: &str = "\x04";
const CTRL_BACKSPACE: &str = "\x1b[127;5u";
const CTRL_P: &str = "\x10";
const CTRL_N: &str = "\x0e";
const TAB: &str = "\t";
const DOWN: &str = "\x1b[B";
const PAGE_DOWN: &str = "\x1b[6~";
const ENTER: &str = "\r";
const ESC: &str = "\x1b";

fn session(id: &str) -> SessionInfo {
    SessionInfo {
        path: format!("/tmp/pi-session-selector-{id}.jsonl"),
        id: id.to_string(),
        cwd: String::new(),
        name: None,
        parent_session_path: None,
        created: Some(BASE),
        modified: BASE,
        message_count: 1,
        first_message: "hello".to_string(),
        all_messages_text: "hello".to_string(),
    }
}

struct Picker {
    component: Rc<SessionSelectorComponent>,
    selected: Rc<RefCell<Vec<String>>>,
    cancelled: Rc<Cell<usize>>,
    requested_all: Rc<Cell<usize>>,
    refreshed: Rc<Cell<usize>>,
    rendered: Rc<Cell<usize>>,
    renames: Rc<RefCell<Vec<(String, String)>>>,
}

fn picker_with(
    current: Vec<SessionInfo>,
    with_rename: bool,
    show_rename_hint: Option<bool>,
    current_path: Option<&str>,
) -> Picker {
    let selected = Rc::new(RefCell::new(Vec::new()));
    let selected_log = Rc::clone(&selected);
    let cancelled = Rc::new(Cell::new(0));
    let cancel_log = Rc::clone(&cancelled);
    let requested_all = Rc::new(Cell::new(0));
    let request_log = Rc::clone(&requested_all);
    let refreshed = Rc::new(Cell::new(0));
    let refresh_log = Rc::clone(&refreshed);
    let rendered = Rc::new(Cell::new(0));
    let render_log = Rc::clone(&rendered);
    let renames = Rc::new(RefCell::new(Vec::new()));
    let rename_log = Rc::clone(&renames);

    let rename_callback: Option<RenameSessionCallback> = if with_rename {
        Some(Rc::new(move |path: &str, name: &str| {
            rename_log
                .borrow_mut()
                .push((path.to_string(), name.to_string()));
        }))
    } else {
        None
    };

    let component = SessionSelectorComponent::new(SessionSelectorOptions {
        current_sessions: current,
        current_session_path: current_path.map(str::to_string),
        on_select: Rc::new(move |path: &str| selected_log.borrow_mut().push(path.to_string())),
        on_cancel: Rc::new(move || cancel_log.set(cancel_log.get() + 1)),
        on_request_all_sessions: Rc::new(move || {
            request_log.set(request_log.get() + 1);
        }),
        on_refresh_sessions: Rc::new(move || {
            refresh_log.set(refresh_log.get() + 1);
        }),
        request_render: Rc::new(move || render_log.set(render_log.get() + 1)),
        rename_session: rename_callback,
        show_rename_hint,
    });
    Picker {
        component,
        selected,
        cancelled,
        requested_all,
        refreshed,
        rendered,
        renames,
    }
}

fn picker(current: Vec<SessionInfo>) -> Picker {
    picker_with(current, false, None, None)
}

#[test]
fn shows_rename_hint_in_interactive_picker_configuration() {
    setup(&[]);
    let picker = picker_with(vec![session("a")], false, Some(true), None);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("ctrl+r"));
    assert!(text.contains("rename"));
}

#[test]
fn does_not_show_rename_hint_in_resume_picker_configuration() {
    setup(&[]);
    let picker = picker_with(vec![session("a")], false, Some(false), None);
    let text = output(picker.component.as_ref(), 120);
    assert!(!text.contains("ctrl+r"));
    assert!(!text.contains("rename"));
}

#[test]
fn enters_rename_mode_on_ctrl_r_and_submits_with_enter() {
    setup(&[]);
    let mut named = session("a");
    named.name = Some("Old".to_string());
    let picker = picker_with(vec![named], true, Some(true), None);

    picker.component.get_session_list().handle_input(CTRL_R);

    let rename_view = output(picker.component.as_ref(), 120);
    assert!(rename_view.contains("Rename Session"));
    assert!(!rename_view.contains("Resume Session"));

    picker.component.handle_input("X");
    picker.component.handle_input(ENTER);

    assert_eq!(
        picker.renames.borrow().as_slice(),
        [(named_path(), "XOld".to_string())]
    );
    assert_eq!(picker.refreshed.get(), 1);
    assert!(output(picker.component.as_ref(), 120).contains("Resume Session"));
}

fn named_path() -> String {
    session("a").path
}

#[test]
fn ctrl_r_does_nothing_without_a_rename_callback() {
    setup(&[]);
    let picker = picker_with(vec![session("a")], false, Some(true), None);
    picker.component.get_session_list().handle_input(CTRL_R);
    assert!(!output(picker.component.as_ref(), 120).contains("Rename Session"));
}

#[test]
fn blank_name_keeps_rename_mode_open() {
    setup(&[]);
    let picker = picker_with(vec![session("a")], true, None, None);
    picker.component.get_session_list().handle_input(CTRL_R);
    picker.component.handle_input(ENTER);
    assert!(picker.renames.borrow().is_empty());
    assert!(output(picker.component.as_ref(), 120).contains("Rename Session"));
}

#[test]
fn escape_leaves_rename_mode() {
    setup(&[]);
    let mut named = session("a");
    named.name = Some("Old".to_string());
    let picker = picker_with(vec![named], true, None, None);
    picker.component.get_session_list().handle_input(CTRL_R);
    picker.component.handle_input(ESC);
    assert!(picker.renames.borrow().is_empty());
    assert!(output(picker.component.as_ref(), 120).contains("Resume Session"));
}

#[test]
fn focus_forwards_to_the_list_and_rename_input() {
    setup(&[]);
    let picker = picker_with(vec![session("a")], true, None, None);
    picker.component.set_focused(true);
    assert!(picker.component.is_focused());
    assert!(picker.component.get_session_list().is_focused());
    picker.component.get_session_list().handle_input(CTRL_R);
    picker.component.handle_input("X");
    picker.component.handle_input(ENTER);
    assert_eq!(picker.renames.borrow().len(), 1);
}

fn watch_confirmation(
    list: &pi_coding_agent::modes::interactive::components::session_selector::SessionList,
) -> Rc<RefCell<Vec<Option<String>>>> {
    let log = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&log);
    *list.on_delete_confirmation_change.borrow_mut() = Some(Rc::new(move |path| {
        captured.borrow_mut().push(path);
    }));
    log
}

#[test]
fn ctrl_d_enters_confirmation_mode_with_a_non_empty_search_query() {
    setup(&[]);
    let picker = picker(vec![session("a"), session("b")]);
    let list = picker.component.get_session_list();
    let confirmations = watch_confirmation(&list);

    list.handle_input("a");
    list.handle_input(CTRL_D);

    assert_eq!(confirmations.borrow().as_slice(), [Some(session("a").path)]);
}

#[test]
fn ctrl_backspace_does_not_delete_when_the_search_query_is_non_empty() {
    setup(&[]);
    let picker = picker(vec![session("a"), session("b")]);
    let list = picker.component.get_session_list();
    let confirmations = watch_confirmation(&list);

    list.handle_input("a");
    list.handle_input(CTRL_BACKSPACE);

    assert!(confirmations.borrow().is_empty());
}

#[test]
fn ctrl_backspace_confirms_and_enter_deletes_when_the_query_is_empty() {
    setup(&[]);
    let picker = picker(vec![session("a"), session("b")]);
    let list = picker.component.get_session_list();
    let deleted = Rc::new(RefCell::new(None));
    let deleted_log = Rc::clone(&deleted);
    *list.on_delete_session.borrow_mut() = Some(Rc::new(move |path: &str| {
        *deleted_log.borrow_mut() = Some(path.to_string());
    }));

    list.handle_input(CTRL_BACKSPACE);
    assert!(output(picker.component.as_ref(), 120).contains("Delete session?"));

    list.handle_input(ENTER);
    assert_eq!(
        deleted.borrow().as_deref(),
        Some(session("a").path.as_str())
    );
    assert!(!output(picker.component.as_ref(), 120).contains("Delete session?"));
}

#[test]
fn delete_confirmation_callback_reports_the_path_then_clear() {
    setup(&[]);
    let picker = picker(vec![session("a")]);
    let list = picker.component.get_session_list();
    let confirmations = watch_confirmation(&list);

    list.handle_input(CTRL_D);
    list.handle_input(ENTER);

    assert_eq!(
        confirmations.borrow().as_slice(),
        [Some(session("a").path), None]
    );
}

#[test]
fn escape_cancels_delete_confirmation() {
    setup(&[]);
    let picker = picker(vec![session("a")]);
    let list = picker.component.get_session_list();

    list.handle_input(CTRL_D);
    assert!(output(picker.component.as_ref(), 120).contains("Delete session?"));
    list.handle_input(ESC);
    assert!(!output(picker.component.as_ref(), 120).contains("Delete session?"));
}

#[test]
fn late_all_scope_load_does_not_switch_the_header_back() {
    setup(&[]);
    let picker = picker(vec![session("current")]);
    let list = picker.component.get_session_list();

    list.handle_input(TAB); // current -> all (starts the host load)
    list.handle_input(TAB); // all -> current
    picker.component.set_all_sessions(vec![session("all")]);

    assert_eq!(picker.requested_all.get(), 1);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("Resume Session (Current Folder)"));
    assert!(!text.contains("Resume Session (All)"));
}

#[test]
fn toggling_scopes_while_all_is_loading_does_not_start_redundant_loads() {
    setup(&[]);
    let picker = picker(vec![session("current")]);
    let list = picker.component.get_session_list();

    list.handle_input(TAB); // current -> all (starts the host load)
    list.handle_input(TAB); // all -> current
    list.handle_input(TAB); // current -> all again while the load is pending

    assert_eq!(picker.requested_all.get(), 1);
    let mut all = session("all");
    all.name = Some("all-row".to_string());
    picker.component.set_all_sessions(vec![all]);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("Resume Session (All)"));
    assert!(text.contains("all-row"));
}

struct SymlinkFixture {
    base: PathBuf,
    parent_alias_a: String,
    parent_alias_b: String,
    child_alias_b: String,
}

impl Drop for SymlinkFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn symlink_fixture(name: &str) -> SymlinkFixture {
    let base =
        std::env::temp_dir().join(format!("pi-session-selector-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    let real = base.join("real");
    let alias_a = base.join("alias-a");
    let alias_b = base.join("alias-b");
    for dir in [&real, &alias_a, &alias_b] {
        fs::create_dir_all(dir).expect("create fixture dir");
    }
    let shared = real.join("sessions");
    fs::create_dir_all(&shared).expect("create shared sessions dir");
    std::os::unix::fs::symlink(&shared, alias_a.join("sessions")).expect("first session alias");
    std::os::unix::fs::symlink(&shared, alias_b.join("sessions")).expect("second session alias");
    fs::write(shared.join("parent.jsonl"), "parent\n").expect("parent file");
    fs::write(shared.join("child.jsonl"), "child\n").expect("child file");

    SymlinkFixture {
        parent_alias_a: alias_a
            .join("sessions")
            .join("parent.jsonl")
            .display()
            .to_string(),
        parent_alias_b: alias_b
            .join("sessions")
            .join("parent.jsonl")
            .display()
            .to_string(),
        child_alias_b: alias_b
            .join("sessions")
            .join("child.jsonl")
            .display()
            .to_string(),
        base,
    }
}

#[test]
fn threads_sessions_when_parent_and_child_use_different_symlink_aliases() {
    setup(&[]);
    let fixture = symlink_fixture("aliases");
    let mut parent = session("parent");
    parent.path = fixture.parent_alias_b.clone();
    parent.name = Some("Parent".to_string());
    let mut child = session("child");
    child.path = fixture.child_alias_b.clone();
    child.parent_session_path = Some(fixture.parent_alias_a.clone());
    child.name = Some("Child".to_string());

    let picker = picker(vec![parent, child]);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("Parent"));
    assert!(text.contains("└─ Child"));
}

#[test]
fn sorts_threaded_sessions_by_latest_activity_in_their_subtree() {
    setup(&[]);
    let mut parent_one = session("parent-one");
    parent_one.name = Some("Parent one".to_string());
    parent_one.modified = BASE + DAY;
    let mut parent_two = session("parent-two");
    parent_two.name = Some("Parent two".to_string());
    let mut child_two = session("child-two");
    child_two.path = session("child-two").path;
    child_two.name = Some("Child two".to_string());
    child_two.parent_session_path = Some(parent_two.path.clone());
    child_two.modified = BASE + 2 * DAY;

    let picker = picker(vec![parent_one, parent_two, child_two]);
    let text = output(picker.component.as_ref(), 120);
    let parent_two_index = text.find("Parent two").expect("parent two row");
    let child_two_index = text.find("└─ Child two").expect("child row");
    let parent_one_index = text.find("Parent one").expect("parent one row");
    assert!(parent_two_index < child_two_index);
    assert!(child_two_index < parent_one_index);
}

#[test]
fn active_session_cannot_be_deleted_across_symlink_aliases() {
    setup(&[]);
    let fixture = symlink_fixture("active");
    let mut parent = session("parent");
    parent.path = fixture.parent_alias_b.clone();
    let picker = picker_with(
        vec![parent],
        false,
        Some(false),
        Some(&fixture.parent_alias_a),
    );
    let list = picker.component.get_session_list();
    let confirmations = watch_confirmation(&list);
    let errors = Rc::new(RefCell::new(Vec::new()));
    let error_log = Rc::clone(&errors);
    *list.on_error.borrow_mut() = Some(Rc::new(move |message: &str| {
        error_log.borrow_mut().push(message.to_string());
    }));

    list.handle_input(CTRL_D);

    assert!(confirmations.borrow().is_empty());
    assert_eq!(
        errors.borrow().as_slice(),
        ["Cannot delete the currently active session"]
    );
}

#[test]
fn shows_the_current_scope_empty_state_and_the_named_filter_hint() {
    setup(&[]);
    let picker = picker(Vec::new());
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("No sessions in current folder. Press Tab to view all."));

    picker.component.get_session_list().handle_input(CTRL_N);
    let named = output(picker.component.as_ref(), 120);
    assert!(named.contains("No named sessions in current folder."));
    assert!(named.contains("Name: Named"));
}

#[test]
fn toggles_path_display_and_shows_the_row_path() {
    setup(&[]);
    let picker = picker(vec![session("a")]);
    let list = picker.component.get_session_list();

    assert!(output(picker.component.as_ref(), 120).contains("path (off)"));
    list.handle_input(CTRL_P);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("path (on)"));
    assert!(text.contains(&session("a").path));
}

#[test]
fn cycles_sort_modes_and_reorders_rows() {
    setup(&[]);
    let mut older = session("older");
    older.name = Some("Older".to_string());
    let mut newer = session("newer");
    newer.name = Some("Newer".to_string());
    newer.modified = BASE + DAY;
    let picker = picker(vec![older, newer]);
    let list = picker.component.get_session_list();

    let threaded = output(picker.component.as_ref(), 120);
    assert!(
        threaded.find("Newer").expect("newer row") < threaded.find("Older").expect("older row")
    );
    assert!(threaded.contains("Sort: Threaded"));

    list.handle_input("\x13"); // ctrl+s
    let recent = output(picker.component.as_ref(), 120);
    assert!(recent.contains("Sort: Recent"));
    assert!(recent.find("Older").expect("older row") < recent.find("Newer").expect("newer row"));

    list.handle_input("\x13");
    assert!(output(picker.component.as_ref(), 120).contains("Sort: Fuzzy"));
    list.handle_input("\x13");
    assert!(output(picker.component.as_ref(), 120).contains("Sort: Threaded"));
}

#[test]
fn toggles_the_named_filter() {
    setup(&[]);
    let mut named = session("named");
    named.name = Some("Named".to_string());
    named.first_message = "named first".to_string();
    let mut unnamed = session("unnamed");
    unnamed.first_message = "unnamed first".to_string();
    let picker = picker(vec![named, unnamed]);
    let list = picker.component.get_session_list();

    list.handle_input(CTRL_N);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("Named"));
    assert!(!text.contains("unnamed first"));
    list.handle_input(CTRL_N);
    assert!(output(picker.component.as_ref(), 120).contains("unnamed first"));
}

#[test]
fn scrolls_the_ten_row_window_and_pages_down() {
    setup(&[]);
    let sessions: Vec<SessionInfo> = (0..12)
        .map(|index| {
            let mut session = session(&format!("s{index}"));
            session.first_message = format!("row{index}");
            session
        })
        .collect();
    let picker = picker(sessions);
    let list = picker.component.get_session_list();

    let first = output(picker.component.as_ref(), 120);
    assert!(first.contains("(1/12)"));
    assert!(!first.contains("row10"));

    list.handle_input(PAGE_DOWN);
    let paged = output(picker.component.as_ref(), 120);
    assert!(paged.contains("(11/12)"));
    assert!(paged.contains("row10"));

    list.handle_input("\x1b[5~");
    assert!(output(picker.component.as_ref(), 120).contains("(1/12)"));
}

#[test]
fn confirm_selects_and_escape_cancels() {
    setup(&[]);
    let picker = picker(vec![session("a"), session("b")]);
    let list = picker.component.get_session_list();

    list.handle_input(DOWN);
    list.handle_input(ENTER);
    assert_eq!(picker.selected.borrow().as_slice(), [session("b").path]);

    list.handle_input("\x1b[A");
    list.handle_input(ENTER);
    assert_eq!(
        picker.selected.borrow().as_slice(),
        [session("b").path, session("a").path]
    );

    list.handle_input(ESC);
    assert_eq!(picker.cancelled.get(), 1);
}

#[test]
fn all_scope_rows_show_the_working_directory() {
    setup(&[]);
    let picker = picker(vec![session("current")]);
    let mut all = session("all");
    all.cwd = "/work/demo".to_string();
    picker.component.get_session_list().handle_input(TAB);
    picker.component.set_all_sessions(vec![all]);

    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("Resume Session (All)"));
    assert!(text.contains("/work/demo"));
}

#[test]
fn reports_load_progress_and_failures_for_the_visible_scope_only() {
    setup(&[]);
    let picker = picker(vec![session("current")]);
    let list = picker.component.get_session_list();
    list.handle_input(TAB);

    picker.component.set_progress(1, 3);
    assert!(output(picker.component.as_ref(), 120).contains("Loading 1/3"));

    picker.component.set_load_error(SessionScope::All, "boom");
    let failed = output(picker.component.as_ref(), 120);
    assert!(failed.contains("Failed to load sessions: boom"));
    assert!(!failed.contains("Loading"));

    list.handle_input(TAB); // back to the current folder
    picker.component.set_load_error(SessionScope::All, "late");
    assert!(!output(picker.component.as_ref(), 120).contains("late"));

    picker
        .component
        .set_load_error(SessionScope::Current, "now");
    assert!(output(picker.component.as_ref(), 120).contains("Failed to load sessions: now"));
}

#[test]
fn set_current_sessions_replaces_the_visible_rows() {
    setup(&[]);
    let mut old = session("old");
    old.first_message = "old-row".to_string();
    let picker = picker(vec![old]);
    let mut new = session("new");
    new.first_message = "new-row".to_string();
    picker.component.set_current_sessions(vec![new]);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("new-row"));
    assert!(!text.contains("old-row"));
    assert!(picker.rendered.get() > 0);
}

#[test]
fn all_scope_empty_states_name_the_named_filter() {
    setup(&[]);
    let picker = picker(Vec::new());
    let list = picker.component.get_session_list();
    list.handle_input(TAB);
    picker.component.set_all_sessions(Vec::new());

    assert!(output(picker.component.as_ref(), 120).contains("No sessions found"));
    list.handle_input(CTRL_N);
    let named = output(picker.component.as_ref(), 120);
    assert!(named.contains("No named sessions found."));
    assert!(named.contains("Name: Named"));
}

#[test]
fn delete_confirmation_removes_the_file_and_requests_a_refresh() {
    setup(&[]);
    let path = std::env::temp_dir().join(format!(
        "pi-session-selector-delete-{}.jsonl",
        std::process::id()
    ));
    fs::write(&path, "session\n").expect("write session file");
    let mut only = session("deleted");
    only.path = path.display().to_string();
    let picker = picker(vec![only]);
    let list = picker.component.get_session_list();

    list.handle_input(CTRL_D);
    list.handle_input(ENTER);

    assert!(!path.exists());
    assert_eq!(picker.refreshed.get(), 1);
    assert!(output(picker.component.as_ref(), 120).contains("No sessions in current folder."));
}

#[test]
fn deleting_a_missing_file_reports_success() {
    let path = std::env::temp_dir().join("pi-session-selector-missing-file.jsonl");
    let _ = fs::remove_file(&path);
    let result = delete_session_file(&path.display().to_string());
    assert!(result.ok);
    assert_eq!(result.method, SessionDeleteMethod::Trash);
    assert!(result.error.is_none());
}

#[test]
fn failed_delete_reports_the_unlink_error_with_the_trash_hint() {
    use std::os::unix::fs::PermissionsExt;

    setup(&[]);
    let dir = std::env::temp_dir().join(format!(
        "pi-session-selector-readonly-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create read-only directory");
    let path = dir.join("locked.jsonl");
    fs::write(&path, "session\n").expect("write locked session");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).expect("lock directory");

    let sanitized = path.display().to_string();
    let mut only = session("locked");
    only.path = sanitized;
    let picker = picker(vec![only]);
    let list = picker.component.get_session_list();
    list.handle_input(CTRL_D);
    list.handle_input(ENTER);

    assert!(path.exists(), "the failed delete leaves the file in place");
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("Failed to delete:"));
    assert!(text.contains("trash:"));
    assert_eq!(picker.refreshed.get(), 0);

    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("unlock directory");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn renders_coarse_age_labels() {
    setup(&[]);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch");
    let now = i64::try_from(now.as_millis()).expect("millis fit i64");
    let cases = [
        (0, "now"),
        (60_000, "1m"),
        (2 * 3_600_000, "2h"),
        (3 * 86_400_000, "3d"),
        (14 * 86_400_000, "2w"),
        (60 * 86_400_000, "2mo"),
        (2 * 365 * 86_400_000, "2y"),
    ];
    let sessions = cases
        .iter()
        .enumerate()
        .map(|(index, (delta, label))| {
            let mut session = session(&format!("age{index}"));
            session.first_message = format!("age-{label}");
            session.modified = now - delta;
            session
        })
        .collect();
    let picker = picker(sessions);
    let text = output(picker.component.as_ref(), 120);
    for (_, label) in cases {
        assert!(text.contains(&format!("1 {label}")), "missing age {label}");
    }
}

#[test]
fn draws_branch_and_continuation_prefixes() {
    setup(&[]);
    let mut parent = session("parent");
    parent.name = Some("Parent".to_string());
    let mut first = session("first");
    first.name = Some("First".to_string());
    first.parent_session_path = Some(parent.path.clone());
    let mut second = session("second");
    second.name = Some("Second".to_string());
    second.parent_session_path = Some(parent.path.clone());
    let mut grandchild = session("grandchild");
    grandchild.name = Some("Grandchild".to_string());
    grandchild.parent_session_path = Some(first.path.clone());
    grandchild.modified = BASE + DAY;
    first.modified = BASE + 2 * DAY;

    let picker = picker(vec![parent, first, second, grandchild]);
    let text = output(picker.component.as_ref(), 120);
    assert!(text.contains("├─ First"));
    assert!(text.contains("│  └─ Grandchild"));
    assert!(text.contains("└─ Second"));
}

#[test]
fn all_scope_shortens_home_relative_working_directories() {
    setup(&[]);
    let home = pi_coding_agent::config::home_dir();
    if home.is_empty() {
        return;
    }
    let mut all = session("all");
    all.cwd = format!("{home}/project");
    let picker = picker(vec![session("current")]);
    picker.component.get_session_list().handle_input(TAB);
    picker.component.set_all_sessions(vec![all]);

    assert!(output(picker.component.as_ref(), 120).contains("~/project"));
}

#[test]
fn current_snapshot_updates_do_not_disturb_the_all_scope() {
    setup(&[]);
    let picker = picker(vec![session("current")]);
    let list = picker.component.get_session_list();
    list.handle_input(TAB);
    let mut all = session("all");
    all.name = Some("all-row".to_string());
    picker.component.set_all_sessions(vec![all]);

    let mut fresh = session("fresh");
    fresh.first_message = "fresh-row".to_string();
    picker.component.set_current_sessions(vec![fresh]);
    let all_text = output(picker.component.as_ref(), 120);
    assert!(all_text.contains("all-row"));
    assert!(!all_text.contains("fresh-row"));

    list.handle_input(TAB);
    assert!(output(picker.component.as_ref(), 120).contains("fresh-row"));

    list.handle_input(TAB);
    let cached = output(picker.component.as_ref(), 120);
    assert!(cached.contains("all-row"));
    assert_eq!(picker.requested_all.get(), 1);
}
