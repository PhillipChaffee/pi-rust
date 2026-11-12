//! The package-resource manager, upstream's
//! `src/modes/interactive/components/config-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#213).
//!
//! The selector renders the resolved extension/skill/prompt/theme paths in
//! two write scopes: the global settings file and the project's
//! `.pi/settings.json`. Global mode toggles a resource by appending `+`/`-`
//! patterns to the settings array or package filters; project mode cycles a
//! project override through `inherit` → `load`/`unload`, dimming resources
//! inherited from the global scope.
//!
//! Restatements: `localeCompare` sorts by byte order; `Set`/`Map` identity
//! rides `Rc::ptr_eq` over shared resource records, so a toggle lands in
//! every list holding the item the way JavaScript's shared object does;
//! settings reads ride the raw JSON map, writes the manager's typed
//! setters. The untrusted-project refusal has no channel here, matching
//! upstream's unawaited write; the manager records it in `errors()`.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable};
use pi_tui::utils::{truncate_to_width, visible_width};
use serde_json::{Map, Value};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{key_hint, raw_key_hint};
use crate::config::{CONFIG_DIR_NAME, home_dir};
use crate::modes::interactive::theme::theme::{bold, theme};
use crate::package_manager::{PathMetadata, ResolvedPaths, ResolvedResource};
use crate::settings_manager::{Settings, SettingsManager, SettingsStorage};
use crate::source_info::{SourceOrigin, SourceScope};
use crate::utils::paths::{
    PathInputOptions, basename_posix, canonicalize_path, dirname_posix, is_local_path,
    relative_posix, resolve_path_with,
};

/// The four resource arrays the selector manages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// Extension files.
    Extensions,
    /// Skill directories.
    Skills,
    /// Prompt templates.
    Prompts,
    /// Themes.
    Themes,
}

impl ResourceType {
    /// The selector's render order, upstream's `RESOURCE_TYPES`.
    const ALL: [Self; 4] = [Self::Extensions, Self::Skills, Self::Prompts, Self::Themes];

    /// The section heading, upstream's `RESOURCE_TYPE_LABELS`.
    const fn label(self) -> &'static str {
        match self {
            Self::Extensions => "Extensions",
            Self::Skills => "Skills",
            Self::Prompts => "Prompts",
            Self::Themes => "Themes",
        }
    }

    /// The settings key and `ResolvedPaths` field name.
    const fn settings_key(self) -> &'static str {
        match self {
            Self::Extensions => "extensions",
            Self::Skills => "skills",
            Self::Prompts => "prompts",
            Self::Themes => "themes",
        }
    }

    /// The position in a subgroup sort, upstream's `typeOrder`.
    const fn order(self) -> u8 {
        match self {
            Self::Extensions => 0,
            Self::Skills => 1,
            Self::Prompts => 2,
            Self::Themes => 3,
        }
    }
}

/// Which settings file a toggle writes, upstream's `ConfigWriteScope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigWriteScope {
    /// The agent dir's global settings.
    Global,
    /// The project's `.pi/settings.json`.
    Project,
}

/// The writable scope a resource resolves against, upstream's
/// `SettingsScope`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsScope {
    User,
    Project,
}

impl SettingsScope {
    /// Upstream's metadata cast: only `project` reads as project.
    const fn from_source_scope(scope: SourceScope) -> Self {
        match scope {
            SourceScope::Project => Self::Project,
            SourceScope::User | SourceScope::Temporary => Self::User,
        }
    }
}

/// A project resource's override state, upstream's
/// `ProjectOverrideState`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProjectOverrideState {
    Inherit,
    Load,
    Unload,
}

/// The two scopes' resolved paths, upstream's `ScopedResolvedPaths`.
#[derive(Debug, Clone, Default)]
pub struct ScopedResolvedPaths {
    /// The global scope's resolved resources.
    pub global: ResolvedPaths,
    /// The project scope's resolved resources.
    pub project: ResolvedPaths,
}

/// One resolved resource, upstream's `ResourceItem`.
#[derive(Debug)]
pub struct ResourceItem {
    /// The absolute path, upstream's `path`.
    pub path: String,
    /// Whether the resource loads, upstream's `enabled`.
    enabled: Cell<bool>,
    /// The provenance, upstream's `metadata`.
    pub metadata: PathMetadata,
    /// The resource array it belongs to, upstream's `resourceType`.
    pub resource_type: ResourceType,
    /// The rendered name, upstream's `displayName`.
    pub display_name: String,
}

impl ResourceItem {
    /// Whether the resource currently loads.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled.get()
    }
}

/// One resource-type section, upstream's `ResourceSubgroup`.
struct ResourceSubgroup {
    label: String,
    items: Vec<Rc<ResourceItem>>,
}

/// One provenance group, upstream's `ResourceGroup`.
struct ResourceGroup {
    label: String,
    scope: SourceScope,
    subgroups: Vec<Rc<ResourceSubgroup>>,
}

/// A flattened render row, upstream's `FlatEntry`.
#[derive(Clone)]
enum FlatEntry {
    Group(Rc<ResourceGroup>),
    Subgroup {
        subgroup: Rc<ResourceSubgroup>,
        group: Rc<ResourceGroup>,
    },
    Item(Rc<ResourceItem>),
}

/// Format a base dir for display: the home prefix becomes `~`, separators
/// normalize, and a trailing slash is forced.
fn format_base_dir(base_dir: &str) -> String {
    let home = home_dir();
    let display_path = if base_dir == home {
        "~".to_string()
    } else if base_dir.starts_with(&home) {
        format!("~{}", base_dir[home.len()..].replace('\\', "/"))
    } else {
        base_dir.replace('\\', "/")
    };
    if display_path.ends_with('/') {
        display_path
    } else {
        format!("{display_path}/")
    }
}

/// The scope label a source carries, upstream's raw `metadata.scope`
/// interpolation.
const fn source_scope_label(scope: SourceScope) -> &'static str {
    match scope {
        SourceScope::User => "user",
        SourceScope::Project => "project",
        SourceScope::Temporary => "temporary",
    }
}

/// The group heading, upstream's `getGroupLabel`.
fn get_group_label(metadata: &PathMetadata, agent_dir: &str) -> String {
    if metadata.origin == SourceOrigin::Package {
        return format!(
            "{} ({})",
            metadata.source,
            source_scope_label(metadata.scope)
        );
    }
    if metadata.source == "auto" {
        let is_user = metadata.scope == SourceScope::User;
        return match &metadata.base_dir {
            Some(base_dir) => format!(
                "{} ({})",
                if is_user { "User" } else { "Project" },
                format_base_dir(base_dir)
            ),
            None if is_user => format!("User ({})", format_base_dir(agent_dir)),
            None => format!("Project ({CONFIG_DIR_NAME}/)"),
        };
    }
    match SettingsScope::from_source_scope(metadata.scope) {
        SettingsScope::User => "User settings".to_string(),
        SettingsScope::Project => "Project settings".to_string(),
    }
}

/// One resource-type section while a group builds.
struct DraftSubgroup {
    resource_type: ResourceType,
    items: Vec<Rc<ResourceItem>>,
}

/// One provenance group while its subgroups build.
struct DraftGroup {
    label: String,
    scope: SourceScope,
    origin: SourceOrigin,
    source: String,
    subgroups: Vec<DraftSubgroup>,
}

/// Build the provenance groups, upstream's `buildGroups`.
fn build_groups(resolved: &ResolvedPaths, agent_dir: &str) -> Vec<Rc<ResourceGroup>> {
    let mut group_map: Vec<(String, DraftGroup)> = Vec::new();

    let mut add_to_group = |resources: &[ResolvedResource], resource_type: ResourceType| {
        for resource in resources {
            let group_key = format!(
                "{:?}:{:?}:{}:{}",
                resource.metadata.origin,
                resource.metadata.scope,
                resource.metadata.source,
                resource.metadata.base_dir.clone().unwrap_or_default()
            );

            let group_index = group_map
                .iter()
                .position(|(key, _)| *key == group_key)
                .unwrap_or_else(|| {
                    group_map.push((
                        group_key.clone(),
                        DraftGroup {
                            label: get_group_label(&resource.metadata, agent_dir),
                            scope: resource.metadata.scope,
                            origin: resource.metadata.origin,
                            source: resource.metadata.source.clone(),
                            subgroups: Vec::new(),
                        },
                    ));
                    group_map.len() - 1
                });
            let group = &mut group_map[group_index].1;

            let file_name = basename_posix(&resource.path);
            let parent_folder = basename_posix(&dirname_posix(&resource.path));
            let display_name =
                if resource_type == ResourceType::Extensions && parent_folder != "extensions" {
                    format!("{parent_folder}/{file_name}")
                } else if resource_type == ResourceType::Skills && file_name == "SKILL.md" {
                    parent_folder
                } else {
                    file_name
                };

            let item = Rc::new(ResourceItem {
                path: resource.path.clone(),
                enabled: Cell::new(resource.enabled),
                metadata: resource.metadata.clone(),
                resource_type,
                display_name,
            });

            if let Some(subgroup) = group
                .subgroups
                .iter_mut()
                .find(|subgroup| subgroup.resource_type == resource_type)
            {
                subgroup.items.push(item);
            } else {
                group.subgroups.push(DraftSubgroup {
                    resource_type,
                    items: vec![item],
                });
            }
        }
    };

    add_to_group(&resolved.extensions, ResourceType::Extensions);
    add_to_group(&resolved.skills, ResourceType::Skills);
    add_to_group(&resolved.prompts, ResourceType::Prompts);
    add_to_group(&resolved.themes, ResourceType::Themes);

    group_map.sort_by(|(_, a), (_, b)| compare_groups(a, b));
    group_map
        .into_iter()
        .map(|(_, mut group)| {
            group
                .subgroups
                .sort_by_key(|subgroup| subgroup.resource_type.order());
            for subgroup in &mut group.subgroups {
                subgroup
                    .items
                    .sort_by(|a, b| a.display_name.cmp(&b.display_name));
            }
            Rc::new(ResourceGroup {
                label: group.label,
                scope: group.scope,
                subgroups: group
                    .subgroups
                    .into_iter()
                    .map(|subgroup| {
                        Rc::new(ResourceSubgroup {
                            label: subgroup.resource_type.label().to_string(),
                            items: subgroup.items,
                        })
                    })
                    .collect(),
            })
        })
        .collect()
}

/// The group ordering: packages first, then top-level; user before project;
/// then the source string.
fn compare_groups(a: &DraftGroup, b: &DraftGroup) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if a.origin != b.origin {
        return if a.origin == SourceOrigin::Package {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if a.scope != b.scope {
        return if a.scope == SourceScope::User {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    a.source.cmp(&b.source)
}

/// The title, scope hint, and key hints, upstream's `ConfigSelectorHeader`.
struct ConfigSelectorHeader {
    write_scope: Cell<ConfigWriteScope>,
    project_mode_available: bool,
}

impl ConfigSelectorHeader {
    const fn new(write_scope: ConfigWriteScope, project_mode_available: bool) -> Self {
        Self {
            write_scope: Cell::new(write_scope),
            project_mode_available,
        }
    }

    fn set_write_scope(&self, write_scope: ConfigWriteScope) {
        self.write_scope.set(write_scope);
    }
}

impl Component for ConfigSelectorHeader {
    fn render(&self, width: usize) -> Vec<String> {
        let write_scope = self.write_scope.get();
        let title = bold(if write_scope == ConfigWriteScope::Project {
            "Project Local Resources"
        } else {
            "Global Resources"
        });
        let sep = theme().fg("muted", " · ");
        let switch_hint = if self.project_mode_available {
            key_hint("tui.input.tab", "switch mode") + &sep
        } else {
            String::new()
        };
        let action_hint = if write_scope == ConfigWriteScope::Project {
            raw_key_hint("space", "cycle inherit/+/-")
        } else {
            raw_key_hint("space", "toggle")
        };
        let hint = switch_hint + &action_hint + &sep + &raw_key_hint("esc", "close");
        let spacing = width
            .saturating_sub(visible_width(&title))
            .saturating_sub(visible_width(&hint))
            .max(1);
        let scope_hint = if write_scope == ConfigWriteScope::Project {
            theme().fg(
                "muted",
                &format!("{CONFIG_DIR_NAME}/settings.json · inherited global resources are dimmed"),
            )
        } else {
            theme().fg("muted", &format!("~/{CONFIG_DIR_NAME}/agent/settings.json"))
        };

        vec![
            truncate_to_width(
                &format!("{title}{}{hint}", " ".repeat(spacing)),
                width,
                "",
                false,
            ),
            truncate_to_width(&scope_hint, width, "", false),
        ]
    }
}

/// The selection callbacks a host binds, upstream's public callback fields.
pub type ResourceToggleCallback = Rc<dyn Fn(&ResourceItem, bool)>;
/// A callback carrying no value.
pub type SelectorCallback = Rc<dyn Fn()>;

/// The searchable resource list, upstream's `ResourceList`.
pub struct ResourceList<S: SettingsStorage> {
    groups_by_scope: [Vec<Rc<ResourceGroup>>; 2],
    flat_items: RefCell<Vec<FlatEntry>>,
    filtered_items: RefCell<Vec<FlatEntry>>,
    selected_index: Cell<usize>,
    search_input: Rc<Input>,
    max_visible: usize,
    settings_manager: Arc<Mutex<SettingsManager<S>>>,
    cwd: String,
    agent_dir: String,
    write_scope: Cell<ConfigWriteScope>,
    inherited_enabled_by_key: HashMap<String, bool>,
    /// Set by the host, upstream's public `onCancel`.
    pub on_cancel: RefCell<Option<SelectorCallback>>,
    /// Set by the host, upstream's public `onExit`.
    pub on_exit: RefCell<Option<SelectorCallback>>,
    /// Set by the host, upstream's public `onToggle`.
    pub on_toggle: RefCell<Option<ResourceToggleCallback>>,
    /// Set by the host, upstream's public `onSwitchMode`.
    pub on_switch_mode: RefCell<Option<SelectorCallback>>,
    focused: Cell<bool>,
    parser: KeyParser,
}

impl<S: SettingsStorage> std::fmt::Debug for ResourceList<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceList")
            .field("write_scope", &self.write_scope.get())
            .field("selected_index", &self.selected_index.get())
            .finish_non_exhaustive()
    }
}

impl<S: SettingsStorage> ResourceList<S> {
    /// Build the list over both scopes' resolved paths.
    #[must_use]
    pub fn new(
        resolved_paths: &ScopedResolvedPaths,
        settings_manager: Arc<Mutex<SettingsManager<S>>>,
        cwd: String,
        agent_dir: String,
        terminal_height: Option<usize>,
        write_scope: ConfigWriteScope,
    ) -> Self {
        let global_groups = build_groups(&resolved_paths.global, &agent_dir);
        let project_groups = build_groups(&resolved_paths.project, &agent_dir);
        let inherited_enabled_by_key = build_inherited_enabled_map(&global_groups);
        let list = Self {
            groups_by_scope: [global_groups, project_groups],
            flat_items: RefCell::new(Vec::new()),
            filtered_items: RefCell::new(Vec::new()),
            selected_index: Cell::new(0),
            search_input: Rc::new(Input::new()),
            // 8 lines of chrome: top spacer + top border + spacer + header
            // (2 lines) + spacer + bottom spacer + bottom border.
            max_visible: (terminal_height.unwrap_or(24)).saturating_sub(8).max(5),
            settings_manager,
            cwd,
            agent_dir,
            write_scope: Cell::new(write_scope),
            inherited_enabled_by_key,
            on_cancel: RefCell::new(None),
            on_exit: RefCell::new(None),
            on_toggle: RefCell::new(None),
            on_switch_mode: RefCell::new(None),
            focused: Cell::new(false),
            parser: KeyParser::new(),
        };
        list.build_flat_list();
        list.filtered_items
            .borrow_mut()
            .clone_from(&list.flat_items.borrow());
        list
    }

    /// Switch the write scope and rebuild both lists, upstream
    /// `setWriteScope`.
    pub fn set_write_scope(&self, write_scope: ConfigWriteScope) {
        self.write_scope.set(write_scope);
        self.build_flat_list();
        self.filter_items(&self.search_input.get_value());
    }

    /// The live write scope.
    #[must_use]
    pub const fn write_scope(&self) -> ConfigWriteScope {
        self.write_scope.get()
    }

    fn groups(&self) -> &[Rc<ResourceGroup>] {
        match self.write_scope.get() {
            ConfigWriteScope::Global => &self.groups_by_scope[0],
            ConfigWriteScope::Project => &self.groups_by_scope[1],
        }
    }

    fn build_flat_list(&self) {
        let mut flat_items = Vec::new();
        for group in self.groups() {
            flat_items.push(FlatEntry::Group(Rc::clone(group)));
            for subgroup in &group.subgroups {
                flat_items.push(FlatEntry::Subgroup {
                    subgroup: Rc::clone(subgroup),
                    group: Rc::clone(group),
                });
                for item in &subgroup.items {
                    flat_items.push(FlatEntry::Item(Rc::clone(item)));
                }
            }
        }
        let first_item = flat_items
            .iter()
            .position(|entry| matches!(entry, FlatEntry::Item(_)))
            .unwrap_or(0);
        self.selected_index.set(first_item);
        *self.flat_items.borrow_mut() = flat_items;
    }

    fn find_next_item(&self, from_index: usize, direction: isize) -> usize {
        let filtered = self.filtered_items.borrow();
        let mut index = from_index;
        loop {
            if direction < 0 {
                if index == 0 {
                    return from_index;
                }
                index -= 1;
            } else {
                index += 1;
                if index >= filtered.len() {
                    return from_index;
                }
            }
            if matches!(filtered[index], FlatEntry::Item(_)) {
                return index;
            }
        }
    }

    fn filter_items(&self, query: &str) {
        let flat_items = self.flat_items.borrow().clone();
        if query.trim().is_empty() {
            *self.filtered_items.borrow_mut() = flat_items;
            self.select_first_item();
            return;
        }

        let lower_query = query.to_lowercase();
        let matches_item = |item: &ResourceItem| {
            item.display_name.to_lowercase().contains(&lower_query)
                || item
                    .resource_type
                    .settings_key()
                    .to_lowercase()
                    .contains(&lower_query)
                || item.path.to_lowercase().contains(&lower_query)
        };

        let mut filtered = Vec::new();
        for entry in &flat_items {
            let keep = match entry {
                FlatEntry::Group(group) => group
                    .subgroups
                    .iter()
                    .any(|subgroup| subgroup.items.iter().any(|item| matches_item(item))),
                FlatEntry::Subgroup { subgroup, .. } => {
                    subgroup.items.iter().any(|item| matches_item(item))
                }
                FlatEntry::Item(item) => matches_item(item),
            };
            if keep {
                filtered.push(entry.clone());
            }
        }
        *self.filtered_items.borrow_mut() = filtered;
        self.select_first_item();
    }

    fn select_first_item(&self) {
        let first_item = self
            .filtered_items
            .borrow()
            .iter()
            .position(|entry| matches!(entry, FlatEntry::Item(_)))
            .unwrap_or(0);
        self.selected_index.set(first_item);
    }

    /// Update the enabled flag on the item and its current-scope twin,
    /// upstream `updateItem`.
    pub fn update_item(&self, item: &ResourceItem, enabled: bool) {
        item.enabled.set(enabled);
        for group in self.groups() {
            for subgroup in &group.subgroups {
                if let Some(found) = subgroup.items.iter().find(|candidate| {
                    candidate.path == item.path && candidate.resource_type == item.resource_type
                }) {
                    found.enabled.set(enabled);
                    return;
                }
            }
        }
    }

    fn settings(&self) -> MutexGuard<'_, SettingsManager<S>> {
        self.settings_manager
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn render_lines(&self, width: usize) -> Vec<String> {
        let mut lines = Vec::new();

        lines.extend(self.search_input.render(width));
        lines.push(String::new());

        let filtered_items = self.filtered_items.borrow();
        if filtered_items.is_empty() {
            lines.push(theme().fg("muted", "  No resources found"));
            return lines;
        }

        let start_index = self
            .selected_index
            .get()
            .saturating_sub(self.max_visible / 2)
            .min(filtered_items.len().saturating_sub(self.max_visible));
        let end_index = (start_index + self.max_visible).min(filtered_items.len());

        for index in start_index..end_index {
            let entry = &filtered_items[index];
            let is_selected = index == self.selected_index.get();
            match entry {
                FlatEntry::Group(group) => {
                    let inherited = self.write_scope.get() == ConfigWriteScope::Project
                        && group.scope == SourceScope::User;
                    let label = bold(&format!(
                        "{}{}",
                        group.label,
                        if inherited {
                            " · inherited global"
                        } else {
                            ""
                        }
                    ));
                    let group_line = theme().fg(if inherited { "dim" } else { "accent" }, &label);
                    lines.push(truncate_to_width(
                        &format!("  {group_line}"),
                        width,
                        "",
                        false,
                    ));
                }
                FlatEntry::Subgroup { subgroup, group } => {
                    let color = if self.write_scope.get() == ConfigWriteScope::Project
                        && group.scope == SourceScope::User
                    {
                        "dim"
                    } else {
                        "muted"
                    };
                    let subgroup_line = theme().fg(color, &subgroup.label);
                    lines.push(truncate_to_width(
                        &format!("    {subgroup_line}"),
                        width,
                        "",
                        false,
                    ));
                }
                FlatEntry::Item(item) => {
                    let cursor = if is_selected { "> " } else { "  " };
                    let dimmed = self.is_dimmed_item(item);
                    let name_text = if is_selected && !dimmed {
                        bold(&item.display_name)
                    } else {
                        item.display_name.clone()
                    };
                    let name = if dimmed {
                        theme().fg("dim", &name_text)
                    } else {
                        name_text
                    };
                    lines.push(truncate_to_width(
                        &format!(
                            "{cursor}    {} {name}{}",
                            self.render_checkbox(item),
                            self.get_item_suffix(item)
                        ),
                        width,
                        "...",
                        false,
                    ));
                }
            }
        }

        if start_index > 0 || end_index < filtered_items.len() {
            let item_count = filtered_items
                .iter()
                .filter(|entry| matches!(entry, FlatEntry::Item(_)))
                .count();
            let current_item_index = filtered_items[..self.selected_index.get()]
                .iter()
                .filter(|entry| matches!(entry, FlatEntry::Item(_)))
                .count()
                + 1;
            lines.push(theme().fg("dim", &format!("  ({current_item_index}/{item_count})")));
        }

        lines
    }

    fn handle_key(&self, data: &str) {
        let matches = |action: &str| get_keybindings().matches(&self.parser, data, action);

        if matches("tui.select.up") {
            self.selected_index
                .set(self.find_next_item(self.selected_index.get(), -1));
        } else if matches("tui.select.down") {
            self.selected_index
                .set(self.find_next_item(self.selected_index.get(), 1));
        } else if matches("tui.select.pageUp") {
            let len = self.filtered_items.borrow().len();
            let mut target = self.selected_index.get().saturating_sub(self.max_visible);
            while target < len
                && !matches!(&self.filtered_items.borrow()[target], FlatEntry::Item(_))
            {
                target += 1;
            }
            if target < len {
                self.selected_index.set(target);
            }
        } else if matches("tui.select.pageDown") {
            let len = self.filtered_items.borrow().len();
            if len == 0 {
                return;
            }
            let mut target = (self.selected_index.get() + self.max_visible).min(len - 1);
            while !matches!(&self.filtered_items.borrow()[target], FlatEntry::Item(_)) {
                if target == 0 {
                    break;
                }
                target -= 1;
            }
            if matches!(&self.filtered_items.borrow()[target], FlatEntry::Item(_)) {
                self.selected_index.set(target);
            }
        } else if matches("tui.select.cancel") {
            if let Some(on_cancel) = self.on_cancel.borrow().as_ref() {
                on_cancel();
            }
        } else if self.parser.matches_key(data, "ctrl+c") {
            if let Some(on_exit) = self.on_exit.borrow().as_ref() {
                on_exit();
            }
        } else if matches("tui.input.tab") {
            if let Some(on_switch_mode) = self.on_switch_mode.borrow().as_ref() {
                on_switch_mode();
            }
        } else if data == " " || matches("tui.select.confirm") {
            let entry = self
                .filtered_items
                .borrow()
                .get(self.selected_index.get())
                .cloned();
            if let Some(FlatEntry::Item(item)) = entry
                && (self.write_scope.get() == ConfigWriteScope::Project
                    || item_scope(&item) == SettingsScope::User)
                && let Some(new_enabled) = self.toggle_resource(&item)
            {
                self.update_item(&item, new_enabled);
                if let Some(on_toggle) = self.on_toggle.borrow().as_ref() {
                    on_toggle(&item, new_enabled);
                }
            }
        } else {
            self.search_input.handle_input(data);
            self.filter_items(&self.search_input.get_value());
        }
    }

    fn toggle_resource(&self, item: &ResourceItem) -> Option<bool> {
        if self.write_scope.get() == ConfigWriteScope::Project {
            let state = self.get_next_override_state(item);
            if !self.set_project_resource_override(item, state) {
                return None;
            }
            return Some(match state {
                ProjectOverrideState::Inherit => self.get_inherited_enabled(item),
                ProjectOverrideState::Load => true,
                ProjectOverrideState::Unload => false,
            });
        }

        let enabled = !item.enabled.get();
        if item.metadata.origin == SourceOrigin::TopLevel {
            self.toggle_top_level_resource(item, enabled);
        } else {
            self.toggle_package_resource(item, enabled);
        }
        Some(enabled)
    }

    fn toggle_top_level_resource(&self, item: &ResourceItem, enabled: bool) {
        let scope = SettingsScope::from_source_scope(item.metadata.scope);
        let settings = self.scoped_settings(scope);
        let current = string_array(&settings, item.resource_type.settings_key());
        let pattern = self.get_resource_pattern(item);
        let updated = strip_resource_patterns(&current, &pattern, enabled);
        self.write_paths(scope, item.resource_type, &updated);
    }

    fn toggle_package_resource(&self, item: &ResourceItem, enabled: bool) {
        let scope = SettingsScope::from_source_scope(item.metadata.scope);
        let settings = self.scoped_settings(scope);
        let mut packages = packages_array(&settings);
        let Some(package_index) = packages
            .iter()
            .position(|package| package_source_string(package) == item.metadata.source)
        else {
            return;
        };

        let mut package = package_object(&packages[package_index]);
        let key = item.resource_type.settings_key();
        let current = object_string_array(&package, key);
        let pattern = package_resource_pattern(item);
        let updated = strip_resource_patterns(&current, &pattern, enabled);
        set_object_string_array(&mut package, key, &updated);
        // Upstream's empty-filter cleanup is unreachable here: the toggle
        // always appends a pattern, so the array never empties.
        packages[package_index] = Value::Object(package);
        self.write_packages(scope, &packages);
    }

    fn render_checkbox(&self, item: &ResourceItem) -> String {
        if self.write_scope.get() == ConfigWriteScope::Project {
            return match self.get_project_override_state(item) {
                ProjectOverrideState::Load => theme().fg("success", "[+]"),
                ProjectOverrideState::Unload => theme().fg("warning", "[-]"),
                ProjectOverrideState::Inherit => {
                    theme().fg("dim", if item.enabled.get() { "[x]" } else { "[ ]" })
                }
            };
        }
        if item.enabled.get() {
            theme().fg("success", "[x]")
        } else {
            theme().fg("dim", "[ ]")
        }
    }

    fn get_item_suffix(&self, item: &ResourceItem) -> String {
        if self.write_scope.get() != ConfigWriteScope::Project {
            return String::new();
        }
        match self.get_project_override_state(item) {
            ProjectOverrideState::Load => theme().fg("muted", "  project load"),
            ProjectOverrideState::Unload => theme().fg("muted", "  project unload"),
            ProjectOverrideState::Inherit => {
                if self.is_inherited_global_item(item) {
                    theme().fg("dim", "  inherited global")
                } else {
                    String::new()
                }
            }
        }
    }

    fn is_dimmed_item(&self, item: &ResourceItem) -> bool {
        self.write_scope.get() == ConfigWriteScope::Project
            && self.is_inherited_global_item(item)
            && self.get_project_override_state(item) == ProjectOverrideState::Inherit
    }

    fn set_project_resource_override(
        &self,
        item: &ResourceItem,
        state: ProjectOverrideState,
    ) -> bool {
        if item.metadata.origin == SourceOrigin::TopLevel {
            self.set_project_top_level_override(item, state)
        } else {
            self.set_project_package_override(item, state)
        }
    }

    fn set_project_top_level_override(
        &self,
        item: &ResourceItem,
        state: ProjectOverrideState,
    ) -> bool {
        let current = string_array(
            &self.scoped_settings(SettingsScope::Project),
            item.resource_type.settings_key(),
        );
        let pattern = if self.is_inherited_global_item(item) {
            item.path.clone()
        } else {
            self.get_resource_pattern_for_scope(item, SettingsScope::Project)
        };
        let patterns = self.get_top_level_override_patterns(item, SettingsScope::Project);
        let mut updated: Vec<String> = current
            .into_iter()
            .filter(|entry| {
                let target = get_pattern_entry_target(entry);
                if (entry.starts_with('!') || entry.starts_with('+') || entry.starts_with('-'))
                    && patterns.contains(&target)
                {
                    return false;
                }
                !(state == ProjectOverrideState::Inherit
                    && self.is_inherited_global_item(item)
                    && target == pattern)
            })
            .collect();
        if state != ProjectOverrideState::Inherit {
            if self.is_inherited_global_item(item) && !updated.contains(&pattern) {
                updated.push(pattern.clone());
            }
            updated.push(format!("{}{pattern}", state.sign()));
        }
        self.write_paths(SettingsScope::Project, item.resource_type, &updated);
        true
    }

    fn set_project_package_override(
        &self,
        item: &ResourceItem,
        state: ProjectOverrideState,
    ) -> bool {
        let mut packages = packages_array(&self.scoped_settings(SettingsScope::Project));
        let source_scope = item_scope(item);
        let mut package_index = packages.iter().position(|package| {
            self.package_source_string_matches(
                &item.metadata.source,
                source_scope,
                package_source_string(package),
                SettingsScope::Project,
            )
        });
        if package_index.is_none() {
            if state == ProjectOverrideState::Inherit {
                return false;
            }
            packages.push(self.create_package_override_source(item));
            package_index = Some(packages.len() - 1);
        }
        let Some(package_index) = package_index else {
            return false;
        };
        let Some(package_value) = packages.get(package_index) else {
            return false;
        };
        let mut package = package_object(package_value);
        let key = item.resource_type.settings_key();
        let pattern = package_resource_pattern(item);
        let mut updated: Vec<String> = object_string_array(&package, key)
            .into_iter()
            .filter(|entry| get_pattern_entry_target(entry) != pattern)
            .collect();
        if state != ProjectOverrideState::Inherit {
            updated.push(format!("{}{pattern}", state.sign()));
        }
        set_object_string_array(&mut package, key, &updated);
        if ResourceType::ALL
            .iter()
            .any(|resource_type| package.contains_key(resource_type.settings_key()))
        {
            packages[package_index] = Value::Object(package);
        } else if package.get("autoload") == Some(&Value::Bool(false)) {
            packages.remove(package_index);
        } else {
            let source = package
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            packages[package_index] = Value::String(source);
        }
        self.write_packages(SettingsScope::Project, &packages);
        true
    }

    fn get_next_override_state(&self, item: &ResourceItem) -> ProjectOverrideState {
        let state = self.get_project_override_state(item);
        let inherited_enabled = self.get_inherited_enabled(item);
        match state {
            ProjectOverrideState::Inherit => {
                if inherited_enabled {
                    ProjectOverrideState::Unload
                } else {
                    ProjectOverrideState::Load
                }
            }
            ProjectOverrideState::Unload => {
                if inherited_enabled {
                    ProjectOverrideState::Load
                } else {
                    ProjectOverrideState::Inherit
                }
            }
            ProjectOverrideState::Load => {
                if inherited_enabled {
                    ProjectOverrideState::Inherit
                } else {
                    ProjectOverrideState::Unload
                }
            }
        }
    }

    fn get_project_override_state(&self, item: &ResourceItem) -> ProjectOverrideState {
        if self.write_scope.get() != ConfigWriteScope::Project {
            return ProjectOverrideState::Inherit;
        }
        if item.metadata.origin == SourceOrigin::TopLevel {
            return override_state_from_entries(
                &string_array(
                    &self.scoped_settings(SettingsScope::Project),
                    item.resource_type.settings_key(),
                ),
                &self.get_top_level_override_patterns(item, SettingsScope::Project),
                false,
            );
        }
        let package = self.find_matching_package_source(item, SettingsScope::Project);
        let Some(Value::Object(package)) = package else {
            return ProjectOverrideState::Inherit;
        };
        let Some(Value::Array(entries)) = package.get(item.resource_type.settings_key()) else {
            return ProjectOverrideState::Inherit;
        };
        let entries: Vec<String> = entries
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        override_state_from_entries(
            &entries,
            &HashSet::from([package_resource_pattern(item)]),
            package.get("autoload") != Some(&Value::Bool(false)),
        )
    }

    fn get_inherited_enabled(&self, item: &ResourceItem) -> bool {
        self.inherited_enabled_by_key
            .get(&get_resource_item_key(item))
            .copied()
            .unwrap_or_else(|| {
                if item_scope(item) == SettingsScope::User {
                    item.enabled.get()
                } else {
                    true
                }
            })
    }

    fn is_inherited_global_item(&self, item: &ResourceItem) -> bool {
        item_scope(item) == SettingsScope::User
            || self
                .inherited_enabled_by_key
                .contains_key(&get_resource_item_key(item))
    }

    fn get_top_level_override_patterns(
        &self,
        item: &ResourceItem,
        scope: SettingsScope,
    ) -> HashSet<String> {
        let base_dir = self.get_top_level_base_dir(scope);
        let mut patterns = HashSet::from([
            self.get_resource_pattern_for_scope(item, scope),
            item.path.clone(),
            relative_posix(&base_dir, &item.path),
        ]);
        if let Some(metadata_base_dir) = &item.metadata.base_dir {
            patterns.insert(relative_posix(metadata_base_dir, &item.path));
        }
        patterns
    }

    fn get_resource_pattern_for_scope(&self, item: &ResourceItem, scope: SettingsScope) -> String {
        let source_scope = item_scope(item);
        if scope != source_scope {
            return item.path.clone();
        }
        let base_dir = item
            .metadata
            .base_dir
            .clone()
            .unwrap_or_else(|| self.get_top_level_base_dir(source_scope));
        relative_posix(&base_dir, &item.path)
    }

    fn create_package_override_source(&self, item: &ResourceItem) -> Value {
        let source = &item.metadata.source;
        let mut object = Map::new();
        object.insert("autoload".to_string(), Value::Bool(false));
        if !is_local_path(source) {
            object.insert("source".to_string(), Value::String(source.clone()));
            return Value::Object(object);
        }
        let resolved = resolve_path_with(
            source,
            &self.get_top_level_base_dir(item_scope(item)),
            &resolve_options(),
        );
        let source_path = resolved.unwrap_or_else(|_| source.clone());
        let relative = relative_posix(
            &self.get_top_level_base_dir(SettingsScope::Project),
            &source_path,
        );
        object.insert(
            "source".to_string(),
            Value::String(if relative.is_empty() {
                ".".to_string()
            } else {
                relative
            }),
        );
        Value::Object(object)
    }

    fn package_source_string_matches(
        &self,
        left_source: &str,
        left_scope: SettingsScope,
        right_source: &str,
        right_scope: SettingsScope,
    ) -> bool {
        if left_source == right_source {
            return true;
        }
        if !is_local_path(left_source) || !is_local_path(right_source) {
            return false;
        }
        let left = resolve_path_with(
            left_source,
            &self.get_top_level_base_dir(left_scope),
            &resolve_options(),
        );
        let right = resolve_path_with(
            right_source,
            &self.get_top_level_base_dir(right_scope),
            &resolve_options(),
        );
        left.ok() == right.ok()
    }

    fn find_matching_package_source(
        &self,
        item: &ResourceItem,
        target_scope: SettingsScope,
    ) -> Option<Value> {
        let settings = self.scoped_settings(target_scope);
        packages_array(&settings).into_iter().find(|package| {
            self.package_source_string_matches(
                &item.metadata.source,
                item_scope(item),
                package_source_string(package),
                target_scope,
            )
        })
    }
}

/// The trimmed resolution options, upstream's `{ trim: true }`.
fn resolve_options() -> PathInputOptions {
    PathInputOptions {
        trim: true,
        ..PathInputOptions::default()
    }
}

impl<S: SettingsStorage> ResourceList<S> {
    fn get_resource_pattern(&self, item: &ResourceItem) -> String {
        let scope = SettingsScope::from_source_scope(item.metadata.scope);
        let base_dir = item
            .metadata
            .base_dir
            .clone()
            .unwrap_or_else(|| self.get_top_level_base_dir(scope));
        relative_posix(&base_dir, &item.path)
    }

    fn get_top_level_base_dir(&self, scope: SettingsScope) -> String {
        match scope {
            SettingsScope::Project => format!("{}/{CONFIG_DIR_NAME}", self.cwd),
            SettingsScope::User => self.agent_dir.clone(),
        }
    }

    fn scoped_settings(&self, scope: SettingsScope) -> Settings {
        let manager = self.settings();
        match scope {
            SettingsScope::User => manager.get_global_settings(),
            SettingsScope::Project => manager.get_project_settings(),
        }
    }

    fn write_paths(&self, scope: SettingsScope, resource_type: ResourceType, paths: &[String]) {
        let mut manager = self.settings();
        match (scope, resource_type) {
            (SettingsScope::User, ResourceType::Extensions) => {
                manager.set_extension_paths(paths);
            }
            (SettingsScope::User, ResourceType::Skills) => manager.set_skill_paths(paths),
            (SettingsScope::User, ResourceType::Prompts) => {
                manager.set_prompt_template_paths(paths);
            }
            (SettingsScope::User, ResourceType::Themes) => manager.set_theme_paths(paths),
            (SettingsScope::Project, ResourceType::Extensions) => {
                let _ = manager.set_project_extension_paths(paths);
            }
            (SettingsScope::Project, ResourceType::Skills) => {
                let _ = manager.set_project_skill_paths(paths);
            }
            (SettingsScope::Project, ResourceType::Prompts) => {
                let _ = manager.set_project_prompt_template_paths(paths);
            }
            (SettingsScope::Project, ResourceType::Themes) => {
                let _ = manager.set_project_theme_paths(paths);
            }
        }
    }

    fn write_packages(&self, scope: SettingsScope, packages: &[Value]) {
        let mut manager = self.settings();
        match scope {
            SettingsScope::User => manager.set_packages(packages),
            SettingsScope::Project => {
                let _ = manager.set_project_packages(packages);
            }
        }
    }
}

impl ProjectOverrideState {
    /// The pattern prefix a load/unload entry carries.
    const fn sign(self) -> &'static str {
        match self {
            Self::Load => "+",
            Self::Inherit | Self::Unload => "-",
        }
    }
}

/// Read a string array from a settings map, upstream's `settings[key] ?? []`.
fn string_array(settings: &Settings, key: &str) -> Vec<String> {
    settings
        .get(key)
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The raw `packages` array, upstream's `settings.packages ?? []`.
fn packages_array(settings: &Settings) -> Vec<Value> {
    settings
        .get("packages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// A package object view: strings become `{ source }`, upstream's
/// string-to-object conversion.
fn package_object(package: &Value) -> Map<String, Value> {
    match package {
        Value::Object(object) => object.clone(),
        Value::String(source) => {
            Map::from_iter([("source".to_string(), Value::String(source.clone()))])
        }
        _ => Map::new(),
    }
}

/// The source string of a package entry, upstream's `typeof pkg === "string"
/// ? pkg : pkg.source`.
fn package_source_string(package: &Value) -> &str {
    match package {
        Value::String(source) => source,
        Value::Object(object) => object
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        _ => "",
    }
}

/// One package object's string-array field, upstream's `pkg[key] ?? []`.
fn object_string_array(package: &Map<String, Value>, key: &str) -> Vec<String> {
    package
        .get(key)
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Write one package object's string-array field: empty arrays drop the key,
/// upstream's `updated.length > 0 ? updated : undefined`.
fn set_object_string_array(package: &mut Map<String, Value>, key: &str, values: &[String]) {
    if values.is_empty() {
        package.remove(key);
    } else {
        package.insert(
            key.to_string(),
            Value::Array(
                values
                    .iter()
                    .map(|value| Value::String(value.clone()))
                    .collect(),
            ),
        );
    }
}

/// Strip a resource's existing patterns and append the toggled one,
/// upstream's filter-and-push in both toggle paths.
fn strip_resource_patterns(current: &[String], pattern: &str, enabled: bool) -> Vec<String> {
    let mut updated: Vec<String> = current
        .iter()
        .filter(|entry| get_pattern_entry_target(entry) != pattern)
        .cloned()
        .collect();
    updated.push(format!("{}{pattern}", if enabled { "+" } else { "-" }));
    updated
}

/// The entry's target with any leading `!`/`+`/`-` stripped.
fn get_pattern_entry_target(entry: &str) -> String {
    entry
        .strip_prefix(['!', '+', '-'])
        .map_or_else(|| entry.to_string(), str::to_string)
}

/// The `type:canonical-path` key, upstream's `getResourceItemKey`.
fn get_resource_item_key(item: &ResourceItem) -> String {
    format!(
        "{}:{}",
        item.resource_type.settings_key(),
        canonicalize_path(&item.path)
    )
}

/// A package resource's pattern relative to its package root, upstream's
/// `getPackageResourcePattern`.
fn package_resource_pattern(item: &ResourceItem) -> String {
    let base_dir = item
        .metadata
        .base_dir
        .clone()
        .unwrap_or_else(|| dirname_posix(&item.path));
    relative_posix(&base_dir, &item.path)
}

/// A resource's writable scope, upstream's `getItemScope`.
const fn item_scope(item: &ResourceItem) -> SettingsScope {
    SettingsScope::from_source_scope(item.metadata.scope)
}

/// The override state an entry list encodes, upstream's
/// `getOverrideStateFromEntries`.
fn override_state_from_entries(
    entries: &[String],
    patterns: &HashSet<String>,
    empty_array_is_unload: bool,
) -> ProjectOverrideState {
    if entries.is_empty() && empty_array_is_unload {
        return ProjectOverrideState::Unload;
    }
    let mut state = ProjectOverrideState::Inherit;
    for entry in entries {
        if !patterns.contains(&get_pattern_entry_target(entry)) {
            continue;
        }
        state = if entry.starts_with('!') || entry.starts_with('-') {
            ProjectOverrideState::Unload
        } else {
            ProjectOverrideState::Load
        };
    }
    state
}

/// Every global resource's enabled flag, upstream's
/// `buildInheritedEnabledMap`.
fn build_inherited_enabled_map(groups: &[Rc<ResourceGroup>]) -> HashMap<String, bool> {
    let mut result = HashMap::new();
    for group in groups {
        for subgroup in &group.subgroups {
            for item in &subgroup.items {
                result.insert(get_resource_item_key(item), item.enabled.get());
            }
        }
    }
    result
}

impl<S: SettingsStorage + 'static> Component for ResourceList<S> {
    fn render(&self, width: usize) -> Vec<String> {
        self.render_lines(width)
    }

    fn handle_input(&self, data: &str) {
        self.handle_key(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {}

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl<S: SettingsStorage> Focusable for ResourceList<S> {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.search_input.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

/// The selector's trailing options, upstream's optional constructor
/// parameters.
#[derive(Debug, Clone, Copy)]
pub struct ConfigSelectorOptions {
    /// The terminal height the visible-row count derives from.
    pub terminal_height: Option<usize>,
    /// The write scope the selector opens in.
    pub write_scope: ConfigWriteScope,
    /// Whether the tab binding switches to the project scope.
    pub project_mode_available: bool,
}

impl Default for ConfigSelectorOptions {
    fn default() -> Self {
        Self {
            terminal_height: None,
            write_scope: ConfigWriteScope::Global,
            project_mode_available: true,
        }
    }
}

/// The config manager dialog, upstream's `ConfigSelectorComponent`.
pub struct ConfigSelectorComponent<S: SettingsStorage> {
    header: Rc<ConfigSelectorHeader>,
    resource_list: Rc<ResourceList<S>>,
    container: Container,
    focused: Cell<bool>,
}

impl<S: SettingsStorage> std::fmt::Debug for ConfigSelectorComponent<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigSelectorComponent")
            .field("write_scope", &self.resource_list.write_scope.get())
            .finish_non_exhaustive()
    }
}

impl<S: SettingsStorage + 'static> ConfigSelectorComponent<S> {
    /// Build the dialog over both scopes' resolved paths.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors upstream's constructor; the trailing optional parameters ride ConfigSelectorOptions"
    )]
    #[must_use]
    pub fn new(
        resolved_paths: &ScopedResolvedPaths,
        settings_manager: Arc<Mutex<SettingsManager<S>>>,
        cwd: String,
        agent_dir: String,
        on_close: SelectorCallback,
        on_exit: SelectorCallback,
        request_render: SelectorCallback,
        options: ConfigSelectorOptions,
    ) -> Self {
        let write_scope = options.write_scope;
        let header = Rc::new(ConfigSelectorHeader::new(
            write_scope,
            options.project_mode_available,
        ));
        let resource_list = Rc::new(ResourceList::new(
            resolved_paths,
            settings_manager,
            cwd,
            agent_dir,
            options.terminal_height,
            write_scope,
        ));
        *resource_list.on_cancel.borrow_mut() = Some(on_close);
        *resource_list.on_exit.borrow_mut() = Some(on_exit);
        let on_toggle_render = Rc::clone(&request_render);
        *resource_list.on_toggle.borrow_mut() =
            Some(Rc::new(move |_item, _enabled| on_toggle_render()));
        if options.project_mode_available {
            let switch_header = Rc::clone(&header);
            let switch_list = Rc::clone(&resource_list);
            *resource_list.on_switch_mode.borrow_mut() = Some(Rc::new(move || {
                toggle_write_scope(&switch_header, &switch_list);
                request_render();
            }));
        }

        let container = Container::default();
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        let header_child: Rc<dyn Component> = header.clone();
        container.add_child(header_child);
        container.add_child(Rc::new(Spacer::new()));
        let list_child: Rc<dyn Component> = resource_list.clone();
        container.add_child(list_child);
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));

        Self {
            header,
            resource_list,
            container,
            focused: Cell::new(false),
        }
    }

    /// The switch binding's write-scope flip, upstream's
    /// `switchWriteScope`.
    pub fn switch_write_scope(&self) {
        toggle_write_scope(&self.header, &self.resource_list);
    }

    /// The live write scope.
    #[must_use]
    pub fn write_scope(&self) -> ConfigWriteScope {
        self.resource_list.write_scope.get()
    }

    /// The resource list, upstream `getResourceList`.
    #[must_use]
    pub fn resource_list(&self) -> Rc<ResourceList<S>> {
        Rc::clone(&self.resource_list)
    }
}

/// Flip both the header and the list to the other write scope.
fn toggle_write_scope<S: SettingsStorage>(header: &ConfigSelectorHeader, list: &ResourceList<S>) {
    let next = match list.write_scope.get() {
        ConfigWriteScope::Global => ConfigWriteScope::Project,
        ConfigWriteScope::Project => ConfigWriteScope::Global,
    };
    header.set_write_scope(next);
    list.set_write_scope(next);
}

impl<S: SettingsStorage + 'static> Component for ConfigSelectorComponent<S> {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_input(&self, data: &str) {
        self.resource_list.handle_input(data);
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
}

impl<S: SettingsStorage> Focusable for ConfigSelectorComponent<S> {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.resource_list.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}
