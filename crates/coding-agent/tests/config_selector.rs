//! The config-selector suite: upstream's `config-selector.ts` at pin
//! `60e7e76` (#213), with the `package-command-paths.test.ts` cycling case
//! ported 1:1 plus boundaries for grouping, search, navigation, both toggle
//! paths, and the project override cycle.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

#[path = "common/selectors.rs"]
mod selectors;

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pi_coding_agent::modes::interactive::components::config_selector::{
    ConfigSelectorComponent, ConfigSelectorOptions, ConfigWriteScope, ResourceType,
    ScopedResolvedPaths, SelectorCallback,
};
use pi_coding_agent::package_manager::{PathMetadata, ResolvedPaths, ResolvedResource};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, SettingsManager, SettingsManagerCreateOptions, SettingsScope,
};
use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
use pi_tui::tui::{Component, Focusable};
use selectors::{cancellation, output, setup};
use serde_json::{Value, json};

type MemoryManager = Arc<Mutex<SettingsManager<InMemorySettingsStorage>>>;

const AGENT_DIR: &str = "/home/user/.pi/agent";
const CWD: &str = "/home/user/project";

fn resource(
    path: &str,
    source: &str,
    scope: SourceScope,
    origin: SourceOrigin,
    base_dir: Option<&str>,
    enabled: bool,
) -> ResolvedResource {
    ResolvedResource {
        path: path.to_string(),
        enabled,
        metadata: PathMetadata {
            source: source.to_string(),
            scope,
            origin,
            base_dir: base_dir.map(str::to_string),
        },
    }
}

const fn resolved_paths(
    extensions: Vec<ResolvedResource>,
    skills: Vec<ResolvedResource>,
    prompts: Vec<ResolvedResource>,
    themes: Vec<ResolvedResource>,
) -> ResolvedPaths {
    ResolvedPaths {
        extensions,
        skills,
        prompts,
        themes,
    }
}

const fn empty_paths() -> ResolvedPaths {
    resolved_paths(Vec::new(), Vec::new(), Vec::new(), Vec::new())
}

fn manager_with(global: &Value, project: &Value) -> MemoryManager {
    let storage = InMemorySettingsStorage::default();
    if *global != Value::Null {
        storage.seed(SettingsScope::Global, global.to_string());
    }
    if *project != Value::Null {
        storage.seed(SettingsScope::Project, project.to_string());
    }
    Arc::new(Mutex::new(SettingsManager::from_storage(
        storage,
        SettingsManagerCreateOptions {
            project_trusted: Some(true),
        },
    )))
}

fn global_settings(manager: &MemoryManager) -> Value {
    Value::Object(manager.lock().expect("manager").get_global_settings())
}

fn project_settings(manager: &MemoryManager) -> Value {
    Value::Object(manager.lock().expect("manager").get_project_settings())
}

fn selector_with(
    scoped: &ScopedResolvedPaths,
    manager: &MemoryManager,
    options: ConfigSelectorOptions,
    on_close: SelectorCallback,
    on_exit: SelectorCallback,
) -> ConfigSelectorComponent<InMemorySettingsStorage> {
    ConfigSelectorComponent::new(
        scoped,
        Arc::clone(manager),
        CWD.to_string(),
        AGENT_DIR.to_string(),
        on_close,
        on_exit,
        Rc::new(|| {}),
        options,
    )
}

fn selector(
    scoped: &ScopedResolvedPaths,
    manager: &MemoryManager,
    write_scope: ConfigWriteScope,
    project_mode_available: bool,
) -> ConfigSelectorComponent<InMemorySettingsStorage> {
    selector_with(
        scoped,
        manager,
        ConfigSelectorOptions {
            terminal_height: Some(24),
            write_scope,
            project_mode_available,
        },
        Rc::new(|| {}),
        Rc::new(|| {}),
    )
}

fn selector_scoped(
    global: ResolvedPaths,
    project: ResolvedPaths,
    manager: &MemoryManager,
    write_scope: ConfigWriteScope,
) -> ConfigSelectorComponent<InMemorySettingsStorage> {
    selector(
        &ScopedResolvedPaths { global, project },
        manager,
        write_scope,
        true,
    )
}

fn selector_both(
    paths: ResolvedPaths,
    manager: &MemoryManager,
    write_scope: ConfigWriteScope,
) -> ConfigSelectorComponent<InMemorySettingsStorage> {
    selector_scoped(paths.clone(), paths, manager, write_scope)
}

fn empty_selector(
    manager: &MemoryManager,
    project_mode_available: bool,
) -> ConfigSelectorComponent<InMemorySettingsStorage> {
    selector(
        &ScopedResolvedPaths {
            global: empty_paths(),
            project: empty_paths(),
        },
        manager,
        ConfigWriteScope::Global,
        project_mode_available,
    )
}

#[test]
fn cycles_project_package_overrides_in_config_local_mode() {
    setup(&[]);
    let manager = manager_with(&json!({ "packages": ["npm:pi-tools"] }), &Value::Null);
    let package_root = "/home/user/pkg";
    let paths = resolved_paths(
        vec![resource(
            &format!("{package_root}/extensions/bar.ts"),
            "npm:pi-tools",
            SourceScope::User,
            SourceOrigin::Package,
            Some(package_root),
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Project);

    selector.resource_list().handle_input(" ");
    assert_eq!(
        project_settings(&manager)["packages"],
        json!([{ "source": "npm:pi-tools", "autoload": false, "extensions": ["-extensions/bar.ts"] }])
    );

    selector.resource_list().handle_input(" ");
    assert_eq!(
        project_settings(&manager)["packages"],
        json!([{ "source": "npm:pi-tools", "autoload": false, "extensions": ["+extensions/bar.ts"] }])
    );

    selector.resource_list().handle_input(" ");
    assert_eq!(project_settings(&manager)["packages"], json!([]));
}

#[test]
fn global_scope_ignores_project_resources() {
    setup(&[]);
    let manager = manager_with(&json!({ "extensions": [] }), &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            &format!("{CWD}/.pi/extensions/a.ts"),
            "local",
            SourceScope::Project,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_scoped(empty_paths(), paths, &manager, ConfigWriteScope::Global);

    selector.resource_list().handle_input(" ");
    assert_eq!(global_settings(&manager)["extensions"], json!([]));
}

#[test]
fn global_scope_toggles_top_level_user_resource() {
    setup(&[]);
    let manager = manager_with(&json!({ "extensions": ["extensions/a.ts"] }), &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            &format!("{AGENT_DIR}/extensions/a.ts"),
            "local",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);

    selector.resource_list().handle_input(" ");
    assert_eq!(
        global_settings(&manager)["extensions"],
        json!(["-extensions/a.ts"])
    );

    selector.resource_list().handle_input(" ");
    assert_eq!(
        global_settings(&manager)["extensions"],
        json!(["+extensions/a.ts"])
    );
}

#[test]
fn project_scope_cycles_top_level_override_through_inherit() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let inherited = resource(
        &format!("{AGENT_DIR}/extensions/a.ts"),
        "local",
        SourceScope::User,
        SourceOrigin::TopLevel,
        None,
        true,
    );
    let paths = resolved_paths(vec![inherited], Vec::new(), Vec::new(), Vec::new());
    let selector = selector_both(paths, &manager, ConfigWriteScope::Project);

    selector.resource_list().handle_input(" ");
    assert_eq!(
        project_settings(&manager)["extensions"],
        json!([
            format!("{AGENT_DIR}/extensions/a.ts"),
            format!("-{AGENT_DIR}/extensions/a.ts")
        ])
    );

    selector.resource_list().handle_input(" ");
    assert_eq!(
        project_settings(&manager)["extensions"],
        json!([
            format!("{AGENT_DIR}/extensions/a.ts"),
            format!("+{AGENT_DIR}/extensions/a.ts")
        ])
    );

    selector.resource_list().handle_input(" ");
    assert_eq!(project_settings(&manager)["extensions"], json!([]));
}

#[test]
fn global_scope_package_toggle_appends_filters_to_the_source_entry() {
    setup(&[]);
    let manager = manager_with(&json!({ "packages": ["npm:pi-tools"] }), &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            "/home/user/pkg/extensions/bar.ts",
            "npm:pi-tools",
            SourceScope::User,
            SourceOrigin::Package,
            Some("/home/user/pkg"),
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);

    selector.resource_list().handle_input(" ");
    assert_eq!(
        global_settings(&manager)["packages"],
        json!([{ "source": "npm:pi-tools", "extensions": ["-extensions/bar.ts"] }])
    );

    selector.resource_list().handle_input(" ");
    assert_eq!(
        global_settings(&manager)["packages"],
        json!([{ "source": "npm:pi-tools", "extensions": ["+extensions/bar.ts"] }])
    );
}

#[test]
fn renders_group_labels_resource_types_and_scope_hint() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let paths = resolved_paths(
        vec![
            resource(
                "/home/user/pkg/extensions/bar.ts",
                "npm:pi-tools",
                SourceScope::User,
                SourceOrigin::Package,
                Some("/home/user/pkg"),
                true,
            ),
            resource(
                &format!("{AGENT_DIR}/extensions/local.ts"),
                "auto",
                SourceScope::User,
                SourceOrigin::TopLevel,
                None,
                false,
            ),
        ],
        vec![resource(
            &format!("{AGENT_DIR}/skills/my-skill/SKILL.md"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);

    let rendered = output(&selector, 100);
    assert!(rendered.contains("Global Resources"), "{rendered}");
    assert!(rendered.contains("~/.pi/agent/settings.json"), "{rendered}");
    assert!(rendered.contains("switch mode"), "{rendered}");
    assert!(rendered.contains("toggle"), "{rendered}");
    assert!(rendered.contains("npm:pi-tools (user)"), "{rendered}");
    assert!(
        rendered.contains(&format!("User ({AGENT_DIR}/)")),
        "{rendered}"
    );
    assert!(rendered.contains("Extensions"), "{rendered}");
    assert!(rendered.contains("Skills"), "{rendered}");
    assert!(rendered.contains("bar.ts"), "{rendered}");
    assert!(rendered.contains("local.ts"), "{rendered}");
    assert!(rendered.contains("my-skill"), "{rendered}");
    assert!(rendered.contains("[x]"), "{rendered}");
    assert!(rendered.contains("[ ]"), "{rendered}");
}

#[test]
fn renders_project_scope_with_inherited_global_dimming() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let inherited = resource(
        &format!("{AGENT_DIR}/extensions/a.ts"),
        "auto",
        SourceScope::User,
        SourceOrigin::TopLevel,
        None,
        true,
    );
    let project_item = resource(
        &format!("{CWD}/.pi/extensions/local.ts"),
        "auto",
        SourceScope::Project,
        SourceOrigin::TopLevel,
        None,
        true,
    );
    let selector = selector(
        &ScopedResolvedPaths {
            global: resolved_paths(vec![inherited.clone()], Vec::new(), Vec::new(), Vec::new()),
            project: resolved_paths(
                vec![inherited, project_item],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
        },
        &manager,
        ConfigWriteScope::Project,
        true,
    );

    let rendered = output(&selector, 120);
    assert!(rendered.contains("Project Local Resources"), "{rendered}");
    assert!(
        rendered.contains(".pi/settings.json · inherited global resources are dimmed"),
        "{rendered}"
    );
    assert!(rendered.contains("cycle inherit/+/-"), "{rendered}");
    assert!(rendered.contains("· inherited global"), "{rendered}");
    assert!(rendered.contains("inherited global"), "{rendered}");
}

#[test]
fn search_filters_resources_and_reports_empty_results() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            &format!("{AGENT_DIR}/extensions/alpha.ts"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);
    let list = selector.resource_list();

    list.handle_input("alpha");
    let rendered = output(&selector, 100);
    assert!(rendered.contains("alpha.ts"), "{rendered}");
    assert!(!rendered.contains("User settings"), "{rendered}");

    list.handle_input("zzz");
    let rendered = output(&selector, 100);
    assert!(rendered.contains("No resources found"), "{rendered}");
}

#[test]
fn tab_switches_to_project_scope_when_available() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let selector = empty_selector(&manager, true);

    assert_eq!(selector.write_scope(), ConfigWriteScope::Global);
    selector.handle_input("\t");
    assert_eq!(selector.write_scope(), ConfigWriteScope::Project);
    let rendered = output(&selector, 100);
    assert!(rendered.contains("Project Local Resources"), "{rendered}");

    selector.handle_input("\t");
    assert_eq!(selector.write_scope(), ConfigWriteScope::Global);
}

#[test]
fn tab_is_inert_without_project_mode() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let selector = empty_selector(&manager, false);

    let rendered = output(&selector, 100);
    assert!(!rendered.contains("switch mode"), "{rendered}");
    selector.handle_input("\t");
    assert_eq!(selector.write_scope(), ConfigWriteScope::Global);
}

#[test]
fn renders_home_prefixed_labels_and_settings_sources() {
    setup(&[]);
    let home = pi_coding_agent::config::home_dir();
    let manager = manager_with(&Value::Null, &Value::Null);
    let paths = resolved_paths(
        vec![
            resource(
                &format!("{home}/custom/extensions/a.ts"),
                "auto",
                SourceScope::User,
                SourceOrigin::TopLevel,
                Some(&format!("{home}/custom")),
                true,
            ),
            resource(
                &format!("{home}/extensions/home.ts"),
                "auto",
                SourceScope::User,
                SourceOrigin::TopLevel,
                Some(&home),
                true,
            ),
            resource(
                &format!("{AGENT_DIR}/extensions/settings.ts"),
                "local",
                SourceScope::User,
                SourceOrigin::TopLevel,
                None,
                true,
            ),
            resource(
                &format!("{AGENT_DIR}/extensions/temporary.ts"),
                "npm:temp-pkg",
                SourceScope::Temporary,
                SourceOrigin::Package,
                Some(AGENT_DIR),
                true,
            ),
            resource(
                "/home/user/other/extensions/other.ts",
                "npm:other",
                SourceScope::User,
                SourceOrigin::Package,
                Some("/home/user/other"),
                true,
            ),
            resource(
                &format!("{CWD}/.pi/extensions/project.ts"),
                "local",
                SourceScope::Project,
                SourceOrigin::TopLevel,
                None,
                true,
            ),
        ],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);

    let rendered = output(&selector, 120);
    assert!(rendered.contains("User (~/custom/)"), "{rendered}");
    assert!(rendered.contains("User (~/)"), "{rendered}");
    assert!(rendered.contains("User settings"), "{rendered}");
    assert!(rendered.contains("npm:temp-pkg (temporary)"), "{rendered}");
    assert!(rendered.contains("npm:other (user)"), "{rendered}");
    assert!(rendered.contains("Project settings"), "{rendered}");
}

#[test]
fn renders_every_resource_type_and_extension_subfolder_names() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            &format!("{AGENT_DIR}/extensions/nested/deep.ts"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        vec![resource(
            &format!("{AGENT_DIR}/skills/my-skill/SKILL.md"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        vec![resource(
            &format!("{AGENT_DIR}/prompts/review.md"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        vec![resource(
            &format!("{AGENT_DIR}/themes/dusk.json"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);

    let rendered = output(&selector, 120);
    assert!(rendered.contains("nested/deep.ts"), "{rendered}");
    assert!(rendered.contains("my-skill"), "{rendered}");
    assert!(rendered.contains("review.md"), "{rendered}");
    assert!(rendered.contains("dusk.json"), "{rendered}");
    assert!(rendered.contains("Prompts"), "{rendered}");
    assert!(rendered.contains("Themes"), "{rendered}");
}

#[test]
fn project_scope_toggles_project_top_level_resource() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            &format!("{CWD}/.pi/extensions/local.ts"),
            "local",
            SourceScope::Project,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_scoped(empty_paths(), paths, &manager, ConfigWriteScope::Project);
    let list = selector.resource_list();

    list.handle_input(" ");
    assert_eq!(
        project_settings(&manager)["extensions"],
        json!(["-extensions/local.ts"])
    );
    let rendered = output(&selector, 120);
    assert!(rendered.contains("[-]"), "{rendered}");
    assert!(rendered.contains("project unload"), "{rendered}");

    list.handle_input(" ");
    assert_eq!(
        project_settings(&manager)["extensions"],
        json!(["+extensions/local.ts"])
    );
    let rendered = output(&selector, 120);
    assert!(rendered.contains("[+]"), "{rendered}");
    assert!(rendered.contains("project load"), "{rendered}");

    list.handle_input("\r");
    assert_eq!(project_settings(&manager)["extensions"], json!([]));
}

#[test]
fn project_scope_cycles_disabled_inherited_resource_through_load_and_unload() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let inherited = resource(
        &format!("{AGENT_DIR}/extensions/a.ts"),
        "auto",
        SourceScope::User,
        SourceOrigin::TopLevel,
        None,
        false,
    );
    let paths = resolved_paths(vec![inherited], Vec::new(), Vec::new(), Vec::new());
    let selector = selector_both(paths, &manager, ConfigWriteScope::Project);
    let list = selector.resource_list();
    let path = format!("{AGENT_DIR}/extensions/a.ts");

    let rendered = output(&selector, 120);
    assert!(rendered.contains("[ ]"), "{rendered}");

    list.handle_input(" ");
    assert_eq!(
        project_settings(&manager)["extensions"],
        json!([path.clone(), format!("+{path}")])
    );

    list.handle_input(" ");
    assert_eq!(
        project_settings(&manager)["extensions"],
        json!([path.clone(), format!("-{path}")])
    );

    list.handle_input(" ");
    assert_eq!(project_settings(&manager)["extensions"], json!([]));
}

#[test]
fn project_scope_package_override_uses_project_relative_local_source() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let package_root = format!("{AGENT_DIR}/../pkgs/tools");
    let paths = resolved_paths(
        vec![resource(
            &format!("{package_root}/extensions/x.ts"),
            "../pkgs/tools",
            SourceScope::User,
            SourceOrigin::Package,
            Some(&package_root),
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Project);
    let list = selector.resource_list();

    list.handle_input(" ");
    assert_eq!(
        project_settings(&manager)["packages"],
        json!([{
            "source": "../../.pi/pkgs/tools",
            "autoload": false,
            "extensions": ["-extensions/x.ts"]
        }])
    );

    list.handle_input(" ");
    list.handle_input(" ");
    assert_eq!(project_settings(&manager)["packages"], json!([]));
}

#[test]
fn project_package_empty_filter_reads_as_unload_and_cycles_to_load() {
    setup(&[]);
    let manager = manager_with(
        &Value::Null,
        &json!({ "packages": [{ "source": "npm:pi-tools", "extensions": [] }] }),
    );
    let paths = resolved_paths(
        vec![resource(
            "/home/user/pkg/extensions/bar.ts",
            "npm:pi-tools",
            SourceScope::User,
            SourceOrigin::Package,
            Some("/home/user/pkg"),
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Project);

    selector.resource_list().handle_input(" ");
    assert_eq!(
        project_settings(&manager)["packages"],
        json!([{ "source": "npm:pi-tools", "extensions": ["+extensions/bar.ts"] }])
    );
}

#[test]
fn search_matches_resource_type_and_path() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            &format!("{AGENT_DIR}/extensions/alpha.ts"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
        vec![resource(
            &format!("{AGENT_DIR}/prompts/beta.md"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);
    let list = selector.resource_list();

    list.handle_input("prompts");
    let rendered = output(&selector, 120);
    assert!(rendered.contains("beta.md"), "{rendered}");
    assert!(!rendered.contains("alpha.ts"), "{rendered}");

    for _ in 0.."prompts".len() {
        list.handle_input("\x7f");
    }
    list.handle_input("extensions/alpha");
    let rendered = output(&selector, 120);
    assert!(rendered.contains("alpha.ts"), "{rendered}");
    assert!(!rendered.contains("beta.md"), "{rendered}");
}

fn one_of_each(base: &str, scope: SourceScope) -> ResolvedPaths {
    resolved_paths(
        vec![resource(
            &format!("{base}/extensions/e.ts"),
            "local",
            scope,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        vec![resource(
            &format!("{base}/skills/s/SKILL.md"),
            "local",
            scope,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        vec![resource(
            &format!("{base}/prompts/p.md"),
            "local",
            scope,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        vec![resource(
            &format!("{base}/themes/t.json"),
            "local",
            scope,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
    )
}

#[test]
fn global_scope_toggles_every_resource_type() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let selector = selector_scoped(
        one_of_each(AGENT_DIR, SourceScope::User),
        empty_paths(),
        &manager,
        ConfigWriteScope::Global,
    );
    let list = selector.resource_list();

    for _ in 0..4 {
        list.handle_input(" ");
        list.handle_input("\x1b[B");
    }

    let settings = global_settings(&manager);
    assert_eq!(settings["extensions"], json!(["-extensions/e.ts"]));
    assert_eq!(settings["skills"], json!(["-skills/s/SKILL.md"]));
    assert_eq!(settings["prompts"], json!(["-prompts/p.md"]));
    assert_eq!(settings["themes"], json!(["-themes/t.json"]));
}

#[test]
fn project_scope_toggles_every_resource_type() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let project_base = format!("{CWD}/.pi");
    let selector = selector_scoped(
        empty_paths(),
        one_of_each(&project_base, SourceScope::Project),
        &manager,
        ConfigWriteScope::Project,
    );
    let list = selector.resource_list();

    for _ in 0..4 {
        list.handle_input(" ");
        list.handle_input("\x1b[B");
    }

    let settings = project_settings(&manager);
    assert_eq!(settings["extensions"], json!(["-extensions/e.ts"]));
    assert_eq!(settings["skills"], json!(["-skills/s/SKILL.md"]));
    assert_eq!(settings["prompts"], json!(["-prompts/p.md"]));
    assert_eq!(settings["themes"], json!(["-themes/t.json"]));
}

#[test]
fn component_surface_answers_contract_queries() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let selector = empty_selector(&manager, true);
    let list = selector.resource_list();

    assert!(selector.wants_input());
    assert!(list.wants_input());
    assert!(selector.as_focusable().is_some());
    assert!(list.as_focusable().is_some());
    assert_eq!(selector.children().len(), 8);
    selector.invalidate();
    list.invalidate();
    assert!(format!("{selector:?}").contains("write_scope"));
    assert!(format!("{list:?}").contains("selected_index"));
    assert!(format!("{:?}", ResourceType::Extensions).contains("Extensions"));
    assert!(format!("{:?}", ConfigWriteScope::Project).contains("Project"));
    assert!(format!("{:?}", ScopedResolvedPaths::default()).contains("global"));
    assert!(format!("{:?}", ConfigSelectorOptions::default()).contains("write_scope"));
}

#[test]
fn global_scope_package_toggle_ignores_unknown_source() {
    setup(&[]);
    let manager = manager_with(&json!({ "packages": ["npm:other"] }), &Value::Null);
    let paths = resolved_paths(
        vec![resource(
            "/home/user/pkg/extensions/bar.ts",
            "npm:pi-tools",
            SourceScope::User,
            SourceOrigin::Package,
            Some("/home/user/pkg"),
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_both(paths, &manager, ConfigWriteScope::Global);

    selector.resource_list().handle_input(" ");
    assert_eq!(global_settings(&manager)["packages"], json!(["npm:other"]));
}

#[test]
fn toggle_callback_reports_the_item_and_its_new_state() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let toggles = Rc::new(std::cell::RefCell::new(Vec::<(String, bool, bool)>::new()));
    let captured = Rc::clone(&toggles);
    let paths = resolved_paths(
        vec![resource(
            &format!("{AGENT_DIR}/extensions/a.ts"),
            "auto",
            SourceScope::User,
            SourceOrigin::TopLevel,
            None,
            true,
        )],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let selector = selector_with(
        &ScopedResolvedPaths {
            global: paths.clone(),
            project: paths,
        },
        &manager,
        ConfigSelectorOptions::default(),
        Rc::new(|| {}),
        Rc::new(|| {}),
    );
    *selector.resource_list().on_toggle.borrow_mut() = Some(Rc::new(move |item, enabled| {
        let _ = format!("{item:?}");
        captured
            .borrow_mut()
            .push((item.path.clone(), enabled, item.enabled()));
    }));

    selector.resource_list().handle_input(" ");
    assert_eq!(
        *toggles.borrow(),
        vec![(format!("{AGENT_DIR}/extensions/a.ts"), false, false)]
    );
}

#[test]
fn switch_write_scope_method_and_default_options() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let options = ConfigSelectorOptions::default();
    assert_eq!(options.write_scope, ConfigWriteScope::Global);
    assert!(options.project_mode_available);
    assert_eq!(options.terminal_height, None);

    let selector = empty_selector(&manager, true);
    selector.switch_write_scope();
    assert_eq!(selector.write_scope(), ConfigWriteScope::Project);
    let rendered = output(&selector, 100);
    assert!(rendered.contains("Project Local Resources"), "{rendered}");
}

#[test]
fn cancel_callback_fires_on_escape() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let (cancel_count, on_close) = cancellation();
    let selector = selector_with(
        &ScopedResolvedPaths::default(),
        &manager,
        ConfigSelectorOptions {
            terminal_height: Some(24),
            write_scope: ConfigWriteScope::Global,
            project_mode_available: true,
        },
        on_close,
        Rc::new(|| {}),
    );

    selector.resource_list().handle_input("\x1b");
    assert_eq!(cancel_count.get(), 1);
}

#[test]
fn focus_propagates_to_the_resource_list() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let selector = empty_selector(&manager, true);
    let list = selector.resource_list();

    assert!(!selector.is_focused());
    selector.set_focused(true);
    assert!(selector.is_focused());
    assert!(list.is_focused());
    selector.set_focused(false);
    assert!(!list.is_focused());
}

#[test]
fn navigation_skips_headers_and_pages_by_the_visible_window() {
    setup(&[]);
    let manager = manager_with(&Value::Null, &Value::Null);
    let extensions = (0..12)
        .map(|index| {
            resource(
                &format!("{AGENT_DIR}/extensions/a{index:02}.ts"),
                "auto",
                SourceScope::User,
                SourceOrigin::TopLevel,
                None,
                true,
            )
        })
        .collect();
    let paths = resolved_paths(extensions, Vec::new(), Vec::new(), Vec::new());
    let selector = selector_with(
        &ScopedResolvedPaths {
            global: paths.clone(),
            project: paths,
        },
        &manager,
        ConfigSelectorOptions {
            terminal_height: Some(13),
            ..ConfigSelectorOptions::default()
        },
        Rc::new(|| {}),
        Rc::new(|| {}),
    );
    let list = selector.resource_list();

    let selected_item = |rendered: &str| {
        rendered
            .lines()
            .find(|line| line.starts_with("> ") && line.contains(".ts"))
            .unwrap_or_default()
            .to_string()
    };

    assert!(selected_item(&output(&selector, 100)).contains("a00.ts"));
    list.handle_input("\x1b[B");
    assert!(selected_item(&output(&selector, 100)).contains("a01.ts"));
    list.handle_input("\x1b[B");
    list.handle_input("\x1b[A");
    assert!(selected_item(&output(&selector, 100)).contains("a01.ts"));
    list.handle_input("\x1b[A");
    list.handle_input("\x1b[A");
    assert!(selected_item(&output(&selector, 100)).contains("a00.ts"));

    list.handle_input("\x1b[6~");
    let rendered = output(&selector, 100);
    assert!(selected_item(&rendered).contains("a05.ts"), "{rendered}");
    assert!(rendered.contains("(6/12)"), "{rendered}");

    list.handle_input("\x1b[5~");
    assert!(selected_item(&output(&selector, 100)).contains("a00.ts"));
}
