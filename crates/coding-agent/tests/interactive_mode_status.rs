//! `test/interactive-mode-status.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements:
//! - Upstream's fake `this` objects with `vi.fn()` spied methods become the
//!   real frame; call-count assertions become rendered transcript rows, the
//!   active indicator kind, and the installed provider's behavior.
//! - `showLoadedResources`'s fake loader object becomes the pure
//!   [`render_loaded_resource_sections`] input built from the same fixtures;
//!   the frame-level quiet-startup branches still drive the real loader.
//! - `showExtensionCustom` has no in-process port (the extension host is
//!   out-of-process and the wire cannot carry component factories), so the
//!   overlay-focus regression is ported against the TUI layer the frame would
//!   mount on.
//! - `requestRender` call counts are not spied; the rendered outcomes are.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the provider and entry shapes; a malformed fixture is a test bug"
)]

mod common;

use std::rc::Rc;
use std::sync::Arc;

use common::interactive::{Harness, HarnessOptions};
use pi_coding_agent::modes::interactive::interactive_mode::{
    LoadedResource, LoadedResourcesOptions, LoadedResourcesView, LoadedSkillEntry,
    render_loaded_resource_sections,
};
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::source_info::SourceInfo;
use pi_coding_agent::utils::tools_manager::ToolStatus;
use pi_tui::tui::Component;

#[tokio::test]
async fn coalesces_immediately_sequential_status_messages() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness.frame.show_status("STATUS_ONE");
    assert_eq!(harness.frame.chat_container().children.borrow().len(), 2);
    assert!(harness.chat_text().contains("STATUS_ONE"));

    harness.frame.show_status("STATUS_TWO");
    assert_eq!(harness.frame.chat_container().children.borrow().len(), 2);
    let text = harness.chat_text();
    assert!(text.contains("STATUS_TWO"));
    assert!(!text.contains("STATUS_ONE"));
}

#[tokio::test]
async fn appends_a_new_status_line_if_something_else_was_added_in_between() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness.frame.show_status("STATUS_ONE");
    assert_eq!(harness.frame.chat_container().children.borrow().len(), 2);

    let other: Rc<dyn Component> = Rc::new(pi_tui::components::Text::with_padding("OTHER", 0, 0));
    harness.frame.chat_container().add_child(other);
    assert_eq!(harness.frame.chat_container().children.borrow().len(), 3);

    harness.frame.show_status("STATUS_TWO");
    assert_eq!(harness.frame.chat_container().children.borrow().len(), 5);
    assert!(harness.chat_text().contains("STATUS_TWO"));
}

#[tokio::test]
async fn renders_tool_updates_as_one_contiguous_group() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness
        .frame
        .show_managed_tool_status(&ToolStatus::Info("fd downloading".to_owned()));
    harness
        .frame
        .show_managed_tool_status(&ToolStatus::Info("rg downloading".to_owned()));
    harness
        .frame
        .show_managed_tool_status(&ToolStatus::Warning("rg failed".to_owned()));

    assert_eq!(harness.frame.chat_container().children.borrow().len(), 4);
    assert_eq!(
        harness.chat_text(),
        "fd downloading\n rg downloading\n Warning: rg failed"
    );
}

#[tokio::test]
async fn applies_expansion_state_to_the_rendered_summary_rows() {
    let harness = Harness::new(HarnessOptions {
        seed_manager: Some(Box::new(|manager| {
            let first_kept = manager
                .append_message(common::agent_session::user_message("kept prompt", 1))
                .expect("user");
            manager
                .append_compaction("expanded summary", &first_kept, 123, None, None, None)
                .expect("compaction");
        })),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.render_initial_messages();

    let collapsed = harness.chat_text();
    assert!(
        collapsed.contains("Compacted from 123 tokens"),
        "{collapsed}"
    );
    assert!(!collapsed.contains("expanded summary"), "{collapsed}");
    harness.frame.set_tools_expanded(true);
    let text = harness.chat_text();
    assert!(text.contains("expanded summary"), "{text}");
    assert!(text.contains("Tool output: expanded"));
}

#[tokio::test]
async fn persists_theme_changes_from_the_extension_context() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;

    let result = harness.frame.ui_context().set_theme("light");
    harness.frame.drain_ui_requests();
    let result = result.await;

    assert!(result.success);
    assert_eq!(
        harness
            .frame
            .session()
            .settings_manager()
            .lock()
            .expect("settings")
            .get_theme()
            .as_deref(),
        Some("light")
    );
}

#[tokio::test]
async fn does_not_persist_invalid_theme_names() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;

    let result = harness.frame.ui_context().set_theme("__missing_theme__");
    harness.frame.drain_ui_requests();
    let result = result.await;

    assert!(!result.success);
    assert_ne!(
        harness
            .frame
            .session()
            .settings_manager()
            .lock()
            .expect("settings")
            .get_theme()
            .as_deref(),
        Some("__missing_theme__")
    );
}

/// A pass-through provider that contributes one trigger character.
struct WrappedProvider {
    inner: Arc<dyn pi_tui::autocomplete::AutocompleteProvider>,
    trigger: String,
}

impl pi_tui::autocomplete::AutocompleteProvider for WrappedProvider {
    fn trigger_characters(&self) -> Vec<String> {
        vec![self.trigger.clone()]
    }

    fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        options: &pi_tui::autocomplete::AutocompleteQueryOptions,
    ) -> Option<pi_tui::autocomplete::AutocompleteSuggestions> {
        self.inner
            .get_suggestions(lines, cursor_line, cursor_col, options)
    }

    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &pi_tui::autocomplete::AutocompleteItem,
        prefix: &str,
    ) -> pi_tui::autocomplete::AutocompleteApplyResult {
        self.inner
            .apply_completion(lines, cursor_line, cursor_col, item, prefix)
    }
}

fn wrapping(
    trigger: &str,
) -> pi_coding_agent::modes::interactive::interactive_mode::AutocompleteProviderFactory {
    let trigger = trigger.to_owned();
    Rc::new(
        move |current: Arc<dyn pi_tui::autocomplete::AutocompleteProvider>| {
            Arc::new(WrappedProvider {
                inner: current,
                trigger: trigger.clone(),
            })
        },
    )
}

#[tokio::test]
async fn stores_wrapper_factories_and_rebuilds_autocomplete_immediately() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness.frame.add_autocomplete_provider(wrapping("$"));

    let provider = harness
        .frame
        .autocomplete_provider()
        .expect("the provider is rebuilt");
    assert_eq!(provider.trigger_characters(), vec!["$".to_owned()]);
}

#[tokio::test]
async fn stacks_wrapper_factories_over_a_fresh_base_provider() {
    use pi_tui::autocomplete::AutocompleteProvider;

    let harness = Harness::new(HarnessOptions::default()).await;
    let calls = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let record = Arc::clone(&calls);
    let outer: pi_coding_agent::modes::interactive::interactive_mode::AutocompleteProviderFactory =
        Rc::new(move |current: Arc<dyn AutocompleteProvider>| {
            let provider: Arc<dyn AutocompleteProvider> = Arc::new(RecordingProvider {
                inner: current,
                name: "wrap2",
                calls: Arc::clone(&record),
            });
            provider
        });
    let record = Arc::clone(&calls);
    let inner: pi_coding_agent::modes::interactive::interactive_mode::AutocompleteProviderFactory =
        Rc::new(move |current: Arc<dyn AutocompleteProvider>| {
            let provider: Arc<dyn AutocompleteProvider> = Arc::new(RecordingProvider {
                inner: current,
                name: "wrap1",
                calls: Arc::clone(&record),
            });
            provider
        });
    harness.frame.add_autocomplete_provider(inner);
    harness.frame.add_autocomplete_provider(outer);

    let provider = harness.frame.autocomplete_provider().expect("provider");
    assert!(provider.should_trigger_file_completion(&["foo".to_owned()], 0, 3));
    assert_eq!(calls.lock().expect("calls").as_slice(), ["wrap2", "wrap1"]);
}

/// Records its name when the wrapped provider is asked to trigger.
struct RecordingProvider {
    inner: Arc<dyn pi_tui::autocomplete::AutocompleteProvider>,
    name: &'static str,
    calls: Arc<std::sync::Mutex<Vec<&'static str>>>,
}

impl pi_tui::autocomplete::AutocompleteProvider for RecordingProvider {
    fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        options: &pi_tui::autocomplete::AutocompleteQueryOptions,
    ) -> Option<pi_tui::autocomplete::AutocompleteSuggestions> {
        self.inner
            .get_suggestions(lines, cursor_line, cursor_col, options)
    }

    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &pi_tui::autocomplete::AutocompleteItem,
        prefix: &str,
    ) -> pi_tui::autocomplete::AutocompleteApplyResult {
        self.inner
            .apply_completion(lines, cursor_line, cursor_col, item, prefix)
    }

    fn should_trigger_file_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> bool {
        self.calls.lock().expect("calls").push(self.name);
        self.inner
            .should_trigger_file_completion(lines, cursor_line, cursor_col)
    }
}

#[tokio::test]
async fn merges_trigger_characters_from_wrapper_factories() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness.frame.add_autocomplete_provider(wrapping("$"));
    harness.frame.add_autocomplete_provider(wrapping("!"));

    assert_eq!(
        harness
            .frame
            .autocomplete_provider()
            .expect("provider")
            .trigger_characters(),
        vec!["$".to_owned(), "!".to_owned()]
    );
}

#[tokio::test]
async fn matches_model_command_arguments_across_provider_model_order() {
    let model = |provider: &str, id: &str, name: &str| {
        let mut model = common::model_layer::model(provider, id);
        model.name = name.to_owned();
        model
    };
    let harness = Harness::new(HarnessOptions {
        extra_models: vec![
            model("github-copilot", "gpt-5.2-codex", "GPT-5.2 Codex"),
            model("openai-codex", "gpt-5.5", "GPT-5.5"),
        ],
        ..HarnessOptions::default()
    })
    .await;
    let provider = harness.frame.create_base_autocomplete_provider();
    let line = "/model codexgpt";
    let suggestions = provider
        .get_suggestions(
            &[line.to_owned()],
            0,
            line.len(),
            &pi_tui::autocomplete::AutocompleteQueryOptions {
                signal: tokio_util::sync::CancellationToken::new(),
                force: false,
            },
        )
        .expect("suggestions");

    assert_eq!(
        suggestions
            .items
            .iter()
            .map(|item| item.value.clone())
            .collect::<Vec<_>>(),
        vec![
            "openai-codex/gpt-5.5".to_owned(),
            "github-copilot/gpt-5.2-codex".to_owned()
        ]
    );
}

#[tokio::test]
async fn matches_login_command_arguments_by_provider_id_and_name() {
    let harness = Harness::new(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        ..HarnessOptions::default()
    })
    .await;
    let provider = harness.frame.create_base_autocomplete_provider();
    let line = "/login subscription anthrop";
    let suggestions = provider
        .get_suggestions(
            &[line.to_owned()],
            0,
            line.len(),
            &pi_tui::autocomplete::AutocompleteQueryOptions {
                signal: tokio_util::sync::CancellationToken::new(),
                force: false,
            },
        )
        .expect("suggestions");

    assert_eq!(
        suggestions.items,
        vec![pi_tui::autocomplete::AutocompleteItem {
            value: "anthropic".to_owned(),
            label: "anthropic".to_owned(),
            description: Some("Anthropic · subscription/API key".to_owned()),
        }]
    );
}

fn source(
    path: &str,
    source: &str,
    scope: pi_coding_agent::source_info::SourceScope,
    origin: pi_coding_agent::source_info::SourceOrigin,
    base_dir: Option<&str>,
) -> SourceInfo {
    SourceInfo {
        path: path.to_owned(),
        source: source.to_owned(),
        scope,
        origin,
        base_dir: base_dir.map(str::to_owned),
    }
}

fn extension_fixtures() -> Vec<LoadedResource> {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let extension = |path: &str, source_name: &str, scope, origin, base: &str| LoadedResource {
        path: path.to_owned(),
        source_info: Some(source(path, source_name, scope, origin, Some(base))),
    };
    vec![
        extension(
            "/tmp/project/.pi/extensions/answer.ts",
            "local",
            SourceScope::Project,
            SourceOrigin::TopLevel,
            "/tmp/project/.pi/extensions",
        ),
        extension(
            "/tmp/project/.pi/extensions/local-index/index.ts",
            "local",
            SourceScope::Project,
            SourceOrigin::TopLevel,
            "/tmp/project/.pi/extensions",
        ),
        extension(
            "/tmp/agent/extensions/user-index/index.ts",
            "local",
            SourceScope::User,
            SourceOrigin::TopLevel,
            "/tmp/agent/extensions",
        ),
        extension(
            "/tmp/project/.pi/npm/node_modules/pi-markdown-preview/extensions/index.ts",
            "npm:pi-markdown-preview",
            SourceScope::Project,
            SourceOrigin::Package,
            "/tmp/project/.pi/npm/node_modules/pi-markdown-preview",
        ),
        extension(
            "/tmp/project/.pi/npm/node_modules/@scope/pi-scoped/extensions/index.ts",
            "npm:@scope/pi-scoped",
            SourceScope::Project,
            SourceOrigin::Package,
            "/tmp/project/.pi/npm/node_modules/@scope/pi-scoped",
        ),
        extension(
            "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents/extensions/index.ts",
            "git:github.com/HazAT/pi-interactive-subagents",
            SourceScope::Project,
            SourceOrigin::Package,
            "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents",
        ),
        extension(
            "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents/extensions/subagents/index.ts",
            "git:github.com/HazAT/pi-interactive-subagents",
            SourceScope::Project,
            SourceOrigin::Package,
            "/tmp/project/.pi/git/github.com/HazAT/pi-interactive-subagents",
        ),
        extension(
            "/tmp/temp/cli-extension.ts",
            "cli",
            SourceScope::Temporary,
            SourceOrigin::TopLevel,
            "/tmp/temp",
        ),
    ]
}

fn resources_view() -> LoadedResourcesView {
    LoadedResourcesView {
        context_files: Vec::new(),
        skills: vec![LoadedSkillEntry {
            name: "commit".to_owned(),
            resource: LoadedResource {
                path: "/tmp/skill/SKILL.md".to_owned(),
                source_info: Some(source(
                    "/tmp/skill/SKILL.md",
                    "local",
                    pi_coding_agent::source_info::SourceScope::Project,
                    pi_coding_agent::source_info::SourceOrigin::TopLevel,
                    Some("/tmp/skill"),
                )),
            },
        }],
        prompts: Vec::new(),
        extensions: Vec::new(),
        themes: Vec::new(),
    }
}

fn section_text(view: &LoadedResourcesView, expanded: bool, name: &str) -> String {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let options = LoadedResourcesOptions {
        expanded,
        cwd: "/tmp/project".to_owned(),
        home: None,
    };
    let sections = render_loaded_resource_sections(view, &options);
    let body = sections
        .into_iter()
        .find(|section| section.name == name)
        .map_or_else(String::new, |section| {
            if expanded {
                section.expanded
            } else {
                section.collapsed
            }
        });
    pi_coding_agent::utils::ansi::strip_ansi(&body)
        .replace('\\', "/")
        .lines()
        .map(|line| line.trim_end().to_owned())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

#[tokio::test]
async fn shows_a_compact_resource_listing_by_default() {
    let text = section_text(&resources_view(), false, "Skills");
    assert!(text.contains("[Skills]"), "{text}");
    assert!(text.contains("commit"), "{text}");
    assert!(!text.contains("resource-list"), "{text}");
}

#[tokio::test]
async fn shows_full_resource_listing_when_expanded() {
    let text = section_text(&resources_view(), true, "Skills");
    assert!(text.contains("[Skills]"), "{text}");
    assert!(text.contains("/tmp/skill/SKILL.md"), "{text}");
    assert!(!text.contains("commit"), "{text}");
}

#[tokio::test]
async fn abbreviates_extensions_in_compact_listing() {
    let view = LoadedResourcesView {
        extensions: vec![
            LoadedResource {
                path: "/tmp/extensions/answer.ts".to_owned(),
                source_info: None,
            },
            LoadedResource {
                path: "/tmp/extensions/btw.ts".to_owned(),
                source_info: None,
            },
        ],
        ..LoadedResourcesView::default()
    };

    let text = section_text(&view, false, "Extensions");

    assert!(text.contains("[Extensions]"), "{text}");
    assert!(text.contains("answer.ts, btw.ts"), "{text}");
    assert!(!text.contains("extensions/answer.ts"), "{text}");
}

#[tokio::test]
async fn captures_mixed_extension_layouts_in_compact_output() {
    let view = LoadedResourcesView {
        extensions: extension_fixtures(),
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  @scope/pi-scoped, answer.ts, cli-extension.ts, HazAT/pi-interactive-subagents, HazAT/pi-interactive-subagents:subagents, local-index, pi-markdown-preview, user-index"
    );
}

#[tokio::test]
async fn captures_mixed_extension_layouts_in_expanded_output() {
    let view = LoadedResourcesView {
        extensions: extension_fixtures(),
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, true, "Extensions"),
        "[Extensions]\n  project\n    /tmp/project/.pi/extensions/answer.ts\n    /tmp/project/.pi/extensions/local-index\n    git:github.com/HazAT/pi-interactive-subagents\n      extensions\n      extensions/subagents\n    npm:@scope/pi-scoped\n      extensions\n    npm:pi-markdown-preview\n      extensions\n  user\n    /tmp/agent/extensions/user-index\n  path\n    /tmp/temp/cli-extension.ts"
    );
}

fn extension(
    path: &str,
    source_name: &str,
    scope: pi_coding_agent::source_info::SourceScope,
    origin: pi_coding_agent::source_info::SourceOrigin,
    base: &str,
) -> LoadedResource {
    LoadedResource {
        path: path.to_owned(),
        source_info: Some(source(path, source_name, scope, origin, Some(base))),
    }
}

#[tokio::test]
async fn adds_more_parent_folders_until_local_extension_labels_are_unique() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let extensions = ["alpha", "beta", "gamma"]
        .into_iter()
        .map(|name| {
            let path = format!("/tmp/{name}/one/index.ts");
            extension(
                &path,
                "cli",
                SourceScope::Temporary,
                SourceOrigin::TopLevel,
                &format!("/tmp/{name}"),
            )
        })
        .collect();
    let view = LoadedResourcesView {
        extensions,
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  alpha/one, beta/one, gamma/one"
    );
}

#[tokio::test]
async fn strips_index_files_from_local_extension_labels() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    for file in ["index.ts", "index.js"] {
        let path = format!("/tmp/extensions/plan-mode/{file}");
        let view = LoadedResourcesView {
            extensions: vec![extension(
                &path,
                "local",
                SourceScope::Project,
                SourceOrigin::TopLevel,
                "/tmp/extensions",
            )],
            ..LoadedResourcesView::default()
        };
        assert_eq!(
            section_text(&view, false, "Extensions"),
            "[Extensions]\n  plan-mode"
        );
    }
}

#[tokio::test]
async fn mixes_single_file_and_subdirectory_index_extensions() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let view = LoadedResourcesView {
        extensions: vec![
            extension(
                "/tmp/extensions/webfetch.ts",
                "local",
                SourceScope::Project,
                SourceOrigin::TopLevel,
                "/tmp/extensions",
            ),
            extension(
                "/tmp/extensions/plan-mode/index.ts",
                "local",
                SourceScope::Project,
                SourceOrigin::TopLevel,
                "/tmp/extensions",
            ),
        ],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  plan-mode, webfetch.ts"
    );
}

#[tokio::test]
async fn keeps_non_index_subdirectory_files_as_file_names() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let view = LoadedResourcesView {
        extensions: vec![extension(
            "/tmp/extensions/my-ext/main.ts",
            "local",
            SourceScope::Project,
            SourceOrigin::TopLevel,
            "/tmp/extensions",
        )],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  main.ts"
    );
}

#[tokio::test]
async fn labels_npm_siblings_relative_to_the_declaring_package() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let base = "/tmp/project/.pi/npm/node_modules/primary-package";
    let view = LoadedResourcesView {
        extensions: vec![
            extension(
                "/tmp/project/.pi/npm/node_modules/primary-package/index.ts",
                "npm:primary-package",
                SourceScope::Project,
                SourceOrigin::Package,
                base,
            ),
            extension(
                "/tmp/project/.pi/npm/node_modules/sibling-package/index.ts",
                "npm:primary-package",
                SourceScope::Project,
                SourceOrigin::Package,
                base,
            ),
        ],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  primary-package, primary-package:../sibling-package"
    );
}

#[tokio::test]
async fn labels_windows_npm_siblings_relative_to_the_declaring_package() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let base = "C:\\Users\\me\\.pi\\agent\\npm\\node_modules\\primary-package";
    let view = LoadedResourcesView {
        extensions: vec![
            extension(
                "C:\\Users\\me\\.pi\\agent\\npm\\node_modules\\primary-package\\index.ts",
                "npm:primary-package",
                SourceScope::User,
                SourceOrigin::Package,
                base,
            ),
            extension(
                "C:\\Users\\me\\.pi\\agent\\npm\\node_modules\\sibling-package\\index.ts",
                "npm:primary-package",
                SourceScope::User,
                SourceOrigin::Package,
                base,
            ),
        ],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  primary-package, primary-package:../sibling-package"
    );
}

#[tokio::test]
async fn shows_context_paths_relative_to_cwd_while_preserving_full_external_paths() {
    let home = "/home/user";
    let cwd = "/home/user/Development/pi-mono";
    let view = LoadedResourcesView {
        context_files: vec![
            "/home/user/.pi/agent/AGENTS.md".to_owned(),
            "/home/user/Development/pi-mono/AGENTS.md".to_owned(),
        ],
        ..LoadedResourcesView::default()
    };
    let options = LoadedResourcesOptions {
        expanded: false,
        cwd: cwd.to_owned(),
        home: Some(home.to_owned()),
    };
    let sections = render_loaded_resource_sections(&view, &options);
    let body = sections
        .iter()
        .find(|section| section.name == "Context")
        .expect("context")
        .collapsed
        .clone();
    let text = pi_coding_agent::utils::ansi::strip_ansi(&body);

    assert!(text.contains("~/.pi/agent/AGENTS.md, AGENTS.md"), "{text}");
    assert!(
        !text.contains("/home/user/Development/pi-mono/AGENTS.md"),
        "{text}"
    );
}

#[tokio::test]
async fn shows_full_context_paths_when_expanded() {
    let home = "/home/user";
    let cwd = "/home/user/Development/pi-mono";
    let view = LoadedResourcesView {
        context_files: vec![
            "/home/user/.pi/agent/AGENTS.md".to_owned(),
            "/home/user/Development/pi-mono/AGENTS.md".to_owned(),
        ],
        ..LoadedResourcesView::default()
    };
    let options = LoadedResourcesOptions {
        expanded: true,
        cwd: cwd.to_owned(),
        home: Some(home.to_owned()),
    };
    let sections = render_loaded_resource_sections(&view, &options);
    let body = &sections
        .iter()
        .find(|section| section.name == "Context")
        .expect("context")
        .expanded;
    let text = pi_coding_agent::utils::ansi::strip_ansi(body);

    assert!(text.contains("~/.pi/agent/AGENTS.md"), "{text}");
    assert!(text.contains("~/Development/pi-mono/AGENTS.md"), "{text}");
}

#[tokio::test]
async fn shows_system_prompt_context_paths_before_project_context_files() {
    let view = LoadedResourcesView {
        context_files: vec![
            "/tmp/project/.pi/SYSTEM.md".to_owned(),
            "/tmp/project/.pi/APPEND_SYSTEM.md".to_owned(),
            "/tmp/project/AGENTS.md".to_owned(),
        ],
        ..LoadedResourcesView::default()
    };
    let options = LoadedResourcesOptions {
        expanded: false,
        cwd: "/tmp/project".to_owned(),
        home: None,
    };
    let sections = render_loaded_resource_sections(&view, &options);
    let body = &sections
        .iter()
        .find(|section| section.name == "Context")
        .expect("context")
        .collapsed;
    let text = pi_coding_agent::utils::ansi::strip_ansi(body);

    assert!(
        text.contains(".pi/SYSTEM.md, .pi/APPEND_SYSTEM.md, AGENTS.md"),
        "{text}"
    );
}

fn skill_fixtures() -> common::interactive::ResourceFixtures {
    common::interactive::ResourceFixtures {
        skills: vec![(
            "commit".to_owned(),
            "/tmp/skill/SKILL.md".to_owned(),
            source(
                "/tmp/skill/SKILL.md",
                "local",
                pi_coding_agent::source_info::SourceScope::Project,
                pi_coding_agent::source_info::SourceOrigin::TopLevel,
                Some("/tmp/skill"),
            ),
        )],
        ..common::interactive::ResourceFixtures::default()
    }
}

fn quiet_settings(quiet: bool) -> Settings {
    let mut settings = Settings::new();
    settings.insert("quietStartup".to_owned(), serde_json::Value::Bool(quiet));
    settings
}

#[tokio::test]
async fn shows_the_compact_listing_on_a_normal_startup() {
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(skill_fixtures()),
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.show_loaded_resources(false);

    let text = harness.resources_text();
    assert!(text.contains("[Skills]"), "{text}");
    assert!(text.contains("commit"), "{text}");
}

#[tokio::test]
async fn shows_the_full_listing_on_verbose_startup_even_when_tool_output_is_collapsed() {
    let harness = Harness::new(HarnessOptions {
        settings: quiet_settings(true),
        fixtures: Some(skill_fixtures()),
        verbose: true,
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.show_loaded_resources(false);

    let text = harness.resources_text();
    assert!(text.contains("[Skills]"), "{text}");
    assert!(text.contains("SKILL.md"), "{text}");
    assert!(!text.contains("commit"), "{text}");
}

#[tokio::test]
async fn does_not_show_a_listing_on_quiet_startup_during_reload() {
    let harness = Harness::new(HarnessOptions {
        settings: quiet_settings(true),
        fixtures: Some(skill_fixtures()),
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.show_loaded_resources(false);

    assert_eq!(
        harness
            .frame
            .loaded_resources_container()
            .children
            .borrow()
            .len(),
        0
    );
}

#[tokio::test]
async fn still_shows_diagnostics_on_quiet_startup_when_requested() {
    let mut fixtures = skill_fixtures();
    fixtures.skill_diagnostics = vec![pi_coding_agent::diagnostics::ResourceDiagnostic {
        kind: pi_coding_agent::diagnostics::ResourceDiagnosticKind::Warning,
        message: "duplicate skill name".to_owned(),
        path: None,
        collision: None,
    }];
    let harness = Harness::new(HarnessOptions {
        settings: quiet_settings(true),
        fixtures: Some(fixtures),
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.show_loaded_resources(true);

    let text = harness.resources_text();
    assert!(text.contains("[Skill conflicts]"), "{text}");
    assert!(!text.contains("[Skills]"), "{text}");
}

/// A focusable label that records its inputs and focus state.
struct FocusableLabel {
    label: &'static str,
    inputs: std::cell::RefCell<Vec<String>>,
    focused: std::cell::Cell<bool>,
}

impl FocusableLabel {
    fn new(label: &'static str) -> Rc<Self> {
        Rc::new(Self {
            label,
            inputs: std::cell::RefCell::new(Vec::new()),
            focused: std::cell::Cell::new(false),
        })
    }
}

impl Component for FocusableLabel {
    fn render(&self, _width: usize) -> Vec<String> {
        vec![self.label.to_string()]
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_input(&self, data: &str) {
        self.inputs.borrow_mut().push(data.to_owned());
    }

    fn invalidate(&self) {}

    fn as_focusable(&self) -> Option<&dyn pi_tui::tui::Focusable> {
        Some(self)
    }
}

impl pi_tui::tui::Focusable for FocusableLabel {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

#[tokio::test]
async fn overlay_custom_ui_reclaims_input_after_non_overlay_custom_ui_closes() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let terminal = common::interactive::RecordingTerminal::new(80, 24);
    let ui = pi_coding_agent::modes::interactive::tui_renderer::create_interactive_tui(
        pi_coding_agent::modes::interactive::tui_renderer::InteractiveTuiOptions {
            tui_mode: pi_tui::tui::TuiMode::Fullscreen,
            show_hardware_cursor: false,
            log_directory: "/tmp".to_string(),
            terminal: Some(Box::new(terminal.clone())),
            on_right_click_paste: None,
            fullscreen_copy_on_select: Some(false),
        },
    );
    let editor_container = Rc::new(pi_tui::tui::Container::default());
    let editor = FocusableLabel::new("EDITOR");
    let palette = FocusableLabel::new("PALETTE");
    let overlay = FocusableLabel::new("OVERLAY");
    let replacement = FocusableLabel::new("REPLACEMENT");
    let editor_container_child: Rc<dyn Component> = editor_container.clone();
    let editor_child: Rc<dyn Component> = editor.clone();
    editor_container.add_child(editor_child);
    ui.tui().add_child(editor_container_child);
    ui.tui().add_child(palette.clone());
    ui.tui().set_focus(Some(palette));
    ui.tui().start();

    let overlay_child: Rc<dyn Component> = overlay.clone();
    let _ = ui.tui().show_overlay(overlay_child, None);
    assert!(overlay.focused.get());

    let replacement_child: Rc<dyn Component> = replacement.clone();
    editor_container.clear();
    editor_container.add_child(replacement_child);
    let replacement_focus: Rc<dyn Component> = replacement.clone();
    ui.tui().set_focus(Some(replacement_focus));
    assert!(replacement.focused.get());

    // The replacement closes: the editor is restored, but the overlay keeps
    // the keyboard.
    let editor_child: Rc<dyn Component> = editor.clone();
    editor_container.clear();
    editor_container.add_child(editor_child);
    let editor_focus: Rc<dyn Component> = editor.clone();
    ui.tui().set_focus(Some(editor_focus));
    ui.tui().render_now(true);
    terminal.send_input("x");
    assert_eq!(overlay.inputs.borrow().as_slice(), ["x"]);
    assert!(editor.inputs.borrow().is_empty());
    assert!(overlay.focused.get());

    ui.tui().hide_overlay();
    assert!(!overlay.focused.get());
    ui.tui().stop(pi_tui::tui::TuiStopOptions::default());
}

#[tokio::test]
async fn multiple_index_files_with_unique_parent_dirs_need_no_disambiguation() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let view = LoadedResourcesView {
        extensions: vec![
            extension(
                "/tmp/extensions/foo/index.ts",
                "local",
                SourceScope::Project,
                SourceOrigin::TopLevel,
                "/tmp/extensions",
            ),
            extension(
                "/tmp/extensions/bar/index.ts",
                "local",
                SourceScope::Project,
                SourceOrigin::TopLevel,
                "/tmp/extensions",
            ),
        ],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  bar, foo"
    );
}

#[tokio::test]
async fn same_parent_dir_names_are_disambiguated_with_the_grandparent() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let view = LoadedResourcesView {
        extensions: vec![
            extension(
                "/tmp/alpha/tools/index.ts",
                "cli",
                SourceScope::Temporary,
                SourceOrigin::TopLevel,
                "/tmp/alpha",
            ),
            extension(
                "/tmp/beta/tools/index.ts",
                "cli",
                SourceScope::Temporary,
                SourceOrigin::TopLevel,
                "/tmp/beta",
            ),
        ],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  alpha/tools, beta/tools"
    );
}

#[tokio::test]
async fn package_extensions_still_strip_index_files() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let view = LoadedResourcesView {
        extensions: vec![extension(
            "/tmp/project/.pi/npm/node_modules/pi-markdown-preview/extensions/index.ts",
            "npm:pi-markdown-preview",
            SourceScope::Project,
            SourceOrigin::Package,
            "/tmp/project/.pi/npm/node_modules/pi-markdown-preview",
        )],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  pi-markdown-preview"
    );
}

#[tokio::test]
async fn package_labels_work_without_base_dir_provenance() {
    use pi_coding_agent::source_info::{SourceOrigin, SourceScope};
    let path = "/tmp/project/.pi/npm/node_modules/pi-markdown-preview/extensions/index.ts";
    let view = LoadedResourcesView {
        extensions: vec![LoadedResource {
            path: path.to_owned(),
            source_info: Some(SourceInfo {
                path: path.to_owned(),
                source: "npm:pi-markdown-preview".to_owned(),
                scope: SourceScope::Project,
                origin: SourceOrigin::Package,
                base_dir: None,
            }),
        }],
        ..LoadedResourcesView::default()
    };

    assert_eq!(
        section_text(&view, false, "Extensions"),
        "[Extensions]\n  pi-markdown-preview"
    );
}
