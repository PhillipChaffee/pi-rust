//! `settings-selector.test.ts` at pin `60e7e76`, ported 1:1, plus boundaries
//! (#206).

#[path = "common/selectors.rs"]
mod selectors;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::{Api, Modality, Model, ModelCost, ModelCostRates, ProviderId, Transport};
use pi_coding_agent::modes::interactive::components::settings_selector::{
    SettingsCallbacks, SettingsConfig, SettingsSelectorComponent, ThemeSubmenu, WarningSettings,
};
use pi_coding_agent::modes::interactive::components::settings_submenu::{
    SelectSubmenu, SteppedSubmenu, SteppedSubmenuOptions, SteppedSubmenuStep, SteppedSubmenuText,
};
use pi_coding_agent::modes::interactive::theme::theme::TerminalTheme;
use pi_coding_agent::settings_manager::{
    DefaultProjectTrust, FullscreenExitOutput, MermaidRenderingMode, QueueMode,
};
use pi_tui::components::scroll_view::ScrollViewScrollbar;
use pi_tui::components::select_list::SelectItem;
use pi_tui::components::settings_list::SettingsSubmenuDone;
use pi_tui::tui::{Component, TuiMode};

use selectors::{cancellation, click, output, setup};

fn config() -> SettingsConfig {
    SettingsConfig {
        auto_compact: true,
        default_model: String::new(),
        current_model: None,
        available_default_models: Vec::new(),
        show_images: true,
        image_width_cells: 80,
        auto_resize_images: true,
        block_images: false,
        enable_skill_commands: true,
        steering_mode: QueueMode::OneAtATime,
        follow_up_mode: QueueMode::OneAtATime,
        transport: Transport::Auto,
        http_idle_timeout_ms: 300_000,
        thinking_level: ThinkingLevel::Medium,
        available_thinking_levels: vec![
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::Xhigh,
            ThinkingLevel::Max,
        ],
        model_thinking_levels: BTreeMap::new(),
        current_theme: "dark".to_string(),
        terminal_theme: TerminalTheme::Dark,
        available_themes: Vec::new(),
        hide_thinking_block: false,
        mermaid_rendering_mode: MermaidRenderingMode::Streaming,
        show_cache_miss_notices: false,
        collapse_changelog: false,
        enable_install_telemetry: true,
        double_escape_action: "tree".to_string(),
        tree_filter_mode: "default".to_string(),
        show_hardware_cursor: false,
        editor_padding_x: 0,
        output_pad: 1,
        autocomplete_max_visible: 5,
        quiet_startup: false,
        default_project_trust: DefaultProjectTrust::Ask,
        clear_on_shrink: false,
        show_terminal_progress: false,
        tui_mode: TuiMode::Regular,
        fullscreen_exit_output: FullscreenExitOutput::Transcript,
        fullscreen_scrollbar: ScrollViewScrollbar::Auto,
        fullscreen_copy_on_select: true,
        warnings: WarningSettings::default(),
    }
}

fn noop_callbacks() -> SettingsCallbacks {
    SettingsCallbacks {
        on_auto_compact_change: Rc::new(|_| {}),
        on_show_images_change: Rc::new(|_| {}),
        on_image_width_cells_change: Rc::new(|_| {}),
        on_auto_resize_images_change: Rc::new(|_| {}),
        on_block_images_change: Rc::new(|_| {}),
        on_enable_skill_commands_change: Rc::new(|_| {}),
        on_steering_mode_change: Rc::new(|_| {}),
        on_follow_up_mode_change: Rc::new(|_| {}),
        on_transport_change: Rc::new(|_| {}),
        on_http_idle_timeout_ms_change: Rc::new(|_| {}),
        on_model_thinking_level_change: Rc::new(|_, _, _| {}),
        on_model_thinking_level_remove: Rc::new(|_, _| {}),
        on_theme_change: Rc::new(|_| {}),
        on_theme_preview: None,
        on_hide_thinking_block_change: Rc::new(|_| {}),
        on_mermaid_rendering_mode_change: Rc::new(|_| {}),
        on_show_cache_miss_notices_change: Rc::new(|_| {}),
        on_collapse_changelog_change: Rc::new(|_| {}),
        on_enable_install_telemetry_change: Rc::new(|_| {}),
        on_double_escape_action_change: Rc::new(|_| {}),
        on_tree_filter_mode_change: Rc::new(|_| {}),
        on_show_hardware_cursor_change: Rc::new(|_| {}),
        on_editor_padding_x_change: Rc::new(|_| {}),
        on_output_pad_change: Rc::new(|_| {}),
        on_autocomplete_max_visible_change: Rc::new(|_| {}),
        on_quiet_startup_change: Rc::new(|_| {}),
        on_default_project_trust_change: Rc::new(|_| {}),
        on_clear_on_shrink_change: Rc::new(|_| {}),
        on_show_terminal_progress_change: Rc::new(|_| {}),
        on_tui_mode_change: Rc::new(|_| {}),
        on_fullscreen_exit_output_change: Rc::new(|_| {}),
        on_fullscreen_scrollbar_change: Rc::new(|_| {}),
        on_fullscreen_copy_on_select_change: Rc::new(|_| {}),
        on_warnings_change: Rc::new(|_| {}),
        on_cancel: Rc::new(|| {}),
    }
}

fn component(config: &SettingsConfig, callbacks: SettingsCallbacks) -> SettingsSelectorComponent {
    SettingsSelectorComponent::new(config, callbacks)
}

#[test]
fn cycles_through_fullscreen_settings() {
    setup(&[]);
    let exit_changes = Rc::new(RefCell::new(Vec::<FullscreenExitOutput>::new()));
    let scrollbar_changes = Rc::new(RefCell::new(Vec::<ScrollViewScrollbar>::new()));
    let copy_changes = Rc::new(RefCell::new(Vec::<bool>::new()));
    let callbacks = SettingsCallbacks {
        on_fullscreen_exit_output_change: {
            let changes = Rc::clone(&exit_changes);
            Rc::new(move |value| changes.borrow_mut().push(value))
        },
        on_fullscreen_scrollbar_change: {
            let changes = Rc::clone(&scrollbar_changes);
            Rc::new(move |value| changes.borrow_mut().push(value))
        },
        on_fullscreen_copy_on_select_change: {
            let changes = Rc::clone(&copy_changes);
            Rc::new(move |value| changes.borrow_mut().push(value))
        },
        ..noop_callbacks()
    };

    let cycle = |label: &str, count: usize, config: SettingsConfig| {
        let selector = component(&config, callbacks.clone());
        let list = selector.get_settings_list();
        for character in label.chars() {
            list.handle_input(&character.to_string());
        }
        for _ in 0..count {
            list.handle_input("\r");
        }
    };

    cycle(
        "Fullscreen exit output",
        2,
        SettingsConfig {
            fullscreen_exit_output: FullscreenExitOutput::Transcript,
            ..config()
        },
    );
    assert_eq!(
        *exit_changes.borrow(),
        vec![
            FullscreenExitOutput::ResumeHint,
            FullscreenExitOutput::Transcript
        ]
    );
    cycle(
        "Fullscreen scrollbar",
        3,
        SettingsConfig {
            fullscreen_scrollbar: ScrollViewScrollbar::Auto,
            ..config()
        },
    );
    assert_eq!(
        *scrollbar_changes.borrow(),
        vec![
            ScrollViewScrollbar::Always,
            ScrollViewScrollbar::Hidden,
            ScrollViewScrollbar::Auto
        ]
    );
    cycle(
        "Fullscreen copy on select",
        2,
        SettingsConfig {
            fullscreen_copy_on_select: true,
            ..config()
        },
    );
    assert_eq!(*copy_changes.borrow(), vec![false, true]);
}

#[test]
fn keeps_the_configured_fixed_theme_marked_while_browsing() {
    setup(&[]);
    let previews = Rc::new(RefCell::new(Vec::<String>::new()));
    let callbacks = SettingsCallbacks {
        on_theme_preview: Some({
            let previews = Rc::clone(&previews);
            Rc::new(move |theme| previews.borrow_mut().push(theme.to_string()))
        }),
        ..noop_callbacks()
    };
    let selector = component(
        &SettingsConfig {
            current_theme: "dark".to_string(),
            available_themes: vec!["dark".to_string(), "light".to_string()],
            ..config()
        },
        callbacks,
    );
    let list = selector.get_settings_list();

    list.select_item("theme");
    list.handle_input("\r");
    let rendered = output(list.as_ref(), 120);
    assert!(rendered.contains("    Automatic"), "{rendered}");
    assert!(rendered.contains("→ ✓ dark"), "{rendered}");

    list.handle_input("\x1b[B");
    let rendered = output(list.as_ref(), 120);
    assert!(rendered.contains("  ✓ dark"), "{rendered}");
    assert!(rendered.contains("→   light"), "{rendered}");
}

#[test]
fn keeps_a_configured_automatic_theme_marked_while_browsing() {
    setup(&[]);
    let selector = component(
        &SettingsConfig {
            current_theme: "light/dark".to_string(),
            available_themes: vec!["dark".to_string(), "light".to_string(), "other".to_string()],
            ..config()
        },
        noop_callbacks(),
    );
    let list = selector.get_settings_list();

    list.select_item("theme");
    list.handle_input("\r");
    list.handle_input("\r");
    let rendered = output(list.as_ref(), 120);
    assert!(rendered.contains("→ ✓ light"), "{rendered}");

    list.handle_input("\x1b[B");
    let rendered = output(list.as_ref(), 120);
    assert!(rendered.contains("  ✓ light"), "{rendered}");
    assert!(rendered.contains("→   other"), "{rendered}");
}

fn thinking_model() -> Model {
    Model {
        id: "thinking-model".to_string(),
        name: "Thinking Model".to_string(),
        api: Api::from("openai-completions"),
        provider: ProviderId("test-provider".to_string()),
        base_url: "https://example.test/v1".to_string(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

#[test]
fn keeps_the_configured_per_model_thinking_level_marked_while_browsing() {
    setup(&[]);
    let model = thinking_model();
    let model_key = "test-provider/thinking-model".to_string();
    let mut model_thinking_levels = BTreeMap::new();
    model_thinking_levels.insert(model_key.clone(), ThinkingLevel::Medium);
    let selector = component(
        &SettingsConfig {
            default_model: model_key,
            available_default_models: vec![model],
            thinking_level: ThinkingLevel::High,
            model_thinking_levels,
            ..config()
        },
        noop_callbacks(),
    );
    let list = selector.get_settings_list();

    list.select_item("model-thinking");
    list.handle_input("\r");
    list.handle_input("\r");
    let rendered = output(list.as_ref(), 120);
    assert!(rendered.contains("→ ✓ medium"), "{rendered}");
    assert!(rendered.contains("    (clear override)"), "{rendered}");

    list.handle_input("\x1b[B");
    let rendered = output(list.as_ref(), 120);
    assert!(rendered.contains("  ✓ medium"), "{rendered}");
    assert!(rendered.contains("→   high"), "{rendered}");
}

#[test]
fn cancel_dismisses_without_a_setting_change() {
    setup(&[]);
    let (cancellations, on_cancel) = cancellation();
    let selector = component(
        &config(),
        SettingsCallbacks {
            on_cancel,
            ..noop_callbacks()
        },
    );
    let list = selector.get_settings_list();

    list.handle_input("\x1b");
    assert_eq!(cancellations.get(), 1);
}

#[test]
fn cancelling_the_theme_menu_previews_the_original_setting() {
    setup(&[]);
    let previews = Rc::new(RefCell::new(Vec::<String>::new()));
    let (cancellations, on_cancel) = cancellation();
    let selector = component(
        &SettingsConfig {
            current_theme: "light/dark".to_string(),
            available_themes: vec!["dark".to_string(), "light".to_string()],
            ..config()
        },
        SettingsCallbacks {
            on_theme_preview: Some({
                let previews = Rc::clone(&previews);
                Rc::new(move |theme| previews.borrow_mut().push(theme.to_string()))
            }),
            on_cancel,
            ..noop_callbacks()
        },
    );
    let list = selector.get_settings_list();

    list.select_item("theme");
    list.handle_input("\r");
    list.handle_input("\x1b");
    assert_eq!(
        previews.borrow().last().map(String::as_str),
        Some("light/dark")
    );
    assert_eq!(cancellations.get(), 0);

    list.handle_input("\x1b");
    assert_eq!(cancellations.get(), 1);
}

#[test]
fn applying_an_automatic_theme_saves_the_light_dark_pair() {
    setup(&[]);
    let changes = Rc::new(RefCell::new(Vec::<String>::new()));
    let selector = component(
        &SettingsConfig {
            current_theme: "light/dark".to_string(),
            available_themes: vec!["dark".to_string(), "light".to_string(), "other".to_string()],
            ..config()
        },
        SettingsCallbacks {
            on_theme_change: {
                let changes = Rc::clone(&changes);
                Rc::new(move |theme| changes.borrow_mut().push(theme.to_string()))
            },
            ..noop_callbacks()
        },
    );
    let list = selector.get_settings_list();

    list.select_item("theme");
    list.handle_input("\r");
    list.handle_input("\x1b[B");
    list.handle_input("\x1b[B");
    list.handle_input("\r");
    assert_eq!(*changes.borrow(), vec!["light/dark".to_string()]);
}

#[test]
fn toggling_a_warning_reports_the_new_warning_settings() {
    setup(&[]);
    let changes = Rc::new(RefCell::new(Vec::<WarningSettings>::new()));
    let selector = component(
        &config(),
        SettingsCallbacks {
            on_warnings_change: {
                let changes = Rc::clone(&changes);
                Rc::new(move |warnings| changes.borrow_mut().push(warnings))
            },
            ..noop_callbacks()
        },
    );
    let list = selector.get_settings_list();

    list.select_item("warnings");
    list.handle_input("\r");
    list.handle_input("\r");
    assert_eq!(
        *changes.borrow(),
        vec![WarningSettings {
            anthropic_extra_usage: false
        }]
    );
}

#[test]
fn clearing_a_per_model_override_removes_it_and_loops_back() {
    setup(&[]);
    let removals = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let model = thinking_model();
    let model_key = "test-provider/thinking-model".to_string();
    let mut model_thinking_levels = BTreeMap::new();
    model_thinking_levels.insert(model_key.clone(), ThinkingLevel::Medium);
    let selector = component(
        &SettingsConfig {
            default_model: model_key,
            available_default_models: vec![model],
            thinking_level: ThinkingLevel::High,
            model_thinking_levels,
            ..config()
        },
        SettingsCallbacks {
            on_model_thinking_level_remove: {
                let removals = Rc::clone(&removals);
                Rc::new(move |provider, model_id| {
                    removals
                        .borrow_mut()
                        .push((provider.to_string(), model_id.to_string()));
                })
            },
            ..noop_callbacks()
        },
    );
    let list = selector.get_settings_list();

    list.select_item("model-thinking");
    list.handle_input("\r");
    list.handle_input("\r");
    list.handle_input("\x1b[B");
    list.handle_input("\x1b[B");
    list.handle_input("\r");
    assert_eq!(
        *removals.borrow(),
        vec![("test-provider".to_string(), "thinking-model".to_string())]
    );

    list.handle_input("\x1b");
    assert!(output(list.as_ref(), 120).contains("none"));
}

#[test]
fn no_models_available_offers_the_login_hint() {
    setup(&[]);
    let selector = component(&config(), noop_callbacks());
    let list = selector.get_settings_list();

    list.select_item("model-thinking");
    list.handle_input("\r");
    let rendered = output(list.as_ref(), 120);
    assert!(rendered.contains("No models available"), "{rendered}");
}

#[test]
fn cycles_every_scalar_setting_through_its_values() {
    setup(&[]);
    let rows: &[(&str, usize)] = &[
        ("Auto-compact", 2),
        ("Steering mode", 2),
        ("Follow-up mode", 2),
        ("Transport", 4),
        ("HTTP idle timeout", 4),
        ("Hide thinking", 2),
        ("Mermaid diagrams", 3),
        ("Cache miss notices", 2),
        ("Collapse changelog", 2),
        ("Quiet startup", 2),
        ("Install telemetry", 2),
        ("Default project trust", 3),
        ("Double-escape action", 3),
        ("Tree filter mode", 5),
        ("TUI mode", 2),
        ("Fullscreen exit output", 2),
        ("Fullscreen scrollbar", 3),
        ("Fullscreen copy on select", 2),
        ("Editor padding", 4),
        ("Output padding", 2),
        ("Autocomplete max items", 6),
        ("Clear on shrink", 2),
        ("Terminal progress", 2),
        ("Show hardware cursor", 2),
        ("Skill commands", 2),
        ("Block images", 2),
        ("Auto-resize images", 2),
    ];
    for (label, times) in rows {
        let selector = component(&config(), noop_callbacks());
        let list = selector.get_settings_list();
        assert!(output(list.as_ref(), 120).contains(label), "{label}");
        for character in label.chars() {
            list.handle_input(&character.to_string());
        }
        for _ in 0..*times {
            list.handle_input("\r");
        }
    }
}

#[test]
fn theme_mode_switches_between_single_and_automatic() {
    setup(&[]);
    let previews = Rc::new(RefCell::new(Vec::<String>::new()));
    let (cancellations, on_cancel) = cancellation();
    let selector = component(
        &SettingsConfig {
            current_theme: "dark".to_string(),
            available_themes: vec!["dark".to_string(), "light".to_string()],
            ..config()
        },
        SettingsCallbacks {
            on_theme_preview: Some({
                let previews = Rc::clone(&previews);
                Rc::new(move |theme| previews.borrow_mut().push(theme.to_string()))
            }),
            on_cancel,
            ..noop_callbacks()
        },
    );
    let list = selector.get_settings_list();

    list.select_item("theme");
    list.handle_input("\r");
    list.handle_input("\x1b[A");
    list.handle_input("\r");
    assert!(output(list.as_ref(), 120).contains("Automatic Theme"));

    for _ in 0..3 {
        list.handle_input("\x1b[B");
    }
    list.handle_input("\r");
    assert!(output(list.as_ref(), 120).contains("→ ✓ dark"));

    list.handle_input("\x1b");
    assert_eq!(previews.borrow().last().map(String::as_str), Some("dark"));
    assert_eq!(cancellations.get(), 0);
}

#[test]
fn automatic_mode_opens_the_dark_theme_picker() {
    setup(&[]);
    let selector = component(
        &SettingsConfig {
            current_theme: "light/dark".to_string(),
            available_themes: vec!["dark".to_string(), "light".to_string(), "other".to_string()],
            ..config()
        },
        noop_callbacks(),
    );
    let list = selector.get_settings_list();

    list.select_item("theme");
    list.handle_input("\r");
    list.handle_input("\x1b[B");
    list.handle_input("\r");
    assert!(output(list.as_ref(), 120).contains("→ ✓ dark"));

    list.handle_input("\x1b");
    assert!(output(list.as_ref(), 120).contains("Automatic Theme"));
}

#[test]
fn model_step_search_filters_and_navigates_back() {
    setup(&[]);
    let model = thinking_model();
    let model_key = "test-provider/thinking-model".to_string();
    let selector = component(
        &SettingsConfig {
            default_model: model_key,
            available_default_models: vec![model],
            ..config()
        },
        noop_callbacks(),
    );
    let list = selector.get_settings_list();

    list.select_item("model-thinking");
    list.handle_input("\r");
    for character in "thinking".chars() {
        list.handle_input(&character.to_string());
    }
    assert!(output(list.as_ref(), 120).contains("thinking-model"));
    list.handle_input("\r");
    assert!(
        output(list.as_ref(), 120).contains("Thinking Level for thinking-model"),
        "{}",
        output(list.as_ref(), 120)
    );

    list.handle_input("\x1b");
    assert!(
        output(list.as_ref(), 120).contains("Per-Model Thinking Level"),
        "{}",
        output(list.as_ref(), 120)
    );
    list.handle_input("\x1b");
}

#[test]
fn changing_a_per_model_level_reports_the_override() {
    setup(&[]);
    let changes = Rc::new(RefCell::new(Vec::<(String, String, ThinkingLevel)>::new()));
    let model = thinking_model();
    let model_key = "test-provider/thinking-model".to_string();
    let selector = component(
        &SettingsConfig {
            default_model: model_key,
            available_default_models: vec![model],
            thinking_level: ThinkingLevel::High,
            ..config()
        },
        SettingsCallbacks {
            on_model_thinking_level_change: {
                let changes = Rc::clone(&changes);
                Rc::new(move |provider, model_id, level| {
                    changes
                        .borrow_mut()
                        .push((provider.to_string(), model_id.to_string(), level));
                })
            },
            ..noop_callbacks()
        },
    );
    let list = selector.get_settings_list();

    list.select_item("model-thinking");
    list.handle_input("\r");
    list.handle_input("\r");
    for _ in 0..4 {
        list.handle_input("\x1b[B");
    }
    list.handle_input("\r");
    assert_eq!(
        *changes.borrow(),
        vec![(
            "test-provider".to_string(),
            "thinking-model".to_string(),
            ThinkingLevel::High
        )]
    );

    list.handle_input("\x1b");
    assert!(output(list.as_ref(), 120).contains("1 configured"));
}

fn select_item(value: &str) -> SelectItem {
    SelectItem {
        value: value.to_string(),
        label: value.to_string(),
        description: None,
    }
}

#[test]
fn component_trait_surfaces_delegate_and_render() {
    setup(&[]);
    let selector = component(&config(), noop_callbacks());
    let _ = selector.render(80);
    selector.handle_input("\x1b");
    selector.invalidate();
    assert!(!selector.children().is_empty());
    let _ = selector.handle_mouse(&click(0, 20));

    let done: SettingsSubmenuDone = Rc::new(|_, _| {});
    let select = SelectSubmenu::new(
        "Title",
        "Description",
        vec![select_item("a")],
        "a",
        Rc::new(|_| {}),
        Rc::new(|| {}),
        None,
        None,
    );
    let _ = select.render(80);
    select.invalidate();
    assert!(!select.children().is_empty());
    let _ = select.handle_mouse(&click(0, 20));
    select.handle_input("\x1b");

    let step = SteppedSubmenuStep {
        key: "key".to_string(),
        title: SteppedSubmenuText::Static("Title".to_string()),
        description: SteppedSubmenuText::Static("Description".to_string()),
        options: Rc::new(|_| vec![select_item("a")]),
        preselect: None,
        searchable: false,
        layout: None,
    };
    let stepped = SteppedSubmenu::new(
        vec![step],
        Rc::new(|_| {}),
        Rc::new(|| {}),
        SteppedSubmenuOptions::default(),
    );
    let _ = stepped.render(80);
    stepped.invalidate();
    assert!(!stepped.children().is_empty());
    let _ = stepped.handle_mouse(&click(0, 20));
    stepped.handle_input("\r");

    let theme = ThemeSubmenu::new(
        "dark",
        TerminalTheme::Dark,
        vec!["dark".to_string()],
        noop_callbacks(),
        done,
    );
    let _ = theme.render(80);
    theme.invalidate();
    assert!(!theme.children().is_empty());
    let _ = theme.handle_mouse(&click(0, 20));
    theme.handle_input("\x1b");
}
