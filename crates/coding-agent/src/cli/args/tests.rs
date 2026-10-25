//! The `args.test.ts` cases, ported 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the help writer case unwraps its in-memory buffer; an unexpected failure should panic the test"
)]

use super::*;

fn parse(args: &[&str]) -> Args {
    parse_args(args)
}

fn texts(values: &[&str]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn error(message: &str) -> Diagnostic {
    Diagnostic {
        kind: DiagnosticKind::Error,
        message: message.to_string(),
    }
}

#[test]
fn parses_version_flag() {
    assert!(parse(&["--version"]).version);
    assert!(parse(&["-v"]).version);
}

#[test]
fn version_does_not_stop_the_parse() {
    let result = parse(&["--version", "--help", "some message"]);
    assert!(result.version);
    assert!(result.help);
    assert!(result.messages.contains(&"some message".to_string()));
}

#[test]
fn parses_help_flag() {
    assert!(parse(&["--help"]).help);
    assert!(parse(&["-h"]).help);
}

#[test]
fn parses_print_flag() {
    assert!(parse(&["--print"]).print);
    assert!(parse(&["-p"]).print);
}

#[test]
fn print_consumes_a_prompt_dressed_as_yaml_frontmatter() {
    let prompt = "---\ntitle: hello\n---\nSay hi.";
    let result = parse(&["-p", prompt]);
    assert!(result.print);
    assert_eq!(result.messages, texts(&[prompt]));
    assert!(result.unknown_flags.is_empty());
}

#[test]
fn print_does_not_consume_options() {
    let result = parse(&["-p", "--provider", "openai", "Say hi."]);
    assert!(result.print);
    assert_eq!(result.provider.as_deref(), Some("openai"));
    assert_eq!(result.messages, texts(&["Say hi."]));
}

#[test]
fn parses_continue_and_resume() {
    assert!(parse(&["--continue"]).continue_session);
    assert!(parse(&["-c"]).continue_session);
    assert!(parse(&["--resume"]).resume);
    assert!(parse(&["-r"]).resume);
}

#[test]
fn parses_single_value_flags() {
    assert_eq!(
        parse(&["--provider", "openai"]).provider.as_deref(),
        Some("openai")
    );
    assert_eq!(
        parse(&["--model", "gpt-4o"]).model.as_deref(),
        Some("gpt-4o")
    );
    assert_eq!(
        parse(&["--api-key", "sk-test-key"]).api_key.as_deref(),
        Some("sk-test-key")
    );
    assert_eq!(
        parse(&["--system-prompt", "You are a helpful assistant"])
            .system_prompt
            .as_deref(),
        Some("You are a helpful assistant")
    );
    assert_eq!(parse(&["--mode", "json"]).mode, Some(Mode::Json));
    assert_eq!(parse(&["--mode", "rpc"]).mode, Some(Mode::Rpc));
    assert_eq!(parse(&["--mode", "text"]).mode, Some(Mode::Text));
    assert_eq!(
        parse(&["--session", "/path/to/session.jsonl"])
            .session
            .as_deref(),
        Some("/path/to/session.jsonl")
    );
    assert_eq!(
        parse(&["--session-id", "orchestrated-session"])
            .session_id
            .as_deref(),
        Some("orchestrated-session")
    );
    assert_eq!(
        parse(&["--fork", "1234abcd"]).fork.as_deref(),
        Some("1234abcd")
    );
    assert_eq!(
        parse(&["--export", "session.jsonl"]).export.as_deref(),
        Some("session.jsonl")
    );
    assert_eq!(
        parse(&["--thinking", "high"]).thinking,
        Some(ThinkingLevel::High)
    );
    assert_eq!(
        parse(&["--models", "gpt-4o,claude-sonnet,gemini-pro"]).models,
        Some(texts(&["gpt-4o", "claude-sonnet", "gemini-pro"]))
    );
}

#[test]
fn fork_flag_leaves_messages_empty() {
    let result = parse(&["--fork", "1234abcd"]);
    assert_eq!(result.fork.as_deref(), Some("1234abcd"));
    assert!(result.messages.is_empty());
}

#[test]
fn parses_repeated_append_system_prompts() {
    assert_eq!(
        parse(&["--append-system-prompt", "Additional context"]).append_system_prompt,
        Some(texts(&["Additional context"]))
    );
    assert_eq!(
        parse(&[
            "--append-system-prompt",
            "Context A",
            "--append-system-prompt",
            "Context B"
        ])
        .append_system_prompt,
        Some(texts(&["Context A", "Context B"]))
    );
}

#[test]
fn parses_name_and_shorthand() {
    assert_eq!(
        parse(&["--name", "my-session"]).name.as_deref(),
        Some("my-session")
    );
    assert_eq!(
        parse(&["-n", "quick-session"]).name.as_deref(),
        Some("quick-session")
    );
    assert_eq!(parse(&["--name", ""]).name.as_deref(), Some(""));
}

#[test]
fn normalizes_display_names_and_rejects_whitespace_only_values() {
    assert_eq!(
        normalize_session_name("  named session  ").as_deref(),
        Some("named session")
    );
    assert_eq!(normalize_session_name("   "), None);
}

#[test]
fn reports_a_missing_name_value() {
    assert_eq!(
        parse(&["--name"]).diagnostics,
        vec![error("--name requires a value")]
    );
}

#[test]
fn name_works_alongside_other_flags() {
    let result = parse(&[
        "--name",
        "named-run",
        "--print",
        "--model",
        "gpt-4o",
        "hello",
    ]);
    assert_eq!(result.name.as_deref(), Some("named-run"));
    assert!(result.print);
    assert_eq!(result.model.as_deref(), Some("gpt-4o"));
    assert_eq!(result.messages, texts(&["hello"]));
}

#[test]
fn parses_no_session() {
    assert!(parse(&["--no-session"]).no_session);
}

#[test]
fn preserves_custom_session_ids_for_non_persisting_commands() {
    let help = parse(&["--session-id", "ephemeral-id", "--help"]);
    assert_eq!(help.session_id.as_deref(), Some("ephemeral-id"));
    assert!(help.help);

    let list = parse(&["--session-id", "ephemeral-id", "--list-models"]);
    assert_eq!(list.session_id.as_deref(), Some("ephemeral-id"));
    assert_eq!(list.list_models, Some(ListModels::All));

    let ephemeral = parse(&["--session-id", "ephemeral-id", "--no-session"]);
    assert_eq!(ephemeral.session_id.as_deref(), Some("ephemeral-id"));
    assert!(ephemeral.no_session);
}

#[test]
fn parses_extension_flags() {
    assert_eq!(
        parse(&["--extension", "./my-extension.ts"]).extensions,
        Some(texts(&["./my-extension.ts"]))
    );
    assert_eq!(
        parse(&["-e", "./my-extension.ts"]).extensions,
        Some(texts(&["./my-extension.ts"]))
    );
    assert_eq!(
        parse(&["--extension", "./ext1.ts", "-e", "./ext2.ts"]).extensions,
        Some(texts(&["./ext1.ts", "./ext2.ts"]))
    );
}

#[test]
fn parses_no_extensions_with_explicit_paths() {
    let result = parse(&["--no-extensions", "-e", "foo.ts", "-e", "bar.ts"]);
    assert!(result.no_extensions);
    assert_eq!(result.extensions, Some(texts(&["foo.ts", "bar.ts"])));
}

#[test]
fn parses_skill_flags() {
    assert_eq!(
        parse(&["--skill", "./skill-dir"]).skills,
        Some(texts(&["./skill-dir"]))
    );
    assert_eq!(
        parse(&["--skill", "./skill-a", "--skill", "./skill-b"]).skills,
        Some(texts(&["./skill-a", "./skill-b"]))
    );
}

#[test]
fn parses_prompt_template_flags() {
    assert_eq!(
        parse(&["--prompt-template", "./prompts"]).prompt_templates,
        Some(texts(&["./prompts"]))
    );
    assert_eq!(
        parse(&["--prompt-template", "./one", "--prompt-template", "./two"]).prompt_templates,
        Some(texts(&["./one", "./two"]))
    );
}

#[test]
fn parses_theme_flags() {
    assert_eq!(
        parse(&["--theme", "./theme.json"]).themes,
        Some(texts(&["./theme.json"]))
    );
    assert_eq!(
        parse(&["--theme", "./dark.json", "--theme", "./light.json"]).themes,
        Some(texts(&["./dark.json", "./light.json"]))
    );
}

#[test]
fn parses_use_theme() {
    assert_eq!(
        parse(&["--use-theme", "light"]).use_theme.as_deref(),
        Some("light")
    );
}

#[test]
fn reports_a_missing_use_theme_name() {
    let result = parse(&["--use-theme", "--print"]);
    assert_eq!(result.use_theme, None);
    assert!(result.print);
    assert_eq!(
        result.diagnostics,
        vec![error("--use-theme requires a theme name")]
    );
}

#[test]
fn parses_resource_discovery_flags() {
    assert!(parse(&["--no-skills"]).no_skills);
    assert!(parse(&["-ns"]).no_skills);
    assert!(parse(&["--no-prompt-templates"]).no_prompt_templates);
    assert!(parse(&["-np"]).no_prompt_templates);
    assert!(parse(&["--no-themes"]).no_themes);
    assert!(parse(&["--no-context-files"]).no_context_files);
    assert!(parse(&["-nc"]).no_context_files);
}

#[test]
fn parses_project_trust_overrides() {
    assert_eq!(parse(&["--approve"]).project_trust_override, Some(true));
    assert_eq!(parse(&["-a"]).project_trust_override, Some(true));
    assert_eq!(parse(&["--no-approve"]).project_trust_override, Some(false));
    assert_eq!(parse(&["-na"]).project_trust_override, Some(false));
    assert_eq!(parse(&[]).project_trust_override, None);
}

#[test]
fn parses_verbose_and_offline() {
    assert!(parse(&["--verbose"]).verbose);
    assert!(parse(&["--offline"]).offline);
}

#[test]
fn parses_tui_modes() {
    assert_eq!(
        parse(&["--tui-mode", "regular"]).tui_mode,
        Some(TuiMode::Regular)
    );
    assert_eq!(
        parse(&["--tui-mode", "fullscreen"]).tui_mode,
        Some(TuiMode::Fullscreen)
    );
}

#[test]
fn rejects_invalid_tui_modes() {
    let result = parse(&["--tui-mode", "other"]);
    assert_eq!(
        result.diagnostics,
        vec![error(
            "Invalid TUI mode \"other\". Valid values: regular, fullscreen"
        )]
    );
}

#[test]
fn requires_a_tui_mode() {
    assert_eq!(
        parse(&["--tui-mode"]).diagnostics,
        vec![error("--tui-mode requires regular or fullscreen")]
    );
}

#[test]
fn does_not_recognize_the_old_ui_mode_flag() {
    let result = parse(&["--ui-mode", "fullscreen"]);
    assert_eq!(result.tui_mode, None);
    assert_eq!(
        result.unknown_flags.get("ui-mode"),
        Some(&UnknownFlagValue::Value("fullscreen".to_string()))
    );
}

#[test]
fn parses_tool_flags() {
    assert!(parse(&["--no-tools"]).no_tools);
    assert!(parse(&["-nt"]).no_tools);
    assert!(parse(&["--no-builtin-tools"]).no_builtin_tools);
    assert!(parse(&["-nbt"]).no_builtin_tools);
    assert_eq!(
        parse(&["--tools", "read,bash"]).tools,
        Some(texts(&["read", "bash"]))
    );
    assert_eq!(
        parse(&["-t", "read,bash"]).tools,
        Some(texts(&["read", "bash"]))
    );
    assert_eq!(
        parse(&["--exclude-tools", "read,bash"]).exclude_tools,
        Some(texts(&["read", "bash"]))
    );
    assert_eq!(
        parse(&["-xt", "read,bash"]).exclude_tools,
        Some(texts(&["read", "bash"]))
    );
}

#[test]
fn tool_lists_keep_explicit_flags_alongside_the_disablers() {
    let no_tools = parse(&["--no-tools", "--tools", "read,bash"]);
    assert!(no_tools.no_tools);
    assert_eq!(no_tools.tools, Some(texts(&["read", "bash"])));

    let no_builtin = parse(&["--no-builtin-tools", "--tools", "read,bash"]);
    assert!(no_builtin.no_builtin_tools);
    assert_eq!(no_builtin.tools, Some(texts(&["read", "bash"])));
}

#[test]
fn parses_messages_and_file_args() {
    assert_eq!(
        parse(&["hello", "world"]).messages,
        texts(&["hello", "world"])
    );
    assert_eq!(
        parse(&["@README.md", "@src/main.ts"]).file_args,
        texts(&["README.md", "src/main.ts"])
    );
    let mixed = parse(&["@file.txt", "explain this", "@image.png"]);
    assert_eq!(mixed.file_args, texts(&["file.txt", "image.png"]));
    assert_eq!(mixed.messages, texts(&["explain this"]));
}

#[test]
fn captures_unknown_long_flags() {
    let string = parse(&["--unknown-flag", "message"]);
    assert!(string.messages.is_empty());
    assert_eq!(
        string.unknown_flags.get("unknown-flag"),
        Some(&UnknownFlagValue::Value("message".to_string()))
    );

    let boolean = parse(&["--unknown-flag"]);
    assert_eq!(
        boolean.unknown_flags.get("unknown-flag"),
        Some(&UnknownFlagValue::Set)
    );

    let equals = parse(&["--unknown-flag=value"]);
    assert_eq!(
        equals.unknown_flags.get("unknown-flag"),
        Some(&UnknownFlagValue::Value("value".to_string()))
    );
}

#[test]
fn treats_double_dash_positionals_as_messages_and_files() {
    let result = parse(&["--", "--provider", "@prompt.md", "-p"]);
    assert_eq!(result.provider, None);
    assert!(!result.print);
    assert_eq!(result.file_args, texts(&["prompt.md"]));
    assert_eq!(result.messages, texts(&["--provider", "-p"]));
}

#[test]
fn rejects_unknown_short_options() {
    assert_eq!(
        parse(&["-z"]).diagnostics,
        vec![error("Unknown option: -z")]
    );
}

#[test]
fn parses_multiple_flags_together() {
    let result = parse(&[
        "--provider",
        "anthropic",
        "--model",
        "claude-sonnet",
        "--print",
        "--thinking",
        "high",
        "@prompt.md",
        "Do the task",
    ]);
    assert_eq!(result.provider.as_deref(), Some("anthropic"));
    assert_eq!(result.model.as_deref(), Some("claude-sonnet"));
    assert!(result.print);
    assert_eq!(result.thinking, Some(ThinkingLevel::High));
    assert_eq!(result.file_args, texts(&["prompt.md"]));
    assert_eq!(result.messages, texts(&["Do the task"]));
}

#[test]
fn warns_on_an_invalid_thinking_level() {
    let result = parse(&["--thinking", "turbo"]);
    assert_eq!(result.thinking, None);
    assert_eq!(
        result.diagnostics,
        vec![Diagnostic {
            kind: DiagnosticKind::Warning,
            message: "Invalid thinking level \"turbo\". Valid values: off, minimal, low, medium, high, xhigh, max"
                .to_string(),
        }]
    );
}

fn extension_flag(name: &str, flag_type: FlagType, description: Option<&str>) -> ExtensionFlag {
    ExtensionFlag {
        name: name.to_string(),
        description: description.map(ToString::to_string),
        flag_type,
        default: None,
        extension_path: format!("/ext/{name}"),
    }
}

#[test]
fn help_renders_extension_flags_and_placeholders() {
    let flag = extension_flag("plan", FlagType::Boolean, Some("Enable plan mode"));
    let text = help_text(Some(&[flag]));
    assert!(text.contains("Extension CLI Flags:"));
    assert!(text.contains("  --plan"));
    assert!(text.contains("Enable plan mode"));
    assert!(text.contains("Usage:"));
    assert!(text.contains(ENV_AGENT_DIR));
    assert!(!text.contains("{APP_NAME}"));
}

#[test]
fn help_falls_back_to_the_extension_path_when_the_description_is_missing() {
    let flag = extension_flag("marker", FlagType::String, None);
    let text = help_text(Some(&[flag]));
    assert!(text.contains("  --marker <value>"));
    assert!(text.contains("Registered by /ext/marker"));
}

#[test]
fn help_omits_the_extension_section_without_flags() {
    let text = help_text(None);
    assert!(!text.contains("Extension CLI Flags:"));
    assert!(text.contains("Extensions can register additional flags"));
}

#[test]
fn print_help_writes_the_rendered_text() {
    let mut buffer = Vec::new();
    print_help(&mut buffer, None).expect("writes");
    assert_eq!(
        String::from_utf8(buffer).expect("utf-8 help"),
        help_text(None)
    );
}

#[test]
fn parses_list_models_search_and_bare_forms() {
    assert_eq!(
        parse(&["--list-models", "sonnet"]).list_models,
        Some(ListModels::Search("sonnet".to_string()))
    );
    assert_eq!(
        parse(&["--list-models", "--verbose"]).list_models,
        Some(ListModels::All)
    );
    assert_eq!(parse(&["--list-models"]).list_models, Some(ListModels::All));
}

#[test]
fn ignores_an_invalid_mode_value() {
    let result = parse(&["--mode", "bogus", "hello"]);
    assert_eq!(result.mode, None);
    assert_eq!(result.messages, texts(&["hello"]));
    assert_eq!(parse(&["--mode"]).mode, None);
}
