use super::*;

#[test]
fn builtin_table_matches_upstream_entries() {
    assert_eq!(BUILTIN_SLASH_COMMANDS.len(), 23);
    let model = BUILTIN_SLASH_COMMANDS
        .iter()
        .find(|command| command.name == "model");
    assert_eq!(
        model.map(|command| command.argument_hint),
        Some(Some("<provider/model>"))
    );
    let quit = BUILTIN_SLASH_COMMANDS
        .iter()
        .find(|command| command.name == "quit");
    assert_eq!(quit.map(|command| command.description), Some("Quit pi"));
    assert_eq!(quit_command_description(), format!("Quit {APP_NAME}"));
}

#[test]
fn source_spellings_match_upstream() {
    assert_eq!(SlashCommandSource::Extension.as_str(), "extension");
    assert_eq!(SlashCommandSource::Prompt.as_str(), "prompt");
    assert_eq!(SlashCommandSource::Skill.as_str(), "skill");
}
