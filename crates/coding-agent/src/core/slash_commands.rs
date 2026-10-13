//! The shared slash-command vocabulary, upstream's `src/core/slash-commands.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The command entries the interactive selector lists: the extension, prompt,
//! and skill sources carry provenance; the built-in table names the commands
//! the app itself handles.

use crate::config::APP_NAME;
use crate::source_info::SourceInfo;

/// Where a slash command came from, upstream's `SlashCommandSource`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlashCommandSource {
    /// An extension-registered command.
    Extension,
    /// A file-based prompt template.
    Prompt,
    /// A skill invocation, `/skill:<name>`.
    Skill,
}

impl SlashCommandSource {
    /// The wire spelling, upstream's literal union values.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Extension => "extension",
            Self::Prompt => "prompt",
            Self::Skill => "skill",
        }
    }
}

/// One slash command offered to the selector, upstream's `SlashCommandInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlashCommandInfo {
    /// The invocation name, without the leading slash.
    pub name: String,
    /// The description shown in the selector; absent for some sources.
    pub description: Option<String>,
    /// Where the command came from.
    pub source: SlashCommandSource,
    /// The command's provenance.
    pub source_info: SourceInfo,
}

/// One built-in command the app handles itself, upstream's
/// `BuiltinSlashCommand`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltinSlashCommand {
    /// The invocation name, without the leading slash.
    pub name: &'static str,
    /// The description shown in the selector.
    pub description: &'static str,
    /// The argument hint shown in the selector, upstream's `argumentHint`.
    pub argument_hint: Option<&'static str>,
}

/// The built-in command table, upstream's `BUILTIN_SLASH_COMMANDS`.
pub const BUILTIN_SLASH_COMMANDS: &[BuiltinSlashCommand] = &[
    BuiltinSlashCommand {
        name: "settings",
        description: "Open settings menu",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "model",
        description: "Select model (opens selector UI)",
        argument_hint: Some("<provider/model>"),
    },
    BuiltinSlashCommand {
        name: "tree",
        description: "Navigate session tree (switch branches)",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "thinking",
        description: "Set thinking level",
        argument_hint: Some("<level>"),
    },
    BuiltinSlashCommand {
        name: "scoped-models",
        description: "Enable/disable models for Ctrl+P cycling",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "export",
        description: "Export session (HTML default, or specify path: .html/.jsonl)",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "import",
        description: "Import and resume a session from a JSONL file",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "share",
        description: "Share session as a secret GitHub gist",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "copy",
        description: "Copy last agent message to clipboard",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "name",
        description: "Set session display name",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "session",
        description: "Show session info and stats",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "changelog",
        description: "Show changelog entries",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "hotkeys",
        description: "Show all keyboard shortcuts",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "fork",
        description: "Create a new fork from a previous user message",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "clone",
        description: "Duplicate the current session at the current position",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "trust",
        description: "Save project trust decision for future sessions",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "login",
        description: "Configure provider authentication",
        argument_hint: Some("<provider>"),
    },
    BuiltinSlashCommand {
        name: "logout",
        description: "Remove provider authentication",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "new",
        description: "Start a new session",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "compact",
        description: "Manually compact the session context",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "resume",
        description: "Resume a different session",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "reload",
        description: "Reload keybindings, extensions, skills, prompts, themes, and context files",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "quit",
        description: "Quit pi",
        argument_hint: None,
    },
];

/// The quit command's description interpolates `APP_NAME`, upstream's
/// template literal.
///
/// The const table can't spell that, so the entry above pins the bare name
/// and this accessor returns the interpolating form.
#[must_use]
pub fn quit_command_description() -> String {
    format!("Quit {APP_NAME}")
}

#[cfg(test)]
mod tests {
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
}
