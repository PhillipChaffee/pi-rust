//! The session's public types, upstream's `src/core/agent-session.ts` type
//! block.

mod config;
mod events;

pub use config::{
    AgentSessionConfig, BashExecutionOptions, CycleDirection, ExtensionBindings, ModelCycleResult,
    ModelMutationOptions, NavigateTreeOptions, NavigateTreeOutcome, PromptOptions,
    SessionStartSeed, SessionStats, SessionTokens,
};
pub use events::{AgentSessionEvent, AgentSessionEventListener, SummarizationSource};

/// The parsed skill block of a user message, upstream's `ParsedSkillBlock`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedSkillBlock {
    /// The skill name from the block's `name` attribute.
    pub name: String,
    /// The skill file from the block's `location` attribute.
    pub location: String,
    /// The skill body between the tags.
    pub content: String,
    /// The user message following the block, when any.
    pub user_message: Option<String>,
}

/// Parse a skill block from message text, upstream's `parseSkillBlock`.
///
/// The grammar is one anchored match: `<skill name="…" location="…">` on
/// its own line, the body, the closing tag, and an optional
/// blank-line-separated user message.
#[must_use]
#[expect(
    clippy::expect_used,
    reason = "the fixed skill-block pattern is a compile-time constant"
)]
pub fn parse_skill_block(text: &str) -> Option<ParsedSkillBlock> {
    static PATTERN: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r#"(?s)^<skill name="([^"]+)" location="([^"]+)">\n(.*?)\n</skill>(?:\n\n(.*))?$"#,
        )
        .expect("the skill-block grammar is a fixed valid pattern")
    });
    let captures = PATTERN.captures(text)?;
    let user_message = captures
        .get(4)
        .map(|capture| capture.as_str().trim())
        .filter(|message| !message.is_empty())
        .map(str::to_owned);
    Some(ParsedSkillBlock {
        name: captures.get(1)?.as_str().to_owned(),
        location: captures.get(2)?.as_str().to_owned(),
        content: captures.get(3)?.as_str().to_owned(),
        user_message,
    })
}
