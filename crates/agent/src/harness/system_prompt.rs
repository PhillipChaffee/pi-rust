//! The model-visible skills block of the system prompt, ported from
//! upstream `src/harness/system-prompt.ts`.

use crate::harness::types::Skill;

/// Format the model-visible skills for the system prompt, upstream's
/// `formatSkillsForSystemPrompt`.
///
/// Skills carrying `disableModelInvocation` are skipped; when none remain
/// the block is empty. Every field escapes as XML text so the names,
/// descriptions, and paths read back unambiguously inside the tags.
#[must_use]
pub fn format_skills_for_system_prompt(skills: &[Skill]) -> String {
    let visible_skills: Vec<&Skill> = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation.unwrap_or(false))
        .collect();
    if visible_skills.is_empty() {
        return String::new();
    }

    let mut lines = vec![
        "The following skills provide specialized instructions for specific tasks.".to_owned(),
        "Read the full skill file when the task matches its description.".to_owned(),
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_owned(),
        String::new(),
        "<available_skills>".to_owned(),
    ];

    for skill in &visible_skills {
        lines.push("  <skill>".to_owned());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path)
        ));
        lines.push("  </skill>".to_owned());
    }

    lines.push("</available_skills>".to_owned());
    lines.join("\n")
}

/// Escape the five XML text characters, upstream's `escapeXml`.
fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests;
