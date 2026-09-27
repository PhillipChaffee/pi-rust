//! The `system-prompt.test.ts` suite ported 1:1, plus the empty-input
//! edge.

use crate::harness::system_prompt::format_skills_for_system_prompt;
use crate::harness::types::Skill;

fn skill(name: &str, description: &str, file_path: &str) -> Skill {
    Skill {
        name: name.to_owned(),
        description: description.to_owned(),
        content: "content".to_owned(),
        file_path: file_path.to_owned(),
        disable_model_invocation: None,
    }
}

fn disabled() -> Skill {
    Skill {
        name: "hidden".to_owned(),
        description: "Hidden".to_owned(),
        content: "hidden content".to_owned(),
        file_path: "/skills/hidden/SKILL.md".to_owned(),
        disable_model_invocation: Some(true),
    }
}

/// Formats visible skills in order and skips model-disabled skills,
/// upstream's "formats visible skills in order and skips model-disabled
/// skills".
#[test]
fn formats_visible_skills_in_order_and_skips_model_disabled_skills() {
    let visible = skill("visible", "Use <this> & that", "/skills/visible/SKILL.md");
    let second = skill("second", "Second skill", "/skills/second/SKILL.md");
    assert_eq!(
        format_skills_for_system_prompt(&[visible, disabled(), second]),
        "The following skills provide specialized instructions for specific tasks.\nRead the full skill file when the task matches its description.\nWhen a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n<available_skills>\n  <skill>\n    <name>visible</name>\n    <description>Use &lt;this&gt; &amp; that</description>\n    <location>/skills/visible/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>second</name>\n    <description>Second skill</description>\n    <location>/skills/second/SKILL.md</location>\n  </skill>\n</available_skills>"
    );
}

/// Returns an empty string when no skills are model-visible, upstream's
/// "returns an empty string when no skills are model-visible" — including
/// the no-skills-at-all input.
#[test]
fn returns_an_empty_string_when_no_skills_are_model_visible() {
    assert_eq!(format_skills_for_system_prompt(&[disabled()]), "");
    assert_eq!(format_skills_for_system_prompt(&[]), "");
}

/// Escapes XML in all model-visible skill fields, upstream's "escapes XML
/// in all model-visible skill fields".
#[test]
fn escapes_xml_in_all_model_visible_skill_fields() {
    let escaping = Skill {
        name: "a&b".to_owned(),
        description: "Quote \"double\" and 'single'".to_owned(),
        content: "content".to_owned(),
        file_path: "/skills/<bad>&\"quote\"/SKILL.md".to_owned(),
        disable_model_invocation: None,
    };
    let formatted = format_skills_for_system_prompt(&[escaping]);
    assert!(formatted.contains(
        "<name>a&amp;b</name>\n    <description>Quote &quot;double&quot; and &apos;single&apos;</description>\n    <location>/skills/&lt;bad&gt;&amp;&quot;quote&quot;/SKILL.md</location>"
    ));
}
