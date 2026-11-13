//! The skill-invocation row: collapsed single line, expanded Markdown body,
//! and the invalidate rebuild.

use pi_coding_agent::core::agent_session::ParsedSkillBlock;
use pi_coding_agent::modes::interactive::components::skill_invocation_message::SkillInvocationMessageComponent;
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::Component;

fn skill_block() -> ParsedSkillBlock {
    ParsedSkillBlock {
        name: "demo-skill".to_string(),
        location: "/skills/demo.md".to_string(),
        content: "Skill body text".to_string(),
        user_message: None,
    }
}

fn text(component: &SkillInvocationMessageComponent) -> String {
    strip_ansi(&component.render(80).join("\n"))
}

#[test]
fn collapsed_skill_names_the_skill_and_hints_expand() {
    init_theme(Some("dark"), false);
    let component = SkillInvocationMessageComponent::new(skill_block(), None);
    let rendered = text(&component);
    assert!(rendered.contains("[skill]"));
    assert!(rendered.contains("demo-skill"));
    assert!(rendered.contains("to expand)"));
    assert!(!rendered.contains("Skill body text"));
    assert!(!component.is_expanded());
}

#[test]
fn expanded_skill_shows_the_body() {
    init_theme(Some("dark"), false);
    let component = SkillInvocationMessageComponent::new(skill_block(), None);
    component.set_expanded(true);
    let rendered = text(&component);
    assert!(component.is_expanded());
    assert!(rendered.contains("demo-skill"));
    assert!(rendered.contains("Skill body text"));
}

#[test]
fn invalidate_keeps_the_expanded_body() {
    init_theme(Some("dark"), false);
    let component = SkillInvocationMessageComponent::new(skill_block(), None);
    component.set_expanded(true);
    Component::invalidate(&component);
    assert!(text(&component).contains("Skill body text"));
}
