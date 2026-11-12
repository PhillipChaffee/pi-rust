//! The three easter eggs: Armin's effect completion, DAXNUTS' scanline
//! reveal, and the text-only Earendil announcement.

use std::sync::Arc;

use pi_coding_agent::modes::interactive::components::armin::{ArminComponent, ArminEffect};
use pi_coding_agent::modes::interactive::components::daxnuts::DaxnutsComponent;
use pi_coding_agent::modes::interactive::components::earendil_announcement::EarendilAnnouncementComponent;
use pi_coding_agent::modes::interactive::theme::theme::{init_theme, theme};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::{Component, RenderRequest};
use pi_tui::utils::visible_width;

fn noop_request() -> RenderRequest {
    Arc::new(|| {})
}

fn run_to_completion(component: &ArminComponent, limit: usize) -> bool {
    for _ in 0..limit {
        if component.tick() {
            return true;
        }
    }
    false
}

fn armin(effect: ArminEffect) -> ArminComponent {
    let component = ArminComponent::with_effect(effect, noop_request());
    component.dispose();
    component
}

#[test]
fn typewriter_finishes_and_renders_the_full_art() {
    init_theme(Some("dark"), false);
    let component = armin(ArminEffect::Typewriter);
    assert!(run_to_completion(&component, 2000));
    let lines = component.render(40);
    assert_eq!(lines.len(), 19);
    assert!(strip_ansi(&lines.join("\n")).contains("ARMIN SAYS HI"));
    assert!(lines[0].contains(theme().get_fg_ansi("accent")));
    for line in &lines {
        assert!(visible_width(line) <= 40);
    }
}

#[test]
fn every_non_rain_effect_terminates() {
    init_theme(Some("dark"), false);
    for effect in [
        ArminEffect::Scanline,
        ArminEffect::Fade,
        ArminEffect::Crt,
        ArminEffect::Glitch,
        ArminEffect::Dissolve,
    ] {
        let component = armin(effect);
        assert!(
            run_to_completion(&component, 2000),
            "effect {effect:?} should finish"
        );
        assert!(
            strip_ansi(&component.render(40).join("\n")).contains("ARMIN SAYS HI"),
            "effect {effect:?} should keep the caption"
        );
    }
}

#[test]
fn scanline_reveals_one_row_per_tick() {
    init_theme(Some("dark"), false);
    let component = armin(ArminEffect::Scanline);
    assert!(!component.tick());
    let first = strip_ansi(&component.render(40)[0].clone());
    assert!(!first.trim().is_empty());
    for _ in 0..17 {
        component.tick();
    }
    assert!(component.tick());
}

#[test]
fn narrow_widths_pad_or_clip_armin_to_the_viewport() {
    init_theme(Some("dark"), false);
    let component = armin(ArminEffect::Scanline);
    run_to_completion(&component, 25);
    for width in [1, 5] {
        assert_eq!(component.render(width).len(), 19);
    }
    for width in [15, 20, 80] {
        for line in component.render(width) {
            assert!(visible_width(&line) <= width);
        }
    }
}

#[test]
fn daxnuts_reveals_the_image_and_the_text_finale() {
    init_theme(Some("dark"), false);
    let component = DaxnutsComponent::new(noop_request());
    component.dispose();
    for _ in 0..30 {
        component.tick();
    }
    let lines = component.render(80);
    assert_eq!(lines.len(), 25);
    let rendered = strip_ansi(&lines.join("\n"));
    assert!(rendered.contains("Free Kimi K2.5 via OpenCode Zen"));
    assert!(rendered.contains("\"Powered by daxnuts\""));
    assert!(rendered.contains("— @thdxr"));
    assert!(rendered.contains("https://mistral.ai/news/mistral-vibe-2-0"));
    for line in &lines {
        assert!(visible_width(line) <= 80);
    }
}

#[test]
fn daxnuts_starts_with_a_scanline_and_hidden_text() {
    init_theme(Some("dark"), false);
    let component = DaxnutsComponent::new(noop_request());
    component.dispose();
    let rendered = strip_ansi(&component.render(80).join("\n"));
    assert!(rendered.contains("▓"));
    assert!(!rendered.contains("Free Kimi K2.5 via OpenCode Zen"));
}

#[test]
fn earendil_text_only_form_keeps_the_announcement() {
    init_theme(Some("dark"), false);
    let component = EarendilAnnouncementComponent::with_image_base64(None);
    let rendered = strip_ansi(&component.render(80).join("\n"));
    assert!(rendered.contains("pi has joined Earendil"));
    assert!(rendered.contains("Read the blog post:"));
    assert!(rendered.contains("https://mariozechner.at/posts/2026-04-08-ive-sold-out/"));
    assert!(rendered.contains("─"));
    assert!(!rendered.contains("clankolas"));
    assert_eq!(component.children().len(), 7);
}

#[test]
fn earendil_image_form_adds_the_image_slot() {
    init_theme(Some("dark"), false);
    let component = EarendilAnnouncementComponent::with_image_base64(Some("aGVsbG8=".to_string()));
    assert_eq!(component.children().len(), 9);
    let rendered = strip_ansi(&component.render(80).join("\n"));
    assert!(rendered.contains("pi has joined Earendil"));
}
