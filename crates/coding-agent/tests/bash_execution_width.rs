//! The `bash-execution-width.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1 (regression
//! #2569: the collapsed preview wraps at render-time width).

use std::sync::Arc;

use pi_coding_agent::modes::interactive::components::bash_execution::BashExecutionComponent;
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_tui::tui::Component;
use pi_tui::utils::visible_width;

fn render_request() -> pi_tui::tui::RenderRequest {
    Arc::new(|| {})
}

#[test]
fn collapsed_preview_lines_respect_render_time_width_not_construction_time_width() {
    init_theme(None, false);
    let narrow_width = 80;
    let component = BashExecutionComponent::new("pwd", render_request(), false);

    let long_line = "x".repeat(150);
    component.append_output(&format!("{long_line}\n{long_line}\n"));
    component.set_complete(Some(0), false, None, None);

    let lines = component.render(narrow_width);
    for (index, line) in lines.iter().enumerate() {
        let width = visible_width(line);
        assert!(
            width <= narrow_width,
            "line {index} visibleWidth={width} > {narrow_width}"
        );
    }
}

#[test]
fn recomputes_lines_when_width_changes_between_renders() {
    init_theme(None, false);
    let component = BashExecutionComponent::new("echo hello", render_request(), false);

    let long_line = "abcdefghij".repeat(20);
    component.append_output(&format!("{long_line}\n"));
    component.set_complete(Some(0), false, None, None);

    let lines_200 = component.render(200);
    for line in &lines_200 {
        assert!(visible_width(line) <= 200);
    }

    let lines_60 = component.render(60);
    for (index, line) in lines_60.iter().enumerate() {
        let width = visible_width(line);
        assert!(width <= 60, "line {index} visibleWidth={width} > 60");
    }
}
