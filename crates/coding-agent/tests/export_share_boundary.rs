//! The export and share boundary suite: the ANSI converter's SGR table, the
//! JS-replace substitution patterns, the color helpers, the tool renderer's
//! degradation paths, and the share flow's arms, pinned against upstream at
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use std::sync::Arc;

use serde_json::json;

use pi_tui::tui::Component;

use pi_coding_agent::export_html::{
    CreatedToolHtmlRenderer, ToolHtmlRenderer, ToolHtmlRendererDeps, ToolRenderDefinition,
    ToolRenderResultHook, ansi_lines_to_html, ansi_to_html,
};

// ---------------------------------------------------------------------------
// ansiToHtml: the SGR table.
// ---------------------------------------------------------------------------

#[test]
fn standard_and_bright_colors_map_to_the_palette() {
    assert_eq!(
        ansi_to_html("\u{1b}[31mred\u{1b}[0m"),
        "<span style=\"color:#800000\">red</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[91mbright\u{1b}[0m"),
        "<span style=\"color:#ff0000\">bright</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[44mblue-bg\u{1b}[0m"),
        "<span style=\"background-color:#000080\">blue-bg</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[104mbright-bg\u{1b}[0m"),
        "<span style=\"background-color:#0000ff\">bright-bg</span>"
    );
}

#[test]
fn the_text_styles_and_resets_compose() {
    // Each escape opens a span carrying the cumulative style; the reset
    // clears everything, upstream's per-escape span ladder.
    assert_eq!(
        ansi_to_html("\u{1b}[1mbold\u{1b}[2mdim\u{1b}[3mitalic\u{1b}[4munderline\u{1b}[0mplain"),
        concat!(
            "<span style=\"font-weight:bold\">bold</span>",
            "<span style=\"font-weight:bold;opacity:0.6\">dim</span>",
            "<span style=\"font-weight:bold;opacity:0.6;font-style:italic\">italic</span>",
            "<span style=\"font-weight:bold;opacity:0.6;font-style:italic;text-decoration:underline\">underline</span>",
            "plain"
        )
    );
    // The selective resets drop back to plain text, upstream's span ladder.
    assert_eq!(
        ansi_to_html(
            "\u{1b}[1mbold\u{1b}[22munbold\u{1b}[23munitalic\u{1b}[24mununderline\u{1b}[0m"
        ),
        "<span style=\"font-weight:bold\">bold</span>unboldunitalicununderline"
    );
}

#[test]
fn extended_colors_cover_the_256_palette_and_rgb() {
    assert_eq!(
        ansi_to_html("\u{1b}[38;5;16mcube\u{1b}[0m"),
        "<span style=\"color:#000000\">cube</span>",
        "cube index 16 is the black corner"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[38;5;231mwhite\u{1b}[0m"),
        "<span style=\"color:#ffffff\">white</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[38;5;232mgray0\u{1b}[0m"),
        "<span style=\"color:#080808\">gray0</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[38;5;255mgray23\u{1b}[0m"),
        "<span style=\"color:#eeeeee\">gray23</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[38;2;10;20;30mrgb\u{1b}[0m"),
        "<span style=\"color:rgb(10,20,30)\">rgb</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[48;5;200mbg\u{1b}[0m"),
        "<span style=\"background-color:#ff00d7\">bg</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[48;2;1;2;3mbg-rgb\u{1b}[0m"),
        "<span style=\"background-color:rgb(1,2,3)\">bg-rgb</span>"
    );
}

#[test]
fn default_colors_and_unrecognized_codes_behave() {
    assert_eq!(
        ansi_to_html("\u{1b}[39mtext\u{1b}[0m"),
        "text",
        "a default-fg reset drops the span"
    );
    assert_eq!(ansi_to_html("\u{1b}[49mtext\u{1b}[0m"), "text");
    assert_eq!(
        ansi_to_html("\u{1b}[7mreverse\u{1b}[0m"),
        "reverse",
        "unrecognized SGR codes are ignored"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[38;5mshort\u{1b}[0m"),
        "short",
        "a truncated extended color is ignored"
    );
    // A truncated RGB falls through: the unconsumed `2` is the dim code,
    // upstream's parameter-loop behavior.
    assert_eq!(
        ansi_to_html("\u{1b}[38;2;10;20mshort-rgb\u{1b}[0m"),
        "<span style=\"opacity:0.6\">short-rgb</span>"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[mreset\u{1b}[0m"),
        "reset",
        "the bare SGR resets"
    );
    assert_eq!(
        ansi_to_html("plain<b>text</b>"),
        "plain&lt;b&gt;text&lt;/b&gt;",
        "HTML escapes"
    );
    assert_eq!(ansi_to_html(""), "");
    assert_eq!(
        ansi_to_html("\u{1b}[31"),
        "\u{1b}[31",
        "an unterminated escape rides the text"
    );
}

#[test]
fn ansi_lines_wrap_in_divs_with_nbsp_for_blanks() {
    assert_eq!(
        ansi_lines_to_html(&[String::new(), "x".to_owned()]),
        "<div class=\"ansi-line\">&nbsp;</div><div class=\"ansi-line\">x</div>"
    );
}

// ---------------------------------------------------------------------------
// The color helpers and theme-var generation.
// ---------------------------------------------------------------------------

struct FixedSource;
impl pi_coding_agent::export_html::ThemeColorsSource for FixedSource {
    fn resolved_colors(&self, _theme: Option<&str>) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::from([
            ("userMessageBg".to_owned(), "#f6f6ef".to_owned()),
            ("name".to_owned(), "terminal-fg".to_owned()),
        ])
    }
    fn export_colors(
        &self,
        _theme: Option<&str>,
    ) -> pi_coding_agent::export_html::ThemeExportColors {
        pi_coding_agent::export_html::ThemeExportColors::default()
    }
}

struct BrokenSource;
impl pi_coding_agent::export_html::ThemeColorsSource for BrokenSource {
    fn resolved_colors(&self, _theme: Option<&str>) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::from([("userMessageBg".to_owned(), "not-a-color".to_owned())])
    }
    fn export_colors(
        &self,
        _theme: Option<&str>,
    ) -> pi_coding_agent::export_html::ThemeExportColors {
        pi_coding_agent::export_html::ThemeExportColors::default()
    }
}

struct ExplicitSource;
impl pi_coding_agent::export_html::ThemeColorsSource for ExplicitSource {
    fn resolved_colors(&self, _theme: Option<&str>) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::from([("userMessageBg".to_owned(), "#343541".to_owned())])
    }
    fn export_colors(
        &self,
        _theme: Option<&str>,
    ) -> pi_coding_agent::export_html::ThemeExportColors {
        pi_coding_agent::export_html::ThemeExportColors {
            page_bg: Some("#111111".to_owned()),
            card_bg: None,
            info_bg: Some("#222222".to_owned()),
        }
    }
}

#[test]
fn light_bases_derive_the_light_export_palette() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let data = pi_coding_agent::export_html::SessionData {
        header: session.get_header(),
        entries: session.entries(),
        leaf_id: session.get_leaf_id(),
        system_prompt: None,
        tools: None,
        rendered_tools: None,
    };
    let html =
        pi_coding_agent::export_html::generate_html(&data, None, &FixedSource).expect("html");
    // Light branch: page dims slightly, card keeps the base, info shifts.
    assert!(
        html.contains("--name: terminal-fg;"),
        "resolved colors emit their vars"
    );
    assert!(
        html.contains("--exportPageBg: rgb(236, 236, 229);"),
        "{html}"
    );
    assert!(html.contains("--exportCardBg: #f6f6ef;"));
    assert!(html.contains("--exportInfoBg: rgb(255, 251, 219);"));
}

#[test]
fn unparseable_bases_fall_back_to_upstreams_darkest_defaults() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let data = pi_coding_agent::export_html::SessionData {
        header: session.get_header(),
        entries: session.entries(),
        leaf_id: session.get_leaf_id(),
        system_prompt: None,
        tools: None,
        rendered_tools: None,
    };
    let html =
        pi_coding_agent::export_html::generate_html(&data, None, &BrokenSource).expect("html");
    assert!(html.contains("--exportPageBg: rgb(24, 24, 30);"));
    assert!(html.contains("--exportCardBg: rgb(30, 30, 36);"));
    assert!(html.contains("--exportInfoBg: rgb(60, 55, 40);"));
}

#[test]
fn explicit_export_colors_override_the_derived_ones() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let data = pi_coding_agent::export_html::SessionData {
        header: session.get_header(),
        entries: session.entries(),
        leaf_id: session.get_leaf_id(),
        system_prompt: None,
        tools: None,
        rendered_tools: None,
    };
    let html =
        pi_coding_agent::export_html::generate_html(&data, None, &ExplicitSource).expect("html");
    assert!(html.contains("--exportPageBg: #111111;"));
    assert!(
        html.contains("--exportCardBg: rgb(44, 45, 55);"),
        "an absent slot derives"
    );
    assert!(html.contains("--exportInfoBg: #222222;"));
}

// ---------------------------------------------------------------------------
// The tool renderer's paths.
// ---------------------------------------------------------------------------

struct FixedComponent(Vec<String>);

impl Component for FixedComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.0.clone()
    }
}

/// A renderer over a definition-constructor: the hooks cannot clone, so the
/// lookup rebuilds fresh definitions whose hooks share leaked state.
fn lookup_with(
    build: impl Fn() -> Option<ToolRenderDefinition> + 'static,
) -> CreatedToolHtmlRenderer<'static> {
    let build: &'static (dyn Fn() -> Option<ToolRenderDefinition> + 'static) =
        Box::leak(Box::new(build));
    let lookup: &'static (dyn Fn(&str) -> Option<ToolRenderDefinition> + 'static) =
        &*Box::leak(Box::new(move |_name: &str| build()));
    CreatedToolHtmlRenderer::new(ToolHtmlRendererDeps {
        get_tool_definition: lookup,
        theme: Arc::new(()),
        cwd: "/tmp".to_owned(),
        width: 80,
    })
}

fn no_hooks() -> ToolRenderDefinition {
    ToolRenderDefinition {
        render_call: None,
        render_result: None,
    }
}

#[test]
fn a_missing_tool_or_hook_degrades_to_none() {
    let mut none = lookup_with(|| None);
    assert!(
        none.render_call("c1", "tool", &serde_json::Map::default())
            .is_none()
    );
    assert!(none.render_result("c1", "tool", &[], None, false).is_none());

    let mut bare = lookup_with(|| Some(no_hooks()));
    assert!(
        bare.render_call("c1", "tool", &serde_json::Map::default())
            .is_none()
    );
    assert!(bare.render_result("c1", "tool", &[], None, false).is_none());
}

#[test]
fn a_panicking_hook_degrades_to_none() {
    let hook: ToolRenderResultHook =
        Box::new(|_result, _options, _context| -> Box<dyn Component> {
            panic!("renderer blew up");
        });
    let hook: &'static ToolRenderResultHook = &*Box::leak(Box::new(hook));
    let mut tool_renderer = lookup_with(move || {
        Some(ToolRenderDefinition {
            render_call: None,
            render_result: Some(Box::new(move |result, options, context| {
                hook(result, options, context)
            })),
        })
    });
    assert!(
        tool_renderer
            .render_result("c1", "tool", &[], None, false)
            .is_none(),
        "upstream's catch falls back to the structured rendering"
    );
}

#[test]
fn distinct_collapsed_and_expanded_results_both_ship() {
    let lines: &'static Vec<String> = Box::leak(Box::new(vec!["x".to_owned()]));
    let variant: &'static usize = Box::leak(Box::new(0));
    let hook: ToolRenderResultHook = Box::new(move |_result, options, _context| {
        if options.expanded {
            Box::new(FixedComponent(vec![format!(
                "expanded-{:?} {variant}",
                lines
            )]))
        } else {
            Box::new(FixedComponent(lines.clone()))
        }
    });
    let hook: &'static ToolRenderResultHook = &*Box::leak(Box::new(hook));
    let mut tool_renderer = lookup_with(move || {
        Some(ToolRenderDefinition {
            render_call: None,
            render_result: Some(Box::new(move |result, options, context| {
                hook(result, options, context)
            })),
        })
    });
    let rendered = tool_renderer
        .render_result("c1", "tool", &[], None, false)
        .expect("rendered");
    assert_eq!(
        rendered.expanded.as_deref(),
        Some("<div class=\"ansi-line\">expanded-[&quot;x&quot;] 0</div>")
    );
    assert_eq!(
        rendered.collapsed.as_deref(),
        Some("<div class=\"ansi-line\">x</div>")
    );
}

#[test]
fn the_call_side_renders_and_tracks_args_and_state() {
    let seen: &'static std::sync::Mutex<Vec<String>> =
        Box::leak(Box::new(std::sync::Mutex::new(Vec::new())));
    let hook = Box::new(
        move |context: &pi_coding_agent::export_html::ToolRenderContext<'_>| {
            seen.lock().expect("seen").push(format!(
                "{}{:?}{}",
                context.tool_call_id,
                context.args.map(serde_json::Map::len),
                context.state.len()
            ));
            Box::new(FixedComponent(vec!["\u{1b}[32mcall\u{1b}[0m".to_owned()]))
        },
    );
    let hook: &'static _ = &*Box::leak(Box::new(hook));
    let mut tool_renderer = lookup_with(move || {
        Some(ToolRenderDefinition {
            render_call: Some(Box::new(move |context| hook(context))),
            render_result: None,
        })
    });

    let mut args = serde_json::Map::new();
    args.insert("a".to_owned(), json!(1));
    let html = tool_renderer
        .render_call("call-9", "tool", &args)
        .expect("call html");
    assert_eq!(
        html,
        "<div class=\"ansi-line\"><span style=\"color:#008000\">call</span></div>"
    );
    let first = seen.lock().expect("seen").pop().expect("first");
    assert!(
        first.starts_with("call-9Some(1)0"),
        "the context carries the args: {first}"
    );
}

#[test]
fn blank_edge_lines_trim_and_blank_results_drop_the_collapsed_half() {
    let mut tool_renderer = lookup_with(|| {
        let lines: &'static Vec<String> = Box::leak(Box::new(vec![
            String::new(),
            "  ".to_owned(),
            "\u{1b}[1mtext\u{1b}[0m".to_owned(),
            String::new(),
        ]));
        Some(ToolRenderDefinition {
            render_call: None,
            render_result: Some(Box::new(move |_result, _options, _context| {
                Box::new(FixedComponent(lines.clone()))
            })),
        })
    });
    let rendered = tool_renderer
        .render_result("c1", "tool", &[], None, false)
        .expect("rendered");
    assert_eq!(
        rendered.expanded.as_deref(),
        Some("<div class=\"ansi-line\"><span style=\"font-weight:bold\">text</span></div>")
    );
    assert_eq!(
        rendered.collapsed, None,
        "collapsed == expanded drops the half"
    );
}

#[test]
fn the_renderer_and_deps_debug_shapes() {
    let tool_renderer = lookup_with(|| {
        Some(ToolRenderDefinition {
            render_call: Some(Box::new(|_context| Box::new(FixedComponent(Vec::new())))),
            render_result: None,
        })
    });
    let debug = format!("{tool_renderer:?}");
    assert!(debug.contains("CreatedToolHtmlRenderer"), "{debug}");
    assert!(debug.contains("width: 80"));
    let definition = ToolRenderDefinition {
        render_call: Some(Box::new(|_context| Box::new(FixedComponent(Vec::new())))),
        render_result: None,
    };
    let debug = format!("{definition:?}");
    assert!(
        debug.contains("render_call: true") && debug.contains("render_result: false"),
        "{debug}"
    );
}

// ---------------------------------------------------------------------------
// The share flow's arms.
// ---------------------------------------------------------------------------

use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::session_share::{
    GistOutcome, RadiusUploadOutcome, ShareHttpClient, ShareProcessRunner, ShareSessionSource,
    ShareUserInterface, share_session,
};

struct StubSource {
    session: SessionManager,
    fail_html: bool,
    radius_provider: bool,
    radius_token: Option<String>,
}

impl ShareSessionSource for StubSource {
    fn session_manager(&self) -> &SessionManager {
        &self.session
    }

    fn system_prompt(&self) -> Option<String> {
        None
    }

    fn tools(&self) -> Vec<pi_coding_agent::export_html::ExportedTool> {
        Vec::new()
    }

    fn export_to_html(
        &mut self,
        file_path: &str,
    ) -> pi_agent_core::types::BoxedFuture<'_, Result<(), String>> {
        let fail = self.fail_html;
        let file_path = file_path.to_owned();
        Box::pin(async move {
            if fail {
                Err("html export broke".to_owned())
            } else {
                std::fs::write(file_path, "html").expect("write");
                Ok(())
            }
        })
    }

    fn has_radius_provider(&self) -> bool {
        self.radius_provider
    }

    fn radius_token(&mut self) -> pi_agent_core::types::BoxedFuture<'_, Option<String>> {
        let token = self.radius_token.clone();
        Box::pin(async move { token })
    }
}

#[derive(Default)]
struct CapturingUi {
    statuses: Vec<String>,
    errors: Vec<String>,
}

impl ShareUserInterface for CapturingUi {
    fn show_status(&mut self, message: &str) {
        self.statuses.push(message.to_owned());
    }

    fn show_error(&mut self, message: &str) {
        self.errors.push(message.to_owned());
    }
}

struct StubRunner {
    auth_status: Option<i32>,
    gist: Option<GistOutcome>,
}

impl ShareProcessRunner for StubRunner {
    fn auth_status(&mut self) -> Option<i32> {
        self.auth_status
    }

    fn create_gist(
        &mut self,
        _file_path: &str,
    ) -> pi_agent_core::types::BoxedFuture<'_, GistOutcome> {
        let gist = self.gist.clone().unwrap_or_default();
        Box::pin(async move { gist })
    }
}

struct StubHttp {
    outcome: RadiusUploadOutcome,
}

impl ShareHttpClient for StubHttp {
    fn upload_artifact(
        &mut self,
        _token: &str,
        _body: &[u8],
    ) -> pi_agent_core::types::BoxedFuture<'_, RadiusUploadOutcome> {
        let outcome = self.outcome.clone();
        Box::pin(async move { outcome })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a_failing_html_export_reports_upstreams_message() {
    let mut source = StubSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        fail_html: true,
        radius_provider: false,
        radius_token: None,
    };
    source
        .session
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("x".to_owned()),
                timestamp: 0,
            }),
        ))
        .expect("append");
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: Some(0),
        gist: None,
    };
    let mut http = StubHttp {
        outcome: RadiusUploadOutcome::Failed("unused".to_owned()),
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share settles");
    assert_eq!(
        ui.errors,
        vec!["Failed to export session: html export broke"],
        "the html leg's failure wraps upstream's message"
    );
    assert!(ui.statuses.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn the_radius_leg_reports_urls_failures_and_abortions() {
    // With a provider and token, the Radius leg short-circuits the gist path.
    let mut source = StubSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        fail_html: false,
        radius_provider: true,
        radius_token: Some("token".to_owned()),
    };
    source
        .session
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("x".to_owned()),
                timestamp: 0,
            }),
        ))
        .expect("append");
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: None,
        gist: None,
    };
    let mut http = StubHttp {
        outcome: RadiusUploadOutcome::Url("https://radius.pi.dev/a/1".to_owned()),
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(
        ui.statuses,
        vec![
            "Share URL: \u{1b}]8;;https://radius.pi.dev/a/1\u{1b}\\https://radius.pi.dev/a/1\u{1b}]8;;\u{1b}\\"
        ],
        "the OSC 8 hyperlink wraps the URL"
    );
    assert!(ui.errors.is_empty());
    assert!(runner.gist.is_none(), "no gist when the Radius leg lands");

    // A failed upload reports the composed message.
    let mut source = StubSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        fail_html: false,
        radius_provider: true,
        radius_token: Some("token".to_owned()),
    };
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: None,
        gist: None,
    };
    let mut http = StubHttp {
        outcome: RadiusUploadOutcome::Failed("404".to_owned()),
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(ui.errors, vec!["Failed to upload Radius artifact: 404"]);

    // An aborted upload settles silently.
    let mut source = StubSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        fail_html: false,
        radius_provider: true,
        radius_token: Some("token".to_owned()),
    };
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: None,
        gist: None,
    };
    let mut http = StubHttp {
        outcome: RadiusUploadOutcome::Aborted,
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert!(
        ui.errors.is_empty() && ui.statuses.is_empty(),
        "the abort settles silently"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn the_gist_leg_reports_the_viewer_url_and_failure_messages() {
    let mut source = StubSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        fail_html: false,
        radius_provider: false,
        radius_token: None,
    };
    source
        .session
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("x".to_owned()),
                timestamp: 0,
            }),
        ))
        .expect("append");
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: Some(0),
        gist: Some(GistOutcome {
            stdout: "https://gist.github.com/test/abc123\n".to_owned(),
            stderr: String::new(),
            code: Some(0),
        }),
    };
    let mut http = StubHttp {
        outcome: RadiusUploadOutcome::Failed("unused".to_owned()),
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(ui.statuses.len(), 1);
    let status = &ui.statuses[0];
    assert!(status.contains("Share URL: "), "{status}");
    assert!(
        status.contains("pi.dev/session/#abc123"),
        "the share viewer url: {status}"
    );
    assert!(status.contains("Gist: "), "{status}");

    // A failing gist reports stderr.
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: Some(0),
        gist: Some(GistOutcome {
            stdout: String::new(),
            stderr: "gh exploded\n".to_owned(),
            code: Some(1),
        }),
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(ui.errors, vec!["Failed to create gist: gh exploded"]);

    // An empty stdout reports the parse failure.
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: Some(0),
        gist: Some(GistOutcome {
            stdout: String::new(),
            stderr: String::new(),
            code: Some(0),
        }),
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(ui.errors, vec!["Failed to parse gist ID from gh output"]);
}

/// The probe key the child-process run keys its scenario on.
const SHARE_PROBE: &str = "PI_CODING_AGENT_SHARE_PROBE";

/// Run this suite's own binary as a child whose `TMPDIR` is a private
/// fixture directory: the sibling share flows share the process temp dir and
/// would race the directory count, so the cleanup check isolates itself.
#[test]
fn the_temp_directory_cleans_up_after_every_flow() {
    let dir = tempfile::tempdir().expect("temp dir");
    let output = std::process::Command::new(std::env::current_exe().expect("the test binary path"))
        .args([
            "--exact",
            "the_share_probe",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(SHARE_PROBE, "cleanup")
        .env("TMPDIR", dir.path().display().to_string())
        .output()
        .expect("the probe child runs");
    assert!(
        output.status.success(),
        "the cleanup probe child passes: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The probe child: the share flow runs alone in the composed TMPDIR.
#[tokio::test(flavor = "current_thread")]
async fn the_share_probe() {
    let Ok(_mode) = std::env::var(SHARE_PROBE) else {
        return;
    };
    let mut source = StubSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        fail_html: false,
        radius_provider: false,
        radius_token: None,
    };
    let mut ui = CapturingUi::default();
    let mut runner = StubRunner {
        auth_status: Some(0),
        gist: Some(GistOutcome {
            stdout: "https://gist.github.com/test/x\n".to_owned(),
            stderr: String::new(),
            code: Some(0),
        }),
    };
    let mut http = StubHttp {
        outcome: RadiusUploadOutcome::Failed("unused".to_owned()),
    };
    share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");

    let leftovers: Vec<_> = std::fs::read_dir(std::env::temp_dir())
        .expect("tmp")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("pi-share-"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "the temp directory removes after the flow: {leftovers:?}"
    );
}

#[test]
fn the_share_viewer_url_default_carries_the_gist_fragment() {
    assert_eq!(
        pi_coding_agent::config::get_share_viewer_url("abc"),
        "https://pi.dev/session/#abc"
    );
}
