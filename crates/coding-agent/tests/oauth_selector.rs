//! `oauth-selector.test.ts` at pin `60e7e76`, ported 1:1, plus boundaries
//! (#204). Upstream's `getLoginProviderOptions` case rides the interactive
//! frame; the other five component cases are below.

#[path = "common/selectors.rs"]
mod selectors;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use pi_ai::auth::types::{ApiKeyAuth, ApiKeyResolveFn, AuthCheck, AuthType};
use pi_coding_agent::modes::interactive::components::oauth_selector::{
    AuthSelectorMethod, AuthSelectorProvider, OAuthSelectorComponent, OAuthSelectorMode,
    format_auth_selector_provider_type,
};
use pi_tui::tui::{Component, Focusable};

use selectors::{cancellation, output, setup};

type SelectionLog = Rc<RefCell<Vec<(String, AuthType)>>>;
type SelectionCallback = Rc<dyn Fn(String, AuthType)>;

fn selection_recording() -> (SelectionLog, SelectionCallback) {
    let values = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&values);
    (
        values,
        Rc::new(move |id, auth_type| captured.borrow_mut().push((id, auth_type))),
    )
}

fn status(auth_type: AuthType, source: Option<&str>) -> AuthCheck {
    AuthCheck {
        source: source.map(str::to_string),
        auth_type,
    }
}

fn provider(
    id: &str,
    name: &str,
    auth_type: AuthType,
    status: Option<AuthCheck>,
) -> AuthSelectorProvider {
    AuthSelectorProvider {
        id: id.to_string(),
        name: name.to_string(),
        auth_type,
        method: None,
        status,
    }
}

fn api_key_method(name: &str) -> AuthSelectorMethod {
    let resolve: ApiKeyResolveFn = Arc::new(|_| Box::pin(async { Ok(None) }));
    AuthSelectorMethod::ApiKey(ApiKeyAuth {
        name: name.to_string(),
        login: None,
        check: None,
        resolve,
    })
}

fn selector(
    mode: OAuthSelectorMode,
    providers: Vec<AuthSelectorProvider>,
) -> OAuthSelectorComponent {
    OAuthSelectorComponent::new(mode, providers, Rc::new(|_, _| {}), Rc::new(|| {}), None)
}

#[test]
fn renders_an_option_without_compiled_auth_status_as_unconfigured() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider("google", "Google", AuthType::ApiKey, None)],
    );
    let rendered = output(&selector, 120);
    assert!(rendered.contains("unconfigured"));
    assert!(!rendered.contains("✓ configured"));
}

#[test]
fn shows_oauth_auth_distinctly_in_the_api_key_selector() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider(
            "anthropic",
            "Anthropic",
            AuthType::ApiKey,
            Some(status(AuthType::OAuth, Some("OAuth"))),
        )],
    );
    assert!(output(&selector, 120).contains("subscription configured"));
}

#[test]
fn shows_environment_api_key_auth_as_configured() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider(
            "openai",
            "OpenAI",
            AuthType::ApiKey,
            Some(status(AuthType::ApiKey, Some("OPENAI_API_KEY"))),
        )],
    );
    let rendered = output(&selector, 120);
    assert!(rendered.contains("✓ env: OPENAI_API_KEY"));
    assert!(!rendered.contains("unconfigured"));
}

#[test]
fn shows_models_json_api_key_auth_as_configured() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider(
            "local-proxy",
            "local-proxy",
            AuthType::ApiKey,
            Some(status(AuthType::ApiKey, Some("key in models.json"))),
        )],
    );
    assert!(output(&selector, 120).contains("✓ key in models.json"));
}

#[test]
fn shows_models_json_command_auth_as_configured() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider(
            "op-proxy",
            "op-proxy",
            AuthType::ApiKey,
            Some(status(AuthType::ApiKey, Some("command in models.json"))),
        )],
    );
    assert!(output(&selector, 120).contains("✓ command in models.json"));
}

#[test]
fn titles_follow_the_login_and_logout_modes() {
    setup(&[]);
    assert!(
        output(&selector(OAuthSelectorMode::Login, Vec::new()), 120)
            .contains("Select provider to configure:")
    );
    assert!(
        output(&selector(OAuthSelectorMode::Logout, Vec::new()), 120)
            .contains("Select provider to logout:")
    );
}

#[test]
fn auth_type_labels_appear_only_when_providers_mix_types() {
    setup(&[]);
    let mixed = selector(
        OAuthSelectorMode::Login,
        vec![
            provider("anthropic", "Anthropic", AuthType::OAuth, None),
            provider("openai", "OpenAI", AuthType::ApiKey, None),
        ],
    );
    let rendered = output(&mixed, 120);
    assert!(rendered.contains("[subscription]"));
    assert!(rendered.contains("[API key]"));

    let uniform = selector(
        OAuthSelectorMode::Login,
        vec![provider("openai", "OpenAI", AuthType::ApiKey, None)],
    );
    assert!(!output(&uniform, 120).contains("[API key]"));
}

#[test]
fn configured_sources_render_without_a_source_label() {
    setup(&[]);
    for source in ["OAuth", "stored credential"] {
        let selector = selector(
            OAuthSelectorMode::Login,
            vec![provider(
                "anthropic",
                "Anthropic",
                AuthType::OAuth,
                Some(status(AuthType::OAuth, Some(source))),
            )],
        );
        assert!(output(&selector, 120).contains("✓ configured"));
    }
}

#[test]
fn mismatched_status_names_the_configured_kind() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider(
            "anthropic",
            "Anthropic",
            AuthType::OAuth,
            Some(status(AuthType::ApiKey, Some("ANTHROPIC_API_KEY"))),
        )],
    );
    assert!(output(&selector, 120).contains("API key configured"));
}

#[test]
fn configured_status_without_a_source_renders_plainly() {
    setup(&[]);
    for source in [None, Some("")] {
        let selector = selector(
            OAuthSelectorMode::Login,
            vec![provider(
                "openai",
                "OpenAI",
                AuthType::ApiKey,
                Some(status(AuthType::ApiKey, source)),
            )],
        );
        assert!(output(&selector, 120).contains("✓ configured"));
    }
}

#[test]
fn env_var_lists_render_as_env_sources() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider(
            "openai",
            "OpenAI",
            AuthType::ApiKey,
            Some(status(
                AuthType::ApiKey,
                Some("OPENAI_API_KEY, ANTHROPIC_API_KEY"),
            )),
        )],
    );
    assert!(output(&selector, 120).contains("✓ env: OPENAI_API_KEY, ANTHROPIC_API_KEY"));
}

#[test]
fn auth_type_labels_cover_both_types() {
    assert_eq!(
        format_auth_selector_provider_type(AuthType::OAuth),
        "subscription"
    );
    assert_eq!(
        format_auth_selector_provider_type(AuthType::ApiKey),
        "API key"
    );
}

#[test]
fn empty_states_follow_the_mode_and_filter() {
    setup(&[]);
    assert!(
        output(&selector(OAuthSelectorMode::Login, Vec::new()), 120)
            .contains("No providers available")
    );
    assert!(
        output(&selector(OAuthSelectorMode::Logout, Vec::new()), 120)
            .contains("No providers logged in. Use /login first.")
    );
    let filtered = selector(
        OAuthSelectorMode::Login,
        vec![provider("openai", "OpenAI", AuthType::ApiKey, None)],
    );
    filtered.handle_input("zzz");
    assert!(output(&filtered, 120).contains("No matching providers"));
}

#[test]
fn initial_search_filters_the_list() {
    setup(&[]);
    let (selected, on_select) = selection_recording();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        vec![
            provider("anthropic", "Anthropic", AuthType::OAuth, None),
            provider("openai", "OpenAI", AuthType::ApiKey, None),
        ],
        on_select,
        Rc::new(|| {}),
        Some("open".to_string()),
    );
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![("openai".to_string(), AuthType::ApiKey)]
    );
}

#[test]
fn empty_initial_search_keeps_every_provider() {
    setup(&[]);
    let (selected, on_select) = selection_recording();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        vec![provider("anthropic", "Anthropic", AuthType::OAuth, None)],
        on_select,
        Rc::new(|| {}),
        Some(String::new()),
    );
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![("anthropic".to_string(), AuthType::OAuth)]
    );
}

#[test]
fn navigation_clamps_at_both_ends() {
    setup(&[]);
    let providers = vec![
        provider("a", "A", AuthType::ApiKey, None),
        provider("b", "B", AuthType::ApiKey, None),
        provider("c", "C", AuthType::ApiKey, None),
    ];
    let (selected, on_select) = selection_recording();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        providers.clone(),
        on_select,
        Rc::new(|| {}),
        None,
    );
    selector.handle_input("\x1b[A");
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![("a".to_string(), AuthType::ApiKey)]
    );

    let (selected, on_select) = selection_recording();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        providers,
        on_select,
        Rc::new(|| {}),
        None,
    );
    for _ in 0..10 {
        selector.handle_input("\x1b[B");
    }
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![("c".to_string(), AuthType::ApiKey)]
    );
}

#[test]
fn cancel_does_not_select() {
    setup(&[]);
    let (selected, on_select) = selection_recording();
    let (cancelled, on_cancel) = cancellation();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        vec![provider("a", "A", AuthType::ApiKey, None)],
        on_select,
        on_cancel,
        None,
    );
    selector.handle_input("\x1b");
    assert!(selected.borrow().is_empty());
    assert_eq!(cancelled.get(), 1);
}

#[test]
fn search_matches_method_names_and_auth_types() {
    setup(&[]);
    let (selected, on_select) = selection_recording();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        vec![
            AuthSelectorProvider {
                method: Some(api_key_method("Google Cloud credentials")),
                ..provider("google-vertex", "Google Vertex AI", AuthType::ApiKey, None)
            },
            provider("anthropic", "Anthropic", AuthType::OAuth, None),
        ],
        on_select,
        Rc::new(|| {}),
        None,
    );
    selector.handle_input("Cloud");
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![("google-vertex".to_string(), AuthType::ApiKey)]
    );

    let (selected, on_select) = selection_recording();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        vec![
            AuthSelectorProvider {
                method: Some(api_key_method("Google Cloud credentials")),
                ..provider("google-vertex", "Google Vertex AI", AuthType::ApiKey, None)
            },
            provider("anthropic", "Anthropic", AuthType::OAuth, None),
        ],
        on_select,
        Rc::new(|| {}),
        None,
    );
    selector.handle_input("oauth");
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![("anthropic".to_string(), AuthType::OAuth)]
    );
}

#[test]
fn scrolling_keeps_the_selection_visible() {
    setup(&[]);
    let providers = (0..10)
        .map(|index| {
            provider(
                &format!("p{index}"),
                &format!("Provider {index}"),
                AuthType::ApiKey,
                None,
            )
        })
        .collect();
    let (selected, on_select) = selection_recording();
    let selector = OAuthSelectorComponent::new(
        OAuthSelectorMode::Login,
        providers,
        on_select,
        Rc::new(|| {}),
        None,
    );
    assert!(output(&selector, 120).contains("(1/10)"));
    for _ in 0..8 {
        selector.handle_input("\x1b[B");
    }
    assert!(output(&selector, 120).contains("(9/10)"));
    selector.handle_input("\n");
    assert_eq!(
        *selected.borrow(),
        vec![("p8".to_string(), AuthType::ApiKey)]
    );
}

#[test]
fn small_lists_have_no_scroll_indicator() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![
            provider("a", "A", AuthType::ApiKey, None),
            provider("b", "B", AuthType::ApiKey, None),
        ],
    );
    assert!(!output(&selector, 120).contains("(1/2)"));
}

#[test]
fn focus_forwards_to_the_search_input() {
    setup(&[]);
    let selector = selector(
        OAuthSelectorMode::Login,
        vec![provider("a", "A", AuthType::ApiKey, None)],
    );
    assert!(!selector.is_focused());
    selector.set_focused(true);
    assert!(selector.is_focused());
    selector.set_focused(false);
    assert!(!selector.is_focused());
}
