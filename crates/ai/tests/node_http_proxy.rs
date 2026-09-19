//! The HTTP proxy resolution port, from `test/node-http-proxy.test.ts`, on
//! the injected process environment instead of process.env mutation.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::collections::BTreeMap;

use pi_ai::types::ProviderEnv;
use pi_ai::utils::node_http_proxy::{
    UNSUPPORTED_PROXY_PROTOCOL_MESSAGE, resolve_http_proxy_url_for_target_with_process_env,
};

fn resolve(target: &str, env: Option<&ProviderEnv>) -> Result<Option<String>, String> {
    resolve_http_proxy_url_for_target_with_process_env(target, env, |_| None)
        .map(|url| url.map(|url| url.to_string()))
        .map_err(|error| error.to_string())
}

fn resolve_with_process_env(
    target: &str,
    env: Option<&ProviderEnv>,
    process_env: BTreeMap<&str, &str>,
) -> Result<Option<String>, String> {
    resolve_http_proxy_url_for_target_with_process_env(target, env, move |name| {
        process_env.get(name).map(|value| (*value).to_owned())
    })
    .map(|url| url.map(|url| url.to_string()))
    .map_err(|error| error.to_string())
}

#[test]
fn respects_no_proxy_exclusions() {
    let env: ProviderEnv = BTreeMap::from([
        (
            String::from("HTTPS_PROXY"),
            String::from("http://proxy.example:8080"),
        ),
        (
            String::from("NO_PROXY"),
            String::from("bedrock-runtime.us-east-1.amazonaws.com"),
        ),
    ]);

    assert_eq![
        resolve(
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            Some(&env)
        ),
        Ok(None)
    ];
}

#[test]
fn resolves_http_and_https_proxy_urls() {
    let env: ProviderEnv = BTreeMap::from([(
        String::from("HTTPS_PROXY"),
        String::from("http://proxy.example:8080"),
    )]);

    assert_eq![
        resolve(
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            Some(&env)
        ),
        Ok(Some(String::from("http://proxy.example:8080/")))
    ];
}

#[test]
fn prefers_scoped_proxy_env_aliases_before_process_env_aliases() {
    let scoped: ProviderEnv = BTreeMap::from([(
        String::from("HTTPS_PROXY"),
        String::from("http://scoped-proxy.example:8080"),
    )]);
    let process_env = BTreeMap::from([("https_proxy", "http://process-proxy.example:8080")]);

    assert_eq![
        resolve_with_process_env(
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            Some(&scoped),
            process_env
        ),
        Ok(Some(String::from("http://scoped-proxy.example:8080/")))
    ];
}

#[test]
fn rejects_socks_and_pac_proxy_urls_explicitly() {
    let env: ProviderEnv = BTreeMap::from([(
        String::from("HTTPS_PROXY"),
        String::from("socks5://proxy.example:1080"),
    )]);

    let error = resolve(
        "https://bedrock-runtime.us-east-1.amazonaws.com",
        Some(&env),
    )
    .expect_err("SOCKS proxies are rejected");
    assert_eq![
        error,
        UNSUPPORTED_PROXY_PROTOCOL_MESSAGE.to_owned() + " Got socks5:"
    ];
}

#[test]
fn handles_subdomain_wildcards_ipv6_and_ports_in_no_proxy() {
    let env: ProviderEnv = BTreeMap::from([
        (
            String::from("HTTPS_PROXY"),
            String::from("http://proxy.example:8080"),
        ),
        (
            String::from("NO_PROXY"),
            String::from(
                "example.com, .wildcard.org, *.star.net, ::1, [2001:db8::1], 127.0.0.1:8080",
            ),
        ),
    ]);

    assert_eq![resolve("https://example.com", Some(&env)), Ok(None)];
    assert_eq![resolve("https://api.example.com", Some(&env)), Ok(None)];
    assert_eq![resolve("https://wildcard.org", Some(&env)), Ok(None)];
    assert_eq![resolve("https://api.wildcard.org", Some(&env)), Ok(None)];
    assert_eq![resolve("https://star.net", Some(&env)), Ok(None)];
    assert_eq![resolve("https://api.star.net", Some(&env)), Ok(None)];
    assert_eq![
        resolve("https://notexample.com", Some(&env)),
        Ok(Some(String::from("http://proxy.example:8080/")))
    ];

    assert_eq![resolve("https://[::1]:80", Some(&env)), Ok(None)];
    assert_eq![resolve("https://[2001:db8::1]", Some(&env)), Ok(None)];
    assert_eq![resolve("https://127.0.0.1:8080", Some(&env)), Ok(None)];
    assert_eq![
        resolve("https://127.0.0.1:3000", Some(&env)),
        Ok(Some(String::from("http://proxy.example:8080/")))
    ];
}

#[test]
fn an_unparsable_proxy_value_fails_with_the_invalid_url_message() {
    let env: ProviderEnv =
        BTreeMap::from([(String::from("HTTPS_PROXY"), String::from("not a url"))]);
    let error =
        resolve("https://api.example.com", Some(&env)).expect_err("the proxy value is not a URL");
    assert![error.contains("Invalid proxy URL"), "{error}"];
    assert![error.contains("not a url"), "{error}"];
}

#[test]
fn all_proxy_is_the_fallback_when_the_scheme_has_no_own_variable() {
    let env: ProviderEnv = BTreeMap::from([(
        String::from("ALL_PROXY"),
        String::from("http://all-proxy.example:3128"),
    )]);
    assert_eq![
        resolve("https://api.example.com", Some(&env)),
        Ok(Some(String::from("http://all-proxy.example:3128/")))
    ];
}

#[test]
fn websocket_targets_use_their_own_port_table_and_variables() {
    let env: ProviderEnv = BTreeMap::from([(
        String::from("WS_PROXY"),
        String::from("http://ws-proxy.example:8080"),
    )]);
    assert_eq![
        resolve("ws://echo.example", Some(&env)),
        Ok(Some(String::from("http://ws-proxy.example:8080/")))
    ];
    let wss_env: ProviderEnv = BTreeMap::from([(
        String::from("WSS_PROXY"),
        String::from("http://wss-proxy.example:8080"),
    )]);
    assert_eq![
        resolve("wss://echo.example", Some(&wss_env)),
        Ok(Some(String::from("http://wss-proxy.example:8080/")))
    ];
}

#[test]
fn schemeless_proxy_values_gain_the_target_scheme() {
    let env: ProviderEnv = BTreeMap::from([(
        String::from("HTTPS_PROXY"),
        String::from("proxy.example:8080"),
    )]);
    assert_eq![
        resolve("https://api.example.com", Some(&env)),
        Ok(Some(String::from("https://proxy.example:8080/")))
    ];
}

#[test]
fn a_target_without_a_usable_url_needs_no_proxy() {
    // Unparsable targets and hostless URLs return None before any
    // environment read, so these assertions stay hermetic.
    assert_eq![resolve("::::", None), Ok(None)];
    assert_eq![resolve("file:///tmp/socket", None), Ok(None)];
    assert_eq![resolve("mailto:user@example.com", None), Ok(None)];
}

#[test]
fn the_public_entry_resolves_through_the_process_environment() {
    use pi_ai::utils::node_http_proxy::resolve_http_proxy_url_for_target;

    // Targets that can never use a proxy settle without reading the
    // process environment.
    assert_eq![
        resolve_http_proxy_url_for_target("::::", None).map(|url| url.is_none()),
        Ok(true)
    ];
    // A proxyable target still resolves, whatever the ambient environment.
    assert![resolve_http_proxy_url_for_target("https://api.example.com", None).is_ok()];
}

#[test]
fn scoped_lowercase_wins_over_scoped_uppercase() {
    let env: ProviderEnv = BTreeMap::from([
        (
            String::from("https_proxy"),
            String::from("http://lower.example:8080"),
        ),
        (
            String::from("HTTPS_PROXY"),
            String::from("http://upper.example:8080"),
        ),
    ]);
    assert_eq![
        resolve("https://api.example.com", Some(&env)),
        Ok(Some(String::from("http://lower.example:8080/")))
    ];
}

#[test]
fn process_env_lowercase_then_uppercase_fill_the_gap() {
    let process = BTreeMap::from([("HTTP_PROXY", "http://process-upper.example:8080")]);
    assert_eq![
        resolve_with_process_env("http://api.example.com", None, process),
        Ok(Some(String::from("http://process-upper.example:8080/")))
    ];
    // Empty variable values fall through like unset ones.
    let emptyish = BTreeMap::from([("HTTP_PROXY", "")]);
    assert_eq![
        resolve_with_process_env("http://api.example.com", None, emptyish),
        Ok(None)
    ];
}

#[test]
fn a_star_no_proxy_disables_every_proxy() {
    let env: ProviderEnv = BTreeMap::from([
        (
            String::from("HTTPS_PROXY"),
            String::from("http://proxy.example:8080"),
        ),
        (String::from("NO_PROXY"), String::from("*")),
    ]);
    assert_eq![resolve("https://api.example.com", Some(&env)), Ok(None)];
}

#[test]
fn blank_and_dot_only_no_proxy_entries_are_skipped() {
    let env: ProviderEnv = BTreeMap::from([
        (
            String::from("HTTPS_PROXY"),
            String::from("http://proxy.example:8080"),
        ),
        (String::from("NO_PROXY"), String::from(" , ., real.host")),
    ]);
    assert_eq![resolve("https://real.host", Some(&env)), Ok(None)];
    assert_eq![resolve("https://api.real.host", Some(&env)), Ok(None)];
    assert_eq![
        resolve("https://other.host", Some(&env)),
        Ok(Some(String::from("http://proxy.example:8080/")))
    ];
}

#[test]
fn bracketed_hosts_carry_ports_and_bare_ipv6_stays_host_only() {
    let env: ProviderEnv = BTreeMap::from([
        (
            String::from("HTTPS_PROXY"),
            String::from("http://proxy.example:8080"),
        ),
        // A bare IPv6 with more than two colons keeps its whole spelling as
        // the host; a bracketed entry may carry a port.
        (
            String::from("NO_PROXY"),
            String::from("[2001:db8::1]:8443, 2001:db8::1"),
        ),
    ]);
    assert_eq![resolve("https://[2001:db8::1]:8443", Some(&env)), Ok(None)];
    assert_eq![
        resolve("https://[2001:db8::2]", Some(&env)),
        Ok(Some(String::from("http://proxy.example:8080/")))
    ];
    assert_eq![resolve("https://[2001:db8::1]", Some(&env)), Ok(None)];
}
