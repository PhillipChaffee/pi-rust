use super::*;
use serde_json::json;

fn lookup_with(entries: &[(&str, &str)]) -> EnvLookup {
    let owned: Vec<(String, String)> = entries
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    Box::new(move |name: &str| {
        owned
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
    })
}

#[test]
fn parses_choices_and_numbers() {
    assert_eq!(parse_http_idle_timeout_ms(&json!("30 sec")), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!("disabled")), Some(0));
    assert_eq!(parse_http_idle_timeout_ms(&json!("Disabled")), Some(0));
    assert_eq!(parse_http_idle_timeout_ms(&json!("  ")), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!("120000")), Some(120_000));
    assert_eq!(parse_http_idle_timeout_ms(&json!("30.9")), Some(30));
    assert_eq!(parse_http_idle_timeout_ms(&json!(0.0)), Some(0));
    assert_eq!(parse_http_idle_timeout_ms(&json!(-1.0)), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!(true)), None);
}

#[test]
fn formats_choices_and_fallbacks() {
    assert_eq!(format_http_idle_timeout_ms(30_000), "30 sec");
    assert_eq!(format_http_idle_timeout_ms(300_000), "5 min");
    assert_eq!(format_http_idle_timeout_ms(0), "disabled");
    assert_eq!(format_http_idle_timeout_ms(45_000), "45 sec");
    assert_eq!(format_http_idle_timeout_ms(1500), "1.5 sec");
}

#[test]
fn applies_proxy_overrides_only_when_unset() {
    let lookup = lookup_with(&[("HTTPS_PROXY", "existing")]);
    assert_eq!(
        http_proxy_env_overrides(Some(" http://proxy:8080 "), &lookup),
        vec![("HTTP_PROXY".to_string(), "http://proxy:8080".to_string())]
    );
    let empty_lookup = lookup_with(&[]);
    assert_eq!(
        http_proxy_env_overrides(Some("http://proxy:8080"), &empty_lookup),
        vec![
            ("HTTP_PROXY".to_string(), "http://proxy:8080".to_string()),
            ("HTTPS_PROXY".to_string(), "http://proxy:8080".to_string()),
        ]
    );
    // A proxy set to the empty string is still set; JS `??=` keeps it.
    let blank_proxy_lookup = lookup_with(&[("HTTP_PROXY", "")]);
    assert_eq!(
        http_proxy_env_overrides(Some("http://proxy:8080"), &blank_proxy_lookup),
        vec![("HTTPS_PROXY".to_string(), "http://proxy:8080".to_string())]
    );
    assert!(http_proxy_env_overrides(Some("   "), &empty_lookup).is_empty());
    assert!(http_proxy_env_overrides(None, &empty_lookup).is_empty());
}
