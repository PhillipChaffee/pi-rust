//! HTTP dispatcher settings, upstream's `src/core/http-dispatcher.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream installs an undici `EnvHttpProxyAgent` as the process-global
//! dispatcher with idle/header timeouts and tunnels; that machinery is
//! Node-specific and has no Rust counterpart — the process-default HTTP
//! client is the reqwest-backed `HttpClient` seam in pi-ai, whose per-call
//! timeout fields carry the timeout knob at consumer sites. What ports is
//! the settings vocabulary: the parse/format pair the settings UI round-trips
//! and the proxy-env application, restated for Rust because edition 2024
//! makes `std::env::set_var` unsafe and this workspace forbids `unsafe` —
//! the caller applies the returned pairs through its own environment seam.

use serde_json::Value;

use crate::config::EnvLookup;

/// The default idle timeout, upstream's `DEFAULT_HTTP_IDLE_TIMEOUT_MS`
/// (Node's undici default would terminate valid long streams; pi raises it).
pub const DEFAULT_HTTP_IDLE_TIMEOUT_MS: i64 = 300_000;

/// One selectable idle-timeout choice, upstream's `HTTP_IDLE_TIMEOUT_CHOICES`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpIdleTimeoutChoice {
    /// The settings-UI label, upstream's `label`.
    pub label: &'static str,
    /// The timeout in milliseconds; `0` disables the idle timeout.
    pub timeout_ms: i64,
}

/// The choices upstream's `HTTP_IDLE_TIMEOUT_CHOICES` table carries.
pub const HTTP_IDLE_TIMEOUT_CHOICES: [HttpIdleTimeoutChoice; 5] = [
    HttpIdleTimeoutChoice {
        label: "30 sec",
        timeout_ms: 30_000,
    },
    HttpIdleTimeoutChoice {
        label: "1 min",
        timeout_ms: 60_000,
    },
    HttpIdleTimeoutChoice {
        label: "2 min",
        timeout_ms: 120_000,
    },
    HttpIdleTimeoutChoice {
        label: "5 min",
        timeout_ms: 300_000,
    },
    HttpIdleTimeoutChoice {
        label: "disabled",
        timeout_ms: 0,
    },
];

/// Parse a settings value into an idle timeout, upstream's
/// `parseHttpIdleTimeoutMs`.
///
/// A `"disabled"` string (case-insensitive) means `0`; an empty or
/// whitespace-only string is no setting; a decimal number string parses like
/// JS `Number` (the hex forms node accepts are not ported); a non-negative
/// finite number floors to milliseconds. Anything else is `None`
/// (upstream's `undefined`).
#[must_use]
pub fn parse_http_idle_timeout_ms(value: &Value) -> Option<i64> {
    match value {
        Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.eq_ignore_ascii_case("disabled") {
                return Some(0);
            }
            if trimmed.is_empty() {
                return None;
            }
            parse_http_idle_timeout_ms(&Value::from(trimmed.parse::<f64>().ok()?))
        }
        Value::Number(number) => {
            let value = number.as_f64()?;
            if !value.is_finite() || value < 0.0 {
                return None;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "floor made the value integral; the cast converts the representation only"
            )]
            let floored = value.floor() as i64;
            Some(floored)
        }
        _ => None,
    }
}

/// Format an idle timeout for display, upstream's `formatHttpIdleTimeoutMs`.
///
/// A table label when the value matches a choice, else seconds with the
/// fractional part kept (`1500` → `"1.5 sec"`), mirroring JS number
/// division's shortest round-trip rendering.
#[must_use]
pub fn format_http_idle_timeout_ms(timeout_ms: i64) -> String {
    if let Some(choice) = HTTP_IDLE_TIMEOUT_CHOICES
        .iter()
        .find(|c| c.timeout_ms == timeout_ms)
    {
        return choice.label.to_string();
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the timeout is a millisecond duration; the mantissa would only lose precision past 2^53 ms"
    )]
    let seconds = timeout_ms as f64 / 1000.0;
    if seconds.fract() == 0.0 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the fract check made the value integral; the cast converts the representation only"
        )]
        let whole = seconds as i64;
        format!("{whole} sec")
    } else {
        format!("{seconds} sec")
    }
}

/// The proxy-env pairs `applyHttpProxySettings` would write, upstream's
/// `process.env.HTTP_PROXY ??= proxy` / `HTTPS_PROXY ??= proxy`.
///
/// A blank proxy applies nothing (upstream's `if (!proxy) return`), and a
/// variable that is set at all — including set to the empty string, which
/// JS nullish assignment keeps — is left untouched. The pairs are ordered
/// `HTTP_PROXY` then `HTTPS_PROXY`, the order the upstream writes run in.
#[must_use]
pub fn http_proxy_env_overrides(
    http_proxy: Option<&str>,
    lookup: &EnvLookup,
) -> Vec<(String, String)> {
    let Some(proxy) = http_proxy.map(str::trim).filter(|proxy| !proxy.is_empty()) else {
        return Vec::new();
    };
    ["HTTP_PROXY", "HTTPS_PROXY"]
        .into_iter()
        .filter(|name| lookup(name).is_none())
        .map(|name| (name.to_string(), proxy.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
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
}
