//! HTTP proxy resolution for provider targets, ported from
//! `packages/ai/src/utils/node-http-proxy.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Resolves the proxy URL a target should route through from the
//! conventional `*_proxy` / `all_proxy` / `no_proxy` environment, honoring
//! scoped `ProviderEnv` overrides before the process environment. SOCKS and
//! PAC proxy URLs are rejected explicitly — the transports that consume the
//! result speak HTTP and HTTPS CONNECT only.
//!
//! The AWS-SDK agent construction this module feeds in TypeScript lands with
//! the Bedrock child; this module owns the tested URL resolution surface.

use crate::types::ProviderEnv;

const DEFAULT_PROXY_PORTS: &[(&str, u16)] = &[
    ("ftp", 21),
    ("gopher", 70),
    ("http", 80),
    ("https", 443),
    ("ws", 80),
    ("wss", 443),
];

/// The failure message for SOCKS and PAC proxy URLs, upstream's
/// `UNSUPPORTED_PROXY_PROTOCOL_MESSAGE`.
pub const UNSUPPORTED_PROXY_PROTOCOL_MESSAGE: &str = "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL.";

/// Why proxy resolution refused to proceed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProxyResolutionError {
    /// The resolved proxy value is not a URL; the message carries the proxy
    /// source and the parse failure, upstream's `Invalid proxy URL ...`
    /// wording.
    InvalidUrl(String),
    /// The proxy URL used a protocol the HTTP transports cannot speak.
    UnsupportedProtocol {
        /// The offending protocol scheme, with its colon.
        got: String,
    },
}

impl std::fmt::Display for ProxyResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUrl(message) => f.write_str(message),
            Self::UnsupportedProtocol { got } => {
                write!(f, "{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {got}")
            }
        }
    }
}

impl std::error::Error for ProxyResolutionError {}

/// Resolve the HTTP or HTTPS proxy URL a target request should route through.
///
/// The value comes from the scoped environment over the process environment.
/// [`None`] means no proxy applies. An unparsable proxy URL or a SOCKS/PAC
/// proxy URL fails with [`ProxyResolutionError`].
///
/// # Errors
/// Returns [`ProxyResolutionError::InvalidUrl`] for a proxy value that is
/// not a URL and [`ProxyResolutionError::UnsupportedProtocol`] for a proxy
/// whose scheme is neither `http` nor `https`.
pub fn resolve_http_proxy_url_for_target(
    target_url: &str,
    env: Option<&ProviderEnv>,
) -> Result<Option<url::Url>, ProxyResolutionError> {
    resolve_http_proxy_url_for_target_with_process_env(target_url, env, |name| {
        std::env::var(name).ok().filter(|value| !value.is_empty())
    })
}

/// The resolution path with the process environment injectable, the seam
/// hermetic tests and the Bedrock child use to read env values from their
/// own source.
///
/// # Errors
/// Returns [`ProxyResolutionError::InvalidUrl`] for a proxy value that does
/// not parse as a URL and [`ProxyResolutionError::UnsupportedProtocol`] for
/// a proxy whose scheme is neither `http` nor `https`.
pub fn resolve_http_proxy_url_for_target_with_process_env(
    target_url: &str,
    env: Option<&ProviderEnv>,
    process_env: impl Fn(&str) -> Option<String>,
) -> Result<Option<url::Url>, ProxyResolutionError> {
    let Some(proxy) = get_proxy_for_url(target_url, env, &process_env) else {
        return Ok(None);
    };

    let proxy_url = url::Url::parse(&proxy).map_err(|error| {
        ProxyResolutionError::InvalidUrl(format!(
            "Invalid proxy URL {}: {error}",
            serde_json::to_string(&proxy).unwrap_or_else(|_| proxy.clone())
        ))
    })?;

    if proxy_url.scheme() != "http" && proxy_url.scheme() != "https" {
        return Err(ProxyResolutionError::UnsupportedProtocol {
            got: format!("{}:", proxy_url.scheme()),
        });
    }
    Ok(Some(proxy_url))
}

fn get_proxy_env(
    key: &str,
    env: Option<&ProviderEnv>,
    process_env: &impl Fn(&str) -> Option<String>,
) -> String {
    let lowercase_key = key.to_ascii_lowercase();
    let uppercase_key = key.to_ascii_uppercase();
    env.and_then(|env| env.get(&lowercase_key))
        .filter(|value| !value.is_empty())
        .or_else(|| {
            env.and_then(|env| env.get(&uppercase_key))
                .filter(|value| !value.is_empty())
        })
        .cloned()
        .or_else(|| process_env(&lowercase_key).filter(|value| !value.is_empty()))
        .or_else(|| process_env(&uppercase_key).filter(|value| !value.is_empty()))
        .unwrap_or_default()
}

fn should_proxy_hostname(
    hostname: &str,
    port: u16,
    env: Option<&ProviderEnv>,
    process_env: &impl Fn(&str) -> Option<String>,
) -> bool {
    let no_proxy = get_proxy_env("no_proxy", env, process_env).to_lowercase();
    if no_proxy.is_empty() {
        return true;
    }
    if no_proxy == "*" {
        return false;
    }

    let normalized_target_host = strip_brackets(&hostname.to_lowercase());

    no_proxy
        .split(|c: char| c == ',' || char::is_whitespace(c))
        .all(|entry| {
            let Some(parsed) = parse_no_proxy_entry(entry) else {
                return true;
            };
            if parsed.port != 0 && parsed.port != port {
                return true;
            }

            let mut domain = strip_brackets(&parsed.host);
            if let Some(stripped) = domain.strip_prefix("*.") {
                domain = stripped.to_owned();
            } else if let Some(stripped) = domain
                .strip_prefix('.')
                .or_else(|| domain.strip_prefix('*'))
            {
                domain = stripped.to_owned();
            }

            if domain.is_empty() {
                return true;
            }
            if normalized_target_host == domain {
                return false;
            }
            if normalized_target_host.ends_with(&format!(".{domain}")) {
                return false;
            }
            true
        })
}

fn get_proxy_for_url(
    target_url: &str,
    env: Option<&ProviderEnv>,
    process_env: &impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let Ok(parsed_url) = url::Url::parse(target_url) else {
        return None;
    };
    if !parsed_url.has_host() {
        return None;
    }

    let protocol = parsed_url.scheme().to_owned();
    let hostname = parsed_url.host_str().unwrap_or_default();
    let hostname = strip_brackets(hostname);
    let port = parsed_url
        .port()
        .or_else(|| {
            DEFAULT_PROXY_PORTS
                .iter()
                .find(|(name, _)| *name == protocol)
                .map(|(_, port)| *port)
        })
        .unwrap_or(0);
    if !should_proxy_hostname(&hostname, port, env, process_env) {
        return None;
    }

    let mut proxy = get_proxy_env(&format!("{protocol}_proxy"), env, process_env);
    if proxy.is_empty() {
        proxy = get_proxy_env("all_proxy", env, process_env);
    }
    if proxy.is_empty() {
        return None;
    }
    if !proxy.contains("://") {
        proxy = format!("{protocol}://{proxy}");
    }
    Some(proxy)
}

fn strip_brackets(host: &str) -> String {
    if host.starts_with('[') && host.ends_with(']') {
        host[1..host.len() - 1].to_owned()
    } else {
        host.to_owned()
    }
}

struct NoProxyEntry {
    host: String,
    port: u16,
}

fn parse_no_proxy_entry(entry: &str) -> Option<NoProxyEntry> {
    let trimmed = entry.trim().to_lowercase();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with('[')
        && let Some(closing_bracket) = trimmed.find(']')
    {
        let host = trimmed[1..closing_bracket].to_owned();
        let rest = &trimmed[closing_bracket + 1..];
        return Some(if let Some(port) = rest.strip_prefix(':') {
            NoProxyEntry {
                host,
                port: port.parse().unwrap_or(0),
            }
        } else {
            NoProxyEntry { host, port: 0 }
        });
    }

    if trimmed.matches(':').count() > 2 {
        return Some(NoProxyEntry {
            host: trimmed,
            port: 0,
        });
    }

    if let Some(colon_index) = trimmed.find(':')
        && trimmed.rfind(':') == Some(colon_index)
    {
        let host = trimmed[..colon_index].to_owned();
        if let Ok(port) = trimmed[colon_index + 1..].parse() {
            return Some(NoProxyEntry { host, port });
        }
    }

    Some(NoProxyEntry {
        host: trimmed,
        port: 0,
    })
}
