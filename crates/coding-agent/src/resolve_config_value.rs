//! Resolve configuration values that may be shell commands, environment
//! variables, or literals — upstream's `src/core/resolve-config-value.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Consumed by auth-storage and model-registry.
//!
//! Porting restatements this module records:
//!
//! - `process.env` reads are the process default of an [`EnvLookup`] seam:
//!   the workspace forbids mutating the process environment, so tests inject
//!   a map-backed lookup through the `_with` variants and the plain functions
//!   read the real environment. The credential's private env map rides the
//!   existing `env` parameter.
//! - The command branch is the POSIX half only: upstream's win32 branch
//!   (the configured-shell `spawnSync` with its stdin transport and the
//!   default-shell fallback) rides the map's Windows exclusion, so
//!   `executeWithDefaultShell` — `execSync` over `/bin/sh -c` with a ten
//!   second deadline — is the whole executor. The stdin-transport test is
//!   win32-gated upstream and documents as not portable.
//! - `execSync`'s timeout kill restates as a poll loop with the deadline;
//!   a killed or failed command resolves `undefined` exactly where upstream's
//!   `catch` swallows the throw. stdout drains on a worker thread so a full
//!   pipe cannot deadlock the poll.
//! - The `commandResultCache` module global restates as a process-wide shared
//!   map behind a mutex with the same exported [`clear_config_value_cache`];
//!   the cache keys on the full `!…` config text and never on environment
//!   values, whose freshness the template branch preserves.

use std::collections::BTreeMap;
use std::process::{Command, Stdio};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

pub use crate::config::{EnvLookup, default_env_lookup};

/// The command-result cache, upstream's module-global `commandResultCache`:
/// results persist for the process lifetime until explicitly cleared.
static COMMAND_RESULT_CACHE: LazyLock<Mutex<BTreeMap<String, Option<String>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// The command deadline, upstream's `timeout: 10000`.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(10);

/// One parsed template segment, upstream's `TemplatePart`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TemplatePart {
    Literal(String),
    Env { name: String },
}

/// One parsed config reference, upstream's `ConfigValueReference`.
enum ConfigValueReference {
    Command { config: String },
    Template { parts: Vec<TemplatePart> },
}

/// Upstream's `ENV_VAR_NAME_RE`: `[A-Za-z_][A-Za-z0-9_]*`, full match.
fn is_env_var_name(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

/// Upstream's `ENV_VAR_NAME_PREFIX_RE` match at `value`: the leading run of
/// `[A-Za-z_][A-Za-z0-9_]*`.
fn env_var_name_prefix(value: &str) -> Option<&str> {
    let mut end = 0;
    for (index, ch) in value.char_indices() {
        let ok = if index == 0 {
            ch.is_ascii_alphabetic() || ch == '_'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if !ok {
            break;
        }
        end = index + ch.len_utf8();
    }
    if end == 0 { None } else { Some(&value[..end]) }
}

/// Merge a literal onto the parts, upstream's `appendLiteral`: adjacent
/// literals concatenate, empty values are skipped.
fn append_literal(parts: &mut Vec<TemplatePart>, value: &str) {
    if value.is_empty() {
        return;
    }
    if let Some(TemplatePart::Literal(previous)) = parts.last_mut() {
        previous.push_str(value);
        return;
    }
    parts.push(TemplatePart::Literal(value.to_string()));
}

/// Upstream's `parseConfigValueTemplate`: `$$`/`$!` escape to literals,
/// `${NAME}` and `$NAME` become env segments when the name matches the env
/// grammar, an unclosed `${` leaves the `$` literal for the next scan, and a
/// bare `$` stays literal.
fn parse_config_value_template(config: &str) -> Vec<TemplatePart> {
    let mut parts: Vec<TemplatePart> = Vec::new();
    let mut index = 0;

    while index < config.len() {
        let Some(dollar) = config[index..].find('$') else {
            append_literal(&mut parts, &config[index..]);
            break;
        };
        let dollar = index + dollar;

        append_literal(&mut parts, &config[index..dollar]);
        let next = config[dollar + 1..].chars().next();

        match next {
            Some('$' | '!') => {
                append_literal(&mut parts, &config[dollar + 1..dollar + 2]);
                index = dollar + 2;
            }
            Some('{') => {
                let close = config[dollar + 2..]
                    .find('}')
                    .map(|close| dollar + 2 + close);
                let Some(close) = close else {
                    append_literal(&mut parts, "$");
                    index = dollar + 1;
                    continue;
                };
                let name = &config[dollar + 2..close];
                if is_env_var_name(name) {
                    parts.push(TemplatePart::Env {
                        name: name.to_string(),
                    });
                } else {
                    append_literal(&mut parts, &config[dollar..=close]);
                }
                index = close + 1;
            }
            _ => {
                let name = env_var_name_prefix(&config[dollar + 1..]);
                if let Some(name) = name {
                    parts.push(TemplatePart::Env {
                        name: name.to_string(),
                    });
                    index = dollar + 1 + name.len();
                } else {
                    append_literal(&mut parts, "$");
                    index = dollar + 1;
                }
            }
        }
    }

    parts
}

/// Upstream's `parseConfigValueReference`: a leading `!` makes the whole
/// value a command; everything else is a template.
fn parse_config_value_reference(config: &str) -> ConfigValueReference {
    if config.starts_with('!') {
        ConfigValueReference::Command {
            config: config.to_string(),
        }
    } else {
        ConfigValueReference::Template {
            parts: parse_config_value_template(config),
        }
    }
}

/// Upstream's `resolveEnvConfigValue`: the credential's private map first,
/// then `process.env`. A value that is present but empty falls through — the
/// `||` chain treats `""` as missing — so an unset variable and a blank one
/// resolve the same way.
fn resolve_env_config_value(
    name: &str,
    env: Option<&BTreeMap<String, String>>,
    process_env: &EnvLookup,
) -> Option<String> {
    if let Some(env) = env
        && let Some(value) = env.get(name)
        && !value.is_empty()
    {
        return Some(value.clone());
    }
    let value = process_env(name)?;
    if value.is_empty() { None } else { Some(value) }
}

/// The distinct env variable names a template reads, upstream's
/// `getTemplateEnvVarNames`.
fn template_env_var_names(parts: &[TemplatePart]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for part in parts {
        let TemplatePart::Env { name } = part else {
            continue;
        };
        if !names.iter().any(|seen| seen == name) {
            names.push(name.clone());
        }
    }
    names
}

/// Substitute the template's env segments, upstream's `resolveTemplate`: one
/// unresolvable name voids the whole value.
fn resolve_template(
    parts: &[TemplatePart],
    env: Option<&BTreeMap<String, String>>,
    process_env: &EnvLookup,
) -> Option<String> {
    let mut resolved = String::new();
    for part in parts {
        match part {
            TemplatePart::Literal(value) => resolved.push_str(value),
            TemplatePart::Env { name } => {
                let value = resolve_env_config_value(name, env, process_env)?;
                resolved.push_str(&value);
            }
        }
    }
    Some(resolved)
}

/// The single env variable a config value is exactly one reference to,
/// upstream's `getConfigValueEnvVarName`.
#[must_use]
pub fn get_config_value_env_var_name(config: &str) -> Option<String> {
    match parse_config_value_reference(config) {
        ConfigValueReference::Command { .. } => None,
        ConfigValueReference::Template { parts } => match parts.as_slice() {
            [TemplatePart::Env { name }] => Some(name.clone()),
            _ => None,
        },
    }
}

/// All env variables a config value reads, upstream's
/// `getConfigValueEnvVarNames`.
#[must_use]
pub fn get_config_value_env_var_names(config: &str) -> Vec<String> {
    match parse_config_value_reference(config) {
        ConfigValueReference::Command { .. } => Vec::new(),
        ConfigValueReference::Template { parts } => template_env_var_names(&parts),
    }
}

/// The env variables a config value reads that resolve to nothing, upstream's
/// `getMissingConfigValueEnvVarNames`.
#[must_use]
pub fn get_missing_config_value_env_var_names(
    config: &str,
    env: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    get_config_value_env_var_names(config)
        .into_iter()
        .filter(|name| resolve_env_config_value(name, env, &default_env_lookup()).is_none())
        .collect()
}

/// The [`get_missing_config_value_env_var_names`] over an injected
/// environment lookup.
#[must_use]
pub fn get_missing_config_value_env_var_names_with(
    config: &str,
    env: Option<&BTreeMap<String, String>>,
    process_env: &EnvLookup,
) -> Vec<String> {
    get_config_value_env_var_names(config)
        .into_iter()
        .filter(|name| resolve_env_config_value(name, env, process_env).is_none())
        .collect()
}

/// Whether the value is a shell-command reference, upstream's
/// `isCommandConfigValue`.
#[must_use]
pub fn is_command_config_value(config: &str) -> bool {
    matches!(
        parse_config_value_reference(config),
        ConfigValueReference::Command { .. }
    )
}

/// Whether every env variable the value reads resolves, upstream's
/// `isConfigValueConfigured`.
#[must_use]
pub fn is_config_value_configured(config: &str, env: Option<&BTreeMap<String, String>>) -> bool {
    get_missing_config_value_env_var_names(config, env).is_empty()
}

/// Resolve a config value to an actual value, upstream's
/// `resolveConfigValue`:
///
/// - `!…` executes the rest as a shell command and uses trimmed stdout,
///   cached per config text for the process lifetime
/// - `$ENV_VAR` / `${ENV_VAR}` interpolate the named variable, read from the
///   credential's private map first and the environment second
/// - `$$` and `$!` escape literal `$` and `!` in non-command values
/// - anything else is a literal
///
/// One unresolvable env segment voids the whole value (`None`).
#[must_use]
pub fn resolve_config_value(
    config: &str,
    env: Option<&BTreeMap<String, String>>,
) -> Option<String> {
    resolve_config_value_with(config, env, &default_env_lookup())
}

/// [`resolve_config_value`] over an injected environment lookup.
#[must_use]
pub fn resolve_config_value_with(
    config: &str,
    env: Option<&BTreeMap<String, String>>,
    process_env: &EnvLookup,
) -> Option<String> {
    match parse_config_value_reference(config) {
        ConfigValueReference::Command { config } => execute_command_cached(&config),
        ConfigValueReference::Template { parts } => resolve_template(&parts, env, process_env),
    }
}

/// Run the command under `/bin/sh -c`, upstream's `executeWithDefaultShell`:
/// stdout piped, stderr discarded, a ten second deadline whose expiry kills
/// the child, and the trimmed output — empty output and every failure mode
/// resolve to `None`.
fn execute_with_default_shell(command: &str) -> Option<String> {
    let mut child = Command::new("/bin/sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;

    // execSync drains stdout while the timeout races; the reader thread owns
    // the pipe so a full buffer cannot stall the poll loop below.
    let reader = std::thread::spawn(move || {
        let mut buffer = String::new();
        let _ = std::io::Read::read_to_string(&mut stdout, &mut buffer);
        buffer
    });

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() >= COMMAND_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(2)),
            Err(_) => break None,
        }
    };

    let output = reader.join().ok()?;
    let success = status.is_some_and(|status| status.success());
    if !success {
        return None;
    }
    let trimmed = output.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// The uncached executor, upstream's `executeCommandUncached` POSIX branch:
/// the command text after the `!`.
fn execute_command_uncached(command_config: &str) -> Option<String> {
    execute_with_default_shell(&command_config[1..])
}

/// The cached executor, upstream's `executeCommand`: successful and failed
/// resolutions both cache, keyed on the full `!…` config text.
fn execute_command_cached(command_config: &str) -> Option<String> {
    let cache = COMMAND_RESULT_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(cached) = cache.get(command_config) {
        return cached.clone();
    }
    drop(cache);
    let result = execute_command_uncached(command_config);
    COMMAND_RESULT_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(command_config.to_string(), result.clone());
    result
}

/// Resolve without consulting or filling the command cache, upstream's
/// `resolveConfigValueUncached`.
#[must_use]
pub fn resolve_config_value_uncached(
    config: &str,
    env: Option<&BTreeMap<String, String>>,
) -> Option<String> {
    resolve_config_value_uncached_with(config, env, &default_env_lookup())
}

/// [`resolve_config_value_uncached`] over an injected environment lookup.
#[must_use]
pub fn resolve_config_value_uncached_with(
    config: &str,
    env: Option<&BTreeMap<String, String>>,
    process_env: &EnvLookup,
) -> Option<String> {
    match parse_config_value_reference(config) {
        ConfigValueReference::Command { config } => execute_command_uncached(&config),
        ConfigValueReference::Template { parts } => resolve_template(&parts, env, process_env),
    }
}

/// Resolve, failing with the named surface's message when the value resolves
/// to nothing, upstream's `resolveConfigValueOrThrow`.
///
/// # Errors
/// The upstream messages: a failed command names the command text, one
/// missing variable names it, several missing variables list them, and a
/// template with no env names falls to the bare form.
pub fn resolve_config_value_or_throw(
    config: &str,
    description: &str,
    env: Option<&BTreeMap<String, String>>,
) -> Result<String, String> {
    let resolved = resolve_config_value_uncached(config, env);
    if let Some(value) = resolved {
        return Ok(value);
    }

    match parse_config_value_reference(config) {
        ConfigValueReference::Command { config } => Err(format!(
            "Failed to resolve {description} from shell command: {}",
            &config[1..]
        )),
        ConfigValueReference::Template { .. } => {
            let missing = get_missing_config_value_env_var_names(config, env);
            match missing.as_slice() {
                [one] => Err(format!(
                    "Failed to resolve {description} from environment variable: {one}"
                )),
                many if many.len() > 1 => Err(format!(
                    "Failed to resolve {description} from environment variables: {}",
                    many.join(", ")
                )),
                _ => Err(format!("Failed to resolve {description}")),
            }
        }
    }
}

/// Resolve every header value with the key-resolution logic, upstream's
/// `resolveHeaders`: blank resolutions drop the entry, and an all-empty
/// result collapses to `None`.
#[must_use]
pub fn resolve_headers(
    headers: Option<&BTreeMap<String, String>>,
    env: Option<&BTreeMap<String, String>>,
) -> Option<BTreeMap<String, String>> {
    let headers = headers?;
    let mut resolved = BTreeMap::new();
    for (key, value) in headers {
        // Upstream's `if (resolvedValue)` truthiness: a resolved empty string
        // drops the entry here, while the or-throw variant below keeps it.
        if let Some(resolved_value) = resolve_config_value(value, env)
            && !resolved_value.is_empty()
        {
            resolved.insert(key.clone(), resolved_value);
        }
    }
    if resolved.is_empty() {
        None
    } else {
        Some(resolved)
    }
}

/// [`resolve_headers`] that fails instead of dropping, upstream's
/// `resolveHeadersOrThrow`.
///
/// # Errors
/// [`resolve_config_value_or_throw`]'s messages, per header value.
pub fn resolve_headers_or_throw(
    headers: Option<&BTreeMap<String, String>>,
    description: &str,
    env: Option<&BTreeMap<String, String>>,
) -> Result<Option<BTreeMap<String, String>>, String> {
    let Some(headers) = headers else {
        return Ok(None);
    };
    let mut resolved = BTreeMap::new();
    for (key, value) in headers {
        let resolved_value =
            resolve_config_value_or_throw(value, &format!("{description} header \"{key}\""), env)?;
        resolved.insert(key.clone(), resolved_value);
    }
    Ok(if resolved.is_empty() {
        None
    } else {
        Some(resolved)
    })
}

/// Clear the command-result cache, upstream's test-facing export.
pub fn clear_config_value_cache() {
    COMMAND_RESULT_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}
