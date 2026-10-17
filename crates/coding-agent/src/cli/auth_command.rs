//! The `pi auth` command grammar, upstream's `src/cli/auth-command.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's `validateAuthCommandArgs` reads a handful of fields off the
//! full `Args` (`cli/args.ts`, 447 lines); the full grammar ports with the
//! CLI-grammar slice, so the fields auth commands consume restated as
//! [`AuthCommandArgs`] — the view the full `Args` will feed.

use std::fmt;
use std::sync::LazyLock;

use regex::Regex;

use pi_ai::auth::types::AuthResult;

use crate::config::APP_NAME;

/// Which auth subcommand ran, upstream's `AuthCommandKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthCommandKind {
    /// `pi auth check`, upstream's `"check"`.
    Check,
    /// `pi auth print-api-key`, upstream's `"api_key"`.
    ApiKey,
    /// `pi auth print-bearer-token`, upstream's `"bearer_token"`.
    BearerToken,
}

impl AuthCommandKind {
    /// The wire's `snake_case` tag, upstream's union member — the JSON
    /// output's `"kind"` and the error messages' subject.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::ApiKey => "api_key",
            Self::BearerToken => "bearer_token",
        }
    }
}

impl fmt::Display for AuthCommandKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A parsed `auth` subcommand, upstream's `AuthCommand`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCommand {
    /// The subcommand, upstream's `kind`.
    pub kind: AuthCommandKind,
    /// The remaining arguments for the full CLI parse, upstream's `args`.
    pub args: Vec<String>,
    /// `auth check`'s `--json`, upstream's `json`.
    pub json: bool,
    /// `auth check`'s `--credentials`, upstream's `credentials`.
    pub credentials: bool,
    /// `auth check`'s `--no-refresh`, upstream's `noRefresh`.
    pub no_refresh: bool,
    /// `print-bearer-token`'s `--min-expiry` in milliseconds, upstream's
    /// `minExpiryMs?`.
    pub min_expiry_ms: Option<i64>,
}

/// An auth-command grammar failure, upstream's `AuthCommandError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCommandError(pub String);

impl fmt::Display for AuthCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AuthCommandError {}

/// The usage lines, upstream's `AUTH_COMMAND_USAGE`.
#[must_use]
pub fn get_auth_command_usage(kind: AuthCommandKind) -> String {
    match kind {
        AuthCommandKind::Check => {
            format!(
                "{APP_NAME} auth check --provider <provider> [--json] [--credentials] [--no-refresh]"
            )
        }
        AuthCommandKind::ApiKey => {
            format!("{APP_NAME} auth print-api-key --provider <provider> [--model <model>]")
        }
        AuthCommandKind::BearerToken => {
            format!(
                "{APP_NAME} auth print-bearer-token --provider <provider> [--model <model>] [--min-expiry <duration>]"
            )
        }
    }
}

/// The command label error messages interpolate, upstream's
/// `getAuthCommandName`.
#[must_use]
pub const fn get_auth_command_name(kind: AuthCommandKind) -> &'static str {
    match kind {
        AuthCommandKind::Check => "auth check",
        AuthCommandKind::ApiKey => "auth print-api-key",
        AuthCommandKind::BearerToken => "auth print-bearer-token",
    }
}

/// Whether the arguments open with an auth help request, upstream's
/// `isAuthCommandHelp`: `auth` alone, `auth help`, or an auth command
/// carrying `--help`/`-h`.
#[must_use]
pub fn is_auth_command_help(args: &[String]) -> bool {
    args.first().is_some_and(|first| first == "auth")
        && (args.get(1).is_none_or(|second| second == "help")
            || args.iter().any(|arg| arg == "--help" || arg == "-h"))
}

/// Print the auth help block, upstream's `printAuthCommandHelp`.
#[expect(
    clippy::print_stdout,
    reason = "the usage text mirrors upstream's direct stdout write; no writer is plumbed to this entry point"
)]
pub fn print_auth_command_help() {
    println!(
        "Usage:\n  pi auth print-api-key [--provider <provider>] [--model <model>]\n  pi auth print-bearer-token [--provider <provider>] [--model <model>] [--min-expiry <duration>]\n  pi auth check [--provider <provider>] [--model <model>] [--json] [--credentials] [--no-refresh]\n\nAuth commands require at least one of --provider or --model. Checks refresh expired OAuth credentials by default; --no-refresh prevents this. --credentials emits the credential, or includes it in JSON output."
    );
}

/// Parse an `auth` subcommand out of the raw arguments, upstream's
/// `parseAuthCommand`.
///
/// `Ok(None)` when the arguments do not open with `auth` (the caller falls
/// through to the full parse); [`AuthCommandError`] on a grammar failure.
///
/// # Errors
/// An unknown subcommand, a flag outside its command, or a malformed
/// `--min-expiry` duration.
pub fn parse_auth_command(args: &[String]) -> Result<Option<AuthCommand>, AuthCommandError> {
    if args.first().is_none_or(|first| first != "auth") {
        return Ok(None);
    }

    let kind = match args.get(1).map(String::as_str) {
        Some("check") => AuthCommandKind::Check,
        Some("print-api-key") => AuthCommandKind::ApiKey,
        Some("print-bearer-token") => AuthCommandKind::BearerToken,
        other => {
            let spelling = other.unwrap_or("");
            return Err(AuthCommandError(format!(
                "Unknown auth command \"{spelling}\". Use \"{APP_NAME} auth print-api-key\", \"{APP_NAME} auth print-bearer-token\", or \"{APP_NAME} auth check\"."
            )));
        }
    };

    let mut command_args: Vec<String> = Vec::new();
    let mut json = false;
    let mut credentials = false;
    let mut no_refresh = false;
    let mut min_expiry_ms: Option<i64> = None;
    let mut index = 2;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--min-expiry" {
            if kind != AuthCommandKind::BearerToken {
                return Err(AuthCommandError(
                    "--min-expiry is only supported by print-bearer-token".to_string(),
                ));
            }
            index += 1;
            let value = args.get(index);
            let parsed = value.and_then(|value| parse_min_expiry(value));
            match parsed {
                Some(millis) => min_expiry_ms = Some(millis),
                None => {
                    return Err(AuthCommandError(
                        "--min-expiry must use a duration such as 30m or 1h".to_string(),
                    ));
                }
            }
        } else if arg == "--json" || arg == "--credentials" || arg == "--no-refresh" {
            if kind != AuthCommandKind::Check {
                return Err(AuthCommandError(format!(
                    "{arg} is only supported by auth check"
                )));
            }
            match arg.as_str() {
                "--json" => json = true,
                "--credentials" => credentials = true,
                _ => no_refresh = true,
            }
        } else {
            command_args.push(arg.clone());
        }
        index += 1;
    }

    Ok(Some(AuthCommand {
        kind,
        args: command_args,
        json,
        credentials,
        no_refresh,
        min_expiry_ms,
    }))
}

/// The `--min-expiry` duration grammar, upstream's `/^(\d+)(ms|s|m|h)$/iu`
/// pattern.
#[expect(
    clippy::expect_used,
    reason = "the pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static MIN_EXPIRY_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(\d+)(ms|s|m|h)$").expect("the min-expiry pattern is a valid regex")
});

/// Parse a `--min-expiry` duration, upstream's `/^(\d+)(ms|s|m|h)$/iu`.
///
/// The grammar is digits then a unit, case-insensitive.
fn parse_min_expiry(value: &str) -> Option<i64> {
    let captures = MIN_EXPIRY_PATTERN.captures(value)?;
    let amount: i64 = captures.get(1)?.as_str().parse().ok()?;
    let unit = captures.get(2)?.as_str().to_ascii_lowercase();
    let multiplier = match unit.as_str() {
        "ms" => 1,
        "s" => 1_000,
        "m" => 60_000,
        _ => 3_600_000,
    };
    amount.checked_mul(multiplier)
}

/// The flag values the full CLI parse can attach to an unknown flag, upstream's
/// `unknownFlags`.
///
/// A bare flag is [`UnknownFlagValue::Flag`], a valued one
/// [`UnknownFlagValue::Value`]; upstream's type is `Map<string, boolean | string>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownFlagValue {
    /// The flag appeared without a value, upstream's `true`.
    Flag,
    /// The flag carried a value, upstream's string.
    Value(String),
}

/// The `Args` fields auth commands read, upstream's `Args` reduced to the
/// slice its validation consumes — the full `cli/args.ts` grammar ports
/// with the CLI-grammar slice and feeds this view.
#[derive(Debug, Clone, Default)]
pub struct AuthCommandArgs {
    /// The `--provider` value, upstream's `provider?`.
    pub provider: Option<String>,
    /// The `--model` value, upstream's `model?`.
    pub model: Option<String>,
    /// The `--api-key` value, upstream's `apiKey?`.
    pub api_key: Option<String>,
    /// The positional prompts, upstream's `messages`.
    pub messages: Vec<String>,
    /// The `@file` inputs, upstream's `fileArgs`.
    pub file_args: Vec<String>,
    /// The flags the grammar did not claim, upstream's `unknownFlags`
    /// (insertion order preserved for the first-key error).
    pub unknown_flags: indexmap::IndexMap<String, UnknownFlagValue>,
}

/// The resolved provider/model pair, upstream's `{ provider?, model? }`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuthCommandTarget {
    /// The trimmed `--provider`, upstream's `provider?`.
    pub provider: Option<String>,
    /// The trimmed `--model`, upstream's `model?`.
    pub model: Option<String>,
}

/// Validate the shared flags of an auth command, upstream's
/// `validateAuthCommandArgs`.
///
/// # Errors
/// An unknown flag, an `--api-key`, prompt, or `@file` input, or a check
/// or print without either `--provider` or `--model` — the messages
/// upstream throws verbatim.
pub fn validate_auth_command_args(
    args: &AuthCommandArgs,
    kind: AuthCommandKind,
) -> Result<AuthCommandTarget, AuthCommandError> {
    let provider = args
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string);
    let model = args
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(str::to_string);
    if let Some((flag, _)) = args.unknown_flags.iter().next() {
        return Err(AuthCommandError(format!(
            "Unknown option --{flag} for \"{}\".",
            get_auth_command_name(kind)
        )));
    }
    if args.api_key.is_some() || !args.messages.is_empty() || !args.file_args.is_empty() {
        return Err(AuthCommandError(
            "Auth commands only accept --provider and --model".to_string(),
        ));
    }
    if kind == AuthCommandKind::Check {
        if provider.is_none() && model.is_none() {
            return Err(AuthCommandError(
                "Auth checks require --provider <provider> or --model <model>".to_string(),
            ));
        }
        return Ok(AuthCommandTarget { provider, model });
    }
    if provider.is_none() && model.is_none() {
        return Err(AuthCommandError(
            "Credential printing requires --provider <provider> or --model <model>".to_string(),
        ));
    }
    Ok(AuthCommandTarget { provider, model })
}

/// The `Bearer ` credential prefix off an `authorization` header, upstream's
/// inline pattern in `getAuthCredential`.
#[expect(
    clippy::expect_used,
    reason = "the pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static BEARER_PREFIX_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^Bearer\s+(.+)$").expect("the bearer-prefix pattern is a valid regex")
});

/// Extract the credential string an auth result carries, upstream's
/// `getAuthCredential`: the API key when set, else the `Bearer ` prefix
/// (case-insensitive, whitespace run) off an `authorization` header.
#[must_use]
pub fn get_auth_credential(auth: Option<&AuthResult>) -> Option<String> {
    let auth = auth?;
    if let Some(api_key) = auth.auth.api_key.as_deref().filter(|key| !key.is_empty()) {
        return Some(api_key.to_string());
    }
    let headers = auth.auth.headers.as_ref()?;
    let (_, value) = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))?;
    let value = value.as_ref()?;
    BEARER_PREFIX_PATTERN
        .captures(value)
        .and_then(|captures| captures.get(1))
        .map(|matched| matched.as_str().to_string())
}

#[cfg(test)]
mod tests;
