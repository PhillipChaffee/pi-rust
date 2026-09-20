//! The `pi-ai` command line, the library half of upstream's
//! `packages/ai/src/cli.ts` at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's file is a standalone script: it filters `builtinProviders()`
//! down to the providers with an OAuth flow, reads and writes `auth.json` in
//! the working directory, and drives stdin/stdout through `node:readline`.
//! The port splits the roles: this module carries the rendering, provider
//! lookup, credential-file io, and the stdin/stdout [`AuthInteraction`]; the
//! `[[bin]]` entry (its ticket wires `builtinProviders` through the provider
//! registry, each OAuth flow via [`crate::auth::oauth`]) builds the
//! [`CliProvider`] list, parses the argument vector, calls [`run`], and
//! prints the returned [`AuthError`] the way upstream's `main().catch(...)`
//! does ("Error: {message}", exit code 1).
//!
//! Porting restatements this module records:
//!
//! - `loadAuth`/`saveAuth` stored raw OAuth credential objects under each
//!   provider id; the port stores the tagged [`Credential`] union, so the
//!   file's `type` field discriminates and both credential kinds
//!   round-trip.
//! - The `AUTH_FILE` constant becomes the `auth_path` parameter — the bin
//!   keeps upstream's working-directory `auth.json`, tests pass a temp path
//!   — and the "Credentials saved to" line prints the file name, the form
//!   upstream's constant produced.
//! - `saveAuth` gains what upstream's plain `writeFileSync` lacked and the
//!   house store rules require: the parent directory is created with 0700
//!   when missing and the file is set to 0600 after writing, so stored key
//!   material stays owner-only. Nothing prints key material.
//! - `node:readline`'s question promise becomes a blocking stdin read
//!   inside [`tokio::task::spawn_blocking`]; stdin EOF fails the pending
//!   prompt as the abort rejection [`PromptFn`](crate::auth::types::PromptFn)
//!   carries (upstream's readline never resolves and the process exits), so
//!   driving one needs a tokio runtime context.
//! - The per-prompt signal on [`AuthPrompt`] is accepted but not observed:
//!   a terminal readline cannot be cancelled, and upstream passes a
//!   never-aborted controller.
//!
//! The output strings — help text, list lines, the notify switch, the
//! saved-to line — are byte-for-byte upstream's.

#![expect(
    clippy::print_stdout,
    reason = "the CLI's job is printing prompts and results to stdout, upstream's console.log surface"
)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::auth::oauth::auth_error;
use crate::auth::types::{
    AuthError, AuthEvent, AuthInteraction, AuthPrompt, AuthPromptKind, Credential, PromptFn,
};

/// One OAuth provider the CLI offers, upstream's filtered
/// `builtinProviders()` entry: a provider whose `auth.oauth` is set.
///
/// It carries the id/name pair the commands print plus the flow the `login`
/// command drives.
#[derive(Debug)]
pub struct CliProvider {
    /// The provider id, upstream's `Provider.id` and the credential's key in
    /// `auth.json`.
    pub id: String,
    /// The display name, upstream's `Provider.name`.
    pub name: String,
    /// The OAuth flow `login` drives.
    pub oauth: crate::auth::types::OAuthAuth,
}

/// The usage text for the help command and the no-command case, byte-for-byte
/// upstream's `console.log` payload.
///
/// The fixed usage block is followed by one provider line per entry,
/// `  {id:<20} {name}` joined with newlines.
#[must_use]
pub fn help_text(providers: &[CliProvider]) -> String {
    let provider_list = providers
        .iter()
        .map(|provider| format!("  {}", padded_line(provider)))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Usage: npx @earendil-works/pi-ai <command> [provider]\n\nCommands:\n  login [provider]  Login to an \
         OAuth provider\n  list              List available providers\n\nProviders:\n{provider_list}"
    )
}

/// One line per provider, `{id:<20} {name}` joined with newlines, the `list`
/// command's payload (upstream prints each line with `padEnd(20)` on the id).
#[must_use]
pub fn list_output(providers: &[CliProvider]) -> String {
    providers
        .iter()
        .map(padded_line)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The provider whose id matches `provider_id`, upstream's
/// `PROVIDERS.find((entry) => entry.id === providerId)`.
#[must_use]
pub fn find_provider<'a>(
    providers: &'a [CliProvider],
    provider_id: &str,
) -> Option<&'a CliProvider> {
    providers.iter().find(|provider| provider.id == provider_id)
}

/// The chosen option's id from a select prompt's entered line, the parse half
/// of upstream's `answerPrompt` select branch.
///
/// The caller prints the `\n{message}` header and the numbered options — the
/// menu needs the prompt's message, which this function does not carry —
/// then passes the entered line here. `raw` parses as a 1-based index
/// (upstream's `Number.parseInt(input, 10) - 1`, so surrounding whitespace
/// is ignored) and the selected option's `id` is returned.
///
/// # Errors
/// `Invalid selection` when the line does not parse to a number inside
/// `1..=options.len()`.
pub fn select_choice(
    options: &[crate::auth::types::AuthPromptOption],
    raw: &str,
) -> Result<String, AuthError> {
    let selected = raw
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|number| number.checked_sub(1))
        .and_then(|index| options.get(index));
    selected.map_or_else(
        || Err(auth_error("Invalid selection")),
        |option| Ok(option.id.clone()),
    )
}

/// The question line for a non-select prompt, upstream's template
/// `` `${message}${placeholder ? ` (${placeholder})` : ""}: ` ``.
#[must_use]
pub fn prompt_line(message: &str, placeholder: Option<&str>) -> String {
    placeholder.map_or_else(
        || format!("{message}: "),
        |placeholder| format!("{message} ({placeholder}): "),
    )
}

/// Read the credential store at `auth_path`, upstream's `loadAuth`: a
/// missing file or a parse failure yields an empty store.
///
/// Entries parse as the tagged [`Credential`] union keyed on the `type`
/// field.
#[must_use]
pub fn load_credentials(auth_path: &str) -> BTreeMap<String, Credential> {
    load_from(Path::new(auth_path))
}

/// The store reader [`load_credentials`] and [`save_credentials`] share: any
/// read or parse failure collapses to an empty store, upstream's
/// `try { JSON.parse(...) } catch { return {} }`.
fn load_from(auth_path: &Path) -> BTreeMap<String, Credential> {
    std::fs::read_to_string(auth_path)
        .ok()
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default()
}

/// An io failure on the credential file, as the [`AuthError`] the store
/// operations report, naming the action and the path.
fn io_error(action: &str, path: &Path, error: &std::io::Error) -> AuthError {
    auth_error(format!("Failed to {action} {}: {error}", path.display()))
}

/// Merge `credential` into the store under `provider_id` and write it back
/// to `auth_path` — upstream's `login()` sequence
/// `auth[providerId] = credential; saveAuth(auth)` in one call.
///
/// The file is pretty JSON with a trailing newline; the parent directory is
/// created with 0700 when missing and the file is set to 0600 after writing,
/// so stored key material stays owner-only.
///
/// # Errors
/// Fails when creating the parent directory, serializing the store, writing
/// the file, or applying permissions fails, with the reason's message.
pub fn save_credentials(
    auth_path: &Path,
    provider_id: &str,
    credential: &Credential,
) -> Result<(), AuthError> {
    let mut storage = load_from(auth_path);
    storage.insert(provider_id.to_owned(), credential.clone());
    if let Some(parent) = auth_path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).map_err(|error| io_error("create", parent, &error))?;
        std::fs::set_permissions(parent, std::os::unix::fs::PermissionsExt::from_mode(0o700))
            .map_err(|error| io_error("set permissions on", parent, &error))?;
    }
    let mut serialized = serde_json::to_string_pretty(&storage)
        .map_err(|error| auth_error(format!("Failed to serialize credentials: {error}")))?;
    serialized.push('\n');
    std::fs::write(auth_path, serialized).map_err(|error| io_error("write", auth_path, &error))?;
    std::fs::set_permissions(
        auth_path,
        std::os::unix::fs::PermissionsExt::from_mode(0o600),
    )
    .map_err(|error| io_error("set permissions on", auth_path, &error))?;
    Ok(())
}

/// The stdin/stdout interaction the `login` command drives, upstream's
/// `createInterface`-backed `prompt`/`notify` pair.
///
/// `prompt` renders the prompt (the numbered select menu, or the
/// message/placeholder question) to stdout and reads one line from stdin;
/// `notify` prints each event per upstream's switch. Prompts read through
/// [`tokio::task::spawn_blocking`], so driving one needs a tokio runtime;
/// stdin EOF fails the pending prompt as the abort rejection.
#[must_use]
pub fn cli_interaction() -> AuthInteraction {
    AuthInteraction {
        signal: None,
        prompt: prompt_fn(),
        notify: notify_fn(),
    }
}

/// The prompt half of [`cli_interaction`]: render and read one line, the
/// `node:readline` question port.
fn prompt_fn() -> PromptFn {
    Arc::new(move |auth_prompt: AuthPrompt| {
        Box::pin(async move {
            match auth_prompt.kind {
                AuthPromptKind::Select { message, options } => {
                    println!("\n{message}");
                    for (index, option) in options.iter().enumerate() {
                        println!("  {}. {}", index + 1, option.label);
                    }
                    let raw = read_line(&format!("Enter number (1-{}): ", options.len())).await?;
                    select_choice(&options, &raw).map_err(|_| crate::utils::abort::AbortError)
                }
                AuthPromptKind::Text {
                    message,
                    placeholder,
                }
                | AuthPromptKind::Secret {
                    message,
                    placeholder,
                }
                | AuthPromptKind::ManualCode {
                    message,
                    placeholder,
                } => {
                    let question = prompt_line(&message, placeholder.as_deref());
                    read_line(&question).await
                }
            }
        })
    })
}

/// The notify half of [`cli_interaction`], upstream's event switch.
fn notify_fn() -> crate::auth::types::NotifyFn {
    Arc::new(move |event: AuthEvent| match event {
        AuthEvent::AuthUrl { url, instructions } => {
            println!("\nOpen this URL in your browser:\n{url}");
            if let Some(instructions) = instructions {
                println!("{instructions}");
            }
        }
        AuthEvent::DeviceCode {
            user_code,
            verification_uri,
            ..
        } => {
            println!("\nOpen this URL in your browser:\n{verification_uri}");
            println!("Enter code: {user_code}");
        }
        AuthEvent::Info { message, .. } | AuthEvent::Progress { message } => {
            println!("{message}");
        }
    })
}

/// The `main()` port: dispatch the argument vector against `providers` and
/// the credential file at `auth_path`.
///
/// No command or a help variant prints [`help_text`]; `list` prints one
/// padded line per provider; `login` without an id picks the provider from
/// a numbered prompt, then runs its OAuth flow over [`cli_interaction`] with
/// a fresh uncancelled signal and persists the credential, printing
/// "\nCredentials saved to {file name}".
///
/// The caller prints the returned error the way upstream's
/// `main().catch(...)` does ("Error: {message}", exit code 1).
///
/// # Errors
/// `Unknown provider: {id}` when the login target is missing — an invalid
/// number or closed stdin selects the empty id upstream sends —
/// `Unknown command: {command}` for any other command, and the OAuth-flow or
/// persistence failures `login_provider` reports.
pub async fn run(
    args: &[String],
    providers: &[CliProvider],
    auth_path: &str,
) -> Result<(), AuthError> {
    match args.first().map(String::as_str) {
        None | Some("help" | "--help" | "-h") => {
            println!("{}", help_text(providers));
            Ok(())
        }
        Some("list") => {
            for provider in providers {
                println!("{}", padded_line(provider));
            }
            Ok(())
        }
        Some("login") => {
            let selected: Option<String> = match args.get(1).map(String::as_str) {
                Some(id) => Some(id.to_owned()),
                None => Some(select_provider(providers).await?),
            };
            let provider_id = selected.unwrap_or_default();
            let Some(provider) = find_provider(providers, &provider_id) else {
                return Err(auth_error(format!("Unknown provider: {provider_id}")));
            };
            login_provider(provider, auth_path).await
        }
        Some(command) => Err(auth_error(format!("Unknown command: {command}"))),
    }
}

/// The padded list line for one provider, upstream's
/// `` `${provider.id.padEnd(20)} ${provider.name}` ``.
fn padded_line(provider: &CliProvider) -> String {
    format!("{:<20} {}", provider.id, provider.name)
}

/// The file name of `auth_path`, falling back to the path itself when it has
/// none — the printed form of upstream's `AUTH_FILE` constant.
fn display_file_name(auth_path: &str) -> &str {
    Path::new(auth_path)
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or(auth_path)
}

/// Print the question and read one line from stdin, the `node:readline`
/// question port: the trailing newline/CR is stripped the way readline
/// strips it, and the blocking read rides [`tokio::task::spawn_blocking`] so
/// the runtime stays responsive.
///
/// # Errors
/// Fails as the abort rejection when the prompt cannot be flushed, when
/// stdin ends before a line arrives, when the read fails, or when the
/// blocking task itself fails — all leave no answer to return, so the prompt
/// rejects the way a cancelled prompt does (upstream's closed-terminal path
/// never resolves).
async fn read_line(question: &str) -> Result<String, crate::utils::abort::AbortError> {
    print!("{question}");
    std::io::stdout()
        .flush()
        .map_err(|_| crate::utils::abort::AbortError)?;
    let line = tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            // stdin closed before a line arrived, or the read failed: both
            // end the interactive session, so no answer exists to return.
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_owned()),
        }
    })
    .await;
    match line {
        Ok(Some(line)) => Ok(line),
        // A missing answer is the abort the prompt contract carries.
        Ok(None) | Err(_) => Err(crate::utils::abort::AbortError),
    }
}

/// The provider picker for `login` without an id: print the numbered provider
/// names, read the choice, and return the selected provider's id — `None` on
/// an invalid number or closed stdin, which the caller reports as
/// `Unknown provider: ` with the empty id upstream sends.
async fn select_provider(providers: &[CliProvider]) -> Result<String, AuthError> {
    for (index, provider) in providers.iter().enumerate() {
        println!("  {}. {}", index + 1, provider.name);
    }
    let raw = read_line(&format!("Enter number (1-{}): ", providers.len()))
        .await
        .map_err(|_| auth_error("Login cancelled"))?;
    let selected = raw
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|number| number.checked_sub(1))
        .and_then(|index| providers.get(index));
    selected
        .map(|provider| provider.id.clone())
        .ok_or_else(|| auth_error(String::new()))
}

/// Run one provider's OAuth login and persist the credential, upstream's
/// `login()`: the flow drives [`cli_interaction`] over a fresh uncancelled
/// signal (upstream's `new AbortController().signal`), then the credential
/// merges into `auth_path`'s store and the saved-to line prints the file
/// name.
///
/// # Errors
/// Fails when the OAuth flow fails or when persisting the credential fails.
async fn login_provider(provider: &CliProvider, auth_path: &str) -> Result<(), AuthError> {
    let interaction = crate::auth::types::ProviderAuthInteraction::from_interaction(
        cli_interaction(),
        tokio_util::sync::CancellationToken::new(),
    );
    let credential = (provider.oauth.login)(interaction).await?;
    save_credentials(
        Path::new(auth_path),
        &provider.id,
        &Credential::OAuth(credential),
    )?;
    println!("\nCredentials saved to {}", display_file_name(auth_path));
    Ok(())
}
