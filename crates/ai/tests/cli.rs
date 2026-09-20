//! The CLI module suite, covering the library half of upstream's
//! `packages/ai/src/cli.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the help/list rendering, the
//! provider lookup and select parsing, the owner-only credential-file io,
//! and the argument dispatch of [`pi_ai::cli::run`].
//!
//! The interactive stdin paths stay uncovered: [`CliInteraction`]'s prompts
//! read the process stdin, so the suites drive `run` only with a provider id
//! (which skips the picker) and a stub OAuth flow (which never prompts).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::auth_fixtures::{StubOAuthAuth, oauth_credentials};
use pi_ai::auth::types::{ApiKeyCredential, AuthPromptOption, Credential};
use pi_ai::cli::{
    CliProvider, cli_interaction, find_provider, help_text, list_output, load_credentials,
    prompt_line, run, save_credentials, select_choice,
};
use pi_ai::utils::abort::AbortError;

/// The environment guard the stdout probe sets on its child run.
const PRINT_PROBE: &str = "PI_AI_CLI_PRINT_PROBE";

/// A CLI provider over a stub flow, the filtered `builtinProviders()` entry
/// the commands render and drive.
fn cli_provider(id: &str, name: &str) -> CliProvider {
    CliProvider {
        id: id.to_owned(),
        name: name.to_owned(),
        oauth: StubOAuthAuth::new(
            format!("{name} OAuth"),
            Ok(oauth_credentials("cli-access", "cli-refresh", 123_456)),
        )
        .auth(),
    }
}

/// A unique temp directory, `std::env::temp_dir().join(unique)`, safe under
/// parallel test binaries.
fn unique_temp_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let unique = format!(
        "pi-ai-cli-{}-{}-{}",
        tag,
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
    );
    std::env::temp_dir().join(unique)
}

#[test]
fn help_text_lists_the_usage_and_the_provider_lines() {
    let providers = vec![
        cli_provider("anthropic", "Anthropic (Claude Pro/Max)"),
        cli_provider("openai-codex", "OpenAI (ChatGPT Plus/Pro)"),
    ];
    assert_eq!(
        help_text(&providers),
        "Usage: npx @earendil-works/pi-ai <command> [provider]\n\nCommands:\n  login [provider]  \
         Login to an OAuth provider\n  list              List available providers\n\nProviders:\n  \
         anthropic            Anthropic (Claude Pro/Max)\n  openai-codex         OpenAI (ChatGPT \
         Plus/Pro)"
    );
}

#[test]
fn list_output_pads_the_ids_to_twenty_columns() {
    let providers = vec![
        cli_provider("anthropic", "Anthropic"),
        cli_provider("openai-codex", "OpenAI Codex"),
    ];
    assert_eq!(
        list_output(&providers),
        "anthropic            Anthropic\nopenai-codex         OpenAI Codex"
    );
}

#[test]
fn find_provider_matches_by_id() {
    let providers = vec![cli_provider("stub", "Stub Provider")];
    let found = find_provider(&providers, "stub").expect("the stub provider is found");
    assert_eq!(found.id, "stub");
    assert_eq!(found.name, "Stub Provider");
    assert!(find_provider(&providers, "ghost").is_none());
}

#[test]
fn select_choice_returns_the_selected_option_id() {
    let options = login_options();
    assert_eq!(
        select_choice(&options, "1").map_err(|error| error.to_string()),
        Ok("browser".to_owned())
    );
    assert_eq!(
        select_choice(&options, " 2 ").map_err(|error| error.to_string()),
        Ok("device_code".to_owned())
    );
    for raw in ["0", "3", "abc", ""] {
        let error = select_choice(&options, raw).expect_err("an invalid line rejects");
        assert_eq!(error.to_string(), "Invalid selection");
    }
}

#[test]
fn prompt_line_appends_the_question_colon_and_the_placeholder() {
    assert_eq!(prompt_line("Enter key", None), "Enter key: ");
    assert_eq!(
        prompt_line("Enter key", Some("sk-...")),
        "Enter key (sk-...): "
    );
}

#[test]
fn credentials_round_trip_with_owner_only_permissions() {
    let dir = unique_temp_dir("round-trip");
    let auth_path = dir.join("auth.json");
    let mut oauth = oauth_credentials("access", "refresh", 12_345);
    oauth.extra.insert(
        "enterpriseUrl".to_owned(),
        serde_json::Value::String("company.ghe.com".to_owned()),
    );
    save_credentials(
        &auth_path,
        "github-copilot",
        &Credential::OAuth(oauth.clone()),
    )
    .expect("the oauth credential saves");
    save_credentials(
        &auth_path,
        "other",
        &Credential::ApiKey(ApiKeyCredential {
            key: Some("k".to_owned()),
            env: None,
        }),
    )
    .expect("the api-key credential saves");

    let file_mode = std::fs::metadata(&auth_path)
        .expect("the file exists")
        .permissions()
        .mode();
    assert_eq!(
        file_mode & 0o777,
        0o600,
        "stored key material stays owner-only"
    );
    let parent_mode = std::fs::metadata(&dir)
        .expect("the directory exists")
        .permissions()
        .mode();
    assert_eq!(
        parent_mode & 0o777,
        0o700,
        "the created directory stays owner-only"
    );

    let loaded = load_credentials(auth_path.to_str().expect("utf-8 path"));
    assert_eq!(loaded.len(), 2);
    assert_eq!(
        loaded.get("github-copilot"),
        Some(&Credential::OAuth(oauth))
    );
    assert_eq!(
        loaded.get("other"),
        Some(&Credential::ApiKey(ApiKeyCredential {
            key: Some("k".to_owned()),
            env: None
        }))
    );
    let content = std::fs::read_to_string(&auth_path).expect("the store reads back");
    assert!(
        content.ends_with('\n'),
        "the pretty JSON keeps its trailing newline"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn run_list_prints_the_padded_lines() {
    // println! has no stable in-process capture, so the probe re-runs this
    // test in a child process and the parent reads its stdout.
    if std::env::var(PRINT_PROBE).is_ok_and(|value| value == "1") {
        let providers = vec![
            cli_provider("test-provider", "Test Provider"),
            cli_provider("openai-codex", "OpenAI Codex"),
        ];
        let outcome = block_run(&[String::from("list")], &providers, "unused-auth.json");
        assert!(outcome.is_ok(), "list succeeds: {outcome:?}");
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("the test binary path"))
        .args(["--exact", "run_list_prints_the_padded_lines", "--nocapture"])
        .env(PRINT_PROBE, "1")
        .output()
        .expect("the probe child runs");
    assert!(
        output.status.success(),
        "the probe child passes: {output:?}"
    );
    let stdout = String::from_utf8(output.stdout).expect("the child's stdout is utf-8");
    assert!(
        stdout.contains("test-provider        Test Provider"),
        "the padded list line prints: {stdout:?}"
    );
    assert!(
        stdout.contains("openai-codex         OpenAI Codex"),
        "every provider prints padded: {stdout:?}"
    );
}

#[tokio::test]
async fn run_login_unknown_provider_fails() {
    let providers = vec![cli_provider("stub", "Stub Provider")];
    let outcome = run(
        &[String::from("login"), String::from("ghost")],
        &providers,
        "unused-auth.json",
    )
    .await;
    assert_eq!(
        outcome.expect_err("an unknown provider fails").to_string(),
        "Unknown provider: ghost"
    );
}

#[tokio::test]
async fn run_unknown_command_fails_and_help_variants_succeed() {
    let providers = vec![cli_provider("stub", "Stub Provider")];
    let error = run(&[String::from("wat")], &providers, "unused-auth.json")
        .await
        .expect_err("an unknown command fails");
    assert_eq!(error.to_string(), "Unknown command: wat");

    let dir = unique_temp_dir("help");
    let auth_path = dir.join("auth.json");
    for args in [
        vec![],
        vec![String::from("help")],
        vec![String::from("--help")],
    ] {
        let outcome = run(&args, &providers, auth_path.to_str().expect("utf-8 path")).await;
        assert!(outcome.is_ok(), "help prints: {outcome:?}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn run_login_with_an_explicit_id_persists_the_credential_without_input() {
    let dir = unique_temp_dir("login");
    let auth_path = dir.join("auth.json");
    let credential = oauth_credentials("cli-access", "cli-refresh", 123_456);
    let oauth = StubOAuthAuth::new("Stub OAuth", Ok(credential.clone())).auth();
    let providers = vec![CliProvider {
        id: "stub".to_owned(),
        name: "Stub Provider".to_owned(),
        oauth,
    }];

    // The explicit provider id skips the picker, and the stub flow never
    // prompts, so the process stdin is never read.
    let outcome = run(
        &[String::from("login"), String::from("stub")],
        &providers,
        auth_path.to_str().expect("utf-8 path"),
    )
    .await;
    assert!(outcome.is_ok(), "login succeeds: {outcome:?}");

    let file_mode = std::fs::metadata(&auth_path)
        .expect("the store saves")
        .permissions()
        .mode();
    assert_eq!(file_mode & 0o777, 0o600, "the saved store stays owner-only");

    let loaded = load_credentials(auth_path.to_str().expect("utf-8 path"));
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.get("stub"), Some(&Credential::OAuth(credential)));

    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// The interactive stdin paths, driven through probe children with piped stdin
// ---------------------------------------------------------------------------

/// The probe mode this binary re-runs itself under, keyed per scenario.
const INTERACTIVE_PROBE: &str = "PI_AI_CLI_INTERACTIVE_PROBE";

/// Spawn the interaction probe: this binary re-running its own test with
/// `stdin_bytes` piped in, the child's captured output.
fn run_probe(mode: &str, stdin_bytes: &[u8]) -> std::process::Output {
    spawn_probe(
        "the_cli_interaction_drives_piped_stdin",
        INTERACTIVE_PROBE,
        mode,
        stdin_bytes,
    )
}

/// Spawn this suite's own binary for `mode` under `probe_env` with
/// `stdin_bytes` piped in, running the named test, and return the captured
/// output.
fn spawn_probe(
    test_name: &str,
    probe_env: &str,
    mode: &str,
    stdin_bytes: &[u8],
) -> std::process::Output {
    use std::io::Write as _;
    use std::process::Stdio;

    let mut child =
        std::process::Command::new(std::env::current_exe().expect("the test binary path"))
            .args(["--exact", test_name, "--nocapture"])
            .env(probe_env, mode)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the probe child spawns");
    child
        .stdin
        .as_mut()
        .expect("the child stdin is piped")
        .write_all(stdin_bytes)
        .expect("the stdin bytes write");
    child.wait_with_output().expect("the probe child runs")
}

/// Drive `run` on a fresh current-thread runtime, the seam the probe modes
/// use to exercise the CLI without a multi-thread runtime.
fn block_run(
    args: &[String],
    providers: &[CliProvider],
    auth_path: &str,
) -> Result<(), pi_ai::auth::types::AuthError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the runtime builds")
        .block_on(run(args, providers, auth_path))
}

/// The two login-method options the select cases render.
fn login_options() -> Vec<AuthPromptOption> {
    vec![
        AuthPromptOption {
            id: "browser".to_owned(),
            label: "Browser login".to_owned(),
            description: None,
        },
        AuthPromptOption {
            id: "device_code".to_owned(),
            label: "Device code login".to_owned(),
            description: None,
        },
    ]
}

#[test]
fn the_cli_interaction_drives_piped_stdin() {
    if let Ok(mode) = std::env::var(INTERACTIVE_PROBE) {
        run_interaction_probe_mode(&mode);
        return;
    }

    // The select menu renders, the entered number picks the option's id.
    let output = run_probe("select", b"2\n");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the select probe passes: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("\nPick a login method:")
            && stdout.contains("  1. Browser login")
            && stdout.contains("  2. Device code login")
            && stdout.contains("Enter number (1-2): "),
        "the numbered menu prints: {stdout:?}"
    );

    // The plain question renders message and placeholder.
    let output = run_probe("question", b"pasted-url\n");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the question probe passes: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Paste the redirect (http://localhost:1/cb): "),
        "the question renders with its placeholder: {stdout:?}"
    );

    // The manual-code prompt renders through the same question line.
    let output = run_probe("manual-code", b"pasted-code\n");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the manual-code probe passes: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Paste the code (http://localhost:1/cb): "),
        "the manual-code prompt renders: {stdout:?}"
    );

    // The secret prompt renders through the same question line.
    let output = run_probe("secret", b"sk-ambient\n");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the secret probe passes: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Enter key (masked): "),
        "the secret prompt renders: {stdout:?}"
    );

    // stdin EOF fails the pending prompt as the closed-terminal rejection.
    let output = run_probe("eof", b"");
    assert!(
        output.status.success(),
        "the eof probe passes: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The notify switch renders every event shape.
    let output = run_probe("notify", b"");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the notify probe passes: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("\nOpen this URL in your browser:\nhttps://authorize.example")
            && stdout.contains("Continue in your browser."),
        "the auth url event renders: {stdout:?}"
    );
    assert!(
        stdout.contains("\nOpen this URL in your browser:\nhttps://verify.example")
            && stdout.contains("Enter code: ABCD-1234"),
        "the device code event renders: {stdout:?}"
    );
    assert!(
        stdout.contains("an info message") && stdout.contains("a progress message"),
        "the info and progress events render: {stdout:?}"
    );
    assert!(
        stdout.contains("https://authorize-bare.example"),
        "the instruction-less event renders: {stdout:?}"
    );

    // A select prompt with no line to read fails as the closed-terminal
    // rejection.
    let output = run_probe("select-eof", b"");
    assert!(
        output.status.success(),
        "the select-eof probe passes: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The child half of [`the_cli_interaction_drives_piped_stdin`]: each mode
/// drives one interactive path against the piped stdin and asserts its own
/// contract; the parent asserts the rendered stdout.
fn run_interaction_probe_mode(mode: &str) {
    run_interaction_probe(mode);
}

#[expect(
    clippy::too_many_lines,
    reason = "one probe mode per branch keeps the piped-stdin contracts legible"
)]
fn run_interaction_probe(mode: &str) {
    use pi_ai::auth::types::{AuthEvent, AuthPrompt, AuthPromptKind, AuthPromptOption};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the runtime builds");
    let prompt = |prompt: AuthPrompt| {
        let interaction = cli_interaction();
        let prompt_fn = interaction.prompt;
        runtime.block_on((prompt_fn)(prompt))
    };
    match mode {
        "select" => {
            let options = login_options();
            let selected = prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Select {
                    message: "Pick a login method:".to_owned(),
                    options,
                },
            });
            assert_eq!(
                selected.expect("the select reads"),
                "device_code",
                "the entered number picks the second option's id"
            );
        }
        "question" => {
            let entered = prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Text {
                    message: "Paste the redirect".to_owned(),
                    placeholder: Some("http://localhost:1/cb".to_owned()),
                },
            });
            assert_eq!(
                entered.expect("the question reads"),
                "pasted-url",
                "the entered line answers the prompt"
            );
        }
        "secret" => {
            let entered = prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Secret {
                    message: "Enter key".to_owned(),
                    placeholder: Some("masked".to_owned()),
                },
            });
            assert_eq!(
                entered.expect("the secret reads"),
                "sk-ambient",
                "the entered line answers the secret prompt"
            );
        }
        "manual-code" => {
            let entered = prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::ManualCode {
                    message: "Paste the code".to_owned(),
                    placeholder: Some("http://localhost:1/cb".to_owned()),
                },
            });
            assert_eq!(
                entered.expect("the manual-code reads"),
                "pasted-code",
                "the entered line answers the manual-code prompt"
            );
        }
        "eof" => {
            // The parent's pipe closes without a line; the pending prompt
            // fails as the abort the prompt contract carries.
            let error = prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Text {
                    message: "Pick:".to_owned(),
                    placeholder: None,
                },
            })
            .expect_err("the eof prompt rejects");
            assert_eq!(error, AbortError);
        }
        "select-eof" => {
            let options = vec![AuthPromptOption {
                id: "browser".to_owned(),
                label: "Browser login".to_owned(),
                description: None,
            }];
            let error = prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Select {
                    message: "Pick:".to_owned(),
                    options,
                },
            })
            .expect_err("the eof select rejects");
            assert_eq!(error, AbortError);
        }
        "notify" => {
            let interaction = cli_interaction();
            let notify = interaction.notify;
            notify(AuthEvent::AuthUrl {
                url: "https://authorize.example".to_owned(),
                instructions: Some("Continue in your browser.".to_owned()),
            });
            // An event without instructions skips the instruction line,
            // upstream's optional-instructions switch arm.
            notify(AuthEvent::AuthUrl {
                url: "https://authorize-bare.example".to_owned(),
                instructions: None,
            });
            notify(AuthEvent::DeviceCode {
                user_code: "ABCD-1234".to_owned(),
                verification_uri: "https://verify.example".to_owned(),
                interval_seconds: Some(5),
                expires_in_seconds: Some(600),
            });
            notify(AuthEvent::Info {
                message: "an info message".to_owned(),
                links: Some(vec![pi_ai::auth::types::AuthInfoLink {
                    url: "https://docs.example".to_owned(),
                    label: None,
                }]),
            });
            notify(AuthEvent::Progress {
                message: "a progress message".to_owned(),
            });
        }
        other => panic!("unknown probe mode: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// run(): the picker over piped stdin, invalid input, and the EOF rejection
// ---------------------------------------------------------------------------

#[test]
fn run_login_picker_serves_the_numbered_menu_and_persists_the_choice() {
    if let Ok(mode) = std::env::var(INTERACTIVE_PROBE) {
        assert_eq!(mode, "picker-valid");
        let dir = unique_temp_dir("picker");
        let auth_path = dir.join("auth.json");
        let providers = vec![cli_provider("stub", "Stub Provider")];
        let outcome = block_run(
            &[String::from("login")],
            &providers,
            auth_path.to_str().expect("utf-8 path"),
        );
        assert!(outcome.is_ok(), "the picked login succeeds: {outcome:?}");
        assert!(
            load_credentials(auth_path.to_str().expect("utf-8 path")).contains_key("stub"),
            "the picked provider's credential persisted"
        );
        std::fs::remove_dir_all(&dir).ok();
        return;
    }

    let output = spawn_probe(
        "run_login_picker_serves_the_numbered_menu_and_persists_the_choice",
        INTERACTIVE_PROBE,
        "picker-valid",
        b"1\n",
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the picker probe passes: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("  1. Stub Provider") && stdout.contains("Enter number (1-1): "),
        "the provider menu prints: {stdout:?}"
    );
    assert!(
        stdout.contains("\nCredentials saved to auth.json"),
        "the saved-to line names the file: {stdout:?}"
    );
}

#[test]
fn run_login_picker_rejects_invalid_and_eof_input() {
    if let Ok(mode) = std::env::var(INTERACTIVE_PROBE) {
        let mode: &str = &mode;
        let providers = vec![cli_provider("stub", "Stub Provider")];
        let outcome = block_run(&[String::from("login")], &providers, "unused-auth.json");
        match mode {
            // An out-of-range number selects nothing; the picker's own
            // rejection carries the empty id upstream sends.
            "picker-invalid" => {
                let error = outcome.expect_err("the invalid pick fails");
                assert_eq!(
                    error.to_string(),
                    "",
                    "the invalid pick carries the empty id"
                );
            }
            "picker-eof" => {
                let error = outcome.expect_err("the closed stdin fails the picker");
                assert_eq!(error.to_string(), "Login cancelled");
            }
            other => panic!("unknown probe mode: {other:?}"),
        }
        return;
    }

    for (mode, stdin_bytes) in [("picker-invalid", &b"99\n"[..]), ("picker-eof", b"")] {
        let output = spawn_probe(
            "run_login_picker_rejects_invalid_and_eof_input",
            INTERACTIVE_PROBE,
            mode,
            stdin_bytes,
        );
        assert!(
            output.status.success(),
            "the {mode:?} probe passes: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("  1. Stub Provider") && stdout.contains("Enter number (1-1): "),
            "the menu prints before the failure: {stdout:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// save_credentials: io failure surfaces
// ---------------------------------------------------------------------------

#[test]
fn save_credentials_fails_loudly_when_the_store_cannot_be_written() {
    // A parent path that is a regular file: the create step fails.
    let dir = unique_temp_dir("io-create");
    let blocker = dir.join("blocker");
    std::fs::create_dir_all(&dir).expect("the temp dir creates");
    std::fs::write(&blocker, b"x").expect("the blocker file writes");
    // The parent's parent is a regular file, so create_dir_all cannot make
    // the missing parent.
    let error = save_credentials(
        &blocker.join("child/auth.json"),
        "stub",
        &Credential::ApiKey(ApiKeyCredential {
            key: Some("k".to_owned()),
            env: None,
        }),
    )
    .expect_err("the blocked create fails");
    assert!(
        error.to_string().starts_with("Failed to create "),
        "the create failure names the action and path: {error:?}"
    );

    // A read-only parent: the write step fails.
    let readonly = dir.join("readonly");
    std::fs::create_dir_all(&readonly).expect("the readonly dir creates");
    std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o500))
        .expect("the permissions tighten");
    let error = save_credentials(
        &readonly.join("auth.json"),
        "stub",
        &Credential::ApiKey(ApiKeyCredential {
            key: Some("k".to_owned()),
            env: None,
        }),
    )
    .expect_err("the read-only write fails");
    assert!(
        error.to_string().starts_with("Failed to write "),
        "the write failure names the action and path: {error:?}"
    );
    std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o700))
        .expect("the permissions restore");
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// run(): the login flow and persistence failure surfaces
// ---------------------------------------------------------------------------

#[test]
fn run_login_surfaces_the_flow_and_persistence_failures() {
    if let Ok(mode) = std::env::var(INTERACTIVE_PROBE) {
        assert_eq!(mode, "login-fail");
        let providers = vec![CliProvider {
            id: "stub".to_owned(),
            name: "Stub Provider".to_owned(),
            oauth: StubOAuthAuth::new("Stub OAuth", Err(String::from("flow refused"))).auth(),
        }];
        let outcome = block_run(
            &[String::from("login"), String::from("stub")],
            &providers,
            "unused-auth.json",
        );
        let error = outcome.expect_err("the flow failure surfaces");
        assert_eq!(error.to_string(), "flow refused");

        // A persistence failure under a blocked path surfaces too.
        let dir = unique_temp_dir("login-io");
        let blocker = dir.join("blocker");
        std::fs::create_dir_all(&dir).expect("the temp dir creates");
        std::fs::write(&blocker, b"x").expect("the blocker file writes");
        let providers = vec![cli_provider("stub", "Stub Provider")];
        let outcome = block_run(
            &[String::from("login"), String::from("stub")],
            &providers,
            blocker
                .join("child/auth.json")
                .to_str()
                .expect("utf-8 path"),
        );
        let error = outcome.expect_err("the persistence failure surfaces");
        assert!(
            error.to_string().starts_with("Failed to create "),
            "{error:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
        return;
    }

    let child = std::process::Command::new(std::env::current_exe().expect("the test binary path"))
        .args([
            "--exact",
            "run_login_surfaces_the_flow_and_persistence_failures",
            "--nocapture",
        ])
        .env(INTERACTIVE_PROBE, "login-fail")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the probe child spawns");
    let output = child.wait_with_output().expect("the probe child runs");
    assert!(
        output.status.success(),
        "the login-failure probe passes: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn save_credentials_reports_a_chmod_failure_on_the_store_file() {
    // /dev/null accepts the write but refuses the owner-only chmod (the
    // device node is root-owned), surfacing the store-file permission error.
    let error = save_credentials(
        std::path::Path::new("/dev/null"),
        "stub",
        &Credential::ApiKey(ApiKeyCredential {
            key: Some("k".to_owned()),
            env: None,
        }),
    )
    .expect_err("the chmod failure surfaces");
    assert!(
        error
            .to_string()
            .starts_with("Failed to set permissions on /dev/null"),
        "{error:?}"
    );
}
