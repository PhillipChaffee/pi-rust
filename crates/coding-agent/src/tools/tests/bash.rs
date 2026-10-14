//! The bash tool block of upstream's `tools.test.ts`, the
//! `5208-late-bash-output` regression, and the boundary tests binding the
//! restated surfaces (the spawn-error resolver injection, the stdin
//! transport through `cat`, the spawn-context session environment).

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::types::AgentHarnessToolUpdateCallback;
use pi_agent_core::types::{AgentToolError, AgentToolResult};
use serde_json::json;

use crate::core::bash_executor::execute_bash_with_operations;
use crate::extensions::types::{CwdContext, ExtensionContext};
use crate::tools::bash::{
    BashExecOptions, BashExecOutcome, BashOperations, BashToolOptions, OnDataListener,
    create_bash_tool, create_local_bash_operations, create_local_shell_operations, format_seconds,
    io_error_message, resolve_spawn_context, resolve_timeout_ms,
};
use crate::utils::shell::{CommandTransport, ShellConfig, get_shell_env};

use super::helpers::{block_on, run_tool, text_output};

use std::time::Duration;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The scripted operations, upstream's `operations: { exec: async ... }`:
/// the chunks stream, then the outcome settles.
fn scripted_operations(
    body: impl Fn(&OnDataListener) -> Result<BashExecOutcome, AgentToolError> + Send + Sync + 'static,
) -> BashOperations {
    let body = Arc::new(body);
    BashOperations {
        exec: Arc::new(
            move |_command: &str, _cwd: &str, options: BashExecOptions| {
                let body = Arc::clone(&body);
                Box::pin(async move { body(&options.on_data) })
            },
        ),
    }
}

async fn run_bash(
    tool: &pi_agent_core::harness::types::AgentHarnessTool,
    args: serde_json::Value,
) -> Result<AgentToolResult, AgentToolError> {
    run_tool(tool, args, None, None, &background_context()).await
}

async fn run_bash_with_updates(
    tool: &pi_agent_core::harness::types::AgentHarnessTool,
    args: serde_json::Value,
    on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
) -> Result<AgentToolResult, AgentToolError> {
    run_tool(tool, args, None, on_update, &background_context()).await
}

// ---------------------------------------------------------------------------
// tools.test.ts: bash tool
// ---------------------------------------------------------------------------

#[test]
fn executes_simple_commands() {
    block_on(async {
        let tool = create_bash_tool(".", None);
        let result = run_bash(&tool, json!({ "command": "echo 'test output'" }))
            .await
            .unwrap();
        assert!(text_output(&result).contains("test output"));
        assert_eq!(result.details, serde_json::Value::Null);
    });
}

#[test]
fn handles_command_errors() {
    block_on(async {
        let tool = create_bash_tool(".", None);
        let error = run_bash(&tool, json!({ "command": "exit 1" }))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("code 1"), "{}", error);
    });
}

#[test]
fn respects_timeout() {
    block_on(async {
        let tool = create_bash_tool(".", None);
        let error = run_bash(&tool, json!({ "command": "sleep 5", "timeout": 0.05 }))
            .await
            .unwrap_err();
        assert!(
            error.to_string().to_lowercase().contains("timed out"),
            "{}",
            error
        );
    });
}

#[test]
fn includes_full_output_path_for_truncated_timeout_and_abort_errors() {
    block_on(async {
        for (sentinel, expected) in [
            ("timeout:5", "Command timed out after 5 seconds"),
            ("aborted", "Command aborted"),
        ] {
            let operations = scripted_operations(move |on_data: &OnDataListener| {
                for i in 1..=3000usize {
                    on_data(format!("{i}\n").as_bytes());
                }
                Err(AgentToolError::from(std::io::Error::other(sentinel)))
            });
            let tool = create_bash_tool(
                "/tmp",
                Some(BashToolOptions {
                    operations: Some(operations),
                    ..BashToolOptions::default()
                }),
            );
            let error = run_bash(&tool, json!({ "command": "chatty-fail" }))
                .await
                .unwrap_err();
            let message = error.to_string();
            assert!(message.contains(expected), "{sentinel}: {message}");
            assert!(
                message.contains("[Showing lines 1001-3000 of 3000. Full output: "),
                "{sentinel}: {message}"
            );
            let path = message
                .split("Full output: ")
                .nth(1)
                .unwrap()
                .split(']')
                .next()
                .unwrap()
                .to_owned();
            assert!(std::path::Path::new(&path).exists(), "{sentinel}: {path}");
            let full_output = std::fs::read_to_string(&path).unwrap();
            assert!(full_output.contains("1\n2\n3"), "{sentinel}");
            assert!(full_output.contains("2998\n2999\n3000"), "{sentinel}");
            let _ = std::fs::remove_file(&path);
        }
    });
}

#[test]
fn throws_error_when_cwd_does_not_exist() {
    block_on(async {
        let tool = create_bash_tool("/this/directory/definitely/does/not/exist/12345", None);
        let error = run_bash(&tool, json!({ "command": "echo test" }))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Working directory does not exist"),
            "{}",
            error
        );
    });
}

#[test]
fn handles_process_spawn_errors() {
    block_on(async {
        // Upstream mocks `getShellConfig`; the resolver parameter is the
        // seam the injection rides.
        let operations = create_local_shell_operations("bash", || {
            Ok(ShellConfig {
                shell: "/nonexistent-shell-path-xyz123".to_owned(),
                args: vec!["-c".to_owned()],
                command_transport: None,
            })
        });
        let tool = create_bash_tool(
            "/tmp",
            Some(BashToolOptions {
                operations: Some(operations),
                ..BashToolOptions::default()
            }),
        );
        let error = run_bash(&tool, json!({ "command": "echo test" }))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("ENOENT"), "{}", error);
    });
}

#[test]
fn custom_shell_path_that_does_not_exist_errors() {
    block_on(async {
        // Upstream's `should pass shellPath through to shell resolution`:
        // the custom-path resolution error is the observable half; the
        // operations-override-skips-resolution half rides the `operations`
        // parameter's short-circuit.
        let operations = create_local_bash_operations(Some("/custom/bash"));
        let error = (operations.exec)(
            "echo test",
            "/tmp",
            BashExecOptions {
                on_data: Arc::new(|_data: &[u8]| {}),
                signal: None,
                timeout: None,
                env: None,
            },
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Custom shell path not found: /custom/bash"
        );
    });
}

#[test]
fn sends_commands_over_stdin_when_shell_resolution_requires_it() {
    block_on(async {
        // Upstream drives node reading its stdin; `cat` is the restated
        // echo-back body.
        let operations = create_local_shell_operations("bash", || {
            Ok(ShellConfig {
                shell: "/bin/cat".to_owned(),
                args: Vec::new(),
                command_transport: Some(CommandTransport::Stdin),
            })
        });
        let chunks = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let sink = Arc::clone(&chunks);
        let command = "name='World'; echo \"Hello, ${name}!\"; count=3; for i in $(seq 1 ${count}); do echo \"Iteration ${i} of ${count}\"; done";
        let settled = (operations.exec)(
            command,
            "/tmp",
            BashExecOptions {
                on_data: Arc::new(move |data: &[u8]| {
                    sink.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .extend_from_slice(data);
                }),
                signal: None,
                timeout: None,
                env: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(settled.exit_code, Some(0));
        let collected = chunks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(String::from_utf8_lossy(&collected), command);
    });
}

#[test]
fn prepends_command_prefix_when_configured() {
    block_on(async {
        let tool = create_bash_tool(
            "/tmp",
            Some(BashToolOptions {
                command_prefix: Some("export TEST_VAR=hello".to_owned()),
                ..BashToolOptions::default()
            }),
        );
        let result = run_bash(&tool, json!({ "command": "echo $TEST_VAR" }))
            .await
            .unwrap();
        assert_eq!(text_output(&result).trim(), "hello");
    });
}

#[test]
fn includes_output_from_both_prefix_and_command() {
    block_on(async {
        let tool = create_bash_tool(
            "/tmp",
            Some(BashToolOptions {
                command_prefix: Some("echo prefix-output".to_owned()),
                ..BashToolOptions::default()
            }),
        );
        let result = run_bash(&tool, json!({ "command": "echo command-output" }))
            .await
            .unwrap();
        assert_eq!(text_output(&result).trim(), "prefix-output\ncommand-output");
    });
}

#[test]
fn works_without_command_prefix() {
    block_on(async {
        let tool = create_bash_tool("/tmp", Some(BashToolOptions::default()));
        let result = run_bash(&tool, json!({ "command": "echo no-prefix" }))
            .await
            .unwrap();
        assert_eq!(text_output(&result).trim(), "no-prefix");
    });
}

#[test]
fn coalesces_streaming_updates_for_chatty_output() {
    block_on(async {
        let operations = scripted_operations(|on_data: &OnDataListener| {
            for i in 0..5000usize {
                on_data(format!("line {i}\n").as_bytes());
            }
            Ok(BashExecOutcome { exit_code: Some(0) })
        });
        let updates = Arc::new(std::sync::Mutex::new(Vec::<AgentToolResult>::new()));
        let sink = Arc::clone(&updates);
        let tool = create_bash_tool(
            "/tmp",
            Some(BashToolOptions {
                operations: Some(operations),
                ..BashToolOptions::default()
            }),
        );
        let on_update: AgentHarnessToolUpdateCallback<'_> =
            &|update: &AgentToolResult, _options| {
                sink.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(update.clone());
            };
        let result = run_bash_with_updates(&tool, json!({ "command": "chatty" }), Some(on_update))
            .await
            .unwrap();
        let collected = updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        assert!(collected < 25, "{collected} updates");
        assert!(text_output(&result).contains("line 4999"));
    });
}

#[test]
fn trailing_newline_is_not_an_extra_truncated_bash_output_line() {
    block_on(async {
        let lines: Vec<String> = (1..=4000usize).map(|i| format!("line-{i:0>4}")).collect();
        let operations = scripted_operations(move |on_data: &OnDataListener| {
            for line in &lines {
                on_data(format!("{line}\n").as_bytes());
            }
            Ok(BashExecOutcome { exit_code: Some(0) })
        });
        let tool = create_bash_tool(
            "/tmp",
            Some(BashToolOptions {
                operations: Some(operations),
                ..BashToolOptions::default()
            }),
        );
        let result = run_bash(&tool, json!({ "command": "many-lines" }))
            .await
            .unwrap();
        let details = &result.details;
        assert_eq!(details["truncation"]["totalLines"], 4000);
        assert_eq!(details["truncation"]["outputLines"], 2000);
        let output = text_output(&result);
        assert!(output.contains("line-2001"));
        assert!(output.contains("line-4000"));
        assert!(output.contains("[Showing lines 2001-4000 of 4000. Full output: "));
        assert!(!output.contains("4001"));
    });
}

#[test]
fn decodes_utf8_characters_split_across_output_chunks() {
    block_on(async {
        let euro = "€\n".as_bytes().to_vec();
        let operations = scripted_operations(move |on_data: &OnDataListener| {
            on_data(&euro[..1]);
            on_data(&euro[1..]);
            Ok(BashExecOutcome { exit_code: Some(0) })
        });
        let tool = create_bash_tool(
            "/tmp",
            Some(BashToolOptions {
                operations: Some(operations),
                ..BashToolOptions::default()
            }),
        );
        let result = run_bash(&tool, json!({ "command": "split-utf8" }))
            .await
            .unwrap();
        assert_eq!(text_output(&result).trim(), "€");
    });
}

#[test]
fn exposes_local_bash_operations_for_extension_reuse() {
    block_on(async {
        let operations = create_local_bash_operations(None);
        let chunks = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let sink = Arc::clone(&chunks);
        let mut env = get_shell_env();
        env.insert(
            "TEST_LOCAL_BASH_OPS".to_owned(),
            "from-local-ops".to_owned(),
        );
        let settled = (operations.exec)(
            "echo $TEST_LOCAL_BASH_OPS",
            "/tmp",
            BashExecOptions {
                on_data: Arc::new(move |data: &[u8]| {
                    sink.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .extend_from_slice(data);
                }),
                signal: None,
                timeout: None,
                env: Some(env),
            },
        )
        .await
        .unwrap();
        assert_eq!(settled.exit_code, Some(0));
        let collected = String::from_utf8_lossy(
            &chunks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
        .into_owned();
        assert_eq!(collected.trim(), "from-local-ops");
    });
}

#[test]
fn preserves_execute_bash_sanitization_when_using_local_bash_operations() {
    block_on(async {
        let result = execute_bash_with_operations(
            "printf '\\033[31mred\\033[0m\\r\\n'",
            ".",
            &create_local_bash_operations(None),
            None,
        )
        .await
        .unwrap();
        assert_eq!(result.exit_code, Some(0));
        assert_eq!(result.output, "red\n");
    });
}

#[test]
fn persists_full_output_when_truncation_happens_by_line_count_only() {
    block_on(async {
        let tool = create_bash_tool("/tmp", None);
        let result = run_bash(&tool, json!({ "command": "printf '%s\\n' {1..3000}" }))
            .await
            .unwrap();
        let output = text_output(&result);
        assert_eq!(result.details["truncation"]["truncated"], true);
        assert_eq!(result.details["truncation"]["truncatedBy"], "lines");
        let path = result.details["fullOutputPath"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(output.contains("[Showing lines 1001-3000 of 3000. Full output: "));
        assert!(std::path::Path::new(&path).exists());
        let full_output = std::fs::read_to_string(&path).unwrap();
        assert!(full_output.contains("1\n2\n3"));
        assert!(full_output.contains("2998\n2999\n3000"));
        let _ = std::fs::remove_file(&path);
    });
}

#[test]
fn execute_bash_persists_full_output_when_truncation_happens_by_line_count_only() {
    block_on(async {
        let result = execute_bash_with_operations(
            "printf '%s\\n' {1..3000}",
            ".",
            &create_local_bash_operations(None),
            None,
        )
        .await
        .unwrap();
        assert!(result.truncated);
        let path = result.full_output_path.expect("persisted");
        assert!(std::path::Path::new(&path).exists());
        let full_output = std::fs::read_to_string(&path).unwrap();
        assert!(full_output.contains("1\n2\n3"));
        assert!(full_output.contains("2998\n2999\n3000"));
        let _ = std::fs::remove_file(&path);
    });
}

// ---------------------------------------------------------------------------
// Regression 5208: late bash output callbacks
// ---------------------------------------------------------------------------

#[test]
fn ignores_output_callbacks_after_bash_operations_resolve() {
    block_on(async {
        // The late callback: the operations stash the shared listener and
        // the test fires it after the tool has settled, upstream's
        // `setTimeout(() => onData(...), 0)`.
        let stored: Arc<std::sync::Mutex<Option<OnDataListener>>> =
            Arc::new(std::sync::Mutex::new(None));
        let slot = Arc::clone(&stored);
        let operations = scripted_operations(move |on_data: &OnDataListener| {
            on_data(b"before\n");
            *slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(on_data));
            Ok(BashExecOutcome { exit_code: Some(0) })
        });
        let tool = create_bash_tool(
            ".",
            Some(BashToolOptions {
                operations: Some(operations),
                ..BashToolOptions::default()
            }),
        );
        let result = run_bash(&tool, json!({ "command": "late-output" }))
            .await
            .unwrap();
        let late = stored
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("the listener was stashed");
        late(b"late\n");
        assert_eq!(text_output(&result).trim(), "before");
    });
}

// ---------------------------------------------------------------------------
// Spawn-context boundary tests
// ---------------------------------------------------------------------------

/// The session-carrying context stub, upstream's `ExtensionContext` with
/// the session manager populated.
struct SessionContext {
    session_id: String,
    session_file: Option<String>,
}

impl ExtensionContext for SessionContext {
    fn cwd(&self) -> &'static str {
        "/workspace"
    }

    fn model(&self) -> Option<&pi_ai::types::Model> {
        None
    }

    fn thinking_level(&self) -> Option<pi_agent_core::types::ThinkingLevel> {
        None
    }

    fn session_id(&self) -> Option<String> {
        Some(self.session_id.clone())
    }

    fn session_file(&self) -> Option<String> {
        self.session_file.clone()
    }
}

#[test]
fn the_spawn_context_re_adds_the_session_variables() {
    // The clearing of ambient `PI_*` values rides the real process
    // environment (mutating it is unsafe in this crate), so the re-adds pin
    // through the context seam: with exposure on, only the context's own
    // values land; with exposure off, none do.
    let context = SessionContext {
        session_id: "sess-1".to_owned(),
        session_file: Some("/tmp/sess.jsonl".to_owned()),
    };
    let spawn = resolve_spawn_context("echo", "/tmp", None, true, Some(&context));
    assert_eq!(
        spawn.env.get("PI_SESSION_ID").map(String::as_str),
        Some("sess-1")
    );
    assert_eq!(
        spawn.env.get("PI_SESSION_FILE").map(String::as_str),
        Some("/tmp/sess.jsonl")
    );
    assert!(
        !spawn.env.contains_key("PI_MODEL"),
        "no model in the stub context"
    );

    let spawn = resolve_spawn_context("echo", "/tmp", None, false, Some(&context));
    assert!(!spawn.env.contains_key("PI_SESSION_ID"));
    assert!(!spawn.env.contains_key("PI_SESSION_FILE"));
}

#[test]
fn the_spawn_context_applies_the_hook() {
    let hook: crate::tools::bash::BashSpawnHook = Arc::new(|mut context| {
        context.command = format!("wrapped {}", context.command);
        context
    });
    let spawn = resolve_spawn_context("echo hi", "/tmp", Some(&hook), false, None);
    assert_eq!(spawn.command, "wrapped echo hi");
}

#[test]
fn the_default_context_is_the_minimal_cwd_shape() {
    let context = CwdContext {
        cwd: "/tmp".to_owned(),
    };
    assert_eq!(context.cwd(), "/tmp");
    assert!(context.model().is_none());
    assert!(context.thinking_level().is_none());
    assert!(context.session_id().is_none());
    assert!(context.session_file().is_none());
}

#[test]
fn the_timeout_resolution_arms_reject_invalid_input() {
    assert_eq!(resolve_timeout_ms(None).unwrap(), None);
    assert_eq!(
        resolve_timeout_ms(Some(2.5)).unwrap(),
        Some(Duration::from_millis(2500))
    );
    for invalid in [Some(-1.0), Some(0.0), Some(f64::NAN), Some(f64::INFINITY)] {
        assert!(resolve_timeout_ms(invalid).is_err(), "{invalid:?}");
    }
    assert!(resolve_timeout_ms(Some(2_147_483_648.0)).is_err());
}

#[test]
fn the_js_number_rendering_matches_the_string_conversion() {
    assert_eq!(format_seconds(5.0), "5");
    assert_eq!(format_seconds(2.5), "2.5");
    assert_eq!(format_seconds(-0.5), "-0.5");
}

#[test]
fn the_io_error_message_maps_the_node_style_codes() {
    let cases = [
        (std::io::ErrorKind::NotFound, "ENOENT"),
        (std::io::ErrorKind::PermissionDenied, "EACCES"),
        (std::io::ErrorKind::AlreadyExists, "EEXIST"),
        (std::io::ErrorKind::DirectoryNotEmpty, "ENOTEMPTY"),
        (std::io::ErrorKind::NotADirectory, "ENOTDIR"),
        (std::io::ErrorKind::IsADirectory, "EISDIR"),
    ];
    for (kind, code) in cases {
        let error = std::io::Error::new(kind, "boom");
        assert_eq!(io_error_message(&error), code);
    }
    let plain = std::io::Error::other("boom");
    assert_eq!(io_error_message(&plain), plain.to_string());
}

#[test]
fn spaced_updates_emit_immediately_after_the_throttle_window() {
    block_on(async {
        let updates = Arc::new(std::sync::Mutex::new(Vec::<AgentToolResult>::new()));
        let sink = Arc::clone(&updates);
        let operations = scripted_operations(move |on_data: &OnDataListener| {
            on_data(b"first\n");
            std::thread::sleep(Duration::from_millis(150));
            on_data(b"second\n");
            Ok(BashExecOutcome { exit_code: Some(0) })
        });
        let tool = create_bash_tool(
            "/tmp",
            Some(BashToolOptions {
                operations: Some(operations),
                ..BashToolOptions::default()
            }),
        );
        let on_update: AgentHarnessToolUpdateCallback<'_> =
            &|update: &AgentToolResult, _options| {
                sink.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(update.clone());
            };
        run_bash_with_updates(&tool, json!({ "command": "spaced" }), Some(on_update))
            .await
            .unwrap();
        let collected = updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        assert!(collected >= 2, "{collected} updates");
    });
}
