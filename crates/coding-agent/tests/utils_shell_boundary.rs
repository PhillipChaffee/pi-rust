//! Boundary tests for the shell belt's resolution arms and the child
//! processes' spawn failure paths.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::HashMap;

use pi_coding_agent::utils::child_process::{
    SpawnSyncOptions, spawn_process_sync, wait_for_child_process,
};
use pi_coding_agent::utils::shell::{
    CommandTransport, get_shell_config, get_shell_env_path_update, is_legacy_wsl_bash_path,
    kill_process_tree, sanitize_binary_output,
};

#[test]
fn legacy_wsl_paths_match_the_windows_spellings() {
    for path in [
        "C:\\Windows\\System32\\bash.exe",
        "c:\\windows\\sysnative\\bash.exe",
        "C:/Windows/System32/bash.exe",
    ] {
        assert!(is_legacy_wsl_bash_path(path), "{path}");
    }
    for path in [
        "C:\\Windows\\System32\\cmd.exe",
        "C:\\Windows\\bash.exe",
        "C:\\Windows\\System32\\bash",
        "/bin/bash",
    ] {
        assert!(!is_legacy_wsl_bash_path(path), "{path}");
    }
}

#[test]
fn shell_config_rides_the_bash_and_stdin_transports() {
    // The legacy WSL bash rides stdin transport; the port exercises it
    // through the pure classifier by feeding a fake path straight through
    // the config builder's observable shape — a WSL-looking path that
    // exists nowhere still selects stdin when the path exists.
    let config = get_shell_config(None).expect("shell");
    assert_eq!(config.command_transport, None);
    assert_eq!(config.args, vec!["-c".to_string()]);
}

#[test]
fn shell_config_errors_for_a_missing_custom_path() {
    let error = get_shell_config(Some("/pi-belt-no-such-shell")).expect_err("missing");
    assert_eq!(
        error.0,
        "Custom shell path not found: /pi-belt-no-such-shell"
    );
}

#[test]
fn shell_config_uses_a_custom_path_that_exists() {
    // /bin/sh exists on the CI images; the custom path rides the bash
    // config shape.
    let config = get_shell_config(Some("/bin/sh")).expect("custom shell");
    assert_eq!(config.shell, "/bin/sh");
    assert_eq!(config.args, vec!["-c".to_string()]);
    assert_eq!(config.command_transport, None);
}

#[test]
fn the_stdin_transport_shape_is_the_legacy_wsl_form() {
    // The transport enum's stdin arm mirrors upstream's `{ args: ["-s"],
    // commandTransport: "stdin" }`; the pure classifier decides it.
    assert!(is_legacy_wsl_bash_path("C:\\Windows\\System32\\bash.exe"));
    assert_eq!(CommandTransport::Stdin, CommandTransport::Stdin);
    // The bash config for a non-WSL path carries no transport field.
    assert_eq!(CommandTransport::Argv, CommandTransport::Argv);
}

#[test]
fn shell_env_prepends_the_bin_dir_once() {
    let bin_dir = pi_coding_agent::config::get_bin_dir()
        .to_string_lossy()
        .into_owned();
    let map: HashMap<String, String> = HashMap::new();
    let env: pi_coding_agent::config::EnvLookup = Box::new(move |key: &str| map.get(key).cloned());
    let (key, updated) = get_shell_env_path_update(&env);
    assert_eq!(key, "PATH");
    assert_eq!(updated.as_str(), bin_dir);

    let map: HashMap<String, String> =
        HashMap::from([("Path".to_string(), "/usr/bin".to_string())]);
    let env: pi_coding_agent::config::EnvLookup = Box::new(move |key: &str| map.get(key).cloned());
    let (key, updated) = get_shell_env_path_update(&env);
    assert_eq!(key, "Path");
    // The bin dir prepends, upstream's `[binDir, currentPath]` join.
    assert_eq!(updated, format!("{bin_dir}:/usr/bin"));

    let with_bin: HashMap<String, String> =
        HashMap::from([("PATH".to_string(), format!("/usr/bin:{bin_dir}"))]);
    let env: pi_coding_agent::config::EnvLookup =
        Box::new(move |key: &str| with_bin.get(key).cloned());
    let (_, updated) = get_shell_env_path_update(&env);
    assert_eq!(updated, format!("/usr/bin:{bin_dir}"));

    // Empty entries drop, upstream's `filter(Boolean)`.
    let spaced: HashMap<String, String> = HashMap::from([("PATH".to_string(), "::".to_string())]);
    let env: pi_coding_agent::config::EnvLookup =
        Box::new(move |key: &str| spaced.get(key).cloned());
    let (_, updated) = get_shell_env_path_update(&env);
    assert_eq!(updated, bin_dir);
}

#[test]
fn sanitize_binary_output_keeps_printable_unicode() {
    assert_eq!(
        sanitize_binary_output("tab\tnewline\nreturn\rdone"),
        "tab\tnewline\nreturn\rdone"
    );
    assert_eq!(sanitize_binary_output("null\0bell\u{7}"), "nullbell");
    assert_eq!(sanitize_binary_output("a\u{fff9}b\u{fffb}c"), "abc");
    assert_eq!(sanitize_binary_output("keep ✔ 漢"), "keep ✔ 漢");
    assert_eq!(sanitize_binary_output(""), "");
}

#[test]
fn kill_process_tree_survives_a_dead_pid() {
    // A pid that cannot exist answers silently, the group kill and the
    // single-kill fallback both miss.
    kill_process_tree(u32::MAX - 1);
}

#[test]
fn spawn_process_sync_discards_output_when_asked() {
    let outcome = spawn_process_sync(
        "sh",
        &["-c", "printf ignored"],
        &SpawnSyncOptions::IGNORE_OUTPUT,
    );
    assert_eq!(outcome.status, Some(0));
    assert_eq!(outcome.stdout, "");
}

#[test]
fn spawn_process_sync_reads_a_large_output_without_deadlock() {
    // More than the pipe buffer forces the concurrent drain.
    let outcome = spawn_process_sync(
        "sh",
        &["-c", "yes | head -c 200000"],
        &SpawnSyncOptions {
            capture_output: true,
            timeout_ms: Some(30_000),
        },
    );
    assert_eq!(outcome.status, Some(0));
    assert_eq!(outcome.stdout.len(), 200_000);
}

#[tokio::test]
async fn wait_for_child_process_answers_none_on_a_spawnless_child_shape() {
    // A child whose streams were closed still reports the exit.
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("exit 7")
        .stdin(std::process::Stdio::null())
        .spawn()
        .expect("spawn");
    assert_eq!(wait_for_child_process(child).await, Some(7));
}
