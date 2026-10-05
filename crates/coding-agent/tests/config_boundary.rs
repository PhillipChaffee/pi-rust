//! Boundary tests for the config foundation: the agent-dir derivation and
//! its on-disk layout, the app and install constants, the thinking defaults,
//! the session-dir cwd encoding, and the resolve-backed session-dir
//! derivation. Upstream exercises these through its own consumers; the
//! vectors here pin the byte-for-byte contract the port carries.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::path::{Path, PathBuf};

use common::{empty_env, env_with};
use pi_agent_core::types::ThinkingLevel;
use pi_coding_agent::config::{
    APP_NAME, APP_TITLE, CONFIG_DIR_NAME, IS_BUN_BINARY, IS_BUN_RUNTIME, IS_BUNDLED_NODE,
    PACKAGE_NAME, VERSION, default_session_dir_path, encode_session_cwd, get_agent_dir_with,
    get_auth_path, get_bin_dir, get_custom_themes_dir, get_debug_log_path, get_models_path,
    get_prompts_dir, get_sessions_dir, get_settings_path, get_tools_dir,
};
use pi_coding_agent::defaults::{DEFAULT_THINKING_LEVEL, THINKING_LEVEL_OPTIONS};

fn process_home() -> String {
    std::env::home_dir()
        .expect("home directory resolves")
        .to_string_lossy()
        .into_owned()
}

fn process_cwd() -> String {
    std::env::current_dir()
        .expect("current directory resolves")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn agent_dir_defaults_to_the_home_layout() {
    let home = process_home();

    assert_eq!(
        get_agent_dir_with(&empty_env()),
        Path::new(&home).join(CONFIG_DIR_NAME).join("agent")
    );
}

#[test]
fn agent_dir_prefers_the_env_override() {
    let env = env_with(&[("PI_CODING_AGENT_DIR", "/custom/agent")]);

    assert_eq!(get_agent_dir_with(&env), PathBuf::from("/custom/agent"));
}

#[test]
fn agent_dir_override_expands_a_leading_tilde() {
    let env = env_with(&[("PI_CODING_AGENT_DIR", "~/agent")]);
    let home = process_home();

    assert_eq!(get_agent_dir_with(&env), PathBuf::from(home).join("agent"));
}

#[test]
fn agent_dir_bare_tilde_override_is_the_home_itself() {
    let env = env_with(&[("PI_CODING_AGENT_DIR", "~")]);
    let home = process_home();

    assert_eq!(get_agent_dir_with(&env), PathBuf::from(home));
}

#[test]
fn agent_dir_override_keeps_a_relative_value_relative() {
    // upstream's expandTildePath is normalizePath, not resolvePath: only the
    // tilde slice runs, so a relative override stays relative
    let env = env_with(&[("PI_CODING_AGENT_DIR", "rel/agent")]);

    assert_eq!(get_agent_dir_with(&env), PathBuf::from("rel/agent"));
}

#[test]
fn the_layout_paths_hang_off_the_agent_dir() {
    let agent = get_agent_dir_with(&empty_env());

    assert_eq!(get_custom_themes_dir(), agent.join("themes"));
    assert_eq!(get_models_path(), agent.join("models.json"));
    assert_eq!(get_auth_path(), agent.join("auth.json"));
    assert_eq!(get_settings_path(), agent.join("settings.json"));
    assert_eq!(get_tools_dir(), agent.join("tools"));
    assert_eq!(get_bin_dir(), agent.join("bin"));
    assert_eq!(get_prompts_dir(), agent.join("prompts"));
    assert_eq!(get_sessions_dir(), agent.join("sessions"));
    assert_eq!(get_debug_log_path(), agent.join("pi-debug.log"));
}

#[test]
fn the_install_probes_restate_as_false_constants() {
    // bound through locals so the assertions stay runtime checks; the
    // constants are compile-time values the assertion would const-fold
    let (bun_binary, bun_runtime, bundled_node) = (IS_BUN_BINARY, IS_BUN_RUNTIME, IS_BUNDLED_NODE);

    assert!(!bun_binary);
    assert!(!bun_runtime);
    assert!(!bundled_node);
}

#[test]
fn the_app_constants_match_upstreams_package_json_values() {
    assert_eq!(PACKAGE_NAME, env!("CARGO_PKG_NAME"));
    assert_eq!(PACKAGE_NAME, "pi-coding-agent");
    assert_eq!(APP_NAME, "pi");
    assert_eq!(APP_TITLE, "π");
    assert_eq!(CONFIG_DIR_NAME, ".pi");
    assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
}

#[test]
fn the_defaults_match_upstreams_thinking_levels() {
    assert_eq!(DEFAULT_THINKING_LEVEL, ThinkingLevel::Medium);
    assert_eq!(
        THINKING_LEVEL_OPTIONS,
        [
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::Xhigh,
            ThinkingLevel::Max,
        ]
    );
}

// =============================================================================
// encode_session_cwd — byte-for-byte per the survey
// =============================================================================

#[test]
fn encode_strips_one_leading_separator_then_replaces() {
    // the strip-then-replace order matters: a leading slash disappears
    // instead of becoming a dash
    assert_eq!(encode_session_cwd("/Users/foo/bar"), "--Users-foo-bar--");
    assert_eq!(encode_session_cwd("/tmp/a b"), "--tmp-a b--");
}

#[test]
fn encode_collapses_separators_and_colons() {
    // the inputs are resolved paths — resolution itself collapses repeated
    // separators (pinned in paths_boundary); the encoder replaces the rest
    assert_eq!(encode_session_cwd("/a/b/c"), "--a-b-c--");
    assert_eq!(encode_session_cwd("/C:\\Users"), "--C--Users--");
    assert_eq!(encode_session_cwd("a:b/c"), "--a-b-c--");
}

#[test]
fn encode_root_and_colon_root_degenerate_to_dashes() {
    // "/" strips to "" → "----"; "/:" strips to ":" → "-----"
    assert_eq!(encode_session_cwd("/"), "----");
    assert_eq!(encode_session_cwd("/:"), "-----");
}

#[test]
fn encode_strips_a_leading_backslash_like_upstream() {
    // reachable only for strings handed to the encoder directly; a resolved
    // POSIX path never starts with a backslash
    assert_eq!(encode_session_cwd("\\Users"), "--Users--");
    assert_eq!(encode_session_cwd("//a"), "---a--");
}

#[test]
fn encode_preserves_non_ascii_and_other_bytes() {
    assert_eq!(
        encode_session_cwd("/Users/日本語/dir"),
        "--Users-日本語-dir--"
    );
    assert_eq!(encode_session_cwd("/a/.../b"), "--a-...-b--");
}

// =============================================================================
// default_session_dir_path — resolve, encode, join
// =============================================================================

#[test]
fn session_dir_joins_the_encoded_cwd_under_sessions() {
    assert_eq!(
        default_session_dir_path("/Users/foo/bar", "/agent"),
        Path::new("/agent")
            .join("sessions")
            .join("--Users-foo-bar--")
    );
}

#[test]
fn session_dir_encodes_the_root_cwd() {
    assert_eq!(
        default_session_dir_path("/", "/agent"),
        Path::new("/agent").join("sessions").join("----")
    );
    assert_eq!(
        default_session_dir_path("/:", "/agent"),
        Path::new("/agent").join("sessions").join("-----")
    );
}

#[test]
fn session_dir_resolves_a_relative_cwd_and_agent_dir() {
    let cwd = process_cwd();
    let encoded = encode_session_cwd(&format!("{cwd}/a/b"));

    assert_eq!(
        default_session_dir_path("a/b", "agent-x"),
        Path::new(&cwd)
            .join("agent-x")
            .join("sessions")
            .join(encoded)
    );
}

#[test]
fn session_dir_expands_a_tilde_cwd_and_agent_dir() {
    let home = process_home();
    let home_encoded = encode_session_cwd(&home);

    assert_eq!(
        default_session_dir_path("~", "~"),
        Path::new(&home).join("sessions").join(home_encoded)
    );
}

#[test]
fn session_dir_dot_inputs_collapse_to_the_process_cwd() {
    let cwd = process_cwd();
    let encoded = encode_session_cwd(&cwd);

    assert_eq!(
        default_session_dir_path(".", "/agent"),
        Path::new("/agent").join("sessions").join(&encoded)
    );
    assert_eq!(
        default_session_dir_path("", "/agent"),
        Path::new("/agent").join("sessions").join(encoded)
    );
}
