//! Upstream `test/config.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! The suite splits at the port boundary. The `findNodePackageDir` case
//! ports against the build-time restatement: the Rust binary has no
//! package.json walk and no bun/dist layouts, so `get_package_dir` is the
//! `PI_PACKAGE_DIR` override over the executable directory — the layout
//! upstream's Bun-binary branch produced, and the case's dist-metadata skip
//! has no counterpart. The npm/pnpm/yarn/bun self-update cases probe
//! machinery that rides its own ticket (the package-manager/self-update
//! port) and have no ported counterpart here; their boundary behavior is
//! re-bound when that machinery lands.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::path::{Path, PathBuf};

use common::{empty_env, env_with};
use pi_coding_agent::config::{
    ENV_AGENT_DIR, ENV_SESSION_DIR, get_agent_dir, get_agent_dir_with, get_package_dir,
    get_package_dir_with,
};

#[test]
fn the_pi_bin_target_runs_and_exits_cleanly() {
    // the scaffold bin ships the provisional `pi` name; it exits 0 doing
    // nothing until the CLI dispatch lands
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_pi"))
        .status()
        .expect("the pi bin target runs");

    assert!(status.success());
}

#[test]
fn the_process_wrappers_read_the_real_environment() {
    // the plain getters are the process-env default over the injected
    // variants; an environment carrying neither override makes them agree
    assert_eq!(get_agent_dir(), get_agent_dir_with(&empty_env()));
    assert_eq!(get_package_dir(), get_package_dir_with(&empty_env()));
}

#[test]
fn package_dir_prefers_the_pi_package_dir_override() {
    let env = env_with(&[("PI_PACKAGE_DIR", "/custom/pkg")]);

    assert_eq!(get_package_dir_with(&env), PathBuf::from("/custom/pkg"));
}

#[test]
fn package_dir_override_expands_a_leading_tilde() {
    let env = env_with(&[("PI_PACKAGE_DIR", "~/pkg")]);
    let home = std::env::home_dir().expect("home directory resolves");

    assert_eq!(get_package_dir_with(&env), home.join("pkg"));
}

#[test]
fn package_dir_falls_back_to_the_executable_directory() {
    let exe = std::env::current_exe().expect("test executable resolves");
    let dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();

    assert_eq!(get_package_dir_with(&empty_env()), dir);
}

#[test]
fn the_agent_dir_variable_names_match_upstreams_derivation() {
    // upstream derives both from `${APP_NAME.toUpperCase()}_CODING_AGENT_DIR`
    assert_eq!(ENV_AGENT_DIR, "PI_CODING_AGENT_DIR");
    assert_eq!(ENV_SESSION_DIR, "PI_CODING_AGENT_SESSION_DIR");
}
