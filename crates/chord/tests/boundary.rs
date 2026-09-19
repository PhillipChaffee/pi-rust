//! The package boundary test, ported from upstream `test/boundary.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream structurally pins that chord imports no other Pi package and
//! ships no files outside its source tree. The port carries the manifest
//! half verbatim: the crate declares no `@earendil-works/pi-*` dependency,
//! and the lib.rs documentation states the same guarantee.

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]

use std::path::Path;

#[test]
fn does_not_depend_on_pi_packages_or_files_outside_chord() {
    let manifest = std::fs::read_to_string(manifest_path())
        .unwrap_or_else(|error| panic!("manifest: {error}"));
    // The `[dependencies]` table carries no Pi workspace package; the
    // upstream test reads package.json's dependency map, the manifest's
    // TOML table is its restatement.
    let in_dependencies = manifest.split("[dependencies]").nth(1).unwrap_or_default();
    let dependency_table = in_dependencies
        .split("[dev-dependencies]")
        .next()
        .unwrap_or_default();
    let violations: Vec<String> = dependency_table
        .lines()
        .filter(|line| line.trim_start().starts_with("@earendil-works/pi-"))
        .map(ToString::to_string)
        .collect();
    assert!(violations.is_empty(), "boundary violation: {violations:?}");
    assert!(
        crate_source_dir().join("lib.rs").exists(),
        "the crate source lives under its own source tree"
    );
}

fn manifest_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

fn crate_source_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}
