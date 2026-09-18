//! CLI integration tests: exercise the binary end-to-end.

use std::process::Command;

/// The binary builds and exits successfully; this is also what puts the
/// entry point under the coverage runtime, since unit tests never run main.
#[test]
fn binary_exits_successfully() {
    let out = Command::new(env!("CARGO_BIN_EXE_pi-rust")).output();
    assert!(out.as_ref().is_ok_and(|o| o.status.success()));
}
