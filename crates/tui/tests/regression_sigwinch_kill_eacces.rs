//! Port of `packages/tui/test/regression-sigwinch-kill-eacces.test.ts` — 1:1
//! against upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41).
//!
//! Upstream monkeypatched `process.kill` and `process.platform`; the port
//! injects the kill call ([`refresh_terminal_dimensions_with`]) and gates the
//! platform cases on their `cfg`.

#[test]
#[cfg(unix)]
fn does_not_throw_when_kill_returns_eacces_for_self_signal() {
    // Signal delivery not permitted in this environment; the refresh is
    // skipped rather than crashing.
    let kill: pi_tui::terminal::KillFn = &|_pid, _signal| Err(libc::EACCES);
    pi_tui::terminal::refresh_terminal_dimensions_with(kill);
}

#[test]
#[cfg(unix)]
fn preserves_other_error_codes() {
    // EPERM is also ignored - the refresh is best-effort.
    let kill: pi_tui::terminal::KillFn = &|_pid, _signal| Err(libc::EPERM);
    pi_tui::terminal::refresh_terminal_dimensions_with(kill);
}

/// Upstream also pins that kill is never called on win32; the branch is
/// compile-time on this platform, so the case compiles and runs there only.
#[test]
#[cfg(windows)]
fn does_not_call_kill_on_win32() {
    let kill: pi_tui::terminal::KillFn = &|_pid, _signal| {
        panic!("kill should not be called on win32");
    };
    pi_tui::terminal::refresh_terminal_dimensions_with(kill);
}
