//! The version-check user agent, upstream's `src/utils/pi-user-agent.ts`.
//!
//! The Rust runtime has no node/bun version string to report, so the
//! runtime slot restates to a plain `rust` — the pi.dev probe upstream
//! points at is a later slice's consumer, and the shape (three
//! semicolon-separated tokens in parentheses) stays what upstream sends.

/// The user agent header value: `pi/<version> (<platform>; <runtime>;
/// <arch>)`.
///
/// Platform and architecture spell node's `os.platform()` / `os.arch()`
/// vocabulary, the same restatement the pi-ai crate's user agent makes, so
/// provider-side analytics see one vocabulary across both runtimes.
#[must_use]
pub fn get_pi_user_agent(version: &str) -> String {
    format!(
        "pi/{version} ({}; rust; {})",
        node_platform_name(),
        node_arch_name()
    )
}

/// node's `os.platform()` spelling for the running platform.
fn node_platform_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    }
}

/// node's `os.arch()` spelling for the running machine.
fn node_arch_name() -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        other => other,
    }
}
