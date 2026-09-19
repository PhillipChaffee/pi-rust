//! The `pi` user agent string, ported from
//! `packages/ai/src/utils/pi-user-agent.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The platform, kernel release, and machine architecture come from the
//! running process, so a Rust pi reports its own platform tuple rather than
//! re-deriving node's. The spelling of platform and architecture matches
//! node's `os.platform()` / `os.arch()` so provider-side analytics see one
//! vocabulary across both runtimes.

/// The user agent header value: `pi (platform release; arch)`.
#[must_use]
pub fn get_pi_user_agent() -> String {
    let platform = node_platform_name();
    let arch = node_arch_name();
    let release = kernel_release();
    format!("pi ({platform} {release}; {arch})")
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

/// The kernel release, node's `os.release()`.
fn kernel_release() -> String {
    let info = rustix::system::uname();
    info.release().to_string_lossy().into_owned()
}
