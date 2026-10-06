//! Default-application opening, upstream's `src/utils/open-browser.ts`.

/// Open a URL or file in the platform browser/default handler.
///
/// This intentionally never invokes a shell — upstream's doc note applies
/// to the win32 `cmd /c start` branch it avoids, which rides the map's
/// Windows exclusion. The launcher spawn is detached and its errors are
/// swallowed: browser launch is best-effort, callers still present the
/// target to the user, and the failure must not become a process crash.
pub fn open_browser(target: &str) {
    #[cfg(target_os = "macos")]
    let (command, args) = ("open", vec![target.to_string()]);
    #[cfg(all(unix, not(target_os = "macos")))]
    let (command, args) = ("xdg-open", vec![target.to_string()]);
    #[cfg(not(unix))]
    let (command, args) = (
        "rundll32",
        vec![
            "url.dll,FileProtocolHandler".to_string(),
            target.to_string(),
        ],
    );

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        let spawned = std::process::Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            // Detached, upstream's `detached: true`: the launcher leads its
            // own process group and outlives this process.
            .process_group(0)
            .spawn();
        let _error_event = spawned.is_err();
    }
    #[cfg(not(unix))]
    {
        let spawned = std::process::Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        let _error_event = spawned.is_err();
    }
}
