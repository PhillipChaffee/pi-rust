//! Clipboard text reads and writes with terminal fallbacks, upstream's
//! `src/utils/clipboard.ts`.
//!
//! The native surface is the arboard replacement pi-tui's decision #51
//! chose — upstream's `getNativeClipboard()` (`getText`/`getImage`,
//! optional `setText`) consumes it here rather than through a seam
//! exported from pi-tui. A process-global override swaps the native
//! clipboard and the OSC 52 stdout sink for the suites, the same seam
//! shape the fake-clock test support carries.

use std::io::Write as _;
use std::sync::{Arc, Mutex};

use base64::Engine as _;

use crate::config::EnvLookup;

use super::clipboard_command::{
    ClipboardCommandOptions, ClipboardCommandRunner, ProcessClipboardCommandRunner,
};

/// The longest OSC 52 payload the terminal fallback emits, upstream's
/// `MAX_OSC52_ENCODED_LENGTH`.
const MAX_OSC52_ENCODED_LENGTH: usize = 100_000;

/// The error a clipboard operation fails with, upstream's thrown `Error`
/// messages verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipboardError(pub String);

impl std::fmt::Display for ClipboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ClipboardError {}

/// The platform a clipboard call targets, node's `process.platform`
/// values; the process default derives from the build target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    /// node's `"darwin"`.
    Darwin,
    /// node's `"linux"`.
    Linux,
    /// node's `"win32"`.
    Win32,
}

impl Platform {
    /// The platform of the running process, upstream's `process.platform`.
    #[must_use]
    pub fn of_process() -> Self {
        match std::env::consts::OS {
            "macos" => Self::Darwin,
            "windows" => Self::Win32,
            _ => Self::Linux,
        }
    }
}

/// The native clipboard, upstream's `getNativeClipboard()` return: text
/// and image reads, and an optional text write.
pub trait NativeClipboard: Send + Sync {
    /// Plain-text read, upstream's `getText`; `Err` is the rejected read,
    /// an empty `Ok` an empty clipboard.
    ///
    /// # Errors
    /// Whatever the backend reports, upstream's rejected promise.
    fn get_text(&self) -> Result<String, ClipboardError>;

    /// Encoded image bytes, upstream's `getImage`; `Err` is the rejected
    /// read, an empty `Ok` a read with no image.
    ///
    /// # Errors
    /// Whatever the backend reports, upstream's rejected promise.
    fn get_image(&self) -> Result<Vec<u8>, ClipboardError>;

    /// Plain-text write, upstream's optional `setText`; `false` when the
    /// backend cannot write.
    fn set_text(&self, text: &str) -> bool;
}

/// The arboard-backed native clipboard; the image read encodes arboard's
/// raw RGBA to PNG, the one format the inline pipeline and the terminal
/// image path both take.
struct ArboardClipboard(Mutex<arboard::Clipboard>);

impl NativeClipboard for ArboardClipboard {
    fn get_text(&self) -> Result<String, ClipboardError> {
        self.0
            .lock()
            .map_err(|_| ClipboardError("the arboard clipboard poisoned".to_string()))?
            .get_text()
            .map_err(|_| ClipboardError("Native clipboard operation failed".to_string()))
    }

    fn get_image(&self) -> Result<Vec<u8>, ClipboardError> {
        let image = self
            .0
            .lock()
            .map_err(|_| ClipboardError("the arboard clipboard poisoned".to_string()))?
            .get_image()
            .map_err(|_| ClipboardError("Native clipboard operation failed".to_string()))?;
        // arboard's dims are usize; a clipboard image beyond u32 cannot exist.
        let width = u32::try_from(image.width)
            .map_err(|_| ClipboardError("Native clipboard operation failed".to_string()))?;
        let height = u32::try_from(image.height)
            .map_err(|_| ClipboardError("Native clipboard operation failed".to_string()))?;
        let buffer = image::RgbaImage::from_raw(width, height, image.bytes.into_owned())
            .ok_or_else(|| ClipboardError("Native clipboard operation failed".to_string()))?;
        let dynamic = image::DynamicImage::ImageRgba8(buffer);
        let mut encoded = std::io::Cursor::new(Vec::new());
        dynamic
            .write_to(&mut encoded, image::ImageFormat::Png)
            .map_err(|_| ClipboardError("Native clipboard operation failed".to_string()))?;
        Ok(encoded.into_inner())
    }

    fn set_text(&self, text: &str) -> bool {
        self.0
            .lock()
            .is_ok_and(|mut clipboard| clipboard.set_text(text).is_ok())
    }
}

type NativeClipboardBox = Arc<dyn NativeClipboard>;

/// The override state a suite installs.
///
/// A clipboard to answer with, or the no-native-helper state the absence
/// cases drive. Thread-local because the suites run concurrently in one
/// process; the production path always reads the unset state.
#[derive(Clone)]
enum NativeOverride {
    Disabled,
    Clipboard(NativeClipboardBox),
}

thread_local! {
    static NATIVE_CLIPBOARD_OVERRIDE: std::cell::RefCell<Option<NativeOverride>> =
        const { std::cell::RefCell::new(None) };
    static OSC52_SINK: std::cell::RefCell<Option<OscSink>> = const { std::cell::RefCell::new(None) };
}

/// The native clipboard of the running session, upstream's
/// `getNativeClipboard()`.
///
/// `None` when the platform has no reachable clipboard. A suite override
/// wins; a disabled override answers `None` outright, the "without a
/// native helper" state.
#[must_use]
pub fn get_native_clipboard() -> Option<NativeClipboardBox> {
    let overridden = NATIVE_CLIPBOARD_OVERRIDE.with(|cell| cell.borrow().clone());
    match overridden {
        Some(NativeOverride::Disabled) => None,
        Some(NativeOverride::Clipboard(clipboard)) => Some(clipboard),
        None => arboard::Clipboard::new().ok().map(|clipboard| {
            let clipboard: NativeClipboardBox = Arc::new(ArboardClipboard(Mutex::new(clipboard)));
            clipboard
        }),
    }
}

/// Install a native-clipboard override for the suites, the seam upstream's
/// `vi.mock("@earendil-works/pi-tui")` carries. Pass `None` to restore the
/// process default.
pub fn set_native_clipboard_override(overridden: Option<NativeClipboardBox>) {
    NATIVE_CLIPBOARD_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = overridden.map(NativeOverride::Clipboard);
    });
}

/// Force the no-native-helper state for the suites, upstream's mocked
/// `getNativeClipboard() -> undefined`.
pub fn set_native_clipboard_disabled() {
    NATIVE_CLIPBOARD_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = Some(NativeOverride::Disabled);
    });
}

type OscSink = Arc<dyn Fn(&[u8]) + Send + Sync>;

/// Install an OSC 52 sink for the suites; pass `None` to restore stdout.
pub fn set_osc52_sink(sink: Option<OscSink>) {
    OSC52_SINK.with(|cell| {
        *cell.borrow_mut() = sink;
    });
}

fn write_osc52(bytes: &[u8]) {
    let sink = OSC52_SINK.with(|cell| cell.borrow().clone());
    if let Some(sink) = sink {
        sink(bytes);
        return;
    }
    osc52_stdout_write(bytes);
    let _flushed = std::io::stdout().flush();
}

fn osc52_stdout_write(bytes: &[u8]) {
    let _written = std::io::stdout().write_all(bytes);
}

/// The truthiness check upstream's `process.env` reads run: an empty
/// value is an unset variable, and `Boolean("")` is false.
pub(crate) fn env_set(env: &EnvLookup, key: &str) -> bool {
    env(key).is_some_and(|value| !value.is_empty())
}

fn is_remote_session(env: &EnvLookup) -> bool {
    env_set(env, "SSH_CONNECTION") || env_set(env, "SSH_CLIENT") || env_set(env, "MOSH_CONNECTION")
}

fn emit_osc52(text: &str) -> bool {
    let encoded = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    if encoded.len() > MAX_OSC52_ENCODED_LENGTH {
        return false;
    }
    write_osc52(format!("\x1b]52;c;{encoded}\x07").as_bytes());
    true
}

/// The platform's command writers, upstream's `if (!copied)` ladder: each
/// writer runs with the text on stdin, and the first success carries.
async fn command_writer(
    env: &EnvLookup,
    platform: Platform,
    runner: &dyn ClipboardCommandRunner,
    text: &str,
) -> bool {
    let mut commands: Vec<(String, Vec<String>)> = Vec::new();
    match platform {
        Platform::Darwin => commands.push(("pbcopy".to_string(), Vec::new())),
        Platform::Win32 => commands.push(("clip".to_string(), Vec::new())),
        Platform::Linux => {
            if env_set(env, "TERMUX_VERSION") {
                commands.push(("termux-clipboard-set".to_string(), Vec::new()));
            }
            if env_set(env, "WAYLAND_DISPLAY") {
                commands.push(("wl-copy".to_string(), Vec::new()));
            }
            if env_set(env, "DISPLAY") {
                commands.push((
                    "xclip".to_string(),
                    clipboard_command_args(&["-selection", "clipboard"]),
                ));
                commands.push((
                    "xsel".to_string(),
                    clipboard_command_args(&["--clipboard", "--input"]),
                ));
            }
        }
    }
    for (command, args) in commands {
        let written = runner
            .run(
                &command,
                &args,
                &ClipboardCommandOptions {
                    input: Some(text.to_string()),
                    timeout_ms: Some(5000),
                    ..ClipboardCommandOptions::default()
                },
            )
            .await;
        if written.is_some() {
            return true;
        }
    }
    false
}

fn clipboard_command_args(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_string()).collect()
}

/// Read plain text from the system clipboard, upstream's
/// `readClipboardText` over the process environment and platform.
pub async fn read_clipboard_text() -> Option<String> {
    read_clipboard_text_with(
        &crate::config::default_env_lookup(),
        Platform::of_process(),
        &ProcessClipboardCommandRunner,
    )
    .await
}

/// [`read_clipboard_text`] over an injected environment, platform, and
/// command runner, the seam the suites drive.
pub async fn read_clipboard_text_with(
    env: &EnvLookup,
    platform: Platform,
    runner: &dyn ClipboardCommandRunner,
) -> Option<String> {
    if platform == Platform::Linux {
        let mut commands: Vec<(String, Vec<String>)> = Vec::new();
        if env_set(env, "TERMUX_VERSION") {
            commands.push(("termux-clipboard-get".to_string(), Vec::new()));
        }
        if env_set(env, "WAYLAND_DISPLAY") {
            commands.push((
                "wl-paste".to_string(),
                clipboard_command_args(&["--no-newline", "--type", "text"]),
            ));
        }
        if env_set(env, "DISPLAY") {
            commands.push((
                "xclip".to_string(),
                clipboard_command_args(&["-selection", "clipboard", "-out"]),
            ));
            commands.push((
                "xsel".to_string(),
                clipboard_command_args(&["--clipboard", "--output"]),
            ));
        }
        for (command, args) in commands {
            if let Some(bytes) = runner
                .run(
                    &command,
                    &args,
                    &super::clipboard_command::read_options(Some(5000)),
                )
                .await
            {
                let text = String::from_utf8_lossy(&bytes).into_owned();
                return if text.is_empty() { None } else { Some(text) };
            }
        }
    }
    // A rejected read and an empty read both answer null, upstream's
    // `|| null` under its try/catch.
    get_native_clipboard()?
        .get_text()
        .ok()
        .filter(|text| !text.is_empty())
}

/// Copy text to the system clipboard, upstream's `copyToClipboard` over the
/// process environment and platform.
///
/// # Errors
/// [`ClipboardError`] with the platform's diagnosis when every writer —
/// native, command, and the OSC 52 terminal fallback in remote sessions —
/// failed.
pub async fn copy_to_clipboard(text: &str) -> Result<(), ClipboardError> {
    copy_to_clipboard_with(
        text,
        &crate::config::default_env_lookup(),
        Platform::of_process(),
        &ProcessClipboardCommandRunner,
    )
    .await
}

/// [`copy_to_clipboard`] over an injected environment, platform, and
/// command runner, the seam the suites drive.
///
/// # Errors
/// [`ClipboardError`] with the platform's diagnosis when every writer
/// failed.
pub async fn copy_to_clipboard_with(
    text: &str,
    env: &EnvLookup,
    platform: Platform,
    runner: &dyn ClipboardCommandRunner,
) -> Result<(), ClipboardError> {
    // Direct writes precede OSC 52 so the terminal cannot race the native
    // writer. Linux tools retain clipboard selection ownership after this
    // call returns.
    let copied = (platform != Platform::Linux
        && get_native_clipboard().is_some_and(|clipboard| clipboard.set_text(text)))
        || command_writer(env, platform, runner, text).await;
    // The OSC 52 fallback only matters when every writer failed; the
    // verified command write is already carried, upstream's
    // `emitOsc52(text) || copied`.
    let copied = if is_remote_session(env) {
        emit_osc52(text) || copied
    } else {
        copied
    };
    if !copied {
        return Err(ClipboardError(match platform {
            Platform::Linux if env_set(env, "TERMUX_VERSION") => {
                "Clipboard unavailable: install the Termux:API app and `termux-api` package"
                    .to_string()
            }
            Platform::Linux if env_set(env, "WAYLAND_DISPLAY") => {
                "Clipboard unavailable: install `wl-clipboard` (`wl-copy`) or check Wayland access"
                    .to_string()
            }
            Platform::Linux if env_set(env, "DISPLAY") => {
                "Clipboard unavailable: install `xclip` or `xsel`, or check X11 access".to_string()
            }
            Platform::Linux => {
                "Clipboard unavailable: no Wayland or X11 display detected".to_string()
            }
            _ => "Clipboard unavailable".to_string(),
        }));
    }
    Ok(())
}
