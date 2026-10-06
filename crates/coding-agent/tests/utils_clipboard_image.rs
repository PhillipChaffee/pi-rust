//! The clipboard-image suites, upstream's `test/clipboard-image.test.ts`
//! and `test/clipboard-image-bmp-conversion.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, over the runner and native
//! seams the vi.mock originals drive.
//!
//! `clipboard-image-native-errors.test.ts` rides the interactive-mode
//! paste handler (`InteractiveMode.handleClipboardPaste`), which lands
//! with its own ticket; the native-read propagation contract it pins is
//! covered here through the ladder (recorded with the ticket).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use pi_coding_agent::config::EnvLookup;
use pi_coding_agent::utils::clipboard::{
    ClipboardError, NativeClipboard, Platform, set_native_clipboard_disabled,
    set_native_clipboard_override,
};
use pi_coding_agent::utils::clipboard_command::ClipboardCommandRunner;
use pi_coding_agent::utils::clipboard_image::{ClipboardImage, read_clipboard_image_with};

fn png_header() -> Vec<u8> {
    vec![
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52,
    ]
}

fn create_tiny_bmp_1x1_red_24bpp() -> Vec<u8> {
    // File size = 14 (BMP header) + 40 (DIB header) + 4 (pixel row) = 58
    const FILE_SIZE: u32 = 58;
    let mut buffer = vec![0u8; FILE_SIZE as usize];
    buffer[0..2].copy_from_slice(b"BM");
    buffer[2..6].copy_from_slice(&FILE_SIZE.to_le_bytes());
    buffer[10..14].copy_from_slice(&54u32.to_le_bytes());
    buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
    buffer[18..22].copy_from_slice(&1i32.to_le_bytes());
    buffer[22..26].copy_from_slice(&1i32.to_le_bytes());
    buffer[26..28].copy_from_slice(&1u16.to_le_bytes());
    buffer[28..30].copy_from_slice(&24u16.to_le_bytes());
    buffer[30..34].copy_from_slice(&0u32.to_le_bytes());
    buffer[34..38].copy_from_slice(&4u32.to_le_bytes());
    buffer[54..57].copy_from_slice(&[0x00, 0x00, 0xff]);
    buffer
}

/// The scripted (command, is-listing) key, the pair the backends probe.
type CommandKey = (String, bool);

/// The scripted command runner, upstream's `mocks.command`.
#[derive(Default)]
struct ScriptedRunner {
    /// The outcome per key; absent commands fall to the default (`None`).
    results: Mutex<HashMap<CommandKey, Option<Vec<u8>>>>,
    default_result: Mutex<Option<Vec<u8>>>,
    calls: Mutex<Vec<String>>,
}

impl ScriptedRunner {
    fn script(&self, command: &str, listing: bool, result: Option<Vec<u8>>) {
        self.results
            .lock()
            .expect("mock")
            .insert((command.to_string(), listing), result);
    }
}

impl ClipboardCommandRunner for ScriptedRunner {
    fn run<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        _options: &'a pi_coding_agent::utils::clipboard_command::ClipboardCommandOptions,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>> {
        self.calls.lock().expect("mock").push(command.to_string());
        // The wl-paste listing probes `--list-types`; the xclip probe asks
        // for TARGETS, so both spellings key the same listing flag.
        let listing = args
            .iter()
            .any(|arg| arg == "--list-types" || arg == "TARGETS");
        let result = self
            .results
            .lock()
            .expect("mock")
            .get(&(command.to_string(), listing))
            .cloned()
            .unwrap_or_else(|| self.default_result.lock().expect("mock").clone());
        Box::pin(std::future::ready(result))
    }
}

/// The native clipboard mock, upstream's `mocks.getNativeClipboard`.
struct MockNative {
    /// `Ok(None)` is the unavailable-display read, `Ok(Some(bytes))` the
    /// image, `Err` the propagating rejection.
    get_image_result: Mutex<Result<Option<Vec<u8>>, ClipboardError>>,
}

impl NativeClipboard for MockNative {
    fn get_text(&self) -> Result<String, ClipboardError> {
        Err(ClipboardError("not under test".to_string()))
    }

    fn get_image(&self) -> Result<Vec<u8>, ClipboardError> {
        let guard = self.get_image_result.lock().expect("mock");
        match guard.clone() {
            Ok(Some(bytes)) => Ok(bytes),
            Ok(None) => Ok(Vec::new()),
            Err(error) => Err(error),
        }
    }

    fn set_text(&self, _text: &str) -> bool {
        false
    }
}

fn native_with_image(bytes: Option<Vec<u8>>) -> Arc<MockNative> {
    Arc::new(MockNative {
        get_image_result: Mutex::new(Ok(bytes)),
    })
}

fn env_lookup(pairs: &[(&str, &str)]) -> EnvLookup {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    Box::new(move |key: &str| map.get(key).cloned())
}

fn empty_env() -> EnvLookup {
    Box::new(|_key| None)
}

// === readClipboardImage: the wayland and x11 command ladders ================

#[tokio::test]
async fn wayland_command_image_present_stops_fallback() {
    for present in [true, false] {
        let png = png_header();
        let runner = ScriptedRunner::default();
        let listing = if present {
            b"text/plain\nimage/png\n".to_vec()
        } else {
            b"text/plain\n".to_vec()
        };
        runner.script("wl-paste", true, Some(listing));
        runner.script("wl-paste", false, Some(png.clone()));
        let env = env_lookup(&[("WAYLAND_DISPLAY", "1"), ("DISPLAY", ":0")]);
        let image = read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error");
        if present {
            assert_eq!(
                image,
                Some(ClipboardImage {
                    bytes: png,
                    mime_type: "image/png".to_string()
                })
            );
            assert_eq!(runner.calls.lock().expect("mock").len(), 2);
        } else {
            assert_eq!(image, None);
            assert_eq!(runner.calls.lock().expect("mock").len(), 1);
        }
    }
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn x11_command_image_present_stops_fallback() {
    for present in [true, false] {
        let png = png_header();
        let runner = ScriptedRunner::default();
        let listing = if present {
            b"text/plain\nimage/png\n".to_vec()
        } else {
            b"text/plain\n".to_vec()
        };
        runner.script("xclip", true, Some(listing));
        runner.script("xclip", false, Some(png.clone()));
        let env = env_lookup(&[("DISPLAY", ":0")]);
        let image = read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error");
        if present {
            assert_eq!(
                image,
                Some(ClipboardImage {
                    bytes: png,
                    mime_type: "image/png".to_string()
                })
            );
            assert_eq!(runner.calls.lock().expect("mock").len(), 2);
        } else {
            assert_eq!(image, None);
            assert_eq!(runner.calls.lock().expect("mock").len(), 1);
        }
    }
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn native_x11_results_stop_fallback() {
    for bytes in [Some(png_header()), None, Some(Vec::new())] {
        let native = native_with_image(bytes.clone());
        set_native_clipboard_override(Some(native));
        let runner = ScriptedRunner::default();
        let env = env_lookup(&[("DISPLAY", ":0")]);
        let image = read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error");
        match bytes {
            Some(bytes) if !bytes.is_empty() => {
                assert_eq!(
                    image,
                    Some(ClipboardImage {
                        bytes,
                        mime_type: "image/png".to_string()
                    })
                );
            }
            _ => assert_eq!(image, None),
        }
        // The xclip probe list plus the four supported types.
        assert_eq!(*runner.calls.lock().expect("mock"), vec!["xclip"; 5]);
        set_native_clipboard_override(None);
    }
}

#[tokio::test]
async fn wayland_falls_back_to_x11_after_the_failure_kinds() {
    // A missing module (no native helper) and an unavailable display (an
    // empty native read) both leave the ladder intact.
    for failure in ["missing module", "unavailable display"] {
        let runner = ScriptedRunner::default();
        runner.script("wl-paste", true, None);
        if failure == "missing module" {
            set_native_clipboard_disabled();
        } else {
            let native = Arc::new(MockNative {
                get_image_result: Mutex::new(Ok(None)),
            });
            set_native_clipboard_override(Some(native));
        }
        runner.script("xclip", true, Some(b"image/png\n".to_vec()));
        runner.script("xclip", false, Some(png_header()));
        let env = env_lookup(&[("WAYLAND_DISPLAY", "1")]);
        let image = read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error");
        assert_eq!(
            image,
            Some(ClipboardImage {
                bytes: png_header(),
                mime_type: "image/png".to_string()
            })
        );
        set_native_clipboard_override(None);
    }
}

#[tokio::test]
async fn darwin_and_win32_read_native_images_once() {
    for platform in [Platform::Darwin, Platform::Win32] {
        for bytes in [Some(png_header()), None, Some(Vec::new())] {
            let native = native_with_image(bytes.clone());
            set_native_clipboard_override(Some(native));
            let runner = ScriptedRunner::default();
            let image = read_clipboard_image_with(&empty_env(), platform, &runner)
                .await
                .expect("no native error");
            match bytes {
                Some(bytes) if !bytes.is_empty() => {
                    assert_eq!(
                        image,
                        Some(ClipboardImage {
                            bytes,
                            mime_type: "image/png".to_string()
                        })
                    );
                }
                _ => assert_eq!(image, None),
            }
            assert!(runner.calls.lock().expect("mock").is_empty());
            set_native_clipboard_override(None);
        }
    }
}

#[tokio::test]
async fn returns_null_without_a_native_helper() {
    set_native_clipboard_disabled();
    let runner = ScriptedRunner::default();
    let image = read_clipboard_image_with(&empty_env(), Platform::Win32, &runner)
        .await
        .expect("no native error");
    assert_eq!(image, None);
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn propagates_native_transfer_errors_without_fallback() {
    for platform in [Platform::Linux, Platform::Win32] {
        let native = Arc::new(MockNative {
            get_image_result: Mutex::new(Err(ClipboardError(
                "Native clipboard operation failed".to_string(),
            ))),
        });
        set_native_clipboard_override(Some(native));
        let runner = ScriptedRunner::default();
        let env = env_lookup(&[("WAYLAND_DISPLAY", "1"), ("DISPLAY", ":0")]);
        let error = read_clipboard_image_with(&env, platform, &runner)
            .await
            .expect_err("native rejections propagate");
        assert_eq!(error.0, "Native clipboard operation failed");
        let expected = match platform {
            Platform::Linux => vec![
                "wl-paste".to_string(),
                "xclip".to_string(),
                "xclip".to_string(),
                "xclip".to_string(),
                "xclip".to_string(),
                "xclip".to_string(),
            ],
            _ => Vec::new(),
        };
        assert_eq!(*runner.calls.lock().expect("mock"), expected);
        set_native_clipboard_override(None);
    }
}

#[tokio::test]
async fn termux_does_not_read_image_clipboards() {
    let native = native_with_image(Some(png_header()));
    set_native_clipboard_override(Some(native));
    let runner = ScriptedRunner::default();
    let env = env_lookup(&[("TERMUX_VERSION", "0.119")]);
    let image = read_clipboard_image_with(&env, Platform::Linux, &runner)
        .await
        .expect("no native error");
    assert_eq!(image, None);
    assert!(runner.calls.lock().expect("mock").is_empty());
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn wsl_tries_power_shell_before_a_broken_native_x11_bridge() {
    // The bridge runs before the native read in the WSL ladder: a broken
    // X11 bridge never answers when the bridge itself succeeds upstream.
    // The port's bridge target is a UUID-named tmp file the scripted
    // `powershell.exe` cannot write, so the bridge fails here and the
    // native rejection propagates — the ordering contract is what the
    // call log pins.
    let native = Arc::new(MockNative {
        get_image_result: Mutex::new(Err(ClipboardError("Broken X11 bridge".to_string()))),
    });
    set_native_clipboard_override(Some(native));
    let runner = ScriptedRunner::default();
    let env = env_lookup(&[("WSL_DISTRO_NAME", "Ubuntu")]);
    let error = read_clipboard_image_with(&env, Platform::Linux, &runner)
        .await
        .expect_err("the broken native propagates after the bridge");
    assert_eq!(error.0, "Broken X11 bridge");
    // The bridge's own probes ran: the wl-paste/xclip ladder drained first.
    let (first_call, hit_xclip) = {
        let calls = runner.calls.lock().expect("mock");
        (
            calls.first().cloned(),
            calls.iter().any(|call| call == "xclip"),
        )
    };
    assert_eq!(first_call.as_deref(), Some("wl-paste"));
    assert!(hit_xclip);
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn an_unsupported_mime_whose_conversion_fails_reads_null() {
    let native = native_with_image(Some(b"garbage bytes".to_vec()));
    // detectSupportedImageMimeType answers octet-stream for the garbage, the
    // conversion fails, and the read answers null.
    set_native_clipboard_override(Some(native));
    let runner = ScriptedRunner::default();
    let image = read_clipboard_image_with(&empty_env(), Platform::Win32, &runner)
        .await
        .expect("no native error");
    assert_eq!(image, None);
    set_native_clipboard_override(None);
}

// === clipboard-image-bmp-conversion.test.ts =================================

#[tokio::test]
async fn converts_command_bmp_to_png() {
    let bmp = create_tiny_bmp_1x1_red_24bpp();
    let runner = ScriptedRunner::default();
    runner.script("wl-paste", true, Some(b"image/bmp\n".to_vec()));
    runner.script("wl-paste", false, Some(bmp));
    let env = env_lookup(&[("WAYLAND_DISPLAY", "wayland-0")]);
    let image = read_clipboard_image_with(&env, Platform::Linux, &runner)
        .await
        .expect("no native error");
    let image = image.expect("the BMP converts");
    assert_eq!(image.mime_type, "image/png");
    assert_eq!(&image.bytes[0..4], &[0x89, 0x50, 0x4e, 0x47]);
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn converts_native_bmp_to_png() {
    let native = native_with_image(Some(create_tiny_bmp_1x1_red_24bpp()));
    set_native_clipboard_override(Some(native));
    let runner = ScriptedRunner::default();
    let image = read_clipboard_image_with(&empty_env(), Platform::Win32, &runner)
        .await
        .expect("no native error");
    let image = image.expect("the BMP converts");
    assert_eq!(image.mime_type, "image/png");
    assert_eq!(&image.bytes[0..4], &[0x89, 0x50, 0x4e, 0x47]);
    set_native_clipboard_override(None);
}
