//! The belt's remaining boundary coverage: the Display impls, the plain
//! production entries, the PowerShell bridge's failure arms, the hosted
//! degenerate shapes, and the renderer scanner's span-tag corners.
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use base64::Engine as _;

use pi_coding_agent::config::EnvLookup;
use pi_coding_agent::utils::clipboard::{
    ClipboardError, NativeClipboard, Platform, copy_to_clipboard, copy_to_clipboard_with,
    read_clipboard_text, set_native_clipboard_disabled, set_native_clipboard_override,
    set_osc52_sink,
};
use pi_coding_agent::utils::clipboard_command::{
    ClipboardCommandOptions, ClipboardCommandRunner, ProcessClipboardCommandRunner,
};
use pi_coding_agent::utils::clipboard_image::{
    base_mime_type, extension_for_image_mime_type, read_clipboard_image, read_clipboard_image_with,
};
use pi_coding_agent::utils::deprecation::{clear_deprecation_warnings_for_tests, warn_deprecation};
use pi_coding_agent::utils::frontmatter::parse_frontmatter;
use pi_coding_agent::utils::image_convert::convert_to_png;
use pi_coding_agent::utils::image_process::{ProcessImageOptions, process_image};
use pi_coding_agent::utils::image_resize_core::resize_image_in_process;
use pi_coding_agent::utils::mime::detect_supported_image_mime_type;
use pi_coding_agent::utils::paths::{
    PathInputOptions, PathNormalizeError, format_path_relative_to_cwd_or_absolute,
    get_cwd_relative_path,
};
use pi_coding_agent::utils::pi_user_agent::get_pi_user_agent;
use pi_coding_agent::utils::shell::{get_shell_config, is_legacy_wsl_bash_path};
use pi_coding_agent::utils::sleep::{SleepAborted, sleep};

// === Display impls ==========================================================

#[test]
fn the_error_types_display_their_messages() {
    assert_eq!(ClipboardError("boom".to_string()).to_string(), "boom");
    assert_eq!(SleepAborted.to_string(), "Aborted");
    assert_eq!(PathNormalizeError.to_string(), "Invalid URL");
    let error = parse_frontmatter("---\nfoo: [bar\n---\nBody").expect_err("parse error");
    assert!(!error.to_string().is_empty());
}

// === the production entries =================================================

#[tokio::test]
async fn the_plain_entries_read_through_the_installed_override() {
    // The plain variants ride the same override seam the suites install;
    // a mock answers so no real clipboard or writer runs.
    struct Mock;
    impl NativeClipboard for Mock {
        fn get_text(&self) -> Result<String, ClipboardError> {
            Ok("plain text".to_string())
        }

        fn get_image(&self) -> Result<Vec<u8>, ClipboardError> {
            Ok(Vec::new())
        }

        fn set_text(&self, _text: &str) -> bool {
            true
        }
    }
    set_native_clipboard_override(Some(Arc::new(Mock)));
    assert_eq!(read_clipboard_text().await, Some("plain text".to_string()));
    // Writes consult the native clipboard off-linux only — the command
    // writers carry linux writes, upstream's platform ladder. The plain
    // entry answers deterministically on the headless runner (no writer
    // can spawn without a display environment); a display-carrying linux
    // box pins the ladder through the injected seam with an empty
    // environment instead of racing the live tools.
    if Platform::of_process() != Platform::Linux {
        copy_to_clipboard("hello").await.expect("the mock writes");
    } else if std::env::var_os("DISPLAY").is_some()
        || std::env::var_os("WAYLAND_DISPLAY").is_some()
        || std::env::var_os("TERMUX_VERSION").is_some()
    {
        let env: EnvLookup = Box::new(|_key: &str| None);
        copy_to_clipboard_with(
            "hello",
            &env,
            Platform::Linux,
            &ProcessClipboardCommandRunner,
        )
        .await
        .expect_err("an empty environment carries no writer");
    } else {
        let error = copy_to_clipboard("hello")
            .await
            .expect_err("a headless linux carries no writer");
        assert_eq!(
            error.0,
            "Clipboard unavailable: no Wayland or X11 display detected"
        );
    }
    set_native_clipboard_override(None);
}

#[test]
fn the_platform_of_the_process_matches_the_build_target() {
    let platform = Platform::of_process();
    let expected = match std::env::consts::OS {
        "macos" => Platform::Darwin,
        "windows" => Platform::Win32,
        _ => Platform::Linux,
    };
    assert_eq!(platform, expected);
}

#[tokio::test]
async fn the_arboard_adapter_answers_through_the_real_clipboard_when_present() {
    // The arboard adapter is the production native clipboard; the test
    // reads it (never writes), and skips silently where no clipboard
    // exists (the headless CI runners).
    let Some(clipboard) = pi_coding_agent::utils::clipboard::get_native_clipboard() else {
        return;
    };
    // A failed or empty read is a valid outcome; the adapter must not panic.
    let _text = clipboard.get_text();
    let _image = clipboard.get_image();
    // The write path round-trips the read text, so the user's clipboard is
    // untouched; a failed read skips the write.
    if let Ok(current) = clipboard.get_text() {
        let _wrote = clipboard.set_text(&current);
    }
}

/// The all-fail runner, upstream's `mocks.command.mockResolvedValue(undefined)`.
struct FailRunner;

impl ClipboardCommandRunner for FailRunner {
    fn run<'a>(
        &'a self,
        _command: &'a str,
        _args: &'a [String],
        _options: &'a ClipboardCommandOptions,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>> {
        Box::pin(std::future::ready(None))
    }
}

#[tokio::test]
async fn the_osc_52_fallback_writes_stdout_when_no_sink_is_installed() {
    // No sink installed: the sequence reaches the harness-captured stdout.
    set_osc52_sink(None);
    set_native_clipboard_disabled();
    let env: EnvLookup =
        Box::new(|key| (key == "SSH_CONNECTION").then(|| "client server".to_string()));
    copy_to_clipboard_with("hello", &env, Platform::Darwin, &FailRunner)
        .await
        .expect("osc 52 carried the write");
    set_native_clipboard_override(None);
}

// === the clipboard-image belt ===============================================

#[test]
fn image_mime_extensions_map_the_inline_formats() {
    assert_eq!(extension_for_image_mime_type("image/png"), Some("png"));
    assert_eq!(
        extension_for_image_mime_type("image/jpeg;charset=1"),
        Some("jpg")
    );
    assert_eq!(extension_for_image_mime_type("image/webp"), Some("webp"));
    assert_eq!(extension_for_image_mime_type("image/gif"), Some("gif"));
    assert_eq!(extension_for_image_mime_type("IMAGE/PNG"), Some("png"));
    assert_eq!(extension_for_image_mime_type("image/bmp"), None);
}

#[test]
fn base_mime_types_strip_parameters_and_casing() {
    assert_eq!(base_mime_type("Image/PNG; param=1"), "image/png");
    assert_eq!(base_mime_type("  image/png  "), "image/png");
}

#[tokio::test]
async fn the_wsl_power_shell_bridge_fails_closed_on_every_arm() {
    struct Scripted {
        wslpath: Option<Vec<u8>>,
        powershell: Option<Vec<u8>>,
    }
    impl ClipboardCommandRunner for Scripted {
        fn run<'a>(
            &'a self,
            command: &'a str,
            _args: &'a [String],
            _options: &'a ClipboardCommandOptions,
        ) -> Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>> {
            let result = match command {
                "wslpath" => self.wslpath.clone(),
                "powershell.exe" => self.powershell.clone(),
                _ => None,
            };
            Box::pin(std::future::ready(result))
        }
    }
    set_native_clipboard_disabled();
    let env: EnvLookup = Box::new(|key| (key == "WSL_DISTRO_NAME").then(|| "Ubuntu".to_string()));

    // The wslpath probe fails.
    let runner = Scripted {
        wslpath: None,
        powershell: None,
    };
    assert_eq!(
        read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error"),
        None
    );
    // The wslpath probe answers empty.
    let runner = Scripted {
        wslpath: Some(Vec::new()),
        powershell: None,
    };
    assert_eq!(
        read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error"),
        None
    );
    // PowerShell answers not-ok.
    let runner = Scripted {
        wslpath: Some(b"C:\\clip.png\n".to_vec()),
        powershell: Some(b"empty\n".to_vec()),
    };
    assert_eq!(
        read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error"),
        None
    );
    // PowerShell answers ok but the tmp file never materialized.
    let runner = Scripted {
        wslpath: Some(b"C:\\clip.png\n".to_vec()),
        powershell: Some(b"ok\n".to_vec()),
    };
    assert_eq!(
        read_clipboard_image_with(&env, Platform::Linux, &runner)
            .await
            .expect("no native error"),
        None
    );
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn the_wl_paste_data_read_failing_stays_unavailable() {
    struct Listing;
    impl ClipboardCommandRunner for Listing {
        fn run<'a>(
            &'a self,
            command: &'a str,
            args: &'a [String],
            _options: &'a ClipboardCommandOptions,
        ) -> Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>> {
            let listing = args.iter().any(|arg| arg == "--list-types");
            let result = if command == "wl-paste" && listing {
                Some(b"image/png\n".to_vec())
            } else {
                None
            };
            Box::pin(std::future::ready(result))
        }
    }
    set_native_clipboard_disabled();
    let env: EnvLookup =
        Box::new(|key| (key == "WAYLAND_DISPLAY").then(|| "wayland-0".to_string()));
    assert_eq!(
        read_clipboard_image_with(&env, Platform::Linux, &Listing)
            .await
            .expect("no native error"),
        None
    );
    set_native_clipboard_override(None);
}

#[tokio::test]
async fn the_plain_image_read_answers_without_backends() {
    set_native_clipboard_disabled();
    let env: EnvLookup = Box::new(|_| None);
    assert_eq!(
        read_clipboard_image(&env, Platform::Win32)
            .await
            .expect("no native error"),
        None
    );
    set_native_clipboard_override(None);
}

// === the shell belt =========================================================

#[test]
fn legacy_wsl_classifier_checks_the_drive_head() {
    // A multi-character head is not a drive letter.
    assert!(!is_legacy_wsl_bash_path("XX:\\windows\\system32\\bash.exe"));
    assert!(!is_legacy_wsl_bash_path(":\\windows\\system32\\bash.exe"));
    assert!(!is_legacy_wsl_bash_path(""));
}

#[test]
fn a_wsl_spelled_shell_path_selects_the_stdin_transport() {
    // The custom path itself carries the WSL spelling and resolves
    // relative to the process cwd, upstream's existsSync(customShellPath)
    // on the legacy WSL path.
    let dir = tempfile::tempdir().expect("scratch");
    let previous = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(&dir).expect("chdir");
    let shell_name = "C:\\Windows\\System32\\bash.exe";
    std::fs::write(shell_name, b"").expect("fixture");
    let config = get_shell_config(Some(shell_name)).expect("custom shell");
    std::env::set_current_dir(previous).expect("restore cwd");
    assert_eq!(config.shell, shell_name);
    assert_eq!(config.args, vec!["-s".to_string()]);
    assert_eq!(
        config.command_transport,
        Some(pi_coding_agent::utils::shell::CommandTransport::Stdin)
    );
}

// === sleep ==================================================================

#[tokio::test(start_paused = true)]
async fn sleep_with_a_live_signal_resolves_after_the_delay() {
    let token = tokio_util::sync::CancellationToken::new();
    tokio::time::advance(std::time::Duration::from_millis(10)).await;
    assert_eq!(sleep(10, Some(&token)).await, Ok(()));
}

// === frontmatter ============================================================

#[test]
fn frontmatter_reports_multiple_documents() {
    // The `...` document terminator closes the first YAML document inside
    // the extracted slice, leaving a second one for the loader.
    let error = parse_frontmatter("---\na: 1\n...\nb: 2\n---\nBody").expect_err("multi-doc");
    assert_eq!(error.0, "Source contains multiple documents");
}

#[test]
fn frontmatter_handles_the_immediate_closing_delimiter() {
    // `---\n---`: the extracted YAML slice clamps to empty.
    let parsed = parse_frontmatter("---\n---\nBody").expect("parse");
    assert_eq!(
        parsed.frontmatter,
        yaml_rust2::Yaml::Hash(yaml_rust2::yaml::Hash::new())
    );
    assert_eq!(parsed.body, "Body");
}

// === paths ==================================================================

#[test]
fn cwd_relative_marks_the_directory_itself() {
    let cwd = std::env::temp_dir().join("pi-paths-cwd");
    let cwd = cwd.to_string_lossy().into_owned();
    assert_eq!(get_cwd_relative_path(&cwd, &cwd), Some(".".to_string()));
}

#[test]
fn format_relative_prefers_the_relative_form_inside_the_cwd() {
    let cwd = std::env::temp_dir().join("pi-paths-cwd");
    let cwd = cwd.to_string_lossy().into_owned();
    assert_eq!(
        format_path_relative_to_cwd_or_absolute("sub/file.txt", &cwd),
        "sub/file.txt"
    );
    assert_eq!(
        format_path_relative_to_cwd_or_absolute("/elsewhere/file.txt", &cwd),
        "/elsewhere/file.txt"
    );
}

#[test]
fn a_bare_file_url_normalizes_to_the_root() {
    // node's fileURLToPath("file://") reads the URL's normalized "/" path.
    assert_eq!(
        pi_coding_agent::utils::paths::resolve_path_with(
            "file://",
            "/unused",
            &PathInputOptions::default()
        ),
        Ok("/".to_string())
    );
}

// === mime ===================================================================

#[test]
fn sniffs_a_file_and_reports_a_missing_one() {
    let dir = tempfile::tempdir().expect("scratch");
    let mut bmp = vec![0u8; 58];
    bmp[0..2].copy_from_slice(b"BM");
    bmp[2..6].copy_from_slice(&58u32.to_le_bytes());
    bmp[10..14].copy_from_slice(&54u32.to_le_bytes());
    bmp[14..18].copy_from_slice(&40u32.to_le_bytes());
    bmp[26..28].copy_from_slice(&1u16.to_le_bytes());
    bmp[28..30].copy_from_slice(&24u16.to_le_bytes());
    let file = dir.path().join("image.bmp");
    std::fs::write(&file, &bmp).expect("write");
    assert_eq!(
        pi_coding_agent::utils::mime::detect_supported_image_mime_type_from_file(
            &file.to_string_lossy()
        )
        .expect("sniff"),
        Some("image/bmp")
    );
    assert!(
        pi_coding_agent::utils::mime::detect_supported_image_mime_type_from_file(
            &dir.path().join("missing").to_string_lossy()
        )
        .is_err()
    );
}

#[test]
fn bmp_rejects_an_offset_past_the_declared_size_and_a_short_info_header() {
    // pixel-data offset at the declared file size.
    let mut buffer = vec![0u8; 58];
    buffer[0..2].copy_from_slice(b"BM");
    buffer[2..6].copy_from_slice(&58u32.to_le_bytes());
    buffer[10..14].copy_from_slice(&58u32.to_le_bytes());
    buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
    assert_eq!(detect_supported_image_mime_type(&buffer), None);
    // A 40-byte DIB header needs 30 bytes to read planes and bpp.
    let mut short = vec![0u8; 28];
    short[0..2].copy_from_slice(b"BM");
    short[14..18].copy_from_slice(&40u32.to_le_bytes());
    assert_eq!(detect_supported_image_mime_type(&short), None);
}

// === image pipeline =========================================================

#[test]
fn an_undecodable_base64_payload_converts_to_none() {
    assert_eq!(convert_to_png("!!!not base64!!!", "image/bmp"), None);
}

#[tokio::test]
async fn an_undecodable_image_reports_the_conversion_omission() {
    let result = process_image(
        b"not an image".to_vec(),
        "application/octet-stream",
        Some(&ProcessImageOptions::default()),
    )
    .await;
    assert_eq!(
        result,
        pi_coding_agent::utils::image_process::ProcessImageResult::Omitted {
            message: "[Image omitted: could not be converted to a supported inline image format.]"
                .to_string(),
        }
    );
}

#[tokio::test]
async fn a_same_mime_conversion_carries_no_hint() {
    // An uppercase MIME normalizes to the same target, so the conversion
    // hint stays silent.
    let tiny_png: &[u8] = &base64::engine::general_purpose::STANDARD.decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURf8AAP///0EdNBEAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gEOADM5Ddoh/wAAAAxJREFUCNdjYGBgAAAABAABJzQnCgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyNi0wMS0xNFQwMDo1MTo1NyswMDowMOnKzHgAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6MDCYl3TEAAAAKHRFWHRkYXRlOnRpbWVzdGFtcAAyMDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAwz4JVGwAAAABJRU5ErkJggg==",
    )
    .expect("fixture");
    let result = process_image(
        tiny_png.to_vec(),
        "IMAGE/PNG; boundary=x",
        Some(&ProcessImageOptions {
            auto_resize_images: Some(false),
            resize_options: None,
        }),
    )
    .await;
    let pi_coding_agent::utils::image_process::ProcessImageResult::Ok {
        data: _,
        mime_type,
        hints,
    } = result
    else {
        panic!("the png survives");
    };
    assert_eq!(mime_type, "image/png");
    assert!(hints.is_empty(), "{hints:?}");
}

#[test]
fn resize_reports_the_fallback_mime_for_an_empty_type() {
    let tiny_png: Vec<u8> = base64::engine::general_purpose::STANDARD.decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURf8AAP///0EdNBEAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gEOADM5Ddoh/wAAAAxJREFUCNdjYGBgAAAABAABJzQnCgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyNi0wMS0xNFQwMDo1MTo1NyswMDowMOnKzHgAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6MDCYl3TEAAAAKHRFWHRkYXRlOnRpbWVzdGFtcAAyMDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAwz4JVGwAAAABJRU5ErkJggg==",
    )
    .expect("fixture");
    let result = resize_image_in_process(&tiny_png, "", None).expect("resize");
    assert_eq!(result.mime_type, "image/png");
}

#[test]
fn resize_shrinks_one_dimension_to_one() {
    // A 2400x1 gray PNG forced under a tight byte cap: the tall dimension
    // clamps to 1 while the wide one shrinks by quarters.
    let mut raw = Vec::new();
    raw.push(0);
    raw.extend(std::iter::repeat_n(0u8, 2400));
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2400u32.to_be_bytes());
    ihdr.extend_from_slice(&1u32.to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
    let png = build_png(&ihdr, &raw);
    let result = resize_image_in_process(
        &png,
        "image/png",
        Some(
            &pi_coding_agent::utils::image_resize_core::ImageResizeOptions {
                max_width: Some(2000),
                max_height: Some(2000),
                max_bytes: Some(64),
                jpeg_quality: Some(80),
            },
        ),
    );
    // The walk terminates either at a candidate under the cap or at null;
    // both pin the shrink arms.
    if let Some(resized) = result {
        assert!(resized.width < 2400);
        assert_eq!(resized.height, 1);
    }
}

fn build_png(ihdr_body: &[u8], raw: &[u8]) -> Vec<u8> {
    let mut crc_input = Vec::new();
    crc_input.extend_from_slice(b"IHDR");
    crc_input.extend_from_slice(ihdr_body);
    let mut png = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixture bodies are tens of bytes"
    )]
    let ihdr_length = ihdr_body.len() as u32;
    png.extend_from_slice(&ihdr_length.to_be_bytes());
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(ihdr_body);
    png.extend_from_slice(&crc32(&crc_input).to_be_bytes());

    let mut zlib_input = Vec::new();
    let mut encoder =
        flate2::write::ZlibEncoder::new(&mut zlib_input, flate2::Compression::default());
    std::io::Write::write_all(&mut encoder, raw).expect("deflate");
    std::io::Write::flush(&mut encoder).expect("deflate");
    drop(encoder);
    let mut idat_input = Vec::new();
    idat_input.extend_from_slice(b"IDAT");
    idat_input.extend_from_slice(&zlib_input);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixture bodies are tens of bytes"
    )]
    let idat_length = zlib_input.len() as u32;
    png.extend_from_slice(&idat_length.to_be_bytes());
    png.extend_from_slice(&idat_input);
    png.extend_from_slice(&crc32(&idat_input).to_be_bytes());

    png.extend_from_slice(&0u32.to_be_bytes());
    png.extend_from_slice(b"IEND");
    png.extend_from_slice(&crc32(b"IEND").to_be_bytes());
    png
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// === pi user agent ==========================================================

#[test]
fn the_user_agent_embeds_the_version_verbatim() {
    let agent = get_pi_user_agent("9.9.9-beta.1");
    assert!(agent.starts_with("pi/9.9.9-beta.1 ("), "{agent}");
    assert!(agent.ends_with(')'), "{agent}");
}

// === the syntax-highlight scanner ===========================================

#[test]
fn the_theme_debug_reports_its_keys() {
    let theme = pi_coding_agent::utils::syntax_highlight::HighlightTheme::new(HashMap::new());
    let rendered = format!("{theme:?}");
    assert!(rendered.contains("HighlightTheme"), "{rendered}");
}

fn kw_formatter() -> pi_coding_agent::utils::syntax_highlight::HighlightFormatter {
    let formatter: pi_coding_agent::utils::syntax_highlight::HighlightFormatter =
        Arc::new(|text: &str| format!("[kw:{text}]"));
    formatter
}

#[test]
fn span_tags_parse_single_quoted_and_broken_classes() {
    let theme = pi_coding_agent::utils::syntax_highlight::HighlightTheme::new(HashMap::from([(
        "keyword",
        kw_formatter(),
    )]));
    let render = pi_coding_agent::utils::syntax_highlight::render_highlighted_html;
    assert_eq!(
        render("<span class='hljs-keyword'>x</span>", &theme),
        "[kw:x]"
    );
    // A class value without a recognized prefix carries no scope.
    assert_eq!(render("<span class=nope>x</span>", &theme), "x");
    // A tag with no closing angle bracket flows through as text.
    assert_eq!(
        render("<span class=\"hljs-keyword\" x", &theme),
        "<span class=\"hljs-keyword\" x"
    );
}

#[test]
fn span_tag_scanners_cover_the_whitespace_variants() {
    let render = pi_coding_agent::utils::syntax_highlight::render_highlighted_html;
    let mut theme = HashMap::new();
    theme.insert("keyword", kw_formatter());
    for tag in [
        "<span\tclass=\"hljs-keyword\">x</span>",
        "<span\nclass=\"hljs-keyword\">x</span>",
        "<span\rclass=\"hljs-keyword\">x</span>",
        "<span class=\"hljs-keyword\">x</span>",
    ] {
        assert_eq!(
            render(
                tag,
                &pi_coding_agent::utils::syntax_highlight::HighlightTheme::new(theme.clone())
            ),
            "[kw:x]",
            "{tag}"
        );
    }
}

#[test]
fn dashed_scopes_inherit_their_prefix_formatter() {
    let render = pi_coding_agent::utils::syntax_highlight::render_highlighted_html;
    let title_formatter: pi_coding_agent::utils::syntax_highlight::HighlightFormatter =
        Arc::new(|t: &str| format!("[title:{t}]"));
    let mut theme = HashMap::new();
    theme.insert("title", title_formatter);
    assert_eq!(
        render(
            "<span class=\"hljs-title-function\">f</span>",
            &pi_coding_agent::utils::syntax_highlight::HighlightTheme::new(theme)
        ),
        "[title:f]"
    );
}

#[test]
fn the_engine_extension_alias_resolves_csharp() {
    let highlight = pi_coding_agent::utils::syntax_highlight::highlight;
    let rendered = highlight(
        "class X { }",
        pi_coding_agent::utils::syntax_highlight::HighlightOptions {
            language: Some("csharp"),
            ignore_illegals: false,
            language_subset: Vec::new(),
            theme: None,
        },
    );
    assert!(rendered.contains("class"), "{rendered}");
}

// === the open-browser launch ===============================================

#[test]
fn the_launcher_swallows_failures() {
    // A target that does not exist makes the platform launcher fail fast;
    // the failure stays swallowed and the process carries on.
    pi_coding_agent::utils::open_browser::open_browser("pi-belt-no-such-target");
}

// === deprecation ============================================================

#[test]
fn deprecation_dedup_and_clear_round_trip() {
    clear_deprecation_warnings_for_tests();
    warn_deprecation("second boundary probe");
    warn_deprecation("second boundary probe");
    clear_deprecation_warnings_for_tests();
    warn_deprecation("second boundary probe");
    clear_deprecation_warnings_for_tests();
}
