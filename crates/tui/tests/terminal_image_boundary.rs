//! Boundary tests for the terminal-image surface (#51), binding the branches
//! upstream's `terminal-image.test.ts` leaves untested at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the tmux probe's spawn, exit,
//! and timeout paths; the capability-override cache semantics; the image-id
//! allocator's range; the chunked Kitty transmission boundaries; the
//! PNG/JPEG/GIF/WebP size parsers; the cell-size clamps; the render and
//! placement `None` paths; and the `Image` component's parser-driven
//! dimensions and caches.
//!
//! Restatement: the tmux probe stands in through
//! [`probe_tmux_hyperlinks_with`] with shell scripts, standing in for the
//! real `tmux` binary the CI runners do not carry; the global stores sit
//! behind [`GLOBALS_LOCK`] like the sibling suites.
#![expect(
    clippy::expect_used,
    reason = "fixture and temp-file setup failures are test-environment failures; expecting keeps the setup readable"
)]

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use base64::Engine;
use pi_tui::components::image::{Image, ImageOptions, ImageTheme};
use pi_tui::terminal_image::{
    CapabilityOverrides, EncodeITerm2Options, EncodeKittyOptions, ImageDimensions,
    ImageRenderOptions, KittyImageMetadata, TerminalCapabilities, allocate_image_id,
    calculate_image_cell_size, calculate_image_rows, crop_kitty_image_line, detect_capabilities,
    encode_iterm2, encode_kitty, get_capabilities, get_image_dimensions, get_kitty_image_metadata,
    get_kitty_image_placement, image_fallback, probe_tmux_hyperlinks, probe_tmux_hyperlinks_with,
    register_kitty_image_metadata, render_image, reset_capabilities_cache, set_capabilities,
    set_capability_overrides, set_cell_dimensions,
};
use pi_tui::tui::Component;

static GLOBALS_LOCK: Mutex<()> = Mutex::new(());

fn theme() -> ImageTheme {
    ImageTheme {
        fallback_color: Arc::new(str::to_string),
    }
}

fn base64_of(bytes: Vec<u8>) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

static FAKE_TMUX_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Writes a stand-in `tmux` script and returns its path; the suite removes
/// the file when the assertion is done.
fn fake_tmux(body: &str) -> PathBuf {
    let seq = FAKE_TMUX_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("pi-tui-fake-tmux-{}-{seq}.sh", std::process::id()));
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write the stand-in tmux script");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
        .expect("make the stand-in executable");
    path
}

/// Probes the stand-in after one untimed warm-up run: the first exec of a
/// freshly written script pays the platform's executable scan, which can
/// alone approach the probe's 250 ms deadline, so the scan is paid outside
/// the timed call.
fn probe_fake(path: &std::path::Path) -> bool {
    let _ = std::process::Command::new(path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let mut command = std::process::Command::new(path);
    probe_tmux_hyperlinks_with(&mut command)
}

// --- the tmux probe ----------------------------------------------------------

#[test]
fn probe_answers_true_when_the_client_termfeatures_list_hyperlinks() {
    let path = fake_tmux("printf 'readonly,hyperlinks'");
    assert!(probe_fake(&path));
    std::fs::remove_file(&path).ok();
}

#[test]
fn probe_answers_false_when_the_client_termfeatures_omit_hyperlinks() {
    let path = fake_tmux("printf 'readonly,sixels'");
    assert!(!probe_fake(&path));
    std::fs::remove_file(&path).ok();
}

#[test]
fn probe_answers_false_when_the_probe_command_fails() {
    let path = fake_tmux("exit 3");
    assert!(!probe_fake(&path));
    std::fs::remove_file(&path).ok();
}

#[test]
fn probe_answers_false_when_the_client_does_not_answer_within_the_deadline() {
    let path = fake_tmux("sleep 1");
    let mut command = std::process::Command::new(&path);
    assert!(!probe_tmux_hyperlinks_with(&mut command));
    std::fs::remove_file(&path).ok();
}

#[test]
fn probe_trims_and_splits_the_termfeatures_list() {
    let path = fake_tmux("printf ' hyperlinks , sixels '");
    assert!(probe_fake(&path));
    std::fs::remove_file(&path).ok();
}

#[test]
fn the_real_probe_answers_without_panic() {
    // The real command answers false where no tmux binary or client exists;
    // a tmux-carrying machine may answer true, so only the absence of a
    // panic binds here.
    let _ = probe_tmux_hyperlinks();
}

// --- the capability-override cache -------------------------------------------

#[test]
fn equal_capability_overrides_keep_the_cached_capabilities() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capability_overrides(CapabilityOverrides::default());
    let explicit = TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Iterm2),
        true_color: true,
        hyperlinks: true,
    };
    set_capabilities(explicit);
    // The override set is unchanged, so the cache the test seeded stays.
    set_capability_overrides(CapabilityOverrides::default());
    assert_eq!(get_capabilities(), explicit);

    // A changed override set resets the cache and re-detects; this suite's
    // process environment carries no terminal markers, so the detection is
    // the neutral default under the forced hyperlink override.
    set_capability_overrides(CapabilityOverrides {
        hyperlinks: Some(true),
        ..CapabilityOverrides::default()
    });
    assert!(get_capabilities().hyperlinks);

    reset_capabilities_cache();
    set_capability_overrides(CapabilityOverrides::default());
}

#[test]
fn capability_overrides_force_a_protocol_over_detection() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    reset_capabilities_cache();
    set_capability_overrides(CapabilityOverrides {
        images: Some(Some(pi_tui::terminal_image::ImageProtocol::Kitty)),
        true_color: Some(true),
        ..CapabilityOverrides::default()
    });
    let caps = get_capabilities();
    assert_eq!(
        caps.images,
        Some(pi_tui::terminal_image::ImageProtocol::Kitty)
    );
    assert!(caps.true_color);
    reset_capabilities_cache();
    set_capability_overrides(CapabilityOverrides::default());
}

#[test]
fn the_cached_capabilities_answer_the_environment_detection_on_reset() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    reset_capabilities_cache();
    let env = pi_tui::terminal::default_env_lookup();
    let expected = detect_capabilities(env.as_ref(), &probe_tmux_hyperlinks);
    assert_eq!(get_capabilities(), expected);
    reset_capabilities_cache();
}

// --- the image-id allocator ---------------------------------------------------

#[test]
fn allocated_image_ids_stay_in_the_upstream_range() {
    for _ in 0..1000 {
        let id = allocate_image_id();
        assert!(id >= 1);
        assert!(id <= 0xffff_fffe);
    }
}

#[test]
fn consecutive_image_ids_differ() {
    let first = allocate_image_id();
    let second = allocate_image_id();
    assert_ne!(first, second);
}

// --- the Kitty encoder boundaries ---------------------------------------------

#[test]
fn kitty_payloads_at_the_chunk_boundary_transmit_in_one_command() {
    let sequence = encode_kitty(&"A".repeat(4096), EncodeKittyOptions::default());
    assert!(sequence.starts_with("\x1b_Ga=T,f=100,q=2;"));
    assert!(sequence.ends_with("\x1b\\"));
    assert!(!sequence.contains("m="));
}

#[test]
fn kitty_payloads_one_byte_past_the_boundary_split_into_two_chunks() {
    let sequence = encode_kitty(&"A".repeat(4097), EncodeKittyOptions::default());
    let first_end = sequence.find("\x1b\\").expect("first chunk terminates");
    let rest = &sequence[first_end + 2..];
    assert!(sequence.starts_with("\x1b_Ga=T,f=100,q=2,m=1;"));
    assert!(rest.starts_with("\x1b_Gm=0;"));
    assert_eq!(sequence.matches("\x1b_G").count(), 2);
}

#[test]
fn kitty_payloads_past_two_chunks_carry_m1_in_the_middle() {
    let sequence = encode_kitty(&"A".repeat(8193), EncodeKittyOptions::default());
    assert!(sequence.contains("\x1b_Ga=T,f=100,q=2,m=1;"));
    assert!(sequence.contains("\x1b_Gm=1;"));
    // 8193 = 4096 + 4096 + 1: the last chunk carries the single leftover A.
    assert!(sequence.ends_with("\x1b_Gm=0;A\x1b\\"));
    assert_eq!(sequence.matches("\x1b_G").count(), 3);
}

// --- the Iterm2 encoder boundaries ---------------------------------------------

#[test]
fn iterm2_size_carries_the_padded_payload_length() {
    // `size` is the last parameter without width/height, so the assert pins
    // up to the payload delimiter.
    let one = encode_iterm2("AA==", EncodeITerm2Options::default());
    assert!(one.contains(";size=1:"));
    let two = encode_iterm2("AAA=", EncodeITerm2Options::default());
    assert!(two.contains(";size=2:"));
    let four = encode_iterm2("AAAAAA==", EncodeITerm2Options::default());
    assert!(four.contains(";size=4:"));
}

#[test]
fn iterm2_inline_defaults_to_on_and_false_emits_inline_0() {
    assert!(
        encode_iterm2("AAAA", EncodeITerm2Options::default())
            .starts_with("\x1b]1337;File=inline=1;")
    );
    let off = encode_iterm2(
        "AAAA",
        EncodeITerm2Options {
            inline: Some(false),
            ..EncodeITerm2Options::default()
        },
    );
    assert!(off.starts_with("\x1b]1337;File=inline=0;"));
}

#[test]
fn iterm2_name_encodes_and_preserve_aspect_ratio_only_fires_when_false() {
    let named = encode_iterm2(
        "AAAA",
        EncodeITerm2Options {
            name: Some("shot.png".to_string()),
            ..EncodeITerm2Options::default()
        },
    );
    let name_base64 = base64::engine::general_purpose::STANDARD.encode(b"shot.png");
    assert!(named.contains(&format!("name={name_base64}:")));

    let preserved = encode_iterm2(
        "AAAA",
        EncodeITerm2Options {
            preserve_aspect_ratio: Some(true),
            ..EncodeITerm2Options::default()
        },
    );
    assert!(!preserved.contains("preserveAspectRatio"));

    let stretched = encode_iterm2(
        "AAAA",
        EncodeITerm2Options {
            preserve_aspect_ratio: Some(false),
            ..EncodeITerm2Options::default()
        },
    );
    assert!(stretched.contains(";preserveAspectRatio=0"));
}

// --- the image-size parsers -----------------------------------------------------

// --- the image-size parsers -----------------------------------------------------

fn png_bytes(width: u32, height: u32) -> String {
    let mut bytes = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&height.to_be_bytes());
    base64_of(bytes)
}

#[test]
fn png_dimensions_read_the_ihdr_extent() {
    assert_eq!(
        get_image_dimensions(&png_bytes(1280, 720), "image/png"),
        Some(ImageDimensions {
            width_px: 1280,
            height_px: 720,
        })
    );
}

#[test]
fn png_dimensions_reject_short_and_mismatched_payloads() {
    let short = base64_of(vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a]);
    assert_eq!(pi_tui::terminal_image::get_png_dimensions(&short), None);
    let wrong_magic = base64_of(vec![
        0x88, 0x50, 0x4e, 0x47, 0, 0, 0, 13, b'I', b'H', b'D', b'R',
    ]);
    assert_eq!(
        pi_tui::terminal_image::get_png_dimensions(&wrong_magic),
        None
    );
    assert_eq!(pi_tui::terminal_image::get_png_dimensions(""), None);
}

#[test]
fn jpeg_dimensions_read_the_sof0_to_sof2_frame_header() {
    let frame = |marker: u8| {
        let bytes = vec![
            0xff, 0xd8, 0xff, marker, 0x00, 0x11, 0x00, 0x02, 0xbc, 0x05, 0x00, 0x00,
        ];
        base64_of(bytes)
    };
    for marker in [0xc0, 0xc1, 0xc2] {
        assert_eq!(
            pi_tui::terminal_image::get_jpeg_dimensions(&frame(marker)),
            Some(ImageDimensions {
                width_px: 0x0500,
                height_px: 0x02bc,
            })
        );
    }
}

#[test]
fn jpeg_dimensions_walk_past_segments_and_non_marker_bytes() {
    // A non-marker byte steps the walk forward one byte before the SOF
    // answers at the shifted offsets.
    let bytes = vec![
        0xff, 0xd8, 0xaa, 0xff, 0xc0, 0x00, 0x00, 0x00, 0x02, 0xbc, 0x05, 0x00, 0x00,
    ];
    assert_eq!(
        pi_tui::terminal_image::get_jpeg_dimensions(&base64_of(bytes)),
        Some(ImageDimensions {
            width_px: 0x0500,
            height_px: 0x02bc,
        })
    );

    // An APP segment of length 6 (the two length bytes plus four data bytes)
    // is stepped over entirely before the SOF answers.
    let bytes = vec![
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x06, 0xd1, 0xd2, 0xd3, 0xd4, 0xff, 0xc0, 0x00, 0x00, 0x00,
        0x03, 0x84, 0x02, 0x80, 0x00,
    ];
    assert_eq!(
        pi_tui::terminal_image::get_jpeg_dimensions(&base64_of(bytes)),
        Some(ImageDimensions {
            width_px: 0x0280,
            height_px: 0x0384,
        })
    );
}

#[test]
fn jpeg_dimensions_reject_truncated_and_zero_length_segments() {
    assert_eq!(pi_tui::terminal_image::get_jpeg_dimensions(""), None);
    // A one-byte payload never enters the walk.
    assert_eq!(pi_tui::terminal_image::get_jpeg_dimensions("AA=="), None);

    let zero_length = vec![
        0xff, 0xd8, 0xff, 0xfe, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    assert_eq!(
        pi_tui::terminal_image::get_jpeg_dimensions(&base64_of(zero_length)),
        None
    );

    // A segment header that fits but whose length runs past the buffer ends
    // the walk with None.
    let truncated = vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x64, 0x00];
    assert_eq!(
        pi_tui::terminal_image::get_jpeg_dimensions(&base64_of(truncated)),
        None
    );
}

#[test]
fn gif_dimensions_read_both_signature_generations() {
    for signature in [b"GIF87a".as_slice(), b"GIF89a".as_slice()] {
        let mut bytes = signature.to_vec();
        bytes.extend_from_slice(&0x0500u16.to_le_bytes());
        bytes.extend_from_slice(&0x02d0u16.to_le_bytes());
        assert_eq!(
            pi_tui::terminal_image::get_gif_dimensions(&base64_of(bytes)),
            Some(ImageDimensions {
                width_px: 0x0500,
                height_px: 0x02d0,
            })
        );
    }

    let wrong = b"GIF89b".to_vec();
    assert_eq!(
        pi_tui::terminal_image::get_gif_dimensions(&base64_of(wrong)),
        None
    );
    assert_eq!(pi_tui::terminal_image::get_gif_dimensions("AAAA"), None);
}

#[test]
fn webp_dimensions_parse_the_vp8_vp8l_and_vp8x_chunks() {
    let mut lossy = b"RIFF\x00\x00\x00\x00WEBPVP8 ".to_vec();
    lossy.extend_from_slice(&[0; 10]);
    lossy.extend_from_slice(&0x0abcu16.to_le_bytes());
    lossy.extend_from_slice(&0x02d0u16.to_le_bytes());
    assert_eq!(
        pi_tui::terminal_image::get_webp_dimensions(&base64_of(lossy)),
        Some(ImageDimensions {
            width_px: 0x0abc & 0x3fff,
            height_px: 0x02d0 & 0x3fff,
        })
    );

    let mut lossless = b"RIFF\x00\x00\x00\x00WEBPVP8L\x00\x00\x00\x00\x00".to_vec();
    let bits: u32 = (720u32 - 1) << 14 | (1280u32 - 1);
    lossless.extend_from_slice(&bits.to_le_bytes());
    lossless.extend_from_slice(&[0; 5]);
    assert_eq!(
        pi_tui::terminal_image::get_webp_dimensions(&base64_of(lossless)),
        Some(ImageDimensions {
            width_px: 1280,
            height_px: 720,
        })
    );

    let mut extended = b"RIFF\x00\x00\x00\x00WEBPVP8X\x00\x00\x00\x00\x00\x00\x00\x00".to_vec();
    extended.extend_from_slice(&[255, 4, 0, 207, 2, 0]);
    assert_eq!(
        pi_tui::terminal_image::get_webp_dimensions(&base64_of(extended)),
        Some(ImageDimensions {
            width_px: 1280,
            height_px: 720,
        })
    );
}

#[test]
fn webp_dimensions_reject_missing_headers_and_unknown_chunks() {
    let mut riffless = b"JUNK\x00\x00\x00\x00WEBPVP8 ".to_vec();
    riffless.extend_from_slice(&[0; 18]);
    assert_eq!(
        pi_tui::terminal_image::get_webp_dimensions(&base64_of(riffless)),
        None
    );

    let mut unknown = b"RIFF\x00\x00\x00\x00WEBPXXXX".to_vec();
    unknown.extend_from_slice(&[0; 18]);
    assert_eq!(
        pi_tui::terminal_image::get_webp_dimensions(&base64_of(unknown)),
        None
    );

    assert_eq!(pi_tui::terminal_image::get_webp_dimensions("AAAA"), None);
}

#[test]
fn image_dimensions_dispatch_by_mime_type() {
    assert_eq!(
        get_image_dimensions(&png_bytes(64, 32), "image/png"),
        Some(ImageDimensions {
            width_px: 64,
            height_px: 32,
        })
    );
    assert_eq!(get_image_dimensions(&png_bytes(64, 32), "image/webp"), None);
    assert_eq!(
        get_image_dimensions(&png_bytes(64, 32), "image/svg+xml"),
        None
    );
    assert_eq!(get_image_dimensions("", "image/jpeg"), None);
}

// --- the cell-size math -----------------------------------------------------------

#[test]
fn cell_size_clamps_zero_and_minimum_extents() {
    let clamped = calculate_image_cell_size(
        ImageDimensions {
            width_px: 100,
            height_px: 100,
        },
        0,
        None,
        pi_tui::terminal_image::CellDimensions {
            width_px: 9,
            height_px: 18,
        },
    );
    assert_eq!(clamped.columns, 1);

    // A zero-pixel image scales as 1x1, so the width bound dominates.
    let zero_image = calculate_image_cell_size(
        ImageDimensions::default(),
        10,
        None,
        pi_tui::terminal_image::CellDimensions {
            width_px: 9,
            height_px: 18,
        },
    );
    assert_eq!(zero_image.columns, 10);
    assert_eq!(zero_image.rows, 5);
}

#[test]
fn cell_size_without_a_height_bound_scales_by_width_alone() {
    let size = calculate_image_cell_size(
        ImageDimensions {
            width_px: 100,
            height_px: 100,
        },
        3,
        None,
        pi_tui::terminal_image::CellDimensions {
            width_px: 9,
            height_px: 18,
        },
    );
    assert_eq!(size.columns, 3);
    assert_eq!(size.rows, 2);
}

#[test]
fn cell_size_clamps_columns_and_rows_to_the_maxima() {
    // The uniform scale fits both bounds: the 10x10 image in 1x1 cells under
    // (3, 2) lands on 2x2, then the clamps hold.
    let size = calculate_image_cell_size(
        ImageDimensions {
            width_px: 10,
            height_px: 10,
        },
        3,
        Some(2),
        pi_tui::terminal_image::CellDimensions {
            width_px: 1,
            height_px: 1,
        },
    );
    assert_eq!(size.columns, 2);
    assert_eq!(size.rows, 2);

    let roomy = calculate_image_cell_size(
        ImageDimensions {
            width_px: 10,
            height_px: 10,
        },
        10,
        Some(100),
        pi_tui::terminal_image::CellDimensions {
            width_px: 10,
            height_px: 10,
        },
    );
    assert_eq!(roomy.columns, 10);
    assert_eq!(roomy.rows, 10);
}

#[test]
fn image_rows_answers_the_unconstrained_height() {
    assert_eq!(
        calculate_image_rows(
            ImageDimensions {
                width_px: 100,
                height_px: 100,
            },
            3,
            pi_tui::terminal_image::CellDimensions {
                width_px: 9,
                height_px: 18,
            },
        ),
        2
    );
}

// --- the renderer and the placement machinery ---------------------------------------

#[test]
fn render_image_answers_none_without_an_image_protocol() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities::default());
    let rendered = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
        &ImageRenderOptions::default(),
    );
    assert_eq!(rendered, None);
    reset_capabilities_cache();
}

#[test]
fn render_image_uses_the_iterm2_encoder_with_auto_height() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Iterm2),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let rendered = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
        &ImageRenderOptions {
            max_width_cells: Some(2),
            ..ImageRenderOptions::default()
        },
    )
    .expect("iterm2 caps render a placement");
    assert_eq!(
        rendered.sequence,
        "\x1b]1337;File=inline=1;size=3;width=2;height=auto:AAAA\x07"
    );
    assert_eq!(rendered.columns, 2);
    assert_eq!(rendered.rows, 2);
    assert_eq!(rendered.image_id, None);
    reset_capabilities_cache();
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn render_image_registers_explicit_kitty_ids_and_defaults_to_cursor_movement() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let rendered = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
        &ImageRenderOptions {
            max_width_cells: Some(2),
            image_id: Some(7),
            ..ImageRenderOptions::default()
        },
    )
    .expect("kitty caps render a placement");
    assert!(!rendered.sequence.contains(",C=1,"));
    assert_eq!(rendered.image_id, Some(7));
    assert_eq!(
        get_kitty_image_metadata(&rendered.sequence),
        Some(KittyImageMetadata {
            image_id: 7,
            columns: 2,
            rows: 2,
            width_px: 20,
            height_px: 20,
        })
    );

    let unregistered = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
        &ImageRenderOptions {
            max_width_cells: Some(2),
            ..ImageRenderOptions::default()
        },
    )
    .expect("kitty caps render a placement");
    assert_eq!(get_kitty_image_metadata(&unregistered.sequence), None);
    assert_eq!(unregistered.image_id, None);

    reset_capabilities_cache();
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn placement_extraction_answers_none_for_unregistered_and_malformed_lines() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    reset_capabilities_cache();
    // No metadata for the id: extraction refuses.
    assert_eq!(
        get_kitty_image_placement("left \x1b_Ga=T,f=100,i=424242;AAAA\x1b\\ right"),
        None
    );
    // A registered id with no terminator never yields a placement.
    register_kitty_image_metadata(KittyImageMetadata {
        image_id: 424_243,
        columns: 1,
        rows: 1,
        width_px: 10,
        height_px: 10,
    });
    assert_eq!(
        get_kitty_image_placement("\x1b_Ga=T,f=100,i=424243;AAA"),
        None
    );
    // A chunked transmission whose continuation is missing never yields one.
    let broken = "\x1b_Ga=T,f=100,i=424243,m=1;AAA\x1b\\ tail".to_string();
    assert_eq!(get_kitty_image_placement(&broken), None);
    // A chunk chain whose next command is not a Kitty command either.
    let broken_chain = "\x1b_Ga=T,f=100,i=424243,m=1;AAA\x1b\\ not-kitty".to_string();
    assert_eq!(get_kitty_image_placement(&broken_chain), None);
    reset_capabilities_cache();
}

#[test]
fn placement_sequences_drop_transmission_only_controls() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    register_kitty_image_metadata(KittyImageMetadata {
        image_id: 424_244,
        columns: 2,
        rows: 2,
        width_px: 20,
        height_px: 20,
    });
    let line = "\x1b_Ga=T,f=100,q=2,C=1,c=2,r=2,i=424244;AAAA\x1b\\";
    let placement =
        get_kitty_image_placement(line).expect("registered metadata yields a placement");
    assert_eq!(
        placement.sequence,
        "\x1b_Ga=p,q=2,C=1,c=2,r=2,i=424244\x1b\\"
    );
    assert_eq!(placement.replacement_line, placement.sequence);
    assert_eq!(placement.transmission_bytes, line.len());
    assert_eq!(placement.estimated_decoded_bytes, 20 * 20 * 4);
    reset_capabilities_cache();
}

#[test]
fn cropping_leaves_unregistered_whole_and_out_of_range_placements_untouched() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    reset_capabilities_cache();
    let unregistered = "\x1b_Ga=T,f=100,i=424245;AAAA\x1b\\";
    assert_eq!(
        crop_kitty_image_line(unregistered, 1, 1),
        unregistered.to_string()
    );

    register_kitty_image_metadata(KittyImageMetadata {
        image_id: 424_246,
        columns: 2,
        rows: 2,
        width_px: 20,
        height_px: 20,
    });
    let whole = "\x1b_Ga=T,f=100,q=2,c=2,r=2,i=424246;AAAA\x1b\\";
    assert_eq!(crop_kitty_image_line(whole, 2, 1), whole.to_string());
    assert_eq!(crop_kitty_image_line(whole, 0, 0), whole.to_string());
    assert_eq!(crop_kitty_image_line(whole, 0, 2), whole.to_string());

    let cropped = crop_kitty_image_line(whole, 1, 1);
    assert!(cropped.contains("y=10,h=10,r=1"));
    assert!(!cropped.contains("c=2,r=2,i=424246;"));
    reset_capabilities_cache();
}

#[test]
fn the_metadata_registry_evicts_its_oldest_entry_past_1000_ids() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    reset_capabilities_cache();
    for id in 1..=1001u64 {
        register_kitty_image_metadata(KittyImageMetadata {
            image_id: id,
            columns: 1,
            rows: 1,
            width_px: 10,
            height_px: 10,
        });
    }
    assert_eq!(get_kitty_image_metadata("\x1b_Gi=1;"), None);
    assert_eq!(
        get_kitty_image_metadata("\x1b_Gi=1001;"),
        Some(KittyImageMetadata {
            image_id: 1001,
            columns: 1,
            rows: 1,
            width_px: 10,
            height_px: 10,
        })
    );
    // Re-registering moves the id to the newest slot; the next insert evicts
    // the id that became oldest, not the re-registered one.
    register_kitty_image_metadata(KittyImageMetadata {
        image_id: 2,
        columns: 2,
        rows: 2,
        width_px: 20,
        height_px: 20,
    });
    register_kitty_image_metadata(KittyImageMetadata {
        image_id: 1002,
        columns: 3,
        rows: 3,
        width_px: 30,
        height_px: 30,
    });
    assert_eq!(get_kitty_image_metadata("\x1b_Gi=3;"), None);
    assert_eq!(
        get_kitty_image_metadata("\x1b_Gi=2;"),
        Some(KittyImageMetadata {
            image_id: 2,
            columns: 2,
            rows: 2,
            width_px: 20,
            height_px: 20,
        })
    );
    reset_capabilities_cache();
}

// --- the Image component --------------------------------------------------------

#[test]
fn image_component_defaults_to_800x600_for_unknown_payloads() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities::default());
    let image = Image::new(
        "not-even-base64",
        "image/png",
        theme(),
        ImageOptions::default(),
    );
    let lines = image.render(40);
    assert_eq!(lines, vec!["[Image: [image/png] 800x600]"]);
    reset_capabilities_cache();
}

#[test]
fn image_component_caches_by_width_and_invalidate_clears_it() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities::default());
    let image = Image::with_dimensions(
        "AAAA",
        "image/png",
        theme(),
        ImageOptions::default(),
        ImageDimensions {
            width_px: 800,
            height_px: 600,
        },
    );
    let first = image.render(40);
    let second = image.render(40);
    assert_eq!(first, second);
    assert!(!image.render(30).is_empty());
    image.invalidate();
    assert_eq!(image.render(40), first);
    reset_capabilities_cache();
}

#[test]
fn image_component_uses_the_parsed_payload_dimensions() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities::default());
    let image = Image::new(
        &png_bytes(64, 32),
        "image/png",
        theme(),
        ImageOptions::default(),
    );
    let lines = image.render(40);
    assert_eq!(lines, vec!["[Image: [image/png] 64x32]"]);
    reset_capabilities_cache();
}

#[test]
fn image_component_reuses_a_provided_kitty_id_across_renders() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let image = Image::with_dimensions(
        "AAAA",
        "image/png",
        theme(),
        ImageOptions {
            image_id: Some(424_247),
            ..ImageOptions::default()
        },
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
    );
    assert_eq!(image.get_image_id(), Some(424_247));
    let lines = image.render(4);
    assert!(lines[0].contains(",i=424247"));
    image.invalidate();
    let re_rendered = image.render(4);
    assert_eq!(re_rendered, lines);
    assert_eq!(image.get_image_id(), Some(424_247));
    reset_capabilities_cache();
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn image_component_uses_the_iterm2_cursor_up_prefix_on_the_last_line() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Iterm2),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let image = Image::with_dimensions(
        "AAAA",
        "image/png",
        theme(),
        ImageOptions::default(),
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
    );
    let lines = image.render(4);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0], "");
    assert!(lines[1].starts_with("\x1b[1A"));
    assert!(lines[1].contains("\x1b]1337;File="));
    reset_capabilities_cache();
    set_cell_dimensions(pi_tui::terminal_image::CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn fallback_shortens_the_home_path_itself_to_a_tilde() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let home = std::env::home_dir()
        .expect("home directory resolves")
        .to_string_lossy()
        .to_string();
    assert_eq!(
        image_fallback(
            "image/png",
            Some(ImageDimensions {
                width_px: 8,
                height_px: 6,
            }),
            Some(&home),
        ),
        "[Image: ~ [image/png] 8x6]"
    );
    reset_capabilities_cache();
}

#[test]
fn image_fallback_percent_encodes_the_file_url() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    });
    let spaced = std::env::home_dir()
        .expect("home directory resolves")
        .join("my shots")
        .join("shot.png")
        .to_string_lossy()
        .to_string();
    let result = image_fallback(
        "image/png",
        Some(ImageDimensions {
            width_px: 1,
            height_px: 1,
        }),
        Some(&spaced),
    );
    assert!(result.contains("%20"), "spaces percent-encode: {result}");
    reset_capabilities_cache();
}
