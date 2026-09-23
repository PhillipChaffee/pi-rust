//! Port of `packages/tui/test/terminal-image.test.ts` 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`). The upstream `describe`
//! groups become test functions with the upstream names.
//!
//! Restatements: upstream's `withEnv` mutates `process.env`; Rust cannot
//! mutate the process environment without the `unsafe` this workspace
//! forbids, so every detection test injects a map-backed env lookup through
//! [`detect_with`] (keys absent from the map read as cleared, exactly the
//! `withEnv` semantics). The global stores (capability cache, Kitty
//! metadata registry, cell dimensions) sit behind [`GLOBALS_LOCK`] the way
//! the sibling suites hold theirs, because the ported upstream mutations are
//! process-global.
#![expect(
    clippy::expect_used,
    reason = "a missing home directory or missing fixture in a test environment is a test failure; expecting keeps the assertions readable"
)]

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use pi_tui::components::image::{Image, ImageOptions, ImageTheme};
use pi_tui::terminal_image::{
    CapabilityOverrides, CellDimensions, EncodeITerm2Options, EncodeKittyOptions, ITerm2Size,
    ImageDimensions, ImageProtocol, ImageRenderOptions, KittyImageMetadata, TerminalCapabilities,
    delete_all_kitty_images, delete_all_kitty_placements, delete_kitty_image, detect_capabilities,
    encode_iterm2, encode_kitty, get_capabilities, get_kitty_image_metadata,
    get_kitty_image_placement, hyperlink, image_fallback, is_image_line, probe_tmux_hyperlinks,
    register_kitty_image_metadata, render_image, reset_capabilities_cache, set_capabilities,
    set_capability_overrides, set_cell_dimensions,
};
use pi_tui::tui::Component;
use pi_tui::utils::visible_width;

static GLOBALS_LOCK: Mutex<()> = Mutex::new(());

fn detect_with(
    env_overrides: &[(&str, &str)],
    tmux_forwards_hyperlink: &dyn Fn() -> bool,
) -> TerminalCapabilities {
    let overrides: HashMap<&str, String> = env_overrides
        .iter()
        .map(|(key, value)| (*key, value.to_string()))
        .collect();
    let env = move |key: &str| overrides.get(key).cloned();
    detect_capabilities(&env, tmux_forwards_hyperlink)
}

#[expect(
    clippy::panic,
    reason = "the probe must never run while PI_HYPERLINKS forces the answer; the test pins the bypass"
)]
fn never_probe() -> bool {
    panic!("the tmux probe must not run without TMUX set")
}

fn theme() -> ImageTheme {
    ImageTheme {
        fallback_color: Arc::new(str::to_string),
    }
}

fn colored_theme() -> ImageTheme {
    ImageTheme {
        fallback_color: Arc::new(|value| format!("\x1b[33m{value}\x1b[0m")),
    }
}

// --- upstream describe("isImageLine") ------------------------------------

#[test]
fn iterm2_image_protocol_detects_image_escape_sequence_at_start_of_line() {
    let iterm2_image_line = "\x1b]1337;File=size=100,100;inline=1:base64encodeddata==\x07";
    assert!(is_image_line(iterm2_image_line));
}

#[test]
fn iterm2_image_protocol_detects_image_escape_sequence_with_text_before_it() {
    let line_with_text_and_image =
        "Some text \x1b]1337;File=size=100,100;inline=1:base64data==\x07 more text";
    assert!(is_image_line(line_with_text_and_image));
}

#[test]
fn iterm2_image_protocol_detects_image_escape_sequence_in_middle_of_long_line() {
    let long_line_with_image =
        "Text before image...\x1b]1337;File=inline=1:verylongbase64data==...text after";
    assert!(is_image_line(long_line_with_image));
}

#[test]
fn iterm2_image_protocol_detects_image_escape_sequence_at_end_of_line() {
    let line_with_image_at_end =
        "Regular text ending with \x1b]1337;File=inline=1:base64data==\x07";
    assert!(is_image_line(line_with_image_at_end));
}

#[test]
fn iterm2_image_protocol_detects_minimal_image_escape_sequence() {
    assert!(is_image_line("\x1b]1337;File=:\x07"));
}

#[test]
fn kitty_image_protocol_detects_image_escape_sequence_at_start_of_line() {
    let kitty_image_line = "\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\";
    assert!(is_image_line(kitty_image_line));
}

#[test]
fn kitty_image_protocol_detects_image_escape_sequence_with_text_before_it() {
    let line_with_text_and_kitty_image = "Output: \x1b_Ga=T,f=100;data...\x1b\\\x1b_Gm=i=1;\x1b\\";
    assert!(is_image_line(line_with_text_and_kitty_image));
}

#[test]
fn kitty_image_protocol_detects_image_escape_sequence_with_padding() {
    let kitty_with_padding = "  \x1b_Ga=T,f=100...\x1b\\\x1b_Gm=i=1;\x1b\\  ";
    assert!(is_image_line(kitty_with_padding));
}

#[test]
fn bug_regression_tests_detect_image_sequences_in_very_long_lines() {
    let base64_char = "A".repeat(100);
    let image_sequence = "\x1b]1337;File=size=800,600;inline=1:";
    let long_line = format!(
        "Text prefix {image_sequence}{} suffix",
        base64_char.repeat(3000)
    );
    assert!(long_line.len() > 300_000);
    assert!(is_image_line(&long_line));
}

#[test]
fn bug_regression_tests_detect_image_sequences_when_terminal_does_not_support_images() {
    let line_with_image = "Read image file [image/jpeg]\x1b]1337;File=inline=1:base64data==\x07";
    assert!(is_image_line(line_with_image));
}

#[test]
fn bug_regression_tests_detect_image_sequences_with_ansi_codes_before_them() {
    let line_with_ansi_and_image = "\x1b[31mError output \x1b]1337;File=inline=1:image==\x07";
    assert!(is_image_line(line_with_ansi_and_image));
}

#[test]
fn bug_regression_tests_detect_image_sequences_with_ansi_codes_after_them() {
    let line_with_image_and_ansi = "\x1b_Ga=T,f=100:data...\x1b\\\x1b_Gm=i=1;\x1b\\\x1b[0m reset";
    assert!(is_image_line(line_with_image_and_ansi));
}

#[test]
fn negative_cases_should_not_detect_images_in_plain_text_lines() {
    assert!(!is_image_line(
        "This is just a regular text line without any escape sequences"
    ));
}

#[test]
fn negative_cases_should_not_detect_images_in_lines_with_only_ansi_codes() {
    let ansi_text = "\x1b[31mRed text\x1b[0m and \x1b[32mgreen text\x1b[0m";
    assert!(!is_image_line(ansi_text));
}

#[test]
fn negative_cases_should_not_detect_images_in_lines_with_cursor_movement_codes() {
    let cursor_codes = "\x1b[1A\x1b[2KLine cleared and moved up";
    assert!(!is_image_line(cursor_codes));
}

#[test]
fn negative_cases_should_not_detect_images_in_lines_with_partial_iterm2_sequences() {
    let partial_sequence = "Some text with ]1337;File but missing ESC at start";
    assert!(!is_image_line(partial_sequence));
}

#[test]
fn negative_cases_should_not_detect_images_in_lines_with_partial_kitty_sequences() {
    let partial_sequence = "Some text with _G but missing ESC at start";
    assert!(!is_image_line(partial_sequence));
}

#[test]
fn negative_cases_should_not_detect_images_in_empty_lines() {
    assert!(!is_image_line(""));
}

#[test]
fn negative_cases_should_not_detect_images_in_lines_with_newlines_only() {
    assert!(!is_image_line("\n"));
    assert!(!is_image_line("\n\n"));
}

#[test]
fn mixed_content_scenarios_detect_images_when_line_has_both_kitty_and_iterm2_sequences() {
    let mixed_line =
        "Kitty: \x1b_Ga=T...\x1b\\\x1b_Gm=i=1;\x1b\\ iTerm2: \x1b]1337;File=inline=1:data==\x07";
    assert!(is_image_line(mixed_line));
}

#[test]
fn mixed_content_scenarios_detect_image_in_line_with_multiple_text_and_image_segments() {
    let complex_line = "Start \x1b]1337;File=img1==\x07 middle \x1b]1337;File=img2==\x07 end";
    assert!(is_image_line(complex_line));
}

#[test]
fn mixed_content_scenarios_do_not_falsely_detect_image_in_line_with_file_path_keywords() {
    let file_path_line = "/path/to/File_1337_backup/image.jpg";
    assert!(!is_image_line(file_path_line));
}

// --- upstream describe("detectCapabilities") ------------------------------

#[test]
fn detect_defaults_to_hyperlinks_false_for_unknown_terminals() {
    let caps = detect_with(&[], &probe_tmux_hyperlinks);
    assert!(!caps.hyperlinks);
    assert_eq!(caps.images, None);
}

#[test]
fn detect_applies_environment_overrides() {
    assert_eq!(
        detect_with(
            &[
                ("PI_HYPERLINKS", "1"),
                ("PI_IMAGE_PROTOCOL", "kitty"),
                ("PI_TRUE_COLOR", "1")
            ],
            &never_probe,
        ),
        TerminalCapabilities {
            images: Some(ImageProtocol::Kitty),
            true_color: true,
            hyperlinks: true,
        }
    );
    assert_eq!(
        detect_with(
            &[
                ("TERM_PROGRAM", "iterm.app"),
                ("PI_HYPERLINKS", "0"),
                ("PI_IMAGE_PROTOCOL", "none"),
                ("PI_TRUE_COLOR", "0"),
            ],
            &never_probe,
        ),
        TerminalCapabilities {
            images: None,
            true_color: false,
            hyperlinks: false,
        }
    );
}

#[test]
fn detect_preserves_auto_detection_for_auto_environment_overrides() {
    assert_eq!(
        detect_with(
            &[
                ("TERM_PROGRAM", "ghostty"),
                ("PI_HYPERLINKS", "auto"),
                ("PI_IMAGE_PROTOCOL", "auto"),
                ("PI_TRUE_COLOR", "auto"),
            ],
            &never_probe,
        ),
        TerminalCapabilities {
            images: Some(ImageProtocol::Kitty),
            true_color: true,
            hyperlinks: true,
        }
    );
}

#[test]
fn detect_applies_and_clears_programmatic_overrides() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    // Upstream runs this test inside a withEnv that forces the detection
    // baseline through PI_* variables; getCapabilities reads the real
    // process environment, so the cleared branch pins against the same
    // detection the cache runs instead of the injected baseline.
    set_capability_overrides(CapabilityOverrides {
        images: Some(None),
        true_color: Some(false),
        hyperlinks: Some(false),
    });
    assert_eq!(
        get_capabilities(),
        TerminalCapabilities {
            images: None,
            true_color: false,
            hyperlinks: false,
        }
    );
    set_capability_overrides(CapabilityOverrides::default());
    let env = pi_tui::terminal::default_env_lookup();
    assert_eq!(
        get_capabilities(),
        detect_capabilities(env.as_ref(), &probe_tmux_hyperlinks)
    );
    set_capability_overrides(CapabilityOverrides::default());
    reset_capabilities_cache();
}

#[test]
fn detect_bypasses_the_tmux_probe_when_hyperlinks_are_overridden() {
    let probed = Cell::new(false);
    let probe = || {
        probed.set(true);
        false
    };
    let caps = detect_with(
        &[
            ("TMUX", "/tmp/tmux-1000/default,1234,0"),
            ("PI_HYPERLINKS", "1"),
            ("PI_IMAGE_PROTOCOL", "kitty"),
        ],
        &probe,
    );
    assert!(!probed.get());
    assert!(caps.hyperlinks);
    assert_eq!(caps.images, Some(ImageProtocol::Kitty));
}

#[test]
fn detect_enables_hyperlinks_under_tmux_when_the_client_forwards_them() {
    let caps = detect_with(
        &[
            ("TMUX", "/tmp/tmux-1000/default,1234,0"),
            ("TERM_PROGRAM", "ghostty"),
        ],
        &|| true,
    );
    assert!(caps.hyperlinks);
    assert_eq!(caps.images, None);
}

#[test]
fn detect_disables_hyperlinks_under_tmux_when_the_client_does_not_forward_them() {
    let caps = detect_with(
        &[
            ("TMUX", "/tmp/tmux-1000/default,1234,0"),
            ("TERM_PROGRAM", "ghostty"),
        ],
        &|| false,
    );
    assert!(!caps.hyperlinks);
    assert_eq!(caps.images, None);
}

#[test]
fn detect_checks_tmux_capability_when_term_starts_with_tmux() {
    let caps = detect_with(
        &[("TERM", "tmux-256color"), ("TERM_PROGRAM", "iterm.app")],
        &|| true,
    );
    assert!(caps.hyperlinks);
    assert_eq!(caps.images, None);

    let caps = detect_with(
        &[("TERM", "tmux-256color"), ("TERM_PROGRAM", "iterm.app")],
        &|| false,
    );
    assert!(!caps.hyperlinks);
}

#[test]
fn detect_forces_hyperlinks_false_when_term_starts_with_screen() {
    let caps = detect_with(&[("TERM", "screen-256color")], &never_probe);
    assert!(!caps.hyperlinks);
    assert_eq!(caps.images, None);
}

#[test]
fn detect_enables_hyperlinks_for_ghostty() {
    let caps = detect_with(&[("TERM_PROGRAM", "ghostty")], &never_probe);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_does_not_disable_ghostty_images_solely_because_cmux_is_present() {
    let caps = detect_with(
        &[
            ("TERM_PROGRAM", "ghostty"),
            ("CMUX_WORKSPACE_ID", "workspace"),
        ],
        &never_probe,
    );
    assert_eq!(caps.images, Some(ImageProtocol::Kitty));
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_hyperlinks_for_kitty() {
    let caps = detect_with(&[("KITTY_WINDOW_ID", "1")], &never_probe);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_hyperlinks_for_wezterm() {
    let caps = detect_with(&[("WEZTERM_PANE", "0")], &never_probe);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_images_and_hyperlinks_for_warp_via_term_program() {
    let caps = detect_with(&[("TERM_PROGRAM", "WarpTerminal")], &never_probe);
    assert_eq!(caps.images, Some(ImageProtocol::Kitty));
    assert!(caps.true_color);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_images_and_hyperlinks_for_warp_via_warp_session_id() {
    let caps = detect_with(&[("WARP_SESSION_ID", "some-session-id")], &never_probe);
    assert_eq!(caps.images, Some(ImageProtocol::Kitty));
    assert!(caps.true_color);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_images_and_hyperlinks_for_warp_via_warp_terminal_session_uuid() {
    let caps = detect_with(
        &[(
            "WARP_TERMINAL_SESSION_UUID",
            "d0e1a2e5-7ca7-44cd-9037-ac7222011161",
        )],
        &never_probe,
    );
    assert_eq!(caps.images, Some(ImageProtocol::Kitty));
    assert!(caps.true_color);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_disables_images_for_warp_inside_tmux() {
    let caps = detect_with(
        &[
            ("TERM_PROGRAM", "WarpTerminal"),
            ("TMUX", "/tmp/tmux-1000/default,1234,0"),
            ("TERM", "tmux-256color"),
        ],
        &|| true,
    );
    assert_eq!(caps.images, None);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_hyperlinks_for_iterm2() {
    let caps = detect_with(&[("TERM_PROGRAM", "iterm.app")], &never_probe);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_hyperlinks_for_vscode() {
    let caps = detect_with(&[("TERM_PROGRAM", "vscode")], &never_probe);
    assert!(caps.hyperlinks);
}

#[test]
fn detect_enables_alacritty_capabilities_for_zed() {
    assert_eq!(
        detect_with(&[("TERM_PROGRAM", "zed")], &never_probe),
        TerminalCapabilities {
            images: None,
            true_color: true,
            hyperlinks: true,
        }
    );
}

#[test]
fn detect_enables_truecolor_and_hyperlinks_for_windows_terminal_outside_multiplexers() {
    let caps = detect_with(
        &[("WT_SESSION", "session"), ("TERM", "xterm-256color")],
        &never_probe,
    );
    assert!(caps.true_color);
    assert!(caps.hyperlinks);
    assert_eq!(caps.images, None);
}

#[test]
fn detect_enables_truecolor_without_hyperlinks_for_jetbrains_terminal() {
    let caps = detect_with(
        &[
            ("TERMINAL_EMULATOR", "JetBrains-JediTerm"),
            ("TERM", "xterm-256color"),
        ],
        &never_probe,
    );
    assert!(caps.true_color);
    assert!(!caps.hyperlinks);
    assert_eq!(caps.images, None);
}

#[test]
fn detect_does_not_inherit_windows_terminal_truecolor_through_tmux() {
    let caps = detect_with(
        &[
            ("WT_SESSION", "session"),
            ("TMUX", "/tmp/tmux-1000/default,1234,0"),
            ("TERM", "tmux-256color"),
        ],
        &|| false,
    );
    assert!(!caps.true_color);
    assert!(!caps.hyperlinks);
    assert_eq!(caps.images, None);
}

#[test]
fn detect_trusts_explicit_truecolor_hints_through_tmux() {
    let caps = detect_with(
        &[
            ("COLORTERM", "truecolor"),
            ("TMUX", "/tmp/tmux-1000/default,1234,0"),
            ("TERM", "tmux-256color"),
        ],
        &|| false,
    );
    assert!(caps.true_color);
    assert!(!caps.hyperlinks);
    assert_eq!(caps.images, None);
}

// --- upstream describe("iTerm2 image encoding") ----------------------------

#[test]
fn iterm2_encoding_includes_the_decoded_payload_size_in_osc_1337_metadata() {
    let sequence = encode_iterm2(
        "AAAA",
        EncodeITerm2Options {
            width: Some(ITerm2Size::Cells(2)),
            height: Some(ITerm2Size::Auto),
            ..EncodeITerm2Options::default()
        },
    );
    assert_eq!(
        sequence,
        "\x1b]1337;File=inline=1;size=3;width=2;height=auto:AAAA\x07"
    );
}

// --- upstream describe("Kitty image cursor movement") ----------------------

#[test]
fn kitty_can_request_no_terminal_side_cursor_movement() {
    let sequence = encode_kitty(
        "AAAA",
        EncodeKittyOptions {
            columns: Some(2),
            rows: Some(2),
            move_cursor: Some(false),
            ..EncodeKittyOptions::default()
        },
    );
    assert!(sequence.starts_with("\x1b_Ga=T,f=100,q=2,C=1,c=2,r=2;"));
}

#[test]
fn kitty_suppresses_kitty_replies_for_delete_commands() {
    assert_eq!(delete_kitty_image(42), "\x1b_Ga=d,d=I,i=42,q=2\x1b\\");
    assert_eq!(delete_all_kitty_images(), "\x1b_Ga=d,d=A,q=2\x1b\\");
    assert_eq!(delete_all_kitty_placements(), "\x1b_Ga=d,d=a,q=2\x1b\\");
}

#[test]
fn kitty_preserves_render_image_default_terminal_side_cursor_movement() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let result = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
        &ImageRenderOptions {
            max_width_cells: Some(2),
            ..ImageRenderOptions::default()
        },
    );
    let result = result.expect("kitty caps render a placement");
    assert!(!result.sequence.contains(",C=1,"));
    assert_eq!(result.rows, 2);
    reset_capabilities_cache();
    set_cell_dimensions(CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn kitty_can_opt_render_image_into_no_terminal_side_cursor_movement() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let result = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
        &ImageRenderOptions {
            max_width_cells: Some(2),
            move_cursor: Some(false),
            ..ImageRenderOptions::default()
        },
    );
    let result = result.expect("kitty caps render a placement");
    assert!(result.sequence.contains(",C=1,"));
    assert_eq!(result.rows, 2);
    reset_capabilities_cache();
    set_cell_dimensions(CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn kitty_registers_metadata_and_crops_a_partially_visible_placement() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let result = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 100,
            height_px: 100,
        },
        &ImageRenderOptions {
            max_width_cells: Some(3),
            image_id: Some(42),
            move_cursor: Some(false),
            ..ImageRenderOptions::default()
        },
    );
    let result = result.expect("kitty caps render a placement");
    assert_eq!(
        get_kitty_image_metadata(&result.sequence),
        Some(KittyImageMetadata {
            image_id: 42,
            columns: 3,
            rows: 3,
            width_px: 100,
            height_px: 100,
        })
    );
    assert!(
        pi_tui::terminal_image::crop_kitty_image_line(&result.sequence, 2, 1)
            .contains("y=66,h=34,r=1")
    );
    reset_capabilities_cache();
    set_cell_dimensions(CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn kitty_creates_placement_only_commands_for_uploaded_and_cropped_images() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    register_kitty_image_metadata(KittyImageMetadata {
        image_id: 42,
        columns: 3,
        rows: 3,
        width_px: 100,
        height_px: 100,
    });
    let transmission = encode_kitty(
        &"A".repeat(8192),
        EncodeKittyOptions {
            columns: Some(3),
            rows: Some(3),
            image_id: Some(42),
            move_cursor: Some(false),
        },
    );
    let cropped = pi_tui::terminal_image::crop_kitty_image_line(&transmission, 2, 1);
    let line = format!("left {cropped} right");
    let placement =
        get_kitty_image_placement(&line).expect("registered metadata yields a placement");
    assert_eq!(
        placement.transmission_bytes,
        line.len() - "left ".len() - " right".len()
    );
    assert_eq!(placement.estimated_decoded_bytes, 100 * 100 * 4);
    assert_eq!(
        placement.sequence,
        "\x1b_Ga=p,q=2,C=1,c=3,i=42,y=66,h=34,r=1\x1b\\"
    );
    assert_eq!(
        placement.replacement_line,
        format!("left {} right", placement.sequence)
    );
    assert!(!placement.replacement_line.contains("AAAA"));
    reset_capabilities_cache();
}

#[test]
fn kitty_honors_max_height_cells_by_reducing_rendered_width() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let result = render_image(
        "AAAA",
        ImageDimensions {
            width_px: 10,
            height_px: 100,
        },
        &ImageRenderOptions {
            max_width_cells: Some(10),
            max_height_cells: Some(5),
            ..ImageRenderOptions::default()
        },
    );
    let result = result.expect("kitty caps render a placement");
    assert_eq!(result.rows, 5);
    assert!(result.sequence.contains(",c=1,r=5"));
    reset_capabilities_cache();
    set_cell_dimensions(CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn kitty_caps_image_component_height_to_a_square_pixel_box_by_default() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(CellDimensions {
        width_px: 10,
        height_px: 20,
    });
    let image = Image::with_dimensions(
        "AAAA",
        "image/png",
        theme(),
        ImageOptions {
            max_width_cells: Some(10),
            ..ImageOptions::default()
        },
        ImageDimensions {
            width_px: 10,
            height_px: 100,
        },
    );
    let lines = image.render(12);
    assert_eq!(lines.len(), 5);
    assert!(lines[0].contains(",c=1,r=5"));
    reset_capabilities_cache();
    set_cell_dimensions(CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn kitty_places_image_sequence_on_first_line_with_empty_padding_rows() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    set_cell_dimensions(CellDimensions {
        width_px: 10,
        height_px: 10,
    });
    let image = Image::with_dimensions(
        "AAAA",
        "image/png",
        theme(),
        ImageOptions {
            max_width_cells: Some(2),
            ..ImageOptions::default()
        },
        ImageDimensions {
            width_px: 20,
            height_px: 20,
        },
    );
    let lines = image.render(4);
    let image_id = image.get_image_id();
    assert!(image_id.is_some());
    assert!(lines[0].starts_with("\x1b_G"));
    assert!(lines[0].contains(",C=1,"));
    assert!(lines[0].contains(&format!(
        ",i={}",
        image_id.expect("kitty render allocates the id")
    )));
    assert!(lines[0].ends_with("\x1b\\"));
    assert_eq!(lines[1..], [""]);
    reset_capabilities_cache();
    set_cell_dimensions(CellDimensions {
        width_px: 9,
        height_px: 18,
    });
}

#[test]
fn kitty_truncates_long_image_fallback_lines_to_render_width() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let long_path = std::env::home_dir()
        .expect("home directory resolves")
        .join("images")
        .join(format!(
            "{}.png",
            "generated-image-with-a-very-long-absolute-path".repeat(4)
        ));
    let long_path = long_path.to_string_lossy().to_string();
    let width = 40;
    let image = Image::with_dimensions(
        "AAAA",
        "image/png",
        colored_theme(),
        ImageOptions {
            filename: Some(long_path),
            ..ImageOptions::default()
        },
        ImageDimensions {
            width_px: 1280,
            height_px: 720,
        },
    );
    let lines = image.render(width);
    assert_eq!(lines.len(), 1);
    assert!(
        visible_width(&lines[0]) <= width,
        "fallback line wider than {width}: visible={} raw={lines:?}",
        visible_width(&lines[0])
    );
    assert!(
        lines[0].contains("..."),
        "expected ellipsis when truncating long fallback path"
    );
    assert!(
        lines[0].contains('~'),
        "expected home-shortened path in fallback"
    );
    reset_capabilities_cache();
}

// --- upstream describe("imageFallback") ------------------------------------

#[test]
fn fallback_shortens_home_prefixed_absolute_paths_without_hyperlinks() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let home = std::env::home_dir().expect("home directory resolves");
    let abs = home
        .join(".pi")
        .join("agent")
        .join("shot.png")
        .to_string_lossy()
        .to_string();
    let result = image_fallback(
        "image/png",
        Some(ImageDimensions {
            width_px: 1280,
            height_px: 720,
        }),
        Some(&abs),
    );
    assert_eq!(result, "[Image: ~/.pi/agent/shot.png [image/png] 1280x720]");
    reset_capabilities_cache();
}

#[test]
fn fallback_wraps_shortened_absolute_paths_in_osc_8_file_links_when_hyperlinks_are_enabled() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    });
    let home = std::env::home_dir().expect("home directory resolves");
    let abs = home
        .join(".pi")
        .join("agent")
        .join("shot.png")
        .to_string_lossy()
        .to_string();
    let result = image_fallback(
        "image/png",
        Some(ImageDimensions {
            width_px: 10,
            height_px: 10,
        }),
        Some(&abs),
    );
    assert!(
        result.contains("\x1b]8;;file://"),
        "expected OSC 8 file link"
    );
    assert!(
        result.contains(abs.replace('\\', "/").as_str()) || result.contains(&abs),
        "file URL should target absolute path"
    );
    let visible = strip_osc8(&result);
    assert_eq!(visible, "[Image: ~/.pi/agent/shot.png [image/png] 10x10]");
    reset_capabilities_cache();
}

/// Restates the upstream test's OSC 8 strip, the
/// `result.replace(/\x1b\]8;;.*?\x1b\\/g, "")` pass.
fn strip_osc8(text: &str) -> String {
    let mut output = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("\x1b]8;;") {
        output.push_str(&rest[..start]);
        let Some(close) = rest[start..].find("\x1b\\") else {
            break;
        };
        rest = &rest[start + close + 2..];
    }
    output.push_str(rest);
    output
}

#[test]
fn fallback_leaves_bare_basenames_unchanged_and_does_not_hyperlink_them() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: true,
    });
    let result = image_fallback(
        "image/png",
        Some(ImageDimensions {
            width_px: 1,
            height_px: 1,
        }),
        Some("clankolas.png"),
    );
    assert_eq!(result, "[Image: clankolas.png [image/png] 1x1]");
    assert!(
        !result.contains("\x1b]8;"),
        "basename must not be hyperlinked"
    );
    reset_capabilities_cache();
}

#[test]
fn fallback_omits_filename_segment_when_not_provided() {
    let _guard = GLOBALS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    assert_eq!(
        image_fallback(
            "image/png",
            Some(ImageDimensions {
                width_px: 8,
                height_px: 6,
            }),
            None,
        ),
        "[Image: [image/png] 8x6]"
    );
    reset_capabilities_cache();
}

// --- upstream describe("hyperlink") -----------------------------------------

#[test]
fn hyperlink_wraps_text_in_osc_8_open_and_close_sequences() {
    let result = hyperlink("click me", "https://example.com");
    assert_eq!(
        result,
        "\x1b]8;;https://example.com\x1b\\click me\x1b]8;;\x1b\\"
    );
}

#[test]
fn hyperlink_preserves_ansi_styling_inside_the_hyperlink() {
    let styled = "\x1b[4m\x1b[34mclick me\x1b[0m";
    let result = hyperlink(styled, "https://example.com");
    assert!(result.starts_with("\x1b]8;;https://example.com\x1b\\"));
    assert!(result.contains(styled));
    assert!(result.ends_with("\x1b]8;;\x1b\\"));
}

#[test]
fn hyperlink_works_with_empty_text() {
    let result = hyperlink("", "https://example.com");
    assert_eq!(result, "\x1b]8;;https://example.com\x1b\\\x1b]8;;\x1b\\");
}

#[test]
fn hyperlink_works_with_file_ur_is() {
    let result = hyperlink("README.md", "file:///home/user/README.md");
    assert!(result.contains("file:///home/user/README.md"));
    assert!(result.contains("README.md"));
}
