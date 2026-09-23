//! Port of `packages/tui/test/bug-regression-isimageline-startswith-bug.test.ts`
//! 1:1 (upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`). The
//! upstream `describe` groups become test functions; the old-implementation
//! simulation restates as a closure over the optional escape prefix.

use pi_tui::terminal_image::is_image_line;

/// Restates the upstream test's local `oldIsImageLine`, the buggy
/// `startsWith`-against-the-detected-prefix implementation.
fn old_is_image_line(line: &str, image_escape_prefix: Option<&str>) -> bool {
    image_escape_prefix.is_some_and(|prefix| line.starts_with(prefix))
}

// --- upstream describe("Bug scenario: Terminal without image support") ----

#[test]
fn old_implementation_would_return_false_causing_crash() {
    // When the terminal does not support images, upstream's
    // getImageEscapePrefix() answered null and the buggy isImageLine missed
    // lines that carry image sequences, so the TUI width-checked a line
    // holding 300 KB of base64 and crashed on it.
    let terminal_without_image_support = None;
    let line_with_image_sequence =
        "Read image file [image/jpeg]\x1b]1337;File=size=800,600;inline=1:base64data...\x07";

    let old_result = old_is_image_line(line_with_image_sequence, terminal_without_image_support);
    assert!(
        !old_result,
        "Bug: old implementation returns false for line containing image sequence when terminal has no image support"
    );
}

#[test]
fn new_implementation_returns_true_correctly() {
    let line_with_image_sequence =
        "Read image file [image/jpeg]\x1b]1337;File=size=800,600;inline=1:base64data...\x07";

    assert!(
        is_image_line(line_with_image_sequence),
        "Fix: new implementation returns true for line containing image sequence"
    );
}

#[test]
fn new_implementation_detects_kitty_sequences_in_any_position() {
    let scenarios = [
        "At start: \x1b_Ga=T,f=100,data...\x1b\\".to_string(),
        "Prefix \x1b_Ga=T,data...\x1b\\".to_string(),
        "Suffix text \x1b_Ga=T,data...\x1b\\ suffix".to_string(),
        "Middle \x1b_Ga=T,data...\x1b\\ more text".to_string(),
        format!(
            "Text before \x1b_Ga=T,f=100{} text after",
            "A".repeat(300_000)
        ),
    ];

    for line in scenarios {
        assert!(
            is_image_line(&line),
            "Should detect Kitty sequence in: {}...",
            &line[..50]
        );
    }
}

#[test]
fn new_implementation_detects_iterm2_sequences_in_any_position() {
    let scenarios = [
        "At start: \x1b]1337;File=size=100,100:base64...\x07".to_string(),
        "Prefix \x1b]1337;File=inline=1:data==\x07".to_string(),
        "Suffix text \x1b]1337;File=inline=1:data==\x07 suffix".to_string(),
        "Middle \x1b]1337;File=inline=1:data==\x07 more text".to_string(),
        format!(
            "Text before \x1b]1337;File=size=800,600;inline=1:{} text after",
            "B".repeat(300_000)
        ),
    ];

    for line in scenarios {
        assert!(
            is_image_line(&line),
            "Should detect iTerm2 sequence in: {}...",
            &line[..50.min(line.len())]
        );
    }
}

// --- upstream describe("Integration: Tool execution scenario") -------------

#[test]
fn integration_detects_image_sequences_in_read_tool_output() {
    let tool_output_line =
        "Read image file [image/jpeg]\x1b]1337;File=size=800,600;inline=1:base64image...\x07";

    assert!(
        is_image_line(tool_output_line),
        "Should detect image sequence in tool output line"
    );
}

#[test]
fn integration_detects_kitty_sequences_from_image_component() {
    let kitty_line = "\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\";

    assert!(
        is_image_line(kitty_line),
        "Should detect Kitty image component output"
    );
}

#[test]
fn integration_handles_ansi_codes_before_image_sequences() {
    let lines = [
        "\x1b[31mError\x1b[0m: \x1b]1337;File=inline=1:base64==\x07",
        "\x1b[33mWarning\x1b[0m: \x1b_Ga=T,data...\x1b\\",
        "\x1b[1mBold\x1b[0m \x1b]1337;File=:base64==\x07\x1b[0m",
    ];

    for line in lines {
        assert!(
            is_image_line(line),
            "Should detect image sequence after ANSI codes: {}...",
            &line[..30.min(line.len())]
        );
    }
}

// --- upstream describe("Crash scenario simulation") -------------------------

#[test]
fn crash_does_not_crash_on_very_long_lines_with_image_sequences() {
    let base64_char = "A".repeat(100);
    let iterm2_sequence = "\x1b]1337;File=size=800,600;inline=1:";
    let crash_line = format!(
        "Output: {iterm2_sequence}{} end of output",
        base64_char.repeat(3040)
    );

    assert!(crash_line.len() > 300_000, "Test line should be > 300KB");

    assert!(
        is_image_line(&crash_line),
        "Should detect image sequence in very long line, preventing TUI crash"
    );
}

#[test]
fn crash_handles_lines_exactly_matching_crash_log_dimensions() {
    let target_width = 58_649;
    let prefix = "Text";
    let sequence = "\x1b_Ga=T,f=100";
    let suffix = "End";
    let padding = "A".repeat(target_width - prefix.len() - sequence.len() - suffix.len());
    let line = format!("{prefix}{sequence}{padding}{suffix}");

    assert_eq!(line.len(), 58_649);
    assert!(
        is_image_line(&line),
        "Should detect image sequence in 58649-char line"
    );
}

// --- upstream describe("Negative cases: Don't false positive") --------------

#[test]
fn negative_does_not_detect_images_in_regular_long_text() {
    let long_text = "A".repeat(100_000);

    assert!(
        !is_image_line(&long_text),
        "Should not detect images in plain long text"
    );
}

#[test]
fn negative_does_not_detect_images_in_lines_with_file_paths() {
    let file_paths = [
        "/path/to/1337/image.jpg",
        "/usr/local/bin/File_converter",
        "~/Documents/1337File_backup.png",
        "./_G_test_file.txt",
    ];

    for path in file_paths {
        assert!(
            !is_image_line(path),
            "Should not falsely detect image sequence in path: {path}"
        );
    }
}
