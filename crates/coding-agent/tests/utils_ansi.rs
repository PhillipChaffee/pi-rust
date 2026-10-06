//! The ANSI belt suite, upstream's `test/ansi-utils.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The compatibility harness restates the reference regex on the `regex`
//! crate and drives the same generated inputs. Upstream's `TypeError`
//! non-string case has no counterpart — Rust's `&str` cannot be anything
//! else — and the port drops it (recorded with the ticket).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_coding_agent::utils::ansi::strip_ansi;

fn reference_ansi_regex() -> regex::Regex {
    // OSC sequences only: ESC ] ... ST (non-greedy until the first ST),
    // with the terminator sequence BEL, ESC backslash, and 0x9c
    const OSC: &str = r"(?:\x1B\][\s\S]*?(?:\x07|\x1B\x5C|\u{9C}))";
    // CSI and related: ESC/C1, optional intermediates, optional params (supports ; and :) then final byte
    const CSI: &str =
        r"[\x1B\u{9B}][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]";
    regex::Regex::new(&format!("{OSC}|{CSI}")).expect("static pattern")
}

fn reference_strip_ansi(value: &str) -> String {
    if !value.contains('\x1b') && !value.contains('\u{9b}') {
        return value.to_string();
    }
    reference_ansi_regex().replace_all(value, "").into_owned()
}

fn get_compatibility_inputs() -> Vec<String> {
    let mut inputs = vec![
        "plain".to_string(),
        "a\x1b[31mred\x1b[0mz".to_string(),
        "a\x1b]8;;https://example.com\x07link\x1b]8;;\x07z".to_string(),
        "a\x1b]unterminated".to_string(),
        "a\x1b]funterminated".to_string(),
        "a\x1bPabc\x1b\\z".to_string(),
        "a\x1b^abc\x07z".to_string(),
        "a\x1b_abc\u{9c}z".to_string(),
        "a\u{90}abc\u{9c}z".to_string(),
        "a\u{9d}abc\u{9c}z".to_string(),
        "a\u{9b}31mred".to_string(),
        "a\x1b(0x".to_string(),
        "a\x1b*0x".to_string(),
        "a\x1b+c".to_string(),
        "a\x1b/0x".to_string(),
        "a\x1bcok".to_string(),
        "a\x1b\\ok".to_string(),
    ];
    let chars = [
        "a", "f", "0", "1", ";", ":", "[", "]", "(", ")", "#", "?", "m", "P", "_", "\\", "\x07",
        "\x1b", "\u{9b}", "\u{9c}", "\u{90}", "\u{9d}",
    ];

    for char in chars {
        inputs.push(format!("x\x1b{char}y"));
        inputs.push(format!("x\u{9b}{char}y"));
        for index in (0..chars.len()).step_by(3) {
            inputs.push(format!("x\x1b{char}{}y", chars[index]));
        }
    }

    inputs
}

#[test]
fn matches_the_reference_regex_for_generated_compatibility_inputs() {
    for input in get_compatibility_inputs() {
        assert_eq!(
            strip_ansi(&input),
            reference_strip_ansi(&input),
            "{input:?}"
        );
    }
}

#[test]
fn strips_ris_without_leaking_the_final_byte() {
    assert_eq!(strip_ansi("\x1bcdone"), "done");
}

#[test]
fn strips_single_byte_esc_sequences_without_leaking_final_bytes() {
    for code in b'g'..=b'm' {
        let input = format!("\x1b{}ok", code as char);
        assert_eq!(strip_ansi(&input), "ok", "{input:?}");
    }
    for code in b'r'..=b't' {
        let input = format!("\x1b{}ok", code as char);
        assert_eq!(strip_ansi(&input), "ok", "{input:?}");
    }
}

#[test]
fn strips_common_ansi_sequences_used_in_tool_output() {
    let input = "a\x1b[31mred\x1b[0m\x1b]8;;https://example.com\x07link\x1b]8;;\x07z";
    assert_eq!(strip_ansi(input), "aredlinkz");
}
