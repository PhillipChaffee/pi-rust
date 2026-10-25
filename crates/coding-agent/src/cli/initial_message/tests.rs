//! The `initial-message.test.ts` cases, ported 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use super::*;

fn args(messages: &[&str]) -> Args {
    Args {
        messages: messages.iter().map(ToString::to_string).collect(),
        ..Args::default()
    }
}

#[test]
fn merges_piped_stdin_with_the_first_cli_message() {
    let mut parsed = args(&["Summarize the text given"]);
    let result = build_initial_message(InitialMessageInput {
        parsed: &mut parsed,
        file_text: None,
        file_images: None,
        stdin_content: Some("README contents\n"),
    });

    assert_eq!(
        result.initial_message.as_deref(),
        Some("README contents\nSummarize the text given")
    );
    assert!(parsed.messages.is_empty());
}

#[test]
fn uses_stdin_as_the_initial_prompt_without_a_cli_message() {
    let mut parsed = args(&[]);
    let result = build_initial_message(InitialMessageInput {
        parsed: &mut parsed,
        file_text: None,
        file_images: None,
        stdin_content: Some("README contents"),
    });

    assert_eq!(result.initial_message.as_deref(), Some("README contents"));
    assert!(parsed.messages.is_empty());
}

#[test]
fn combines_stdin_file_text_and_first_cli_message() {
    let mut parsed = args(&["Explain it", "Second message"]);
    let result = build_initial_message(InitialMessageInput {
        parsed: &mut parsed,
        file_text: Some("file\n"),
        file_images: None,
        stdin_content: Some("stdin\n"),
    });

    assert_eq!(
        result.initial_message.as_deref(),
        Some("stdin\nfile\nExplain it")
    );
    assert_eq!(parsed.messages, vec!["Second message".to_string()]);
}

#[test]
fn empty_file_text_and_empty_images_are_dropped() {
    let mut parsed = args(&[]);
    let result = build_initial_message(InitialMessageInput {
        parsed: &mut parsed,
        file_text: Some(""),
        file_images: Some(Vec::new()),
        stdin_content: None,
    });

    assert_eq!(result.initial_message, None);
    assert_eq!(result.initial_images, None);
}
