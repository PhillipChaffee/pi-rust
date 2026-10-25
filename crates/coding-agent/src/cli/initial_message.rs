//! The initial-prompt assembly, upstream's `src/cli/initial-message.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use pi_ai::types::ImageContent;

use crate::cli::args::Args;

/// The assembly inputs, upstream's `InitialMessageInput`.
#[derive(Debug)]
pub struct InitialMessageInput<'a> {
    /// The parsed command line; its first message is consumed.
    pub parsed: &'a mut Args,
    /// The rendered `@file` text, upstream's `fileText`.
    pub file_text: Option<&'a str>,
    /// The `@file` image attachments.
    pub file_images: Option<Vec<ImageContent>>,
    /// The piped stdin content, when the caller read any.
    pub stdin_content: Option<&'a str>,
}

/// The assembled prompt, upstream's `InitialMessageResult`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InitialMessageResult {
    /// Stdin, file text, and the first CLI message joined with no separator.
    pub initial_message: Option<String>,
    /// The file images, when the files carried any.
    pub initial_images: Option<Vec<ImageContent>>,
}

/// Combine stdin content, `@file` text, and the first CLI message into the
/// single initial prompt non-interactive mode sends.
#[must_use]
pub fn build_initial_message(input: InitialMessageInput<'_>) -> InitialMessageResult {
    let InitialMessageInput {
        parsed,
        file_text,
        file_images,
        stdin_content,
    } = input;

    let mut parts: Vec<String> = Vec::new();
    if let Some(stdin) = stdin_content {
        parts.push(stdin.to_string());
    }
    if let Some(text) = file_text.filter(|text| !text.is_empty()) {
        parts.push(text.to_string());
    }
    if !parsed.messages.is_empty() {
        parts.push(parsed.messages.remove(0));
    }

    InitialMessageResult {
        initial_message: if parts.is_empty() {
            None
        } else {
            Some(parts.concat())
        },
        initial_images: file_images.filter(|images| !images.is_empty()),
    }
}

#[cfg(test)]
mod tests;
