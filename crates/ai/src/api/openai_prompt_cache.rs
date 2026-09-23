//! The OpenAI prompt-cache key clamp, ported from
//! `packages/ai/src/api/openai-prompt-cache.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

/// The longest prompt-cache key OpenAI accepts, upstream's
/// `OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH`.
pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// Clamp a prompt-cache key to the length OpenAI accepts, upstream's
/// `clampOpenAIPromptCacheKey`.
///
/// The cap counts characters, not bytes: a multi-byte key truncates whole
/// characters, matching upstream's `Array.from` slicing.
#[must_use]
pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<&str> {
    let key = key?;
    let cut = key
        .char_indices()
        .nth(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
        .map_or(key.len(), |(index, _)| index);
    Some(&key[..cut])
}

#[cfg(test)]
mod prompt_cache_tests {
    use super::*;

    #[test]
    fn clamps_by_characters_not_bytes() {
        let ascii = "k".repeat(64);
        assert_eq!(
            clamp_openai_prompt_cache_key(Some(&ascii)),
            Some(ascii.as_str())
        );

        let longer = format!("{ascii}tail");
        assert_eq!(
            clamp_openai_prompt_cache_key(Some(&longer)),
            Some(ascii.as_str())
        );

        // Three-byte characters: 64 of them truncate to the same characters,
        // never mid-codepoint.
        let multi = "字".repeat(64);
        let over = format!("{multi}字");
        assert_eq!(
            clamp_openai_prompt_cache_key(Some(&over)),
            Some(multi.as_str())
        );

        assert_eq!(clamp_openai_prompt_cache_key(None), None);
    }
}
