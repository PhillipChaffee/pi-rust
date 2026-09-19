//! Fuzzy matching utilities, ported from `packages/tui/src/fuzzy.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#42).
//!
//! Matches if all query characters appear in order (not necessarily
//! consecutive). Lower score = better match.

use std::cmp::Ordering;
use std::sync::LazyLock;

use regex::Regex;

use crate::utils::{is_whitespace_char, static_regex};

/// One [`fuzzy_match`] outcome, upstream `FuzzyMatch`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FuzzyMatch {
    /// Whether every query character appeared in order.
    pub matches: bool,
    /// Match quality; lower is better.
    pub score: f64,
}

/// Characters that start a word for the boundary bonus, upstream
/// `/[\s\-_./:]/`.
const fn is_word_boundary_char(ch: char) -> bool {
    is_whitespace_char(ch) || matches!(ch, '-' | '_' | '.' | '/' | ':')
}

static SWAPPED_LETTERS_FIRST_RE: LazyLock<Regex> =
    LazyLock::new(|| static_regex("^(?<letters>[a-z]+)(?<digits>[0-9]+)$"));

static SWAPPED_DIGITS_FIRST_RE: LazyLock<Regex> =
    LazyLock::new(|| static_regex("^(?<digits>[0-9]+)(?<letters>[a-z]+)$"));

/// Score one query against one text, upstream `fuzzyMatch`.
///
/// Restatement: JS indexes strings by UTF-16 code unit; the port indexes by
/// `char` (see the [`crate::utils`] module docs). The `-1` sentinel for "no
/// previous match" becomes `Option<usize>`, whose `None == i.checked_sub(1)`
/// comparison reproduces upstream's `lastMatchIndex === i - 1` truthiness at
/// `i == 0`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "match scores sum bounded per-character terms over a text length far below f64's exact-integer range; upstream's JS doubles behave the same"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "mul_add would fuse the rounding step; upstream's JS performs two roundings per term and score parity with upstream is the acceptance gate"
)]
pub fn fuzzy_match(query: &str, text: &str) -> FuzzyMatch {
    let query_lower = query.to_lowercase();
    let text_lower = text.to_lowercase();

    let match_query = |normalized_query: &str| -> FuzzyMatch {
        let query_chars: Vec<char> = normalized_query.chars().collect();
        let text_chars: Vec<char> = text_lower.chars().collect();

        if query_chars.is_empty() {
            return FuzzyMatch {
                matches: true,
                score: 0.0,
            };
        }

        if query_chars.len() > text_chars.len() {
            return FuzzyMatch {
                matches: false,
                score: 0.0,
            };
        }

        let mut query_index = 0usize;
        let mut score = 0.0f64;
        let mut last_match_index: Option<usize> = None;
        let mut consecutive_matches = 0usize;

        for (i, &text_char) in text_chars.iter().enumerate() {
            if query_index >= query_chars.len() {
                break;
            }
            if text_char == query_chars[query_index] {
                let is_word_boundary = i == 0 || is_word_boundary_char(text_chars[i - 1]);

                // Reward consecutive matches
                if i.checked_sub(1) == last_match_index {
                    consecutive_matches += 1;
                    score -= (consecutive_matches * 5) as f64;
                } else {
                    consecutive_matches = 0;
                    // Penalize gaps
                    if let Some(last) = last_match_index {
                        score += ((i - last - 1) * 2) as f64;
                    }
                }

                // Reward word boundary matches
                if is_word_boundary {
                    score -= 10.0;
                }

                // Slight penalty for later matches
                score += i as f64 * 0.1;

                last_match_index = Some(i);
                query_index += 1;
            }
        }

        if query_index < query_chars.len() {
            return FuzzyMatch {
                matches: false,
                score: 0.0,
            };
        }

        if normalized_query == text_lower {
            score -= 100.0;
        }

        FuzzyMatch {
            matches: true,
            score,
        }
    };

    let primary_match = match_query(&query_lower);
    if primary_match.matches {
        return primary_match;
    }

    // "codex52" vs a "5.2-codex" style label: match the digits-first
    // spelling too, at a small fixed penalty.
    let swapped_query = SWAPPED_LETTERS_FIRST_RE
        .captures(&query_lower)
        .map(|caps| format!("{}{}", &caps["digits"], &caps["letters"]))
        .or_else(|| {
            SWAPPED_DIGITS_FIRST_RE
                .captures(&query_lower)
                .map(|caps| format!("{}{}", &caps["letters"], &caps["digits"]))
        });

    let Some(swapped_query) = swapped_query else {
        return primary_match;
    };

    let swapped_match = match_query(&swapped_query);
    if !swapped_match.matches {
        return primary_match;
    }

    FuzzyMatch {
        matches: true,
        score: swapped_match.score + 5.0,
    }
}

/// Filter and sort items by fuzzy match quality (best matches first),
/// upstream `fuzzyFilter`. Supports whitespace- and slash-separated tokens:
/// all tokens must match.
///
/// Restatement: upstream returns the matched items in a new array; the port
/// returns references into `items` (same order guarantees, no clone bound).
/// The sort is stable, matching JS `Array.prototype.sort`.
pub fn fuzzy_filter<'a, T>(
    items: &'a [T],
    query: &str,
    get_text: impl Fn(&T) -> String,
) -> Vec<&'a T> {
    if query.trim().is_empty() {
        return items.iter().collect();
    }

    let tokens: Vec<&str> = query
        .trim()
        .split(|ch: char| is_whitespace_char(ch) || ch == '/')
        .filter(|token| !token.is_empty())
        .collect();

    if tokens.is_empty() {
        return items.iter().collect();
    }

    let mut results: Vec<(f64, &'a T)> = Vec::new();

    for item in items {
        let text = get_text(item);
        let mut total_score = 0.0f64;
        let mut all_match = true;

        for token in &tokens {
            let m = fuzzy_match(token, &text);
            if m.matches {
                total_score += m.score;
            } else {
                all_match = false;
                break;
            }
        }

        if all_match {
            results.push((total_score, item));
        }
    }

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    results.into_iter().map(|(_, item)| item).collect()
}
