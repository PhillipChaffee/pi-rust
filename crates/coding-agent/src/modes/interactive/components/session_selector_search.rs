//! Session list search and ordering, upstream's
//! `session-selector-search.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#202).
//!
//! Restatements: JS `RegExp` accepts lookarounds and backreferences that the
//! Rust `regex` engine rejects; those patterns parse as an error and match
//! nothing, exactly like invalid patterns. Match indexes count `char`s where
//! JS counts UTF-16 code units; both are monotonic in the text, so scores
//! order identically.

use pi_tui::fuzzy::fuzzy_match;
use regex::RegexBuilder;

use crate::session_manager::SessionInfo;

/// The list ordering, upstream's `SortMode`. Threaded mode falls back to
/// relevance whenever a query is present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortMode {
    /// Threaded display without a query; relevance ordering with one.
    #[default]
    Threaded,
    /// Keeps the incoming (recency-sorted) order.
    Recent,
    /// Best match first, ties broken by most recently modified.
    Relevance,
}

/// The name filter, upstream's `NameFilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NameFilter {
    /// Every session.
    #[default]
    All,
    /// Only sessions carrying a non-blank user name.
    Named,
}

/// One parsed query token, upstream's `{ kind, value }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchToken {
    /// Ordered fuzzy-character match.
    Fuzzy(String),
    /// Whitespace-normalized exact substring.
    Phrase(String),
}

/// Which half of [`ParsedSearchQuery`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Fuzzy and quoted-phrase tokens.
    Tokens,
    /// One case-insensitive `re:` pattern.
    Regex,
}

/// A parsed query, upstream's `ParsedSearchQuery`.
#[derive(Debug)]
pub struct ParsedSearchQuery {
    /// Regex queries match with [`ParsedSearchQuery::regex`]; token queries
    /// use [`ParsedSearchQuery::tokens`].
    pub mode: SearchMode,
    /// Token-mode terms, in query order.
    pub tokens: Vec<SearchToken>,
    /// The compiled case-insensitive pattern; `None` for token mode and for
    /// a failed compile.
    pub regex: Option<regex::Regex>,
    /// The compile failure (or `"Empty regex"`); a set error matches nothing.
    pub error: Option<String>,
}

/// One session's match outcome, upstream's `MatchResult`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchResult {
    /// Whether every query term matched.
    pub matches: bool,
    /// Lower is better; only meaningful when [`MatchResult::matches`].
    pub score: f64,
}

const NO_MATCH: MatchResult = MatchResult {
    matches: false,
    score: 0.0,
};

fn normalize_whitespace_lower(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// The text a session is searched over: id, name, every message, and cwd.
fn session_search_text(session: &SessionInfo) -> String {
    format!(
        "{} {} {} {}",
        session.id,
        session.name.as_deref().unwrap_or(""),
        session.all_messages_text,
        session.cwd
    )
}

/// Whether the session carries a display name with visible characters.
#[must_use]
pub fn has_session_name(session: &SessionInfo) -> bool {
    session
        .name
        .as_deref()
        .is_some_and(|name| !name.trim().is_empty())
}

/// Splits a query into fuzzy tokens, quoted phrases, or one `re:` regex.
///
/// An unbalanced quote falls back to plain whitespace tokenization with the
/// quote characters kept, upstream's recovery path.
#[must_use]
pub fn parse_search_query(query: &str) -> ParsedSearchQuery {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return ParsedSearchQuery {
            mode: SearchMode::Tokens,
            tokens: Vec::new(),
            regex: None,
            error: None,
        };
    }

    if let Some(rest) = trimmed.strip_prefix("re:") {
        let pattern = rest.trim();
        if pattern.is_empty() {
            return ParsedSearchQuery {
                mode: SearchMode::Regex,
                tokens: Vec::new(),
                regex: None,
                error: Some("Empty regex".to_string()),
            };
        }
        return match RegexBuilder::new(pattern).case_insensitive(true).build() {
            Ok(regex) => ParsedSearchQuery {
                mode: SearchMode::Regex,
                tokens: Vec::new(),
                regex: Some(regex),
                error: None,
            },
            Err(error) => ParsedSearchQuery {
                mode: SearchMode::Regex,
                tokens: Vec::new(),
                regex: None,
                error: Some(error.to_string()),
            },
        };
    }

    let mut tokens = Vec::new();
    let mut buf = String::new();
    let mut in_quote = false;
    let flush = |kind: SearchTokenKind, buf: &mut String, tokens: &mut Vec<SearchToken>| {
        let value = buf.trim().to_string();
        buf.clear();
        if value.is_empty() {
            return;
        }
        tokens.push(match kind {
            SearchTokenKind::Fuzzy => SearchToken::Fuzzy(value),
            SearchTokenKind::Phrase => SearchToken::Phrase(value),
        });
    };

    for ch in trimmed.chars() {
        if ch == '"' {
            if in_quote {
                flush(SearchTokenKind::Phrase, &mut buf, &mut tokens);
                in_quote = false;
            } else {
                flush(SearchTokenKind::Fuzzy, &mut buf, &mut tokens);
                in_quote = true;
            }
            continue;
        }
        if !in_quote && ch.is_whitespace() {
            flush(SearchTokenKind::Fuzzy, &mut buf, &mut tokens);
            continue;
        }
        buf.push(ch);
    }

    if in_quote {
        return ParsedSearchQuery {
            mode: SearchMode::Tokens,
            tokens: trimmed
                .split_whitespace()
                .filter(|token| !token.is_empty())
                .map(|token| SearchToken::Fuzzy(token.to_string()))
                .collect(),
            regex: None,
            error: None,
        };
    }

    flush(SearchTokenKind::Fuzzy, &mut buf, &mut tokens);
    ParsedSearchQuery {
        mode: SearchMode::Tokens,
        tokens,
        regex: None,
        error: None,
    }
}

#[derive(Clone, Copy)]
enum SearchTokenKind {
    Fuzzy,
    Phrase,
}

/// Scores one session against a parsed query. Every term must match; the
/// score sums term scores, with phrase and regex positions weighted 0.1.
#[must_use]
pub fn match_session(session: &SessionInfo, parsed: &ParsedSearchQuery) -> MatchResult {
    let text = session_search_text(session);

    if parsed.mode == SearchMode::Regex {
        let Some(regex) = parsed.regex.as_ref() else {
            return NO_MATCH;
        };
        return regex.find(&text).map_or(NO_MATCH, |found| MatchResult {
            matches: true,
            score: u32::try_from(text[..found.start()].chars().count()).map_or(f64::MAX, f64::from)
                * 0.1,
        });
    }

    if parsed.tokens.is_empty() {
        return MatchResult {
            matches: true,
            score: 0.0,
        };
    }

    let mut total_score = 0.0;
    let mut normalized_text: Option<String> = None;

    for token in &parsed.tokens {
        match token {
            SearchToken::Phrase(value) => {
                let normalized =
                    normalized_text.get_or_insert_with(|| normalize_whitespace_lower(&text));
                let phrase = normalize_whitespace_lower(value);
                if phrase.is_empty() {
                    continue;
                }
                let Some(index) = normalized.find(&phrase) else {
                    return NO_MATCH;
                };
                let position =
                    u32::try_from(normalized[..index].chars().count()).map_or(f64::MAX, f64::from);
                total_score = position.mul_add(0.1, total_score);
            }
            SearchToken::Fuzzy(value) => {
                let matched = fuzzy_match(value, &text);
                if !matched.matches {
                    return NO_MATCH;
                }
                total_score += matched.score;
            }
        }
    }

    MatchResult {
        matches: true,
        score: total_score,
    }
}

/// Filters by name and query, then orders per `sort_mode`. Recent keeps the
/// incoming order; relevance sorts by score with newest-modified first on
/// ties. An empty query keeps the incoming order.
#[must_use]
pub fn filter_and_sort_sessions(
    sessions: &[SessionInfo],
    query: &str,
    sort_mode: SortMode,
    name_filter: NameFilter,
) -> Vec<SessionInfo> {
    let name_filtered: Vec<&SessionInfo> = match name_filter {
        NameFilter::All => sessions.iter().collect(),
        NameFilter::Named => sessions
            .iter()
            .filter(|session| has_session_name(session))
            .collect(),
    };
    if query.trim().is_empty() {
        return name_filtered.into_iter().cloned().collect();
    }

    let parsed = parse_search_query(query);
    if parsed.error.is_some() {
        return Vec::new();
    }

    if sort_mode == SortMode::Recent {
        return name_filtered
            .into_iter()
            .filter(|session| match_session(session, &parsed).matches)
            .cloned()
            .collect();
    }

    let mut scored: Vec<(&SessionInfo, f64)> = name_filtered
        .into_iter()
        .filter_map(|session| {
            let result = match_session(session, &parsed);
            result.matches.then_some((session, result.score))
        })
        .collect();
    scored.sort_by(|(left_session, left_score), (right_session, right_score)| {
        left_score
            .total_cmp(right_score)
            .then_with(|| right_session.modified.cmp(&left_session.modified))
    });
    scored
        .into_iter()
        .map(|(session, _)| session.clone())
        .collect()
}
