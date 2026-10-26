//! The upstream `session-selector-search.test.ts` cases at pin
//! `60e7e76`, plus boundaries for the empty regex, unbalanced quotes, and
//! threaded mode's relevance fallback.

use pi_coding_agent::modes::interactive::components::session_selector_search::{
    NameFilter, SortMode, filter_and_sort_sessions, has_session_name, match_session,
    parse_search_query,
};
use pi_coding_agent::session_manager::SessionInfo;

const BASE: i64 = 1_767_225_600_000; // 2026-01-01T00:00:00Z
const DAY: i64 = 86_400_000;

fn session(id: &str) -> SessionInfo {
    SessionInfo {
        path: format!("/tmp/{id}.jsonl"),
        id: id.to_string(),
        cwd: String::new(),
        name: None,
        parent_session_path: None,
        created: Some(BASE),
        modified: BASE,
        message_count: 1,
        first_message: "hello".to_string(),
        all_messages_text: "hello".to_string(),
    }
}

fn ids(sessions: &[SessionInfo]) -> Vec<String> {
    sessions.iter().map(|session| session.id.clone()).collect()
}

fn filter(sessions: &[SessionInfo], query: &str, sort: SortMode, name: NameFilter) -> Vec<String> {
    ids(&filter_and_sort_sessions(sessions, query, sort, name))
}

#[test]
fn filters_by_quoted_phrase_with_whitespace_normalization() {
    let mut first = session("a");
    first.all_messages_text = "node\n\n   cve was discussed".to_string();
    let mut second = session("b");
    second.modified = BASE + DAY;
    second.all_messages_text = "node something else".to_string();

    assert_eq!(
        filter(
            &[first, second],
            "\"node cve\"",
            SortMode::Recent,
            NameFilter::All
        ),
        ["a"]
    );
}

#[test]
fn filters_by_regex_and_is_case_insensitive() {
    let mut first = session("a");
    first.modified = BASE + DAY;
    first.all_messages_text = "Brave is great".to_string();
    let mut second = session("b");
    second.modified = BASE + 2 * DAY;
    second.all_messages_text = "bravery is not the same".to_string();

    assert_eq!(
        filter(
            &[first, second],
            r"re:\bbrave\b",
            SortMode::Recent,
            NameFilter::All
        ),
        ["a"]
    );
}

#[test]
fn recent_sort_preserves_input_order() {
    let mut newer = session("newer");
    newer.modified = BASE + 2 * DAY;
    newer.all_messages_text = "brave".to_string();
    let mut older = session("older");
    older.all_messages_text = "brave".to_string();
    let mut nomatch = session("nomatch");
    nomatch.modified = BASE + 3 * DAY;
    nomatch.all_messages_text = "something else".to_string();

    assert_eq!(
        filter(
            &[newer, older, nomatch],
            "\"brave\"",
            SortMode::Recent,
            NameFilter::All
        ),
        ["newer", "older"]
    );
}

#[test]
fn relevance_sort_orders_by_score_and_tie_breaks_by_modified_desc() {
    let mut late = session("late");
    late.modified = BASE + 2 * DAY;
    late.all_messages_text = "xxxx brave".to_string();
    let mut early = session("early");
    early.all_messages_text = "brave xxxx".to_string();
    assert_eq!(
        filter(
            &[late, early],
            "\"brave\"",
            SortMode::Relevance,
            NameFilter::All
        ),
        ["early", "late"]
    );

    let mut newer = session("newer");
    newer.modified = BASE + 2 * DAY;
    newer.all_messages_text = "brave".to_string();
    let mut older = session("older");
    older.all_messages_text = "brave".to_string();
    assert_eq!(
        filter(
            &[newer, older],
            "\"brave\"",
            SortMode::Relevance,
            NameFilter::All
        ),
        ["newer", "older"]
    );
}

#[test]
fn returns_empty_list_for_invalid_regex() {
    let mut only = session("a");
    only.all_messages_text = "brave".to_string();
    assert!(
        filter_and_sort_sessions(&[only], "re:(", SortMode::Recent, NameFilter::All).is_empty()
    );
}

fn named_fixture() -> Vec<SessionInfo> {
    let mut named1 = session("named1");
    named1.name = Some("My Project".to_string());
    named1.modified = BASE + 2 * DAY;
    named1.all_messages_text = "blueberry".to_string();
    let mut named2 = session("named2");
    named2.name = Some("Another Named".to_string());
    named2.modified = BASE + DAY;
    named2.all_messages_text = "blueberry".to_string();
    let mut other1 = session("other1");
    other1.modified = BASE + 3 * DAY;
    other1.all_messages_text = "blueberry".to_string();
    let mut other2 = session("other2");
    other2.all_messages_text = "blueberry".to_string();
    vec![named1, named2, other1, other2]
}

#[test]
fn name_filter_returns_all_sessions_for_all() {
    assert_eq!(
        filter(&named_fixture(), "", SortMode::Recent, NameFilter::All),
        ["named1", "named2", "other1", "other2"]
    );
}

#[test]
fn name_filter_returns_only_named_sessions() {
    assert_eq!(
        filter(&named_fixture(), "", SortMode::Recent, NameFilter::Named),
        ["named1", "named2"]
    );
}

#[test]
fn name_filter_applies_before_search_query() {
    assert_eq!(
        filter(
            &named_fixture(),
            "blueberry",
            SortMode::Recent,
            NameFilter::Named
        ),
        ["named1", "named2"]
    );
}

#[test]
fn name_filter_excludes_whitespace_only_names() {
    let mut whitespace = session("whitespace");
    whitespace.name = Some("   ".to_string());
    whitespace.all_messages_text = "test".to_string();
    let mut empty = session("empty");
    empty.name = Some(String::new());
    empty.all_messages_text = "test".to_string();
    let mut named = session("named");
    named.name = Some("Real Name".to_string());
    named.all_messages_text = "test".to_string();

    assert_eq!(
        filter(
            &[whitespace, empty, named],
            "",
            SortMode::Recent,
            NameFilter::Named
        ),
        ["named"]
    );
}

#[test]
fn empty_regex_matches_nothing() {
    let mut only = session("a");
    only.all_messages_text = "brave".to_string();
    assert!(filter_and_sort_sessions(&[only], "re:", SortMode::Recent, NameFilter::All).is_empty());
    assert_eq!(
        parse_search_query("re:   ").error.as_deref(),
        Some("Empty regex")
    );
}

#[test]
fn unbalanced_quote_falls_back_to_whitespace_tokens() {
    let mut first = session("a");
    first.all_messages_text = "node cve".to_string();
    let mut second = session("b");
    second.all_messages_text = "node something".to_string();

    let parsed = parse_search_query("\"node cve");
    // Upstream's fallback splits on whitespace and keeps the quote character,
    // so no ordinary message text matches the first token.
    assert_eq!(parsed.tokens.len(), 2);
    assert!(parsed.error.is_none());
    assert!(
        filter(
            &[first, second],
            "\"node cve",
            SortMode::Recent,
            NameFilter::All
        )
        .is_empty()
    );
}

#[test]
fn threaded_mode_orders_by_relevance_when_a_query_is_present() {
    let mut later_match = session("later");
    later_match.modified = BASE + 2 * DAY;
    later_match.all_messages_text = "xxxx brave".to_string();
    let mut earlier_match = session("earlier");
    earlier_match.all_messages_text = "brave xxxx".to_string();
    assert_eq!(
        filter(
            &[later_match, earlier_match],
            "\"brave\"",
            SortMode::Threaded,
            NameFilter::All
        ),
        ["earlier", "later"]
    );
}

#[test]
fn search_text_includes_name_and_cwd() {
    let mut named = session("a");
    named.name = Some("Release work".to_string());
    named.cwd = "/work/pi-rust".to_string();
    let mut other = session("b");
    other.all_messages_text = "unrelated".to_string();

    assert_eq!(
        filter(
            &[named.clone(), other.clone()],
            "release",
            SortMode::Recent,
            NameFilter::All
        ),
        ["a"]
    );
    assert_eq!(
        filter(
            &[named, other],
            "pi-rust",
            SortMode::Recent,
            NameFilter::All
        ),
        ["a"]
    );
}

#[test]
fn every_token_must_match_and_phrase_positions_score() {
    let mut matching = session("a");
    matching.all_messages_text = "prefix alpha beta".to_string();
    let mut missing = session("b");
    missing.all_messages_text = "alpha only".to_string();
    let parsed = parse_search_query("alpha \"beta\"");
    assert!(match_session(&matching, &parsed).matches);
    assert!(!match_session(&missing, &parsed).matches);
    let phrase_only = parse_search_query("\"alpha beta\"");
    let result = match_session(&matching, &phrase_only);
    assert!(result.matches);
    assert!(result.score > 0.0);
}

#[test]
fn has_session_name_trims_whitespace() {
    let mut unnamed = session("a");
    assert!(!has_session_name(&unnamed));
    unnamed.name = Some("  ".to_string());
    assert!(!has_session_name(&unnamed));
    unnamed.name = Some("Named".to_string());
    assert!(has_session_name(&unnamed));
}
