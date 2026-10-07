//! Boundary tests binding the `resolve_config_value` branches the 1:1 suites
//! leave untested, at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759.
//!
//! The environment never rides the process: template cases run through the
//! `_with` variants over an injected lookup, and command cases run directly —
//! the command branch never reads the seam. The ten-second command deadline
//! and the node-vs-port divergence on invalid UTF-8 are pinned in the module
//! doc rather than paid for in suite seconds; see the notes below.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::collections::BTreeMap;

use common::{empty_env, env_with};
use pi_coding_agent::resolve_config_value::{
    EnvLookup, clear_config_value_cache, get_config_value_env_var_name,
    get_config_value_env_var_names, get_missing_config_value_env_var_names,
    get_missing_config_value_env_var_names_with, is_command_config_value,
    is_config_value_configured, resolve_config_value, resolve_config_value_or_throw,
    resolve_config_value_uncached, resolve_config_value_uncached_with, resolve_config_value_with,
    resolve_headers, resolve_headers_or_throw,
};

/// A map-backed lookup carrying the given entries, with names the process
/// environment will never carry.
fn process_env(entries: &[(&str, &str)]) -> EnvLookup {
    env_with(entries)
}

fn credential_env(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

// =============================================================================
// The template parser's boundary shapes
// =============================================================================

#[test]
fn the_template_parser_pins_its_literal_shapes() {
    let empty = empty_env();

    // An unclosed `${` leaves the `$` literal and reparses the rest.
    assert_eq!(
        resolve_config_value_with("${PI_BELT_UNCLOSED", None, &empty),
        Some("${PI_BELT_UNCLOSED".to_string()),
    );
    // A `$` at the end stays literal.
    assert_eq!(
        resolve_config_value_with("abc$", None, &empty),
        Some("abc$".to_string()),
    );
    // The `$$`/`$!` escapes.
    assert_eq!(
        resolve_config_value_with("$$", None, &empty),
        Some("$".to_string()),
    );
    assert_eq!(
        resolve_config_value_with("$!", None, &empty),
        Some("!".to_string()),
    );
    // A brace reference whose name breaks the env grammar passes through,
    // whether the break is a dash or a leading digit.
    assert_eq!(
        resolve_config_value_with("${not-a-name}", None, &empty),
        Some("${not-a-name}".to_string()),
    );
    assert_eq!(
        resolve_config_value_with("${1abc}", None, &empty),
        Some("${1abc}".to_string()),
    );
    // A bare `$` before a non-name start passes through.
    assert_eq!(
        resolve_config_value_with("$1abc", None, &empty),
        Some("$1abc".to_string()),
    );
    assert_eq!(
        resolve_config_value_with("a$ b", None, &empty),
        Some("a$ b".to_string()),
    );
    assert_eq!(
        resolve_config_value_with("$", None, &empty),
        Some("$".to_string()),
    );
}

#[test]
fn the_template_parser_accepts_the_env_grammar_shapes() {
    let process_env = process_env(&[("_", "under"), ("A_1B", "mixed")]);

    // A name of one underscore is the grammar's shortest member.
    assert_eq!(
        resolve_config_value_with("$_", None, &process_env),
        Some("under".to_string()),
    );
    // Digits after the first character, and underscores throughout.
    assert_eq!(
        resolve_config_value_with("${A_1B}", None, &process_env),
        Some("mixed".to_string()),
    );
}

#[test]
fn a_multi_part_template_voids_on_one_unresolvable_name() {
    let process_env = process_env(&[("PI_BELT_A", "a")]);

    assert_eq!(
        resolve_config_value_with("pre-$PI_BELT_A-mid-$PI_BELT_A-post", None, &process_env),
        Some("pre-a-mid-a-post".to_string()),
        "every set name substitutes in place",
    );
    assert_eq!(
        resolve_config_value_with("$PI_BELT_A/$PI_BELT_B", None, &process_env),
        None,
        "one unresolvable segment voids the whole template",
    );
}

#[test]
fn an_empty_config_resolves_to_the_empty_string() {
    let empty = empty_env();

    assert_eq!(
        resolve_config_value_with("", None, &empty),
        Some(String::new()),
    );
}

#[test]
fn a_blank_env_value_falls_through_like_an_unset_one() {
    let process_env = process_env(&[("PI_BELT_BLANK", "")]);

    assert_eq!(
        resolve_config_value_with("$PI_BELT_BLANK", None, &process_env),
        None,
        "the || chain treats an empty value as missing"
    );
}

// =============================================================================
// The or-throw messages
// =============================================================================

#[test]
fn the_or_throw_messages_name_the_failed_surface() {
    let credential = credential_env(&[("PI_BELT_SET", "value")]);

    // The success path resolves through the credential map.
    assert_eq!(
        resolve_config_value_or_throw("$PI_BELT_SET", "Test Key", Some(&credential))
            .expect("resolved"),
        "value",
    );

    // A failed command names the command text after the `!`.
    let error = resolve_config_value_or_throw("!exit 7", "Test Key", None)
        .expect_err("the command must fail");
    assert_eq!(
        error,
        "Failed to resolve Test Key from shell command: exit 7"
    );

    // One missing variable names it.
    let error = resolve_config_value_or_throw("$PI_BELT_MISSING_ALPHA", "Test Key", None)
        .expect_err("the variable must be missing");
    assert_eq!(
        error,
        "Failed to resolve Test Key from environment variable: PI_BELT_MISSING_ALPHA"
    );

    // Several missing variables list them.
    let error = resolve_config_value_or_throw(
        "$PI_BELT_MISSING_ALPHA/$PI_BELT_MISSING_BETA",
        "Test Key",
        None,
    )
    .expect_err("the variables must be missing");
    assert_eq!(
        error,
        "Failed to resolve Test Key from environment variables: PI_BELT_MISSING_ALPHA, PI_BELT_MISSING_BETA"
    );
}

// =============================================================================
// The headers resolvers
// =============================================================================

#[test]
fn resolve_headers_drops_blank_and_unresolved_entries() {
    let headers = BTreeMap::from([
        ("x-blank".to_string(), String::new()),
        (
            "x-missing".to_string(),
            "$PI_BELT_MISSING_HEADER".to_string(),
        ),
        ("x-literal".to_string(), "lit".to_string()),
        ("x-escape".to_string(), "$$".to_string()),
    ]);

    // Upstream's `if (resolvedValue)` truthiness: a resolved empty string
    // drops the entry, and so does a resolution to nothing.
    let resolved = resolve_headers(Some(&headers), None).expect("resolved headers");
    assert_eq!(
        resolved,
        BTreeMap::from([
            ("x-escape".to_string(), "$".to_string()),
            ("x-literal".to_string(), "lit".to_string()),
        ]),
    );

    // An all-dropped result collapses to None.
    let all_dropped = BTreeMap::from([
        ("x-blank".to_string(), String::new()),
        (
            "x-missing".to_string(),
            "$PI_BELT_MISSING_HEADER".to_string(),
        ),
    ]);
    assert_eq!(resolve_headers(Some(&all_dropped), None), None);
    // No headers at all resolves to None.
    assert_eq!(resolve_headers(None, None), None);
}

#[test]
fn resolve_headers_or_throw_keeps_empty_values_and_names_the_header() {
    // The or-throw variant keeps what the dropping variant drops.
    let empty_only = BTreeMap::from([("x-empty".to_string(), String::new())]);
    assert_eq!(
        resolve_headers_or_throw(Some(&empty_only), "Model", None).expect("resolved"),
        Some(empty_only.clone()),
    );

    // A missing variable names the header it was resolving for.
    let missing = BTreeMap::from([("x-key".to_string(), "$PI_BELT_MISSING_HEADER".to_string())]);
    let error =
        resolve_headers_or_throw(Some(&missing), "Model", None).expect_err("the header must fail");
    assert_eq!(
        error,
        "Failed to resolve Model header \"x-key\" from environment variable: PI_BELT_MISSING_HEADER"
    );

    // A failed command carries the same header description.
    let command = BTreeMap::from([("x-cmd".to_string(), "!exit 3".to_string())]);
    let error = resolve_headers_or_throw(Some(&command), "Model", None)
        .expect_err("the command header must fail");
    assert_eq!(
        error,
        "Failed to resolve Model header \"x-cmd\" from shell command: exit 3"
    );

    // Empty and absent header maps resolve to None without error.
    assert_eq!(
        resolve_headers_or_throw(Some(&BTreeMap::new()), "Model", None).expect("empty"),
        None,
    );
    assert_eq!(
        resolve_headers_or_throw(None, "Model", None).expect("absent"),
        None
    );
}

// =============================================================================
// The env-name helpers
// =============================================================================

#[test]
fn get_config_value_env_var_name_answers_single_env_references_only() {
    assert_eq!(
        get_config_value_env_var_name("$PI_BELT_A"),
        Some("PI_BELT_A".to_string()),
    );
    assert_eq!(
        get_config_value_env_var_name("${PI_BELT_A}"),
        Some("PI_BELT_A".to_string()),
    );
    // Multi-part templates, commands, and literals name nothing.
    assert_eq!(get_config_value_env_var_name("$PI_BELT_A/$PI_BELT_B"), None);
    assert_eq!(get_config_value_env_var_name("!echo hi"), None);
    assert_eq!(get_config_value_env_var_name("literal"), None);
}

#[test]
fn get_config_value_env_var_names_deduplicates_in_order() {
    assert_eq!(
        get_config_value_env_var_names("$PI_BELT_A/$PI_BELT_A/$PI_BELT_B"),
        vec!["PI_BELT_A".to_string(), "PI_BELT_B".to_string()],
    );
    assert!(get_config_value_env_var_names("!echo hi").is_empty());
    assert!(get_config_value_env_var_names("$$").is_empty());
}

#[test]
fn the_missing_name_helpers_honor_the_credential_map() {
    let credential = credential_env(&[("PI_BELT_A", "a")]);

    assert_eq!(
        get_missing_config_value_env_var_names("$PI_BELT_A/$PI_BELT_B", Some(&credential)),
        vec!["PI_BELT_B".to_string()],
    );
    assert!(get_missing_config_value_env_var_names("$PI_BELT_A", Some(&credential)).is_empty(),);
    assert_eq!(
        get_missing_config_value_env_var_names("$PI_BELT_B", None),
        vec!["PI_BELT_B".to_string()],
    );

    // The `_with` variant swaps the process lookup: the injected entries
    // satisfy B, the credential map satisfies A, and only the unbacked name
    // is missing.
    let process_env = process_env(&[("PI_BELT_B", "b")]);
    assert!(
        get_missing_config_value_env_var_names_with(
            "$PI_BELT_A/$PI_BELT_B",
            Some(&credential),
            &process_env
        )
        .is_empty(),
    );
    assert_eq!(
        get_missing_config_value_env_var_names_with(
            "$PI_BELT_A/$PI_BELT_B/$PI_BELT_MISSING_GAMMA",
            Some(&credential),
            &process_env
        ),
        vec!["PI_BELT_MISSING_GAMMA".to_string()],
    );
    // With no process entry for B, the injected lookup alone decides.
    assert!(
        get_missing_config_value_env_var_names_with("$PI_BELT_B", None, &process_env).is_empty(),
    );
    assert_eq!(
        get_missing_config_value_env_var_names_with("$PI_BELT_B", None, &empty_env()),
        vec!["PI_BELT_B".to_string()],
    );
}

#[test]
fn is_command_config_value_and_is_config_value_configured() {
    assert!(is_command_config_value("!echo hi"));
    assert!(!is_command_config_value("$PI_BELT_A"));
    assert!(!is_command_config_value("literal"));

    let credential = credential_env(&[("PI_BELT_A", "a")]);
    assert!(is_config_value_configured("$PI_BELT_A", Some(&credential)));
    assert!(!is_config_value_configured("$PI_BELT_B", Some(&credential)));
    // Commands read no env names, so they are always configured.
    assert!(is_config_value_configured("!echo hi", None));
    assert!(is_config_value_configured("", None));
}

// =============================================================================
// The command branch
// =============================================================================

#[test]
fn an_invalid_utf8_command_output_resolves_to_none() {
    // node's execSync replaces invalid sequences; the port's read_to_string
    // fails, so the buffer stays empty and the resolution is None. Pinned as
    // the documented divergence. The byte rides printf's POSIX octal escape:
    // bash interprets `\xHH` in the format string but dash does not, so the
    // hex form yields the literal text on Linux's /bin/sh.
    assert_eq!(
        resolve_config_value_uncached("!printf 'a\\377b'", None),
        None
    );
}

#[test]
fn the_command_cache_never_bleeds_across_commands() {
    clear_config_value_cache();
    let first = resolve_config_value("!printf 'first'", None).expect("first");
    let second = resolve_config_value("!printf 'second'", None).expect("second");
    assert_eq!(first, "first");
    assert_eq!(second, "second");
    assert_eq!(
        resolve_config_value("!printf 'first'", None).expect("cached first"),
        "first",
        "the cache keys on the full command text"
    );
    clear_config_value_cache();
}

#[test]
fn the_uncached_variant_resolves_templates_without_the_cache() {
    let credential = credential_env(&[("PI_BELT_SET", "value")]);

    assert_eq!(
        resolve_config_value_uncached_with("$PI_BELT_SET", Some(&credential), &process_env(&[])),
        Some("value".to_string()),
    );
    assert_eq!(
        resolve_config_value_uncached("$PI_BELT_SET", Some(&credential)),
        Some("value".to_string()),
    );
}
