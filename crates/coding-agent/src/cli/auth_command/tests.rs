#![expect(
    clippy::expect_used,
    reason = "the unit tests pin parse/validation outcomes; an unexpected result panics the test by design"
)]

use super::*;

use pi_ai::auth::types::ModelAuth;

fn args(flags: &[&str]) -> Vec<String> {
    flags.iter().map(ToString::to_string).collect()
}

#[test]
fn parses_the_three_kinds_and_flags() {
    let check = parse_auth_command(&args(&[
        "auth",
        "check",
        "--json",
        "--credentials",
        "--no-refresh",
        "x",
    ]))
    .expect("parses")
    .expect("auth");
    assert_eq!(
        check,
        AuthCommand {
            kind: AuthCommandKind::Check,
            args: vec!["x".to_string()],
            json: true,
            credentials: true,
            no_refresh: true,
            min_expiry_ms: None,
        }
    );
    let print = parse_auth_command(&args(&["auth", "print-api-key"]))
        .expect("parses")
        .expect("auth");
    assert_eq!(print.kind, AuthCommandKind::ApiKey);
    assert!(print.args.is_empty());
    let bearer = parse_auth_command(&args(&[
        "auth",
        "print-bearer-token",
        "--min-expiry",
        "30m",
    ]))
    .expect("parses")
    .expect("auth");
    assert_eq!(bearer.min_expiry_ms, Some(30 * 60_000));
    let hours = parse_auth_command(&args(&["auth", "print-bearer-token", "--min-expiry", "2H"]))
        .expect("parses")
        .expect("auth");
    assert_eq!(hours.min_expiry_ms, Some(7_200_000));
    assert!(
        parse_auth_command(&args(&["install", "x"]))
            .expect("not auth")
            .is_none()
    );
}

#[test]
fn rejects_grammar_violations() {
    let unknown = parse_auth_command(&args(&["auth", "unknown"]));
    assert_eq!(
        unknown.expect_err("unknown"),
        AuthCommandError("Unknown auth command \"unknown\". Use \"pi auth print-api-key\", \"pi auth print-bearer-token\", or \"pi auth check\".".to_string())
    );
    let min_expiry_on_check = parse_auth_command(&args(&["auth", "check", "--min-expiry", "30m"]));
    assert_eq!(
        min_expiry_on_check.expect_err("min-expiry"),
        AuthCommandError("--min-expiry is only supported by print-bearer-token".to_string())
    );
    let json_on_print = parse_auth_command(&args(&["auth", "print-api-key", "--json"]));
    assert_eq!(
        json_on_print.expect_err("json"),
        AuthCommandError("--json is only supported by auth check".to_string())
    );
    let bad_duration =
        parse_auth_command(&args(&["auth", "print-bearer-token", "--min-expiry", "30"]));
    assert_eq!(
        bad_duration.expect_err("duration"),
        AuthCommandError("--min-expiry must use a duration such as 30m or 1h".to_string())
    );
    let missing_value = parse_auth_command(&args(&["auth", "print-bearer-token", "--min-expiry"]));
    assert_eq!(
        missing_value.expect_err("missing"),
        AuthCommandError("--min-expiry must use a duration such as 30m or 1h".to_string())
    );
}

#[test]
fn detects_help_requests() {
    assert!(is_auth_command_help(&args(&["auth"])));
    assert!(is_auth_command_help(&args(&["auth", "help"])));
    assert!(is_auth_command_help(&args(&["auth", "check", "--help"])));
    assert!(is_auth_command_help(&args(&[
        "auth",
        "print-api-key",
        "-h"
    ])));
    assert!(!is_auth_command_help(&args(&["auth", "check"])));
    assert!(!is_auth_command_help(&args(&["install"])));
}

#[test]
fn validates_targets_and_rejects_extras() {
    let base = AuthCommandArgs {
        provider: Some(" openai ".to_string()),
        ..AuthCommandArgs::default()
    };
    let target = validate_auth_command_args(&base, AuthCommandKind::Check).expect("valid");
    assert_eq!(target.provider.as_deref(), Some("openai"));

    let flagged = AuthCommandArgs {
        unknown_flags: indexmap::IndexMap::from([(
            "credentails".to_string(),
            UnknownFlagValue::Flag,
        )]),
        ..AuthCommandArgs::default()
    };
    assert_eq!(
        validate_auth_command_args(&flagged, AuthCommandKind::Check).expect_err("flagged"),
        AuthCommandError("Unknown option --credentails for \"auth check\".".to_string())
    );
    let with_api_key = AuthCommandArgs {
        api_key: Some("k".to_string()),
        ..AuthCommandArgs::default()
    };
    assert_eq!(
        validate_auth_command_args(&with_api_key, AuthCommandKind::Check).expect_err("api key"),
        AuthCommandError("Auth commands only accept --provider and --model".to_string())
    );
    assert_eq!(
        validate_auth_command_args(&AuthCommandArgs::default(), AuthCommandKind::Check)
            .expect_err("empty"),
        AuthCommandError(
            "Auth checks require --provider <provider> or --model <model>".to_string()
        )
    );
    assert_eq!(
        validate_auth_command_args(&AuthCommandArgs::default(), AuthCommandKind::ApiKey)
            .expect_err("empty"),
        AuthCommandError(
            "Credential printing requires --provider <provider> or --model <model>".to_string()
        )
    );
}

#[test]
fn extracts_credentials() {
    let api_key = AuthResult {
        auth: ModelAuth {
            api_key: Some("test-key".to_string()),
            ..ModelAuth::default()
        },
        ..AuthResult::default()
    };
    assert_eq!(
        get_auth_credential(Some(&api_key)).as_deref(),
        Some("test-key")
    );

    let bearer = AuthResult {
        auth: ModelAuth {
            api_key: None,
            headers: Some(
                [
                    ("X-Other".to_string(), None),
                    (
                        "AUTHORIZATION".to_string(),
                        Some("Bearer  header-token".to_string()),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            ..ModelAuth::default()
        },
        ..AuthResult::default()
    };
    assert_eq!(
        get_auth_credential(Some(&bearer)).as_deref(),
        Some("header-token")
    );
    assert_eq!(get_auth_credential(None), None);
    // An empty API key falls through to the header scan, upstream's
    // truthiness.
    let empty_key = AuthResult {
        auth: ModelAuth {
            api_key: Some(String::new()),
            ..ModelAuth::default()
        },
        ..AuthResult::default()
    };
    assert_eq!(get_auth_credential(Some(&empty_key)), None);
}
