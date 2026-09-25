//! The message helper suite, ported from the behaviors upstream's
//! `messages.ts` carries (the runtime suites exercise it end to end;
//! upstream has no dedicated unit file).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use serde_json::json;

use crate::harness::messages::{
    BRANCH_SUMMARY_PREFIX, COMPACTION_SUMMARY_PREFIX, Timestamp, bash_execution_to_text,
    convert_to_llm, create_branch_summary_message, create_compaction_summary_message,
    create_custom_message, parse_date_millis,
};
use crate::types::AgentMessage;
use pi_ai::types::{Message, UserContent};

fn bash_execution_wire() -> serde_json::Value {
    json!({
        "role": "bashExecution",
        "command": "cargo test",
        "output": "ok",
        "exitCode": 0,
        "cancelled": false,
        "truncated": false,
        "timestamp": 1_700_000_000_000_i64,
    })
}

fn custom_message(wire: serde_json::Value) -> AgentMessage {
    serde_json::from_value(wire).expect("a custom message parses")
}

/// The bash-execution text renderer carries the command, output, cancel
/// and exit-code suffixes, and the truncation pointer.
#[test]
fn bash_execution_text_renders_the_suffixes() {
    let message = custom_message(bash_execution_wire());
    let AgentMessage::Custom(custom) = &message else {
        panic!("a non-standard role parses as the custom variant");
    };
    let parsed: crate::harness::messages::BashExecutionMessage =
        crate::harness::messages::BashExecutionMessage::try_from(custom)
            .expect("the bash execution shape parses");
    assert_eq!(
        bash_execution_to_text(&parsed),
        "Ran `cargo test`\n```\nok\n```"
    );

    let failed = custom_message(json!({
        "role": "bashExecution",
        "command": "cargo test",
        "output": "",
        "exitCode": 1,
        "cancelled": false,
        "truncated": true,
        "fullOutputPath": "/tmp/full.log",
        "timestamp": 1,
    }));
    let AgentMessage::Custom(failed) = &failed else {
        panic!("custom variant");
    };
    let parsed: crate::harness::messages::BashExecutionMessage =
        crate::harness::messages::BashExecutionMessage::try_from(failed)
            .expect("the bash execution shape");
    assert_eq!(
        bash_execution_to_text(&parsed),
        "Ran `cargo test`\n(no output)\n\nCommand exited with code 1\n\n[Output truncated. Full output: /tmp/full.log]"
    );
}

/// The summary and custom message constructors carry the wire fields and
/// the summary prefixes ride `convert_to_llm`.
#[test]
fn the_message_constructors_and_conversion_keep_the_wire() {
    let branch =
        create_branch_summary_message("summary", Some("from".to_owned()), Timestamp::Millis(1));
    assert_eq!(branch.summary, "summary");
    assert_eq!(branch.from_id.as_deref(), Some("from"));
    let compaction = create_compaction_summary_message("summary", 100, Timestamp::Millis(2));
    assert_eq!(compaction.tokens_before, 100);
    let custom = create_custom_message(
        "note",
        crate::harness::messages::CustomMessageContent::Text("hello".to_owned()),
        true,
        None,
        Timestamp::Millis(3),
    );
    assert_eq!(custom.timestamp, 3);

    let branch_message: AgentMessage = serde_json::from_value(json!({
        "role": "branchSummary",
        "summary": "back",
        "timestamp": 4,
    }))
    .expect("branch summary");
    let converted = convert_to_llm(&[branch_message]);
    assert_eq!(converted.len(), 1);
    let Message::User(user) = &converted[0] else {
        panic!("a custom message converts to a user message");
    };
    assert!(matches!(
        &user.content,
        UserContent::Text(text)
            if text.starts_with(BRANCH_SUMMARY_PREFIX) && text.ends_with("back</summary>")
    ));

    let compaction_message: AgentMessage = serde_json::from_value(json!({
        "role": "compactionSummary",
        "summary": "compacted",
        "tokensBefore": 100,
        "timestamp": 5,
    }))
    .expect("compaction summary");
    let converted = convert_to_llm(std::slice::from_ref(&compaction_message));
    let Some(Message::User(user)) = converted.first() else {
        panic!("a compaction summary converts to a user message");
    };
    assert!(matches!(
        &user.content,
        UserContent::Text(text)
            if text.starts_with(COMPACTION_SUMMARY_PREFIX)
    ));
}

/// Unconvertible custom roles drop from the context, upstream's filter.
#[test]
fn unknown_custom_roles_drop_from_the_context() {
    let unknown: AgentMessage = serde_json::from_value(json!({
        "role": "mystery",
        "timestamp": 1,
    }))
    .expect("unknown role");
    assert!(convert_to_llm(&[unknown]).is_empty());
}

/// The date parser covers ISO-8601 with fractional seconds and offsets,
/// restating JavaScript's NaN as `i64::MIN`.
#[test]
fn the_date_parser_covers_the_round_tripped_formats() {
    assert_eq!(parse_date_millis("2024-01-01"), 1_704_067_200_000);
    assert_eq!(
        parse_date_millis("2024-01-01T00:00:01.5Z"),
        1_704_067_201_500
    );
    assert_eq!(
        parse_date_millis("2024-01-01T01:00:00+01:00"),
        1_704_067_200_000
    );
    assert_eq!(parse_date_millis("not a date"), i64::MIN);
}
