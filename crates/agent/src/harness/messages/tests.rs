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
use pi_ai::types::{Message, UserBlock, UserContent, UserMessage};

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

/// The cancelled bash-execution rendering takes the cancel suffix and
/// suppresses the exit-code suffix, upstream's cancelled branch.
#[test]
fn bash_execution_text_renders_the_cancelled_suffix() {
    let cancelled = custom_message(json!({
        "role": "bashExecution",
        "command": "cargo test",
        "output": "",
        "exitCode": 1,
        "cancelled": true,
        "truncated": false,
        "timestamp": 1,
    }));
    let AgentMessage::Custom(custom) = &cancelled else {
        panic!("custom variant");
    };
    let parsed: crate::harness::messages::BashExecutionMessage =
        crate::harness::messages::BashExecutionMessage::try_from(custom)
            .expect("the bash execution shape");
    assert_eq!(
        bash_execution_to_text(&parsed),
        "Ran `cargo test`\n(no output)\n\n(command cancelled)"
    );
}

/// The timestamp constructors accept epoch millis and date strings,
/// upstream's `string | number` union.
#[test]
fn timestamps_accept_millis_and_date_strings() {
    let from_millis: Timestamp = 5i64.into();
    assert_eq!(from_millis.millis(), 5);
    assert_eq!(
        Timestamp::Date("2024-01-01".to_owned()).millis(),
        1_704_067_200_000
    );
}

/// The date parser rejects malformed months, days, separators, time
/// pieces, and offsets, restating JavaScript's NaN as `i64::MIN`.
#[test]
fn the_date_parser_rejects_malformed_pieces() {
    assert_eq!(parse_date_millis("2024-13-01"), i64::MIN);
    assert_eq!(parse_date_millis("2024-00-15"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-00"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-32"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01X"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01T:00"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01T00Zx"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01T00:00:0x"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01T00:00:00.5+zz"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01T00:00:00X"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01T00:00+zz"), i64::MIN);
    assert_eq!(parse_date_millis("2024-01-01T00:Q"), i64::MIN);
}

/// The date parser applies numeric UTC offsets with and without minutes
/// and accepts minuteless and secondless forms.
#[test]
fn the_date_parser_applies_utc_offsets() {
    assert_eq!(
        parse_date_millis("2024-01-01T00:00:00+05:30"),
        1_704_067_200_000 - 5 * 3_600_000 - 30 * 60_000
    );
    assert_eq!(
        parse_date_millis("2024-01-01T00:00:00-05:00"),
        1_704_067_200_000 + 5 * 3_600_000
    );
    assert_eq!(
        parse_date_millis("2024-01-01T00:00:00.5-01:00"),
        1_704_067_200_500 + 3_600_000
    );
    assert_eq!(
        parse_date_millis("2024-01-01T00:00+01:00"),
        1_704_063_600_000
    );
    assert_eq!(parse_date_millis("2024-01-01T00:00-05"), 1_704_085_200_000);
    assert_eq!(parse_date_millis("2024-01-01T00Z"), 1_704_067_200_000);
    assert_eq!(parse_date_millis("2024-01-01T00"), 1_704_067_200_000);
}

/// The LLM conversion carries bash executions, custom text and block
/// content, and standard messages; excluded, malformed, and unknown
/// entries drop, upstream's filter.
#[test]
fn convert_to_llm_maps_and_drops_by_role() {
    let excluded = custom_message(json!({
        "role": "bashExecution",
        "command": "cargo test",
        "output": "x",
        "cancelled": false,
        "truncated": false,
        "excludeFromContext": true,
        "timestamp": 2,
    }));
    let malformed_bash: AgentMessage = serde_json::from_value(json!({
        "role": "bashExecution",
        "timestamp": 3,
    }))
    .expect("bash wire");
    let text_custom = custom_message(json!({
        "role": "custom",
        "customType": "note",
        "content": "hello",
        "display": true,
        "timestamp": 4,
    }));
    let blocks_custom = custom_message(json!({
        "role": "custom",
        "customType": "note",
        "content": [
            {"type": "text", "text": "a"},
            {"type": "image", "data": "Zm9v", "mimeType": "image/png"},
        ],
        "display": false,
        "timestamp": 4,
    }));
    let malformed_branch: AgentMessage = serde_json::from_value(json!({
        "role": "branchSummary",
        "timestamp": 5,
    }))
    .expect("branch summary wire");
    let malformed_compaction: AgentMessage = serde_json::from_value(json!({
        "role": "compactionSummary",
        "timestamp": 6,
    }))
    .expect("compaction summary wire");
    let malformed_custom: AgentMessage = serde_json::from_value(json!({
        "role": "custom",
        "timestamp": 9,
    }))
    .expect("custom wire");
    let standard = AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text("hi".to_owned()),
        timestamp: 7,
    }));
    let bash = custom_message(bash_execution_wire());
    let converted = convert_to_llm(&[
        bash,
        excluded,
        malformed_bash,
        text_custom,
        blocks_custom,
        malformed_branch,
        malformed_compaction,
        malformed_custom,
        standard,
    ]);
    assert_eq!(converted.len(), 4);
    let Message::User(bash_user) = &converted[0] else {
        panic!("a bash execution converts to a user message");
    };
    assert!(matches!(
        &bash_user.content,
        UserContent::Text(text) if text.contains("Ran `cargo test`")
    ));
    let Message::User(text_user) = &converted[1] else {
        panic!("a custom message converts to a user message");
    };
    assert!(matches!(
        &text_user.content,
        UserContent::Text(text) if text == "hello"
    ));
    let Message::User(blocks_user) = &converted[2] else {
        panic!("a block custom message converts to a user message");
    };
    let UserContent::Blocks(blocks) = &blocks_user.content else {
        panic!("block content stays typed");
    };
    assert_eq!(blocks.len(), 2);
    assert!(matches!(
        &blocks[0],
        UserBlock::Text(text) if text.text == "a"
    ));
    assert!(matches!(&blocks[1], UserBlock::Image(_)));
    let Message::User(standard_user) = &converted[3] else {
        panic!("a standard message passes through");
    };
    assert!(matches!(
        &standard_user.content,
        UserContent::Text(text) if text == "hi"
    ));
}
