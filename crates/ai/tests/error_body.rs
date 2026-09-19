//! The provider error-body normalizer port, from `test/error-body.test.ts`.
//! Each synthesized SDK error object becomes an [`SdkError`] input; the
//! JS-runtime sniffing is carried by the [`ErrorBody`] variants.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_ai::utils::error_body::{
    ErrorBody, MAX_PROVIDER_ERROR_BODY_CHARS, SdkError, SdkResponse, format_provider_error,
    normalize_provider_error, normalize_thrown_value, truncate_error_text,
};
use serde_json::json;

#[test]
fn extracts_status_and_body_from_a_mistral_shaped_error() {
    let error = SdkError {
        message: String::from("Mistral request failed"),
        status_code: Some(403),
        body: Some(String::from("{\"error\":\"blocked by gateway WAF\"}")),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.status, Some(403)];
    assert_eq![
        norm.body.as_deref(),
        Some("{\"error\":\"blocked by gateway WAF\"}")
    ];
    assert!(!norm.message_carries_body);
}

#[test]
fn reads_the_parsed_body_off_an_openai_api_error_when_the_message_is_opaque() {
    // makeMessage(status, error, message) yields "<status> status code (no
    // body)" when the parsed body is unparsed, while the body stays on
    // error.error.
    let error = SdkError {
        message: String::from("403 status code (no body)"),
        status: Some(403),
        error: Some(ErrorBody::Parsed(
            json!({"error": "blocked by gateway WAF"}),
        )),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.status, Some(403)];
    assert_eq![
        norm.body.as_deref(),
        Some("{\"error\":\"blocked by gateway WAF\"}")
    ];
    assert!(!norm.message_carries_body);
}

#[test]
fn preserves_the_message_when_genai_already_folds_the_body_into_it() {
    let body = json!({"error": {"code": 403, "message": "Permission denied"}});
    let error = SdkError {
        message: serde_json::to_string(&body).expect("body serializes"),
        status: Some(403),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.status, Some(403)];
    assert!(norm.message_carries_body);
    assert_eq![
        norm.message,
        serde_json::to_string(&body).expect("body again")
    ];
}

#[test]
fn extracts_status_and_body_from_a_bedrock_shaped_service_exception() {
    let error = SdkError {
        message: String::from("UnknownError"),
        metadata_http_status_code: Some(403),
        response: Some(SdkResponse {
            status_code: Some(403),
            body: Some(ErrorBody::Text(String::from(
                "{\"message\":\"blocked by gateway WAF\"}",
            ))),
        }),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.status, Some(403)];
    assert_eq![
        norm.body.as_deref(),
        Some("{\"message\":\"blocked by gateway WAF\"}")
    ];
    assert!(!norm.message_carries_body);
}

#[test]
fn ignores_a_bedrock_response_stream_instead_of_serializing_its_internals() {
    let error = SdkError {
        message: String::from(
            "Invocation of model ID anthropic.claude-opus-5 with on-demand throughput isn't supported.",
        ),
        metadata_http_status_code: Some(400),
        response: Some(SdkResponse {
            status_code: Some(400),
            // A readable stream arrives as an unreadable body.
            body: Some(ErrorBody::Unreadable),
        }),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.status, Some(400)];
    assert_eq![norm.body, None];
    assert!(
        norm.message
            .contains("on-demand throughput isn't supported")
    );
    assert!(norm.message_carries_body);
}

#[test]
fn ignores_a_wrapper_response_body_without_a_pipe_method_instead_of_serializing_it() {
    // Not every SDK response wrapper is a node stream: web ReadableStreams
    // and SDK-specific wrapper classes have no `pipe`, but serializing them
    // still yields internals-noise that would replace the real message.
    let error = SdkError {
        message: String::from("Input is too long for requested model."),
        metadata_http_status_code: Some(400),
        response: Some(SdkResponse {
            status_code: Some(400),
            body: Some(ErrorBody::Unreadable),
        }),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.status, Some(400)];
    assert_eq![norm.body, None];
    assert!(norm.message.contains("Input is too long"));
    assert!(norm.message_carries_body);
}

#[test]
fn ignores_a_wrapper_error_field_instead_of_serializing_it() {
    let error = SdkError {
        message: String::from("TLS handshake failed"),
        status: Some(502),
        error: Some(ErrorBody::Unreadable),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.body, None];
    assert_eq![norm.message, "TLS handshake failed"];
    assert!(norm.message_carries_body);
}

#[test]
fn still_surfaces_a_plain_parsed_json_body_object() {
    let error = SdkError {
        message: String::from("400 status code (no body)"),
        status: Some(400),
        error: Some(ErrorBody::Parsed(json!({
            "message": "schema validation failed",
            "field": "tools[0]"
        }))),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![
        norm.body.as_deref(),
        Some("{\"message\":\"schema validation failed\",\"field\":\"tools[0]\"}")
    ];
    assert!(!norm.message_carries_body);
}

#[test]
fn json_stringifies_a_non_error_thrown_value() {
    let norm = normalize_thrown_value(&json!({"reason": "boom"}));

    assert_eq![norm.status, None];
    assert_eq![norm.body, None];
    assert_eq![norm.message, "{\"reason\":\"boom\"}"];
    assert!(!norm.message_carries_body);
}

#[test]
fn treats_an_empty_parsed_body_object_as_no_body() {
    let error = SdkError {
        message: String::from("403 status code (no body)"),
        status: Some(403),
        error: Some(ErrorBody::Parsed(json!({}))),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert_eq![norm.body, None];
    assert!(norm.message_carries_body);
}

#[test]
fn truncates_the_body_at_the_cap() {
    let long_body = "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS + 50);
    let error = SdkError {
        message: String::from("failed"),
        status_code: Some(500),
        body: Some(long_body.clone()),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    let body = norm.body.expect("a body");
    assert!(body.contains("... [truncated 50 chars]"), "{body}");
    assert!(body.len() < long_body.len());
}

#[test]
fn sets_message_carries_body_when_the_message_already_contains_the_extracted_body() {
    let error = SdkError {
        message: String::from("500: upstream exploded"),
        status_code: Some(500),
        body: Some(String::from("upstream exploded")),
        ..SdkError::default()
    };

    let norm = normalize_provider_error(error);

    assert!(norm.message_carries_body);
}

#[test]
fn surfaces_status_and_body_without_a_prefix() {
    let norm = normalize_provider_error(SdkError {
        message: String::from("403 status code (no body)"),
        status: Some(403),
        error: Some(ErrorBody::Parsed(
            json!({"error": "blocked by gateway WAF"}),
        )),
        ..SdkError::default()
    });

    let formatted = format_provider_error(&norm, None);

    assert!(formatted.contains("403"), "{formatted}");
    assert!(formatted.contains("blocked by gateway WAF"), "{formatted}");
    assert_ne![formatted, "403 status code (no body)"];
}

#[test]
fn applies_a_provider_prefix_with_status_and_body() {
    let norm = normalize_provider_error(SdkError {
        message: String::from("403 status code (no body)"),
        status: Some(403),
        error: Some(ErrorBody::Parsed(
            json!({"error": "blocked by gateway WAF"}),
        )),
        ..SdkError::default()
    });

    assert_eq![
        format_provider_error(&norm, Some("OpenAI API error")),
        "OpenAI API error (403): {\"error\":\"blocked by gateway WAF\"}"
    ];
}

#[test]
fn preserves_the_message_with_prefix_and_status_when_it_already_carries_the_body() {
    let body = json!({"error": {"message": "Permission denied"}});
    let body_text = serde_json::to_string(&body).expect("body serializes");
    let norm = normalize_provider_error(SdkError {
        message: body_text.clone(),
        status: Some(403),
        ..SdkError::default()
    });

    assert_eq![
        format_provider_error(&norm, Some("OpenAI API error")),
        format!("OpenAI API error (403): {body_text}")
    ];
}

#[test]
fn returns_the_bare_message_for_a_non_error_value() {
    let norm = normalize_thrown_value(&json!({"reason": "boom"}));

    assert_eq![format_provider_error(&norm, None), "{\"reason\":\"boom\"}"];
}

#[test]
fn truncation_notes_the_dropped_length() {
    assert_eq![truncate_error_text("short", 4000), "short"];
    let long = "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS + 50);
    let truncated = truncate_error_text(&long, MAX_PROVIDER_ERROR_BODY_CHARS);
    assert!(truncated.contains("... [truncated 50 chars]"));
    assert![truncated.len() < long.len()];
}

#[test]
fn a_thrown_value_stringifies_as_the_sdk_message() {
    let error = SdkError::from_thrown(&json!({"reason": "boom"}));
    assert_eq![error.message, "{\"reason\":\"boom\"}"];
    assert_eq![error.status_code, None];
    assert_eq![error.status, None];

    let norm = normalize_provider_error(error);
    assert_eq![norm.status, None];
    assert_eq![norm.body, None];
    assert_eq![norm.message, "{\"reason\":\"boom\"}"];
}

#[test]
fn the_bedrock_response_status_is_the_last_status_probe() {
    // Only $response.statusCode is populated: the lowest-priority probe.
    let error = SdkError {
        message: String::from("failed"),
        response: Some(SdkResponse {
            status_code: Some(503),
            body: None,
        }),
        ..SdkError::default()
    };
    let norm = normalize_provider_error(error);
    assert_eq![norm.status, Some(503)];
    assert![norm.message_carries_body];
}

#[test]
fn an_empty_body_string_is_no_body() {
    let error = SdkError {
        message: String::from("gateway timeout"),
        status_code: Some(504),
        body: Some(String::from("   ")),
        ..SdkError::default()
    };
    let norm = normalize_provider_error(error);
    assert_eq![norm.body, None];
    assert!(norm.message_carries_body);
}

#[test]
fn the_bedrock_parsed_body_object_stringifies_like_the_openai_one() {
    let error = SdkError {
        message: String::from("404 status code (no body)"),
        response: Some(SdkResponse {
            status_code: Some(404),
            body: Some(ErrorBody::Parsed(json!({"message": "model not found"}))),
        }),
        ..SdkError::default()
    };
    let norm = normalize_provider_error(error);
    assert_eq![norm.status, Some(404)];
    assert_eq![
        norm.body.as_deref(),
        Some("{\"message\":\"model not found\"}")
    ];
    assert!(!norm.message_carries_body);

    // An empty parsed body under $response stays no-body, like under error.
    let empty = SdkError {
        message: String::from("404 status code (no body)"),
        response: Some(SdkResponse {
            status_code: Some(404),
            body: Some(ErrorBody::Parsed(json!({}))),
        }),
        ..SdkError::default()
    };
    let norm = normalize_provider_error(empty);
    assert_eq![norm.body, None];
    assert!(norm.message_carries_body);
}

#[test]
fn formatting_falls_back_to_the_bare_message_without_a_status() {
    let norm = normalize_thrown_value(&json!("just a string"));
    // The thrown value stringifies into the message; without a status the
    // prefix never composes.
    assert_eq![
        format_provider_error(&norm, Some("Prefix")),
        "\"just a string\""
    ];
    assert_eq![format_provider_error(&norm, None), "\"just a string\""];

    // A message that already carries the body keeps it unchanged; the
    // prefix composes with the status only.
    let carrying = SdkError {
        message: String::from("500: upstream exploded"),
        status_code: Some(500),
        body: Some(String::from("upstream exploded")),
        ..SdkError::default()
    };
    let norm = normalize_provider_error(carrying);
    assert_eq![
        format_provider_error(&norm, Some("Bedrock")),
        "Bedrock (500): 500: upstream exploded"
    ];
}
