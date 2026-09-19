//! Rust-native suites for the JSON-parse belt: the `partial-json` parser
//! behaviors verified against the pinned 0.1.7 build, `repairJson`, and the
//! streaming fallback ladder.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_ai::utils::json_parse::{
    Allow, PartialJsonError, parse_json_with_repair, parse_partial_json, parse_streaming_json,
    repair_json,
};
use serde_json::{Value, json};

fn parse(input: &str) -> Result<Value, PartialJsonError> {
    parse_partial_json(input, Allow::ALL)
}

#[test]
fn parses_completed_prefixes_of_objects_arrays_and_strings() {
    let cases: &[(&str, Value)] = &[
        ("{\"path\":\"READ", json!({"path": "READ"})),
        ("{\"path\":\"READ\",", json!({"path": "READ"})),
        ("{\"a\":1,\"b\":2,\"c\":", json!({"a": 1, "b": 2})),
        ("[1, 2,", json!([1, 2])),
        ("\"hello \\u12", json!("hello ")),
        ("123", json!(123)),
        ("-1", json!(-1)),
        ("1e", json!(1)),
        ("1e5", json!(100_000)),
        ("1e5x", json!(1)),
        ("nul", json!(null)),
        ("tru", json!(true)),
        ("nullx", json!(null)),
        ("  {\"x\": 1}  ", json!({"x": 1})),
        ("{\"a\": {\"b\": \"partial", json!({"a": {"b": "partial"}})),
        ("{\"na", json!({})),
        ("{\"a\": tru}", json!({})),
        ("[{\"a\":1},{\"b\":", json!([{"a": 1}, {}])),
    ];
    for (input, expected) in cases {
        let parsed = parse(input).expect("the input parses to its completed prefix");
        assert_eq![&parsed, expected, "input {input}"];
    }
}

#[test]
fn refuses_inputs_it_cannot_salvage() {
    for input in ["123.", "1.", "012", "-", "]", "garbage"] {
        assert![parse(input).is_err(), "input {input} must fail"];
    }
    let error = parse("-").expect_err("a bare dash is malformed");
    assert![error.0.contains("Not sure what '-' is"), "{}", error.0];
}

#[test]
fn the_allow_mask_gates_partial_forms() {
    // A truncated object with OBJ cleared fails; the whole form parses.
    assert![parse_partial_json("{\"a\": 1, \"b\":", Allow::ALL).is_ok()];
    assert![parse_partial_json("{\"a\": 1, \"b\":", Allow::ALL ^ Allow::OBJ).is_err()];
    assert![parse_partial_json("[1, 2,", Allow::ARR).is_ok()];
    assert![parse_partial_json("[1, 2,", Allow::ALL ^ Allow::ARR).is_err()];
    assert![parse_partial_json("\"hello \\u12", Allow::ALL ^ Allow::STR).is_err()];
}

#[test]
fn repair_escapes_raw_control_characters_inside_strings() {
    assert_eq![
        repair_json("{\"a\": \"b\u{1}c\"}"),
        "{\"a\": \"b\\u0001c\"}"
    ];
    assert_eq![repair_json("a\nb"), "a\nb"];
    assert_eq![
        repair_json("{\"a\": \"line\nbreak\"}"),
        "{\"a\": \"line\\nbreak\"}"
    ];
    // Each named escape and the generic \\uXXXX form have their own arm.
    assert_eq![repair_json("\"\u{8}\u{c}\r\t\""), "\"\\b\\f\\r\\t\""];
    assert_eq![repair_json("{\"é\": \"\u{7}\"}"), "{\"é\": \"\\u0007\"}"];
    // Control characters outside strings pass through untouched.
    assert_eq![repair_json("{\u{1}}"), "{\u{1}}"];
}

#[test]
fn repair_doubles_backslashes_before_invalid_escapes() {
    assert_eq![repair_json("{\"a\": \"b\\xc\"}"), "{\"a\": \"b\\\\xc\"}"];
    assert_eq![
        repair_json("{\"a\": \"trailing\\\"}"),
        "{\"a\": \"trailing\\\"}"
    ];
    assert_eq![repair_json("{\"a\": \"\\uZZZZ\"}"), "{\"a\": \"\\uZZZZ\"}"];
    assert_eq![repair_json("{\"a\": \"\\n\"}"), "{\"a\": \"\\n\"}"];
    // A complete \\uXXXX escape keeps its digits.
    assert_eq![repair_json("{\"a\": \"\\u00e9\"}"), "{\"a\": \"\\u00e9\"}"];
    // A \\u escape with too few digits keeps the escape and lets the
    // digits pass as ordinary text.
    assert_eq![repair_json("{\"a\": \"\\u0\"}"), "{\"a\": \"\\u0\"}"];
}

#[test]
fn parse_with_repair_falls_back_only_when_the_repair_changed_the_input() {
    assert_eq![
        parse_json_with_repair("{\"a\": 1}").expect("parses"),
        json!({"a": 1})
    ];
    assert_eq![
        parse_json_with_repair("{\"a\": \"b\x01\"}").expect("repairs"),
        json!({"a": "b\u{1}"})
    ];
    assert![parse_json_with_repair("{not json at all}").is_err()];
}

#[test]
fn streaming_parse_always_returns_a_value() {
    assert_eq![parse_streaming_json(None), json!({})];
    assert_eq![parse_streaming_json(Some("")), json!({})];
    assert_eq![parse_streaming_json(Some("   ")), json!({})];
    assert_eq![parse_streaming_json(Some("{\"a\": 1}")), json!({"a": 1})];
    assert_eq![
        parse_streaming_json(Some("{\"a\": \"partial")),
        json!({"a": "partial"})
    ];
    assert_eq![parse_streaming_json(Some("garbage")), json!({})];
    // A complete null parses as null through the repair path; a truncated
    // one only reaches the partial fallback, where `result ?? {}` folds the
    // parsed null into the empty object.
    assert_eq![parse_streaming_json(Some("null")), json!(null)];
    assert_eq![parse_streaming_json(Some("nu")), json!({})];
    // A raw control character breaks the direct parse and the first
    // partial attempt (a bare string with a control char); the repaired
    // partial fallback salvages the string with the escaped form inside.
    assert_eq![parse_streaming_json(Some("\"b\u{1}")), json!("b\u{1}")];
}

#[test]
fn non_integral_numbers_keep_their_fraction_and_wide_integers_keep_precision() {
    assert_eq![parse("1.5").expect("a float"), json!(1.5)];
    let wide = parse("123456789012345678901234567890")
        .expect("a number beyond the i64 window stays a number");
    assert![wide.is_number()];
    assert_eq![parse("-2.5").expect("a negative float"), json!(-2.5)];
}

#[test]
fn a_blank_input_is_an_error_with_the_input_in_the_message() {
    let error = parse_partial_json("   ", Allow::ALL).expect_err("blank is not JSON");
    assert![error.0.contains("is empty"), "{}", error.0];
    assert_eq![error.to_string(), error.0, "the display is the message"];
    let _ = Box::<dyn std::error::Error>::from(error.clone());
    assert![std::error::Error::source(&error).is_none()];
}

#[test]
fn the_special_literals_parse_complete_and_partial() {
    assert_eq![parse("false").expect("false"), json!(false)];
    assert_eq![parse("fa").expect("false prefix"), json!(false)];
    assert_eq![parse("Infinity").expect("infinity"), json!(null)];
    assert_eq![parse("Inf").expect("infinity prefix"), json!(null)];
    assert_eq![parse("-Infinity").expect("negative infinity"), json!(null)];
    assert_eq![
        parse("-Inf").expect("negative infinity prefix"),
        json!(null)
    ];
    let dash = parse("-").expect_err("a bare dash is malformed");
    assert![dash.0.contains("Not sure what '-' is"), "{}", dash.0];
    assert_eq![parse("NaN").expect("not a number"), json!(null)];
    assert_eq![parse("Na").expect("nan prefix"), json!(null)];
}

#[test]
fn truncated_object_keys_report_their_inner_shape() {
    // A raw control character makes the key string unparsable: the
    // degenerate fallback candidate is invalid too, the key error
    // propagates to the object, and the object truncates.
    let with_control_key = "{\"a\": 1, \"b\u{1}";
    assert_eq![
        parse(with_control_key).expect("the object truncates"),
        json!({"a": 1})
    ];
    // A key fragment that parses as a non-string literal reports the
    // not-a-string failure, and the object still truncates.
    assert_eq![parse("{\"nu").expect("the object truncates"), json!({})];
}

#[test]
fn the_string_fallback_repairs_from_the_last_backslash_of_the_input() {
    // An invalid escape sequence after the string start trims the escape
    // away and closes the string there.
    assert_eq![parse("\"a\\q").expect("the escape is trimmed"), json!("a")];
    // A raw control character invalidates the string head and the input
    // holds no backslash, so the degenerate candidate is the whole prefix
    // before the string start — the bare quote at the top level — and the
    // parse fails with the escape-sequence error.
    let error = parse("\"a\u{1}b").expect_err("the fallback cannot salvage a control char");
    assert![error.0.contains("Invalid escape sequence"), "{}", error.0];
    // Inside an object the same fallback trims at a backslash that sits in
    // an earlier value, degenerate to a swap that fails; the object still
    // truncates to its completed prefix.
    assert_eq![
        parse("{\"a\": \"\\\\q\", \"b\u{1}").expect("the object truncates"),
        json!({"a": "\\q"})
    ];
}

#[test]
fn container_numbers_take_the_partial_and_exponent_paths() {
    assert_eq![
        parse("[1, -2]").expect("a negative element"),
        json!([1, -2])
    ];
    assert_eq![parse("[1e5x]").expect("the exponent retry"), json!([1])];
    assert_eq![parse("[]").expect("an empty array"), json!([])];
    // A truncated number with the NUM bit cleared fails the element and the
    // array truncates.
    assert_eq![
        parse_partial_json("[1, 2", Allow::ARR).expect("the array truncates"),
        json!([1])
    ];
    assert_eq![
        parse_partial_json("[1, -", Allow::ARR | Allow::NUM).expect("the dash is dropped"),
        json!([1])
    ];
}
