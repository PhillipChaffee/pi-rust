//! `test/rpc-jsonl.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, plus the chunk-boundary
//! cases the Node stream reader gets from `StringDecoder`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

use pi_coding_agent::modes::rpc::jsonl::JsonlLineDecoder;
use pi_coding_agent::modes::rpc::jsonl::attach_jsonl_line_reader;
use pi_coding_agent::modes::rpc::jsonl::serialize_json_line;
use serde_json::json;
use tokio::sync::mpsc;

#[test]
fn serializes_strict_jsonl_records_without_escaping_unicode_separators() {
    let line = serialize_json_line(&json!({ "text": "a\u{2028}b\u{2029}c" }));

    assert!(line.contains("a\u{2028}b\u{2029}c"));
    assert!(line.ends_with('\n'));
    let parsed: serde_json::Value = serde_json::from_str(line.trim()).expect("the line parses");
    assert_eq!(parsed, json!({ "text": "a\u{2028}b\u{2029}c" }));
}

#[test]
fn splits_on_lf_only_and_preserves_unicode_separators_inside_payloads() {
    let mut decoder = JsonlLineDecoder::default();
    let line = serialize_json_line(&json!({ "text": "a\u{2028}b\u{2029}c" }));
    let lines = decoder.feed(line.as_bytes());

    assert_eq!(lines.len(), 1);
    let parsed: serde_json::Value = serde_json::from_str(&lines[0]).expect("the line parses");
    assert_eq!(parsed, json!({ "text": "a\u{2028}b\u{2029}c" }));
}

#[test]
fn handles_crlf_delimited_input() {
    let mut decoder = JsonlLineDecoder::default();
    let lines = decoder.feed(b"{\"a\":1}\r\n{\"b\":2}\r\n");
    assert_eq!(lines, vec!["{\"a\":1}".to_owned(), "{\"b\":2}".to_owned()]);
}

#[test]
fn emits_a_final_line_without_trailing_lf() {
    let mut decoder = JsonlLineDecoder::default();
    assert!(decoder.feed(b"{\"a\":1}").is_empty());
    assert_eq!(decoder.finish(), Some("{\"a\":1}".to_owned()));
    assert_eq!(decoder.finish(), None);
}

#[test]
fn a_chunk_boundary_inside_a_line_waits_for_the_newline() {
    let mut decoder = JsonlLineDecoder::default();
    assert_eq!(decoder.feed(b"{\"a\":"), Vec::<String>::new());
    assert_eq!(decoder.feed(b"1}\n"), vec!["{\"a\":1}".to_owned()]);
}

#[test]
fn a_multibyte_character_split_across_chunks_decodes_whole() {
    let mut decoder = JsonlLineDecoder::default();
    let bytes = "{\"t\":\"é\"}\n".as_bytes();
    let split = bytes.len() - 2;
    assert!(decoder.feed(&bytes[..split]).is_empty());
    let lines = decoder.feed(&bytes[split..]);
    assert_eq!(lines, vec!["{\"t\":\"é\"}".to_owned()]);
}

#[test]
fn only_one_trailing_carriage_return_is_stripped() {
    let mut decoder = JsonlLineDecoder::default();
    assert_eq!(
        decoder.feed(b"a\r\r\n"),
        vec!["a\r".to_owned()],
        "upstream strips exactly one \\r"
    );
}

#[tokio::test]
async fn the_attached_reader_emits_every_line_and_closes_at_eof() {
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let input: &[u8] = b"{\"a\":1}\n{\"b\":2}";
    attach_jsonl_line_reader(input, sender).await;

    let mut lines = Vec::new();
    while let Some(line) = receiver.recv().await {
        lines.push(line);
    }
    assert_eq!(lines, vec!["{\"a\":1}".to_owned(), "{\"b\":2}".to_owned()]);
}
