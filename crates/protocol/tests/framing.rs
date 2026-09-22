//! The framing suite, ported from upstream `test/framing.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]

use pi_protocol::{DEFAULT_MAX_FRAME_LENGTH, FrameDecoder, FrameDecoderOptions, encode_frame};

fn concatenate(chunks: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::with_capacity(chunks.iter().map(|chunk| chunk.len()).sum());
    for chunk in chunks {
        result.extend_from_slice(chunk);
    }
    result
}

fn new_decoder() -> FrameDecoder {
    FrameDecoder::new(FrameDecoderOptions::default()).expect("default options are valid")
}

fn bounded_decoder(max_frame_length: usize) -> FrameDecoder {
    FrameDecoder::new(FrameDecoderOptions { max_frame_length }).expect("bounded options are valid")
}

#[test]
fn prefixes_payloads_with_a_four_byte_big_endian_length() {
    assert_eq!(
        encode_frame(&[0xaa, 0xbb, 0xcc]).expect("within the u32 length limit"),
        vec![0x00, 0x00, 0x00, 0x03, 0xaa, 0xbb, 0xcc]
    );
    assert_eq!(
        encode_frame(&[]).expect("within the u32 length limit"),
        vec![0, 0, 0, 0]
    );
}

#[test]
fn decodes_fragmented_coalesced_and_empty_frames_in_order() {
    let wire = concatenate(&[
        &encode_frame(&[1, 2, 3]).expect("within the u32 length limit"),
        &encode_frame(&[]).expect("within the u32 length limit"),
        &encode_frame(&[4]).expect("within the u32 length limit"),
    ]);
    let mut decoder = new_decoder();
    let mut frames = Vec::new();
    for byte in &wire {
        frames.extend(decoder.push(&[*byte]).expect("clean stream"));
    }
    decoder.end().expect("clean stream");
    assert_eq!(frames, vec![vec![1, 2, 3], Vec::<u8>::new(), vec![4]]);

    let mut coalesced = new_decoder();
    assert_eq!(coalesced.push(&wire).expect("clean stream"), frames);
    coalesced.end().expect("clean stream");
}

#[test]
fn assembles_payloads_spanning_multiple_internal_blocks() {
    let payload: Vec<u8> = (0..70_000u32)
        .map(|index| {
            #[allow(
                clippy::cast_sign_loss,
                reason = "index % 251 is a non-negative value inside u8"
            )]
            let value = (index % 251) as u8;
            value
        })
        .collect();
    let wire = encode_frame(&payload).expect("within the u32 length limit");
    let mut frame_decoder = new_decoder();
    let frames = [
        frame_decoder.push(&wire[..101]).expect("clean stream"),
        frame_decoder
            .push(&wire[101..65_541])
            .expect("clean stream"),
        frame_decoder.push(&wire[65_541..]).expect("clean stream"),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    frame_decoder.end().expect("clean stream");
    assert_eq!(frames, vec![payload]);
}

#[test]
fn handles_every_split_point_across_a_frame() {
    let wire = encode_frame(&[10, 20, 30, 40]).expect("within the u32 length limit");
    for split in 0..=wire.len() {
        let mut frame_decoder = new_decoder();
        let frames = [
            frame_decoder.push(&wire[..split]).expect("clean stream"),
            frame_decoder.push(&wire[split..]).expect("clean stream"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        frame_decoder.end().expect("clean stream");
        assert_eq!(frames, vec![vec![10, 20, 30, 40]]);
    }
}

#[test]
fn copies_payload_bytes_instead_of_retaining_or_aliasing_input_chunks() {
    let mut chunk = encode_frame(&[1, 2, 3]).expect("within the u32 length limit");
    let mut frame_decoder = new_decoder();
    let frames = frame_decoder.push(&chunk).expect("clean stream");
    chunk.fill(9);
    assert_eq!(frames, vec![vec![1, 2, 3]]);
}

#[test]
fn accepts_empty_chunks_and_a_clean_empty_stream() {
    let mut frame_decoder = new_decoder();
    assert_eq!(
        frame_decoder.push(&[]).expect("clean stream"),
        Vec::<Vec<u8>>::new()
    );
    frame_decoder.end().expect("clean stream");
}

#[test]
fn rejects_a_truncated_stream_at_end_partial_header() {
    let mut frame_decoder = new_decoder();
    assert_eq!(
        frame_decoder
            .push(&[0x00, 0x00, 0x00])
            .expect("header stays partial"),
        Vec::<Vec<u8>>::new()
    );
    let error = frame_decoder.end().expect_err("truncated at end");
    assert!(error.message().contains("Truncated"));
}

#[test]
fn rejects_a_truncated_stream_at_end_partial_payload() {
    let mut frame_decoder = new_decoder();
    assert_eq!(
        frame_decoder
            .push(&[0x00, 0x00, 0x00, 0x02, 0x01])
            .expect("payload stays partial"),
        Vec::<Vec<u8>>::new()
    );
    let error = frame_decoder.end().expect_err("truncated at end");
    assert!(error.message().contains("Truncated"));
}

#[test]
fn rejects_an_oversized_declared_length_as_soon_as_its_header_is_complete() {
    let mut frame_decoder = bounded_decoder(3);
    let error = frame_decoder
        .push(&[0x00, 0x00, 0x00, 0x04])
        .expect_err("declared length is over the limit");
    assert!(error.message().to_lowercase().contains("limit"));
    let error = frame_decoder.push(&[1]).expect_err("failed state latches");
    assert!(error.message().to_lowercase().contains("failed"));
}

#[test]
fn accepts_a_frame_exactly_at_the_configured_maximum() {
    let mut frame_decoder = bounded_decoder(3);
    let payload = encode_frame(&[1, 2, 3]).expect("within the u32 length limit");
    assert_eq!(
        frame_decoder.push(&payload).expect("clean stream"),
        vec![vec![1, 2, 3]]
    );
    frame_decoder.end().expect("clean stream");
}

#[test]
fn cannot_be_pushed_after_end() {
    let mut frame_decoder = new_decoder();
    frame_decoder.end().expect("clean stream");
    let error = frame_decoder.push(&[]).expect_err("ended latches");
    assert!(error.message().to_lowercase().contains("ended"));
    let error = frame_decoder.end().expect_err("ended latches");
    assert!(error.message().to_lowercase().contains("ended"));
}

#[test]
fn end_reports_the_failed_state_after_a_framing_failure() {
    let mut bounded = bounded_decoder(3);
    assert!(bounded.push(&[0x00, 0x00, 0x00, 0x04]).is_err());
    let error = bounded.end().expect_err("failed state latches");
    assert!(error.message().contains("failed"));
}

#[test]
fn rejects_invalid_maximum_frame_length() {
    // Upstream also rejects -1, 1.5, and NaN: the `usize` limit cannot hold
    // negative, fractional, or non-finite numbers, so those inputs are
    // unrepresentable.
    let options = FrameDecoderOptions {
        max_frame_length: DEFAULT_MAX_FRAME_LENGTH * 1000,
    };
    let error = FrameDecoder::new(options).expect_err("past the u32 range");
    assert!(error.message().contains("must be an integer between 0 and"));
}
