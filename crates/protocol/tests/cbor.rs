//! The CBOR codec suite, ported from upstream `test/cbor/cbor.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The known-vector hex
//! round-trips and the decoder-rejection fixtures are the byte-exact
//! porting oracle ([ADR 0002](../../docs/adr/0002-hand-ported-cbor-codec.md)).

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]

use pi_protocol::{
    CborError, CborOptions, CborValue, DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH, DEFAULT_MAX_CBOR_DEPTH, decode_cbor, encode_cbor,
};

fn from_hex(hex: &str) -> Vec<u8> {
    assert!(
        hex.len().is_multiple_of(2),
        "hex fixture must contain whole bytes"
    );
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("hex digit pair"))
        .collect()
}

fn to_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = std::fmt::Write::write_fmt(&mut hex, format_args!("{byte:02x}"));
    }
    hex
}

fn options() -> CborOptions {
    CborOptions::default()
}

#[rustfmt::skip]
fn known_vectors() -> Vec<(CborValue, &'static str)> {
    let map = |entries: Vec<(&str, CborValue)>| CborValue::map(entries);
    vec![
        (CborValue::Null, "f6"),
        (CborValue::Bool(false), "f4"),
        (CborValue::Bool(true), "f5"),
        (CborValue::Int(0), "00"),
        (CborValue::Int(1), "01"),
        (CborValue::Int(10), "0a"),
        (CborValue::Int(23), "17"),
        (CborValue::Int(24), "1818"),
        (CborValue::Int(25), "1819"),
        (CborValue::Int(100), "1864"),
        (CborValue::Int(1000), "1903e8"),
        (CborValue::Int(1_000_000), "1a000f4240"),
        (CborValue::Int(1_000_000_000_000), "1b000000e8d4a51000"),
        (CborValue::Int(9_007_199_254_740_991), "1b001fffffffffffff"),
        (CborValue::Int(-1), "20"),
        (CborValue::Int(-10), "29"),
        (CborValue::Int(-24), "37"),
        (CborValue::Int(-25), "3818"),
        (CborValue::Int(-100), "3863"),
        (CborValue::Int(-1000), "3903e7"),
        (CborValue::Int(-9_007_199_254_740_991), "3b001ffffffffffffe"),
        (CborValue::Float(1.1), "fb3ff199999999999a"),
        (CborValue::Float(-0.0), "fb8000000000000000"),
        (CborValue::Bytes(vec![1, 2, 3, 4]), "4401020304"),
        (CborValue::Text(String::new()), "60"),
        (CborValue::Text("IETF".to_string()), "6449455446"),
        (CborValue::Text("ü".to_string()), "62c3bc"),
        (CborValue::Text("水".to_string()), "63e6b0b4"),
        (CborValue::Text("𐅑".to_string()), "64f0908591"),
        (CborValue::Array(vec![]), "80"),
        (CborValue::Array(vec![CborValue::Int(1), CborValue::Int(2), CborValue::Int(3)]), "83010203"),
        (CborValue::Array(vec![
            CborValue::Int(1),
            CborValue::Array(vec![CborValue::Int(2), CborValue::Int(3)]),
            CborValue::Array(vec![CborValue::Int(4), CborValue::Int(5)]),
        ]), "8301820203820405"),
        (map(vec![("a", CborValue::Int(1)), ("b", CborValue::Array(vec![CborValue::Int(2), CborValue::Int(3)]))]), "a26161016162820203"),
    ]
}

#[test]
fn encodes_and_decodes_rfc_8949_vectors() {
    for (value, wire) in known_vectors() {
        let encoded = encode_cbor(&value, &options()).expect("known vector encodes");
        assert_eq!(to_hex(&encoded), wire);
        let decoded = decode_cbor(&from_hex(wire), &options()).expect("known vector decodes");
        assert_eq!(decoded, value);
        // `-0` round-trips as float64 `fb8000000000000000`, whose bit pattern
        // the generic float equality above cannot pin (`-0.0 == 0.0`).
        if value == CborValue::Float(-0.0) {
            let CborValue::Float(decoded) = decoded else {
                panic!("-0 decodes to a float");
            };
            assert_eq!(decoded.to_bits(), 0x8000_0000_0000_0000);
        }
    }
}

#[test]
fn encodes_the_object_entries_it_holds_without_omitting_falsey_values() {
    // Upstream's encoder skips `undefined` map values while keeping falsey
    // ones; `undefined` has no variant in the owned tree, so the omission is
    // the entry list itself.
    let value = CborValue::map(vec![
        ("zero", CborValue::Int(0)),
        ("empty", CborValue::Text(String::new())),
        ("no", CborValue::Bool(false)),
        ("nil", CborValue::Null),
    ]);
    let decoded = decode_cbor(
        &encode_cbor(&value, &options()).expect("encodes"),
        &options(),
    )
    .expect("decodes");
    assert_eq!(decoded, value);
}

#[test]
fn preserves_a_leading_unicode_bom_and_treats_proto_as_data() {
    let decoded = decode_cbor(&from_hex("63efbbbf"), &options()).expect("BOM text decodes");
    assert_eq!(decoded, CborValue::Text("\u{feff}".to_string()));

    // A JS object's own `__proto__` property stays data: the owned map has
    // no prototype to collide with, so the round trip carries the key.
    let value = CborValue::map(vec![("__proto__", CborValue::Text("safe".to_string()))]);
    let encoded = encode_cbor(&value, &options()).expect("encodes");
    assert_eq!(decode_cbor(&encoded, &options()).expect("decodes"), value);
}

#[test]
fn rejects_unsupported_encoder_values() {
    // The representable cases from upstream's list: non-finite numbers and
    // integers outside the safe range.
    for (value, wire) in [
        (CborValue::Float(f64::NAN), "NaN"),
        (CborValue::Float(f64::INFINITY), "positive infinity"),
        (CborValue::Float(f64::NEG_INFINITY), "negative infinity"),
        (
            CborValue::Int(9_007_199_254_740_992),
            "unsafe positive integer",
        ),
        (
            CborValue::Int(-9_007_199_254_740_992),
            "unsafe negative integer",
        ),
        (CborValue::Float(9_007_199_254_740_992.0), "unsafe integer"),
    ] {
        let error =
            encode_cbor(&value, &options()).expect_err(&format!("rejects {value:?} ({wire})"));
        assert!(error.message().contains("must") || error.message().contains("finite"));
    }
    // Upstream's remaining cases — top-level `undefined`, an undefined
    // array element, an array hole, a bigint, a symbol, a function, a
    // `Date`, and a `Map` — have no variant in the owned tree, and a lossy
    // string (`"\ud800"`) cannot exist in a Rust `String`: the enum and
    // `String` are the encoder's rejection surface.
}

#[test]
fn rejects_hand_built_duplicate_map_keys() {
    // Upstream rejects maps with enumerable symbol keys because a JS plain
    // object's keys are strings; the owned map's key hazard is a duplicate,
    // which the wire decoder would reject on read.
    let value = CborValue::Map(vec![
        ("a".to_string(), CborValue::Int(1)),
        ("a".to_string(), CborValue::Int(2)),
    ]);
    let error = encode_cbor(&value, &options()).expect_err("duplicate key");
    assert!(error.message().contains("duplicate key"));
}

#[test]
fn rejects_excessive_encoder_depth() {
    // Upstream also rejects lossy strings (`"\ud800"`) and cycles, which
    // are unrepresentable in a `String` and an owned tree.
    let mut too_deep = CborValue::Null;
    for _ in 0..=DEFAULT_MAX_CBOR_DEPTH {
        too_deep = CborValue::Array(vec![too_deep]);
    }
    let error = encode_cbor(&too_deep, &options()).expect_err("past the depth limit");
    assert!(error.message().contains("depth"));
}

#[test]
fn rejects_invalid_decoder_input() {
    let rejections: [(&str, &str); 29] = [
        ("empty input", ""),
        ("truncated integer", "18"),
        ("reserved additional information", "1c"),
        ("indefinite byte string", "5f"),
        ("indefinite text string", "7f"),
        ("indefinite array", "9f"),
        ("indefinite map", "bf"),
        ("tag", "c000"),
        ("undefined", "f7"),
        ("unsupported simple value", "e0"),
        ("break outside an indefinite item", "ff"),
        ("float16", "f93c00"),
        ("float32", "fa3f800000"),
        ("positive infinity", "fb7ff0000000000000"),
        ("NaN", "fb7ff8000000000000"),
        ("truncated float64", "fb3ff00000"),
        ("truncated byte string", "44010203"),
        ("truncated text string", "636162"),
        ("truncated array", "8201"),
        ("truncated map", "a16161"),
        ("trailing data", "0000"),
        ("non-string map key", "a10102"),
        ("duplicate map key", "a2616101616102"),
        ("invalid UTF-8 byte", "61ff"),
        ("overlong UTF-8", "62c080"),
        ("UTF-8 surrogate", "63eda080"),
        ("unsafe positive integer", "1b0020000000000000"),
        ("unsafe negative integer", "3b001fffffffffffff"),
        ("unsafe integer encoded as float64", "fb4340000000000000"),
    ];
    for (label, wire) in rejections {
        let error: CborError = decode_cbor(&from_hex(wire), &options())
            .expect_err(&format!("rejects {label}: {wire}"));
        assert!(!error.message().is_empty());
    }
}

#[test]
fn enforces_depth_and_declared_length_limits_before_traversing_values() {
    let mut too_deep = vec![0x81_u8; DEFAULT_MAX_CBOR_DEPTH + 2];
    *too_deep.last_mut().expect("non-empty") = 0xf6;
    let error = decode_cbor(&too_deep, &options()).expect_err("past the depth limit");
    assert!(error.message().contains("depth"));

    let oversized = |head: u8, length: u64| {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the declared lengths sit a single step past the u32-bound defaults"
        )]
        let (head_length, length) = (head, length as u32);
        let mut wire = vec![head_length];
        wire.extend_from_slice(&length.to_be_bytes());
        wire
    };
    let wires = [
        oversized(0x5a, DEFAULT_MAX_CBOR_BYTE_LENGTH as u64 + 1),
        oversized(0x7a, DEFAULT_MAX_CBOR_BYTE_LENGTH as u64 + 1),
        oversized(0x9a, DEFAULT_MAX_CBOR_CONTAINER_LENGTH as u64 + 1),
        oversized(0xba, DEFAULT_MAX_CBOR_CONTAINER_LENGTH as u64 + 1),
    ];
    for wire in wires {
        let error = decode_cbor(&wire, &options()).expect_err("declared length over the limit");
        assert!(error.message().to_lowercase().contains("limit"));
    }
}

#[test]
fn supports_stricter_caller_provided_limits() {
    let container = CborOptions {
        max_container_length: 2,
        ..CborOptions::default()
    };
    let error =
        decode_cbor(&from_hex("83010203"), &container).expect_err("array past the container limit");
    assert!(error.message().to_lowercase().contains("limit"));

    let byte = CborOptions {
        max_byte_length: 2,
        ..CborOptions::default()
    };
    let error = decode_cbor(&from_hex("626162"), &byte).expect_err("bytes over the limit");
    assert!(error.message().to_lowercase().contains("limit"));

    let error = encode_cbor(
        &CborValue::Array(vec![
            CborValue::Int(1),
            CborValue::Int(2),
            CborValue::Int(3),
        ]),
        &container,
    )
    .expect_err("array past the container limit");
    assert!(error.message().to_lowercase().contains("limit"));

    let error = encode_cbor(&CborValue::Text("ab".to_string()), &byte)
        .expect_err("text over the byte limit");
    assert!(error.message().to_lowercase().contains("limit"));
}
