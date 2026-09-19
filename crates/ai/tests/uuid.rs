//! The uuidv7 port, from `test/uuid.test.ts`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_ai::utils::uuid::{
    MAX_UUID_V7_TIMESTAMP, RandomSource, UuidV7Error, UuidV7Generator, uuidv7,
};

const TIMESTAMP: u64 = 0x0123_4567_89ab;

fn parse_timestamp(uuid: &str) -> u64 {
    let compact = uuid.replace('-', "");
    u64::from_str_radix(&compact[..12], 16).expect("the first 12 hex digits are the timestamp")
}

/// The upstream suite pins the RFC shape with a regex; the same check by
/// hand: lowercase hex, dashes at 8/13/18/23, version 7, variant in 89ab.
fn assert_uuid_shape(uuid: &str) {
    let bytes = uuid.as_bytes();
    assert_eq!(bytes.len(), 36, "expected uuidv7 shape, got {uuid}");
    for dash in [8, 13, 18, 23] {
        assert_eq!(bytes[dash], b'-', "expected uuidv7 shape, got {uuid}");
    }
    for (index, ch) in uuid.chars().enumerate() {
        if [8, 13, 18, 23].contains(&index) {
            continue;
        }
        if index == 14 {
            assert_eq!(ch, '7', "expected version 7, got {uuid}");
            continue;
        }
        if index == 19 {
            assert!(
                ch == '8' || ch == '9' || ch == 'a' || ch == 'b',
                "expected variant 89ab, got {uuid}"
            );
            continue;
        }
        assert!(
            ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase(),
            "expected lowercase hex, got {uuid}"
        );
    }
}

/// A randomness source that fills every byte with an incrementing counter,
/// the `vi.stubGlobal("crypto", ...)` stub's behavior.
struct CountingRandom {
    next: std::sync::Mutex<u8>,
}

impl CountingRandom {
    const fn new() -> Self {
        Self {
            next: std::sync::Mutex::new(0),
        }
    }
}

impl RandomSource for CountingRandom {
    fn fill(&self, bytes: &mut [u8]) {
        let mut next = self
            .next
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *next = next.wrapping_add(1);
        bytes.fill(*next);
    }
}

#[test]
fn generates_ordered_uuidv7s_while_preserving_follower_timestamps() {
    let clock = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(TIMESTAMP));
    let clock_for_generator = clock.clone();
    let generator = UuidV7Generator::with_clock_and_random(
        move || clock_for_generator.load(std::sync::atomic::Ordering::SeqCst),
        CountingRandom::new(),
    );

    let first = generator.generate(None).expect("first id");
    let second = generator.generate(None).expect("second id");
    let after_rollback = {
        clock.store(TIMESTAMP - 1, std::sync::atomic::Ordering::SeqCst);
        generator.generate(None).expect("rollback id")
    };
    let after_advance = {
        clock.store(TIMESTAMP + 1, std::sync::atomic::Ordering::SeqCst);
        generator.generate(None).expect("advanced id")
    };
    let ordinary_ids = [first, second, after_rollback, after_advance];
    let follower_timestamp = TIMESTAMP - 1_000;
    let followers = [
        generator
            .generate(Some(follower_timestamp))
            .expect("follower"),
        generator
            .generate(Some(follower_timestamp))
            .expect("follower"),
    ];

    for id in ordinary_ids.iter().chain(followers.iter()) {
        assert_uuid_shape(id);
    }
    let parsed: Vec<u64> = ordinary_ids.iter().map(|id| parse_timestamp(id)).collect();
    let distinct: std::collections::BTreeSet<String> = ordinary_ids.iter().cloned().collect();
    let mut sorted = ordinary_ids.clone();
    sorted.sort();
    assert_eq!(parsed, [TIMESTAMP, TIMESTAMP, TIMESTAMP, TIMESTAMP + 1]);
    assert_eq!(ordinary_ids, sorted, "ordinary ids sort by timestamp");
    assert_eq!(distinct.len(), 4, "ordinary ids are unique");
    assert_eq!(
        followers
            .iter()
            .map(|id| parse_timestamp(id))
            .collect::<Vec<_>>(),
        [follower_timestamp, follower_timestamp]
    );
    let follower_set: std::collections::BTreeSet<&String> = followers.iter().collect();
    assert_eq!(follower_set.len(), 2, "followers are unique");
}

#[test]
fn uses_fresh_randomness_for_every_uuid_tail() {
    let generator = UuidV7Generator::with_clock_and_random(|| TIMESTAMP, CountingRandom::new());

    let first = generator.generate(Some(TIMESTAMP)).expect("first");
    let second = generator.generate(Some(TIMESTAMP)).expect("second");
    // The last 8 hex chars are the unmasked random bytes.
    assert_eq![(&first[28..], &second[28..]), ("01010101", "02020202")];
}

#[test]
fn accepts_timestamp_boundaries() {
    for timestamp in [0u64, 0xffff_ffff_ffff] {
        let id = uuidv7(Some(timestamp)).expect("boundary id");
        assert_eq!(parse_timestamp(&id), timestamp);
    }
    assert_eq!(MAX_UUID_V7_TIMESTAMP, 0xffff_ffff_ffff);
}

#[test]
fn rejects_invalid_timestamps() {
    assert_eq!(
        uuidv7(Some(MAX_UUID_V7_TIMESTAMP + 1)).expect_err("above the 48-bit range"),
        UuidV7Error::TimestampOutOfRange
    );
    // Upstream also rejects -1, 1.5, NaN, and Infinity; those are
    // unrepresentable as a u64 millisecond count, so only the range error
    // can fire here.
}

#[test]
fn the_error_message_carries_the_wire_range() {
    let error = uuidv7(Some(u64::MAX)).expect_err("out of range");
    assert_eq!(
        error.to_string(),
        format!("UUIDv7 timestamp must be an integer between 0 and {MAX_UUID_V7_TIMESTAMP}")
    );
}

#[test]
fn the_generator_debugs_as_an_opaque_struct() {
    let generator = UuidV7Generator::with_clock_and_random(|| TIMESTAMP, CountingRandom::new());
    let debug = format!("{generator:?}");
    assert_eq![debug, "UuidV7Generator { .. }", "{debug}"];

    let system_debug = format!("{:?}", UuidV7Generator::system());
    assert_eq![system_debug, "UuidV7Generator { .. }"];
}

fn unix_millis() -> u64 {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is after the epoch")
        .as_millis();
    u64::try_from(millis).expect("milliseconds since the epoch fit u64")
}

#[test]
fn the_system_generator_reads_the_running_clock() {
    // The system generator reads Date.now()-equivalent milliseconds, so the
    // id it returns must carry a timestamp within a second of now.
    let generator = UuidV7Generator::system();
    let before = unix_millis();
    let id = generator
        .generate(None)
        .expect("the system generator works");
    let after = unix_millis();

    assert_uuid_shape(&id);
    let timestamp = parse_timestamp(&id);
    let lower = before.saturating_sub(1_000);
    assert!(
        (lower..=after.saturating_add(1_000)).contains(&timestamp),
        "timestamp {timestamp} must sit near [{before}, {after}]"
    );

    // The process generator never issues two identical ordinary ids.
    let first = uuidv7(None).expect("the process generator works");
    let second = uuidv7(None).expect("the process generator repeats");
    assert![
        second > first,
        "ids sort by time, got {first} then {second}"
    ];
}
