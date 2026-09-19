//! Time-ordered UUIDv7 generation, ported from
//! `packages/ai/src/utils/uuid.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The generator produces RFC 9562 UUIDv7 ids whose first 48 bits are a
//! millisecond timestamp. Ordinary calls read the clock and never go
//! backwards (a follower id issued before an earlier timestamp still sorts
//! after the ids issued at that timestamp); explicit timestamps are preserved
//! for follower ids. The 41-bit sequence counter starts from fresh randomness
//! per process and increments within one millisecond, and the remaining bits
//! stay random.
//!
//! The clock, randomness, and monotonic state are injectable for tests;
//! [`uuidv7`] is the process-wide entry point.

use std::sync::{LazyLock, Mutex};

/// The largest millisecond timestamp a UUIDv7 can carry: 48 bits.
pub const MAX_UUID_V7_TIMESTAMP: u64 = 0xffff_ffff_ffff;
/// The 41-bit sequence counter's ceiling.
const MAX_SEQUENCE: u64 = (1 << 41) - 1;

/// Where random bytes come from, upstream's `crypto.getRandomValues`.
pub trait RandomSource: Send + Sync {
    /// Fill `bytes` with random data.
    fn fill(&self, bytes: &mut [u8]);
}

struct SystemRandom;

impl RandomSource for SystemRandom {
    fn fill(&self, bytes: &mut [u8]) {
        #[expect(
            clippy::expect_used,
            reason = "system entropy is a platform facility; an entropy failure is a platform breakage, not a runtime condition"
        )]
        getrandom::fill(bytes).expect("system random source");
    }
}

/// A uuidv7 generator with injectable clock and randomness.
///
/// The clock reads milliseconds since the Unix epoch, upstream's `Date.now`.
pub struct UuidV7Generator {
    clock: Box<dyn Fn() -> u64 + Send + Sync>,
    random: Box<dyn RandomSource>,
    state: Mutex<UuidV7State>,
}

struct UuidV7State {
    last_ordinary_timestamp: i64,
    sequence: Option<u64>,
}

impl Default for UuidV7State {
    fn default() -> Self {
        Self {
            last_ordinary_timestamp: -1,
            sequence: None,
        }
    }
}

impl std::fmt::Debug for UuidV7Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UuidV7Generator").finish_non_exhaustive()
    }
}

impl UuidV7Generator {
    /// A generator reading the system clock and the system entropy source.
    #[must_use]
    pub fn system() -> Self {
        Self {
            clock: Box::new(system_time_millis),
            random: Box::new(SystemRandom),
            state: Mutex::new(UuidV7State::default()),
        }
    }

    /// A generator with a test clock and a test randomness source.
    #[must_use]
    pub fn with_clock_and_random(
        clock: impl Fn() -> u64 + Send + Sync + 'static,
        random: impl RandomSource + 'static,
    ) -> Self {
        Self {
            clock: Box::new(clock),
            random: Box::new(random),
            state: Mutex::new(UuidV7State::default()),
        }
    }

    /// Generate a time-ordered UUIDv7.
    ///
    /// A supplied timestamp is preserved for follower ids and never rolled
    /// forward; an omitted timestamp reads the clock and is rolled forward
    /// past the last ordinary id this generator issued.
    ///
    /// # Errors
    /// Returns [`UuidV7Error::TimestampOutOfRange`] when the timestamp is not
    /// a representable 48-bit millisecond count, and
    /// [`UuidV7Error::SequenceExhausted`] when the 41-bit sequence counter
    /// overflows without the clock advancing.
    pub fn generate(&self, timestamp_ms: Option<u64>) -> Result<String, UuidV7Error> {
        let requested_timestamp = timestamp_ms.unwrap_or_else(|| (self.clock)());
        if requested_timestamp > MAX_UUID_V7_TIMESTAMP {
            return Err(UuidV7Error::TimestampOutOfRange);
        }

        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let effective_timestamp = timestamp_ms.map_or_else(
            || {
                let effective = requested_timestamp
                    .cast_signed()
                    .max(state.last_ordinary_timestamp);
                state.last_ordinary_timestamp = effective;
                effective
            },
            u64::cast_signed,
        );

        let mut bytes = [0u8; 16];
        self.random.fill(&mut bytes);
        let sequence = match state.sequence {
            None => {
                u64::from(bytes[1]) << 32
                    | u64::from(bytes[2]) << 24
                    | u64::from(bytes[3]) << 16
                    | u64::from(bytes[4]) << 8
                    | u64::from(bytes[5])
            }
            Some(sequence) => {
                if sequence == MAX_SEQUENCE {
                    return Err(UuidV7Error::SequenceExhausted);
                }
                sequence + 1
            }
        };
        state.sequence = Some(sequence);
        drop(state);

        let timestamp = effective_timestamp.cast_unsigned();
        for index in (0..=5).rev() {
            bytes[index] = ((timestamp >> ((5 - index) * 8)) & 0xff) as u8;
        }
        bytes[6] = 0x70u8 | (((sequence >> 37) & 0x0f) as u8);
        bytes[7] = ((sequence >> 29) & 0xff) as u8;
        bytes[8] = 0x80u8 | (((sequence >> 23) & 0x3f) as u8);
        bytes[9] = ((sequence >> 15) & 0xff) as u8;
        bytes[10] = ((sequence >> 7) & 0xff) as u8;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the masked shift yields at most 8 bits, so the cast is exact"
        )]
        let variant_bits = (((sequence & 0x7f) << 1) | u64::from(bytes[11] & 0x01)) as u8;
        bytes[11] = variant_bits;

        Some(format_uuid_v7(&bytes)).ok_or(UuidV7Error::TimestampOutOfRange)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "millisecond counts since the epoch fit u64 for any system clock this code will run on"
)]
fn system_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_millis() as u64)
}

/// Why a uuidv7 could not be generated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UuidV7Error {
    /// The requested timestamp is not a representable 48-bit millisecond
    /// count, upstream's `RangeError` for out-of-range timestamps.
    TimestampOutOfRange,
    /// The 41-bit sequence counter exhausted within one millisecond,
    /// upstream's `RangeError` for a full sequence.
    SequenceExhausted,
}

impl std::fmt::Display for UuidV7Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimestampOutOfRange => {
                write!(
                    f,
                    "UUIDv7 timestamp must be an integer between 0 and {MAX_UUID_V7_TIMESTAMP}"
                )
            }
            Self::SequenceExhausted => f.write_str("UUIDv7 generator sequence exhausted"),
        }
    }
}

impl std::error::Error for UuidV7Error {}

static SYSTEM_GENERATOR: LazyLock<UuidV7Generator> = LazyLock::new(UuidV7Generator::system);

/// Generate a time-ordered UUIDv7 from the process generator.
///
/// A supplied timestamp is preserved for follower ids; an omitted timestamp
/// reads the system clock, rolled forward past the last ordinary id issued.
///
/// # Errors
/// Returns [`UuidV7Error::TimestampOutOfRange`] for timestamps above 48 bits.
pub fn uuidv7(timestamp_ms: Option<u64>) -> Result<String, UuidV7Error> {
    SYSTEM_GENERATOR.generate(timestamp_ms)
}

fn format_uuid_v7(bytes: &[u8; 16]) -> String {
    use std::fmt::Write as _;
    let mut hex = String::with_capacity(32);
    for byte in bytes {
        let _ = write!(hex, "{byte:02x}");
    }
    let (head, tail) = hex.split_at(16);
    let mut formatted = String::with_capacity(36);
    formatted.push_str(&head[..8]);
    formatted.push('-');
    formatted.push_str(&head[8..12]);
    formatted.push('-');
    formatted.push_str(&head[12..]);
    formatted.push('-');
    formatted.push_str(&tail[..4]);
    formatted.push('-');
    formatted.push_str(&tail[4..]);
    formatted
}
