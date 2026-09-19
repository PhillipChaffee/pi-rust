//! A fast deterministic hash to shorten long strings, ported from
//! `packages/ai/src/utils/hash.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The algorithm mixes over UTF-16 code units (upstream's `charCodeAt`),
//! which Rust's `encode_utf16` reproduces byte for byte, so the digest is
//! identical to the TypeScript original for every input.

/// Hash a string to a short base-36 digest of its two mixed halves.
#[must_use]
pub fn short_hash(input: &str) -> String {
    let mut h1: u32 = 0xdead_beef;
    let mut h2: u32 = 0x41c6_ce57;
    for unit in input.encode_utf16() {
        let ch = u32::from(unit);
        h1 = (h1 ^ ch).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ ch).wrapping_mul(1_597_334_677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909);
    base36(h2) + &base36(h1)
}

/// The unsigned base-36 form of a 32-bit value, matching JS
/// `(value >>> 0).toString(36)`.
fn base36(value: u32) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if value == 0 {
        return "0".to_owned();
    }
    let mut digits = Vec::new();
    let mut rest = value;
    while rest > 0 {
        digits.push(DIGITS[(rest % 36) as usize]);
        rest /= 36;
    }
    digits.reverse();
    // The digit table is ASCII, so every byte in the buffer is a char.
    digits.iter().map(|byte| *byte as char).collect()
}
