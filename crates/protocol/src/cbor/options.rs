//! Encoder/decoder options and the crate's `CborError`, ported from upstream
//! `src/cbor/options.ts`.

use std::fmt;

/// The `0x1_0000_0000` divisor upstream splits 64-bit arguments with.
pub(crate) const UINT32_BASE: u64 = 0x1_0000_0000;

/// The largest unsigned 32-bit argument, the `MAX_UINT32` bound upstream
/// resolves option ranges against.
pub(crate) const MAX_UINT32: u64 = 0xffff_ffff;

const MAX_CONFIGURED_DEPTH: usize = 512;

/// JavaScript's `Number.MAX_SAFE_INTEGER`, the ±(2^53-1) integer bound the
/// codec shares between the integer and float64 paths.
pub(crate) const MAX_SAFE_INTEGER: i64 = (1 << 53) - 1;

/// Safe defaults for untrusted protocol payloads.
pub const DEFAULT_MAX_CBOR_BYTE_LENGTH: usize = 16 * 1024 * 1024;
/// Safe defaults for untrusted protocol payloads.
pub const DEFAULT_MAX_CBOR_CONTAINER_LENGTH: usize = 1_000_000;
/// Safe defaults for untrusted protocol payloads.
pub const DEFAULT_MAX_CBOR_DEPTH: usize = 64;

/// Limits for encoding and decoding, ported from upstream's `CborOptions`.
///
/// Upstream spells each limit as an optional number defaulting to the safe
/// default; the plain fields with [`Default`] carry the same contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CborOptions {
    /// Maximum encoded input/output bytes and maximum byte/text string length.
    pub max_byte_length: usize,
    /// Maximum number of elements in an array or entries in a map.
    pub max_container_length: usize,
    /// Maximum recursive item depth.
    pub max_depth: usize,
}

impl Default for CborOptions {
    fn default() -> Self {
        Self {
            max_byte_length: DEFAULT_MAX_CBOR_BYTE_LENGTH,
            max_container_length: DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
            max_depth: DEFAULT_MAX_CBOR_DEPTH,
        }
    }
}

/// The resolved form of [`CborOptions`], ported from upstream's
/// `ResolvedCborOptions`.
///
/// The field names mirror upstream's `maxByteLength`/`maxContainerLength`/
/// `maxDepth` option vocabulary, so the `max` prefix is the wire-facing
/// spelling, not noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_field_names,
    reason = "the fields mirror upstream's maxByteLength/maxContainerLength/maxDepth option names"
)]
pub(crate) struct ResolvedCborOptions {
    /// The resolved maximum encoded byte length.
    pub max_byte_length: usize,
    /// The resolved maximum container length.
    pub max_container_length: usize,
    /// The resolved maximum depth.
    pub max_depth: usize,
}

/// The error the CBOR codec raises, ported from upstream's `CborError`.
///
/// Upstream extends `Error` with `name = "CborError"`; the distinct type is
/// the same discrimination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CborError {
    message: String,
}

impl CborError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The error message, the upstream `Error.message` surface.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CborError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for CborError {}

/// Resolves option limits against their configured ranges, ported from
/// upstream's `resolveOptions`.
///
/// # Errors
///
/// Returns [`CborError`] when a limit falls outside its range: byte and
/// container lengths must sit in `0..=0xffff_ffff`, depth in `0..=512`.
/// Upstream throws a `RangeError` with the same message; the resolved
/// `usize` fields make non-integer inputs unrepresentable.
pub(crate) fn resolve_options(options: &CborOptions) -> Result<ResolvedCborOptions, CborError> {
    Ok(ResolvedCborOptions {
        max_byte_length: resolve_limit("maxByteLength", options.max_byte_length, MAX_UINT32)?,
        max_container_length: resolve_limit(
            "maxContainerLength",
            options.max_container_length,
            MAX_UINT32,
        )?,
        max_depth: resolve_limit("maxDepth", options.max_depth, MAX_CONFIGURED_DEPTH as u64)?,
    })
}

fn resolve_limit(name: &str, value: usize, maximum: u64) -> Result<usize, CborError> {
    if value as u64 > maximum {
        return Err(CborError::new(format!(
            "{name} must be an integer between 0 and {maximum}"
        )));
    }
    Ok(value)
}
