//! The owned CBOR value tree.
//!
//! Upstream encodes and decodes plain JavaScript values — `null`, booleans,
//! numbers, strings, `Uint8Array`, arrays, and plain objects with own
//! enumerable string keys. Owned Rust data restates that domain as this
//! enum: the variants are the encoder's rejection surface, because the JS
//! values upstream rejects (`undefined`, holes, bigints, symbols, functions,
//! `Date`, `Map`, cycles, non-plain prototypes) have no variant to occupy.
//! Map entries keep insertion order, the wire order the decoder preserves,
//! and duplicate keys stay unrepresentable on the decode side while the
//! encoder polices them for hand-built values.

/// One value in the protocol's strict, definite-length CBOR subset.
#[derive(Debug, Clone, PartialEq)]
pub enum CborValue {
    /// CBOR `null` (`0xf6`).
    Null,
    /// CBOR `true` (`0xf5`) or `false` (`0xf4`).
    Bool(bool),
    /// A safe integer, bounded to ±2^53-1 like upstream's `Number.isSafeInteger`.
    Int(i64),
    /// A finite IEEE-754 float64 (`0xfb`), including `-0`.
    Float(f64),
    /// A UTF-8 text string (major type 3).
    Text(String),
    /// A byte string (major type 2), upstream's `Uint8Array`.
    Bytes(Vec<u8>),
    /// An array (major type 4).
    Array(Vec<Self>),
    /// A map with string keys in insertion order (major type 5).
    Map(Vec<(String, Self)>),
}

impl CborValue {
    /// Builds a [`CborValue::Map`] from `(key, value)` pairs in iteration
    /// order.
    #[must_use]
    pub fn map(entries: Vec<(&str, Self)>) -> Self {
        Self::Map(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }
}
