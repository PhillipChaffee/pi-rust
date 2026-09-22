//! The strict, definite-length CBOR subset, ported from upstream `src/cbor/`
//! 1:1 ([ADR 0002](../../docs/adr/0002-hand-ported-cbor-codec.md)).
//!
//! The 33 RFC 8949 hex round-trip vectors and the 28 decoder-rejection
//! fixtures from upstream's suite are the porting oracle.

mod decoder;
mod encoder;
#[allow(
    clippy::redundant_pub_crate,
    reason = "encoder and decoder share the option bounds across sibling modules; crate visibility is the narrowest that carries them"
)]
mod options;
mod value;

pub use decoder::decode_cbor;
pub use encoder::encode_cbor;
pub use options::{
    CborError, CborOptions, DEFAULT_MAX_CBOR_BYTE_LENGTH, DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
};
pub use value::CborValue;
