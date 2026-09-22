//! The CBOR decoder, ported from upstream `src/cbor/decoder.ts` 1:1.

use super::options::{
    CborError, CborOptions, MAX_SAFE_INTEGER, ResolvedCborOptions, UINT32_BASE, resolve_options,
};
use super::value::CborValue;

struct CborReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    options: ResolvedCborOptions,
}

impl CborReader<'_> {
    fn decode(&mut self) -> Result<CborValue, CborError> {
        let value = self.read_item(0)?;
        if self.offset != self.bytes.len() {
            return Err(CborError::new("CBOR payload contains trailing data"));
        }
        Ok(value)
    }

    fn read_item(&mut self, depth: usize) -> Result<CborValue, CborError> {
        if depth > self.options.max_depth {
            return Err(CborError::new(format!(
                "CBOR nesting depth exceeds configured limit of {}",
                self.options.max_depth
            )));
        }
        let initial = self.read_byte()?;
        // The dispatch covers every byte 0x00..=0xff, so the major-type
        // switch's provably-dead default arm upstream carries has no Rust
        // counterpart.
        match initial {
            0x00..=0x1f => {
                let value = self.read_argument(initial & 0x1f)?;
                #[allow(
                    clippy::cast_sign_loss,
                    reason = "the argument surface bounds the value to the safe-integer range"
                )]
                Ok(CborValue::Int(value.cast_signed()))
            }
            0x20..=0x3f => {
                #[allow(
                    clippy::cast_sign_loss,
                    reason = "the argument surface bounds the value to the safe-integer range"
                )]
                let argument = self.read_argument(initial & 0x1f)?.cast_signed();
                let value = -1 - argument;
                if !(-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&value) {
                    return Err(CborError::new(
                        "Decoded CBOR integer is outside the safe range",
                    ));
                }
                Ok(CborValue::Int(value))
            }
            0x40..=0x5f => {
                let length =
                    self.read_length(initial & 0x1f, "byte string", self.options.max_byte_length)?;
                Ok(CborValue::Bytes(self.read_bytes(length)?.to_vec()))
            }
            0x60..=0x7f => {
                let length =
                    self.read_length(initial & 0x1f, "text string", self.options.max_byte_length)?;
                let bytes = self.read_bytes(length)?;
                String::from_utf8(bytes.to_vec())
                    .map(CborValue::Text)
                    .map_err(|_| CborError::new("CBOR text string contains invalid UTF-8"))
            }
            0x80..=0x9f => {
                let length =
                    self.read_length(initial & 0x1f, "array", self.options.max_container_length)?;
                let mut result = Vec::new();
                for _ in 0..length {
                    result.push(self.read_item(depth + 1)?);
                }
                Ok(CborValue::Array(result))
            }
            0xa0..=0xbf => {
                let length =
                    self.read_length(initial & 0x1f, "map", self.options.max_container_length)?;
                let mut result = Vec::new();
                let mut keys = std::collections::HashSet::new();
                for _ in 0..length {
                    let key = self.read_item(depth + 1)?;
                    let CborValue::Text(key) = key else {
                        return Err(CborError::new("CBOR map keys must be strings"));
                    };
                    if !keys.insert(key.clone()) {
                        return Err(CborError::new("CBOR map contains a duplicate key"));
                    }
                    let value = self.read_item(depth + 1)?;
                    result.push((key, value));
                }
                Ok(CborValue::Map(result))
            }
            0xc0..=0xdf => Err(CborError::new("CBOR tags are not supported")),
            0xe0..=0xff => self.read_simple(initial & 0x1f),
        }
    }

    fn read_simple(&mut self, additional_information: u8) -> Result<CborValue, CborError> {
        match additional_information {
            20 => Ok(CborValue::Bool(false)),
            21 => Ok(CborValue::Bool(true)),
            22 => Ok(CborValue::Null),
            27 => {
                let bytes = self.read_bytes(8)?;
                #[allow(
                    clippy::indexing_slicing,
                    reason = "read_bytes returned exactly the 8 bytes the slice indexes"
                )]
                let value = f64::from_bits(u64::from_be_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]));
                if !value.is_finite() {
                    return Err(CborError::new("Decoded CBOR number must be finite"));
                }
                // `Number.isInteger` is true for every float64 whose fraction
                // is zero; such a value outside the safe range is rejected
                // exactly as upstream rejects it (e.g. 2^53 itself).
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "the safe-integer bound makes the f64 representation exact"
                )]
                let max_safe_integer = MAX_SAFE_INTEGER as f64;
                if value.fract() == 0.0 && value.abs() > max_safe_integer {
                    return Err(CborError::new(
                        "Decoded CBOR integer is outside the safe range",
                    ));
                }
                Ok(CborValue::Float(value))
            }
            31 => Err(CborError::new("CBOR break marker is not supported")),
            _ => Err(CborError::new(
                "Unsupported CBOR simple value or floating-point width",
            )),
        }
    }

    fn read_length(
        &mut self,
        additional_information: u8,
        kind: &str,
        limit: usize,
    ) -> Result<usize, CborError> {
        if additional_information == 31 {
            return Err(CborError::new(format!(
                "Indefinite-length CBOR {kind}s are not supported"
            )));
        }
        let length = self.read_argument(additional_information)?;
        if length > limit as u64 {
            return Err(CborError::new(format!(
                "CBOR {kind} length exceeds configured limit of {limit}"
            )));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "length <= limit, a usize, so the value fits usize"
        )]
        Ok(length as usize)
    }

    fn read_argument(&mut self, additional_information: u8) -> Result<u64, CborError> {
        if additional_information < 24 {
            return Ok(u64::from(additional_information));
        }
        match additional_information {
            24 => Ok(u64::from(self.read_byte()?)),
            25 => {
                let bytes = self.read_bytes(2)?;
                #[allow(
                    clippy::indexing_slicing,
                    reason = "read_bytes returned exactly the 2 bytes the slice indexes"
                )]
                Ok(u64::from(u16::from_be_bytes([bytes[0], bytes[1]])))
            }
            26 => {
                let bytes = self.read_bytes(4)?;
                #[allow(
                    clippy::indexing_slicing,
                    reason = "read_bytes returned exactly the 4 bytes the slice indexes"
                )]
                Ok(u64::from(u32::from_be_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                ])))
            }
            27 => {
                let high = self.read_argument(26)?;
                let low = self.read_argument(26)?;
                if high > 0x1f_ffff {
                    return Err(CborError::new(
                        "Decoded CBOR integer or length is outside the safe range",
                    ));
                }
                Ok(high * UINT32_BASE + low)
            }
            31 => Err(CborError::new(
                "Indefinite-length CBOR items are not supported",
            )),
            _ => Err(CborError::new("Malformed CBOR additional information")),
        }
    }

    fn read_byte(&mut self) -> Result<u8, CborError> {
        if self.offset >= self.bytes.len() {
            return Err(CborError::new("Truncated CBOR payload"));
        }
        #[allow(
            clippy::indexing_slicing,
            reason = "the offset guard above bounds the index"
        )]
        let value = self.bytes[self.offset];
        self.offset += 1;
        Ok(value)
    }

    fn read_bytes(&mut self, length: usize) -> Result<&[u8], CborError> {
        if length > self.bytes.len() - self.offset {
            return Err(CborError::new("Truncated CBOR payload"));
        }
        #[allow(
            clippy::indexing_slicing,
            reason = "the length guard above bounds the range"
        )]
        let value = &self.bytes[self.offset..self.offset + length];
        self.offset += length;
        Ok(value)
    }
}

/// Decodes exactly one item from the protocol's strict RFC 8949 subset.
///
/// # Errors
///
/// Returns [`CborError`] for every input the subset rejects: indefinite
/// lengths, tags, break markers, float16/float32, non-finite or unsafe
/// numbers, non-string or duplicate map keys, invalid UTF-8, truncation,
/// trailing data, and byte lengths or limits past the configured options.
pub fn decode_cbor(bytes: &[u8], options: &CborOptions) -> Result<CborValue, CborError> {
    let resolved = resolve_options(options)?;
    if bytes.len() > resolved.max_byte_length {
        return Err(CborError::new(format!(
            "CBOR byte length exceeds configured limit of {}",
            resolved.max_byte_length
        )));
    }
    CborReader {
        bytes,
        offset: 0,
        options: resolved,
    }
    .decode()
}
