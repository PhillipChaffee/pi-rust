//! The CBOR encoder, ported from upstream `src/cbor/encoder.ts` 1:1.

use super::options::{
    CborError, CborOptions, MAX_SAFE_INTEGER, MAX_UINT32, ResolvedCborOptions, UINT32_BASE,
    resolve_options,
};
use super::value::CborValue;

struct CborWriter {
    buffer: Vec<u8>,
    max_byte_length: usize,
}

impl CborWriter {
    fn new(max_byte_length: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_byte_length.min(256)),
            max_byte_length,
        }
    }

    fn write_byte(&mut self, value: u8) -> Result<(), CborError> {
        self.ensure_capacity(1)?;
        self.buffer.push(value);
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), CborError> {
        self.ensure_capacity(bytes.len())?;
        self.buffer.extend_from_slice(bytes);
        Ok(())
    }

    fn write_uint16(&mut self, value: u16) -> Result<(), CborError> {
        self.ensure_capacity(2)?;
        self.buffer.extend_from_slice(&value.to_be_bytes());
        Ok(())
    }

    fn write_uint32(&mut self, value: u32) -> Result<(), CborError> {
        self.ensure_capacity(4)?;
        self.buffer.extend_from_slice(&value.to_be_bytes());
        Ok(())
    }

    fn write_uint64(&mut self, value: u64) -> Result<(), CborError> {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "arguments reach here bounded to the safe-integer range, so high < 2^21 and low < 2^32"
        )]
        let (high, low) = ((value / UINT32_BASE) as u32, (value % UINT32_BASE) as u32);
        self.write_uint32(high)?;
        self.write_uint32(low)
    }

    fn write_float64(&mut self, value: f64) -> Result<(), CborError> {
        self.write_byte(0xfb)?;
        self.buffer
            .extend_from_slice(&value.to_bits().to_be_bytes());
        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        self.buffer
    }

    fn ensure_capacity(&mut self, additional_bytes: usize) -> Result<(), CborError> {
        let required = self.buffer.len() + additional_bytes;
        if required > self.max_byte_length {
            return Err(CborError::new(format!(
                "CBOR byte length exceeds configured limit of {}",
                self.max_byte_length
            )));
        }
        self.buffer.reserve(additional_bytes);
        Ok(())
    }
}

fn write_argument(writer: &mut CborWriter, major_type: u8, value: u64) -> Result<(), CborError> {
    let prefix = major_type << 5;
    if value < 24 {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the arm guards value < 24, well inside u8"
        )]
        let head = prefix | value as u8;
        writer.write_byte(head)
    } else if value <= 0xff {
        writer.write_byte(prefix | 0x18)?;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the arm guards value <= 0xff"
        )]
        let argument = value as u8;
        writer.write_byte(argument)
    } else if value <= 0xffff {
        writer.write_byte(prefix | 0x19)?;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the arm guards value <= 0xffff"
        )]
        let argument = value as u16;
        writer.write_uint16(argument)
    } else if value <= MAX_UINT32 {
        writer.write_byte(prefix | 0x1a)?;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the arm guards value <= 0xffff_ffff"
        )]
        let argument = value as u32;
        writer.write_uint32(argument)
    } else {
        writer.write_byte(prefix | 0x1b)?;
        writer.write_uint64(value)
    }
}

fn encode_text(
    writer: &mut CborWriter,
    value: &str,
    options: &ResolvedCborOptions,
) -> Result<(), CborError> {
    let bytes = value.as_bytes();
    if bytes.len() > options.max_byte_length {
        return Err(CborError::new(format!(
            "CBOR text string length exceeds configured limit of {}",
            options.max_byte_length
        )));
    }
    // Upstream re-decodes the bytes to catch surrogate-only JS strings; a
    // Rust `String` is valid UTF-8 by construction, so the bytes are the
    // string and the round-trip cannot be lossy.
    write_argument(writer, 3, bytes.len() as u64)?;
    writer.write_bytes(bytes)
}

fn encode_value(
    writer: &mut CborWriter,
    value: &CborValue,
    options: &ResolvedCborOptions,
    depth: usize,
) -> Result<(), CborError> {
    if depth > options.max_depth {
        return Err(CborError::new(format!(
            "CBOR nesting depth exceeds configured limit of {}",
            options.max_depth
        )));
    }

    match value {
        CborValue::Null => writer.write_byte(0xf6),
        CborValue::Bool(flag) => writer.write_byte(if *flag { 0xf5 } else { 0xf4 }),
        CborValue::Int(integer) => {
            if *integer < -MAX_SAFE_INTEGER || *integer > MAX_SAFE_INTEGER {
                return Err(CborError::new(
                    "CBOR integers must be safe JavaScript integers",
                ));
            }
            if *integer >= 0 {
                write_argument(writer, 0, (*integer).cast_unsigned())
            } else {
                // `-1 - integer` cannot overflow past the safe bound above.
                write_argument(writer, 1, (-1 - *integer).cast_unsigned())
            }
        }
        CborValue::Float(number) => encode_float(writer, *number),
        CborValue::Text(text) => encode_text(writer, text, options),
        CborValue::Bytes(bytes) => {
            if bytes.len() > options.max_byte_length {
                return Err(CborError::new(format!(
                    "CBOR byte string length exceeds configured limit of {}",
                    options.max_byte_length
                )));
            }
            write_argument(writer, 2, bytes.len() as u64)?;
            writer.write_bytes(bytes)
        }
        CborValue::Array(items) => {
            if items.len() > options.max_container_length {
                return Err(CborError::new(format!(
                    "CBOR array length exceeds configured limit of {}",
                    options.max_container_length
                )));
            }
            // Holes and `undefined` elements are unrepresentable in the
            // owned tree; upstream rejects them from the JS array surface.
            write_argument(writer, 4, items.len() as u64)?;
            for item in items {
                encode_value(writer, item, options, depth + 1)?;
            }
            Ok(())
        }
        CborValue::Map(entries) => {
            // JS plain objects cannot carry duplicate keys; a hand-built map
            // can, and the wire would read it back as a decoder rejection.
            let mut keys = std::collections::HashSet::with_capacity(entries.len());
            for (key, _) in entries {
                if !keys.insert(key.as_str()) {
                    return Err(CborError::new("CBOR map contains a duplicate key"));
                }
            }
            if entries.len() > options.max_container_length {
                return Err(CborError::new(format!(
                    "CBOR map length exceeds configured limit of {}",
                    options.max_container_length
                )));
            }
            // Symbol keys are unrepresentable; upstream rejects them from
            // the JS object surface.
            write_argument(writer, 5, entries.len() as u64)?;
            for (key, entry_value) in entries {
                encode_text(writer, key, options)?;
                encode_value(writer, entry_value, options, depth + 1)?;
            }
            Ok(())
        }
    }
}

fn encode_float(writer: &mut CborWriter, number: f64) -> Result<(), CborError> {
    if !number.is_finite() {
        return Err(CborError::new("CBOR numbers must be finite"));
    }
    // Upstream takes the integer path when `Number.isInteger(value)` holds
    // and the value is not `-0` (`Object.is`); `-0` keeps the float64 path
    // so it round-trips as `0xfb8000000000000000`.
    let is_negative_zero = number == 0.0 && number.is_sign_negative();
    if number.fract() == 0.0 && !is_negative_zero {
        #[allow(
            clippy::cast_precision_loss,
            reason = "the safe-integer bound makes the f64 representation exact"
        )]
        let max_safe_integer = MAX_SAFE_INTEGER as f64;
        if number.abs() > max_safe_integer {
            return Err(CborError::new(
                "CBOR integers must be safe JavaScript integers",
            ));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the guard bounds the value inside the safe-integer range"
        )]
        #[allow(
            clippy::cast_sign_loss,
            reason = "both branches produce non-negative safe integers: the guard bounds the value and -1.0 - number flips the sign"
        )]
        let argument = if number >= 0.0 {
            number as u64
        } else {
            (-1.0 - number) as u64
        };
        if number >= 0.0 {
            write_argument(writer, 0, argument)
        } else {
            write_argument(writer, 1, argument)
        }
    } else {
        writer.write_float64(number)
    }
}

/// Encodes the protocol's strict, definite-length RFC 8949 subset.
///
/// # Errors
///
/// Returns [`CborError`] for values or limits the subset rejects: non-finite
/// or unsafe numbers, string or container lengths past the configured
/// limits, nesting past the depth limit, and duplicate map keys on
/// hand-built maps.
pub fn encode_cbor(value: &CborValue, options: &CborOptions) -> Result<Vec<u8>, CborError> {
    let resolved = resolve_options(options)?;
    let mut writer = CborWriter::new(resolved.max_byte_length);
    encode_value(&mut writer, value, &resolved, 0)?;
    Ok(writer.finish())
}
