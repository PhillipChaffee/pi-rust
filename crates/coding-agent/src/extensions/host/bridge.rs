//! The integer-preserving `serde_json::Value` ⇄ [`CborValue`] bridge
//! ([ADR 0010](../../../../../docs/adr/0010-extension-host-wire-protocol.md)).
//!
//! pi-protocol's remote-envelope conversion encodes every JSON number as a
//! CBOR float, which type-shifts integers on the return trip. This bridge
//! writes integral numbers as CBOR integers and floats as floats; the
//! decoder direction rejects the one value CBOR can carry that strict JSON
//! cannot, a byte string.

use pi_protocol::CborValue;
use serde_json::{Map, Number, Value};

use super::WireError;

/// Converts one owned JSON value to the CBOR tree the encoder writes.
///
/// Integral numbers in `i64` range become [`CborValue::Int`]; every other
/// number becomes [`CborValue::Float`]. A `u64` past `i64::MAX` loses
/// precision on the float path, and an integer past the codec's safe range
/// fails the frame at encode time.
#[must_use]
pub fn json_to_cbor(value: &Value) -> CborValue {
    match value {
        Value::Null => CborValue::Null,
        Value::Bool(flag) => CborValue::Bool(*flag),
        Value::Number(number) => number_to_cbor(number),
        Value::String(text) => CborValue::Text(text.clone()),
        Value::Array(items) => CborValue::Array(items.iter().map(json_to_cbor).collect()),
        Value::Object(object) => CborValue::Map(
            object
                .iter()
                .map(|(key, item)| (key.clone(), json_to_cbor(item)))
                .collect(),
        ),
    }
}

fn number_to_cbor(number: &Number) -> CborValue {
    if let Some(integer) = number.as_i64() {
        return CborValue::Int(integer);
    }
    if let Some(integer) = number.as_u64() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "ADR 0010 sends u64 values past i64::MAX as floats, the wire's only remaining number shape"
        )]
        let float = integer as f64;
        return CborValue::Float(float);
    }
    // serde_json's Number is always i64, u64, or f64 without
    // arbitrary_precision; NaN keeps the impossible arm from inventing a
    // value, and the codec rejects it at encode time.
    CborValue::Float(number.as_f64().unwrap_or(f64::NAN))
}

/// Converts one decoded CBOR item to strict JSON.
///
/// # Errors
///
/// Returns [`WireError`] when the item is not strict JSON: a byte string, or
/// a non-finite float the CBOR decoder let through.
pub fn cbor_to_json(value: &CborValue) -> Result<Value, WireError> {
    Ok(match value {
        CborValue::Null => Value::Null,
        CborValue::Bool(flag) => Value::Bool(*flag),
        CborValue::Int(integer) => Value::Number(Number::from(*integer)),
        CborValue::Float(float) => Value::Number(Number::from_f64(*float).ok_or_else(|| {
            WireError::new("extension frame payload carries a non-finite number")
        })?),
        CborValue::Text(text) => Value::String(text.clone()),
        CborValue::Bytes(_) => {
            return Err(WireError::new(
                "extension frame payload carries a CBOR byte string, which is not JSON",
            ));
        }
        CborValue::Array(items) => {
            Value::Array(items.iter().map(cbor_to_json).collect::<Result<_, _>>()?)
        }
        CborValue::Map(entries) => {
            let mut object = Map::new();
            for (key, item) in entries {
                object.insert(key.clone(), cbor_to_json(item)?);
            }
            Value::Object(object)
        }
    })
}
