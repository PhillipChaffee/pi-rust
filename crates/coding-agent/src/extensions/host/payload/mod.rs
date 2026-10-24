//! The wire mappings for the payload domain, [ADR 0010]'s serialization
//! slice (#169): extension events, handler results, the declarative tool
//! surface, the `ExtensionContext` snapshot, and the command, flag, and
//! provider descriptors.
//!
//! Mappings live here, not on the domain types, so `serde_json::Value` is
//! the only wire currency and camelCase field names stay pinned to
//! upstream's TypeScript shapes (pin `60e7e76`). Types the port already
//! serializes — messages, models, session entries — ride their own serde;
//! the extension-owned vocabularies (reasons, preparations, descriptors)
//! map by hand. `AbortSignal` crosses as its wire signal id through the
//! pluggable resolver closures the dispatch slice (#170) binds.
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

pub(in crate::extensions::host) mod descriptors;
pub(in crate::extensions::host) mod events;
pub(in crate::extensions::host) mod results;
#[cfg(test)]
mod tests;

use super::WireError;

/// Error text prefix every mapping failure carries, so a dispatch
/// diagnostic can tell a payload defect from a frame defect.
const PAYLOAD_ERROR_PREFIX: &str = "extension payload";

fn payload_error(field: &str) -> WireError {
    WireError::new(format!("{PAYLOAD_ERROR_PREFIX}: {field}"))
}

/// The object view a payload map is read through.
fn fields<'a>(
    payload: &'a serde_json::Value,
    kind: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, WireError> {
    payload
        .as_object()
        .ok_or_else(|| payload_error(&format!("{kind} payload must be an object")))
}

fn required_string(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<String, WireError> {
    match fields.get(key) {
        Some(serde_json::Value::String(text)) => Ok(text.clone()),
        _ => Err(payload_error(&format!(
            "{kind} field `{key}` must be a string"
        ))),
    }
}

fn optional_string(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<Option<String>, WireError> {
    match fields.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(payload_error(&format!(
            "{kind} field `{key}` must be a string"
        ))),
    }
}

fn required_bool(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<bool, WireError> {
    match fields.get(key) {
        Some(serde_json::Value::Bool(value)) => Ok(*value),
        _ => Err(payload_error(&format!(
            "{kind} field `{key}` must be a boolean"
        ))),
    }
}

fn optional_bool(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<Option<bool>, WireError> {
    match fields.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(payload_error(&format!(
            "{kind} field `{key}` must be a boolean"
        ))),
    }
}

fn required_u64(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<u64, WireError> {
    fields
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            payload_error(&format!(
                "{kind} field `{key}` must be a non-negative number"
            ))
        })
}

fn optional_u64(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<Option<u64>, WireError> {
    match fields.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(value)) => value.as_u64().map(Some).ok_or_else(|| {
            payload_error(&format!(
                "{kind} field `{key}` must be a non-negative number"
            ))
        }),
        Some(_) => Err(payload_error(&format!(
            "{kind} field `{key}` must be a non-negative number"
        ))),
    }
}

fn required_value<'a>(
    fields: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<&'a serde_json::Value, WireError> {
    fields
        .get(key)
        .ok_or_else(|| payload_error(&format!("{kind} must carry `{key}`")))
}

fn required_string_array(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<Vec<String>, WireError> {
    let entries = fields
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| payload_error(&format!("{kind} field `{key}` must be an array")))?;
    entries
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| payload_error(&format!("{kind} field `{key}` must hold strings")))
        })
        .collect()
}

fn optional_string_array(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: &str,
) -> Result<Option<Vec<String>>, WireError> {
    match fields.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(array) if array.is_array() => required_string_array(fields, key, kind).map(Some),
        Some(_) => Err(payload_error(&format!(
            "{kind} field `{key}` must be an array"
        ))),
    }
}

/// Serializes one serde-backed domain member the shared wire payload,
/// wherever the domain surface already carries the wire shape.
fn serde_to_wire<T: serde::Serialize>(
    value: &T,
    what: &str,
) -> Result<serde_json::Value, WireError> {
    serde_json::to_value(value)
        .map_err(|error| payload_error(&format!("{what} is not wire-serializable: {error}")))
}

/// Deserializes one serde-backed domain member off the wire.
fn serde_from_wire<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    what: &str,
) -> Result<T, WireError> {
    serde_json::from_value(value)
        .map_err(|error| payload_error(&format!("{what} does not ride the wire: {error}")))
}
