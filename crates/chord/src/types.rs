//! Core data model of the runtime, ported from upstream `src/types.ts`.
//!
//! The owned [`JsonValue`] tree, the service vocabulary (`ServiceType`
//! singleton and keyed variants, `ServiceCall`, `ServiceProviderUpdate`,
//! `ServiceCatalogueEntry`, `ServiceInstanceAddress`), the `Service` and
//! `Context` contracts, and the facet-host plus remote-transport surfaces.
//! Contracts TypeScript enforces only at compile time (`RemoteServiceContract`,
//! `JsonRepresentation`) restate here as trait bounds over owned wire-safe
//! data.

/// A finite JSON number.
///
/// Upstream spells this half of `JsonValue` as the JavaScript `number`; the
/// newtype keeps the constructor surface the place where non-finite values
/// die, so a [`JsonValue`] tree can never carry `NaN` or an infinity and
/// every consumer may assume `Number.isFinite` semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsonNumber(f64);

impl JsonNumber {
    /// The finite value, or [`None`] for `NaN` and the infinities.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// The value as a plain float. Every returned value is finite.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl From<i64> for JsonNumber {
    fn from(value: i64) -> Self {
        #[allow(
            clippy::cast_precision_loss,
            reason = "JavaScript numbers are f64, so the wire loses precision past 2^53 for any producer; the constructor mirrors that"
        )]
        Self(value as f64)
    }
}

impl From<u64> for JsonNumber {
    fn from(value: u64) -> Self {
        #[allow(
            clippy::cast_precision_loss,
            reason = "JavaScript numbers are f64, so the wire loses precision past 2^53 for any producer; the constructor mirrors that"
        )]
        Self(value as f64)
    }
}

/// A JSON object preserving insertion order.
///
/// JavaScript object semantics are the contract the port mirrors: assigning
/// to an existing key updates it in place, a fresh key appends at the end,
/// and equality ignores order because delta replication preserves JSON
/// values, not insertion order.
#[derive(Debug, Clone, Default)]
pub struct JsonObject {
    entries: Vec<(String, JsonValue)>,
}

impl JsonObject {
    /// An empty object.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an object from `(key, value)` pairs in iteration order.
    #[must_use]
    pub fn from_entries(entries: Vec<(String, JsonValue)>) -> Self {
        let mut object = Self::default();
        for (key, value) in entries {
            object.set(key, value);
        }
        object
    }

    /// The value stored under `key`, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.entries
            .iter()
            .find(|(stored, _)| stored == key)
            .map(|(_, value)| value)
    }

    /// The value stored under `key`, mutable, if present.
    pub fn as_value_mut(&mut self, key: &str) -> Option<&mut JsonValue> {
        self.entries
            .iter_mut()
            .find(|(stored, _)| stored == key)
            .map(|(_, value)| value)
    }

    /// Assigns `key`: an existing entry updates in place, a fresh one appends.
    pub fn set(&mut self, key: impl Into<String>, value: JsonValue) {
        let key = key.into();
        match self.entries.iter_mut().find(|(stored, _)| *stored == key) {
            Some((_, stored)) => *stored = value,
            None => self.entries.push((key, value)),
        }
    }

    /// Removes `key`, returning its value if it was present.
    pub fn remove(&mut self, key: &str) -> Option<JsonValue> {
        let at = self.entries.iter().position(|(stored, _)| stored == key)?;
        Some(self.entries.remove(at).1)
    }

    /// Whether `key` is stored.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(stored, _)| stored == key)
    }

    /// The keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(key, _)| key.as_str())
    }

    /// The number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the object holds no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The `(key, value)` entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &JsonValue)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }
}

/// Strict JSON: finite numbers, plain objects, no cycles.
///
/// Owned data restates the guarantees `isJsonValue` checks for at runtime in
/// TypeScript: a [`JsonNumber`] cannot be non-finite, a [`JsonObject`] cannot
/// carry an exotic prototype, and a tree owned by value cannot cycle, so
/// [`crate::json::is_json_value`] only has depth left to police.
#[derive(Debug, Clone)]
#[allow(
    clippy::enum_variant_names,
    reason = "the variant names mirror the upstream JsonValue union, which spells JsonValue"
)]
pub enum JsonValue {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A finite number.
    Number(JsonNumber),
    /// A string.
    Str(String),
    /// An ordered array; holes cannot be represented.
    Array(Vec<Self>),
    /// An object with string keys in insertion order.
    Object(JsonObject),
}

impl JsonValue {
    /// A finite number, or [`None`] when `value` is `NaN` or infinite.
    #[must_use]
    pub fn number(value: f64) -> Option<Self> {
        Some(Self::Number(JsonNumber::new(value)?))
    }

    /// A string value.
    #[must_use]
    pub fn string(value: impl Into<String>) -> Self {
        Self::Str(value.into())
    }

    /// The string content, or [`None`] when this is not a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    /// The number as a plain float, or [`None`] when this is not a number.
    #[must_use]
    pub const fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(number) => Some(number.get()),
            _ => None,
        }
    }

    /// Whether the value is a container: an array or an object.
    ///
    /// Upstream spells this `isObj`; every walk, diff, and resolve branches on
    /// it.
    #[must_use]
    pub const fn is_container(&self) -> bool {
        matches!(self, Self::Array(_) | Self::Object(_))
    }

    /// The array elements, or [`None`] when this is not an array.
    #[must_use]
    pub const fn as_array(&self) -> Option<&Vec<Self>> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The object entries, or [`None`] when this is not an object.
    #[must_use]
    pub const fn as_object(&self) -> Option<&JsonObject> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// The object as a mutable entry list, or [`None`] when this is not an
    /// object.
    pub const fn as_object_mut(&mut self) -> Option<&mut JsonObject> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// The array as a mutable element list, or [`None`] when this is not an
    /// array.
    pub const fn as_array_mut(&mut self) -> Option<&mut Vec<Self>> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Compact JSON text, the form `JSON.stringify` produces for the tree.
    ///
    /// Object keys serialize in insertion order, matching upstream's
    /// `JSON.stringify`.
    ///
    /// # Panics
    /// Never: every [`JsonNumber`] is finite by construction.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Number(value) => {
                let text = value.get().to_string();
                out.push_str(&text);
            }
            Self::Str(text) => write_json_string(text, out),
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write_json(out);
                }
                out.push(']');
            }
            Self::Object(object) => {
                out.push('{');
                for (index, (key, value)) in object.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_json_string(key, out);
                    out.push(':');
                    value.write_json(out);
                }
                out.push('}');
            }
        }
    }
}

/// Writes `text` as a JSON string literal, escaping quotes, backslashes, and
/// the control characters `JSON.stringify` escapes.
fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            character if (character as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", character as u32);
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

/// Equality of JSON values as data: arrays element-wise in order, objects by
/// key sets.
///
/// JavaScript's structural comparison and vitest's `toEqual` ignore object
/// insertion order — delta replication preserves JSON values, not order — so
/// the derived order-sensitive comparison would fail ports of upstream
/// assertions that hold. Strings stay compared exactly; numbers keep float
/// equality, where `-0.0 == 0.0` matches `===`.
impl PartialEq for JsonValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Number(left), Self::Number(right)) => left == right,
            (Self::Str(left), Self::Str(right)) => left == right,
            (Self::Array(left), Self::Array(right)) => left == right,
            (Self::Object(left), Self::Object(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(key, value)| {
                        right
                            .get(key)
                            .is_some_and(|other_value| value == other_value)
                    })
            }
            _ => false,
        }
    }
}

/// Equality is an equivalence relation: numbers are finite (no `NaN`), so
/// reflexivity, symmetry, and transitivity all hold, and object comparison is
/// key-set equality, which is an equivalence on maps.
impl Eq for JsonValue {}

impl Eq for JsonNumber {}
