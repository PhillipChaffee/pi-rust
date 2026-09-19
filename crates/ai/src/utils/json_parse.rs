//! The hand-ported partial-JSON reader that [`super::json_parse`] wraps.
//!
//! Ported from the `partial-json` 0.1.7 algorithm (MIT, Szymon Wultur),
//! pinned through the upstream lockfile at the pi port pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The `Allow` bit mask and the
//! parser's control flow mirror the original: objects and strings return
//! their completed prefix, and truncated numbers, booleans, and nulls
//! resolve when the prefix is unambiguous.
//!
//! Porting restatements: `Infinity`, `-Infinity`, and `NaN` parse to
//! [`crate::types::JsonValue::Null`] — serde_json numbers cannot carry non-finite values,
//! and the JSON wire form of those literals is `null` anyway (what
//! `JSON.stringify` would have emitted). Number parsing goes through
//! `serde_json`, which is JSON-strict exactly like `JSON.parse`.

use serde_json::{Map, Number, Value};

/// A parsed number canonicalized the way a JS number prints: an integral
/// float becomes the integer form, so `1e5` compares equal to `100000`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the i64 bounds as floats reproduce the JS number conversion range this check ports, and the fract check bounds the float to the i64 window"
)]
fn canonicalize_number(number: Number) -> Value {
    if let Some(float) = number.as_f64()
        && float.is_finite()
        && float.fract() == 0.0
        && (i64::MIN as f64..=i64::MAX as f64).contains(&float)
    {
        return Value::from(float as i64);
    }
    Value::Number(number)
}

/// Which types the parser may return in a truncated form, upstream's `Allow`.
///
/// Compose the bits; the default is all of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Allow;

impl Allow {
    /// Allow partial strings like `"hello \u12` to parse as `"hello "`.
    pub const STR: u16 = 0b0_0000_0001;
    /// Allow partial numbers like `123.` to parse as `123`.
    pub const NUM: u16 = 0b0_0000_0010;
    /// Allow partial arrays like `[1, 2,` to parse as `[1, 2]`.
    pub const ARR: u16 = 0b0_0000_0100;
    /// Allow partial objects like `{"a": 1, "b":` to parse as `{"a": 1}`.
    pub const OBJ: u16 = 0b0_0000_1000;
    /// Allow `nu` to parse as `null`.
    pub const NULL: u16 = 0b0_0001_0000;
    /// Allow `tr` to parse as `true` and `fa` to parse as `false`.
    pub const BOOL: u16 = 0b0_0010_0000;
    /// Allow `Na` to parse as `NaN`.
    pub const NAN: u16 = 0b0_0100_0000;
    /// Allow `Inf` to parse as `Infinity`.
    pub const INFINITY: u16 = 0b0_1000_0000;
    /// Allow `-Inf` to parse as `-Infinity`.
    pub const NEG_INFINITY: u16 = 0b1_0000_0000;
    /// Both infinity directions.
    pub const INF: u16 = Self::INFINITY | Self::NEG_INFINITY;
    /// The special literal types: null, booleans, infinities, NaN.
    pub const SPECIAL: u16 = Self::NULL | Self::BOOL | Self::INF | Self::NAN;
    /// The atom types: strings, numbers, and the special literals.
    pub const ATOM: u16 = Self::STR | Self::NUM | Self::SPECIAL;
    /// The collection types: arrays and objects.
    pub const COLLECTION: u16 = Self::ARR | Self::OBJ;
    /// Every type may be partially parsed.
    pub const ALL: u16 = Self::ATOM | Self::COLLECTION;
}

/// A malformed or truncated input the parser refuses to salvage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialJsonError(pub String);

impl std::fmt::Display for PartialJsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PartialJsonError {}

/// Parse possibly-truncated JSON, returning the completed prefix of whatever
/// the `allow` mask permits to be partial. Whitespace around the input is
/// trimmed; an all-whitespace input is an error.
///
/// # Errors
/// Returns [`PartialJsonError`] when the input is malformed beyond salvage or
/// truncated where `allow` forbids a partial form.
pub fn parse_partial_json(json: &str, allow: u16) -> Result<Value, PartialJsonError> {
    if json.trim().is_empty() {
        return Err(PartialJsonError(format!("{json} is empty")));
    }
    let trimmed = json.trim();
    Parser {
        json: trimmed.as_bytes(),
        source: trimmed,
        index: 0,
        allow,
    }
    .parse_any()
}

struct Parser<'a> {
    json: &'a [u8],
    source: &'a str,
    index: usize,
    allow: u16,
}

impl Parser<'_> {
    const fn at_end(&self) -> bool {
        self.index >= self.json.len()
    }

    fn byte(&self) -> Option<u8> {
        self.json.get(self.index).copied()
    }

    fn skip_blank(&mut self) {
        while let Some(b' ' | b'\n' | b'\r' | b'\t') = self.byte() {
            self.index += 1;
        }
    }

    fn partial_error(&self, msg: &str) -> PartialJsonError {
        PartialJsonError(format!("{msg} at position {}", self.index))
    }

    fn malformed_error(&self, msg: &str) -> PartialJsonError {
        PartialJsonError(format!("{msg} at position {}", self.index))
    }

    /// The exact-match arm upstream checks first, plus the truncated arm
    /// gated on the matching `Allow` bit: the remaining input is a strict
    /// prefix of `keyword`.
    fn literal(&self, keyword: &str, allow_bit: u16) -> bool {
        let remaining = self.json.len() - self.index;
        self.source[self.index..].starts_with(keyword)
            || (self.allow & allow_bit != 0
                && remaining < keyword.len()
                && keyword.starts_with(&self.source[self.index..]))
    }

    fn parse_any(&mut self) -> Result<Value, PartialJsonError> {
        self.skip_blank();
        if self.at_end() {
            return Err(self.partial_error("Unexpected end of input"));
        }
        match self.byte() {
            Some(b'"') => self.parse_str(),
            Some(b'{') => self.parse_obj(),
            Some(b'[') => self.parse_arr(),
            _ => self.parse_literal_or_number(),
        }
    }

    fn parse_literal_or_number(&mut self) -> Result<Value, PartialJsonError> {
        if self.literal("null", Allow::NULL) {
            self.index += 4;
            return Ok(Value::Null);
        }
        if self.literal("true", Allow::BOOL) {
            self.index += 4;
            return Ok(Value::Bool(true));
        }
        if self.literal("false", Allow::BOOL) {
            self.index += 5;
            return Ok(Value::Bool(false));
        }
        if self.literal("Infinity", Allow::INFINITY) {
            self.index += 8;
            return Ok(Value::Null);
        }
        let remaining = self.json.len() - self.index;
        if self.source[self.index..].starts_with("-Infinity")
            || (self.allow & Allow::NEG_INFINITY != 0
                && 1 < remaining
                && remaining < 9
                && "-Infinity".starts_with(&self.source[self.index..]))
        {
            self.index += 9;
            return Ok(Value::Null);
        }
        if self.literal("NaN", Allow::NAN) {
            self.index += 3;
            return Ok(Value::Null);
        }
        self.parse_num()
    }

    fn parse_str(&mut self) -> Result<Value, PartialJsonError> {
        let start = self.index;
        let mut escape = false;
        self.index += 1;
        while self.index < self.json.len()
            && (self.json[self.index] != b'"' || (escape && self.json[self.index - 1] == b'\\'))
        {
            escape = if self.json[self.index] == b'\\' {
                !escape
            } else {
                false
            };
            self.index += 1;
        }
        if self.byte() == Some(b'"') {
            self.index += 1;
            let end = self.index - usize::from(escape);
            let slice = &self.source[start..end];
            return serde_json::from_str(slice).map_err(|e| self.malformed_error(&e.to_string()));
        }
        if self.allow & Allow::STR != 0 {
            let end = self.index - usize::from(escape);
            let head = &self.source[start..end];
            if let Ok(value) = serde_json::from_str(&format!("{head}\"")) {
                return Ok(value);
            }
            // The upstream fallback trims at the last backslash of the whole
            // input; JS `substring` semantics (argument swap when the second
            // position precedes the first) reproduce the degenerate cases.
            let candidate = match self.source.rfind('\\') {
                Some(pos) if pos >= start => format!("{}\"", &self.source[start..pos]),
                Some(pos) => format!("{}\"", &self.source[pos..start]),
                None => format!("{}\"", &self.source[..start]),
            };
            if let Ok(value) = serde_json::from_str(&candidate) {
                return Ok(value);
            }
            return Err(self.malformed_error("Invalid escape sequence"));
        }
        Err(self.partial_error("Unterminated string literal"))
    }

    fn parse_obj(&mut self) -> Result<Value, PartialJsonError> {
        self.index += 1;
        self.skip_blank();
        let mut object = Map::new();
        let mut propagated: Option<PartialJsonError> = None;
        while self.byte() != Some(b'}') {
            self.skip_blank();
            if self.at_end() && self.allow & Allow::OBJ != 0 {
                return Ok(Value::Object(object));
            }
            let key = match self.parse_str() {
                Ok(Value::String(key)) => key,
                Err(error) => {
                    propagated = Some(error);
                    break;
                }
                // Upstream's key reader always yields a string; any other
                // shape cannot surface from a quoted literal, so the object
                // truncates the same way a malformed key does.
                Ok(_) => {
                    propagated = Some(self.malformed_error("object key is not a string"));
                    break;
                }
            };
            self.skip_blank();
            self.index += 1;
            match self.parse_any() {
                Ok(value) => {
                    object.insert(key, value);
                }
                Err(error) => {
                    propagated = Some(error);
                    break;
                }
            }
            self.skip_blank();
            if self.byte() == Some(b',') {
                self.index += 1;
            }
        }
        if propagated.is_some() {
            if self.allow & Allow::OBJ != 0 {
                return Ok(Value::Object(object));
            }
            return Err(self.partial_error("Expected '}' at end of object"));
        }
        self.index += 1;
        Ok(Value::Object(object))
    }

    fn parse_arr(&mut self) -> Result<Value, PartialJsonError> {
        self.index += 1;
        let mut array = Vec::new();
        while self.byte() != Some(b']') {
            match self.parse_any() {
                Ok(value) => array.push(value),
                Err(_) => {
                    // The array path reports its own truncation message
                    // instead of the inner error, unlike the object path.
                    return if self.allow & Allow::ARR != 0 {
                        Ok(Value::Array(array))
                    } else {
                        Err(self.partial_error("Expected ']' at end of array"))
                    };
                }
            }
            self.skip_blank();
            if self.byte() == Some(b',') {
                self.index += 1;
            }
        }
        self.index += 1;
        Ok(Value::Array(array))
    }

    fn parse_num(&mut self) -> Result<Value, PartialJsonError> {
        if self.index == 0 {
            if self.source == "-" {
                return Err(self.malformed_error("Not sure what '-' is"));
            }
            return match serde_json::from_str::<Number>(self.source) {
                Ok(value) => Ok(canonicalize_number(value)),
                Err(error) => {
                    if self.allow & Allow::NUM != 0
                        && let Some(position) = self.source.rfind('e')
                        && let Ok(value) = serde_json::from_str::<Number>(&self.source[..position])
                    {
                        return Ok(Value::Number(value));
                    }
                    Err(self.malformed_error(&error.to_string()))
                }
            };
        }
        let start = self.index;
        if self.byte() == Some(b'-') {
            self.index += 1;
        }
        while let Some(byte) = self.byte() {
            if byte == b',' || byte == b']' || byte == b'}' {
                break;
            }
            self.index += 1;
        }
        if self.at_end() && self.allow & Allow::NUM == 0 {
            return Err(self.partial_error("Unterminated number literal"));
        }
        let slice = &self.source[start..self.index];
        if let Ok(value) = serde_json::from_str::<Number>(slice) {
            return Ok(canonicalize_number(value));
        }
        if slice == "-" {
            return Err(self.partial_error("Not sure what '-' is"));
        }
        // The upstream retry trims at the last exponent marker of the whole
        // input; positions before the literal degenerate to an empty slice.
        let position = self.source.rfind('e').map_or(start, |pos| pos.max(start));
        if let Ok(value) = serde_json::from_str::<Number>(&self.source[start..position]) {
            return Ok(canonicalize_number(value));
        }
        Err(self.malformed_error("Invalid number"))
    }
}

/// JSON escape characters a string literal may carry after a backslash.
const VALID_JSON_ESCAPES: [char; 9] = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'];

fn escape_control_character(ch: char) -> String {
    match ch {
        '\u{0008}' => "\\b".to_owned(),
        '\u{000C}' => "\\f".to_owned(),
        '\n' => "\\n".to_owned(),
        '\r' => "\\r".to_owned(),
        '\t' => "\\t".to_owned(),
        other => format!("\\u{:04x}", other as u32),
    }
}

/// Repairs malformed JSON string literals by escaping raw control characters
/// inside strings and doubling backslashes before invalid escape characters.
///
/// # Examples
///
/// ```
/// use pi_ai::utils::json_parse::repair_json;
///
/// assert_eq!(repair_json("{\"a\": \"b\u{1}c\"}"), "{\"a\": \"b\\u0001c\"}");
/// ```
#[must_use]
pub fn repair_json(json: &str) -> String {
    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let mut chars = json.char_indices().peekable();

    while let Some((_, ch)) = chars.next() {
        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            continue;
        }

        if ch == '"' {
            repaired.push('"');
            in_string = false;
            continue;
        }

        if ch == '\\' {
            match chars.peek().map(|(_, next)| *next) {
                Some('u') => {
                    let digits: String = chars.clone().skip(1).take(4).map(|(_, c)| c).collect();
                    repaired.push_str("\\u");
                    if digits.chars().count() == 4 && digits.chars().all(|c| c.is_ascii_hexdigit())
                    {
                        repaired.push_str(&digits);
                        for _ in 0..5 {
                            chars.next();
                        }
                    } else {
                        // A `u` escape with non-hex digits keeps the escape
                        // and lets the digits pass through as ordinary text.
                        chars.next();
                    }
                }
                Some(next) if VALID_JSON_ESCAPES.contains(&next) => {
                    repaired.push('\\');
                    repaired.push(next);
                    chars.next();
                }
                None | Some(_) => {
                    repaired.push_str("\\\\");
                }
            }
            continue;
        }

        // The JSON grammar's control-character range only: C0 controls up to
        // 0x1F, not every Unicode control character.
        if (ch as u32) <= 0x1F {
            repaired.push_str(&escape_control_character(ch));
        } else {
            repaired.push(ch);
        }
    }

    repaired
}

/// Parse JSON, falling back to [`repair_json`] when the input is malformed
/// and the repair changed it.
///
/// # Errors
/// Returns the original parse error when the input is malformed and repair
/// changed nothing, or when the repaired form still fails to parse.
pub fn parse_json_with_repair(json: &str) -> Result<Value, serde_json::Error> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(error) => {
            let repaired = repair_json(json);
            if repaired == json {
                Err(error)
            } else {
                serde_json::from_str(&repaired)
            }
        }
    }
}

/// Attempts to parse potentially incomplete JSON during streaming. Always
/// returns a valid value: the empty object when the input is empty or every
/// salvage attempt fails.
#[must_use]
pub fn parse_streaming_json(partial: Option<&str>) -> Value {
    let Some(partial) = partial else {
        return Value::Object(Map::new());
    };
    if partial.trim().is_empty() {
        return Value::Object(Map::new());
    }

    if let Ok(value) = parse_json_with_repair(partial) {
        return value;
    }
    if let Ok(value) = parse_partial_json(partial, Allow::ALL) {
        return nullish_to_empty_object(value);
    }
    if let Ok(value) = parse_partial_json(&repair_json(partial), Allow::ALL) {
        return nullish_to_empty_object(value);
    }
    Value::Object(Map::new())
}

/// Upstream feeds the partial result through `result ?? {}`, which folds a
/// parsed `null` into the empty object.
fn nullish_to_empty_object(value: Value) -> Value {
    if value.is_null() {
        Value::Object(Map::new())
    } else {
        value
    }
}
