//! HTML entity decoding for highlighted-output scanners, upstream's
//! `src/utils/html.ts`.

/// A decoded entity and the source length it consumed.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct DecodedHtmlEntity {
    /// The decoded text.
    pub text: String,
    /// The number of source characters the entity spanned, `&` through `;`.
    pub length: usize,
}

fn decode_code_point(code_point: u32) -> Option<String> {
    char::from_u32(code_point).map(String::from)
}

/// Decode the named and numeric HTML entities the highlighters emit.
///
/// Upstream's `Number.isInteger`/`> 0x10ffff` guards restate as
/// [`char::from_u32`]'s own range check, which also rejects the surrogate
/// code points `String.fromCodePoint` alone permits.
#[must_use]
pub fn decode_html_entity(entity: &str) -> Option<String> {
    match entity {
        "amp" => return Some("&".to_string()),
        "lt" => return Some("<".to_string()),
        "gt" => return Some(">".to_string()),
        "quot" => return Some("\"".to_string()),
        "apos" => return Some("'".to_string()),
        _ => {}
    }

    if let Some(hex) = entity
        .strip_prefix("#x")
        .or_else(|| entity.strip_prefix("#X"))
    {
        return u32::from_str_radix(hex, 16)
            .ok()
            .and_then(decode_code_point);
    }

    if let Some(decimal) = entity.strip_prefix('#') {
        return decimal.parse::<u32>().ok().and_then(decode_code_point);
    }

    None
}

/// Decode the entity starting at `index` (an `&`), upstream's
/// `decodeHtmlEntityAt`. The scan gives up past a 16-character body or a
/// missing semicolon, so a bare `&` in text costs one lookahead at most.
#[must_use]
pub fn decode_html_entity_at(html: &str, index: usize) -> Option<DecodedHtmlEntity> {
    let body = html.get(index + 1..)?;
    let semicolon_index = body.find(';')? + index + 1;
    if semicolon_index - index > 16 {
        return None;
    }

    let entity = &html[index + 1..semicolon_index];
    let decoded = decode_html_entity(entity)?;
    Some(DecodedHtmlEntity {
        text: decoded,
        length: semicolon_index - index + 1,
    })
}
