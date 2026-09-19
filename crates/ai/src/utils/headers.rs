//! Header conversions, ported from
//! `packages/ai/src/utils/headers.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::BTreeMap;

use crate::types::ProviderHeaders;

/// Collect header pairs into a record. HTTP header maps that carry case-insensitive
/// names keep the case of the first occurrence, like `Headers.entries()`.
#[must_use]
pub fn headers_to_record<'a>(
    headers: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> BTreeMap<String, String> {
    headers
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect()
}

/// Drop the suppressed (`null`) entries of [`ProviderHeaders`] and return the
/// remaining concrete headers; an empty result is no headers at all.
#[must_use]
pub fn provider_headers_to_record(
    headers: Option<&ProviderHeaders>,
) -> Option<BTreeMap<String, String>> {
    let headers = headers?;
    let result: BTreeMap<String, String> = headers
        .iter()
        .filter_map(|(key, value)| value.as_ref().map(|value| (key.clone(), value.clone())))
        .collect();
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}
