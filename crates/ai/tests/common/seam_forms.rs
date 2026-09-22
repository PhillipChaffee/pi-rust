//! Form-body readers for the seam mock's recorded requests, the port of
//! upstream's `new URLSearchParams(String(init.body))` fetch-stub asserts in
//! `test/xai-oauth.test.ts` and `test/radius-oauth.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::BTreeMap;

use pi_ai::http::mock::RecordedRequest;

/// The request's form body as a field map, upstream's `URLSearchParams`.
#[must_use]
pub fn form_fields(request: &RecordedRequest) -> BTreeMap<String, String> {
    url::form_urlencoded::parse(request.body.as_deref().unwrap_or_default())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect()
}

/// One form field, empty when the request does not carry it.
#[must_use]
pub fn form_field<'a>(fields: &'a BTreeMap<String, String>, name: &str) -> &'a str {
    fields.get(name).map(String::as_str).unwrap_or_default()
}
