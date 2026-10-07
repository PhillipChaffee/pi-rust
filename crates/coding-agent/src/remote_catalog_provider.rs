//! The persisted pi.dev catalog overlay over a static built-in provider,
//! upstream's `src/core/remote-catalog-provider.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the wrapper is a delegating [`Provider`] whose
//! dynamic overlay lives behind a mutex, refreshed through the
//! [`RefreshModelsContext`] publications; the wire models that do not
//! deserialize into a full [`Model`] are dropped rather than carried as
//! broken shapes (upstream's spread kept whatever the JSON carried); the
//! `Last-Modified` date parser handles the RFC 1123 format HTTP sends, and
//! an unparsable date stores `0` like upstream's `Number.isNaN` branch.

use std::sync::{Arc, Mutex};

use pi_ai::http::{HttpClient, default_http_client};
use pi_ai::models::{
    CatalogPersist, ModelsPublication, Provider, ProviderError, ProviderModelError,
    RefreshModelsContext,
};
use pi_ai::models_store::ModelsStoreEntry;
use pi_ai::types::{BoxedFuture, Model, ProviderHeaders};

use crate::config::VERSION;
use crate::utils::management_http::{FetchRetryOptions, fetch_with_retry};
use crate::utils::pi_user_agent::get_pi_user_agent;

const DEFAULT_CATALOG_BASE_URL: &str = "https://pi.dev";
const REMOTE_CATALOG_ATTEMPT_TIMEOUT_MS: u64 = 4_000;
/// The freshness window a stored catalog's check satisfies before another
/// network fetch, upstream's `REMOTE_CATALOG_REFRESH_INTERVAL_MS` (four
/// hours).
pub const REMOTE_CATALOG_REFRESH_INTERVAL_MS: i64 = 4 * 60 * 60 * 1000;

/// Merge a dynamic overlay into a baseline list by model id, upstream's
/// `mergeModels`.
fn merge_models(baseline: Vec<Model>, dynamic: &[Model]) -> Vec<Model> {
    let mut merged = baseline;
    for model in dynamic {
        match merged.iter_mut().find(|entry| entry.id == model.id) {
            Some(slot) => *slot = model.clone(),
            None => merged.push(model.clone()),
        }
    }
    merged
}

/// Parse a fetched catalog body, upstream's `parseCatalog`: an array, an
/// object with a `models` array, or an object map of models. Entries that
/// do not deserialize into a full model are dropped.
fn parse_catalog(provider_id: &str, value: serde_json::Value) -> Result<Vec<Model>, ProviderError> {
    let entries: Vec<serde_json::Value> = match value {
        serde_json::Value::Array(entries) => entries,
        serde_json::Value::Object(map) => {
            if let Some(models) = map.get("models") {
                models
                    .as_array()
                    .cloned()
                    .ok_or_else(|| invalid_catalog(provider_id))?
            } else {
                map.into_values().collect()
            }
        }
        _ => return Err(invalid_catalog(provider_id)),
    };
    Ok(entries
        .into_iter()
        .filter(|entry| entry.get("id").is_some_and(serde_json::Value::is_string))
        .filter_map(|entry| match serde_json::from_value::<Model>(entry) {
            Ok(mut model) => {
                model.provider = pi_ai::types::ProviderId(provider_id.to_owned());
                Some(model)
            }
            Err(_) => None,
        })
        .collect())
}

fn invalid_catalog(provider_id: &str) -> ProviderError {
    Box::new(std::io::Error::other(format!(
        "Invalid model catalog for provider \"{provider_id}\""
    )))
}

/// The overlay a stored entry contributes, upstream's `remoteModels`: stale
/// or absent entries contribute nothing so the generated catalog is not
/// emptied by an older cached overlay.
fn remote_models(entry: Option<&ModelsStoreEntry>, local_generated_at: Option<i64>) -> Vec<Model> {
    let Some(entry) = entry else {
        return Vec::new();
    };
    if local_generated_at.is_some_and(|local_generated_at| {
        entry
            .last_modified
            .is_none_or(|last_modified| last_modified <= local_generated_at)
    }) {
        return Vec::new();
    }
    entry.models.clone()
}

/// Parse an RFC 1123 HTTP date (`Wed, 21 Oct 2015 07:28:00 GMT`) into Unix
/// milliseconds, `0` when the header is absent or unparsable, the port of
/// upstream's `Date.parse(...) ?? NaN → 0`.
fn parse_http_date_ms(header: Option<&str>) -> i64 {
    let Some(text) = header else {
        return 0;
    };
    let Some((_, rest)) = text.split_once(", ") else {
        return 0;
    };
    let mut parts = rest.split_whitespace();
    let Some(day) = parts
        .next()
        .and_then(|day| day.parse::<u32>().ok())
        .map(i64::from)
    else {
        return 0;
    };
    let Some(month) = parts.next().and_then(|month| match month {
        "Jan" => Some(1),
        "Feb" => Some(2),
        "Mar" => Some(3),
        "Apr" => Some(4),
        "May" => Some(5),
        "Jun" => Some(6),
        "Jul" => Some(7),
        "Aug" => Some(8),
        "Sep" => Some(9),
        "Oct" => Some(10),
        "Nov" => Some(11),
        "Dec" => Some(12),
        _ => None,
    }) else {
        return 0;
    };
    let Some(year) = parts.next().and_then(|year| year.parse::<i64>().ok()) else {
        return 0;
    };
    let Some(time) = parts.next() else {
        return 0;
    };
    let mut clock = time.split(':');
    let (Some(hours), Some(minutes), Some(seconds)) = (clock.next(), clock.next(), clock.next())
    else {
        return 0;
    };
    let Ok(hours) = hours.parse::<i64>() else {
        return 0;
    };
    let Ok(minutes) = minutes.parse::<i64>() else {
        return 0;
    };
    let Ok(seconds) = seconds.split('.').next().unwrap_or(seconds).parse::<i64>() else {
        return 0;
    };
    // Days-from-civil over the Gregorian calendar (Howard Hinnant's
    // algorithm), the date math `Date.parse` performs.
    let adjusted_year = i64::from(month <= 2) + year;
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year.rem_euclid(400);
    let day_of_era = (365 * year_of_era + year_of_era.div_euclid(4) - year_of_era.div_euclid(100))
        + ((153 * (i64::from(month) + if month > 2 { -3 } else { 9 }) + 2) / 5)
        + day
        - 1;
    let days = era * 146_097 + day_of_era - 719_468;
    days * 86_400_000 + (hours * 60 + minutes) * 60_000 + seconds * 1000
}

fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// The state one wrapped provider keeps across refresh phases, upstream's
/// closure-captured `dynamicModels`.
struct RemoteCatalogState {
    dynamic_models: Mutex<Vec<Model>>,
}

/// Add a persisted pi.dev catalog overlay to a static built-in provider,
/// upstream's `withRemoteCatalog`.
#[must_use]
pub fn with_remote_catalog(
    provider: Arc<dyn Provider>,
    catalog_base_url: Option<String>,
    local_generated_at: Option<i64>,
) -> Arc<dyn Provider> {
    with_remote_catalog_client(
        provider,
        catalog_base_url,
        local_generated_at,
        default_http_client(),
    )
}

/// [`with_remote_catalog`] over an explicit client, the seam the boundary
/// tests drive instead of upstream's stubbed `globalThis.fetch`.
#[must_use]
pub fn with_remote_catalog_client(
    provider: Arc<dyn Provider>,
    catalog_base_url: Option<String>,
    local_generated_at: Option<i64>,
    client: Arc<dyn HttpClient>,
) -> Arc<dyn Provider> {
    Arc::new(RemoteCatalogProvider {
        inner: provider,
        catalog_base_url: catalog_base_url.unwrap_or_else(|| DEFAULT_CATALOG_BASE_URL.to_owned()),
        local_generated_at,
        client,
        state: Arc::new(RemoteCatalogState {
            dynamic_models: Mutex::new(Vec::new()),
        }),
    })
}

/// The wrapped provider, upstream's spread object returned from
/// `withRemoteCatalog`.
struct RemoteCatalogProvider {
    inner: Arc<dyn Provider>,
    catalog_base_url: String,
    local_generated_at: Option<i64>,
    client: Arc<dyn HttpClient>,
    state: Arc<RemoteCatalogState>,
}

impl std::fmt::Debug for RemoteCatalogProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteCatalogProvider")
            .field("id", &self.inner.id())
            .finish_non_exhaustive()
    }
}

impl Provider for RemoteCatalogProvider {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn base_url(&self) -> Option<&str> {
        self.inner.base_url()
    }

    fn headers(&self) -> Option<&ProviderHeaders> {
        self.inner.headers()
    }

    fn auth(&self) -> &pi_ai::auth::types::ProviderAuth {
        self.inner.auth()
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        let dynamic = self
            .state
            .dynamic_models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        Ok(merge_models(self.inner.get_models()?, &dynamic))
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxedFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { self.refresh(context).await })
    }

    fn supports_refresh_models(&self) -> bool {
        true
    }

    fn filter_models(
        &self,
        models: Vec<Model>,
        credential: Option<&pi_ai::auth::types::Credential>,
    ) -> Vec<Model> {
        self.inner.filter_models(models, credential)
    }

    fn stream(
        &self,
        model: &Model,
        context: &pi_ai::types::Context,
        options: Option<&pi_ai::types::StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &pi_ai::types::Context,
        options: Option<&pi_ai::types::SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        self.inner.stream_simple(model, context, options)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &pi_ai::types::DeferredHandle,
        options: Option<&pi_ai::types::DeferredFetchOptions>,
    ) -> Option<pi_ai::utils::event_stream::AssistantMessageEventStream> {
        self.inner.fetch_deferred(model, handle, options)
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a pi_ai::types::DeferredHandle,
        options: Option<&'a pi_ai::types::DeferredCancelOptions>,
    ) -> Option<BoxedFuture<'a, Result<(), pi_ai::utils::provider_retry::ProviderRequestError>>>
    {
        self.inner.cancel_deferred(model, handle, options)
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.inner.supports_fetch_deferred()
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.inner.supports_cancel_deferred()
    }
}

impl RemoteCatalogProvider {
    /// The refresh flow, upstream's `refreshModels` body: restore the stale
    /// overlay, respect the freshness window, then revalidate or fetch.
    #[expect(
        clippy::too_many_lines,
        reason = "the 1:1 port of upstream's status-laddered refresh body reads longer than the lint's slice"
    )]
    async fn refresh(&self, context: RefreshModelsContext) -> Result<(), ProviderError> {
        let stored = context.stored.clone();
        let restored = remote_models(stored.as_ref(), self.local_generated_at)
            .into_iter()
            .filter(|model| model.provider.0 == self.inner.id())
            .collect::<Vec<_>>();
        let published = (context.publish)(ModelsPublication {
            persist: CatalogPersist::Omit,
            update: Some(Box::new({
                let state = Arc::clone(&self.state);
                move || {
                    state
                        .dynamic_models
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone_from(&restored);
                }
            })),
        })
        .await?;
        if !published {
            return Ok(());
        }
        if !context.allow_network || context.signal.is_cancelled() {
            return Ok(());
        }
        let now = pi_ai::auth::resolve::now_ms();
        if context.force != Some(true)
            && stored.as_ref().is_some_and(|stored| {
                stored.checked_at.is_some()
                    && stored.last_modified.is_some()
                    && now - stored.checked_at.unwrap_or_default()
                        < REMOTE_CATALOG_REFRESH_INTERVAL_MS
            })
        {
            return Ok(());
        }

        // Only revalidate when a cached body backs the validator, so a 304
        // can never leave the overlay empty.
        let validator = stored
            .as_ref()
            .filter(|stored| !stored.models.is_empty())
            .and_then(|stored| stored.etag.clone());
        let url = build_catalog_url(&self.catalog_base_url, self.inner.id())?;
        let mut headers: Vec<(String, String)> = vec![
            ("accept".to_owned(), "application/json".to_owned()),
            ("User-Agent".to_owned(), get_pi_user_agent(VERSION)),
        ];
        if let Some(validator) = validator {
            headers.push(("if-none-match".to_owned(), validator));
        }
        let response = fetch_with_retry(
            &self.client,
            &url,
            headers,
            context.signal.clone(),
            FetchRetryOptions {
                attempt_timeout_ms: Some(REMOTE_CATALOG_ATTEMPT_TIMEOUT_MS),
                ..FetchRetryOptions::default()
            },
        )
        .await?;
        if context.signal.is_cancelled() {
            return Ok(());
        }
        let checked_at = pi_ai::auth::resolve::now_ms();
        // Unchanged: dynamicModels already holds the stored overlay, so only
        // the freshness window moves.
        if response.status == 304 && stored.is_some() {
            (context.publish)(ModelsPublication {
                persist: CatalogPersist::Write(ModelsStoreEntry {
                    checked_at: Some(checked_at),
                    ..stored.unwrap_or_default()
                }),
                update: None,
            })
            .await?;
            return Ok(());
        }
        if response.status == 404 || response.status == 501 {
            (context.publish)(ModelsPublication {
                persist: CatalogPersist::Write(ModelsStoreEntry {
                    models: stored.clone().unwrap_or_default().models,
                    checked_at: Some(checked_at),
                    last_modified: Some(0),
                    etag: None,
                }),
                update: None,
            })
            .await?;
            return Ok(());
        }
        if !(200..300).contains(&response.status) {
            // Transient failure: the cached body and its validator stay
            // valid, so keep the etag and let the next refresh revalidate
            // instead of downloading the catalog.
            (context.publish)(ModelsPublication {
                persist: CatalogPersist::Write(ModelsStoreEntry {
                    models: stored.clone().unwrap_or_default().models,
                    checked_at: Some(checked_at),
                    ..stored.unwrap_or_default()
                }),
                update: None,
            })
            .await?;
            return Err(Box::new(std::io::Error::other(format!(
                "Model catalog request failed for {}: {}",
                self.inner.id(),
                response.status
            ))));
        }
        let body = pi_ai::http::client::read_body_text(response.body).await?;
        let refreshed = parse_catalog(
            self.inner.id(),
            serde_json::from_str(&body).map_err(|error| -> ProviderError { Box::new(error) })?,
        )?;
        let last_modified = parse_http_date_ms(header_value(&response.headers, "last-modified"));
        if context.signal.is_cancelled() {
            return Ok(());
        }
        let entry = ModelsStoreEntry {
            models: refreshed.clone(),
            checked_at: Some(checked_at),
            last_modified: Some(last_modified),
            etag: header_value(&response.headers, "etag").map(str::to_owned),
        };
        let published_models = remote_models(Some(&entry), self.local_generated_at);
        (context.publish)(ModelsPublication {
            persist: CatalogPersist::Write(entry),
            update: Some(Box::new({
                let state = Arc::clone(&self.state);
                move || {
                    state
                        .dynamic_models
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone_from(&published_models);
                }
            })),
        })
        .await?;
        Ok(())
    }
}

/// The catalog URL for a provider, upstream's `new URL(path, base)` join: a
/// path-absolute path replaces the base's path.
fn build_catalog_url(base: &str, provider_id: &str) -> Result<String, ProviderError> {
    let encoded = urlencoding_encode_component(provider_id);
    let joined = url::Url::parse(base)
        .and_then(|base| base.join(&format!("/api/models/providers/{encoded}")))
        .map_err(|error| -> ProviderError { Box::new(error) })?;
    Ok(joined.to_string())
}

/// The hex digits the percent-encoding emits.
const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// `encodeURIComponent`, upstream's provider-id encoding in the catalog
/// path.
fn urlencoding_encode_component(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        let unreserved =
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'*');
        if unreserved {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0F)]));
        }
    }
    encoded
}
