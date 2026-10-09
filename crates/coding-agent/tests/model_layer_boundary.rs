//! Boundary tests binding the model-layer branches the 1:1 suites leave
//! untested (#121), at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759: the
//! remote-catalog ladder (upstream
//! `packages/coding-agent/test/remote-catalog-provider.test.ts`), the
//! management-HTTP retry ladder (upstream
//! `packages/coding-agent/test/management-http.test.ts`), minimatch compile
//! vectors, and the models.json schema ladder.
//!
//! Porting restatements this suite records:
//!
//! - Upstream stubs `globalThis.fetch`; the port drives the
//!   `pi_ai::http::MockHttpClient` seam through `with_remote_catalog_client`
//!   and `fetch_with_retry`, which take the client explicitly.
//! - The shared-timeout identity upstream pins with `signals[0] ===
//!   signals[1]` has no recorded-signal surface in Rust: the budget rides one
//!   parent cancellation token cloned into every attempt's request, and a
//!   recorded request carries no token — the port pins the attempt count
//!   only.
//! - The bypass case upstream runs through two concurrent `models.refresh`
//!   calls; the port keeps that shape. Upstream's supersede also cancels the
//!   older request through its signal, so in both runtimes the gated older
//!   body lands on an already-aborted fetch and resolves nothing — the same
//!   no-stale-publication outcome.
//! - The minimatch vectors pin the port's compiled semantics, checked against
//!   the released npm minimatch 10.2.6 the module header cites; where the
//!   port and npm diverge the divergence is pinned and noted in the case
//!   (globstar separators, the trailing globstar's bare prefix, and extglob
//!   literals).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "a wrong error variant is reported by panicking"
)]

mod common;

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::model_layer::model;
use pi_ai::auth::types::{ApiKeyAuth, ApiKeyCredential, AuthResult, Credential, ProviderAuth};
use pi_ai::http::mock::{MockRouteBuilder, RecordedRequest};
use pi_ai::http::{
    HttpClient, HttpError, HttpResponse, MockHttpClient, MockResponse, json_response,
};
use pi_ai::models::{
    CatalogPersist, CreateModelsOptions, CreateProviderOptions, ModelsRefreshOptions, Provider,
    ProviderApi, ProviderError, PublishFn, RefreshModelsContext, create_models, create_provider,
};
use pi_ai::models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry};
use pi_ai::types::{Context, Model, ProviderStreams, SimpleStreamOptions, StreamOptions};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::config::VERSION;
use pi_coding_agent::model_config::ModelConfig;
use pi_coding_agent::remote_catalog_provider::with_remote_catalog_client;
use pi_coding_agent::utils::management_http::{FetchRetryOptions, fetch_with_retry};
use pi_coding_agent::utils::minimatch::matches;
use tokio_util::sync::CancellationToken;

/// Area 1: the remote-catalog ladder, upstream's seven cases plus the 404
/// rung, driven over the injected client.
mod remote_catalog {
    use super::*;

    /// The API double upstream's `api: { stream: () => { throw } }`: streams
    /// are never driven by these cases, and a call is a test failure.
    struct PanickingStreams;

    impl ProviderStreams for PanickingStreams {
        fn stream(
            &self,
            _model: &Model,
            _context: &Context,
            _options: Option<&StreamOptions>,
        ) -> AssistantMessageEventStream {
            panic!("not used")
        }

        fn stream_simple(
            &self,
            _model: &Model,
            _context: &Context,
            _options: Option<&SimpleStreamOptions>,
        ) -> AssistantMessageEventStream {
            panic!("not used")
        }
    }

    /// The static provider upstream's `testProvider` builds — one `static`
    /// model over a resolve-anything api-key auth — wrapped in the catalog
    /// overlay fetched through `client`.
    fn catalog_provider(
        client: Arc<dyn HttpClient>,
        local_generated_at: Option<i64>,
    ) -> Arc<dyn Provider> {
        with_remote_catalog_client(
            Arc::new(create_provider(CreateProviderOptions {
                id: "test-provider".to_owned(),
                name: None,
                base_url: None,
                headers: None,
                auth: ProviderAuth {
                    api_key: Some(ApiKeyAuth {
                        name: "Test".to_owned(),
                        login: None,
                        check: None,
                        resolve: Arc::new(|_input| {
                            Box::pin(async { Ok(Some(AuthResult::default())) })
                        }),
                    }),
                    oauth: None,
                },
                models: vec![model("test-provider", "static")],
                fetch_models: None,
                filter_models: None,
                api: ProviderApi::Single(Arc::new(PanickingStreams)),
            })),
            Some("https://pi.dev".to_owned()),
            local_generated_at,
            client,
        )
    }

    /// The wire entry for one fixture model, the shape `parse_catalog` reads.
    fn catalog_model(id: &str) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "name": id,
            "api": "openai-completions",
            "provider": "test-provider",
            "baseUrl": "https://example.test/v1",
            "reasoning": false,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1000,
            "maxTokens": 100
        })
    }

    /// The keyed catalog body upstream's `{ dynamic: model("dynamic") }` — an
    /// object map of models, one of `parse_catalog`'s three accepted shapes.
    fn catalog_map(id: &str) -> serde_json::Value {
        let mut entries = serde_json::Map::new();
        entries.insert(id.to_owned(), catalog_model(id));
        serde_json::Value::Object(entries)
    }

    /// The catalog route every case drives, upstream's `https://pi.dev` base
    /// joined with the provider id.
    fn catalog_route(mock: &MockHttpClient) -> MockRouteBuilder<'_> {
        mock.on(|request| request.url.contains("/api/models/providers/test-provider"))
    }

    /// The wired catalog rig: provider and fresh store over the mock, the
    /// setup every ladder case shares.
    fn catalog_rig(
        mock: &MockHttpClient,
        local_generated_at: Option<i64>,
    ) -> (Arc<dyn Provider>, Arc<InMemoryModelsStore>) {
        let client: Arc<dyn HttpClient> = Arc::new(mock.clone());
        let provider = catalog_provider(client, local_generated_at);
        let store = Arc::new(InMemoryModelsStore::default());
        (provider, store)
    }

    /// The 200 catalog body stamped with the fixture etag, the first
    /// response the etag cases send.
    fn etag_body(id: &str) -> MockResponse {
        json_response(200, &catalog_map(id)).with_header("etag", "\"catalog-1\"")
    }

    /// The 304 revalidation answer stamped with the fixture etag.
    fn etag_304() -> MockResponse {
        MockResponse::status(304).with_header("etag", "\"catalog-1\"")
    }

    /// The network knobs of one refresh, upstream's `overrides` param: the
    /// defaults run the network and leave `force` unset.
    #[derive(Clone, Copy)]
    struct RefreshOverrides {
        allow_network: bool,
        force: Option<bool>,
    }

    impl Default for RefreshOverrides {
        fn default() -> Self {
            Self {
                allow_network: true,
                force: None,
            }
        }
    }

    /// Upstream's `refreshProvider`: one refresh phase over `store` with the
    /// publication spelled inline — persist writes and deletes land in the
    /// store, `update` runs after the store mutation, and the publication
    /// always resolves `true`.
    ///
    /// # Panics
    /// A store failure, which the in-memory backend cannot produce.
    async fn refresh_provider(
        provider: &Arc<dyn Provider>,
        store: &Arc<InMemoryModelsStore>,
        overrides: RefreshOverrides,
    ) -> Result<(), ProviderError> {
        let provider_id = provider.id().to_owned();
        let publish_store = Arc::clone(store);
        let publish: PublishFn = Arc::new(move |publication| {
            let store = Arc::clone(&publish_store);
            let provider_id = provider_id.clone();
            Box::pin(async move {
                match publication.persist {
                    CatalogPersist::Omit => {}
                    CatalogPersist::Delete => {
                        ModelsStore::delete(&*store, &provider_id, None).await?;
                    }
                    CatalogPersist::Write(entry) => {
                        ModelsStore::write(&*store, &provider_id, entry, None).await?;
                    }
                }
                if let Some(update) = publication.update {
                    update();
                }
                Ok(true)
            })
        });
        let stored = ModelsStore::read(&**store, provider.id(), None)
            .await
            .expect("the in-memory store reads");
        provider
            .refresh_models(RefreshModelsContext {
                credential: Some(Credential::ApiKey(ApiKeyCredential::default())),
                stored,
                publish,
                allow_network: overrides.allow_network,
                force: overrides.force,
                signal: CancellationToken::new(),
            })
            .await
    }

    /// The model ids a provider lists, upstream's `.map((entry) => entry.id)`.
    fn model_ids(provider: &Arc<dyn Provider>) -> Vec<String> {
        provider
            .get_models()
            .expect("the wrapped provider lists models")
            .into_iter()
            .map(|entry| entry.id)
            .collect()
    }

    /// The stored entry a refresh left behind, upstream's
    /// `store.read(provider.id)` unwrapped: every assertion runs after a
    /// publication wrote the entry.
    ///
    /// # Panics
    /// A store failure or a missing entry, which these flows cannot produce.
    async fn stored_entry(store: &Arc<InMemoryModelsStore>) -> ModelsStoreEntry {
        ModelsStore::read(&**store, "test-provider", None)
            .await
            .expect("the in-memory store reads")
            .expect("the refresh wrote the provider entry")
    }

    /// The stored catalog's model ids, upstream's
    /// `(await store.read(provider.id))?.models.map((entry) => entry.id)`.
    async fn stored_model_ids(store: &Arc<InMemoryModelsStore>) -> Vec<String> {
        stored_entry(store)
            .await
            .models
            .into_iter()
            .map(|entry| entry.id)
            .collect()
    }

    /// The header value a recorded request carried, case-insensitive like
    /// the wire.
    fn header<'a>(request: &'a RecordedRequest, name: &str) -> Option<&'a str> {
        request
            .headers
            .iter()
            .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// Upstream's first case: the keyed catalog parses, the version headers
    /// ride every request, the freshness window skips the middle refresh,
    /// and a forced refresh bypasses the window.
    #[tokio::test]
    async fn keyed_catalog_version_headers_refresh_ttl_and_forced_refresh() {
        let mock = MockHttpClient::new();
        catalog_route(&mock).respond(json_response(200, &catalog_map("dynamic")));
        let (provider, store) = catalog_rig(&mock, None);

        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the first refresh lands");
        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the refresh inside the freshness window skips the network");
        refresh_provider(
            &provider,
            &store,
            RefreshOverrides {
                force: Some(true),
                ..RefreshOverrides::default()
            },
        )
        .await
        .expect("the forced refresh lands");

        assert_eq!(model_ids(&provider), ["static", "dynamic"]);
        assert_eq!(stored_model_ids(&store).await, ["dynamic"]);
        assert_eq!(
            mock.request_count(),
            2,
            "the freshness window skipped the middle refresh"
        );
        let recorded = mock.recorded();
        assert!(
            header(&recorded[0], "User-Agent")
                .is_some_and(|agent| agent.contains(&format!("pi/{VERSION}"))),
            "the catalog request carries the pi user agent, headers: {:?}",
            recorded[0].headers
        );
    }

    /// Upstream's second case: a stored overlay older than the generated
    /// catalog contributes nothing, and a remote catalog newer than the
    /// generated one wins by its `Last-Modified` date.
    #[tokio::test]
    async fn prefers_the_newer_of_the_generated_and_remote_catalogs() {
        // Upstream pins the boundary with `Date.parse("2026-07-23T10:00:00Z")`
        // and `new Date(...).toUTCString()` header strings; the port spells
        // both as constants.
        let local_generated_at = 1_784_800_800_000;
        let older = json_response(200, &catalog_map("old"))
            .with_header("last-modified", "Thu, 23 Jul 2026 09:59:00 GMT");
        let newer = json_response(200, &catalog_map("newer"))
            .with_header("last-modified", "Thu, 23 Jul 2026 10:01:00 GMT");
        let mock = MockHttpClient::new();
        catalog_route(&mock).respond_sequence(vec![older, newer]);
        let (provider, store) = catalog_rig(&mock, Some(local_generated_at));

        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the stale overlay refresh lands");
        assert_eq!(
            model_ids(&provider),
            ["static"],
            "the older Last-Modified loses to the generated catalog"
        );

        refresh_provider(
            &provider,
            &store,
            RefreshOverrides {
                force: Some(true),
                ..RefreshOverrides::default()
            },
        )
        .await
        .expect("the newer-catalog refresh lands");
        assert_eq!(model_ids(&provider), ["static", "newer"]);
        assert_eq!(
            stored_entry(&store).await.last_modified,
            Some(1_784_800_860_000),
            "the store keeps the parsed Last-Modified date"
        );
    }

    /// Upstream's third case: the first request has no `if-none-match`, the
    /// revalidation sends the stored etag, and a 304 keeps the overlay and
    /// moves only the freshness window.
    #[tokio::test]
    async fn revalidates_a_stored_catalog_with_its_etag_and_keeps_the_overlay_on_304() {
        let mock = MockHttpClient::new();
        catalog_route(&mock).respond_sequence(vec![etag_body("dynamic"), etag_304()]);
        let (provider, store) = catalog_rig(&mock, None);

        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the first refresh lands");
        let recorded = mock.recorded();
        assert!(
            header(&recorded[0], "if-none-match").is_none(),
            "the first request has nothing to revalidate against"
        );
        assert_eq!(
            stored_entry(&store).await.etag.as_deref(),
            Some("\"catalog-1\"")
        );

        let checked_at = stored_entry(&store).await.checked_at;
        refresh_provider(
            &provider,
            &store,
            RefreshOverrides {
                force: Some(true),
                ..RefreshOverrides::default()
            },
        )
        .await
        .expect("the revalidation lands");

        let recorded = mock.recorded();
        assert_eq!(
            header(&recorded[1], "if-none-match"),
            Some("\"catalog-1\""),
            "the revalidation sends the stored etag"
        );
        assert_eq!(
            model_ids(&provider),
            ["static", "dynamic"],
            "the 304 keeps the overlay"
        );
        let stored = stored_entry(&store).await;
        assert_eq!(stored_model_ids(&store).await, ["dynamic"]);
        assert_eq!(stored.etag.as_deref(), Some("\"catalog-1\""));
        assert!(
            stored
                .checked_at
                .is_some_and(|checked| checked >= checked_at.unwrap_or(0)),
            "the 304 still moves the freshness window"
        );
    }

    /// Upstream's fourth case: a 501 answer drops the stored etag, so the
    /// next refresh fetches instead of revalidating.
    #[tokio::test]
    async fn drops_a_stale_etag_when_the_overlay_becomes_unavailable() {
        let unimplemented = MockResponse::status(501).with_body("not implemented");
        let mock = MockHttpClient::new();
        catalog_route(&mock).respond_sequence(vec![etag_body("dynamic"), unimplemented]);
        let (provider, store) = catalog_rig(&mock, None);

        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the first refresh lands");
        refresh_provider(
            &provider,
            &store,
            RefreshOverrides {
                force: Some(true),
                ..RefreshOverrides::default()
            },
        )
        .await
        .expect("the unavailable refresh resolves");

        assert_eq!(
            stored_entry(&store).await.etag,
            None,
            "the unavailable answer drops the validator"
        );
    }

    /// Upstream's fifth case: the 429 exhausts the retry ladder and fails
    /// the refresh with the status text, but the stored overlay and its
    /// etag survive, and the next forced refresh revalidates to a 304 that
    /// keeps everything.
    #[tokio::test]
    async fn keeps_the_etag_and_overlay_after_a_transient_failure() {
        let rate_limited = MockResponse::status(429).with_body("rate limited");
        let mock = MockHttpClient::new();
        catalog_route(&mock).respond_sequence(vec![
            etag_body("dynamic"),
            rate_limited.clone(),
            rate_limited.clone(),
            rate_limited,
            etag_304(),
        ]);
        let (provider, store) = catalog_rig(&mock, None);

        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the first refresh lands");

        let forced = RefreshOverrides {
            force: Some(true),
            ..RefreshOverrides::default()
        };
        let error = refresh_provider(&provider, &store, forced)
            .await
            .expect_err("the 429 exhausts the retry ladder");
        assert!(
            error.to_string().contains("429"),
            "the status text rides the failure: {error}"
        );
        let stored = stored_entry(&store).await;
        assert_eq!(
            stored.etag.as_deref(),
            Some("\"catalog-1\""),
            "the validator survives the transient failure"
        );
        assert_eq!(stored_model_ids(&store).await, ["dynamic"]);

        refresh_provider(&provider, &store, forced)
            .await
            .expect("the revalidation lands");
        let recorded = mock.recorded();
        assert_eq!(
            header(&recorded[4], "if-none-match"),
            Some("\"catalog-1\""),
            "the revalidation sends the surviving etag"
        );
        assert_eq!(model_ids(&provider), ["static", "dynamic"]);
    }

    /// Upstream's sixth case: a newer refresh supersedes a stalled older
    /// one; finishing the older request with a stale body must not publish.
    #[tokio::test]
    async fn lets_a_newer_request_bypass_a_stalled_older_request_without_stale_publication() {
        // The first request parks on this watch until the test sends the
        // older body; the channel models upstream's pending-Response gate,
        // which the superseded request's cancellation also abandons.
        let (gate_tx, gate_rx) = tokio::sync::watch::channel(None::<MockResponse>);
        let (started_tx, mut started_rx) = tokio::sync::watch::channel(());
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = MockHttpClient::new();
        let handler_calls = Arc::clone(&calls);
        catalog_route(&mock).respond_fn(move |_request| {
            let calls = Arc::clone(&handler_calls);
            let gate_rx = gate_rx.clone();
            let started_tx = started_tx.clone();
            async move {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    let _ = started_tx.send(());
                    let mut gate = gate_rx;
                    while gate.borrow().is_none() {
                        if gate.changed().await.is_err() {
                            return Err(HttpError::Transport(
                                "the gated response was dropped".to_owned(),
                            ));
                        }
                    }
                    return gate.borrow().clone().ok_or_else(|| {
                        HttpError::Transport("the gated response vanished".to_owned())
                    });
                }
                Ok(json_response(200, &catalog_map("newer")))
            }
        });
        let (provider, store) = catalog_rig(&mock, None);
        let store_handle: Arc<dyn ModelsStore> = Arc::<InMemoryModelsStore>::clone(&store);
        let models = create_models(Some(CreateModelsOptions {
            models_store: Some(store_handle),
            ..CreateModelsOptions::default()
        }));
        models.set_provider(Arc::clone(&provider));

        let forced = || ModelsRefreshOptions {
            providers: Some(vec!["test-provider".to_owned()]),
            force: Some(true),
            ..ModelsRefreshOptions::default()
        };
        let first = {
            let models = models.clone();
            let options = forced();
            tokio::spawn(async move { models.refresh(Some(&options)).await })
        };
        started_rx
            .changed()
            .await
            .expect("the first refresh reaches the gate");

        let second = models.refresh(Some(&forced())).await;
        assert!(
            second.errors.is_empty() && !second.aborted,
            "the newer refresh completes: {second:?}"
        );
        assert_eq!(model_ids(&provider), ["static", "newer"]);

        // The superseded refresh resolves however its cancellation races the
        // operation; neither outcome publishes, which the state assertions
        // below pin.
        let _ = first.await.expect("the first refresh task joins");
        let _ = gate_tx.send(Some(json_response(200, &catalog_map("older"))));
        tokio::task::yield_now().await;

        assert_eq!(
            model_ids(&provider),
            ["static", "newer"],
            "the stale body never publishes"
        );
        assert_eq!(stored_model_ids(&store).await, ["newer"]);
    }

    /// Upstream's seventh case: a 501 answer on a fresh provider stores the
    /// unavailable overlay — no models, a checked date, no validator.
    #[tokio::test]
    async fn treats_unimplemented_catalog_routes_as_an_unavailable_overlay() {
        let mock = MockHttpClient::new();
        catalog_route(&mock).respond(MockResponse::status(501).with_body("not implemented"));
        let (provider, store) = catalog_rig(&mock, None);

        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the unavailable overlay refresh resolves");

        assert_eq!(model_ids(&provider), ["static"]);
        let stored = stored_entry(&store).await;
        assert!(stored.models.is_empty(), "the overlay is unavailable");
        assert!(stored.checked_at.is_some(), "the check is recorded");
        assert_eq!(stored.last_modified, Some(0));
        assert_eq!(stored.etag, None);
    }

    /// The 404 rung the 1:1 suites leave untested: the unavailable-overlay
    /// write also fires, keeping the stored models and the listing.
    #[tokio::test]
    async fn a_404_marks_the_catalog_unavailable_but_keeps_the_stored_models() {
        let mock = MockHttpClient::new();
        catalog_route(&mock).respond(MockResponse::status(404).with_body("gone"));
        let (provider, store) = catalog_rig(&mock, None);
        ModelsStore::write(
            &*store,
            "test-provider",
            ModelsStoreEntry {
                models: vec![model("test-provider", "seed")],
                ..ModelsStoreEntry::default()
            },
            None,
        )
        .await
        .expect("the seed entry writes");

        // The seeded entry has no checked date, so the freshness window
        // cannot skip the fetch and no force is needed.
        refresh_provider(&provider, &store, RefreshOverrides::default())
            .await
            .expect("the 404 refresh resolves");

        let stored = stored_entry(&store).await;
        assert_eq!(
            stored_model_ids(&store).await,
            ["seed"],
            "the stored models survive"
        );
        assert!(stored.checked_at.is_some(), "the check is recorded");
        assert_eq!(stored.last_modified, Some(0));
        assert_eq!(stored.etag, None);
        assert_eq!(
            model_ids(&provider),
            ["static", "seed"],
            "the listing is unchanged"
        );
    }
}

/// Area 2: the management-HTTP retry ladder, upstream's five cases over the
/// injected client.
mod management_http {
    use super::*;

    /// The `example.test` mock as the trait object `fetch_with_retry` takes,
    /// paired with the handle the assertions read back.
    fn mock_client() -> (Arc<dyn HttpClient>, MockHttpClient) {
        let mock = MockHttpClient::new();
        let client: Arc<dyn HttpClient> = Arc::new(mock.clone());
        (client, mock)
    }

    /// The route matcher every case drives, upstream's `example.test` URL.
    fn test_route(mock: &MockHttpClient) -> MockRouteBuilder<'_> {
        mock.on(|request| request.url.contains("example.test"))
    }

    /// The attempt-counting rig the scripted retry cases share: the client
    /// pair plus the counter the handler bumps and the case reads back.
    fn counted_rig() -> (Arc<dyn HttpClient>, MockHttpClient, Arc<AtomicUsize>) {
        let (client, mock) = mock_client();
        let attempts = Arc::new(AtomicUsize::new(0));
        (client, mock, attempts)
    }

    /// The scripted route over the counted rig: every request bumps the
    /// counter and runs the case's body, upstream's fetch handler that
    /// counts its calls.
    fn scripted_route<F, Fut>(mock: &MockHttpClient, attempts: Arc<AtomicUsize>, body: F)
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<MockResponse, HttpError>> + Send,
    {
        test_route(mock).respond_fn({
            let body = Arc::new(body);
            move |_request| {
                let attempts = Arc::clone(&attempts);
                let body = Arc::clone(&body);
                async move { body(attempts.fetch_add(1, Ordering::SeqCst)).await }
            }
        });
    }

    /// The 200 `{"ok": true}` body the successful attempt answers with.
    fn ok_body() -> MockResponse {
        json_response(200, &serde_json::json!({"ok": true}))
    }

    /// The GET over the test client with the given retry options, the call
    /// every retry case drives.
    ///
    /// # Panics
    /// The request failure, which these cases' successful answers cannot
    /// produce.
    async fn retry_get(
        client: &Arc<dyn HttpClient>,
        options: FetchRetryOptions,
        message: &str,
    ) -> HttpResponse {
        fetch_with_retry(
            client,
            "https://example.test",
            Vec::new(),
            CancellationToken::new(),
            options,
        )
        .await
        .expect(message)
    }

    /// Upstream's first case: two transport failures then the answer — the
    /// default ladder carries two retries.
    #[tokio::test]
    async fn retries_a_transient_transport_failure() {
        let (client, mock, attempts) = counted_rig();
        scripted_route(&mock, Arc::clone(&attempts), |attempt| async move {
            match attempt {
                0 | 1 => Err(HttpError::Transport("fetch failed".to_owned())),
                _ => Ok(ok_body()),
            }
        });

        let response = retry_get(
            &client,
            FetchRetryOptions::default(),
            "the retried request lands",
        )
        .await;

        assert_eq!(response.status, 200);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "two failures then the answer"
        );
        assert_eq!(mock.request_count(), 3);
    }

    /// Upstream's second case: the shared budget survives a first failure
    /// and the retry still runs. The signal-identity assertion upstream
    /// makes (`signals[0] === signals[1]`) has no recorded-signal surface in
    /// Rust — the header restatement covers it.
    #[tokio::test]
    async fn shares_the_timeout_budget_across_attempts() {
        let (client, mock, attempts) = counted_rig();
        scripted_route(&mock, Arc::clone(&attempts), |attempt| async move {
            if attempt == 0 {
                return Err(HttpError::Transport("fetch failed".to_owned()));
            }
            Ok(ok_body())
        });

        let response = retry_get(
            &client,
            FetchRetryOptions {
                timeout_ms: Some(1_000),
                ..FetchRetryOptions::default()
            },
            "the budget survives the first failure",
        )
        .await;

        assert_eq!(response.status, 200);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "the shared budget still had room for the retry"
        );
        assert_eq!(mock.request_count(), 2);
    }

    /// Upstream's third case: a hung first attempt past the attempt budget
    /// is aborted and retried; the paused clock makes the sleeps instant.
    #[tokio::test(start_paused = true)]
    async fn retries_an_attempt_timeout() {
        let (client, mock, attempts) = counted_rig();
        scripted_route(&mock, Arc::clone(&attempts), |attempt| async move {
            if attempt == 0 {
                // The hung attempt: the sleep passes the attempt
                // budget, so the outer select's attempt deadline
                // wins before the handler answers.
                tokio::time::sleep(Duration::from_millis(5_000)).await;
            }
            Ok(ok_body())
        });

        let response = retry_get(
            &client,
            FetchRetryOptions {
                attempt_timeout_ms: Some(4_000),
                ..FetchRetryOptions::default()
            },
            "the retried attempt answers",
        )
        .await;

        assert_eq!(response.status, 200);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "the attempt timed out once and retried"
        );
        assert_eq!(mock.request_count(), 2);
    }

    /// Upstream's fourth case: a retryable status response retries into the
    /// successful one.
    #[tokio::test]
    async fn retries_transient_http_responses_and_returns_the_successful_response() {
        let (client, mock) = mock_client();
        test_route(&mock)
            .respond_sequence(vec![MockResponse::status(503).with_body("busy"), ok_body()]);

        let response = retry_get(
            &client,
            FetchRetryOptions::default(),
            "the 503 retried into the answer",
        )
        .await;

        assert_eq!(response.status, 200);
        assert_eq!(mock.request_count(), 2);
    }

    /// Upstream's fifth case: a pre-cancelled token fails with the abort
    /// failure before any request leaves.
    #[tokio::test]
    async fn does_not_retry_caller_cancellation() {
        let (client, mock) = mock_client();
        let token = CancellationToken::new();
        token.cancel();

        let error = fetch_with_retry(
            &client,
            "https://example.test",
            Vec::new(),
            token,
            FetchRetryOptions::default(),
        )
        .await
        .expect_err("the pre-cancelled token fails before any request");

        match error {
            HttpError::Aborted => {}
            other => panic!("expected the abort failure, got: {other:?}"),
        }
        assert_eq!(
            mock.request_count(),
            0,
            "a pre-cancelled token never reaches the client"
        );
    }
}

/// Area 3: minimatch compile vectors binding the branches the unit tests in
/// `src/utils/minimatch.rs` leave thin. Every vector's npm minimatch 10.2.6
/// value was checked against the released package; the three divergences are
/// pinned with a note.
mod minimatch_vectors {
    use super::*;

    /// A mid-pattern globstar matches any number of segments between its
    /// fixed neighbors, zero included, and the segment may be empty (npm
    /// minimatch 10.2.6 matches both the single-slash and the doubled-slash
    /// spellings the same way).
    #[test]
    fn a_mid_pattern_globstar_matches_any_number_of_segments() {
        assert!(matches("a/b/c", "a/**/c", false));
        assert!(!matches("a/x", "a/**/c", false));
        assert!(matches("a/c", "a/**/c", false));
        assert!(matches("a//c", "a/**/c", false));
    }

    /// A leading globstar spans any number of leading segments, zero
    /// included — npm minimatch 10.2.6 matches the bare `c` the same way.
    #[test]
    fn a_leading_globstar_matches_any_number_of_leading_segments() {
        assert!(matches("a/b/c", "**/c", false));
        assert!(matches("x/c", "**/c", false));
        assert!(matches("c", "**/c", false));
    }

    /// Escaped metacharacters match their literal spellings only. The
    /// escapes ride the segment compiler, after brace expansion — so escaped
    /// braces are out of reach (npm minimatch 10.2.6 matches
    /// `"{a}"` against `"\{a\}"`).
    #[test]
    fn escaped_metacharacters_are_literals() {
        assert!(matches("*", "\\*", false));
        assert!(!matches("x", "\\*", false));
        assert!(matches("a?", "\\a\\?", false));
        assert!(!matches("ax", "\\a\\?", false));
    }

    /// `!` negation composes with character ranges.
    #[test]
    fn negated_classes_carry_ranges() {
        assert!(matches("model-5", "model-[!a-z]", false));
        assert!(!matches("model-d", "model-[!a-z]", false));
        assert!(matches("model-A", "model-[!a-z]", false));
    }

    /// Brace alternatives expand recursively over nested braces.
    #[test]
    fn nested_braces_expand() {
        assert!(matches("a", "{a,b{c,d}}", false));
        assert!(matches("bc", "{a,b{c,d}}", false));
        assert!(matches("bd", "{a,b{c,d}}", false));
        assert!(!matches("be", "{a,b{c,d}}", false));
    }

    /// The `{}` edge: the empty brace group expands to the empty
    /// alternative, so the pattern degenerates to the empty pattern. npm
    /// minimatch 10.2.6 keeps the literal braces instead.
    #[test]
    fn empty_braces_degenerate_to_the_empty_pattern() {
        assert!(matches("", "{}", false));
        assert!(!matches("x", "{}", false));
        assert!(!matches("{}", "{}", false));
    }

    /// `?` spans one non-separator character.
    #[test]
    fn question_marks_do_not_cross_slashes() {
        assert!(matches("abc", "a?c", false));
        assert!(!matches("a/c", "a?c", false));
    }

    /// The extglob forms compile as literals: the intended meaning never
    /// matches, and only the literal spelling does. npm minimatch 10.2.6
    /// supports extglobs and rejects even the literal spelling.
    #[test]
    fn extglob_patterns_compile_as_literals() {
        assert!(!matches("ab", "@(a|b)", false));
        assert!(!matches("a", "+(a)", false));
        assert!(matches("@(a|b)", "@(a|b)", false));
    }

    /// An empty pattern matches only the empty candidate.
    #[test]
    fn an_empty_pattern_matches_only_the_empty_candidate() {
        assert!(matches("", "", false));
        assert!(!matches("x", "", false));
    }

    /// A trailing globstar swallows its separator, so the bare prefix is a
    /// match. npm minimatch 10.2.6 does not match the bare prefix here.
    #[test]
    fn a_trailing_globstar_extends_a_bare_prefix() {
        assert!(matches("anthropic", "anthropic/**", false));
        assert!(matches("anthropic/claude", "anthropic/**", false));
        assert!(matches("anthropic/x/y", "anthropic/**", false));
        assert!(!matches("anthropics", "anthropic/**", false));
    }
}

/// Area 4: the models.json schema ladder over temp files, pinning the
/// validation rungs' dotted paths and the success path's comment/BOM
/// stripping.
mod models_json_schema {
    use super::*;

    /// One models.json load over a throwaway file.
    ///
    /// # Panics
    /// The temp-dir and file-write failures, which the fixture cannot
    /// produce.
    fn load_config(body: &str) -> ModelConfig {
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("models.json");
        std::fs::write(&path, body).expect("write models.json");
        ModelConfig::load(Some(path.to_str().expect("utf-8 temp path")))
            .expect("the temp path normalizes")
    }

    /// One provider's config body, the wrapper every provider-level rung
    /// shares.
    fn provider_body(provider: &str) -> String {
        format!(r#"{{"providers": {{"p": {provider}}}}}"#)
    }

    /// One model entry's config body, the wrapper every model-level rung
    /// shares.
    fn model_body(entry: &str) -> String {
        provider_body(&format!(r#"{{"models": [{entry}]}}"#))
    }

    /// Every validation rung reports through the wrapper with its dotted
    /// path, and an invalid file loads no providers.
    #[test]
    fn schema_errors_carry_the_dotted_path_and_empty_the_config() {
        let cases: &[(&str, &str)] = &[
            ("[]", "root: Expected object"),
            ("{}", "providers: Required property missing"),
            (r#"{"providers": []}"#, "providers: Expected object"),
            (&provider_body("5"), "providers.p: Expected object"),
            (
                &provider_body(r#"{"oauth": "magic"}"#),
                "providers.p.oauth: Expected union value: \"radius\"",
            ),
            (
                &provider_body(r#"{"headers": {"X-A": 5}}"#),
                "providers.p.headers: Expected object",
            ),
            (
                &provider_body(r#"{"compat": {"maxTokensField": "bogus"}}"#),
                "providers.p.compat: Expected object",
            ),
            (
                &provider_body(r#"{"compat": {"supportsUsageInStreaming": "yes"}}"#),
                "providers.p.compat: Expected object",
            ),
            (
                &provider_body(r#"{"compat": {"chatTemplateKwargs": {"x": {"$var": "nope"}}}}"#),
                "providers.p.compat: Expected object",
            ),
            (
                &model_body(r#"{"name": "m"}"#),
                "providers.p.models.0.id: Expected string",
            ),
            (
                &model_body(r#"{"id": ""}"#),
                "providers.p.models.0.id: Expected string",
            ),
            (
                &model_body(r#"{"id": "m", "input": ["audio"]}"#),
                "providers.p.models.0.input: Expected union value: \"text\" | \"image\"",
            ),
            (
                &model_body(r#"{"id": "m", "cost": {"input": 0}}"#),
                "providers.p.models.0.cost: Expected object",
            ),
            (
                &model_body(
                    r#"{"id": "m", "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "tiers": [{"input": 1}]}}"#,
                ),
                "providers.p.models.0.cost: Expected object",
            ),
            (
                &model_body(r#"{"id": "m", "samplingParams": 5}"#),
                "providers.p.models.0.samplingParams: Expected object",
            ),
            (
                &model_body(r#"{"id": "m", "compat": {"thinkingFormat": "bogus"}}"#),
                "providers.p.models.0.compat: Expected object",
            ),
        ];
        for (body, expected) in cases {
            let config = load_config(body);
            let Some(error) = config.get_error() else {
                panic!("the body must fail validation: {body}");
            };
            assert!(
                error.contains("Invalid models.json schema:"),
                "the wrapper names the file's failure: {error}"
            );
            assert!(
                error.contains(&format!("  - {expected}")),
                "the dotted path and message ride the list: {error}"
            );
            assert!(
                config.get_provider_ids().is_empty(),
                "an invalid file loads no providers: {error}"
            );
        }
    }

    /// An empty provider name fails the walk's `NonEmptyStr` node, upstream's
    /// typebox `minLength: 1`.
    #[test]
    fn an_empty_name_fails_the_schema_walk() {
        let config = load_config(&provider_body(r#"{"name": ""}"#));
        let error = config.get_error().expect("the empty name fails the walk");
        assert!(
            error.contains("providers.p.name: Expected string"),
            "the dotted path names the offending member: {error}"
        );
        assert!(
            config.get_provider_ids().is_empty(),
            "an invalid file loads no providers"
        );
    }

    /// The success path: a BOM and a leading comment strip before the parse,
    /// and a valid provider carries its models through.
    #[test]
    fn comments_and_a_bom_are_stripped_and_a_valid_provider_parses() {
        let config = load_config(&format!(
            "\u{FEFF}// provider config\n{}\n",
            serde_json::json!({
                "providers": {
                    "custom": {
                        "name": "Custom",
                        "baseUrl": "https://api.test/v1",
                        "apiKey": "$CUSTOM_KEY",
                        "api": "openai-completions",
                        "models": [{"id": "m1", "contextWindow": 1000, "maxTokens": 100}]
                    }
                }
            })
        ));

        assert_eq!(config.get_error(), None);
        assert_eq!(config.get_provider_ids(), ["custom"]);
        let provider = config.get_provider("custom").expect("the provider parses");
        assert_eq!(provider.name.as_deref(), Some("Custom"));
        let entries = provider.models.as_ref().expect("the model list parses");
        assert_eq!(entries[0].id, "m1");
        assert_eq!(entries[0].context_window, Some(1000));
        assert_eq!(entries[0].max_tokens, Some(100));
    }
}
