//! The pluggable session-search contract, upstream's `src/search/index.ts`.
//!
//! A pure interface with no implementation in this crate — pi itself ships
//! none; hosts (the coding agent, extensions) provide one. The optional
//! `searchEntries` upstream restates as a provided default that returns no
//! hits, because no pi consumer calls it: implementers that do not search
//! entries omit the override, and callers of the future coding-agent port
//! degrade to an empty result the way an omitted method does upstream.

use pi_ai::types::BoxedFuture;

/// The error a search-service method surfaces, upstream's rejected promise.
///
/// The service is host-provided, so the rejection type is the host's — the
/// boxed-error stand-in upstream's `Error` throw gets.
pub type SessionSearchError = Box<dyn std::error::Error + Send + Sync>;

/// A search request, upstream's `SearchQuery`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchQuery {
    /// The search text.
    pub text: String,
    /// The hit cap. Omitted: the service's own limit applies.
    pub limit: Option<u64>,
}

/// A matched session, upstream's `SessionSearchHit`.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionSearchHit {
    /// The session's id.
    pub session_id: String,
    /// The relevance score. Omitted when the service does not score.
    pub score: Option<f64>,
    /// The best-matching entry of the session. Omitted when the service does
    /// not report one.
    pub top: Option<SessionSearchTopHit>,
}

/// The best-matching entry of a [`SessionSearchHit`], upstream's inline
/// `top` object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionSearchTopHit {
    /// The matched entry's id.
    pub entry_id: String,
    /// The matched text excerpt. Omitted when the service does not excerpt.
    pub snippet: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// A matched transcript entry, upstream's `EntrySearchHit`.
#[derive(Clone, Debug, PartialEq)]
pub struct EntrySearchHit {
    /// The session the entry belongs to.
    pub session_id: String,
    /// The matched entry's id.
    pub entry_id: String,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// The matched text excerpt. Omitted when the service does not excerpt.
    pub snippet: Option<String>,
    /// The match score. Omitted when the service does not score.
    pub score: Option<f64>,
}

/// The session-search surface hosts install for the coding agent's transcript
/// search, upstream's `SessionSearchService`.
pub trait SessionSearchService: Send + Sync {
    /// Search sessions for `query`, upstream's `searchSessions`.
    ///
    /// # Errors
    /// The returned future resolves to the service's failure.
    fn search_sessions<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxedFuture<'a, Result<Vec<SessionSearchHit>, SessionSearchError>>;

    /// Search transcript entries for `query`, upstream's `searchEntries`.
    ///
    /// Upstream declares this method optional (no pi consumer calls it);
    /// the default restates "the service does not search entries" as an
    /// empty result.
    fn search_entries<'a>(
        &'a self,
        _query: &'a SearchQuery,
    ) -> BoxedFuture<'a, Result<Vec<EntrySearchHit>, SessionSearchError>> {
        Box::pin(std::future::ready(Ok(Vec::new())))
    }

    /// Bring the search index up to date with recent session writes,
    /// upstream's `sync`.
    ///
    /// # Errors
    /// The returned future resolves to the service's failure.
    fn sync(&self) -> BoxedFuture<'_, Result<(), SessionSearchError>>;

    /// Record that `session_id` changed, upstream's `notify` — a synchronous
    /// enqueue; indexing happens on the service's own schedule.
    fn notify(&self, session_id: &str);

    /// Drop `session_id` from the index, upstream's `remove`.
    ///
    /// # Errors
    /// The returned future resolves to the service's failure.
    fn remove<'a>(&'a self, session_id: &'a str)
    -> BoxedFuture<'a, Result<(), SessionSearchError>>;

    /// Release the service's resources, upstream's `close`.
    ///
    /// # Errors
    /// The returned future resolves to the service's failure.
    fn close(&self) -> BoxedFuture<'_, Result<(), SessionSearchError>>;
}
