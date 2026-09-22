//! The OpenCode per-conversation routing header, ported from
//! `packages/ai/src/providers/opencode-headers.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::types::{ProviderStreams, SimpleStreamOptions, StreamOptions};
use crate::utils::event_stream::AssistantMessageEventStream;

const OPENCODE_SESSION_HEADER: &str = "x-opencode-session";

fn has_header(headers: Option<&crate::types::ProviderHeaders>, name: &str) -> bool {
    headers.is_some_and(|headers| headers.keys().any(|key| key.eq_ignore_ascii_case(name)))
}

fn with_session_header(options: StreamOptions) -> StreamOptions {
    if options.session_id.is_none() || has_header(options.headers.as_ref(), OPENCODE_SESSION_HEADER)
    {
        return options;
    }
    let session_id = options.session_id.clone().unwrap_or_default();
    let mut updated = options;
    let mut headers = updated.headers.take().unwrap_or_default();
    headers.insert(OPENCODE_SESSION_HEADER.to_owned(), Some(session_id));
    updated.headers = Some(headers);
    updated
}

fn with_simple_session_header(options: SimpleStreamOptions) -> SimpleStreamOptions {
    if options.session_id.is_none() || has_header(options.headers.as_ref(), OPENCODE_SESSION_HEADER)
    {
        return options;
    }
    let session_id = options.session_id.clone().unwrap_or_default();
    let mut updated = options;
    let mut headers = updated.headers.take().unwrap_or_default();
    headers.insert(OPENCODE_SESSION_HEADER.to_owned(), Some(session_id));
    updated.headers = Some(headers);
    updated
}

/// Adds OpenCode's required per-conversation routing header before API
/// dispatch, upstream's `withOpenCodeSessionHeader`.
#[must_use]
pub fn with_opencode_session_header(streams: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
    Arc::new(OpenCodeSessionHeader(streams))
}

/// The wrapped streams of [`with_opencode_session_header`].
struct OpenCodeSessionHeader(Arc<dyn ProviderStreams>);

impl ProviderStreams for OpenCodeSessionHeader {
    fn stream(
        &self,
        model: &crate::types::Model,
        context: &crate::types::Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let adjusted = options.cloned().map(with_session_header);
        self.0.stream(model, context, adjusted.as_ref())
    }

    fn stream_simple(
        &self,
        model: &crate::types::Model,
        context: &crate::types::Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let adjusted = options.cloned().map(with_simple_session_header);
        self.0.stream_simple(model, context, adjusted.as_ref())
    }
}
