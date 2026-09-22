//! The uniform adapter scaffolding the wire-API modules share, ported from
//! upstream's per-API option plumbing at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The `From<StreamOptions>` conversion each options struct carries and the
//! [`crate::types::ProviderStreams`] dispatch each module forwards repeat one
//! shape per API upstream; the macros keep the variation (the adapter's extra
//! option fields, its option type) at the call site and the plumbing
//! single-sourced.

/// Generate the [`crate::types::ProviderStreams`] dispatch a wire-API module
/// carries: `stream` converts the base [`crate::types::StreamOptions`] into
/// the adapter's option type and forwards to the module's `stream`, and
/// `stream_simple` forwards verbatim.
macro_rules! impl_provider_streams {
    ($streams:ident, $options:ident) => {
        impl crate::types::ProviderStreams for $streams {
            fn stream(
                &self,
                model: &Model,
                context: &Context,
                options: Option<&StreamOptions>,
            ) -> AssistantMessageEventStream {
                let options = options.cloned().map($options::from);
                stream(model, context, options.as_ref())
            }

            fn stream_simple(
                &self,
                model: &Model,
                context: &Context,
                options: Option<&SimpleStreamOptions>,
            ) -> AssistantMessageEventStream {
                stream_simple(model, context, options)
            }
        }
    };
}

/// Generate the `From<StreamOptions>` conversion a wire-API options struct
/// carries, upstream's `...Options extends StreamOptions` spread: the base
/// fields move field-for-field and the adapter's extras initialize from the
/// listed expressions.
macro_rules! impl_stream_options_from {
    ($options:ident from $source:ident { $($field:ident : $value:expr),* $(,)? }) => {
        impl From<crate::types::StreamOptions> for $options {
            fn from($source: crate::types::StreamOptions) -> Self {
                Self {
                    transport_options: $source.transport_options,
                    api_key: $source.api_key,
                    telemetry_context: $source.telemetry_context,
                    env: $source.env,
                    headers: $source.headers,
                    timeout_ms: $source.timeout_ms,
                    max_retries: $source.max_retries,
                    max_retry_delay_ms: $source.max_retry_delay_ms,
                    temperature: $source.temperature,
                    max_tokens: $source.max_tokens,
                    cache_retention: $source.cache_retention,
                    session_id: $source.session_id,
                    metadata: $source.metadata,
                    $($field: $value,)*
                }
            }
        }
    };
}

pub(crate) use impl_provider_streams;
pub(crate) use impl_stream_options_from;
