//! Go-like cancellation and invocation-scoped values, ported from upstream
//! `src/context/index.ts`: the shared background context and placeholder
//! context, `ContextKey` runtime identities with descriptions, the layered
//! value chain and its scoping helpers, cancellation wiring over
//! `tokio_util`'s `CancellationToken` (upstream's `AbortSignal` machinery,
//! including the fan-in and masking behaviors), and running a future inside
//! a context with `awaitWithContext`.
