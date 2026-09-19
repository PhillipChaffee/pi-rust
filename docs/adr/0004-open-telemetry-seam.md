# Telemetry crosses crates as an open trait behind a dispatch handle

The telemetry capability is selected at run time — the adapter is pulled from the ambient context bag with a no-op fallback — so the thing stored in the bag and handed between crates must be one concrete type. The natural trait's `start_span` is generic, which makes `dyn TelemetryContext` impossible (E0038). Full parity now includes a Rust-native extension mechanism, so third-party adapters are a real future implementer, not a hypothetical. Decision: `pi-telemetry` exposes an open, unsealed trait plus a `Clone` dispatch handle (`Arc<dyn>` with boxed futures) as the concrete type downstream crates hold; the generic trait stays public for statically-known sites, where dispatch is monomorphic.

## Considered options

- **Generics threaded through ai and agent**: zero-cost where the type is static, but the ambient bag is a runtime store, so pure generics cannot carry the design; threading would also add a type parameter to every downstream signature.
- **Closed enum** (`Noop | InMemory | Span`): fastest dispatch and exactly matched the adapter set at the pin, but hard-codes the adapter list in `pi-telemetry`, excluding the extension-authored adapters the destination requires.

## Consequences

- Every span operation pays one vtable hop plus one future allocation — noise against the provider calls and tool runs spans wrap, and the same dynamic dispatch TypeScript already performs on every interface call.
- A downstream crate with one statically-known adapter can still use the raw trait generically; no erasure is forced on statically-typed code.
