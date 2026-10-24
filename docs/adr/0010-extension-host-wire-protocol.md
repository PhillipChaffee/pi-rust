# Third-party extensions speak a versioned CBOR session over stdio

Decision, at upstream pin `60e7e76`, extending [ADR 0007](0007-rust-native-extension-mechanism.md): an installed extension is a child process whose stdin/stdout carry one framed CBOR session; the host (`pi-coding-agent`'s `extensions::host` module) spawns it, handshakes, binds its declarations into the in-process `Extension` record, and proxies every dispatch back over the wire. First-party compile-ins keep registering in-process and never touch the wire.

## Substrate

Frames are pi-protocol's 4-byte big-endian length prefix plus one definite-length CBOR item ([`pi_protocol::framing`]), under its 16 MiB frame cap and full rejection surface (indefinite lengths, tags, break markers, duplicate keys, trailing data). stderr is diagnostics, forwarded verbatim and never parsed. The wire version is `EXTENSION_PROTOCOL_VERSION: u64 = 1`, its own constant: the remote-session `PROTOCOL_VERSION` (8) versions a different system, and [ADR 0007](0007-rust-native-extension-mechanism.md) keeps the two separate.

Every frame is a CBOR `map` with a `type` text discriminator. Payload fields carry the upstream TypeScript JSON shape: `serde_json::Value` on both sides, converted to and from `CborValue` by an integer-preserving bridge (integral numbers in `i64` range become `Int`; everything else `Float`). pi-protocol's remote-envelope conversion is not reused: it encodes every JSON number as a CBOR float, which type-shifts integers on the return trip. A JSON payload that is not strict JSON (unrepresentable numbers, byte strings) fails the frame.

## Handshake

1. Child → host: `hello { protocol_version, extension: { name, version } }`.
2. Host compares `protocol_version` with its constant and with the receipt. Mismatch or a refused receipt → `hello_error { code, message }`, close, fail closed. This mirrors the server's `hello_error` behavior.
3. Host → child: `welcome { protocol_version }`.
4. Child → host: zero or more registration frames, then `ready`.
5. Host → child: dispatch only after `ready`.

Registration frames: `register_event { channel }`, `register_tool { tool }`, `register_command { command }`, `register_flag { flag }`, `register_provider { provider }`. `tool` carries the declarative `ToolDefinition` fields (name, label, description, prompt snippet/guidelines, JSON-Schema parameters, constrained sampling, render shell, execution mode); `execute` and `prepare_arguments` are wire calls, not values.

## Dispatch and context

Each dispatch opens a call scope keyed by a host-assigned `call` id:

- `call_event { call, channel, payload }` → `call_result { call, result }` or `call_error { call, code, message, stack? }`.
- `call_tool { call, tool, tool_call_id, params, context }` → `call_update { call, update }` frames, then `call_result`. `context` is the serializable `ExtensionContext` snapshot (cwd, model, thinking level, session ids).
- `call_command { call, name, args }` → `call_result` / `call_error`.
- `call_provider { call, provider, request }` → `stream_event { call, event }` frames, then `stream_end { call }` or `call_error`.

Inside a scope the child reaches host state with request/response frames:

- `context_call { call, id, method, params? }` → `context_result { call, id, value }` / `context_error`.
- `ui_call { call, id, method, params? }` → `ui_result` / `ui_error`.

Methods are a versioned whitelist (`cwd`, `mode`, `has_ui`, model/thinking/scoped-model queries, `is_idle`, `is_project_trusted`, `has_pending_messages`, `get_context_usage`, `get_system_prompt`, `get_system_prompt_options`, `compact`, `abort`, `shutdown`, `get_signal`, plus `session.*` and `models.*` namespaces for the session-manager and model-registry reads). An unknown or unimplemented method answers `unsupported` or `unknown_method` in the scope, never a protocol failure, so additive methods do not bump the wire version. UI mutators stay [the interactive ticket's](https://github.com/PhillipChaffee/pi-rust/issues/132) surface; without a bound UI, `has_ui` is false and UI calls answer `unsupported`.

Signals: `get_signal` returns `{ signal: id }` scoped to its call; the host pushes `signal { call, signal, aborted: true }` when it trips, and the SDK flips its local signal. Handler failures are reported through the runner's error listener and read as `HandlerResult::None`, matching the in-process contract; the per-channel merge semantics stay host-side.

## Lifecycle

A handshake timeout, malformed frame, or early exit fails closed: the extension is reported as a load error or a dispatch diagnostic, and its handlers are skipped for the rest of the session (no automatic respawn until reload). An exit mid-call fails that call with `extension_exited`. Shutdown sends `shutdown`, closes stdin, waits a bounded grace, then kills the child; `Drop` does the same so a replaced session leaves no orphan.

The loader's `LoadExtensionsFn` seam owns the cache: one live session per resolved path, reused across the pre-trust and final load passes (so a trusted project never double-spawns), invalidated on cwd change or reload, and shut down for paths that leave the set. Cache state is loader-owned, not global.

Receipts gate installed paths only: verify every file's sha256 and the recorded protocol version before every spawn, fail closed on any byte or version change. Explicit local paths load in place without a receipt; first-party compile-ins bypass the process entirely. Sandbox enforcement stays deferred per [ADR 0007](0007-rust-native-extension-mechanism.md).

The child SDK is this crate's `extensions::host` module ([ADR 0003](0003-crate-graph-and-porting-route.md) adds no crate), published with the workspace ([ADR 0008](0008-pi-crates-publish-to-crates-io.md)), so an extension crate's `main` hands a factory to `host::serve` and the SDK runs the session.

## Considered options

- **Reuse the remote-session wire (`pi-client`/`pi-server`)** — it routes chord service calls and session attachments; the extension session carries handler calls and context reads. [ADR 0007](0007-rust-native-extension-mechanism.md) keeps the systems separate, and the payload meanings do not overlap.
- **JSON lines over stdio** — easier to eyeball, but throws away pi-protocol's tested framing and rejection surface and the byte-exact oracle; [ADR 0007](0007-rust-native-extension-mechanism.md) pinned CBOR.
- **CBOR-encode the domain structs directly (serde on every type)** — one representation instead of JSON-plus-bridge, but it invents a second wire shape beside upstream's TypeScript one and forces CBOR support into every payload type across `pi-ai`, `pi-agent-core`, and this crate.
- **Spawn per dispatch (stateless)** — no crash-state to track, but every event pays a handshake, provider streams have no home, and a crashed child is indistinguishable from a slow one.

## Consequences

- The extension event union, handler results, tool/provider descriptors, and `ExtensionContext` snapshot need JSON wire mappings (serialize on the host, deserialize in the SDK, and the reverse for results); `AbortSignal` becomes a wire signal id, closures become calls.
- One child process per extension: N extensions cost N handshakes at startup, kept cheap by the loader cache. Isolation is real — a crashing extension dies alone.
- Testing rides an injected duplex link (the whole protocol, hermetic) plus a spawn gate tested on its failure paths; the child-side fixture is a test binary, never a shipped extension.
