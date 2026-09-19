# Survey: pi remote-session surface — protocol, client, server

Reference: earendil-works/pi at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (HEAD verified).
Packages: `~/git/pi/packages/{protocol,client,server}` — all version 0.85.1, MIT, ESM, Node >= 22.19.

## 1. What this surface is

These three packages form one CBOR-framed remote-session stack. `@earendil-works/pi-protocol`
(`packages/protocol`) owns the wire: routed envelopes, a strict definite-length RFC 8949 CBOR subset,
and 4-byte length-prefixed framing (`packages/protocol/src/protocol.ts`, `src/cbor/`, `src/framing.ts`,
`src/codec.ts`). `@earendil-works/pi-client` (`packages/client`) is a transport-neutral client that
opens a `ByteTransport`, performs a versioned handshake, then correlates requests/cancels/subscriptions
over framed bytes (`src/client.ts`, `src/connection.ts`, `src/transport.ts`). `@earendil-works/pi-server`
(`packages/server`) mirrors it: `Server` accepts `ByteConnection`s from pluggable `ServerListener`s,
handshakes, decodes client messages, and routes service calls to a host-provided
`ServerHost`/`SessionRouter` (`src/server.ts`, `src/listener.ts`, `src/session-router.ts`, `src/connection.ts`).
Payload semantics inside envelopes are NOT owned by these packages: calls
(`{serviceId, instance?, member, args}`), subscription snapshots/updates, and error codes belong to the
workspace package `@earendil-works/chord`; `Context`/`SessionMetadata`/abort plumbing belongs to
`@earendil-works/pi-agent-core`. The protocol validates only that opaque payloads are strict JSON
(README: `packages/protocol/README.md:15`). Transports must preserve byte order; auth is out of scope.

## 2. Per-package public API

**protocol** — `package.json` exports only `"."`.
- `src/index.ts`: re-exports `src/cbor/index.ts` (`encodeCbor`, `decodeCbor`, `CborError`, `CborOptions`,
  `DEFAULT_MAX_CBOR_BYTE_LENGTH`/`_CONTAINER_LENGTH`/`_DEPTH`), `src/codec.ts` (`encodeClientMessage`,
  `encodeServerMessage`, `ClientMessageDecoder`, `ServerMessageDecoder`, `parseClientMessage`,
  `parseServerMessage`, `ProtocolValidationError`, `isSupportedProtocolVersion`), `src/framing.ts`
  (`encodeFrame`, `FrameDecoder`, `FrameError`, `DEFAULT_MAX_FRAME_LENGTH`, `FrameDecoderOptions`), and
  selected types from `src/protocol.ts` (`PROTOCOL_VERSION`, `ClientHello`, `ClientMessage`,
  `ServerHello`/`ServerHelloError`, `ServerMessage`, `RequestEnvelope`, `CancelEnvelope`,
  `ResponseEnvelope`, `ServiceEventEnvelope`, `AttachmentEnvelope`, `RpcTarget`, `SessionTarget`,
  `ServerId`, `isServerId`, `ProtocolError`, `ProtocolErrorCode`).

**client** — exports `"."` and `"./unix"`.
- `src/index.ts`: `Client`, `createClientServiceTransport` (`src/client.ts`);
  `ClientDisposedError`, `DisconnectedError`, `ServerError` (`src/errors.ts`); type-only
  `ByteTransport`/`ByteTransportFactory`/`ByteTransportHandlers` (`src/transport.ts`),
  `ClientOptions`, `ConnectionState`/`ConnectionStateChange`, `ServiceSubscription`, `Unsubscribe`,
  `AttachmentChangeListener`, `ListenerErrorHandler` (`src/types.ts`).
- `src/unix.ts` (subpath export, not in index): `createUnixTransportFactory`, `discoverUnixServers`,
  `UnixTransportOptions`, `UnixServerRoute`, `DiscoverUnixServersOptions`.
- Internal but port-relevant: `Connection` state machine (`src/connection.ts`), promise resolvers
  (`src/promise.ts` — shims `Promise.withResolvers`).

**server** — exports `"."`, `"./testing"`, `"./unix"`.
- `src/index.ts`: star re-exports of `errors.ts` (`ServerError`, `WrongServerError`,
  `SessionNotFoundError`, `SessionAmbiguousError`, `SessionNotAttachedError`, `ServerDrainingError`,
  `INTERNAL_SERVER_ERROR_MESSAGE`), `listener.ts` (`ServerListener`), `server.ts` (`Server`,
  generic over `SessionMetadata`), `types.ts` (`ServerOptions`, `ByteConnection` lives in
  `src/connection.ts` along `ByteConnectionHandler`/`ByteConnectionAcceptor`/`ConnectionState`;
  `RoutedSessionAttachment`, `RoutedServerServiceAttachment`, `RoutedServerServiceHost`,
  `RoutedSessionHandle`, `ServerHost` in `src/types.ts`).
- `src/transports/unix/index.ts`: `createUnixListener`, `createUnixServer` (preset composing
  `Server` + listener), `getUnixSocketPath`, option types (`listener.ts`, `preset.ts`, `address.ts`,
  `types.ts`). `UnixByteConnection` is exported `@internal` for transport tests (`listener.ts:191`).
- `src/testing/index.ts` (test-support subpath): `createTestServer`, `TestServerHost`, `TestHarness`,
  `Deferred`, `createTestServerServices`, `ProtocolTestClient`, `WireChannel`, `connectUnixTestClient`
  (`testing/server.ts`, `testing/host.ts`, `testing/client.ts`).
- `src/session-router.ts`: `SessionRouter` (not exported from index; internal routing core).

## 3. Framing and message shapes (byte-exact)

- **Frame** = 4-byte unsigned big-endian payload length, then exactly that many payload bytes
  (`framing.ts:28-39`). Length 0 is a legal empty frame. `DEFAULT_MAX_FRAME_LENGTH` = 16 MiB
  (16_777_216); a decoder fed a declared length above its limit fails the moment the header completes
  (`framing.ts:77-79`) and stays failed. `FrameDecoder` accepts arbitrary chunk boundaries, assembles
  payloads through 64 KiB internal blocks (copying, never aliasing input), rejects truncated streams at
  `end()`, and rejects `maxFrameLength` outside 0..2^32-1.
- **Payload** = one definite-length CBOR item (RFC 8949 subset, `cbor/encoder.ts`, `cbor/decoder.ts`):
  major types 0/1 (uint/nint, minimal-length argument encoding, safe integers only, ±2^53-1 bounds),
  2 (byte strings), 3 (text, UTF-8 validated strictly, BOM preserved), 4 (arrays), 5 (maps, string
  keys only, duplicate keys rejected), 7/20-22 (bool/null), 7/27 = float64 big-endian (`0xfb`).
  Rejected: indefinite lengths, tags, break, float16/float32, non-finite numbers, `undefined`, holes,
  cycles, symbol keys, trailing data. `-0` round-trips as float64 `fb8000000000000000`.
  Encoder omits `undefined` map values (JSON-like); decoder builds plain objects with own enumerable
  props (so `__proto__` stays data). Limits: `DEFAULT_MAX_CBOR_BYTE_LENGTH` 16 MiB,
  `DEFAULT_MAX_CBOR_CONTAINER_LENGTH` 1_000_000, `DEFAULT_MAX_CBOR_DEPTH` 64 (cap 512), checked before
  traversal where possible (`cbor/options.ts`, `cbor/decoder.ts:112-117`).
- **Validation layer** (`codec.ts`): typebox schemas (`protocol.ts`) with `additionalProperties: false`
  everywhere; payloads must also pass chord's `isJsonValue` (strict JSON: no Uint8Array, NaN, undefined,
  cycles). Failures throw `ProtocolValidationError`; decoders latch failed state. Message schemas:
  client = `hello{version:int>=0}` | `request{id,target,call}` | `cancel{id,target}`;
  server = `hello{version:8,serverId}` | `hello_error{error{code,message}}` |
  `response{id,ok:true,result?}` | `response{id,ok:false,error}` | `service_update{subscriptionId,update}` |
  `attachment{attachment: SessionTarget|null}`. `RpcTarget` = `{serverId}` or
  `{serverId,sessionId,attachmentId}`; `ServerId` must match the canonical lowercase UUIDv4 regex
  (`protocol.ts:12-14`). `PROTOCOL_VERSION = 8`.
- **Transport plug-in**: client side is `ByteTransportFactory(handlers) -> ByteTransport` where handlers
  are `onData(chunk)`, `onClose()`, `onError(error)` and the transport offers ordered
  `send(chunk): Promise<void>` + idempotent `close()` (`client/src/transport.ts`). Server side is the
  mirror `ByteConnection{closed, send, close(finalChunk?)}` + `ByteConnectionHandler`, delivered by a
  `ServerListener.start(accept)` (`server/src/connection.ts`, `server/src/listener.ts`). Handshake: client
  sends `hello` first; server replies `hello` (matching serverId) or `hello_error` then closes
  (`client/src/connection.ts:163-213`, `server/src/server.ts:262-296`). A 5 s default server handshake
  timeout closes with a final `hello_error{code:"invalid_request"}` frame.

## 4. Dependency edges

- **protocol** -> `@earendil-works/chord` (only `isJsonValue` + `JsonValue` type, `codec.ts:1`),
  `typebox` 1.3.27 (schema build + `Check`, `protocol.ts`). No other runtime deps.
- **client** -> `@earendil-works/pi-protocol`, `@earendil-works/chord` (service-call parse/build,
  catalogue/subscription snapshot+update codecs, `ServiceStateDecoder`, `RemoteServiceTransport`,
  `BACKGROUND_CONTEXT` from `chord/context` — `client.ts:1-19`). Node builtins only in `src/unix.ts`
  (`node:fs/promises`, `node:net`, `node:path`). devDeps: vitest 4.1.9, shx.
- **server** -> `@earendil-works/pi-protocol`, `@earendil-works/chord` (service-call parse, control-call
  decode, `ServiceStateEncoder`), `@earendil-works/pi-agent-core` (`Context`, `SessionMetadata`,
  `BACKGROUND_CONTEXT`, `TODO_CONTEXT`, `withAbortSignal`, `MemorySessionRepo` in testing host —
  `server.ts:11`, `testing/host.ts:3`). Node builtins: `node:crypto` (randomUUID, sha256),
  `node:net` (server/sockets), `node:fs/promises` (mkdir/chmod/link/rename/unlink/lstat), `node:path`.
- Crate-DAG shape for Rust: `protocol` is a leaf (plus a JSON-value check); `client` depends on
  `protocol` + the chord-semantics crate; `server` depends on `protocol` + chord-semantics + the
  agent-core crate. The chord/pi-agent-core crates are separate tickets; their API surface used here is
  limited to the symbols named above.
- Test files reach across packages by relative import (`client/test/unix.test.ts:7-9` imports
  `../../server/src/...`), so a Rust workspace should expect cross-crate test usage.

## 5. Test-suite inventory

**protocol** (3 files, ~40 test declarations; `.each` expands, e.g. 33 RFC 8949 vectors):
- `test/framing.test.ts` (11): header/empty-frame bytes, fragmented/coalesced/byte-at-a-time decode,
  multi-block 70 KB assembly, every split point, copy-not-alias, truncation at `end`, oversized declared
  length, exact-max acceptance, push-after-end, invalid `maxFrameLength` values. Pure; ports directly.
- `test/cbor/cbor.test.ts` (9): known-vector hex round-trips (33 vectors incl. -0 and MAX_SAFE_INTEGER),
  undefined-omission, BOM + `__proto__`-as-data, encoder rejections (holes, NaN, bigint, symbol,
  Date, Map, cycles, depth, symbol keys, lossy strings), 28 decoder-rejection hex cases (indefinite,
  tags, float16/32, unsafe ints, dup keys, bad UTF-8 incl. overlong/surrogate), limit enforcement with
  pre-declared lengths, stricter caller limits. Hex fixtures = byte-exact porting oracle.
- `test/protocol.test.ts` (20): version negotiation, hello/request/cancel/response/attachment schema
  acceptance + rejection (extra fields, empty ids, non-canonical UUIDs, JSON-string-as-message),
  opaque payload strict-JSON rule, framed encode/decode round-trips, split-at-every-point incremental
  decode, hostile frames (empty/malformed/schema-invalid CBOR), truncation + oversized framing.
- Hazards: byte-exact CBOR vectors (Rust CBOR lib must emit minimal-length args, omit undefined,
  reject non-JSON), `-0` float preservation, safe-integer bounds, latched failed decoder state.

**client** (4 files, 27 declarations):
- `test/client.test.ts` (15): `MemoryByteServer` in-memory transport (`test/support.ts:10-84`) —
  connects only to expected serverId, attachment out-of-band updates, subscription update buffering
  before snapshot (`hydrated`/`ready`/`queued` machinery, `client.ts:172-236`), out-of-order response
  correlation, bounded ServerError, pre-aborted request sends nothing, cancel envelope, disconnect/dispose
  rejection, data-before-hello, hello_error, reconnect via fresh transport, invalid/truncated framing,
  unmatched response fails connection.
- `test/unix.test.ts` (8): `discoverUnixServers` against real sockets — missing dir, serverId-ordered
  discovery, malformed/non-socket/mismatched entries ignored, stale socket (SIGKILLed fork of
  `test/fixtures/stale-socket-server.mjs`) left in place, unresponsive-socket timeout, 16-probe
  concurrency cap (polls live connection count), close-before-handshake ignored, ENOTDIR propagation.
- `test/unix-transport.test.ts` (4): real Unix socket end-to-end handshake + request with byte-by-byte
  and split-frame writes, truncated final frame surfaces through Client, ENOENT rejection, option
  validation (win32-gated describe).
- `test/support.ts` is the fixture hub; `fixtures/stale-socket-server.mjs` is a forkable child process.
- Hazards: Node `Socket` event semantics (connect/data/end/close/error/drain), write backpressure
  (`maxPendingBytes` + `writeTail` ordering, `client/src/unix.ts:161-235`), `AbortSignal` plumbing,
  forked child process fixture, cross-package imports of server internals.

**server** (6 files, 42 declarations):
- `test/server.test.ts` (6): option validation, concurrent-start rejection without leaking the Unix
  listener, handshake timeout closes with final `hello_error` frame (fake `ByteConnection` captures
  `finalChunk`), timeout-above-Node-max rejection, `maxPendingBytes < maxFrameLength+4` rejection,
  close/closed reject when listener shutdown fails.
- `test/protocol.test.ts` (7): first-message-must-be-hello, version rejection, fragmented hello/request,
  hostile frames, second-hello rejection, coalesced hello+request in one chunk, truncated final frame
  reported via `onError`. Uses in-memory `ByteConnection` + `ProtocolTestClient`.
- `test/conformance.test.ts` (20): the behavioral core — handshake w/o session listing, semantically
  invalid call after envelope decode, metadata typed through host, opaque server services + out-of-band
  attachment publish, multiple attachments per Session, attachment-release failure clears ownership,
  attachment fencing (stale route rejected), opaque results + adapter-defect bounding, concurrent calls,
  demand held until admitted call settles after disconnect, wrong-server rejection before repo access,
  unknown/ambiguous session without Harness creation, terminated-handle invalidation, connection-loss vs
  shutdown release ordering, and 4 routed-acquisition failure races (lease vs termination, shared
  creation failure, shutdown racing acquisition, in-flight acquisition blocking shutdown). Gates
  (`OpenGate`/`Deferred`) make the races deterministic.
- `test/listener.test.ts` (2): all listeners started/closed, startup failure closes started ones.
- `test/unix.test.ts` (6): `getUnixSocketPath` derivation, live-listener rejection without unlink,
  never unlinks a regular file, nested-parent creation + 0600 mode + self-cleanup, replacement-inode
  preservation on shutdown, stale-socket removal before bind.
- `test/unix-connection.test.ts` (1): final protocol error queued behind pending output before close.
- `src/testing/` is a shipped test-support subpath (fixtures live in-package, not `test/`):
  `TestServerHost` wraps `MemorySessionRepo`; `ProtocolTestClient` offers `sendFragmented`,
  message waiters, `waitForClose` (`testing/client.ts`).
- Hazards: timers (`unref`ed handshake/graceful-close timeouts), deterministic race gates, inode-stability
  checks (dev/ino), socket rename/link dance for atomic bind (`transports/unix/listener.ts:47-86`),
  graceful close with final chunk behind pending writes.

## 6. Porting flags (TS constructs needing a Rust-native answer)

- **Incremental byte decoding without Node streams**: pi never uses Node streams — it feeds
  `FrameDecoder`/`*MessageDecoder` with arbitrary `Uint8Array` chunks from socket `data` events. In Rust,
  model this as a stateful decoder with `push(&[u8]) -> Vec<...>` + `end()` (or integrate with
  `tokio_util::codec::Decoder`, but the tests demand chunk-boundary independence, so keep the explicit
  buffer API for the port). Byte-exactness: big-endian u32 header, definite-length CBOR only.
- **CBOR library choice**: the codec is a hand-rolled strict subset (no tags, no indefinite, no
  non-UTF-8, safe-int bounds, duplicate-key rejection, undefined-omitted maps, `-0` as f64). Off-the-shelf
  crates (ciborium, serde_cbor) will not reproduce the rejection surface; port the encoder/decoder
  1:1 or pin a lib with strict-mode config and add the 33-vector + 28-rejection tests as the oracle.
  Frame length/CBOR limits must be enforced pre-traversal to avoid allocating hostile payloads.
- **Schema validation**: typebox + `Check` + strict-JSON gate. Rust analog: serde with
  `deny_unknown_fields` enums for the five server/three client shapes, plus a custom JSON-value guard
  (no floats that are NaN/inf, no maps, cycles impossible by construction).
- **AbortSignal/AbortController**: client cancels and server request cancellation rely on them
  (`client.ts:262-269`, `server.ts:302,328-331`). Rust: `tokio_util::sync::CancellationToken` or
  `futures` abort handles per request; cancellation must send a `cancel` envelope exactly once.
- **Promise ordering invariants**: `writeTail` serializes socket writes and keeps `pendingBytes` bounded
  (`client/src/unix.ts:161-177`, `server/.../listener.ts:212-228`); `deliveryTail` orders listener
  callbacks (`client.ts:417-421`); `sendFragmented`/split-frame tests rely on ordering. Rust: per-
  connection `mpsc` write task or mutex-guarded ordered writes with backpressure accounting.
- **Unix socket lifecycle**: bind-to-scratch-path then `link` + `chmod` + unlink-scratch for atomic
  takeover (`server/.../listener.ts:52-86`), dev/ino identity checks before cleanup, stale-socket probe
  via live connect (`isSocketLive`). Rust: `std::os::unix::net`/tokio UnixListener + libc/stat; Windows
  is explicitly unsupported (`client/src/unix.ts:38`).
- **Timers**: handshake timeout (5 s default), graceful-close timeout (5 s), discovery probe timeout —
  all `.unref()`ed so they never hold the process open. Rust: tokio timeouts; nothing to unref, but
  ensure shutdown does not await them.
- **DOMException/AggregateError/Error.cause**: Rust needs an error taxonomy —
  `ProtocolValidationError`/`FrameError`/`CborError` vs wire `ServerError{code}` vs disconnect causes;
  `AggregateError` in server shutdown maps to error-collection in `close()` results.
- **`process.platform` gates and `MAX_TIMER_DELAY_MS` (2^31-1) validation**: fold into config validation;
  the platform gate disappears on Unix-only Rust (or gate with `#[cfg(unix)]`).
- **`Symbol.asyncDispose`** (`client.ts:394`) and `Promise.withResolvers` shim: no direct analog;
  `async fn connect` + explicit `dispose()` (or `Drop`) in Rust.