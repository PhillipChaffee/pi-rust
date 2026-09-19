# Survey: pi's `chord` package

Reference pin: pi @ `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (HEAD verified).
All paths relative to `~/git/pi/packages/chord/`.

## 1. What the package is

`@earendil-works/chord` is an application-composition runtime for agentic systems
assembled from plugins: facets (setup units), typed services (singleton/keyed),
replicated state (delta-tracked JSON), a transport-agnostic remote-service wire
grammar, Go-like cancellation contexts, and an esbuild-based facet bundler/loader
(`README.md:1-65`, `PLANNING.md`). It is deliberately *not* a Pi package: it
depends on no other workspace package and a boundary test enforces that
(`test/boundary.test.ts`). Inside pi it is the substrate every runtime piece
stands on — `protocol` carries its payload grammar, `agent`/`server`/`client`
speak its service vocabulary, and `coding-agent` uses it to host and hot-reload
plugin facets (`packages/coding-agent/src/experimental/services/*.ts`).

## 2. Public API surface

Five subpath exports (`package.json:7-27`):

- Root `src/index.ts` (79 lines) — re-exports everything below.
- `./context` `src/context/index.ts` — `BACKGROUND_CONTEXT`, `TODO_CONTEXT`,
  `createContextKey`, `withContextValue`, `withAbortSignal`,
  `withoutAbortSignal`, `withCancel`, `awaitWithContext` (121 lines).
- `./delta` `src/delta/index.ts` — standalone delta primitive (1267 lines):
  `Op`/`WireOp` tuples, `track()` tracker, `apply`/`applyImmutable`,
  `encoder`/`decoder` (path interning), `assertValidOp`/`assertValidWireOp`,
  `assertSafePath`, `overlap`, `isBase`, `isReplace` (`delta/index.ts:12-120,
  435, 764-1198`).
- `./bundler` `src/bundler.ts` — `bundleFacets` (`node/bundle.ts:39`),
  `bundleFacetPackage` (`node/package.ts:30`); manifest types
  (`node/manifest.ts`).
- `./node` `src/node.ts` — `createFacetBundleLoader`,
  `createFacetBundleArtifactLoader`, `readFacetBundleArtifact`,
  `readFacetBundleManifest` (`node/bundle-loader.ts:44,56`); format constants.

Root factory functions (`src/api.ts`): `createFacetHost`, `defineFacet`,
`defineService`, `createStaticFacetLoader`, `combineFacetLoaders`,
`createRemoteServiceBinding`, `replicatedState`. Core types in
`src/types.ts` (`Facet`, `FacetHost`, `Service`, `RemoteServiceTransport`,
`ReplicatedState`, `Context`, `JsonValue`, `JsonRepresentation`, and the
`$chord.service` snapshot/update/call vocabulary). Errors in
`services/errors.ts` (8 `RemoteServiceErrorCode`s).

## 3. The service / replication / plugin model

- **Facets** declare requirements/provisions in synchronous `setup(env)`
  (`types.ts:206-228`). `FacetKernel` (`facets/host.ts`, 906 lines) validates
  the graph (missing deps, cycles, duplicate providers), binds handles, then
  activates providers before consumers and disposes in reverse order; lifecycle
  gates (`facets/host.ts:59-140`) block handle use outside `setting_up`/`active`.
- **Services**: `singleton` or `keyed`; `local` ones keep unrestricted JS
  contracts and never cross the wire (`types.ts:62-68`, `api.ts:77-82`,
  `$chord.` prefix reserved). `RemoteServiceContract<T>` compile-time check
  (`types.ts:70-110`) forces JSON-only args/results/state.
- **Replication**: providers mutate a Proxy-tracked object and `publish(context)`
  (`services/state.ts:6-57`); the tracker flushes ops (`delta/index.ts:435`),
  consumers hold `ReplicatedStateReplica` with strict sequence fencing — gaps
  clear the replica (`services/state.ts:93-100`). Per-subscription path-codec
  registries reset on replacement/unavailable/close (`services/state-codec.ts`).
- **Remote boundary**: `RemoteServiceTransport` (`types.ts:184-192`) is the
  pluggable wire; the `$chord.service` control vocabulary
  (`services/wire.ts:39-64`) drives catalogue/subscribe/unsubscribe.
  `RemoteServiceProvider` (`services/provider.ts`) hosts instances and buffers
  subscriber updates; `RemoteServiceBindingImpl` (`services/consumer.ts`) keeps
  stable facades across provider replacement/rebind. Loopback transport ties the
  two together in-process (`services/loopback.ts`).
- **Plugins**: esbuild bundles each facet into a content-addressed CommonJS file
  plus `chord-facets.json` manifest (`node/bundle.ts:39-60`); the Node loader
  verifies SHA-256 integrity and compiles via `node:vm.compileFunction`, bypassing
  the require cache, with host-resolved externals (`node/bundle-loader.ts:81+`).
  Reload swaps singletons without an unavailable interval; keyed instances get
  fresh generations (`ServiceInstanceAddress.generation`).

## 4. Dependency edges

**(a) Workspace (crate) DAG — chord imports none; the arrows point inward:**

| Consumer (package.json dep) | Imports used |
|---|---|
| `protocol` | `JsonValue`, `isJsonValue` (`protocol/src/protocol.ts:1`, `codec.ts:1`) |
| `agent` | types + `/context` (`agent/src/harness/agent-harness.ts:1`, `context.ts:1-11`) |
| `client` | root API + `/context` (`client/src/client.ts:18-19`) |
| `server` | types: `ServiceCall`, `ServiceProviderUpdate`, `RemoteServiceErrorCode`, `ServiceStateEncoder` (`server/src/server.ts:10`, `errors.ts:1`, `connection.ts:1`) |
| `coding-agent` | deepest consumer: root + `/context` + `/node` + `/bundler`; experimental services define chord services (`coding-agent/src/experimental/services/*.ts:1`) |

Non-consumers: `ai`, `evals`, `session-backends`, `telemetry`, `tui`.

**(b) External npm deps:** only `esbuild` `0.28.2` (bundler); dev-only `shx`,
`vitest` `4.1.9`. Node engine `>=22.19.0` (`package.json:60-63`) for
`AbortSignal.any`, `node:vm.compileFunction` semantics, `import.meta.resolve`.

## 5. Test-suite inventory

9 test files, 160 test cases (`vitest --run`), no `*.mock` fixtures; `vi.fn`
used only as spy callbacks (`facets.test.ts`, `services.test.ts:284,618`,
`facet-loader.test.ts:61`); helpers are one re-export (`test/helpers.ts`).

| File | Cases | Behavior covered | Fixtures/mocks | Porting hazards |
|---|---|---|---|---|
| `delta.test.ts` | 100 | overlap algorithm; tracker intent/roots; base flush; immutable apply; flush-time minimization (append/front-truncate); array-index safety; op/wire-op assertion; codec interning; two property tests (seeded LCG `0x5eed1234`, `delta.test.ts:1034-1103`; Math.random round-trip 3000 iterations, `delta.test.ts:1106-1141`) | synthetic JSON, `structuredClone` | Property tests need a seeded PRNG in Rust (`Math.random` → `rand` with fixed seed); no async/time — deterministic |
| `services.test.ts` | 18 | singleton/keyed providers; replicated state publish/hydrate; facade stability across replacement/dispose; update buffering vs hydration races (`services.test.ts:553`); readiness failure; mode/member validation | loopback transport, `vi.fn` | Concurrency-ordered buffering (JS event loop semantics → Rust must pin an explicit update queue); async disposal ordering |
| `facets.test.ts` | 14 | dependency discovery, cycles/missing deps, keyed observation lifecycle, host termination on publication failure, reload stability, local-service isolation | loopback, `vi.fn` | Host reload cutover ordering; error aggregation (`AggregateError`) |
| `facet-loader.test.ts` | 9 | loader combination/reverse dispose; stable handles across reload; shape-change rejection before cutover; failed-candidate cleanup | `vi.fn` | Generation semantics; double-dispose guards |
| `bundle.test.ts` | 5 | real esbuild builds: content-addressed entries, external require, package conventions, corrupt-integrity rejection | temp dirs, real files, real `node:vm` | Requires esbuild equivalent or a different plugin-build story in Rust |
| `service-wire.test.ts` | 7 | `$chord.service` control calls; snapshot/update validation; per-state codec isolation; endpoint subscribe/cleanup | pure data | Straight port; tuple→enum |
| `context.test.ts` | 5 | layered values, cancellation inheritance/isolation, `withoutAbortSignal` masking, `awaitWithContext` | `vi.fn`, real `AbortController` | `AbortSignal.any`/`DOMException` → `tokio_util::CancellationToken`-style port |
| `json.test.ts` | 1 | `isJsonValue` acceptance/rejection (cycles, prototypes, sparse arrays) | pure data | Recursion depth cap 512 (`json.ts:9`) — watch stack depth |
| `boundary.test.ts` | 1 | no Pi-package imports, no files outside `src/` (`boundary.test.ts:11-33`) | reads own sources | Trivial to mirror as a test/lint rule |

## 6. Porting flags — TS constructs needing a Rust-native answer

1. **Proxy-based tracking** (`delta/index.ts:435` `track()`) and remote member
   proxies (`consumer.ts:50-63`, `handle.ts:42`) — Rust has no dynamic proxies;
   needs macro-generated tracked structs, a diffing write-guard, or explicit
   mutate-and-flush handles. This is the core semantic to re-design.
2. **Compile-time contract types** — `RemoteServiceContract<T>` and
   `JsonRepresentation<T>` conditional/mapped types (`types.ts:23-110`); Rust
   analogues are serde derive + trait bounds, not type-level tricks.
3. **Dynamic method dispatch** — `Reflect.apply(member.method, impl, [...args,
   context])` (`provider.ts:233`); Rust needs a member registry of boxed
   closures or an enum-per-service.
4. **Context/abort machinery** — `AbortSignal`/`AbortController`/`AbortSignal.any`
   (`context/index.ts:71-116`) → `tokio_util::sync::CancellationToken` fan-in or
   equivalent; `Context` is an immutable linked chain — fits Rust fine.
5. **Event-loop-concurrency assumptions** — provider subscriber buffers,
   `Promise.allSettled` disposal fan-out, `ready()` barrier (`provider.ts:42-63`,
   `facets/loader.ts`, `types.ts:119-121`) must become explicit queues/tasks with
   pinned ordering (tests depend on exact replay order).
6. **Node-native plugin loading** — `node:vm.compileFunction`, CommonJS,
   SHA-256 integrity, `import.meta.resolve` externals (`node/bundle-loader.ts`),
   esbuild builds (`node/bundle.ts`). A Rust port needs an embedded JS engine
   (deno_core/rquickjs/wasmtime) or a wasm-plugin contract instead.
7. **JS runtime odds and ends** — `structuredClone`, `AggregateError`,
   `DOMException`, `WeakMap` internal registration (`state-internals.ts:11`),
   `Reflect.ownKeys`/descriptor scans in `isJsonValue` (`json.ts`), frozen
   objects, `#private` fields, branded `SERVICE_TYPE` symbol, tuple ops.
   Tuple ops map cleanly to Rust enums; `Op` validation
   (`delta/index.ts:785-921`) guards `__proto__`-style prototype pollution —
   map to serde_json path checks.
8. **Numbers on the wire** — sequences/generations are plain counters the wire
   carries; keep them wire-sourced in the port (no fabricated counts).