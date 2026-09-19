# Survey: pi's `ai` package

Reference pin: `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (verified; `git -C ~/git/pi rev-parse HEAD` matches). Paths below are relative to `~/git/pi/packages/ai/`.

## 1. What the package is

`@earendil-works/pi-ai` v0.85.1 (`package.json:2,4`) is pi's unified LLM layer: one `Context`/`Message`/`Tool`/`AssistantMessageEvent` model, ten wire-protocol implementations ("APIs"), ~46 provider configurations with generated model catalogs, credential/OAuth resolution, retry and context-overflow handling, cost/usage tracking, and image generation — exposed both as a modern `Models` runtime (`createModels()`) and a deprecated global `stream()`/`complete()` compat API (`src/compat.ts:1-11`, slated for deletion at the coding-agent ModelManager migration). It also ships a `pi-ai` bin (`package.json:47-49`, `src/cli.ts` — OAuth login/list CLI) and a Bun-specific OAuth entry (`src/bun-oauth.ts`). The README positions it as "Unified LLM API with provider collections, automatic auth resolution, token and cost tracking, and simple context persistence and hand-off to other models mid-session" (`README.md:3`), tool-calling models only.

## 2. Public API surface

Entry points (from the `exports` map, `package.json:13-46`):

- `.` → `src/index.ts` — core, side-effect-free: `types.ts` (881 lines: `Context`, `Model`, `Message`, `Tool`, `Usage`, `AssistantMessageEvent`, `KnownApi` x10, `KnownProvider` x39, `ThinkingLevel`, `Transport`), `models.ts` (957 lines: `createModels()`, `Models` runtime, `Provider` registration, refresh/transforms), `models-store.ts` (catalog persistence), `auth/*` (context, credential-store, resolve, types, oauth/ flows), `providers/faux.ts` (708-line in-memory fake provider), `session-resources.ts`, `images-models.ts`, and utils: `event-stream.ts` (`EventStream`/`AssistantMessageEventStream`), `assistant-message-frame.ts`, `diagnostics`, `json-parse`, `overflow`, `retry`, `text`, `typebox-helpers`, `uuidv7`, `validation`.
- `./compat` → `src/compat.ts` — legacy global API: `stream()`/`complete()` with env-key injection, api-registry, `getModel`/`getModels`/`getProviders`, per-API lazy wrappers, image generation (`legacy-api-aliases.ts` deprecated stream aliases).
- `./providers/*` → `src/providers/` — one factory file per provider (e.g. `anthropic.ts`, `openai.ts`, `google.ts`, `amazon-bedrock.ts`, `github-copilot.ts`, `openai-codex.ts`, `radius.ts`, `faux.ts`), `all.ts` (155-line builtin registry: `builtinModels()`/`builtinProviders()`), per-provider generated catalogs `*.models.ts` importing `./data/<provider>.json` via JSON import attributes (data files generated at build time by `scripts/generate-models.ts`, which fetches models.dev + provider APIs; `src/providers/data/` does not exist in a fresh clone), plus auth helpers (`cloudflare-auth.ts`, `opencode-headers.ts`, `opencode-go.ts`, `radius-config.ts`, qwen/xiaomi token plans) and `providers/images/register-builtins.ts`.
- `./api/*` → `src/api/` — one module per wire protocol: `anthropic-messages.ts`, `openai-completions.ts` (1717 lines), `openai-responses.ts` + `openai-responses-shared.ts` (793 lines), `openai-codex-responses.ts` (1657 lines, SSE + WebSocket transports), `azure-openai-responses.ts`, `google-generative-ai.ts`, `google-vertex.ts` + `google-shared.ts`, `mistral-conversations.ts` (941 lines), `bedrock-converse-stream.ts` (1329 lines), `pi-messages.ts`, `openrouter-images.ts` (images), plus `lazy.ts` (`lazyStream`/`lazyApi` — sync stream return with async setup behind it), `simple-options.ts`, `transform-messages.ts` (cross-provider message rewriting), `constrained-sampling.ts`, `github-copilot-headers.ts`, `openai-prompt-cache.ts`; each impl has a `.lazy.ts` dynamic-import wrapper.
- `./utils/*`, `./oauth` (`src/oauth.ts`), `./bedrock-provider`, `./bun-oauth`.
- Side-effect modules registered in `package.json:8-12`: `compat.js`, `images.js`, `providers/images/register-builtins.js`.

Notable core plumbing in `src/utils/`: `provider-retry.ts`, `retry.ts`, `abort.ts`/`abort-signals.ts`, `event-stream.ts` (hand-rolled async-iterable broadcast queue), `overflow.ts`, `estimate.ts`, `json-parse.ts`, `deferred-tools.ts` (deferred tool results: `DeferredHandle`/`fetchDeferred`/`cancelDeferred` in `ProviderStreams`), `uuid.ts` (uuidv7 with timestamp-follower ordering), `sanitize-unicode.ts`, `provider-env.ts`, `node-http-proxy.ts`.

## 3. Dependency edges

### Workspace (crate DAG)

- `ai` imports exactly one workspace package: `@earendil-works/pi-telemetry` — as a type in `src/types.ts:1` and one test (`test/telemetry-options.test.ts:1`); `vitest.config.ts:15` aliases it to `../telemetry/src/index.ts` so tests run against telemetry source. So in the crate graph: **pi-ai -> pi-telemetry** (edge of a `TelemetryContext` type — a natural candidate to keep as a small trait).
- Consumers (who imports `ai`): `agent` (`pi-agent-core` deps: chord, pi-ai, telemetry), `coding-agent` (deps pi-agent-core, pi-ai, tui, client, protocol, server, chord), `evals` (pi-ai, coding-agent). `ai` is therefore the near-leaf of the workspace DAG; only `telemetry`, `tui`, `chord` sit below it. (`packages/session-backends` contains `sqlite-node` with no top-level package.json.)

### External npm dependencies

| Dependency | Used for | Where |
|---|---|---|
| `@anthropic-ai/sdk` 0.124.0 | Anthropic client, `beta.messages.create().asResponse()` raw SSE | `src/api/anthropic-messages.ts` |
| `openai` 6.40.0 | OpenAI client for completions/responses/codex/azure/openrouter-images | `src/api/openai-completions.ts:1`, `openai-responses-shared.ts:1`, `azure-openai-responses.ts`, `openrouter-images.ts` |
| `@google/genai` 2.21.0 | Google Generative AI SDK | `src/api/google-generative-ai.ts` |
| `@aws-sdk/client-bedrock-runtime` + `@smithy/node-http-handler` | Bedrock Converse API, custom node HTTP handler | `src/api/bedrock-converse-stream.ts:25-26` |
| `http-proxy-agent` / `https-proxy-agent` | HTTP(S) proxy support for Bedrock | `src/api/bedrock-converse-stream.ts:27-28`, `src/utils/node-http-proxy.ts` |
| `partial-json` | parse truncated streaming tool-call JSON | `src/utils/json-parse.ts:1` |
| `typebox` 1.3.27 | tool-parameter JSON schema (Type/Static/TSchema), re-exported from index | `src/index.ts:1-2`, `src/utils/typebox-helpers.ts` |

Codex WebSocket transport uses `globalThis.WebSocket` (`src/api/openai-codex-responses.ts:992` — Node 22+/Bun global, no `ws` dep). Dev deps: `vitest` 4.1.9, `canvas` 3.2.3 (test image generation), `@types/node`. Engines: node >= 22.19.0.

## 4. Test-suite inventory

147 `test/*.test.ts` files + helpers (`test/oauth.ts` — reads `~/.pi/agent/auth.json`, auto-refreshes expired OAuth tokens; `azure-utils.ts`, `bedrock-utils.ts`, `cloudflare-utils.ts` credential guards; `codex-websocket-cached-probe.ts`; `scratch.ts` manual demo; `data/red-circle.png` used by 5 image tests). Vitest 4, node env, 30s timeout. No snapshot files; no fixtures beyond the PNG.

Testing styles, by share:

1. **Offline SDK-mock tests (~45 files use vi mocks/stubs/fake timers)** — `vi.mock("<sdk>")` with `vi.hoisted` state: `anthropic-auth-token.test.ts`, `bedrock-raw-stop-reason.test.ts`, `openai-completions-retry.test.ts`, `openrouter-images.test.ts`; `vi.stubGlobal("fetch", ...)` for raw-fetch APIs and OAuth flows: `oauth-auth.test.ts`, `radius-oauth.test.ts`, `xai-oauth.test.ts`, `xai-responses.test.ts`; fake `Response` objects with hand-built SSE bodies injected via fake client `asResponse()`: `anthropic-sse-parsing.test.ts` (585 lines).
2. **Pure unit tests** — `event-stream.test.ts` (queue ordering, #9055 regression), `uuid.test.ts`, `text.test.ts`, `validation.test.ts`, `overflow.test.ts`, `context-estimate.test.ts`, `assistant-message-frame.test.ts`, `deferred-tools.test.ts`, `tool-call-id-normalization.test.ts`, `unicode-surrogate.test.ts`, `lax-message-content.test.ts`, `sampling-options.test.ts`, plus ~30 provider-thinking/reasoning/retry conformance tests (`anthropic-thinking-disable`, `google-thinking-level-map`, `mistral-tool-schema`, `openai-completions-cache-control-format`, `cross-provider-handoff`, ...) driven by mock transports.
3. **Model-catalog/data tests** — `generate-models-strict.test.ts` and `fireworks-model-generation.test.ts` spawn child Node processes with a patched `globalThis.fetch` (`spawnSync`, line 39-55); `generate-models-strict.test.ts` regenerates catalogs and diffs; `model-data-validation`, `models-runtime`, `*-models.test.ts` (~15 files) validate generated `.models.ts` data.
4. **Credential-gated live-network tests (~24 files import the guards)** — `stream.test.ts` (1716 lines: text/tool/streaming/abort/image flows across every provider; top-level OAuth token resolution; even spawns `ollama serve`/`ollama pull` at lines 1621-1647), `empty.test.ts`, `tokens.test.ts` (usage on abort), `total-tokens.test.ts`, `abort.test.ts` share this helper pattern; `*-e2e.test.ts` (6 files: `openai-responses-cache-affinity-e2e`, `openai-codex-cache-affinity-e2e`, `anthropic-eager-tool-input-e2e`, `anthropic-long-cache-retention-e2e`, `openai-responses-reasoning-replay-e2e`, `anthropic-thinking-binding-e2e`) gate via `describe.skipIf(!process.env.OPENAI_API_KEY)`-style checks; `context-overflow.test.ts` probes `http://localhost:11434` (Ollama).

Hard-to-port specifics: fake timers + `vi.setSystemTime` (retry backoff, OAuth polling, uuidv7 ordering — Rust needs `tokio::time::pause` or an injected clock); `vi.mock` of entire npm SDK modules (Rust has no module mocking — needs trait seams or a test HTTP server, e.g. wiremock); `vi.stubGlobal("fetch")`; top-level `await` module init in live tests; child-process spawning with patched fetch.

## 5. Porting flags (TypeScript -> Rust-native answers needed)

- **Async-iterable event streams**: `EventStream<T,R>` (`src/utils/event-stream.ts:26-89`) implements `AsyncIterable` with a hand-rolled FIFO of waiter promises + a `result()` promise. Rust: a channel (`mpsc`/`broadcast`) + a oneshot for the final result; note push-after-done is silently dropped and buffered events drain in order (regression-tested).
- **Sync-stream-with-async-setup**: `lazyStream`/`lazyApi` (`src/api/lazy.ts:46-98`) return the stream synchronously while auth/lazy-imports resolve behind it; errors arrive as in-stream `error` events. Rust: no lazy dynamic import — translate to a trait object or enum with the same error-in-stream contract.
- **Global fetch + streaming**: `FetchFunction = typeof globalThis.fetch` (`src/types.ts:115`), injectable per-request; SSE parsed from `Response.body` byte streams. Rust: `reqwest` + SSE parsing behind a `HttpClient` trait for tests.
- **WebSocket transport** for Codex via `globalThis.WebSocket` (`src/api/openai-codex-responses.ts:992`), with SSE-fallback state machine and cached-probe transport (`Transport = "sse" | "websocket" | "websocket-cached" | "auto"`, `src/types.ts:110`).
- **Dynamic typing**: `Api = KnownApi | (string & {})` and `ProviderId = KnownProvider | string` open-ended string unions (`src/types.ts:17-33,76`); `StreamOptionsWithExtras = StreamOptions & Record<string, unknown>` in tests; `Tool<typeof schema>` generic over typebox schemas with `Type.Object` const-literal schemas at call sites. Rust: `String` IDs plus a serde-validated tool schema representation (typebox -> JSON Schema -> `schemars`/`serde_json::Value`).
- **Provider factories + registration**: side-effectful module registration (`sideEffects`, `providers/images/register-builtins.ts`, api-registry), JSON import attributes for generated catalog data (`src/providers/anthropic.models.ts:4`), and the generated-data build step (`scripts/generate-models.ts`, models.dev fetch). Rust: explicit registry/builder + `include_str!`/build-script generated data.
- **AbortSignal plumbing everywhere** (`AbortController` per request, `operationSignal`, `raceWithAbortSignal` in `src/utils/abort.ts`) -> tokio `CancellationToken`.
- **uuidv7 with timestamp-follower ordering** (`src/utils/uuid.ts`) — needs a monotonic clock + randomness source with test hooks.
- **Env/OAuth credential surface**: `~/.pi/agent/auth.json` credential store (see `test/oauth.ts:14`), OAuth device-code + PKCE flows per provider (`src/auth/oauth/`), Bun-specific `process.env` quirks (`src/utils/provider-env.ts:7-16`). Rust: keep the same JSON file shape; port flows with an injectable clock and HTTP client.
- **Node built-ins**: `child_process` spawning (`stream.test.ts` ollama, generation tests), `fs` auth store with 0600 perms, Node 22 globals (`fetch`, `WebSocket`). Rust: `std::process::Command`, `tokio-tungstenite`, etc.
