# Survey: pi `agent` package (pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759)

Source: `~/git/pi/packages/agent` (read-only). HEAD verified at the expected pin; the only
working-tree difference is an untracked `opencode.json` (not part of the pin).

## 1. What the package is

`@earendil-works/pi-agent-core` v0.85.1 (`package.json:4`) is pi's agent runtime: a stateful
`Agent` class plus a low-level `agentLoop` that drive LLM turns, tool execution, and event
streaming on top of `@earendil-works/pi-ai`. At this pin it also carries a large `src/harness/`
subtree — a second, newer runtime ("AgentHarness") with lane-based operation admission, a
`Session` storage layer (memory + JSONL with v3 migration), compaction, skills, prompt
templates, built-in bash/edit/read/write tools, and telemetry schemas. The core (~2.2k lines in
`agent.ts`/`agent-loop.ts`/`types.ts`/`proxy.ts`) is the transport-abstracted agent loop the
ticket describes; the harness (~20k lines) is the state-management half. A port that targets
only the core is a fraction of the work; the harness is most of it.

## 2. Public API surface

Package exports map (`package.json:8-37`): `.`, `./node`, `./harness/context`,
`./harness/env/nodejs`, `./harness/runtime/reducer`, `./harness/session`,
`./harness/session/testing`.

- `src/index.ts` — main barrel: re-exports agent core, harness, session, tools, compaction,
  telemetry, skills, system-prompt, result/error types, `setDefaultStreamFn`.
- `src/agent.ts` (592 lines) — `Agent` class: `prompt`, `continue`, `steer`, `followUp`,
  `subscribe`, `abort`, `waitForIdle`, queue-clearing, mutable `AgentState` with copy-on-assign
  `tools`/`messages` accessors (`agent.ts:68-95`).
- `src/agent-loop.ts` (803 lines) — `agentLoop`, `agentLoopContinue`, `runAgentLoop*`; returns
  `EventStream<AgentEvent, AgentMessage[]>` (`agent-loop.ts:32-94`).
- `src/types.ts` (446 lines) — `AgentMessage`, `AgentEvent` union (`types.ts:431-446`),
  `StreamFn` contract (`types.ts:28-32`), `AgentTool`, `AgentState`,
  `beforeToolCall`/`afterToolCall`/`shouldStopAfterTurn` hook types, `ToolExecutionMode`,
  `QueueMode`, `ThinkingLevel`.
- `src/proxy.ts` (402 lines) — `streamProxy` + compact wire events for browser→server proxying.
- `src/stream-fn.ts` — global default `StreamFn` injection (`setDefaultStreamFn`).
- `src/node.ts` — `NodeExecutionEnv` re-export plus main barrel.
- `src/search/index.ts` — `SessionSearchService` interface (pluggable, no impl here).
- `src/harness/agent-harness.ts` (622 lines) — `AgentHarness`/`AgentLane` types + `Result`
  error taxonomy (`Closed`, `LaneBusy`, `InvalidNavigation`, …).
- `src/harness/runtime/harness.ts` (408) + `runtime/drive/*.ts` — harness impl and the
  generation/tool/retry/deferred/reconcile/terminal drive phases.
- `src/harness/session/` — `StorageBackedSession` (`session.ts`), `MemorySessionRepo`
  (`memory.ts`), `JsonlSessionRepo` + codec/torn-tail handling (`jsonl/`), commit/fork policy,
  `values.ts` durable addresses, `legacy-v3.ts` migration.
- `src/harness/session/testing/` — exported conformance suites (`conformance/session-repo.ts`,
  `storage.ts`) and benchmark datasets; part of the public exports map.
- `src/harness/tools/` — `createBashTool`, `createEditTool`, `createReadTool`, `createWriteTool`,
  `edit-diff.ts`, `file-mutation-queue.ts`, `output-capture.ts`.
- `src/harness/compaction/` — `compact`, `shouldCompact`, branch summarization.
- `src/harness/skills.ts`, `prompt-templates.ts`, `system-prompt.ts`, `telemetry.ts`,
  `hooks.ts`, `events.ts`, `config.ts`, `env/nodejs.ts`.

## 3. Dependency edges

### Workspace imports (crate-DAG edges)

- `@earendil-works/pi-ai` — dominant edge (30+ files): message/model/usage types, `EventStream`,
  `validateToolArguments` (`agent-loop.ts:12`), `RetryPolicy`/`retryDelayMs` (`config.ts:1`,
  `runtime/drive/retry.ts:1`), `DeferredHandle` (`runtime/drive/deferred.ts:1`),
  `AssistantMessageFrame`/`reduceAssistantMessageFrames` (`runtime/recovery.ts`,
  `progress.ts`), `uuidv7` (`session/*.ts`), `Transport` (`agent.ts:9`, `harness/types.ts:1`),
  faux provider helpers (test-only).
- `@earendil-works/chord` — `JsonValue`/`JsonRepresentation` (`session/types.ts:1`,
  `agent-harness.ts:1`); `Context`/`ContextKey` async-context machinery —
  `BACKGROUND_CONTEXT`, `withCancel`, `withAbortSignal`, `awaitWithContext`,
  `withContextValue` re-exported from `harness/context.ts:1-25` and threaded through nearly
  every harness API.
- `@earendil-works/pi-telemetry` — span schemas/types (`index.ts:2-40`, `harness/telemetry.ts`,
  `harness/context.ts:12`).

So the Rust crate graph is: `agent` → {`pi-ai`, `chord`, `pi-telemetry`}; nothing else in
`packages/` imports it upstream except apps.

### External npm deps

- `typebox` — schema-typed tool parameters (`Static<TSchema>`, `types.ts:16`,
  `harness/tools/*.ts`); also drives argument validation via pi-ai.
- `diff` — edit-diff computation for the edit tool (`tools/edit-diff.ts:5`).
- `ignore` + `yaml` — SKILL.md loading with ignore-file filtering (`skills.ts:1-2`).
- `yaml` — prompt template frontmatter parsing (`prompt-templates.ts:1`).
- Dev: vitest 4.1.9, `@vitest/coverage-v8`, typescript 5.9.3 (`tsconfig.build.json` builds with
  `tsgo`). Node >= 22.19.

## 4. Test-suite inventory

57 vitest files, 21,660 lines total. Two configs: `vitest.config.ts` (30s timeout, aliases
workspace deps to source) runs `test/*.test.ts`; `vitest.harness.config.ts` runs
`test/harness/**` with v8 coverage over `src/harness/**` + `agent.ts` + `agent-loop.ts`
(`vitest.harness.config.ts:14-19`); `vitest.benchmark.config.ts` for session benches.

Core suite (`test/`):

- `agent.test.ts` (810) — `Agent` class end-to-end: event ordering, steering/follow-up queues,
  hooks, `terminate` batching, awaited-subscriber settlement (`agent.test.ts:14` defines
  `MockAssistantStream extends EventStream` — the canonical fake transport).
- `agent-loop.test.ts` (1610) — loop-level: streaming deltas, tool preflight/parallel vs
  sequential, abort, `prepareNextTurn`, default-stream-fn fallback via
  `queueMicrotask` (`agent-loop.test.ts:84-116`).
- `proxy.test.ts` (129) — `streamProxy` wire-event decoding.
- `e2e.test.ts` (415) — Agent against pi-ai's `fauxProvider` (`e2e.test.ts:1-13`), registration
  cleanup in `afterEach`, `test/utils/calculate.ts` + `get-current-time.ts` fixture tools.

Harness suite (`test/harness/`, 44 files):

- Session/storage: `memory-storage`, `jsonl-storage` (torn-tail recovery),
  `jsonl-io` (atomic publication), `jsonl-session-repo` (+`-conformance`),
  `memory-session-repo`, `memory-conformance`, `storage-backed-session`,
  `session-create-branch`, `branch`, `jsonl-v3-migration` (2013 lines — biggest file; legacy v3
  transcript fixtures), `jsonl-v3-stream`, `values`, `mutation-line`, `gating-storage`,
  `instrumented-storage`, `text-line-reader`. Conformance suites are parameterized fixtures from
  `src/harness/session/testing/` run against both memory and JSONL backends.
- Runtime: `runtime/lane`, `runtime/harness` (lane management, global metadata),
  `runtime/drive-*` (structural 1454, generation 796, tools 763, reconcile 747,
  retry-deferred 613, public 612, terminal, retry, progress), `accept`, `watch`, `restore`,
  `reducer`, `progress`. Fixtures: `MemoryStorage` + `FailingMemoryStorage`/
  `ControlledMemoryStorage` overrides (`test/harness/runtime/test-utils.ts:4-24`), `deferred()`
  promise gates, faux provider models.
- Execution/tools/compaction/skills: `execution-primitives` (gate, HookRegistry, HarnessEventBus),
  `execution-assistant`, `execution-tools`, `tools` (782 — built-in tool behavior against
  `NodeExecutionEnv`), `nodejs-env` (706), `compaction` (827), `branch-summarization`,
  `skills`, `prompt-templates`, `system-prompt`, `truncate`, `output-capture`,
  `adaptive-publisher`, `types` (523 — `Result`/`ExecutionEnv` surface), `telemetry`,
  `context`, `session-context`, `resource-formatting`, `text-line-reader`.
- Stray: `docs/mobile-handoff/01-harness/01-delta/delta.test.ts` (563) — chord-style delta-op
  tracker tests outside `test/`; decide whether to port or drop.

Porting hazards:

- Async timing: settlement ordering around `agent_end` awaited subscribers (`agent.test.ts`),
  `queueMicrotask`-driven default-stream test, `waitForTick()` = `setTimeout(0)`
  (`test/utils/wait-for-tick.ts`) — Rust tests need tokio::time or deterministic wakeups.
- Event ordering: parallel tool mode emits `tool_execution_end` in completion order but
  toolResult messages in assistant source order (README.md:119-124; `drive-tools.test.ts`).
- Fake transports: `MockAssistantStream`/`EventStream` subclassing and `fauxProvider` — the
  port needs an equivalent test-double seam at the `StreamFn` boundary.
- Storage: torn-tail JSONL recovery, atomic publication, v3 migration fixtures (large literal
  transcript blobs to transcribe).
- `AbortSignal` is a parameter of every hook and storage call; cancellation tests rely on
  deferred promises resolving mid-run.
- No fake timers observed; timing control is deferred-promise based — good news for porting.

## 5. Porting flags — TS constructs needing Rust-native answers

- `EventStream<Event, Final>` (pi-ai; `agent-loop.ts:38`, `proxy.ts:20`) — push-based stream
  with a terminal value. Rust: `Stream<Item=AgentEvent>` via `async-stream`/mpsc, final value
  as a separate oneshot or terminal variant.
- Subscriber model: `agent.subscribe` awaits listeners in registration order and gates idle
  state on them (README.md:178, `types.ts:431` note). Rust: no direct emitter; needs an
  ordered listener registry + explicit join/settlement, or a broadcast channel that cannot
  preserve await-semantics — design decision required.
- `Context`/`ContextKey` (chord; `harness/context.ts`) — Go-style ambient cancellation +
  typed values threaded through ~every harness call (`BACKGROUND_CONTEXT`,
  `withTelemetryContext`). Rust has no ambient context; must choose explicit
  `&Context` params, task-locals, or a `CancellationToken` + separate telemetry span arg.
- `AbortController`/`AbortSignal` (`agent.ts:491`, `harness/execution/effect-gate.ts:33`,
  `runtime/types.ts:116`) → `tokio_util::sync::CancellationToken`.
- `StreamFn` indirection + module-global default (`stream-fn.ts`) — Rust: trait object or
  generic; the global default becomes `once_cell` + `RwLock` or is dropped in favor of
  explicit injection.
- `Transport` (pi-ai, imported in `agent.ts:9`, `harness/types.ts:1`) — transport abstraction
  the harness options carry; Rust port needs the pi-ai-side trait first.
- Declaration-merging extension of `AgentMessage` (`types.ts:317-326`) — no Rust equivalent;
  use an untagged enum with an app-extension variant or a generic.
- `typebox` `Static<TSchema>` compile-time inference (`types.ts:387-412`) — Rust: schemars +
  serde, or a tool-args trait with JSON validation.
- Get/set accessor properties with copy-on-assign semantics (`agent.ts:78-89`) — Rust: methods
  (`set_tools`, `set_messages`) that clone; document the copy semantics.
- Hand-rolled `Result<T,E>`/`ok`/`err` (`harness/types.ts:9-30`) maps 1:1 to Rust `Result` —
  the taxonomy of harness error types (`agent-harness.ts:21-52`) ports naturally as enums.
- Async generators over `for await` (`README.md:503-510`) — Rust `Stream` + `StreamExt`.
- Node-specific bits (`env/nodejs.ts`, `tools/bash.ts`, `jsonl/io.ts` atomic publication) map
  to std/tokio::process + fsync/rename; watch for Windows assumptions.