# Survey: pi `coding-agent` package

Reference pin: `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (HEAD of `~/git/pi` verified 2026-09-18).
All paths relative to `~/git/pi/packages/coding-agent/` unless noted.

## 1. What the package is

`@earendil-works/pi-coding-agent` (v0.85.1, `package.json:2-4`) is the product layer of pi: the `pi`
CLI that a developer actually runs. It composes the workspace's engine packages — `pi-ai`
(providers/streaming/auth), `pi-agent-core` (agent loop), `pi-tui` (terminal UI) — into a coding
agent with built-in read/bash/powershell/edit/write/grep/find/ls tools, session persistence, an
extension system, skills/themes/prompt-template loading, project-trust gating, and three run modes
(interactive TUI, print/JSON, RPC-over-stdio). `src/main.ts` is the orchestrator wiring all of this;
`src/core/` holds the domain logic (session manager, settings, auth storage, compaction, tools,
extensions, model runtime); `src/modes/` holds the run modes; `src/cli/` holds command-line
plumbing. It also ships an SDK surface (`src/index.ts` exports) for programmatic embedding, and an
experimental client/server split under `src/experimental/` (excluded from the published npm files).

## 2. Public API and CLI surface

**Package entry points** (`package.json:9-28`): bin `pi` → `dist/bundle/cli.js`; main export
`dist/index.js`; subpath exports `./rpc-entry` (RPC child process) and source-only `./client`,
`./experimental/plugin`. Entry script `src/cli.ts` (6 lines): `setupCli()` then `main(process.argv.slice(2))`.

**Main export barrel** — `src/index.ts` (439 lines). Groups: `parseArgs`/`Args`; config paths
(`CONFIG_DIR_NAME`, `getAgentDir`, `VERSION`); `AgentSession` (`src/core/agent-session.ts`);
`readStoredCredential` (`src/core/auth-storage.ts`); compaction API
(`src/core/compaction/index.ts`); the extension system types + runtime
(`src/core/extensions/index.ts`); `ModelRegistry`, `ModelRuntime`
(`src/core/model-registry.ts`, `src/core/model-runtime.ts`); `DefaultPackageManager`
(`src/core/package-manager.ts`), `DefaultResourceLoader` (`src/core/resource-loader.ts`); SDK
factories `createAgentSession`, `createAgentSessionServices`, `createCodingTools`,
`createReadTool` etc. (`src/core/sdk.ts`); `SessionManager` + all session entry types
(`src/core/session-manager.ts`); `SettingsManager` (`src/core/settings-manager.ts`); skills
(`src/core/skills.ts`); tool definitions (`src/core/tools/index.ts`); `ProjectTrustStore`
(`src/core/trust-manager.ts`); `main()` (`src/main.ts`); run modes `InteractiveMode`, `runPrintMode`,
`runRpcMode`, `RpcClient` (`src/modes/index.ts`); ~40 interactive UI components
(`src/modes/interactive/components/index.ts`); theme/clipboard/image/frontmatter utils.

**CLI commands and flags** — grammar in `src/cli/args.ts` (`parseArgs`, `printHelp`), dispatch in
`src/main.ts:562-640`. Subcommands: `install <source>`, `remove`/`uninstall`, `update [source|self|pi]`,
`list` (package management, `handlePackageCommand`), `config` (TUI resource selector,
`src/cli/config-selector.ts`), `auth print-api-key|print-bearer-token|check` (`src/cli/auth-command.ts`,
`src/cli/credential-print.ts`, `src/cli/auth-check.ts`). Flags: `--provider --model --api-key
--system-prompt --append-system-prompt --mode text|json|rpc --print/-p --continue/-c --resume/-r
--session --session-id --fork --session-dir --no-session --name/-n --models --tools/-t
--exclude-tools/-xt --no-tools --no-builtin-tools --thinking --extension/-e --no-extensions --skill
--prompt-template --theme --use-theme --no-* variants --export --list-models --tui-mode --verbose
--approve/-a --no-approve --offline -- --help --version` (`args.ts:91-246`). Unknown flags are
collected into `unknownFlags` for extensions to claim. `@file` args become attached file context
(`src/cli/file-processor.ts`). Help text also enumerates ~30 provider env-var names
(`args.ts:387-435`). Modes: interactive (`src/modes/interactive/`), print (`src/modes/print-mode.ts`),
JSON events (`src/modes/json-event.ts` — serializes `AgentSessionEvent`s, splitting streaming updates),
RPC (`src/modes/rpc/rpc-mode.ts` + `rpc-types.ts`).

## 3. On-disk data (critical for the TS→Rust import tool)

All user state lives under one root. `src/config.ts` derives everything:

| Location | Format | Written/read by |
|---|---|---|
| `~/.pi/agent/` | dir (env override `PI_CODING_AGENT_DIR`) | `config.ts:528-534` `getAgentDir()` |
| `~/.pi/agent/sessions/<encoded-cwd>/<timestamp>_<uuidv7>.jsonl` | JSONL | `core/session-manager.ts` |
| `~/.pi/agent/auth.json` | JSON object, 2-space, 0o600 | `core/auth-storage.ts` |
| `~/.pi/agent/settings.json` | JSON object | `core/settings-manager.ts` |
| `.pi/settings.json` (project cwd) | JSON object | `core/settings-manager.ts:233` |
| `~/.pi/agent/models.json` | JSON object (custom providers/models) | `core/model-runtime.ts:175`; docs/models.md |
| `~/.pi/agent/models-store.json` | JSON object keyed by providerId (refreshed catalogs) | `core/models-store.ts:47-60` `FileModelsStore` |
| `~/.pi/agent/trust.json` | JSON `Record<abs-path, true\|false\|null>` | `core/trust-manager.ts:212-213` |
| `~/.pi/agent/{skills,prompts,themes,extensions}/` | files | `core/resource-loader.ts:813-822` |
| `.pi/{skills,prompts,themes,extensions}/`, `.pi/SYSTEM.md`, `.pi/APPEND_SYSTEM.md` | files (project) | `core/resource-loader.ts:819-822, 1024-1043` |
| `~/.pi/agent/tools/`, `~/.pi/agent/bin/` (fd, rg), `~/.pi/agent/themes/`, `~/.pi/agent/prompts/` | dirs | `config.ts:557-569` |
| `~/.pi/agent/pi-debug.log` | text log | `config.ts:577-579` |
| Legacy: `~/.pi/agent/oauth.json`; `settings.json#apiKeys` | JSON | migrated to auth.json by `src/migrations.ts:21-60` |

Session files are the import-critical format. Default dir per project:
`getAgentDir()/sessions/--<cwd-with-slashes-colons-replaced-by-dashes>--/` — leading `/` stripped,
`/`, `\`, `:` → `-`, wrapped in `--...--` (`session-manager.ts:476-481`; doc:
`docs/session-format.md:7-11`). Filename: ISO timestamp with `:` and `.` replaced by `-`, `_`, then
the session id (`session-manager.ts:948-949`).

JSONL structure (`session-manager.ts:30-156`): line 1 is the header
`{type:"session", version:3, id, timestamp, cwd, parentSession?}` (v1 files lack `version`). Every
subsequent line is one entry `{type, id, parentId: string|null, timestamp, ...}` forming a tree:
`message` (wraps an `AgentMessage`: user/assistant/toolResult from pi-ai + pi-agent-core extensions;
assistant messages carry `api`, `provider`, `model`, `usage`, `stopReason`; images/thinking are
base64/signature content blocks — see `docs/session-format.md:43-100`), `thinking_level_change`,
`model_change {provider, modelId}`, `compaction {summary, firstKeptEntryId, tokensBefore, details?,
usage?, fromHook?}`, `branch_summary`, `custom {customType, data?}`, `custom_message {customType,
content, display}`, `label {targetId, label}`, `session_info {name}`. Written one
`JSON.stringify(entry)` per line via `appendFileSync`; migration v1→v2→v3 rewrites the whole file on
load (`session-manager.ts:230-260`, `_rewriteFile`). Malformed lines are skipped on read
(`parseSessionEntryLine`, `session-manager.ts:503-511`). Header discovery reads ≤1 MB with a
bounded scan (`SessionHeaderScanLimitError`, `session-manager.ts:491-610`).

auth.json (`core/auth-storage.ts`): one JSON object keyed by providerId, pretty-printed, created
with mode 0o600 inside a 0o700 dir (`auth-storage.ts:25,59`). Values are one of
`{type:"api_key", key?: string, env?: Record<string,string>}` or
`{type:"oauth", access: string, refresh: string, expires: number}`. Reads strip a UTF-8 BOM.
`key`/header values may be *config values* (`core/resolve-config-value.ts`): a string beginning `!`
runs a shell command (`!cmd...`); `${VAR}`/`$VAR` are env-var templates resolved against the
credential's private `env` map then `process.env` (`resolve-config-value.ts:80-113`). Concurrency is
a `proper-lockfile` lock next to the file (`.lock`), with stale-lock handling and a shared
process-wide read-state keyed on file revision (mtime/size via `getFileRevision`,
`auth-storage.ts:39,341`). The Rust import tool must parse this file and may echo the *shape* but
should never transcribe key material into logs.

Settings: global `~/.pi/agent/settings.json` merged (deep merge) with project `<cwd>/.pi/settings.json`
(`settings-manager.ts:227-233,345`). ~35 documented keys (`settings-manager.ts:106-140`):
`defaultProvider/defaultModel/defaultThinkingLevel/modelThinkingLevels`, `compaction {enabled,
reserveTokens, keepRecentTokens, modelOverrides}`, `branchSummary`, `retry {enabled, maxRetries,
baseDelayMs, maxAgentDelayMs, provider}`, `theme`, `quietStartup`, `defaultProjectTrust`,
`shellCommandPrefix`, `npmCommand`, `packages: (string|{source,...})[]`, `skills/prompts/themes/
extensions` (local resource lists), `defaultTools`, image/TUI flags. Corrupt files warn, don't crash
(`suite/regressions/7829`, `settings-diagnostics.test.ts`).

trust.json: `Record<canonical-abs-path, boolean|null>`; lookup walks up parent dirs
(`trust-manager.ts:44-54`); writes take a `trust.json.lock`; `null` means "stop ascending here".

## 4. Dependency edges

**(a) Workspace imports** (import counts across `src/`, `rg "from \"@earendil-works/..."`):

- `@earendil-works/pi-tui` — 78 imports; TUI widgets, editor, differential rendering
  (`packages/tui`, "Terminal User Interface library with differential rendering").
- `@earendil-works/pi-ai` — 74; `Model`, `Provider`, `Credential`/`CredentialStore`, streaming,
  uuidv7, OAuth (`packages/ai`, "Unified LLM API with automatic model discovery").
- `@earendil-works/pi-agent-core` — 57; `Agent`, `AgentMessage`, `AgentTool`, `ThinkingLevel`
  (`packages/agent`, "General-purpose agent with transport abstraction").
- `@earendil-works/chord` — 28, **only under `src/experimental/`** ("Application composition runtime
  for services, replicated state, RPC, and plugins").
- `@earendil-works/pi-protocol`, `pi-client`, `pi-server` — 8/7/5, only under `src/client/`,
  `src/cli/experimental/`, `src/experimental/` (devDependencies; the remote-session CBOR stack).
- `evals`, `session-backends`, `telemetry` are workspace siblings but not imported by coding-agent.

DAG shape: coding-agent sits on top of agent-core + ai + tui (+ protocol/client/server in
experimental only); no workspace package imports coding-agent except the evals/server dev tooling
(the one `pi-coding-agent` self-import is an internal `src/` reference).

**(b) Notable npm dependencies** (`package.json:51-70`): `chalk` (CLI colors), `diff` (edit-tool
diffs), `minimatch` + `ignore` (find tool glob/gitignore semantics), `cross-spawn` (Windows-safe
child processes), `proper-lockfile` (auth.json/models-store/trust.json locks), `semver`
(version checks), `undici` (HTTP dispatcher/proxy), `yaml` (frontmatter), `jiti` (extension TS
loading), `typebox` (tool schemas), `highlight.js` + `grok-mermaid` + `@silvia-odwyer/photon-node`
(WASM image resize) for rendering, `hosted-git-info` (git package sources). Bun-compiled binary
build via `src/bun/cli.ts`.

## 5. Test-suite inventory

Runner: vitest (`vitest.config.ts`) merging `~/git/vitest.base.ts`; `PI_OFFLINE=1` by default with
opt-in `test/test-network-env.ts` `allowNetwork()`; workspace source aliased to pi-ai/agent/tui
sources; 30 s timeout. ~185 test files: 103 top-level `test/*.test.ts`, 13 under `test/suite/`
(1 harness + 2 misc + 76 regression files named `<issue>-<slug>.test.ts`), plus non-test helpers
(`test-theme-colors.ts`, `streaming-render-debug.ts`, `model-runtime-test-utils.ts`,
`test-harness.ts`, `utilities.ts`, `test-network-env.ts`, `experimental-*.ts`, `rpc-example.ts`,
`sdk-codex-cache-probe-tool-loop.ts`).

Key groups (file → behavior):

- **Core data formats**: `config.test.ts` (package-dir detection incl. bun/dist layouts),
  `session-cwd.test.ts`, `session-file-invalid.test.ts`, `session-id-readonly.test.ts`,
  `suite/regressions/5996-session-name-newlines`, `7497-session-discovery-symlink`,
  `8337-utf8-bom-parsing` (BOM stripping), `session-info-modified-timestamp.test.ts`;
  `auth-storage.test.ts`, `settings-manager.test.ts` + `-bug.test.ts` (external-edit preservation),
  `settings-manager-compaction.test.ts`, `settings-diagnostics.test.ts`, `models-store.test.ts`
  (`FileModelsStore` locking), `trust-manager.test.ts`, `resolve-config-value.test.ts`,
  `config-value-migration.test.ts`.
- **AgentSession behavior**: `agent-session-*.test.ts` (auto-compaction queue, branching, compaction,
  concurrent prompt guard, dynamic provider/tool registration, retry, runtime events, stats, tree
  navigation), plus 76 `test/suite/regressions/*` files each pinning a numbered bug (compaction,
  tool ordering, RPC, extensions, selectors).
- **Compaction**: `compaction*.test.ts`, `branch-summarization*.test.ts`, `cache-stats.test.ts`.
- **Tools**: `tools.test.ts` (tool definitions), `builtin-tool-strict-mode.test.ts`,
  `edit-tool-legacy-input/no-full-redraw`, `powershell-tool.test.ts`, `file-mutation-queue.test.ts`,
  `tool-system-prompt-contributions.test.ts`, `suite/regressions/5303-bash-output-truncation`,
  `5208-late-bash-output`, `3302-find-path-glob`, `3303-find-nested-gitignore`.
- **Modes/protocol**: `rpc*.test.ts` (framing `rpc-jsonl.test.ts` matches `src/modes/rpc/jsonl.ts`
  strict-LF framing), `print-mode.test.ts`, `stdout-cleanliness.test.ts`, `export-html-*.test.ts`
  (incl. XSS sanitization), `export-jsonl-share.test.ts`, `session-share.test.ts`.
- **CLI/auth**: `args.test.ts` (full flag grammar), `auth-check.test.ts`, `credential-print.test.ts`,
  `package-command-paths.test.ts`, `package-manager*.test.ts`, `git-ssh-url.test.ts`,
  `version-check.test.ts`, `first-time-setup*.test.ts`.
- **UI components**: selectors (`model-selector`, `thinking-selector`, `session-selector-*`,
  `settings-selector`, `trust-selector`, `oauth-selector`, `tree-selector`, `scoped-models-selector`),
  message renderers (`assistant-message`, `user-message`, `custom-message`, `tool-execution-component`,
  `bash-execution-width`, `mermaid`), `keybindings*.test.ts`, theme tests
  (`theme-controller/detection/export/picker`, `scrollbar-theme`), `truncate-to-width.test.ts`,
  `ansi-utils.test.ts`.
- **Experimental**: ~20 `experimental-*.test.ts` covering server lifecycle, session workers,
  plugin reload, client TUI, slash-command facets.
- **Docs/examples as tests**: `documentation.test.ts`, `compaction-extensions-example.test.ts`,
  `plan-mode-extension.test.ts`, `git-merge-and-resolve-extension.test.ts`, `input-transform-streaming-example.test.ts`,
  `trigger-compact-extension.test.ts` — run example/extension code inline.

**Fixtures/mocks**: `test/suite/harness.ts` (225 lines) is the suite backbone: `createHarness()`
builds a temp dir, an in-memory `SessionManager.inMemory()`, `SettingsManager.inMemory()`,
`AuthStorage.inMemory()`, and a **faux LLM provider** from `@earendil-works/pi-ai/compat`
(`registerFauxProvider`, scripted `FauxResponseStep`s) — no network. Extensions are inline
`InlineExtension`s via `createTestExtensionsResult` (`test/utilities.ts`);
`model-runtime-test-utils.ts` builds registries against temp `models.json`. Clipboard and fs are
`vi.mock`ed per test (e.g. `interactive-tui.test.ts:27`).

**Porting hazards**:
- **Process spawning**: 22 test files spawn children (`bash-close-hang-windows.test.ts` keeps stdio
  handles open after shell exit; `rpc-client-process-exit`, `suite/regressions/6596-taskkill-enoent`,
  `5724-sigterm-signal-exit`, `8237-node-sea-extension-loading`). Rust: mirror `std::process` +
  signal semantics per OS.
- **TTY**: no pty use; TUI tests drive components directly and mock clipboard; `stdout-cleanliness`
  asserts non-interactive modes write nothing to stdout. Rust tests should target the same
  invariant without a pty harness.
- **Time**: 9 files use `vi.useFakeTimers` (compaction/retry/status tests); `version-check`,
  `session-info-modified-timestamp`, `7027-credential-refresh-hang` depend on real/faked clocks.
  Rust: inject a clock or use `tokio::time::pause`.
- **Files/env**: 99 files create temp dirs; `unstubEnvs: true` + `PI_OFFLINE=1` baseline means env
  mutation is per-test. Locking tests (`proper-lockfile` retry/ELOCKED semantics, stale 30 s lock
  recovery in `auth-storage.ts:120-155`) need a Rust file-lock strategy with equivalent behavior.
- **WASM/native**: photon-node image resize is `server.deps.external` (loaded as WASM at runtime);
  clipboard tests cover BMP conversion via native CLIs.

## 6. Porting flags — TS constructs needing a Rust-native answer

1. **ESM/TSX extension loading** — extensions are TS files loaded via `jiti`
   (`package.json:64`, `core/extensions/`); Rust needs a scripting seam (e.g. WASM plugins or an
   embedded JS runtime) or must re-scope extension discovery to data-only.
2. **Structured-clone + JSON round-trip** — session entries and settings are cloned with
   `structuredClone` (`settings-manager.ts:502`, `models-store.ts:32`); Rust wants
   `serde_json::Value` or typed structs with `#[serde(default)]` to absorb legacy/unknown fields.
3. **Discriminated-union typing** — `SessionEntry`, `RpcCommand`, `AgentSessionEvent` are
   TS tagged unions (`session-manager.ts:144-153`, `rpc-types.ts:20-75`); model as Rust enums with
   `serde(tag = "type")`, keeping unknown-tag tolerance (skip malformed lines, don't fail the file).
4. **`import.meta.url`-based install detection** — `config.ts:13-28` (bunfs, bundled-node flags);
   Rust replaces with build-time `env!("OUT_DIR")`-style constants.
5. **Global singletons & module state** — shared auth read-state (`auth-storage.ts:39`), models
   read-state, `commandResultCache` (`resolve-config-value.ts:10`); Rust needs explicit
   `OnceLock`/runtime-owned state, not ambient module globals.
6. **Sync-fs blocking helpers** — `appendFileSync`, `spawnProcessSync`, sync lock spin-wait
   (`auth-storage.ts:69-94`); Rust should pick async-with-blocking-section or dedicated IO threads.
7. **Node streams/readline** — RPC JSONL framing deliberately bypasses readline
   (`modes/rpc/jsonl.ts:14-20`): LF-only, strips one trailing `\r`, splits strictly on `\n`;
   session header reads use bounded `createReadStream` scans. Port with byte-oriented buffered
   readers, not line iterators that split on Unicode separators.
8. **AbortController/AbortSignal** — pervasive (`raceWithAbortSignal`, lock acquisition); map to
   `tokio_util::sync::CancellationToken`.
9. **EventEmitter-style extension event bus** — ~80 typed events (`src/index.ts:53-186`); decide
   trait-object handlers vs channel-based bus; `AgentSessionEvent` callback subscription
   (`harness.ts:196-199`) becomes a broadcast channel.
10. **Bun/Node dual runtime** — bun binary build, `process.versions.bun`, Windows quarantine
    cleanup (`main.ts:575-577`, `utils/windows-self-update.ts`) — not portable; Rust has a single
    native binary target, so the whole `install-lock`/self-update/shell-detection surface
    (`config.ts:34-360`, `utils/windows-self-update.ts`) needs redesign rather than translation.
11. **Regex/path quirks** — session-dir encoding and tilde expansion are hand-rolled string ops
    over both `/` and `\` (`config.ts:511-513`, `session-manager.ts:476-481`); port byte-for-byte
    to keep on-disk dirs identical, including Windows drive-colon handling.
12. **uuidv7 ids** — session ids are UUIDv7 from pi-ai (`session-manager.ts:2,208-210`); Rust
    `uuid` crate v7 feature for import compatibility.