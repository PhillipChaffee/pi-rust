# Survey: pi `session-backends` package

Reference: `~/git/pi` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. Read-only survey; no upstream edits.

## 1. What the package is

`packages/session-backends` contains exactly one npm package: `@earendil-works/pi-session-backend-sqlite-node` (`sqlite-node/package.json`, v0.85.1, "Node sqlite session backend for @earendil-works/pi-agent-core sessions"). It is a **session storage backend**: a persistent, transactional implementation of the `SessionRepo` / `Storage` / `Session` contracts defined in `@earendil-works/pi-agent-core` (the workspace's `packages/agent`), backed by Node's built-in `node:sqlite` (`DatabaseSync`, engine `>=22.19.0`). It stores each coding-agent-style session — an entry tree with branches, scalar/list values, and a usage ledger — in SQLite database files, either one file per session (default) or many sessions in one shared container (`databasePath` option, `sqlite-node/src/sqlite/repo.ts:32`). It was renamed from `pi-storage-sqlite-node` in 0.84.0 and re-based onto the v4 lane-based `SessionRepo` contract (`sqlite-node/CHANGELOG.md`). It implements no writer ownership by design: the host (pi's server/worker lifecycle) owns writable authority, per `packages/agent/docs/work-packages/07-sqlite-host-ownership-live-forks.md`. In pi today `coding-agent` opens sessions through the `JsonlSessionRepo` from agent-core (`packages/coding-agent/src/experimental/mini/worker/run.ts:64`, `packages/agent/src/harness/session/jsonl/`); this package is the alternative SQLite backend for the same contract. It has no compile-time dependency on `protocol`/`client`/`server`; it sits at the storage layer beneath the RPC/server surface — the sessions it produces are what RPC mode and the server expose over the wire. Search is explicitly out of scope: "the package does not export a search service or FTS index; search is the separate S3 projection" (`sqlite-node/README.md:33`).

## 2. Public API surface

Entry point `sqlite-node/src/index.ts`; everything re-exported from `sqlite-node/src/sqlite/index.ts`:

- **`SqliteSessionRepo`** — `sqlite-node/src/sqlite/repo.ts:157`. `create/open/list/delete/fork/close` over `SqliteSessionMetadata` (`SessionMetadata` + container `path`). Options: `directory`, optional `databasePath` (shared container), `databaseFactory`, injectable `now` clock (`repo.ts:32`).
- **`SqliteStorage`** — `sqlite-node/src/sqlite/storage.ts:49`. Implements agent-core `Storage`: `commit` (serialized per-instance commit queue, `storage.ts:67`), `getEntries`, `getValue/scanValues`, `readList`, `scanBranch/scanBranchStructure/scanEntries/scanUsage`, `getStats`, `snapshot`, `close`.
- **`SqliteOpenSession`** — `sqlite-node/src/sqlite/session.ts:24`. `Session<SqliteSessionMetadata>` facade wrapping `StorageBackedSession` (from agent-core); tracks in-flight operations so `close()` waits for them (`session.ts:174`).
- **SQLite adapter for `node:sqlite`** — `createNodeSqliteFactory()` / `wrapNodeSqliteDatabase()`, `sqlite-node/src/index.ts:102-120`: three open modes (create, no-create rw via file-URL `?mode=rw`, read-only) behind `SqliteDatabaseFactory` (`sqlite-node/src/sqlite/types.ts:24`); sync `transaction()` wrapper that rejects async callbacks (`index.ts:78`).
- **`sql` template tag + `joinSqlFragments`** — `sqlite-node/src/sqlite/sql.ts:38,56`: parameterized query builder (`SqlQuery`).
- **`applyInitialSchema`** — `sqlite-node/src/sqlite/migrations.ts:5`: executes `migrations/001_initial.sql`.
- Row/decode helpers under `sqlite-node/src/sqlite/session/`: `session-row.ts`, `entries.ts` (`EntryRowWriter`, `decodeEntryRow`), `values.ts` (incl. `nextPrefixBoundary` prefix scan, `values.ts:90`), `usage-ledger.ts`, `branch-entries.ts` (branch segment machinery, `branch-entries.ts:166,314`), `session-sequences.ts`, `session-stats.ts`.
- Constants: `SQLITE_STORAGE_VERSION = 1`, `SQLITE_SESSION_EXTENSION = ".sqlite"` (`repo.ts:24-25`).

## 3. Data formats (import-tool relevance)

On-disk format: **SQLite container files** (WAL mode, `busy_timeout=5000`, `repo.ts:64-70`). Default layout: `{directory}/{sessionId}.sqlite`; IDs matching `[A-Za-z0-9_-]+` keep the raw name, anything else is `~` + base64url of the ID's **UTF-16 code units** + `.sqlite` (`repo.ts:40-44` — the durable ID itself is unchanged; `list`/`open` return the canonical physical path). Deletion also removes `-wal`/`-shm` sidecars (`repo.ts:58`). Optional shared container: all sessions in one file, rows scoped by `session_id`.

Schema (storageVersion 1 = "AgentHarness storage format 4", `migrations/001_initial.sql`; every table `WITHOUT ROWID`, keyed by `session_id`):

- `sessions`: `id`, `created_at` (epoch-ms INTEGER), `parent_session_id`, `storage_version`, `metadata` (TEXT, currently always NULL — reserved), `message_count`, `usage_payload` (JSON `Usage`: token counts + cost, `session-row.ts:72`), `next_seq`.
- `entries`: PK `(session_id, id)`; `parent_id`, `seq`, `type` ∈ `message|compaction|branch_summary|custom`, `custom_type` (required for custom, `entries.ts:118`), `timestamp`, `payload` = JSON of the entry minus `id/parentId/seq/timestamp/type/customType` (`entries.ts:28-63`): message → `{message, terminate?}`; compaction → `{summary, retainedTail, tokensBefore, details?, usage?, fromHook}`; branch_summary → `{fromId, summary, details?, usage?, fromHook}`; custom → `{data?}`. Indexes on parent and `(seq, type)`.
- `scalar_values`: PK `(session_id, namespace, key)`; `seq`, `value` JSON. Upsert semantics.
- `list_values`: PK `(session_id, namespace, key, seq)`; `value` JSON (append-only).
- `usage_ledger`: PK `(session_id, id)`; `seq`, `entry_id?`, `adjustment` (0/1), `usage` JSON, `details` JSON?. Entry and usage IDs share one namespace, enforced by insert triggers (`001_initial.sql:69-91`).
- `branch_entries` / `branch_meta`: private branch-index projections (not part of the portable value surface — "no equivalent in the other backends", `001_initial.sql:93`); segments carry base-branch linkage with compaction-based bases; unique index on tip.

Version gate: a row with `storage_version` newer than the code refuses to open; older refuses pending migrations (`session-row.ts:57`). Forks never copy `usage_ledger` rows (`repo.test.ts:552`). No credentials or key material are stored anywhere in this format. The import tool must reproduce: the filename encoding, the JSON payload shapes, epoch-ms timestamps, and the version gate; `branch_*` tables are reproducible projections that a Rust port can rebuild.

## 4. Dependency edges

**(a) Workspace (crate-DAG) imports** — `sqlite-node/package.json` `dependencies`:
- `@earendil-works/pi-agent-core` = `packages/agent`: interfaces (`Storage`, `Session`, `SessionRepo`, `Entry`, `Value`/`ValueList`, `ForkOptions`, ...), `StorageBackedSession`, `prepareStorageCommit`, `branchTip`, `createForkSnapshot`, `BACKGROUND_CONTEXT`, and the conformance harness `@earendil-works/pi-agent-core/harness/session/testing` (= `agent/src/harness/session/testing/`).
- `@earendil-works/pi-ai` = `packages/ai`: `uuidv7` (`ai/src/index.ts:47`) and the `Usage` type.
- Reverse edges: **none**. No other workspace package's `package.json` imports `pi-session-backend`; the only workspace references are the root `tsconfig.json`, `package-lock.json`, and `scripts/local-release.mjs`. It is a leaf of the workspace DAG, published standalone. The vitest configs also alias `pi-telemetry`, but only transitively via agent-core.

**(b) External npm dependencies**: none at runtime. devDependencies: `vitest` 4.1.9 + `@vitest/coverage-v8` 4.1.9. Everything else is Node builtins: `node:sqlite`, `node:fs/promises`, `node:path`, `node:url` (`src/`), plus `node:os`/`node:tmpdir` in tests.

## 5. Test-suite inventory

Vitest, `sqlite-node/vitest.config.ts` (v8 coverage on `src/**`, source-alias resolution of workspace packages). Six files, 49 hand-written cases plus two suite-driven conformance registrations. All run against real SQLite (`:memory:` or `mkdtemp` temp dirs); no network, no snapshot fixtures.

- **`test/adapter.test.ts`** (5): `createNodeSqliteFactory` semantics — `openExisting`/`openReadOnly` never create missing files; read-only connections reject writes; sync transaction commits and returns its value; positional + named parameters forwarded; async transaction callbacks rejected with rollback. Porting: the file-URL `?mode=rw` trick (`index.ts:112`) and the named-param detection heuristic (`index.ts:7`) both disappear with rusqlite's typed API.
- **`test/sql.test.ts`** (2): `sql` template composition without renumbering parameters; `joinSqlFragments` filter composition; parameterized `exec/run/get/all`.
- **`test/storage.test.ts`** (20): single-transaction-per-commit (via `TransactionCountingDatabase` wrapper); branch-index maintenance for root/append/divergent/compaction-based segments; `getEntries` preserves requested id order; entry scans with filters/seq bounds; branch scans across materialized segments with stop boundaries applied before filtering; structure scans without payloads; usage scans; `prepareStorageCommit` sequence/timestamp assignment; `next_seq` read/advance; stats maintenance incl. historical totals after reopen; scalar get/prefix scan including a `U+FFFF` boundary key; **`EXPLAIN QUERY PLAN` assertions** that branch and list queries use covering indexes/PK and never a temp B-tree (`storage.test.ts:71,679`). Porting hazard: query-plan text is SQLite-version-sensitive.
- **`test/storage-conformance.test.ts`**: registers agent-core's shared `createStorageConformance` suite against a `:memory:` `SqliteStorage` fixture (`StorageFixture` with `Symbol.asyncDispose`).
- **`test/repo.test.ts`** (22, incl. a per-file/shared-layout parameterized fork case): create/list/open lifecycle and metadata shape; duplicate create/open rejection; `writer_lease` legacy table ignored; corrupt/newer-version files skipped by best-effort `list`; delete failures; repo close closing all sessions and rejecting later ops; shared-container isolation and selective deletion; WAL/SHM sidecar removal; unsafe-ID filename encoding and round-trip (unicode, slashes, `..`, `%`); foreign-source fork path identity and "outside this repository" rejection; **live-source fork snapshot coherence** — a writer commits after the reader's snapshot boundary via `SnapshotBoundaryFactory`/`SnapshotBoundaryStatement` statement interception and deferred commits (`repo.test.ts:71-148,210-234`); deletion reservation against concurrent create/open/fork via `GatedOpenExistingFactory` deferred gates; close-error aggregation (`AggregateError` with one-vs-many shape) via `CloseTrackingFactory`.
- **`test/repo-conformance.test.ts`**: registers agent-core's shared `createSessionRepoConformance` for both per-file and shared-container repo layouts.

Fixtures/mocks are hand-rolled wrapper classes implementing the structural `SqliteDatabase`/`SqliteStatement` interfaces — no mocking library. Porting hazards: (1) concurrency — the deferred-promise gates and snapshot-boundary statement hooks encode subtle WAL snapshot timing that a Rust port must reproduce at the trait seam; (2) `realpath`-based identity checks (case/alias sensitivity); (3) time is injected (`now`), so no wall-clock flakiness; (4) no network anywhere. Benchmark harnesses exist under `benchmark/session/` (`vitest bench`) but reuse the agent package's benchmark driver — they are not part of the test gate.

## 6. Porting flags (TypeScript constructs needing a Rust-native answer)

1. **`node:sqlite` `DatabaseSync`** — synchronous API with create / no-create-rw (file-URL `?mode=rw`) / read-only opens (`index.ts:106-120`). Rust: `rusqlite` `OpenFlags` map directly; the URL trick and named-param sniffing go away.
2. **Structural interfaces** `SqliteDatabase`/`SqliteStatement`/`SqliteFactory` (`types.ts`) with hand-rolled test doubles → Rust traits; keep the trait seam because several tests depend on wrapping/injecting connections.
3. **`sql` tagged-template builder + `joinSqlFragments`** (`sql.ts`) → a Rust query builder or `const`-composed statements; must preserve parameter ordering.
4. **`Buffer.from(id, "utf16le").toString("base64url")`** filename encoding (`repo.ts:42`) — Rust `String` is UTF-8; the UTF-16 code-unit encoding must be reproduced exactly for on-disk name compatibility.
5. **Promise-chained `commitQueue`** serializing commits/snapshots per storage (`storage.ts:55,67,132`) → `Mutex`/serialized connection or an actor; the snapshot-must-queue-behind-commits guarantee is asserted by tests.
6. **`SqliteOpenSession.admitted` set**: close waits for all in-flight async operations (`session.ts:174-184`) → `JoinSet`/generation counter; close must be idempotent and single-flight (`closePromise`).
7. **`AggregateError`** one-vs-many close-error aggregation (`repo.ts:399-404`) → custom aggregate error type.
8. **`Promise.allSettled`** + best-effort error swallowing (list discovery, `repo.ts:277`) → explicit `Result` accumulation.
9. **Entry payload structural `Omit`/conditional spread** (`entries.ts:23-63`) → serde `flatten`/`skip_serializing_if`; payload is JSON in a TEXT column, so serde round-trip fidelity matters for the import tool.
10. **`EXPLAIN QUERY PLAN` assertions** (`storage.test.ts:64-76`) — keep the intent (index usage, no temp B-tree) but expect plan-text drift across SQLite versions.
11. **`uuidv7` from `@earendil-works/pi-ai`** → `uuid` crate v7 feature (or port `ai/src/utils/uuid.ts`).
12. **Injectable `now: () => number`** → injectable clock trait; epoch-ms integers.
13. **`Symbol.asyncDispose`** fixtures/conformance → `Drop` (or explicit teardown fns).
14. **Deferred promise gates in tests** (`deferred()`, `repo.test.ts:142`) → `tokio::sync::oneshot`/`Notify`; the live-fork snapshot tests are the hardest to port faithfully.
15. **`Math.max(0, limit)` clamping, `NoInfer<T>`, `Omit<...>` row typing** — trivial or subsumed by Rust types; `Number(result.changes)` BigInt coercion disappears with rusqlite.
16. **Path identity via `realpath`** (`repo.ts:426-435`) → `std::fs::canonicalize`; note Windows/normalization differences if the import tool must match stored paths.