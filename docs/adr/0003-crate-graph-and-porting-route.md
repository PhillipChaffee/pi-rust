# The workspace mirrors the upstream crate graph; the port routes for signal

pi-rust ports the pi TypeScript monorepo pinned at `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The workspace mirrors upstream one crate per package — published names as crate names (`crates/agent` holds `pi-agent-core`; the sqlite backend stays nested at `crates/session-backends/sqlite-node`) — and the crate DAG mirrors the TS runtime import graph with three deliberate adjustments. First, `pi-coding-agent` takes `pi-client`/`pi-protocol`/`pi-server` as plain runtime dependencies: upstream shipped them as devDependencies consumed by an esbuild bundle, and Rust has no bundler to hide behind. Second, telemetry crosses crates behind the open dispatch handle of [ADR 0004](0004-open-telemetry-seam.md). Third, `pi-evals` starts as a zero-dependency library leaf and grows dev-dependencies on `pi-ai` and `pi-coding-agent` only when the eval rig lands ([ADR 0005](0005-evals-re-scope.md)), mirroring upstream's devDeps. There is no shared-types crate: `pi-protocol`'s duplication of agent-side wire schemas is intentional upstream and is mirrored 1:1, with the 33-vector CBOR hex oracle as the drift guard. Porting route: telemetry (done) → ai → chord → agent → sqlite-node → protocol → client → server → tui → coding-agent.

## Considered options

- **Bottom-up order matching upstream's build order** (chord first): de-risks the hardest port sooner but delays every end-to-end signal; rejected because the frontier's parallelism already lets a second session take chord while ai ports.
- **A shared-types crate below agent and protocol**: would deduplicate wire schemas but breaks the mirror of upstream's intentional per-package type ownership; conformance vectors guard drift instead.
- **Cargo feature gates for the remote stack**: unused code dead-strips anyway, and features multiply the test matrix for no parity benefit.

## Consequences

- chord is ported before agent and protocol; ai before agent; coding-agent assembles everything last.
- The Rust-native extension mechanism is part of the destination; its API lives inside the chord, agent, and coding-agent port scopes rather than as a twelfth crate.
- The evals crate's contents are decided in [ADR 0005](0005-evals-re-scope.md).
- The TS→Rust import tool is a twelfth workspace crate, `pi-import`, the one crate with no upstream counterpart (decided by the import-tool ticket). The eleven-crate mirror above covers upstream ports only; the extension mechanism's API still lives inside the chord, agent, and coding-agent port scopes, not in a crate.
