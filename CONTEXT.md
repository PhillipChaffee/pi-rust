# pi-rust

Rust port of pi, the minimal terminal coding harness.

## Language

**Extension**:
User-written Rust code that extends a running Rust pi with tools, providers, commands, hooks, or UI. Distinguished from a TypeScript extension, which a Rust pi cannot execute.
_Avoid_: plugin (reserved for the upstream chord-facet plugin system)

**Extension receipt**:
The hash-bound record that gates spawning an installed extension — artifact checksum, source inputs, and protocol version, recorded at enable time and re-verified fail-closed at every start. A Rust extension is a reviewed binary, not readable source; the receipt is what makes the review stick.

**Extension host**:
The pi process's half of an extension conversation: it verifies the receipt, spawns the child, binds the child's declarations into the in-process extension record, and proxies every dispatch. The child-side half is the extension SDK, this crate's `extensions::host` module.
_Avoid_: plugin host, extension server

**Extension wire**:
The versioned CBOR-over-stdio conversation between the host and one extension child: framed envelopes, the handshake, registrations, call scopes, and context calls ([ADR 0010](docs/adr/0010-extension-host-wire-protocol.md)).
_Avoid_: extension protocol session, IPC

**Parity**:
Behavioral equivalence with upstream pi at the pinned commit. Full parity includes the Rust-native extension mechanism; it never includes executing TS/JS extension files.

**Import tool**:
The `pi-import` binary that migrates TS-pi sessions, settings, and provider credentials into the Rust pi's formats. It reads TS-pi artifacts at the pinned commit and writes the Rust formats as the port tickets define them; it never executes `!cmd` indirections and never prints key material. TS extensions are reported as skipped items — a Rust pi cannot execute them.

**Eval rig**:
The Rust-native harness, judges, and runner that re-express upstream evals over the Rust pi. Not a 1:1 port: upstream's vitest-evals machinery is a JavaScript framework contract with no Rust equivalent.
_Avoid_: eval framework

**Docs-lift experiment**:
The paired A/B measurement of how much pi's bundled documentation improves the model's pass rate on eval tasks, run in fresh read-only containers against live model credentials.

**Lift**:
The measured pass-rate gain of an experimental arm over its paired control arm.

**Blocked pair**:
A treatment/control pair that cannot be compared because one side's observation is missing or invalid; reports fail closed and record the reason rather than guessing.
