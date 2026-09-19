# pi-rust

Rust port of pi, the minimal terminal coding harness.

## Language

**Extension**:
User-written Rust code that extends a running Rust pi with tools, providers, commands, hooks, or UI. Distinguished from a TypeScript extension, which a Rust pi cannot execute.
_Avoid_: plugin (reserved for the upstream chord-facet plugin system)

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
