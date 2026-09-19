# The evals re-scope: pure logic ports 1:1; the rig is re-expressed on mira-eval

`pi-evals` starts as a zero-dependency library crate porting upstream's pure experiment logic (plan, comparison, report) and its 21 portable test cases 1:1, inlining the minimal vitest-evals-core report-reader slice those tests pin. The 32 Node-welded test cases are not ported — they exercise the JavaScript vitest-evals contract, which has no Rust substrate — and their substance re-enters through a Rust-native eval rig declared as destination scope: harness adapters over the Rust `AgentSession`, judges, the Docker docs-lift A/B runner (sandbox protocol ported; the treatment becomes which files exist in the installed Rust distribution), and the eval suites re-expressed outside the acceptance gate. The rig is fogged until the coding-agent port exposes `AgentSession`, `ModelRuntime` construction, the faux provider, and the system-prompt builder; its substrate is mira-eval (MIT), chosen because an ecosystem survey found no maintained, adoption-proven Rust eval framework, with the rig's first ticket piloting it against an in-process tokio session and a hand-rolled Harness/judge substrate as the recorded fallback.

## Considered options

- **Drop evals from the port**: loses portable pure logic and leaves the Rust pi without behavioral-eval capability; rejected.
- **Port the vitest-evals adapter 1:1**: impossible — the contract is a JavaScript framework interface with no Rust substrate.
- **Hand-roll the rig on `cargo test` with no external crate**: the ecosystem baseline (OpenAI Codex, rig, and Swiftide ship no eval framework) and the recorded fallback; mira-eval covers the same surface off the shelf and was preferred, with the pilot as the evidence gate.

## Consequences

- `pi-evals` stays a zero-workspace-dep leaf until the rig lands, then grows dev-dependencies on `pi-ai` and `pi-coding-agent` mirroring upstream devDeps ([ADR 0003](0003-crate-graph-and-porting-route.md)'s static dev-dep line is amended on that point).
- The docs-lift experiment's treatment re-defines: which files exist in the installed Rust distribution, not in npm tarballs.
- The eval suites (smoke, documentation-audit, docs-lift) sit outside the vitest-suite acceptance gate: they are model-in-the-loop experiments needing live credentials and Docker, not porting gates.
