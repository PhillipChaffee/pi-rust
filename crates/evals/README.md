# pi-evals

Rust port of upstream `packages/evals` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

Pure experiment logic for pi behavioral evals: case planning, paired
comparison, and report reading. Upstream evals is a private
harness-and-runner package; only its pure experiment logic is portable, and
only that ports here (per [ADR 0005](../../docs/adr/0005-evals-re-scope.md)).
