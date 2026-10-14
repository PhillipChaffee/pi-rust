# pi-telemetry

Rust port of upstream `packages/telemetry` in earendil-works/pi (MIT, (c)
2025 Mario Zechner), pinned at commit
`60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

The vendor-neutral callback-span telemetry contract. The contract is
callback-managed: a context hands a span to a body closure and settles the
span when that body's future completes — there is no `end()`. Reference
pieces ship beside the contract: a shared no-op context, a recording
implementation for tests, and span helpers.
