# pi-coding-agent

Rust port of upstream `packages/coding-agent` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

The product layer of the port: config paths, the session manager, built-in
tools, the CLI, and the interactive TUI, assembling pi-ai, pi-agent-core,
pi-tui, chord, pi-client, pi-protocol, and pi-server as plain runtime
dependencies ([ADR 0003](../../docs/adr/0003-crate-graph-and-porting-route.md)).

Work in progress: the crate lands ticket by ticket on the map
([Port pi-coding-agent](https://github.com/PhillipChaffee/pi-rust/issues/23));
this README tracks the crate, not the ticket order. Windows is out of scope
for this effort (map ticket "Decide the Rust stack").
