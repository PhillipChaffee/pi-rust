# pi-agent-core

Rust port of upstream `packages/agent` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

The agent runtime of the port: a stateful [`Agent`](crate::types) driving LLM
turns, tool execution, and event streaming over the transport-abstracted
[`StreamFn`](crate::types::StreamFn) seam, plus the harness runtime with its
session storage, tools, and compaction. Message, model, and usage types come
from `pi-ai`; cancellation rides the same token pi-ai's transport options
carry.

Work in progress: the crate lands ticket by ticket on the map
([Port pi-agent-core](https://github.com/PhillipChaffee/pi-rust/issues/17));
this README tracks the crate, not the ticket order. Windows is out of scope
for this effort (map ticket "Decide the Rust stack").
