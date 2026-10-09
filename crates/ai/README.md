# pi-ai

Rust port of upstream `packages/ai` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

The unified LLM layer: one `Context`/`Message`/`Tool`/`AssistantMessageEvent`
model shared by every wire-protocol implementation, with the data shapes of
the provider catalogs, credential resolution, retry/overflow handling, and
image generation. Wire-protocol implementations, OAuth flows, the provider
registry, and the Models runtime land with their own tickets.
