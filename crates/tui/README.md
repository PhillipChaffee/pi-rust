# pi-tui

Rust port of upstream `packages/tui` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

Terminal UI library with differential rendering (per [ADR
0001](../../docs/adr/0001-tui-renderer-port-not-ratatui.md)): stateful
components render rows of pre-styled ANSI strings, the renderer diffs whole
lines and writes only what changed, and crossterm stays a thin input and
raw-mode backend.
