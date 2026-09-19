# TUI renderer: port pi-tui's line-diff renderer, not ratatui

The tui crate ports pi-tui's renderer 1:1 — stateful components emitting rows of
pre-styled ANSI strings, whole-line diffing, synchronized output (`CSI 2026`), a
`BoundedTerminalWriter`, alt-screen layout with cached leaf lines, and a
main-screen mode that writes into scrollback — on top of crossterm for raw mode,
terminal size, and terminal state only. ratatui was rejected: its immediate-mode
cell buffer replaces the line-diff model and would rework main-screen scrollback
semantics, which are the package's defining behavior. The hand-rolled input
parser (Kitty keyboard protocol negotiation, bracketed paste, 50/100 ms lone-ESC
latency, split-sequence reassembly) also ports 1:1; crossterm's event enum does
not expose that behavior surface.

## Consequences

Crossterm's role is thin plumbing; every visible byte flows through the ported
writer, so a later substrate swap stays contained. Windows is out of scope for
this effort ([#1](https://github.com/PhillipChaffee/pi-rust/issues/1)), which
trims crossterm's remaining role to macOS/Linux convenience; Windows console
input does not deliver a POSIX byte stream, so a future Windows port would need
its own input adapter.

Decided in [#10](https://github.com/PhillipChaffee/pi-rust/issues/10) against
evidence on the `research/survey-tui` branch (upstream pin
`60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).
