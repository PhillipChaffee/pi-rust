# Survey: pi `tui` package (pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759)

Reference: `~/git/pi/packages/tui` (`@earendil-works/pi-tui` 0.85.1). Ticket: PhillipChaffee/pi-rust#5.

## 1. What the package is

`packages/tui` is pi's standalone terminal-UI library: a retained, stateful component
tree whose components each render to plain ANSI-string lines, plus two renderer
backends (main-screen, alt-screen) that diff those lines against the previous frame
and write only what changed. It also owns terminal I/O concerns (raw mode, Kitty
keyboard protocol negotiation, bracketed paste, mouse, images, clipboard) and a
library of built-in components (`Text`, `Editor`, `Markdown`, `SelectList`, layout
stacks, scroll views, overlays). It is a self-contained leaf package: everything
downstream (the `coding-agent` package) builds on it; it depends on nothing else in
the workspace. ~38k lines TS across `src/` (41 files) and `test/`.

## 2. Public API surface

Single barrel: `src/index.ts` (~40 exported symbols, re-exports only).

- Core framework — `src/tui.ts` (1,456 L): `TUI`/`TuiBase` (abstract base),
  `Component`/`Container`/`Focusable` interfaces, overlay stack + `OverlayHandle`,
  `TuiMouseEvent` types + dispatch helpers, `compositeTuiLine`, `CURSOR_MARKER`,
  `ViewportTUI` capability + `isViewportTUI`.
- Renderers — `src/tui-main-screen.ts` (655 L): `TuiMainScreen`; `src/tui-alt-screen.ts`
  (1,727 L): `TuiAltScreen`, alt-screen options incl. search panel
  (`src/alt-screen-search.ts`).
- Terminal abstraction — `src/terminal.ts` (547 L): `Terminal` interface +
  `ProcessTerminal`; `src/stdin-buffer.ts` (444 L): escape-sequence reassembly +
  paste batch splitting; `src/terminal-colors.ts`, `src/terminal-image.ts` (696 L).
- Input — `src/keys.ts` (1,401 L): `parseKey`/`matchesKey`/`Key`; `src/keybindings.ts`
  (320 L): action registry (`TUI_KEYBINDINGS`, `getKeybindings`/`setKeybindings`,
  `KeybindingsManager`).
- Layout — internal engine `src/layout.ts` (449 L) + `src/layout-node.ts`; public
  components `src/components/{v-stack,h-stack,scroll-view,stack}.ts`.
- Components (`src/components/`): box, text, truncated-text, input, editor,
  editor-component, markdown, loader, cancellable-loader, select-list,
  settings-list, mouse-region, spacer, image, alt-screen-flash, plus stacks and
  scroll-view.
- Editor support machinery (internal, not exported): `kill-ring.ts`, `undo-stack.ts`,
  `word-navigation.ts`; autocomplete is public via `src/autocomplete.ts` (826 L).
- Utilities — `src/utils.ts` (1,337 L): `visibleWidth`, `truncateToWidth`,
  `wrapTextWithAnsi`, `sliceByColumn`, OSC-8 link extraction, grapheme segmentation.
- Extras: `src/fuzzy.ts`, `src/latex.ts` (1,394 L), `src/native-platform.ts` /
  `src/native-modifiers.ts` / `src/native-module-path.ts` (N-API addon loading).

## 3. The differential-rendering model

Model: **retained component tree + line-string diff**. Each frame, the renderer
calls `component.render(width)` top-down, producing `string[]` where each element is
one terminal row of pre-styled ANSI text. The renderer keeps the previous frame's
lines and diffs by whole-line equality:

- `TuiMainScreen` (`tui-main-screen.ts:247` doRender): three strategies — first
  render writes all lines into scrollback; width change or any change above the
  viewport triggers clear-screen + full re-render (`fullRedrawCount`); normal
  update moves the cursor to the first changed line, clears to end-of-screen, and
  rewrites changed lines. Kitty image placements in overwritten regions are
  explicitly deleted. Output goes through `BoundedTerminalWriter` (1 MiB chunks,
  UTF-16 surrogate-safe splitting).
- `TuiAltScreen` (`tui-alt-screen.ts:1658` doRender): builds a per-frame internal
  `LayoutFrame` (rebuild geometry every render, reuse cached leaf lines by
  reference), paints exactly `terminal.rows` rows, composites overlays and
  selection, then diffs `screen[row] === previousScreen[row]` and rewrites only
  changed rows in place.
- Both wrap frames in synchronized output (`CSI 2026 h/l`) for atomic flicker-free
  updates (`tui.ts:384` SEGMENT_RESET; `tui-alt-screen.ts:70-71`).

Scheduling (`tui.ts:944-1004`): `requestRender()` coalesces onto a 16 ms
min-interval timer; keyboard input takes an immediate `process.nextTick` path to
skip the throttle (`tui.ts:1078-1081`).

**vs ratatui**: ratatui is immediate-mode into a cell `Buffer` — widgets draw
styled cells, diffs are cell-by-cell, and components are stateless draw functions,
not retained objects. pi-tui's diff is whole-ANSI-line equality, its components are
stateful with their own caches, and it ships a main-screen mode that preserves
terminal scrollback (ratatui's inline viewport only partly covers this) plus
application-owned alt-screen scrolling, IME cursor-marker positioning, and Kitty
image management — none of which ratatui provides natively. Porting pi's own
renderer over crossterm would preserve behavior most faithfully; ratatui would
replace the line-diff model with cells and rework main-screen scrollback semantics.
Input in pi is hand-rolled (not a ratatui/crossterm-style event enum).

## 4. Input handling and keybindings

- Capture: `ProcessTerminal` (`terminal.ts`) sets raw mode, negotiates the Kitty
  keyboard protocol (`CSI > 7 u` query, flags 7, `terminal.ts:14`), enables
  bracketed paste, and feeds bytes through `StdinBuffer`, which reassembles
  split escape sequences with a 50 ms timeout and splits paste batches
  (`stdin-buffer.ts`). Lone-ESC latency window: 10 ms local, 100 ms over SSH,
  overridable via `PI_TUI_ESC_TIMEOUT` (`terminal.ts:115-130`).
- Dispatch: `TuiBase.handleTerminalInput` (`tui.ts:1006`) consumes terminal
  replies first (OSC 11 background, CSI 996/997 color scheme, CSI 16 t cell size),
  then runs registered input listeners (chainable, may rewrite/consume), applies
  the global debug key (Shift+Ctrl+D), fixes overlay focus, and finally delivers
  to the focused component's `handleInput(data)`. Key-release events are filtered
  unless `wantsKeyRelease` (Kitty protocol).
- Parsing: `keys.ts` decodes legacy CSI/SS3 sequences and Kitty `CSI u` events;
  `matchesKey(data, Key.ctrl("c"))` compares against string `KeyId`s
  (`"ctrl+shift+p"`).
- Keybindings: `keybindings.ts` is a global action registry keyed by action IDs
  (`tui.editor.cursorUp`, `tui.altScreen.pageUp`, …) with defaults in
  `TUI_KEYBINDINGS`, downstream extension via TS declaration merging, and
  conflict detection. Alt-screen navigation (page/top/bottom/search/prompts) and
  editor/input/select actions all route through it.
- Mouse: SGR mouse sequences normalized into `TuiMouseEvent`
  (press/release/move/drag/click/wheel, component-local coords); alt-screen
  hit-tests the committed layout frame and supports capture, focus, render
  control, selection with OSC 52 copy, scrollbar drags, and OSC 8 hyperlink
  clicks. Main-screen mode does not capture mouse.
- Native helpers: per-platform N-API prebuilds (`native/{darwin,linux,win32}`)
  provide clipboard access and physical-modifier state (native Shift+Enter on
  Apple Terminal / Windows).

## 5. Dependency edges

(a) Workspace: **none**. `src/` imports only `node:*` builtins and npm packages —
`tui` is the leaf at the bottom of the pi crate DAG (everything else may depend on
it; it depends on nothing else under `packages/`). Self-references are only
`Symbol.for("@earendil-works/pi-tui/viewport")` and the package name in
`native-module-path.ts`.

(b) External (runtime): `marked` 18.0.11 (markdown AST for the Markdown
component, `components/markdown.ts:1`); `get-east-asian-width` 1.6.0 (CJK width
in `utils.ts:1`). Dev: `@xterm/headless` 5.5.0 (VirtualTerminal test harness),
`chalk` 6 (test themes). Native `.node` prebuilds for darwin/linux/win32 are
committed under `native/` (clipboard + modifier queries). `terminal-image.ts`
shells out via `execSync` to probe terminal image support. Engines: Node >= 22.19
with strip-only TS (`*.ts` imports, no parameter properties — deliberate).

## 6. Test-suite inventory

Runner is **`node --test`** (`package.json:13`), not vitest: 37 `*.test.ts` files
plus helpers. Harness style: `VirtualTerminal` (`test/virtual-terminal.ts`) wraps
`@xterm/headless` for real terminal emulation — `sendInput`, `resize`,
`getViewport`/`getScrollBuffer`, `waitForRender` (nextTick + 20 ms + flush, matching
the 16 ms throttle). Shared fixture themes in `test/test-themes.ts`.

- Renderers: `tui-render.test.ts` (952 L — first render, diff/append/shrink,
  Kitty image deletion, 1 MiB bounded writer), `tui-alt-screen.test.ts` (1,938 L
  — viewport scroll, selection, hyperlinks, scrollbar, images, stop-document),
  `tui-shrink.test.ts`, `tui-cell-size-input.test.ts`, `viewport-overwrite-repro.ts`
  (manual repro script, not a test).
- Layout/scroll: `layout.test.ts` (367 L — stack allocator, grow/shrink/min/max),
  mouse routing in `mouse-components.test.ts` (249 L).
- Components: `editor.test.ts` (4,165 L — the giant: editing, autocomplete,
  paste markers, history), `editor-history-keybindings.test.ts`,
  `input.test.ts` (664 L), `markdown.test.ts` (1,760 L), `select-list.test.ts`,
  `settings-list.test.ts`, `truncated-text.test.ts`, `word-navigation.test.ts`.
- Input/keys: `keys.test.ts` (633 L), `keybindings.test.ts`,
  `stdin-buffer.test.ts` (526 L — partial sequences, paste splits),
  `terminal.test.ts` (300 L — escape timeouts, ProcessTerminal).
- Overlays: `overlay-options.test.ts` (541 L), `overlay-non-capturing.test.ts`
  (1,203 L), `overlay-short-content.test.ts`, `tui-overlay-style-leak.test.ts`.
- Utilities/i18n: `truncate-to-width.test.ts`, `wrap-ansi.test.ts`,
  `tab-width.test.ts`, `fuzzy.test.ts`, `terminal-colors.test.ts` (252 L),
  `regression-overlay-cjk-boundary.test.ts`,
  `regression-regional-indicator-width.test.ts` (emoji flags),
  `bug-regression-isimageline-startswith-bug.test.ts`.
- Images/latex/autocomplete: `terminal-image.test.ts` (700 L), `latex.test.ts`
  (505 L), `autocomplete.test.ts` (578 L), `fuzzy.test.ts`.
- Native/platform: `native-module-path.test.ts`, `native-platform.test.ts`
  (ELF page-size binary assertions; darwin/win32-gated clipboard tests, win32
  write test opt-in via `PI_TEST_NATIVE_CLIPBOARD=1`),
  `native-clipboard-linux.test.ts` (247 L — spawns Xvfb + xclip, needs `cc`,
  `pkg-config xcb`; uses `test/fixtures/*.c|cjs` clipboard helpers).
- Regression: `regression-sigwinch-kill-eacces.test.ts` (process.kill EACCES).
- Manual/demo scripts (excluded from `*.test.ts` glob): `chat-simple.ts`,
  `key-tester.ts`, `image-test.ts`, `render-churn-bench.ts`,
  `alt-screen-large-transcript-bench.ts`.

TTY/porting hazards: the Linux clipboard suite needs Xvfb/xclip and is
platform-gated; native tests assert on committed binary prebuilds (ELF header
parsing); SIGWINCH test touches signal behavior; everything else is TTY-free via
`VirtualTerminal`. A Rust port needs an equivalent headless terminal emulator for
the viewport/scrollback golden assertions (e.g., `vte`-based, or termwiz).
Capability detection tests manipulate env vars (`TERM_PROGRAM`, `KITTY_*`).

## 7. Relationship to `tui-plan.md`

`tui-plan.md` (repo root, 1,001 L) is the design handoff for the **alternate-screen
constrained layout system**: `VStack`/`HStack`/`ScrollView` primitives with
`basis`/`grow`/`shrink`/`minSize`/`maxSize` entries, an internal per-frame layout
tree (`layout.ts` — suggested there by name), the `ViewportTUI` capability +
`setLayoutRoot`, a legacy implicit-root path for plain `addChild()` users, and the
coding-agent transcript+dock composition in `interactive-mode.ts`. The current
package shows the plan **fully implemented**: `layout.ts`/`layout-node.ts` and the
three layout components exist; `TuiAltScreen` imports `renderLayoutFrame`,
`getLayoutBoxesAt`, scrollbar geometry (`tui-alt-screen.ts:12-20`); `VIEWPORT_TUI`
symbol and `isViewportTUI` are shipped (`tui.ts:454-463`); README documents
`setLayoutRoot`; dedicated `layout.test.ts` exists. The plan's test matrix
(cursor-marker survival under clipping, scroll chaining, ANSI no-leak) matches the
regression tests present. Treat the plan as the record of intended scope — the
port should honor its core decisions (main-screen stays scrollback-mode; layout is
alt-screen-only; rebuild geometry, reuse leaf caches; layout internals stay private).

## 8. Porting flags — TS constructs needing a Rust-native answer

1. **Class inheritance + optional methods**: abstract `TuiBase`, `Container`
   subclasses, `Component` with optional `handleInput?/handleMouse?/invalidate?`
   and duck-typed `Focusable` (`isFocusable` checks `"focused" in component`) —
   Rust: traits with default methods + `Any` downcasting for focus/mouse
   dispatch; `isViewportTUI` symbol check (`Symbol.for`) → marker trait.
2. **Declaration-merging keybindings registry** (`interface Keybindings` extended
   by downstream packages) → runtime registry with string action IDs + builder
   config, since Rust has no declaration merging.
3. **Global mutable state**: `_kittyProtocolActive` (`keys.ts:25`),
   `TUI_KEYBINDINGS`, capability caches (`terminal-image.ts`) → `RwLock`/`Arc` or
   owned state on the terminal/renderer.
4. **Event-driven scheduling**: `process.nextTick` immediate-render preemption +
   16 ms `setTimeout` throttle (`tui.ts:952-1004`), `NodeJS.Timeout`,
   `EventEmitter` (`StdinBuffer`) → tokio/async task + channel design; the
   input-preempts-throttle behavior is load-bearing (documented at
   `tui.ts:971-972`).
5. **`AbortSignal`** (`CancellableLoader`) → `tokio_util::sync::CancellationToken`.
6. **Template-literal type** `SizeValue = number | "50%"` (`tui.ts:197`) → enum
   `SizeValue { Cells, Percent }`.
7. **ANSI-string line model with UTF-16 care**: `BoundedTerminalWriter` splits
   writes without splitting surrogate pairs (`tui-main-screen.ts:18-74`); grapheme
   and CJK width logic leans on `get-east-asian-width` → Rust UTF-8 simplifies
   chunking; width via `unicode-width`; decide between porting the string-diff
   renderer verbatim (crossterm-only) vs adopting ratatui's cell model — the
   string model is what makes main-screen scrollback mode work.
8. **N-API native addons** (darwin Obj-C, linux X11 C, win32 C; committed
   prebuilds) for clipboard + modifier state → Rust-native (e.g., `arboard`,
   platform crates) removes the prebuild matrix; `execSync` shell-outs in
   `terminal-image.ts` → `std::process::Command`.
9. **Dependencies to replace**: `marked` → `pulldown-cmark`; `get-east-asian-width`
   → `unicode-width`; `@xterm/headless` test harness → headless vte/termwiz
   emulator.
10. **Node runtime specifics**: raw-mode stdin via `process.stdin`, `readline`
    interactions, `SIGWINCH` self-kill for dimension refresh
    (`terminal.ts:46-53`), stdin drain on exit (SSH key-release leak guard,
    `Terminal.drainInput`) → crossterm event source + explicit drain policy.
11. **Node strip-only TS constraint** shaped the source (`.ts` extension imports,
    no parameter properties) — irrelevant in Rust, but explains style quirks.
12. **Render-cache discipline**: leaf components own `invalidate()` caches the
    layout engine deliberately does not duplicate (`tui-plan.md` decision 7) —
    Rust port must keep single-ownership of caches or accept a second
    invalidation story.