# The pi-* crates publish to crates.io as the distribution channel

Decision, at upstream pin `60e7e76`: the workspace's library crates publish to
crates.io under their upstream package names — `pi-telemetry`, `pi-ai`,
`pi-chord`, `pi-evals`, `pi-tui`, `pi-protocol`, `pi-client`, `pi-agent-core`,
`pi-server`, `pi-coding-agent`, `pi-session-backend-sqlite-node`, and
`pi-import` — making crates.io the registry half of the `pi install
crate:<name>` channel (ADR 0007) and the extension SDK a plain dependency: an
extension crate declares `pi-coding-agent`, compiles its published source
`--locked` against the extension's committed lockfile, and the receipt's
version pin stays reproducible. Versions are lockstep: every crate carries the
workspace version and a release is a workspace snapshot tagged `v<version>` on
this repo. The series stays `0.x` until full parity; within `0.x` a
semver-breaking change bumps the minor, so an extension's lockfile upgrade is
always a deliberate act. Cadence: a publish rides the ticket that moves a
crate's public API — cut from `main` after the merge, all crates published
together in dependency order. Yank policy: a published version is yanked only
when it is wrong on the machine — fails to compile, corrupts data, or leaks
credentials; behavioral regressions and API mistakes get a forward fix
instead. A yanked version stays resolvable to committed lockfiles (`cargo`
keeps building `--locked` against it), so installed receipts never rot.

## Considered options

- **Git-dependency-only distribution** — no registry account, no version
  history, and the `crate:` channel loses crates.io's provenance and
  checksums; lockfile pinning degrades to a git rev. Upstream's npm-publish
  role would have no carrier (ADR 0007's consequence line).
- **Independent per-crate versions** — the ecosystem norm, but it couples one
  release to hand-editing twelve version fields and cross-checking twelve
  compatibility ranges; lockstep keeps one version string per snapshot in
  every receipt.
- **Publish only at 1.0 parity** — defers semver stability past the port;
  extension authors would build against git revs in the meantime and the
  `crate:` channel ships nothing.

## Consequences

- Every public item in a published crate is API. `missing_docs` is denied
  workspace-wide already, so the doc gate doubles as the API-doc gate, and
  `pub(crate)`/`#[doc(hidden)]` boundaries harden at each publish.
- The root `pi-rust` aggregate (bootstrap binary, no library surface) does
  not publish; it carries `publish = false`.
- The crates.io metadata set (description, license, repository, readme,
  keywords, categories) is part of the publish gate — every crate carries all
  six plus a LICENSE file and a README; this slice closed the four missing
  per-crate READMEs (`pi-ai`, `pi-evals`, `pi-telemetry`, `pi-tui`).
- Lockstep releases mean a one-line bug fix in one crate republishes all of
  them at the next version; accepted — the workspace is one port, not twelve
  products.
