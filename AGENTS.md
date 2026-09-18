# pi-rust

Rust port of [earendil-works/pi](https://github.com/earendil-works/pi) (MIT,
(c) 2025 Mario Zechner).

## Reference codebase

The TypeScript original lives at `~/git/pi`. Read it freely as a read-only
primary source when porting; this repo is the only place work happens and
commits land, and `origin` points only at this repo. When a ticket depends on
exact upstream code, record the pi commit SHA in the ticket.

## Comment and doc-comment rules

The coding agent's global rules own the style basics — "Present state only"
(a comment never narrates what the code used to do) and "Why, not what" (a
comment earns its lines with a trade-off, a measurement, or a contract, never
a restatement of the signature) — and those apply everywhere, prose and code.
This repo adds four rules of its own:

- Cite the issue, don't narrate it. #NNN references are house currency, but
  only as a subordinate tag on a sentence that still works without them.
- Doc comments carry what the signature cannot: the wire's name for a field,
  what the other side sends there, defaults, degradation behavior, `# Errors`
  / `# Panics` sections where they apply.
- A flagged identifier is config, not a rewrite: when clippy::doc_markdown
  fires on a genuine house identifier, extend doc-valid-idents in clippy.toml
  (verbatim code spelling) rather than rewording the prose.
- Nothing may display a number no provider or session backend sends. Token
  counts, usage, timings, and model metadata come from the wire; the rule
  excludes instrumentation-fabricated numbers from docs and examples.

## Agent skills

### Issue tracker

GitHub issues on `PhillipChaffee/pi-rust` via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, labels equal to role names. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
