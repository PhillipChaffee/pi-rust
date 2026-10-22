# CI speed fact base (research for #160)

Fact base of mechanisms that could speed this repo's GitHub Actions CI, per the
ticket ask: no direction fixed going in, official docs and the tools'
repositories as primary sources, and "the current shape is already near the
floor" a valid closable outcome. A follow-up decision ticket chooses what, if
anything, is worth taking off this file.

Provenance: repo-side numbers measured from `gh api` over the last six CI runs
on `main` (commits `8a32f97`..`407d2a1`; run IDs 38343105275, 38359595561,
38544915754, 38631821927, 39003265925, 39743799835) and the workflow text at
`407d2a1`; tool-side facts carry one URL per claim and flag where the docs are
silent.

## Where the time is (measured)

Three parallel jobs on `ubuntu-latest`; wall clock is the `test` job.

| job          | range over 6 runs | latest |
|--------------|-------------------|--------|
| test         | 470–985 s         | 647 s  |
| doc-comments | 41–120 s          | 72 s   |
| hygiene      | 40–63 s           | 43 s   |

Latest `test` job steps (647 s total): disk-free provision 84 s, rust-cache
restore 26 s + save 21 s, tool installs ~5 s, `cargo fmt --check` 7 s, clippy
80 s, doctests 75 s, Xvfb 12 s, coverage gate 328 s, Coveralls upload 4 s.

The gate step's split between instrumented build and the 7,843-test nextest run
is not decomposed in CI logs; the closest local analogue is #159's 78.5 s for
the same nextest sweep on a warm tree (M-series, live doors skipped), so the
instrumented build is plausibly the larger share — unmeasured; get a one-run
build/run split from CI logs before any lever gets credit.

## Structure: what each step can and cannot share

- sccache does front the coverage build, by wrapper chaining: cargo-llvm-cov
  sets `RUSTC_WRAPPER` to itself and re-chains the pre-existing wrapper
  (wrapper.rs is an identical blob at the pinned v0.9.0 and main;
  https://github.com/taiki-e/cargo-llvm-cov/blob/main/src/wrapper.rs), which
  resolves this repo's `RUSTC_WRAPPER=sccache`. Instrumentation flags apply
  only to workspace-member crates; dependencies compile uninstrumented, so the
  doctest step's dep compiles are the same-shape rustc invocations the gate's
  dep leg can hit in sccache.
- clippy and fmt contribute nothing to the coverage build's cache. sccache
  does not support clippy/rls — open since 2019
  (https://github.com/mozilla/sccache/issues/423), with a documented warm-cached
  clippy run slower than a plain one (https://github.com/mozilla/sccache/issues/2041);
  fmt invokes no rustc. The 87 s of clippy + fmt sit on the `test` job's
  critical path warming nothing.
- rust-cache stores dependency build artifacts only — "In particular, the
  workspace crates themselves are not cached"
  (https://github.com/Swatinem/rust-cache#rust-cache-action-title) — keyed on
  job id + toolchain + `RUSTFLAGS`. cargo-llvm-cov additionally redirects the
  gate's build to its own `llvm-cov-target` directory
  (https://github.com/taiki-e/cargo-llvm-cov/blob/main/src/cargo.rs), so
  cross-step artifact sharing at cargo's level is moot; the workspace crates'
  instrumented compile is the per-PR, cross-run-sccache unit.
- sccache can never cache the link stage: "Crates that invoke the system
  linker cannot be cached. This includes bin, dylib, cdylib, and proc-macro
  crates" (https://github.com/mozilla/sccache/blob/main/docs/Rust.md). Link
  time is the uncacheable floor of every warm run.

Already banked, nil further headroom per official docs: `-Cdebuginfo=0` at the
documented maximum reduction
(https://doc.rust-lang.org/cargo/guide/build-performance.html), `CARGO_INCREMENTAL=0`
per sccache, `-Cembed-bitcode=no` applied by cargo by default in this profile
shape (https://doc.rust-lang.org/rustc/codegen-options/index.html#embed-bitcode),
sparse crates.io index default since 1.70.0, nextest as the run carrier
(https://nexte.st/docs/benchmarks/), `cargo test --doc` as the doctest carrier
nextest itself requests (https://nexte.st/docs/integrations/test-coverage/).

## The 470–985 s swing on an unchanged dependency tree

Documented candidate mechanisms, none confirmed as the cause here:

- sccache's GHA backend: "In case sccache reaches the rate limit of the
  service, the build will continue, but the storage might not be performed"
  (https://github.com/mozilla/sccache/blob/main/docs/GHA.md) — a green run
  that stores nothing leaves the next run cold.
- Measured service behavior under cache pressure: 429 rate-limit errors,
  13–21 s single write latencies, and a correlation between cache error count
  and job time (https://github.com/mozilla/sccache/issues/1485); GitHub staff:
  200 cache-entry uploads/min/repository cap and "cache thrashing" under the
  10 GB pool (https://github.com/mozilla/sccache/issues/2566).
- rust-cache and sccache entries share one repository cache pool: 10 GB total,
  LRU eviction, 7-day disuse deletion
  (https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#usage-limits-and-eviction-policy).
- rust-cache defaults `cache-on-failure=false`, so red runs bank nothing and
  extend the cold streak (https://github.com/Swatinem/rust-cache#readme).

Confirmation needs run-log archaeology (`sccache --show-stats` in the action
logs), not more docs. No doc names any instrument-coverage-specific behavior;
these caveats are invocation-shaped, not coverage-shaped.

## Ranked candidates (documented direction, unmeasured in this repo)

1. **Move clippy + fmt + doctests off the `test` job's critical path** into
   their own parallel jobs. ~170 s off the wall clock at zero runner cost on
   public repos (standard runner minutes free and unlimited, 20 concurrent
   jobs — https://docs.github.com/en/actions/reference/github-hosted-runners-reference#standard-github-hosted-runners-for-public-repositories,
   https://docs.github.com/en/actions/reference/limits#job-concurrency-limits-for-github-hosted-runners).
   Costs: each split job pays its own cache restore+save round trip (existing
   caches are immutable, new key each —
   https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#cache-hits-and-misses),
   and job-id keying fragments the 10 GB pool further. A second job's identical
   dep invocations hit the same repo-global sccache cache; that assumption is
   inference, unverified without a run.
2. **Faster linker for the gate** (`-Clinker-features=+lld` is stable on
   `x86_64-unknown-linux-gnu` —
   https://doc.rust-lang.org/rustc/codegen-options/index.html#linker-features;
   cargo's guide recommends exactly this shape —
   https://doc.rust-lang.org/cargo/guide/build-performance.html). Targets the
   one stage sccache can never cache, so warm runs keep the savings. No
   percentage documented anywhere; A/B it here.
3. **`codegen-units` 16 → 256 for dev/test** — the documented default under
   `CARGO_INCREMENTAL=0` is 16, overridable via
   `CARGO_PROFILE_DEV_CODEGEN_UNITS`/`--config`
   (https://doc.rust-lang.org/cargo/reference/profiles.html#codegen-units);
   cargo docs hedge "possibly reducing compile time".
4. **`cache-on-failure=true` on rust-cache** to close the red-run cold streak
   documented above (deliberate choice; the docs flag the trade, not the winner).
5. **Scope `RUSTC_WRAPPER` off the clippy step** if a run-log check shows
   sccache proxying clippy-driver costs rather than helps there
   (https://github.com/mozilla/sccache/issues/2041; today's pass-through
   behavior is undocumented).

Not a lever on a free public repo: larger runners (Team/Enterprise only;
"Usage of larger runners is always blocked until you set up a payment method"
— https://docs.github.com/en/billing/concepts/product-billing/github-actions),
nightly-only knobs (parallel frontend, Cranelift — barred by the stable
toolchain coverage story), within-runner nextest partitioning (documented only
as the cross-machine feature; cross-runner coverage merging is undocumented in
both nextest and cargo-llvm-cov — https://nexte.st/docs/ci-features/partitioning/).

## Docs-silence ledger

- sccache: step-ordering guidance; clippy-driver pass-through behavior today;
  whether the Rust hash excludes `--out-dir`-style paths (this decides whether
  the doctest step's dep compiles actually hit in-job); any
  instrument-coverage-specific behavior.
- rust-cache: restore-key fallback order across job-keyed caches; whether its
  cleanup retains `target/llvm-cov-target/**` subdirectories.
- cargo-llvm-cov: no sccache mention at all; no cross-runner merge story.
- No official page documents the GB footprint of the `dotnet`/`android`/`ghc`
  image payload the disk-free step removes, nor a conditional free-disk
  pattern; keep that step unconditional on the #153/#143 evidence (runner SSD
  documented at 14 GB —
  https://docs.github.com/en/actions/reference/github-hosted-runners-reference#standard-github-hosted-runners-for-public-repositories).

## Open measurements before any lever ranks above another

1. Decompose the 328 s gate step (build vs run) from CI logs once.
2. `sccache --show-stats` deltas across a couple of runs to test whether the
   slow runs ride a store-failure cold cache (the #1485/#2566 mechanisms).
3. clippy step with and without `RUSTC_WRAPPER` on the runner.
4. Local A/B of `-Clinker-features=+lld` and 256 CGUs against this workspace's
   instrumented build.

## Sources

Repo: `.github/workflows/ci.yml` at `407d2a1` (env :35–42, steps :87–92, gate
:130–131), `run-gates.sh` at `407d2a1`, run/job/step timings from `gh api` for
the run IDs above, local gate numbers recorded on #159. Tooling: the URLs
above, all reached 2026-10-22.
