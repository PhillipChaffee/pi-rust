# Unit tests live in `foo/tests.rs` siblings; the coverage gate measures production code

Every `#[cfg(test)]` unit-test block in `crates/*/src/**` lives as a
directory-module sibling: the parent file carries

```rust
#[cfg(test)]
mod tests;
```

and the tests live in a sibling `foo/tests.rs` (`foo/mod.rs` parents keep a
`tests.rs` in the same directory). No inline `#[cfg(test)] mod tests { … }`
block remains under `crates/*/src/**`.

## Why

cargo-llvm-cov's test exclusion is file-shaped — directories named `tests` and
files named `tests.rs` / `*_tests.rs` — and an inline test module sits inside
the production source file, so the gate charged its budget for test code:
helper closures, async blocks, and panic-message closures land in the
uncovered-functions denominator with counters no path can ever reach (the call
sites bind the cross-CGU inlined copies). The gate became a tax on passing
tests rather than a measure of them.

Upstream has no remedy: [taiki-e/cargo-llvm-cov#123](https://github.com/taiki-e/cargo-llvm-cov/issues/123)
asked for library-only counting in 2021 and remains open and blocked, and
`#[coverage(off)]` is nightly-only — its stabilization was reverted
([rust-lang/rust#134672](https://github.com/rust-lang/rust/pull/134672)). The
file layout is the stable mechanism available to us.

Moved tests are pure copy-paste, verified by `scripts/verify_test_move.py`:
the parent must equal the old text with the block swapped for the decl byte
for byte, and the sibling must equal the moved block minus one indent level or
the canonical rustfmt rendering of exactly that content.

## Policy

Tests are never deleted, thinned, or adapted to fit the gate. A budget or
floor shortfall after real work calls for a layout or tooling fix, not weaker
tests. Entries below the moves' re-baselining point reflect that policy:
`--fail-under-lines 95`, `--fail-under-regions 95`, and an
uncovered-functions cap with cross-platform headroom, ratcheted down as the
measured count shrinks.

Decided in [#156](https://github.com/PhillipChaffee/pi-rust/issues/156).
