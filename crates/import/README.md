# pi-import

The TS-pi to Rust-pi migration tool, the one workspace crate with no upstream
counterpart (decided by [the import-tool
ticket](https://github.com/PhillipChaffee/pi-rust/issues/13), ADR 0003
addendum). It migrates a TS-pi install's sessions, settings, trust, and
provider credentials into the Rust pi's formats as the port tickets define
them, so a Rust pi can replace a TS pi.

The source is TS pi's agent dir — `PI_CODING_AGENT_DIR` when set, else
`~/.pi/agent`, overridable with `--source` — and the target is the Rust pi's
own agent dir, derived the same way. Project-side settings migrate only with
`--project <path>`. The report is human-readable per artifact; `--json`
emits the machine-readable form; `--dry-run` reports without writing; the
exit is non-zero when anything failed, with failures itemized.

The tool never executes `!cmd`/`${VAR}` indirections (they carry verbatim and
re-resolve at runtime) and never prints key material — credential entries
appear as provider ids and counts only. TS extensions are reported as
skipped items; a Rust pi cannot execute them.

Work in progress: the tool lands on the map
([Port the TS→Rust import tool](https://github.com/PhillipChaffee/pi-rust/issues/136)).
Windows is out of scope for this effort (map ticket "Decide the Rust stack").
