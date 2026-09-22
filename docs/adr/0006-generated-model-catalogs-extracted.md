# Generated model catalogs: committed shards extracted from the pinned upstream build

The provider catalogs land in `crates/ai/src/providers/data/` as 38 per-provider
JSON shards plus `.manifest.json`, extracted by running upstream's
`scripts/generate-models.ts --strict` against a scratch copy of `packages/ai`
at the pin, and embedded at compile time with `include_str!`. The committed
`image-models.generated.ts` converts to one JSON file the same way. The
`scripts/model-data.ts` validation half ports as `model_data.rs` and gates the
committed shards in tests; upstream's `scripts/generate-models.ts` generator
does not port.

The generator is upstream-side maintenance machinery: 3141 lines of
per-provider normalization against live third-party catalogs (models.dev,
OpenRouter, Vercel AI Gateway) fetched at build time. Porting it would
re-fetch network state in CI — non-hermetic, and its output is a function of
the build date, not of the pin. Upstream's own strict allowlists are what make
its generated data reproducible; a Rust port would duplicate that logic with
no runtime consumer. Regeneration therefore happens upstream-side at a new
pin: rerun the strict generator, re-extract the shards, and the manifest's
schema version and structure hash validate the copy through the ported
`model_data` module. Upstream's `generate-models-strict.test.ts` and
`fireworks-model-generation.test.ts` have the generator as their subject and
do not port 1:1; the parity obligation they cover — committed data matching
the manifest and the catalog's documented invariants — is carried by the
`model-data-validation` port and the embedded-data gate tests. The fireworks
generation test's runtime half (adaptive-thinking payload shaping) lands with
the Anthropic Messages API ticket.

Decided in [#28](https://github.com/PhillipChaffee/pi-rust/issues/28) against
the pinned upstream checkout (`60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`,
strict run of 2026-09-19).
