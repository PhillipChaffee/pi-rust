# pi-chord

Rust port of upstream `packages/chord` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

Chord is an application-composition runtime for systems assembled from
plugins: facets (synchronous setup units), typed services (singleton and
keyed), replicated JSON state backed by a delta tracker, Go-like
cancellation contexts, and a transport-agnostic remote-service boundary.
Upstream is deliberately not a Pi package — it imports no other Pi
workspace package — and the port keeps that boundary: `pi-chord` declares
zero dependencies, with the test runtime living in dev-dependencies.

## The loading seam

Upstream bundles each facet with esbuild into a content-addressed CommonJS
file and loads it on Node with SHA-256 integrity checks and
`node:vm.compileFunction`. A Rust runtime cannot execute those artifacts,
so the port replaces facet execution with a deliberately open seam:
`node::bundle_loader` defines the `FacetLoader` trait plus an in-process
loader registry, and the manifest, artifact, integrity, and resolution
machinery ports as data validation (`node::manifest`, `node::bundle`,
`node::package`). The esbuild and `node:vm` pipeline itself is not ported;
the bundling pipeline runs over built sources, the packaging pipeline
carries the package.json conventions 1:1, and the ported bundle suite
(`tests/bundle.rs`) drives the loader seam through a fixture module host
whose program language stands in for the extension mechanism's.

## The string-segment contract

Delta string operations carry segments of one string value. Upstream
counts UTF-16 code units — the `t` operation removes a number of code
units from the front of a string. This port tracks owned UTF-8 strings, so
string segments are byte offsets: an append carries the bytes to add, a
front-truncate carries a number of bytes to remove, and a count that would
split a UTF-8 character boundary fails validation like any other
malformed operation.

## Module map

- `context`, `delta` — mirror upstream's `./context` and `./delta` subpaths.
- `node`, `bundler` — the manifest/artifact layer upstream exposes through
  its `./node` and `./bundler` subpaths.
- `types`, `json`, `errors`, `services`, `facets`, `api` — the root API
  surface ported from `src/types.ts`, `src/json.ts`, `src/services/*`,
  `src/facets/*`, and `src/api.ts`.

`src/lib.rs` is the crate's single import point; its re-export list mirrors
upstream's `src/index.ts`.
