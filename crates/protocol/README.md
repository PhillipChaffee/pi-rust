# pi-protocol

Rust port of upstream `packages/protocol` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

The wire for remote pi sessions: a 4-byte big-endian length-prefixed frame
carries exactly one definite-length RFC 8949 CBOR item, and the framed
payload validates as one of a small set of routed envelope schemas (client
hello/request/cancel; server hello/hello_error/response/service_update/
attachment). Envelope payloads stay opaque strict JSON — call semantics,
snapshots, and error codes belong to `pi-chord`, and the protocol validates
only that opaque values are strict JSON. Upstream is transport-neutral: no
sockets, no streams; the decoders accept arbitrary chunk boundaries through
an explicit `push`/`end` buffer API.

The CBOR codec is hand-ported 1:1 (per [ADR
0002](../../docs/adr/0002-hand-ported-cbor-codec.md)): off-the-shelf CBOR
crates do not reproduce the byte-exact encoding or the rejection surface, so
the 33 RFC 8949 hex round-trip vectors and the 28 decoder-rejection fixtures
from upstream's suite are the porting oracle, ported as-is. The crate
mirrors upstream's deliberate duplication of agent-side wire schemas: wire
schema ownership stays here, and no shared-types crate deduplicates it.
