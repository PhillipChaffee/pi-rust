# CBOR codec: hand-ported strict subset, not a CBOR crate

The protocol crate hand-ports pi-protocol's CBOR encoder/decoder 1:1: 4-byte
unsigned big-endian frame length prefix, strict definite-length subset
(minimal-length integer encoding, safe integers ±2^53-1, string-keyed maps only,
duplicate keys rejected, float64-only `0xfb`), and the full rejection surface —
indefinite lengths, tags, break, float16/32, non-finite numbers, `undefined`,
cycles, trailing data. Off-the-shelf crates (ciborium, serde_cbor) do not
reproduce the rejection surface or the byte-exact encoding, and the wire is a
compatibility surface with TS pi, so the 33 hex round-trip vectors and 28
decoder-rejection fixtures serve as the porting oracle. serde_json with tolerant
reads (skip malformed lines, LF-only byte framing, `#[serde(tag = "type")]`)
handles the JSONL session files. The decoder keeps an explicit `push`/`end`
buffer API rather than `tokio_util::codec` because the tests demand
chunk-boundary independence.

Decided in [#10](https://github.com/PhillipChaffee/pi-rust/issues/10) against
evidence on the `research/survey-protocol-client-server` branch (upstream pin
`60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).
