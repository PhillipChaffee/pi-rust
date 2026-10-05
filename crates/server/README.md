# pi-server

Rust port of upstream `packages/server` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

The transport-neutral server half of the remote-session wire: a
[`ServerListener`](crate::listener) supplies established
[`ByteConnection`](crate::ByteConnection)s, the [`Server`](crate::Server)
handshakes each one (client hello → server hello or `hello_error`), then
routes requests — server-wide through the host's service attachment,
session-scoped through the router to a host-provided
[`ServerHost`](crate::ServerHost) — and publishes attachment changes and
subscription updates out of band (`pi-protocol` owns the framing and the
envelope schemas; the service meaning of the opaque payloads belongs to
`pi-chord`). The client-side mirror lives in `pi-client`.

Upstream is single-threaded Node: connection events, promise chaining, and
per-client operation serialization all run on one event loop in arrival
order. The port keeps that shape — the server is a single-threaded object
(`Rc`/`RefCell` internally, `!Send`) whose listener tasks and per-client
operation chains run as local tasks, so every surface expects the caller's
current-thread tokio runtime behind a `LocalSet`. Four upstream contracts
restate because JavaScript provides them at runtime and Rust cannot:
per-client promise chains become one driver task per client; reused settled
promises become latch waiters; the error classes become one owned
[`Failure`](crate::Failure) taxonomy whose identity-carrying opaque errors
ride an `Rc`; and upstream's structural identity — the wire snapshot,
update, and call objects *are* their JSON — becomes the `pi-chord`
serializers the response paths encode through.

The Unix listener transport, socket-path derivation, and the composed preset
live in the [`unix`](crate::unix) module, mirroring upstream's `./unix`
subpath export; the test doubles (host, harness, wire client) live in
[`testing`](crate::testing), mirroring upstream's shipped `./testing`
subpath. Windows is out of scope for this effort (map ticket "Decide the
Rust stack"), so `unix` compiles on Unix targets only.
