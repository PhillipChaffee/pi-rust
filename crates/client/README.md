# pi-client

Rust port of upstream `packages/client` in earendil-works/pi (MIT, (c) 2025
Mario Zechner), pinned at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

The transport-neutral client half of the remote-session wire: it opens a
pluggable [`ByteTransport`](crate::transport), performs the versioned
handshake, then correlates requests, cancellations, and service
subscriptions over framed bytes (`pi-protocol` owns the framing and the
envelope schemas; the service meaning of the opaque payloads belongs to
`pi-chord`). Server-side routing mirrors it in `pi-server`.

Upstream is single-threaded Node: transport events, promise callbacks, and
subscriber delivery all run on one event loop in invocation order. The port
keeps that shape — the client is a single-threaded object (`Rc`/`RefCell`
internally, `!Send`) whose transport tasks run as local tasks, so every
surface expects the caller's current-thread tokio runtime behind a
`LocalSet`. Ordering guarantees are the ones the upstream suite pins:
ordered sends, ordered subscriber delivery, and correlation of
out-of-order responses.

The Unix transports and discovery live in the [`unix`](self::unix) module,
mirroring upstream's `./unix` subpath export. Windows is out of scope for
this effort (map ticket "Decide the Rust stack").
