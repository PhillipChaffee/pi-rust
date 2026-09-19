//! The service runtime, ported from upstream `src/services/*`: the
//! `$chord.service` control vocabulary and its strict wire parsing,
//! replicated state with sequence fencing on both provider and replica
//! sides, per-subscription state codec registries, the remote service
//! provider and consumer facades, the loopback transport, and the handle
//! types that keep provider rebinds invisible. Errors re-export through
//! this module for path parity with upstream's `services/errors.ts`.
