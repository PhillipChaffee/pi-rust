//! Error model of the runtime: the eight `RemoteServiceErrorCode` variants
//! ported from upstream `src/services/errors.ts` with the ordered
//! `REMOTE_SERVICE_ERROR_CODES` const and its parse surface, plus the
//! crate-level `ChordError` that aggregates several failures into one
//! (upstream's `AggregateError` role) when disposal or activation collects
//! multiple failures.
