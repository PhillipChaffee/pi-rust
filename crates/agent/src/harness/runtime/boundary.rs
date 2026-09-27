//! The runtime boundary suite: behavior upstream exercises from other
//! layers (the harness lane surface, the drive procedures, and the reducer
//! callers) bound here where the runtime child owns the seams — the staged
//! seam raises, the drive spawn's failure path, the mismatch shape, the
//! durable lane record's wire round-trip, the inbox admission split, and
//! the captured-model projection over the state leaves.

#[cfg(test)]
mod tests;
