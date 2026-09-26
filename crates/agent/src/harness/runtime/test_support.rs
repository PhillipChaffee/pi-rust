//! The runtime suites' session fixture: a contract-implementing memory
//! session and controlled storage, standing in for the session-layer child's
//! backend until that child lands (retired by its merge). The shapes mirror
//! the contract only — the mutation barrier serializes one callback at a
//! time, commits assign sequences and timestamps, and the reads surface the
//! committed state.

#![allow(dead_code)]