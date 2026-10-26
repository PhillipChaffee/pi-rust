//! RPC mode, upstream's `src/modes/rpc/` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`jsonl`] owns the strict LF-only framing, [`rpc_types`] the command and
//! response vocabulary, [`rpc_mode`] the stdin loop and dispatch, and
//! [`rpc_client`] the child-process client.

pub mod jsonl;
pub mod rpc_client;
pub mod rpc_mode;
pub mod rpc_types;
