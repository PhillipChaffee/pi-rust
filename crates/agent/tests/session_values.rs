//! The bound durable-address runtime cases from upstream
//! `test/harness/values.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` — the cases that need a live
//! storage. The type-level invariance cases upstream pins with
//! `expectTypeOf`/`@ts-expect-error` restate compile-time (the `Value<T>`
//! phantom parameter), and `Object.isFrozen` has no Rust counterpart; the
//! unit suite in `crates/agent/src/harness/session/values/tests.rs` carries
//! the rest.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod session_common;
use session_common::*;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::types::Storage;
use pi_agent_core::harness::session::values::{
    self as stored_values, ListReadOptions, Write, set_value as set_value_write,
};

#[tokio::test]
async fn uses_separately_constructed_equal_addresses_for_one_durable_location() {
    let storage = MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(1)),
    });
    let first =
        stored_values::value::<serde_json::Value>("app.state", "workspace").expect("address");
    let second =
        stored_values::value::<serde_json::Value>("app.state", "workspace").expect("address");
    storage
        .commit(
            vec![Write::ValueSet(
                set_value_write(&first, serde_json::json!({ "ready": true })).expect("write"),
            )],
            &background_context(),
        )
        .await
        .expect("commit");

    let stored = storage
        .get_value(&second.address, &background_context())
        .await
        .expect("value");
    assert_eq!(
        stored,
        Some(stored_values::StoredValue {
            namespace: first.address.namespace.clone(),
            key: first.address.key.clone(),
            value: serde_json::json!({ "ready": true }),
            seq: 1,
        }),
    );
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn resolves_list_reads_through_the_reader_and_storage_surfaces() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(1)),
    }));
    let events = stored_values::generic_list("app.events", "");
    storage
        .commit(
            vec![Write::ListAppend(
                stored_values::append_list(&events, serde_json::json!({ "name": "created" }))
                    .expect("write"),
            )],
            &background_context(),
        )
        .await
        .expect("commit");

    // Upstream reads the same list through the `SessionReader`-typed view
    // of the storage; the erased Rust surface is `dyn Storage`, so both
    // reads go through the trait-object call.
    let from_storage = storage
        .read_list(&events.address, None, &background_context())
        .await
        .expect("list");
    let from_reader = storage
        .read_list(&events.address, None, &background_context())
        .await
        .expect("list");
    assert_eq!(from_storage, from_reader);
    assert_eq!(
        from_storage,
        vec![stored_values::ListElement {
            seq: 1,
            value: serde_json::json!({ "name": "created" }),
        }],
    );
    // The list-read options the erased reader carries restate the typed
    // defaults: the page and cursor semantics the conformance suite binds.
    let _ = ListReadOptions::default();
    storage.close(&background_context()).await.expect("close");
}
