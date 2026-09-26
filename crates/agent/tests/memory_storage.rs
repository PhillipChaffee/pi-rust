//! The `MemoryStorage` suite, ported 1:1 from upstream
//! `test/harness/memory-storage.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod session_common;
use session_common::*;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::types::{CustomEntryBody, NewEntry, Storage};
use pi_agent_core::harness::session::values::Write;
use pi_agent_core::harness::session::{commit as session_writes, values as stored_values};

#[tokio::test]
async fn uses_the_injected_clock_once_per_transaction() {
    let storage = MemoryStorage::new(MemoryStorageOptions {
        now: Some(ticking_clock()),
    });

    let first = storage
        .commit(
            vec![
                Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
                    id: "first".to_owned(),
                    parent_id: None,
                    body: CustomEntryBody {
                        custom_type: "note".to_owned(),
                        data: None,
                    },
                }))),
                Write::ValueSet(
                    stored_values::set_value(&stored_values::session_name(), "first".to_owned())
                        .expect("write"),
                ),
            ],
            &background_context(),
        )
        .await
        .expect("first commit");
    let second = storage
        .commit(
            vec![Write::ValueSet(
                stored_values::set_value(&stored_values::session_name(), "second".to_owned())
                    .expect("write"),
            )],
            &background_context(),
        )
        .await
        .expect("second commit");

    assert_eq!(first.timestamp, NOW);
    assert_eq!(second.timestamp, NOW + 1);
    assert_eq!(
        storage
            .get_entries(vec!["first".to_owned()], &background_context())
            .await
            .expect("entries")
            .get("first")
            .expect("entry")
            .timestamp(),
        first.timestamp,
    );
    storage.close(&background_context()).await.expect("close");
}
