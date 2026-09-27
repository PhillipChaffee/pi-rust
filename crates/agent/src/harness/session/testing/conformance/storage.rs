//! The durable Storage contract's conformance cases, ported from upstream
//! `src/harness/session/testing/conformance/storage.ts`.
//!
//! The fixture bodies assert with `assert*!` (upstream's `node:assert`);
//! a failing case panics its runner.

#![expect(
    clippy::expect_used,
    reason = "conformance fixtures construct fixed addresses and fixed writes whose Results are infallible; a failure is a bug the case panics on"
)]
#![expect(
    clippy::too_many_lines,
    reason = "the creator mirrors upstream's createStorageConformance case array; the body is the cases"
)]

use std::sync::Arc;

use pi_ai::types::BoxedFuture;
use serde_json::json;

use crate::harness::context::background_context;
use crate::harness::session::testing::conformance::{
    asc_scan, asc_usage_scan, assert_historical_unchanged, assert_list_values,
    assert_strictly_increasing, assert_value_absent, branch_query_seed_writes, commit_ok,
    compaction_entry_write, custom_entry, custom_entry_write, ids, insert_entry_write,
    insert_usage_write, list_element, scan_branch_ids, scan_entry_ids, scan_structure_ids,
    snapshot_historical_state, stored_map, stored_value, stored_values, test_list, test_name,
    test_value, test_value_prefix, usage, usage_ledger_writes, user_entry, user_entry_write,
    user_message, zero_usage,
};
use crate::harness::session::testing::types::{ConformanceCase, StorageFixture};
use crate::harness::session::types::{
    CommitResult, NewEntry, Storage, StorageBranchScan, UsageScan,
};
use crate::harness::session::values::{ListCursor, ListElement, ListReadOptions, Write};

/// The factory one storage fixture case builds through, upstream's
/// `() => Promise<StorageFixture>`.
pub type StorageFixtureFactory =
    Arc<dyn Fn() -> BoxedFuture<'static, StorageFixture> + Send + Sync>;

/// The per-case body over the storage, upstream's
/// `test: (fixture) => Promise<void>`.
pub type StorageCaseTest =
    Arc<dyn for<'a> Fn(&'a dyn Storage) -> BoxedFuture<'a, ()> + Send + Sync>;

fn case(
    factory: &StorageFixtureFactory,
    group: &str,
    name: &str,
    test: StorageCaseTest,
) -> ConformanceCase {
    let factory = factory.clone();
    ConformanceCase::new(group, name, move || {
        let factory = factory.clone();
        let test = test.clone();
        Box::pin(async move {
            let fixture = factory().await;
            test(fixture.storage.as_ref()).await;
            fixture.dispose().await;
        })
    })
}

/// Asserts the commit's stats equal the storage's post-apply totals,
/// upstream's `assertCommitStats`.
async fn assert_commit_stats(storage: &dyn Storage, result: &CommitResult) {
    assert_eq!(
        result.stats,
        storage
            .get_stats(&background_context())
            .await
            .expect("stats")
    );
}

/// Creates fresh, runner-independent cases for the durable Storage contract,
/// upstream's `createStorageConformance`.
///
/// # Panics
/// A case body panics on its first broken assertion, upstream's
/// `node:assert` throw.
#[must_use]
pub fn create_storage_conformance(factory: &StorageFixtureFactory) -> Vec<ConformanceCase> {
    vec![
        case(
            factory,
            "transactions",
            "commits mixed writes atomically in write order",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let result = commit_ok(
                        &*storage,
                        vec![
                            user_entry_write("entry", None, "entry"),
                            Write::ValueSet(
                                stored_values::set_value(&test_name(), json!("session"))
                                    .expect("write"),
                            ),
                            insert_usage_write(
                                "usage",
                                usage(2, 3),
                                false,
                                Some("entry".to_owned()),
                            ),
                        ],
                    )
                    .await;

                    assert_eq!(result.seqs.len(), 3);
                    assert_eq!(result.first_seq, result.seqs[0]);
                    assert_commit_stats(&*storage, &result).await;
                    assert_strictly_increasing(&result.seqs);
                    assert!(result.timestamp >= 0);
                    assert_eq!(
                        storage
                            .get_entries(vec!["entry".to_owned()], &background_context())
                            .await
                            .expect("entries"),
                        stored_map(vec![(
                            "entry",
                            user_entry("entry", None, "entry")
                                .materialize(result.seqs[0], result.timestamp),
                        )]),
                    );
                    assert_eq!(
                        storage
                            .get_value(&test_name().address, &background_context())
                            .await
                            .expect("value"),
                        Some(stored_value(
                            &test_name().address,
                            json!("session"),
                            result.seqs[1]
                        )),
                    );
                    assert_eq!(
                        storage
                            .scan_usage(
                                &UsageScan {
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Asc
                                    ),
                                    ..Default::default()
                                },
                                &background_context()
                            )
                            .await
                            .expect("usage"),
                        vec![crate::harness::session::types::UsageRow {
                            id: "usage".to_owned(),
                            seq: result.seqs[2],
                            usage: usage(2, 3),
                            entry_id: Some("entry".to_owned()),
                            adjustment: false,
                            details: None,
                        }],
                    );
                })
            }),
        ),
        case(
            factory,
            "transactions",
            "rolls back every store when a mixed transaction fails",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    storage
                        .commit(
                            vec![
                                user_entry_write("root", None, "root"),
                                insert_usage_write("taken", usage(1, 1), false, None),
                            ],
                            &background_context(),
                        )
                        .await
                        .expect("setup commit");
                    let before = snapshot_historical_state(&*storage).await;

                    let rejected = storage
                        .commit(
                            vec![
                                Write::ValueSet(
                                    stored_values::set_value(&test_name(), json!("transient"))
                                        .expect("write"),
                                ),
                                custom_entry_write("transient-entry", Some("root"), "note"),
                                insert_usage_write("transient-usage", usage(5, 8), true, None),
                                custom_entry_write("taken", Some("root"), "note"),
                            ],
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the mixed transaction to reject"
                    );

                    assert_historical_unchanged(&*storage, &before).await;
                    assert_value_absent(&*storage, &test_name().address).await;
                })
            }),
        ),
        case(
            factory,
            "transactions",
            "preserves overwritten and deleted values when a transaction fails",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    storage
                        .commit(
                            vec![
                                Write::ValueSet(
                                    stored_values::set_value(
                                        &test_value("overwritten"),
                                        json!("original"),
                                    )
                                    .expect("write"),
                                ),
                                Write::ValueSet(
                                    stored_values::set_value(
                                        &test_value("deleted"),
                                        json!({ "kept": true }),
                                    )
                                    .expect("write"),
                                ),
                                user_entry_write("taken", None, "taken"),
                            ],
                            &background_context(),
                        )
                        .await
                        .expect("setup commit");
                    let overwritten_before = storage
                        .get_value(&test_value("overwritten").address, &background_context())
                        .await
                        .expect("value");
                    let deleted_before = storage
                        .get_value(&test_value("deleted").address, &background_context())
                        .await
                        .expect("value");

                    let rejected = storage
                        .commit(
                            vec![
                                Write::ValueSet(
                                    stored_values::set_value(
                                        &test_value("overwritten"),
                                        json!("transient"),
                                    )
                                    .expect("write"),
                                ),
                                Write::ValueDelete(stored_values::delete_value(&test_value(
                                    "deleted",
                                ))),
                                custom_entry_write("transient", Some("taken"), "note"),
                                insert_entry_write(custom_entry(
                                    "taken",
                                    None,
                                    "note",
                                    Some(json!({ "id": "taken" })),
                                )),
                            ],
                            &background_context(),
                        )
                        .await;
                    assert!(rejected.is_err(), "expected the transaction to reject");

                    assert_eq!(
                        storage
                            .get_value(&test_value("overwritten").address, &background_context())
                            .await
                            .expect("value"),
                        overwritten_before,
                    );
                    assert_eq!(
                        storage
                            .get_value(&test_value("deleted").address, &background_context())
                            .await
                            .expect("value"),
                        deleted_before,
                    );
                    assert!(
                        storage
                            .get_entries(vec!["transient".to_owned()], &background_context())
                            .await
                            .expect("entries")
                            .is_empty(),
                    );
                })
            }),
        ),
        case(
            factory,
            "transactions",
            "enforces one shared entry and usage id namespace",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    storage
                        .commit(
                            vec![
                                user_entry_write("existing-entry", None, "existing-entry"),
                                insert_usage_write("existing-usage", usage(1, 1), false, None),
                            ],
                            &background_context(),
                        )
                        .await
                        .expect("setup commit");

                    let rejected = storage
                        .commit(
                            vec![insert_usage_write(
                                "existing-entry",
                                usage(2, 2),
                                false,
                                None,
                            )],
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the duplicate usage id to reject"
                    );
                    let rejected = storage
                        .commit(
                            vec![insert_entry_write(custom_entry(
                                "existing-usage",
                                None,
                                "note",
                                Some(json!({ "id": "existing-usage" })),
                            ))],
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the duplicate entry id to reject"
                    );

                    for (id, writes) in [
                        (
                            "entry-then-usage",
                            vec![
                                insert_entry_write(custom_entry(
                                    "entry-then-usage",
                                    None,
                                    "note",
                                    Some(json!({ "id": "entry-then-usage" })),
                                )),
                                insert_usage_write("entry-then-usage", usage(3, 3), false, None),
                            ],
                        ),
                        (
                            "usage-then-entry",
                            vec![
                                insert_usage_write("usage-then-entry", usage(4, 4), false, None),
                                insert_entry_write(custom_entry(
                                    "usage-then-entry",
                                    None,
                                    "note",
                                    Some(json!({ "id": "usage-then-entry" })),
                                )),
                            ],
                        ),
                    ] {
                        let rejected = storage.commit(writes, &background_context()).await;
                        assert!(rejected.is_err(), "Expected duplicate id {id} to reject");
                    }

                    assert_eq!(
                        ids(&storage
                            .scan_entries(&asc_scan(), &background_context())
                            .await
                            .expect("entries")),
                        ["existing-entry"],
                    );
                    assert_eq!(
                        storage
                            .scan_usage(&asc_usage_scan(), &background_context())
                            .await
                            .expect("usage")
                            .into_iter()
                            .map(|row| row.id)
                            .collect::<Vec<_>>(),
                        ["existing-usage"],
                    );
                })
            }),
        ),
        case(
            factory,
            "transactions",
            "resolves parents only from prior entries and earlier writes",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    storage
                        .commit(
                            vec![user_entry_write("root", None, "root")],
                            &background_context(),
                        )
                        .await
                        .expect("setup commit");
                    storage
                        .commit(
                            vec![
                                custom_entry_write("child", Some("root"), "note"),
                                custom_entry_write("grandchild", Some("child"), "note"),
                            ],
                            &background_context(),
                        )
                        .await
                        .expect("setup commit");
                    assert_eq!(
                        scan_branch_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "grandchild".to_owned(),
                                order: Some(
                                    crate::harness::session::types::BranchScanOrder::OldestFirst,
                                ),
                                ..Default::default()
                            },
                        )
                        .await,
                        ["root", "child", "grandchild"],
                    );

                    let rejected = storage
                        .commit(
                            vec![
                                custom_entry_write("before-parent", Some("later-parent"), "note"),
                                custom_entry_write("later-parent", Some("root"), "note"),
                                Write::ValueSet(
                                    stored_values::set_value(
                                        &stored_values::entry_label("before-parent"),
                                        "transient".to_owned(),
                                    )
                                    .expect("write"),
                                ),
                            ],
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the forward-parent transaction to reject"
                    );
                    let rejected = storage
                        .commit(
                            vec![custom_entry_write("orphan", Some("missing"), "note")],
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the orphan transaction to reject"
                    );
                    storage
                        .commit(
                            vec![insert_usage_write(
                                "usage-is-not-parent",
                                usage(1, 1),
                                false,
                                None,
                            )],
                            &background_context(),
                        )
                        .await
                        .expect("usage commit");
                    let rejected = storage
                        .commit(
                            vec![custom_entry_write(
                                "usage-child",
                                Some("usage-is-not-parent"),
                                "note",
                            )],
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the usage-parent transaction to reject"
                    );

                    assert_eq!(
                        storage
                            .get_entries(
                                vec![
                                    "before-parent".to_owned(),
                                    "later-parent".to_owned(),
                                    "orphan".to_owned(),
                                    "usage-child".to_owned(),
                                ],
                                &background_context(),
                            )
                            .await
                            .expect("entries"),
                        stored_map(Vec::<(&str, crate::harness::session::types::Entry)>::new()),
                    );
                    assert!(
                        storage
                            .get_value(
                                &stored_values::entry_label("before-parent").address,
                                &background_context()
                            )
                            .await
                            .expect("value")
                            .is_none(),
                    );
                })
            }),
        ),
        case(
            factory,
            "transactions",
            "places pending content under its reserved entry id",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let message = user_message("queued");
                    let entry = NewEntry::Message {
                        id: "reserved".to_owned(),
                        parent_id: None,
                        body: Box::new(crate::harness::session::types::MessageEntry {
                            message: message.clone(),
                            terminate: None,
                        }),
                    };
                    let pending = crate::harness::session::types::PendingEntry::Message {
                        payload: Box::new(message),
                    };
                    storage
                        .commit(
                            vec![
                                Write::ValueSet(
                                    stored_values::set_value(
                                        &stored_values::pending_entry("reserved"),
                                        pending.clone(),
                                    )
                                    .expect("write"),
                                ),
                                Write::ValueSet(
                                    stored_values::set_value(
                                        &stored_values::branch_tip("main"),
                                        Option::<String>::None,
                                    )
                                    .expect("write"),
                                ),
                            ],
                            &background_context(),
                        )
                        .await
                        .expect("setup commit");

                    assert!(
                        storage
                            .get_entries(vec!["reserved".to_owned()], &background_context())
                            .await
                            .expect("entries")
                            .is_empty(),
                    );
                    assert_eq!(
                        storage
                            .get_value(
                                &stored_values::pending_entry("reserved").address,
                                &background_context()
                            )
                            .await
                            .expect("value")
                            .expect("pending stored")
                            .value,
                        serde_json::to_value(&pending).expect("pending wire"),
                    );
                    assert_eq!(
                        storage
                            .get_value(
                                &stored_values::branch_tip("main").address,
                                &background_context()
                            )
                            .await
                            .expect("value")
                            .expect("tip stored")
                            .value,
                        serde_json::Value::Null,
                    );

                    let placement = commit_ok(
                        &*storage,
                        vec![
                            insert_entry_write(entry.clone()),
                            Write::ValueDelete(stored_values::delete_value(
                                &stored_values::pending_entry("reserved"),
                            )),
                            Write::ValueSet(
                                stored_values::set_value(
                                    &stored_values::branch_tip("main"),
                                    Some("reserved".to_owned()),
                                )
                                .expect("write"),
                            ),
                        ],
                    )
                    .await;

                    assert_eq!(
                        storage
                            .get_entries(vec!["reserved".to_owned()], &background_context())
                            .await
                            .expect("entries"),
                        stored_map(vec![(
                            "reserved",
                            entry.materialize(placement.seqs[0], placement.timestamp),
                        )]),
                    );
                    assert!(
                        storage
                            .get_value(
                                &stored_values::pending_entry("reserved").address,
                                &background_context()
                            )
                            .await
                            .expect("value")
                            .is_none(),
                    );
                    assert_eq!(
                        storage
                            .get_value(
                                &stored_values::branch_tip("main").address,
                                &background_context()
                            )
                            .await
                            .expect("value"),
                        Some(stored_value(
                            &stored_values::branch_tip("main").address,
                            json!("reserved"),
                            placement.seqs[2]
                        )),
                    );
                })
            }),
        ),
        case(
            factory,
            "values",
            "sets, replaces, deletes, and recreates values without tombstones",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let first = commit_ok(
                        &*storage,
                        vec![
                            Write::ValueSet(
                                stored_values::set_value(&test_value("prefix/b"), json!(1))
                                    .expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&test_value("prefix/a"), json!(2))
                                    .expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&test_value("other"), json!(3))
                                    .expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&test_value("prefix/\u{e000}"), json!(4))
                                    .expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&test_value("prefix/\u{10000}"), json!(5))
                                    .expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(
                                    &test_value("prefix/a"),
                                    serde_json::Value::Null,
                                )
                                .expect("write"),
                            ),
                        ],
                    )
                    .await;
                    assert_eq!(
                        storage
                            .get_value(&test_value("prefix/a").address, &background_context())
                            .await
                            .expect("value"),
                        Some(stored_value(
                            &test_value("prefix/a").address,
                            serde_json::Value::Null,
                            first.seqs[5]
                        )),
                    );

                    let second = commit_ok(
                        &*storage,
                        vec![
                            Write::ValueDelete(stored_values::delete_value(&test_value(
                                "prefix/a",
                            ))),
                            Write::ValueDelete(stored_values::delete_value(&test_value("absent"))),
                            Write::ValueSet(
                                stored_values::set_value(
                                    &test_value("prefix/a"),
                                    json!("recreated"),
                                )
                                .expect("write"),
                            ),
                        ],
                    )
                    .await;

                    assert_eq!(
                        storage
                            .scan_values(
                                &test_value_prefix("prefix/").address,
                                &background_context()
                            )
                            .await
                            .expect("values"),
                        vec![
                            stored_values::StoredValue {
                                namespace: test_value("prefix/a").address.namespace,
                                key: "prefix/a".to_owned(),
                                value: json!("recreated"),
                                seq: second.seqs[2],
                            },
                            stored_values::StoredValue {
                                namespace: test_value("prefix/b").address.namespace,
                                key: "prefix/b".to_owned(),
                                value: json!(1),
                                seq: first.seqs[0],
                            },
                            stored_values::StoredValue {
                                namespace: test_value("prefix/\u{e000}").address.namespace,
                                key: "prefix/\u{e000}".to_owned(),
                                value: json!(4),
                                seq: first.seqs[3],
                            },
                            stored_values::StoredValue {
                                namespace: test_value("prefix/\u{10000}").address.namespace,
                                key: "prefix/\u{10000}".to_owned(),
                                value: json!(5),
                                seq: first.seqs[4],
                            },
                        ],
                    );
                    assert_value_absent(&*storage, &test_value("absent").address).await;
                })
            }),
        ),
        case(
            factory,
            "values",
            "applies same-transaction value and list operations in write order",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let kept_value = test_value("write-order/kept");
                    let deleted_value = test_value("write-order/deleted");
                    let kept_list = test_list("write-order/kept");
                    let deleted_list = test_list("write-order/deleted");
                    let result = commit_ok(
                        &*storage,
                        vec![
                            Write::ValueSet(
                                stored_values::set_value(&deleted_value, json!("transient"))
                                    .expect("write"),
                            ),
                            Write::ValueDelete(stored_values::delete_value(&deleted_value)),
                            Write::ValueSet(
                                stored_values::set_value(&kept_value, json!("transient"))
                                    .expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&kept_value, json!("kept"))
                                    .expect("write"),
                            ),
                            Write::ListAppend(
                                stored_values::append_list(&kept_list, json!("transient"))
                                    .expect("write"),
                            ),
                            Write::ListDelete(stored_values::delete_list(&kept_list)),
                            Write::ListAppend(
                                stored_values::append_list(&kept_list, json!("kept"))
                                    .expect("write"),
                            ),
                            Write::ListAppend(
                                stored_values::append_list(&deleted_list, json!("transient"))
                                    .expect("write"),
                            ),
                            Write::ListDelete(stored_values::delete_list(&deleted_list)),
                        ],
                    )
                    .await;

                    assert_value_absent(&*storage, &deleted_value.address).await;
                    assert_eq!(
                        storage
                            .get_value(&kept_value.address, &background_context())
                            .await
                            .expect("value"),
                        Some(stored_values::StoredValue {
                            namespace: kept_value.address.namespace.clone(),
                            key: kept_value.address.key.clone(),
                            value: json!("kept"),
                            seq: result.seqs[3],
                        }),
                    );
                    assert_eq!(
                        storage
                            .read_list(&kept_list.address, None, &background_context())
                            .await
                            .expect("list"),
                        vec![list_element(result.seqs[6], json!("kept"))],
                    );
                    assert!(
                        storage
                            .read_list(&deleted_list.address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                })
            }),
        ),
        case(
            factory,
            "values",
            "does not change historical stores during value-only commits",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    storage
                        .commit(
                            vec![
                                user_entry_write("root", None, "root"),
                                insert_usage_write("historical-usage", usage(2, 3), false, None),
                            ],
                            &background_context(),
                        )
                        .await
                        .expect("setup commit");
                    let before = snapshot_historical_state(&*storage).await;

                    let result = commit_ok(
                        &*storage,
                        vec![
                            Write::ValueSet(
                                stored_values::set_value(&test_name(), json!("first"))
                                    .expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&test_name(), json!("second"))
                                    .expect("write"),
                            ),
                        ],
                    )
                    .await;

                    assert_historical_unchanged(&*storage, &before).await;
                    assert_eq!(
                        storage
                            .get_value(&test_name().address, &background_context())
                            .await
                            .expect("value"),
                        Some(stored_value(
                            &test_name().address,
                            json!("second"),
                            result.seqs[1]
                        )),
                    );
                })
            }),
        ),
        case(
            factory,
            "lists",
            "pages appends by global sequence and deletes whole lists",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let address = test_list("events");
                    assert!(
                        storage
                            .read_list(&address.address, None, &background_context())
                            .await
                            .expect("list")
                            .is_empty(),
                    );
                    let result = commit_ok(
                        &*storage,
                        vec![
                            Write::ListAppend(
                                stored_values::append_list(&address, json!("a")).expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&test_name(), json!("gap"))
                                    .expect("write"),
                            ),
                            Write::ListAppend(
                                stored_values::append_list(&address, json!("b")).expect("write"),
                            ),
                            Write::ListAppend(
                                stored_values::append_list(&address, json!("c")).expect("write"),
                            ),
                        ],
                    )
                    .await;
                    assert_eq!(
                        storage
                            .read_list(&address.address, None, &background_context())
                            .await
                            .expect("list"),
                        vec![
                            ListElement {
                                seq: result.seqs[0],
                                value: json!("a")
                            },
                            ListElement {
                                seq: result.seqs[2],
                                value: json!("b")
                            },
                            ListElement {
                                seq: result.seqs[3],
                                value: json!("c")
                            },
                        ],
                    );
                    assert_eq!(
                        storage
                            .read_list(
                                &address.address,
                                Some(ListReadOptions {
                                    limit: Some(2),
                                    ..Default::default()
                                }),
                                &background_context()
                            )
                            .await
                            .expect("list"),
                        vec![
                            ListElement {
                                seq: result.seqs[0],
                                value: json!("a")
                            },
                            ListElement {
                                seq: result.seqs[2],
                                value: json!("b")
                            },
                        ],
                    );
                    assert_eq!(
                        storage
                            .read_list(
                                &address.address,
                                Some(ListReadOptions {
                                    cursor: Some(ListCursor {
                                        seq: result.seqs[0]
                                    }),
                                    limit: Some(2),
                                    ..Default::default()
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("list"),
                        vec![
                            ListElement {
                                seq: result.seqs[2],
                                value: json!("b")
                            },
                            ListElement {
                                seq: result.seqs[3],
                                value: json!("c")
                            },
                        ],
                    );
                    assert_eq!(
                        storage
                            .read_list(
                                &address.address,
                                Some(ListReadOptions {
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Desc
                                    ),
                                    limit: Some(2),
                                    ..Default::default()
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("list"),
                        vec![
                            ListElement {
                                seq: result.seqs[3],
                                value: json!("c")
                            },
                            ListElement {
                                seq: result.seqs[2],
                                value: json!("b")
                            },
                        ],
                    );
                    assert_eq!(
                        storage
                            .read_list(
                                &address.address,
                                Some(ListReadOptions {
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Desc
                                    ),
                                    cursor: Some(ListCursor {
                                        seq: result.seqs[3]
                                    }),
                                    limit: Some(2),
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("list"),
                        vec![
                            ListElement {
                                seq: result.seqs[2],
                                value: json!("b")
                            },
                            ListElement {
                                seq: result.seqs[0],
                                value: json!("a")
                            },
                        ],
                    );
                    let rejected = storage
                        .read_list(
                            &address.address,
                            Some(ListReadOptions {
                                limit: Some(0),
                                ..Default::default()
                            }),
                            &background_context(),
                        )
                        .await;
                    assert!(rejected.is_err(), "expected the zero limit to reject");
                    let rejected = storage
                        .read_list(
                            &address.address,
                            Some(ListReadOptions {
                                limit: Some(u64::MAX),
                                ..Default::default()
                            }),
                            &background_context(),
                        )
                        .await;
                    assert!(rejected.is_err(), "expected the overflow limit to reject");

                    storage
                        .commit(
                            vec![
                                Write::ListDelete(stored_values::delete_list(&address)),
                                Write::ListDelete(stored_values::delete_list(&test_list("absent"))),
                                Write::ListAppend(
                                    stored_values::append_list(&address, json!("new"))
                                        .expect("write"),
                                ),
                            ],
                            &background_context(),
                        )
                        .await
                        .expect("delete commit");
                    assert_list_values(&*storage, &address.address, &[json!("new")]).await;
                })
            }),
        ),
        case(
            factory,
            "lists",
            "clamps one read page without limiting list growth",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let address = test_list("large");
                    storage
                        .commit(
                            (0..10_001u64)
                                .map(|index| {
                                    Write::ListAppend(
                                        stored_values::append_list(&address, json!(index))
                                            .expect("write"),
                                    )
                                })
                                .collect::<Vec<Write>>(),
                            &background_context(),
                        )
                        .await
                        .expect("commit");
                    let first_page = storage
                        .read_list(&address.address, None, &background_context())
                        .await
                        .expect("list");
                    assert_eq!(first_page.len(), 1_000);
                    assert_eq!(
                        storage
                            .read_list(
                                &address.address,
                                Some(ListReadOptions {
                                    limit: Some(20_000),
                                    ..Default::default()
                                }),
                                &background_context()
                            )
                            .await
                            .expect("list")
                            .len(),
                        10_000,
                    );
                    assert_eq!(
                        storage
                            .read_list(
                                &address.address,
                                Some(ListReadOptions {
                                    cursor: Some(ListCursor {
                                        seq: first_page.last().expect("page").seq,
                                    }),
                                    ..Default::default()
                                }),
                                &background_context(),
                            )
                            .await
                            .expect("list")
                            .len(),
                        1_000,
                    );
                })
            }),
        ),
        case(
            factory,
            "lists",
            "commits mixed list writes atomically and rolls them back with siblings",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let address = test_list("atomic");
                    let committed = commit_ok(
                        &*storage,
                        vec![
                            user_entry_write("mixed", None, "mixed"),
                            Write::ListAppend(
                                stored_values::append_list(&address, json!("kept")).expect("write"),
                            ),
                            Write::ValueSet(
                                stored_values::set_value(&test_name(), json!("kept"))
                                    .expect("write"),
                            ),
                            insert_usage_write("mixed-usage", usage(1, 2), false, None),
                        ],
                    )
                    .await;
                    assert_eq!(
                        storage
                            .read_list(&address.address, None, &background_context())
                            .await
                            .expect("list"),
                        vec![list_element(committed.seqs[1], json!("kept"))],
                    );

                    let rejected = storage
                        .commit(
                            vec![
                                Write::ListAppend(
                                    stored_values::append_list(&address, json!("transient"))
                                        .expect("write"),
                                ),
                                Write::ValueDelete(stored_values::delete_value(&test_name())),
                                user_entry_write("mixed", None, "mixed"),
                            ],
                            &background_context(),
                        )
                        .await;
                    assert!(
                        rejected.is_err(),
                        "expected the mixed list transaction to reject"
                    );
                    assert_eq!(
                        storage
                            .read_list(&address.address, None, &background_context())
                            .await
                            .expect("list"),
                        vec![list_element(committed.seqs[1], json!("kept"))],
                    );
                    assert_eq!(
                        storage
                            .get_value(&test_name().address, &background_context())
                            .await
                            .expect("value")
                            .expect("kept value")
                            .value,
                        json!("kept"),
                    );
                })
            }),
        ),
        case(
            factory,
            "entry queries",
            "stores custom entries with and without data",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let result = commit_ok(
                        &*storage,
                        vec![
                            insert_entry_write(custom_entry("without-data", None, "marker", None)),
                            insert_entry_write(custom_entry(
                                "with-data",
                                Some("without-data".to_owned()),
                                "note",
                                Some(json!({ "nested": [1, 2] })),
                            )),
                        ],
                    )
                    .await;

                    assert_eq!(
                        storage
                            .get_entries(
                                vec!["without-data".to_owned(), "with-data".to_owned()],
                                &background_context(),
                            )
                            .await
                            .expect("entries"),
                        stored_map(vec![
                            (
                                "without-data",
                                custom_entry("without-data", None, "marker", None)
                                    .materialize(result.seqs[0], result.timestamp)
                            ),
                            (
                                "with-data",
                                custom_entry(
                                    "with-data",
                                    Some("without-data".to_owned()),
                                    "note",
                                    Some(json!({ "nested": [1, 2] })),
                                )
                                .materialize(result.seqs[1], result.timestamp),
                            ),
                        ]),
                    );
                })
            }),
        ),
        case(
            factory,
            "entry queries",
            "scans global entries with explicit ranges, filters, orders, and limits",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let result = commit_ok(
                        &*storage,
                        vec![
                            user_entry_write("root", None, "root"),
                            custom_entry_write("note-1", Some("root"), "note"),
                            custom_entry_write("other", Some("note-1"), "other"),
                            custom_entry_write("note-2", Some("other"), "note"),
                            user_entry_write("tail", Some("note-2"), "tail"),
                        ],
                    )
                    .await;

                    assert_eq!(
                        scan_entry_ids(
                            &*storage,
                            &crate::harness::session::types::EntryScan {
                                kind: Some(crate::harness::session::types::EntryType::Custom),
                                custom_type: Some("note".to_owned()),
                                from_seq: Some(result.seqs[1]),
                                to_seq: Some(result.seqs[3]),
                                order: Some(crate::harness::session::types::EntryScanOrder::Desc),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["note-2", "note-1"],
                    );
                    assert_eq!(
                        scan_entry_ids(
                            &*storage,
                            &crate::harness::session::types::EntryScan {
                                order: Some(crate::harness::session::types::EntryScanOrder::Asc),
                                limit: Some(2),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["root", "note-1"],
                    );
                    assert_eq!(
                        scan_entry_ids(
                            &*storage,
                            &crate::harness::session::types::EntryScan {
                                order: Some(crate::harness::session::types::EntryScanOrder::Desc),
                                limit: Some(2),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["tail", "note-2"],
                    );
                })
            }),
        ),
        case(
            factory,
            "branch queries",
            "applies stops before filters and cursors before limits",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let result = commit_ok(&*storage, branch_query_seed_writes()).await;

                    assert_eq!(
                        scan_branch_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                stop_at_type: Some(
                                    crate::harness::session::types::EntryType::Compaction
                                ),
                                kind: Some(crate::harness::session::types::EntryType::Message),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["leaf"],
                    );
                    assert_eq!(
                        scan_branch_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                order: Some(
                                    crate::harness::session::types::BranchScanOrder::OldestFirst
                                ),
                                stop_at_id: Some("middle".to_owned()),
                                kind: Some(crate::harness::session::types::EntryType::Custom),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["marker"],
                    );
                    assert_eq!(
                        scan_branch_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                order: Some(
                                    crate::harness::session::types::BranchScanOrder::NewestFirst
                                ),
                                cursor: Some(crate::harness::session::types::EntryCursor {
                                    seq: result.seqs[4]
                                }),
                                limit: Some(2),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["compact", "middle"],
                    );
                    assert_eq!(
                        scan_branch_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                order: Some(
                                    crate::harness::session::types::BranchScanOrder::OldestFirst
                                ),
                                cursor: Some(crate::harness::session::types::EntryCursor {
                                    seq: result.seqs[1]
                                }),
                                limit: Some(2),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["middle", "compact"],
                    );
                    assert_eq!(
                        scan_branch_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                stop_at_id: Some("leaf".to_owned()),
                                kind: Some(crate::harness::session::types::EntryType::Custom),
                                ..Default::default()
                            }
                        )
                        .await,
                        Vec::<String>::new(),
                    );
                    assert_eq!(
                        scan_branch_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                custom_type: Some("note".to_owned()),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["note"],
                    );
                    let rejected = storage
                        .scan_branch(
                            &StorageBranchScan {
                                start: "missing".to_owned(),
                                ..Default::default()
                            },
                            &background_context(),
                        )
                        .await;
                    assert!(rejected.is_err(), "expected the unknown start to reject");
                })
            }),
        ),
        case(
            factory,
            "branch queries",
            "returns branch structure without payload fields",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let result = commit_ok(
                        &*storage,
                        vec![
                            user_entry_write("root", None, "root"),
                            custom_entry_write("child", Some("root"), "note"),
                        ],
                    )
                    .await;

                    assert_eq!(
                        storage
                            .scan_branch_structure(
                                &StorageBranchScan {
                                    start: "child".to_owned(),
                                    order: Some(crate::harness::session::types::BranchScanOrder::OldestFirst),
                                    ..Default::default()
                                },
                                &background_context(),
                            )
                            .await
                            .expect("structure"),
                        vec![
                            crate::harness::session::types::EntryStructure {
                                id: "root".to_owned(),
                                parent_id: None,
                                seq: result.seqs[0],
                                timestamp: result.timestamp,
                                kind: crate::harness::session::types::EntryType::Message,
                                custom_type: None,
                            },
                            crate::harness::session::types::EntryStructure {
                                id: "child".to_owned(),
                                parent_id: Some("root".to_owned()),
                                seq: result.seqs[1],
                                timestamp: result.timestamp,
                                kind: crate::harness::session::types::EntryType::Custom,
                                custom_type: Some("note".to_owned()),
                            },
                        ],
                    );
                })
            }),
        ),
        case(
            factory,
            "branch queries",
            "applies branch query semantics to structure scans",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let result = commit_ok(&*storage, branch_query_seed_writes()).await;

                    assert_eq!(
                        scan_structure_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                stop_at_type: Some(
                                    crate::harness::session::types::EntryType::Compaction
                                ),
                                kind: Some(crate::harness::session::types::EntryType::Message),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["leaf"],
                    );
                    assert_eq!(
                        scan_structure_ids(
                            &*storage,
                            &StorageBranchScan {
                                start: "leaf".to_owned(),
                                order: Some(
                                    crate::harness::session::types::BranchScanOrder::OldestFirst
                                ),
                                cursor: Some(crate::harness::session::types::EntryCursor {
                                    seq: result.seqs[1]
                                }),
                                limit: Some(2),
                                ..Default::default()
                            }
                        )
                        .await,
                        ["middle", "compact"],
                    );
                    let rejected = storage
                        .scan_branch_structure(
                            &StorageBranchScan {
                                start: "missing".to_owned(),
                                ..Default::default()
                            },
                            &background_context(),
                        )
                        .await;
                    assert!(rejected.is_err(), "expected the unknown start to reject");
                })
            }),
        ),
        case(
            factory,
            "usage and stats",
            "scans the usage ledger with explicit ranges, orders, and limits",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let result = commit_ok(&*storage, usage_ledger_writes()).await;

                    assert_eq!(
                        storage
                            .scan_usage(
                                &UsageScan {
                                    from_seq: Some(result.seqs[1]),
                                    to_seq: Some(result.seqs[2]),
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Asc
                                    ),
                                    ..Default::default()
                                },
                                &background_context(),
                            )
                            .await
                            .expect("usage")
                            .into_iter()
                            .map(|row| row.id)
                            .collect::<Vec<_>>(),
                        ["usage-2"],
                    );
                    assert_eq!(
                        storage
                            .scan_usage(
                                &UsageScan {
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Desc
                                    ),
                                    limit: Some(2),
                                    ..Default::default()
                                },
                                &background_context()
                            )
                            .await
                            .expect("usage")
                            .into_iter()
                            .map(|row| row.id)
                            .collect::<Vec<_>>(),
                        ["usage-3", "usage-2"],
                    );
                    assert_eq!(
                        storage
                            .scan_usage(
                                &UsageScan {
                                    order: Some(
                                        crate::harness::session::types::EntryScanOrder::Asc
                                    ),
                                    limit: Some(2),
                                    ..Default::default()
                                },
                                &background_context()
                            )
                            .await
                            .expect("usage")
                            .into_iter()
                            .map(|row| row.id)
                            .collect::<Vec<_>>(),
                        ["usage-1", "usage-2"],
                    );
                })
            }),
        ),
        case(
            factory,
            "usage and stats",
            "keeps stats equal to message count and ledger totals",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    assert_eq!(
                        storage
                            .get_stats(&background_context())
                            .await
                            .expect("stats"),
                        crate::harness::session::types::SessionStats {
                            message_count: 0,
                            usage: zero_usage(),
                        },
                    );

                    let first_usage =
                        crate::harness::session::testing::conformance::usage_with_extras(
                            2,
                            3,
                            Some(4),
                            Some(1),
                        );
                    let first = commit_ok(
                        &*storage,
                        vec![
                            user_entry_write("message", None, "message"),
                            insert_usage_write("usage-1", first_usage, false, None),
                        ],
                    )
                    .await;
                    assert_commit_stats(&*storage, &first).await;
                    assert_eq!(
                        first.stats,
                        crate::harness::session::types::SessionStats {
                            message_count: 1,
                            usage: first_usage,
                        },
                    );

                    let second_usage =
                        crate::harness::session::testing::conformance::usage_with_extras(
                            5,
                            7,
                            Some(6),
                            Some(2),
                        );
                    let second = commit_ok(
                        &*storage,
                        vec![
                            custom_entry_write("custom", Some("message"), "note"),
                            compaction_entry_write("compaction", Some("custom")),
                            insert_usage_write("usage-2", second_usage, true, None),
                        ],
                    )
                    .await;
                    assert_commit_stats(&*storage, &second).await;
                    assert_eq!(
                        second.stats,
                        crate::harness::session::types::SessionStats {
                            message_count: 1,
                            usage: pi_ai::types::Usage {
                                input: 7,
                                output: 10,
                                cache_read: 9,
                                cache_write: 12,
                                cache_write_1h: Some(10),
                                reasoning: Some(3),
                                total_tokens: 17,
                                cost: pi_ai::types::UsageCost {
                                    input: first_usage.cost.input + second_usage.cost.input,
                                    output: first_usage.cost.output + second_usage.cost.output,
                                    cache_read: first_usage.cost.cache_read
                                        + second_usage.cost.cache_read,
                                    cache_write: first_usage.cost.cache_write
                                        + second_usage.cost.cache_write,
                                    total: first_usage.cost.total + second_usage.cost.total,
                                },
                            },
                        },
                    );
                })
            }),
        ),
        case(
            factory,
            "serialization",
            "serializes back-to-back commits in admission order",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let first = storage.commit(
                        vec![user_entry_write("first", None, "first")],
                        &background_context(),
                    );
                    let second = storage.commit(
                        vec![user_entry_write("second", Some("first"), "second")],
                        &background_context(),
                    );
                    let (first_result, second_result) = tokio::join!(first, second);

                    let first_result = first_result.expect("first commit");
                    let second_result = second_result.expect("second commit");
                    assert!(first_result.seqs[0] < second_result.seqs[0]);
                    assert_eq!(
                        first_result.stats,
                        crate::harness::session::types::SessionStats {
                            message_count: 1,
                            usage: zero_usage(),
                        },
                    );
                    assert_eq!(
                        second_result.stats,
                        crate::harness::session::types::SessionStats {
                            message_count: 2,
                            usage: zero_usage(),
                        },
                    );
                    assert_commit_stats(&*storage, &second_result).await;
                    assert_eq!(
                        ids(&storage
                            .scan_entries(&asc_scan(), &background_context())
                            .await
                            .expect("entries")),
                        ["first", "second"],
                    );
                })
            }),
        ),
        case(
            factory,
            "lifecycle",
            "seals admission, drains admitted commits, and closes idempotently",
            Arc::new(|storage: &dyn Storage| -> BoxedFuture<'_, ()> {
                Box::pin(async move {
                    let admitted = storage.commit(
                        vec![user_entry_write("admitted", None, "admitted")],
                        &background_context(),
                    );
                    let first_close = storage.close(&background_context());
                    let second_close = storage.close(&background_context());

                    let rejected = storage.get_stats(&background_context()).await;
                    assert!(rejected.is_err(), "expected getStats after close to reject");
                    let rejected = storage.commit(Vec::new(), &background_context()).await;
                    assert!(rejected.is_err(), "expected commit after close to reject");
                    assert_eq!(admitted.await.expect("admitted commit").seqs.len(), 1);
                    let (first_closed, second_closed) = tokio::join!(first_close, second_close);
                    assert!(first_closed.is_ok() && second_closed.is_ok());

                    let (
                        entries,
                        value,
                        values,
                        list,
                        branch,
                        structure,
                        flat_entries,
                        flat_usage,
                        stats,
                    ) = (
                        storage.get_entries(Vec::new(), &background_context()),
                        storage.get_value(&test_name().address, &background_context()),
                        storage.scan_values(&test_name().address, &background_context()),
                        storage.read_list(
                            &test_list("events").address,
                            None,
                            &background_context(),
                        ),
                        storage.scan_branch(
                            &StorageBranchScan {
                                start: "admitted".to_owned(),
                                ..Default::default()
                            },
                            &background_context(),
                        ),
                        storage.scan_branch_structure(
                            &StorageBranchScan {
                                start: "admitted".to_owned(),
                                ..Default::default()
                            },
                            &background_context(),
                        ),
                        storage.scan_entries(&asc_scan(), &background_context()),
                        storage.scan_usage(&asc_usage_scan(), &background_context()),
                        storage.get_stats(&background_context()),
                    );
                    for read in [
                        entries.await.err(),
                        value.await.err(),
                        values.await.err(),
                        list.await.err(),
                        branch.await.err(),
                        structure.await.err(),
                        flat_entries.await.err(),
                        flat_usage.await.err(),
                        stats.await.err(),
                    ] {
                        assert!(read.is_some(), "expected every read after close to reject");
                    }
                })
            }),
        ),
    ]
}
