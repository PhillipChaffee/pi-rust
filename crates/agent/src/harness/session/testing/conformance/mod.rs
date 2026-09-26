//! The runner-independent conformance suites, ported from upstream
//! `src/harness/session/testing/conformance/`.
//!
//! They carry the durable Storage contract and the `SessionRepo`
//! lifecycle/fork surface as
//! [`ConformanceCase`](crate::harness::session::testing::ConformanceCase)
//! values any test runner registers.

#![expect(
    clippy::panic,
    reason = "conformance cases panic on a broken fixture contract, upstream's node:assert throws"
)]
#![expect(
    clippy::expect_used,
    reason = "the fixture helpers carry the cases' expects; a failure is a bug the case panics on"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "the fixture usage rows' dollar costs divide small token counts; the f64 rounding upstream's JS arithmetic produces is the fixture"
)]

pub mod session_repo;
pub mod storage;

use pi_ai::types::Usage;

use crate::harness::session::commit::insert_entry;
use crate::harness::session::commit::insert_usage;
use crate::harness::session::types::{
    CustomEntryBody, Entry, MessageEntry, NewEntry, UsageWriteRow,
};
use crate::harness::session::values::{self, Write};
use crate::types::AgentMessage;

/// The fixed message timestamp the fixture entries carry, upstream's
/// `MESSAGE_TIMESTAMP`.
pub const MESSAGE_TIMESTAMP: i64 = 1_650_000_000_000;

/// The `test.session.name` address, upstream's `testName`.
#[must_use]
pub fn test_name() -> values::Value<serde_json::Value> {
    values::generic_value("test.session.name", "")
}

/// The `test.value` address, upstream's `testValue`.
#[must_use]
pub fn test_value(key: &str) -> values::Value<serde_json::Value> {
    values::generic_value("test.value", key)
}

/// The `test.value` prefix address, upstream's `testValuePrefix`.
#[must_use]
pub fn test_value_prefix(prefix: &str) -> values::Value<serde_json::Value> {
    values::generic_value("test.value", prefix)
}

/// The `test.list` address, upstream's `testList`.
#[must_use]
pub fn test_list(key: &str) -> values::ValueList<serde_json::Value> {
    values::generic_list("test.list", key)
}

/// The usage row a fixture builds, upstream's `usage(input, output, options)`.
#[must_use]
pub fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read: input + 1,
        cache_write: output + 1,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output,
        cost: pi_ai::types::UsageCost {
            input: input as f64 / 100.0,
            output: output as f64 / 100.0,
            cache_read: (input + 1) as f64 / 100.0,
            cache_write: (output + 1) as f64 / 100.0,
            total: (input + output + 2) as f64 / 100.0,
        },
    }
}

/// The zero usage row, upstream's `zeroUsage`.
#[must_use]
pub fn zero_usage() -> Usage {
    Usage::default()
}

/// The usage row with the optional components a fixture sets, upstream's
/// `usage(input, output, { cacheWrite1h, reasoning })`.
#[must_use]
pub fn usage_with_extras(
    input: u64,
    output: u64,
    cache_write_1h: Option<u64>,
    reasoning: Option<u64>,
) -> Usage {
    Usage {
        cache_write_1h,
        reasoning,
        ..usage(input, output)
    }
}

/// The entries map a fixture compares against, upstream's `new Map(...)`.
#[must_use]
pub fn stored_map<S: Into<String>, E>(
    entries: impl IntoIterator<Item = (S, E)>,
) -> std::collections::BTreeMap<String, E> {
    entries
        .into_iter()
        .map(|(id, entry)| (id.into(), entry))
        .collect()
}

/// The ascending entry scan, upstream's `{ order: "asc" }` literal.
#[must_use]
pub fn asc_scan() -> crate::harness::session::types::EntryScan {
    crate::harness::session::types::EntryScan {
        order: Some(crate::harness::session::types::EntryScanOrder::Asc),
        ..Default::default()
    }
}

/// The ascending usage scan, upstream's `{ order: "asc" }` literal.
#[must_use]
pub fn asc_usage_scan() -> crate::harness::session::types::UsageScan {
    crate::harness::session::types::UsageScan {
        order: Some(crate::harness::session::types::EntryScanOrder::Asc),
        ..Default::default()
    }
}

/// The message entry a fixture builds, upstream's `userEntry`.
#[must_use]
pub fn user_entry(id: &str, parent_id: Option<String>, text: &str) -> NewEntry {
    NewEntry::Message {
        id: id.to_owned(),
        parent_id,
        body: Box::new(MessageEntry {
            message: user_message(text),
            terminate: None,
        }),
    }
}

/// The custom entry a fixture builds, upstream's `customEntry`.
#[must_use]
pub fn custom_entry(
    id: &str,
    parent_id: Option<String>,
    custom_type: &str,
    data: Option<serde_json::Value>,
) -> NewEntry {
    NewEntry::Custom {
        id: id.to_owned(),
        parent_id,
        body: CustomEntryBody {
            custom_type: custom_type.to_owned(),
            data,
        },
    }
}

/// The compaction entry a fixture builds, upstream's `compactionEntry`.
#[must_use]
pub fn compaction_entry(id: &str, parent_id: Option<String>) -> NewEntry {
    NewEntry::Compaction {
        id: id.to_owned(),
        parent_id,
        body: crate::harness::session::types::CompactionEntryBody {
            summary: format!("summary:{id}"),
            retained_tail: Vec::new(),
            tokens_before: 10,
            details: None,
            usage: None,
            from_hook: false,
        },
    }
}

/// The entry ids of one scan, upstream's `ids` (both `Entry` and
/// `EntryStructure` carry `id`).
pub trait EntryIds {
    /// The entry id.
    fn entry_id(&self) -> String;
}

impl EntryIds for Entry {
    fn entry_id(&self) -> String {
        self.id().to_owned()
    }
}

impl EntryIds for crate::harness::session::types::EntryStructure {
    fn entry_id(&self) -> String {
        self.id.clone()
    }
}

/// The ids of one scan, upstream's `ids`.
pub fn ids<T: EntryIds>(entries: &[T]) -> Vec<String> {
    entries.iter().map(EntryIds::entry_id).collect()
}

/// Whether a scan's sequences are strictly increasing, upstream's
/// `assertStrictlyIncreasing`.
///
/// # Panics
/// When any adjacent pair is not strictly increasing.
pub fn assert_strictly_increasing(values: &[u64]) {
    for pair in values.windows(2) {
        assert!(
            pair[0] < pair[1],
            "Expected {values:?} to be strictly increasing"
        );
    }
}

/// One insert-entry write, the suites' shorthand for upstream's
/// `insertEntry(...)`.
#[must_use]
pub fn insert_entry_write(entry: NewEntry) -> Write {
    Write::Entry(Box::new(insert_entry(entry)))
}

/// One insert-usage write, the suites' shorthand for upstream's
/// `insertUsage(...)`.
#[must_use]
pub fn insert_usage_write(
    id: &str,
    usage: Usage,
    adjustment: bool,
    entry_id: Option<String>,
) -> Write {
    Write::Usage(insert_usage(UsageWriteRow {
        id: id.to_owned(),
        usage,
        entry_id,
        adjustment,
        details: None,
    }))
}

/// The user message a fixture builds, upstream's `{ role: "user", content:
/// [{ type: "text", text }], timestamp }`.
#[must_use]
pub fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(pi_ai::types::Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Blocks(vec![pi_ai::types::UserBlock::Text(
            pi_ai::types::TextContent {
                text: text.to_owned(),
                text_signature: None,
            },
        )]),
        timestamp: MESSAGE_TIMESTAMP,
    }))
}

/// The commit result one seeded callback returned, upstream's awaited
/// `mutator.commit(...)` return value.
///
/// # Panics
/// When the boxed result is not a commit result — a broken fixture.
#[must_use]
pub fn downcast_commit_result(
    result: Box<dyn std::any::Any + Send>,
) -> crate::harness::session::types::CommitResult {
    result
        .downcast::<crate::harness::session::types::CommitResult>()
        .map_or_else(
            |_| panic!("expected the seeded callback to return its commit result"),
            |result| *result,
        )
}

/// The commit module the suites alias, upstream's
/// `* as sessionWrites from "../../commit.ts"`.
pub use crate::harness::session::commit as session_writes;
/// The stored-value module the suites alias, upstream's
/// `* as storedValues from "../../values.ts"`.
pub use crate::harness::session::values as stored_values;

/// The session creation the repo suites issue, upstream's
/// `repo.create({ id }, ctx)` with the fixture's expect.
///
/// # Panics
/// When the repo rejects the creation.
pub async fn create_repo_session(
    repo: &dyn crate::harness::session::types::SessionRepo,
    id: Option<&str>,
    parent_session_id: Option<&str>,
) -> Box<dyn crate::harness::session::types::Session> {
    repo.create(
        crate::harness::session::types::SessionCreateOptions {
            id: id.map(str::to_owned),
            parent_session_id: parent_session_id.map(str::to_owned),
        },
        &crate::harness::context::background_context(),
    )
    .await
    .expect("create")
}

/// The branch-fork options fixture, upstream's `{ scope: "branch", ... }`
/// literals.
#[must_use]
pub fn branch_fork(
    branch: &str,
    entry_id: Option<&str>,
    position: Option<crate::harness::session::types::ForkPosition>,
    id: Option<&str>,
) -> crate::harness::session::types::ForkOptions {
    crate::harness::session::types::ForkOptions::Branch {
        branch: branch.to_owned(),
        entry_id: entry_id.map(str::to_owned),
        position,
        id: id.map(str::to_owned),
    }
}

/// The tree-fork options fixture, upstream's `{ scope: "tree", id }`.
#[must_use]
pub fn tree_fork(id: &str) -> crate::harness::session::types::ForkOptions {
    crate::harness::session::types::ForkOptions::Tree {
        id: Some(id.to_owned()),
    }
}

/// The session-wide entry ids one query returns, upstream's
/// `findEntries({ order: "asc" })` id maps.
///
/// # Panics
/// When the query fails.
pub async fn find_entry_ids(
    session: &dyn crate::harness::session::types::Session,
    query: &crate::harness::session::types::EntryQuery,
) -> Vec<String> {
    session
        .find_entries(Some(query), &crate::harness::context::background_context())
        .await
        .expect("entries")
        .into_iter()
        .map(|entry| entry.id().to_owned())
        .collect()
}

/// The session close the suites issue, upstream's
/// `session.close(BACKGROUND_CONTEXT)` with the fixture's expect.
///
/// # Panics
/// When the close fails.
pub async fn close_session(session: &dyn crate::harness::session::types::Session) {
    session
        .close(&crate::harness::context::background_context())
        .await
        .expect("close");
}

/// The commit the storage cases issue, upstream's
/// `storage.commit(writes, BACKGROUND_CONTEXT)` with the fixture's expect.
///
/// # Panics
/// When the commit fails.
pub async fn commit_ok(
    storage: &dyn crate::harness::session::types::Storage,
    writes: Vec<Write>,
) -> crate::harness::session::types::CommitResult {
    storage
        .commit(writes, &crate::harness::context::background_context())
        .await
        .expect("commit")
}

/// The stored value a fixture compares, upstream's
/// `{ address, value, seq }` literals.
#[must_use]
pub fn stored_value(
    address: &values::ValueAddress,
    value: serde_json::Value,
    seq: u64,
) -> values::StoredValue {
    values::StoredValue {
        namespace: address.namespace.clone(),
        key: address.key.clone(),
        value,
        seq,
    }
}

/// The list element a fixture compares, upstream's `{ seq, value }`.
#[must_use]
pub const fn list_element(seq: u64, value: serde_json::Value) -> values::ListElement {
    values::ListElement { seq, value }
}

/// Asserts the address holds nothing, upstream's
/// `strictEqual(await getValue(...), undefined)`.
///
/// # Panics
/// When the address holds a value or the read fails.
pub async fn assert_value_absent(
    storage: &dyn crate::harness::session::types::Storage,
    address: &values::ValueAddress,
) {
    assert!(
        storage
            .get_value(address, &crate::harness::context::background_context())
            .await
            .expect("value")
            .is_none(),
    );
}

/// Asserts a read list holds exactly the given element values, upstream's
/// `deepStrictEqual(await readList(...), [{ value }])` value maps.
///
/// # Panics
/// When the read fails or the values differ.
pub async fn assert_list_values(
    storage: &dyn crate::harness::session::types::Storage,
    address: &values::ListAddress,
    expected: &[serde_json::Value],
) {
    let elements = storage
        .read_list(
            address,
            None,
            &crate::harness::context::background_context(),
        )
        .await
        .expect("list");
    let got: Vec<serde_json::Value> = elements.into_iter().map(|element| element.value).collect();
    assert_eq!(got, expected);
}

/// The branch-tip write a fixture builds, upstream's
/// `setValue(branchTip(branch), tip)`.
///
/// # Panics
/// Never: the reserved address validates by construction and the value is
/// a plain string.
#[must_use]
pub fn tip_write(branch: &str, tip: Option<&str>) -> Write {
    Write::ValueSet(
        stored_values::set_value(&stored_values::branch_tip(branch), tip.map(str::to_owned))
            .expect("write"),
    )
}

/// The lane-configuration write a fixture builds, upstream's
/// `setValue(laneConfig(branch), configuration)`.
///
/// # Panics
/// Never: the reserved address validates by construction.
#[must_use]
pub fn lane_config_write(
    branch: &str,
    configuration: &crate::harness::session::types::LaneConfiguration,
) -> Write {
    Write::ValueSet(
        stored_values::set_value(&stored_values::lane_config(branch), configuration.clone())
            .expect("write"),
    )
}

/// The lane-state write a fixture builds, upstream's
/// `setValue(laneState(branch), state)`.
///
/// # Panics
/// Never: the reserved address validates by construction.
#[must_use]
pub fn lane_state_write(branch: &str, state: &crate::harness::session::types::LaneState) -> Write {
    Write::ValueSet(
        stored_values::set_value(&stored_values::lane_state(branch), state.clone()).expect("write"),
    )
}

/// The entry-label write a fixture builds, upstream's
/// `setValue(entryLabel(entryId), label)`.
///
/// # Panics
/// Never: the reserved address validates by construction.
#[must_use]
pub fn label_write(entry_id: &str, label: &str) -> Write {
    Write::ValueSet(
        stored_values::set_value(&stored_values::entry_label(entry_id), label.to_owned())
            .expect("write"),
    )
}

/// The seeded custom entry's write, upstream's
/// `insertEntry({ id, parentId, type: "custom", customType })` without data.
#[must_use]
pub fn seed_custom_entry(id: &str, parent_id: Option<&str>, custom_type: &str) -> Write {
    insert_entry_write(custom_entry(
        id,
        parent_id.map(str::to_owned),
        custom_type,
        None,
    ))
}

/// Asserts a session's lane configuration matches the fixture, upstream's
/// `deepStrictEqual((await getValue(laneConfig(branch)))?.value, configuration)`.
///
/// # Panics
/// When the read fails or the value differs.
pub async fn assert_lane_config(
    session: &dyn crate::harness::session::types::Session,
    branch: &str,
    configuration: &crate::harness::session::types::LaneConfiguration,
) {
    let stored = session
        .get_value(
            &stored_values::lane_config(branch).address,
            &crate::harness::context::background_context(),
        )
        .await
        .expect("value")
        .expect("configured lane");
    assert_eq!(
        stored.value,
        serde_json::to_value(configuration).expect("config wire")
    );
}

/// Asserts a session's lane state matches the fixture, upstream's
/// `deepStrictEqual((await getValue(laneState(branch)))?.value, state)`.
///
/// # Panics
/// When the read fails or the value differs.
pub async fn assert_lane_state(
    session: &dyn crate::harness::session::types::Session,
    branch: &str,
    state: &crate::harness::session::types::LaneState,
) {
    let stored = session
        .get_value(
            &stored_values::lane_state(branch).address,
            &crate::harness::context::background_context(),
        )
        .await
        .expect("value")
        .expect("lane state");
    assert_eq!(
        stored.value,
        serde_json::to_value(state).expect("state wire")
    );
}

/// Asserts a session holds nothing at the address, upstream's
/// `strictEqual(await getValue(...), undefined)`.
///
/// # Panics
/// When the read fails or the address holds a value.
pub async fn assert_session_value_absent(
    session: &dyn crate::harness::session::types::Session,
    address: &values::ValueAddress,
) {
    assert!(
        session
            .get_value(address, &crate::harness::context::background_context())
            .await
            .expect("value")
            .is_none(),
    );
}

/// Seeds the usage-ledger commit the scan and stats cases share, upstream's
/// inline `insertUsage` transaction shapes.
///
/// # Panics
/// Never: the fixture writes are fixed and valid.
#[must_use]
pub fn usage_ledger_writes() -> Vec<Write> {
    vec![
        insert_usage_write("usage-1", usage(1, 1), false, None),
        Write::ValueSet(
            stored_values::set_value(&test_name(), serde_json::json!("sequence gap"))
                .expect("write"),
        ),
        insert_usage_write("usage-2", usage(2, 2), false, None),
        insert_usage_write("usage-3", usage(3, 3), true, None),
    ]
}

/// The historical-state snapshot the rollback cases compare, upstream's
/// `entriesBefore`/`usageBefore`/`statsBefore` tuples.
#[derive(Debug)]
pub struct HistoricalSnapshot {
    /// The flat entry scan before the transaction.
    pub entries: Vec<Entry>,
    /// The usage scan before the transaction.
    pub usage: Vec<crate::harness::session::types::UsageRow>,
    /// The totals before the transaction.
    pub stats: crate::harness::session::types::SessionStats,
}

/// Snapshots the historical stores, upstream's inline triples.
///
/// # Panics
/// When a read fails.
pub async fn snapshot_historical_state(
    storage: &dyn crate::harness::session::types::Storage,
) -> HistoricalSnapshot {
    let context = &crate::harness::context::background_context();
    HistoricalSnapshot {
        entries: storage
            .scan_entries(&asc_scan(), context)
            .await
            .expect("entries"),
        usage: storage
            .scan_usage(&asc_usage_scan(), context)
            .await
            .expect("usage"),
        stats: storage.get_stats(context).await.expect("stats"),
    }
}

/// Asserts the historical stores are unchanged, upstream's three
/// `deepStrictEqual` compares.
///
/// # Panics
/// When any store differs from the snapshot.
pub async fn assert_historical_unchanged(
    storage: &dyn crate::harness::session::types::Storage,
    snapshot: &HistoricalSnapshot,
) {
    let context = &crate::harness::context::background_context();
    assert_eq!(
        storage
            .scan_entries(&asc_scan(), context)
            .await
            .expect("entries"),
        snapshot.entries,
    );
    assert_eq!(
        storage
            .scan_usage(&asc_usage_scan(), context)
            .await
            .expect("usage"),
        snapshot.usage,
    );
    assert_eq!(
        storage.get_stats(context).await.expect("stats"),
        snapshot.stats
    );
}

/// The entry ids one flat scan returns, upstream's
/// `ids(await storage.scanEntries(query, ctx))`.
///
/// # Panics
/// When the scan fails.
pub async fn scan_entry_ids(
    storage: &dyn crate::harness::session::types::Storage,
    query: &crate::harness::session::types::EntryScan,
) -> Vec<String> {
    ids(&storage
        .scan_entries(query, &crate::harness::context::background_context())
        .await
        .expect("scan"))
}

/// The entry ids one branch scan returns, upstream's
/// `ids(await storage.scanBranch(query, ctx))`.
///
/// # Panics
/// When the scan fails.
pub async fn scan_branch_ids(
    storage: &dyn crate::harness::session::types::Storage,
    query: &crate::harness::session::types::StorageBranchScan,
) -> Vec<String> {
    ids(&storage
        .scan_branch(query, &crate::harness::context::background_context())
        .await
        .expect("scan"))
}

/// The entry ids one structural branch scan returns, upstream's
/// `ids(await storage.scanBranchStructure(query, ctx))`.
///
/// # Panics
/// When the scan fails.
pub async fn scan_structure_ids(
    storage: &dyn crate::harness::session::types::Storage,
    query: &crate::harness::session::types::StorageBranchScan,
) -> Vec<String> {
    ids(&storage
        .scan_branch_structure(query, &crate::harness::context::background_context())
        .await
        .expect("structure"))
}

/// The custom entry's write with the fixture's data payload, upstream's
/// `insertEntry({ ..., customType, data: { id } })` shapes.
#[must_use]
pub fn custom_entry_write(id: &str, parent_id: Option<&str>, custom_type: &str) -> Write {
    insert_entry_write(custom_entry(
        id,
        parent_id.map(str::to_owned),
        custom_type,
        Some(serde_json::json!({ "id": id })),
    ))
}

/// The message entry's write, upstream's `insertEntry(userEntry(...))`.
#[must_use]
pub fn user_entry_write(id: &str, parent_id: Option<&str>, text: &str) -> Write {
    insert_entry_write(user_entry(id, parent_id.map(str::to_owned), text))
}

/// The tree fork the streaming-fork cases issue after their optional close,
/// upstream's `if (sourceState === "closed") await source.close(...); fork =
/// await repo.fork(...)` sequence.
///
/// # Panics
/// When the fork fails.
pub async fn fork_tree_session(
    repo: &dyn crate::harness::session::types::SessionRepo,
    source: &dyn crate::harness::session::types::Session,
    id: &str,
    close_first: bool,
) -> Box<dyn crate::harness::session::types::Session> {
    if close_first {
        close_session(source).await;
    }
    repo.fork(
        source.metadata(),
        &tree_fork(id),
        &crate::harness::context::background_context(),
    )
    .await
    .expect("fork")
}

/// The branch fork the repo suites issue, upstream's
/// `repo.fork(source.metadata(), { scope: "branch", ... }, ctx)` with the
/// fixture's expect.
///
/// # Panics
/// When the fork fails.
pub async fn fork_branch_session(
    repo: &dyn crate::harness::session::types::SessionRepo,
    source: &dyn crate::harness::session::types::Session,
    branch: &str,
    entry_id: Option<&str>,
    id: Option<&str>,
) -> Box<dyn crate::harness::session::types::Session> {
    repo.fork(
        source.metadata(),
        &branch_fork(branch, entry_id, None, id),
        &crate::harness::context::background_context(),
    )
    .await
    .expect("fork")
}

/// The compaction entry's write, upstream's
/// `insertEntry(compactionEntry(id, parentId))`.
#[must_use]
pub fn compaction_entry_write(id: &str, parent_id: Option<&str>) -> Write {
    insert_entry_write(compaction_entry(id, parent_id.map(str::to_owned)))
}

/// The ascending session-wide query, upstream's `{ order: "asc" }` literal.
#[must_use]
pub fn asc_query() -> crate::harness::session::types::EntryQuery {
    crate::harness::session::types::EntryQuery {
        order: Some(crate::harness::session::types::EntryScanOrder::Asc),
        ..Default::default()
    }
}

/// Asserts a session's list holds exactly the given element values, the
/// Session-side twin of [`assert_list_values`], upstream's
/// `deepStrictEqual((await readList(...)).map(({ value }) => value), [...])`.
///
/// # Panics
/// When the read fails or the values differ.
pub async fn assert_session_list_values(
    session: &dyn crate::harness::session::types::Session,
    address: &values::ListAddress,
    expected: &[serde_json::Value],
) {
    let elements = session
        .read_list(
            address,
            None,
            &crate::harness::context::background_context(),
        )
        .await
        .expect("list");
    let got: Vec<serde_json::Value> = elements.into_iter().map(|element| element.value).collect();
    assert_eq!(got, expected);
}

/// The six-entry branch path the branch-query cases seed, upstream's
/// root/marker/middle/compact/note/leaf chain.
#[must_use]
pub fn branch_query_seed_writes() -> Vec<Write> {
    vec![
        user_entry_write("root", None, "root"),
        custom_entry_write("marker", Some("root"), "marker"),
        user_entry_write("middle", Some("marker"), "middle"),
        compaction_entry_write("compact", Some("middle")),
        custom_entry_write("note", Some("compact"), "note"),
        user_entry_write("leaf", Some("note"), "leaf"),
    ]
}
