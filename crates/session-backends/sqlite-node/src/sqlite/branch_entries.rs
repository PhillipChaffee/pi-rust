//! The private branch projections, upstream's
//! `src/sqlite/session/branch-entries.ts`.
//!
//! Authoritative durable state is `entries` plus the value tables; the
//! `branch_entries`/`branch_meta` tables are the maintained branch index this
//! module writes and reads. A branch's id is the id of its first entry; an
//! entry belongs to a branch iff `base_seq < entry_seq <= tip_seq`.

use pi_agent_core::harness::session::types::{
    BranchScanOrder, Entry, EntryStructure, SessionError, StorageBranchScan,
};

use crate::sql;
use crate::sqlite::entries::{decode_entry_row, entry_row, entry_structure_row, wire_of};
use crate::sqlite::sql::{SqlQuery, join_sql_fragments};
use crate::sqlite::types::{SqliteDatabase, SqliteRow, sql_integer};

/// One contiguous branch segment between a base boundary and a tip, upstream's
/// `BranchSegment`.
#[derive(Clone, Debug)]
struct BranchSegment {
    branch_id: String,
    lower_seq: i64,
    upper_seq: i64,
}

/// Reads the branch membership of one entry, upstream's `readBranchMembership`.
///
/// # Errors
/// `Branch cache missing entry {entry_id}` when absent; a driver failure
/// otherwise.
fn read_branch_membership(
    db: &dyn SqliteDatabase,
    session_id: &str,
    entry_id: &str,
) -> Result<(String, i64), SessionError> {
    let row = sql!(
        "SELECT b.branch_id, b.entry_seq
		FROM branch_entries b
		JOIN branch_meta m ON m.session_id = b.session_id AND m.branch_id = b.branch_id
		WHERE b.session_id = ?
			AND b.entry_id = ?
			AND ((m.base_seq IS NULL AND b.entry_seq > 0) OR (m.base_seq IS NOT NULL AND b.entry_seq > m.base_seq))
			AND b.entry_seq <= m.tip_seq
		ORDER BY m.tip_seq DESC, b.branch_id
		LIMIT 1",
        session_id,
        entry_id
    )
    .get(db)?
    .ok_or_else(|| SessionError::Message(format!("Branch cache missing entry {entry_id}")))?;
    Ok((row.string("branch_id")?, row.integer("entry_seq")?))
}

/// Reads one branch's meta row, upstream's `readBranchMeta`.
///
/// # Errors
/// `Branch metadata missing for branch {branch_id}` when absent; a driver
/// failure otherwise.
fn read_branch_meta(
    db: &dyn SqliteDatabase,
    session_id: &str,
    branch_id: &str,
) -> Result<SqliteRow, SessionError> {
    sql!(
        "SELECT branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq
		FROM branch_meta
		WHERE session_id = ? AND branch_id = ?",
        session_id,
        branch_id
    )
    .get(db)?
    .ok_or_else(|| SessionError::Message(format!("Branch metadata missing for branch {branch_id}")))
}

/// Inserts one membership row, upstream's `insertBranchEntry`.
///
/// # Errors
/// A driver failure or an out-of-range integer.
fn insert_branch_entry(
    db: &dyn SqliteDatabase,
    session_id: &str,
    branch_id: &str,
    entry: &Entry,
) -> Result<(), SessionError> {
    sql!(
        "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)
		VALUES (?, ?, ?, ?, ?)",
        session_id,
        branch_id,
        entry.id(),
        sql_integer(entry.seq())?,
        wire_of(entry.entry_type())
    )
    .run(db)?;
    Ok(())
}

/// Finds the branch whose tip is the parent, upstream's
/// `readBranchTipForParent`.
///
/// # Errors
/// A driver failure.
fn read_branch_tip_for_parent(
    db: &dyn SqliteDatabase,
    session_id: &str,
    parent_id: &str,
) -> Result<Option<String>, SessionError> {
    Ok(sql!(
        "SELECT branch_id
		FROM branch_meta
		WHERE session_id = ? AND tip_entry_id = ?",
        session_id,
        parent_id
    )
    .get(db)?
    .map(|row| row.string("branch_id"))
    .transpose()?)
}

/// Opens a root branch at one entry, upstream's `createRootBranchForEntry`.
///
/// # Errors
/// A driver failure or an out-of-range integer.
fn create_root_branch_for_entry(
    db: &dyn SqliteDatabase,
    session_id: &str,
    entry: &Entry,
) -> Result<(), SessionError> {
    sql!(
        "INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq)
		VALUES (?, ?, ?, ?, ?, ?)",
        session_id,
        entry.id(),
        entry.id(),
        sql_integer(entry.seq())?,
        None::<String>,
        None::<i64>
    )
    .run(db)
    .map(|_| ())?;
    insert_branch_entry(db, session_id, entry.id(), entry)
}

/// Appends one entry to its parent's branch and moves the tip, upstream's
/// `appendEntryToExistingBranch`.
///
/// # Errors
/// `Expected to update branch {branch_id}, updated {n}` when the tip update
/// misses; a driver failure otherwise.
fn append_entry_to_existing_branch(
    db: &dyn SqliteDatabase,
    session_id: &str,
    branch_id: &str,
    entry: &Entry,
) -> Result<(), SessionError> {
    insert_branch_entry(db, session_id, branch_id, entry)?;
    let result = sql!(
        "UPDATE branch_meta
		SET tip_entry_id = ?, tip_seq = ?
		WHERE session_id = ? AND branch_id = ?",
        entry.id(),
        sql_integer(entry.seq())?,
        session_id,
        branch_id
    )
    .run(db)?;
    if result.changes != 1 {
        return Err(SessionError::Message(format!(
            "Expected to update branch {branch_id}, updated {}",
            result.changes
        )));
    }
    Ok(())
}

/// Walks the base chain from the start entry to the root, newest segment
/// first, upstream's `readBranchSegmentsNewestFirst`.
///
/// # Errors
/// `Branch {branch_id} has base branch without base_seq` on an inconsistent
/// meta row; the membership/meta errors otherwise.
fn read_branch_segments_newest_first(
    db: &dyn SqliteDatabase,
    session_id: &str,
    start: &str,
) -> Result<Vec<BranchSegment>, SessionError> {
    let (mut branch_id, mut upper_seq) = read_branch_membership(db, session_id, start)?;
    let mut segments: Vec<BranchSegment> = Vec::new();
    loop {
        let meta = read_branch_meta(db, session_id, &branch_id)?;
        let base_seq = meta.opt_integer("base_seq")?;
        let base_branch_id = meta.opt_string("base_branch_id")?;
        let lower_seq = base_seq.unwrap_or(0);
        segments.push(BranchSegment {
            branch_id: branch_id.clone(),
            lower_seq,
            upper_seq,
        });
        match base_branch_id {
            None => break,
            Some(base_branch_id) => {
                let base_seq = base_seq.ok_or_else(|| {
                    SessionError::Message(format!(
                        "Branch {branch_id} has base branch without base_seq"
                    ))
                })?;
                branch_id = base_branch_id;
                upper_seq = base_seq;
            }
        }
    }
    Ok(segments)
}

/// The newest compaction boundary at or below one entry's path, upstream's
/// `readNewestCompactionBoundary`.
///
/// # Errors
/// A driver failure.
fn read_newest_compaction_boundary(
    db: &dyn SqliteDatabase,
    session_id: &str,
    segments_newest_first: &[BranchSegment],
) -> Result<Option<(String, i64)>, SessionError> {
    for segment in segments_newest_first {
        let row = sql!(
            "SELECT MAX(entry_seq) AS entry_seq
			FROM branch_entries
			WHERE session_id = ?
				AND branch_id = ?
				AND entry_seq > ?
				AND entry_seq <= ?
				AND entry_type = ?",
            session_id,
            segment.branch_id.as_str(),
            segment.lower_seq,
            segment.upper_seq,
            "compaction"
        )
        .get(db)?;
        let seq = match &row {
            Some(row) => row.opt_integer("entry_seq")?,
            None => None,
        };
        if let Some(seq) = seq {
            return Ok(Some((segment.branch_id.clone(), seq)));
        }
    }
    Ok(None)
}

/// Copies the branch index rows after one sequence into a target branch,
/// oldest segment first, upstream's `copyBranchEntriesAfterSeqThroughParent`.
///
/// # Errors
/// A driver failure.
fn copy_branch_entries_after_seq_through_parent(
    db: &dyn SqliteDatabase,
    session_id: &str,
    target_branch_id: &str,
    segments_newest_first: &[BranchSegment],
    after_seq: i64,
) -> Result<(), SessionError> {
    for segment in segments_newest_first.iter().rev() {
        let lower_seq = segment.lower_seq.max(after_seq);
        if segment.upper_seq <= lower_seq {
            continue;
        }
        sql!(
            "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)
			SELECT ?, ?, entry_id, entry_seq, entry_type
			FROM branch_entries
			WHERE session_id = ?
				AND branch_id = ?
				AND entry_seq > ?
				AND entry_seq <= ?",
            session_id,
            target_branch_id,
            session_id,
            segment.branch_id.as_str(),
            lower_seq,
            segment.upper_seq
        )
        .run(db)?;
    }
    Ok(())
}

/// Opens a divergent branch at one entry, upstream's
/// `createDivergentBranchForEntry`: the base is the newest compaction
/// boundary at or below the parent, and the boundary..parent range is copied
/// in. A null base means this segment stores its own root-through-parent
/// prefix.
///
/// # Errors
/// `Root entries do not create divergent branches` when the entry has no
/// parent; the segment/boundary errors otherwise.
fn create_divergent_branch_for_entry(
    db: &dyn SqliteDatabase,
    session_id: &str,
    entry: &Entry,
) -> Result<(), SessionError> {
    let Some(parent_id) = entry.parent_id() else {
        return Err(SessionError::Message(
            "Root entries do not create divergent branches".to_owned(),
        ));
    };
    let segments_newest_first = read_branch_segments_newest_first(db, session_id, parent_id)?;
    let compaction = read_newest_compaction_boundary(db, session_id, &segments_newest_first)?;
    let branch_id = entry.id();
    sql!(
        "INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq)
		VALUES (?, ?, ?, ?, ?, ?)",
        session_id,
        branch_id,
        entry.id(),
        sql_integer(entry.seq())?,
        compaction.as_ref().map(|(branch_id, _)| branch_id.clone()),
        compaction.as_ref().map(|(_, seq)| *seq),
    )
    .run(db)?;
    copy_branch_entries_after_seq_through_parent(
        db,
        session_id,
        branch_id,
        &segments_newest_first,
        compaction.map_or(0, |(_, seq)| seq),
    )?;
    insert_branch_entry(db, session_id, branch_id, entry)
}

/// Maintains the branch index for one inserted entry, upstream's
/// `appendEntryToBranchIndex`: a root opens a branch, a parent's tip
/// appends, anything else diverges.
///
/// # Errors
/// The segment machinery's errors.
pub fn append_entry_to_branch_index(
    db: &dyn SqliteDatabase,
    session_id: &str,
    entry: &Entry,
) -> Result<(), SessionError> {
    match entry.parent_id() {
        None => create_root_branch_for_entry(db, session_id, entry),
        Some(parent_id) => read_branch_tip_for_parent(db, session_id, parent_id)?.map_or_else(
            || create_divergent_branch_for_entry(db, session_id, entry),
            |branch_id| append_entry_to_existing_branch(db, session_id, &branch_id, entry),
        ),
    }
}

/// The stop predicates, upstream's `stopPredicates`.
fn stop_predicates(query: &StorageBranchScan) -> Vec<SqlQuery> {
    let mut predicates: Vec<SqlQuery> = Vec::new();
    if let Some(stop_at_type) = query.stop_at_type {
        predicates.push(sql!("b.entry_type = ?", wire_of(stop_at_type)));
    }
    if let Some(stop_at_id) = &query.stop_at_id {
        predicates.push(sql!("b.entry_id = ?", stop_at_id.as_str()));
    }
    predicates
}

/// The sequence the scan stops at inside one segment, upstream's
/// `readStopSeq`.
///
/// # Errors
/// A driver failure.
fn read_stop_seq(
    db: &dyn SqliteDatabase,
    session_id: &str,
    segment: &BranchSegment,
    query: &StorageBranchScan,
    oldest_first: bool,
) -> Result<Option<i64>, SessionError> {
    let stop = stop_predicates(query);
    if stop.is_empty() {
        return Ok(None);
    }
    let mut composed = SqlQuery::text("SELECT ");
    composed.push(SqlQuery::text(if oldest_first {
        "MIN(b.entry_seq)"
    } else {
        "MAX(b.entry_seq)"
    }));
    composed.push(sql!(
        " AS stop_seq
		FROM branch_entries b
		WHERE b.session_id = ?
			AND b.branch_id = ?
			AND b.entry_seq > ?
			AND b.entry_seq <= ?
			AND (",
        session_id,
        &segment.branch_id,
        segment.lower_seq,
        segment.upper_seq
    ));
    composed.push(join_sql_fragments(stop, " OR "));
    composed.push(SqlQuery::text(")"));
    composed.get(db)?.map_or(Ok(None), |row| {
        row.opt_integer("stop_seq").map_err(SessionError::from)
    })
}

/// The scan predicates for one segment, upstream's `branchScanPredicates`.
fn branch_scan_predicates(
    session_id: &str,
    segment: &BranchSegment,
    query: &StorageBranchScan,
    oldest_first: bool,
    stop_seq: Option<i64>,
) -> Result<Vec<SqlQuery>, SessionError> {
    let mut predicates = vec![
        sql!("b.session_id = ?", session_id),
        sql!("b.branch_id = ?", segment.branch_id.as_str()),
        sql!("b.entry_seq > ?", segment.lower_seq),
        sql!("b.entry_seq <= ?", segment.upper_seq),
        sql!("e.session_id = b.session_id"),
    ];
    if let Some(stop_seq) = stop_seq {
        predicates.push(if oldest_first {
            sql!("b.entry_seq <= ?", stop_seq)
        } else {
            sql!("b.entry_seq >= ?", stop_seq)
        });
    }
    if let Some(kind) = query.kind {
        predicates.push(sql!("b.entry_type = ?", wire_of(kind)));
    }
    if let Some(custom_type) = &query.custom_type {
        predicates.push(sql!("e.custom_type = ?", custom_type.as_str()));
    }
    if let Some(cursor) = &query.cursor {
        predicates.push(if oldest_first {
            sql!("b.entry_seq > ?", sql_integer(cursor.seq)?)
        } else {
            sql!("b.entry_seq < ?", sql_integer(cursor.seq)?)
        });
    }
    Ok(predicates)
}

/// The limit fragment, upstream's `limitSql`.
fn limit_sql(limit: Option<u64>) -> Result<SqlQuery, SessionError> {
    limit.map_or_else(
        || Ok(SqlQuery::empty()),
        |limit| Ok(sql!(" LIMIT ?", sql_integer(limit)?)),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the segment reader's arity mirrors upstream's readSegment callback"
)]
fn scan_segment_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    segment: &BranchSegment,
    query: &StorageBranchScan,
    oldest_first: bool,
    stop_seq: Option<i64>,
    limit: Option<u64>,
    payload: bool,
) -> Result<Vec<SqliteRow>, SessionError> {
    let predicates = branch_scan_predicates(session_id, segment, query, oldest_first, stop_seq)?;
    let order = if oldest_first { "ASC" } else { "DESC" };
    let columns = if payload {
        "SELECT e.id, e.parent_id, e.seq, e.type, e.custom_type, e.timestamp, e.payload"
    } else {
        "SELECT e.id, e.parent_id, e.seq, e.type, e.custom_type, e.timestamp"
    };
    let mut composed = SqlQuery::text(columns);
    composed.push(SqlQuery::text(
        "
		FROM branch_entries b
		CROSS JOIN entries e ON e.session_id = b.session_id AND e.id = b.entry_id
		WHERE ",
    ));
    composed.push(join_sql_fragments(predicates, " AND "));
    composed.push(SqlQuery::text(
        "
		ORDER BY b.entry_seq ",
    ));
    composed.push(SqlQuery::text(order));
    composed.push(limit_sql(limit)?);
    composed.all(db)
}

/// The payload-carrying segment scan, upstream's `scanEntrySegmentRows`.
///
/// # Errors
/// The predicate builder's errors; a driver failure otherwise.
fn scan_entry_segment_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    segment: &BranchSegment,
    query: &StorageBranchScan,
    oldest_first: bool,
    stop_seq: Option<i64>,
    limit: Option<u64>,
) -> Result<Vec<SqliteRow>, SessionError> {
    scan_segment_rows(
        db,
        session_id,
        segment,
        query,
        oldest_first,
        stop_seq,
        limit,
        true,
    )
}

/// The structural segment scan, upstream's `scanStructureSegmentRows`.
///
/// # Errors
/// The predicate builder's errors; a driver failure otherwise.
fn scan_structure_segment_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    segment: &BranchSegment,
    query: &StorageBranchScan,
    oldest_first: bool,
    stop_seq: Option<i64>,
    limit: Option<u64>,
) -> Result<Vec<EntryStructure>, SessionError> {
    let rows = scan_segment_rows(
        db,
        session_id,
        segment,
        query,
        oldest_first,
        stop_seq,
        limit,
        false,
    )?;
    rows.iter().map(entry_structure_row).collect()
}

/// Walks the branch segments applying one segment reader, upstream's
/// `scanBranchSegments`.
///
/// # Errors
/// The segment reader's errors.
fn scan_branch_segments<T>(
    db: &dyn SqliteDatabase,
    session_id: &str,
    query: &StorageBranchScan,
    read_segment: impl Fn(
        &dyn SqliteDatabase,
        &str,
        &BranchSegment,
        &StorageBranchScan,
        bool,
        Option<i64>,
        Option<u64>,
    ) -> Result<Vec<T>, SessionError>,
) -> Result<Vec<T>, SessionError> {
    let oldest_first = query.order == Some(BranchScanOrder::OldestFirst);
    let mut segments = read_branch_segments_newest_first(db, session_id, &query.start)?;
    if oldest_first {
        segments.reverse();
    }
    let limit = query.limit;
    if limit == Some(0) {
        return Ok(Vec::new());
    }

    let mut rows: Vec<T> = Vec::new();
    for segment in &segments {
        let read = u64::try_from(rows.len()).unwrap_or(u64::MAX);
        let remaining = limit.map(|limit| limit.saturating_sub(read));
        if remaining.is_some_and(|remaining| remaining == 0) {
            break;
        }
        let stop_seq = read_stop_seq(db, session_id, segment, query, oldest_first)?;
        rows.extend(read_segment(
            db,
            session_id,
            segment,
            query,
            oldest_first,
            stop_seq,
            remaining,
        )?);
        if stop_seq.is_some() {
            break;
        }
    }
    Ok(rows)
}

/// Scans a branch path to entries, upstream's `scanBranchEntries`.
///
/// # Errors
/// The segment machinery's errors; a decode failure otherwise.
pub fn scan_branch_entries(
    db: &dyn SqliteDatabase,
    session_id: &str,
    query: &StorageBranchScan,
) -> Result<Vec<Entry>, SessionError> {
    let rows = scan_branch_segments(db, session_id, query, scan_entry_segment_rows)?;
    rows.iter()
        .map(|row| entry_row(row).and_then(|entry_row| decode_entry_row(&entry_row)))
        .collect()
}

/// Scans a branch path to structures, upstream's `scanBranchEntryStructures`.
///
/// # Errors
/// The segment machinery's errors; a decode failure otherwise.
pub fn scan_branch_entry_structures(
    db: &dyn SqliteDatabase,
    session_id: &str,
    query: &StorageBranchScan,
) -> Result<Vec<EntryStructure>, SessionError> {
    scan_branch_segments(db, session_id, query, scan_structure_segment_rows)
}
