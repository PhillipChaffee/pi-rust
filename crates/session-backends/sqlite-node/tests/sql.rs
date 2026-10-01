//! The SQLite query-builder suite, upstream's `test/sql.test.ts`
//! `describe("sql")`, ported 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's tagged template takes interpolations inline; the port's `sql!`
//! macro carries the `?` markers in the literal and the parameters in order,
//! and a mid-template fragment interpolation composes through
//! `SqlQuery::push`.

#![expect(clippy::expect_used, reason = "tests assert on results")]

use pi_session_backend_sqlite_node::sqlite::types::{SqliteDatabaseFactory, SqliteValue};
use pi_session_backend_sqlite_node::{create_rusqlite_factory, join_sql_fragments, sql};

#[tokio::test]
async fn composes_sqlite_queries_without_renumbering_parameters() {
    let db = create_rusqlite_factory().open(":memory:").expect("open");
    sql!("CREATE TABLE entries (id TEXT PRIMARY KEY, kind TEXT NOT NULL, active INTEGER NOT NULL)")
        .exec(&*db)
        .expect("create table");
    sql!(
        "INSERT INTO entries (id, kind, active) VALUES (?, ?, ?)",
        "one",
        "message",
        1i64
    )
    .run(&*db)
    .expect("insert one");
    sql!(
        "INSERT INTO entries (id, kind, active) VALUES (?, ?, ?)",
        "two",
        "message",
        0i64
    )
    .run(&*db)
    .expect("insert two");

    // Upstream embeds the joined fragment mid-template
    // (`WHERE ${filters} LIMIT ${10}`); the port splices text and parameters
    // with `push`, which is what keeps the numbering stable.
    let filters = join_sql_fragments(
        vec![sql!("kind = ?", "message"), sql!("active = ?", 1i64)],
        " AND ",
    );
    let mut query = sql!("SELECT id FROM entries WHERE ");
    query.push(filters);
    query.push(sql!(" LIMIT ?", 10i64));

    let rows = query.all(&*db).expect("all");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].column_names().collect::<Vec<_>>(), vec!["id"]);
    assert_eq!(
        rows[0].get("id"),
        Some(&SqliteValue::Text("one".to_owned()))
    );

    db.close().expect("close");
}

#[tokio::test]
async fn executes_parameterized_queries() {
    let db = create_rusqlite_factory().open(":memory:").expect("open");
    sql!("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        .exec(&*db)
        .expect("create table");
    sql!(
        "INSERT INTO values_table (id, value) VALUES (?, ?)",
        1i64,
        "one"
    )
    .run(&*db)
    .expect("insert one");
    sql!(
        "INSERT INTO values_table (id, value) VALUES (?, ?)",
        2i64,
        "two"
    )
    .run(&*db)
    .expect("insert two");

    let row = sql!("SELECT value FROM values_table WHERE id = ?", 1i64)
        .get(&*db)
        .expect("get")
        .expect("row");
    assert_eq!(row.column_names().collect::<Vec<_>>(), vec!["value"]);
    assert_eq!(row.get("value"), Some(&SqliteValue::Text("one".to_owned())));

    let rows = sql!("SELECT value FROM values_table ORDER BY id")
        .all(&*db)
        .expect("all");
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0].get("value"),
        Some(&SqliteValue::Text("one".to_owned()))
    );
    assert_eq!(
        rows[1].get("value"),
        Some(&SqliteValue::Text("two".to_owned()))
    );

    db.close().expect("close");
}
