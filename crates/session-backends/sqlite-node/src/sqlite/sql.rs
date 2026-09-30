//! The parameterized query builder, upstream's `src/sqlite/sql.ts`.
//!
//! Upstream builds queries with a `sql` tagged template: interpolations
//! become `?` parameters and nested `SqlQuery` fragments inline their text
//! and splice their parameters. The port's `sql!` macro takes the query
//! text with `?` markers written in the literal and the parameters in order
//! (a macro cannot introspect format-args holes); fragment composition rides
//! [`SqlQuery::push`] and [`join_sql_fragments`].

use pi_agent_core::harness::session::types::SessionError;

use crate::sqlite::types::{
    SqliteAdapterError, SqliteDatabase, SqliteParams, SqliteRow, SqliteRunResult, SqliteValue,
};

fn session_error(error: SqliteAdapterError) -> SessionError {
    SessionError::from(error)
}

/// A parameterized SQLite query, upstream's `SqlQuery`.
#[derive(Clone, Debug, PartialEq)]
pub struct SqlQuery {
    /// The SQL text with `?` markers, upstream's `queryText`.
    pub query_text: String,
    /// The parameters in marker order, upstream's `params`.
    pub params: Vec<SqliteValue>,
}

impl SqlQuery {
    /// A query fragment from plain SQL text, no parameters.
    #[must_use]
    pub fn text(text: &str) -> Self {
        Self {
            query_text: text.to_owned(),
            params: Vec::new(),
        }
    }

    /// A single-`?` parameter fragment, upstream's `` sql`${value}` ``.
    #[must_use]
    pub fn param(value: impl Into<SqliteValue>) -> Self {
        Self {
            query_text: "?".to_owned(),
            params: vec![value.into()],
        }
    }

    /// The empty fragment — upstream's empty template, the no-clause placeholder.
    #[must_use]
    pub fn empty() -> Self {
        Self::text("")
    }

    /// Inlines another query: its text is appended and its parameters spliced,
    /// upstream's nested-`SqlQuery` interpolation.
    pub fn push(&mut self, other: Self) {
        self.query_text.push_str(&other.query_text);
        self.params.extend(other.params);
    }

    /// Execute statements that return no rows, upstream's `exec`.
    ///
    /// # Errors
    /// When parameters are present (`SQLite exec queries cannot have
    /// parameters`) or the driver fails.
    pub fn exec(&self, db: &dyn SqliteDatabase) -> Result<(), SessionError> {
        if !self.params.is_empty() {
            return Err(SessionError::Message(
                "SQLite exec queries cannot have parameters".to_owned(),
            ));
        }
        db.exec(&self.query_text).map_err(session_error)
    }

    /// Execute a row-changing statement, upstream's `run`.
    ///
    /// # Errors
    /// A driver failure.
    pub fn run(&self, db: &dyn SqliteDatabase) -> Result<SqliteRunResult, SessionError> {
        db.prepare(&self.query_text)
            .run(&self.params())
            .map_err(session_error)
    }

    /// Read the first row, upstream's `get`.
    ///
    /// # Errors
    /// A driver failure.
    pub fn get(&self, db: &dyn SqliteDatabase) -> Result<Option<SqliteRow>, SessionError> {
        db.prepare(&self.query_text)
            .get(&self.params())
            .map_err(session_error)
    }

    /// Read every row, upstream's `all`.
    ///
    /// # Errors
    /// A driver failure.
    pub fn all(&self, db: &dyn SqliteDatabase) -> Result<Vec<SqliteRow>, SessionError> {
        db.prepare(&self.query_text)
            .all(&self.params())
            .map_err(session_error)
    }

    fn params(&self) -> SqliteParams {
        SqliteParams::Positional(self.params.clone())
    }
}

/// Joins trusted query fragments while preserving their parameter order,
/// upstream's `joinSqlFragments`.
#[must_use]
pub fn join_sql_fragments(fragments: Vec<SqlQuery>, separator: &str) -> SqlQuery {
    let mut query_text = String::new();
    let mut params = Vec::new();
    for (index, fragment) in fragments.into_iter().enumerate() {
        if index > 0 {
            query_text.push_str(separator);
        }
        query_text.push_str(&fragment.query_text);
        params.extend(fragment.params);
    }
    SqlQuery { query_text, params }
}

/// Builds a parameterized query, upstream's `sql` tagged template.
///
/// The text carries `?` markers verbatim and each following expression is one
/// parameter, in order: `sql!("SELECT id FROM t WHERE k = ?", key)`. Nested
/// fragments compose through [`SqlQuery::push`].
#[macro_export]
macro_rules! sql {
    ($text:literal $(, $param:expr)* $(,)?) => {
        $crate::sqlite::sql::SqlQuery {
            query_text: ::std::string::String::from($text),
            params: ::std::vec![$($crate::sqlite::types::SqliteValue::from_param($param)),*],
        }
    };
}
