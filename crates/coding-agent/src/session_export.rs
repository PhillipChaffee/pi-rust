//! The JSONL session export, upstream's `src/core/session-export.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value as JsonValue;

use crate::session_manager::{
    CURRENT_SESSION_VERSION, FileEntry, SessionHeader, SessionManager, SessionManagerError,
    file_timestamp, now_iso8601, resolve_here,
};

/// The trailing-entries callback, upstream's `createTrailingEntries`
/// `(parentId: string | null, timestamp: string) => readonly object[]`.
pub type TrailingEntries<'a> = &'a (dyn Fn(Option<&str>, &str) -> Vec<JsonValue> + 'a);

/// Write the current session branch and optional trailing export-only
/// entries as JSONL, upstream's `exportSessionToJsonl`. Returns the written
/// file path.
///
/// The branch flattens to a linear chain: every entry re-parents onto the
/// previous one, so the exported file replays the conversation without the
/// source session's tree structure. The header carries no parent link.
///
/// # Errors
/// [`SessionManagerError::Io`] when the directory or file cannot be written.
pub fn export_session_to_jsonl(
    session_manager: &SessionManager,
    output_path: Option<&str>,
    create_trailing_entries: Option<TrailingEntries<'_>>,
) -> Result<String, SessionManagerError> {
    let default_name = format!("session-{}.jsonl", file_timestamp(&now_iso8601()));
    let file_path = resolve_here(output_path.unwrap_or(&default_name));
    let dir = Path::new(&file_path)
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }

    let timestamp = now_iso8601();
    let header = SessionHeader {
        version: Some(CURRENT_SESSION_VERSION),
        id: session_manager.session_id().to_owned(),
        timestamp: timestamp.clone(),
        cwd: Some(session_manager.cwd().to_owned()),
        parent_session: None,
        extras: serde_json::Map::default(),
    };
    let mut lines = vec![
        serde_json::to_string(&FileEntry::Session(header)).map_err(SessionManagerError::from)?,
    ];

    let mut parent_id: Option<String> = None;
    for entry in session_manager.get_branch(None) {
        let mut chained = entry.clone();
        match &mut chained {
            FileEntry::Entry(typed) => typed.base_mut().parent_id.clone_from(&parent_id),
            FileEntry::Other(value) => {
                if let Some(object) = value.as_object_mut() {
                    object.insert(
                        "parentId".to_owned(),
                        parent_id.clone().map_or(JsonValue::Null, JsonValue::String),
                    );
                }
            }
            FileEntry::Session(_) => {}
        }
        parent_id = entry.entry_id().map(str::to_owned);
        lines.push(serde_json::to_string(&chained).map_err(SessionManagerError::from)?);
    }
    for entry in create_trailing_entries
        .map_or_else(Vec::new, |create| create(parent_id.as_deref(), &timestamp))
    {
        lines.push(serde_json::to_string(&entry).map_err(SessionManagerError::from)?);
    }

    let mut file = fs::File::create(&file_path)?;
    for line in &lines {
        writeln!(file, "{line}")?;
    }
    Ok(file_path)
}
