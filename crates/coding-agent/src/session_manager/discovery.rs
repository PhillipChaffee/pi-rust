//! Session discovery and listing, upstream's `findMostRecentSession`,
//! `buildSessionInfo`, and `listSessionsFromDir` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::fs;
use std::path::{Path, PathBuf};

use pi_agent_core::harness::session::jsonl::codec::parse_iso8601_millis;
use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Message, UserContent};

use super::entries::{MessageEntry, SessionEntry};
use super::file_entry::FileEntry;
use super::parse::{parse_session_entry_line, read_session_header_for_discovery};
use crate::config::{default_session_dir_path, get_agent_dir, home_dir, process_cwd};
use crate::utils::paths::{PathInputOptions, normalize_path, resolve_path};

/// One session as discovery reports it, upstream's `SessionInfo`.
///
/// Timestamps are Unix epoch milliseconds, upstream's `Date` values; an
/// unparseable header timestamp reports [`SessionInfo::created`] as absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionInfo {
    /// The session file path.
    pub path: String,
    /// The session id.
    pub id: String,
    /// The working directory the session started in; empty for old sessions.
    pub cwd: String,
    /// The user-defined display name from `session_info` entries.
    pub name: Option<String>,
    /// The parent session path when this session was forked or branched.
    pub parent_session_path: Option<String>,
    /// The header timestamp, upstream's `created`; absent when the header
    /// timestamp does not parse.
    pub created: Option<i64>,
    /// The last activity time, falling back to the header timestamp and then
    /// the file mtime.
    pub modified: i64,
    /// The message entry count.
    pub message_count: u64,
    /// The first user message text, or `"(no messages)"`.
    pub first_message: String,
    /// All user/assistant message text, joined with spaces.
    pub all_messages_text: String,
}

/// The listing progress callback, upstream's `SessionListProgress`
/// `(loaded, total) => void`.
pub type SessionListProgress<'a> = &'a (dyn Fn(u64, u64) + 'a);

/// The default session directory for a cwd without creating it, upstream's
/// private `getDefaultSessionDirPath` (the path half rides the config
/// foundation).
pub(crate) fn get_default_session_dir_path(cwd: &str, agent_dir: &str) -> PathBuf {
    default_session_dir_path(cwd, agent_dir)
}

/// The default session directory for a cwd, creating it when missing,
/// upstream's `getDefaultSessionDir`.
///
/// # Errors
/// [`super::file_entry::SessionManagerError::Io`] when the directory cannot
/// be created.
pub fn get_default_session_dir(
    cwd: &str,
) -> Result<PathBuf, super::file_entry::SessionManagerError> {
    let session_dir = get_default_session_dir_path(cwd, &get_agent_dir().display().to_string());
    if !session_dir.exists() {
        fs::create_dir_all(&session_dir)?;
    }
    Ok(session_dir)
}

/// Whether a reported cwd matches the resolved cwd, upstream's
/// `sessionCwdMatches`: present, non-empty, and resolving equal.
pub(crate) fn session_cwd_matches(cwd: Option<&str>, resolved_cwd: &str) -> bool {
    cwd.is_some_and(|cwd| !cwd.is_empty())
        && resolve_path(cwd.unwrap_or_default(), &process_cwd(), &home_dir()) == resolved_cwd
}

/// A file mtime as Unix epoch milliseconds, upstream's `stat.mtime`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "millisecond counts since the epoch fit i64 for any system clock this code will run on"
)]
pub(crate) fn system_time_millis(time: std::time::SystemTime) -> i64 {
    time.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_millis() as i64)
}

/// The message text content for listing, upstream's `extractTextContent`:
/// plain strings as-is, text blocks joined with spaces.
fn extract_text_content(message: &Message) -> Option<String> {
    let text = match message {
        Message::User(user) => match &user.content {
            UserContent::Text(text) => text.clone(),
            UserContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    pi_ai::types::UserBlock::Text(text) => Some(text.text.clone()),
                    pi_ai::types::UserBlock::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        },
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .filter_map(|block| match block {
                pi_ai::types::AssistantBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        Message::ToolResult(_) => return None,
    };
    (!text.is_empty()).then_some(text)
}

/// The message text content for an agent message, upstream's
/// `isMessageWithContent` gate: custom messages carry no extractable text.
fn message_text_content(message: &AgentMessage) -> Option<String> {
    match message {
        AgentMessage::Standard(standard) => extract_text_content(standard),
        AgentMessage::Custom(_) => None,
    }
}

/// The message activity time for listing, upstream's
/// `getMessageActivityTime`: the message's own timestamp. The typed message
/// layer always carries a numeric timestamp, so the entry-timestamp fallback
/// upstream keeps for untyped messages is unreachable.
fn message_activity_time(entry: &MessageEntry) -> Option<i64> {
    let AgentMessage::Standard(standard) = entry.message.as_ref()? else {
        return None;
    };
    match standard {
        Message::User(user) => Some(user.timestamp),
        Message::Assistant(assistant) => Some(assistant.timestamp),
        Message::ToolResult(_) => None,
    }
}

/// Build the listing info for one session file, upstream's
/// `buildSessionInfo`. Files whose first parsed entry is not a session
/// header are not sessions.
#[must_use]
pub(crate) fn build_session_info(file_path: &str) -> Option<SessionInfo> {
    let metadata = fs::metadata(file_path).ok()?;
    let bytes = fs::read(file_path).ok()?;
    let content = String::from_utf8_lossy(&bytes).into_owned();
    let mut header: Option<super::types::SessionHeader> = None;
    let mut message_count: u64 = 0;
    let mut first_message = String::new();
    let mut all_messages: Vec<String> = Vec::new();
    let mut name: Option<String> = None;
    let mut last_activity_time: Option<i64> = None;

    for line in content.split('\n') {
        let Some(entry) = parse_session_entry_line(line) else {
            continue;
        };
        let Some(parsed_header) = header.as_ref() else {
            match entry {
                FileEntry::Session(parsed) => header = Some(parsed),
                _ => return None,
            }
            continue;
        };
        let _ = parsed_header;

        let FileEntry::Entry(typed) = entry else {
            continue;
        };

        // Extract session name (use latest, including explicit clears).
        if let SessionEntry::SessionInfo(info) = &typed {
            name = info
                .name
                .as_deref()
                .map(str::trim)
                .filter(|trimmed| !trimmed.is_empty())
                .map(str::to_owned);
        }

        if let SessionEntry::Message(message_entry) = &typed {
            message_count += 1;
            if let Some(activity_time) = message_activity_time(message_entry) {
                last_activity_time = Some(
                    last_activity_time
                        .map_or(activity_time, |previous| previous.max(activity_time)),
                );
            }
            let Some(message) = &message_entry.message else {
                continue;
            };
            let Some(text_content) = message_text_content(message) else {
                continue;
            };
            if matches!(message, AgentMessage::Standard(Message::User(_)))
                && first_message.is_empty()
            {
                first_message.clone_from(&text_content);
            }
            all_messages.push(text_content);
        }
    }

    let header = header?;
    let cwd = header.cwd.clone().unwrap_or_default();
    let parent_session_path = header.parent_session.clone();
    let header_time = parse_iso8601_millis(&header.timestamp);
    let modified = match last_activity_time {
        Some(activity) if activity > 0 => activity,
        _ => header_time.unwrap_or_else(|| {
            system_time_millis(metadata.modified().unwrap_or(std::time::UNIX_EPOCH))
        }),
    };

    Some(SessionInfo {
        path: file_path.to_owned(),
        id: header.id,
        cwd,
        name,
        parent_session_path,
        created: header_time,
        modified,
        message_count,
        first_message: if first_message.is_empty() {
            "(no messages)".to_owned()
        } else {
            first_message
        },
        all_messages_text: all_messages.join(" "),
    })
}

/// The most recently modified session file in a directory, upstream's
/// `findMostRecentSession` (exported for testing).
///
/// Directory access and stat races make recent-session discovery
/// unavailable, upstream's catch-all `null`.
#[must_use]
pub fn find_most_recent_session(session_dir: &str, cwd: Option<&str>) -> Option<String> {
    let resolved_session_dir = normalize_path(session_dir, &PathInputOptions::default()).ok()?;
    let resolved_cwd = cwd.map(|cwd| resolve_path(cwd, &process_cwd(), &home_dir()));

    let dir_entries = fs::read_dir(&resolved_session_dir).ok()?;
    let mut candidates: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
    for entry in dir_entries {
        let Ok(entry) = entry else { return None };
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if entry
            .path()
            .extension()
            .is_none_or(|extension| extension != "jsonl")
        {
            continue;
        }
        let path = Path::new(&resolved_session_dir).join(name);
        let Some(header) = read_session_header_for_discovery(&path) else {
            continue;
        };
        if resolved_cwd
            .as_ref()
            .is_none_or(|resolved| session_cwd_matches(header.cwd(), resolved))
        {
            let Ok(metadata) = fs::metadata(&path) else {
                return None;
            };
            let Ok(modified) = metadata.modified() else {
                return None;
            };
            candidates.push((path, modified));
        }
    }
    candidates.sort_by_key(|(_, modified)| std::cmp::Reverse(*modified));
    candidates
        .first()
        .map(|(path, _)| path.display().to_string())
}

/// List one directory's sessions with progress, upstream's
/// `listSessionsFromDir`. Upstream loads with 10-way concurrency; the port
/// loads sequentially, which preserves the filtered result set and only
/// reorders the progress callbacks.
pub(crate) async fn list_sessions_from_dir(
    dir: &str,
    on_progress: Option<SessionListProgress<'_>>,
    progress_offset: u64,
    progress_total: Option<u64>,
) -> Vec<SessionInfo> {
    let mut sessions: Vec<SessionInfo> = Vec::new();
    if !Path::new(dir).exists() {
        return sessions;
    }

    let Ok(dir_entries) = fs::read_dir(dir) else {
        return sessions;
    };
    let files: Vec<String> = dir_entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .map(|entry| Path::new(dir).join(entry.file_name()).display().to_string())
        .collect();
    let total = progress_total.unwrap_or(files.len() as u64);

    let mut loaded: u64 = 0;
    for file in files {
        if let Some(info) = build_session_info(&file) {
            sessions.push(info);
        }
        loaded += 1;
        if let Some(on_progress) = on_progress {
            on_progress(progress_offset + loaded, total);
        }
    }

    sessions
}
