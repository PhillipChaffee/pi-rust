//! The JSONL storage, ported from upstream
//! `src/harness/session/jsonl/storage.ts`.
//!
//! The storage appends one serialized transaction line per commit and
//! replays the file on open. A torn final line (a crash mid-write) is
//! discarded on open and the file truncates atomically before new writes
//! are admitted. Upstream serializes commits through a promise queue; the
//! port holds a FIFO mutex across each apply, so admission order is poll
//! order and the queue machinery is gone. A legacy v3 backing serves its
//! imported usage without durable rows and upgrades to format 4 on the
//! first non-empty commit.

use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::types::BoxedFuture;
use serde_json::json;

use crate::harness::context::Context;
use crate::harness::session::commit::{CommittedWrite, insert_usage};
use crate::harness::session::in_memory_storage_state::InMemoryStorageState;
use crate::harness::session::jsonl::codec::LegacyV3SessionHeader;
use crate::harness::session::jsonl::io::{
    file_value, parse_jsonl_transaction, publish_file_atomically, publish_jsonl, read_jsonl_header,
    serialize_jsonl_transaction, split_complete_lines,
};
use crate::harness::session::jsonl::legacy_v3::LegacyV3Source;
use crate::harness::session::jsonl::types::{
    JSONL_STORAGE_VERSION, JsonlStorageHeader, JsonlStorageOptions,
};
use crate::harness::session::memory::default_now;
use crate::harness::session::types::{
    CommitResult, Entry, EntryScan, EntryStructure, SessionError, SessionStats, Storage,
    StorageBranchScan, UsageRow, UsageScan,
};
use crate::harness::session::values::Write;
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress,
};
use crate::harness::types::FileSystem;

/// The backing a JSONL storage reads, upstream's `JsonlBacking`.
#[derive(Clone)]
enum Backing {
    /// A format-4 file.
    V4,
    /// A legacy v3 file captured through its normalized source.
    V3(Arc<LegacyV3Source>),
}

/// The storage lifecycle, upstream's `"open" | "closing" | "closed"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    /// Accepting operations.
    Open,
    /// Closing; operations not yet admitted reject.
    Closing,
    /// Closed.
    Closed,
}

/// JSONL storage backed by an injected filesystem capability, upstream's
/// `JsonlStorage`.
pub struct JsonlStorage {
    file_system: Arc<dyn FileSystem>,
    path: String,
    now: crate::harness::session::memory::NowFn,
    /// The file's first line, upstream's `header`.
    pub header: JsonlStorageHeader,
    backing: Mutex<Backing>,
    storage_state: Mutex<InMemoryStorageState>,
    commit_line: Arc<tokio::sync::Mutex<()>>,
    lifecycle: Mutex<Lifecycle>,
    close_cell: tokio::sync::OnceCell<Result<(), SessionError>>,
}

impl std::fmt::Debug for JsonlStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonlStorage")
            .field("path", &self.path)
            .field("header", &self.header)
            .finish_non_exhaustive()
    }
}

impl JsonlStorage {
    /// The locked state the read paths share.
    fn lock_state(&self) -> std::sync::MutexGuard<'_, InMemoryStorageState> {
        self.storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs one state read under the lifecycle check, upstream's inline
    /// `assertOpen + this.state.lock()` per method.
    fn with_state<T>(
        &self,
        read: impl FnOnce(&mut InMemoryStorageState) -> Result<T, SessionError>,
    ) -> Result<T, SessionError> {
        self.assert_open()?;
        let mut state = self
            .storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        read(&mut state)
    }

    fn new(options: &JsonlStorageOptions, header: JsonlStorageHeader, backing: Backing) -> Self {
        Self {
            file_system: options.file_system.clone(),
            path: options.path.clone(),
            now: options.now.clone().unwrap_or_else(default_now),
            header,
            backing: Mutex::new(backing),
            storage_state: Mutex::new(InMemoryStorageState::new()),
            commit_line: Arc::new(tokio::sync::Mutex::new(())),
            lifecycle: Mutex::new(Lifecycle::Open),
            close_cell: tokio::sync::OnceCell::new(),
        }
    }

    /// Create a fresh file with the header and the initial writes, upstream's
    /// `JsonlStorage.create`.
    ///
    /// # Errors
    /// The session-layer errors: publication failures and write validation.
    pub async fn create(
        options: &JsonlStorageOptions,
        header: JsonlStorageHeader,
        initial_writes: Vec<Write>,
        context: &Context,
    ) -> Result<Self, SessionError> {
        let storage = Self::new(options, header, Backing::V4);
        let timestamp = (storage.now)();
        let prepared = storage
            .storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .prepare_commit(initial_writes, timestamp)?;
        let writes = prepared.writes.clone();
        publish_jsonl(
            &options.file_system,
            &options.path,
            &storage.header,
            context,
            move |append| async move {
                if !writes.is_empty() {
                    append(&writes).await?;
                }
                Ok(())
            },
        )
        .await
        .map_err(SessionError::from)?;
        storage
            .storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .apply_validated(&prepared.writes);
        Ok(storage)
    }

    /// Open an existing file, detecting the format from its header, upstream's
    /// `JsonlStorage.open`.
    ///
    /// # Errors
    /// The session-layer errors: read failures, invalid headers, replay
    /// failures, unsupported storage versions.
    pub async fn open(
        options: &JsonlStorageOptions,
        context: &Context,
    ) -> Result<Self, SessionError> {
        let mut reader = file_value(
            options
                .file_system
                .open_text_line_reader(&options.path, context)
                .await,
            &format!("Failed to read JSONL storage {}", options.path),
        )
        .map_err(SessionError::from)?;
        let parsed = read_jsonl_header(&mut *reader, &options.path, context)
            .await
            .map_err(SessionError::from);
        reader.close(context).await;
        match parsed? {
            crate::harness::session::jsonl::codec::JsonlParsedSessionHeader::V3Legacy(header) => {
                Self::open_legacy_v3(options, header, context).await
            }
            crate::harness::session::jsonl::codec::JsonlParsedSessionHeader::V4(header) => {
                Self::open_v4(options, header, context).await
            }
        }
    }

    async fn open_v4(
        options: &JsonlStorageOptions,
        header: JsonlStorageHeader,
        context: &Context,
    ) -> Result<Self, SessionError> {
        let file_content = file_value(
            options
                .file_system
                .read_text_file(&options.path, context)
                .await,
            &format!("Failed to read JSONL storage {}", options.path),
        )
        .map_err(SessionError::from)?;
        let (lines, torn) = split_complete_lines(&file_content);
        if header.storage_version != JSONL_STORAGE_VERSION {
            return Err(SessionError::Message(format!(
                "Session {} uses unsupported storage version {}",
                header.id, header.storage_version
            )));
        }
        let storage = Self::new(options, header, Backing::V4);
        for (index, line) in lines.iter().enumerate().skip(1) {
            let writes = parse_jsonl_transaction(line).map_err(|error| {
                SessionError::Message(format!(
                    "Invalid JSONL storage {}: line {} ({error})",
                    options.path,
                    index + 1
                ))
            })?;
            storage.replay_committed(&writes)?;
        }
        if let Some(next_seq) = storage.header.next_seq {
            storage
                .storage_state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .advance_next_seq(next_seq)?;
        }
        if torn {
            publish_file_atomically(
                &options.file_system,
                &options.path,
                context,
                |append| async move { append(format!("{}\n", lines.join("\n"))).await },
            )
            .await
            .map_err(SessionError::from)?;
        }
        Ok(storage)
    }

    async fn open_legacy_v3(
        options: &JsonlStorageOptions,
        _header: LegacyV3SessionHeader,
        context: &Context,
    ) -> Result<Self, SessionError> {
        let source = Arc::new(
            LegacyV3Source::read(options.file_system.clone(), &options.path, context).await?,
        );
        let mut normalized_header = source.header.clone();
        normalized_header.next_seq = Some(source.next_seq);
        let storage = Self::new(options, normalized_header, Backing::V3(source.clone()));
        for write in source
            .writes(context, None)
            .await
            .map_err(SessionError::from)?
        {
            storage.replay_committed(std::slice::from_ref(&write))?;
        }
        Ok(storage)
    }

    fn replay_committed(&self, writes: &[CommittedWrite]) -> Result<(), SessionError> {
        let mut state = self
            .storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let result = state.validate_committed(writes);
        if result.is_ok() {
            state.apply_validated(writes);
        }
        drop(state);
        result
    }

    fn lock_backing(&self) -> std::sync::MutexGuard<'_, Backing> {
        self.backing.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn with_imported_usage(&self, stats: SessionStats) -> SessionStats {
        match &*self.lock_backing() {
            Backing::V4 => stats,
            Backing::V3(source) => SessionStats {
                usage: source.imported_usage,
                ..stats
            },
        }
    }

    /// Whether the storage still backs onto a legacy v3 file, upstream's
    /// `isLegacyV3`.
    #[must_use]
    pub fn is_legacy_v3(&self) -> bool {
        matches!(&*self.lock_backing(), Backing::V3(_))
    }

    /// Capture the first sequence a later source commit would use, upstream's
    /// `captureForkNextSeq`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the storage is closed.
    pub async fn capture_fork_next_seq(&self) -> Result<u64, SessionError> {
        self.assert_open()?;
        let _line = self.commit_line.lock().await;
        Ok(self
            .storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_next_seq())
    }

    /// The first v3 commit's imported-usage adjustment and caller writes,
    /// upstream's `upgradeLegacyV3ToV4`.
    async fn upgrade_legacy_v3_to_v4(
        &self,
        source: &Arc<LegacyV3Source>,
        caller_writes: Vec<Write>,
        context: &Context,
    ) -> Result<CommitResult, SessionError> {
        let timestamp = (self.now)();
        let adjustment = insert_usage(crate::harness::session::types::UsageWriteRow {
            id: pi_ai::utils::uuid::uuidv7(Some(u64::try_from(timestamp).unwrap_or_default()))
                .map_err(|error| SessionError::Message(error.to_string()))?,
            usage: source.imported_usage,
            entry_id: None,
            adjustment: true,
            details: Some(json!({ "source": "v3-import" })),
        });
        let mut writes = vec![Write::Usage(adjustment)];
        writes.extend(caller_writes);
        let prepared = self
            .storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .prepare_commit(writes, timestamp)?;

        let next_seq = prepared.result.first_seq + prepared.writes.len() as u64;
        let mut upgraded_header = self.header.clone();
        upgraded_header.next_seq = Some(next_seq);
        let source_writes = source
            .writes(context, None)
            .await
            .map_err(SessionError::from)?;
        let prepared_writes = prepared.writes.clone();
        publish_jsonl(
            &self.file_system,
            &self.path,
            &upgraded_header,
            context,
            move |append| async move {
                for write in &source_writes {
                    append(std::slice::from_ref(write)).await?;
                }
                append(&prepared_writes).await
            },
        )
        .await
        .map_err(SessionError::from)?;

        let stats = self
            .storage_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .apply_validated(&prepared.writes);
        *self.lock_backing() = Backing::V4;
        // The first sequence belongs to the internal usage adjustment;
        // return only caller-write sequences.
        Ok(CommitResult {
            first_seq: prepared.result.first_seq + 1,
            seqs: prepared.result.seqs[1..].to_vec(),
            timestamp: prepared.result.timestamp,
            stats,
        })
    }

    fn assert_open(&self) -> Result<(), SessionError> {
        if *self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            != Lifecycle::Open
        {
            return Err(SessionError::Message("JsonlStorage is closed".to_owned()));
        }
        Ok(())
    }
}

impl Storage for JsonlStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        // Upstream enqueues the apply on the commit queue; the port holds
        // the FIFO line across the apply, so a later commit's state reads
        // observe this commit's writes.
        let line = self.commit_line.clone();
        let context = context.clone();
        Box::pin(async move {
            let _guard = line.lock().await;
            // Copy the v3 backing out of the std guard before awaiting: the
            // guard itself must not cross an await point.
            let v3_source = {
                let backing = self.lock_backing();
                match &*backing {
                    Backing::V3(source) if !writes.is_empty() => Some(source.clone()),
                    _ => None,
                }
            };
            if let Some(source) = v3_source {
                return self
                    .upgrade_legacy_v3_to_v4(&source, writes, &context)
                    .await;
            }
            let prepared = self
                .storage_state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .prepare_commit(writes, (self.now)())?;
            if !prepared.writes.is_empty() {
                file_value(
                    self.file_system
                        .append_file(
                            &self.path,
                            crate::harness::types::FileContent::Text(format!(
                                "{}\n",
                                serialize_jsonl_transaction(&prepared.writes)
                            )),
                            &context,
                        )
                        .await,
                    &format!("Failed to append JSONL storage {}", self.path),
                )
                .map_err(SessionError::from)?;
            }
            let stats = self
                .storage_state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .apply_validated(&prepared.writes);
            Ok(CommitResult {
                first_seq: prepared.result.first_seq,
                seqs: prepared.result.seqs,
                timestamp: prepared.result.timestamp,
                stats: self.with_imported_usage(stats),
            })
        })
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        let result = self.with_state(|state| Ok(state.get_entries(&ids)));
        Box::pin(std::future::ready(result))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        let result = self.with_state(|state| Ok(state.get_value(address)));
        Box::pin(std::future::ready(result))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        let result = self.with_state(|state| Ok(state.scan_values(prefix)));
        Box::pin(std::future::ready(result))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        let result = self.with_state(|state| state.read_list(address, options));
        Box::pin(std::future::ready(result))
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        let result = self.with_state(|state| state.scan_branch(query));
        Box::pin(std::future::ready(result))
    }

    fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
        let result = self.with_state(|state| state.scan_branch_structure(query));
        Box::pin(std::future::ready(result))
    }

    fn scan_entries(
        &self,
        query: &EntryScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        let result = self.with_state(|state| Ok(state.scan_entries(query)));
        Box::pin(std::future::ready(result))
    }

    fn scan_usage(
        &self,
        query: &UsageScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>> {
        let result = self.with_state(|state| Ok(state.scan_usage(query)));
        Box::pin(std::future::ready(result))
    }

    fn get_stats(&self, _context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        let result = self
            .assert_open()
            .map(|()| self.with_imported_usage(self.lock_state().get_stats()));
        Box::pin(std::future::ready(result))
    }

    fn close(&self, _context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        if *self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            == Lifecycle::Open
        {
            *self
                .lifecycle
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closing;
        }
        let line = self.commit_line.clone();
        Box::pin(async move {
            self.close_cell
                .get_or_init(|| async move {
                    drop(line.lock().await);
                    *self
                        .lifecycle
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closed;
                    Ok(())
                })
                .await
                .clone()
        })
    }
}
