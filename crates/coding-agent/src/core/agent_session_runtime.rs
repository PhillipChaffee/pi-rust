//! The session runtime, upstream's `src/core/agent-session-runtime.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`AgentSessionRuntime`] owns the current [`AgentSession`] plus its
//! cwd-bound services and replaces both on `/new`, `/resume`, `/fork`, and
//! import. Each replacement settles the outgoing turn, emits
//! `session_shutdown`, invalidates the outgoing extension contexts, then
//! builds the next runtime through the caller's factory.
//!
//! Restatements: the factory and callbacks are Rust closures over boxed
//! futures; the new-session `setup` callback runs against a cloned manager
//! that is written back before the session reads it, because the manager sits
//! behind a `std::sync::Mutex` that cannot be held across an await.

use std::sync::Arc;

use pi_ai::types::BoxedFuture;

use crate::config::home_dir;
use crate::config::process_cwd;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::ForkPosition;
use crate::extensions::reasons::SessionShutdownReason;
use crate::extensions::results::SessionBeforeResult;
use crate::extensions::runner::ExtensionCommandContext;
use crate::extensions::runner::emit_session_shutdown_event;
use crate::project_trust::ProjectTrustSelector;
use crate::session_cwd::MissingSessionCwdError;
use crate::session_cwd::assert_session_cwd_exists;
use crate::session_manager::NewSessionOptions;
use crate::session_manager::SessionManager;
use crate::session_manager::SessionManagerError;
use crate::utils::paths::basename_posix;
use crate::utils::paths::resolve_path;

use super::agent_session::AgentSession;
use super::agent_session::SessionStartSeed;
use super::agent_session_services::AgentSessionRuntimeDiagnostic;
use super::agent_session_services::AgentSessionServices;
use super::sdk::CreateAgentSessionError;

/// The result a runtime factory returns, upstream's
/// `CreateAgentSessionRuntimeResult`.
pub struct CreateAgentSessionRuntimeResult<S: crate::settings_manager::SettingsStorageKind> {
    /// The created session.
    pub session: Arc<AgentSession<S>>,
    /// The loaded extensions result.
    pub extensions_result: crate::resource_loader::ExtensionLoadResult,
    /// The restored-model fallback warning.
    pub model_fallback_message: Option<String>,
    /// The cwd-bound services the session was built over.
    pub services: AgentSessionServices<S>,
    /// The diagnostics collected during setup.
    pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

impl<S: crate::settings_manager::SettingsStorageKind> std::fmt::Debug
    for CreateAgentSessionRuntimeResult<S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateAgentSessionRuntimeResult")
            .field("model_fallback_message", &self.model_fallback_message)
            .field("diagnostics", &self.diagnostics)
            .finish_non_exhaustive()
    }
}

/// The runtime-factory inputs, upstream's inline factory options.
pub struct CreateAgentSessionRuntimeOptions {
    /// The effective cwd.
    pub cwd: String,
    /// The global config directory.
    pub agent_dir: String,
    /// The session manager the new session reads.
    pub session_manager: SessionManager,
    /// Session-start metadata.
    pub session_start_event: Option<SessionStartSeed>,
    /// The project-trust context, when the caller has a UI to ask through.
    pub project_trust_context: Option<Arc<dyn ProjectTrustSelector>>,
}

impl std::fmt::Debug for CreateAgentSessionRuntimeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateAgentSessionRuntimeOptions")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .finish_non_exhaustive()
    }
}

/// The runtime factory, upstream's `CreateAgentSessionRuntimeFactory`.
pub type CreateAgentSessionRuntimeFactory<S> = Arc<
    dyn Fn(
            CreateAgentSessionRuntimeOptions,
        ) -> BoxedFuture<
            'static,
            Result<CreateAgentSessionRuntimeResult<S>, CreateAgentSessionError>,
        > + Send
        + Sync,
>;

/// The project-trust context factory a switch runs for the target cwd,
/// upstream's `projectTrustContextFactory`.
pub type ProjectTrustContextFactory =
    Arc<dyn Fn(&str) -> Option<Arc<dyn ProjectTrustSelector>> + Send + Sync>;

/// The replaced-session callback, upstream's `withSession`.
pub type WithSessionCallback =
    Box<dyn FnOnce(ExtensionCommandContext) -> BoxedFuture<'static, ()> + Send>;

/// The setup callback a new session runs before its messages load.
///
/// Restatement: upstream's callback mutates the manager in place; the port
/// hands the manager over and takes it back so the future stays `'static`.
pub type NewSessionSetup =
    Box<dyn FnOnce(SessionManager) -> BoxedFuture<'static, SessionManager> + Send>;

/// The rebind callback a replacement runs before `withSession`.
pub type RebindSession<S> =
    Arc<dyn Fn(&Arc<AgentSession<S>>) -> BoxedFuture<'static, ()> + Send + Sync>;

/// The error `/import` raises for a missing source file, upstream's
/// `SessionImportFileNotFoundError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionImportFileNotFoundError {
    /// The resolved path that does not exist.
    pub file_path: String,
}

impl std::fmt::Display for SessionImportFileNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "File not found: {}", self.file_path)
    }
}

impl std::error::Error for SessionImportFileNotFoundError {}

/// The runtime operation failure, upstream's thrown errors.
#[derive(Debug)]
pub enum AgentSessionRuntimeError {
    /// The session manager failed.
    SessionManager(SessionManagerError),
    /// The stored session cwd is gone.
    MissingSessionCwd(MissingSessionCwdError),
    /// `/import` pointed at a missing file.
    ImportFileNotFound(SessionImportFileNotFoundError),
    /// The factory failed to create the replacement session.
    SessionCreation(CreateAgentSessionError),
    /// An invalid client state, upstream's plain `throw new Error`.
    Message(String),
}

impl std::fmt::Display for AgentSessionRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionManager(error) => write!(f, "{error}"),
            Self::MissingSessionCwd(error) => write!(f, "{error}"),
            Self::ImportFileNotFound(error) => write!(f, "{error}"),
            Self::SessionCreation(error) => write!(f, "{error}"),
            Self::Message(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AgentSessionRuntimeError {}

impl From<SessionManagerError> for AgentSessionRuntimeError {
    fn from(error: SessionManagerError) -> Self {
        Self::SessionManager(error)
    }
}

impl From<MissingSessionCwdError> for AgentSessionRuntimeError {
    fn from(error: MissingSessionCwdError) -> Self {
        Self::MissingSessionCwd(error)
    }
}

impl From<CreateAgentSessionError> for AgentSessionRuntimeError {
    fn from(error: CreateAgentSessionError) -> Self {
        Self::SessionCreation(error)
    }
}

/// The outcome of a replacement, upstream's `{ cancelled }`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SwitchOutcome {
    /// Whether a `session_before_switch` handler cancelled it.
    pub cancelled: bool,
}

/// The fork outcome, upstream's `{ cancelled, selectedText? }`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForkOutcome {
    /// Whether a `session_before_fork` handler cancelled it.
    pub cancelled: bool,
    /// The user text the target message carried.
    pub selected_text: Option<String>,
}

/// The switch options, upstream's inline options object.
#[derive(Default)]
pub struct SwitchSessionOptions {
    /// Override the cwd the opened session reports.
    pub cwd_override: Option<String>,
    /// Work to run against the replacement session.
    pub with_session: Option<WithSessionCallback>,
    /// Build the project-trust context for the target cwd.
    pub project_trust_context_factory: Option<ProjectTrustContextFactory>,
}

impl std::fmt::Debug for SwitchSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwitchSessionOptions")
            .field("cwd_override", &self.cwd_override)
            .finish_non_exhaustive()
    }
}

/// The new-session options, upstream's inline options object.
#[derive(Default)]
pub struct NewSessionRequest {
    /// The session file the new one forks from.
    pub parent_session: Option<String>,
    /// Work to run against the new manager before the session reads it.
    pub setup: Option<NewSessionSetup>,
    /// Work to run against the replacement session.
    pub with_session: Option<WithSessionCallback>,
}

impl std::fmt::Debug for NewSessionRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewSessionRequest")
            .field("parent_session", &self.parent_session)
            .finish_non_exhaustive()
    }
}

/// The fork options, upstream's inline options object.
#[derive(Default)]
pub struct ForkOptions {
    /// Whether the fork positions before or at the entry. Default: before.
    pub position: Option<ForkPosition>,
    /// Work to run against the replacement session.
    pub with_session: Option<WithSessionCallback>,
}

impl std::fmt::Debug for ForkOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForkOptions")
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}

/// The owned current session plus its cwd-bound services, upstream's
/// `AgentSessionRuntime`.
pub struct AgentSessionRuntime<S: crate::settings_manager::SettingsStorageKind> {
    session: Arc<AgentSession<S>>,
    services: AgentSessionServices<S>,
    create_runtime: CreateAgentSessionRuntimeFactory<S>,
    diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
    model_fallback_message: Option<String>,
    rebind_session: Option<RebindSession<S>>,
    before_session_invalidate: Option<Box<dyn Fn() + Send + Sync>>,
}

impl<S: crate::settings_manager::SettingsStorageKind> std::fmt::Debug for AgentSessionRuntime<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSessionRuntime")
            .field("cwd", &self.services.cwd)
            .field("diagnostics", &self.diagnostics)
            .finish_non_exhaustive()
    }
}

impl<S: crate::settings_manager::SettingsStorageKind> AgentSessionRuntime<S> {
    /// The cwd-bound services.
    #[must_use]
    pub const fn services(&self) -> &AgentSessionServices<S> {
        &self.services
    }

    /// The current session.
    #[must_use]
    pub const fn session(&self) -> &Arc<AgentSession<S>> {
        &self.session
    }

    /// The effective cwd.
    #[must_use]
    pub fn cwd(&self) -> &str {
        &self.services.cwd
    }

    /// The diagnostics the runtime was created with.
    #[must_use]
    pub fn diagnostics(&self) -> &[AgentSessionRuntimeDiagnostic] {
        &self.diagnostics
    }

    /// The restored-model fallback warning.
    #[must_use]
    pub fn model_fallback_message(&self) -> Option<&str> {
        self.model_fallback_message.as_deref()
    }

    /// Set the callback a replacement runs before `withSession`.
    pub fn set_rebind_session(&mut self, rebind_session: Option<RebindSession<S>>) {
        self.rebind_session = rebind_session;
    }

    /// Set the synchronous callback that runs after `session_shutdown`
    /// handlers finish but before the current session is invalidated.
    pub fn set_before_session_invalidate(
        &mut self,
        before_session_invalidate: Option<Box<dyn Fn() + Send + Sync>>,
    ) {
        self.before_session_invalidate = before_session_invalidate;
    }
}

impl<S: crate::settings_manager::SettingsStorageKind> AgentSessionRuntime<S> {
    fn session_manager(&self) -> SessionManager {
        super::agent_session::lock(&self.session.session_manager).clone()
    }

    async fn emit_before_switch(
        &self,
        reason: SessionShutdownReason,
        target_session_file: Option<String>,
    ) -> bool {
        let runner = self.session.extension_runner();
        if !runner.has_handlers("session_before_switch") {
            return false;
        }
        matches!(
            runner
                .emit(ExtensionEvent::SessionBeforeSwitch {
                    reason,
                    target_session_file,
                })
                .await,
            SessionBeforeResult::Switch { cancel: true }
        )
    }

    async fn emit_before_fork(&self, entry_id: &str, position: ForkPosition) -> bool {
        let runner = self.session.extension_runner();
        if !runner.has_handlers("session_before_fork") {
            return false;
        }
        matches!(
            runner
                .emit(ExtensionEvent::SessionBeforeFork {
                    entry_id: entry_id.to_owned(),
                    position,
                })
                .await,
            SessionBeforeResult::Fork { cancel: true, .. }
        )
    }

    async fn teardown_current(
        &self,
        reason: SessionShutdownReason,
        target_session_file: Option<String>,
    ) {
        // Settle the active response first so the aborted turn (tool results
        // included) persists to the outgoing session before replacement.
        self.session.abort().await;
        emit_session_shutdown_event(
            &self.session.extension_runner(),
            ExtensionEvent::SessionShutdown {
                reason,
                target_session_file,
            },
        )
        .await;
        if let Some(before_session_invalidate) = &self.before_session_invalidate {
            before_session_invalidate();
        }
        self.session.dispose();
    }

    fn apply(&mut self, result: CreateAgentSessionRuntimeResult<S>) {
        self.session = result.session;
        self.services = result.services;
        self.diagnostics = result.diagnostics;
        self.model_fallback_message = result.model_fallback_message;
    }

    async fn finish_session_replacement(&self, with_session: Option<WithSessionCallback>) {
        if let Some(rebind_session) = &self.rebind_session {
            rebind_session(&self.session).await;
        }
        if let Some(with_session) = with_session {
            with_session(self.session.create_replaced_session_context()).await;
        }
    }

    /// Replace the current runtime with the session file at `session_path`,
    /// upstream's `switchSession`.
    ///
    /// # Errors
    /// The open, cwd, or factory failure.
    pub async fn switch_session(
        &mut self,
        session_path: &str,
        options: SwitchSessionOptions,
    ) -> Result<SwitchOutcome, AgentSessionRuntimeError> {
        if self
            .emit_before_switch(SessionShutdownReason::Resume, Some(session_path.to_owned()))
            .await
        {
            return Ok(SwitchOutcome { cancelled: true });
        }

        let previous_session_file = self.session.session_file();
        let session_manager =
            SessionManager::open(session_path, None, options.cwd_override.as_deref())?;
        assert_session_cwd_exists(&session_manager, self.cwd())?;
        let target_session_file = session_manager.session_file().map(str::to_owned);
        self.teardown_current(SessionShutdownReason::Resume, target_session_file)
            .await;
        let project_trust_context = options
            .project_trust_context_factory
            .as_ref()
            .and_then(|factory| factory(session_manager.cwd()));
        let result = (self.create_runtime)(CreateAgentSessionRuntimeOptions {
            cwd: session_manager.cwd().to_owned(),
            agent_dir: self.services.agent_dir.clone(),
            session_manager,
            session_start_event: Some(SessionStartSeed {
                reason: crate::extensions::reasons::SessionStartReason::Resume,
                previous_session_file,
            }),
            project_trust_context,
        })
        .await?;
        self.apply(result);
        self.finish_session_replacement(options.with_session).await;
        Ok(SwitchOutcome { cancelled: false })
    }

    /// Start a fresh session in the current cwd, upstream's `newSession`.
    ///
    /// # Errors
    /// The create or factory failure.
    pub async fn new_session(
        &mut self,
        options: NewSessionRequest,
    ) -> Result<SwitchOutcome, AgentSessionRuntimeError> {
        if self
            .emit_before_switch(SessionShutdownReason::New, None)
            .await
        {
            return Ok(SwitchOutcome { cancelled: true });
        }

        let previous_session_file = self.session.session_file();
        let current_manager = self.session_manager();
        let session_dir = current_manager.session_dir().to_owned();
        let mut session_manager = if current_manager.is_persisted() {
            SessionManager::create(self.cwd(), Some(&session_dir), None)?
        } else {
            SessionManager::in_memory(Some(self.cwd()), None, None)?
        };
        if let Some(parent_session) = &options.parent_session {
            session_manager.new_session(Some(NewSessionOptions {
                id: None,
                parent_session: Some(parent_session.clone()),
            }))?;
        }

        let target_session_file = session_manager.session_file().map(str::to_owned);
        self.teardown_current(SessionShutdownReason::New, target_session_file)
            .await;
        let result = (self.create_runtime)(CreateAgentSessionRuntimeOptions {
            cwd: self.cwd().to_owned(),
            agent_dir: self.services.agent_dir.clone(),
            session_manager,
            session_start_event: Some(SessionStartSeed {
                reason: crate::extensions::reasons::SessionStartReason::New,
                previous_session_file,
            }),
            project_trust_context: None,
        })
        .await?;
        self.apply(result);
        if let Some(setup) = options.setup {
            let session_manager = setup(self.session_manager()).await;
            *super::agent_session::lock(&self.session.session_manager) = session_manager;
            let context =
                super::agent_session::lock(&self.session.session_manager).build_session_context();
            self.session.agent.set_messages(&context.messages);
        }
        self.finish_session_replacement(options.with_session).await;
        Ok(SwitchOutcome { cancelled: false })
    }
}

impl<S: crate::settings_manager::SettingsStorageKind> AgentSessionRuntime<S> {
    /// Fork the current session at an entry, upstream's `fork`.
    ///
    /// # Errors
    /// An invalid entry, an unflushed persisted session, or a manager or
    /// factory failure.
    #[expect(
        clippy::too_many_lines,
        reason = "the 1:1 port of upstream's fork keeps its persisted and in-memory arms inline"
    )]
    pub async fn fork(
        &mut self,
        entry_id: &str,
        options: ForkOptions,
    ) -> Result<ForkOutcome, AgentSessionRuntimeError> {
        let position = options.position.unwrap_or(ForkPosition::Before);
        if self.emit_before_fork(entry_id, position).await {
            return Ok(ForkOutcome {
                cancelled: true,
                selected_text: None,
            });
        }

        let selected_entry = self
            .session_manager()
            .get_entry(entry_id)
            .cloned()
            .ok_or_else(|| {
                AgentSessionRuntimeError::Message("Invalid entry ID for forking".to_owned())
            })?;
        let mut selected_text = None;
        let target_leaf_id = match position {
            ForkPosition::At => Some(entry_id.to_owned()),
            ForkPosition::Before => {
                let user_text = match &selected_entry {
                    crate::session_manager::FileEntry::Entry(
                        crate::session_manager::SessionEntry::Message(entry),
                    ) => match &entry.message {
                        Some(pi_agent_core::types::AgentMessage::Standard(
                            pi_ai::types::Message::User(user),
                        )) => Some(pi_ai::utils::text::content_text_with(&user.content, "")),
                        _ => None,
                    },
                    _ => None,
                };
                let Some(user_text) = user_text else {
                    return Err(AgentSessionRuntimeError::Message(
                        "Invalid entry ID for forking".to_owned(),
                    ));
                };
                selected_text = Some(user_text);
                match &selected_entry {
                    crate::session_manager::FileEntry::Entry(
                        crate::session_manager::SessionEntry::Message(entry),
                    ) => entry.base.parent_id.clone(),
                    _ => None,
                }
            }
        };

        let previous_session_file = self.session.session_file();
        let current_manager = self.session_manager();
        if current_manager.is_persisted() {
            let Some(current_session_file) = self.session.session_file() else {
                return Err(AgentSessionRuntimeError::Message(
                    "Persisted session is missing a session file".to_owned(),
                ));
            };
            let session_dir = current_manager.session_dir().to_owned();
            let session_manager = match &target_leaf_id {
                None => {
                    let mut manager = SessionManager::create(self.cwd(), Some(&session_dir), None)?;
                    manager.new_session(Some(NewSessionOptions {
                        id: None,
                        parent_session: Some(current_session_file),
                    }))?;
                    manager
                }
                Some(target_leaf_id) => {
                    if !std::path::Path::new(&current_session_file).exists() {
                        return Err(AgentSessionRuntimeError::Message(
                            "This session has not been saved yet. Wait for the first assistant \
                             response before cloning or forking it."
                                .to_owned(),
                        ));
                    }
                    let mut manager =
                        SessionManager::open(&current_session_file, Some(&session_dir), None)?;
                    if manager.create_branched_session(target_leaf_id)?.is_none() {
                        return Err(AgentSessionRuntimeError::Message(
                            "Failed to create forked session".to_owned(),
                        ));
                    }
                    manager
                }
            };
            let target_session_file = session_manager.session_file().map(str::to_owned);
            let cwd = session_manager.cwd().to_owned();
            self.teardown_current(SessionShutdownReason::Fork, target_session_file)
                .await;
            let result = (self.create_runtime)(CreateAgentSessionRuntimeOptions {
                cwd,
                agent_dir: self.services.agent_dir.clone(),
                session_manager,
                session_start_event: Some(SessionStartSeed {
                    reason: crate::extensions::reasons::SessionStartReason::Fork,
                    previous_session_file,
                }),
                project_trust_context: None,
            })
            .await?;
            self.apply(result);
            self.finish_session_replacement(options.with_session).await;
            return Ok(ForkOutcome {
                cancelled: false,
                selected_text,
            });
        }

        let mut session_manager = current_manager;
        let target_session_file = session_manager.session_file().map(str::to_owned);
        self.teardown_current(SessionShutdownReason::Fork, target_session_file)
            .await;
        match &target_leaf_id {
            None => {
                session_manager.new_session(Some(NewSessionOptions {
                    id: None,
                    parent_session: previous_session_file.clone(),
                }))?;
            }
            Some(target_leaf_id) => {
                session_manager.create_branched_session(target_leaf_id)?;
            }
        }
        let result = (self.create_runtime)(CreateAgentSessionRuntimeOptions {
            cwd: self.cwd().to_owned(),
            agent_dir: self.services.agent_dir.clone(),
            session_manager,
            session_start_event: Some(SessionStartSeed {
                reason: crate::extensions::reasons::SessionStartReason::Fork,
                previous_session_file,
            }),
            project_trust_context: None,
        })
        .await?;
        self.apply(result);
        self.finish_session_replacement(options.with_session).await;
        Ok(ForkOutcome {
            cancelled: false,
            selected_text,
        })
    }
}

impl<S: crate::settings_manager::SettingsStorageKind> AgentSessionRuntime<S> {
    /// Import a session JSONL file and switch to it, upstream's
    /// `importFromJsonl`.
    ///
    /// # Errors
    /// The missing source file, the open, cwd, copy, or factory failure.
    pub async fn import_from_jsonl(
        &mut self,
        input_path: &str,
        cwd_override: Option<&str>,
    ) -> Result<SwitchOutcome, AgentSessionRuntimeError> {
        let resolved_path = resolve_path(input_path, &process_cwd(), &home_dir());
        if !std::path::Path::new(&resolved_path).exists() {
            return Err(AgentSessionRuntimeError::ImportFileNotFound(
                SessionImportFileNotFoundError {
                    file_path: resolved_path,
                },
            ));
        }

        let session_dir = self.session_manager().session_dir().to_owned();
        if !std::path::Path::new(&session_dir).exists() {
            std::fs::create_dir_all(&session_dir)
                .map_err(|error| AgentSessionRuntimeError::Message(error.to_string()))?;
        }

        let file_name = basename_posix(&resolved_path);
        let mut destination_path = format!("{session_dir}/{file_name}");
        let source_already_stored =
            resolve_path(&destination_path, &process_cwd(), &home_dir()) == resolved_path;
        if !source_already_stored {
            let (stem, extension) = split_name_and_extension(&file_name);
            let mut suffix = 1_u32;
            while std::path::Path::new(&destination_path).exists() {
                destination_path = format!("{session_dir}/{stem}-{suffix}{extension}");
                suffix += 1;
            }
        }
        if self
            .emit_before_switch(
                SessionShutdownReason::Resume,
                Some(destination_path.clone()),
            )
            .await
        {
            return Ok(SwitchOutcome { cancelled: true });
        }

        let previous_session_file = self.session.session_file();
        if !source_already_stored {
            copy_exclusive(&resolved_path, &destination_path)?;
        }

        let session_manager =
            SessionManager::open(&destination_path, Some(&session_dir), cwd_override)?;
        assert_session_cwd_exists(&session_manager, self.cwd())?;
        let target_session_file = session_manager.session_file().map(str::to_owned);
        self.teardown_current(SessionShutdownReason::Resume, target_session_file)
            .await;
        let result = (self.create_runtime)(CreateAgentSessionRuntimeOptions {
            cwd: session_manager.cwd().to_owned(),
            agent_dir: self.services.agent_dir.clone(),
            session_manager,
            session_start_event: Some(SessionStartSeed {
                reason: crate::extensions::reasons::SessionStartReason::Resume,
                previous_session_file,
            }),
            project_trust_context: None,
        })
        .await?;
        self.apply(result);
        self.finish_session_replacement(None).await;
        Ok(SwitchOutcome { cancelled: false })
    }

    /// Emit the quit shutdown, invalidate the contexts, and dispose the
    /// session, upstream's `dispose`.
    pub async fn dispose(&mut self) {
        emit_session_shutdown_event(
            &self.session.extension_runner(),
            ExtensionEvent::SessionShutdown {
                reason: SessionShutdownReason::Quit,
                target_session_file: None,
            },
        )
        .await;
        if let Some(before_session_invalidate) = &self.before_session_invalidate {
            before_session_invalidate();
        }
        self.session.dispose();
    }
}

/// Split a file name into its stem and extension (extension includes the
/// dot), Node's `path.parse`.
fn split_name_and_extension(file_name: &str) -> (&str, &str) {
    match file_name.rfind('.') {
        Some(index) if index > 0 => (&file_name[..index], &file_name[index..]),
        _ => (file_name, ""),
    }
}

/// Copy a file, failing when the destination exists, Node's
/// `fs.copyFileSync(..., COPYFILE_EXCL)`.
fn copy_exclusive(source: &str, destination: &str) -> Result<(), AgentSessionRuntimeError> {
    use std::io::Read as _;
    use std::io::Write as _;

    let io_error = |error: std::io::Error| AgentSessionRuntimeError::Message(error.to_string());
    let mut input = std::fs::File::open(source).map_err(io_error)?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(io_error)?;
    let mut buffer = Vec::new();
    input.read_to_end(&mut buffer).map_err(io_error)?;
    output.write_all(&buffer).map_err(io_error)?;
    Ok(())
}

/// Create the initial runtime from a factory and its first session target,
/// upstream's `createAgentSessionRuntime`.
///
/// # Errors
/// The missing stored cwd or the factory failure.
pub async fn create_agent_session_runtime<S: crate::settings_manager::SettingsStorageKind>(
    create_runtime: CreateAgentSessionRuntimeFactory<S>,
    options: CreateAgentSessionRuntimeOptions,
) -> Result<AgentSessionRuntime<S>, AgentSessionRuntimeError> {
    assert_session_cwd_exists(&options.session_manager, &options.cwd)?;
    let result = create_runtime(options).await?;
    Ok(AgentSessionRuntime {
        session: result.session,
        services: result.services,
        create_runtime,
        diagnostics: result.diagnostics,
        model_fallback_message: result.model_fallback_message,
        rebind_session: None,
        before_session_invalidate: None,
    })
}
