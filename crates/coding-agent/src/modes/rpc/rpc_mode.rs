//! RPC mode, upstream's `src/modes/rpc/rpc-mode.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Commands arrive as JSON lines on stdin; responses, session events, and
//! extension UI requests leave as JSON lines on stdout. The mode keeps
//! running until stdin closes, a `SIGTERM`/`SIGHUP` arrives, or an extension
//! asks for shutdown after the agent settles.
//!
//! Restatements recorded here:
//! - Upstream's `process.exit` and `Promise<never>` become the returned exit
//!   code; the entry dispatch calls the process exit (#185).
//! - The six command-context replacement actions bind with the entry host
//!   (#185), the same restatement print mode carries: the handlers need the
//!   owned runtime, which the rebind callback cannot borrow.
//! - The runtime's replacement path already runs the rebind callback, so the
//!   mode skips upstream's redundant explicit rebind after `new_session`,
//!   `switch_session`, and `fork`.
//! - Input lines are handled one at a time. Upstream fires them concurrently,
//!   so a long `compact` blocks the next command here; prompt, steer, and
//!   follow-up already return before their run settles, which is what the
//!   protocol's callers wait on.
//! - `user_bash` carries only `handled` (#148), so the handler-supplied result
//!   arm has no counterpart and the command always executes.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use pi_agent_core::harness::context::AbortSignal;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::core::agent_session::AgentSession;
use crate::core::agent_session::AgentSessionEvent;
use crate::core::agent_session::BashExecutionOptions;
use crate::core::agent_session::CycleDirection;
use crate::core::agent_session::ExtensionBindings;
use crate::core::agent_session::ModelMutationOptions;
use crate::core::agent_session::PromptOptions;
use crate::core::agent_session::lock;
use crate::core::agent_session_runtime::AgentSessionRuntime;
use crate::core::agent_session_runtime::ForkOptions;
use crate::core::agent_session_runtime::NewSessionRequest;
use crate::core::agent_session_runtime::SwitchSessionOptions;
use crate::core::output_guard::flush_raw_stdout;
use crate::core::output_guard::take_over_stdout;
use crate::core::output_guard::wait_for_raw_stdout_backpressure;
use crate::core::output_guard::write_raw_stdout;
use crate::export_html::NoThemeSource;
use crate::extensions::reasons::ExtensionMode;
use crate::extensions::reasons::ForkPosition;
use crate::extensions::reasons::InputSource;
use crate::modes::json_event::to_json_event;
use crate::modes::rpc::jsonl::serialize_json_line;
use crate::modes::rpc::rpc_types::COMMAND_TYPES;
use crate::modes::rpc::rpc_types::RpcCommand;
use crate::modes::rpc::rpc_types::RpcCommandSource;
use crate::modes::rpc::rpc_types::RpcExtensionUiRequest;
use crate::modes::rpc::rpc_types::RpcExtensionUiResponse;
use crate::modes::rpc::rpc_types::RpcNotifyType;
use crate::modes::rpc::rpc_types::RpcSessionState;
use crate::modes::rpc::rpc_types::RpcSlashCommand;
use crate::modes::rpc::rpc_types::RpcUiMethod;
use crate::modes::rpc::rpc_types::RpcWidgetPlacement;
use crate::settings_manager::SettingsStorageKind;
use crate::utils::shell::kill_tracked_detached_children;

/// Write one protocol line to the raw stdout, upstream's `output`.
fn output(value: &Value) {
    write_raw_stdout(&serialize_json_line(value));
}

/// A success response, upstream's `success` helper. An absent `data` stays
/// off the wire; a JSON `null` is a deliberate value.
fn success_response(id: Option<&str>, command: &str, data: Option<Value>) -> Value {
    let mut map = serde_json::Map::new();
    if let Some(id) = id {
        map.insert("id".to_owned(), Value::from(id));
    }
    map.insert("type".to_owned(), Value::from("response"));
    map.insert("command".to_owned(), Value::from(command));
    map.insert("success".to_owned(), Value::Bool(true));
    if let Some(data) = data {
        map.insert("data".to_owned(), data);
    }
    Value::Object(map)
}

/// An error response, upstream's `error` helper.
fn error_response(id: Option<&str>, command: &str, message: &str) -> Value {
    let mut map = serde_json::Map::new();
    if let Some(id) = id {
        map.insert("id".to_owned(), Value::from(id));
    }
    map.insert("type".to_owned(), Value::from("response"));
    map.insert("command".to_owned(), Value::from(command));
    map.insert("success".to_owned(), Value::Bool(false));
    map.insert("error".to_owned(), Value::from(message));
    Value::Object(map)
}

fn to_json<T: Serialize>(value: &T) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|error| error.to_string())
}

fn lock_map<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The dialog options extensions pass, upstream's
/// `ExtensionUIDialogOptions`.
#[derive(Clone, Debug, Default)]
pub struct DialogOptions {
    /// Dismisses the dialog when it fires.
    pub signal: Option<AbortSignal>,
    /// Auto-dismisses after this many milliseconds.
    pub timeout: Option<u64>,
}

/// The RPC mode's extension UI surface, upstream's
/// `createExtensionUIContext`.
///
/// The TUI-only knobs (working message and indicator, footer/header,
/// custom components, autocomplete, theme, editor component, tool expansion)
/// are inert in upstream's RPC context and have no counterpart until the
/// interactive surface types `ctx.ui` (#132).
#[derive(Debug, Default)]
pub struct RpcUiContext {
    pending: Mutex<HashMap<String, oneshot::Sender<RpcExtensionUiResponse>>>,
}

impl RpcUiContext {
    /// An empty context.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Resolve the pending request a stdin reply names, upstream's
    /// `pendingExtensionRequests.get(response.id)`.
    pub fn handle_response(&self, response: RpcExtensionUiResponse) {
        let sender = lock_map(&self.pending).remove(&response.id);
        if let Some(sender) = sender {
            let _ = sender.send(response);
        }
    }

    /// Emit one dialog request and wait for its reply, upstream's
    /// `createDialogPromise`.
    async fn dialog<T>(
        &self,
        options: Option<DialogOptions>,
        default: T,
        request: impl FnOnce(&str) -> RpcExtensionUiRequest,
        parse: impl FnOnce(RpcExtensionUiResponse) -> T,
    ) -> T {
        let options = options.unwrap_or_default();
        if options.signal.as_ref().is_some_and(AbortSignal::aborted) {
            return default;
        }
        let id = uuid::Uuid::new_v4().to_string();
        let (sender, receiver) = oneshot::channel();
        lock_map(&self.pending).insert(id.clone(), sender);
        output(&to_json(&request(&id)).unwrap_or(Value::Null));
        let aborted = async {
            match &options.signal {
                Some(signal) => {
                    let _ = signal.wait().await;
                }
                None => std::future::pending::<()>().await,
            }
        };
        let timed_out = async {
            match options.timeout {
                Some(milliseconds) => {
                    tokio::time::sleep(std::time::Duration::from_millis(milliseconds)).await;
                }
                None => std::future::pending::<()>().await,
            }
        };
        let response = tokio::select! {
            response = receiver => response.ok(),
            () = aborted => None,
            () = timed_out => None,
        };
        lock_map(&self.pending).remove(&id);
        response.map_or(default, parse)
    }

    /// Show a selector; a dismissal or absent value reads as `None`.
    pub async fn select(
        &self,
        title: String,
        options: Vec<String>,
        dialog: Option<DialogOptions>,
    ) -> Option<String> {
        let timeout = dialog.as_ref().and_then(|options| options.timeout);
        self.dialog(
            dialog,
            None,
            |id| RpcExtensionUiRequest {
                id: id.to_owned(),
                method: RpcUiMethod::Select {
                    title,
                    options,
                    timeout,
                },
            },
            |response| {
                if response.is_cancelled() {
                    None
                } else {
                    response.value
                }
            },
        )
        .await
    }

    /// Show a confirmation; a dismissal reads as `false`.
    pub async fn confirm(
        &self,
        title: String,
        message: String,
        dialog: Option<DialogOptions>,
    ) -> bool {
        let timeout = dialog.as_ref().and_then(|options| options.timeout);
        self.dialog(
            dialog,
            false,
            |id| RpcExtensionUiRequest {
                id: id.to_owned(),
                method: RpcUiMethod::Confirm {
                    title,
                    message,
                    timeout,
                },
            },
            |response| {
                if response.is_cancelled() {
                    false
                } else {
                    response.confirmed.unwrap_or(false)
                }
            },
        )
        .await
    }

    /// Show a text input; a dismissal reads as `None`.
    pub async fn input(
        &self,
        title: String,
        placeholder: Option<String>,
        dialog: Option<DialogOptions>,
    ) -> Option<String> {
        let timeout = dialog.as_ref().and_then(|options| options.timeout);
        self.dialog(
            dialog,
            None,
            |id| RpcExtensionUiRequest {
                id: id.to_owned(),
                method: RpcUiMethod::Input {
                    title,
                    placeholder,
                    timeout,
                },
            },
            |response| {
                if response.is_cancelled() {
                    None
                } else {
                    response.value
                }
            },
        )
        .await
    }

    /// Show the multi-line editor; a dismissal reads as `None`.
    pub async fn editor(&self, title: String, prefill: Option<String>) -> Option<String> {
        self.dialog(
            None,
            None,
            |id| RpcExtensionUiRequest {
                id: id.to_owned(),
                method: RpcUiMethod::Editor { title, prefill },
            },
            |response| {
                if response.is_cancelled() {
                    None
                } else {
                    response.value
                }
            },
        )
        .await
    }

    /// Emit a fire-and-forget notice.
    pub fn notify(&self, message: &str, notify_type: Option<RpcNotifyType>) {
        Self::emit(RpcUiMethod::Notify {
            message: message.to_owned(),
            notify_type,
        });
    }

    /// Emit a status-bar entry; an absent text clears it.
    pub fn set_status(&self, status_key: &str, status_text: Option<String>) {
        Self::emit(RpcUiMethod::SetStatus {
            status_key: status_key.to_owned(),
            status_text,
        });
    }

    /// Emit a widget; absent lines clear it.
    pub fn set_widget(
        &self,
        widget_key: &str,
        widget_lines: Option<Vec<String>>,
        widget_placement: Option<RpcWidgetPlacement>,
    ) {
        Self::emit(RpcUiMethod::SetWidget {
            widget_key: widget_key.to_owned(),
            widget_lines,
            widget_placement,
        });
    }

    /// Emit the terminal title.
    pub fn set_title(&self, title: &str) {
        Self::emit(RpcUiMethod::SetTitle {
            title: title.to_owned(),
        });
    }

    /// Emit the editor's text.
    pub fn set_editor_text(&self, text: &str) {
        Self::emit(RpcUiMethod::SetEditorText {
            text: text.to_owned(),
        });
    }

    /// Pasted text takes the editor-text path, upstream's `pasteToEditor`.
    pub fn paste_to_editor(&self, text: &str) {
        self.set_editor_text(text);
    }

    fn emit(method: RpcUiMethod) {
        let request = RpcExtensionUiRequest {
            id: uuid::Uuid::new_v4().to_string(),
            method,
        };
        if let Ok(value) = to_json(&request) {
            output(&value);
        }
    }
}

/// The live subscriptions a replacement rebinds, plus the channel the
/// settlement listener uses to wake the command loop.
#[derive(Default)]
struct Subscriptions {
    unsubscribe: Option<Box<dyn FnOnce() + Send>>,
    unsubscribe_backpressure: Option<Box<dyn FnOnce() + Send>>,
    settled: Option<mpsc::UnboundedSender<()>>,
}

/// Upstream's `console.error`, the mode's only error channel.
#[expect(
    clippy::print_stderr,
    reason = "upstream reports extension failures through console.error, which is stderr"
)]
fn report_error(message: &str) {
    eprintln!("{message}");
}

/// Bind the RPC extension surface and move the output subscriptions onto
/// `session`, upstream's `rebindSession`.
async fn rebind<S>(
    session: &Arc<AgentSession<S>>,
    ui: &Arc<RpcUiContext>,
    state: &Arc<Mutex<Subscriptions>>,
    shutdown_requested: &Arc<AtomicBool>,
) -> Result<(), String>
where
    S: SettingsStorageKind,
{
    let shutdown = Arc::clone(shutdown_requested);
    let ui_handle: Arc<dyn Any + Send + Sync> = ui.clone();
    session
        .bind_extensions(ExtensionBindings {
            mode: Some(ExtensionMode::Rpc),
            ui_context: Some(ui_handle),
            shutdown_handler: Some(Arc::new(move || {
                shutdown.store(true, Ordering::Relaxed);
            })),
            on_error: Some(Arc::new(|error| {
                output(&json!({
                    "type": "extension_error",
                    "extensionPath": error.extension_path,
                    "event": error.event,
                    "error": error.error,
                }));
            })),
            ..ExtensionBindings::default()
        })
        .await?;

    let (unsubscribe, unsubscribe_backpressure) = {
        let mut subscriptions = lock_map(state);
        (
            subscriptions.unsubscribe.take(),
            subscriptions.unsubscribe_backpressure.take(),
        )
    };
    if let Some(unsubscribe) = unsubscribe {
        unsubscribe();
    }
    if let Some(unsubscribe) = unsubscribe_backpressure {
        unsubscribe();
    }

    let settled = lock_map(state).settled.clone();
    let unsubscribe = session.subscribe(Arc::new(move |event| {
        match to_json_event(&event) {
            Ok(payload) => output(&payload),
            Err(error) => report_error(&error),
        }
        if matches!(event, AgentSessionEvent::AgentSettled)
            && let Some(settled) = &settled
        {
            let _ = settled.send(());
        }
    }));
    let unsubscribe_backpressure = session.agent().subscribe(Arc::new(|_event, _token| {
        Box::pin(async { wait_for_raw_stdout_backpressure().await })
    }));

    let mut subscriptions = lock_map(state);
    subscriptions.unsubscribe = Some(Box::new(unsubscribe));
    subscriptions.unsubscribe_backpressure = Some(Box::new(unsubscribe_backpressure));
    drop(subscriptions);
    Ok(())
}

/// Run RPC mode over the command lines a reader feeds.
///
/// Returns the process exit code: `0` on stdin EOF or an extension shutdown,
/// `143`/`129` on `SIGTERM`/`SIGHUP`.
pub async fn run_rpc_mode<S>(
    runtime: &mut AgentSessionRuntime<S>,
    mut lines: mpsc::UnboundedReceiver<String>,
) -> i32
where
    S: SettingsStorageKind,
{
    take_over_stdout();
    let ui = RpcUiContext::new();
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (settled_sender, mut settled_receiver) = mpsc::unbounded_channel();
    let state = Arc::new(Mutex::new(Subscriptions {
        settled: Some(settled_sender),
        ..Subscriptions::default()
    }));

    let rebind_state = Arc::clone(&state);
    let rebind_ui = Arc::clone(&ui);
    let rebind_shutdown = Arc::clone(&shutdown_requested);
    runtime.set_rebind_session(Some(Arc::new(move |session| {
        let state = Arc::clone(&rebind_state);
        let ui = Arc::clone(&rebind_ui);
        let shutdown_requested = Arc::clone(&rebind_shutdown);
        let session = Arc::clone(session);
        Box::pin(async move {
            if let Err(error) = rebind(&session, &ui, &state, &shutdown_requested).await {
                report_error(&error);
            }
        })
    })));

    if let Err(error) = rebind(runtime.session(), &ui, &state, &shutdown_requested).await {
        report_error(&error);
    }

    let (signal_sender, mut signal_receiver) = mpsc::unbounded_channel::<(i32, bool)>();
    let signal_tasks = spawn_signal_tasks(&signal_sender);

    let mut exit_code = 0;
    let mut flush = true;
    loop {
        tokio::select! {
            biased;
            Some((code, skip_flush)) = signal_receiver.recv() => {
                exit_code = code;
                flush = !skip_flush;
                break;
            }
            Some(()) = settled_receiver.recv() => {
                if shutdown_requested.load(Ordering::Relaxed) {
                    break;
                }
            }
            line = lines.recv() => {
                let Some(line) = line else {
                    break;
                };
                handle_input_line(&line, runtime, &ui).await;
                if shutdown_requested.load(Ordering::Relaxed) {
                    break;
                }
            }
        }
    }

    for task in signal_tasks {
        task.abort();
    }
    {
        let mut subscriptions = lock_map(&state);
        if let Some(unsubscribe) = subscriptions.unsubscribe.take() {
            unsubscribe();
        }
        if let Some(unsubscribe) = subscriptions.unsubscribe_backpressure.take() {
            unsubscribe();
        }
    }
    runtime.dispose().await;
    if flush {
        flush_raw_stdout().await;
    }
    exit_code
}

/// Register `SIGTERM` and `SIGHUP`, upstream's `registerSignalHandlers`.
///
/// Each signal sends its exit code and whether the shutdown may skip the raw
/// stdout flush (`SIGTERM` does, upstream's `signal !== "SIGTERM"` guard).
fn spawn_signal_tasks(sender: &mpsc::UnboundedSender<(i32, bool)>) -> Vec<JoinHandle<()>> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::SignalKind;

        let mut tasks = Vec::new();
        for (kind, code) in [(SignalKind::terminate(), 143), (SignalKind::hangup(), 129)] {
            let Ok(mut stream) = tokio::signal::unix::signal(kind) else {
                continue;
            };
            let sender = sender.clone();
            tasks.push(tokio::spawn(async move {
                if stream.recv().await.is_some() {
                    kill_tracked_detached_children();
                    let _ = sender.send((code, code == 143));
                }
            }));
        }
        tasks
    }
    #[cfg(not(unix))]
    {
        let _ = sender;
        Vec::new()
    }
}

/// Handle one stdin line: a UI reply, a command, or a parse failure,
/// upstream's `handleInputLine`.
async fn handle_input_line<S>(
    line: &str,
    runtime: &mut AgentSessionRuntime<S>,
    ui: &Arc<RpcUiContext>,
) where
    S: SettingsStorageKind,
{
    let parsed: Value = match serde_json::from_str(line) {
        Ok(parsed) => parsed,
        Err(error) => {
            output(&error_response(
                None,
                "parse",
                &format!("Failed to parse command: {error}"),
            ));
            wait_for_raw_stdout_backpressure().await;
            return;
        }
    };

    if parsed.get("type").and_then(Value::as_str) == Some("extension_ui_response") {
        if let Ok(response) = serde_json::from_value::<RpcExtensionUiResponse>(parsed) {
            ui.handle_response(response);
        }
        return;
    }

    let id = parsed.get("id").and_then(Value::as_str).map(str::to_owned);
    let raw_type = parsed
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let command: RpcCommand = match serde_json::from_value(parsed) {
        Ok(command) => command,
        Err(error) => {
            let message = match &raw_type {
                Some(name) if !COMMAND_TYPES.contains(&name.as_str()) => {
                    format!("Unknown command: {name}")
                }
                _ => error.to_string(),
            };
            output(&error_response(
                id.as_deref(),
                raw_type.as_deref().unwrap_or_default(),
                &message,
            ));
            wait_for_raw_stdout_backpressure().await;
            return;
        }
    };

    let command_id = command.id().map(str::to_owned);
    let command_type = command.type_name();
    match handle_command(command, runtime).await {
        Ok(Some(response)) => {
            output(&response);
            wait_for_raw_stdout_backpressure().await;
        }
        Ok(None) => {}
        Err(message) => {
            output(&error_response(
                command_id.as_deref(),
                command_type,
                &message,
            ));
            wait_for_raw_stdout_backpressure().await;
        }
    }
}

/// Dispatch one parsed command, upstream's `handleCommand`.
///
/// The prompt arm returns no response of its own: the success or failure
/// follows the prompt's preflight result, and the run itself streams as
/// events.
#[expect(
    clippy::too_many_lines,
    clippy::significant_drop_tightening,
    reason = "the 1:1 port keeps upstream's command switch arm for arm; the entry and tree locks stay held while their borrowed views serialize"
)]
async fn handle_command<S>(
    command: RpcCommand,
    runtime: &mut AgentSessionRuntime<S>,
) -> Result<Option<Value>, String>
where
    S: SettingsStorageKind,
{
    let id = command.id().map(str::to_owned);
    let id = id.as_deref();
    Ok(Some(match command {
        RpcCommand::Prompt {
            message,
            images,
            streaming_behavior,
            ..
        } => {
            let session = Arc::clone(runtime.session());
            let preflight_succeeded = Arc::new(AtomicBool::new(false));
            let preflight_flag = Arc::clone(&preflight_succeeded);
            let preflight_id = id.map(str::to_owned);
            let options = PromptOptions {
                images,
                streaming_behavior,
                source: Some(InputSource::Rpc),
                preflight_result: Some(Arc::new(move |succeeded| {
                    if succeeded {
                        preflight_flag.store(true, Ordering::Relaxed);
                        output(&success_response(preflight_id.as_deref(), "prompt", None));
                    }
                })),
                ..PromptOptions::default()
            };
            let spawn_id = id.map(str::to_owned);
            tokio::spawn(async move {
                if let Err(error) = session.prompt(message, Some(options)).await
                    && !preflight_succeeded.load(Ordering::Relaxed)
                {
                    output(&error_response(spawn_id.as_deref(), "prompt", &error));
                }
            });
            return Ok(None);
        }
        RpcCommand::Steer {
            id: _,
            message,
            images,
        } => {
            runtime
                .session()
                .steer(message, images, Some(InputSource::Rpc))
                .await?;
            success_response(id, "steer", None)
        }
        RpcCommand::FollowUp {
            id: _,
            message,
            images,
        } => {
            runtime
                .session()
                .follow_up(message, images, Some(InputSource::Rpc))
                .await?;
            success_response(id, "follow_up", None)
        }
        RpcCommand::Abort { id: _ } => {
            runtime.session().abort().await;
            success_response(id, "abort", None)
        }
        RpcCommand::ClearQueue { id: _ } => {
            let (steering, follow_up) = runtime.session().clear_queue();
            success_response(
                id,
                "clear_queue",
                Some(json!({ "steering": steering, "followUp": follow_up })),
            )
        }
        RpcCommand::NewSession {
            id: _,
            parent_session,
        } => {
            let result = runtime
                .new_session(NewSessionRequest {
                    parent_session,
                    ..NewSessionRequest::default()
                })
                .await
                .map_err(|error| error.to_string())?;
            success_response(
                id,
                "new_session",
                Some(json!({ "cancelled": result.cancelled })),
            )
        }
        RpcCommand::GetState { id: _ } => {
            let session = runtime.session();
            let state = RpcSessionState {
                model: session.model(),
                thinking_level: session.thinking_level(),
                is_streaming: session.is_streaming(),
                is_compacting: session.is_compacting(),
                steering_mode: session.steering_mode(),
                follow_up_mode: session.follow_up_mode(),
                session_file: session.session_file(),
                session_id: session.session_id(),
                session_name: session.session_name(),
                auto_compaction_enabled: session.auto_compaction_enabled(),
                message_count: session.agent().state().messages.len(),
                pending_message_count: session.pending_message_count(),
            };
            success_response(id, "get_state", Some(to_json(&state)?))
        }
        RpcCommand::SetModel {
            id: _,
            provider,
            model_id,
        } => {
            let session = runtime.session();
            let Some(model) = session
                .model_runtime()
                .get_available_snapshot()
                .into_iter()
                .find(|model| model.provider.0 == provider && model.id == model_id)
            else {
                return Err(format!("Model not found: {provider}/{model_id}"));
            };
            session
                .set_model(model.clone(), ModelMutationOptions::default())
                .await?;
            success_response(id, "set_model", Some(to_json(&model)?))
        }
        RpcCommand::CycleModel { id: _ } => {
            let result = runtime
                .session()
                .cycle_model(CycleDirection::Forward, ModelMutationOptions::default());
            let data = match result {
                Some(result) => to_json(&result)?,
                None => Value::Null,
            };
            success_response(id, "cycle_model", Some(data))
        }
        RpcCommand::GetAvailableModels { id: _ } => success_response(
            id,
            "get_available_models",
            Some(json!({
                "models": runtime.session().model_runtime().get_available_snapshot()
            })),
        ),
        RpcCommand::SetThinkingLevel { id: _, level } => {
            runtime
                .session()
                .set_thinking_level(level, ModelMutationOptions::default());
            success_response(id, "set_thinking_level", None)
        }
        RpcCommand::CycleThinkingLevel { id: _ } => {
            let level = runtime
                .session()
                .cycle_thinking_level(ModelMutationOptions::default());
            let data = level.map_or(Value::Null, |level| json!({ "level": level }));
            success_response(id, "cycle_thinking_level", Some(data))
        }
        RpcCommand::GetAvailableThinkingLevels { id: _ } => success_response(
            id,
            "get_available_thinking_levels",
            Some(json!({ "levels": runtime.session().available_thinking_levels() })),
        ),
        RpcCommand::SetSteeringMode { id: _, mode } => {
            runtime.session().set_steering_mode(mode);
            success_response(id, "set_steering_mode", None)
        }
        RpcCommand::SetFollowUpMode { id: _, mode } => {
            runtime.session().set_follow_up_mode(mode);
            success_response(id, "set_follow_up_mode", None)
        }
        RpcCommand::Compact {
            id: _,
            custom_instructions,
        } => {
            let result = runtime.session().compact(custom_instructions).await?;
            success_response(id, "compact", Some(to_json(&result)?))
        }
        RpcCommand::SetAutoCompaction { id: _, enabled } => {
            runtime.session().set_auto_compaction_enabled(enabled);
            success_response(id, "set_auto_compaction", None)
        }
        RpcCommand::SetAutoRetry { id: _, enabled } => {
            runtime.session().set_auto_retry_enabled(enabled);
            success_response(id, "set_auto_retry", None)
        }
        RpcCommand::AbortRetry { id: _ } => {
            runtime.session().abort_retry();
            success_response(id, "abort_retry", None)
        }
        RpcCommand::Bash {
            id: _,
            command,
            exclude_from_context,
        } => {
            let session = runtime.session();
            let _ = session
                .extension_runner()
                .emit_user_bash(command.clone())
                .await;
            let result = session
                .execute_bash(
                    &command,
                    None,
                    Some(BashExecutionOptions {
                        exclude_from_context: exclude_from_context.unwrap_or(false),
                        id: id.map(str::to_owned),
                        operations: None,
                    }),
                )
                .await?;
            success_response(id, "bash", Some(to_json(&result)?))
        }
        RpcCommand::AbortBash { id: _ } => {
            runtime.session().abort_bash();
            success_response(id, "abort_bash", None)
        }
        RpcCommand::GetSessionStats { id: _ } => success_response(
            id,
            "get_session_stats",
            Some(to_json(&runtime.session().session_stats())?),
        ),
        RpcCommand::ExportHtml { id: _, output_path } => {
            let path = runtime
                .session()
                .export_to_html(output_path, None, &NoThemeSource)
                .map_err(|error| error.to_string())?;
            success_response(id, "export_html", Some(json!({ "path": path })))
        }
        RpcCommand::SwitchSession {
            id: _,
            session_path,
        } => {
            let result = runtime
                .switch_session(&session_path, SwitchSessionOptions::default())
                .await
                .map_err(|error| error.to_string())?;
            success_response(
                id,
                "switch_session",
                Some(json!({ "cancelled": result.cancelled })),
            )
        }
        RpcCommand::Fork { id: _, entry_id } => {
            let result = runtime
                .fork(&entry_id, ForkOptions::default())
                .await
                .map_err(|error| error.to_string())?;
            success_response(
                id,
                "fork",
                Some(json!({
                    "text": result.selected_text,
                    "cancelled": result.cancelled,
                })),
            )
        }
        RpcCommand::Clone { id: _ } => {
            let session = runtime.session();
            let Some(leaf_id) = session.leaf_id() else {
                return Err("Cannot clone session: no current entry selected".to_owned());
            };
            let result = runtime
                .fork(
                    &leaf_id,
                    ForkOptions {
                        position: Some(ForkPosition::At),
                        ..ForkOptions::default()
                    },
                )
                .await
                .map_err(|error| error.to_string())?;
            success_response(id, "clone", Some(json!({ "cancelled": result.cancelled })))
        }
        RpcCommand::GetForkMessages { id: _ } => {
            let messages: Vec<Value> = runtime
                .session()
                .user_messages_for_forking()
                .into_iter()
                .map(|(entry_id, text)| json!({ "entryId": entry_id, "text": text }))
                .collect();
            success_response(
                id,
                "get_fork_messages",
                Some(json!({ "messages": messages })),
            )
        }
        RpcCommand::GetEntries { id: _, since } => {
            let session = runtime.session();
            let manager = lock(&session.session_manager);
            let mut entries = manager.entries();
            if let Some(since) = since {
                let Some(index) = entries
                    .iter()
                    .position(|entry| entry.entry_id() == Some(since.as_str()))
                else {
                    return Err(format!("Entry not found: {since}"));
                };
                entries = entries.split_off(index + 1);
            }
            let leaf_id = manager.get_leaf_id().map(str::to_owned);
            let entries = to_json(&entries)?;
            success_response(
                id,
                "get_entries",
                Some(json!({ "entries": entries, "leafId": leaf_id })),
            )
        }
        RpcCommand::GetTree { id: _ } => {
            let manager = lock(&runtime.session().session_manager);
            let tree = to_json(&manager.get_tree())?;
            let leaf_id = manager.get_leaf_id().map(str::to_owned);
            success_response(
                id,
                "get_tree",
                Some(json!({ "tree": tree, "leafId": leaf_id })),
            )
        }
        RpcCommand::GetLastAssistantText { id: _ } => success_response(
            id,
            "get_last_assistant_text",
            Some(json!({ "text": runtime.session().last_assistant_text() })),
        ),
        RpcCommand::SetSessionName { id: _, name } => {
            let name = name.trim();
            if name.is_empty() {
                return Err("Session name cannot be empty".to_owned());
            }
            runtime.session().set_session_name(name);
            success_response(id, "set_session_name", None)
        }
        RpcCommand::GetMessages { id: _ } => success_response(
            id,
            "get_messages",
            Some(json!({ "messages": runtime.session().agent().state().messages })),
        ),
        RpcCommand::GetCommands { id: _ } => {
            let commands = slash_commands(runtime.session());
            success_response(id, "get_commands", Some(to_json(&commands)?))
        }
    }))
}

/// The invocable commands: extension commands, prompt templates, then
/// skills, upstream's `get_commands` collection order.
fn slash_commands<S: SettingsStorageKind>(session: &AgentSession<S>) -> Vec<RpcSlashCommand> {
    let mut commands: Vec<RpcSlashCommand> = session
        .extension_runner()
        .get_registered_commands()
        .into_iter()
        .map(|command| RpcSlashCommand {
            name: command.invocation_name,
            description: command.description,
            source: RpcCommandSource::Extension,
            source_info: command.source_info,
        })
        .collect();
    commands.extend(
        session
            .prompt_templates()
            .into_iter()
            .map(|template| RpcSlashCommand {
                name: template.name,
                description: Some(template.description),
                source: RpcCommandSource::Prompt,
                source_info: template.source_info,
            }),
    );
    commands.extend(
        session
            .with_resource_loader(|loader| loader.get_skills().skills)
            .into_iter()
            .map(|skill| RpcSlashCommand {
                name: format!("skill:{}", skill.name),
                description: Some(skill.description),
                source: RpcCommandSource::Skill,
                source_info: skill.source_info,
            }),
    );
    commands
}
