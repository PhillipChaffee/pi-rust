//! Print mode (single-shot), upstream's `src/modes/print-mode.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `pi -p "prompt"` writes the final assistant text; `pi --mode json
//! "prompt"` writes one JSON event per session event. Both modes dispose the
//! runtime and emit `session_shutdown` before returning.
//! Restatement: upstream binds the six command-context replacement actions
//! by capturing `runtimeHost`; the handlers need the owned host, so they bind
//! with the entry dispatch (#185) where the host handle exists. A bind
//! failure during a replacement reports to stderr and the replacement
//! continues, because the runtime's rebind callback has no error channel.

use std::future::Future;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::AssistantBlock;
use pi_ai::types::ImageContent;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::core::agent_session::AgentSession;
use crate::core::agent_session::ExtensionBindings;
use crate::core::agent_session::PromptOptions;
use crate::core::agent_session_runtime::AgentSessionRuntime;
use crate::core::output_guard::flush_raw_stdout;
use crate::core::output_guard::wait_for_raw_stdout_backpressure;
use crate::core::output_guard::write_raw_stdout;
use crate::extensions::reasons::ExtensionMode;
use crate::modes::json_event::to_json_event;
use crate::settings_manager::SettingsStorageKind;
use crate::utils::shell::kill_tracked_detached_children;

/// The output protocol, upstream's `mode: "text" | "json"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrintModeKind {
    /// Final assistant text only.
    #[default]
    Text,
    /// Every session event as a JSON line.
    Json,
}

/// Options for print mode, upstream's `PrintModeOptions`.
#[derive(Clone, Debug, Default)]
pub struct PrintModeOptions {
    /// The output protocol.
    pub mode: PrintModeKind,
    /// Additional prompts sent after the initial message.
    pub messages: Vec<String>,
    /// The first prompt, which may carry `@file` content.
    pub initial_message: Option<String>,
    /// Images attached to the initial prompt.
    pub initial_images: Vec<ImageContent>,
}

/// The live subscriptions and current session a replacement rebinds.
struct Subscriptions<S: SettingsStorageKind> {
    session: Arc<AgentSession<S>>,
    unsubscribe: Option<Box<dyn FnOnce() + Send>>,
    unsubscribe_backpressure: Option<Box<dyn FnOnce() + Send>>,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Upstream's `console.error`, the mode's only error channel.
#[expect(
    clippy::print_stderr,
    reason = "upstream reports mode and extension failures through console.error, which is stderr"
)]
fn report_error(message: &str) {
    eprintln!("{message}");
}

/// Run the single-shot mode, upstream's `runPrintMode`.
///
/// Returns the process exit code: `1` when the final assistant message
/// failed or a prompt rejected, `0` otherwise.
#[expect(
    clippy::too_many_lines,
    reason = "the 1:1 port keeps the signal/prompt/cleanup sequence inline like upstream's try/finally"
)]
pub async fn run_print_mode<S>(
    runtime: &mut AgentSessionRuntime<S>,
    options: PrintModeOptions,
) -> i32
where
    S: SettingsStorageKind,
{
    let PrintModeOptions {
        mode,
        messages,
        initial_message,
        initial_images,
    } = options;
    let state = Arc::new(Mutex::new(Subscriptions {
        session: Arc::clone(runtime.session()),
        unsubscribe: None,
        unsubscribe_backpressure: None,
    }));

    let rebind_state = Arc::clone(&state);
    runtime.set_rebind_session(Some(Arc::new(move |session| {
        let state = Arc::clone(&rebind_state);
        let session = Arc::clone(session);
        Box::pin(async move {
            if let Err(error) = rebind(&session, mode, &state).await {
                report_error(&error);
            }
        })
    })));

    let (signal_sender, mut signal_receiver) = mpsc::unbounded_channel::<i32>();
    let signal_tasks = spawn_signal_tasks(&signal_sender);

    let mut run_error: Option<String> = None;
    let mut signal_code: Option<i32> = None;

    if mode == PrintModeKind::Json {
        let session = lock(&state).session.clone();
        if let Some(header) = session.session_header() {
            match serde_json::to_value(&header) {
                Ok(header) => write_raw_stdout(&format!("{header}\n")),
                Err(error) => run_error = Some(error.to_string()),
            }
        }
    }

    if run_error.is_none()
        && let Err(error) = rebind(runtime.session(), mode, &state).await
    {
        run_error = Some(error);
    }

    if run_error.is_none() {
        let session = lock(&state).session.clone();
        if let Some(initial_message) = initial_message {
            let prompt_options = PromptOptions {
                images: Some(initial_images),
                ..PromptOptions::default()
            };
            match prompt_or_signal(
                &mut signal_receiver,
                Box::pin(session.prompt(initial_message, Some(prompt_options))),
            )
            .await
            {
                PromptOutcome::Done(result) => {
                    if let Err(error) = result {
                        run_error = Some(error);
                    }
                }
                PromptOutcome::Signal(code) => signal_code = Some(code),
            }
        }

        if run_error.is_none() && signal_code.is_none() {
            for message in messages {
                match prompt_or_signal(
                    &mut signal_receiver,
                    Box::pin(session.prompt(message, None)),
                )
                .await
                {
                    PromptOutcome::Done(result) => {
                        if let Err(error) = result {
                            run_error = Some(error);
                        }
                    }
                    PromptOutcome::Signal(code) => {
                        signal_code = Some(code);
                        break;
                    }
                }
            }
        }
    }

    if run_error.is_none() && signal_code.is_none() && mode == PrintModeKind::Text {
        let session = lock(&state).session.clone();
        let messages = session.agent().state().messages.clone();
        if let Some(last) = messages.last()
            && let Some(output) = text_mode_output(last)
        {
            match output {
                Ok(lines) => {
                    for line in lines {
                        write_raw_stdout(&format!("{line}\n"));
                    }
                }
                Err(error) => run_error = Some(error),
            }
        }
    }

    if signal_code.is_none() {
        signal_code = signal_receiver.try_recv().ok();
    }

    for task in signal_tasks {
        task.abort();
    }
    {
        let mut subscriptions = lock(&state);
        if let Some(unsubscribe) = subscriptions.unsubscribe.take() {
            unsubscribe();
        }
        if let Some(unsubscribe) = subscriptions.unsubscribe_backpressure.take() {
            unsubscribe();
        }
    }
    runtime.dispose().await;
    flush_raw_stdout().await;

    if let Some(code) = signal_code {
        std::process::exit(code);
    }
    run_error.map_or(0, |error| {
        report_error(&error);
        1
    })
}

/// The outcome of one prompt racing the process signals.
enum PromptOutcome {
    Done(Result<(), String>),
    Signal(i32),
}

async fn prompt_or_signal<F>(
    signal_receiver: &mut mpsc::UnboundedReceiver<i32>,
    prompt: F,
) -> PromptOutcome
where
    F: Future<Output = Result<(), String>>,
{
    tokio::select! {
        biased;
        code = signal_receiver.recv() => PromptOutcome::Signal(code.unwrap_or(143)),
        result = prompt => PromptOutcome::Done(result),
    }
}

/// Bind the mode's extension surface and move the output subscriptions onto
/// `session`.
async fn rebind<S>(
    session: &Arc<AgentSession<S>>,
    mode: PrintModeKind,
    state: &Arc<Mutex<Subscriptions<S>>>,
) -> Result<(), String>
where
    S: SettingsStorageKind,
{
    let extension_mode = if mode == PrintModeKind::Json {
        ExtensionMode::Json
    } else {
        ExtensionMode::Print
    };
    session
        .bind_extensions(ExtensionBindings {
            mode: Some(extension_mode),
            on_error: Some(Arc::new(|error| {
                report_error(&format!(
                    "Extension error ({}): {}",
                    error.extension_path, error.error
                ));
            })),
            ..ExtensionBindings::default()
        })
        .await?;

    let (unsubscribe, unsubscribe_backpressure) = {
        let mut subscriptions = lock(state);
        subscriptions.session = Arc::clone(session);
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

    let event_mode = mode;
    let unsubscribe = session.subscribe(Arc::new(move |event| {
        if event_mode == PrintModeKind::Json {
            match to_json_event(&event) {
                Ok(payload) => write_raw_stdout(&format!("{payload}\n")),
                Err(error) => report_error(&error),
            }
        }
    }));
    let unsubscribe_backpressure = if mode == PrintModeKind::Json {
        let unsubscribe = session.agent().subscribe(Arc::new(|_event, _token| {
            Box::pin(async { wait_for_raw_stdout_backpressure().await })
        }));
        let boxed: Box<dyn FnOnce() + Send> = Box::new(unsubscribe);
        Some(boxed)
    } else {
        None
    };

    let mut subscriptions = lock(state);
    subscriptions.unsubscribe = Some(Box::new(unsubscribe));
    subscriptions.unsubscribe_backpressure = unsubscribe_backpressure;
    drop(subscriptions);
    Ok(())
}

/// Register `SIGTERM` and `SIGHUP`, upstream's `registerSignalHandlers`.
fn spawn_signal_tasks(sender: &mpsc::UnboundedSender<i32>) -> Vec<JoinHandle<()>> {
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
                    let _ = sender.send(code);
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

/// The text mode's tail projection: the final assistant message's text blocks,
/// or the failure the process reports.
fn text_mode_output(message: &AgentMessage) -> Option<Result<Vec<String>, String>> {
    let AgentMessage::Standard(Message::Assistant(assistant)) = message else {
        return None;
    };
    match assistant.stop_reason {
        StopReason::Error | StopReason::Aborted => {
            let reason = if assistant.stop_reason == StopReason::Aborted {
                "aborted"
            } else {
                "error"
            };
            let error = assistant
                .error_message
                .clone()
                .unwrap_or_else(|| format!("Request {reason}"));
            Some(Err(error))
        }
        _ => Some(Ok(assistant
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantBlock::Text(text) => Some(text.text.clone()),
                AssistantBlock::Thinking(_) | AssistantBlock::ToolCall(_) => None,
            })
            .collect())),
    }
}

#[cfg(test)]
mod tests;
