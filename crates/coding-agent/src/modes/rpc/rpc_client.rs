//! The RPC client, upstream's `src/modes/rpc/rpc-client.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Spawns the agent in RPC mode and speaks the JSONL command protocol to it.
//!
//! Restatements recorded here:
//! - Upstream spawns `node <cliPath>`; the port's single binary spawns
//!   `cliPath` directly, defaulting to the current executable.
//! - Child-exit detection polls `try_wait` while a request is in flight
//!   rather than running a `Child::wait` listener, because tokio's `wait`
//!   needs `&mut Child` and the client keeps the handle for `stop`.
//! - Data-returning commands hand back the response's `data` as JSON;
//!   upstream's TypeScript interfaces have no run-time shape to carry over.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::time::Duration;
use std::time::Instant;

use serde_json::Value;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::modes::rpc::jsonl::attach_jsonl_line_reader;
use crate::modes::rpc::jsonl::serialize_json_line;

/// The per-request response deadline, upstream's 30 s timer.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// How often an in-flight request checks whether the child exited.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
/// How long `stop` waits for a `SIGTERM` exit before killing.
const STOP_GRACE: Duration = Duration::from_secs(1);

/// One agent event listener, upstream's `RpcEventListener`.
pub type RpcEventListener = Arc<dyn Fn(Value) + Send + Sync>;

/// Client options, upstream's `RpcClientOptions`.
#[derive(Clone, Debug, Default)]
pub struct RpcClientOptions {
    /// The program to spawn; defaults to the current executable.
    pub cli_path: Option<String>,
    /// The agent's working directory.
    pub cwd: Option<String>,
    /// Environment overrides layered over the inherited environment.
    pub env: Vec<(String, String)>,
    /// The provider to start with.
    pub provider: Option<String>,
    /// The model id to start with.
    pub model: Option<String>,
    /// Additional CLI arguments.
    pub args: Vec<String>,
}

/// A pending request's responder.
type PendingResponder = oneshot::Sender<Result<Value, String>>;
type Pending = Arc<Mutex<HashMap<String, PendingResponder>>>;
type Listeners = Arc<Mutex<Vec<(Arc<()>, RpcEventListener)>>>;

/// The child-process RPC client, upstream's `RpcClient`.
pub struct RpcClient {
    options: RpcClientOptions,
    process: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout_reader: Option<JoinHandle<()>>,
    stderr_reader: Option<JoinHandle<()>>,
    listeners: Listeners,
    pending: Pending,
    request_id: u64,
    stderr: Arc<Mutex<String>>,
    exit_error: Option<String>,
}

impl std::fmt::Debug for RpcClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcClient")
            .field("started", &self.process.is_some())
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl RpcClient {
    /// Build a client; nothing spawns until [`Self::start`].
    #[must_use]
    pub fn new(options: RpcClientOptions) -> Self {
        Self {
            options,
            process: None,
            stdin: None,
            stdout_reader: None,
            stderr_reader: None,
            listeners: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            request_id: 0,
            stderr: Arc::new(Mutex::new(String::new())),
            exit_error: None,
        }
    }

    /// Start the RPC agent process, upstream's `start`.
    ///
    /// # Errors
    /// The spawn failure, or the exit status when the child dies during the
    /// 100 ms initialization wait.
    pub async fn start(&mut self) -> Result<(), String> {
        if self.process.is_some() {
            return Err("Client already started".to_owned());
        }
        self.exit_error = None;

        let program = self
            .options
            .cli_path
            .clone()
            .unwrap_or_else(default_program);
        let mut args = vec!["--mode".to_owned(), "rpc".to_owned()];
        if let Some(provider) = &self.options.provider {
            args.push("--provider".to_owned());
            args.push(provider.clone());
        }
        if let Some(model) = &self.options.model {
            args.push("--model".to_owned());
            args.push(model.clone());
        }
        args.extend(self.options.args.iter().cloned());

        let mut command = Command::new(&program);
        command
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &self.options.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &self.options.env {
            command.env(key, value);
        }
        let mut child = command
            .spawn()
            .map_err(|error| format!("Agent process error: {error}"))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(stderr) = stderr {
            let buffer = Arc::clone(&self.stderr);
            self.stderr_reader = Some(tokio::spawn(async move {
                read_stderr(stderr, buffer).await;
            }));
        }

        if let Some(stdout) = stdout {
            let (line_sender, mut line_receiver) = mpsc::unbounded_channel();
            tokio::spawn(attach_jsonl_line_reader(stdout, line_sender));
            let pending = Arc::clone(&self.pending);
            let listeners = Arc::clone(&self.listeners);
            self.stdout_reader = Some(tokio::spawn(async move {
                while let Some(line) = line_receiver.recv().await {
                    handle_line(&line, &pending, &listeners);
                }
            }));
        }

        self.process = Some(child);
        self.stdin = stdin;

        tokio::time::sleep(Duration::from_millis(100)).await;
        if let Some(status) = self
            .process
            .as_mut()
            .and_then(|process| process.try_wait().ok().flatten())
        {
            let error = self.process_exit_error(status);
            self.exit_error = Some(error.clone());
            return Err(error);
        }
        Ok(())
    }

    /// Stop the agent process, upstream's `stop`.
    pub async fn stop(&mut self) {
        if self.process.is_none() {
            return;
        }
        if let Some(reader) = self.stdout_reader.take() {
            reader.abort();
        }
        if let Some(reader) = self.stderr_reader.take() {
            reader.abort();
        }
        self.stdin.take();

        let Some(mut child) = self.process.take() else {
            return;
        };
        terminate(&mut child).await;
        lock(&self.pending).clear();
    }

    /// Subscribe to agent events; the returned closure removes the listener.
    pub fn on_event(&mut self, listener: RpcEventListener) -> Box<dyn FnOnce() + Send> {
        let token = Arc::new(());
        lock(&self.listeners).push((Arc::clone(&token), listener));
        let listeners = Arc::clone(&self.listeners);
        Box::new(move || {
            lock(&listeners).retain(|(registered, _)| !Arc::ptr_eq(registered, &token));
        })
    }

    /// The child's collected stderr, upstream's `getStderr`.
    #[must_use]
    pub fn stderr(&self) -> String {
        lock(&self.stderr).clone()
    }

    /// Send a prompt; events follow. Wait with [`Self::wait_for_idle`].
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn prompt(&mut self, message: &str, images: Option<Value>) -> Result<(), String> {
        self.send_command(json!({ "type": "prompt", "message": message, "images": images }))
            .await
            .map(|_| ())
    }

    /// Queue a steering message into the running stream.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn steer(&mut self, message: &str, images: Option<Value>) -> Result<(), String> {
        self.send_command(json!({ "type": "steer", "message": message, "images": images }))
            .await
            .map(|_| ())
    }

    /// Queue a follow-up message for after the run settles.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn follow_up(&mut self, message: &str, images: Option<Value>) -> Result<(), String> {
        self.send_command(json!({ "type": "follow_up", "message": message, "images": images }))
            .await
            .map(|_| ())
    }

    /// Abort the current operation.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn abort(&mut self) -> Result<(), String> {
        self.send_command(json!({ "type": "abort" }))
            .await
            .map(|_| ())
    }

    /// Drop the queued messages and return their texts.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn clear_queue(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "clear_queue" })).await
    }

    /// Start a new session, optionally with parent tracking.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn new_session(&mut self, parent_session: Option<&str>) -> Result<Value, String> {
        self.send_command(json!({ "type": "new_session", "parentSession": parent_session }))
            .await
    }

    /// Read the current session state.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_state(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_state" })).await
    }

    /// Select a model by provider and id.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn set_model(&mut self, provider: &str, model_id: &str) -> Result<Value, String> {
        self.send_command(json!({
            "type": "set_model",
            "provider": provider,
            "modelId": model_id
        }))
        .await
    }

    /// Cycle to the next model.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn cycle_model(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "cycle_model" })).await
    }

    /// List the available models.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_available_models(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_available_models" }))
            .await
    }

    /// Set the thinking level.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn set_thinking_level(&mut self, level: &str) -> Result<(), String> {
        self.send_command(json!({ "type": "set_thinking_level", "level": level }))
            .await
            .map(|_| ())
    }

    /// Cycle the thinking level.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn cycle_thinking_level(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "cycle_thinking_level" }))
            .await
    }

    /// List the levels the current model offers.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_available_thinking_levels(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_available_thinking_levels" }))
            .await
    }

    /// Set how queued steering messages drain.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn set_steering_mode(&mut self, mode: &str) -> Result<(), String> {
        self.send_command(json!({ "type": "set_steering_mode", "mode": mode }))
            .await
            .map(|_| ())
    }

    /// Set how queued follow-up messages drain.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn set_follow_up_mode(&mut self, mode: &str) -> Result<(), String> {
        self.send_command(json!({ "type": "set_follow_up_mode", "mode": mode }))
            .await
            .map(|_| ())
    }

    /// Compact the session context.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn compact(&mut self, custom_instructions: Option<&str>) -> Result<Value, String> {
        self.send_command(json!({
            "type": "compact",
            "customInstructions": custom_instructions
        }))
        .await
    }

    /// Toggle threshold compaction.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn set_auto_compaction(&mut self, enabled: bool) -> Result<(), String> {
        self.send_command(json!({ "type": "set_auto_compaction", "enabled": enabled }))
            .await
            .map(|_| ())
    }

    /// Toggle automatic retries.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn set_auto_retry(&mut self, enabled: bool) -> Result<(), String> {
        self.send_command(json!({ "type": "set_auto_retry", "enabled": enabled }))
            .await
            .map(|_| ())
    }

    /// Abort the retry in flight.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn abort_retry(&mut self) -> Result<(), String> {
        self.send_command(json!({ "type": "abort_retry" }))
            .await
            .map(|_| ())
    }

    /// Run a bash command.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn bash(&mut self, command: &str) -> Result<Value, String> {
        self.send_command(json!({ "type": "bash", "command": command }))
            .await
    }

    /// Abort the running bash command.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn abort_bash(&mut self) -> Result<(), String> {
        self.send_command(json!({ "type": "abort_bash" }))
            .await
            .map(|_| ())
    }

    /// Read the session statistics.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_session_stats(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_session_stats" }))
            .await
    }

    /// Export the session to HTML; the data carries the written `path`.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn export_html(&mut self, output_path: Option<&str>) -> Result<Value, String> {
        self.send_command(json!({ "type": "export_html", "outputPath": output_path }))
            .await
    }

    /// Switch to another session file.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn switch_session(&mut self, session_path: &str) -> Result<Value, String> {
        self.send_command(json!({ "type": "switch_session", "sessionPath": session_path }))
            .await
    }

    /// Fork from a specific entry.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn fork(&mut self, entry_id: &str) -> Result<Value, String> {
        self.send_command(json!({ "type": "fork", "entryId": entry_id }))
            .await
    }

    /// Clone the active branch into a new session.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn clone(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "clone" })).await
    }

    /// List the user messages available for forking.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_fork_messages(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_fork_messages" }))
            .await
    }

    /// Read the session entries, optionally after a cursor.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_entries(&mut self, since: Option<&str>) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_entries", "since": since }))
            .await
    }

    /// Read the session entry tree.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_tree(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_tree" })).await
    }

    /// Read the last assistant message's text.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_last_assistant_text(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_last_assistant_text" }))
            .await
    }

    /// Set the session display name.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn set_session_name(&mut self, name: &str) -> Result<(), String> {
        self.send_command(json!({ "type": "set_session_name", "name": name }))
            .await
            .map(|_| ())
    }

    /// Read every message in the session.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_messages(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_messages" })).await
    }

    /// List the invocable commands.
    ///
    /// # Errors
    /// The transport or response failure.
    pub async fn get_commands(&mut self) -> Result<Value, String> {
        self.send_command(json!({ "type": "get_commands" })).await
    }

    /// Wait until the agent settles, upstream's `waitForIdle`.
    ///
    /// # Errors
    /// The deadline elapsing.
    pub async fn wait_for_idle(&mut self, timeout: Duration) -> Result<(), String> {
        self.collect_settled(timeout, false).await.map(|_| ())
    }

    /// Collect events until the agent settles, upstream's `collectEvents`.
    ///
    /// # Errors
    /// The deadline elapsing.
    pub async fn collect_events(&mut self, timeout: Duration) -> Result<Vec<Value>, String> {
        self.collect_settled(timeout, true).await
    }

    /// Subscribe, optionally collect, and resolve when `agent_settled`
    /// arrives.
    async fn collect_settled(
        &mut self,
        timeout: Duration,
        collect: bool,
    ) -> Result<Vec<Value>, String> {
        let (sender, receiver) = oneshot::channel();
        let sender = Arc::new(Mutex::new(Some(sender)));
        let events: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let events_sink = Arc::clone(&events);
        let unsubscribe = self.on_event(Arc::new(move |event| {
            if collect {
                events_sink
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(event.clone());
            }
            if event["type"] == "agent_settled"
                && let Some(sender) = sender.lock().unwrap_or_else(PoisonError::into_inner).take()
            {
                let _ = sender.send(());
            }
        }));
        let settled = tokio::time::timeout(timeout, receiver).await;
        unsubscribe();
        match settled {
            Ok(Ok(())) => Ok(std::mem::take(
                &mut *events.lock().unwrap_or_else(PoisonError::into_inner),
            )),
            _ => Err(format!(
                "Timeout waiting for agent to become idle. Stderr: {}",
                self.stderr()
            )),
        }
    }

    /// Send a prompt and collect events until it settles.
    ///
    /// # Errors
    /// The send or collection failure.
    pub async fn prompt_and_wait(
        &mut self,
        message: &str,
        images: Option<Value>,
        timeout: Duration,
    ) -> Result<Vec<Value>, String> {
        let (sender, receiver) = oneshot::channel();
        let sender = Arc::new(Mutex::new(Some(sender)));
        let events: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let events_sink = Arc::clone(&events);
        let unsubscribe = self.on_event(Arc::new(move |event| {
            events_sink
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event.clone());
            if event["type"] == "agent_settled"
                && let Some(sender) = sender.lock().unwrap_or_else(PoisonError::into_inner).take()
            {
                let _ = sender.send(());
            }
        }));
        self.prompt(message, images).await?;
        let settled = tokio::time::timeout(timeout, receiver).await;
        unsubscribe();
        match settled {
            Ok(Ok(())) => Ok(std::mem::take(
                &mut *events.lock().unwrap_or_else(PoisonError::into_inner),
            )),
            _ => Err(format!(
                "Timeout collecting events. Stderr: {}",
                self.stderr()
            )),
        }
    }

    /// Send one command and resolve its response, upstream's `send`.
    ///
    /// # Errors
    /// The client not being started, a dead child, a write failure, the
    /// request timeout, or the response's own error arm.
    pub async fn send_command(&mut self, command: Value) -> Result<Value, String> {
        if self.process.is_none() || self.stdin.is_none() {
            return Err("Client not started".to_owned());
        }
        if let Some(error) = &self.exit_error {
            return Err(error.clone());
        }
        if let Some(status) = self
            .process
            .as_mut()
            .and_then(|process| process.try_wait().ok().flatten())
        {
            let error = self.process_exit_error(status);
            self.exit_error = Some(error.clone());
            return Err(error);
        }

        self.request_id += 1;
        let id = format!("req_{}", self.request_id);
        let command_type = command["type"].as_str().unwrap_or_default().to_owned();
        let mut full = command;
        full["id"] = Value::from(id.clone());

        let (sender, mut receiver) = oneshot::channel();
        lock(&self.pending).insert(id.clone(), sender);
        {
            let Some(stdin) = self.stdin.as_mut() else {
                return Err("Client not started".to_owned());
            };
            if let Err(error) = stdin.write_all(serialize_json_line(&full).as_bytes()).await {
                lock(&self.pending).remove(&id);
                let error = format!(
                    "Agent process stdin error: {error}. Stderr: {}",
                    self.stderr()
                );
                self.exit_error = Some(error.clone());
                return Err(error);
            }
        }

        let deadline = Instant::now() + REQUEST_TIMEOUT;
        loop {
            tokio::select! {
                response = &mut receiver => {
                    let response = response
                        .map_err(|_| "Agent process dropped the request".to_owned())??;
                    return match response["success"].as_bool() {
                        Some(true) => Ok(response.get("data").cloned().unwrap_or(Value::Null)),
                        _ => Err(response["error"].as_str().unwrap_or_default().to_owned()),
                    };
                }
                () = tokio::time::sleep(EXIT_POLL_INTERVAL) => {
                    if let Some(status) = self
                        .process
                        .as_mut()
                        .and_then(|process| process.try_wait().ok().flatten())
                    {
                        let error = self.process_exit_error(status);
                        self.exit_error = Some(error.clone());
                        lock(&self.pending).clear();
                        return Err(error);
                    }
                    if Instant::now() >= deadline {
                        lock(&self.pending).remove(&id);
                        return Err(format!(
                            "Timeout waiting for response to {command_type}. Stderr: {}",
                            self.stderr()
                        ));
                    }
                }
            }
        }
    }

    /// The exit status message, upstream's `createProcessExitError`.
    fn process_exit_error(&self, status: std::process::ExitStatus) -> String {
        let (code, signal) = exit_description(status);
        format!(
            "Agent process exited (code={code} signal={signal}). Stderr: {}",
            self.stderr()
        )
    }
}

/// Route one stdout line to its pending response or to the event listeners,
/// upstream's `handleLine`.
fn handle_line(line: &str, pending: &Pending, listeners: &Listeners) {
    let Ok(data) = serde_json::from_str::<Value>(line) else {
        return;
    };
    if data["type"] == "response"
        && let Some(id) = data["id"].as_str()
        && let Some(sender) = lock(pending).remove(id)
    {
        let _ = sender.send(Ok(data));
        return;
    }
    for (_, listener) in lock(listeners).iter() {
        listener(data.clone());
    }
}

/// Collect the child's stderr into `buffer` while echoing it to stderr,
/// upstream's `stderr.on("data")` handler.
#[expect(
    clippy::print_stderr,
    reason = "the client tees the child's stderr to its own, upstream's process.stderr.write"
)]
async fn read_stderr(mut stderr: tokio::process::ChildStderr, buffer: Arc<Mutex<String>>) {
    use tokio::io::AsyncReadExt;
    let mut chunk = [0_u8; 4096];
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(read) => {
                let text = String::from_utf8_lossy(&chunk[..read]).into_owned();
                buffer
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push_str(&text);
                eprint!("{text}");
            }
        }
    }
}

/// The default program: this executable, upstream's `dist/cli.js` sibling.
fn default_program() -> String {
    std::env::current_exe().map_or_else(|_| "pi".to_owned(), |path| path.display().to_string())
}

/// The exit code and signal name, upstream's `(code, signal)` pair.
fn exit_description(status: std::process::ExitStatus) -> (String, String) {
    let code = status
        .code()
        .map_or_else(|| "null".to_owned(), |code| code.to_string());
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt;
        status.signal().map_or_else(
            || "null".to_owned(),
            |number| {
                nix::sys::signal::Signal::try_from(number).ok().map_or_else(
                    || format!("SIG{number}"),
                    |signal| signal.as_str().to_owned(),
                )
            },
        )
    };
    #[cfg(not(unix))]
    let signal = "null".to_owned();
    (code, signal)
}

/// `SIGTERM`, then `SIGKILL` after the grace deadline.
async fn terminate(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child
        .id()
        .and_then(|pid| i32::try_from(pid).ok())
        .map(nix::unistd::Pid::from_raw)
    {
        let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
    }
    let deadline = Instant::now() + STOP_GRACE;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            _ if Instant::now() >= deadline => {
                let _ = child.kill().await;
                return;
            }
            _ => tokio::time::sleep(EXIT_POLL_INTERVAL).await,
        }
    }
}
