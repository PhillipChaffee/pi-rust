//! The in-process RPC harness: a real runtime over the faux provider, the
//! captured raw stdout, and the command-line channel the mode reads.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::time::Duration;
use std::time::Instant;

use pi_coding_agent::core::agent_session_runtime::AgentSessionRuntime;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeOptions;
use pi_coding_agent::core::agent_session_runtime::create_agent_session_runtime;
use pi_coding_agent::core::output_guard::RawStdoutWriter;
use pi_coding_agent::core::output_guard::set_raw_writer_for_test;
use pi_coding_agent::core::output_guard::take_over_stdout;
use pi_coding_agent::modes::rpc::jsonl::serialize_json_line;
use pi_coding_agent::modes::rpc::rpc_mode::run_rpc_mode;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::agent_session::RuntimeFactoryOptions;
use super::agent_session::runtime_factory;
use super::sdk_fixture::Fixture;
use super::sdk_fixture::faux_model;
use super::sdk_fixture::registry_with_faux_delay;
use super::sdk_fixture::registry_with_gated_stream;

/// The raw stdout chunks captured through the injectable writer; the
/// previous writer is restored when the capture drops.
pub struct CapturedStdout {
    buffer: Arc<Mutex<Vec<u8>>>,
    previous: Option<RawStdoutWriter>,
}

impl CapturedStdout {
    /// Install the capturing writer.
    pub fn new() -> Self {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&buffer);
        let previous = set_raw_writer_for_test(Some(Arc::new(move |text: &str| {
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(text.as_bytes());
            Ok(())
        })));
        Self { buffer, previous }
    }

    /// The captured text.
    pub fn text(&self) -> String {
        String::from_utf8(
            self.buffer
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone(),
        )
        .expect("the raw stdout carries UTF-8")
    }

    /// The parsed lines written so far.
    ///
    /// # Panics
    /// A stdout line that is not JSON, which the protocol forbids.
    pub fn parsed(&self) -> Vec<Value> {
        self.text()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("every stdout line is JSON"))
            .collect()
    }
}

impl Drop for CapturedStdout {
    fn drop(&mut self) {
        set_raw_writer_for_test(self.previous.take());
    }
}

/// One running RPC mode over a real runtime.
pub struct RpcHarness {
    lines: mpsc::UnboundedSender<String>,
    captured: CapturedStdout,
    task: JoinHandle<i32>,
    _fixture: Fixture,
}

impl RpcHarness {
    /// Start the mode with the faux stream delayed by `delay_ms`.
    ///
    /// # Panics
    /// The fixture's own setup failures, which the tests cannot satisfy.
    pub async fn start(delay_ms: u64) -> Self {
        Self::start_registry(registry_with_faux_delay(delay_ms).await, false).await
    }

    /// Start the mode over a persisted session manager, the export arms need
    /// a session file.
    ///
    /// # Panics
    /// The fixture's own setup failures, which the tests cannot satisfy.
    pub async fn start_persisted(delay_ms: u64) -> Self {
        Self::start_registry(registry_with_faux_delay(delay_ms).await, true).await
    }

    /// Start the mode over the gated stream so queue tests can act while the
    /// run parks in the provider call.
    ///
    /// # Panics
    /// The fixture's own setup failures, which the tests cannot satisfy.
    pub async fn start_gated(release: Arc<tokio::sync::Notify>) -> Self {
        Self::start_registry(registry_with_gated_stream(release).await, false).await
    }

    async fn start_registry(
        registry: pi_coding_agent::model_registry::ModelRegistry,
        persisted: bool,
    ) -> Self {
        let fixture = Fixture::new();
        let factory = runtime_factory(RuntimeFactoryOptions {
            runtime: registry.runtime().clone(),
            model: faux_model(),
            extensions: Arc::new(Vec::new),
            no_extensions: true,
            tools: None,
        });
        let session_manager = if persisted {
            let session_dir = format!("{}/sessions", fixture.agent_dir);
            SessionManager::create(&fixture.cwd, Some(&session_dir), None)
                .expect("persisted manager")
        } else {
            SessionManager::in_memory(Some(&fixture.cwd), None, None).expect("session manager")
        };
        let mut runtime: AgentSessionRuntime<InMemorySettingsStorage> =
            create_agent_session_runtime(
                factory,
                CreateAgentSessionRuntimeOptions {
                    cwd: fixture.cwd.clone(),
                    agent_dir: fixture.agent_dir.clone(),
                    session_manager,
                    session_start_event: None,
                    project_trust_context: None,
                },
            )
            .await
            .expect("runtime");

        // The mode takes over stdout at entry; install the capture after the
        // takeover so the mode's own takeover is the idempotent no-op.
        take_over_stdout();
        let captured = CapturedStdout::new();
        let (lines, receiver) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move { run_rpc_mode(&mut runtime, receiver).await });
        Self {
            lines,
            captured,
            task,
            _fixture: fixture,
        }
    }

    /// Send one command object.
    ///
    /// # Panics
    /// The mode having already stopped reading.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the call sites build a literal Value; the helper forwards it by reference"
    )]
    pub fn send(&self, value: Value) {
        self.lines
            .send(serialize_json_line(&value))
            .expect("the mode is reading");
    }

    /// Send one raw line.
    ///
    /// # Panics
    /// The mode having already stopped reading.
    pub fn send_raw(&self, line: &str) {
        self.lines
            .send(line.to_owned())
            .expect("the mode is reading");
    }

    /// The parsed lines written so far.
    ///
    /// # Panics
    /// A stdout line that is not JSON, which the protocol forbids.
    pub fn parsed(&self) -> Vec<Value> {
        self.captured.parsed()
    }

    /// Poll until `predicate` holds over the parsed lines.
    ///
    /// # Panics
    /// The five-second deadline elapsing.
    pub async fn wait_for(&self, predicate: impl Fn(&[Value]) -> bool) -> Vec<Value> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let lines = self.parsed();
            if predicate(&lines) {
                return lines;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for the output; saw {lines:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Stop the mode by closing its input and await the exit code.
    ///
    /// # Panics
    /// The mode task panicking.
    pub async fn shutdown(self) -> i32 {
        let Self { lines, task, .. } = self;
        drop(lines);
        tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("the mode stops at EOF")
            .expect("the mode task does not panic")
    }
}
