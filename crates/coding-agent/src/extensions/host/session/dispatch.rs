//! The proxy dispatch: event, tool, and command calls from the in-process
//! record to the wire child ([ADR 0010]). Provider streams ride the sibling
//! module; the registration pass that builds the record rides `binding`.
//!
//! Every method opens one call scope through [`SessionCore::run_call`]. A
//! handler failure reads as [`HandlerResult::None`] for events, an
//! [`AgentToolError`] for tools, and a report for commands, matching the
//! in-process contract where a throw never escapes a handler.
//!
//! [ADR 0010]: ../../../../../../docs/adr/0010-extension-host-wire-protocol.md

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::AgentHarnessToolUpdateCallback;
use pi_agent_core::types::{AgentToolError, AgentToolResult};
use serde_json::Value;

use crate::extensions::events::ExtensionEvent;
use crate::extensions::results::HandlerResult;
use crate::extensions::runtime::HandlerContext;
use crate::extensions::types::ExtensionContext;

use super::super::context::HostCallError;
use super::super::payload::descriptors::{ContextSnapshot, context_snapshot_to_wire};
use super::super::payload::events::event_to_wire;
use super::super::payload::results::result_from_wire;
use super::super::{Frame, WireError};
use super::call::{CallOutcome, CallSink, SignalScope};
use super::connection::SessionCore;

pub(super) mod binding;
mod provider;

impl<T> SessionCore<T>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    /// Dispatch one event handler, reading a failure as no result.
    pub(super) async fn call_event(
        &self,
        event: ExtensionEvent,
        context: HandlerContext,
    ) -> HandlerResult {
        let channel = event.channel();
        if self.is_dead() {
            return HandlerResult::None;
        }
        let call = self.next_call();
        let mut scope = SignalScope::new(call, self.abort_tx.clone());
        let payload = match event_to_wire(&event, &mut |signal| scope.allocate(signal)) {
            Ok(payload) => payload,
            Err(error) => {
                self.report(channel, &Self::payload_failure(&error), None);
                return HandlerResult::None;
            }
        };
        let frame = Frame::CallEvent {
            call,
            channel: channel.to_string(),
            payload,
        };
        match self
            .run_call(
                call,
                frame,
                Some(&context),
                &mut scope,
                CallSink::default(),
                channel,
            )
            .await
        {
            Ok(CallOutcome::Result(value)) => {
                result_from_wire(channel, &value).unwrap_or_else(|error| {
                    self.report(channel, &Self::payload_failure(&error), None);
                    HandlerResult::None
                })
            }
            Ok(CallOutcome::StreamEnd) => {
                self.report(
                    channel,
                    &HostCallError::new(
                        "extension_protocol_error",
                        "the child ended a stream outside a provider call",
                    ),
                    None,
                );
                HandlerResult::None
            }
            Err(error) => {
                self.report(channel, &error, error.stack().map(str::to_string));
                HandlerResult::None
            }
        }
    }

    /// Execute one registered tool over the wire, forwarding its updates.
    pub(super) async fn call_tool(
        &self,
        tool: &str,
        tool_call_id: &str,
        params: Value,
        context: ContextSnapshot,
        signal: Option<&AbortSignal>,
        update: Option<AgentHarnessToolUpdateCallback<'_>>,
    ) -> Result<AgentToolResult, AgentToolError> {
        if self.is_dead() {
            return Err(tool_error(self.dead_failure().message()));
        }
        let call = self.next_call();
        let mut scope = SignalScope::new(call, self.abort_tx.clone());
        if let Some(signal) = signal {
            scope.allocate(signal);
        }
        let context = match context_snapshot_to_wire(&context) {
            Ok(context) => context,
            Err(error) => {
                self.report("tool", &Self::payload_failure(&error), None);
                return Err(tool_error(error.message()));
            }
        };
        let frame = Frame::CallTool {
            call,
            tool: tool.to_string(),
            tool_call_id: tool_call_id.to_string(),
            params,
            context,
        };
        let forward = |value: Value| -> Result<(), HostCallError> {
            let result: AgentToolResult = serde_json::from_value(value).map_err(|error| {
                HostCallError::new(
                    "extension_protocol_error",
                    format!("the tool update is not a tool result: {error}"),
                )
            })?;
            if let Some(update) = update {
                update(&result, None);
            }
            Ok(())
        };
        let sink = CallSink {
            update: Some(&forward),
            stream: None,
        };
        match self
            .run_call(
                call,
                frame,
                self.context().as_ref(),
                &mut scope,
                sink,
                "tool",
            )
            .await
        {
            Ok(CallOutcome::Result(value)) => serde_json::from_value(value).map_err(|error| {
                let failure = HostCallError::new(
                    "extension_protocol_error",
                    format!("the tool result is not a tool result: {error}"),
                );
                self.report("tool", &failure, None);
                tool_error(failure.message())
            }),
            Ok(CallOutcome::StreamEnd) => {
                let failure = HostCallError::new(
                    "extension_protocol_error",
                    "the child ended a stream during a tool call",
                );
                self.report("tool", &failure, None);
                Err(tool_error(failure.message()))
            }
            Err(error) => {
                self.report("tool", &error, error.stack().map(str::to_string));
                Err(tool_error(error.message()))
            }
        }
    }

    /// Invoke one registered command; its result is discarded like the
    /// in-process handler's unit return.
    pub(super) async fn call_command(&self, name: &str, args: &str) {
        if self.is_dead() {
            return;
        }
        let call = self.next_call();
        let mut scope = SignalScope::new(call, self.abort_tx.clone());
        let frame = Frame::CallCommand {
            call,
            name: name.to_string(),
            args: args.to_string(),
        };
        if let Err(error) = self
            .run_call(
                call,
                frame,
                self.context().as_ref(),
                &mut scope,
                CallSink::default(),
                "command",
            )
            .await
        {
            self.report("command", &error, error.stack().map(str::to_string));
        }
    }

    /// The context snapshot a tool call carries: the tool's own context when
    /// it has one, else the session context's accessors.
    pub(super) fn snapshot(
        &self,
        extension_context: Option<&dyn ExtensionContext>,
    ) -> ContextSnapshot {
        if let Some(context) = extension_context {
            return ContextSnapshot::from_extension_context(context);
        }
        let Some(context) = self.context() else {
            return ContextSnapshot {
                cwd: String::new(),
                model: None,
                thinking_level: None,
                session_id: None,
                session_file: None,
            };
        };
        let manager = context.session_manager().ok();
        ContextSnapshot {
            cwd: context.cwd().unwrap_or_default(),
            model: context.get_model().ok().flatten(),
            thinking_level: context.get_thinking_level().ok(),
            session_id: manager.as_ref().map(|manager| manager.session_id.clone()),
            session_file: manager.and_then(|manager| manager.session_file),
        }
    }

    pub(super) fn payload_failure(error: &WireError) -> HostCallError {
        HostCallError::new("extension_protocol_error", error.message().to_string())
    }
}

fn tool_error(message: &str) -> AgentToolError {
    AgentToolError::from(std::io::Error::other(message.to_string()))
}
