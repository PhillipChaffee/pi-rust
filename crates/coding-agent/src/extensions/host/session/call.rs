//! The call-scope machinery: one dispatch's send, read loop, and in-scope
//! answers ([ADR 0010]).
//!
//! A scope owns its `call` id, its signal ids, and the frames it forwards to
//! the caller (tool updates, provider stream events). Context and UI calls
//! that arrive while the scope runs are answered here; any other frame is a
//! protocol failure that kills the session.
//!
//! [ADR 0010]: ../../../../../../docs/adr/0010-extension-host-wire-protocol.md

use pi_agent_core::harness::context::AbortSignal;
use serde_json::Value;
use tokio::task::JoinHandle;

use super::super::context::{HostCallError, call_context};
use super::super::{Frame, WireError};
use super::connection::{SessionCore, SignalAbort};
use crate::extensions::runtime::HandlerContext;

/// What a finished scope produced.
pub(super) enum CallOutcome {
    /// The child's `call_result` value.
    Result(Value),
    /// The child's `stream_end` for a provider stream.
    StreamEnd,
}

/// The in-scope frames the caller consumes while the scope runs.
#[derive(Clone, Copy, Default)]
pub(super) struct CallSink<'a> {
    /// A tool's streaming update.
    pub(super) update: Option<&'a (dyn Fn(Value) -> Result<(), HostCallError> + Send + Sync)>,
    /// A provider's stream event.
    pub(super) stream: Option<&'a (dyn Fn(Value) -> Result<(), HostCallError> + Send + Sync)>,
}

/// The signal ids a call scope allocated and the watchers that report their
/// aborts. Dropping the scope cancels every watcher.
pub(super) struct SignalScope {
    call: u64,
    next: u64,
    tx: tokio::sync::mpsc::UnboundedSender<SignalAbort>,
    watchers: Vec<JoinHandle<()>>,
}

impl SignalScope {
    pub(super) const fn new(
        call: u64,
        tx: tokio::sync::mpsc::UnboundedSender<SignalAbort>,
    ) -> Self {
        Self {
            call,
            next: 0,
            tx,
            watchers: Vec::new(),
        }
    }

    /// Allocate the next wire signal id and watch it for the scope's abort.
    pub(super) fn allocate(&mut self, signal: &AbortSignal) -> u64 {
        self.next += 1;
        let id = self.next;
        let signal = signal.clone();
        let call = self.call;
        let tx = self.tx.clone();
        self.watchers.push(tokio::spawn(async move {
            signal.wait().await;
            let _ = tx.send(SignalAbort { call, signal: id });
        }));
        id
    }
}

impl Drop for SignalScope {
    fn drop(&mut self) {
        for watcher in self.watchers.drain(..) {
            watcher.abort();
        }
    }
}

impl<T> SessionCore<T>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    /// Run one call scope to its result, answering nested requests.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the link and abort guards must span the whole call scope"
    )]
    pub(super) async fn run_call(
        &self,
        call: u64,
        frame: Frame,
        context: &HandlerContext,
        scope: &mut SignalScope,
        sink: CallSink<'_>,
        event: &str,
    ) -> Result<CallOutcome, HostCallError> {
        if self.is_dead() {
            return Err(self.dead_failure());
        }
        let mut link = self.link.lock().await;
        link.write_frame(&frame)
            .await
            .map_err(|error| self.fail(event, Self::write_failure(&error)))?;
        let mut aborts = self.abort_rx.lock().await;
        loop {
            tokio::select! {
                notice = aborts.recv() => {
                    let Some(notice) = notice else {
                        return Err(self.fail(
                            event,
                            HostCallError::new("extension_exited", "the extension session closed"),
                        ));
                    };
                    if notice.call == call {
                        link.write_frame(&Frame::Signal {
                            call,
                            signal: notice.signal,
                            aborted: true,
                        })
                        .await
                        .map_err(|error| self.fail(event, Self::write_failure(&error)))?;
                    }
                }
                incoming = link.read_frame() => {
                    let Some(incoming) = incoming.map_err(|error| self.fail(event, Self::link_failure(&error)))? else {
                        return Err(self.fail(
                            event,
                            HostCallError::new(
                                "extension_exited",
                                format!("the extension exited during a {event} call"),
                            ),
                        ));
                    };
                    match incoming {
                        Frame::ContextCall { call: owner, id, method, params } if owner == call => {
                            let value = call_context(
                                context,
                                &method,
                                params.as_ref(),
                                self.ui.is_some(),
                                &mut |signal| scope.allocate(signal),
                            );
                            let answer = match value {
                                Ok(value) => Frame::ContextResult { call, id, value },
                                Err(error) => Frame::ContextError {
                                    call,
                                    id,
                                    code: error.code().to_string(),
                                    message: error.message().to_string(),
                                },
                            };
                            link.write_frame(&answer)
                                .await
                                .map_err(|error| self.fail(event, Self::write_failure(&error)))?;
                        }
                        Frame::UiCall { call: owner, id, method, params } if owner == call => {
                            let answer = self.answer_ui(call, id, &method, params).await;
                            link.write_frame(&answer)
                                .await
                                .map_err(|error| self.fail(event, Self::write_failure(&error)))?;
                        }
                        Frame::CallUpdate { call: owner, update } if owner == call => {
                            match sink.update {
                                Some(update_sink) => update_sink(update)?,
                                None => return Err(self.protocol_failure(event, "a tool update")),
                            }
                        }
                        Frame::StreamEvent { call: owner, event: stream_event } if owner == call => {
                            match sink.stream {
                                Some(stream_sink) => stream_sink(stream_event)?,
                                None => return Err(self.protocol_failure(event, "a stream event")),
                            }
                        }
                        Frame::StreamEnd { call: owner } if owner == call => {
                            if sink.stream.is_none() {
                                return Err(self.protocol_failure(event, "a stream end"));
                            }
                            return Ok(CallOutcome::StreamEnd);
                        }
                        Frame::CallResult { call: owner, result } if owner == call => {
                            return Ok(CallOutcome::Result(result));
                        }
                        Frame::CallError { call: owner, code, message, stack } if owner == call => {
                            return Err(HostCallError::new(code, message).with_stack(stack));
                        }
                        other => {
                            return Err(self.protocol_failure(event, &format!("{other:?}")));
                        }
                    }
                }
            }
        }
    }

    async fn answer_ui(&self, call: u64, id: u64, method: &str, params: Option<Value>) -> Frame {
        let Some(ui) = &self.ui else {
            let error = HostCallError::unsupported("no UI is bound to this extension session");
            return Frame::UiError {
                call,
                id,
                code: error.code().to_string(),
                message: error.message().to_string(),
            };
        };
        match ui.call(method, params).await {
            Ok(value) => Frame::UiResult { call, id, value },
            Err(error) => Frame::UiError {
                call,
                id,
                code: error.code().to_string(),
                message: error.message().to_string(),
            },
        }
    }

    fn link_failure(error: &WireError) -> HostCallError {
        HostCallError::new("extension_protocol_error", error.message().to_string())
    }

    fn write_failure(error: &WireError) -> HostCallError {
        HostCallError::new("extension_exited", error.message().to_string())
    }

    fn protocol_failure(&self, event: &str, what: &str) -> HostCallError {
        self.fail(
            event,
            HostCallError::new(
                "extension_protocol_error",
                format!("the extension sent an unexpected {what} during dispatch"),
            ),
        )
    }
}
