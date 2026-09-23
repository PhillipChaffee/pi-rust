//! The harness hook registry and aggregate runner, ported from upstream
//! `src/harness/hooks.ts`.
//!
//! The registry keeps ordered registrations per hook name and runs the
//! per-name aggregates: fail-closed for `before_drive`, first-match
//! structural results for `before_compaction`/`before_navigation`,
//! thread-through folding for the rest, with isolated handler failures
//! reported through the injected reporter. Tool hooks run one telemetry
//! span per registered handler, carrying the admission outcome.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;
use serde_json::Value as JsonValue;

use crate::harness::agent_harness::{
    HookEvent, HookHandler, HookOptions, HookResult, HookInvocation, StepKind,
};
use crate::harness::context::{with_abort_signal, Context};
use crate::harness::gate::Gate;
use crate::harness::result::HarnessError;
use crate::harness::types::{AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch};
use crate::types::AgentMessage;

/// The error reporter the registry forwards handler failures to, upstream's
/// `HookErrorReporter`.
pub type HookErrorReporter = Arc<
    dyn Fn(Box<dyn std::error::Error + Send + Sync>, HookName, &str, &Context) -> BoxedFuture<'_, ()>
        + Send
        + Sync,
>;

struct HookRegistration {
    id: Option<String>,
    handler: HookHandler,
    token: u64,
}

/// Ordered harness hook registry and aggregate runner, upstream's
/// `HookRegistry`.
pub struct HookRegistry {
    registrations: Arc<Mutex<HashMap<HookName, Vec<HookRegistration>>>>,
    report_error: HookErrorReporter,
    closed_error: Mutex<Option<String>>,
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistry").finish_non_exhaustive()
    }
}

impl HookRegistry {
    /// Builds a registry over the supplied error reporter.
    #[must_use]
    pub fn new(report_error: HookErrorReporter) -> Self {
        Self {
            registrations: Arc::new(Mutex::new(HashMap::new())),
            report_error,
            closed_error: Mutex::new(None),
        }
    }

    /// Registers one handler and returns its unsubscription, upstream's
    /// `on`.
    ///
    /// # Errors
    /// The close error when the registry has closed.
    pub fn on(
        &self,
        name: HookName,
        handler: HookHandler,
        options: crate::harness::agent_harness::HookOptions,
    ) -> Result<crate::harness::agent_harness::Subscription, String> {
        if let Some(closed_error) = self.closed_error.lock().expect("hook closed lock").clone() {
            return Err(closed_error);
        }
        static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);
        let registration = HookRegistration {
            id: options.id.clone(),
            handler,
            token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        };
        let token = registration.token;
        self.registrations
            .lock()
            .expect("hook registrations lock")
            .entry(name)
            .or_default()
            .push(registration);
        let registrations = Arc::clone(&self.registrations);
        Ok(crate::harness::agent_harness::Subscription::new(Arc::new(
            move || {
                if let Some(list) = registrations
                    .lock()
                    .expect("hook registrations lock")
                    .get_mut(&name)
                {
                    list.retain(|registration| registration.token != token);
                }
            },
        )))
    }

    /// Whether any handler is registered under the name, upstream's `has`.
    #[must_use]
    pub fn has(&self, name: HookName) -> bool {
        self.registrations
            .lock()
            .expect("hook registrations lock")
            .get(&name)
            .is_some_and(|list| !list.is_empty())
    }

    /// Closes the registry with its error; the first error wins, upstream's
    /// `close`.
    pub fn close(&self, error: String) {
        let mut closed = self.closed_error.lock().expect("hook closed lock");
        if closed.is_none() {
            *closed = Some(error);
        }
    }

    /// Invokes one accepted-operation aggregate after synchronously
    /// passing its effect gate, upstream's `runWithGate`.
    ///
    /// # Errors
    /// The gate rejection, a closed registry, or a fail-closed handler
    /// failure.
    pub async fn run_with_gate(
        &self,
        name: HookName,
        event: HookInvocation,
        gate: &Gate,
        context: &Context,
    ) -> Result<HookResult, HookRunError> {
        let admitted_context =
            gate.admit(|| with_abort_signal(gate.signal().clone(), context))?;
        if let Some(signal) = admitted_context.abort_signal()
            && let Some(reason) = signal.reason()
        {
            return Err(HookRunError::Aborted(reason));
        }
        self.run_admitted(name, event, &admitted_context).await
    }

    /// Invokes a tool-hook aggregate with one telemetry span per
    /// registered handler, upstream's `runToolWithGate`.
    ///
    /// # Errors
    /// The gate rejection when admission refused, or a closed registry.
    pub async fn run_tool_with_gate(
        &self,
        name: HookName,
        event: HookInvocation,
        gate: &Gate,
        context: &Context,
    ) -> Result<HookResult, HookRunError> {
        let admitted_context =
            gate.admit(|| with_abort_signal(gate.signal().clone(), context))?;
        if let Some(signal) = admitted_context.abort_signal()
            && let Some(reason) = signal.reason()
        {
            return Err(HookRunError::Aborted(reason));
        }
        match name {
            HookName::BeforeTool | HookName::AfterTool => self.run_admitted(name, event, &admitted_context).await,
            _ => Err(HookRunError::Closed(HarnessError::Closed {
                message: format!("{} is not a tool hook", name.as_str()),
            })),
        }
    }

    async fn run_admitted(
        &self,
        name: HookName,
        event: HookInvocation,
        context: &Context,
    ) -> Result<HookResult, HookRunError> {
        if let Some(closed_error) = self.closed_error.lock().expect("hook closed lock").clone() {
            return Err(HookRunError::Closed(HarnessError::Closed {
                message: closed_error,
            }));
        }
        match name {
            HookName::BeforeRun => Ok(HookResult::BeforeRun(
                self.before_run(&event, context).await,
            )),
            HookName::BeforeDrive => {
                self.before_drive(&event, context).await?;
                Ok(HookResult::BeforeDrive)
            }
            HookName::BeforeRunEnd => Ok(HookResult::BeforeRunEnd(
                self.before_run_end(&event, context).await,
            )),
            HookName::TransformContext => Ok(HookResult::TransformContext(Some(
                self.transform_context(&event, context).await,
            ))),
            HookName::BeforeRequest => Ok(HookResult::BeforeRequest(
                self.before_request(&event, context).await,
            )),
            HookName::BeforePayload => Ok(HookResult::BeforePayload(Some(
                self.before_payload(&event, context).await,
            ))),
            HookName::AfterResponse => Ok(HookResult::AfterResponse(Some(
                self.after_response(&event, context).await,
            ))),
            HookName::BeforeTool => Ok(HookResult::BeforeTool(
                self.before_tool(&event, context).await,
            )),
            HookName::AfterTool => Ok(HookResult::AfterTool(
                self.after_tool(&event, context).await,
            )),
            HookName::BeforeCompaction => Ok(HookResult::BeforeCompaction(
                self.first_structural(HookName::BeforeCompaction, &event, context)
                    .await,
            )),
            HookName::BeforeNavigation => Ok(HookResult::BeforeNavigation(
                self.first_structural(HookName::BeforeNavigation, &event, context)
                    .await,
            )),
        }
    }

    async fn before_run(
        &self,
        event: &HookInvocation,
        context: &Context,
    ) -> Option<crate::harness::agent_harness::BeforeRunResult> {
        let HookEvent::BeforeRun { prompt, resources } = &event.event else {
            return None;
        };
        let mut prompt = prompt.clone();
        let mut injected: Vec<AgentMessage> = Vec::new();
        for registration in self.registrations_for(HookName::BeforeRun) {
            let current = HookInvocation {
                lane: event.lane.clone(),
                run_id: event.run_id.clone(),
                event: HookEvent::BeforeRun {
                    prompt: prompt.clone(),
                    resources: resources.clone(),
                },
            };
            match (registration.handler)(&current, context).await {
                Ok(HookResult::BeforeRun(Some(result))) => {
                    injected.extend(result.messages.iter().cloned());
                    prompt.extend(result.messages.iter().cloned());
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::BeforeRun, &event.lane, context).await;
                }
            }
        }
        (!injected.is_empty()).then_some(crate::harness::agent_harness::BeforeRunResult {
            messages: injected,
        })
    }

    async fn before_tool(
        &self,
        event: &HookInvocation,
        context: &Context,
    ) -> Option<crate::harness::agent_harness::BeforeToolResult> {
        let HookEvent::BeforeTool {
            tool_call_id,
            tool_name,
            args: original_args,
        } = &event.event
        else {
            return None;
        };
        let mut args = original_args.clone();
        let mut block: Option<crate::harness::agent_harness::ToolBlock> = None;
        for registration in self.registrations_for(HookName::BeforeTool) {
            let current = HookInvocation {
                lane: event.lane.clone(),
                run_id: event.run_id.clone(),
                event: HookEvent::BeforeTool {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                },
            };
            match invoke_tool_registration(HookName::BeforeTool, &registration, &current, context)
                .await
            {
                Ok(HookResult::BeforeTool(Some(result))) => {
                    if let Some(replacement) = result.args {
                        args = replacement;
                    }
                    if let Some(block_result) = result.block {
                        block = Some(block_result);
                        break;
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::BeforeTool, &event.lane, context).await;
                    block = Some(crate::harness::agent_harness::ToolBlock {
                        reason: error.to_string(),
                        terminate: None,
                    });
                    break;
                }
            }
        }
        Some(crate::harness::agent_harness::BeforeToolResult {
            args: (args != *original_args).then_some(args),
            block,
        })
    }

    async fn transform_context(
        &self,
        event: &HookInvocation,
        context: &Context,
    ) -> crate::harness::agent_harness::TransformContextResult {
        let HookEvent::TransformContext {
            messages: original_messages,
            system_prompt: original_system_prompt,
        } = &event.event
        else {
            return crate::harness::agent_harness::TransformContextResult {
                messages: None,
                system_prompt: None,
            };
        };
        let mut messages = original_messages.clone();
        let mut system_prompt = original_system_prompt.clone();
        for registration in self.registrations_for(HookName::TransformContext) {
            let current = HookInvocation {
                lane: event.lane.clone(),
                run_id: event.run_id.clone(),
                event: HookEvent::TransformContext {
                    messages: messages.clone(),
                    system_prompt: system_prompt.clone(),
                },
            };
            match (registration.handler)(&current, context).await {
                Ok(HookResult::TransformContext(Some(result))) => {
                    if let Some(replacement) = result.messages {
                        messages = replacement;
                    }
                    if let Some(replacement) = result.system_prompt {
                        system_prompt = replacement;
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::TransformContext, &event.lane, context)
                        .await;
                }
            }
        }
        crate::harness::agent_harness::TransformContextResult {
            messages: (messages != *original_messages).then_some(messages),
            system_prompt: (system_prompt != *original_system_prompt).then_some(system_prompt),
        }
    }

    async fn before_request(
        &self,
        event: &HookInvocation,
        context: &Context,
    ) -> Option<crate::harness::agent_harness::BeforeRequestResult> {
        let HookEvent::BeforeRequest {
            model,
            step,
            attempt,
            stream_options: original_options,
        } = &event.event
        else {
            return None;
        };
        let mut stream_options = original_options.clone();
        let mut changed = false;
        for registration in self.registrations_for(HookName::BeforeRequest) {
            let current = HookInvocation {
                lane: event.lane.clone(),
                run_id: event.run_id.clone(),
                event: HookEvent::BeforeRequest {
                    model: model.clone(),
                    step: *step,
                    attempt: *attempt,
                    stream_options: stream_options.clone(),
                },
            };
            match (registration.handler)(&current, context).await {
                Ok(HookResult::BeforeRequest(Some(result))) => {
                    stream_options =
                        apply_stream_options_patch(&stream_options, &result.stream_options);
                    changed = true;
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::BeforeRequest, &event.lane, context).await;
                }
            }
        }
        changed.then(|| crate::harness::agent_harness::BeforeRequestResult {
            stream_options: create_stream_options_patch(original_options, &stream_options),
        })
    }

    async fn before_payload(
        &self,
        event: &HookInvocation,
        context: &Context,
    ) -> crate::harness::agent_harness::PayloadResult {
        let HookEvent::BeforePayload { model, payload: original_payload } = &event.event else {
            return crate::harness::agent_harness::PayloadResult {
                payload: JsonValue::Null,
            };
        };
        let mut payload = original_payload.clone();
        for registration in self.registrations_for(HookName::BeforePayload) {
            let current = HookInvocation {
                lane: event.lane.clone(),
                run_id: event.run_id.clone(),
                event: HookEvent::BeforePayload {
                    model: model.clone(),
                    payload: payload.clone(),
                },
            };
            match (registration.handler)(&current, context).await {
                Ok(HookResult::BeforePayload(Some(result))) => {
                    payload = result.payload;
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::BeforePayload, &event.lane, context).await;
                }
            }
        }
        crate::harness::agent_harness::PayloadResult { payload }
    }

    async fn after_response(
        &self,
        event: &HookInvocation,
        context: &Context,
    ) -> crate::harness::agent_harness::AfterResponseResult {
        let HookEvent::AfterResponse {
            status,
            headers,
            message: original_message,
        } = &event.event
        else {
            return crate::harness::agent_harness::AfterResponseResult { message: None };
        };
        let mut message = original_message.clone();
        for registration in self.registrations_for(HookName::AfterResponse) {
            let current = HookInvocation {
                lane: event.lane.clone(),
                run_id: event.run_id.clone(),
                event: HookEvent::AfterResponse {
                    status: *status,
                    headers: headers.clone(),
                    message: message.clone(),
                },
            };
            match (registration.handler)(&current, context).await {
                Ok(HookResult::AfterResponse(Some(result))) => {
                    if let Some(replacement) = result.message {
                        message = replacement;
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::AfterResponse, &event.lane, context).await;
                }
            }
        }
        crate::harness::agent_harness::AfterResponseResult {
            message: Some(message),
        }
    }

    async fn after_tool(
        &self,
        event: &HookInvocation,
        context: &Context,
    ) -> Option<crate::harness::agent_harness::AfterToolResult> {
        let HookEvent::AfterTool {
            tool_call_id,
            tool_name,
            args,
            content: original_content,
            details: original_details,
            is_error: original_is_error,
            usage: original_usage,
        } = &event.event
        else {
            return None;
        };
        let mut current_content = original_content.clone();
        let mut current_details = original_details.clone();
        let mut current_is_error = *original_is_error;
        let mut current_usage = original_usage.clone();
        let mut aggregate = crate::harness::agent_harness::AfterToolResult::default();
        for registration in self.registrations_for(HookName::AfterTool) {
            let current = HookInvocation {
                lane: event.lane.clone(),
                run_id: event.run_id.clone(),
                event: HookEvent::AfterTool {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                    content: current_content.clone(),
                    details: current_details.clone(),
                    is_error: current_is_error,
                    usage: current_usage.clone(),
                },
            };
            match invoke_tool_registration(HookName::AfterTool, &registration, &current, context)
                .await
            {
                Ok(HookResult::AfterTool(Some(result))) => {
                    if let Some(content) = result.content {
                        aggregate.content = Some(content.clone());
                        current_content = content;
                    }
                    if let Some(details) = result.details {
                        aggregate.details = Some(details.clone());
                        current_details = Some(details);
                    }
                    if let Some(is_error) = result.is_error {
                        aggregate.is_error = Some(is_error);
                        current_is_error = is_error;
                    }
                    if let Some(usage) = result.usage {
                        aggregate.usage = Some(usage.clone());
                        current_usage = Some(usage);
                    }
                    if let Some(terminate) = result.terminate {
                        aggregate.terminate = Some(terminate);
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::AfterTool, &event.lane, context).await;
                }
            }
        }
        let has_any = aggregate.content.is_some()
            || aggregate.details.is_some()
            || aggregate.is_error.is_some()
            || aggregate.usage.is_some()
            || aggregate.terminate.is_some();
        has_any.then_some(aggregate)
    }

    async fn first_structural(
        &self,
        name: HookName,
        event: &HookInvocation,
        context: &Context,
    ) -> Option<HookResult> {
        let field = match name {
            HookName::BeforeCompaction => StructuralField::Compaction,
            HookName::BeforeNavigation => StructuralField::Summary,
            _ => return None,
        };
        for registration in self.registrations_for(name) {
            match (registration.handler)(event, context).await {
                Ok(HookResult::BeforeCompaction(Some(result))) if field == StructuralField::Compaction => {
                    if result.decline == Some(true) && result.compaction.is_some() {
                        (self.report_error)(
                            Box::new(std::io::Error::other(format!(
                                "{} hook cannot return both decline and {}",
                                name.as_str(),
                                field.as_str()
                            ))),
                            name,
                            &event.lane,
                            context,
                        )
                        .await;
                        continue;
                    }
                    if result.decline == Some(true) || result.compaction.is_some() {
                        return Some(HookResult::BeforeCompaction(Some(result)));
                    }
                }
                Ok(HookResult::BeforeNavigation(Some(result)))
                    if field == StructuralField::Summary =>
                {
                    if result.decline == Some(true) && result.summary.is_some() {
                        (self.report_error)(
                            Box::new(std::io::Error::other(format!(
                                "{} hook cannot return both decline and {}",
                                name.as_str(),
                                field.as_str()
                            ))),
                            name,
                            &event.lane,
                            context,
                        )
                        .await;
                        continue;
                    }
                    if result.decline == Some(true) || result.summary.is_some() {
                        return Some(HookResult::BeforeNavigation(Some(result)));
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, name, &event.lane, context).await;
                }
            }
        }
        None
    }

    async fn before_drive(&self, event: &HookInvocation, context: &Context) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for registration in self.registrations_for(HookName::BeforeDrive) {
            match (registration.handler)(event, context).await {
                Ok(_) => {}
                Err(error) => {
                    (self.report_error)(error, HookName::BeforeDrive, &event.lane, context).await;
                    return Err(Box::new(std::io::Error::other("before_drive handler failed")));
                }
            }
        }
        Ok(())
    }

    fn registrations_for(&self, name: HookName) -> Vec<HookRegistration> {
        self.registrations
            .lock()
            .expect("hook registrations lock")
            .get(&name)
            .cloned()
            .unwrap_or_default()
    }
}

/// The error one hook run reports, upstream's `runWithGate` rejections.
#[derive(Debug)]
pub enum HookRunError {
    /// The gate refused admission for cancellation.
    Aborted(pi_chord::context::AbortReason),
    /// The registry or gate closed.
    Closed(HarnessError),
    /// A fail-closed handler failed.
    Handler(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for HookRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aborted(reason) => write!(f, "{reason}"),
            Self::Closed(error) => write!(f, "{error}"),
            Self::Handler(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for HookRunError {}

impl From<crate::harness::gate::GateRejection> for HookRunError {
    fn from(rejection: crate::harness::gate::GateRejection) -> Self {
        match rejection {
            crate::harness::gate::GateRejection::AbortRequested(_) => {
                Self::Aborted(pi_chord::context::AbortReason::Aborted)
            }
            crate::harness::gate::GateRejection::Closed(error) => Self::Closed(HarnessError::Closed {
                message: error.0,
            }),
            crate::harness::gate::GateRejection::Aborted(reason) => Self::Aborted(reason),
        }
    }
}

/// The structural result field a first-match runner consults, upstream's
/// `"compaction" | "summary"` discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructuralField {
    /// The compaction result field.
    Compaction,
    /// The summary result field.
    Summary,
}

impl StructuralField {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Compaction => "compaction",
            Self::Summary => "summary",
        }
    }
}

/// Invokes one tool-hook registration under one telemetry span, upstream's
/// `invokeToolRegistration`: the span carries the lane, operation id, hook
/// name, and optional registration id, and records the blocked/completed/
/// failed outcome.
async fn invoke_tool_registration(
    name: HookName,
    registration: &HookRegistration,
    event: &HookInvocation,
    context: &Context,
) -> Result<HookResult, Box<dyn std::error::Error + Send + Sync>> {
    let attributes = crate::harness::telemetry::harness_schema::hook::Start {
        lane_name: event.lane.clone(),
        operation_id: Some(event.run_id.clone()),
        hook_name: hook_name_span(name),
        hook_registration_id: registration.id.clone(),
    };
    let handler = registration.handler.clone();
    let event = event.clone();
    crate::harness::telemetry::start_harness_span(
        crate::harness::telemetry::harness_schema::hook::Span,
        attributes,
        move |span, span_context| async move {
            let result = (handler)(&event, &span_context).await;
            let blocked = matches!(
                (&event.event, &result),
                (
                    HookEvent::BeforeTool { .. },
                    Ok(HookResult::BeforeTool(Some(result)))
                ) if result.block.is_some()
            );
            let outcome = if result.is_err() {
                crate::harness::telemetry::harness_schema::hook::HookOutcome::Failed
            } else if blocked {
                crate::harness::telemetry::harness_schema::hook::HookOutcome::Blocked
            } else {
                crate::harness::telemetry::harness_schema::hook::HookOutcome::Completed
            };
            span.set_attributes(
                crate::harness::telemetry::harness_schema::hook::End {
                    hook_outcome: Some(outcome),
                }
                .into_span_attributes(),
            );
            result
        },
        context,
    )
    .await
}

fn hook_name_span(
    name: HookName,
) -> crate::harness::telemetry::harness_schema::hook::HookNameSpan {
    use crate::harness::telemetry::harness_schema::hook::HookNameSpan as Span;
    match name {
        HookName::BeforeRun => Span::BeforeRun,
        HookName::BeforeDrive => Span::BeforeDrive,
        HookName::BeforeRunEnd => Span::BeforeRunEnd,
        HookName::TransformContext => Span::TransformContext,
        HookName::BeforeRequest => Span::BeforeRequest,
        HookName::BeforePayload => Span::BeforePayload,
        HookName::AfterResponse => Span::AfterResponse,
        HookName::BeforeTool => Span::BeforeTool,
        HookName::AfterTool => Span::AfterTool,
        HookName::BeforeCompaction => Span::BeforeCompaction,
        HookName::BeforeNavigation => Span::BeforeNavigation,
    }
}

/// Applies one stream-options patch over the base, upstream's
/// `applyStreamOptionsPatch`.
#[must_use]
pub fn apply_stream_options_patch(
    base: &AgentHarnessStreamOptions,
    patch: &crate::harness::types::AgentHarnessStreamOptionsPatch,
) -> AgentHarnessStreamOptions {
    let mut next = base.clone();
    apply_scalar(&mut next.transport, patch.transport.clone());
    apply_scalar(&mut next.timeout_ms, patch.timeout_ms);
    apply_scalar(&mut next.max_retries, patch.max_retries);
    apply_scalar(&mut next.max_retry_delay_ms, patch.max_retry_delay_ms);
    apply_scalar(&mut next.cache_retention, patch.cache_retention);
    apply_scalar(&mut next.deferred, patch.deferred.clone());
    if let Some(headers_patch) = &patch.headers {
        match headers_patch {
            None => next.headers = None,
            Some(entries) => {
                let mut headers = next.headers.clone().unwrap_or_default();
                for (key, entry) in entries {
                    match entry {
                        Some(value) => {
                            headers.insert(key.clone(), Some(value.clone()));
                        }
                        None => {
                            headers.remove(key);
                        }
                    }
                }
                next.headers = Some(headers);
            }
        }
    }
    if let Some(metadata_patch) = &patch.metadata {
        match metadata_patch {
            None => next.metadata = None,
            Some(entries) => {
                let mut metadata = next.metadata.clone().unwrap_or_default();
                for (key, entry) in entries {
                    match entry {
                        Some(value) => {
                            metadata.insert(key.clone(), value.clone());
                        }
                        None => {
                            metadata.remove(key);
                        }
                    }
                }
                next.metadata = Some(metadata);
            }
        }
    }
    next
}

fn apply_scalar<T>(target: &mut Option<T>, patch: Option<Option<T>>) {
    match patch {
        Some(None) => *target = None,
        Some(Some(value)) => *target = Some(value),
        None => {}
    }
}

/// Derives the patch between two stream-option snapshots, upstream's
/// `createStreamOptionsPatch`.
#[must_use]
pub fn create_stream_options_patch(
    base: &AgentHarnessStreamOptions,
    value: &AgentHarnessStreamOptions,
) -> crate::harness::types::AgentHarnessStreamOptionsPatch {
    let mut patch = crate::harness::types::AgentHarnessStreamOptionsPatch::default();
    if base.transport != value.transport {
        patch.transport = Some(value.transport.clone());
    }
    if base.timeout_ms != value.timeout_ms {
        patch.timeout_ms = Some(value.timeout_ms);
    }
    if base.max_retries != value.max_retries {
        patch.max_retries = Some(value.max_retries);
    }
    if base.max_retry_delay_ms != value.max_retry_delay_ms {
        patch.max_retry_delay_ms = Some(value.max_retry_delay_ms);
    }
    if base.cache_retention != value.cache_retention {
        patch.cache_retention = Some(value.cache_retention);
    }
    if base.deferred != value.deferred {
        patch.deferred = Some(value.deferred.clone());
    }
    if base.headers != value.headers {
        patch.headers = Some(map_patch(
            base.headers.as_ref(),
            value.headers.as_ref(),
        ));
    }
    if base.metadata != value.metadata {
        patch.metadata = Some(map_patch(
            base.metadata.as_ref(),
            value.metadata.as_ref(),
        ));
    }
    patch
}

/// The per-key patch between two maps: deleted keys carry `None`, changed
/// or added keys carry their new value. When the base map was absent and
/// the derived patch is empty, upstream still emits the explicit empty
/// patch (`{}`), so `Some` with an empty map is meaningful.
fn map_patch<V: PartialEq + Clone>(
    base: Option<&BTreeMap<String, V>>,
    value: Option<&BTreeMap<String, V>>,
) -> Option<BTreeMap<String, Option<V>>> {
    let Some(value_map) = value else {
        return None;
    };
    let base_map = base.unwrap_or(&BTreeMap::new());
    let mut patch = BTreeMap::new();
    for key in base_map.keys() {
        if !value_map.contains_key(key) {
            patch.insert(key.clone(), None);
        }
    }
    for (key, entry) in value_map {
        if base_map.get(key) != Some(entry) {
            patch.insert(key.clone(), Some(entry.clone()));
        }
    }
    if base.is_none() && patch.is_empty() {
        Some(BTreeMap::new())
    } else if !patch.is_empty() {
        Some(patch)
    } else {
        None
    }
}

#[cfg(test)]
mod tests;