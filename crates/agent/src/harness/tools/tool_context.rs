//! The filesystem-and-shell context the built-in execution tools resolve
//! their environment through, ported from upstream
//! `src/harness/tools/tool-context.ts`.
//!
//! Upstream parameterizes every built-in tool over
//! `TContext extends ExecutionToolContext` and destructures `env` from the
//! context; the erased harness carries the context as
//! [`crate::harness::types::ToolContext`] (`Option<Arc<dyn Any>>`), so the
//! bound restates as this trait: a tool creator is generic over the context
//! type and downcasts the erased value to it at execution time, and an
//! application with a richer context implements the trait for its own type.

use std::sync::Arc;

use crate::harness::context::Context;
use crate::harness::types::ExecutionEnv;

/// The context bound the built-in execution tools execute against,
/// upstream's `ExecutionToolContext` interface.
pub trait ExecutionToolContext: Send + Sync + 'static {
    /// The filesystem and shell the tools run against, upstream's `env`.
    fn env(&self) -> &Arc<dyn ExecutionEnv>;
}

/// The plain `{ env }` context, the shape upstream's production consumers
/// pass (`toolContext: { env: executionEnv }`) and the default
/// `TContext extends ExecutionToolContext = ExecutionToolContext`.
pub struct EnvToolContext {
    /// The execution environment.
    pub env: Arc<dyn ExecutionEnv>,
}

impl std::fmt::Debug for EnvToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvToolContext").finish_non_exhaustive()
    }
}

impl ExecutionToolContext for EnvToolContext {
    fn env(&self) -> &Arc<dyn ExecutionEnv> {
        &self.env
    }
}

/// The erased-to-typed context downcast the tool creators run, upstream's
/// type-level `TContext extends ExecutionToolContext` constraint.
///
/// # Errors
/// When the context is absent or not the creator's context type — an
/// internal contract violation the harness cannot produce; tools surface it
/// as a failed execution rather than panic.
pub fn execution_tool_context<C: ExecutionToolContext>(
    context: &crate::harness::types::ToolContext,
) -> Result<Arc<C>, crate::types::AgentToolError> {
    let context = context.as_ref().ok_or_else(|| missing_context_error())?;
    Arc::clone(context)
        .downcast::<C>()
        .map_err(|_| downcast_error())
}

fn missing_context_error() -> crate::types::AgentToolError {
    io_error("the built-in tool ran without its execution tool context")
}

fn downcast_error() -> crate::types::AgentToolError {
    io_error("the tool context does not carry the expected execution tool context type")
}

/// Whether the context's abort signal has fired, upstream's
/// `context.abortSignal?.aborted`.
pub(crate) fn is_aborted(context: &Context) -> bool {
    context
        .abort_signal()
        .is_some_and(|signal| signal.aborted())
}

/// The abort throw, upstream's `throw new Error("Operation aborted")`.
pub(crate) fn aborted_error() -> crate::types::AgentToolError {
    io_error("Operation aborted")
}

/// The boxed error the tools surface, upstream's `throw new Error(message)`.
pub(crate) fn io_error(message: impl Into<String>) -> crate::types::AgentToolError {
    Box::new(std::io::Error::other(message.into()))
}
