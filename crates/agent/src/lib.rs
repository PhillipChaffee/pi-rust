//! The transport-abstracted agent runtime, upstream's `packages/agent`
//! (`@earendil-works/pi-agent-core`) at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The crate carries two layers. The core is the stateful [`Agent`] plus the
//! low-level `agentLoop`: they drive LLM turns, tool execution, and event
//! streaming through the [`types::StreamFn`] seam — a boxed function returning
//! pi-ai's `AssistantMessageEventStream`, the same shape `Models.streamSimple`
//! satisfies — so the runtime never depends on a provider catalog. The
//! harness subtree (lanes, session storage, tools, compaction) lands with its
//! own tickets; this root tracks the crate.
//!
//! Upstream runs on one JavaScript event loop; the port keeps every future
//! single-threaded-contract (`Send` boxed futures on the caller's runtime)
//! and carries cancellation on the same token pi-ai's transport options do.
//! Windows is out of scope for this effort (map ticket "Decide the Rust
//! stack").
//!
//! [`Agent`]: crate::agent::Agent
#![forbid(unsafe_code)]

pub mod agent;
pub mod agent_loop;
pub mod harness;
pub mod search;
pub mod stream_fn;
pub mod types;

pub use agent::{Agent, AgentError, AgentInitialState, AgentOptions};
pub use agent_loop::{
    AgentEventSink, AgentEventStream, AgentLoopError, agent_loop, agent_loop_continue,
    run_agent_loop, run_agent_loop_continue,
};
pub use search::{
    EntrySearchHit, SearchQuery, SessionSearchError, SessionSearchHit, SessionSearchService,
    SessionSearchTopHit,
};
pub use stream_fn::{NoDefaultStreamFn, get_default_stream_fn, set_default_stream_fn};
pub use types::{
    AfterToolCall, AfterToolCallContext, AfterToolCallResult, AgentContext, AgentEvent,
    AgentListener, AgentLoopConfig, AgentLoopTurnUpdate, AgentMessage, AgentPrepareNextTurn,
    AgentPrepareNextTurnWithContext, AgentShouldStopAfterTurn, AgentState, AgentTool,
    AgentToolCall, AgentToolContent, AgentToolError, AgentToolExecuteFn, AgentToolPrepareArguments,
    AgentToolResult, AgentToolUpdateCallback, BeforeToolCall, BeforeToolCallContext,
    BeforeToolCallResult, BoxedFuture, ConvertToLlm, CustomAgentMessage, GetApiKey,
    GetFollowUpMessages, GetSteeringMessages, PrepareNextTurn, PrepareNextTurnContext, PromptInput,
    QueueMode, ShouldStopAfterTurn, ShouldStopAfterTurnContext, StreamFn, ThinkingLevel,
    ToolExecutionMode, ToolReplay, TransformContext,
};
