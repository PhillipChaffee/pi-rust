//! The agent hooks the session installs: tool interception and per-turn
//! compaction/system-prompt refresh.

use std::sync::Arc;

use pi_agent_core::types::AgentContext;
use pi_agent_core::types::AgentLoopTurnUpdate;
use pi_agent_core::types::PrepareNextTurnContext;
use pi_ai::types::ToolResultBlock;

use super::helpers::model_key;
use super::lock;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    #[expect(
        clippy::too_many_lines,
        reason = "the two hook bodies mirror upstream's interception passes"
    )]
    pub(crate) fn install_agent_tool_hooks(&self) {
        let weak = self.self_weak();
        self.agent
            .set_before_tool_call(Some(Arc::new(move |context, _token| {
                let weak = weak.clone();
                Box::pin(async move {
                    let session = weak.upgrade()?;
                    let runner = session.extension_runner();
                    if !runner.has_handlers("tool_call") {
                        return None;
                    }
                    let tool_call = context.tool_call.clone();
                    let result = runner
                        .emit_tool_call(crate::extensions::events::ExtensionEvent::ToolCall {
                            tool_name: tool_call.name.clone(),
                            tool_call_id: tool_call.id.clone(),
                            input: context.args.clone(),
                        })
                        .await?;
                    Some(pi_agent_core::types::BeforeToolCallResult {
                        block: Some(result.block),
                        reason: result.reason,
                        terminate: Some(result.terminate),
                    })
                })
            })));

        let weak = self.self_weak();
        self.agent
            .set_after_tool_call(Some(Arc::new(move |context, _token| {
                let weak = weak.clone();
                Box::pin(async move {
                    let session = weak.upgrade()?;
                    let runner = session.extension_runner();
                    let hook_result = if runner.has_handlers("tool_result") {
                        runner
                            .emit_tool_result(
                                crate::extensions::events::ExtensionEvent::ToolResult {
                                    tool_name: context.tool_call.name.clone(),
                                    tool_call_id: context.tool_call.id.clone(),
                                    input: context.args.clone(),
                                    content: context.result.content.clone(),
                                    details: context.result.details.clone(),
                                    is_error: context.is_error,
                                    usage: context.result.usage,
                                },
                            )
                            .await
                    } else {
                        None
                    };
                    let content = hook_result
                        .as_ref()
                        .and_then(|result| result.content.clone())
                        .unwrap_or_else(|| context.result.content.clone());
                    let auto_resize_images =
                        lock(&session.settings_manager).get_image_auto_resize();
                    let normalization_options =
                        crate::utils::tool_result_images::NormalizeToolResultImagesOptions {
                            auto_resize_images: Some(auto_resize_images),
                        };
                    let normalized =
                        crate::utils::tool_result_images::normalize_tool_result_images(
                            content
                                .into_iter()
                                .map(|block| match block {
                                    ToolResultBlock::Text(text) => {
                                        crate::utils::tool_result_images::ToolResultContent::Text(
                                            text,
                                        )
                                    }
                                    ToolResultBlock::Image(image) => {
                                        crate::utils::tool_result_images::ToolResultContent::Image(
                                            image,
                                        )
                                    }
                                })
                                .collect(),
                            Some(&normalization_options),
                        )
                        .await
                        .into_iter()
                        .map(|block| match block {
                            crate::utils::tool_result_images::ToolResultContent::Text(text) => {
                                ToolResultBlock::Text(text)
                            }
                            crate::utils::tool_result_images::ToolResultContent::Image(image) => {
                                ToolResultBlock::Image(image)
                            }
                        })
                        .collect::<Vec<_>>();
                    if hook_result.is_none() && normalized == context.result.content {
                        return None;
                    }
                    Some(pi_agent_core::types::AfterToolCallResult {
                        content: Some(normalized),
                        details: hook_result
                            .as_ref()
                            .and_then(|result| result.details.clone()),
                        is_error: hook_result
                            .as_ref()
                            .and_then(|result| result.is_error)
                            .or(Some(context.is_error)),
                        usage: hook_result.and_then(|result| result.usage),
                        terminate: None,
                    })
                })
            })));
    }

    pub(crate) fn install_agent_next_turn_refresh(&self) {
        let previous_with_context = self.agent.prepare_next_turn_with_context();
        let previous_signal_only = self.agent.prepare_next_turn();
        let weak = self.self_weak();
        self.agent
            .set_prepare_next_turn_with_context(Some(Arc::new(move |turn, token| {
                let weak = weak.clone();
                let previous_with_context = previous_with_context.clone();
                let previous_signal_only = previous_signal_only.clone();
                Box::pin(async move {
                    let session = weak.upgrade()?;
                    let context = session
                        .compact_before_next_assistant_response(turn.context.clone())
                        .await;
                    let previous = if let Some(hook) = &previous_with_context {
                        hook(
                            PrepareNextTurnContext {
                                context: context.clone(),
                                ..turn.clone()
                            },
                            token.clone(),
                        )
                        .await
                    } else if let Some(hook) = &previous_signal_only {
                        hook(token.clone()).await
                    } else {
                        None
                    };
                    let next = previous
                        .as_ref()
                        .and_then(|snapshot| snapshot.context.clone())
                        .unwrap_or(context);
                    let (system_prompt, tools, model, thinking_level) = {
                        let state = session.agent.state();
                        (
                            lock(&session.system_prompt_override)
                                .clone()
                                .unwrap_or_else(|| lock(&session.base_system_prompt).clone()),
                            state.tools.clone(),
                            state.model.clone(),
                            state.thinking_level,
                        )
                    };
                    Some(AgentLoopTurnUpdate {
                        context: Some(AgentContext {
                            system_prompt,
                            messages: next.messages,
                            tools: Some(tools),
                        }),
                        model: Some(model),
                        thinking_level: Some(thinking_level),
                    })
                })
            })));
    }

    async fn compact_before_next_assistant_response(&self, context: AgentContext) -> AgentContext {
        let Some(model) = self.model() else {
            return context;
        };
        let settings =
            { lock(&self.settings_manager).get_compaction_settings(Some(&model_key(&model))) };
        let Ok(settings) = settings else {
            return context;
        };
        if model.context_window == 0
            || !crate::compaction::should_compact(
                crate::compaction::estimate_context_tokens(&context.messages).tokens,
                model.context_window,
                &settings,
            )
        {
            return context;
        }
        self.run_auto_compaction(
            crate::extensions::reasons::CompactionReason::Threshold,
            false,
        )
        .await;
        AgentContext {
            messages: self.agent.state().messages.clone(),
            ..context
        }
    }
}
