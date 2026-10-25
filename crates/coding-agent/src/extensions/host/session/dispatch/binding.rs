//! The registration pass's binding: the child's declarations become the
//! in-process [`Extension`] record, with every handler, tool, command, and
//! provider proxying back over the wire ([ADR 0010]).
//!
//! [ADR 0010]: ../../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::sync::Arc;

use crate::extensions::api::PendingProviderRegistration;
use crate::extensions::runtime::{Extension, ExtensionFlag, RegisteredTool, ResolvedCommand};
use crate::extensions::types::ToolDefinition;

use super::super::super::payload::descriptors::{
    CommandDeclaration, FlagDeclaration, ToolDeclaration,
};
use super::super::super::session::HostConnectOptions;
use super::super::connection::{Registrations, SessionCore};
use crate::extensions::runtime::HandlerContext;

/// Build the in-process record over the child's declarations.
pub(in crate::extensions::host) fn build_extension<T>(
    core: &Arc<SessionCore<T>>,
    options: &HostConnectOptions,
    registrations: &Registrations,
) -> Arc<Extension>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let extension = Arc::new(Extension::new(
        options.path.clone(),
        options.resolved_path.clone(),
        false,
        options.source_info.clone(),
    ));
    for channel in &registrations.events {
        let core = Arc::clone(core);
        let channel = channel.clone();
        extension.on(
            &channel,
            Arc::new(
                move |event: crate::extensions::events::ExtensionEvent, context: HandlerContext| {
                    let core = Arc::clone(&core);
                    Box::pin(async move { core.call_event(event, context).await })
                },
            ),
        );
    }
    for tool in &registrations.tools {
        extension.register_tool(RegisteredTool {
            definition: tool_definition(core, tool),
            source_info: options.source_info.clone(),
        });
    }
    for command in &registrations.commands {
        extension.register_command(command_record(core, options, command));
    }
    for flag in &registrations.flags {
        extension.register_flag(flag_record(options, flag));
    }
    extension
}

/// Build the queued provider registrations, attaching the stream proxy.
pub(in crate::extensions::host) fn build_providers<T>(
    core: &Arc<SessionCore<T>>,
    options: &HostConnectOptions,
    registrations: &Registrations,
) -> Vec<PendingProviderRegistration>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    registrations
        .providers
        .iter()
        .cloned()
        .zip(registrations.provider_names.iter().cloned())
        .map(|(mut config, name)| {
            let core = Arc::clone(core);
            let stream_name = name.clone();
            config.stream_simple = Some(Arc::new(move |model, context, options| {
                core.stream_provider(&stream_name, model, context, options)
            }));
            PendingProviderRegistration {
                name,
                config,
                extension_path: options.path.clone(),
            }
        })
        .collect()
}

fn tool_definition<T>(core: &Arc<SessionCore<T>>, tool: &ToolDeclaration) -> ToolDefinition
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let core = Arc::clone(core);
    let name = tool.name.clone();
    let execute_core = Arc::clone(&core);
    ToolDefinition {
        name: tool.name.clone(),
        label: tool.label.clone(),
        description: tool.description.clone(),
        prompt_snippet: tool.prompt_snippet.clone(),
        prompt_guidelines: tool.prompt_guidelines.clone(),
        parameters: tool.parameters.clone(),
        constrained_sampling: tool.constrained_sampling.clone(),
        render_shell: tool.render_shell,
        prepare_arguments: None,
        execution_mode: tool.execution_mode,
        execute: Arc::new(move |tool_call_id, params, signal, update, context| {
            let core = Arc::clone(&execute_core);
            let name = name.clone();
            let tool_call_id = tool_call_id.to_string();
            let params = params.clone();
            let signal = signal.cloned();
            let snapshot = core.snapshot(context);
            Box::pin(async move {
                core.call_tool(
                    &name,
                    &tool_call_id,
                    params,
                    snapshot,
                    signal.as_ref(),
                    update,
                )
                .await
            })
        }),
    }
}

fn command_record<T>(
    core: &Arc<SessionCore<T>>,
    options: &HostConnectOptions,
    command: &CommandDeclaration,
) -> ResolvedCommand
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let core = Arc::clone(core);
    let name = command.name.clone();
    let handler_name = command.name.clone();
    ResolvedCommand {
        name,
        invocation_name: command.name.clone(),
        description: command.description.clone(),
        source_info: options.source_info.clone(),
        handler: Arc::new(move |args: String, _context: HandlerContext| {
            let core = Arc::clone(&core);
            let name = handler_name.clone();
            Box::pin(async move { core.call_command(&name, &args).await })
        }),
    }
}

fn flag_record(options: &HostConnectOptions, flag: &FlagDeclaration) -> ExtensionFlag {
    ExtensionFlag {
        name: flag.name.clone(),
        description: flag.description.clone(),
        flag_type: flag.flag_type,
        default: flag.default.clone(),
        extension_path: options.path.clone(),
    }
}
