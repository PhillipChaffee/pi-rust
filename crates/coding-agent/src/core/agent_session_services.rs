//! The cwd-bound runtime services, upstream's
//! `src/core/agent-session-services.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`create_agent_session_services`] builds the model runtime, settings, and
//! resource loader for one effective session cwd, flushes the extensions'
//! queued provider registrations into the runtime, and reports every
//! non-fatal issue as a diagnostic. [`create_agent_session_from_services`]
//! then creates the session over those services.
//!
//! Restatements: the native-provider queue is deferred (its `Provider` type
//! lands with the composer's native-provider ticket), so only the
//! configuration registrations flush here.

use std::sync::Arc;
use std::sync::Mutex;

use tokio_util::sync::CancellationToken;

use crate::config::get_agent_dir;
use crate::config::home_dir;
use crate::config::process_cwd;
use crate::extensions::runtime::ExtensionFlag;
use crate::extensions::runtime::FlagType;
use crate::extensions::runtime::FlagValue;
use crate::model_runtime::CreateModelRuntimeOptions;
use crate::model_runtime::ModelRuntime;
use crate::resource_loader::DefaultResourceLoader;
use crate::resource_loader::DefaultResourceLoaderOptions;
use crate::resource_loader::ResourceLoaderHandle;
use crate::resource_loader::ResourceLoaderReloadOptions;
use crate::resource_loader::shared_resource_loader;
use crate::session_manager::SessionManager;
use crate::settings_manager::SettingsManager;
use crate::settings_manager::SettingsManagerCreateOptions;
use crate::settings_manager::SettingsStorageKind;
use crate::utils::paths::resolve_path;

use super::agent_session::SessionStartSeed;
use super::sdk::CreateAgentSessionError;
use super::sdk::CreateAgentSessionOptions;
use super::sdk::CreateAgentSessionResult;
use super::sdk::NoTools;
use super::sdk::create_agent_session;

/// One non-fatal issue collected while creating services or sessions,
/// upstream's `AgentSessionRuntimeDiagnostic`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentSessionRuntimeDiagnostic {
    /// The severity.
    pub kind: AgentSessionRuntimeDiagnosticKind,
    /// The message.
    pub message: String,
}

/// The diagnostic severity, upstream's `"info" | "warning" | "error"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentSessionRuntimeDiagnosticKind {
    /// Informational.
    Info,
    /// A warning.
    Warning,
    /// An error the app layer decides to abort on.
    Error,
}

/// The service-creation inputs, upstream's
/// `CreateAgentSessionServicesOptions`.
pub struct CreateAgentSessionServicesOptions<S: SettingsStorageKind> {
    /// The effective cwd.
    pub cwd: String,
    /// The global config directory. Default: `~/.pi/agent`.
    pub agent_dir: Option<String>,
    /// The settings manager. Default: one over the effective dirs.
    pub settings_manager: Option<SettingsManager<S>>,
    /// The model/auth runtime. Default: one over the effective agent dir.
    pub model_runtime: Option<ModelRuntime>,
    /// Cancellation for the runtime's initial refresh.
    pub model_runtime_signal: Option<CancellationToken>,
    /// CLI-provided extension flag values, applied after the load.
    pub extension_flag_values: Option<std::collections::BTreeMap<String, FlagValue>>,
    /// Loader options; `cwd` and `agent_dir` are overwritten with the
    /// effective values.
    pub resource_loader_options: Option<DefaultResourceLoaderOptions>,
    /// The reload options, upstream's `resourceLoaderReloadOptions`.
    pub resource_loader_reload_options: Option<ResourceLoaderReloadOptions>,
}

impl<S: SettingsStorageKind> std::fmt::Debug for CreateAgentSessionServicesOptions<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateAgentSessionServicesOptions")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .finish_non_exhaustive()
    }
}

/// The session-creation inputs over already-built services, upstream's
/// `CreateAgentSessionFromServicesOptions`.
pub struct CreateAgentSessionFromServicesOptions<S: SettingsStorageKind> {
    /// The cwd-bound services.
    pub services: AgentSessionServices<S>,
    /// The session manager.
    pub session_manager: SessionManager,
    /// Session-start metadata.
    pub session_start_event: Option<SessionStartSeed>,
    /// The model. Default: from settings.
    pub model: Option<pi_ai::types::Model>,
    /// The thinking level. Default: from settings.
    pub thinking_level: Option<pi_agent_core::types::ThinkingLevel>,
    /// Models available for cycling.
    pub scoped_models: Option<Vec<crate::model_resolver::ScopedModel>>,
    /// The tool allowlist.
    pub tools: Option<Vec<String>>,
    /// The tool denylist.
    pub exclude_tools: Option<Vec<String>>,
    /// The default tool suppression mode.
    pub no_tools: Option<NoTools>,
    /// Custom tools.
    pub custom_tools: Option<Vec<crate::extensions::types::ToolDefinition>>,
}

impl<S: SettingsStorageKind> std::fmt::Debug for CreateAgentSessionFromServicesOptions<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateAgentSessionFromServicesOptions")
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .finish_non_exhaustive()
    }
}

/// The cwd-bound runtime services, upstream's `AgentSessionServices`.
pub struct AgentSessionServices<S: SettingsStorageKind> {
    /// The effective cwd.
    pub cwd: String,
    /// The effective agent dir.
    pub agent_dir: String,
    /// The model/auth runtime.
    pub model_runtime: ModelRuntime,
    /// The settings manager the loader shares.
    pub settings_manager: Arc<Mutex<SettingsManager<S>>>,
    /// The resource loader, shared with the session built over it.
    pub resource_loader: ResourceLoaderHandle<S>,
    /// The collected diagnostics.
    pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

impl<S: SettingsStorageKind> Clone for AgentSessionServices<S> {
    fn clone(&self) -> Self {
        Self {
            cwd: self.cwd.clone(),
            agent_dir: self.agent_dir.clone(),
            model_runtime: self.model_runtime.clone(),
            settings_manager: Arc::clone(&self.settings_manager),
            resource_loader: Arc::clone(&self.resource_loader),
            diagnostics: self.diagnostics.clone(),
        }
    }
}

impl<S: SettingsStorageKind> std::fmt::Debug for AgentSessionServices<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSessionServices")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .field("diagnostics", &self.diagnostics)
            .finish_non_exhaustive()
    }
}

/// Apply the CLI-provided extension flag values, upstream's
/// `applyExtensionFlagValues`.
///
/// A boolean flag takes `true`; a string flag needs its value and rejects a
/// boolean; unknown names report once as an unknown-option error.
fn apply_extension_flag_values<S: SettingsStorageKind>(
    resource_loader: &ResourceLoaderHandle<S>,
    extension_flag_values: Option<&std::collections::BTreeMap<String, FlagValue>>,
) -> Vec<AgentSessionRuntimeDiagnostic> {
    let Some(values) = extension_flag_values else {
        return Vec::new();
    };
    let extensions_result = crate::core::agent_session::lock(resource_loader)
        .as_ref()
        .map(DefaultResourceLoader::get_extensions)
        .unwrap_or_default();
    let registered: Vec<ExtensionFlag> = extensions_result
        .extensions
        .iter()
        .filter_map(|extension| extension.record.clone())
        .flat_map(|extension| extension.flags())
        .collect();

    let mut diagnostics = Vec::new();
    let mut unknown_flags = Vec::new();
    for (name, value) in values {
        let Some(flag) = registered.iter().find(|flag| flag.name == *name) else {
            unknown_flags.push(name.clone());
            continue;
        };
        if flag.flag_type == FlagType::Boolean {
            extensions_result
                .runtime
                .set_flag_value(name, FlagValue::Bool(true));
            continue;
        }
        match value {
            FlagValue::Text(text) => {
                extensions_result
                    .runtime
                    .set_flag_value(name, FlagValue::Text(text.clone()));
            }
            FlagValue::Bool(_) => diagnostics.push(AgentSessionRuntimeDiagnostic {
                kind: AgentSessionRuntimeDiagnosticKind::Error,
                message: format!("Extension flag \"--{name}\" requires a value"),
            }),
        }
    }

    if !unknown_flags.is_empty() {
        let names = unknown_flags
            .iter()
            .map(|name| format!("--{name}"))
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = if unknown_flags.len() == 1 { "" } else { "s" };
        diagnostics.push(AgentSessionRuntimeDiagnostic {
            kind: AgentSessionRuntimeDiagnosticKind::Error,
            message: format!("Unknown option{suffix}: {names}"),
        });
    }
    diagnostics
}

/// Create the cwd-bound runtime services, upstream's
/// `createAgentSessionServices`.
///
/// # Errors
/// The model runtime's credential-store open or the loader's construction
/// failure.
pub async fn create_agent_session_services<S: SettingsStorageKind>(
    options: CreateAgentSessionServicesOptions<S>,
) -> Result<AgentSessionServices<S>, String> {
    let cwd = resolve_path(&options.cwd, &process_cwd(), &home_dir());
    let agent_dir = options.agent_dir.as_ref().map_or_else(
        || get_agent_dir().display().to_string(),
        |agent_dir| resolve_path(agent_dir, &process_cwd(), &home_dir()),
    );
    let model_runtime = match options.model_runtime {
        Some(runtime) => runtime,
        None => ModelRuntime::create(CreateModelRuntimeOptions {
            auth_path: Some(format!("{agent_dir}/auth.json")),
            models_path: Some(Some(format!("{agent_dir}/models.json"))),
            signal: options.model_runtime_signal,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .map_err(|error| error.to_string())?,
    };
    let settings_manager = options.settings_manager.unwrap_or_else(|| {
        S::default_manager(&cwd, &agent_dir, SettingsManagerCreateOptions::default())
    });
    let mut loader_options =
        options
            .resource_loader_options
            .unwrap_or_else(|| DefaultResourceLoaderOptions {
                cwd: cwd.clone(),
                agent_dir: agent_dir.clone(),
                ..DefaultResourceLoaderOptions::default()
            });
    loader_options.cwd.clone_from(&cwd);
    loader_options.agent_dir.clone_from(&agent_dir);
    let mut resource_loader = DefaultResourceLoader::new(loader_options, settings_manager);
    resource_loader
        .reload(options.resource_loader_reload_options)
        .await;
    let settings_manager = resource_loader.settings_manager();
    let resource_loader = shared_resource_loader(resource_loader);

    let mut diagnostics = Vec::new();
    let extensions_result = crate::core::agent_session::lock(&resource_loader)
        .as_ref()
        .map(DefaultResourceLoader::get_extensions)
        .unwrap_or_default();
    for registration in extensions_result
        .runtime
        .take_pending_provider_registrations()
    {
        if let Err(message) =
            model_runtime.register_provider(&registration.name, registration.config.clone())
        {
            diagnostics.push(AgentSessionRuntimeDiagnostic {
                kind: AgentSessionRuntimeDiagnosticKind::Error,
                message: format!(
                    "Extension \"{}\" error: {message}",
                    registration.extension_path
                ),
            });
        }
    }
    let _ = model_runtime
        .refresh(pi_ai::models::ModelsRefreshOptions {
            allow_network: Some(false),
            ..pi_ai::models::ModelsRefreshOptions::default()
        })
        .await;
    diagnostics.extend(apply_extension_flag_values(
        &resource_loader,
        options.extension_flag_values.as_ref(),
    ));

    Ok(AgentSessionServices {
        cwd,
        agent_dir,
        model_runtime,
        settings_manager,
        resource_loader,
        diagnostics,
    })
}

/// Create a session from already-created services, upstream's
/// `createAgentSessionFromServices`.
///
/// # Errors
/// The session factory's failure.
pub async fn create_agent_session_from_services<S: SettingsStorageKind>(
    options: CreateAgentSessionFromServicesOptions<S>,
) -> Result<CreateAgentSessionResult<S>, CreateAgentSessionError> {
    let AgentSessionServices {
        cwd,
        agent_dir,
        model_runtime,
        resource_loader,
        ..
    } = options.services;
    create_agent_session(CreateAgentSessionOptions {
        cwd: Some(cwd),
        agent_dir: Some(agent_dir),
        model_runtime: Some(model_runtime),
        model: options.model,
        thinking_level: options.thinking_level,
        scoped_models: options.scoped_models,
        no_tools: options.no_tools,
        tools: options.tools,
        exclude_tools: options.exclude_tools,
        custom_tools: options.custom_tools,
        resource_loader: Some(resource_loader),
        session_manager: Some(options.session_manager),
        settings_manager: None,
        session_start_event: options.session_start_event,
    })
    .await
}
