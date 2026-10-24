//! The session factory, upstream's `src/core/sdk.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`create_agent_session`] assembles the product session: the model/thinking
//! restore ladder, the block-images `convertToLlm` filter, the stream wiring
//! that adds provider-retry settings, attribution headers, and the extension
//! request hooks, and the initial persisted model/thinking entries.
//!
//! Restatements: upstream shares one settings manager between the resource
//! loader and the session; the port builds the loader first and reads its
//! manager, which is the same object. `setDefaultStreamFn(streamSimple)` runs
//! at module load upstream; with no module initializer the port installs it
//! on the first factory call ([`install_default_stream_fn`]).

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::AgentInitialState;
use pi_agent_core::AgentOptions;
use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::Context;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::ProviderHeaders;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::ToolResultBlock;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::Value;

use super::agent_session::AgentSession;
use super::agent_session::AgentSessionConfig;
use super::agent_session::SessionStartSeed;
use crate::auth_guidance::format_no_models_available_message;
use crate::config::default_env_lookup;
use crate::config::get_agent_dir;
use crate::config::home_dir;
use crate::config::process_cwd;
use crate::defaults::DEFAULT_THINKING_LEVEL;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::types::ToolDefinition;
use crate::messages::convert_to_llm;
use crate::model_resolver::FindInitialModelOptions;
use crate::model_resolver::InitialModelError;
use crate::model_resolver::ScopedModel;
use crate::model_resolver::find_initial_model;
use crate::model_runtime::CreateModelRuntimeOptions;
use crate::model_runtime::ModelRuntime;
use crate::provider_attribution::merge_provider_attribution_headers;
use crate::resource_loader::DefaultResourceLoader;
use crate::resource_loader::DefaultResourceLoaderOptions;
use crate::resource_loader::ExtensionLoadResult;
use crate::resource_loader::ResourceLoaderHandle;
use crate::resource_loader::shared_resource_loader;
use crate::session_manager::FileEntry;
use crate::session_manager::SessionEntry;
use crate::session_manager::SessionManager;
use crate::session_manager::SessionManagerError;
use crate::session_manager::get_default_session_dir_with_agent_dir;
use crate::settings_manager::SettingsManager;
use crate::settings_manager::SettingsManagerCreateOptions;
use crate::settings_manager::SettingsStorage;
use crate::settings_manager::SettingsStorageKind;
use crate::utils::paths::resolve_path;

/// The default built-in tool selection, upstream's `defaultActiveToolNames`.
const DEFAULT_ACTIVE_TOOL_NAMES: [&str; 4] = ["read", "bash", "edit", "write"];

/// The `noTools` suppression mode, upstream's `"all" | "builtin"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoTools {
    /// Start with no tools enabled.
    All,
    /// Disable the default built-in tools but keep extension/custom tools.
    Builtin,
}

/// The factory options, upstream's `CreateAgentSessionOptions`.
pub struct CreateAgentSessionOptions<S: SettingsStorageKind> {
    /// Working directory for project-local discovery. Default: the process
    /// cwd, or the supplied session manager's cwd.
    pub cwd: Option<String>,
    /// Global config directory. Default: `~/.pi/agent`.
    pub agent_dir: Option<String>,
    /// Canonical model/auth runtime. Default: one over the effective agent
    /// dir's `auth.json` and `models.json`.
    pub model_runtime: Option<ModelRuntime>,
    /// The model to use. Default: from settings, else the first available.
    pub model: Option<Model>,
    /// The thinking level. Default: restored from the session, then the
    /// per-model override, then the global default, clamped to the model.
    pub thinking_level: Option<ThinkingLevel>,
    /// Models available for cycling.
    pub scoped_models: Option<Vec<ScopedModel>>,
    /// The default tool suppression mode when no allowlist is given.
    pub no_tools: Option<NoTools>,
    /// The tool-name allowlist.
    pub tools: Option<Vec<String>>,
    /// The tool-name denylist, applied after `tools`.
    pub exclude_tools: Option<Vec<String>>,
    /// Custom tools registered alongside the built-ins.
    pub custom_tools: Option<Vec<ToolDefinition>>,
    /// The resource loader. Default: a reloaded `DefaultResourceLoader`.
    pub resource_loader: Option<ResourceLoaderHandle<S>>,
    /// The session manager. Default: `SessionManager.create(cwd)`.
    pub session_manager: Option<SessionManager>,
    /// The settings manager. Default: `SettingsManager.create(cwd, agentDir)`.
    pub settings_manager: Option<SettingsManager<S>>,
    /// Session-start metadata for extension runtime startup.
    pub session_start_event: Option<SessionStartSeed>,
}

impl<S: SettingsStorageKind> std::fmt::Debug for CreateAgentSessionOptions<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateAgentSessionOptions")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .finish_non_exhaustive()
    }
}

/// The factory result, upstream's `CreateAgentSessionResult`.
pub struct CreateAgentSessionResult<S: SettingsStorageKind> {
    /// The created session.
    pub session: Arc<AgentSession<S>>,
    /// The loaded extensions result.
    pub extensions_result: ExtensionLoadResult,
    /// The warning when the session restored a different model than saved.
    pub model_fallback_message: Option<String>,
}

impl<S: SettingsStorageKind> std::fmt::Debug for CreateAgentSessionResult<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateAgentSessionResult")
            .field("model_fallback_message", &self.model_fallback_message)
            .finish_non_exhaustive()
    }
}

/// The factory failure, upstream's thrown errors on the creation path.
#[derive(Debug)]
pub enum CreateAgentSessionError {
    /// The session manager failed to open or create.
    SessionManager(SessionManagerError),
    /// The model runtime failed to open its credential store.
    ModelRuntime(String),
    /// The initial-model resolution failed, upstream's print-and-exit path.
    InitialModel(InitialModelError),
}

impl std::fmt::Display for CreateAgentSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionManager(error) => write!(f, "{error}"),
            Self::ModelRuntime(error) => f.write_str(error),
            Self::InitialModel(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CreateAgentSessionError {}

impl From<SessionManagerError> for CreateAgentSessionError {
    fn from(error: SessionManagerError) -> Self {
        Self::SessionManager(error)
    }
}

impl From<InitialModelError> for CreateAgentSessionError {
    fn from(error: InitialModelError) -> Self {
        Self::InitialModel(error)
    }
}

/// Install pi-ai's simple stream as the process-global default, upstream's
/// module-level `setDefaultStreamFn(streamSimple)`.
///
/// Idempotent: repeated calls replace the same default.
pub fn install_default_stream_fn() {
    pi_agent_core::set_default_stream_fn(Some(Arc::new(|model, context, options| {
        pi_ai::compat::stream_simple(model, context, options)
    })));
}

/// The `convertToLlm` wrapper that filters images when `blockImages` is on,
/// upstream's `convertToLlmWithBlockImages`.
///
/// The setting is read per conversion so mid-session changes take effect.
#[must_use]
pub fn convert_to_llm_with_block_images<S: SettingsStorage>(
    messages: &[pi_agent_core::types::AgentMessage],
    settings_manager: &SettingsManager<S>,
) -> Vec<Message> {
    let converted = convert_to_llm(messages);
    if !settings_manager.get_block_images() {
        return converted;
    }
    converted
        .into_iter()
        .map(|message| match message {
            Message::User(mut message) => {
                message.content = block_images_in_user_content(message.content);
                Message::User(message)
            }
            Message::ToolResult(mut message) => {
                message.content = block_images_in_tool_content(message.content);
                Message::ToolResult(message)
            }
            other @ Message::Assistant(_) => other,
        })
        .collect()
}

/// The user content with images replaced by the disabled placeholder,
/// consecutive placeholders deduped.
fn block_images_in_user_content(content: UserContent) -> UserContent {
    let UserContent::Blocks(blocks) = content else {
        return content;
    };
    let filtered: Vec<UserBlock> = blocks
        .into_iter()
        .map(|block| match block {
            UserBlock::Image(_) => UserBlock::Text(disabled_image_text()),
            other @ UserBlock::Text(_) => other,
        })
        .collect();
    UserContent::Blocks(dedupe_user_blocks(filtered))
}

/// The tool-result content with images replaced by the disabled placeholder,
/// consecutive placeholders deduped.
fn block_images_in_tool_content(content: Vec<ToolResultBlock>) -> Vec<ToolResultBlock> {
    let filtered: Vec<ToolResultBlock> = content
        .into_iter()
        .map(|block| match block {
            ToolResultBlock::Image(_) => ToolResultBlock::Text(disabled_image_text()),
            other @ ToolResultBlock::Text(_) => other,
        })
        .collect();
    let mut deduped: Vec<ToolResultBlock> = Vec::with_capacity(filtered.len());
    for block in filtered {
        if is_placeholder(&block) && deduped.last().is_some_and(is_placeholder) {
            continue;
        }
        deduped.push(block);
    }
    deduped
}

/// The placeholder text upstream writes for a filtered image.
fn disabled_image_text() -> pi_ai::types::TextContent {
    pi_ai::types::TextContent {
        text: "Image reading is disabled.".to_owned(),
        text_signature: None,
    }
}

/// Whether a user block is the disabled-image placeholder.
fn is_user_placeholder(block: &UserBlock) -> bool {
    matches!(block, UserBlock::Text(text) if text.text == "Image reading is disabled.")
}

/// Whether a tool-result block is the disabled-image placeholder.
fn is_placeholder(block: &ToolResultBlock) -> bool {
    matches!(block, ToolResultBlock::Text(text) if text.text == "Image reading is disabled.")
}

/// Dedupe consecutive placeholder text blocks, upstream's filter.
fn dedupe_user_blocks(blocks: Vec<UserBlock>) -> Vec<UserBlock> {
    let mut deduped: Vec<UserBlock> = Vec::with_capacity(blocks.len());
    for block in blocks {
        if is_user_placeholder(&block) && deduped.last().is_some_and(is_user_placeholder) {
            continue;
        }
        deduped.push(block);
    }
    deduped
}

/// Create an `AgentSession` over `options`.
///
/// # Errors
/// A session manager, model runtime, or initial-model resolution failure.
///
/// # Panics
/// When the shared loader slot is empty, which only a concurrent reload can
/// cause.
#[expect(
    clippy::too_many_lines,
    clippy::expect_used,
    reason = "the 1:1 port of upstream's createAgentSession keeps its ladder inline; the loader is present outside the reload window"
)]
pub async fn create_agent_session<S: SettingsStorageKind>(
    options: CreateAgentSessionOptions<S>,
) -> Result<CreateAgentSessionResult<S>, CreateAgentSessionError> {
    install_default_stream_fn();

    let cwd_source = options
        .cwd
        .clone()
        .or_else(|| {
            options
                .session_manager
                .as_ref()
                .map(|manager| manager.cwd().to_owned())
        })
        .unwrap_or_else(process_cwd);
    let cwd = resolve_path(&cwd_source, &process_cwd(), &home_dir());
    let agent_dir = options.agent_dir.as_ref().map_or_else(
        || get_agent_dir().display().to_string(),
        |agent_dir| resolve_path(agent_dir, &process_cwd(), &home_dir()),
    );

    let auth_path = options
        .agent_dir
        .as_ref()
        .map(|_| format!("{agent_dir}/auth.json"));
    let models_path = options
        .agent_dir
        .as_ref()
        .map(|_| format!("{agent_dir}/models.json"));
    let model_runtime = match options.model_runtime {
        Some(runtime) => runtime,
        None => ModelRuntime::create(CreateModelRuntimeOptions {
            auth_path,
            models_path: models_path.map(Some),
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .map_err(|error| CreateAgentSessionError::ModelRuntime(error.to_string()))?,
    };

    let mut settings_manager = options.settings_manager;
    if settings_manager.is_none() && options.resource_loader.is_none() {
        settings_manager = Some(S::default_manager(
            &cwd,
            &agent_dir,
            SettingsManagerCreateOptions::default(),
        ));
    }
    let session_manager = if let Some(manager) = options.session_manager {
        manager
    } else {
        let session_dir = get_default_session_dir_with_agent_dir(&cwd, &agent_dir)?;
        SessionManager::create(&cwd, Some(&session_dir.display().to_string()), None)?
    };

    let resource_loader = if let Some(loader) = options.resource_loader {
        loader
    } else {
        let Some(settings_manager) = settings_manager else {
            return Err(CreateAgentSessionError::ModelRuntime(
                "no settings manager or resource loader supplied".to_owned(),
            ));
        };
        let mut loader = DefaultResourceLoader::new(
            DefaultResourceLoaderOptions {
                cwd: cwd.clone(),
                agent_dir,
                ..DefaultResourceLoaderOptions::default()
            },
            settings_manager,
        );
        loader.reload(None).await;
        shared_resource_loader(loader)
    };
    let extensions_result = crate::core::agent_session::lock(&resource_loader)
        .as_ref()
        .map(DefaultResourceLoader::get_extensions)
        .unwrap_or_default();
    let settings = crate::core::agent_session::lock(&resource_loader)
        .as_ref()
        .map(DefaultResourceLoader::settings_manager)
        .expect("the loader is present at construction");

    let existing_session = session_manager.build_session_context();
    let has_existing_session = !existing_session.messages.is_empty();
    let has_thinking_entry = session_manager.get_branch(None).into_iter().any(|entry| {
        matches!(
            entry,
            FileEntry::Entry(SessionEntry::ThinkingLevelChange(_))
        )
    });

    let mut model = options.model;
    let mut model_fallback_message: Option<String> = None;

    if model.is_none()
        && has_existing_session
        && let Some(saved) = &existing_session.model
    {
        let restored = model_runtime.get_model(&saved.provider, &saved.model_id);
        if let Some(restored) = &restored
            && model_runtime.has_configured_auth(&restored.provider.0)
        {
            model = Some(restored.clone());
        }
        if model.is_none() {
            model_fallback_message = Some(format!(
                "Could not restore model {}/{}",
                saved.provider, saved.model_id
            ));
        }
    }

    if model.is_none() {
        let per_model_levels = model_thinking_levels(&settings);
        let defaults = {
            let locked = crate::core::agent_session::lock(&settings);
            (
                locked.get_default_provider(),
                locked.get_default_model(),
                locked.get_default_thinking_level(),
            )
        };
        let result = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[],
            is_continuing: has_existing_session,
            default_provider: defaults.0.as_deref(),
            default_model_id: defaults.1.as_deref(),
            default_thinking_level: defaults.2,
            model_thinking_levels: Some(&per_model_levels),
            model_runtime: &model_runtime,
        })?;
        model = result.model;
        match (&model, &mut model_fallback_message) {
            (None, message) => *message = Some(format_no_models_available_message()),
            (Some(found), Some(message)) => {
                let _ = write!(message, ". Using {}/{}", found.provider.0, found.id);
            }
            (Some(_), None) => {}
        }
    }

    let mut thinking_level = options.thinking_level;
    if thinking_level.is_none() && has_existing_session {
        thinking_level = Some(if has_thinking_entry {
            thinking_level_from_str(&existing_session.thinking_level)
        } else {
            crate::core::agent_session::lock(&settings)
                .get_default_thinking_level()
                .unwrap_or(DEFAULT_THINKING_LEVEL)
        });
    }
    if thinking_level.is_none()
        && let Some(current) = &model
        && let Some(per_model) = crate::core::agent_session::lock(&settings)
            .get_model_thinking_level(&current.provider.0, &current.id)
    {
        thinking_level = Some(per_model);
    }
    if thinking_level.is_none() {
        thinking_level = Some(
            crate::core::agent_session::lock(&settings)
                .get_default_thinking_level()
                .unwrap_or(DEFAULT_THINKING_LEVEL),
        );
    }
    let thinking_level = match (&model, thinking_level) {
        (None, _) => ThinkingLevel::Off,
        (Some(model), Some(level)) => {
            crate::core::agent_session::clamp_thinking_level_for_model(model, level)
        }
        (Some(_), None) => DEFAULT_THINKING_LEVEL,
    };

    let configured_default_tool_names =
        crate::core::agent_session::lock(&settings).get_default_tools();
    let allowed_tool_names = options
        .tools
        .clone()
        .or_else(|| (options.no_tools == Some(NoTools::All)).then(Vec::new));
    let excluded_tool_names = options.exclude_tools.clone();
    let excluded_set: Option<std::collections::BTreeSet<String>> = excluded_tool_names
        .as_ref()
        .map(|names| names.iter().cloned().collect());
    let initial_active_tool_names = options
        .tools
        .clone()
        .unwrap_or_else(|| {
            if options.no_tools.is_some() {
                Vec::new()
            } else {
                configured_default_tool_names.unwrap_or_else(|| {
                    DEFAULT_ACTIVE_TOOL_NAMES
                        .iter()
                        .map(|name| (*name).to_owned())
                        .collect()
                })
            }
        })
        .into_iter()
        .filter(|name| !excluded_set.as_ref().is_some_and(|set| set.contains(name)))
        .collect();

    let extension_runner_ref: Arc<Mutex<Option<Arc<ExtensionRunner>>>> = Arc::new(Mutex::new(None));

    let settings_for_convert = Arc::clone(&settings);
    let convert_to_llm_fn: pi_agent_core::types::ConvertToLlm = Arc::new(move |messages| {
        let settings = Arc::clone(&settings_for_convert);
        Box::pin(async move {
            convert_to_llm_with_block_images(
                &messages,
                &crate::core::agent_session::lock(&settings),
            )
        })
    });

    let session_id = session_manager.session_id().to_owned();
    let stream_fn = build_stream_fn(
        model_runtime.clone(),
        Arc::clone(&settings),
        Arc::clone(&extension_runner_ref),
        session_id.clone(),
    );

    let transform_context = {
        let slot = Arc::clone(&extension_runner_ref);
        let transform: pi_agent_core::types::TransformContext = Arc::new(
            move |messages: Vec<pi_agent_core::types::AgentMessage>,
                  _token: Option<tokio_util::sync::CancellationToken>|
                  -> pi_ai::types::BoxedFuture<
                'static,
                Vec<pi_agent_core::types::AgentMessage>,
            > {
                let slot = Arc::clone(&slot);
                Box::pin(async move {
                    let runner = crate::core::agent_session::lock(&slot).clone();
                    let Some(runner) = runner else {
                        return messages;
                    };
                    runner.emit_context(messages).await
                })
            },
        );
        Some(transform)
    };

    let on_payload = {
        let slot = Arc::clone(&extension_runner_ref);
        Some(pi_ai::types::OnPayload::new(
            move |payload: Value, _model: Model| {
                let slot = Arc::clone(&slot);
                Box::pin(async move {
                    let runner = crate::core::agent_session::lock(&slot).clone();
                    let Some(runner) = runner else {
                        return Some(payload);
                    };
                    if !runner.has_handlers("before_provider_request") {
                        return Some(payload);
                    }
                    Some(runner.emit_before_provider_request(payload).await)
                })
            },
        ))
    };

    let on_response = {
        let slot = Arc::clone(&extension_runner_ref);
        Some(pi_ai::types::OnResponse::new(
            move |response: pi_ai::types::ProviderResponse, _model: Model| {
                let slot = Arc::clone(&slot);
                Box::pin(async move {
                    let runner = crate::core::agent_session::lock(&slot).clone();
                    let Some(runner) = runner else {
                        return;
                    };
                    if !runner.has_handlers("after_provider_response") {
                        return;
                    }
                    let headers: ProviderHeaders = response
                        .headers
                        .iter()
                        .map(|(key, value)| (key.clone(), Some(value.clone())))
                        .collect();
                    let _ = runner
                        .emit(ExtensionEvent::AfterProviderResponse {
                            status: Some(response.status),
                            headers,
                        })
                        .await;
                })
            },
        ))
    };

    let (steering_mode, follow_up_mode, transport, thinking_budgets, max_retry_delay_ms) = {
        let locked = crate::core::agent_session::lock(&settings);
        (
            locked.get_steering_mode(),
            locked.get_follow_up_mode(),
            locked.get_transport(),
            locked
                .get_thinking_budgets()
                .and_then(|budgets| serde_json::from_value(Value::Object(budgets)).ok()),
            locked.get_provider_retry_settings().max_retry_delay_ms,
        )
    };

    let initial_messages = if has_existing_session {
        existing_session.messages
    } else {
        Vec::new()
    };
    let agent = pi_agent_core::Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            system_prompt: String::new(),
            model: model.clone(),
            thinking_level: Some(thinking_level),
            tools: Vec::new(),
            messages: initial_messages,
        }),
        convert_to_llm: Some(convert_to_llm_fn),
        transform_context,
        stream_fn: Some(stream_fn),
        on_payload,
        on_response,
        steering_mode: Some(to_agent_queue_mode(steering_mode)),
        follow_up_mode: Some(to_agent_queue_mode(follow_up_mode)),
        session_id: Some(session_id),
        transport: Some(transport),
        thinking_budgets,
        max_retry_delay_ms: non_negative_u64(max_retry_delay_ms),
        ..AgentOptions::default()
    });

    let mut session_manager = session_manager;
    if has_existing_session {
        if !has_thinking_entry {
            session_manager.append_thinking_level_change(&thinking_level_str(thinking_level))?;
        }
    } else {
        if let Some(model) = &model {
            session_manager.append_model_change(&model.provider.0, &model.id)?;
        }
        session_manager.append_thinking_level_change(&thinking_level_str(thinking_level))?;
    }

    let session = AgentSession::new(AgentSessionConfig {
        agent,
        session_manager,
        resource_loader,
        cwd,
        scoped_models: options.scoped_models,
        model_runtime,
        custom_tools: options.custom_tools,
        initial_active_tool_names: Some(initial_active_tool_names),
        allowed_tool_names,
        excluded_tool_names,
        base_tools_override: None,
        extension_runner_ref: Some(extension_runner_ref),
        session_start_event: options.session_start_event,
    });

    Ok(CreateAgentSessionResult {
        session,
        extensions_result,
        model_fallback_message,
    })
}

/// Build the stream function, upstream's `streamFn` option body: settings
/// defaults layered under the request's own values, then the assembled-header
/// attribution/extension transform.
fn build_stream_fn<S: SettingsStorageKind>(
    model_runtime: ModelRuntime,
    settings: Arc<Mutex<SettingsManager<S>>>,
    extension_runner_ref: Arc<Mutex<Option<Arc<ExtensionRunner>>>>,
    session_id: String,
) -> pi_agent_core::types::StreamFn {
    Arc::new(
        move |model: &Model,
              context: &Context,
              options: Option<&SimpleStreamOptions>|
              -> AssistantMessageEventStream {
            let http_idle_timeout_ms =
                crate::core::agent_session::lock(&settings).get_http_idle_timeout_ms();
            let http_idle_timeout_ms = match http_idle_timeout_ms {
                Ok(value) => value,
                Err(message) => {
                    return pi_ai::api::wire_common::setup_error_stream(model, &message);
                }
            };
            let websocket_connect_timeout_ms =
                crate::core::agent_session::lock(&settings).get_websocket_connect_timeout_ms();
            let websocket_connect_timeout_ms = match websocket_connect_timeout_ms {
                Ok(value) => value,
                Err(message) => {
                    return pi_ai::api::wire_common::setup_error_stream(model, &message);
                }
            };
            let provider_retry =
                crate::core::agent_session::lock(&settings).get_provider_retry_settings();
            // SDKs treat timeout=0 as an immediate timeout, not "no timeout".
            let effective_timeout_ms = if http_idle_timeout_ms == 0 {
                2_147_483_647
            } else {
                http_idle_timeout_ms
            };

            let mut owned = options.cloned().unwrap_or_default();
            owned.timeout_ms = Some(
                options
                    .and_then(|options| options.timeout_ms)
                    .or_else(|| non_negative_u64(provider_retry.timeout_ms.unwrap_or(-1)))
                    .unwrap_or_else(|| {
                        #[expect(
                            clippy::cast_sign_loss,
                            reason = "effective_timeout_ms is non-negative by construction"
                        )]
                        let timeout = effective_timeout_ms.max(0) as u64;
                        timeout
                    }),
            );
            owned.websocket_connect_timeout_ms = options
                .and_then(|options| options.websocket_connect_timeout_ms)
                .or_else(|| websocket_connect_timeout_ms.and_then(non_negative_u64))
                .or(owned.websocket_connect_timeout_ms);
            owned.max_retries = options
                .and_then(|options| options.max_retries)
                .or_else(|| provider_retry.max_retries.and_then(non_negative_u32))
                .or(owned.max_retries);
            let retry_delay = non_negative_u64(provider_retry.max_retry_delay_ms);
            owned.max_retry_delay_ms = options
                .and_then(|options| options.max_retry_delay_ms)
                .or(retry_delay)
                .or(owned.max_retry_delay_ms);

            let model_for_headers = model.clone();
            let settings_for_headers = Arc::clone(&settings);
            let slot = Arc::clone(&extension_runner_ref);
            let session_id_for_headers = session_id.clone();
            let transform_headers: pi_ai::models::TransformHeadersFn =
                Arc::new(move |headers: ProviderHeaders| {
                    let model = model_for_headers.clone();
                    let settings = Arc::clone(&settings_for_headers);
                    let slot = Arc::clone(&slot);
                    let session_id = session_id_for_headers.clone();
                    Box::pin(async move {
                        let merged = merge_provider_attribution_headers(
                            &model,
                            &crate::core::agent_session::lock(&settings),
                            &default_env_lookup(),
                            Some(&session_id),
                            &[&headers],
                        );
                        let merged = merged.unwrap_or_default();
                        let runner = crate::core::agent_session::lock(&slot).clone();
                        match runner {
                            Some(runner) if runner.has_handlers("before_provider_headers") => {
                                runner.emit_before_provider_headers(merged).await
                            }
                            _ => merged,
                        }
                    })
                });

            model_runtime.stream_simple(
                model,
                context,
                Some(&pi_ai::models::WithTransforms {
                    options: owned,
                    transform_headers: Some(transform_headers),
                }),
            )
        },
    )
}

/// Convert a non-negative setting to `u64`.
fn non_negative_u64(value: i64) -> Option<u64> {
    (value >= 0).then(|| {
        #[expect(
            clippy::cast_sign_loss,
            reason = "the guard ensures the value is non-negative"
        )]
        let value = value as u64;
        value
    })
}

/// Convert a non-negative setting to `u32`.
fn non_negative_u32(value: i64) -> Option<u32> {
    (value >= 0).then(|| {
        #[expect(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "retry counts are small non-negative integers"
        )]
        let value = value as u32;
        value
    })
}

/// The per-model thinking-level overrides as a typed map.
fn model_thinking_levels<S: SettingsStorage>(
    settings: &Arc<Mutex<SettingsManager<S>>>,
) -> std::collections::BTreeMap<String, ThinkingLevel> {
    crate::core::agent_session::lock(settings)
        .get_all_model_thinking_levels()
        .into_iter()
        .filter_map(|(model, level)| {
            serde_json::from_value::<ThinkingLevel>(level)
                .ok()
                .map(|level| (model, level))
        })
        .collect()
}

/// Parse a stored thinking-level string, defaulting to off.
fn thinking_level_from_str(value: &str) -> ThinkingLevel {
    serde_json::from_value(Value::String(value.to_owned())).unwrap_or(ThinkingLevel::Off)
}

/// Render a thinking level as its wire string.
fn thinking_level_str(level: ThinkingLevel) -> String {
    serde_json::to_value(level)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "off".to_owned())
}

/// Map the settings queue mode onto the agent's.
const fn to_agent_queue_mode(
    mode: crate::settings_manager::QueueMode,
) -> pi_agent_core::types::QueueMode {
    match mode {
        crate::settings_manager::QueueMode::All => pi_agent_core::types::QueueMode::All,
        crate::settings_manager::QueueMode::OneAtATime => {
            pi_agent_core::types::QueueMode::OneAtATime
        }
    }
}
