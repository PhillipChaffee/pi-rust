//! The tool registry and system-prompt rebuild.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use super::ToolDefinitionEntry;
use super::lock;
use crate::extensions::runner::ToolInfo;
use crate::extensions::types::ToolDefinition;
use crate::settings_manager::SettingsStorage;
use crate::source_info::SourceInfo;
use crate::source_info::SyntheticSourceOptions;
use crate::source_info::create_synthetic_source_info;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Set the active tools by name and rebuild the system prompt.
    pub fn set_active_tools_by_name(&self, tool_names: &[String]) {
        let registry = lock(&self.tool_registry);
        let mut tools = Vec::new();
        let mut valid = Vec::new();
        for name in tool_names {
            if let Some(tool) = registry.get(name) {
                tools.push(tool.clone());
                valid.push(name.clone());
            }
        }
        drop(registry);
        self.agent.set_tools(&tools);
        let prompt = self.rebuild_system_prompt(&valid);
        (*lock(&self.base_system_prompt)).clone_from(&prompt);
        let override_prompt = lock(&self.system_prompt_override).clone();
        self.agent
            .set_system_prompt(override_prompt.unwrap_or(prompt));
    }

    /// Every configured tool with its provenance.
    #[must_use]
    pub fn get_all_tools(&self) -> Vec<ToolInfo> {
        lock(&self.tool_definitions)
            .values()
            .map(|entry| ToolInfo {
                name: entry.definition.name.clone(),
                description: entry.definition.description.clone(),
                parameters: entry.definition.parameters.clone(),
                prompt_guidelines: entry.definition.prompt_guidelines.clone(),
                source_info: entry.source_info.clone(),
            })
            .collect()
    }

    /// One tool definition by name.
    #[must_use]
    pub fn get_tool_definition(&self, name: &str) -> Option<ToolDefinition> {
        lock(&self.tool_definitions)
            .get(name)
            .map(|entry| entry.definition.clone())
    }

    pub(crate) fn refresh_tool_registry(
        &self,
        active_tool_names: Option<Vec<String>>,
        include_all_extension_tools: bool,
    ) {
        let previous_registry_names: BTreeSet<String> =
            lock(&self.tool_registry).keys().cloned().collect();
        let previous_active = self.active_tool_names();
        let is_allowed = |name: &str| {
            self.allowed_tool_names
                .as_ref()
                .is_none_or(|allowed| allowed.contains(name))
                && !self
                    .excluded_tool_names
                    .as_ref()
                    .is_some_and(|excluded| excluded.contains(name))
        };

        let runner = self.extension_runner();
        let mut custom: Vec<ToolDefinitionEntry> = runner
            .get_all_registered_tools()
            .into_iter()
            .map(|tool| ToolDefinitionEntry {
                definition: tool.definition,
                source_info: tool.source_info,
            })
            .collect();
        custom.extend(
            self.custom_tools
                .iter()
                .map(|definition| ToolDefinitionEntry {
                    definition: definition.clone(),
                    source_info: synthetic_source(&definition.name, "sdk"),
                }),
        );
        custom.retain(|entry| is_allowed(&entry.definition.name));

        let mut definitions: BTreeMap<String, ToolDefinitionEntry> = BTreeMap::new();
        for (name, definition) in lock(&self.base_tool_definitions).iter() {
            if is_allowed(name) {
                definitions.insert(
                    name.clone(),
                    ToolDefinitionEntry {
                        definition: definition.clone(),
                        source_info: synthetic_source(name, "builtin"),
                    },
                );
            }
        }
        for entry in custom {
            definitions.insert(entry.definition.name.clone(), entry);
        }
        *lock(&self.tool_definitions) = definitions.clone();
        *lock(&self.tool_prompt_snippets) = definitions
            .values()
            .filter_map(|entry| {
                normalize_prompt_snippet(entry.definition.prompt_snippet.as_deref())
                    .map(|snippet| (entry.definition.name.clone(), snippet))
            })
            .collect();
        *lock(&self.tool_prompt_guidelines) = definitions
            .values()
            .filter_map(|entry| {
                let guidelines =
                    normalize_prompt_guidelines(entry.definition.prompt_guidelines.as_deref());
                (!guidelines.is_empty()).then(|| (entry.definition.name.clone(), guidelines))
            })
            .collect();

        let registry: BTreeMap<String, pi_agent_core::types::AgentTool> = definitions
            .values()
            .map(|entry| {
                (
                    entry.definition.name.clone(),
                    self.definition_to_agent_tool(&entry.definition),
                )
            })
            .collect();
        *lock(&self.tool_registry) = registry;

        let include_new = active_tool_names.is_none();
        let mut next: Vec<String> = active_tool_names.unwrap_or(previous_active);
        next.retain(|name| is_allowed(name));
        if let Some(allowed) = &self.allowed_tool_names {
            for name in lock(&self.tool_registry).keys() {
                if allowed.contains(name) {
                    next.push(name.clone());
                }
            }
        } else if include_all_extension_tools {
            next.extend(definitions.keys().cloned());
        } else if include_new {
            for name in definitions.keys() {
                if !previous_registry_names.contains(name) {
                    next.push(name.clone());
                }
            }
        }
        let unique: Vec<String> = next
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        self.set_active_tools_by_name(&unique);
    }

    pub(crate) fn rebuild_system_prompt(&self, tool_names: &[String]) -> String {
        let registry = lock(&self.tool_registry);
        let valid: Vec<String> = tool_names
            .iter()
            .filter(|name| registry.contains_key(*name))
            .cloned()
            .collect();
        drop(registry);
        let mut snippets = std::collections::HashMap::new();
        let mut guidelines = Vec::new();
        for name in &valid {
            if let Some(snippet) = lock(&self.tool_prompt_snippets).get(name) {
                snippets.insert(name.clone(), snippet.clone());
            }
            if let Some(tool_guidelines) = lock(&self.tool_prompt_guidelines).get(name) {
                guidelines.extend(tool_guidelines.iter().cloned());
            }
        }
        let loader = lock(&self.resource_loader);
        let append = loader.get_append_system_prompt();
        let options = crate::system_prompt::BuildSystemPromptOptions {
            cwd: self.cwd.clone(),
            skills: Some(loader.get_skills().skills),
            context_files: Some(
                loader
                    .get_agents_files()
                    .into_iter()
                    .map(|file| crate::system_prompt::ContextFile {
                        path: file.path,
                        content: file.content,
                    })
                    .collect(),
            ),
            custom_prompt: loader.get_system_prompt(),
            append_system_prompt: (!append.is_empty()).then(|| append.join("\n\n")),
            selected_tools: Some(valid),
            tool_snippets: Some(snippets),
            prompt_guidelines: Some(guidelines),
        };
        drop(loader);
        let prompt = crate::system_prompt::build_system_prompt(&options);
        *lock(&self.base_system_prompt_options) = options;
        prompt
    }
}

pub(super) fn synthetic_source(name: &str, source: &str) -> SourceInfo {
    create_synthetic_source_info(
        &format!("<{source}:{name}>"),
        &SyntheticSourceOptions {
            source: source.to_owned(),
            scope: None,
            origin: None,
            base_dir: None,
        },
    )
}

fn normalize_prompt_snippet(text: Option<&str>) -> Option<String> {
    let one_line = text?.split(['\r', '\n']).collect::<Vec<_>>().join(" ");
    let one_line = one_line.split_whitespace().collect::<Vec<_>>().join(" ");
    (!one_line.is_empty()).then_some(one_line)
}

fn normalize_prompt_guidelines(guidelines: Option<&[String]>) -> Vec<String> {
    let mut unique = Vec::new();
    for guideline in guidelines.unwrap_or_default() {
        let normalized = guideline.trim();
        if !normalized.is_empty() && !unique.iter().any(|item| item == normalized) {
            unique.push(normalized.to_owned());
        }
    }
    unique
}
