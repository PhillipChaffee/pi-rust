//! CLI argument parsing and help display, upstream's `src/cli/args.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The parse is upstream's single order-sensitive ladder: the first match
//! wins, value-taking flags consume the next argument only when it exists,
//! and everything `--` precedes stays positional. Unknown long flags collect
//! into [`Args::unknown_flags`] for the extension flag surface; unknown short
//! options are diagnostics.
//!
//! The help text drops upstream's `chalk.bold` calls: colors ride the theme
//! system slice, and chalk itself prints plain when stdout is not a TTY.

use indexmap::IndexMap;

use pi_agent_core::types::ThinkingLevel;
use pi_tui::tui::TuiMode;

use crate::config::{APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, ENV_SESSION_DIR};
use crate::extensions::runtime::{ExtensionFlag, FlagType};

/// The output mode, upstream's `Mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The default interactive-capable text mode.
    Text,
    /// Line-delimited JSON events.
    Json,
    /// JSON-RPC over stdio.
    Rpc,
}

/// The `--list-models` operand, upstream's `string | true`: the bare flag
/// lists everything, the value filters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListModels {
    /// The bare flag.
    All,
    /// A fuzzy search pattern.
    Search(String),
}

/// A diagnostic severity, upstream's `"warning" | "error"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticKind {
    /// The parse continues; the caller prints a warning.
    Warning,
    /// The caller exits after printing the error.
    Error,
}

/// One parse diagnostic, upstream's `{ type, message }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// The severity, upstream's `type`.
    pub kind: DiagnosticKind,
    /// The caller-facing message, upstream's `message`.
    pub message: String,
}

/// An unknown long flag's value, upstream's `boolean | string` map value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownFlagValue {
    /// The flag appeared without a value, upstream's `true`.
    Set,
    /// The flag's value, from `--flag=value` or the following argument.
    Value(String),
}

impl UnknownFlagValue {
    /// The value as text, `None` for the bare flag.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Set => None,
            Self::Value(value) => Some(value),
        }
    }
}

/// The parsed command line, upstream's `Args`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the bool fields mirror upstream's `Args` interface 1:1; the parse is the contract, not a state machine to model"
)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Args {
    /// `--provider`.
    pub provider: Option<String>,
    /// `--model`.
    pub model: Option<String>,
    /// `--api-key`.
    pub api_key: Option<String>,
    /// `--system-prompt`.
    pub system_prompt: Option<String>,
    /// `--append-system-prompt`, repeatable.
    pub append_system_prompt: Option<Vec<String>>,
    /// `--thinking`.
    pub thinking: Option<ThinkingLevel>,
    /// `--continue`/`-c`, upstream's `continue`.
    pub continue_session: bool,
    /// `--resume`/`-r`.
    pub resume: bool,
    /// `--help`/`-h`.
    pub help: bool,
    /// `--version`/`-v`.
    pub version: bool,
    /// `--mode`.
    pub mode: Option<Mode>,
    /// `--name`/`-n`.
    pub name: Option<String>,
    /// `--no-session`.
    pub no_session: bool,
    /// `--session`.
    pub session: Option<String>,
    /// `--session-id`.
    pub session_id: Option<String>,
    /// `--fork`.
    pub fork: Option<String>,
    /// `--session-dir`.
    pub session_dir: Option<String>,
    /// `--models`, split on comma with entries trimmed.
    pub models: Option<Vec<String>>,
    /// `--tools`/`-t`, comma-split with empty entries dropped.
    pub tools: Option<Vec<String>>,
    /// `--exclude-tools`/`-xt`, comma-split with empty entries dropped.
    pub exclude_tools: Option<Vec<String>>,
    /// `--no-tools`/`-nt`.
    pub no_tools: bool,
    /// `--no-builtin-tools`/`-nbt`.
    pub no_builtin_tools: bool,
    /// `--extension`/`-e`, repeatable.
    pub extensions: Option<Vec<String>>,
    /// `--no-extensions`/`-ne`.
    pub no_extensions: bool,
    /// `--print`/`-p`.
    pub print: bool,
    /// `--export`.
    pub export: Option<String>,
    /// `--no-skills`/`-ns`.
    pub no_skills: bool,
    /// `--skill`, repeatable.
    pub skills: Option<Vec<String>>,
    /// `--prompt-template`, repeatable.
    pub prompt_templates: Option<Vec<String>>,
    /// `--no-prompt-templates`/`-np`.
    pub no_prompt_templates: bool,
    /// `--theme`, repeatable.
    pub themes: Option<Vec<String>>,
    /// `--use-theme`.
    pub use_theme: Option<String>,
    /// `--no-themes`.
    pub no_themes: bool,
    /// `--no-context-files`/`-nc`.
    pub no_context_files: bool,
    /// `--list-models`, with its optional search pattern.
    pub list_models: Option<ListModels>,
    /// `--offline`.
    pub offline: bool,
    /// `--tui-mode`.
    pub tui_mode: Option<TuiMode>,
    /// `--verbose`.
    pub verbose: bool,
    /// `--approve`/`-a` or `--no-approve`/`-na`; `None` leaves the stored
    /// decision in force.
    pub project_trust_override: Option<bool>,
    /// Positional messages, in order.
    pub messages: Vec<String>,
    /// `@file` arguments with the `@` stripped, in order.
    pub file_args: Vec<String>,
    /// Unknown long flags, in first-seen order, potentially extension flags.
    pub unknown_flags: IndexMap<String, UnknownFlagValue>,
    /// Parse diagnostics, in order.
    pub diagnostics: Vec<Diagnostic>,
}

/// The thinking levels `--thinking` accepts, upstream's
/// `VALID_THINKING_LEVELS`, in upstream order.
pub const VALID_THINKING_LEVELS: [&str; 7] =
    ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Parse a `--thinking` value, upstream's `isValidThinkingLevel` type guard.
#[must_use]
pub fn parse_thinking_level(level: &str) -> Option<ThinkingLevel> {
    match level {
        "off" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::Xhigh),
        "max" => Some(ThinkingLevel::Max),
        _ => None,
    }
}

/// Trim a display name and reject whitespace-only values, upstream's
/// `normalizeSessionName`.
#[must_use]
pub fn normalize_session_name(value: &str) -> Option<String> {
    let name = value.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn comma_list(value: &str, drop_empty: bool) -> Vec<String> {
    value
        .split(',')
        .map(|entry| entry.trim().to_string())
        .filter(|entry| !drop_empty || !entry.is_empty())
        .collect()
}

/// Parse a command line, upstream's `parseArgs`.
#[expect(
    clippy::too_many_lines,
    reason = "upstream's single order-sensitive ladder; splitting it would hide that the first matching flag wins and which branches consume the next argument"
)]
#[must_use]
pub fn parse_args<S: AsRef<str>>(args: &[S]) -> Args {
    let mut result = Args::default();
    let args: Vec<&str> = args.iter().map(AsRef::as_ref).collect();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];

        if arg == "--" {
            for positional in &args[i + 1..] {
                if let Some(file) = positional.strip_prefix('@') {
                    result.file_args.push(file.to_string());
                } else {
                    result.messages.push((*positional).to_string());
                }
            }
            break;
        } else if arg == "--help" || arg == "-h" {
            result.help = true;
        } else if arg == "--version" || arg == "-v" {
            result.version = true;
        } else if arg == "--mode" && i + 1 < args.len() {
            let mode = args[i + 1];
            if let Some(mode) = match mode {
                "text" => Some(Mode::Text),
                "json" => Some(Mode::Json),
                "rpc" => Some(Mode::Rpc),
                _ => None,
            } {
                result.mode = Some(mode);
            }
            i += 1;
        } else if arg == "--continue" || arg == "-c" {
            result.continue_session = true;
        } else if arg == "--resume" || arg == "-r" {
            result.resume = true;
        } else if arg == "--provider" && i + 1 < args.len() {
            result.provider = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--model" && i + 1 < args.len() {
            result.model = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--api-key" && i + 1 < args.len() {
            result.api_key = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--system-prompt" && i + 1 < args.len() {
            result.system_prompt = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--append-system-prompt" && i + 1 < args.len() {
            result
                .append_system_prompt
                .get_or_insert_with(Vec::new)
                .push(args[i + 1].to_string());
            i += 1;
        } else if arg == "--name" || arg == "-n" {
            if i + 1 < args.len() {
                result.name = Some(args[i + 1].to_string());
                i += 1;
            } else {
                result.diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::Error,
                    message: "--name requires a value".to_string(),
                });
            }
        } else if arg == "--no-session" {
            result.no_session = true;
        } else if arg == "--session" && i + 1 < args.len() {
            result.session = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--session-id" && i + 1 < args.len() {
            result.session_id = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--fork" && i + 1 < args.len() {
            result.fork = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--session-dir" && i + 1 < args.len() {
            result.session_dir = Some(args[i + 1].to_string());
            i += 1;
        } else if arg == "--models" && i + 1 < args.len() {
            result.models = Some(comma_list(args[i + 1], false));
            i += 1;
        } else if arg == "--no-tools" || arg == "-nt" {
            result.no_tools = true;
        } else if arg == "--no-builtin-tools" || arg == "-nbt" {
            result.no_builtin_tools = true;
        } else if (arg == "--tools" || arg == "-t") && i + 1 < args.len() {
            result.tools = Some(comma_list(args[i + 1], true));
            i += 1;
        } else if (arg == "--exclude-tools" || arg == "-xt") && i + 1 < args.len() {
            result.exclude_tools = Some(comma_list(args[i + 1], true));
            i += 1;
        } else if arg == "--thinking" && i + 1 < args.len() {
            let level = args[i + 1];
            i += 1;
            if let Some(thinking) = parse_thinking_level(level) {
                result.thinking = Some(thinking);
            } else {
                result.diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::Warning,
                    message: format!(
                        "Invalid thinking level \"{level}\". Valid values: {}",
                        VALID_THINKING_LEVELS.join(", ")
                    ),
                });
            }
        } else if arg == "--print" || arg == "-p" {
            result.print = true;
            if let Some(next) = args.get(i + 1)
                && !next.starts_with('@')
                && (!next.starts_with('-') || next.starts_with("---"))
            {
                result.messages.push((*next).to_string());
                i += 1;
            }
        } else if arg == "--export" && i + 1 < args.len() {
            result.export = Some(args[i + 1].to_string());
            i += 1;
        } else if (arg == "--extension" || arg == "-e") && i + 1 < args.len() {
            result
                .extensions
                .get_or_insert_with(Vec::new)
                .push(args[i + 1].to_string());
            i += 1;
        } else if arg == "--no-extensions" || arg == "-ne" {
            result.no_extensions = true;
        } else if arg == "--skill" && i + 1 < args.len() {
            result
                .skills
                .get_or_insert_with(Vec::new)
                .push(args[i + 1].to_string());
            i += 1;
        } else if arg == "--prompt-template" && i + 1 < args.len() {
            result
                .prompt_templates
                .get_or_insert_with(Vec::new)
                .push(args[i + 1].to_string());
            i += 1;
        } else if arg == "--theme" && i + 1 < args.len() {
            result
                .themes
                .get_or_insert_with(Vec::new)
                .push(args[i + 1].to_string());
            i += 1;
        } else if arg == "--use-theme" {
            match args.get(i + 1) {
                Some(theme_name) if !theme_name.starts_with('-') => {
                    result.use_theme = Some((*theme_name).to_string());
                    i += 1;
                }
                _ => result.diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::Error,
                    message: "--use-theme requires a theme name".to_string(),
                }),
            }
        } else if arg == "--no-skills" || arg == "-ns" {
            result.no_skills = true;
        } else if arg == "--no-prompt-templates" || arg == "-np" {
            result.no_prompt_templates = true;
        } else if arg == "--no-themes" {
            result.no_themes = true;
        } else if arg == "--no-context-files" || arg == "-nc" {
            result.no_context_files = true;
        } else if arg == "--list-models" {
            result.list_models = match args.get(i + 1) {
                Some(next) if !next.starts_with('-') && !next.starts_with('@') => {
                    i += 1;
                    Some(ListModels::Search((*next).to_string()))
                }
                _ => Some(ListModels::All),
            };
        } else if arg == "--tui-mode" {
            match args.get(i + 1).copied() {
                Some("regular") => {
                    result.tui_mode = Some(TuiMode::Regular);
                    i += 1;
                }
                Some("fullscreen") => {
                    result.tui_mode = Some(TuiMode::Fullscreen);
                    i += 1;
                }
                Some(mode) if !mode.starts_with('-') => {
                    i += 1;
                    result.diagnostics.push(Diagnostic {
                        kind: DiagnosticKind::Error,
                        message: format!(
                            "Invalid TUI mode \"{mode}\". Valid values: regular, fullscreen"
                        ),
                    });
                }
                _ => result.diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::Error,
                    message: "--tui-mode requires regular or fullscreen".to_string(),
                }),
            }
        } else if arg == "--verbose" {
            result.verbose = true;
        } else if arg == "--approve" || arg == "-a" {
            result.project_trust_override = Some(true);
        } else if arg == "--no-approve" || arg == "-na" {
            result.project_trust_override = Some(false);
        } else if arg == "--offline" {
            result.offline = true;
        } else if let Some(file) = arg.strip_prefix('@') {
            result.file_args.push(file.to_string());
        } else if let Some(flag) = arg.strip_prefix("--") {
            if let Some((name, value)) = flag.split_once('=') {
                result
                    .unknown_flags
                    .insert(name.to_string(), UnknownFlagValue::Value(value.to_string()));
            } else {
                match args.get(i + 1) {
                    Some(next) if !next.starts_with('-') && !next.starts_with('@') => {
                        result.unknown_flags.insert(
                            flag.to_string(),
                            UnknownFlagValue::Value((*next).to_string()),
                        );
                        i += 1;
                    }
                    _ => {
                        result
                            .unknown_flags
                            .insert(flag.to_string(), UnknownFlagValue::Set);
                    }
                }
            }
        } else if let Some(option) = arg.strip_prefix('-') {
            result.diagnostics.push(Diagnostic {
                kind: DiagnosticKind::Error,
                message: format!("Unknown option: -{option}"),
            });
        } else {
            result.messages.push(arg.to_string());
        }

        i += 1;
    }

    result
}

/// The help text, upstream's `printHelp` template body.
const HELP_TEMPLATE: &str = concat!(
    r#"{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [--] [@files...] [messages...]

Commands:
  {APP_NAME} install <source> [-l]     Install extension source and add to settings
  {APP_NAME} remove <source> [-l]      Remove extension source from settings
  {APP_NAME} uninstall <source> [-l]   Alias for remove
  {APP_NAME} update [source|self|pi]   Update pi, extensions, or model catalogs
  {APP_NAME} list                      List installed extensions from settings
  {APP_NAME} config [-l]               Open TUI to enable/disable package resources (Tab switches scope)
  {APP_NAME} auth <command>            Print credentials or check provider readiness
  {APP_NAME} <command> --help          Show help for install/remove/uninstall/update/list/config/auth

Options:
  --provider <name>              Provider name (default: google)
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt (can be used multiple times)
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path|id>            Use specific session file or partial UUID
  --session-id <id>              Use exact project session ID, creating it if missing
  --fork <path|id>               Fork specific session file or partial UUID into a new session
  --session-dir <dir>            Directory for session storage and lookup
  --no-session                   Don't save session (ephemeral)
  --name, -n <name>              Set session display name
  --models <patterns>            Comma-separated model patterns for Ctrl+P cycling
                                 Supports globs (anthropic/*, *sonnet*) and fuzzy matching
  --no-tools, -nt                Disable all tools by default (built-in and extension)
  --no-builtin-tools, -nbt       Disable built-in tools by default but keep extension/custom tools enabled
  --tools, -t <tools>            Comma-separated allowlist of tool names to enable
                                 Applies to built-in, extension, and custom tools
  --exclude-tools, -xt <tools>   Comma-separated denylist of tool names to disable
                                 Applies to built-in, extension, and custom tools
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh, max
  --extension, -e <path>         Load an extension file (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery (explicit -e paths still work)
  --skill <path>                 Load a skill file or directory (can be used multiple times)
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory (can be used multiple times)
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory (can be used multiple times)
  --use-theme <name[/name]>      Set the initial interactive theme for this run
  --no-themes                    Disable theme discovery and loading
  --no-context-files, -nc        Disable AGENTS.md and CLAUDE.md discovery and loading
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models (with optional fuzzy search)
  --verbose                      Force verbose startup (overrides quietStartup setting)
  --tui-mode <mode>              TUI mode: regular (default) or fullscreen
  --approve, -a                  Trust project-local files for this run
  --no-approve, -na              Ignore project-local files for this run
  --offline                      Disable startup network operations (same as PI_OFFLINE=1)
  --                             End option parsing; treat remaining arguments as messages/files
  --help, -h                     Show this help
  --version, -v                  Show version number

Extensions can register additional flags (e.g., --plan from plan-mode extension).{EXTENSION_FLAGS}

Examples:
  # Print a provider API key for an external client
  {APP_NAME} auth print-api-key --provider openai

  # Print an OAuth bearer token for an external client (refreshes if expired)
  {APP_NAME} auth print-bearer-token --provider openai-codex

  # Interactive mode
  {APP_NAME}

  # Interactive mode with initial prompt
  {APP_NAME} "List all .ts files in src/"

  # Include files in initial message
  {APP_NAME} @prompt.md @image.png "What color is the sky?"

  # Non-interactive mode (process and exit)
  {APP_NAME} -p "List all .ts files in src/"

  # Prompt beginning with a dash
  {APP_NAME} -p -- "- Summarize these points"

  # Multiple messages (interactive)
  {APP_NAME} "Read package.json" "What dependencies do we have?"

  # Continue previous session
  {APP_NAME} --continue "What did we discuss?"

  # Start a named session
  {APP_NAME} --name "Refactor auth module"

  # Use different model
  {APP_NAME} --provider openai --model gpt-4o-mini "Help me refactor this code"

  # Use model with provider prefix (no --provider needed)
  {APP_NAME} --model openai/gpt-4o "Help me refactor this code"

  # Use model with thinking level shorthand
  {APP_NAME} --model sonnet:high "Solve this complex problem"

  # Limit model cycling to specific models
  {APP_NAME} --models claude-sonnet,claude-haiku,gpt-4o

  # Limit to a specific provider with glob pattern
  {APP_NAME} --models "github-copilot/*"

  # Cycle models with fixed thinking levels
  {APP_NAME} --models sonnet:high,haiku:low

  # Start with a specific thinking level
  {APP_NAME} --thinking high "Solve this complex problem"

  # Read-only mode (no file modifications possible)
  {APP_NAME} --tools read,grep,find,ls -p "Review the code in src/"

  # Disable one tool while keeping the rest available
  {APP_NAME} --exclude-tools ask_question

  # Export a session file to HTML
  {APP_NAME} --export ~/{CONFIG_DIR_NAME}/agent/sessions/--path--/session.jsonl
  {APP_NAME} --export session.jsonl output.html

"#,
    r#"Environment Variables:
  ANTHROPIC_AUTH_TOKEN             - Anthropic bearer auth token
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  ANTHROPIC_OAUTH_TOKEN            - Anthropic OAuth token (alternative to API key)
  ANT_LING_API_KEY                 - Ant Ling API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  AZURE_OPENAI_API_KEY             - Azure OpenAI API key
  AZURE_OPENAI_BASE_URL            - Azure OpenAI/Cognitive Services base URL (e.g. https://{resource}.openai.azure.com)
  AZURE_OPENAI_RESOURCE_NAME       - Azure OpenAI resource name (alternative to base URL)
  AZURE_OPENAI_API_VERSION         - Azure OpenAI API version (default: v1)
  AZURE_OPENAI_DEPLOYMENT_NAME_MAP - Azure OpenAI model=deployment map (comma-separated)
  DEEPSEEK_API_KEY                 - DeepSeek API key
  NVIDIA_API_KEY                   - NVIDIA NIM API key
  GEMINI_API_KEY                   - Google Gemini API key
  GROQ_API_KEY                     - Groq API key
  CEREBRAS_API_KEY                 - Cerebras API key
  XAI_API_KEY                      - xAI Grok API key
  FIREWORKS_API_KEY                - Fireworks API key
  TOGETHER_API_KEY                 - Together AI API key
  BASETEN_API_KEY                  - Baseten API key
  OPENROUTER_API_KEY               - OpenRouter API key
  AI_GATEWAY_API_KEY               - Vercel AI Gateway API key
  ZAI_API_KEY                      - ZAI Coding Plan API key (Global)
  ZAI_CODING_CN_API_KEY            - ZAI Coding Plan API key (China)
  MISTRAL_API_KEY                  - Mistral API key
  MINIMAX_API_KEY                  - MiniMax API key
  MOONSHOT_API_KEY                 - Moonshot AI API key
  OPENCODE_API_KEY                 - OpenCode Zen/OpenCode Go API key
  KIMI_API_KEY                     - Kimi For Coding API key
  CLOUDFLARE_API_KEY               - Cloudflare API token (Workers AI and AI Gateway)
  CLOUDFLARE_ACCOUNT_ID            - Cloudflare account id (required for both)
  CLOUDFLARE_GATEWAY_ID            - Cloudflare AI Gateway slug (required for AI Gateway)
  QWEN_TOKEN_PLAN_API_KEY          - Qwen Token Plan API key (international region)
  QWEN_TOKEN_PLAN_CN_API_KEY       - Qwen Token Plan API key (China region)
  XIAOMI_API_KEY                   - Xiaomi MiMo API key (api.xiaomimimo.com billing)
  XIAOMI_TOKEN_PLAN_CN_API_KEY     - Xiaomi MiMo Token Plan API key (China region)
  XIAOMI_TOKEN_PLAN_AMS_API_KEY    - Xiaomi MiMo Token Plan API key (Amsterdam region)
  XIAOMI_TOKEN_PLAN_SGP_API_KEY    - Xiaomi MiMo Token Plan API key (Singapore region)
  AWS_PROFILE                      - AWS profile for Amazon Bedrock
  AWS_ACCESS_KEY_ID                - AWS access key for Amazon Bedrock
  AWS_SECRET_ACCESS_KEY            - AWS secret key for Amazon Bedrock
  AWS_BEARER_TOKEN_BEDROCK         - Bedrock API key (bearer token)
  AWS_REGION                       - AWS region for Amazon Bedrock (e.g., us-east-1)
  {ENV_AGENT_DIR_PADDED} - Config directory (default: ~/{CONFIG_DIR_NAME}/agent)
  {ENV_SESSION_DIR_PADDED} - Session storage directory (overridden by --session-dir)
  PI_PACKAGE_DIR                   - Override package directory (for Nix/Guix store paths)
  PI_OFFLINE                       - Disable startup network operations when set to 1/true/yes
  PI_TELEMETRY                     - Override install telemetry when set to 1/true/yes or 0/false/no
  PI_SHARE_VIEWER_URL              - Base URL for /share command (default: https://pi.dev/session/)

Built-in Tool Names:
  read       - Read file contents
  bash       - Execute bash commands
  powershell - Execute PowerShell commands on Windows
  edit       - Edit files with find/replace
  write      - Write files (creates/overwrites)
  grep       - Search file contents (read-only, off by default)
  find       - Find files by glob pattern (read-only, off by default)
  ls         - List directory contents (read-only, off by default)
"#
);

/// Render the help text, upstream's `printHelp` body.
#[must_use]
pub fn help_text(extension_flags: Option<&[ExtensionFlag]>) -> String {
    use std::fmt::Write as _;

    let extension_flags_text = extension_flags
        .filter(|flags| !flags.is_empty())
        .map(|flags| {
            let mut text = String::from("\nExtension CLI Flags:\n");
            for flag in flags {
                let value = if flag.flag_type == FlagType::String {
                    " <value>"
                } else {
                    ""
                };
                let description = flag
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("Registered by {}", flag.extension_path));
                let flag_text = format!("  --{}{value}", flag.name);
                let _ = writeln!(text, "{flag_text:<30}{description}");
            }
            text
        })
        .unwrap_or_default();

    HELP_TEMPLATE
        .replace("{APP_NAME}", APP_NAME)
        .replace("{CONFIG_DIR_NAME}", CONFIG_DIR_NAME)
        .replace("{EXTENSION_FLAGS}", &extension_flags_text)
        .replace("{ENV_AGENT_DIR_PADDED}", &format!("{ENV_AGENT_DIR:<32}"))
        .replace(
            "{ENV_SESSION_DIR_PADDED}",
            &format!("{ENV_SESSION_DIR:<32}"),
        )
}

/// Print [`help_text`] to stdout, upstream's `printHelp`.
///
/// # Errors
/// [`std::io::Error`] when stdout cannot be written.
pub fn print_help(extension_flags: Option<&[ExtensionFlag]>) -> std::io::Result<()> {
    use std::io::Write as _;

    let mut stdout = std::io::stdout().lock();
    write!(stdout, "{}", help_text(extension_flags))
}

#[cfg(test)]
mod tests;
