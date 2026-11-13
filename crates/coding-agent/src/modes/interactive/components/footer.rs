//! The status footer, upstream's
//! `src/modes/interactive/components/footer.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Two lines normally: the working directory with git branch and session
//! name, then the token/cache/cost stats left-aligned and the model (plus
//! provider and thinking level) right-aligned. Extension statuses add a
//! third, sanitized line.
//!
//! Restatements against upstream:
//!
//! - Upstream holds the concrete `AgentSession` and
//!   `ReadonlyFooterDataProvider`; the port takes the [`FooterSession`] and
//!   [`FooterData`] seams so the width/format suite can drive the render
//!   without a live agent. [`FooterSession`] exposes the model fields the
//!   render reads ([`FooterModel`]) instead of the whole
//!   `pi_ai::types::Model`. [`AgentSession`] and [`FooterDataProvider`]
//!   implement the traits.
//! - The session entry walk is a visitor (`for_each_entry`) rather than a
//!   cloned entry list, so a render does not copy every stored message.
//! - The session reference is swappable behind a `RefCell`, matching
//!   upstream's `setSession`; the field is shared as an `Rc` because the
//!   frame and the render both hold the component. `onBranchChange` stays on
//!   the concrete provider: the footer redraws on the frame's next render and
//!   never subscribes itself.
//! - Extension-status keys sort with `str`'s byte order where upstream calls
//!   `localeCompare`; there is no locale collator in `std`. The status text
//!   sanitizer matches upstream exactly (CR/LF/tab to space, runs collapsed,
//!   trimmed).
//! - `process.env.HOME || process.env.USERPROFILE` is
//!   [`crate::config::home_dir`]; `path.resolve` is the crate's
//!   Node-compatible [`resolve_path`] and
//!   the home check is a component-wise prefix comparison.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::Message;
use pi_tui::tui::Component;
use pi_tui::utils::{truncate_to_width, visible_width};

use crate::config::home_dir;
use crate::core::agent_session::AgentSession;
use crate::core::experimental::are_experimental_features_enabled;
use crate::extensions::runtime::ContextUsage;
use crate::footer_data_provider::FooterDataProvider;
use crate::modes::interactive::theme::theme::theme;
use crate::session_manager::{FileEntry, SessionEntry};
use crate::settings_manager::SettingsStorage;
use crate::usage_totals::{add_usage_to_totals, create_usage_totals};
use crate::utils::paths::resolve_path;

/// The model fields the footer reads, the subset of upstream's
/// `state.model` the render touches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FooterModel {
    /// The model id, upstream `state.model.id`.
    pub id: String,
    /// The provider id, upstream `state.model.provider`.
    pub provider: String,
    /// The context window in tokens, upstream `state.model.contextWindow`.
    pub context_window: u64,
    /// Whether the model supports reasoning, upstream
    /// `state.model.reasoning`.
    pub reasoning: bool,
}

/// The session read surface the footer needs, upstream's `AgentSession`
/// fields and getters.
pub trait FooterSession {
    /// The current model, upstream `session.state.model`.
    fn model(&self) -> Option<FooterModel>;
    /// The current thinking level, upstream `session.state.thinkingLevel`.
    fn thinking_level(&self) -> ThinkingLevel;
    /// The context usage reading, upstream `session.getContextUsage()`.
    fn context_usage(&self) -> Option<ContextUsage>;
    /// The working directory, upstream `sessionManager.getCwd()`.
    fn cwd(&self) -> String;
    /// The display name, upstream `sessionManager.getSessionName()`.
    fn session_name(&self) -> Option<String>;
    /// Whether the provider bills through a subscription, upstream
    /// `session.modelRuntime.isUsingSubscription(provider)`.
    fn is_using_subscription(&self, provider: &str) -> bool;
    /// Visit every session entry in file order, upstream
    /// `sessionManager.getEntries()`.
    fn for_each_entry(&self, visit: &mut dyn FnMut(&FileEntry));
}

/// The read-only footer data upstream types as
/// `ReadonlyFooterDataProvider`.
pub trait FooterData {
    /// Current git branch, absent outside a repo.
    fn get_git_branch(&self) -> Option<String>;
    /// Extension status texts in insertion order.
    fn get_extension_statuses(&self) -> Vec<(String, String)>;
    /// Number of providers with available models.
    fn get_available_provider_count(&self) -> usize;
}

impl FooterData for FooterDataProvider {
    fn get_git_branch(&self) -> Option<String> {
        Self::get_git_branch(self)
    }

    fn get_extension_statuses(&self) -> Vec<(String, String)> {
        Self::get_extension_statuses(self)
    }

    fn get_available_provider_count(&self) -> usize {
        Self::get_available_provider_count(self)
    }
}

#[expect(
    clippy::use_self,
    reason = "the fully qualified inherent getters keep their names where the trait methods shadow them"
)]
impl<S: SettingsStorage + 'static> FooterSession for AgentSession<S> {
    fn model(&self) -> Option<FooterModel> {
        AgentSession::model(self).map(|model| FooterModel {
            id: model.id,
            provider: model.provider.0,
            context_window: model.context_window,
            reasoning: model.reasoning,
        })
    }

    fn thinking_level(&self) -> ThinkingLevel {
        AgentSession::thinking_level(self)
    }

    fn context_usage(&self) -> Option<ContextUsage> {
        AgentSession::context_usage(self)
    }

    fn cwd(&self) -> String {
        AgentSession::cwd(self).to_string()
    }

    fn session_name(&self) -> Option<String> {
        AgentSession::session_name(self)
    }

    fn is_using_subscription(&self, provider: &str) -> bool {
        self.model_runtime().is_using_subscription(provider)
    }

    fn for_each_entry(&self, visit: &mut dyn FnMut(&FileEntry)) {
        let manager = crate::core::agent_session::lock(&self.session_manager);
        for entry in manager.entries() {
            visit(entry);
        }
    }
}

/// Format token counts for the compact footer, upstream `formatTokens`.
#[expect(
    clippy::cast_precision_loss,
    reason = "token counts stay far below 2^52; the display mirrors JS `toFixed`/`Math.round`"
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rounded quotient is non-negative and below the input count"
)]
#[must_use]
pub fn format_tokens(count: u64) -> String {
    if count < 1_000 {
        return count.to_string();
    }
    if count < 10_000 {
        return format!("{:.1}k", count as f64 / 1_000.0);
    }
    if count < 1_000_000 {
        return format!("{}k", (count as f64 / 1_000.0).round() as u64);
    }
    if count < 10_000_000 {
        return format!("{:.1}M", count as f64 / 1_000_000.0);
    }
    format!("{}M", (count as f64 / 1_000_000.0).round() as u64)
}

/// Replace the home-directory prefix with `~`, upstream
/// `formatCwdForFooter`.
///
/// A sibling path that merely shares the home string (`/home/user2` under
/// `/home/user`) keeps its full form: the check compares path components,
/// not a string prefix.
#[must_use]
pub fn format_cwd_for_footer(cwd: &str, home: Option<&str>) -> String {
    let Some(home) = home else {
        return cwd.to_string();
    };
    let base = std::env::current_dir()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    let resolved_cwd = resolve_path(cwd, &base, home);
    let resolved_home = resolve_path(home, &base, home);

    let Ok(relative) = Path::new(&resolved_cwd).strip_prefix(Path::new(&resolved_home)) else {
        return cwd.to_string();
    };
    if relative.as_os_str().is_empty() {
        return "~".to_string();
    }
    format!("~{}{}", std::path::MAIN_SEPARATOR, relative.display())
}

/// Sanitize one extension status for a single line, upstream
/// `sanitizeStatusText`.
fn sanitize_status_text(text: &str) -> String {
    let flattened: String = text
        .chars()
        .map(|character| {
            if matches!(character, '\r' | '\n' | '\t') {
                ' '
            } else {
                character
            }
        })
        .collect();
    flattened
        .split(' ')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The upstream wire spelling of a thinking level.
const fn thinking_level_label(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

/// The footer component, upstream `FooterComponent`.
pub struct FooterComponent {
    auto_compact_enabled: Cell<bool>,
    session: RefCell<Rc<dyn FooterSession>>,
    footer_data: Rc<dyn FooterData>,
}

impl std::fmt::Debug for FooterComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FooterComponent")
            .field("auto_compact_enabled", &self.auto_compact_enabled.get())
            .finish_non_exhaustive()
    }
}

impl FooterComponent {
    /// Build the footer over the session and provider seams.
    #[must_use]
    pub fn new(session: Rc<dyn FooterSession>, footer_data: Rc<dyn FooterData>) -> Self {
        Self {
            auto_compact_enabled: Cell::new(true),
            session: RefCell::new(session),
            footer_data,
        }
    }

    /// Point the footer at a new session, upstream `setSession`.
    pub fn set_session(&self, session: Rc<dyn FooterSession>) {
        *self.session.borrow_mut() = session;
    }

    /// Set whether the `(auto)` context suffix shows, upstream
    /// `setAutoCompactEnabled`.
    pub fn set_auto_compact_enabled(&self, enabled: bool) {
        self.auto_compact_enabled.set(enabled);
    }

    /// No-op kept for upstream's call sites; the provider owns git-branch
    /// caching, upstream `invalidate`.
    pub const fn invalidate(&self) {}

    /// No-op kept for upstream's call sites; the provider owns watcher
    /// cleanup, upstream `dispose`.
    pub const fn dispose(&self) {}
}

impl Component for FooterComponent {
    #[expect(
        clippy::cast_precision_loss,
        reason = "usage counts stay far below 2^52; the percentage mirrors upstream's float math"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's one render pass so the layout arithmetic stays comparable"
    )]
    fn render(&self, width: usize) -> Vec<String> {
        let session = Rc::clone(&self.session.borrow());
        let model = session.model();
        let thinking_level = session.thinking_level();
        let active_theme = theme();

        // Cumulative usage over every entry, not just post-compaction
        // messages; the cache hit rate is the latest assistant request's.
        let mut usage_totals = create_usage_totals();
        let mut latest_cache_hit_rate: Option<f64> = None;
        session.for_each_entry(&mut |entry| match entry {
            FileEntry::Entry(SessionEntry::Message(message_entry)) => {
                let Some(AgentMessage::Standard(message)) = &message_entry.message else {
                    return;
                };
                match message {
                    Message::Assistant(assistant) => {
                        add_usage_to_totals(&mut usage_totals, &assistant.usage);
                        let latest_prompt_tokens = assistant.usage.input
                            + assistant.usage.cache_read
                            + assistant.usage.cache_write;
                        latest_cache_hit_rate = (latest_prompt_tokens > 0).then(|| {
                            assistant.usage.cache_read as f64 / latest_prompt_tokens as f64 * 100.0
                        });
                    }
                    Message::ToolResult(result) => {
                        if let Some(usage) = &result.usage {
                            add_usage_to_totals(&mut usage_totals, usage);
                        }
                    }
                    Message::User(_) => {}
                }
            }
            FileEntry::Entry(SessionEntry::BranchSummary(branch)) => {
                if let Some(usage) = &branch.usage {
                    add_usage_to_totals(&mut usage_totals, usage);
                }
            }
            FileEntry::Entry(SessionEntry::Compaction(compaction)) => {
                if let Some(usage) = &compaction.usage {
                    add_usage_to_totals(&mut usage_totals, usage);
                }
            }
            FileEntry::Entry(_) | FileEntry::Session(_) | FileEntry::Other(_) => {}
        });

        // Context usage comes from the session so compaction is handled;
        // after compaction the percent is unknown until the next response.
        let context_usage = session.context_usage();
        let context_window = context_usage.as_ref().map_or_else(
            || model.as_ref().map_or(0, |model| model.context_window),
            |usage| usage.context_window,
        );
        let context_percent_value = context_usage
            .as_ref()
            .and_then(|usage| usage.percent)
            .unwrap_or(0.0);
        let context_percent = if context_usage
            .as_ref()
            .is_some_and(|usage| usage.percent.is_none())
        {
            "?".to_string()
        } else {
            format!("{context_percent_value:.1}")
        };

        let home = home_dir();
        let home = (!home.is_empty()).then_some(home.as_str());
        let mut pwd = format_cwd_for_footer(&session.cwd(), home);
        if let Some(branch) = self
            .footer_data
            .get_git_branch()
            .filter(|branch| !branch.is_empty())
        {
            pwd = format!("{pwd} ({branch})");
        }
        if let Some(name) = session.session_name().filter(|name| !name.is_empty()) {
            pwd = format!("{pwd} • {name}");
        }

        let mut stats_parts: Vec<String> = Vec::new();
        if usage_totals.input > 0 {
            stats_parts.push(format!("↑{}", format_tokens(usage_totals.input)));
        }
        if usage_totals.output > 0 {
            stats_parts.push(format!("↓{}", format_tokens(usage_totals.output)));
        }
        if usage_totals.cache_read > 0 {
            stats_parts.push(format!("R{}", format_tokens(usage_totals.cache_read)));
        }
        if usage_totals.cache_write > 0 {
            stats_parts.push(format!("W{}", format_tokens(usage_totals.cache_write)));
        }
        if (usage_totals.cache_read > 0 || usage_totals.cache_write > 0)
            && let Some(rate) = latest_cache_hit_rate
        {
            stats_parts.push(format!("CH{rate:.1}%"));
        }

        // Kimi Coding is subscription-backed despite API-key auth.
        let using_subscription = model.as_ref().is_some_and(|model| {
            model.provider == "kimi-coding" || session.is_using_subscription(&model.provider)
        });
        if usage_totals.cost != 0.0 || using_subscription {
            let suffix = if using_subscription { " (sub)" } else { "" };
            stats_parts.push(format!("${:.3}{suffix}", usage_totals.cost));
        }

        let auto_indicator = if self.auto_compact_enabled.get() {
            " (auto)"
        } else {
            ""
        };
        let context_percent_display = if context_percent == "?" {
            format!("?/{}{auto_indicator}", format_tokens(context_window))
        } else {
            format!(
                "{context_percent}%/{}{auto_indicator}",
                format_tokens(context_window)
            )
        };
        let context_percent_str = if context_percent_value > 90.0 {
            active_theme.fg("error", &context_percent_display)
        } else if context_percent_value > 70.0 {
            active_theme.fg("warning", &context_percent_display)
        } else {
            context_percent_display
        };
        stats_parts.push(context_percent_str);
        if are_experimental_features_enabled() {
            stats_parts.push(format!(
                "{} {}",
                active_theme.fg("dim", "•"),
                crate::modes::interactive::theme::theme::bold(&active_theme.fg("warning", "xp"))
            ));
        }

        let mut stats_left = stats_parts.join(" ");
        let mut stats_left_width = visible_width(&stats_left);
        if stats_left_width > width {
            stats_left = truncate_to_width(&stats_left, width, "...", false);
            stats_left_width = visible_width(&stats_left);
        }

        // Right side: model name, thinking level, then the provider when
        // more than one offers models and the line still fits.
        let model_name = model
            .as_ref()
            .map(|model| model.id.clone())
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| "no-model".to_string());
        let right_side_without_provider = if model.as_ref().is_some_and(|model| model.reasoning) {
            let level = thinking_level_label(thinking_level);
            if level == "off" {
                format!("{model_name} • thinking off")
            } else {
                format!("{model_name} • {level}")
            }
        } else {
            model_name
        };
        let min_padding = 2;
        let right_side = if self.footer_data.get_available_provider_count() > 1
            && let Some(model) = &model
        {
            let prefixed = format!("({}) {right_side_without_provider}", model.provider);
            if stats_left_width + min_padding + visible_width(&prefixed) <= width {
                prefixed
            } else {
                right_side_without_provider
            }
        } else {
            right_side_without_provider
        };

        let right_side_width = visible_width(&right_side);
        let total_needed = stats_left_width + min_padding + right_side_width;
        let (stats_left, remainder) = if total_needed <= width {
            let padding = " ".repeat(width - stats_left_width - right_side_width);
            (stats_left, format!("{padding}{right_side}"))
        } else {
            let needed = stats_left_width + min_padding;
            if width > needed {
                let available_for_right = width - needed;
                let truncated = truncate_to_width(&right_side, available_for_right, "", false);
                let truncated_width = visible_width(&truncated);
                let padding = " ".repeat(width.saturating_sub(stats_left_width + truncated_width));
                (stats_left, format!("{padding}{truncated}"))
            } else {
                (stats_left, String::new())
            }
        };

        // Dim the parts around the colored context percentage separately,
        // so its reset does not clear the outer dim wrapper.
        let dim_stats_left = active_theme.fg("dim", &stats_left);
        let dim_remainder = active_theme.fg("dim", &remainder);
        let pwd_line = truncate_to_width(
            &active_theme.fg("dim", &pwd),
            width,
            &active_theme.fg("dim", "..."),
            false,
        );
        let mut lines = vec![pwd_line, dim_stats_left + &dim_remainder];

        let mut statuses = self.footer_data.get_extension_statuses();
        if !statuses.is_empty() {
            statuses.sort_by(|left, right| left.0.cmp(&right.0));
            let status_line = statuses
                .iter()
                .map(|(_, text)| sanitize_status_text(text))
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(truncate_to_width(
                &status_line,
                width,
                &active_theme.fg("dim", "..."),
                false,
            ));
        }

        lines
    }
}
