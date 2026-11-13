//! The editor's status indicators, upstream
//! `src/modes/interactive/components/status-indicator.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#213).
//!
//! Restatements: `class StatusIndicator extends Loader` becomes composition
//! over [`Loader`]; upstream's `TUI` handle narrows to the [`RenderRequest`]
//! closure, and its `WorkingIndicatorOptions` is the pi-tui
//! [`LoaderIndicatorOptions`] the loader already takes. `renderSpinnerInBorder`
//! reads the loader line and keeps the leading styled token, because the
//! port's loader keeps its current frame private. The retry countdown is the
//! manual-tick [`CountdownTimer`]; tests fire its ticks inline.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::{ColorFn, Loader, LoaderIndicatorOptions};
use pi_tui::tui::{Component, RenderRequest};
use pi_tui::utils::truncate_to_width;

use super::countdown_timer::CountdownTimer;
use super::keybinding_hints::key_text;
use crate::modes::interactive::theme::theme::theme;

/// The status the indicator reports, upstream's `StatusIndicatorKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusIndicatorKind {
    /// A model request is in flight.
    Working,
    /// A retry is counting down.
    Retry,
    /// Context compaction is running.
    Compaction,
    /// A branch summary is being generated.
    BranchSummary,
}

impl StatusIndicatorKind {
    /// The upstream wire spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Retry => "retry",
            Self::Compaction => "compaction",
            Self::BranchSummary => "branchSummary",
        }
    }
}

/// The shared loader-backed indicator, upstream's `StatusIndicator`.
pub struct StatusIndicator {
    kind: StatusIndicatorKind,
    loader: Loader,
    message: RefCell<String>,
    indicator: RefCell<Option<LoaderIndicatorOptions>>,
}

impl std::fmt::Debug for StatusIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatusIndicator")
            .field("kind", &self.kind)
            .field("message", &self.message.borrow())
            .finish_non_exhaustive()
    }
}

impl StatusIndicator {
    /// Build the loader, upstream's constructor.
    #[must_use]
    pub fn new(
        kind: StatusIndicatorKind,
        render_request: RenderRequest,
        spinner_color_fn: ColorFn,
        message_color_fn: ColorFn,
        message: impl Into<String>,
        indicator: Option<LoaderIndicatorOptions>,
    ) -> Self {
        let message = message.into();
        let loader = Loader::new(
            render_request,
            spinner_color_fn,
            message_color_fn,
            message.clone(),
            indicator.clone(),
        );
        Self {
            kind,
            loader,
            message: RefCell::new(message),
            indicator: RefCell::new(indicator),
        }
    }

    /// The status kind, upstream's readonly `kind`.
    #[must_use]
    pub const fn kind(&self) -> StatusIndicatorKind {
        self.kind
    }

    /// Replace the message, upstream `setMessage`.
    pub fn set_message(&self, message: impl Into<String>) {
        let message = message.into();
        self.message.borrow_mut().clone_from(&message);
        self.loader.set_message(message);
    }

    /// Replace the animation options, upstream `setIndicator`.
    pub fn set_indicator(&self, indicator: Option<LoaderIndicatorOptions>) {
        self.indicator.borrow_mut().clone_from(&indicator);
        self.loader.set_indicator(indicator);
    }

    /// The full indicator line clipped to `width`, upstream's
    /// `renderInBorder`: drop the loader's left padding and trailing spaces
    /// first, so the editor can place its own border segments.
    #[must_use]
    pub fn render_in_border(&self, width: usize) -> String {
        let rendered = self.loader.render(width.saturating_add(2));
        let line = rendered.get(1).cloned().unwrap_or_default();
        let trimmed = line.strip_prefix(' ').unwrap_or(&line).trim_end();
        truncate_to_width(trimmed, width, "", false)
    }

    /// Only the animated spinner, clipped to `width`, upstream's
    /// `renderSpinnerInBorder`.
    #[must_use]
    pub fn render_spinner_in_border(&self, width: usize) -> String {
        if width == 0 || indicator_hidden(self.indicator.borrow().as_ref()) {
            return String::new();
        }
        let rendered = self.loader.render(width.saturating_add(2));
        let line = rendered.get(1).cloned().unwrap_or_default();
        let line = line.strip_prefix(' ').unwrap_or(&line);
        truncate_to_width(leading_styled_token(line).trim_end(), width, "", false)
    }

    /// Stop the loader, upstream `dispose`.
    pub fn dispose(&self) {
        self.loader.stop();
    }
}

impl super::custom_editor::EmbeddedStatusIndicator for StatusIndicator {
    fn render_in_border(&self, width: usize) -> String {
        Self::render_in_border(self, width)
    }

    fn render_spinner_in_border(&self, width: usize) -> String {
        Self::render_spinner_in_border(self, width)
    }
}

impl Component for StatusIndicator {
    fn render(&self, width: usize) -> Vec<String> {
        self.loader.render(width)
    }

    fn invalidate(&self) {
        self.loader.invalidate();
    }
}

/// Whether the indicator's frames are an explicit empty list, which upstream
/// renders as no spinner at all.
fn indicator_hidden(indicator: Option<&LoaderIndicatorOptions>) -> bool {
    indicator
        .and_then(|options| options.frames.as_ref())
        .is_some_and(Vec::is_empty)
}

/// The leading styled run of `line`, stopping at the first plain space outside
/// an ANSI escape sequence. The loader composes `indicator + " " + message`,
/// so this is the rendered spinner.
fn leading_styled_token(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0x1b && bytes.get(index + 1) == Some(&b'[') {
            index += 2;
            while index < bytes.len() && !(0x40..=0x7e).contains(&bytes[index]) {
                index += 1;
            }
            index += 1;
            continue;
        }
        if bytes[index] == b' ' || bytes[index] == b'\t' {
            break;
        }
        let Some(character) = line[index..].chars().next() else {
            break;
        };
        index += character.len_utf8();
    }
    &line[..index]
}

/// Delegate [`Component`] through a wrapper's shared base.
macro_rules! impl_status_component {
    ($type:ty) => {
        impl Component for $type {
            fn render(&self, width: usize) -> Vec<String> {
                self.inner.render(width)
            }

            fn invalidate(&self) {
                self.inner.invalidate();
            }
        }
    };
}

/// The working indicator, upstream's `WorkingStatusIndicator`.
pub struct WorkingStatusIndicator {
    inner: StatusIndicator,
}

impl std::fmt::Debug for WorkingStatusIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkingStatusIndicator")
            .field("inner", &self.inner)
            .finish()
    }
}

impl std::ops::Deref for WorkingStatusIndicator {
    type Target = StatusIndicator;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl WorkingStatusIndicator {
    /// Build the working indicator, upstream's constructor; `color_fn`
    /// overrides both the spinner and message colors when present.
    #[must_use]
    pub fn new(
        render_request: RenderRequest,
        message: impl Into<String>,
        indicator: Option<LoaderIndicatorOptions>,
        color_fn: Option<ColorFn>,
    ) -> Self {
        let spinner = color_fn
            .clone()
            .unwrap_or_else(|| Arc::new(|text: &str| theme().fg("accent", text)));
        let message_color =
            color_fn.unwrap_or_else(|| Arc::new(|text: &str| theme().fg("muted", text)));
        Self {
            inner: StatusIndicator::new(
                StatusIndicatorKind::Working,
                render_request,
                spinner,
                message_color,
                message,
                indicator,
            ),
        }
    }
}

impl_status_component!(WorkingStatusIndicator);

/// The retry countdown indicator, upstream's `RetryStatusIndicator`.
pub struct RetryStatusIndicator {
    inner: Rc<StatusIndicator>,
    countdown: Rc<RefCell<Option<Rc<CountdownTimer>>>>,
}

impl std::fmt::Debug for RetryStatusIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryStatusIndicator")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl std::ops::Deref for RetryStatusIndicator {
    type Target = StatusIndicator;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

impl RetryStatusIndicator {
    /// Build the retry indicator and arm its countdown, upstream's
    /// constructor.
    #[must_use]
    pub fn new(
        render_request: RenderRequest,
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
    ) -> Self {
        let seconds = i64::try_from(delay_ms.div_ceil(1000)).unwrap_or(i64::MAX);
        let inner = Rc::new(StatusIndicator::new(
            StatusIndicatorKind::Retry,
            Arc::clone(&render_request),
            Arc::new(|text: &str| theme().fg("warning", text)),
            Arc::new(|text: &str| theme().fg("muted", text)),
            retry_message(attempt, max_attempts, seconds),
            None,
        ));

        let countdown: Rc<RefCell<Option<Rc<CountdownTimer>>>> = Rc::new(RefCell::new(None));
        let tick_inner = Rc::clone(&inner);
        let expire_slot = Rc::downgrade(&countdown);
        let request_render: Option<Rc<dyn Fn()>> = Some(Rc::new(move || render_request()));
        let timer = Rc::new(CountdownTimer::new(
            delay_ms,
            request_render,
            Rc::new(move |seconds| {
                tick_inner.set_message(retry_message(attempt, max_attempts, seconds));
            }),
            Rc::new(move || {
                if let Some(slot) = expire_slot.upgrade() {
                    *slot.borrow_mut() = None;
                }
            }),
        ));
        *countdown.borrow_mut() = Some(timer);
        Self { inner, countdown }
    }

    /// The armed countdown, when it has not expired.
    #[must_use]
    pub fn countdown(&self) -> Option<Rc<CountdownTimer>> {
        self.countdown.borrow().as_ref().map(Rc::clone)
    }

    /// Stop the countdown and the loader, upstream `dispose`.
    pub fn dispose(&self) {
        if let Some(countdown) = self.countdown.borrow_mut().take() {
            countdown.dispose();
        }
        self.inner.dispose();
    }
}

impl_status_component!(RetryStatusIndicator);

/// The retry message, upstream's inline `retryMessage`.
fn retry_message(attempt: u32, max_attempts: u32, seconds: i64) -> String {
    format!(
        "Retrying ({attempt}/{max_attempts}) in {seconds}s... ({} to cancel)",
        key_text("app.interrupt")
    )
}

/// Why compaction started, upstream's `CompactionStatusReason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionStatusReason {
    /// The user ran the compact command.
    Manual,
    /// The context crossed the auto-compaction threshold.
    Threshold,
    /// The provider reported a context overflow.
    Overflow,
}

/// The compaction indicator, upstream's `CompactionStatusIndicator`.
pub struct CompactionStatusIndicator {
    inner: StatusIndicator,
}

impl std::fmt::Debug for CompactionStatusIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionStatusIndicator")
            .field("inner", &self.inner)
            .finish()
    }
}

impl std::ops::Deref for CompactionStatusIndicator {
    type Target = StatusIndicator;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl CompactionStatusIndicator {
    /// Build the compaction indicator, upstream's constructor.
    #[must_use]
    pub fn new(render_request: RenderRequest, reason: CompactionStatusReason) -> Self {
        let cancel_hint = format!("({} to cancel)", key_text("app.interrupt"));
        let label = if reason == CompactionStatusReason::Manual {
            format!("Compacting context... {cancel_hint}")
        } else {
            let overflow = if reason == CompactionStatusReason::Overflow {
                "Context overflow detected, "
            } else {
                ""
            };
            format!("{overflow}Auto-compacting... {cancel_hint}")
        };
        Self {
            inner: StatusIndicator::new(
                StatusIndicatorKind::Compaction,
                render_request,
                Arc::new(|text: &str| theme().fg("accent", text)),
                Arc::new(|text: &str| theme().fg("muted", text)),
                label,
                None,
            ),
        }
    }
}

impl_status_component!(CompactionStatusIndicator);

/// The branch-summary indicator, upstream's `BranchSummaryStatusIndicator`.
pub struct BranchSummaryStatusIndicator {
    inner: StatusIndicator,
}

impl std::fmt::Debug for BranchSummaryStatusIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BranchSummaryStatusIndicator")
            .field("inner", &self.inner)
            .finish()
    }
}

impl std::ops::Deref for BranchSummaryStatusIndicator {
    type Target = StatusIndicator;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl BranchSummaryStatusIndicator {
    /// Build the branch-summary indicator, upstream's constructor.
    #[must_use]
    pub fn new(render_request: RenderRequest) -> Self {
        Self {
            inner: StatusIndicator::new(
                StatusIndicatorKind::BranchSummary,
                render_request,
                Arc::new(|text: &str| theme().fg("accent", text)),
                Arc::new(|text: &str| theme().fg("muted", text)),
                format!(
                    "Summarizing branch... ({} to cancel)",
                    key_text("app.interrupt")
                ),
                None,
            ),
        }
    }
}

impl_status_component!(BranchSummaryStatusIndicator);

/// The idle stand-in that keeps the editor at the same height as a mounted
/// indicator, upstream's `IdleStatus`.
#[derive(Debug, Default)]
pub struct IdleStatus;

impl Component for IdleStatus {
    fn render(&self, width: usize) -> Vec<String> {
        let empty_line = " ".repeat(width);
        vec![empty_line.clone(), empty_line]
    }
}
