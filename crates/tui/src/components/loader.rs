//! Loader component, ported from `packages/tui/src/components/loader.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#42).
//!
//! A loader that updates with an optional spinning animation.
//!
//! Restatements against upstream:
//!
//! - `class Loader extends Text` becomes composition: a [`Loader`] embeds an
//!   `Arc`-shared [`Text`] that its animation thread writes through
//!   [`Text::set_text`]; the [`crate::tui::Component`] methods delegate.
//! - Upstream's `setInterval` becomes a worker thread per animation with the
//!   terminal-port-style stop channel: `recv_timeout(interval)` is the tick
//!   and disconnecting the sender stops it. A set already in flight when the
//!   indicator is swapped may land one extra frame; upstream's
//!   `clearInterval` has the same best-effort timing.
//! - Upstream's `ui: TUI` narrows to a [`crate::tui::RenderRequest`] closure
//!   (the one method the loader calls); see [`crate::tui`] docs.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::components::{ColorFn, Text};
use crate::tui::{Component, RenderRequest};

/// Animation frames. Use an empty array to hide the indicator, upstream
/// `LoaderIndicatorOptions.frames`.
///
/// Upstream defaults: the braille spinner at 80 ms per frame.
const DEFAULT_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const DEFAULT_INTERVAL_MS: u64 = 80;

/// Loader indicator options, upstream `LoaderIndicatorOptions`.
#[derive(Debug, Clone, Default)]
pub struct LoaderIndicatorOptions {
    /// Animation frames. `None` keeps the default spinner; an empty list
    /// hides the indicator.
    pub frames: Option<Vec<String>>,
    /// Frame interval in milliseconds for animated indicators. Zero or
    /// `None` selects the default.
    pub interval_ms: Option<u64>,
}

/// The loader state shared with the animation thread, upstream's private
/// fields read by the `setInterval` closure.
struct LoaderCore {
    frames: Vec<String>,
    interval_ms: u64,
    current_frame: usize,
    render_indicator_verbatim: bool,
    spinner_color_fn: ColorFn,
    message_color_fn: ColorFn,
    message: String,
}

/// The spawned animation worker; dropping it disconnects the stop channel
/// and the thread exits on its next wake.
struct Animation {
    /// Held purely so its drop disconnects the worker's stop channel.
    #[expect(
        dead_code,
        reason = "the sender's drop is the stop signal; nothing reads it"
    )]
    stop: std::sync::mpsc::Sender<()>,
}

impl std::fmt::Debug for Loader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loader").finish_non_exhaustive()
    }
}

/// Loader component that updates with an optional spinning animation.
pub struct Loader {
    text: Arc<Text>,
    core: Arc<Mutex<LoaderCore>>,
    render_request: RenderRequest,
    animation: Mutex<Option<Animation>>,
}

impl Loader {
    /// Upstream's `new Loader(ui, spinnerColorFn, messageColorFn, message,
    /// indicator)`; `message` has the upstream default `"Loading..."`.
    #[must_use]
    pub fn new(
        render_request: RenderRequest,
        spinner_color_fn: ColorFn,
        message_color_fn: ColorFn,
        message: impl Into<String>,
        indicator: Option<LoaderIndicatorOptions>,
    ) -> Self {
        let text = Arc::new(Text::with_padding("", 1, 0));
        let core = Arc::new(Mutex::new(LoaderCore {
            frames: DEFAULT_FRAMES
                .iter()
                .map(|frame| (*frame).to_string())
                .collect(),
            interval_ms: DEFAULT_INTERVAL_MS,
            current_frame: 0,
            render_indicator_verbatim: false,
            spinner_color_fn,
            message_color_fn,
            message: message.into(),
        }));
        let loader = Self {
            text,
            core,
            render_request,
            animation: Mutex::new(None),
        };
        loader.set_indicator(indicator);
        loader
    }

    /// Upstream `start`: refresh the display and arm the animation.
    pub fn start(&self) {
        self.update_display();
        self.restart_animation();
    }

    /// Upstream `stop`: disarm the animation worker, if any.
    pub fn stop(&self) {
        self.animation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }

    /// Upstream `setMessage`.
    pub fn set_message(&self, message: impl Into<String>) {
        self.lock_core().message = message.into();
        self.update_display();
    }

    /// Upstream `setIndicator`.
    pub fn set_indicator(&self, indicator: Option<LoaderIndicatorOptions>) {
        let interval_ms = indicator
            .as_ref()
            .and_then(|options| options.interval_ms)
            .filter(|interval| *interval > 0)
            .unwrap_or(DEFAULT_INTERVAL_MS);
        {
            let mut core = self.lock_core();
            core.render_indicator_verbatim = indicator.is_some();
            core.frames = match indicator {
                Some(options) => options.frames.unwrap_or_else(default_frames),
                None => default_frames(),
            };
            core.interval_ms = interval_ms;
            core.current_frame = 0;
        }
        self.start();
    }

    /// Upstream `getRenderedIndicator` composed into the full display
    /// string, the body of upstream's `updateDisplay`.
    fn compose_display(core: &LoaderCore) -> String {
        let frame = core
            .frames
            .get(core.current_frame)
            .map_or("", String::as_str);
        let spinner = &core.spinner_color_fn;
        let rendered = if core.render_indicator_verbatim {
            frame.to_string()
        } else {
            spinner(frame)
        };
        let indicator = if rendered.is_empty() {
            String::new()
        } else {
            format!("{rendered} ")
        };
        let message_fn = &core.message_color_fn;
        format!("{indicator}{}", message_fn(&core.message))
    }

    fn update_display(&self) {
        let display = Self::compose_display(&self.lock_core());
        self.text.set_text(&display);
        (self.render_request)();
    }

    fn restart_animation(&self) {
        self.stop();
        let core = Arc::clone(&self.core);
        let frames_len = self.lock_core().frames.len();
        if frames_len <= 1 {
            return;
        }

        let text = Arc::clone(&self.text);
        let render_request = Arc::clone(&self.render_request);
        let (stop, stop_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            loop {
                let interval = {
                    let core = core.lock().unwrap_or_else(PoisonError::into_inner);
                    if core.frames.len() <= 1 {
                        return;
                    }
                    Duration::from_millis(core.interval_ms)
                };
                match stop_rx.recv_timeout(interval) {
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        let display = {
                            let mut core = core.lock().unwrap_or_else(PoisonError::into_inner);
                            core.current_frame = (core.current_frame + 1) % core.frames.len();
                            let display = Self::compose_display(&core);
                            drop(core);
                            display
                        };
                        text.set_text(&display);
                        (render_request)();
                    }
                }
            }
        });
        *self
            .animation
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Animation { stop });
    }

    fn lock_core(&self) -> MutexGuard<'_, LoaderCore> {
        self.core.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn default_frames() -> Vec<String> {
    DEFAULT_FRAMES
        .iter()
        .map(|frame| (*frame).to_string())
        .collect()
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Component for Loader {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = vec![String::new()];
        lines.extend(self.text.render(width));
        lines
    }

    fn invalidate(&self) {
        self.text.invalidate();
        self.update_display();
    }
}
