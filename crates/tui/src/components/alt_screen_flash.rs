//! `AltScreenFlashContainer`, ported from
//! `packages/tui/src/components/alt-screen-flash.ts` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#46).
//!
//! Transient messages the alternate-screen renderer composites at the right
//! edge of the viewport; each message hides itself after its duration.
//!
//! Restatement: each entry's `setTimeout` timer becomes a worker thread with
//! the terminal-port-style stop channel — the single-shot `recv_timeout` is
//! the timer, and dropping the sender stops it. The worker cannot touch the
//! owner-thread entry list, so expiry delivers an id through a channel and
//! raises the render demand; the renderer drains the channel on the owner
//! thread at the top of its frame, exactly upstream's timer callback running
//! on the event loop before the next paint.

use std::cell::{Cell, RefCell};
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::tui::RenderRequest;
use crate::utils::truncate_to_width;

const DEFAULT_DURATION_MS: u64 = 1000;

/// One armed flash message, upstream `FlashEntry`.
struct FlashEntry {
    id: u64,
    message: String,
    /// Held purely so its drop is the worker's stop signal, upstream's
    /// `clearTimeout`.
    #[expect(
        dead_code,
        reason = "the sender's drop is the stop signal; nothing reads it"
    )]
    stop: Sender<()>,
}

/// Transient messages composited by the alternate-screen renderer, upstream
/// `class AltScreenFlashContainer`.
pub struct AltScreenFlashContainer {
    entries: RefCell<Vec<FlashEntry>>,
    next_id: Cell<u64>,
    /// The render request upstream captured at construction; the port fills
    /// it when the renderer receives its base.
    request_render: RefCell<Option<RenderRequest>>,
    /// The expiry-signal sender the armed workers clone; kept so it can be
    /// handed out per arm.
    expired_tx: RefCell<Option<Sender<u64>>>,
    expired: RefCell<Receiver<u64>>,
}

impl std::fmt::Debug for AltScreenFlashContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AltScreenFlashContainer")
            .field("entries", &self.entries.borrow().len())
            .finish_non_exhaustive()
    }
}

impl AltScreenFlashContainer {
    /// Upstream's constructor: `new AltScreenFlashContainer(() =>
    /// this.requestRender())`.
    #[must_use]
    pub fn new() -> Self {
        let (expired_tx, expired_rx) = channel();
        Self {
            entries: RefCell::new(Vec::new()),
            next_id: Cell::new(0),
            request_render: RefCell::new(None),
            expired_tx: RefCell::new(Some(expired_tx)),
            expired: RefCell::new(expired_rx),
        }
    }

    /// Store the render request, upstream's constructor capture.
    pub fn set_request_render(&self, request_render: RenderRequest) {
        *self.request_render.borrow_mut() = Some(request_render);
    }

    /// The expiry-signal sender the armed workers send through.
    fn expired_sender(&self) -> Sender<u64> {
        self.expired_tx
            .borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                // The container was never attached to a renderer: workers get
                // a sender nothing reads, so their expiry is a no-op.
                let (tx, rx) = channel();
                std::mem::forget(rx);
                tx
            })
    }

    fn request_render(&self) {
        if let Some(request_render) = self.request_render.borrow().as_ref() {
            request_render();
        }
    }

    /// Show a message for `duration_ms`, upstream `flash`: `None` rides the
    /// one-second default.
    pub fn flash(&self, message: &str, duration_ms: Option<u64>) {
        let id = self.next_id.replace(self.next_id.get() + 1);
        let duration = duration_ms.unwrap_or(DEFAULT_DURATION_MS);
        let expired_tx = self.expired_sender();
        let request_render = self.request_render.borrow().clone();
        let (stop, stop_rx) = channel();
        std::thread::spawn(move || {
            match stop_rx.recv_timeout(std::time::Duration::from_millis(duration)) {
                Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let _ = expired_tx.send(id);
                    if let Some(request_render) = request_render.as_ref() {
                        request_render();
                    }
                }
            }
        });
        self.entries.borrow_mut().push(FlashEntry {
            id,
            message: message.to_string(),
            stop,
        });
        self.request_render();
    }

    /// Remove the entries whose workers reported expiry, upstream's timer
    /// callback splicing the entry out. The renderer calls this on the owner
    /// thread at the top of its frame.
    pub fn drain_expired(&self) {
        let mut expired = Vec::new();
        while let Ok(id) = self.expired.borrow_mut().try_recv() {
            expired.push(id);
        }
        if expired.is_empty() {
            return;
        }
        self.entries
            .borrow_mut()
            .retain(|entry| !expired.contains(&entry.id));
        self.request_render();
    }

    /// Clear every entry and stop its worker, upstream `dispose`.
    pub fn dispose(&self) {
        self.entries.borrow_mut().clear();
    }

    /// The styled message lines, upstream `AltScreenFlashContainer.render`.
    #[must_use]
    pub fn render(&self, width: usize) -> Vec<String> {
        self.entries
            .borrow()
            .iter()
            .map(|entry| {
                let message = truncate_to_width(&format!(" {} ", entry.message), width, "", false);
                format!("\x1b[7m{message}\x1b[27m")
            })
            .collect()
    }
}

impl Default for AltScreenFlashContainer {
    fn default() -> Self {
        Self::new()
    }
}
