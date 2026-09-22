//! Ring buffer for Emacs-style kill/yank operations, ported from
//! `packages/tui/src/kill-ring.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! Tracks killed (deleted) text entries. Consecutive kills can accumulate
//! into a single entry. Supports yank (paste most recent) and yank-pop
//! (cycle through older entries).

/// Ring buffer for the editor's kill/yank operations, upstream `KillRing`.
#[derive(Debug, Default)]
pub struct KillRing {
    ring: Vec<String>,
}

/// Options for [`KillRing::push`], upstream `opts`.
#[derive(Debug, Clone, Copy, Default)]
pub struct KillRingPushOptions {
    /// If accumulating, prepend (backward deletion) or append (forward
    /// deletion), upstream `opts.prepend`.
    pub prepend: bool,
    /// Merge with the most recent entry instead of creating a new one,
    /// upstream `opts.accumulate`.
    pub accumulate: bool,
}

impl KillRing {
    /// Add text to the kill ring; empty text is dropped, upstream `push`.
    pub fn push(&mut self, text: &str, opts: KillRingPushOptions) {
        if text.is_empty() {
            return;
        }

        if opts.accumulate && !self.ring.is_empty() {
            let last = self.ring.pop().unwrap_or_default();
            let merged = if opts.prepend {
                format!("{text}{last}")
            } else {
                format!("{last}{text}")
            };
            self.ring.push(merged);
        } else {
            self.ring.push(text.to_string());
        }
    }

    /// Most recent entry without modifying the ring, upstream `peek`.
    #[must_use]
    pub fn peek(&self) -> Option<&str> {
        self.ring.last().map(String::as_str)
    }

    /// Move the last entry to the front for yank-pop cycling, upstream
    /// `rotate`.
    pub fn rotate(&mut self) {
        if self.ring.len() > 1
            && let Some(last) = self.ring.pop()
        {
            self.ring.insert(0, last);
        }
    }

    /// Entry count, upstream `length`.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.ring.len()
    }

    /// Whether the ring holds no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}
