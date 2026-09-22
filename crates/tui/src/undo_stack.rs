//! Generic undo stack with clone-on-push semantics, ported from
//! `packages/tui/src/undo-stack.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! Stores clones of state snapshots. Popped snapshots are returned directly
//! since they are already detached from the live state.

/// LIFO snapshot stack, upstream `UndoStack<S>`.
#[derive(Debug, Default)]
pub struct UndoStack<S> {
    stack: Vec<S>,
}

impl<S> UndoStack<S> {
    /// An empty stack.
    #[must_use]
    pub const fn new() -> Self {
        Self { stack: Vec::new() }
    }
}

impl<S: Clone> UndoStack<S> {
    /// Push a clone of the given state onto the stack, upstream `push`.
    pub fn push(&mut self, state: &S) {
        self.stack.push(state.clone());
    }

    /// Pop and return the most recent snapshot, or `None` if empty, upstream
    /// `pop`.
    pub fn pop(&mut self) -> Option<S> {
        self.stack.pop()
    }

    /// Remove all snapshots, upstream `clear`.
    pub fn clear(&mut self) {
        self.stack.clear();
    }

    /// Snapshot count, upstream `length`.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.stack.len()
    }

    /// Whether the stack holds no snapshots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}
