//! The effect admission gate, ported from upstream
//! `src/harness/execution/effect-gate.ts`.
//!
//! Upstream homes the module in `execution/`, but `hooks.ts` consumes the
//! procedure-facing view, so the map's harness-foundations child carries
//! the whole module and the execution child consumes it (recorded on that
//! ticket). Upstream's `AbortRequested` throw restates as the
//! [`GateRejection::AbortRequested`] half of `admit`'s result; its carried
//! cancellation promise restates as a shared, cloneable future. The
//! controller's abort reason carries the fixed message, since chord's Rust
//! port models abort reasons as strings.

use std::sync::{Arc, Mutex};

use pi_chord::context::AbortSignal;

/// The cancellation promise [`GateControl::begin_abort`] carries, upstream's
/// `Promise<void>`: the abort path awaits it while aborted effects settle.
/// A watch receiver restates the promise's one-shot settlement — every
/// clone resolves when the abort work sends or the sender drops — and
/// stays cloneable and awaitable where a boxed future is single-consumer.
pub type Cancellation = tokio::sync::watch::Receiver<()>;

/// The expected internal control flow when cancellation wins effect
/// admission, upstream's `AbortRequested` error class.
pub struct AbortRequested {
    /// The cancellation future the abort path awaits.
    pub cancellation: Cancellation,
}

impl std::fmt::Debug for AbortRequested {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AbortRequested").finish_non_exhaustive()
    }
}

impl std::fmt::Display for AbortRequested {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Abort requested")
    }
}

impl std::error::Error for AbortRequested {}

/// What [`Gate::admit`] reports when admission is refused.
#[derive(Debug)]
pub enum GateRejection {
    /// Cancellation won admission; the carried future settles when the
    /// cancellation work completes.
    AbortRequested(AbortRequested),
    /// The gate closed with its error.
    Closed(GateClosedError),
}

/// The error a closed gate carries, restated as a string-backed error
/// (upstream passes an arbitrary `Error` through `close`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateClosedError(pub String);

impl std::fmt::Display for GateClosedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GateClosedError {}

/// Procedure-facing synchronous admission capability for one drive pass,
/// upstream's `Gate`.
#[derive(Clone)]
pub struct Gate {
    signal: AbortSignal,
    shared: Arc<GateShared>,
}

struct GateShared {
    state: Mutex<GateState>,
    controller: pi_chord::context::AbortController,
}

enum GateState {
    Open,
    Aborting {
        cancellation: Cancellation,
    },
    Closed {
        error: String,
    },
}

/// Owner-facing lifecycle controls for one drive pass, upstream's
/// `GateControl`.
#[derive(Clone)]
pub struct GateControl {
    shared: Arc<GateShared>,
}

impl std::fmt::Debug for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gate").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for GateControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GateControl").finish_non_exhaustive()
    }
}

impl Gate {
    /// The gate's cancellation signal, upstream's `Gate.signal`.
    #[must_use]
    pub fn signal(&self) -> &AbortSignal {
        &self.signal
    }

    /// Admits one effect, upstream's `admit<T>(invoke)`.
    ///
    /// # Errors
    /// Returns [`GateRejection::AbortRequested`] when cancellation won the
    /// admission race and [`GateRejection::Closed`] when the gate closed.
    pub fn admit<T>(&self, invoke: impl FnOnce() -> T) -> Result<T, GateRejection> {
        let cancellation = match &*self.shared.state.lock().expect("gate state lock") {
            GateState::Aborting { cancellation } => Some(cancellation.clone()),
            GateState::Closed { error } => {
                return Err(GateRejection::Closed(GateClosedError(error.clone())));
            }
            GateState::Open => None,
        };
        if let Some(cancellation) = cancellation {
            return Err(GateRejection::AbortRequested(AbortRequested { cancellation }));
        }
        Ok(invoke())
    }
}

impl GateControl {
    /// Records the abort's cancellation future, ignoring later calls,
    /// upstream's `beginAbort`.
    pub fn begin_abort(&self, cancellation: Cancellation) {
        let mut state = self.shared.state.lock().expect("gate state lock");
        if !matches!(*state, GateState::Open) {
            return;
        }
        *state = GateState::Aborting { cancellation };
    }

    /// Fires the gate's signal once aborting, upstream's `signalAbort`.
    pub fn signal_abort(&self) {
        {
            let state = self.shared.state.lock().expect("gate state lock");
            if !matches!(*state, GateState::Aborting { .. })
                || self.shared.controller.signal().aborted()
            {
                return;
            }
        }
        self.shared.controller.abort("Abort requested");
    }

    /// Closes the gate with its error, aborting the signal when still
    /// live, upstream's `close`.
    pub fn close(&self, error: String) {
        {
            let mut state = self.shared.state.lock().expect("gate state lock");
            if matches!(*state, GateState::Closed { .. }) {
                return;
            }
            *state = GateState::Closed {
                error: error.clone(),
            };
        }
        if !self.shared.controller.signal().aborted() {
            self.shared.controller.abort(error);
        }
    }
}

/// Creates separate procedure-facing and owner-facing views of one effect
/// gate, upstream's `createGate`.
#[must_use]
pub fn create_gate() -> (Gate, GateControl) {
    // chord mints fresh controllers only through `with_cancel`; the derived
    // context is dropped and the controller's signal carries the gate.
    let (_, controller) = pi_chord::context::with_cancel(&pi_chord::context::background_context());
    let shared = Arc::new(GateShared {
        state: Mutex::new(GateState::Open),
        controller,
    });
    let signal = shared.controller.signal().clone();
    (
        Gate {
            signal,
            shared: Arc::clone(&shared),
        },
        GateControl { shared },
    )
}

#[cfg(test)]
mod tests;