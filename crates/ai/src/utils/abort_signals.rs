//! Combined cancellation tokens, ported from
//! `packages/ai/src/utils/abort-signals.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`combine_abort_signals`] folds several optional tokens into one token
//! that cancels when any of them does, with a cleanup that drops the linking
//! tasks so the combined token stops observing its sources.

use tokio_util::sync::CancellationToken;

/// A combined signal plus the cleanup that stops it from observing its
/// sources, upstream's `CombinedAbortSignal`.
pub struct CombinedAbortSignal {
    /// The combined token; [`None`] when no source signal was active.
    pub signal: Option<CancellationToken>,
    /// Stop observing the source signals. After cleanup, cancelling a source
    /// no longer cancels the combined token.
    pub cleanup: Box<dyn FnOnce() + Send>,
}

impl std::fmt::Debug for CombinedAbortSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedAbortSignal")
            .field("signal", &self.signal)
            .finish_non_exhaustive()
    }
}

/// Combine optional cancellation tokens into one token that cancels when any active source cancels.
///
/// With no active sources there is no combined token; with one active source it is passed through
/// unchanged.
#[must_use]
pub fn combine_abort_signals(signals: &[Option<&CancellationToken>]) -> CombinedAbortSignal {
    let active: Vec<&CancellationToken> = signals.iter().filter_map(|signal| *signal).collect();
    match active.len() {
        0 => CombinedAbortSignal {
            signal: None,
            cleanup: Box::new(|| {}),
        },
        1 => CombinedAbortSignal {
            signal: Some(active[0].clone()),
            cleanup: Box::new(|| {}),
        },
        _ => {
            let combined = CancellationToken::new();
            let mut linkers = Vec::new();
            for source in active {
                link_source_to_target(source.clone(), combined.clone(), &mut linkers);
            }
            CombinedAbortSignal {
                signal: Some(combined),
                cleanup: Box::new(move || {
                    for handle in linkers {
                        handle.abort();
                    }
                }),
            }
        }
    }
}

/// Link one source token into the combined target; a source that is already
/// cancelled cancels the target immediately when the link task first polls.
fn link_source_to_target(
    source: CancellationToken,
    target: CancellationToken,
    handles: &mut Vec<tokio::task::JoinHandle<()>>,
) {
    handles.push(tokio::spawn(async move {
        source.cancelled().await;
        target.cancel();
    }));
}
