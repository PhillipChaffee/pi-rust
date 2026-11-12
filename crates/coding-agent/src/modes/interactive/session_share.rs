//! The interactive session-share surface, upstream
//! `src/modes/interactive/session-share.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! The export → Radius upload → GitHub CLI gist ladder landed with the
//! session manager and export slice as [`crate::session_share`]; this module
//! is its upstream-faithful path under `modes::interactive` and re-exports
//! that port unchanged rather than keeping a second copy in sync.
//!
//! Restatement: upstream's `SessionShareContext` swaps a `BorderedLoader`
//! into the editor container, focuses it, and restores the editor on abort
//! or completion. The port restates that editor choreography as the
//! injected [`ShareUserInterface`], [`ShareProcessRunner`], and
//! [`ShareHttpClient`] seams — the loader's abort signal and editor swap
//! belong to the app frame, which is the remaining #131 slice — while the
//! export shape and the Radius/gist ladder stay 1:1.

pub use crate::session_share::{
    GistOutcome, RadiusUploadOutcome, ShareHttpClient, ShareProcessRunner, ShareSessionSource,
    ShareUserInterface, export_session_for_share, share_session,
};
