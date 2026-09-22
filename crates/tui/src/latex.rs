//! The LaTeX-to-Unicode-math rendering seam for the markdown component,
//! ported from `packages/tui/src/latex.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The renderer itself lands with the latex port ticket
//! ([#50](https://github.com/PhillipChaffee/pi-rust/issues/50)); this module
//! holds only the seam the markdown component renders through, upstream
//! `renderLatex`. The stub answers [`render_latex`] with `None`, which is
//! markdown's documented degradation: an expression the renderer declines —
//! or, until #50, every expression — renders as its raw LaTeX source.

/// Render a LaTeX expression to terminal-safe Unicode text, upstream
/// `renderLatex(latex, options?)`.
///
/// `display` selects display-style layout (upstream `{ display: true }`),
/// where fractions stack over rule lines and limits sit beneath operators;
/// inline mode renders linearly. The answer is `None` when the expression
/// contains syntax the renderer does not support, upstream's `null`, and
/// callers fall back to the raw source text.
///
/// The body is the #50 seam: the markdown slice's goldens that assert
/// rendered math restate to raw passthrough until the renderer lands.
#[must_use]
pub const fn render_latex(_latex: &str, _display: bool) -> Option<String> {
    None
}
