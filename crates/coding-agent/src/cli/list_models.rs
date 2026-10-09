//! The `--list-models` table, upstream's `src/cli/list-models.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The output contract keeps upstream's column table verbatim. The chalk
//! color on the load warning rides the theme system slice (the crate has
//! no global color vocabulary yet), so the warning prints plain; the
//! `localeCompare` sort restates to byte order, which the ASCII provider
//! and model ids the table sorts agree with.

use std::io::Write;

use tokio_util::sync::CancellationToken;

use pi_ai::auth::types::AuthOptions;
use pi_ai::types::{Modality, Model};

use crate::auth_guidance::format_no_models_available_message;
use crate::model_runtime::ModelRuntime;

use pi_tui::fuzzy::fuzzy_filter;

/// Format a token count the way the table renders it, upstream's `formatTokenCount`.
///
/// Millions and thousands scale with an optional single decimal (`200000` →
/// `"200K"`, `1000000` → `"1M"`, `1500000` → `"1.5M"`); anything smaller
/// prints whole.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the scale math mirrors upstream's JS f64 arithmetic; token counts stay far below the range where the u64-to-f64 mantissa loss could bite, and the `as u64` casts only run when the quotient is integral"
)]
pub fn format_token_count(count: u64) -> String {
    if count >= 1_000_000 {
        let millions = count as f64 / 1_000_000.0;
        return if millions.fract() == 0.0 {
            format!("{}M", millions as u64)
        } else {
            format!("{millions:.1}M")
        };
    }
    if count >= 1_000 {
        let thousands = count as f64 / 1_000.0;
        return if thousands.fract() == 0.0 {
            format!("{}K", thousands as u64)
        } else {
            format!("{thousands:.1}K")
        };
    }
    count.to_string()
}

/// The `yes`/`no` cell a boolean flag renders to, upstream's ternaries.
const fn yes_no(flag: bool) -> &'static str {
    if flag { "yes" } else { "no" }
}

/// Render the model table lines, upstream's `listModels` body.
///
/// The empty-models guidance comes first (before any filter), then the
/// optional fuzzy filter, the provider-then-id sort, and the six-column
/// space-padded grid.
#[must_use]
pub fn render_models_table(models: &[Model], search_pattern: Option<&str>) -> Option<Vec<String>> {
    if models.is_empty() {
        return Some(vec![format_no_models_available_message()]);
    }
    let filtered: Vec<&Model> = search_pattern
        .filter(|pattern| !pattern.is_empty())
        .map_or_else(
            || models.iter().collect(),
            |pattern| {
                fuzzy_filter(models, pattern, |model| {
                    format!("{} {}", model.provider.0, model.id)
                })
            },
        );
    if filtered.is_empty() {
        return search_pattern.map_or_else(
            || Some(vec![format_no_models_available_message()]),
            |search_pattern| Some(vec![format!("No models matching \"{search_pattern}\"")]),
        );
    }

    let mut sorted: Vec<&Model> = filtered;
    sorted.sort_by(|left, right| {
        left.provider
            .0
            .cmp(&right.provider.0)
            .then_with(|| left.id.cmp(&right.id))
    });

    let headers = [
        "provider", "model", "context", "max-out", "thinking", "images",
    ];
    let rows: Vec<[String; 6]> = sorted
        .iter()
        .map(|model| {
            [
                model.provider.0.clone(),
                model.id.clone(),
                format_token_count(model.context_window),
                format_token_count(model.max_tokens),
                yes_no(model.reasoning).to_string(),
                yes_no(model.input.contains(&Modality::Image)).to_string(),
            ]
        })
        .collect();
    let widths: [usize; 6] = std::array::from_fn(|column| {
        headers[column].len().max(
            rows.iter()
                .map(|row| row[column].chars().count())
                .max()
                .unwrap_or(0),
        )
    });
    let render_row = |cells: [String; 6]| -> String {
        cells
            .iter()
            .enumerate()
            .map(|(column, cell)| format!("{cell:<width$}", width = widths[column]))
            .collect::<Vec<_>>()
            .join("  ")
    };

    let header_cells: [String; 6] = std::array::from_fn(|column| headers[column].to_string());
    let mut lines = vec![render_row(header_cells)];
    lines.extend(rows.into_iter().map(render_row));
    Some(lines)
}

/// List available models with optional fuzzy search, upstream's
/// `listModels`: the table prints to the writer, the load warning to the
/// error writer (upstream's stderr).
///
/// # Errors
/// The availability pass's failure, upstream's propagating rejection.
pub async fn list_models_into(
    out: &mut dyn Write,
    err: &mut dyn Write,
    model_runtime: &ModelRuntime,
    search_pattern: Option<&str>,
    signal: Option<CancellationToken>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(load_error) = model_runtime.get_error() {
        let _ = writeln!(err, "Warning: errors loading models.json:\n{load_error}");
    }

    let models = model_runtime
        .get_available(None, Some(&AuthOptions { signal }))
        .await
        .map_err(|failure| -> Box<dyn std::error::Error + Send + Sync> { Box::new(failure) })?;

    let Some(lines) = render_models_table(&models, search_pattern) else {
        return Ok(());
    };
    // The empty-models case renders the guidance message; upstream prints
    // it via console.log, the no-match case likewise.
    for line in lines {
        let _ = writeln!(out, "{line}");
    }
    Ok(())
}

/// [`list_models_into`] over stdout and stderr, upstream's `console`.
///
/// # Errors
/// As [`list_models_into`].
pub async fn list_models(
    model_runtime: &ModelRuntime,
    search_pattern: Option<&str>,
    signal: Option<CancellationToken>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    list_models_into(
        &mut stdout,
        &mut stderr,
        model_runtime,
        search_pattern,
        signal,
    )
    .await
}
