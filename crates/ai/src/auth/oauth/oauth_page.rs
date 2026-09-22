//! The success/error pages an OAuth callback server returns, ported from
//! `packages/ai/src/auth/oauth/oauth-page.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

/// The pi logo, inlined into the callback pages.
const LOGO_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800" aria-hidden="true"><path fill="#fff" fill-rule="evenodd" d="M165.29 165.29 H517.36 V400 H400 V517.36 H282.65 V634.72 H165.29 Z M282.65 282.65 V400 H400 V282.65 Z"/><path fill="#fff" d="M517.36 400 H634.72 V634.72 H517.36 Z"/></svg>"##;

/// Escape HTML special characters, upstream's `escapeHtml`.
#[must_use]
fn escape_html(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Render the callback page for the given title/heading/message/details.
#[must_use]
fn render_page(options: (&str, &str, &str, Option<&str>)) -> String {
    let (title, heading, message, details) = options;
    let title = escape_html(title);
    let heading = escape_html(heading);
    let message = escape_html(message);
    let details = details.map(escape_html);
    let details_html = details
        .map(|details| format!(r#"<div class="details">{details}</div>"#))
        .unwrap_or_default();
    let logo = LOGO_SVG;

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --text: #fafafa;
      --text-dim: #a1a1aa;
      --page-bg: #09090b;
      --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
    * {{ box-sizing: border-box; }}
    html {{ color-scheme: dark; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: var(--page-bg);
      color: var(--text);
      font-family: var(--font-sans);
      text-align: center;
    }}
    main {{
      width: 100%;
      max-width: 560px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }}
    .logo {{
      width: 72px;
      height: 72px;
      display: block;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 28px;
      line-height: 1.15;
      font-weight: 650;
      color: var(--text);
    }}
    p {{
      margin: 0;
      line-height: 1.7;
      color: var(--text-dim);
      font-size: 15px;
    }}
    .details {{
      margin-top: 16px;
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--text-dim);
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <main>
    <div class="logo">{logo}</div>
    <h1>{heading}</h1>
    <p>{message}</p>
    {details_html}
  </main>
</body>
</html>"#
    )
}

/// The page a completed login returns.
#[must_use]
pub fn oauth_success_html(message: &str) -> String {
    render_page((
        "Authentication successful",
        "Authentication successful",
        message,
        None,
    ))
}

/// The page a failed login returns.
#[must_use]
pub fn oauth_error_html(message: &str, details: Option<&str>) -> String {
    render_page((
        "Authentication failed",
        "Authentication failed",
        message,
        details,
    ))
}
