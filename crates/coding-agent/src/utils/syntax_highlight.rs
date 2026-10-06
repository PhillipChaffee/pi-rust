//! Syntax highlighting, upstream's `src/utils/syntax-highlight.ts`.
//!
//! highlight.js restates onto the `syntect` engine — the Sublime syntax
//! set, with two-face's extended syntax pack carrying the eager languages'
//! full coverage (`nix` among them, which the Sublime default set lacks)
//! and the pure-Rust `fancy-regex` backend standing in for oniguruma. The
//! recorded restatement for this module: highlight.js's HTML output and
//! its `hljs-*` scopes do not exist here; [`highlight`] walks syntect's
//! scope stack per token and maps each scope through the same prefix rules
//! [`render_highlighted_html`] applies to class names, so a theme stays a
//! map keyed by hljs vocabulary (`keyword`, `number`, `string`, …) and the
//! theme module ports unchanged. The eager/deferred load split exists
//! upstream because highlight.js lazy-loads its language index; syntect
//! loads its whole set at init, so [`load_all_highlight_languages`] is the
//! no-op the restatement makes it and every language it ships answers
//! [`supports_language`] from the start.

use std::collections::HashMap;
use std::sync::OnceLock;

use syntect::easy::ScopeRangeIterator;
use syntect::parsing::{ParseState, ScopeStack, SyntaxSet};

use super::html::decode_html_entity_at;

/// The scope vocabulary the theme keys, upstream's `hljs-` class prefix
/// restated to the bare names.
const HIGHLIGHT_CLASS_PREFIX: &str = "hljs-";

const SPAN_CLOSE: &str = "</span>";

fn syntax_set() -> &'static SyntaxSet {
    static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
    SYNTAX_SET.get_or_init(two_face::syntax::extra_newlines)
}

/// Load the highlighter's full language set, upstream's
/// `loadAllHighlightLanguages`.
///
/// The restated engine has no deferred index to load — the whole syntax
/// set initializes with the process — so this settles immediately. The
/// async shape stays, upstream's awaited surface its callers share.
#[expect(
    clippy::unused_async,
    reason = "upstream's awaited load surface; the port keeps the shape its callers await"
)]
pub async fn load_all_highlight_languages() {
    let _initialized = syntax_set();
}

/// Whether the highlighter carries a language, upstream's `supportsLanguage`.
#[must_use]
pub fn supports_language(name: &str) -> bool {
    language_for_name(name).is_some()
}

/// The highlight.js language names whose spellings the engine does not
/// index under either the syntax names or the file extensions; the alias
/// carries the engine's own spelling.
const LANGUAGE_ALIASES: [(&str, &str); 1] = [("csharp", "cs")];

/// The syntax a highlight.js language name picks, over the aliases, the
/// syntax names, and the file extensions syntect indexes.
fn language_for_name(name: &str) -> Option<&'static syntect::parsing::SyntaxReference> {
    let aliased = LANGUAGE_ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map(|(_, engine)| *engine);
    let set = syntax_set();
    let candidates = aliased.into_iter().chain(std::iter::once(name));
    for candidate in candidates {
        if let Some(syntax) = set.find_syntax_by_token(candidate).into_iter().next() {
            return Some(syntax);
        }
        if let Some(syntax) = set.find_syntax_by_extension(candidate).into_iter().next() {
            return Some(syntax);
        }
    }
    None
}

/// A text formatter a theme maps scopes to, upstream's
/// `HighlightFormatter` — the arc'd boxed closure the shared themes ride.
pub type HighlightFormatter = std::sync::Arc<dyn Fn(&str) -> String>;

/// The theme's scope map, upstream's `HighlightTheme`.
///
/// Bare hljs scope names to formatters, with `default` as the unmapped
/// fallback. The formatter closures carry no `Debug`, so the theme reports
/// its keys.
pub struct HighlightTheme<'a> {
    map: HashMap<&'a str, HighlightFormatter>,
    default: Option<HighlightFormatter>,
}

impl std::fmt::Debug for HighlightTheme<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HighlightTheme")
            .field("scopes", &self.map.keys().collect::<Vec<_>>())
            .field("default", &self.default.is_some())
            .finish()
    }
}

impl<'a> HighlightTheme<'a> {
    /// Build a theme from the scope-name map, `"default"` routing to the
    /// fallback formatter.
    #[must_use]
    pub fn new(map: HashMap<&'a str, HighlightFormatter>) -> Self {
        let default = map.get("default").cloned();
        Self { map, default }
    }
}

/// The highlight options, upstream's `HighlightOptions`.
#[derive(Debug)]
pub struct HighlightOptions<'a> {
    /// The language name; absent picks the auto-detection.
    pub language: Option<&'a str>,
    /// highlight.js's illegals tolerance; the restated engine has no
    /// illegal-sequence failure mode, so the flag changes nothing here.
    pub ignore_illegals: bool,
    /// The subset the auto-detection scores, upstream's `languageSubset`.
    pub language_subset: Vec<&'a str>,
    /// The scope map the output formats through.
    pub theme: Option<HighlightTheme<'a>>,
}

/// Get the scope name a `hljs-` class attribute carries, upstream's
/// `getScopeFromSpanTag` — the first `hljs-`-prefixed class in the value.
fn get_scope_from_span_tag(tag: &str) -> Option<String> {
    let class_start = tag.find("class")?;
    let after = &tag[class_start + 5..];
    let after = after.trim_start();
    let after = after.strip_prefix('=')?.trim_start();
    let class_value = match after.as_bytes().first() {
        Some(b'"') => after[1..].split('"').next()?,
        Some(b'\'') => after[1..].split('\'').next()?,
        _ => return None,
    };

    for class_name in class_value.split_whitespace() {
        if let Some(scope) = class_name.strip_prefix(HIGHLIGHT_CLASS_PREFIX) {
            return Some(scope.to_string());
        }
    }

    None
}

/// The formatter a scope maps to, upstream's `getScopeFormatter`: the exact
/// key, else the scope's prefix before the first `.` or `-`.
fn get_scope_formatter(scope: &str, theme: &HighlightTheme<'_>) -> Option<HighlightFormatter> {
    if let Some(exact) = theme.map.get(scope) {
        return Some(std::sync::Arc::clone(exact));
    }

    if let Some(dot_index) = scope.find('.')
        && let Some(prefix_formatter) = theme.map.get(&scope[..dot_index])
    {
        return Some(std::sync::Arc::clone(prefix_formatter));
    }

    if let Some(dash_index) = scope.find('-')
        && let Some(prefix_formatter) = theme.map.get(&scope[..dash_index])
    {
        return Some(std::sync::Arc::clone(prefix_formatter));
    }

    None
}

/// The formatter of the innermost mapped scope, upstream's
/// `getActiveFormatter`, scanning the stack innermost-first.
fn get_active_formatter(
    scopes: &[Option<String>],
    theme: &HighlightTheme<'_>,
) -> Option<HighlightFormatter> {
    for scope in scopes.iter().rev() {
        let Some(scope) = scope else { continue };
        if let Some(formatter) = get_scope_formatter(scope, theme) {
            return Some(formatter);
        }
    }
    theme.default.clone()
}

fn is_span_open_tag_start(html: &str, index: usize) -> bool {
    if !html[index..].starts_with("<span") {
        return false;
    }
    matches!(
        html.as_bytes().get(index + "<span".len()),
        Some(b'>' | b' ' | b'\t' | b'\n' | b'\r')
    )
}

/// Render highlighter HTML through a theme, upstream's
/// `renderHighlightedHtml`.
///
/// The span walker its renderer tests pin: class scopes push and pop,
/// entities decode, and text formats through the active scope's formatter.
///
/// # Panics
/// Never: the byte-index scan re-derives each character's length, so the
/// `expect` guard on the char read only fires on a non-char boundary,
/// which a `&str` cannot contain.
#[must_use]
pub fn render_highlighted_html(html: &str, theme: &HighlightTheme) -> String {
    let mut output = String::new();
    let mut text_buffer = String::new();
    let mut scopes: Vec<Option<String>> = Vec::new();

    let flush_text = |output: &mut String, text_buffer: &mut String, scopes: &[Option<String>]| {
        if text_buffer.is_empty() {
            return;
        }
        if let Some(formatter) = get_active_formatter(scopes, theme) {
            output.push_str(&formatter(text_buffer));
        } else {
            output.push_str(text_buffer);
        }
        text_buffer.clear();
    };

    let mut index = 0usize;
    let bytes = html.as_bytes();
    while index < html.len() {
        if is_span_open_tag_start(html, index)
            && let Some(tag_end_index) = html[index + 5..].find('>').map(|at| at + index + 5)
        {
            flush_text(&mut output, &mut text_buffer, &scopes);
            let tag = &html[index..=tag_end_index];
            let scope = get_scope_from_span_tag(tag);
            scopes.push(scope);
            index = tag_end_index + 1;
            continue;
        }

        if html[index..].starts_with(SPAN_CLOSE) {
            flush_text(&mut output, &mut text_buffer, &scopes);
            if !scopes.is_empty() {
                scopes.pop();
            }
            index += SPAN_CLOSE.len();
            continue;
        }

        if bytes.get(index) == Some(&b'&')
            && let Some(decoded) = decode_html_entity_at(html, index)
        {
            text_buffer.push_str(&decoded.text);
            index += decoded.length;
            continue;
        }

        // The scan walks byte indices a `&str` bounds; the next char
        // always exists inside the slice.
        let Some(ch) = html[index..].chars().next() else {
            break;
        };
        text_buffer.push(ch);
        index += ch.len_utf8();
    }

    flush_text(&mut output, &mut text_buffer, &scopes);
    output
}

/// The syntect scope → hljs scope translation, the restatement's bridge:
/// the theme keys stay hljs vocabulary (`keyword`, `number`, `string`, …),
/// so the theme module ports unchanged, and each engine scope resolves
/// through its dotted prefixes, most specific first.
fn translate_scope(scope: &str) -> Option<&'static str> {
    match scope {
        "keyword.control"
        | "keyword.other"
        | "keyword.declaration"
        | "keyword.include"
        | "keyword.import"
        | "keyword"
        | "storage.type"
        | "storage.modifier"
        | "storage" => Some("keyword"),
        "keyword.operator" => Some("operator"),
        "constant.numeric" => Some("number"),
        "constant.language" => Some("literal"),
        "constant.character.escape" | "escape" => Some("char.escape"),
        "string" => Some("string"),
        "comment" => Some("comment"),
        "entity.name.function" | "support.function" => Some("title.function"),
        "entity.name.type" | "entity.name.class" | "entity.name.struct" | "entity.name.enum"
        | "support.class" | "support.type" => Some("title.class"),
        "entity.name.tag" => Some("tag"),
        "entity.name.section" => Some("section"),
        "entity.other.attribute-name" => Some("attribute"),
        "entity.name.constant" | "support.constant" => Some("variable.constant"),
        "variable.language" | "variable.other" | "variable.parameter" | "variable" => {
            Some("variable")
        }
        "punctuation" => Some("punctuation"),
        "meta" => Some("meta"),
        "markup.inserted" | "markup.changed" => Some("addition"),
        "markup.deleted" => Some("deletion"),
        "markup.heading" => Some("title"),
        "markup.bold" => Some("strong"),
        "markup.italic" => Some("emphasis"),
        "markup.quote" => Some("quote"),
        "markup.list" => Some("bullet"),
        "markup.link" => Some("link"),
        _ => None,
    }
}

/// The formatter of the innermost mapped scope on a syntect stack: each
/// scope walks its dotted prefixes (most specific first), the translation
/// table first and the raw engine scope second, so themes keyed with
/// engine vocabulary keep working.
fn active_scope_formatter(
    stack: &ScopeStack,
    theme: &HighlightTheme<'_>,
) -> Option<HighlightFormatter> {
    for scope in stack.as_slice().iter().rev() {
        let name = scope.to_string();
        let mut parts: Vec<&str> = name.split('.').collect();
        while !parts.is_empty() {
            let candidate = parts.join(".");
            if let Some(hljs) = translate_scope(&candidate)
                && let Some(formatter) = get_scope_formatter(hljs, theme)
            {
                return Some(formatter);
            }
            if let Some(formatter) = get_scope_formatter(&candidate, theme) {
                return Some(formatter);
            }
            parts.pop();
        }
    }
    theme.default.clone()
}

/// Highlight code through the engine and a theme, upstream's `highlight`.
///
/// The `language`-absent branch restates `highlightAuto` on the engine's
/// first-line detection; the engine has no `ignoreIllegals` failure mode
/// for the flag to soften.
#[must_use]
pub fn highlight(code: &str, options: HighlightOptions) -> String {
    let theme = options
        .theme
        .unwrap_or_else(|| HighlightTheme::new(HashMap::new()));
    let syntax = options.language.map_or_else(
        || syntax_set().find_syntax_by_first_line(code),
        language_for_name,
    );
    let Some(syntax) = syntax else {
        return code.to_string();
    };

    let mut parse_state = ParseState::new(syntax);
    let mut scope_stack = ScopeStack::new();
    let mut output = String::new();
    // The scope-region iterator pairs every region with the op that scopes
    // it, so each region formats through the stack after its op applies —
    // upstream's style-per-token shape with scopes instead of styles.
    let Ok(ops) = parse_state.parse_line(code, syntax_set()) else {
        return code.to_string();
    };
    for (range, op) in ScopeRangeIterator::new(&ops, code) {
        let _scoped = scope_stack.apply(op);
        let text = &code[range];
        if text.is_empty() {
            continue;
        }
        match active_scope_formatter(&scope_stack, &theme) {
            Some(formatter) => output.push_str(&formatter(text)),
            None => output.push_str(text),
        }
    }
    output
}
