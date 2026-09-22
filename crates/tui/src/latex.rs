//! The LaTeX-to-Unicode-math renderer for the markdown component, ported from
//! `packages/tui/src/latex.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The renderer turns a basic LaTeX math expression into terminal-friendly
//! Unicode text: Greek letters, operators, and arrows become their Unicode
//! glyphs, scripts become superscript/subscript characters, and in display
//! mode fractions stack over rule lines, operator limits sit beneath their
//! operator, and matrices draw box-drawing delimiters. The markdown component
//! renders through [`render_latex`] and falls back to the raw source when the
//! renderer answers `None`.
//!
//! Restatements against upstream, all invisible to the golden tests:
//!
//! - JS UTF-16 index arithmetic becomes UTF-8 byte indexing; every scan lands
//!   on `char` boundaries, and an escaped character is skipped whole (upstream
//!   skips one UTF-16 code unit, splitting astral pairs).
//! - The layout markers (`U+F0000..=U+F0005`) are single `char`s here, not
//!   surrogate pairs; the marker scans hand-roll the two private-use regexes
//!   (the `regex` crate has no lookaround) over the same marker grammar.
//! - JS `\p{L}`/`\p{N}` classes become `regex`-crate classes, which carry the
//!   same Unicode general categories.
//! - Nested parses run on their own parser but inherit the parent's layout
//!   node list, so marker indices keep counting through nesting exactly as
//!   upstream's shared node array.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use crate::utils::static_regex;
use crate::utils::visible_width;

/// Layout-marker delimiters and the protected-space padding character. All
/// private-use plane-15 code points: they cannot appear in user source, and
/// every consumer strips them before the text reaches a terminal.
const LAYOUT_MARKER_START: char = '\u{f0000}';
const LAYOUT_MARKER_END: char = '\u{f0001}';
const PROTECTED_SPACE: char = '\u{f0002}';
const NAMED_OPERATOR_START: char = '\u{f0004}';
const NAMED_OPERATOR_END: char = '\u{f0005}';

/// Upstream's negative-space sentinel: a NUL character the sequence parser
/// trims around, never emitted in output.
const NEGATIVE_SPACE: &str = "\u{0000}";

static SYMBOLS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("alpha", "α"),
        ("beta", "β"),
        ("gamma", "γ"),
        ("delta", "δ"),
        ("epsilon", "ϵ"),
        ("varepsilon", "ε"),
        ("zeta", "ζ"),
        ("eta", "η"),
        ("theta", "θ"),
        ("vartheta", "ϑ"),
        ("iota", "ι"),
        ("kappa", "κ"),
        ("varkappa", "ϰ"),
        ("lambda", "λ"),
        ("mu", "μ"),
        ("nu", "ν"),
        ("xi", "ξ"),
        ("pi", "π"),
        ("varpi", "ϖ"),
        ("rho", "ρ"),
        ("varrho", "ϱ"),
        ("sigma", "σ"),
        ("varsigma", "ς"),
        ("tau", "τ"),
        ("upsilon", "υ"),
        ("phi", "ϕ"),
        ("varphi", "φ"),
        ("chi", "χ"),
        ("psi", "ψ"),
        ("omega", "ω"),
        ("Gamma", "Γ"),
        ("Delta", "Δ"),
        ("Theta", "Θ"),
        ("Lambda", "Λ"),
        ("Xi", "Ξ"),
        ("Pi", "Π"),
        ("Sigma", "Σ"),
        ("Upsilon", "Υ"),
        ("Phi", "Φ"),
        ("Psi", "Ψ"),
        ("Omega", "Ω"),
        ("pm", "±"),
        ("mp", "∓"),
        ("times", "×"),
        ("div", "÷"),
        ("cdot", "·"),
        ("ast", "∗"),
        ("star", "⋆"),
        ("circ", "∘"),
        ("bullet", "•"),
        ("oplus", "⊕"),
        ("ominus", "⊖"),
        ("otimes", "⊗"),
        ("oslash", "⊘"),
        ("odot", "⊙"),
        ("bigcirc", "○"),
        ("dagger", "†"),
        ("ddagger", "‡"),
        ("amalg", "⨿"),
        ("uplus", "⊎"),
        ("sqcap", "⊓"),
        ("sqcup", "⊔"),
        ("bowtie", "⋈"),
        ("Join", "⋈"),
        ("ltimes", "⋉"),
        ("rtimes", "⋊"),
        ("leftouterjoin", "⟕"),
        ("rightouterjoin", "⟖"),
        ("fullouterjoin", "⟗"),
        ("triangleleft", "◁"),
        ("triangleright", "▷"),
        ("wr", "≀"),
        ("cap", "∩"),
        ("cup", "∪"),
        ("bigcap", "⋂"),
        ("bigcup", "⋃"),
        ("bigwedge", "⋀"),
        ("bigvee", "⋁"),
        ("bigsqcup", "⨆"),
        ("biguplus", "⨄"),
        ("bigoplus", "⨁"),
        ("bigotimes", "⨂"),
        ("bigodot", "⨀"),
        ("setminus", "∖"),
        ("in", "∈"),
        ("notin", "∉"),
        ("ni", "∋"),
        ("subset", "⊂"),
        ("supset", "⊃"),
        ("subseteq", "⊆"),
        ("supseteq", "⊇"),
        ("sqsubset", "⊏"),
        ("sqsupset", "⊐"),
        ("sqsubseteq", "⊑"),
        ("sqsupseteq", "⊒"),
        ("prec", "≺"),
        ("preceq", "≼"),
        ("succ", "≻"),
        ("succeq", "≽"),
        ("ll", "≪"),
        ("gg", "≫"),
        ("le", "≤"),
        ("leq", "≤"),
        ("leqslant", "≤"),
        ("ge", "≥"),
        ("geq", "≥"),
        ("geqslant", "≥"),
        ("ne", "≠"),
        ("neq", "≠"),
        ("equiv", "≡"),
        ("approx", "≈"),
        ("sim", "∼"),
        ("simeq", "≃"),
        ("cong", "≅"),
        ("asymp", "≍"),
        ("doteq", "≐"),
        ("propto", "∝"),
        ("parallel", "∥"),
        ("perp", "⊥"),
        ("mid", "∣"),
        ("vdash", "⊢"),
        ("dashv", "⊣"),
        ("models", "⊨"),
        ("Vdash", "⊩"),
        ("Vvdash", "⊪"),
        ("nvdash", "⊬"),
        ("nvDash", "⊭"),
        ("forall", "∀"),
        ("exists", "∃"),
        ("nexists", "∄"),
        ("neg", "¬"),
        ("land", "∧"),
        ("wedge", "∧"),
        ("lor", "∨"),
        ("vee", "∨"),
        ("to", "→"),
        ("rightarrow", "→"),
        ("longrightarrow", "→"),
        ("leftarrow", "←"),
        ("longleftarrow", "←"),
        ("gets", "←"),
        ("leftrightarrow", "↔"),
        ("longleftrightarrow", "↔"),
        ("hookleftarrow", "↩"),
        ("hookrightarrow", "↪"),
        ("twoheadleftarrow", "↞"),
        ("twoheadrightarrow", "↠"),
        ("leftharpoonup", "↼"),
        ("leftharpoondown", "↽"),
        ("rightharpoonup", "⇀"),
        ("rightharpoondown", "⇁"),
        ("rightleftharpoons", "⇌"),
        ("leftrightharpoons", "⇋"),
        ("nearrow", "↗"),
        ("searrow", "↘"),
        ("swarrow", "↙"),
        ("nwarrow", "↖"),
        ("rightsquigarrow", "⇝"),
        ("leadsto", "⇝"),
        ("Rightarrow", "⇒"),
        ("Longrightarrow", "⇒"),
        ("Leftarrow", "⇐"),
        ("Longleftarrow", "⇐"),
        ("Leftrightarrow", "⇔"),
        ("Longleftrightarrow", "⇔"),
        ("implies", "⇒"),
        ("iff", "⇔"),
        ("mapsto", "↦"),
        ("longmapsto", "↦"),
        ("uparrow", "↑"),
        ("downarrow", "↓"),
        ("partial", "∂"),
        ("nabla", "∇"),
        ("int", "∫"),
        ("iint", "∬"),
        ("iiint", "∭"),
        ("oint", "∮"),
        ("sum", "∑"),
        ("prod", "∏"),
        ("coprod", "∐"),
        ("infty", "∞"),
        ("emptyset", "∅"),
        ("varnothing", "∅"),
        ("angle", "∠"),
        ("therefore", "∴"),
        ("because", "∵"),
        ("aleph", "ℵ"),
        ("beth", "ℶ"),
        ("gimel", "ℷ"),
        ("daleth", "ℸ"),
        ("top", "⊤"),
        ("bot", "⊥"),
        ("triangle", "△"),
        ("square", "□"),
        ("lozenge", "◊"),
        ("checkmark", "✓"),
        ("complement", "∁"),
        ("wp", "℘"),
        ("prime", "′"),
        ("ldots", "…"),
        ("dots", "…"),
        ("cdots", "⋯"),
        ("vdots", "⋮"),
        ("ddots", "⋱"),
        ("ell", "ℓ"),
        ("hbar", "ℏ"),
        ("Im", "ℑ"),
        ("Re", "ℜ"),
        ("langle", "⟨"),
        ("rangle", "⟩"),
        ("vert", "|"),
        ("lvert", "|"),
        ("rvert", "|"),
        ("Vert", "‖"),
        ("lVert", "‖"),
        ("rVert", "‖"),
        ("lbrace", "{"),
        ("rbrace", "}"),
        ("backslash", "\\"),
        ("lfloor", "⌊"),
        ("rfloor", "⌋"),
        ("lceil", "⌈"),
        ("rceil", "⌉"),
        ("colon", ":"),
    ])
});

static NAMED_OPERATORS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "arccos", "arcsin", "arctan", "arg", "cos", "cosh", "cot", "coth", "csc", "deg", "det",
        "dim", "exp", "gcd", "hom", "inf", "ker", "lg", "lim", "liminf", "limsup", "ln", "log",
        "max", "min", "Pr", "sec", "sin", "sinh", "sup", "tan", "tanh",
    ])
});

static LIMIT_OPERATORS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "argmax", "argmin", "inf", "injlim", "lim", "liminf", "limsup", "max", "min", "projlim",
        "sup",
    ])
});

static DISPLAY_LIMIT_SYMBOLS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "bigcap",
        "bigcup",
        "bigodot",
        "bigoplus",
        "bigotimes",
        "bigsqcup",
        "biguplus",
        "bigvee",
        "bigwedge",
        "coprod",
        "int",
        "iint",
        "iiint",
        "oint",
        "prod",
        "sum",
    ])
});

static RELATION_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "Leftarrow",
        "Leftrightarrow",
        "Longleftarrow",
        "Longleftrightarrow",
        "Longrightarrow",
        "Rightarrow",
        "Join",
        "Vdash",
        "Vvdash",
        "approx",
        "asymp",
        "bowtie",
        "cong",
        "dashv",
        "fullouterjoin",
        "doteq",
        "downarrow",
        "equiv",
        "ge",
        "geq",
        "geqslant",
        "gets",
        "gg",
        "hookleftarrow",
        "hookrightarrow",
        "iff",
        "implies",
        "in",
        "leadsto",
        "le",
        "leftarrow",
        "leftharpoondown",
        "leftharpoonup",
        "leftrightarrow",
        "leftrightharpoons",
        "leftouterjoin",
        "leq",
        "leqslant",
        "ll",
        "longleftarrow",
        "longleftrightarrow",
        "longmapsto",
        "longrightarrow",
        "ltimes",
        "mapsto",
        "mid",
        "models",
        "ne",
        "nearrow",
        "neq",
        "ni",
        "notin",
        "nvdash",
        "nvDash",
        "nwarrow",
        "parallel",
        "perp",
        "prec",
        "preceq",
        "propto",
        "rightharpoondown",
        "rightharpoonup",
        "rightleftharpoons",
        "rightouterjoin",
        "rightarrow",
        "rightsquigarrow",
        "rtimes",
        "searrow",
        "sim",
        "simeq",
        "sqsubset",
        "sqsubseteq",
        "sqsupset",
        "sqsupseteq",
        "subset",
        "subseteq",
        "succ",
        "succeq",
        "supset",
        "supseteq",
        "swarrow",
        "to",
        "triangleleft",
        "triangleright",
        "twoheadleftarrow",
        "twoheadrightarrow",
        "uparrow",
        "vdash",
    ])
});

static NEGATED_SYMBOLS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("<", "≮"),
        (">", "≯"),
        ("=", "≠"),
        ("∈", "∉"),
        ("∋", "∌"),
        ("∣", "∤"),
        ("∥", "∦"),
        ("∼", "≁"),
        ("≃", "≄"),
        ("≅", "≇"),
        ("≈", "≉"),
        ("≡", "≢"),
        ("≤", "≰"),
        ("≥", "≱"),
        ("≺", "⊀"),
        ("≻", "⊁"),
        ("⊂", "⊄"),
        ("⊃", "⊅"),
        ("⊆", "⊈"),
        ("⊇", "⊉"),
        ("⊢", "⊬"),
        ("⊨", "⊭"),
        ("↔", "↮"),
        ("←", "↚"),
        ("→", "↛"),
        ("⇒", "⇏"),
        ("⇐", "⇍"),
        ("⇔", "⇎"),
        ("≼", "⋠"),
        ("≽", "⋡"),
    ])
});

static BLACKBOARD: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("C", "ℂ"),
        ("H", "ℍ"),
        ("N", "ℕ"),
        ("P", "ℙ"),
        ("Q", "ℚ"),
        ("R", "ℝ"),
        ("Z", "ℤ"),
    ])
});

static SUPERSCRIPTS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("0", "⁰"),
        ("1", "¹"),
        ("2", "²"),
        ("3", "³"),
        ("4", "⁴"),
        ("5", "⁵"),
        ("6", "⁶"),
        ("7", "⁷"),
        ("8", "⁸"),
        ("9", "⁹"),
        ("+", "⁺"),
        ("-", "⁻"),
        ("=", "⁼"),
        ("(", "⁽"),
        (")", "⁾"),
        ("a", "ᵃ"),
        ("b", "ᵇ"),
        ("c", "ᶜ"),
        ("d", "ᵈ"),
        ("e", "ᵉ"),
        ("f", "ᶠ"),
        ("g", "ᵍ"),
        ("h", "ʰ"),
        ("i", "ⁱ"),
        ("j", "ʲ"),
        ("k", "ᵏ"),
        ("l", "ˡ"),
        ("m", "ᵐ"),
        ("n", "ⁿ"),
        ("o", "ᵒ"),
        ("p", "ᵖ"),
        ("r", "ʳ"),
        ("s", "ˢ"),
        ("t", "ᵗ"),
        ("u", "ᵘ"),
        ("v", "ᵛ"),
        ("w", "ʷ"),
        ("x", "ˣ"),
        ("y", "ʸ"),
        ("z", "ᶻ"),
    ])
});

static SUBSCRIPTS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("0", "₀"),
        ("1", "₁"),
        ("2", "₂"),
        ("3", "₃"),
        ("4", "₄"),
        ("5", "₅"),
        ("6", "₆"),
        ("7", "₇"),
        ("8", "₈"),
        ("9", "₉"),
        ("+", "₊"),
        ("-", "₋"),
        ("=", "₌"),
        ("(", "₍"),
        (")", "₎"),
        ("a", "ₐ"),
        ("e", "ₑ"),
        ("h", "ₕ"),
        ("i", "ᵢ"),
        ("j", "ⱼ"),
        ("k", "ₖ"),
        ("l", "ₗ"),
        ("m", "ₘ"),
        ("n", "ₙ"),
        ("o", "ₒ"),
        ("p", "ₚ"),
        ("r", "ᵣ"),
        ("s", "ₛ"),
        ("t", "ₜ"),
        ("u", "ᵤ"),
        ("v", "ᵥ"),
        ("x", "ₓ"),
    ])
});

static SPACING_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        ",",
        ":",
        ";",
        " ",
        ">",
        "enspace",
        "enskip",
        "medspace",
        "quad",
        "qquad",
        "thickspace",
        "thinspace",
    ])
});

static NEGATIVE_SPACING_COMMANDS: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| HashSet::from(["!", "negmedspace", "negthickspace", "negthinspace"]));

static IGNORED_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "displaystyle",
        "limits",
        "nolimits",
        "scriptstyle",
        "scriptscriptstyle",
        "textstyle",
    ])
});

static SIZE_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "big", "Big", "bigg", "Bigg", "bigl", "Bigl", "biggl", "Biggl", "bigr", "Bigr", "biggr",
        "Biggr",
    ])
});

static PLAIN_WRAPPERS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "emph",
        "mathcal",
        "mathbf",
        "mathfrak",
        "mathit",
        "mathrm",
        "mathnormal",
        "mathscr",
        "mathsf",
        "mathtt",
        "mathup",
        "mbox",
        "overbrace",
        "pmb",
        "smash",
        "substack",
        "text",
        "textbf",
        "textit",
        "textmd",
        "textnormal",
        "textrm",
        "textsc",
        "textsf",
        "textsl",
        "texttt",
        "textup",
        "underbrace",
        "bm",
        "boldsymbol",
    ])
});

static ACCENTS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("acute", "\u{0301}"),
        ("bar", "\u{0305}"),
        ("breve", "\u{0306}"),
        ("check", "\u{030c}"),
        ("ddot", "\u{0308}"),
        ("dot", "\u{0307}"),
        ("grave", "\u{0300}"),
        ("hat", "\u{0302}"),
        ("mathring", "\u{030a}"),
        ("overleftarrow", "\u{20d6}"),
        ("overleftrightarrow", "\u{20e1}"),
        ("overline", "\u{0305}"),
        ("overrightarrow", "\u{20d7}"),
        ("tilde", "\u{0303}"),
        ("underline", "\u{0332}"),
        ("vec", "\u{20d7}"),
        ("widehat", "\u{0302}"),
        ("widetilde", "\u{0303}"),
    ])
});

static SIMPLE_TEXT_RE: LazyLock<Regex> = LazyLock::new(|| static_regex("^[\\p{L}\\p{N}.]+$"));
static SIMPLE_NUMBER_RE: LazyLock<Regex> = LazyLock::new(|| static_regex("^[\\p{N}.]+$"));

/// Whether a character counts as a letter or number for the named-operator
/// spacing scans. Upstream's `[\p{L}\p{N}]` inside a lookaround the `regex`
/// crate cannot express; the scans only ever run over the renderer's own
/// output, where the categories coincide.
fn is_letter_or_number(ch: char) -> bool {
    ch.is_alphabetic() || ch.is_numeric()
}

/// JS `\s`: the Unicode `White_Space` set plus U+FEFF. Rust's
/// `char::is_whitespace` covers `White_Space`; the BOM rides along explicitly.
const fn is_js_whitespace(ch: char) -> bool {
    ch.is_whitespace() || ch == '\u{feff}'
}

/// `/\s*([=+-])\s*/g` → `"$1"`: whitespace touching a sign collapses onto it.
fn collapse_sign_whitespace(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    loop {
        let mut whitespace = String::new();
        while let Some(&ch) = chars.peek() {
            if is_js_whitespace(ch) {
                whitespace.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        match chars.peek() {
            Some(&sign @ ('=' | '+' | '-')) => {
                chars.next();
                out.push(sign);
                while let Some(&ch) = chars.peek() {
                    if is_js_whitespace(ch) {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            Some(_) => {
                out.push_str(&whitespace);
                out.push(chars.next().unwrap_or_default());
            }
            None => {
                out.push_str(&whitespace);
                break;
            }
        }
    }
    out
}

/// Map every character through `replacements`, or `None` when any character
/// has no entry — upstream `replaceCharacters`.
fn replace_characters(
    value: &str,
    replacements: &HashMap<&'static str, &'static str>,
) -> Option<String> {
    let mut result = String::with_capacity(value.len());
    let mut buffer = [0_u8; 4];
    for character in value.chars() {
        let encoded: &str = character.encode_utf8(&mut buffer);
        result.push_str(replacements.get(encoded)?);
    }
    Some(result)
}

/// Render a script argument as Unicode sub/superscripts, falling back to
/// `_value`/`^value` spellings when the script has no Unicode form — upstream
/// `formatScript`.
fn format_script(value: &str, sub: bool) -> String {
    let value = value.trim();
    let replacements = if sub { &SUBSCRIPTS } else { &SUPERSCRIPTS };
    let collapsed = collapse_sign_whitespace(value);
    if let Some(unicode) = replace_characters(&collapsed, replacements) {
        return unicode;
    }

    let prefix = if sub { "_" } else { "^" };
    if value.chars().count() == 1
        || (sub && !value.is_empty() && value.chars().all(|ch| ch.is_ascii_alphabetic()))
    {
        format!("{prefix}{value}")
    } else {
        format!("{prefix}({value})")
    }
}

/// Numerator/denominator as a simple `a/b` form, parenthesizing the complex
/// side — upstream `formatFraction`.
fn format_fraction(numerator: &str, denominator: &str) -> String {
    let numerator = numerator.trim();
    let denominator = denominator.trim();
    let simple_numerator = SIMPLE_TEXT_RE.is_match(numerator);
    let simple_denominator =
        SIMPLE_NUMBER_RE.is_match(denominator) || denominator.chars().count() == 1;
    let open_numerator = if simple_numerator {
        numerator.to_string()
    } else {
        format!("({numerator})")
    };
    let open_denominator = if simple_denominator {
        denominator.to_string()
    } else {
        format!("({denominator})")
    };
    format!("{open_numerator}/{open_denominator}")
}

/// A root as `√value` or `√(value)` — upstream `formatRoot`.
fn format_root(value: &str, symbol: &str) -> String {
    let value = value.trim();
    if SIMPLE_TEXT_RE.is_match(value) {
        format!("{symbol}{value}")
    } else {
        format!("{symbol}({value})")
    }
}

/// Strip the named-operator sentinels, collapse runs of horizontal whitespace
/// per line, drop empty edge lines, and trim — upstream `normalizeOutput`.
fn normalize_output(value: &str) -> String {
    // Left spacing: U+F0004 preceded by a letter, number, closing bracket, or
    // layout end marker becomes a space; every other U+F0004 is removed.
    let mut spaced = String::with_capacity(value.len());
    let mut previous: Option<char> = None;
    for ch in value.chars() {
        if ch == NAMED_OPERATOR_START {
            if previous.is_some_and(is_letter_or_number) {
                spaced.push(' ');
            }
        } else {
            spaced.push(ch);
        }
        previous = Some(ch);
    }
    // NAMED_OPERATOR_START is now gone; the right spacing pattern looks ahead
    // from each NAMED_OPERATOR_END.
    let chars: Vec<char> = spaced.chars().collect();
    let mut stripped = String::with_capacity(spaced.len());
    for (index, &ch) in chars.iter().enumerate() {
        if ch == NAMED_OPERATOR_END {
            let next = chars.get(index + 1).copied();
            let follows = next
                .is_some_and(|n| n == '√' || n == LAYOUT_MARKER_START || is_letter_or_number(n));
            if follows {
                stripped.push(' ');
            }
        } else {
            stripped.push(ch);
        }
    }
    let normalized: String = stripped
        .chars()
        .filter(|&ch| ch != NAMED_OPERATOR_END)
        .collect();
    let mapped: Vec<String> = normalized
        .split('\n')
        .map(|line| {
            let mut collapsed = String::with_capacity(line.len());
            let mut chars = line.chars().peekable();
            while let Some(&ch) = chars.peek() {
                if ch == ' ' || ch == '\t' {
                    while let Some(&next) = chars.peek() {
                        if next == ' ' || next == '\t' {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    collapsed.push(' ');
                } else {
                    collapsed.push(ch);
                    chars.next();
                }
            }
            collapsed.trim().to_string()
        })
        .collect();
    let last = mapped.len().saturating_sub(1);
    let filtered: Vec<&str> = mapped
        .iter()
        .enumerate()
        .filter(|(index, line)| !line.is_empty() || (*index > 0 && *index < last))
        .map(|(_, line)| line.as_str())
        .collect();
    filtered.join("\n").trim().to_string()
}

/// One node the two-pass layout stacks vertically: a fraction over a rule
/// line, an operator with stacked limits, or a box-drawing matrix.
enum LayoutNode {
    Fraction {
        numerator: String,
        denominator: String,
    },
    Operator {
        operator: String,
        lower: Option<String>,
        upper: Option<String>,
    },
    Matrix {
        lines: Vec<String>,
        baseline: usize,
    },
}

/// The rendered geometry of one layout unit: its lines, its cell width, and
/// the line index that sits on the text baseline.
struct Layout {
    lines: Vec<String>,
    width: usize,
    baseline: usize,
}

/// One `\u{f0000}<digits>\u{f0001}` marker inside a source line: the byte
/// range it occupies and the node index it encodes.
struct MarkerMatch {
    start: usize,
    end: usize,
    index: usize,
}

/// Scan a source line for layout markers — upstream `LAYOUT_MARKER_PATTERN`
/// (`/\u{f0000}(\d+)\u{f0001}/gu`), hand-scanned because the `regex` crate
/// offers no equivalent and the marker grammar is ours.
fn find_layout_markers(source_line: &str) -> Vec<MarkerMatch> {
    let mut matches = Vec::new();
    let mut offset = 0;
    while let Some(relative) = source_line[offset..].find(LAYOUT_MARKER_START) {
        let start = offset + relative;
        let mut cursor = start + LAYOUT_MARKER_START.len_utf8();
        let digits_start = cursor;
        while cursor < source_line.len() && source_line.as_bytes()[cursor].is_ascii_digit() {
            cursor += 1;
        }
        let closed = source_line[cursor..].starts_with(LAYOUT_MARKER_END);
        match (source_line[digits_start..cursor].parse::<usize>(), closed) {
            (Ok(index), true) => {
                let end = cursor + LAYOUT_MARKER_END.len_utf8();
                matches.push(MarkerMatch { start, end, index });
                offset = end;
            }
            _ => {
                offset = cursor.max(start + LAYOUT_MARKER_START.len_utf8());
            }
        }
    }
    matches
}

/// The node index of a `\u{f0000}<digits>\u{f0001}` marker ending the string,
/// or `None` — upstream `TRAILING_LAYOUT_MARKER_PATTERN`.
fn trailing_layout_marker(source: &str) -> Option<usize> {
    if !source.ends_with(LAYOUT_MARKER_END) {
        return None;
    }
    let digits_end = source.len() - LAYOUT_MARKER_END.len_utf8();
    let mut start = digits_end;
    while start > 0 && source.as_bytes()[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if start == digits_end || !source[..start].ends_with(LAYOUT_MARKER_START) {
        return None;
    }
    source[start..digits_end].parse::<usize>().ok()
}

/// Pad a line to `width` cells, optionally centering it — upstream
/// `padLayoutLine`.
fn pad_layout_line(line: &str, width: usize, centered: bool) -> String {
    let padding = width.saturating_sub(visible_width(line));
    let left = if centered { padding / 2 } else { 0 };
    format!("{}{}{}", " ".repeat(left), line, " ".repeat(padding - left))
}

/// Place layouts side by side on a shared baseline — upstream `joinLayouts`.
fn join_layouts(layouts: &[Layout]) -> Layout {
    if layouts.is_empty() {
        return Layout {
            lines: vec![String::new()],
            width: 0,
            baseline: 0,
        };
    }
    let baseline = layouts
        .iter()
        .map(|layout| layout.baseline)
        .max()
        .unwrap_or_default();
    let below = layouts
        .iter()
        .map(|layout| layout.lines.len().saturating_sub(layout.baseline + 1))
        .max()
        .unwrap_or_default();
    let mut lines = Vec::with_capacity(baseline + below + 1);
    for row in 0..=baseline + below {
        let mut line = String::new();
        for layout in layouts {
            let offset = baseline - layout.baseline;
            let source_row = row.checked_sub(offset);
            match source_row.and_then(|row| layout.lines.get(row)) {
                Some(line_text) if row >= offset => {
                    line.push_str(&pad_layout_line(line_text, layout.width, false));
                }
                _ => line.push_str(&" ".repeat(layout.width)),
            }
        }
        lines.push(line.trim_end().to_string());
    }
    Layout {
        lines,
        width: layouts.iter().map(|layout| layout.width).sum(),
        baseline,
    }
}

/// Stack fraction/limit/matrix nodes vertically over their inline markers —
/// upstream `renderLayout`.
#[expect(
    clippy::too_many_lines,
    reason = "the function mirrors upstream renderLayout branch-for-branch; extracting the three node arms would hide the 1:1 correspondence"
)]
fn render_layout(source: &str, nodes: &[LayoutNode]) -> Layout {
    let mut rendered_lines: Vec<String> = Vec::new();
    let mut first_baseline = 0;
    for source_line in source.split('\n') {
        let mut layouts: Vec<Layout> = Vec::new();
        let mut position = 0;
        let mut previous: Option<&LayoutNode> = None;
        for MarkerMatch { start, end, index } in find_layout_markers(source_line) {
            let Some(node) = nodes.get(index) else {
                continue;
            };
            if start > position {
                let sliced = &source_line[position..start];
                let trimmed = if previous.is_some() {
                    sliced.trim_start()
                } else {
                    sliced
                }
                .trim_end();
                let preserve_leading = matches!(previous, Some(LayoutNode::Matrix { .. }))
                    && sliced.starts_with(is_js_whitespace);
                let preserve_trailing =
                    matches!(node, LayoutNode::Matrix { .. }) && sliced.ends_with(is_js_whitespace);
                let text = if !trimmed.is_empty() {
                    format!(
                        "{}{trimmed}{}",
                        if preserve_leading { " " } else { "" },
                        if preserve_trailing { " " } else { "" }
                    )
                } else if preserve_leading || preserve_trailing {
                    " ".to_string()
                } else {
                    String::new()
                };
                layouts.push(Layout {
                    lines: vec![text.clone()],
                    width: visible_width(&text),
                    baseline: 0,
                });
            }
            match node {
                LayoutNode::Fraction {
                    numerator,
                    denominator,
                } => {
                    let numerator = render_layout(numerator, nodes);
                    let denominator = render_layout(denominator, nodes);
                    let content_width = numerator.width.max(denominator.width).max(1);
                    let width = content_width + 2;
                    let mut lines: Vec<String> = numerator
                        .lines
                        .iter()
                        .map(|line| pad_layout_line(line, width, true))
                        .collect();
                    lines.push(format!(" {} ", "─".repeat(content_width)));
                    lines.extend(
                        denominator
                            .lines
                            .iter()
                            .map(|line| pad_layout_line(line, width, true)),
                    );
                    layouts.push(Layout {
                        lines,
                        width,
                        baseline: numerator.lines.len(),
                    });
                }
                LayoutNode::Operator {
                    operator,
                    lower,
                    upper,
                } => {
                    let content_width = visible_width(operator)
                        .max(lower.as_deref().map_or(0, visible_width))
                        .max(upper.as_deref().map_or(0, visible_width));
                    let mut lines = Vec::new();
                    if let Some(upper) = upper {
                        lines.push(format!("{} ", pad_layout_line(upper, content_width, true)));
                    }
                    lines.push(format!(
                        "{} ",
                        pad_layout_line(operator, content_width, true)
                    ));
                    if let Some(lower) = lower {
                        lines.push(format!("{} ", pad_layout_line(lower, content_width, true)));
                    }
                    layouts.push(Layout {
                        lines,
                        width: content_width + 1,
                        baseline: usize::from(upper.is_some()),
                    });
                }
                LayoutNode::Matrix {
                    lines: matrix_lines,
                    baseline,
                } => {
                    let width = matrix_lines
                        .iter()
                        .map(|line| visible_width(line))
                        .max()
                        .unwrap_or(0);
                    layouts.push(Layout {
                        lines: matrix_lines
                            .iter()
                            .map(|line| pad_layout_line(line, width, false))
                            .collect(),
                        width,
                        baseline: *baseline,
                    });
                }
            }
            position = end;
            previous = Some(node);
        }
        if position < source_line.len() {
            let sliced = &source_line[position..];
            let trimmed = if previous.is_some() {
                sliced.trim_start()
            } else {
                sliced
            };
            let text = if matches!(previous, Some(LayoutNode::Matrix { .. }))
                && sliced.starts_with(is_js_whitespace)
            {
                format!(" {trimmed}")
            } else {
                trimmed.to_string()
            };
            layouts.push(Layout {
                lines: vec![text.clone()],
                width: visible_width(&text),
                baseline: 0,
            });
        }
        let line_layout = join_layouts(&layouts);
        if rendered_lines.is_empty() {
            first_baseline = line_layout.baseline;
        }
        rendered_lines.extend(line_layout.lines);
    }
    let width = rendered_lines
        .iter()
        .map(|line| visible_width(line))
        .max()
        .unwrap_or(0);
    Layout {
        lines: rendered_lines,
        width,
        baseline: first_baseline,
    }
}

/// Which inline spelling a stacked-operator lower limit takes: `lim[x]` for
/// named operators, `xᵢ` for symbols — upstream `inlineLowerStyle`.
#[derive(Clone, Copy)]
enum LowerStyle {
    /// `op[lower]`.
    Bracket,
    /// `op` plus a Unicode subscript.
    Script,
}

/// The recursive-descent parser over one LaTeX source — upstream
/// `LatexParser`. Byte-indexed over the borrowed source; the layout nodes the
/// parse pushes outlive the parser and feed the second rendering pass.
struct LatexParser<'a> {
    source: &'a str,
    layout_nodes: Vec<LayoutNode>,
    display: bool,
    position: usize,
    supported: bool,
    stack_fractions: bool,
}

impl<'a> LatexParser<'a> {
    const fn new(source: &'a str, layout_nodes: Vec<LayoutNode>, display: bool) -> Self {
        Self {
            source,
            layout_nodes,
            display,
            position: 0,
            supported: true,
            stack_fractions: true,
        }
    }

    /// Parse and normalize; `None` when the source used unsupported or
    /// malformed syntax, or stopped mid-expression.
    fn render(&mut self) -> Option<String> {
        let rendered = self.parse_sequence(None);
        if !self.supported || self.position != self.source.len() {
            return None;
        }
        Some(normalize_output(&rendered))
    }

    fn peek(&self) -> Option<char> {
        self.source[self.position..].chars().next()
    }

    fn bump(&mut self) {
        if let Some(character) = self.peek() {
            self.position += character.len_utf8();
        }
    }

    fn parse_sequence(&mut self, end: Option<char>) -> String {
        let mut result = String::new();
        while self.position < self.source.len() {
            let character = self.peek().unwrap_or_default();
            if end.is_some_and(|end| character == end) {
                self.bump();
                return result;
            }

            if character == '}' {
                self.supported = false;
                return result;
            }

            if character == '{' {
                self.bump();
                result.push_str(&self.parse_sequence(Some('}')));
                continue;
            }

            if character == '\\' {
                let command = self.parse_command();
                if command == NEGATIVE_SPACE {
                    result = result.trim_end().to_string();
                    if result.ends_with(NAMED_OPERATOR_END) {
                        // The marker is one char here; upstream slices off
                        // its two UTF-16 units.
                        result.pop();
                    }
                } else {
                    result.push_str(&command);
                }
                continue;
            }

            if character == '^' || character == '_' {
                self.bump();
                result = result.trim_end().to_string();
                let script = format_script(&self.parse_required_argument(false), character == '_');
                if result.ends_with(NAMED_OPERATOR_END) {
                    result.pop();
                    result.push_str(&script);
                    result.push(NAMED_OPERATOR_END);
                } else {
                    result.push_str(&script);
                }
                continue;
            }

            if is_js_whitespace(character) {
                result.push_str(&self.parse_whitespace());
                continue;
            }

            if character == '=' || character == '<' || character == '>' {
                result = format!("{} {character} ", result.trim_end());
                self.bump();
                continue;
            }

            if character == '&' {
                self.bump();
                continue;
            }

            if character == '~' {
                self.bump();
                result.push(' ');
                continue;
            }

            if character == '.' {
                let marker_index = trailing_layout_marker(&result);
                if let Some(node) = marker_index.and_then(|index| self.layout_nodes.get_mut(index))
                    && let LayoutNode::Matrix { lines, .. } = node
                    && let Some(last_line) = lines.last_mut()
                {
                    last_line.push(character);
                    self.bump();
                    continue;
                }
            }

            result.push(character);
            self.bump();
        }

        if end.is_some() {
            self.supported = false;
        }
        result
    }

    fn parse_whitespace(&mut self) -> String {
        while self.peek().is_some_and(is_js_whitespace) {
            self.bump();
        }
        " ".to_string()
    }

    fn parse_command(&mut self) -> String {
        self.bump();
        if self.position >= self.source.len() {
            self.supported = false;
            return String::new();
        }

        let first = self.peek().unwrap_or_default();
        if first == '\n' || first == '\r' {
            self.bump();
            if first == '\r' && self.peek() == Some('\n') {
                self.bump();
            }
            return " ".to_string();
        }
        let command: String = if first.is_ascii_alphabetic() {
            let start = self.position;
            while self
                .peek()
                .is_some_and(|character| character.is_ascii_alphabetic())
            {
                self.bump();
            }
            self.source[start..self.position].to_string()
        } else {
            self.bump();
            first.to_string()
        };

        if command == "\\" {
            return "\n".to_string();
        }
        if SPACING_COMMANDS.contains(command.as_str()) {
            return " ".to_string();
        }
        if NEGATIVE_SPACING_COMMANDS.contains(command.as_str()) {
            return NEGATIVE_SPACE.to_string();
        }
        if IGNORED_COMMANDS.contains(command.as_str()) {
            return String::new();
        }
        if matches!(command.as_str(), "{" | "}" | "$" | "%" | "#" | "_" | "&") {
            return command;
        }
        if command == "|" {
            return "‖".to_string();
        }
        if command == "not" {
            let value = self.parse_required_argument(false).trim().to_string();
            if let Some(negated) = NEGATED_SYMBOLS.get(value.as_str()) {
                return format!(" {negated} ");
            }
            let characters: Vec<char> = value.chars().collect();
            if characters.is_empty() {
                self.supported = false;
                return String::new();
            }
            return format!(
                " {}{}{} ",
                characters[0],
                '\u{0338}',
                characters[1..].iter().collect::<String>()
            );
        }
        if LIMIT_OPERATORS.contains(command.as_str()) {
            return self.parse_operator(&command, LowerStyle::Bracket, true, true);
        }

        let Some(symbol) = SYMBOLS.get(command.as_str()) else {
            return self.parse_command_tail(&command);
        };
        if DISPLAY_LIMIT_SYMBOLS.contains(command.as_str()) {
            return self.parse_operator(symbol, LowerStyle::Script, true, false);
        }
        if command == "cdot" || command == "times" || RELATION_COMMANDS.contains(command.as_str()) {
            return format!(" {symbol} ");
        }
        symbol.to_string()
    }

    /// The command dispatch tail for commands with no symbol-table entry —
    /// upstream `parseCommand`'s fall-through ladder.
    #[expect(
        clippy::too_many_lines,
        reason = "the ladder mirrors upstream parseCommand's fall-through arm order; grouping it would reorder the checks"
    )]
    fn parse_command_tail(&mut self, command: &str) -> String {
        if NAMED_OPERATORS.contains(command) {
            return format!("{NAMED_OPERATOR_START}{command}{NAMED_OPERATOR_END}");
        }
        if SIZE_COMMANDS.contains(command) {
            return String::new();
        }
        if matches!(command, "left" | "middle" | "right") {
            if self.peek() == Some('.') {
                self.bump();
            }
            return String::new();
        }
        if matches!(command, "frac" | "dfrac" | "tfrac") {
            let should_stack = self.display && self.stack_fractions && command != "tfrac";
            let numerator = self.parse_required_argument(!should_stack);
            let denominator = self.parse_required_argument(!should_stack);
            if should_stack {
                let index = self.layout_nodes.len();
                self.layout_nodes.push(LayoutNode::Fraction {
                    numerator: normalize_output(&numerator),
                    denominator: normalize_output(&denominator),
                });
                return format!("{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}");
            }
            return format_fraction(&numerator, &denominator);
        }
        if command == "sqrt" {
            let degree = self
                .parse_optional_argument()
                .map(|degree| degree.trim().to_string());
            let value = self.parse_required_argument(true);
            let Some(degree) = degree else {
                return format_root(&value, "√");
            };
            return match degree.as_str() {
                "2" => format_root(&value, "√"),
                "3" => format_root(&value, "∛"),
                "4" => format_root(&value, "∜"),
                _ => format!(
                    "{}{}",
                    format_script(&degree, false),
                    format_root(&value, "√")
                ),
            };
        }
        if matches!(command, "boxed" | "fbox") {
            return format!("[{}]", self.parse_required_argument(true).trim());
        }
        if matches!(command, "binom" | "dbinom" | "tbinom") {
            return format!(
                "({} choose {})",
                self.parse_required_argument(true),
                self.parse_required_argument(true)
            );
        }
        if let Some(accent) = ACCENTS.get(command) {
            let value = self.parse_required_argument(true);
            return if value.chars().count() == 1 {
                format!("{value}{accent}")
            } else {
                format!("{command}({value})")
            };
        }
        if command == "mathbb" {
            let value = self.parse_required_argument(true);
            return value
                .chars()
                .map(|character| {
                    let mut buffer = [0_u8; 4];
                    let encoded: &str = character.encode_utf8(&mut buffer);
                    BLACKBOARD.get(encoded).map_or_else(
                        || character.to_string(),
                        |replacement| (*replacement).to_string(),
                    )
                })
                .collect();
        }
        if command == "operatorname" {
            let starred = self.peek() == Some('*');
            if starred {
                self.bump();
            }
            let operator = normalize_output(&self.parse_required_argument(true))
                .trim()
                .to_string();
            return self.parse_operator(&operator, LowerStyle::Bracket, starred, true);
        }
        if matches!(command, "mod" | "bmod") {
            return " mod ".to_string();
        }
        if matches!(command, "pmod" | "pod") {
            let value = self.parse_required_argument(true).trim().to_string();
            return if command == "pmod" {
                format!(" (mod {value})")
            } else {
                format!(" ({value})")
            };
        }
        if matches!(command, "overset" | "stackrel") {
            let upper = self.parse_required_argument(true);
            let value = self.parse_required_argument(true).trim().to_string();
            return format!("{value}{}", format_script(&upper, false));
        }
        if command == "underset" {
            let lower = self.parse_required_argument(true);
            let value = self.parse_required_argument(true).trim().to_string();
            return format!("{value}{}", format_script(&lower, true));
        }
        if PLAIN_WRAPPERS.contains(command) {
            let value = self.parse_required_argument(true);
            return if command.starts_with("text") || command == "mbox" {
                value
            } else {
                value.trim().to_string()
            };
        }
        if command == "begin" {
            return self.parse_environment();
        }
        if command == "end" {
            self.supported = false;
            return String::new();
        }

        self.supported = false;
        format!("\\{command}")
    }

    /// Parse a `\limits`/`\nolimits` modifier and any `_`/`^` scripts after an
    /// operator — upstream `parseOperator`.
    fn parse_operator(
        &mut self,
        operator: &str,
        inline_lower_style: LowerStyle,
        display_limits: bool,
        spaced: bool,
    ) -> String {
        let mut use_display_limits = display_limits;
        let mut modifier_position = self.position;
        while modifier_position < self.source.len()
            && matches!(
                self.source.as_bytes().get(modifier_position),
                Some(b' ' | b'\t')
            )
        {
            modifier_position += 1;
        }
        let modifier_source = &self.source[modifier_position..];
        let modifier = ["\\limits", "\\nolimits"].into_iter().find(|name| {
            modifier_source.starts_with(name)
                && !modifier_source[name.len()..]
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_alphabetic())
        });
        if let Some(modifier) = modifier {
            use_display_limits = modifier == "\\limits";
            self.position = modifier_position + modifier.len();
        }

        let mut lower: Option<String> = None;
        let mut upper: Option<String> = None;
        loop {
            let mut script_position = self.position;
            while script_position < self.source.len()
                && matches!(
                    self.source.as_bytes().get(script_position),
                    Some(b' ' | b'\t')
                )
            {
                script_position += 1;
            }
            let kind = self.source[script_position..].chars().next();
            let Some(kind @ ('_' | '^')) = kind else {
                break;
            };
            self.position = script_position + kind.len_utf8();
            let value = normalize_output(&self.parse_required_argument(false)).replace(' ', "");
            if kind == '_' {
                if lower.is_some() {
                    self.supported = false;
                }
                lower = Some(value);
            } else {
                if upper.is_some() {
                    self.supported = false;
                }
                upper = Some(value);
            }
        }

        if self.display && use_display_limits && (lower.is_some() || upper.is_some()) {
            let index = self.layout_nodes.len();
            self.layout_nodes.push(LayoutNode::Operator {
                operator: operator.to_string(),
                lower,
                upper,
            });
            return format!("{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}");
        }

        let mut rendered = operator.to_string();
        if let Some(lower) = &lower {
            match inline_lower_style {
                LowerStyle::Bracket => {
                    rendered.push('[');
                    rendered.push_str(lower);
                    rendered.push(']');
                }
                LowerStyle::Script => rendered.push_str(&format_script(lower, true)),
            }
        }
        if let Some(upper) = &upper {
            rendered.push_str(&format_script(upper, false));
        }
        if spaced {
            format!(" {rendered} ")
        } else {
            rendered
        }
    }

    /// One argument with the fraction-stacking context suspended or kept —
    /// upstream `parseRequiredArgument`.
    fn parse_required_argument(&mut self, stack_fractions: bool) -> String {
        let previous = self.stack_fractions;
        self.stack_fractions = previous && stack_fractions;
        let value = self.parse_required_argument_value();
        self.stack_fractions = previous;
        value
    }

    fn parse_required_argument_value(&mut self) -> String {
        while self.peek().is_some_and(is_js_whitespace) {
            self.bump();
        }
        if self.position >= self.source.len() {
            self.supported = false;
            return String::new();
        }
        match self.peek() {
            Some('{') => {
                self.bump();
                self.parse_sequence(Some('}'))
            }
            Some('\\') => self.parse_command(),
            Some(character) => {
                self.bump();
                character.to_string()
            }
            None => String::new(),
        }
    }

    /// A `[degree]` bracket, rendered — upstream `parseOptionalArgument`.
    fn parse_optional_argument(&mut self) -> Option<String> {
        while self.position < self.source.len()
            && matches!(
                self.source.as_bytes().get(self.position),
                Some(b' ' | b'\t')
            )
        {
            self.position += 1;
        }
        if self.peek() != Some('[') {
            return None;
        }
        let Some(end) = self.source[self.position + 1..].find(']') else {
            self.supported = false;
            return None;
        };
        let end = end + self.position + 1;
        let value = self.source[self.position + 1..end].to_string();
        self.position = end + 1;
        Some(self.render_nested(&value, true))
    }

    /// A raw `{group}` with escapes skipped whole — upstream `readRawGroup`.
    fn read_raw_group(&mut self) -> Option<String> {
        while self.position < self.source.len()
            && matches!(
                self.source.as_bytes().get(self.position),
                Some(b' ' | b'\t')
            )
        {
            self.position += 1;
        }
        if self.peek() != Some('{') {
            self.supported = false;
            return None;
        }

        self.bump();
        let start = self.position;
        let mut depth = 1;
        while self.position < self.source.len() {
            let character = self.peek().unwrap_or_default();
            if character == '\\' {
                // Upstream skips one UTF-16 unit per backslash; skipping the
                // whole escaped char keeps the scan on char boundaries.
                self.bump();
                self.bump();
                continue;
            }
            if character == '{' {
                depth += 1;
            }
            if character == '}' {
                depth -= 1;
            }
            if depth == 0 {
                let value = self.source[start..self.position].to_string();
                self.bump();
                return Some(value);
            }
            self.bump();
        }
        self.supported = false;
        None
    }

    /// Split a body on `\\` row separators with optional `[space]` brackets —
    /// upstream `splitEnvironmentRows`.
    fn split_environment_rows(body: &str) -> Vec<&str> {
        static ROW_SPLIT: LazyLock<Regex> =
            LazyLock::new(|| static_regex(r"\\\\(?:\[[^\]\n]*\])?"));
        ROW_SPLIT.split(body).collect()
    }

    /// Body of `\begin{...}` up to the matching `\end` — upstream
    /// `parseEnvironment`.
    #[expect(
        clippy::too_many_lines,
        reason = "the environment ladder mirrors upstream parseEnvironment arm-for-arm; each environment family is one upstream if-block"
    )]
    fn parse_environment(&mut self) -> String {
        static LEADING_GROUP: LazyLock<Regex> = LazyLock::new(|| static_regex(r"^\s*\{[^}]*\}"));
        let Some(environment) = self.read_raw_group() else {
            return String::new();
        };
        let end_marker = format!("\\end{{{environment}}}");
        let Some(end) = self.source[self.position..].find(&end_marker) else {
            self.supported = false;
            return String::new();
        };
        let end = self.position + end;
        let body = self.source[self.position..end].to_string();
        self.position = end + end_marker.len();

        if matches!(
            environment.as_str(),
            "equation" | "equation*" | "displaymath"
        ) {
            return self.render_nested(&body, true).trim().to_string();
        }

        if matches!(
            environment.as_str(),
            "aligned"
                | "align"
                | "align*"
                | "alignedat"
                | "alignat"
                | "alignat*"
                | "gather"
                | "gathered"
                | "multline"
                | "multline*"
                | "split"
        ) {
            let aligned_at = matches!(environment.as_str(), "alignedat" | "alignat" | "alignat*");
            let aligned_body: String = if aligned_at {
                LEADING_GROUP.replace(&body, "").into_owned()
            } else {
                body.clone()
            };
            return Self::split_environment_rows(&aligned_body)
                .iter()
                .map(|row| {
                    let cells: Vec<&str> = row.split('&').collect();
                    let source = if aligned_at {
                        cells
                            .chunks(2)
                            .map(<[&str]>::concat)
                            .collect::<Vec<_>>()
                            .join(" ")
                    } else {
                        cells.concat()
                    };
                    self.render_nested(&source, true).trim().to_string()
                })
                .filter(|row| !row.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
        }

        if matches!(environment.as_str(), "cases" | "cases*") {
            let rows: Vec<Vec<String>> = Self::split_environment_rows(&body)
                .iter()
                .map(|row| {
                    row.split('&')
                        .map(|cell| self.render_nested(cell, false).trim().to_string())
                        .collect::<Vec<String>>()
                })
                .filter(|row| row.iter().any(|cell| !cell.is_empty()))
                .collect();
            return rows
                .iter()
                .enumerate()
                .map(|(index, row)| {
                    static TRAILING_COMMA: LazyLock<Regex> =
                        LazyLock::new(|| static_regex(r",\s*$"));
                    let value = row.first().map_or(String::new(), |value| {
                        TRAILING_COMMA.replace(value, "").into_owned()
                    });
                    let condition = row.get(1).cloned().unwrap_or_default();
                    let delimiter = if index == 0 {
                        "⎧"
                    } else if index == rows.len() - 1 {
                        "⎩"
                    } else {
                        "⎨"
                    };
                    if condition.is_empty() {
                        format!("{delimiter} {value}")
                    } else {
                        let prefix = if case_condition_takes_space(&condition) {
                            " "
                        } else {
                            " if "
                        };
                        format!("{delimiter} {value}{prefix}{condition}")
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
        }

        if matches!(
            environment.as_str(),
            "array"
                | "matrix"
                | "smallmatrix"
                | "pmatrix"
                | "bmatrix"
                | "Bmatrix"
                | "vmatrix"
                | "Vmatrix"
        ) {
            let matrix_body: String = if environment == "array" {
                LEADING_GROUP.replace(&body, "").into_owned()
            } else {
                body
            };
            return self.render_matrix(&environment, &matrix_body);
        }

        self.supported = false;
        body
    }

    /// Lay out a matrix environment's cells in box-drawing columns — upstream
    /// `renderMatrix`.
    fn render_matrix(&mut self, environment: &str, body: &str) -> String {
        let matrix: Vec<Vec<String>> = Self::split_environment_rows(body)
            .iter()
            .map(|row| {
                row.split('&')
                    .map(|cell| self.render_nested(cell, false).trim().to_string())
                    .collect::<Vec<String>>()
            })
            .filter(|row| row.iter().any(|cell| !cell.is_empty()))
            .collect();
        let column_count = matrix.iter().map(Vec::len).max().unwrap_or(0);
        let column_widths: Vec<usize> = (0..column_count)
            .map(|column| {
                matrix
                    .iter()
                    .map(|row| visible_width(row.get(column).map_or("", String::as_str)))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let rows: Vec<String> = matrix
            .iter()
            .map(|row| {
                (0..column_count)
                    .map(|column| {
                        let cell = row.get(column).map_or(String::new(), String::clone);
                        let padding = column_widths[column].saturating_sub(visible_width(&cell));
                        format!("{cell}{}", PROTECTED_SPACE.to_string().repeat(padding))
                    })
                    .collect::<Vec<_>>()
                    .join(" │ ")
            })
            .collect();

        let lines: Vec<String> = if matches!(environment, "array" | "matrix" | "smallmatrix") {
            rows
        } else {
            let Some(delimiter) = (match environment {
                "pmatrix" => Some(("⎛", "⎞", "⎜", "⎟", "⎝", "⎠")),
                "bmatrix" => Some(("⎡", "⎤", "⎢", "⎥", "⎣", "⎦")),
                "Bmatrix" => Some(("⎧", "⎫", "⎨", "⎬", "⎩", "⎭")),
                "vmatrix" => Some(("│", "│", "│", "│", "│", "│")),
                "Vmatrix" => Some(("║", "║", "║", "║", "║", "║")),
                _ => None,
            }) else {
                self.supported = false;
                return rows.join("\n");
            };
            rows.iter()
                .enumerate()
                .map(|(index, row)| {
                    let left = if index == 0 {
                        delimiter.0
                    } else if index == rows.len() - 1 {
                        delimiter.4
                    } else {
                        delimiter.2
                    };
                    let right = if index == 0 {
                        delimiter.1
                    } else if index == rows.len() - 1 {
                        delimiter.5
                    } else {
                        delimiter.3
                    };
                    format!("{left} {row} {right}")
                })
                .collect()
        };

        if lines.len() <= 1 {
            return lines.into_iter().next().unwrap_or_default();
        }
        let index = self.layout_nodes.len();
        self.layout_nodes
            .push(LayoutNode::Matrix { lines, baseline: 0 });
        format!("{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}")
    }

    /// Parse a sub-source with this parser's node list, keeping display mode
    /// only when the caller keeps fraction stacking — upstream `renderNested`.
    fn render_nested(&mut self, source: &str, stack_fractions: bool) -> String {
        let mut nested = LatexParser::new(
            source,
            std::mem::take(&mut self.layout_nodes),
            self.display && stack_fractions,
        );
        let rendered = nested.render();
        self.layout_nodes = nested.layout_nodes;
        if let Some(rendered) = rendered {
            rendered
        } else {
            self.supported = false;
            source.to_string()
        }
    }
}

/// Upstream's `^(?:if|when|for|otherwise)\b/i` case-condition check: the
/// keyword must end at a non-word character (letters, digits, or underscore).
fn case_condition_takes_space(condition: &str) -> bool {
    const KEYWORDS: [&str; 4] = ["if", "when", "for", "otherwise"];
    let lower = condition.to_lowercase();
    KEYWORDS.iter().any(|keyword| {
        lower.starts_with(keyword)
            && !lower[keyword.len()..]
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    })
}

/// Render a basic LaTeX math expression as terminal-friendly Unicode text, or
/// `None` when the expression contains unsupported or malformed syntax —
/// upstream `renderLatex(latex, options?)`.
///
/// `display` selects display-style layout (upstream `{ display: true }`),
/// where fractions stack over rule lines and limits sit beneath their
/// operators; inline mode renders linearly. Callers fall back to the raw
/// source on `None` — markdown's documented degradation.
#[must_use]
pub fn render_latex(latex: &str, display: bool) -> Option<String> {
    let mut parser = LatexParser::new(latex, Vec::new(), display);
    let rendered = parser.render()?;
    if parser.layout_nodes.is_empty() {
        return Some(rendered.replace(PROTECTED_SPACE, " "));
    }
    let lines = render_layout(&rendered, &parser.layout_nodes).lines;
    let indentation = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min();
    let Some(indentation) = indentation else {
        // Every line blank: upstream slices at `Math.min()` of an empty set,
        // which empties every line.
        return Some(String::new());
    };
    Some(
        lines
            .iter()
            .map(|line| {
                let start = line
                    .char_indices()
                    .nth(indentation)
                    .map_or(line.len(), |(index, _)| index);
                line[start..].trim_end()
            })
            .collect::<Vec<_>>()
            .join("\n")
            .trim_end()
            .replace(PROTECTED_SPACE, " "),
    )
}
