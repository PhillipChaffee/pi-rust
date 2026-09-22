//! Boundary tests binding the LaTeX renderer branches the upstream suite
//! leaves untested (upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`,
//! [#50](https://github.com/PhillipChaffee/pi-rust/issues/50)): the spacing
//! sentinels, escaped punctuation and size commands, the five matrix
//! delimiter families and the array column spec, the gather/multline/cases*
//! environments, single-row matrices without layout markers, swapped and
//! duplicated operator scripts, the named-operator spacing scans, negative
//! spaces, the matrix-dot fall-through for non-matrix trailing markers, and
//! the unsupported/malformed rejections the parser reports `None` for. The
//! expected values are upstream `renderLatex` outputs, run at the pin.

use pi_tui::latex::render_latex;

fn assert_cases(cases: &[(&str, &str)]) {
    for (source, expected) in cases {
        assert_eq!(
            render_latex(source, false).as_deref(),
            Some(*expected),
            "source: {source:?}"
        );
    }
}

#[test]
fn negative_and_positive_spacing_commands() {
    assert_cases(&[
        (r"a\!b", "ab"),
        (r"a\quad b", "a b"),
        (r"a\enspace b", "a b"),
        // The negative space trims the accumulated text only; the space
        // before the following token stays.
        (r"a\negthinspace b", "a b"),
    ]);
}

#[test]
fn escaped_punctuation_and_special_commands() {
    assert_cases(&[
        (r"50\% + 100\#", "50% + 100#"),
        (r"a\|b", "a‖b"),
        (r"a~b", "a b"),
        (r"a&b", "ab"),
        (r"a\\b", "a\nb"),
        // The `[4pt]` row bracket is only stripped inside environment bodies;
        // a raw `\\` outside one emits the newline and leaves the bracket.
        (r"a\\[4pt]b", "a\n[4pt]b"),
    ]);
}

#[test]
fn size_commands_drop_but_leave_their_delimiter_text() {
    assert_cases(&[(r"\big( x \big)", "( x )")]);
}

#[test]
fn symbol_sequences_without_relation_commands_get_no_spaces() {
    assert_cases(&[(r"\alpha\beta", "αβ")]);
}

#[test]
fn named_operators_space_around_adjacent_letters_and_numbers() {
    assert_cases(&[(r"\alpha\sin", "α sin"), (r"\sin\alpha", "sin α")]);
}

#[test]
fn the_not_fallback_combines_the_first_character_with_the_negation_slash() {
    assert_cases(&[(r"x \not\ast", "x ∗̸"), (r"x\not\sim", "x ≁")]);
}

#[test]
fn invisible_and_size_delimiters_drop() {
    assert_cases(&[(r"x \left. y", "x y")]);
}

#[test]
fn wrappers_pass_their_argument_through() {
    assert_cases(&[
        (r"\smash{x}", "x"),
        (r"\mathfrak{R}", "R"),
        (r"\mathbb{x}", "x"),
        (r"\hat{xy}", "hat(xy)"),
        (r"\substack{a\\b}", "a\nb"),
    ]);
}

#[test]
fn operatorname_without_a_star_stays_inline() {
    assert_cases(&[(r"\operatorname{snr}", "snr"), (r"a\pod{n}", "a (n)")]);
}

#[test]
fn indexed_roots_beyond_the_fourth_degree_restate_to_a_superscript() {
    assert_cases(&[(r"x\sqrt[5]{y}", "x⁵√y")]);
}

#[test]
fn the_five_matrix_delimiter_families_and_the_array_column_spec() {
    assert_cases(&[
        (r"\begin{bmatrix}1&2\end{bmatrix}", "⎡ 1 │ 2 ⎤"),
        (r"\begin{Bmatrix}1\\2\end{Bmatrix}", "⎧ 1 ⎫\n⎩ 2 ⎭"),
        (r"\begin{vmatrix}1\\2\end{vmatrix}", "│ 1 │\n│ 2 │"),
        (r"\begin{Vmatrix}1\\2\end{Vmatrix}", "║ 1 ║\n║ 2 ║"),
        // The array's `{cc}` column spec strips; the trailing padding lands
        // after the final trim because the protected spaces turn into real
        // spaces last.
        (
            r"\begin{array}{cc}1&22\\333&4\end{array}",
            "1   │ 22\n333 │ 4 ",
        ),
        // A single-row delimiter matrix skips the layout pass entirely.
        (r"\begin{pmatrix}1&2\end{pmatrix}", "⎛ 1 │ 2 ⎞"),
    ]);
}

#[test]
fn matrix_adjacent_text_keeps_its_separating_space() {
    // The trailing " y" after a matrix marker keeps its leading space, and
    // the text before the marker keeps its trailing one.
    assert_cases(&[
        (r"x\begin{matrix}1\\2\end{matrix} y", "x1 y\n 2"),
        (r"x \begin{matrix}1\\2\end{matrix}", "x 1\n  2"),
    ]);
}

#[test]
fn the_matrix_dot_falls_through_after_a_non_matrix_marker() {
    // A stacked fraction's trailing dot is ordinary text: it lands on the
    // rule's row, not inside the fraction.
    assert_eq!(
        render_latex(r"x\frac12.", true).as_deref(),
        Some("  1\nx ─ .\n  2"),
        "source: x\\frac12."
    );
}

#[test]
fn display_mode_places_a_lower_only_limit_inline() {
    assert_eq!(
        render_latex(r"\sum_i x", true).as_deref(),
        Some("∑ x\ni"),
        "source: \\sum_i x"
    );
}

#[test]
fn operator_scripts_parse_in_either_order() {
    assert_cases(&[(r"\sum^a_b", "∑_bᵃ")]);
}

#[test]
fn gather_multline_and_starred_cases_environments_align_rows() {
    assert_cases(&[
        (r"\begin{gather}a\\b\end{gather}", "a\nb"),
        (r"\begin{multline*}a\\b\end{multline*}", "a\nb"),
        (r"\begin{cases}a\end{cases}", "⎧ a"),
        (r"\begin{cases*}a & x\end{cases*}", "⎧ a if x"),
    ]);
}

#[test]
fn nested_parse_failures_propagate_to_none() {
    let cases = [
        // The boxed argument's sub-parse hits an unknown command.
        r"\boxed{\unknown{x}}",
        // The sqrt optional bracket never closes.
        r"\sqrt[n{y}",
        // The environment name group is missing.
        r"\begin x",
        // An `\end` with no open environment.
        r"\end{matrix}",
        // The environment family is not supported.
        r"\begin{unknown}x\end{unknown}",
        // Two lower scripts on one operator.
        r"\lim_{n}_{m}",
    ];
    for source in cases {
        assert_eq!(render_latex(source, false), None, "source: {source:?}");
    }
}
