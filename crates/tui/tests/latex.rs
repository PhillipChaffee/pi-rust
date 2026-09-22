//! Port of `packages/tui/test/latex.test.ts` 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`). The upstream `describe`
//! groups become test functions; `defineCases` becomes the [`assert_cases`]
//! helper. One restatement: the `renders <JSON.stringify(source)>` test names
//! carry no assertion weight here, so each helper reports the raw source in
//! its failure message instead.

use pi_tui::latex::render_latex;

type LatexCase<'a> = (&'a str, &'a str);

fn assert_cases(cases: &[LatexCase<'_>]) {
    for (source, expected) in cases {
        assert_eq!(
            render_latex(source, false).as_deref(),
            Some(*expected),
            "source: {source:?}"
        );
    }
}

#[test]
fn jacobian_conjecture_session_using_dollar_delimiters() {
    assert_cases(&[
        (r"\mathbb{C}^3 \to \mathbb{C}^3", "ℂ³ → ℂ³"),
        (
            r"\{3x+2y,\; 27x^2-4z-1,\; x(x-1)(x+1)\} \quad\Rightarrow\quad x \in \{0, \pm 1\},",
            "{3x+2y, 27x²-4z-1, x(x-1)(x+1)} ⇒ x ∈ {0, ± 1},",
        ),
        (r"F_1 = -\frac{1}{4x^2}.", "F₁ = -1/(4x²)."),
        ("-2", "-2"),
        ("(0,0,-1/4)", "(0,0,-1/4)"),
        ("(1,-3/2,13/2)", "(1,-3/2,13/2)"),
        ("(1,1,1)", "(1,1,1)"),
        ("(2,1,0)", "(2,1,0)"),
        ("(-1/4, 0, 0)", "(-1/4, 0, 0)"),
        (
            r"\{(0,0,-1/4), (1,-3/2,13/2), (-1,3/2,13/2)\}",
            "{(0,0,-1/4), (1,-3/2,13/2), (-1,3/2,13/2)}",
        ),
        ("(2,1,1)", "(2,1,1)"),
        ("(7/3,-2/5,11/7)", "(7/3,-2/5,11/7)"),
        (r"\{y - p(x),\; q(x)\}", "{y - p(x), q(x)}"),
        (r"\deg q = 3", "deg q = 3"),
        (
            r"[\mathbb{C}(x,y,z):\mathbb{C}(F_1,F_2,F_3)] = 3",
            "[ℂ(x,y,z):ℂ(F₁,F₂,F₃)] = 3",
        ),
        ("u = 1+xy", "u = 1+xy"),
        ("G = u^2 z + y^2(4+3xy)", "G = u² z + y²(4+3xy)"),
        ("F_1 = uG", "F₁ = uG"),
        ("F_2 = y + 3xG", "F₂ = y + 3xG"),
        ("x=0", "x = 0"),
        ("F_2 = F_3 = 0", "F₂ = F₃ = 0"),
        ("xy = -3/2", "xy = -3/2"),
        ("x^2 z = 13/2", "x² z = 13/2"),
        (r"\mathbb{C}^*", "ℂ^*"),
        (
            r"s \mapsto (s,\, -\tfrac{3}{2s},\, \tfrac{13}{2s^2})",
            "s ↦ (s, -3/(2s), 13/(2s²))",
        ),
        ("X", "X"),
        (r"p_\pm", "p_±"),
        (
            "F(-x,-y,z) = (F_1, -F_2, -F_3)",
            "F(-x,-y,z) = (F₁, -F₂, -F₃)",
        ),
        ("p_0", "p₀"),
        (r"s \to \infty", "s → ∞"),
        ("(0,0,0)", "(0,0,0)"),
        (r"\Rightarrow", "⇒"),
        (r"\ge 2", "≥ 2"),
        (r"\ge 3", "≥ 3"),
        ("1", "1"),
        (r"\mathrm{diag}(-1/2,1,1)", "diag(-1/2,1,1)"),
        ("4+3xy", "4+3xy"),
    ]);
}

#[test]
fn satellite_calculation_session_using_bracket_delimiters() {
    assert_cases(&[
        (
            r"E \approx \frac{0.1\ \text{lux}}{100\ \text{lm/W}} = 0.001\ \text{W/m}^2",
            "E ≈ (0.1 lux)/(100 lm/W) = 0.001 W/m²",
        ),
        (
            r"\boxed{1\ \text{milliwatt per square metre}}",
            "[1 milliwatt per square metre]",
        ),
        (
            r"5\ \text{km}^2 = 5{,}000{,}000\ \text{m}^2",
            "5 km² = 5,000,000 m²",
        ),
        (
            r"P_{\text{light}} = 0.001 \times 5{,}000{,}000
= \boxed{5{,}000\ \text{W}}",
            "P_light = 0.001 × 5,000,000 = [5,000 W]",
        ),
        (
            r"P_{\text{electric}} = 5\ \text{kW} \times 0.2
= \boxed{1\ \text{kW}}",
            "P_electric = 5 kW × 0.2 = [1 kW]",
        ),
        (
            r"\pi(2.5\ \text{km})^2 = 19.6\ \text{km}^2",
            "π(2.5 km)² = 19.6 km²",
        ),
        (
            r"0.001\ \text{W/m}^2 \times 19.6 \times 10^6\ \text{m}^2
\approx \boxed{20\ \text{kW optical}}",
            "0.001 W/m² × 19.6 × 10⁶ m² ≈ [20 kW optical]",
        ),
        (
            r"1\ \text{kW} \times \frac{1}{3600}\ \text{hour}
= \boxed{0.28\ \text{Wh}}",
            "1 kW × 1/3600 hour = [0.28 Wh]",
        ),
    ]);
}

#[test]
fn jacobian_conjecture_sessions_using_parenthesis_and_bracket_delimiters() {
    assert_cases(&[
        (
            r"\det\!\left(\frac{\partial(F_1,F_2,F_3)}{\partial(x,y,z)}\right)=-2.",
            "det((∂(F₁,F₂,F₃))/(∂(x,y,z))) = -2.",
        ),
        (
            r"\begin{aligned}
F(0,0,-\tfrac14)&=(-\tfrac14,0,0),\\
F(1,-\tfrac32,\tfrac{13}2)&=(-\tfrac14,0,0),\\
F(-1,\tfrac32,\tfrac{13}2)&=(-\tfrac14,0,0).
\end{aligned}",
            "F(0,0,-1/4) = (-1/4,0,0),\nF(1,-3/2,13/2) = (-1/4,0,0),\nF(-1,3/2,13/2) = (-1/4,0,0).",
        ),
        ("F=(F_1,F_2,F_3)", "F = (F₁,F₂,F₃)"),
        ("F", "F"),
        ("3", "3"),
    ]);
}

#[test]
fn jacobian_matrix_session_using_dollar_delimiters() {
    assert_cases(&[
        (
            r"J = \begin{pmatrix}
\frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} & \frac{\partial f_1}{\partial z} \\
\frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y} & \frac{\partial f_2}{\partial z} \\
\frac{\partial f_3}{\partial x} & \frac{\partial f_3}{\partial y} & \frac{\partial f_3}{\partial z}
\end{pmatrix}",
            "J = ⎛ (∂ f₁)/(∂ x) │ (∂ f₁)/(∂ y) │ (∂ f₁)/(∂ z) ⎞\n    ⎜ (∂ f₂)/(∂ x) │ (∂ f₂)/(∂ y) │ (∂ f₂)/(∂ z) ⎟\n    ⎝ (∂ f₃)/(∂ x) │ (∂ f₃)/(∂ y) │ (∂ f₃)/(∂ z) ⎠",
        ),
        (
            r"\begin{aligned}
f_1 &= (1+xy)^3 z + y^2(1+xy)(4+3xy) \\
f_2 &= y + 3x(1+xy)^2 z + 3xy^2(4+3xy) \\
f_3 &= 2x - 3x^2y - x^3z
\end{aligned}",
            "f₁ = (1+xy)³ z + y²(1+xy)(4+3xy)\nf₂ = y + 3x(1+xy)² z + 3xy²(4+3xy)\nf₃ = 2x - 3x²y - x³z",
        ),
        ("x, y, z", "x, y, z"),
        ("(x, y, z)", "(x, y, z)"),
        (r"(0,\; 0,\; -\tfrac14)", "(0, 0, -1/4)"),
        (r"(-\tfrac14,\; 0,\; 0)", "(-1/4, 0, 0)"),
        (r"(1,\; -\tfrac32,\; \tfrac{13}{2})", "(1, -3/2, 13/2)"),
        (r"(-1,\; \tfrac32,\; \tfrac{13}{2})", "(-1, 3/2, 13/2)"),
        (r"(-\frac14, 0, 0)", "(-1/4, 0, 0)"),
        (r"F: \mathbb{C}^3 \to \mathbb{C}^3", "F: ℂ³ → ℂ³"),
        (
            r"F(0,0,-\tfrac14) = F(1,-\tfrac32,\tfrac{13}{2}) = F(-1,\tfrac32,\tfrac{13}{2}) = (-\tfrac14, 0, 0)",
            "F(0,0,-1/4) = F(1,-3/2,13/2) = F(-1,3/2,13/2) = (-1/4, 0, 0)",
        ),
        (r"\mathbb{C}^3", "ℂ³"),
        (
            r"\begin{aligned}
f_1 &= \frac{f_1^{\text{ut}}(u,t)}{x^2}, \quad
f_2 = \frac{f_2^{\text{ut}}(u,t)}{x}, \quad
f_3 = x\,(2 - 3u - t)
\end{aligned}",
            "f₁ = (f₁ᵘᵗ(u,t))/(x²), f₂ = (f₂ᵘᵗ(u,t))/x, f₃ = x (2 - 3u - t)",
        ),
        (r"\det J_F", "det J_F"),
        (r"(-\tfrac14, 0, 0)", "(-1/4, 0, 0)"),
        ("u = xy", "u = xy"),
        ("t = x^2z", "t = x²z"),
        (r"x \neq 0", "x ≠ 0"),
        (r"f_1^{\text{ut}}, f_2^{\text{ut}}", "f₁ᵘᵗ, f₂ᵘᵗ"),
        ("u,t", "u,t"),
        ("x", "x"),
        ("x, x^2", "x, x²"),
        (r"\mathbb{C}^n \to \mathbb{C}^n", "ℂⁿ → ℂⁿ"),
        (r"n \geq 2", "n ≥ 2"),
        (r"\mathbb{P}^3", "ℙ³"),
    ]);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the stress-test group carries 16 upstream cases verbatim; splitting the array would break the 1:1 transcription"
)]
fn extended_formulas_from_a_renderer_stress_test_session() {
    // The contour-integral case's continuation lines indent 40 cells; the
    // rest of the group is flat (source, expected) pairs.
    let pad = " ".repeat(40);
    let contour_source = r"f(z)
=
\frac{1}{2\pi i}
\oint_{\gamma}
\frac{f(\zeta)}{\zeta-z}\,d\zeta,
\qquad
\det\!\begin{pmatrix}
\lambda-a & -b & 0\\
-c & \lambda-d & -e\\
0 & -f & \lambda-g
\end{pmatrix}
=0.";
    let contour_expected = [
        "f(z) = 1/(2π i) ∮_γ (f(ζ))/(ζ-z) dζ, det⎛ λ-a │ -b  │ 0   ⎞ = 0.".to_string(),
        format!("{pad}⎜ -c  │ λ-d │ -e  ⎟"),
        format!("{pad}⎝ 0   │ -f  │ λ-g ⎠"),
    ]
    .join("\n");
    assert_eq!(
        render_latex(contour_source, false).as_deref(),
        Some(contour_expected.as_str()),
        "source: {contour_source:?}"
    );

    assert_cases(&[
        (r"e^{i\pi}+1=0", "e^(iπ)+1 = 0"),
        (
            r"\boxed{
\mathcal{Z}(\beta)
=
\int_{\mathcal M}
\exp\!\left(
-\beta\left[
\frac12 g^{ij}(x)\,\partial_i\phi\,\partial_j\phi
+V(\phi)
\right]\right)
\mathcal D\phi
}",
            "[Z(β) = ∫_M exp( -β[ 1/2 gⁱʲ(x) ∂ᵢϕ ∂ⱼϕ +V(ϕ) ]) Dϕ]",
        ),
        (
            r"\begin{aligned}
\nabla_\mu T^{\mu\nu}
&=
\frac{1}{\sqrt{-g}}
\partial_\mu\!\left(\sqrt{-g}\,T^{\mu\nu}\right)
+\Gamma^\nu_{\mu\lambda}T^{\mu\lambda}
=0, \\[4pt]
R_{\mu\nu}-\frac12 Rg_{\mu\nu}+\Lambda g_{\mu\nu}
&=
\frac{8\pi G}{c^4}T_{\mu\nu}.
\end{aligned}",
            "∇_μ T^(μν) = 1/(√(-g)) ∂_μ(√(-g) T^(μν)) +Γ^ν_(μλ)T^(μλ) = 0,\nR_(μν)-1/2 Rg_(μν)+Λ g_(μν) = (8π G)/(c⁴)T_(μν).",
        ),
        (
            r"\Psi(x,t)=
\sum_{n=1}^{\infty}
\underbrace{
c_n
\sqrt{\frac{2}{L}}
\sin\!\left(\frac{n\pi x}{L}\right)
}_{\text{spatial eigenmode}}
\exp\!\left(-\frac{i\hbar n^2\pi^2}{2mL^2}t\right),
\qquad
|\Psi(x,t)|^2
=
\begin{cases}
\Psi^\ast\Psi, & 0<x<L,\\
0, & \text{otherwise}.
\end{cases}",
            "Ψ(x,t) = ∑ₙ₌₁^∞ cₙ √(2/L) sin((nπ x)/L)_(spatial eigenmode) exp(-(iℏ n²π²)/(2mL²)t), |Ψ(x,t)|² = ⎧ Ψ^∗Ψ if 0 < x < L,\n⎩ 0 otherwise.",
        ),
        (
            r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
            "x = (-b±√(b²-4ac))/(2a)",
        ),
        (
            r"\int_0^\infty e^{-x^2}\,dx=\frac{\sqrt{\pi}}{2}",
            "∫₀^∞ e^(-x²) dx = (√π)/2",
        ),
        (
            r"e^{i\theta}=\cos\theta+i\sin\theta",
            "e^(iθ) = cos θ+i sin θ",
        ),
        (
            r"\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}",
            "∑ₙ₌₁^∞1/(n²) = π²/6",
        ),
        (r"\lim_{x\to 0}\frac{\sin x}{x}=1", "lim[x→0] (sin x)/x = 1"),
        (
            r"\lim_{n\to\infty}
\left(1+\frac{1}{n}\right)^n=e",
            "lim[n→∞] (1+1/n)ⁿ = e",
        ),
        (
            r"\int_0^1 \frac{x^2}{1+x^3}\,dx
=\frac{1}{3}\ln 2",
            "∫₀¹ x²/(1+x³) dx = 1/3 ln 2",
        ),
        (
            r"\sum_{k=1}^{n}\frac{k}{k+1}
=n+1-H_{n+1}",
            "∑ₖ₌₁ⁿk/(k+1) = n+1-Hₙ₊₁",
        ),
        (
            r"\frac{
  \displaystyle \frac{x^2+1}{x-1}
  -
  \displaystyle \frac{2x}{x+1}
}{
  \displaystyle \frac{x}{x^2-1}
}",
            "((x²+1)/(x-1) - 2x/(x+1))/(x/(x²-1))",
        ),
        (
            r"\lim_{x\to 0}
\frac{
  \displaystyle \frac{\sin x}{x}-1
}{
  \displaystyle \frac{e^x-1}{x}-1
}
=0",
            "lim[x→0] ((sin x)/x-1)/((eˣ-1)/x-1) = 0",
        ),
        (
            r"\frac{
  1+\displaystyle\frac{1}{1+\frac{1}{x}}
}{
  1-\displaystyle\frac{1}{1-\frac{1}{x}}
}",
            "(1+1/(1+1/x))/(1-1/(1-1/x))",
        ),
        (
            r"\sum_{n=1}^{\infty}
\frac{
  \displaystyle \frac{1}{n}-\frac{1}{n+1}
}{
  \displaystyle 1+\frac{1}{n^2}
}",
            "∑ₙ₌₁^∞ (1/n-1/(n+1))/(1+1/(n²))",
        ),
    ]);
}

#[test]
fn renders_common_symbols_roots_sums_and_integrals() {
    assert_eq!(
        render_latex(
            r"\sum_{i=0}^n \alpha_i + \int_0^\infty e^{-x^2}\,dx = \sqrt{\pi}",
            false
        ),
        Some("∑ᵢ₌₀ⁿ αᵢ + ∫₀^∞ e^(-x²) dx = √π".to_string())
    );
}

#[test]
fn renders_common_accents_and_binomial_notation() {
    // The accent glyphs ride as combining marks after their base character —
    // upstream's expected literal is the decomposed byte form (y + U+0302),
    // which terminals display as the precomposed ŷ.
    assert_eq!(
        render_latex(r"\binom{n}{k}+\vec{x}+\hat{y}+\overline{AB}", false),
        Some("(n choose k)+x\u{20d7}+y\u{0302}+overline(AB)".to_string())
    );
}

#[test]
fn renders_extended_symbols_and_negated_relations() {
    assert_eq!(
        render_latex(
            r"\epsilon+\varepsilon+\varsigma+\varkappa+\oplus+\otimes+\therefore+\because",
            false
        ),
        Some("ϵ+ε+ς+ϰ+⊕+⊗+∴+∵".to_string())
    );
    assert_eq!(
        render_latex(r"A\not\subseteq B,\quad x\not\in X", false),
        Some("A ⊈ B, x ∉ X".to_string())
    );
}

#[test]
fn renders_relational_algebra_join_operators() {
    assert_eq!(
        render_latex(r"R\bowtie S,\quad R\Join S", false),
        Some("R ⋈ S, R ⋈ S".to_string())
    );
    assert_eq!(
        render_latex(r"R\ltimes S,\quad R\rtimes S", false),
        Some("R ⋉ S, R ⋊ S".to_string())
    );
    assert_eq!(
        render_latex(
            r"R\leftouterjoin S,\quad R\rightouterjoin S,\quad R\fullouterjoin S",
            false
        ),
        Some("R ⟕ S, R ⟖ S, R ⟗ S".to_string())
    );
}

#[test]
fn renders_delimiter_commands_and_invisible_delimiters() {
    assert_eq!(
        render_latex(
            r"\lvert{x}\rvert+\lVert{v}\rVert+\left.\frac{dy}{dx}\right|_{x=0}",
            false
        ),
        Some("|x|+‖v‖+dy/(dx)|ₓ₌₀".to_string())
    );
    assert_eq!(
        render_latex(r"\left\lbrace x \middle| x>0 \right\rbrace", false),
        Some("{ x | x > 0 }".to_string())
    );
}

#[test]
fn renders_named_modular_overlaid_and_underlaid_operators() {
    assert_eq!(
        render_latex(r"\operatorname*{arg\,max}_{x\in X} f(x)", false),
        Some("arg max[x∈X] f(x)".to_string())
    );
    assert_eq!(
        render_latex(r"a\bmod n,\quad a\equiv b\pmod n", false),
        Some("a mod n, a ≡ b (mod n)".to_string())
    );
    assert_eq!(
        render_latex(r"\overset{!}{=}+\underset{n}{x}+\stackrel{def}{=}", false),
        Some("=^!+xₙ+=ᵈᵉᶠ".to_string())
    );
}

#[test]
fn renders_indexed_roots_and_additional_accents_and_wrappers() {
    assert_eq!(
        render_latex(
            r"\sqrt[2]{x}+\sqrt[3]{x}+\sqrt[4]{x}+\sqrt[n]{x}+\sqrt[k]{x+1}",
            false
        ),
        Some("√x+∛x+∜x+ⁿ√x+ᵏ√(x+1)".to_string())
    );
    // The accent glyphs ride as combining marks after their base character —
    // upstream's expected literal is the decomposed byte form (x + U+0301,
    // y + U+0300), which terminals display as the accented letters.
    assert_eq!(
        render_latex(
            r"\acute{x}+\grave{y}+\widehat{xyz}+\overrightarrow{AB}",
            false
        ),
        Some("x\u{0301}+y\u{0300}+widehat(xyz)+overrightarrow(AB)".to_string())
    );
    assert_eq!(
        render_latex(r"\textnormal{hello}+\mbox{world}+\boldsymbol{x}", false),
        Some("hello+world+x".to_string())
    );
}

#[test]
fn renders_additional_display_environments() {
    assert_eq!(
        render_latex(
            r"\begin{equation}\begin{split}a&=b\\&=c\end{split}\end{equation}",
            false
        ),
        Some("a = b\n= c".to_string())
    );
    assert_eq!(
        render_latex(
            r"\begin{alignedat}{2}a&=b&\quad c&=d\\e&=f&g&=h\end{alignedat}",
            false
        ),
        Some("a = b c = d\ne = f g = h".to_string())
    );
}

#[test]
fn uses_natural_case_conditions_and_aligns_matrix_columns() {
    assert_eq!(
        render_latex(
            r"\begin{cases}a & x<0 \\ b & \text{if }x=0 \\ c & \text{otherwise}\end{cases}",
            false
        ),
        Some("⎧ a if x < 0\n⎨ b if x = 0\n⎩ c otherwise".to_string())
    );
    assert_eq!(
        render_latex(r"\begin{pmatrix}1&200\\3000&4\end{pmatrix}", false),
        Some("⎛ 1    │ 200 ⎞\n⎝ 3000 │ 4   ⎠".to_string())
    );
}

#[test]
fn composes_matrices_with_fractions_and_adjacent_matrices() {
    assert_eq!(
        render_latex(
            r"R\left(\frac{\pi}{4}\right)
=
\begin{pmatrix}
\frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2}\\
\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}
\end{pmatrix}.",
            true
        ),
        Some("   π\nR( ─ ) = ⎛ (√2)/2 │ -(√2)/2 ⎞\n   4     ⎝ (√2)/2 │ (√2)/2  ⎠.".to_string())
    );
    assert_eq!(
        render_latex(
            r"\mathbf w
=
R\left(\frac{\pi}{4}\right)
\begin{pmatrix}1\\0\end{pmatrix}
=
\begin{pmatrix}\frac{\sqrt{2}}{2}\\\frac{\sqrt{2}}{2}\end{pmatrix}.",
            true
        ),
        Some("       π\nw = R( ─ ) ⎛ 1 ⎞ = ⎛ (√2)/2 ⎞\n       4   ⎝ 0 ⎠   ⎝ (√2)/2 ⎠.".to_string())
    );
    assert_eq!(
        render_latex(
            r"A\mathbf e_1=\begin{pmatrix}\pi\\0\end{pmatrix},\qquad A\mathbf e_2=\begin{pmatrix}0\\\frac{1}{\pi}\end{pmatrix}.",
            true
        ),
        Some("Ae₁ = ⎛ π ⎞, Ae₂ = ⎛ 0   ⎞\n      ⎝ 0 ⎠        ⎝ 1/π ⎠.".to_string())
    );
    assert_eq!(
        render_latex(
            r"\sum_{i=0}^n x_i=\begin{pmatrix}a&b\\c&d\end{pmatrix}.",
            true
        ),
        Some(" n\n ∑  xᵢ = ⎛ a │ b ⎞\ni=0      ⎝ c │ d ⎠.".to_string())
    );
}

#[test]
fn normalizes_relation_multiplication_and_named_operator_spacing() {
    for source in ["x=y", "x =y", "x=\ny", "x\n=\ny"] {
        assert_eq!(
            render_latex(source, false).as_deref(),
            Some("x = y"),
            "source: {source:?}"
        );
    }
    assert_eq!(render_latex("x_{i=0}", false), Some("xᵢ₌₀".to_string()));
    assert_eq!(render_latex(r"x\neq0", false), Some("x ≠ 0".to_string()));
    assert_eq!(render_latex(r"A\to B", false), Some("A → B".to_string()));
    assert_eq!(
        render_latex(r"\pi\cdot\frac{1}{\pi}", false),
        Some("π · 1/π".to_string())
    );
    assert_eq!(
        render_latex(r"\sin\theta", false),
        Some("sin θ".to_string())
    );
    assert_eq!(render_latex(r"\sin^2 x", false), Some("sin² x".to_string()));
    assert_eq!(
        render_latex(r"-\sin\theta", false),
        Some("-sin θ".to_string())
    );
    assert_eq!(
        render_latex(r"i\sin\theta", false),
        Some("i sin θ".to_string())
    );
    assert_eq!(render_latex(r"\det(A)", false), Some("det(A)".to_string()));
}

#[test]
fn treats_a_backslash_followed_by_a_line_ending_as_control_space() {
    let source = "\\boxed{
(1,1,1),\\ (1,1,2),\\ (1,2,5),\\ (1,5,13),\\ (2,5,29),\\
(1,13,34),\\ (1,34,89)
}.";
    assert_eq!(
        render_latex(source, true),
        Some("[(1,1,1), (1,1,2), (1,2,5), (1,5,13), (2,5,29), (1,13,34), (1,34,89)].".to_string())
    );
    assert_eq!(render_latex("a\\\r\nb", false), Some("a b".to_string()));
}

#[test]
fn stacks_operator_limits_in_display_mode() {
    assert_eq!(
        render_latex(r"\sum_{i=0}^n x_i", true),
        Some(" n\n ∑  xᵢ\ni=0".to_string())
    );
    assert_eq!(
        render_latex(r"\min_{x\in X} f(x)", true),
        Some("min f(x)\nx∈X".to_string())
    );
    assert_eq!(
        render_latex(r"\operatorname*{arg\,max}_{x\in X} f(x)", true),
        Some("arg max f(x)\n  x∈X".to_string())
    );
    assert_eq!(
        render_latex(r"\int\nolimits_0^1 f(x)\,dx", true),
        Some("∫₀¹ f(x) dx".to_string())
    );
    assert_eq!(
        render_latex(r"\int\limits_0^1 f(x)\,dx", true),
        Some("1\n∫ f(x) dx\n0".to_string())
    );
}

#[test]
fn uses_the_middle_brace_for_intermediate_case_rows() {
    assert_eq!(
        render_latex(
            r"\begin{cases}a & x<0 \\ b & x=0 \\ c & x>0\end{cases}",
            false
        ),
        Some("⎧ a if x < 0\n⎨ b if x = 0\n⎩ c if x > 0".to_string())
    );
}

#[test]
fn stacks_fractions_in_display_mode() {
    assert_eq!(
        render_latex(r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}", true),
        Some("    -b±√(b²-4ac)\nx = ────────────\n         2a".to_string())
    );
    assert_eq!(
        render_latex(r"\frac{x^2+1}{x-1}", true),
        Some("x²+1\n────\nx-1".to_string())
    );
    assert_eq!(
        render_latex("\\frac{1}\n{2}", true),
        Some("1\n─\n2".to_string())
    );
}

#[test]
fn keeps_nested_display_fractions_linear() {
    let cases: [(&str, &str); 3] = [
        (
            r"\frac{\frac{x^2+1}{x-1}-\frac{2x}{x+1}}{\frac{x}{x^2-1}}",
            "(x²+1)/(x-1)-2x/(x+1)\n─────────────────────\n      x/(x²-1)",
        ),
        (
            r"\lim_{x\to 0}\frac{\frac{\sin x}{x}-1}{\frac{e^x-1}{x}-1}=0",
            "     (sin x)/x-1\nlim  ─────────── = 0\nx→0  (eˣ-1)/x-1",
        ),
        (
            r"\frac{1+\frac{1}{1+\frac{1}{x}}}{1-\frac{1}{1-\frac{1}{x}}}",
            "1+1/(1+1/x)\n───────────\n1-1/(1-1/x)",
        ),
    ];
    for (source, expected) in cases {
        assert_eq!(
            render_latex(source, true).as_deref(),
            Some(expected),
            "source: {source:?}"
        );
    }
}

#[test]
fn keeps_fractions_linear_in_scripts_and_text_style_fractions() {
    assert_eq!(
        render_latex(r"e^{\frac{1}{2}}", true),
        Some("e^(1/2)".to_string())
    );
    assert_eq!(render_latex(r"\tfrac{1}{2}", true), Some("1/2".to_string()));
}

#[test]
fn returns_none_for_unsupported_commands() {
    assert_eq!(render_latex(r"x + \unknown{y}", false), None);
}

#[test]
fn returns_none_for_malformed_groups_and_environments() {
    let malformed = [r"\frac{1}{x", "x}", r"\begin{matrix}1 & 2", "x\\"];
    for source in malformed {
        assert_eq!(render_latex(source, false), None, "source: {source:?}");
    }
}
