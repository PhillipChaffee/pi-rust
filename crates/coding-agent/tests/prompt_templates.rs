//! Upstream `test/prompt-templates.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::literal_string_with_formatting_args,
    reason = "the substitution grammar's placeholders carry braced patterns, upstream's dollar-brace default and slice shapes"
)]

use std::path::Path;

use pi_coding_agent::prompt_templates::{
    PromptTemplate, expand_prompt_template, load_prompt_templates, parse_command_args,
    substitute_args,
};
use pi_coding_agent::source_info::{SourceInfo, SourceOrigin, SourceScope};

fn template_named(name: &str, content: &str) -> PromptTemplate {
    PromptTemplate {
        name: name.to_string(),
        description: "test".to_string(),
        argument_hint: None,
        content: content.to_string(),
        source_info: SourceInfo {
            path: "/tmp/arg-test.md".to_string(),
            source: "local".to_string(),
            scope: SourceScope::Temporary,
            origin: SourceOrigin::TopLevel,
            base_dir: None,
        },
        file_path: "/tmp/arg-test.md".to_string(),
    }
}

// === substituteArgs =========================================================

#[test]
fn replaces_arguments_with_all_args_joined() {
    assert_eq!(
        substitute_args("Test: $ARGUMENTS", &["a".into(), "b".into(), "c".into()]),
        "Test: a b c"
    );
}

#[test]
fn replaces_at_with_all_args_joined() {
    assert_eq!(
        substitute_args("Test: $@", &["a".into(), "b".into(), "c".into()]),
        "Test: a b c"
    );
}

#[test]
fn replaces_at_and_arguments_identically() {
    let args = ["foo".to_string(), "bar".to_string(), "baz".to_string()];
    assert_eq!(
        substitute_args("Test: $@", &args),
        substitute_args("Test: $ARGUMENTS", &args)
    );
}

#[test]
fn does_not_recursively_substitute_patterns_in_argument_values() {
    assert_eq!(
        substitute_args("$ARGUMENTS", &["$1".to_string(), "$ARGUMENTS".to_string()]),
        "$1 $ARGUMENTS"
    );
    assert_eq!(
        substitute_args("$@", &["$100".to_string(), "$1".to_string()]),
        "$100 $1"
    );
    assert_eq!(
        substitute_args("$ARGUMENTS", &["$100".to_string(), "$1".to_string()]),
        "$100 $1"
    );
}

#[test]
fn supports_mixed_positional_and_arguments() {
    assert_eq!(
        substitute_args(
            "$1: $ARGUMENTS",
            &["prefix".to_string(), "a".to_string(), "b".to_string()]
        ),
        "prefix: prefix a b"
    );
}

#[test]
fn supports_mixed_positional_and_at() {
    assert_eq!(
        substitute_args(
            "$1: $@",
            &["prefix".to_string(), "a".to_string(), "b".to_string()]
        ),
        "prefix: prefix a b"
    );
}

#[test]
fn handles_empty_arguments_array_with_arguments() {
    assert_eq!(substitute_args("Test: $ARGUMENTS", &[]), "Test: ");
}

#[test]
fn handles_empty_arguments_array_with_at() {
    assert_eq!(substitute_args("Test: $@", &[]), "Test: ");
}

#[test]
fn handles_empty_arguments_array_with_positional() {
    assert_eq!(substitute_args("Test: $1", &[]), "Test: ");
}

#[test]
fn handles_multiple_occurrences_of_arguments() {
    assert_eq!(
        substitute_args(
            "$ARGUMENTS and $ARGUMENTS",
            &["a".to_string(), "b".to_string()]
        ),
        "a b and a b"
    );
}

#[test]
fn handles_multiple_occurrences_of_at() {
    assert_eq!(
        substitute_args("$@ and $@", &["a".to_string(), "b".to_string()]),
        "a b and a b"
    );
}

#[test]
fn handles_mixed_occurrences_of_at_and_arguments() {
    assert_eq!(
        substitute_args("$@ and $ARGUMENTS", &["a".to_string(), "b".to_string()]),
        "a b and a b"
    );
}

#[test]
fn handles_special_characters_in_arguments() {
    assert_eq!(
        substitute_args(
            "$1 $2: $ARGUMENTS",
            &["arg100".to_string(), "@user".to_string()]
        ),
        "arg100 @user: arg100 @user"
    );
}

#[test]
fn handles_out_of_range_numbered_placeholders() {
    assert_eq!(
        substitute_args("$1 $2 $3 $4 $5", &["a".to_string(), "b".to_string()]),
        "a b   "
    );
}

#[test]
fn handles_unicode_characters() {
    assert_eq!(
        substitute_args(
            "$ARGUMENTS",
            &["日本語".to_string(), "🎉".to_string(), "café".to_string()]
        ),
        "日本語 🎉 café"
    );
}

#[test]
fn preserves_newlines_and_tabs_in_argument_values() {
    assert_eq!(
        substitute_args(
            "$1 $2",
            &["line1\nline2".to_string(), "tab\tthere".to_string()]
        ),
        "line1\nline2 tab\tthere"
    );
}

#[test]
fn handles_consecutive_dollar_patterns() {
    assert_eq!(
        substitute_args("$1$2", &["a".to_string(), "b".to_string()]),
        "ab"
    );
}

#[test]
fn handles_quoted_arguments_with_spaces() {
    assert_eq!(
        substitute_args(
            "$ARGUMENTS",
            &["first arg".to_string(), "second arg".to_string()]
        ),
        "first arg second arg"
    );
}

#[test]
fn handles_single_argument_with_arguments() {
    assert_eq!(
        substitute_args("Test: $ARGUMENTS", &["only".to_string()]),
        "Test: only"
    );
}

#[test]
fn handles_single_argument_with_at() {
    assert_eq!(
        substitute_args("Test: $@", &["only".to_string()]),
        "Test: only"
    );
}

#[test]
fn handles_zero_index() {
    assert_eq!(
        substitute_args("$0", &["a".to_string(), "b".to_string()]),
        ""
    );
}

#[test]
fn handles_decimal_number_in_pattern_only_integer_part_matches() {
    assert_eq!(substitute_args("$1.5", &["a".to_string()]), "a.5");
}

#[test]
fn handles_arguments_as_part_of_word() {
    assert_eq!(
        substitute_args("pre$ARGUMENTS", &["a".to_string(), "b".to_string()]),
        "prea b"
    );
}

#[test]
fn handles_at_as_part_of_word() {
    assert_eq!(
        substitute_args("pre$@", &["a".to_string(), "b".to_string()]),
        "prea b"
    );
}

#[test]
fn handles_empty_arguments_in_middle_of_list() {
    assert_eq!(
        substitute_args(
            "$ARGUMENTS",
            &["a".to_string(), String::new(), "c".to_string()]
        ),
        "a  c"
    );
}

#[test]
fn handles_trailing_and_leading_spaces_in_arguments() {
    assert_eq!(
        substitute_args(
            "$ARGUMENTS",
            &["  leading  ".to_string(), "trailing  ".to_string()]
        ),
        "  leading   trailing  "
    );
}

#[test]
fn handles_argument_containing_pattern_partially() {
    assert_eq!(
        substitute_args("Prefix $ARGUMENTS suffix", &["ARGUMENTS".to_string()]),
        "Prefix ARGUMENTS suffix"
    );
}

#[test]
fn handles_non_matching_patterns() {
    assert_eq!(
        substitute_args("$A $$ $ $ARGS", &["a".to_string()]),
        "$A $$ $ $ARGS"
    );
}

#[test]
fn handles_case_variations_case_sensitive() {
    assert_eq!(
        substitute_args(
            "$arguments $Arguments $ARGUMENTS",
            &["a".to_string(), "b".to_string()]
        ),
        "$arguments $Arguments a b"
    );
}

#[test]
fn handles_both_syntaxes_in_same_command_with_same_result() {
    let args = ["x".to_string(), "y".to_string(), "z".to_string()];
    let result1 = substitute_args("$@ and $ARGUMENTS", &args);
    let result2 = substitute_args("$ARGUMENTS and $@", &args);
    assert_eq!(result1, result2);
    assert_eq!(result1, "x y z and x y z");
}

#[test]
fn handles_very_long_argument_lists() {
    let args: Vec<String> = (0..100).map(|i| format!("arg{i}")).collect();
    let result = substitute_args("$ARGUMENTS", &args);
    assert_eq!(result, args.join(" "));
}

#[test]
fn handles_numbered_placeholders_with_single_digit() {
    assert_eq!(
        substitute_args(
            "$1 $2 $3",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "a b c"
    );
}

#[test]
fn handles_numbered_placeholders_with_multiple_digits() {
    let args: Vec<String> = (0..15).map(|i| format!("val{i}")).collect();
    assert_eq!(substitute_args("$10 $12 $15", &args), "val9 val11 val14");
}

#[test]
fn handles_escaped_dollar_signs_literal_backslash_preserved() {
    // No escape mechanism exists - backslash is treated literally
    assert_eq!(substitute_args("Price: \\$100", &[]), "Price: \\");
}

#[test]
fn handles_mixed_numbered_and_wildcard_placeholders() {
    assert_eq!(
        substitute_args(
            "$1: $@ ($ARGUMENTS)",
            &[
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ]
        ),
        "first: first second third (first second third)"
    );
}

#[test]
fn handles_command_with_no_placeholders() {
    assert_eq!(
        substitute_args("Just plain text", &["a".to_string(), "b".to_string()]),
        "Just plain text"
    );
}

#[test]
fn handles_command_with_only_placeholders() {
    assert_eq!(
        substitute_args(
            "$1 $2 $@",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "a b a b c"
    );
}

// === substituteArgs - positional defaults ===================================

#[test]
fn uses_default_when_positional_arg_is_missing() {
    assert_eq!(
        substitute_args("List exactly ${1:-7} next steps", &[]),
        "List exactly 7 next steps"
    );
}

#[test]
fn supports_defaults_for_all_arguments() {
    let template = "${@:-default}\n${ARGUMENTS:-default}";

    assert_eq!(substitute_args(template, &[]), "default\ndefault");
    assert_eq!(
        substitute_args(
            template,
            &[
                "This".to_string(),
                "would".to_string(),
                "be".to_string(),
                "the".to_string(),
                "arguments".to_string()
            ]
        ),
        "This would be the arguments\nThis would be the arguments"
    );
}

#[test]
fn uses_positional_arg_when_present() {
    assert_eq!(
        substitute_args("List exactly ${1:-7} next steps", &["3".to_string()]),
        "List exactly 3 next steps"
    );
}

#[test]
fn uses_default_when_positional_arg_is_empty() {
    assert_eq!(
        substitute_args("Mode: ${1:-brief}", &[String::new()]),
        "Mode: brief"
    );
}

#[test]
fn supports_multiple_positional_defaults() {
    let template = "${1:-7} ${2:-brief}";
    assert_eq!(substitute_args(template, &[]), "7 brief");
    assert_eq!(substitute_args(template, &["3".to_string()]), "3 brief");
    assert_eq!(
        substitute_args(template, &["3".to_string(), "verbose".to_string()]),
        "3 verbose"
    );
}

#[test]
fn does_not_recursively_substitute_patterns_in_arg_values() {
    assert_eq!(
        substitute_args("${1:-7}", &["$ARGUMENTS".to_string()]),
        "$ARGUMENTS"
    );
    assert_eq!(substitute_args("${1:-7}", &["$1".to_string()]), "$1");
}

#[test]
fn does_not_recursively_substitute_patterns_in_default_values() {
    assert_eq!(
        substitute_args("${1:-$ARGUMENTS}", &["a".to_string(), "b".to_string()]),
        "a"
    );
    assert_eq!(
        substitute_args("${3:-$ARGUMENTS}", &["a".to_string(), "b".to_string()]),
        "$ARGUMENTS"
    );
}

#[test]
fn supports_defaults_with_spaces() {
    assert_eq!(substitute_args("${1:-seven steps}", &[]), "seven steps");
}

#[test]
fn supports_out_of_range_positional_defaults() {
    assert_eq!(
        substitute_args("${3:-fallback}", &["a".to_string(), "b".to_string()]),
        "fallback"
    );
}

#[test]
fn mixes_positional_defaults_with_existing_placeholders() {
    assert_eq!(
        substitute_args("$1 ${2:-x} $ARGUMENTS", &["a".to_string()]),
        "a x a"
    );
}

// === substituteArgs - array slicing =========================================

#[test]
fn slices_from_index() {
    assert_eq!(
        substitute_args(
            "${@:2}",
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        ),
        "b c d"
    );
    assert_eq!(
        substitute_args(
            "${@:1}",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "a b c"
    );
    assert_eq!(
        substitute_args(
            "${@:3}",
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        ),
        "c d"
    );
}

#[test]
fn slices_with_length() {
    assert_eq!(
        substitute_args(
            "${@:2:2}",
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        ),
        "b c"
    );
    assert_eq!(
        substitute_args(
            "${@:1:1}",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "a"
    );
    assert_eq!(
        substitute_args(
            "${@:3:1}",
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        ),
        "c"
    );
    assert_eq!(
        substitute_args(
            "${@:2:3}",
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string()
            ]
        ),
        "b c d"
    );
}

#[test]
fn handles_out_of_range_slices() {
    assert_eq!(
        substitute_args("${@:99}", &["a".to_string(), "b".to_string()]),
        ""
    );
    assert_eq!(
        substitute_args("${@:5}", &["a".to_string(), "b".to_string()]),
        ""
    );
    assert_eq!(
        substitute_args("${@:10:5}", &["a".to_string(), "b".to_string()]),
        ""
    );
}

#[test]
fn handles_zero_length_slices() {
    assert_eq!(
        substitute_args(
            "${@:2:0}",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        ""
    );
    assert_eq!(
        substitute_args("${@:1:0}", &["a".to_string(), "b".to_string()]),
        ""
    );
}

#[test]
fn handles_length_exceeding_array() {
    assert_eq!(
        substitute_args(
            "${@:2:99}",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "b c"
    );
    assert_eq!(
        substitute_args("${@:1:10}", &["a".to_string(), "b".to_string()]),
        "a b"
    );
}

#[test]
fn processes_slice_before_simple_at() {
    assert_eq!(
        substitute_args(
            "${@:2} vs $@",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "b c vs a b c"
    );
    assert_eq!(
        substitute_args(
            "First: ${@:1:1}, All: $@",
            &["x".to_string(), "y".to_string(), "z".to_string()]
        ),
        "First: x, All: x y z"
    );
}

#[test]
fn does_not_recursively_substitute_slice_patterns_in_args() {
    assert_eq!(
        substitute_args("${@:1}", &["${@:2}".to_string(), "test".to_string()]),
        "${@:2} test"
    );
    assert_eq!(
        substitute_args(
            "${@:2}",
            &["a".to_string(), "${@:3}".to_string(), "c".to_string()]
        ),
        "${@:3} c"
    );
}

#[test]
fn handles_mixed_usage_with_positional_args() {
    assert_eq!(
        substitute_args(
            "$1: ${@:2}",
            &["cmd".to_string(), "arg1".to_string(), "arg2".to_string()]
        ),
        "cmd: arg1 arg2"
    );
    assert_eq!(
        substitute_args(
            "$1 $2 ${@:3}",
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        ),
        "a b c d"
    );
}

#[test]
fn treats_zero_index_slice_as_all_args() {
    assert_eq!(
        substitute_args(
            "${@:0}",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "a b c"
    );
}

#[test]
fn handles_empty_args_array_in_slices() {
    assert_eq!(substitute_args("${@:2}", &[]), "");
    assert_eq!(substitute_args("${@:1}", &[]), "");
}

#[test]
fn handles_single_arg_array_in_slices() {
    assert_eq!(substitute_args("${@:1}", &["only".to_string()]), "only");
    assert_eq!(substitute_args("${@:2}", &["only".to_string()]), "");
}

#[test]
fn handles_slice_in_middle_of_text() {
    assert_eq!(
        substitute_args(
            "Process ${@:2} with $1",
            &["tool".to_string(), "file1".to_string(), "file2".to_string()]
        ),
        "Process file1 file2 with tool"
    );
}

#[test]
fn handles_multiple_slices_in_one_template() {
    assert_eq!(
        substitute_args(
            "${@:1:1} and ${@:2}",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "a and b c"
    );
    assert_eq!(
        substitute_args(
            "${@:1:2} vs ${@:3:2}",
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string()
            ]
        ),
        "a b vs c d"
    );
}

#[test]
fn handles_quoted_arguments_in_slices() {
    assert_eq!(
        substitute_args(
            "${@:2}",
            &[
                "cmd".to_string(),
                "first arg".to_string(),
                "second arg".to_string()
            ]
        ),
        "first arg second arg"
    );
}

#[test]
fn handles_special_characters_in_sliced_args() {
    assert_eq!(
        substitute_args(
            "${@:2}",
            &[
                "cmd".to_string(),
                "$100".to_string(),
                "@user".to_string(),
                "#tag".to_string()
            ]
        ),
        "$100 @user #tag"
    );
}

#[test]
fn handles_unicode_in_sliced_args() {
    assert_eq!(
        substitute_args(
            "${@:1}",
            &["日本語".to_string(), "🎉".to_string(), "café".to_string()]
        ),
        "日本語 🎉 café"
    );
}

#[test]
fn combines_positional_slice_and_wildcard_placeholders() {
    let template = "Run $1 on ${@:2:2}, then process $@";
    let args: Vec<String> = [
        "eslint".to_string(),
        "file1.ts".to_string(),
        "file2.ts".to_string(),
        "file3.ts".to_string(),
    ]
    .into();
    assert_eq!(
        substitute_args(template, &args),
        "Run eslint on file1.ts file2.ts, then process eslint file1.ts file2.ts file3.ts"
    );
}

#[test]
fn handles_slice_with_no_spacing() {
    assert_eq!(
        substitute_args(
            "prefix${@:2}suffix",
            &["a".to_string(), "b".to_string(), "c".to_string()]
        ),
        "prefixb csuffix"
    );
}

#[test]
fn handles_large_slice_lengths_gracefully() {
    let args: Vec<String> = (0..10).map(|i| format!("arg{}", i + 1)).collect();
    assert_eq!(
        substitute_args("${@:5:100}", &args),
        "arg5 arg6 arg7 arg8 arg9 arg10"
    );
}

// === parseCommandArgs =======================================================

#[test]
fn parses_simple_space_separated_arguments() {
    assert_eq!(parse_command_args("a b c"), vec!["a", "b", "c"]);
}

#[test]
fn parses_quoted_arguments_with_spaces() {
    assert_eq!(
        parse_command_args("\"first arg\" second"),
        vec!["first arg", "second"]
    );
}

#[test]
fn parses_single_quoted_arguments() {
    assert_eq!(
        parse_command_args("'first arg' second"),
        vec!["first arg", "second"]
    );
}

#[test]
fn parses_mixed_quote_styles() {
    assert_eq!(
        parse_command_args("\"double\" 'single' \"double again\""),
        vec!["double", "single", "double again"]
    );
}

#[test]
fn parses_the_empty_string_as_no_arguments() {
    assert_eq!(parse_command_args(""), Vec::<String>::new());
}

#[test]
fn handles_extra_spaces() {
    assert_eq!(parse_command_args("a  b   c"), vec!["a", "b", "c"]);
}

#[test]
fn handles_tabs_as_separators() {
    assert_eq!(parse_command_args("a\tb\tc"), vec!["a", "b", "c"]);
}

#[test]
fn handles_quoted_empty_string() {
    // Empty quotes are skipped by current implementation
    assert_eq!(parse_command_args("\"\" \" \""), vec![" "]);
}

#[test]
fn handles_arguments_with_special_characters() {
    assert_eq!(
        parse_command_args("$100 @user #tag"),
        vec!["$100", "@user", "#tag"]
    );
}

#[test]
fn handles_unicode_characters_in_parse() {
    assert_eq!(
        parse_command_args("日本語 🎉 café"),
        vec!["日本語", "🎉", "café"]
    );
}

#[test]
fn handles_newlines_in_quoted_arguments() {
    assert_eq!(
        parse_command_args("\"line1\nline2\" second"),
        vec!["line1\nline2", "second"]
    );
}

#[test]
fn treats_unquoted_newlines_as_separators() {
    assert_eq!(
        parse_command_args("label-2\n\nHere is some description #2."),
        vec!["label-2", "Here", "is", "some", "description", "#2."]
    );
}

#[test]
fn collapses_mixed_unquoted_whitespace() {
    assert_eq!(parse_command_args("a\n\n\tb  c"), vec!["a", "b", "c"]);
}

#[test]
fn handles_escaped_quotes_inside_quoted_strings() {
    // This implementation doesn't handle escaped quotes - backslash is literal
    assert_eq!(
        parse_command_args("\"quoted \\\"text\\\"\""),
        vec!["quoted \\text\\"]
    );
}

#[test]
fn handles_trailing_spaces() {
    assert_eq!(parse_command_args("a b c   "), vec!["a", "b", "c"]);
}

#[test]
fn handles_leading_spaces() {
    assert_eq!(parse_command_args("   a b c"), vec!["a", "b", "c"]);
}

// === expandPromptTemplate ===================================================

#[test]
fn splits_template_arguments_on_unquoted_newlines() {
    let result = expand_prompt_template(
        "/arg-test label-2\n\nHere is some description #2.",
        &[template_named("arg-test", "- arg1: $1\n- rest: ${@:2}")],
    );

    assert_eq!(
        result,
        "- arg1: label-2\n- rest: Here is some description #2."
    );
}

#[test]
fn supports_template_command_separated_from_args_by_newline() {
    let result = expand_prompt_template(
        "/arg-test\nlabel-2",
        &[template_named("arg-test", "arg1: $1")],
    );

    assert_eq!(result, "arg1: label-2");
}

// === parseCommandArgs + substituteArgs integration ==========================

#[test]
fn parses_and_substitutes_together_correctly() {
    let input = "Button \"onClick handler\" \"disabled support\"";
    let args = parse_command_args(input);
    let template = "Create component $1 with features: $ARGUMENTS";
    let result = substitute_args(template, &args);
    assert_eq!(
        result,
        "Create component Button with features: Button onClick handler disabled support"
    );
}

#[test]
fn handles_the_example_from_readme() {
    let input = "Button \"onClick handler\" \"disabled support\"";
    let args = parse_command_args(input);
    let template = "Create a React component named $1 with features: $ARGUMENTS";
    let result = substitute_args(template, &args);
    assert_eq!(
        result,
        "Create a React component named Button with features: Button onClick handler disabled support"
    );
}

#[test]
fn produces_same_result_with_at_and_arguments() {
    let args = parse_command_args("feature1 feature2 feature3");
    let template1 = "Implement: $@";
    let template2 = "Implement: $ARGUMENTS";
    assert_eq!(
        substitute_args(template1, &args),
        substitute_args(template2, &args)
    );
}

// === loadPromptTemplates - argument-hint ====================================

fn write_template(dir: &Path, name: &str, content: &str) {
    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(dir.join(format!("{name}.md")), content).expect("write template");
}

fn load_templates_from(dir: &Path) -> Vec<PromptTemplate> {
    load_prompt_templates(
        &pi_coding_agent::prompt_templates::LoadPromptTemplatesOptions {
            cwd: &std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy(),
            agent_dir: &pi_coding_agent::config::get_agent_dir().to_string_lossy(),
            prompt_paths: &[dir.to_string_lossy().into_owned()],
            include_defaults: false,
        },
    )
}

#[test]
fn parses_required_argument_hint_from_frontmatter() {
    let test_dir = std::env::temp_dir().join("pi-test-prompts-required-arg-hint");
    write_template(
        &test_dir,
        "pr",
        "---\ndescription: Review PRs from URLs with structured issue and code analysis\nargument-hint: \"<PR-URL>\"\n---\nYou are given one or more GitHub PR URLs: $@",
    );

    let templates = load_templates_from(&test_dir);

    let pr = templates
        .iter()
        .find(|t| t.name == "pr")
        .expect("pr template");
    assert_eq!(pr.argument_hint.as_deref(), Some("<PR-URL>"));
    assert_eq!(
        pr.description,
        "Review PRs from URLs with structured issue and code analysis"
    );
    let _ = std::fs::remove_dir_all(&test_dir);
}

#[test]
fn parses_optional_argument_hint_from_frontmatter() {
    let test_dir = std::env::temp_dir().join("pi-test-prompts-optional-arg-hint");
    write_template(
        &test_dir,
        "wr",
        "---\ndescription: Finish the current task end-to-end with changelog, commit, and push\nargument-hint: \"[instructions]\"\n---\nWrap it. Additional instructions: $ARGUMENTS",
    );

    let templates = load_templates_from(&test_dir);

    let wr = templates
        .iter()
        .find(|t| t.name == "wr")
        .expect("wr template");
    assert_eq!(wr.argument_hint.as_deref(), Some("[instructions]"));
    assert_eq!(
        wr.description,
        "Finish the current task end-to-end with changelog, commit, and push"
    );
    let _ = std::fs::remove_dir_all(&test_dir);
}

#[test]
fn leaves_argument_hint_absent_when_not_specified() {
    let test_dir = std::env::temp_dir().join("pi-test-prompts-no-arg-hint");
    write_template(
        &test_dir,
        "cl",
        "---\ndescription: Audit changelog entries before release\n---\nAudit changelog entries for all commits since the last release.",
    );

    let templates = load_templates_from(&test_dir);

    let cl = templates
        .iter()
        .find(|t| t.name == "cl")
        .expect("cl template");
    assert!(cl.argument_hint.is_none());
    let _ = std::fs::remove_dir_all(&test_dir);
}

#[test]
fn ignores_empty_argument_hint() {
    let test_dir = std::env::temp_dir().join("pi-test-prompts-empty-arg-hint");
    write_template(
        &test_dir,
        "empty-hint",
        "---\ndescription: A command with empty hint\nargument-hint: \"\"\n---\nDo something",
    );

    let templates = load_templates_from(&test_dir);

    let tmpl = templates
        .iter()
        .find(|t| t.name == "empty-hint")
        .expect("empty-hint template");
    assert!(tmpl.argument_hint.is_none());
    let _ = std::fs::remove_dir_all(&test_dir);
}

#[test]
fn preserves_argument_hint_with_special_characters() {
    let test_dir = std::env::temp_dir().join("pi-test-prompts-special-arg-hint");
    write_template(
        &test_dir,
        "is",
        "---\ndescription: Analyze GitHub issues (bugs or feature requests)\nargument-hint: \"<issue>\"\n---\nAnalyze GitHub issue(s): $ARGUMENTS",
    );

    let templates = load_templates_from(&test_dir);

    let is = templates
        .iter()
        .find(|t| t.name == "is")
        .expect("is template");
    assert_eq!(is.argument_hint.as_deref(), Some("<issue>"));
    let _ = std::fs::remove_dir_all(&test_dir);
}
