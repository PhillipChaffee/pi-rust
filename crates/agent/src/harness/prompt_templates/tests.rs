//! The `prompt-templates.test.ts` suite ported 1:1, plus boundary tests
//! binding the restated surfaces upstream's suite does not reach: the
//! first-line description fallback edges, the case-sensitive `.md`
//! selection, the substitution passes' ordering and clamping, and the
//! quote parser's edges.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use crate::harness::context::background_context;
use crate::harness::fs_scan::{DiagnosticSeverity, SourcedPath};
use crate::harness::prompt_templates::{
    PromptTemplateDiagnostic, PromptTemplateDiagnosticCode, SourcedPromptTemplate,
    format_prompt_template_invocation, load_prompt_templates, load_sourced_prompt_templates,
    load_sourced_prompt_templates_mapped, load_template_from_file, parse_command_args,
    substitute_args,
};
use crate::harness::test_support::{TestSource, env_for, file_path, mkdir, write};
use crate::harness::types::PromptTemplate;

/// Loads markdown templates non-recursively from one or more dirs,
/// upstream's "loads markdown templates non-recursively from one or more
/// dirs".
#[tokio::test]
async fn loads_markdown_templates_non_recursively_from_one_or_more_dirs() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "a/nested", &context).await;
    mkdir(&env, "b", &context).await;
    write(
        &env,
        "a/one.md",
        "---\ndescription: One template\n---\nHello $1",
        &context,
    )
    .await;
    write(&env, "a/nested/ignored.md", "Ignored", &context).await;
    write(&env, "b/two.md", "First line description\nBody", &context).await;

    let paths = vec!["a".to_owned(), "b".to_owned()];
    let loaded = load_prompt_templates(&env, &paths, &context).await;

    assert!(loaded.diagnostics.is_empty());
    assert_eq!(
        loaded.prompt_templates,
        vec![
            PromptTemplate {
                name: "one".to_owned(),
                description: Some("One template".to_owned()),
                content: "Hello $1".to_owned(),
            },
            PromptTemplate {
                name: "two".to_owned(),
                description: Some("First line description".to_owned()),
                content: "First line description\nBody".to_owned(),
            },
        ]
    );
}

/// Preserves source info for sourced prompt templates, upstream's
/// "preserves source info for sourced prompt templates".
#[tokio::test]
async fn preserves_source_info_for_sourced_prompt_templates() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "prompts", &context).await;
    write(
        &env,
        "prompts/example.md",
        "---\ndescription: Example\n---\nExample body",
        &context,
    )
    .await;

    let inputs = vec![SourcedPath {
        path: "prompts".to_owned(),
        source: TestSource::Project,
    }];
    let loaded = load_sourced_prompt_templates(&env, &inputs, &context).await;

    assert!(loaded.diagnostics.is_empty());
    assert_eq!(
        loaded.prompt_templates,
        vec![SourcedPromptTemplate {
            prompt_template: PromptTemplate {
                name: "example".to_owned(),
                description: Some("Example".to_owned()),
                content: "Example body".to_owned(),
            },
            source: TestSource::Project,
        }]
    );
}

/// Attaches source info to diagnostics, upstream's "attaches source info to
/// diagnostics".
#[tokio::test]
async fn attaches_source_info_to_diagnostics() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    write(
        &env,
        "broken.md",
        "---\ndescription: [unterminated\n---\nBody",
        &context,
    )
    .await;

    let inputs = vec![SourcedPath {
        path: "broken.md".to_owned(),
        source: TestSource::User,
    }];
    let loaded = load_sourced_prompt_templates(&env, &inputs, &context).await;

    assert!(loaded.prompt_templates.is_empty());
    assert_eq!(loaded.diagnostics.len(), 1);
    let diagnostic = &loaded.diagnostics[0];
    assert_eq!(diagnostic.diagnostic.r#type, DiagnosticSeverity::Warning);
    assert_eq!(
        diagnostic.diagnostic.path,
        file_path(root.path(), "broken.md")
    );
    assert_eq!(diagnostic.source, TestSource::User);
}

/// Loads explicit markdown files and symlinked files, upstream's "loads
/// explicit markdown files and symlinked files".
#[tokio::test]
async fn loads_explicit_markdown_files_and_symlinked_files() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    write(
        &env,
        "target.md",
        "---\ndescription: Target\n---\nTarget body",
        &context,
    )
    .await;
    std::os::unix::fs::symlink(
        file_path(root.path(), "target.md"),
        file_path(root.path(), "link.md"),
    )
    .expect("symlink");

    let paths = vec!["target.md".to_owned(), "link.md".to_owned()];
    let loaded = load_prompt_templates(&env, &paths, &context).await;

    assert_eq!(
        loaded.prompt_templates,
        vec![
            PromptTemplate {
                name: "target".to_owned(),
                description: Some("Target".to_owned()),
                content: "Target body".to_owned(),
            },
            PromptTemplate {
                name: "link".to_owned(),
                description: Some("Target".to_owned()),
                content: "Target body".to_owned(),
            },
        ]
    );
}

/// Substitutes command arguments, upstream's "substitutes command
/// arguments".
#[test]
fn substitutes_command_arguments() {
    let template = PromptTemplate {
        name: "one".to_owned(),
        description: None,
        content: "$1 ${@:2} $ARGUMENTS".to_owned(),
    };
    let args = vec!["hello world".to_owned(), "test".to_owned()];
    assert_eq!(
        format_prompt_template_invocation(&template, &args),
        "hello world test hello world test"
    );
}

/// The first-line fallback clips at sixty characters with an ellipsis and
/// skips blank lines; a body with no non-blank line carries the empty
/// description, upstream's `description: ""`.
#[tokio::test]
async fn the_first_line_fallback_clips_at_sixty_characters_and_skips_blank_lines() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    let long_first = "x".repeat(61);
    write(&env, "long.md", &format!("{long_first}\nBody"), &context).await;
    write(&env, "blank.md", "\n   \nSecond line\nBody", &context).await;
    write(&env, "empty.md", "", &context).await;

    let paths = vec![
        "long.md".to_owned(),
        "blank.md".to_owned(),
        "empty.md".to_owned(),
    ];
    let loaded = load_prompt_templates(&env, &paths, &context).await;

    let descriptions: Vec<&str> = loaded
        .prompt_templates
        .iter()
        .map(|template| {
            template
                .description
                .as_deref()
                .expect("description present")
        })
        .collect();
    // Sixty characters plus the ellipsis.
    assert_eq!(descriptions[0].len(), 63);
    assert!(descriptions[0].starts_with(&"x".repeat(60)));
    assert!(descriptions[0].ends_with("..."));
    assert_eq!(descriptions[1], "Second line");
    assert_eq!(descriptions[2], "");
}

/// Directory traversal stays non-recursive and non-markdown files skip.
#[tokio::test]
async fn directory_traversal_stays_non_recursive_and_non_markdown_files_skip() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "prompts/nested", &context).await;
    write(&env, "prompts/notes.txt", "Not markdown", &context).await;
    write(&env, "prompts/nested/inner.md", "Nested body", &context).await;
    write(&env, "prompts/keep.md", "Keep body", &context).await;

    let paths = vec!["prompts".to_owned()];
    let loaded = load_prompt_templates(&env, &paths, &context).await;

    let names: Vec<&str> = loaded
        .prompt_templates
        .iter()
        .map(|template| template.name.as_str())
        .collect();
    assert_eq!(names, ["keep"]);
}

/// The `.md` selection is case-sensitive while the name strip is not, the
/// upstream pair `endsWith(".md")` / `/\.md$/i` — so an uppercase-suffix
/// explicit file is skipped outright.
#[tokio::test]
async fn an_uppercase_suffix_file_is_skipped_by_the_case_sensitive_selection() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    write(
        &env,
        "upper.MD",
        "---\ndescription: Upper\n---\nBody",
        &context,
    )
    .await;

    let paths = vec!["upper.MD".to_owned()];
    let loaded = load_prompt_templates(&env, &paths, &context).await;

    assert!(loaded.prompt_templates.is_empty());
    assert!(loaded.diagnostics.is_empty());
}

/// Positional substitution: `$0` and out-of-range positions read empty,
/// multi-digit positions index the tail.
#[test]
fn positional_substitution_reads_out_of_range_positions_as_empty() {
    let args: Vec<String> = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(substitute_args("$1/$2/$0/$3/$10", &args), "a/b//c/j");
}

/// The `${@:N}` slice pass clamps: zero starts at the first argument,
/// out-of-range starts and lengths read empty.
#[test]
fn the_slice_pass_clamps_starts_and_lengths() {
    let args: Vec<String> = ["1", "2", "3"].iter().map(ToString::to_string).collect();
    assert_eq!(
        substitute_args("${@:1}|${@:2}|${@:3}|${@:0}|${@:9}", &args),
        "1 2 3|2 3|3|1 2 3|"
    );
    assert_eq!(
        substitute_args("${@:2:1}|${@:1:0}|${@:2:99}|${@:9:1}", &args),
        "2||2 3|"
    );
}

/// `$ARGUMENTS` and `$@` join every argument; the passes run in upstream's
/// order, so substituted text participates in the later passes.
#[test]
fn the_arguments_passes_run_in_upstreams_order() {
    let args: Vec<String> = vec!["x".to_owned(), "y".to_owned()];
    assert_eq!(substitute_args("A $ARGUMENTS B", &args), "A x y B");
    assert_eq!(substitute_args("A $@ B", &args), "A x y B");
    // An argument value carrying `$@` is inserted by the `$1` pass and then
    // rescanned by the `$@` pass.
    assert_eq!(substitute_args("$1", ["$@".to_owned()].as_slice()), "$@");
    // A `$` that starts no placeholder stays literal; the `$` after it
    // starts one, upstream's regex rescanning from the next byte.
    assert_eq!(substitute_args("$$1 cost", &args), "$x cost");
    // The `$ARGUMENTS` pass replaces every occurrence, including the one a
    // preceding `$` leaves inside the text.
    assert_eq!(substitute_args("$$ARGUMENTS", &args), "$x y");
    // Substituted values insert literally: upstream's replacement-string
    // `$` patterns (e.g. `$&` re-inserting the match) restate as literal
    // insertion.
    assert_eq!(
        substitute_args("$ARGUMENTS", ["a$&b".to_owned()].as_slice()),
        "a$&b"
    );
}

/// Malformed `${@:` shapes stay literal, the inputs the global regex
/// declines; `${@:1}` reads every argument from the start.
#[test]
fn malformed_slice_shapes_stay_literal() {
    assert_eq!(
        substitute_args(
            "${@:x} ${@:1:} ${@:1:2:3} ${@:1",
            ["a".to_owned()].as_slice()
        ),
        "${@:x} ${@:1:} ${@:1:2:3} ${@:1"
    );
    assert_eq!(
        substitute_args(
            "${@:1}:${@:2:2}",
            ["a".to_owned(), "b".to_owned(), "c".to_owned()].as_slice()
        ),
        "a b c:b c"
    );
}

/// The quote parser toggles single and double quotes, splits on space and
/// tab, and keeps unterminated quotes' content.
#[test]
fn the_quote_parser_covers_quotes_tabs_and_unterminated_quotes() {
    assert_eq!(
        parse_command_args("run 'a b' \"c d\" plain\ttabbed  end"),
        ["run", "a b", "c d", "plain", "tabbed", "end"]
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    assert_eq!(parse_command_args("'unterminated"), ["unterminated"]);
    assert_eq!(parse_command_args("a\"b\"c"), ["abc"]);
    assert_eq!(parse_command_args(""), Vec::<String>::new());
}

/// The sourced mapper converts to the application's template type.
#[tokio::test]
async fn the_sourced_mapper_converts_to_the_application_template_type() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct RichTemplate {
        name: String,
        origin: &'static str,
    }

    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "prompts", &context).await;
    write(
        &env,
        "prompts/example.md",
        "---\ndescription: Example\n---\nBody",
        &context,
    )
    .await;

    let inputs = vec![SourcedPath {
        path: "prompts".to_owned(),
        source: TestSource::User,
    }];
    let loaded = load_sourced_prompt_templates_mapped(
        &env,
        &inputs,
        &|template, source, _context| RichTemplate {
            name: template.name.clone(),
            origin: match source {
                TestSource::User => "user",
                TestSource::Project => "project",
            },
        },
        &context,
    )
    .await;

    assert_eq!(
        loaded.prompt_templates,
        vec![SourcedPromptTemplate {
            prompt_template: RichTemplate {
                name: "example".to_owned(),
                origin: "user",
            },
            source: TestSource::User,
        }]
    );
}

// --- boundary: the loader's error-diagnostic branches, bound through the
// --- fault-injecting environment.

use crate::harness::test_support::{Fault, FaultEnv};
use crate::harness::types::FileErrorCode;

const BOOM: Fault = Fault {
    path_contains: "",
    code: FileErrorCode::Unknown,
    message: "boom",
};

/// A root file-info failure outside `not_found` reports the diagnostic;
/// `not_found` stays silent.
#[tokio::test]
async fn a_file_info_failure_reports_the_diagnostic() {
    let root = tempfile::tempdir().expect("temp root");
    let context = background_context();

    let mut env = FaultEnv::new(root.path());
    env.file_info_fault = Some(BOOM);
    let loaded = load_prompt_templates(&env, &["prompts".to_owned()], &context).await;
    assert!(loaded.prompt_templates.is_empty());
    assert_eq!(
        loaded.diagnostics,
        vec![PromptTemplateDiagnostic::warning(
            PromptTemplateDiagnosticCode::FileInfoFailed,
            "boom",
            "prompts",
        )]
    );

    env.file_info_fault = Some(Fault {
        code: FileErrorCode::NotFound,
        ..BOOM
    });
    let loaded = load_prompt_templates(&env, &["prompts".to_owned()], &context).await;
    assert!(loaded.prompt_templates.is_empty());
    assert!(loaded.diagnostics.is_empty());
}

/// A listing failure reports the `list_failed` diagnostic.
#[tokio::test]
async fn a_listing_failure_reports_the_diagnostic() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    mkdir(&env, "prompts", &context).await;

    let mut faulted = FaultEnv::new(root.path());
    faulted.list_dir_fault = Some(BOOM);
    let loaded = load_prompt_templates(&faulted, &["prompts".to_owned()], &context).await;
    assert!(loaded.prompt_templates.is_empty());
    assert_eq!(
        loaded.diagnostics,
        vec![PromptTemplateDiagnostic::warning(
            PromptTemplateDiagnosticCode::ListFailed,
            "boom",
            root.path().join("prompts").to_string_lossy().as_ref(),
        )]
    );
}

/// A template read failure reports `read_failed`.
#[tokio::test]
async fn a_read_failure_reports_the_diagnostic() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    write(&env, "prompts/keep.md", "Body", &context).await;

    let mut faulted = FaultEnv::new(root.path());
    faulted.read_text_file_fault = Some(BOOM);
    let loaded = load_prompt_templates(&faulted, &["prompts/keep.md".to_owned()], &context).await;
    assert!(loaded.prompt_templates.is_empty());
    assert_eq!(
        loaded.diagnostics,
        vec![PromptTemplateDiagnostic::warning(
            PromptTemplateDiagnosticCode::ReadFailed,
            "boom",
            root.path()
                .join("prompts/keep.md")
                .to_string_lossy()
                .as_ref(),
        )]
    );
}

/// An explicit path that resolves through a broken symlink skips silently,
/// the loader's `_ => {}` arm.
#[tokio::test]
async fn an_explicit_broken_symlink_path_skips_silently() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    std::os::unix::fs::symlink(
        file_path(root.path(), "missing.md"),
        file_path(root.path(), "broken-link.md"),
    )
    .expect("symlink");

    let paths = vec!["broken-link.md".to_owned()];
    let loaded = load_prompt_templates(&env, &paths, &context).await;
    assert!(loaded.prompt_templates.is_empty());
    assert!(loaded.diagnostics.is_empty());
}

/// The name strip is case-insensitive, upstream's `/\.md$/i`: the four
/// case shapes strip. The loader's case-sensitive selection never routes
/// these through, so the restatement binds by calling the loader's file
/// reader directly.
#[tokio::test]
async fn the_name_strip_is_case_insensitive() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_for(root.path());
    let context = background_context();
    write(&env, "notes.md", "Body", &context).await;

    for file_name in ["notes.md", "notes.MD", "notes.Md", "notes.mD"] {
        let (template, diagnostics) =
            load_template_from_file(&env, "notes.md", file_name, &context).await;
        assert!(diagnostics.is_empty());
        assert_eq!(template.expect("template").name, "notes");
    }
}
