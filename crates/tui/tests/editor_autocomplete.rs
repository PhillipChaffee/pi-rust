//! The editor's autocomplete integration, ported from the autocomplete
//! tests of `packages/tui/test/editor.test.ts` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#49) — the set the
//! editor slice (#47) deferred to this ticket.
//!
//! Restatements: upstream awaits `flushAutocomplete()` (microtask +
//! `setImmediate`) between keystrokes; the port awaits
//! [`tui_support::flush_autocomplete`], which waits for the worker thread
//! to settle and drains. The worker's debounce keeps the upstream
//! `setTimeout(20ms)` semantics; the sleeps below mirror the upstream
//! real-time waits.

#![expect(
    clippy::expect_used,
    reason = "the suite asserts on finds like upstream's assert.strictEqual(...) with messages; expecting keeps the failure modes readable"
)]
#![expect(
    clippy::panic,
    reason = "the declined-gate provider panics if reached, proving the gate blocked the query"
)]
#![expect(
    clippy::redundant_clone,
    reason = "each mock's prefix feeds the returned suggestion set; dropping the trailing clone is a nursery micro-nit in test mocks"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use pi_tui::autocomplete::{
    AutocompleteApplyResult, AutocompleteItem, AutocompleteProvider, AutocompleteQueryOptions,
    AutocompleteSuggestions, CombinedAutocompleteProvider, CommandEntry, SlashCommand,
};
use pi_tui::components::{Editor, EditorOptions};
use pi_tui::tui::Component;
use tui_support::{default_editor_theme, flush_autocomplete, new_editor_test_tui, strip_ansi};

fn editor() -> Editor {
    let tui = new_editor_test_tui(80, 24);
    Editor::new(&tui, default_editor_theme())
}

fn editor_with_options(options: EditorOptions) -> Editor {
    let tui = new_editor_test_tui(80, 24);
    Editor::with_options(&tui, default_editor_theme(), options)
}

/// The upstream tests' standard `applyCompletion`: replace the prefix with
/// the item value.
fn standard_apply_completion(
    lines: &[String],
    cursor_line: usize,
    cursor_col: usize,
    item: &AutocompleteItem,
    prefix: &str,
) -> AutocompleteApplyResult {
    let line = lines.get(cursor_line).cloned().unwrap_or_default();
    let before = &line[..cursor_col.saturating_sub(prefix.len())];
    let after = &line[cursor_col.min(line.len())..];
    let mut new_lines = lines.to_vec();
    new_lines[cursor_line] = format!("{before}{}{after}", item.value);
    AutocompleteApplyResult {
        lines: new_lines,
        cursor_line,
        cursor_col: cursor_col - prefix.len() + item.value.len(),
    }
}

type GetSuggestionsHook = Arc<
    dyn Fn(&[String], usize, usize, &AutocompleteQueryOptions) -> Option<AutocompleteSuggestions>
        + Send
        + Sync,
>;

/// The upstream tests' inline mock provider shape: custom
/// `getSuggestions`, the standard `applyCompletion`, optional trigger
/// characters.
struct HookProvider {
    trigger_characters: Vec<String>,
    get_suggestions: GetSuggestionsHook,
}

impl AutocompleteProvider for HookProvider {
    fn trigger_characters(&self) -> Vec<String> {
        self.trigger_characters.clone()
    }

    fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        options: &AutocompleteQueryOptions,
    ) -> Option<AutocompleteSuggestions> {
        (self.get_suggestions)(lines, cursor_line, cursor_col, options)
    }

    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> AutocompleteApplyResult {
        standard_apply_completion(lines, cursor_line, cursor_col, item, prefix)
    }
}

fn hook_provider(
    get_suggestions: impl Fn(
        &[String],
        usize,
        usize,
        &AutocompleteQueryOptions,
    ) -> Option<AutocompleteSuggestions>
    + Send
    + Sync
    + 'static,
) -> HookProvider {
    HookProvider {
        trigger_characters: Vec::new(),
        get_suggestions: Arc::new(get_suggestions),
    }
}

fn item(value: &str, label: &str) -> AutocompleteItem {
    AutocompleteItem {
        value: value.to_string(),
        label: label.to_string(),
        description: None,
    }
}

// --- upstream editor.test.ts, prompt-history describe block --------------

#[test]
fn undoes_autocomplete() {
    let editor = editor();

    // Mock provider returning suggestions only for the exact "di" prefix.
    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let prefix = &lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        if prefix == "di" {
            return Some(AutocompleteSuggestions {
                items: vec![item("dist/", "dist/")],
                prefix: prefix.clone(),
            });
        }
        None
    });
    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "di"
    editor.handle_input("d");
    editor.handle_input("i");
    assert_eq!(editor.get_text(), "di");

    // Press Tab to trigger autocomplete
    editor.handle_input("\t");
    flush_autocomplete(&editor);
    assert_eq!(editor.get_text(), "dist/");
    assert!(!editor.is_showing_autocomplete());

    // Undo should restore to "di"
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "di");
}

#[test]
fn does_not_trigger_autocomplete_during_single_line_paste() {
    let editor = editor();
    let suggestion_calls = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&suggestion_calls);
    let provider = hook_provider(move |_lines, _cursor_line, _cursor_col, _options| {
        calls.fetch_add(1, Ordering::SeqCst);
        None
    });

    editor.set_autocomplete_provider(Arc::new(provider));
    editor.handle_input("\x1b[200~look at @node_modules/react/index.js please\x1b[201~");

    assert_eq!(
        editor.get_text(),
        "look at @node_modules/react/index.js please"
    );
    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 0);
    assert!(!editor.is_showing_autocomplete());
}

// --- upstream editor.test.ts describe("Autocomplete") ---------------------

#[test]
fn auto_applies_single_force_file_suggestion_without_showing_menu() {
    let editor = editor();

    let provider = hook_provider(|lines, _cursor_line, cursor_col, options| {
        if !options.force {
            return None;
        }
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        if prefix == "Work" {
            return Some(AutocompleteSuggestions {
                items: vec![item("Workspace/", "Workspace/")],
                prefix: prefix.clone(),
            });
        }
        None
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "Work"
    for character in "Work".chars() {
        editor.handle_input(&character.to_string());
    }
    assert_eq!(editor.get_text(), "Work");

    // Press Tab - should auto-apply without showing menu
    editor.handle_input("\t");
    flush_autocomplete(&editor);
    assert_eq!(editor.get_text(), "Workspace/");
    assert!(!editor.is_showing_autocomplete());

    // Undo should restore to "Work"
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "Work");
}

#[test]
fn shows_menu_when_force_file_has_multiple_suggestions() {
    let editor = editor();

    let provider = hook_provider(|lines, _cursor_line, cursor_col, options| {
        if !options.force {
            return None;
        }
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        if prefix == "src" {
            return Some(AutocompleteSuggestions {
                items: vec![item("src/", "src/"), item("src.txt", "src.txt")],
                prefix: prefix.clone(),
            });
        }
        None
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "src"
    for character in "src".chars() {
        editor.handle_input(&character.to_string());
    }
    assert_eq!(editor.get_text(), "src");

    // Press Tab - should show menu because there are multiple suggestions
    editor.handle_input("\t");
    flush_autocomplete(&editor);
    assert_eq!(editor.get_text(), "src");
    assert!(editor.is_showing_autocomplete());

    // Press Tab again to accept first suggestion
    editor.handle_input("\t");
    assert_eq!(editor.get_text(), "src/");
    assert!(!editor.is_showing_autocomplete());
}

#[test]
fn keeps_suggestions_open_when_typing_in_force_mode() {
    let editor = editor();

    let all_files = [
        item("readme.md", "readme.md"),
        item("package.json", "package.json"),
        item("src/", "src/"),
        item("dist/", "dist/"),
    ];
    let provider = hook_provider(move |lines, _cursor_line, cursor_col, options| {
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        let should_match = options.force || prefix.contains('/') || prefix.starts_with('.');
        if !should_match {
            return None;
        }
        let filtered: Vec<AutocompleteItem> = all_files
            .iter()
            .filter(|file| {
                file.value
                    .to_lowercase()
                    .starts_with(&prefix.to_lowercase())
            })
            .cloned()
            .collect();
        if !filtered.is_empty() {
            return Some(AutocompleteSuggestions {
                items: filtered,
                prefix: prefix.clone(),
            });
        }
        None
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    // Press Tab on empty prompt - should show all files (force mode)
    editor.handle_input("\t");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Type "r" - should narrow to "readme.md" (force mode keeps suggestions open)
    editor.handle_input("r");
    flush_autocomplete(&editor);
    assert_eq!(editor.get_text(), "r");
    assert!(editor.is_showing_autocomplete());

    // Type "e" - should still show "readme.md"
    editor.handle_input("e");
    flush_autocomplete(&editor);
    assert_eq!(editor.get_text(), "re");
    assert!(editor.is_showing_autocomplete());

    // Accept with Tab
    editor.handle_input("\t");
    assert_eq!(editor.get_text(), "readme.md");
    assert!(!editor.is_showing_autocomplete());
}

#[test]
fn debounces_at_autocomplete_while_typing() {
    let editor = editor();
    let suggestion_calls = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&suggestion_calls);
    let provider = hook_provider(move |lines, _cursor_line, cursor_col, _options| {
        calls.fetch_add(1, Ordering::SeqCst);
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        Some(AutocompleteSuggestions {
            items: vec![item("@main.ts", "main.ts")],
            prefix,
        })
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("@");
    editor.handle_input("m");
    editor.handle_input("a");
    editor.handle_input("i");

    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 0);
    assert!(!editor.is_showing_autocomplete());

    thread::sleep(Duration::from_millis(50));
    flush_autocomplete(&editor);

    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 1);
    assert!(editor.is_showing_autocomplete());
}

#[test]
fn re_queries_the_autocomplete_picker_when_the_cursor_moves_back_into_the_command_name() {
    // Regression for earendil-works/pi#5496: arrowing left out of a slash
    // command's argument region must re-query the picker, not leave the
    // stale argument list showing.
    let editor = editor();

    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let before = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        if !before.starts_with('/') {
            return None;
        }
        // Past the command name (a space before the cursor): offer
        // arguments.
        if before.contains(' ') {
            let argument_prefix = &before[before.find(' ').expect("space checked") + 1..];
            return Some(AutocompleteSuggestions {
                items: vec![
                    item("repo", "repo"),
                    item("message", "message"),
                    item("help", "help"),
                ],
                prefix: argument_prefix.to_string(),
            });
        }
        // Inside the command name: offer the command name only.
        Some(AutocompleteSuggestions {
            items: vec![item("cmd", "cmd")],
            prefix: before.clone(),
        })
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type `/cmd ` so the picker ends up showing the argument list.
    for character in "/cmd ".chars() {
        editor.handle_input(&character.to_string());
        flush_autocomplete(&editor);
    }
    assert_eq!(editor.get_text(), "/cmd ");
    assert!(editor.is_showing_autocomplete());
    let at_argument = editor
        .render(80)
        .iter()
        .map(|line| strip_ansi(line))
        .collect::<Vec<String>>()
        .join("\n");
    assert!(
        at_argument.contains("repo"),
        "argument menu should be visible at `/cmd `: {at_argument}"
    );

    // Arrow Left back into the command name (`/cmd`).
    editor.handle_input("\x1b[D");
    flush_autocomplete(&editor);

    // The picker must have re-queried: the stale argument items are gone
    // (replaced by the command-name suggestion, or the picker closed).
    let after_move = editor
        .render(80)
        .iter()
        .map(|line| strip_ansi(line))
        .collect::<Vec<String>>()
        .join("\n");
    assert!(
        !after_move.contains("repo"),
        "stale argument menu must not survive the cursor move: {after_move}"
    );
    assert!(
        !after_move.contains("message"),
        "stale argument menu must not survive the cursor move: {after_move}"
    );
}

#[test]
fn debounces_hash_autocomplete_while_typing() {
    let editor = editor();
    let suggestion_calls = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&suggestion_calls);
    let provider = hook_provider(move |lines, _cursor_line, cursor_col, _options| {
        calls.fetch_add(1, Ordering::SeqCst);
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        Some(AutocompleteSuggestions {
            items: vec![item("#2983", "#2983")],
            prefix,
        })
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("#");
    editor.handle_input("2");
    editor.handle_input("9");
    editor.handle_input("8");

    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 0);
    assert!(!editor.is_showing_autocomplete());

    thread::sleep(Duration::from_millis(50));
    flush_autocomplete(&editor);

    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 1);
    assert!(editor.is_showing_autocomplete());
}

#[test]
fn debounces_custom_trigger_characters_autocomplete_while_typing() {
    let editor = editor();
    let suggestion_calls = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&suggestion_calls);
    let mut provider = hook_provider(move |lines, _cursor_line, cursor_col, _options| {
        calls.fetch_add(1, Ordering::SeqCst);
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        Some(AutocompleteSuggestions {
            items: vec![item("$skill-name", "skill-name")],
            prefix,
        })
    });
    provider.trigger_characters = vec!["$".to_string()];

    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("$");
    editor.handle_input("s");
    editor.handle_input("k");

    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 0);
    thread::sleep(Duration::from_millis(50));
    flush_autocomplete(&editor);

    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 1);
    assert!(editor.is_showing_autocomplete());
}

#[test]
fn resets_custom_trigger_characters_when_provider_changes() {
    let editor = editor();
    let suggestion_calls = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&suggestion_calls);

    let mut first = hook_provider(|_lines, _cursor_line, _cursor_col, _options| {
        Some(AutocompleteSuggestions {
            items: vec![item("$skill-name", "skill-name")],
            prefix: "$".to_string(),
        })
    });
    first.trigger_characters = vec!["$".to_string()];
    editor.set_autocomplete_provider(Arc::new(first));
    let second = hook_provider(move |_lines, _cursor_line, _cursor_col, _options| {
        calls.fetch_add(1, Ordering::SeqCst);
        Some(AutocompleteSuggestions {
            items: vec![item("$skill-name", "skill-name")],
            prefix: "$".to_string(),
        })
    });
    editor.set_autocomplete_provider(Arc::new(second));

    editor.handle_input("$");
    editor.handle_input("s");
    thread::sleep(Duration::from_millis(50));
    flush_autocomplete(&editor);

    assert_eq!(suggestion_calls.load(Ordering::SeqCst), 0);
    assert!(!editor.is_showing_autocomplete());
}

#[test]
fn aborts_active_at_autocomplete_when_typing_continues() {
    let editor = editor();
    let aborts = Arc::new(AtomicUsize::new(0));
    let abort_counter = Arc::clone(&aborts);
    let provider = hook_provider(move |_lines, _cursor_line, _cursor_col, options| {
        // Resolve on abort or after 500ms, whichever comes first — the
        // upstream mock's setTimeout + abort listener.
        let started = std::time::Instant::now();
        let deadline = started + Duration::from_millis(500);
        loop {
            if options.signal.is_cancelled() {
                abort_counter.fetch_add(1, Ordering::SeqCst);
                return None;
            }
            if std::time::Instant::now() >= deadline {
                return Some(AutocompleteSuggestions {
                    items: vec![item("@main.ts", "main.ts")],
                    prefix: "@main".to_string(),
                });
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("@");
    editor.handle_input("m");
    editor.handle_input("a");
    editor.handle_input("i");
    thread::sleep(Duration::from_millis(250));
    editor.handle_input("n");
    thread::sleep(Duration::from_millis(50));

    assert_eq!(aborts.load(Ordering::SeqCst), 1);
}

#[test]
fn hides_autocomplete_when_backspacing_slash_command_to_empty() {
    let editor = editor();

    // Mock provider with slash commands
    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        // Only return slash command suggestions when line starts with /
        if let Some(query) = prefix.strip_prefix('/') {
            let commands = [
                ("/model", "model", "Change model"),
                ("/help", "help", "Show help"),
            ];
            let filtered: Vec<AutocompleteItem> = commands
                .iter()
                .filter(|(value, _label, _description)| value.starts_with(query))
                .map(|(value, label, description)| AutocompleteItem {
                    value: (*value).to_string(),
                    label: (*label).to_string(),
                    description: Some((*description).to_string()),
                })
                .collect();
            if !filtered.is_empty() {
                return Some(AutocompleteSuggestions {
                    items: filtered,
                    prefix: prefix.clone(),
                });
            }
        }
        None
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "/" - should show slash command suggestions
    editor.handle_input("/");
    flush_autocomplete(&editor);
    assert_eq!(editor.get_text(), "/");
    assert!(editor.is_showing_autocomplete());

    // Backspace to delete "/" - should hide autocomplete completely
    editor.handle_input("\x7f"); // Backspace
    flush_autocomplete(&editor);
    assert_eq!(editor.get_text(), "");
    assert!(!editor.is_showing_autocomplete());
}

/// The shared `/argtest` mock: argument completions filtered by the typed
/// prefix, upstream's `argtestMatch` providers.
fn argtest_provider(items: Vec<AutocompleteItem>, filter: bool) -> HookProvider {
    let items = Arc::new(items);
    hook_provider(move |lines, _cursor_line, cursor_col, _options| {
        let before = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        let _argument = match_argtest(&before)?;
        let filtered: Vec<AutocompleteItem> = if filter {
            items
                .iter()
                .filter(|item| item.value.starts_with(&argument_prefix(&before)))
                .cloned()
                .collect()
        } else {
            items.as_ref().clone()
        };
        if filtered.is_empty() {
            return None;
        }
        Some(AutocompleteSuggestions {
            items: filtered,
            prefix: argument_prefix(&before),
        })
    })
}

/// Upstream's `beforeCursor.match(/^\/argtest\s+(\S+)$/)` — the typed
/// argument (or `None` outside the argument context).
fn match_argtest(before: &str) -> Option<String> {
    let rest = before.strip_prefix("/argtest")?;
    if !rest.starts_with(' ') {
        return None;
    }
    let argument = rest.trim_start();
    if argument.is_empty() || argument.contains(' ') {
        return None;
    }
    Some(argument.to_string())
}

fn argument_prefix(before: &str) -> String {
    match_argtest(before).unwrap_or_default()
}

fn type_argtest(editor: &Editor, typed: &str) {
    for character in "/argtest ".chars().chain(typed.chars()) {
        editor.handle_input(&character.to_string());
    }
}

#[test]
fn applies_exact_typed_slash_argument_value_on_enter_even_when_first_item_is_highlighted() {
    let editor = editor();
    let provider = argtest_provider(
        vec![
            item("one", "one"),
            item("two", "two"),
            item("three", "three"),
        ],
        true,
    );

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "/argtest two"
    type_argtest(&editor, "two");

    assert_eq!(editor.get_text(), "/argtest two");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Press Enter - should apply the exact typed value "two", not the
    // first item
    editor.handle_input("\r");

    // The exact typed value "two" should be retained
    assert_eq!(editor.get_text(), "/argtest two");
}

#[test]
fn selects_first_prefix_match_on_enter_when_typed_arg_is_not_exact_match() {
    let editor = editor();
    let provider = argtest_provider(
        vec![
            item("two", "two"),
            item("three", "three"),
            item("twelve", "twelve"),
        ],
        true,
    );

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "/argtest t" - filtered to [two, three, twelve], prefix "t"
    // matches "two" first
    type_argtest(&editor, "t");

    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Press Enter - "t" prefix matches "two" (first in list), so "two" is
    // applied
    editor.handle_input("\r");
    assert_eq!(editor.get_text(), "/argtest two");
}

#[test]
fn highlights_unique_prefix_match_as_user_types_before_full_exact_match() {
    let editor = editor();
    // The provider returns all items unfiltered (like real extensions do).
    let provider = argtest_provider(
        vec![
            item("one", "one"),
            item("two", "two"),
            item("three", "three"),
        ],
        false,
    );

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "/argtest tw" - "tw" is a prefix of only "two"
    type_argtest(&editor, "tw");

    assert_eq!(editor.get_text(), "/argtest tw");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Press Enter - "tw" uniquely matches "two", so "two" should be applied
    editor.handle_input("\r");
    assert_eq!(editor.get_text(), "/argtest two");
}

#[test]
fn selects_first_prefix_match_when_multiple_items_match() {
    let editor = editor();
    // The provider returns all items unfiltered.
    let provider = argtest_provider(
        vec![
            item("one", "one"),
            item("two", "two"),
            item("three", "three"),
        ],
        false,
    );

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "/argtest t" - "t" is a prefix of both "two" and "three"
    type_argtest(&editor, "t");

    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Press Enter - "t" matches "two" first, so "two" is selected
    editor.handle_input("\r");
    assert_eq!(editor.get_text(), "/argtest two");
}

#[test]
fn works_for_builtin_style_command_argument_completion_path() {
    let editor = editor();
    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let before = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        // `/model <model>` context, `[^ ]+` matching any non-space text.
        let model_text = match_model(&before)?;
        let all_models = vec![
            item("gpt-4o", "gpt-4o"),
            item("gpt-4o-mini", "gpt-4o-mini"),
            item("claude-sonnet", "claude-sonnet"),
        ];
        let filtered: Vec<AutocompleteItem> = all_models
            .into_iter()
            .filter(|model| model.value.starts_with(&model_text))
            .collect();
        if filtered.is_empty() {
            return None;
        }
        Some(AutocompleteSuggestions {
            items: filtered,
            prefix: model_text,
        })
    });

    editor.set_autocomplete_provider(Arc::new(provider));

    // Type "/model gpt-4o-mini" - exact match for second item in list
    for character in "/model gpt-4o-mini".chars() {
        editor.handle_input(&character.to_string());
    }

    assert_eq!(editor.get_text(), "/model gpt-4o-mini");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Press Enter - should retain exact typed value, not apply first
    // highlighted item
    editor.handle_input("\r");

    // The exact typed value should be retained
    assert_eq!(editor.get_text(), "/model gpt-4o-mini");
}

/// Upstream's `/model\s+(\S+)$` match.
fn match_model(before: &str) -> Option<String> {
    let rest = before.strip_prefix("/model")?;
    let argument = rest.trim_start();
    if argument.is_empty() || argument.contains(' ') {
        return None;
    }
    Some(argument.to_string())
}

#[test]
fn awaits_async_slash_command_argument_completions() {
    let editor = editor();
    let provider = CombinedAutocompleteProvider::new(
        vec![CommandEntry::Slash(SlashCommand {
            name: "load-skills".to_string(),
            description: Some("Load skills".to_string()),
            argument_hint: None,
            get_argument_completions: Some(Box::new(|prefix| {
                if prefix.starts_with('s') {
                    Some(vec![item("skill-a", "skill-a")])
                } else {
                    None
                }
            })),
        })],
        std::env::temp_dir().to_string_lossy().to_string(),
        None,
    );
    editor.set_autocomplete_provider(Arc::new(provider));
    editor.set_text("/load-skills ");

    editor.handle_input("s");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    editor.handle_input("\t");
    assert_eq!(editor.get_text(), "/load-skills skill-a");
    assert!(!editor.is_showing_autocomplete());
}

#[test]
fn ignores_invalid_slash_command_argument_completion_results() {
    let editor = editor();
    // Upstream hands the provider a `getArgumentCompletions` that returns
    // `"not-an-array"` and relies on the runtime `Array.isArray` check; the
    // typed hook makes a non-array unrepresentable, so the port drives the
    // degradation path with `None`, the same closed picker.
    let provider = CombinedAutocompleteProvider::new(
        vec![CommandEntry::Slash(SlashCommand {
            name: "load-skills".to_string(),
            description: Some("Load skills".to_string()),
            argument_hint: None,
            get_argument_completions: Some(Box::new(|_prefix| None)),
        })],
        std::env::temp_dir().to_string_lossy().to_string(),
        None,
    );
    editor.set_autocomplete_provider(Arc::new(provider));
    editor.set_text("/load-skills ");

    editor.handle_input("s");
    flush_autocomplete(&editor);
    assert!(!editor.is_showing_autocomplete());
    assert_eq!(editor.get_text(), "/load-skills s");
}

#[test]
fn does_not_show_argument_completions_when_command_has_no_argument_completer() {
    let editor = editor();
    let provider = CombinedAutocompleteProvider::new(
        vec![
            CommandEntry::Slash(SlashCommand {
                name: "help".to_string(),
                description: Some("Show help".to_string()),
                argument_hint: None,
                get_argument_completions: None,
            }),
            CommandEntry::Slash(SlashCommand {
                name: "model".to_string(),
                description: Some("Switch model".to_string()),
                argument_hint: None,
                get_argument_completions: Some(Box::new(|_prefix| {
                    Some(vec![item("claude-opus", "claude-opus")])
                })),
            }),
        ],
        std::env::temp_dir().to_string_lossy().to_string(),
        None,
    );
    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("/");
    editor.handle_input("h");
    editor.handle_input("e");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    editor.handle_input("\t");
    assert_eq!(editor.get_text(), "/help ");
    assert!(!editor.is_showing_autocomplete());
}

// --- boundary suite: the dropdown's render and mouse paths ---------------

#[test]
fn renders_the_dropdown_below_the_editor_and_routes_its_clicks() {
    let editor = editor();
    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        if prefix.starts_with('/') {
            return Some(AutocompleteSuggestions {
                items: vec![item("cmd", "cmd"), item("cat", "cat")],
                prefix: prefix.clone(),
            });
        }
        None
    });
    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("/");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // The dropdown renders below the bottom border: top border, one text
    // row, bottom border, then the picker rows.
    let rendered = editor.render(80);
    let rendered: Vec<String> = rendered.iter().map(|line| strip_ansi(line)).collect();
    let dropdown_rows = &rendered[3..];
    assert!(
        dropdown_rows.iter().any(|line| line.contains("cmd")),
        "the dropdown renders after the bottom border: {rendered:?}"
    );

    // A click on the second picker row applies that suggestion.
    let press = pi_tui::tui::TuiMouseEvent {
        event_type: pi_tui::tui::TuiMouseEventType::Press,
        button: pi_tui::tui::TuiMouseButton::Left,
        x: 3,
        y: 4,
        screen_x: 3,
        screen_y: 4,
        width: 80,
        height: 24,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    };
    assert!(Component::handle_mouse(&editor, &press).is_some());
    let click = pi_tui::tui::TuiMouseEvent {
        event_type: pi_tui::tui::TuiMouseEventType::Click,
        click_count: Some(1),
        ..press
    };
    assert!(Component::handle_mouse(&editor, &click).is_some());
    // The hook provider's applyCompletion replaces the "/" prefix with the
    // bare value, so the line becomes "cat".
    assert_eq!(editor.get_text(), "cat");
    assert!(!editor.is_showing_autocomplete());
}

#[test]
fn cancel_key_closes_the_picker_and_restores_typing() {
    let editor = editor();
    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        if prefix.starts_with('/') {
            return Some(AutocompleteSuggestions {
                items: vec![item("cmd", "cmd")],
                prefix: prefix.clone(),
            });
        }
        None
    });
    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("/");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Escape closes the picker; typing continues normally.
    editor.handle_input("\x1b");
    assert!(!editor.is_showing_autocomplete());
    editor.handle_input("c");
    assert_eq!(editor.get_text(), "/c");
}

#[test]
fn clamps_the_dropdown_row_budget_from_options() {
    let clamped = editor_with_options(EditorOptions {
        autocomplete_max_visible: 2,
        ..EditorOptions::default()
    });
    // Upstream clamps the constructor value into [3, 20] as well.
    assert_eq!(clamped.get_autocomplete_max_visible(), 3);
    clamped.set_autocomplete_max_visible(50);
    assert_eq!(clamped.get_autocomplete_max_visible(), 20);
    clamped.set_autocomplete_max_visible(0);
    assert_eq!(clamped.get_autocomplete_max_visible(), 3);

    let fresh = editor();
    assert_eq!(fresh.get_autocomplete_max_visible(), 5);
}

#[test]
fn the_force_gate_blocks_tab_completion_when_the_provider_declines() {
    struct DecliningProvider;

    impl AutocompleteProvider for DecliningProvider {
        fn get_suggestions(
            &self,
            _lines: &[String],
            _cursor_line: usize,
            _cursor_col: usize,
            _options: &AutocompleteQueryOptions,
        ) -> Option<AutocompleteSuggestions> {
            panic!("the forced query must not reach the provider");
        }

        fn apply_completion(
            &self,
            lines: &[String],
            cursor_line: usize,
            cursor_col: usize,
            _item: &AutocompleteItem,
            _prefix: &str,
        ) -> AutocompleteApplyResult {
            AutocompleteApplyResult {
                lines: lines.to_vec(),
                cursor_line,
                cursor_col,
            }
        }

        fn should_trigger_file_completion(
            &self,
            _lines: &[String],
            _cursor_line: usize,
            _cursor_col: usize,
        ) -> bool {
            false
        }
    }

    let editor = editor();
    editor.set_autocomplete_provider(Arc::new(DecliningProvider));

    // Plain text plus Tab: the gate declines, no request runs, the picker
    // never opens, and the Tab key is consumed.
    editor.handle_input("a");
    editor.handle_input("b");
    editor.handle_input("\t");
    flush_autocomplete(&editor);
    assert!(!editor.is_showing_autocomplete());
}

#[test]
fn the_force_gate_passes_and_completes_plain_text() {
    struct YesProvider;

    impl AutocompleteProvider for YesProvider {
        fn get_suggestions(
            &self,
            lines: &[String],
            _cursor_line: usize,
            cursor_col: usize,
            options: &AutocompleteQueryOptions,
        ) -> Option<AutocompleteSuggestions> {
            if !options.force {
                return None;
            }
            let prefix = lines.first().map_or(String::new(), |line| {
                line[..cursor_col.min(line.len())].to_string()
            });
            Some(AutocompleteSuggestions {
                items: vec![item("file.txt", "file.txt"), item("docs/", "docs/")],
                prefix,
            })
        }

        fn apply_completion(
            &self,
            lines: &[String],
            cursor_line: usize,
            cursor_col: usize,
            item: &AutocompleteItem,
            prefix: &str,
        ) -> AutocompleteApplyResult {
            standard_apply_completion(lines, cursor_line, cursor_col, item, prefix)
        }

        fn should_trigger_file_completion(
            &self,
            _lines: &[String],
            _cursor_line: usize,
            _cursor_col: usize,
        ) -> bool {
            true
        }
    }

    let editor = editor();
    editor.set_autocomplete_provider(Arc::new(YesProvider));

    // Tab on plain text forces the file query; two suggestions open the
    // menu, and a second Tab applies the highlighted first row.
    editor.handle_input("a");
    editor.handle_input("b");
    editor.handle_input("\t");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());
    editor.handle_input("\t");
    assert_eq!(editor.get_text(), "file.txt");
    assert!(!editor.is_showing_autocomplete());
}

#[test]
fn undo_restores_the_pre_completion_text_across_the_request_pipeline() {
    let editor = editor();
    // The provider completes "@" tokens; the request resolves after the
    // keystrokes land.
    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        if !prefix.starts_with('@') {
            return None;
        }
        Some(AutocompleteSuggestions {
            items: vec![item("@main.ts", "main.ts")],
            prefix,
        })
    });
    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("@");
    editor.handle_input("m");
    thread::sleep(Duration::from_millis(50));
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // Tab applies, then undo reverts the splice.
    editor.handle_input("\t");
    assert_eq!(editor.get_text(), "@main.ts");
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "@m");
}

#[test]
fn the_stale_request_after_more_typing_does_not_overwrite_the_picker() {
    let editor = editor();
    // The provider always answers with the "@x" suggestion.
    let provider = hook_provider(|lines, _cursor_line, cursor_col, _options| {
        let prefix = lines.first().map_or(String::new(), |line| {
            line[..cursor_col.min(line.len())].to_string()
        });
        Some(AutocompleteSuggestions {
            items: vec![item("@x", "x")],
            prefix,
        })
    });
    editor.set_autocomplete_provider(Arc::new(provider));

    editor.handle_input("@");
    thread::sleep(Duration::from_millis(50));
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    // More typing re-queries; the older result must not replace the
    // newer picker state (upstream's snapshot staleness check).
    editor.handle_input("y");
    flush_autocomplete(&editor);
    editor.handle_input("z");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());
    assert_eq!(editor.get_text(), "@yz");
}

#[test]
fn multiline_text_positions_the_picker_on_the_last_row() {
    let editor = editor();
    let provider = hook_provider(|lines, cursor_line, cursor_col, _options| {
        let current = lines.get(cursor_line).cloned().unwrap_or_default();
        let prefix = current[..cursor_col.min(current.len())].to_string();
        if prefix.starts_with('@') {
            return Some(AutocompleteSuggestions {
                items: vec![item("@main.ts", "main.ts")],
                prefix,
            });
        }
        None
    });
    editor.set_autocomplete_provider(Arc::new(provider));

    // setText never triggers; the typed "@" on the second line does.
    editor.set_text("first\n");
    editor.handle_input("@");
    flush_autocomplete(&editor);
    assert!(editor.is_showing_autocomplete());

    let rendered = editor.render(80);
    let rendered: Vec<String> = rendered.iter().map(|line| strip_ansi(line)).collect();
    // The picker renders after the second text row and its borders.
    assert!(
        rendered.iter().any(|line| line.contains("main.ts")),
        "the dropdown renders for multi-line text: {rendered:?}"
    );
    assert_eq!(editor.get_text(), "first\n@");
}
