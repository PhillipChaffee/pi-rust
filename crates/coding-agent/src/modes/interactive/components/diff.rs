//! Diff rendering for the edit tool, upstream
//! `src/modes/interactive/components/diff.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The renderer colors context, removed, and added lines and inverse-highlights
//! the changed words of a single removed/added pair. Restatement: the word
//! diff is a token-level longest-common-subsequence over ASCII word runs and
//! whitespace, where upstream's jsdiff also segments Unicode words and breaks
//! ties with its own Myers walk; only the coloring contract is pinned.

use regex::Regex;

use crate::modes::interactive::theme::theme::{inverse, theme};

/// One parsed diff row: the `+`/`-`/space prefix, its line number, and the
/// content after it.
struct ParsedDiffLine {
    prefix: char,
    line_num: String,
    content: String,
}

/// Parse a `+123 content` / `-123 content` / ` 123 content` row, upstream's
/// `parseDiffLine`.
fn parse_diff_line(line: &str) -> Option<ParsedDiffLine> {
    let pattern = Regex::new(r"^([+\s-])(\s*\d*)\s(.*)$").ok()?;
    let captures = pattern.captures(line)?;
    Some(ParsedDiffLine {
        prefix: captures.get(1)?.as_str().chars().next()?,
        line_num: captures.get(2)?.as_str().to_string(),
        content: captures.get(3)?.as_str().to_string(),
    })
}

/// Style `text` inverse for every changed token of a removed/added pair.
fn render_intra_line_diff(old_content: &str, new_content: &str) -> (String, String) {
    let old_tokens = tokenize(old_content);
    let new_tokens = tokenize(new_content);
    let common = lcs_marks(&old_tokens, &new_tokens);

    let mut removed_line = String::new();
    let mut added_line = String::new();
    let mut first_removed = true;
    let mut first_added = true;

    for (index, token) in old_tokens.iter().enumerate() {
        if common.old_changed[index] {
            removed_line.push_str(&changed_text(token, &mut first_removed));
        } else {
            removed_line.push_str(token);
        }
    }
    for (index, token) in new_tokens.iter().enumerate() {
        if common.new_changed[index] {
            added_line.push_str(&changed_text(token, &mut first_added));
        } else {
            added_line.push_str(token);
        }
    }

    (removed_line, added_line)
}

/// Style one changed token, keeping the first changed token's leading
/// whitespace unhighlighted, upstream's `isFirstRemoved` branch.
fn changed_text(token: &str, first: &mut bool) -> String {
    if *first {
        let trimmed = token.trim_start();
        let leading = &token[..token.len() - trimmed.len()];
        *first = false;
        return format!(
            "{leading}{}",
            if trimmed.is_empty() {
                String::new()
            } else {
                inverse(trimmed)
            }
        );
    }
    if token.is_empty() {
        String::new()
    } else {
        inverse(token)
    }
}

/// Split into word runs and whitespace runs, upstream's `diffWords` tokens.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_is_space: Option<bool> = None;
    for character in text.chars() {
        let is_space = character.is_whitespace();
        if current_is_space.is_some_and(|previous| previous != is_space) {
            tokens.push(std::mem::take(&mut current));
        }
        current_is_space = Some(is_space);
        current.push(character);
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Which tokens are changed on each side, from a longest-common-subsequence
/// alignment of the two token lists.
struct ChangeMarks {
    old_changed: Vec<bool>,
    new_changed: Vec<bool>,
}

fn lcs_marks(old: &[String], new: &[String]) -> ChangeMarks {
    let mut lengths = vec![vec![0usize; new.len() + 1]; old.len() + 1];
    for old_index in (0..old.len()).rev() {
        for new_index in (0..new.len()).rev() {
            lengths[old_index][new_index] = if old[old_index] == new[new_index] {
                lengths[old_index + 1][new_index + 1] + 1
            } else {
                lengths[old_index + 1][new_index].max(lengths[old_index][new_index + 1])
            };
        }
    }

    let mut old_changed = vec![true; old.len()];
    let mut new_changed = vec![true; new.len()];
    let (mut old_index, mut new_index) = (0, 0);
    while old_index < old.len() && new_index < new.len() {
        if old[old_index] == new[new_index] {
            old_changed[old_index] = false;
            new_changed[new_index] = false;
            old_index += 1;
            new_index += 1;
        } else if lengths[old_index + 1][new_index] >= lengths[old_index][new_index + 1] {
            old_index += 1;
        } else {
            new_index += 1;
        }
    }

    ChangeMarks {
        old_changed,
        new_changed,
    }
}

/// Render a numbered diff string with colored lines and intra-line change
/// highlighting, upstream's `renderDiff`.
#[must_use]
pub fn render_diff(diff_text: &str) -> String {
    let theme = theme();
    let lines: Vec<&str> = diff_text.split('\n').collect();
    let mut result: Vec<String> = Vec::new();
    let mut index = 0;

    while index < lines.len() {
        let line = lines[index];
        let Some(parsed) = parse_diff_line(line) else {
            result.push(theme.fg("toolDiffContext", line));
            index += 1;
            continue;
        };

        if parsed.prefix == '-' {
            let mut removed: Vec<(String, String)> = Vec::new();
            while index < lines.len() {
                let Some(next) = parse_diff_line(lines[index]) else {
                    break;
                };
                if next.prefix != '-' {
                    break;
                }
                removed.push((next.line_num, next.content));
                index += 1;
            }

            let mut added: Vec<(String, String)> = Vec::new();
            while index < lines.len() {
                let Some(next) = parse_diff_line(lines[index]) else {
                    break;
                };
                if next.prefix != '+' {
                    break;
                }
                added.push((next.line_num, next.content));
                index += 1;
            }

            if removed.len() == 1 && added.len() == 1 {
                let (removed_line_num, removed_content) = &removed[0];
                let (added_line_num, added_content) = &added[0];
                let (removed_line, added_line) = render_intra_line_diff(
                    &removed_content.replace('\t', "   "),
                    &added_content.replace('\t', "   "),
                );
                result.push(theme.fg(
                    "toolDiffRemoved",
                    &format!("-{removed_line_num} {removed_line}"),
                ));
                result.push(theme.fg("toolDiffAdded", &format!("+{added_line_num} {added_line}")));
            } else {
                for (line_num, content) in &removed {
                    result.push(theme.fg(
                        "toolDiffRemoved",
                        &format!("-{line_num} {}", content.replace('\t', "   ")),
                    ));
                }
                for (line_num, content) in &added {
                    result.push(theme.fg(
                        "toolDiffAdded",
                        &format!("+{line_num} {}", content.replace('\t', "   ")),
                    ));
                }
            }
        } else if parsed.prefix == '+' {
            result.push(theme.fg(
                "toolDiffAdded",
                &format!(
                    "+{} {}",
                    parsed.line_num,
                    parsed.content.replace('\t', "   ")
                ),
            ));
            index += 1;
        } else {
            result.push(theme.fg(
                "toolDiffContext",
                &format!(
                    " {} {}",
                    parsed.line_num,
                    parsed.content.replace('\t', "   ")
                ),
            ));
            index += 1;
        }
    }

    result.join("\n")
}
