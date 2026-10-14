//! The git metadata probe, upstream's `src/core/footer-data-provider.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! This module carries the piece the resource loader consumes —
//! [`find_git_paths`], the walk-up git-metadata reader that decides the
//! worktree context-file dedup. The rest of upstream's footer data
//! provider (branch/status collection, the file watchers, the change
//! events) renders the interactive footer and rides its consumer ticket
//! (#131).

use std::path::Path;

/// The git metadata paths one repo walk found, upstream's `GitPaths`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitPaths {
    /// The working tree root carrying the `.git` entry.
    pub repo_dir: String,
    /// The shared git directory — the worktree git dir's `commondir`
    /// target when linked, else the `.git` directory itself.
    pub common_git_dir: String,
    /// The git dir's `HEAD` file.
    pub head_path: String,
}

/// Find git metadata paths by walking up from the cwd, upstream's
/// `findGitPaths`.
///
/// Handles both regular repos (`.git` is a directory) and linked
/// worktrees (`.git` is a `gitdir:` file). `None` when no repo root sits
/// above the cwd, when a `gitdir:` target or its `HEAD` is missing, or
/// when the stat of a `.git` entry fails.
#[must_use]
pub fn find_git_paths(cwd: &str) -> Option<GitPaths> {
    let mut dir = cwd.to_string();
    loop {
        let git_path = Path::new(&dir).join(".git");
        if git_path.exists() {
            let Ok(fs_meta) = std::fs::metadata(&git_path) else {
                return None;
            };
            if fs_meta.is_file() {
                let Ok(content) = std::fs::read_to_string(&git_path) else {
                    return None;
                };
                let content = content.trim();
                if let Some(git_dir_raw) = content.strip_prefix("gitdir: ") {
                    let git_dir = resolve_against(&dir, git_dir_raw.trim());
                    let head_path = Path::new(&git_dir).join("HEAD");
                    if !head_path.exists() {
                        return None;
                    }
                    let common_dir_path = Path::new(&git_dir).join("commondir");
                    let common_git_dir = if common_dir_path.exists() {
                        let Ok(common_dir) = std::fs::read_to_string(&common_dir_path) else {
                            return None;
                        };
                        resolve_against(&git_dir, common_dir.trim())
                    } else {
                        git_dir
                    };
                    return Some(GitPaths {
                        repo_dir: dir,
                        common_git_dir,
                        head_path: head_path.to_string_lossy().into_owned(),
                    });
                }
            } else if fs_meta.is_dir() {
                let head_path = git_path.join("HEAD");
                if !head_path.exists() {
                    return None;
                }
                return Some(GitPaths {
                    repo_dir: dir,
                    common_git_dir: git_path.to_string_lossy().into_owned(),
                    head_path: head_path.to_string_lossy().into_owned(),
                });
            }
        }
        let parent = parent_dir(&dir)?;
        dir = parent;
    }
}

/// Resolve `target` against `base`, Node's `path.resolve` for the
/// absolute-or-relative targets git writes: the join normalizes lexically
/// (`..` pops, `.` drops), the way Node's resolver collapses the
/// `commondir` relative paths git writes.
fn resolve_against(base: &str, target: &str) -> String {
    if target.starts_with('/') {
        return normalize_segments(target);
    }
    let joined = if base.ends_with('/') {
        format!("{base}{target}")
    } else {
        format!("{base}/{target}")
    };
    normalize_segments(&joined)
}

/// Collapse `.` and `..` segments lexically, Node's `path.normalize`'
/// segment walk.
fn normalize_segments(joined: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in joined.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    let mut normalized = segments.join("/");
    if joined.starts_with('/') {
        normalized = format!("/{normalized}");
    }
    if normalized.is_empty() {
        normalized = "/".to_string();
    }
    normalized
}

/// The parent of a POSIX path, `None` at the root.
fn parent_dir(dir: &str) -> Option<String> {
    let trimmed = dir.trim_end_matches('/');
    let parent = match trimmed.rfind('/') {
        Some(0) => "/",
        Some(at) => &trimmed[..at],
        None => return None,
    };
    if parent == trimmed {
        return None;
    }
    Some(parent.to_string())
}
