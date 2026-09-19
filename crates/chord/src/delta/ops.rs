//! Delta path vocabulary, path-safety rejection, and the string-overlap
//! scan, ported from upstream `src/delta/index.ts`.
//!

//
//! A delta operation addresses one value through a [`Path`] of object keys
//! and array indices. Reserved segments stay rejected wherever a path is
//! data: the applier performs `parent[key] = value` writes, so
//! `["s", ["__proto__", "isAdmin"], true]` would pollute `Object.prototype`
//! for the whole process. Ops arrive from a facet, a plugin compartment, or
//! a tool whose details may echo model output, so none of it is trusted.

use std::fmt;

/// One path segment: an object key or an array index.
///
/// Upstream spells the segment `string | number`; the enum pins the number
/// half as a non-negative integer, so the negative and fractional segments
/// upstream rejects at validation time cannot be constructed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seg {
    /// An object property key, spelled as the producer sent it.
    Key(String),
    /// A zero-based array index.
    Index(usize),
}

impl fmt::Display for Seg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Key(key) => f.write_str(key),
            Self::Index(index) => write!(f, "{index}"),
        }
    }
}

/// A sequence of object keys and array indices addressing one value in the
/// state tree, e.g. `["message", "content", 0, "text"]`.
pub type Path = Vec<Seg>;

/// Segments that reach the prototype chain.
///
/// `JSON.parse` on its own makes `__proto__` an own property; what is not
/// safe is `parent[key] = value`, which is exactly what an applier does, and
/// paths are data: `["s", ["__proto__", "isAdmin"], true]` would pollute
/// `Object.prototype` for the whole process.
pub const RESERVED_SEGMENTS: [&str; 3] = ["__proto__", "constructor", "prototype"];

/// A path segment that would reach the prototype chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsafePathError {
    /// The rejected segment, spelled as the producer sent it.
    pub segment: Seg,
}

impl fmt::Display for UnsafePathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsafe path segment: {}", self.segment)
    }
}

impl std::error::Error for UnsafePathError {}

/// Rejects paths whose segments would escape the value into the prototype
/// chain.
///
/// Upstream also rejects negative and non-integer segments here; [`Seg`]
/// pins indices as `usize`, so those wire forms cannot be built in the port.
///
/// # Errors
/// Returns [`UnsafePathError`] when a key segment is one of
/// [`RESERVED_SEGMENTS`].
pub fn assert_safe_path(path: &[Seg]) -> Result<(), UnsafePathError> {
    for segment in path {
        if let Seg::Key(key) = segment
            && RESERVED_SEGMENTS.contains(&key.as_str())
        {
            return Err(UnsafePathError {
                segment: segment.clone(),
            });
        }
    }
    Ok(())
}

/// The prefix of `s` spanning its first `count` characters, or all of `s`
/// when it is shorter.
fn char_prefix(s: &str, count: usize) -> &str {
    match s.char_indices().nth(count) {
        Some((end, _)) => &s[..end],
        None => s,
    }
}

/// Longest suffix of `a` that is a prefix of `b`, counted in [`char`]
/// units.
///
/// Upstream counts UTF-16 code units; this port counts Unicode scalar values
/// and every producer and consumer inside the port counts the same way, so
/// only text carried across from a TypeScript side sees different counts on
/// astral-plane characters.
///
/// Probes with substring search and then verifies exact equality, so the
/// hot loops stay native. A probe of length `probe` can only find overlaps
/// of at least that length, so a long head is tried first — few candidates,
/// and it catches the large overlaps a rolling window produces — then a
/// single character, which finds any overlap at the cost of more
/// candidates. Candidates are bounded because repetitive output, a build
/// log or any run of one character, makes a long head match at thousands
/// of positions. Giving up returns 0, which emits a set: larger, never
/// wrong.
///
/// `scan` bounds how much of `a`'s tail may participate, in chars. Upstream
/// defaults `probe` to 64 and `max_candidates` to 8; the tracker passes
/// those defaults.
#[must_use]
pub fn overlap(a: &str, b: &str, scan: usize, probe: usize, max_candidates: usize) -> usize {
    if a.is_empty() || b.is_empty() || scan == 0 {
        return 0;
    }
    // `rev().nth(scan)` yields a start offset only when `a` holds more than
    // `scan` characters, in which case the tail keeps exactly `scan` of them.
    let tail: &str = match a.char_indices().rev().nth(scan) {
        Some((start, _)) => &a[start..],
        None => a,
    };
    let b_chars = b.chars().count();
    // A zero probe degenerates to an empty head, which finds nothing useful;
    // the single-character pass below still produces a verified answer.
    for head_len in [probe.min(b_chars), 1] {
        if head_len == 0 {
            continue;
        }
        let head = char_prefix(b, head_len);
        let mut tried = 0usize;
        let mut cursor = 0usize;
        while let Some(found) = tail[cursor..].find(head) {
            let start = cursor + found;
            tried += 1;
            if tried > max_candidates {
                break;
            }
            let n = tail[start..].chars().count();
            if n <= b_chars && tail[start..] == *char_prefix(b, n) {
                return n;
            }
            // Step one character past the match: overlapping occurrences of a
            // self-overlapping head are legitimate candidates.
            cursor = match tail[start..].char_indices().nth(1) {
                Some((offset, _)) => start + offset,
                None => tail.len(),
            };
        }
        if head_len == 1 {
            break;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_an_overlap_shorter_than_the_long_probe() {
        assert_eq!(overlap("abcdefgh", "defghxyz", 65_536, 64, 8), 5);
    }

    #[test]
    fn honors_a_disabled_scan() {
        assert_eq!(overlap("abcdef", "defghi", 0, 64, 8), 0);
    }

    // Upstream exercises `assertSafePath` only through the apply and decoder
    // rejection cases, which land with the op vocabulary; this direct check
    // pins the reserved-key surface on its own until then.
    #[test]
    fn rejects_reserved_keys_and_accepts_safe_paths() {
        let safe = [
            Seg::Key("message".into()),
            Seg::Index(0),
            Seg::Key("text".into()),
        ];
        assert_eq!(assert_safe_path(&safe), Ok(()));
        for key in RESERVED_SEGMENTS {
            let path = [Seg::Key(key.into())];
            assert_eq!(
                assert_safe_path(&path),
                Err(UnsafePathError {
                    segment: Seg::Key(key.into()),
                })
            );
        }
    }
}
