use super::*;

#[test]
fn finds_an_overlap_shorter_than_the_long_probe() {
    assert_eq!(overlap("abcdefgh", "defghxyz", 65_536, 64, 8), 5);
}

#[test]
fn honors_a_disabled_scan() {
    assert_eq!(overlap("abcdef", "defghi", 0, 64, 8), 0);
}

#[test]
fn spells_segments_and_errors() {
    assert_eq!(Seg::Key("text".to_string()).to_string(), "text");
    assert_eq!(Seg::Index(7).to_string(), "7");
    let error = UnsafePathError {
        segment: Seg::Key("__proto__".to_string()),
    };
    assert_eq!(error.to_string(), "unsafe path segment: __proto__");
}

#[test]
fn overlap_returns_zero_without_any_tail_or_probe() {
    assert_eq!(overlap("", "b", 64, 64, 8), 0);
    assert_eq!(overlap("a", "", 64, 64, 8), 0);
    assert_eq!(overlap("abc", "b", 0, 64, 8), 0);
    // A zero probe degenerates to the single-character pass.
    assert_eq!(overlap("abc", "cb", 3, 0, 8), 1);
}

#[test]
fn overlap_scan_window_and_candidate_bound() {
    // A scan smaller than the tail keeps exactly that many characters.
    assert_eq!(overlap("abcd", "cd", 2, 64, 8), 2);
    // A scan larger than the input keeps all of it.
    assert_eq!(overlap("abc", "c", 4, 64, 8), 1);
    // A repetitive tail exhausts the candidate bound and gives up,
    // emitting a set instead of a guessed overlap.
    assert_eq!(overlap(&"ab".repeat(60), "ab", 128, 1, 1), 0);
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
