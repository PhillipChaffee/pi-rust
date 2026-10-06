//! One-time deprecation warnings, upstream's `src/utils/deprecation.ts`.

use std::collections::HashSet;
use std::io::IsTerminal;
use std::sync::Mutex;

static EMITTED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

fn with_emitted<T>(body: impl FnOnce(&mut HashSet<String>) -> T) -> T {
    // The set lives one write at a time under the mutex; a poisoned lock
    // means a previous access panicked mid-write, which the belt treats as
    // the end of the process.
    let mut guard = EMITTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    body(guard.get_or_insert_with(HashSet::new))
}

/// Warn about a deprecated surface once per process.
///
/// The first call writes `Deprecation warning: <message>` to stderr —
/// chalk-yellow when stderr is a terminal, plain text otherwise — and every
/// later call with the same message is silent.
pub fn warn_deprecation(message: &str) {
    let mut warned = false;
    with_emitted(|set| warned = set.insert(message.to_string()));
    if warned {
        emit_deprecation_warning(message);
    }
}

/// Clear the deprecation warning state, upstream's
/// `clearDeprecationWarningsForTests`.
pub fn clear_deprecation_warnings_for_tests() {
    with_emitted(HashSet::clear);
}

fn emit_deprecation_warning(message: &str) {
    #[expect(
        clippy::print_stderr,
        reason = "console.warn is the surface upstream carries; the warning must reach the operator on stderr"
    )]
    fn write_line(line: &str) {
        eprintln!("{line}");
    }
    if std::io::stderr().is_terminal() {
        write_line(&format!("\x1b[33mDeprecation warning: {message}\x1b[39m"));
    } else {
        write_line(&format!("Deprecation warning: {message}"));
    }
}
