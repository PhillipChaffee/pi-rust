//! pi-rust: Rust port of [earendil-works/pi](https://github.com/earendil-works/pi) (MIT).
//!
//! Bootstrap skeleton; the crate layout is decided by the port plan (see
//! AGENTS.md for the reference codebase and comment rules).

/// Crate version, from Cargo.
#[must_use]
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Placeholder entry point; the crate layout is decided by the port plan.
const fn main() {}

#[cfg(test)]
mod tests {
    use super::version;

    #[test]
    fn version_is_set() {
        assert!(!version().is_empty());
    }
}
