//! The `experimental.ts` cases, ported at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use super::*;

fn env(value: Option<&str>) -> EnvLookup {
    let value = value.map(ToString::to_string);
    Box::new(move |key| {
        if key == ENV_EXPERIMENTAL {
            value.clone()
        } else {
            None
        }
    })
}

#[test]
fn only_the_exact_opt_in_value_enables_experimental_features() {
    assert!(are_experimental_features_enabled_with(&env(Some("1"))));
    assert!(!are_experimental_features_enabled_with(&env(Some("0"))));
    assert!(!are_experimental_features_enabled_with(&env(None)));
}
