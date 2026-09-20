//! 1:1 port of `packages/tui/test/fuzzy.test.ts` (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).

use pi_tui::fuzzy::{fuzzy_filter, fuzzy_match};

mod fuzzy_match {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the empty-query score is the literal 0.0; no arithmetic accumulates"
    )]
    fn empty_query_matches_everything_with_score_0() {
        let result = fuzzy_match("", "anything");
        assert!(result.matches);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn query_longer_than_text_does_not_match() {
        let result = fuzzy_match("longquery", "short");
        assert!(!result.matches);
    }

    #[test]
    fn exact_match_has_good_score() {
        let result = fuzzy_match("test", "test");
        assert!(result.matches);
        // Should be negative due to consecutive bonuses
        assert!(result.score < 0.0);
    }

    #[test]
    fn characters_must_appear_in_order() {
        let match_in_order = fuzzy_match("abc", "aXbXc");
        assert!(match_in_order.matches);

        let match_out_of_order = fuzzy_match("abc", "cba");
        assert!(!match_out_of_order.matches);
    }

    #[test]
    fn case_insensitive_matching() {
        let result = fuzzy_match("ABC", "abc");
        assert!(result.matches);

        let result2 = fuzzy_match("abc", "ABC");
        assert!(result2.matches);
    }

    #[test]
    fn consecutive_matches_score_better_than_scattered_matches() {
        let consecutive = fuzzy_match("foo", "foobar");
        let scattered = fuzzy_match("foo", "f_o_o_bar");

        assert!(consecutive.matches);
        assert!(scattered.matches);
        assert!(consecutive.score < scattered.score);
    }

    #[test]
    fn word_boundary_matches_score_better() {
        let at_boundary = fuzzy_match("fb", "foo-bar");
        let not_at_boundary = fuzzy_match("fb", "afbx");

        assert!(at_boundary.matches);
        assert!(not_at_boundary.matches);
        assert!(at_boundary.score < not_at_boundary.score);
    }

    #[test]
    fn matches_swapped_alpha_numeric_tokens() {
        let result = fuzzy_match("codex52", "gpt-5.2-codex");
        assert!(result.matches);
    }
}

mod fuzzy_filter {
    use super::*;

    #[test]
    fn empty_query_returns_all_items_unchanged() {
        let items = vec!["apple", "banana", "cherry"];
        let result = fuzzy_filter(&items, "", |x| (*x).to_string());
        assert_eq!(result, items.iter().collect::<Vec<_>>());
    }

    #[test]
    fn filters_out_non_matching_items() {
        let items = vec!["apple", "banana", "cherry"];
        let result = fuzzy_filter(&items, "an", |x| (*x).to_string());
        assert!(result.contains(&&"banana"));
        assert!(!result.contains(&&"apple"));
        assert!(!result.contains(&&"cherry"));
    }

    #[test]
    fn sorts_results_by_match_quality() {
        let items = vec!["a_p_p", "app", "application"];
        let result = fuzzy_filter(&items, "app", |x| (*x).to_string());

        // "app" should be first (exact consecutive match at start)
        assert_eq!(result[0], &"app");
    }

    #[test]
    fn prioritizes_exact_matches_over_longer_prefix_matches() {
        let items = vec!["clone", "cl"];
        let result = fuzzy_filter(&items, "cl", |x| (*x).to_string());

        assert_eq!(result, vec![&"cl", &"clone"]);
    }

    #[test]
    fn works_with_custom_get_text_function() {
        struct Item {
            name: String,
            /// Mirrors upstream's `{ name, id }` item shape; the matcher only
            /// reads `name`.
            #[expect(dead_code, reason = "upstream payload this test never reads")]
            id: u32,
        }
        let items = vec![
            Item {
                name: "foo".into(),
                id: 1,
            },
            Item {
                name: "bar".into(),
                id: 2,
            },
            Item {
                name: "foobar".into(),
                id: 3,
            },
        ];
        let result = fuzzy_filter(&items, "foo", |item| item.name.clone());

        assert_eq!(result.len(), 2);
        let names: Vec<&str> = result.iter().map(|item| item.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"foobar"));
    }

    #[test]
    fn matches_slash_separated_provider_model_queries_against_reordered_text() {
        struct Model {
            id: String,
            provider: String,
        }
        let item = Model {
            id: "gpt-5.5".to_string(),
            provider: "openai-codex".to_string(),
        };
        let result = fuzzy_filter(
            std::slice::from_ref(&item),
            "openai-codex/gpt-5.5",
            |model| format!("{} {}", model.id, model.provider),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, item.id);
        assert_eq!(result[0].provider, item.provider);
    }
}
