use super::*;

use pi_ai::types::ImageContent;

#[test]
fn extension_mode_defaults_to_print() {
    assert_eq!(ExtensionMode::default(), ExtensionMode::Print);
    assert_eq!(InputSource::default(), InputSource::Interactive);
    for mode in [
        ExtensionMode::Tui,
        ExtensionMode::Rpc,
        ExtensionMode::Json,
        ExtensionMode::Print,
    ] {
        assert!(!format!("{mode:?}").is_empty());
    }
    for source in [
        InputSource::Interactive,
        InputSource::Rpc,
        InputSource::Extension,
    ] {
        assert!(!format!("{source:?}").is_empty());
    }
}

#[test]
fn every_reason_enum_debugs() {
    for reason in [
        SessionStartReason::Startup,
        SessionStartReason::Resume,
        SessionStartReason::New,
        SessionStartReason::Fork,
        SessionStartReason::Reload,
    ] {
        assert!(!format!("{reason:?}").is_empty());
    }
    for reason in [
        SessionShutdownReason::New,
        SessionShutdownReason::Resume,
        SessionShutdownReason::Fork,
        SessionShutdownReason::Quit,
    ] {
        assert!(!format!("{reason:?}").is_empty());
    }
    for reason in [
        CompactionReason::Manual,
        CompactionReason::Threshold,
        CompactionReason::Overflow,
    ] {
        assert!(!format!("{reason:?}").is_empty());
    }
    for source in [
        ModelSelectSource::Set,
        ModelSelectSource::Cycle,
        ModelSelectSource::Restore,
    ] {
        assert!(!format!("{source:?}").is_empty());
    }
    for position in [ForkPosition::Before, ForkPosition::At] {
        assert!(!format!("{position:?}").is_empty());
    }
    for behavior in [StreamingBehavior::Steer, StreamingBehavior::FollowUp] {
        assert!(!format!("{behavior:?}").is_empty());
    }
}

#[test]
fn payloads_debug_and_block_constructors_build() {
    let preparation = TreePreparation {
        target_id: "e-1".to_string(),
        old_leaf_id: Some("e-0".to_string()),
        common_ancestor_id: None,
        entries_to_summarize: Vec::new(),
        user_wants_summary: true,
        custom_instructions: Some("ci".to_string()),
        replace_instructions: Some(false),
        label: Some("l".to_string()),
    };
    let payload = CustomMessagePayload {
        custom_type: "artifact".to_string(),
        content: vec![
            text_block("hello"),
            image_block(ImageContent {
                data: "aGk=".to_string(),
                mime_type: "image/png".to_string(),
            }),
        ],
        display: true,
        details: None,
    };
    assert!(!format!("{preparation:?}").is_empty());
    assert!(!format!("{payload:?}").is_empty());
    assert!(matches!(
        text_block("hello"),
        pi_ai::types::UserBlock::Text(_)
    ));
    assert!(matches!(
        image_block(ImageContent {
            data: "aGk=".to_string(),
            mime_type: "image/png".to_string(),
        }),
        pi_ai::types::UserBlock::Image(_)
    ));
}
