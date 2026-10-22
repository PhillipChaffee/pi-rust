use super::*;

#[test]
fn spells_every_wire_code_round_trip() {
    let codes = [
        "service_not_allowed",
        "service_not_found",
        "service_mode_mismatch",
        "service_member_not_found",
        "service_member_mismatch",
        "service_instance_not_found",
        "service_stale_instance",
        "service_invalid_value",
    ];
    for code in codes {
        let Some(parsed) = RemoteServiceErrorCode::parse(code) else {
            unreachable!("the wire code parses: {code}")
        };
        assert_eq!(parsed.as_str(), code);
        assert_eq!(parsed.to_string(), code);
    }
    assert_eq!(RemoteServiceErrorCode::parse("unknown"), None);
}

#[test]
fn spells_and_converts_error_shapes() {
    let remote = ChordError::from(RemoteServiceError::new(
        RemoteServiceErrorCode::ServiceNotFound,
        "service_not_found",
    ));
    assert_eq!(remote.to_string(), "service_not_found");
    // The remote error spells itself the same way on its own.
    let bare = RemoteServiceError::new(
        RemoteServiceErrorCode::ServiceMemberMismatch,
        "member mismatch",
    );
    assert_eq!(bare.to_string(), "member mismatch");
    assert_eq!(ChordError::from("plain").to_string(), "plain");
    assert_eq!(ChordError::from(String::from("owned")).to_string(), "owned");
    assert_eq!(ChordError::message("plain").to_string(), "plain");
    let delta = ChordError::from(crate::delta::DeltaError::from(crate::delta::OpShapeError {
        message: "t arity".to_string(),
    }));
    assert_eq!(delta.to_string(), "t arity");
}

#[test]
fn aggregates_one_or_many_failures() {
    assert!(collect_errors(Vec::new(), "unused").is_none());
    let Some(single) = collect_errors(vec![ChordError::Message("only".to_string())], "wrapper")
    else {
        unreachable!("one failure")
    };
    assert_eq!(single.to_string(), "only");
    let Some(aggregate) = collect_errors(
        vec![
            ChordError::Message("first".to_string()),
            ChordError::Message("second".to_string()),
        ],
        "wrapper",
    ) else {
        unreachable!("two failures aggregate")
    };
    assert_eq!(aggregate.to_string(), "wrapper (2 errors: first,  second)");
    assert!(
        matches!(aggregate, ChordError::Aggregate(..)),
        "several failures aggregate"
    );
}
