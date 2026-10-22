use super::*;
use crate::context::Context;
use crate::test_support::{jb, jn, js};

#[test]
fn spells_json_shapes_and_escapes_control_characters() {
    assert_eq!(JsonValue::string("text"), js("text"));
    assert_eq!(jn(1.0).as_number(), Some(1.0));
    assert_eq!(js("x").as_number(), None);
    let mut null = JsonValue::Null;
    assert!(null.as_object_mut().is_none());
    assert_eq!(JsonValue::Null.to_json_string(), "null");
    assert_eq!(jb(true).to_json_string(), "true");
    assert_eq!(jb(false).to_json_string(), "false");
    assert_eq!(
        js("quote\"back\\newline\nreturn\rtab\tbackspace\u{08}form\u{0C}unit\u{01}")
            .to_json_string(),
        "\"quote\\\"back\\\\newline\\nreturn\\rtab\\tbackspace\\bform\\funit\\u0001\""
    );
}

#[test]
fn spells_the_vocabulary_debug_surfaces() {
    let snapshot = ServiceSubscriptionSnapshot {
        mode: ServiceMode::Singleton,
        service_id: "test.subscription".to_string(),
        instances: Vec::new(),
    };
    let subscription = ServiceSubscription {
        snapshot,
        activate: Box::new(|| Ok(())),
        close: Box::new(|_context: Option<Context>| Box::pin(std::future::ready(Ok(())))),
    };
    assert!(format!("{subscription:?}").contains("test.subscription"));
    let options = RemoteServiceSourceOpenOptions {
        services: Vec::new(),
        assert_access: crate::handle::allow_access(),
        on_error: crate::handle::no_error_reporter(),
    };
    assert!(format!("{options:?}").contains("services"));
    let facet = FacetDef {
        id: "test.facet".to_string(),
        setup: Rc::new(|_env: &mut crate::facets::host::FacetEnvironment| ()),
    };
    assert!(format!("{facet:?}").contains("test.facet"));
    let loaded = LoadedFacets {
        facets: vec![facet.clone()],
        dispose: crate::handle::sync_disposal(|| Ok(())),
    };
    assert!(format!("{loaded:?}").contains("test.facet"));
    // The delivery kind spells its wire form.
    assert_eq!(ReplicatedStateDeliveryKind::Hydrate.as_str(), "hydrate");
    assert_eq!(ReplicatedStateDeliveryKind::Update.to_string(), "update");
    // ServiceMode parses its wire spelling and spells itself back.
    assert_eq!(
        ServiceMode::parse("singleton"),
        Some(ServiceMode::Singleton)
    );
    assert_eq!(ServiceMode::parse("keyed"), Some(ServiceMode::Keyed));
    assert_eq!(ServiceMode::parse("weird"), None);
    assert_eq!(ServiceMode::Keyed.to_string(), "keyed");
    // Mutable object handles build, inspect, and remove entries.
    let mut object = JsonObject::new();
    assert!(object.is_empty());
    object.set("alpha", js("one"));
    assert_eq!(object.remove("alpha"), Some(js("one")));
    assert!(object.get("alpha").is_none());
    assert!(object.remove("absent").is_none());
    // Numbers build from integers and floats, and booleans read back.
    assert_eq!(JsonValue::number(1.5), Some(jn(1.5)));
    assert_eq!(JsonValue::number(f64::NAN), None);
    assert_eq!(
        JsonValue::Number(JsonNumber::from(3i64)).to_json_string(),
        "3"
    );
    assert_eq!(
        JsonValue::Number(JsonNumber::from(4u64)).to_json_string(),
        "4"
    );
    assert_eq!(jb(true).as_bool(), Some(true));
    assert_eq!(js("not a bool").as_bool(), None);
    // A facet definition clones with its setup shared.
    let cloned = facet.clone();
    assert_eq!(cloned.id, facet.id);
}
