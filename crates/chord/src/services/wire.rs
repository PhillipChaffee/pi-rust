//! The `$chord.service` wire grammar, ported from upstream
//! `src/services/wire.ts`.
//!
//! The control vocabulary (`catalogue` / `subscribe` / `unsubscribe` on the
//! reserved `$chord.service` ID) plus the parse surface every remote
//! payload passes: snapshots, provider updates, calls, and catalogues, each
//! validated against strict key sets before use. Upstream parses `unknown`
//! wire data and asserts op shape at the boundary; the port's wire form is
//! the owned [`JsonValue`] tree, and the delta crate's op parsers do the
//! operation validation — [`Op::from_json`] for decoded payloads,
//! [`WireOp::from_json`] for wire payloads.

use crate::delta::{Op, WireOp};
use crate::errors::ChordError;
use crate::types::{
    JsonNumber, JsonObject, JsonValue, ServiceCall, ServiceCatalogueEntry, ServiceInstanceAddress,
    ServiceInstanceSnapshot, ServiceMemberSnapshot, ServiceMode, ServiceProviderUpdate,
    ServiceSubscriptionSnapshot,
};

/// One member snapshot on the wire, operations still encoded as wire ops.
#[derive(Debug, Clone)]
pub enum WireServiceMemberSnapshot {
    /// A method member.
    Method {
        /// The member name.
        name: String,
    },
    /// Replicated state.
    State {
        /// The member name.
        name: String,
        /// The provider's publication sequence.
        sequence: u64,
        /// The wire-form operation batch.
        ops: Vec<WireOp>,
    },
}

/// One live instance's members on the wire.
#[derive(Debug, Clone)]
pub struct WireServiceInstanceSnapshot {
    /// The address, present for keyed instances.
    pub instance: Option<ServiceInstanceAddress>,
    /// The members in registration order.
    pub members: Vec<WireServiceMemberSnapshot>,
}

/// The wire-form subscription snapshot: like
/// [`ServiceSubscriptionSnapshot`] with wire ops.
#[derive(Debug, Clone)]
pub struct WireServiceSubscriptionSnapshot {
    /// The subscribed service.
    pub service_id: String,
    /// The mode the subscription asked for.
    pub mode: ServiceMode,
    /// The live instances at subscription time.
    pub instances: Vec<WireServiceInstanceSnapshot>,
}

/// One provider update on the wire.
#[derive(Debug, Clone)]
pub enum WireServiceProviderUpdate {
    /// State changed on one member of one instance.
    State {
        /// The instance address, absent for singletons.
        instance: Option<ServiceInstanceAddress>,
        /// The state member that published.
        member: String,
        /// The provider's sequence for this publication.
        sequence: u64,
        /// The wire-form operation batch.
        ops: Vec<WireOp>,
    },
    /// The singleton became unavailable.
    Unavailable,
    /// The singleton was replaced.
    Replaced {
        /// The replacement's wire snapshot.
        snapshot: WireServiceInstanceSnapshot,
    },
    /// A keyed instance came up.
    Spawned {
        /// The new instance's snapshot.
        instance: WireServiceInstanceSnapshot,
    },
    /// A keyed instance closed.
    Closed {
        /// The address that closed.
        instance: ServiceInstanceAddress,
    },
}

/// One decoded `$chord.service` control call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceControlCall {
    /// List the provider's catalogue.
    Catalogue,
    /// Open a subscription.
    Subscribe {
        /// The consumer's subscription identity.
        subscription_id: String,
        /// The service to subscribe to.
        service_id: String,
        /// The mode to subscribe in.
        mode: ServiceMode,
    },
    /// Close one subscription.
    Unsubscribe {
        /// The subscription to close.
        subscription_id: String,
    },
}

/// The reserved control service ID the grammar routes through.
pub const SERVICE_CONTROL_ID: &str = "$chord.service";

const SERVICE_CATALOGUE_MEMBER: &str = "catalogue";
const SERVICE_SUBSCRIBE_MEMBER: &str = "subscribe";
const SERVICE_UNSUBSCRIBE_MEMBER: &str = "unsubscribe";

/// Builds the catalogue control call.
#[must_use]
pub fn create_service_catalogue_call() -> ServiceCall {
    ServiceCall {
        service_id: SERVICE_CONTROL_ID.to_string(),
        instance: None,
        member: SERVICE_CATALOGUE_MEMBER.to_string(),
        args: Vec::new(),
    }
}

/// Builds the subscribe control call.
#[must_use]
pub fn create_service_subscribe_call(subscription_id: &str, service_id: &str, mode: ServiceMode) -> ServiceCall {
    ServiceCall {
        service_id: SERVICE_CONTROL_ID.to_string(),
        instance: None,
        member: SERVICE_SUBSCRIBE_MEMBER.to_string(),
        args: vec![
            JsonValue::Str(subscription_id.to_string()),
            JsonValue::Str(service_id.to_string()),
            JsonValue::Str(mode.as_str().to_string()),
        ],
    }
}

/// Builds the unsubscribe control call.
#[must_use]
pub fn create_service_unsubscribe_call(subscription_id: &str) -> ServiceCall {
    ServiceCall {
        service_id: SERVICE_CONTROL_ID.to_string(),
        instance: None,
        member: SERVICE_UNSUBSCRIBE_MEMBER.to_string(),
        args: vec![JsonValue::Str(subscription_id.to_string())],
    }
}

/// Decodes one call as a control call, or [`None`] when it is a plain
/// member invocation. The reserved service ID without a control shape falls
/// through too: upstream returns `undefined`.
#[must_use]
pub fn decode_service_control_call(call: &ServiceCall) -> Option<ServiceControlCall> {
    if call.service_id != SERVICE_CONTROL_ID || call.instance.is_some() {
        return None;
    }
    match call.member.as_str() {
        "catalogue" if call.args.is_empty() => Some(ServiceControlCall::Catalogue),
        "subscribe" if call.args.len() == 3
            && is_id(&call.args[0])
            && is_id(&call.args[1])
            && ServiceMode::parse(call.args[2].as_str()?).is_some() =>
        {
            Some(ServiceControlCall::Subscribe {
                subscription_id: call.args[0].as_str()?.to_string(),
                service_id: call.args[1].as_str()?.to_string(),
                mode: ServiceMode::parse(call.args[2].as_str()?).expect("mode parsed above"),
            })
        }
        "unsubscribe" if call.args.len() == 1 && is_id(&call.args[0]) => {
            Some(ServiceControlCall::Unsubscribe {
                subscription_id: call.args[0].as_str()?.to_string(),
            })
        }
        _ => None,
    }
}

/// Validates one service call decoded from the wire.
///
/// # Errors
/// [`ChordError::Message`] `"Invalid service call"` on any shape violation.
pub fn parse_service_call(value: &JsonValue) -> Result<ServiceCall, ChordError> {
    let call = record(value, "service call")?;
    assert_keys(call, &["serviceId", "member", "args"], &["instance"], "service call")?;
    let service_id = string_field(call, "serviceId").ok_or_else(|| invalid("service call"))?;
    let member = string_field(call, "member").ok_or_else(|| invalid("service call"))?;
    let JsonValue::Array(args) = call.get("args").unwrap_or(&JsonValue::Null) else {
        return Err(invalid("service call"));
    };
    let instance = match call.get("instance") {
        Some(address) => Some(parse_address(address)?),
        None => None,
    };
    Ok(ServiceCall {
        service_id: service_id.to_string(),
        instance,
        member: member.to_string(),
        args: args.clone(),
    })
}

/// Validates a service catalogue decoded from the wire: an array of
/// `{ serviceId, mode }` entries with unique IDs.
///
/// # Errors
/// [`ChordError::Message`] `"Invalid service catalogue"` on any shape
/// violation or duplicate ID.
pub fn parse_service_catalogue(value: &JsonValue) -> Result<Vec<ServiceCatalogueEntry>, ChordError> {
    let JsonValue::Array(items) = value else {
        return Err(invalid("service catalogue"));
    };
    let mut ids = Vec::new();
    let mut entries = Vec::with_capacity(items.len());
    for item in items {
        let entry = record(item, "service catalogue entry")?;
        assert_keys(entry, &["serviceId", "mode"], &[], "service catalogue entry")?;
        let service_id = string_field(entry, "serviceId").ok_or_else(|| invalid("service catalogue"))?;
        let mode = mode_field(entry).ok_or_else(|| invalid("service catalogue"))?;
        if ids.contains(&service_id.to_string()) {
            return Err(invalid("service catalogue"));
        }
        ids.push(service_id.to_string());
        entries.push(ServiceCatalogueEntry {
            service_id: service_id.to_string(),
            mode,
        });
    }
    Ok(entries)
}

/// Validates a decoded subscription snapshot, operations validated as ops.
///
/// # Errors
/// [`ChordError::Message`] `"Invalid service subscription snapshot"` on any
/// shape violation.
pub fn parse_service_subscription_snapshot(value: &JsonValue) -> Result<ServiceSubscriptionSnapshot, ChordError> {
    let snapshot = parse_snapshot(value, decode_op)?;
    let instances = snapshot
        .instances
        .into_iter()
        .map(|instance| ServiceInstanceSnapshot {
            instance: instance.instance,
            members: instance
                .members
                .into_iter()
                .map(|member| match member {
                    ParsedMember::Method { name } => ServiceMemberSnapshot::Method { name },
                    ParsedMember::State {
                        name,
                        sequence,
                        ops,
                    } => ServiceMemberSnapshot::State {
                        name,
                        sequence,
                        ops,
                    },
                })
                .collect(),
        })
        .collect();
    Ok(ServiceSubscriptionSnapshot {
        service_id: snapshot.service_id,
        mode: snapshot.mode,
        instances,
    })
}

/// Validates a wire-form subscription snapshot, operations validated as
/// wire ops.
///
/// # Errors
/// [`ChordError::Message`] `"Invalid service subscription snapshot"` on any
/// shape violation.
pub fn parse_wire_service_subscription_snapshot(
    value: &JsonValue,
) -> Result<WireServiceSubscriptionSnapshot, ChordError> {
    let snapshot = parse_snapshot(value, decode_wire_op)?;
    let instances = snapshot
        .instances
        .into_iter()
        .map(|instance| WireServiceInstanceSnapshot {
            instance: instance.instance,
            members: instance
                .members
                .into_iter()
                .map(|member| match member {
                    ParsedMember::Method { name } => WireServiceMemberSnapshot::Method { name },
                    ParsedMember::State {
                        name,
                        sequence,
                        ops,
                    } => WireServiceMemberSnapshot::State {
                        name,
                        sequence,
                        ops,
                    },
                })
                .collect(),
        })
        .collect();
    Ok(WireServiceSubscriptionSnapshot {
        service_id: snapshot.service_id,
        mode: snapshot.mode,
        instances,
    })
}

/// Validates a decoded provider update.
///
/// # Errors
/// [`ChordError`] on any shape violation, with the message the update kind
/// names.
pub fn parse_service_provider_update(value: &JsonValue) -> Result<ServiceProviderUpdate, ChordError> {
    match parse_update(value, decode_op)? {
        ParsedUpdate::State {
            instance,
            member,
            sequence,
            ops,
        } => Ok(ServiceProviderUpdate::State {
            instance,
            member,
            sequence,
            ops,
        }),
        ParsedUpdate::Unavailable => Ok(ServiceProviderUpdate::Unavailable),
        ParsedUpdate::Replaced { snapshot } => Ok(ServiceProviderUpdate::Replaced {
            snapshot: decoded_instance(snapshot),
        }),
        ParsedUpdate::Spawned { instance } => Ok(ServiceProviderUpdate::Spawned {
            instance: decoded_instance(instance),
        }),
        ParsedUpdate::Closed { instance } => Ok(ServiceProviderUpdate::Closed { instance }),
    }
}

/// Validates a wire-form provider update.
///
/// # Errors
/// [`ChordError`] on any shape violation, including malformed wire ops.
pub fn parse_wire_service_provider_update(value: &JsonValue) -> Result<WireServiceProviderUpdate, ChordError> {
    match parse_update(value, decode_wire_op)? {
        ParsedUpdate::State {
            instance,
            member,
            sequence,
            ops,
        } => Ok(WireServiceProviderUpdate::State {
            instance,
            member,
            sequence,
            ops,
        }),
        ParsedUpdate::Unavailable => Ok(WireServiceProviderUpdate::Unavailable),
        ParsedUpdate::Replaced { snapshot } => Ok(WireServiceProviderUpdate::Replaced {
            snapshot: wire_instance(snapshot),
        }),
        ParsedUpdate::Spawned { instance } => Ok(WireServiceProviderUpdate::Spawned {
            instance: wire_instance(instance),
        }),
        ParsedUpdate::Closed { instance } => Ok(WireServiceProviderUpdate::Closed { instance }),
    }
}

struct ParsedSnapshot<OP> {
    service_id: String,
    mode: ServiceMode,
    instances: Vec<ParsedInstance<OP>>,
}

struct ParsedInstance<OP> {
    instance: Option<ServiceInstanceAddress>,
    members: Vec<ParsedMember<OP>>,
}

enum ParsedMember<OP> {
    Method {
        name: String,
    },
    State {
        name: String,
        sequence: u64,
        ops: Vec<OP>,
    },
}

enum ParsedUpdate<OP> {
    State {
        instance: Option<ServiceInstanceAddress>,
        member: String,
        sequence: u64,
        ops: Vec<OP>,
    },
    Unavailable,
    Replaced {
        snapshot: ParsedInstance<OP>,
    },
    Spawned {
        instance: ParsedInstance<OP>,
    },
    Closed {
        instance: ServiceInstanceAddress,
    },
}

fn parse_snapshot<OP>(
    value: &JsonValue,
    decode_op: impl Fn(&JsonValue) -> Result<OP, ChordError>,
) -> Result<ParsedSnapshot<OP>, ChordError> {
    let snapshot = record(value, "service subscription snapshot")?;
    assert_keys(snapshot, &["serviceId", "mode", "instances"], &[], "service subscription snapshot")?;
    let service_id =
        string_field(snapshot, "serviceId").ok_or_else(|| invalid("service subscription snapshot"))?;
    let mode = mode_field(snapshot).ok_or_else(|| invalid("service subscription snapshot"))?;
    let JsonValue::Array(instances) = snapshot.get("instances").unwrap_or(&JsonValue::Null) else {
        return Err(invalid("service subscription snapshot"));
    };
    let mut parsed = Vec::with_capacity(instances.len());
    for instance in instances {
        parsed.push(parse_instance(instance, &decode_op)?);
    }
    Ok(ParsedSnapshot {
        service_id: service_id.to_string(),
        mode,
        instances: parsed,
    })
}

fn parse_instance<OP>(
    value: &JsonValue,
    decode_op: &impl Fn(&JsonValue) -> Result<OP, ChordError>,
) -> Result<ParsedInstance<OP>, ChordError> {
    let instance = record(value, "service instance snapshot")?;
    assert_keys(instance, &["members"], &["instance"], "service instance snapshot")?;
    let address = match instance.get("instance") {
        Some(address) => Some(parse_address(address)?),
        None => None,
    };
    let JsonValue::Array(members) = instance.get("members").unwrap_or(&JsonValue::Null) else {
        return Err(invalid("service instance snapshot"));
    };
    let mut parsed = Vec::with_capacity(members.len());
    for candidate in members {
        let member = record(candidate, "service member snapshot")?;
        let kind = member.get("kind").and_then(JsonValue::as_str).unwrap_or("");
        match kind {
            "method" => {
                assert_keys(member, &["name", "kind"], &[], "service method snapshot")?;
                let name = string_field(member, "name").ok_or_else(|| invalid("service method snapshot"))?;
                parsed.push(ParsedMember::Method { name: name.to_string() });
            }
            "state" => {
                assert_keys(member, &["name", "kind", "sequence", "ops"], &[], "service state snapshot")?;
                let name = string_field(member, "name").ok_or_else(|| invalid("service state snapshot"))?;
                let sequence =
                    integer_field(member, "sequence", 0).ok_or_else(|| invalid("service state snapshot"))?;
                let JsonValue::Array(ops) = member.get("ops").unwrap_or(&JsonValue::Null) else {
                    return Err(invalid("service state snapshot"));
                };
                let mut decoded = Vec::with_capacity(ops.len());
                for op in ops {
                    decoded.push(decode_op(op)?);
                }
                parsed.push(ParsedMember::State {
                    name: name.to_string(),
                    sequence,
                    ops: decoded,
                });
            }
            _ => return Err(invalid("service member snapshot")),
        }
    }
    Ok(ParsedInstance {
        instance: address,
        members: parsed,
    })
}

fn parse_update<OP>(
    value: &JsonValue,
    decode_op: impl Fn(&JsonValue) -> Result<OP, ChordError>,
) -> Result<ParsedUpdate<OP>, ChordError> {
    let update = record(value, "service provider update")?;
    match update.get("type").and_then(JsonValue::as_str) {
        Some("state") => {
            assert_keys(update, &["type", "member", "sequence", "ops"], &["instance"], "state update")?;
            let member = string_field(update, "member").ok_or_else(|| invalid("service state update"))?;
            let sequence =
                integer_field(update, "sequence", 1).ok_or_else(|| invalid("service state update"))?;
            let JsonValue::Array(ops) = update.get("ops").unwrap_or(&JsonValue::Null) else {
                return Err(invalid("service state update"));
            };
            let mut decoded = Vec::with_capacity(ops.len());
            for op in ops {
                decoded.push(decode_op(op)?);
            }
            let instance = match update.get("instance") {
                Some(address) => Some(parse_address(address)?),
                None => None,
            };
            Ok(ParsedUpdate::State {
                instance,
                member: member.to_string(),
                sequence,
                ops: decoded,
            })
        }
        Some("unavailable") => {
            assert_keys(update, &["type"], &[], "unavailable update")?;
            Ok(ParsedUpdate::Unavailable)
        }
        Some("replaced") => {
            assert_keys(update, &["type", "snapshot"], &[], "replacement update")?;
            let snapshot = parse_instance(update.get("snapshot").unwrap_or(&JsonValue::Null), &decode_op)?;
            Ok(ParsedUpdate::Replaced { snapshot })
        }
        Some("spawned") => {
            assert_keys(update, &["type", "instance"], &[], "spawn update")?;
            let instance = parse_instance(update.get("instance").unwrap_or(&JsonValue::Null), &decode_op)?;
            Ok(ParsedUpdate::Spawned { instance })
        }
        Some("closed") => {
            assert_keys(update, &["type", "instance"], &[], "close update")?;
            let address = parse_address(update.get("instance").unwrap_or(&JsonValue::Null))?;
            Ok(ParsedUpdate::Closed { instance: address })
        }
        _ => Err(invalid("service provider update")),
    }
}

fn decoded_instance(instance: ParsedInstance<Op>) -> ServiceInstanceSnapshot {
    ServiceInstanceSnapshot {
        instance: instance.instance,
        members: instance
            .members
            .into_iter()
            .map(|member| match member {
                ParsedMember::Method { name } => ServiceMemberSnapshot::Method { name },
                ParsedMember::State {
                    name,
                    sequence,
                    ops,
                } => ServiceMemberSnapshot::State {
                    name,
                    sequence,
                    ops,
                },
            })
            .collect(),
    }
}

fn wire_instance(instance: ParsedInstance<WireOp>) -> WireServiceInstanceSnapshot {
    WireServiceInstanceSnapshot {
        instance: instance.instance,
        members: instance
            .members
            .into_iter()
            .map(|member| match member {
                ParsedMember::Method { name } => WireServiceMemberSnapshot::Method { name },
                ParsedMember::State {
                    name,
                    sequence,
                    ops,
                } => WireServiceMemberSnapshot::State {
                    name,
                    sequence,
                    ops,
                },
            })
            .collect(),
    }
}

fn parse_address(value: &JsonValue) -> Result<ServiceInstanceAddress, ChordError> {
    let address = record(value, "service instance address")?;
    assert_keys(address, &["key", "generation"], &[], "service instance address")?;
    let key = string_field(address, "key").ok_or_else(|| invalid("service instance address"))?;
    let generation =
        integer_field(address, "generation", 1).ok_or_else(|| invalid("service instance address"))?;
    Ok(ServiceInstanceAddress {
        key: key.to_string(),
        generation,
    })
}

fn record<'a>(value: &'a JsonValue, description: &str) -> Result<&'a JsonObject, ChordError> {
    value.as_object().ok_or_else(|| invalid(description))
}

fn assert_keys(
    value: &JsonObject,
    required: &[&str],
    optional: &[&str],
    description: &str,
) -> Result<(), ChordError> {
    for key in required {
        if !value.contains_key(key) {
            return Err(invalid(description));
        }
    }
    for key in value.keys() {
        if !required.contains(&key) && !optional.contains(&key) {
            return Err(invalid(description));
        }
    }
    Ok(())
}

fn string_field<'a>(value: &'a JsonObject, key: &str) -> Option<&'a str> {
    value.get(key).and_then(JsonValue::as_str)
}

fn mode_field(value: &JsonObject) -> Option<ServiceMode> {
    ServiceMode::parse(value.get("mode")?.as_str()?)
}

fn integer_field(value: &JsonObject, key: &str, minimum: u64) -> Option<u64> {
    let number = value.get(key)?.as_number()?;
    if number.fract() != 0.0 || number < minimum as f64 {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the integer check above rules out fractions and negatives, so the cast is exact"
    )]
    Some(number as u64)
}

fn is_id(value: &JsonValue) -> bool {
    value.as_str().is_some_and(|text| !text.is_empty())
}

fn decode_op(value: &JsonValue) -> Result<Op, ChordError> {
    Op::from_json(value).map_err(|error| ChordError::Message(error.to_string()))
}

fn decode_wire_op(value: &JsonValue) -> Result<WireOp, ChordError> {
    WireOp::from_json(value).map_err(|error| ChordError::Message(error.to_string()))
}

fn invalid(description: &str) -> ChordError {
    ChordError::Message(format!("Invalid {description}"))
}

/// Renders a service catalogue as the JSON the endpoint returns for the
/// catalogue control call, upstream's `provider.catalogue` handed to the
/// wire as-is.
#[must_use]
pub fn catalogue_to_json(entries: &[ServiceCatalogueEntry]) -> JsonValue {
    JsonValue::Array(
        entries
            .iter()
            .map(|entry| {
                object(vec![
                    ("serviceId", JsonValue::Str(entry.service_id.clone())),
                    ("mode", JsonValue::Str(entry.mode.as_str().to_string())),
                ])
            })
            .collect(),
    )
}

/// Renders a decoded subscription snapshot as its wire form.
#[must_use]
pub fn snapshot_to_json(snapshot: &ServiceSubscriptionSnapshot) -> JsonValue {
    object(vec![
        ("serviceId", JsonValue::Str(snapshot.service_id.clone())),
        ("mode", JsonValue::Str(snapshot.mode.as_str().to_string())),
        (
            "instances",
            JsonValue::Array(snapshot.instances.iter().map(instance_to_json).collect()),
        ),
    ])
}

fn instance_to_json(instance: &ServiceInstanceSnapshot) -> JsonValue {
    let mut entries = Vec::new();
    if let Some(address) = &instance.instance {
        entries.push(("instance", address_to_json(address)));
    }
    entries.push((
        "members",
        JsonValue::Array(
            instance
                .members
                .iter()
                .map(|member| match member {
                    ServiceMemberSnapshot::Method { name } => object(vec![
                        ("name", JsonValue::Str(name.clone())),
                        ("kind", JsonValue::Str("method".to_string())),
                    ]),
                    ServiceMemberSnapshot::State {
                        name,
                        sequence,
                        ops,
                    } => object(vec![
                        ("name", JsonValue::Str(name.clone())),
                        ("kind", JsonValue::Str("state".to_string())),
                        ("sequence", JsonValue::Number(JsonNumber::from(*sequence))),
                        ("ops", JsonValue::Array(ops.iter().map(Op::to_json).collect())),
                    ]),
                })
                .collect(),
        ),
    ));
    object(entries)
}

fn address_to_json(address: &ServiceInstanceAddress) -> JsonValue {
    object(vec![
        ("key", JsonValue::Str(address.key.clone())),
        ("generation", JsonValue::Number(JsonNumber::from(address.generation))),
    ])
}

/// Builds a JSON object from `(key, value)` pairs in order.
#[must_use]
pub fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(crate::types::JsonObject::from_entries(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    ))
}