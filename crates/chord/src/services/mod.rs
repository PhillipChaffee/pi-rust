//! The remote service surface, ported from upstream `src/services/`.
//!
//! [`provider`] hosts one provider's registrations, instances, and
//! subscribers, plus the `$chord.service` endpoint; [`state`] carries the
//! replicated state source and replica; [`state_codec`] the per-subscription
//! operation codecs; [`loopback`] ties a provider to a binding without
//! changing remote semantics. The facade and binding machinery lives in
//! [`crate::consumer`], and the member model in [`crate::handle`], mirroring
//! upstream's `consumer.ts` and `handle.ts` with the crate-level placement
//! the shared slot vocabulary needs.

pub mod instances;
pub mod loopback;
pub mod provider;
pub mod state;
pub mod state_codec;
pub mod wire;

pub use provider::{RemoteServiceEndpoint, RemoteServiceProvider, create_remote_service_endpoint};
pub use state::{MutableReplicatedState, ReplicatedStateReplica};
pub use state_codec::{
    ServiceStateDecoder, ServiceStateEncoder, create_service_state_decoder,
    create_service_state_encoder,
};
pub use wire::{
    ServiceControlCall, WireServiceInstanceSnapshot, WireServiceMemberSnapshot,
    WireServiceProviderUpdate, WireServiceSubscriptionSnapshot, catalogue_to_json,
    create_service_catalogue_call, create_service_subscribe_call, create_service_unsubscribe_call,
    decode_service_control_call, parse_service_call, parse_service_catalogue,
    parse_service_provider_update, parse_service_subscription_snapshot,
    parse_wire_service_provider_update, parse_wire_service_subscription_snapshot,
};
