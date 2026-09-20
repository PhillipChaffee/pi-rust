//! Stateful operation codecs for one service subscription, ported from
//! upstream `src/services/state-codec.ts`.
//!
//! Every replicated state in one subscription rides its own encoder/decoder
//! pair, because the wire interning dictionaries are per-state. A
//! replacement or unavailable update resets the dictionaries; a close
//! removes that instance's codecs, so a late state update on a closed
//! instance reports unknown.

use std::cell::RefCell;
use std::rc::Rc;

use crate::delta::{Decoder, Encoder};
use crate::errors::ChordError;
use crate::services::wire::{
    WireServiceInstanceSnapshot, WireServiceMemberSnapshot, WireServiceProviderUpdate,
    WireServiceSubscriptionSnapshot,
};
use crate::types::{
    ServiceInstanceAddress, ServiceInstanceSnapshot, ServiceMemberSnapshot, ServiceProviderUpdate,
    ServiceSubscriptionSnapshot,
};

/// Stateful operation encoders for every replicated state in one service
/// subscription, upstream's `ServiceStateEncoder`.
pub struct ServiceStateEncoder {
    codecs: CodecRegistry<Encoder>,
}

impl std::fmt::Debug for ServiceStateEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceStateEncoder")
            .finish_non_exhaustive()
    }
}

impl ServiceStateEncoder {
    /// Encodes a snapshot, resetting the dictionaries first.
    ///
    /// # Errors
    /// [`ChordError`] when the snapshot names a state twice.
    pub fn encode_snapshot(
        &mut self,
        snapshot: &ServiceSubscriptionSnapshot,
    ) -> Result<WireServiceSubscriptionSnapshot, ChordError> {
        self.codecs.reset();
        let mut instances = Vec::with_capacity(snapshot.instances.len());
        for instance in &snapshot.instances {
            instances.push(self.encode_instance(instance)?);
        }
        Ok(WireServiceSubscriptionSnapshot {
            service_id: snapshot.service_id.clone(),
            mode: snapshot.mode,
            instances,
        })
    }

    /// Encodes one update against its subscription state.
    ///
    /// # Errors
    /// [`ChordError`] when the update names a state this subscription never
    /// saw.
    pub fn encode_update(
        &mut self,
        update: &ServiceProviderUpdate,
    ) -> Result<WireServiceProviderUpdate, ChordError> {
        match update {
            ServiceProviderUpdate::State {
                instance,
                member,
                sequence,
                ops,
            } => {
                let codec = self.codecs.get(instance.as_ref(), member)?;
                let ops = codec.borrow_mut().encode(ops);
                Ok(WireServiceProviderUpdate::State {
                    instance: instance.clone(),
                    member: member.clone(),
                    sequence: *sequence,
                    ops,
                })
            }
            ServiceProviderUpdate::Unavailable => {
                self.codecs.reset();
                Ok(WireServiceProviderUpdate::Unavailable)
            }
            ServiceProviderUpdate::Replaced { snapshot } => {
                self.codecs.reset();
                Ok(WireServiceProviderUpdate::Replaced {
                    snapshot: self.encode_instance(snapshot)?,
                })
            }
            ServiceProviderUpdate::Spawned { instance } => Ok(WireServiceProviderUpdate::Spawned {
                instance: self.encode_instance(instance)?,
            }),
            ServiceProviderUpdate::Closed { instance } => {
                self.codecs.remove_instance(instance);
                Ok(WireServiceProviderUpdate::Closed {
                    instance: instance.clone(),
                })
            }
        }
    }

    fn encode_instance(
        &mut self,
        instance: &ServiceInstanceSnapshot,
    ) -> Result<WireServiceInstanceSnapshot, ChordError> {
        let mut members = Vec::with_capacity(instance.members.len());
        for member in &instance.members {
            match member {
                ServiceMemberSnapshot::Method { name } => {
                    members.push(WireServiceMemberSnapshot::Method { name: name.clone() });
                }
                ServiceMemberSnapshot::State {
                    name,
                    sequence,
                    ops,
                } => {
                    let codec = self.codecs.add(instance.instance.as_ref(), name)?;
                    let ops = codec.borrow_mut().encode(ops);
                    members.push(WireServiceMemberSnapshot::State {
                        name: name.clone(),
                        sequence: *sequence,
                        ops,
                    });
                }
            }
        }
        Ok(WireServiceInstanceSnapshot {
            instance: instance.instance.clone(),
            members,
        })
    }
}

/// Stateful operation decoders for every replicated state in one service
/// subscription, upstream's `ServiceStateDecoder`.
pub struct ServiceStateDecoder {
    codecs: CodecRegistry<Decoder>,
}

impl std::fmt::Debug for ServiceStateDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceStateDecoder")
            .finish_non_exhaustive()
    }
}

impl ServiceStateDecoder {
    /// Decodes a snapshot, resetting the dictionaries first.
    ///
    /// # Errors
    /// [`ChordError`] when the snapshot names a state twice or an op fails
    /// to decode.
    pub fn decode_snapshot(
        &mut self,
        snapshot: &WireServiceSubscriptionSnapshot,
    ) -> Result<ServiceSubscriptionSnapshot, ChordError> {
        self.codecs.reset();
        let mut instances = Vec::with_capacity(snapshot.instances.len());
        for instance in &snapshot.instances {
            let mut members = Vec::with_capacity(instance.members.len());
            for member in &instance.members {
                match member {
                    WireServiceMemberSnapshot::Method { name } => {
                        members.push(ServiceMemberSnapshot::Method { name: name.clone() });
                    }
                    WireServiceMemberSnapshot::State {
                        name,
                        sequence,
                        ops,
                    } => {
                        let codec = self.codecs.add(instance.instance.as_ref(), name)?;
                        let ops = codec.borrow_mut().decode(ops)?;
                        members.push(ServiceMemberSnapshot::State {
                            name: name.clone(),
                            sequence: *sequence,
                            ops,
                        });
                    }
                }
            }
            instances.push(ServiceInstanceSnapshot {
                instance: instance.instance.clone(),
                members,
            });
        }
        Ok(ServiceSubscriptionSnapshot {
            service_id: snapshot.service_id.clone(),
            mode: snapshot.mode,
            instances,
        })
    }

    /// Decodes one update against this subscription's state.
    ///
    /// # Errors
    /// [`ChordError`] when the update names a state this subscription never
    /// saw or an op fails to decode.
    pub fn decode_update(
        &mut self,
        update: &WireServiceProviderUpdate,
    ) -> Result<ServiceProviderUpdate, ChordError> {
        match update {
            WireServiceProviderUpdate::State {
                instance,
                member,
                sequence,
                ops,
            } => {
                let codec = self.codecs.get(instance.as_ref(), member)?;
                let ops = codec.borrow_mut().decode(ops)?;
                Ok(ServiceProviderUpdate::State {
                    instance: instance.clone(),
                    member: member.clone(),
                    sequence: *sequence,
                    ops,
                })
            }
            WireServiceProviderUpdate::Unavailable => {
                self.codecs.reset();
                Ok(ServiceProviderUpdate::Unavailable)
            }
            WireServiceProviderUpdate::Replaced { snapshot } => {
                self.codecs.reset();
                Ok(ServiceProviderUpdate::Replaced {
                    snapshot: self.decode_instance(snapshot)?,
                })
            }
            WireServiceProviderUpdate::Spawned { instance } => Ok(ServiceProviderUpdate::Spawned {
                instance: self.decode_instance(instance)?,
            }),
            WireServiceProviderUpdate::Closed { instance } => {
                self.codecs.remove_instance(instance);
                Ok(ServiceProviderUpdate::Closed {
                    instance: instance.clone(),
                })
            }
        }
    }

    fn decode_instance(
        &mut self,
        instance: &WireServiceInstanceSnapshot,
    ) -> Result<ServiceInstanceSnapshot, ChordError> {
        let mut members = Vec::with_capacity(instance.members.len());
        for member in &instance.members {
            match member {
                WireServiceMemberSnapshot::Method { name } => {
                    members.push(ServiceMemberSnapshot::Method { name: name.clone() });
                }
                WireServiceMemberSnapshot::State {
                    name,
                    sequence,
                    ops,
                } => {
                    let codec = self.codecs.add(instance.instance.as_ref(), name)?;
                    let ops = codec.borrow_mut().decode(ops)?;
                    members.push(ServiceMemberSnapshot::State {
                        name: name.clone(),
                        sequence: *sequence,
                        ops,
                    });
                }
            }
        }
        Ok(ServiceInstanceSnapshot {
            instance: instance.instance.clone(),
            members,
        })
    }
}

/// The encoder half of the subscription codec pair.
#[must_use]
pub fn create_service_state_encoder() -> ServiceStateEncoder {
    ServiceStateEncoder {
        codecs: CodecRegistry::new(encoder_factory),
    }
}

/// The decoder side of the subscription codec pair.
#[must_use]
pub fn create_service_state_decoder() -> ServiceStateDecoder {
    ServiceStateDecoder {
        codecs: CodecRegistry::new(decoder_factory),
    }
}

fn encoder_factory() -> Encoder {
    crate::delta::encoder()
}

fn decoder_factory() -> Decoder {
    crate::delta::decoder()
}

type SharedCodec<C> = Rc<RefCell<C>>;

struct CodecRegistry<C> {
    create: Box<dyn Fn() -> C>,
    entries: Vec<(StateKey, SharedCodec<C>)>,
}

/// The identity one codec entry lives under: instance address plus member
/// name, upstream's `stateKey` JSON tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StateKey {
    key: Option<String>,
    generation: Option<u64>,
    member: String,
}

impl<C> CodecRegistry<C> {
    fn new(create: impl Fn() -> C + 'static) -> Self {
        Self {
            create: Box::new(create),
            entries: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.entries.clear();
    }

    fn add(
        &mut self,
        instance: Option<&ServiceInstanceAddress>,
        member: &str,
    ) -> Result<SharedCodec<C>, ChordError> {
        let key = state_key(instance, member);
        if self.entries.iter().any(|(stored, _)| *stored == key) {
            return Err(ChordError::Message(format!(
                "Duplicate service state {}",
                describe_state(instance, member)
            )));
        }
        let codec = Rc::new(RefCell::new((self.create)()));
        self.entries.push((key, codec.clone()));
        Ok(codec)
    }

    fn get(
        &self,
        instance: Option<&ServiceInstanceAddress>,
        member: &str,
    ) -> Result<SharedCodec<C>, ChordError> {
        let key = state_key(instance, member);
        self.entries
            .iter()
            .find(|(stored, _)| *stored == key)
            .map(|(_, codec)| codec.clone())
            .ok_or_else(|| {
                ChordError::Message(format!(
                    "Unknown service state {}",
                    describe_state(instance, member)
                ))
            })
    }

    fn remove_instance(&mut self, instance: &ServiceInstanceAddress) {
        self.entries.retain(|(key, _)| {
            !(key.key.as_deref() == Some(instance.key.as_str())
                && key.generation == Some(instance.generation))
        });
    }
}

fn state_key(instance: Option<&ServiceInstanceAddress>, member: &str) -> StateKey {
    StateKey {
        key: instance.map(|address| address.key.clone()),
        generation: instance.map(|address| address.generation),
        member: member.to_string(),
    }
}

fn describe_state(instance: Option<&ServiceInstanceAddress>, member: &str) -> String {
    instance.map_or_else(
        || member.to_string(),
        |address| format!("{}@{}.{}", address.key, address.generation, member),
    )
}
