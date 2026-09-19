//! The in-memory recording adapter, ported from upstream `memory.ts`.
//!
//! One mutex guards the whole state, matching upstream's per-call atomicity:
//! every recording step either fully applies or, for `None`-valued entries,
//! is skipped entirely — there is no partial merge. A poisoned mutex is
//! recovered from rather than propagated, which is the Rust-native
//! statement of upstream's "recording is passive" contract: recording never
//! panics, and hostile payloads cannot exist for owned plain data.

use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan};

/// A recorded event as exposed by snapshot readers.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedTelemetryEvent {
    /// The event name.
    pub name: String,
    /// A detached copy of the event attributes.
    pub attributes: SpanAttributes,
}

/// A detached snapshot of one recorded span.
///
/// Snapshots are deep copies in span-start order; mutating one never
/// affects the adapter's recording state.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedTelemetrySpan {
    /// Deterministic id assigned in registration order, starting at 1.
    pub id: u64,
    /// The parent span id, or `None` for roots.
    pub parent_id: Option<u64>,
    /// The span name.
    pub name: String,
    /// A detached copy of the merged attributes.
    pub attributes: SpanAttributes,
    /// A detached copy of the events in record order.
    pub events: Vec<RecordedTelemetryEvent>,
    /// The last status recorded.
    pub status: SpanStatus,
    /// Whether the span has settled.
    pub settled: bool,
    /// The settlement order sequence; present only once settled.
    pub end_sequence: Option<u64>,
}

/// The mutable recording state of one span.
#[derive(Debug)]
struct SpanRecord {
    id: usize,
    parent_id: Option<usize>,
    name: String,
    attributes: SpanAttributes,
    events: Vec<RecordedTelemetryEvent>,
    status: SpanStatus,
    explicit_status: bool,
    settled: bool,
    end_sequence: Option<u64>,
}

#[derive(Debug)]
struct InMemoryState {
    spans: Vec<SpanRecord>,
    next_span_id: usize,
    next_end_sequence: u64,
}

impl Default for InMemoryState {
    fn default() -> Self {
        Self {
            spans: Vec::new(),
            next_span_id: 1,
            next_end_sequence: 1,
        }
    }
}

/// Locks the state, recovering from poisoning so recording stays passive.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Skips `None` entries, the port of upstream's undefined-skipping copy.
fn copy_attributes(attributes: Option<&SpanAttributes>) -> SpanAttributes {
    let mut copy = SpanAttributes::new();
    if let Some(attributes) = attributes {
        for (name, value) in attributes {
            if let Some(value) = value {
                copy.insert(name.clone(), Some(value.clone()));
            }
        }
    }
    copy
}

/// Merges later attributes over current ones; `None` entries never clobber.
fn merge_attributes(current: &SpanAttributes, attributes: &SpanAttributes) -> SpanAttributes {
    let mut merged = copy_attributes(Some(current));
    for (name, value) in attributes {
        if let Some(value) = value {
            merged.insert(name.clone(), Some(value.clone()));
        }
    }
    merged
}

/// Copies a status, detaching the error details.
fn copy_status(status: &SpanStatus) -> SpanStatus {
    match status {
        SpanStatus::Ok => SpanStatus::Ok,
        SpanStatus::Error { error } => SpanStatus::Error {
            error: error.clone(),
        },
    }
}

/// Marks a span settled, assigning the automatic error status when the body
/// failed without an explicit status.
fn settle(state: &Mutex<InMemoryState>, id: usize, failed: bool) {
    let mut state = lock(state);
    if state.spans[span_index(id)].settled {
        return;
    }
    let sequence = state.next_end_sequence;
    state.next_end_sequence += 1;
    let record = &mut state.spans[span_index(id)];
    if failed && !record.explicit_status {
        record.status = SpanStatus::Error { error: None };
    }
    record.settled = true;
    record.end_sequence = Some(sequence);
    drop(state);
}

/// Registers a span and returns its id, in registration order.
fn register_span(
    state: &Mutex<InMemoryState>,
    parent_id: Option<usize>,
    options: &SpanOptions,
) -> usize {
    let mut state = lock(state);
    let id = state.next_span_id;
    state.next_span_id += 1;
    state.spans.push(SpanRecord {
        id,
        parent_id,
        name: options.name.clone(),
        attributes: copy_attributes(options.attributes.as_ref()),
        events: Vec::new(),
        status: SpanStatus::Ok,
        explicit_status: false,
        settled: false,
        end_sequence: None,
    });
    id
}

/// Runs a body against a freshly registered span and settles it afterwards.
async fn run_body<T, E, Fut, F>(
    state: Arc<Mutex<InMemoryState>>,
    id: usize,
    body: F,
) -> Result<T, E>
where
    F: FnOnce(InMemorySpan) -> Fut + Send,
    Fut: Future<Output = Result<T, E>> + Send,
{
    let span = InMemorySpan {
        state: Arc::clone(&state),
        id,
    };
    let result = body(span).await;
    settle(&state, id, result.is_err());
    result
}

/// Backend-neutral reference implementation that records spans in process
/// memory. Create a fresh instance to isolate tests or independent recording
/// scopes.
#[derive(Clone, Debug)]
pub struct InMemoryTelemetryContext {
    state: Arc<Mutex<InMemoryState>>,
}

impl Default for InMemoryTelemetryContext {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(InMemoryState::default())),
        }
    }
}

impl InMemoryTelemetryContext {
    /// Creates an isolated recording scope with fresh ids and sequences.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns detached snapshots in span-start order.
    #[must_use]
    pub fn get_spans(&self) -> Vec<RecordedTelemetrySpan> {
        let state = lock(&self.state);
        state.spans.iter().map(SpanRecord::snapshot).collect()
    }
}

impl TelemetryContext for InMemoryTelemetryContext {
    type Span = InMemorySpan;

    fn start_span<T, E, Fut, F>(
        &self,
        options: SpanOptions,
        body: F,
    ) -> impl Future<Output = Result<T, E>> + Send
    where
        F: FnOnce(Self::Span) -> Fut + Send,
        Fut: Future<Output = Result<T, E>> + Send,
    {
        let state = Arc::clone(&self.state);
        let id = register_span(&self.state, None, &options);
        run_body(state, id, body)
    }
}

/// The span handle handed to bodies by [`InMemoryTelemetryContext`].
///
/// The handle shares the adapter's state, so recordings stay visible after
/// the body returns, and stays usable after settlement, where calls are
/// inert.
#[derive(Clone, Debug)]
pub struct InMemorySpan {
    state: Arc<Mutex<InMemoryState>>,
    id: usize,
}

impl TelemetrySpan for InMemorySpan {
    fn add_event(&self, name: &str, attributes: SpanAttributes) {
        let mut state = lock(&self.state);
        let record = &mut state.spans[span_index(self.id)];
        if record.settled {
            return;
        }
        record.events.push(RecordedTelemetryEvent {
            name: name.to_owned(),
            attributes: copy_attributes(Some(&attributes)),
        });
        drop(state);
    }

    fn set_attributes(&self, attributes: SpanAttributes) {
        let mut state = lock(&self.state);
        let record = &mut state.spans[span_index(self.id)];
        if record.settled {
            return;
        }
        let merged = merge_attributes(&record.attributes, &attributes);
        record.attributes = merged;
        drop(state);
    }

    fn set_status(&self, status: SpanStatus) {
        let mut state = lock(&self.state);
        let record = &mut state.spans[span_index(self.id)];
        if record.settled {
            return;
        }
        record.status = copy_status(&status);
        record.explicit_status = true;
        drop(state);
    }
}

impl TelemetryContext for InMemorySpan {
    type Span = Self;

    fn start_span<T, E, Fut, F>(
        &self,
        options: SpanOptions,
        body: F,
    ) -> impl Future<Output = Result<T, E>> + Send
    where
        F: FnOnce(Self) -> Fut + Send,
        Fut: Future<Output = Result<T, E>> + Send,
    {
        let parent_settled = lock(&self.state).spans[span_index(self.id)].settled;
        // Children of settled spans record nowhere reachable, matching
        // upstream's delegation to the noop context: the body runs against a
        // throwaway state nothing can observe.
        let (state, parent_id) = if parent_settled {
            (Arc::new(Mutex::new(InMemoryState::default())), None)
        } else {
            (Arc::clone(&self.state), Some(self.id))
        };
        let id = register_span(&state, parent_id, &options);
        run_body(state, id, body)
    }
}

const fn span_id(id: usize) -> u64 {
    id as u64
}

/// Locates a span record by its registration id; ids start at 1 and are
/// assigned sequentially, so the record index is always `id - 1`.
const fn span_index(id: usize) -> usize {
    id - 1
}

impl SpanRecord {
    fn snapshot(&self) -> RecordedTelemetrySpan {
        RecordedTelemetrySpan {
            id: span_id(self.id),
            parent_id: self.parent_id.map(span_id),
            name: self.name.clone(),
            attributes: copy_attributes(Some(&self.attributes)),
            events: self
                .events
                .iter()
                .map(|event| RecordedTelemetryEvent {
                    name: event.name.clone(),
                    attributes: copy_attributes(Some(&event.attributes)),
                })
                .collect(),
            status: copy_status(&self.status),
            settled: self.settled,
            end_sequence: self.end_sequence,
        }
    }
}
