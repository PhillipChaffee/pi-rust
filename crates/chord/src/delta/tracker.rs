//! Flush-time change tracking over owned JSON, ported from upstream
//! `src/delta/index.ts` (`track` and the dirty-walk machinery).
//!
//! Upstream records mutation intents through recursive JavaScript `Proxy`
//! traps: reading a child returns another proxy, every assignment and array
//! mutator call marks a dirty path on the pending tree. Owned Rust data has
//! no dynamic proxies, so the same intent tree is recorded by the typed
//! mutation surface: [`Tracker::set`], [`Tracker::delete`], the array
//! mutators, and [`Tracker::replace_root`] each mutate the owned value and
//! mark the same dirty node the corresponding trap would. `flush` is
//! unchanged: it walks the dirty paths, diffs the last published baseline
//! against the current value, and advances the baseline.

use crate::delta::apply::apply;
use crate::delta::diff::{diff_value, emit_set};
use crate::delta::ops::{RESERVED_SEGMENTS, Seg, UnsafePathError};
use crate::delta::{Op, path_to_json_text};
use crate::types::JsonValue;

/// One dirty intent: the value at this path changed, an array changed
/// structurally, or a child did.
#[derive(Debug, Default)]
struct DirtyNode {
    value_dirty: bool,
    array: Option<ArrayDirty>,
    children: Vec<(Seg, Self)>,
}

/// How an array node changed.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ArrayDirty {
    /// Elements appended at `start`; tail edits fold into the append payload.
    Append {
        /// The index the appended region starts at.
        start: usize,
    },
    /// A non-append structural change: flush diffs the whole array.
    Diff,
    /// The array was replaced wholesale.
    Replace,
}

impl DirtyNode {
    fn new() -> Self {
        Self::default()
    }
}

/// Options for [`Tracker::with_options`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackerOptions {
    /// How far back a flush may scan for shared string prefixes. Upstream
    /// defaults to 65,536; the tracker passes the default when unset.
    pub max_overlap_scan: Option<usize>,
}

/// Error raised by tracker mutation methods.
///
/// Upstream throws from the proxy traps before the value changes; every
/// variant here means the value was left untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackerError {
    /// A path segment would reach the prototype chain.
    UnsafePath(UnsafePathError),
    /// The write cannot be expressed on tracked state: deleting from an
    /// array, resolving through a missing container, assigning the root.
    TypeError(String),
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeError(message) => f.write_str(message),
            Self::UnsafePath(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TrackerError {}

impl From<UnsafePathError> for TrackerError {
    fn from(error: UnsafePathError) -> Self {
        Self::UnsafePath(error)
    }
}

/// Flush-time change tracking over a plain JSON value.
///
/// The tracker owns the current value. Every write goes through the typed
/// mutation methods; reads go through [`Tracker::get`] and [`Tracker::value`].
#[derive(Debug)]
pub struct Tracker {
    root: JsonValue,
    baseline: Option<JsonValue>,
    pending: DirtyNode,
    has_pending: bool,
    force_base: bool,
    scan: usize,
}

impl Tracker {
    /// Tracks `root`; the first `flush` is a base batch.
    #[must_use]
    pub fn new(root: JsonValue) -> Self {
        Self::with_options(root, TrackerOptions::default())
    }

    /// Tracks `root` with a bounded overlap scan.
    #[must_use]
    pub fn with_options(root: JsonValue, options: TrackerOptions) -> Self {
        Self {
            root,
            baseline: None,
            pending: DirtyNode::default(),
            has_pending: false,
            force_base: true,
            scan: options.max_overlap_scan.unwrap_or(65_536),
        }
    }

    /// The tracked value.
    #[must_use]
    pub const fn value(&self) -> &JsonValue {
        &self.root
    }

    /// The value at `path`, if it resolves through own properties only.
    #[must_use]
    pub fn get(&self, path: &[Seg]) -> Option<&JsonValue> {
        let mut node = &self.root;
        for segment in path {
            node = own_value(node, segment)?;
        }
        Some(node)
    }

    /// Whether a flush would produce operations.
    ///
    /// Upstream spells this the `dirty` getter.
    #[must_use]
    pub const fn dirty(&self) -> bool {
        self.force_base || self.has_pending
    }

    /// Assigns the value at `path`, the object-key and array-index assignment
    /// traps: an existing index writes in place, an index one past the end
    /// appends, a gap or a reserved segment is rejected with the value
    /// untouched.
    ///
    /// # Errors
    /// [`TrackerError`] when a segment is reserved, the parent is missing or
    /// not a container, or an array index would leave a hole.
    pub fn set(&mut self, path: &[Seg], value: JsonValue) -> Result<(), TrackerError> {
        if path.is_empty() {
            return Err(Self::type_error(
                "the tracked root cannot be assigned through set; use replace_root",
            ));
        }
        Self::guard_path(path)?;
        let (container_path, segment) = split_path(path);
        let container = resolve_container_mut(&mut self.root, container_path)?;
        if let Some(items) = container.as_array_mut() {
            let Seg::Index(index) = segment else {
                return Err(Self::type_error("tracked arrays are addressed by index"));
            };
            if *index > items.len() {
                return Err(UnsafePathError {
                    segment: segment.clone(),
                }
                .into());
            }
            let start = append_start(&self.pending, container_path);
            let appends = *index == items.len();
            if appends {
                items.push(value);
            } else {
                items[*index] = value;
            }
            self.has_pending = true;
            if appends {
                mark_array_append(&mut self.pending, container_path, *index);
            } else {
                match start {
                    None => mark_value(&mut self.pending, path),
                    Some(start) if *index < start => mark_value(&mut self.pending, path),
                    Some(_) => {}
                }
            }
            return Ok(());
        }
        let Some(object) = container.as_object_mut() else {
            return Err(Self::type_error("the tracked parent is not a container"));
        };
        let Seg::Key(key) = segment else {
            return Err(Self::type_error("tracked objects are addressed by key"));
        };
        self.has_pending = true;
        mark_value(&mut self.pending, path);
        object.set(key.clone(), value);
        Ok(())
    }

    /// Removes the value at `path` from its parent object, the delete trap.
    /// Arrays reject the delete: a hole does not survive a JSON round trip.
    /// Deleting a missing key marks and emits nothing, as upstream's
    /// `deleteProperty` does.
    ///
    /// # Errors
    /// [`TrackerError`] when the path is empty (the root cannot be deleted),
    /// a segment is reserved, or the parent is missing or an array.
    pub fn delete(&mut self, path: &[Seg]) -> Result<(), TrackerError> {
        if path.is_empty() {
            return Err(Self::type_error("the tracked root cannot be deleted"));
        }
        Self::guard_path(path)?;
        let (container_path, segment) = split_path(path);
        let container = resolve_container_mut(&mut self.root, container_path)?;
        let Seg::Key(key) = segment else {
            return Err(Self::type_error(
                "delete would create a sparse array; use splice instead",
            ));
        };
        if let Some(object) = container.as_object_mut() {
            self.has_pending = true;
            mark_value(&mut self.pending, path);
            object.remove(key);
            Ok(())
        } else {
            Err(Self::type_error(
                "delete would create a sparse array; use splice instead",
            ))
        }
    }

    /// Appends `items` to the array at `path`, the `push` trap.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn push(&mut self, path: &[Seg], items: Vec<JsonValue>) -> Result<usize, TrackerError> {
        Self::guard_path(path)?;
        let inserted = items.len();
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("push runs on an array"));
        };
        let before = array.len();
        array.extend(items);
        if inserted > 0 {
            self.has_pending = true;
            mark_array_append(&mut self.pending, path, before);
        }
        Ok(before + inserted)
    }

    /// Prepends `items` to the array at `path`, the `unshift` trap.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn unshift(&mut self, path: &[Seg], items: Vec<JsonValue>) -> Result<usize, TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("unshift runs on an array"));
        };
        let inserted = items.len();
        for (offset, item) in items.into_iter().enumerate() {
            array.insert(offset, item);
        }
        if inserted > 0 {
            self.has_pending = true;
            mark_array_diff(&mut self.pending, path);
        }
        Ok(array.len())
    }

    /// Removes and returns the last element, the `pop` trap. Popping inside
    /// the appended region folds into the pending append; popping before it
    /// degrades the node to a diff.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn pop(&mut self, path: &[Seg]) -> Result<Option<JsonValue>, TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("pop runs on an array"));
        };
        let before = array.len();
        let removed = array.pop();
        if before > 0 {
            let start = append_start(&self.pending, path);
            if start.is_none_or(|start| before - 1 < start) {
                self.has_pending = true;
                mark_array_diff(&mut self.pending, path);
            }
        }
        Ok(removed)
    }

    /// Removes and returns the first element, the `shift` trap.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn shift(&mut self, path: &[Seg]) -> Result<Option<JsonValue>, TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("shift runs on an array"));
        };
        let removed = if array.is_empty() {
            None
        } else {
            self.has_pending = true;
            mark_array_diff(&mut self.pending, path);
            Some(array.remove(0))
        };
        Ok(removed)
    }

    /// Splices the array at `path`, the `splice` trap: removes `remove`
    /// elements at `index` and inserts `items`. The index may address one
    /// past the end (append); the returned value is what was removed.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn splice(
        &mut self,
        path: &[Seg],
        index: usize,
        remove: usize,
        items: &[JsonValue],
    ) -> Result<Vec<JsonValue>, TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("splice runs on an array"));
        };
        let before = array.len();
        let index = index.min(before);
        let remove = remove.min(before - index);
        let removed: Vec<JsonValue> = array.drain(index..index + remove).collect();
        for (offset, item) in items.iter().enumerate() {
            array.insert(index + offset, item.clone());
        }
        if remove > 0 || !items.is_empty() {
            let start = append_start(&self.pending, path);
            if index == 0 && remove == before {
                self.has_pending = true;
                mark_array_replace(&mut self.pending, path);
            } else if start.is_some_and(|start| index >= start) {
                // The final append payload includes all tail edits.
            } else if index == before && remove == 0 {
                self.has_pending = true;
                mark_array_append(&mut self.pending, path, before);
            } else {
                self.has_pending = true;
                mark_array_diff(&mut self.pending, path);
            }
        }
        Ok(removed)
    }

    /// Resizes the array at `path`, the `length` assignment trap: shrinking
    /// drops the tail (a full shrink replaces, an in-append shrink degrades to
    /// a diff), growth appends explicit nulls.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn set_length(&mut self, path: &[Seg], length: usize) -> Result<(), TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("length runs on an array"));
        };
        let before = array.len();
        if length < before {
            if length == 0 {
                self.has_pending = true;
                mark_array_replace(&mut self.pending, path);
            } else {
                let start = append_start(&self.pending, path);
                if start.is_none_or(|start| length < start) {
                    self.has_pending = true;
                    mark_array_diff(&mut self.pending, path);
                }
            }
            array.truncate(length);
        } else if length > before {
            self.has_pending = true;
            mark_array_append(&mut self.pending, path, before);
            array.resize(length, JsonValue::Null);
        }
        Ok(())
    }

    /// Sorts the array at `path`, the `sort` trap: the default sort compares
    /// elements by their JSON text, upstream's string-coercion ordering.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn sort(&mut self, path: &[Seg]) -> Result<(), TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("sort runs on an array"));
        };
        array.sort_by_key(JsonValue::to_json_string);
        self.has_pending = true;
        mark_array_diff(&mut self.pending, path);
        Ok(())
    }

    /// Reverses the array at `path`, the `reverse` trap.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn reverse(&mut self, path: &[Seg]) -> Result<(), TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("reverse runs on an array"));
        };
        array.reverse();
        self.has_pending = true;
        mark_array_diff(&mut self.pending, path);
        Ok(())
    }

    /// Fills the array at `path` with `value` between `start` and the end,
    /// the `fill` trap.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn fill(&mut self, path: &[Seg], value: JsonValue) -> Result<(), TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("fill runs on an array"));
        };
        array.fill(value);
        self.has_pending = true;
        mark_array_diff(&mut self.pending, path);
        Ok(())
    }

    /// Moves `count` elements from `source` to `destination` within the array
    /// at `path`, the `copyWithin` trap.
    ///
    /// # Errors
    /// [`TrackerError`] when the parent is missing or not an array, or a
    /// segment is reserved.
    pub fn copy_within(
        &mut self,
        path: &[Seg],
        source: usize,
        destination: usize,
        count: usize,
    ) -> Result<(), TrackerError> {
        Self::guard_path(path)?;
        let container = resolve_container_mut(&mut self.root, path)?;
        let Some(array) = container.as_array_mut() else {
            return Err(Self::type_error("copyWithin runs on an array"));
        };
        for offset in 0..count {
            let Some(from) = source.checked_add(offset) else {
                break;
            };
            let Some(to) = destination.checked_add(offset) else {
                break;
            };
            let Some(value) = array.get(from).cloned() else {
                break;
            };
            if to < array.len() {
                array[to] = value;
            }
        }
        self.has_pending = true;
        mark_array_diff(&mut self.pending, path);
        Ok(())
    }

    /// Replaces the whole tracked value, the `state = next` setter.
    ///
    /// Upstream's identity no-op — assigning the tracker's own proxy back —
    /// is unobservable in owned data: both the identity and the replacement
    /// branch produce the same batch stream, a base batch on the next flush
    /// carrying the current value, so the port always takes the replacement
    /// branch.
    pub fn replace_root(&mut self, value: JsonValue) {
        self.clear_pending();
        self.root = value;
        self.baseline = None;
        self.force_base = true;
    }

    /// Emits the operation batch that turns the last published value into the
    /// current one, and advances the baseline.
    #[must_use]
    pub fn flush(&mut self) -> Vec<Op> {
        if self.force_base {
            let value = self.root.clone();
            self.baseline = Some(self.root.clone());
            self.force_base = false;
            self.clear_pending();
            return vec![Op::Replace(value)];
        }
        let mut out = Vec::new();
        if self.has_pending
            && let Some(baseline) = self.baseline.as_ref()
        {
            walk_dirty(
                baseline,
                &self.root,
                &self.pending,
                &[],
                self.scan,
                &mut out,
            );
            let synced = match self.baseline.as_mut() {
                Some(baseline) => sync_baseline(baseline, &self.root, &self.pending),
                None => false,
            };
            if !synced && !out.is_empty() {
                let replayed = self.baseline.take();
                self.baseline = Some(Self::replay(replayed, &out));
            }
        }
        self.clear_pending();
        out
    }

    /// Advances the baseline by replaying the emitted ops.
    ///
    /// # Panics
    /// Only if the tracker's own emission fails to apply, which its emitted
    /// vocabulary cannot make happen; a panic here is a port bug, not a runtime
    /// condition.
    fn replay(baseline: Option<JsonValue>, ops: &[Op]) -> JsonValue {
        #[allow(
            clippy::expect_used,
            reason = "the ops are this tracker's own emission; apply cannot reject them"
        )]
        {
            apply(baseline, ops.to_vec()).expect("flush-emitted ops are valid by construction")
        }
    }

    /// Makes the next flush a complete base batch without changing the value.
    pub fn rebase(&mut self) {
        self.clear_pending();
        self.force_base = true;
    }

    /// Accepts pending mutations locally without emitting them.
    pub fn discard(&mut self) {
        self.baseline = Some(self.root.clone());
        self.clear_pending();
    }

    fn clear_pending(&mut self) {
        self.pending = DirtyNode::new();
        self.has_pending = false;
    }

    fn type_error(message: impl Into<String>) -> TrackerError {
        TrackerError::TypeError(message.into())
    }

    fn guard_path(path: &[Seg]) -> Result<(), TrackerError> {
        for segment in path {
            if let Seg::Key(key) = segment
                && RESERVED_SEGMENTS.contains(&key.as_str())
            {
                return Err(UnsafePathError {
                    segment: segment.clone(),
                }
                .into());
            }
        }
        Ok(())
    }
}

const fn split_path(path: &[Seg]) -> (&[Seg], &Seg) {
    #[allow(
        clippy::expect_used,
        reason = "mutation methods guard non-empty paths before splitting"
    )]
    let (last, rest) = path.split_last().expect("mutation paths are non-empty");
    (rest, last)
}

fn resolve_container_mut<'a>(
    root: &'a mut JsonValue,
    path: &[Seg],
) -> Result<&'a mut JsonValue, TrackerError> {
    let mut node = root;
    for segment in path {
        let next = own_value_mut(node, segment).ok_or_else(|| {
            TrackerError::TypeError(format!(
                "the tracked value at {} does not exist",
                path_to_json_text(path)
            ))
        })?;
        node = next;
    }
    Ok(node)
}

fn own_value<'a>(value: &'a JsonValue, segment: &Seg) -> Option<&'a JsonValue> {
    match (value, segment) {
        (JsonValue::Object(object), Seg::Key(key)) => object.get(key),
        (JsonValue::Array(items), Seg::Index(index)) => items.get(*index),
        _ => None,
    }
}

fn own_value_mut<'a>(value: &'a mut JsonValue, segment: &Seg) -> Option<&'a mut JsonValue> {
    match (value, segment) {
        (JsonValue::Object(object), Seg::Key(key)) => object.as_value_mut(key),
        (JsonValue::Array(items), Seg::Index(index)) => items.get_mut(*index),
        _ => None,
    }
}

/// Descends the pending tree along `path`, moving visited children to the end
/// so flush emits re-marked paths last, exactly as upstream's `Map` delete
/// and re-set does. Returns [`None`] when an ancestor already carries a whole
/// value or non-append array intent; the write still lands, only the finer
/// mark is dropped.
fn ensure_node<'a>(pending: &'a mut DirtyNode, path: &[Seg]) -> Option<&'a mut DirtyNode> {
    let mut node = pending;
    for segment in path {
        if node.value_dirty || matches!(node.array, Some(ArrayDirty::Diff | ArrayDirty::Replace)) {
            return None;
        }
        let at = node.children.iter().position(|(seen, _)| seen == segment);
        match at {
            Some(at) => {
                let child = node.children.remove(at).1;
                node.children.push((segment.clone(), child));
            }
            None => node.children.push((segment.clone(), DirtyNode::default())),
        }
        let last = node.children.len() - 1;
        node = &mut node.children[last].1;
    }
    Some(node)
}

fn find_node<'a>(pending: &'a DirtyNode, path: &[Seg]) -> Option<&'a DirtyNode> {
    let mut node = pending;
    for segment in path {
        let at = node.children.iter().position(|(seen, _)| seen == segment)?;
        node = &node.children[at].1;
    }
    Some(node)
}

/// Marks the whole value at `path` dirty.
fn mark_value(pending: &mut DirtyNode, path: &[Seg]) {
    if let Some(node) = ensure_node(pending, path) {
        node.value_dirty = true;
        node.array = None;
        node.children.clear();
    }
}

/// Records an array append at `start`; the first append on a node wins.
fn mark_array_append(pending: &mut DirtyNode, path: &[Seg], start: usize) {
    if let Some(node) = ensure_node(pending, path)
        && !node.value_dirty
        && !matches!(node.array, Some(ArrayDirty::Diff | ArrayDirty::Replace))
        && node.array.is_none()
    {
        node.array = Some(ArrayDirty::Append { start });
    }
}

/// Marks a non-append array change.
fn mark_array_diff(pending: &mut DirtyNode, path: &[Seg]) {
    if let Some(node) = ensure_node(pending, path)
        && !node.value_dirty
        && !matches!(node.array, Some(ArrayDirty::Replace))
    {
        node.array = Some(ArrayDirty::Diff);
        node.children.clear();
    }
}

/// Marks an array replaced wholesale.
fn mark_array_replace(pending: &mut DirtyNode, path: &[Seg]) {
    if let Some(node) = ensure_node(pending, path)
        && !node.value_dirty
    {
        node.array = Some(ArrayDirty::Replace);
        node.children.clear();
    }
}

/// The start of the pending append at `path`, if one is recorded.
fn append_start(pending: &DirtyNode, path: &[Seg]) -> Option<usize> {
    match find_node(pending, path)?.array {
        Some(ArrayDirty::Append { start }) => Some(start),
        _ => None,
    }
}

/// Walks the dirty tree, diffing `baseline` against `root` along the recorded
/// intents.
fn walk_dirty(
    before: &JsonValue,
    after: &JsonValue,
    node: &DirtyNode,
    path: &[Seg],
    scan: usize,
    out: &mut Vec<Op>,
) {
    if node.value_dirty {
        diff_value(Some(before), Some(after), path, scan, out);
        return;
    }
    if node.array == Some(ArrayDirty::Replace) {
        if before != after {
            emit_set(path, after, out);
        }
        return;
    }
    if let Some(ArrayDirty::Append { start }) = node.array {
        let (Some(before_items), Some(after_items)) = (before.as_array(), after.as_array()) else {
            diff_value(Some(before), Some(after), path, scan, out);
            return;
        };
        if before_items.len() != start || after_items.len() < start {
            diff_value(Some(before), Some(after), path, scan, out);
            return;
        }
        for (segment, child) in &node.children {
            let Seg::Index(index) = segment else {
                continue;
            };
            if *index >= start {
                continue;
            }
            let previous = own_value(before, segment);
            let current = own_value(after, segment);
            let child_path = child_path(path, segment);
            match (previous, current, child.value_dirty) {
                (Some(previous), Some(current), false)
                    if previous.is_container() && current.is_container() =>
                {
                    walk_dirty(previous, current, child, &child_path, scan, out);
                }
                (previous, current, _) => {
                    diff_value(previous, current, &child_path, scan, out);
                }
            }
        }
        let items = &after_items[start..];
        if !items.is_empty() {
            out.push(Op::Splice {
                path: path.to_vec(),
                index: start,
                remove: 0,
                items: items.to_vec(),
            });
        }
        return;
    }
    for (segment, child) in &node.children {
        let previous = own_value(before, segment);
        let current = own_value(after, segment);
        let child_path = child_path(path, segment);
        match (previous, current, child.value_dirty) {
            (Some(previous), Some(current), false)
                if previous.is_container() && current.is_container() =>
            {
                walk_dirty(previous, current, child, &child_path, scan, out);
            }
            (previous, current, _) => {
                diff_value(previous, current, &child_path, scan, out);
            }
        }
    }
}

fn child_path(path: &[Seg], segment: &Seg) -> Vec<Seg> {
    let mut child = path.to_vec();
    child.push(segment.clone());
    child
}

/// Brings the baseline up to the root along the dirty paths by cloning the
/// changed values. Upstream shares references for scalars, strings, and
/// array appends and declines the rest so the caller can replay ops; owned
/// data cannot share, so a successful sync deep-copies the touched subtrees
/// and a failed sync — any non-append array change — leaves the replay path
/// in charge, which is O(changes) instead of O(array).
fn sync_baseline(baseline: &mut JsonValue, root: &JsonValue, node: &DirtyNode) -> bool {
    if !can_sync(node) {
        return false;
    }
    sync_into(baseline, root, node);
    true
}

fn can_sync(node: &DirtyNode) -> bool {
    if node
        .array
        .is_some_and(|array| !matches!(array, ArrayDirty::Append { .. }))
    {
        return false;
    }
    node.children.iter().all(|(_, child)| can_sync(child))
}

fn sync_child(parent: &mut JsonValue, root: &JsonValue, segment: &Seg, child: &DirtyNode) {
    let Some(current) = own_value(root, segment) else {
        match (parent, segment) {
            (JsonValue::Array(items), Seg::Index(index)) => {
                items.remove(*index);
            }
            (JsonValue::Object(object), Seg::Key(key)) => {
                object.remove(key);
            }
            _ => {}
        }
        return;
    };
    let previous = own_value_mut(parent, segment);
    let shape_mismatch = match (&previous, current) {
        (Some(previous), _) => previous.as_array().is_some() != current.as_array().is_some(),
        (None, _) => true,
    };
    if child.value_dirty || !current.is_container() || shape_mismatch {
        match (parent, segment) {
            (JsonValue::Array(items), Seg::Index(index)) => {
                if *index == items.len() {
                    items.push(current.clone());
                } else if *index < items.len() {
                    items[*index] = current.clone();
                }
            }
            (JsonValue::Object(object), Seg::Key(key)) => {
                object.set(key.clone(), current.clone());
            }
            _ => {}
        }
        return;
    }
    if let Some(previous) = previous {
        sync_into(previous, current, child);
    }
}

/// The shared body of [`sync_baseline`]: mirrors upstream's `syncInto`,
/// cloning rather than sharing every value it moves into the baseline.
fn sync_into(baseline: &mut JsonValue, root: &JsonValue, node: &DirtyNode) {
    if let Some(ArrayDirty::Append { start }) = node.array
        && matches!((baseline.as_array(), root.as_array()), (Some(_), Some(_)))
    {
        for (segment, child) in &node.children {
            if matches!(segment, Seg::Index(index) if *index < start) {
                sync_child(baseline, root, segment, child);
            }
        }
        if let (Some(root_items), Some(baseline_items)) = (root.as_array(), baseline.as_array_mut())
        {
            baseline_items.extend(root_items[start..].iter().cloned());
        }
        return;
    }
    for (segment, child) in &node.children {
        sync_child(baseline, root, segment, child);
    }
}
