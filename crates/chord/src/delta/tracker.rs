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
    if matches!(node.array, Some(ArrayDirty::Diff)) {
        diff_value(Some(before), Some(after), path, scan, out);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::codec::{decoder, encoder};
    use crate::delta::{PathRef, WireOp, is_base, track};
    use crate::test_support::{applied, idx, ja, jb, jn, jo, js, key, ok, some};

    fn pad() -> JsonValue {
        js(&"p".repeat(400))
    }

    /// The current string at `path`, or the empty string when absent.
    fn text_at(tracker: &Tracker, path: &[Seg]) -> String {
        match tracker.get(path) {
            Some(JsonValue::Str(text)) => text.clone(),
            _ => String::new(),
        }
    }

    /// The `state.s += suffix` idiom through the typed surface.
    fn append_text(tracker: &mut Tracker, path: &[Seg], suffix: &str) {
        let current = text_at(tracker, path);
        ok(tracker.set(path, JsonValue::Str(current + suffix)));
    }

    /// The array length at `path`, read through the tracker.
    fn length_at(tracker: &Tracker, path: &[Seg]) -> usize {
        tracker
            .get(path)
            .and_then(JsonValue::as_array)
            .map_or(0, Vec::len)
    }

    #[test]
    fn records_an_append_not_a_replacement() {
        let initial = jo(vec![("s", js("")), ("pad", pad())]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        append_text(&mut t, &[key("s")], "ab");
        append_text(&mut t, &[key("s")], "cd");
        let ops = t.flush();
        assert_eq!(
            ops,
            vec![Op::Append {
                path: vec![key("s")],
                suffix: "abcd".to_string(),
            }]
        );
        assert_eq!(
            applied(Some(initial), ops),
            jo(vec![("s", js("abcd")), ("pad", pad())])
        );
    }

    #[test]
    fn recovers_truncate_and_append_from_a_rolling_window() {
        let initial = jo(vec![("s", js("abcdefgh")), ("pad", pad())]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        let rolled = format!("{}xyz", &text_at(&t, &[key("s")])[3..]);
        ok(t.set(&[key("s")], js(&rolled)));
        let ops = t.flush();
        assert_eq!(
            ops,
            vec![
                Op::Truncate {
                    path: vec![key("s")],
                    count: 3,
                },
                Op::Append {
                    path: vec![key("s")],
                    suffix: "xyz".to_string(),
                },
            ]
        );
        assert_eq!(
            applied(Some(initial), ops),
            jo(vec![("s", js("defghxyz")), ("pad", pad())])
        );
    }

    // Upstream pins proxy-identity caching here: reading `state.child` twice
    // returns the same proxy and reassigning the child replaces it. Owned
    // data has no proxies; the observable rest is that the write lands and
    // reads stay consistent.
    #[test]
    fn caches_nested_proxies_while_replacing_changed_children() {
        let mut t = Tracker::new(jo(vec![("child", jo(vec![("value", jn(1.0))]))]));
        let first = t.get(&[key("child")]).cloned();
        assert_eq!(t.get(&[key("child")]).cloned(), first);
        ok(t.set(&[key("child")], jo(vec![("value", jn(2.0))])));
        assert_eq!(
            t.get(&[key("child")]).cloned(),
            Some(jo(vec![("value", jn(2.0))]))
        );
        assert_eq!(
            t.get(&[key("child")]).cloned(),
            t.get(&[key("child")]).cloned()
        );
    }

    #[test]
    fn records_array_intent_as_one_splice() {
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0)])), ("pad", pad())]));
        let _ = t.flush();
        ok(t.push(&[key("xs")], vec![jn(3.0)]));
        assert_eq!(
            t.flush(),
            vec![Op::Splice {
                path: vec![key("xs")],
                index: 2,
                remove: 0,
                items: vec![jn(3.0)],
            }]
        );
    }

    #[test]
    fn normalises_a_delete_of_a_set_value() {
        let mut t = Tracker::new(jo(vec![("a", jn(1.0)), ("pad", pad())]));
        let _ = t.flush();
        ok(t.delete(&[key("a")]));
        assert_eq!(
            t.flush(),
            vec![Op::Delete {
                path: vec![key("a")],
            }]
        );
    }

    #[test]
    fn uses_absence_and_delete_for_optional_object_properties() {
        let mut t = Tracker::new(jo(vec![("foo", jn(1.0))]));
        let replica = applied(None, t.flush());
        assert!(t.get(&[key("something")]).is_none());
        ok(t.set(&[key("something")], js("enabled")));
        let replica = applied(Some(replica), t.flush());
        assert_eq!(
            replica,
            jo(vec![("foo", jn(1.0)), ("something", js("enabled"))])
        );
        ok(t.delete(&[key("something")]));
        let ops = t.flush();
        assert_eq!(
            ops,
            vec![Op::Delete {
                path: vec![key("something")],
            }]
        );
        assert_eq!(applied(Some(replica), ops), jo(vec![("foo", jn(1.0))]));
    }

    #[test]
    fn round_trips_interleaved_writes() {
        let initial = jo(vec![("out", js(&"x".repeat(500))), ("total", jn(0.0))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        for index in 0..1_000 {
            let out = text_at(&t, &[key("out")]);
            ok(t.set(&[key("out")], js(&format!("{}{:010}", &out[10..], index))));
            let total = some(t.get(&[key("total")]).and_then(JsonValue::as_number));
            ok(t.set(&[key("total")], jn(total + 10.0)));
        }
        let ops = t.flush();
        assert!(ops.len() <= 3);
        let final_value = t.value().clone();
        assert_eq!(applied(Some(initial), ops), final_value);
    }

    #[test]
    fn deep_diffs_reassigned_objects() {
        let initial = jo(vec![(
            "message",
            jo(vec![
                ("content", ja(vec![jo(vec![("text", js("hello"))])])),
                ("count", jn(0.0)),
            ]),
        )]);
        let mut t = Tracker::new(initial);
        let _ = t.flush();
        ok(t.set(
            &[key("message")],
            jo(vec![
                ("content", ja(vec![jo(vec![("text", js("hello world"))])])),
                ("count", jn(1.0)),
            ]),
        ));
        assert_eq!(
            t.flush(),
            vec![
                Op::Append {
                    path: vec![key("message"), key("content"), idx(0), key("text")],
                    suffix: " world".to_string(),
                },
                Op::Set {
                    path: vec![key("message"), key("count")],
                    value: jn(1.0),
                },
            ]
        );
    }

    #[test]
    fn deep_diffs_retained_array_edits_combined_with_append_in_a_replacement() {
        let initial = jo(vec![(
            "view",
            jo(vec![(
                "messages",
                ja(vec![
                    jo(vec![("text", js("a"))]),
                    jo(vec![("text", js("b"))]),
                ]),
            )]),
        )]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.set(
            &[key("view")],
            jo(vec![(
                "messages",
                ja(vec![
                    jo(vec![("text", js("ax"))]),
                    jo(vec![("text", js("b"))]),
                    jo(vec![("text", js("c"))]),
                ]),
            )]),
        ));
        let ops = t.flush();
        assert_eq!(
            ops,
            vec![
                Op::Append {
                    path: vec![key("view"), key("messages"), idx(0), key("text")],
                    suffix: "x".to_string(),
                },
                Op::Splice {
                    path: vec![key("view"), key("messages")],
                    index: 2,
                    remove: 0,
                    items: vec![jo(vec![("text", js("c"))])],
                },
            ]
        );
        assert_eq!(applied(Some(initial), ops), t.value().clone());
    }

    #[test]
    fn emits_nothing_when_a_replacement_is_deeply_equal() {
        let nested = || ja(vec![jn(1.0), jo(vec![("text", js("same"))])]);
        let mut t = Tracker::new(jo(vec![("value", jo(vec![("nested", nested())]))]));
        let _ = t.flush();
        ok(t.set(&[key("value")], jo(vec![("nested", nested())])));
        assert!(t.dirty());
        assert_eq!(t.flush(), Vec::new());
        assert!(!t.dirty());
    }

    #[test]
    fn invalidates_a_pending_child_when_its_parent_is_overwritten() {
        let initial = jo(vec![("a", jo(vec![("x", jn(1.0))]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.set(&[key("a"), key("b")], jn(99.0)));
        ok(t.set(&[key("a")], jo(vec![("c", jn(2.0))])));
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn splices_a_value_that_is_itself_an_array() {
        let initial = ja(vec![jn(1.0), jn(2.0), jn(3.0), pad()]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.push(&[], vec![jn(4.0)]));
        let ops = t.flush();
        assert_eq!(
            ops,
            vec![Op::Splice {
                path: Vec::new(),
                index: 4,
                remove: 0,
                items: vec![jn(4.0)],
            }]
        );
        assert_eq!(
            applied(Some(initial), ops),
            ja(vec![jn(1.0), jn(2.0), jn(3.0), pad(), jn(4.0)])
        );
    }

    #[test]
    fn normalises_a_splice_covering_the_whole_root_to_a_replacement() {
        let mut t = Tracker::new(ja(vec![jn(1.0), jn(2.0), jn(3.0)]));
        let _ = t.flush();
        ok(t.splice(&[], 0, 3, &[jn(9.0)]));
        let ops = t.flush();
        assert_eq!(ops, vec![Op::Replace(ja(vec![jn(9.0)]))]);
        assert!(is_base(&ops));
    }

    #[test]
    fn normalises_a_nested_splice_all_to_a_set_not_a_replacement() {
        let mut t = Tracker::new(jo(vec![
            ("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)])),
            ("pad", pad()),
        ]));
        let _ = t.flush();
        ok(t.splice(&[key("xs")], 0, 3, &[jn(9.0)]));
        assert_eq!(
            t.flush(),
            vec![Op::Set {
                path: vec![key("xs")],
                value: ja(vec![jn(9.0)]),
            }]
        );
    }

    #[test]
    fn matches_splice_with_no_arguments() {
        let initial = jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.splice(&[key("xs")], 0, 0, &[]));
        assert_eq!(t.value(), &initial);
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn normalises_splice_arguments_to_integers() {
        let initial = jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        // Upstream normalizes `splice(0, undefined, 9)` and `splice(1.5, 1)`;
        // the typed surface carries the normalized integers directly.
        ok(t.splice(&[key("xs")], 0, 0, &[jn(9.0)]));
        ok(t.splice(&[key("xs")], 1, 1, &[]));
        let ops = t.flush();
        for op in &ops {
            if let Op::Splice { index, remove, .. } = op {
                assert!(index.to_string().parse::<usize>().is_ok());
                assert!(remove.to_string().parse::<usize>().is_ok());
            }
        }
        assert_eq!(applied(Some(initial), ops), t.value().clone());
    }

    #[test]
    fn normalises_length_zero_on_the_root() {
        let initial = ja(vec![jn(1.0), jn(2.0), jn(3.0)]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.set_length(&[], 0));
        let ops = t.flush();
        assert_eq!(ops, vec![Op::Replace(ja(Vec::new()))]);
        assert_eq!(applied(Some(initial), ops), ja(Vec::new()));
    }

    #[test]
    fn round_trips_repeated_nested_splices() {
        let initial = jo(vec![("xs", ja(vec![jn(1.0)]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        for index in 2..=100 {
            ok(t.push(&[key("xs")], vec![crate::test_support::u(index)]));
        }
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn does_not_fold_a_child_path_across_a_parent_splice() {
        let initial = jo(vec![("xs", ja(vec![js("ab"), js("cd")]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        append_text(&mut t, &[key("xs"), idx(0)], "x");
        ok(t.shift(&[key("xs")]));
        append_text(&mut t, &[key("xs"), idx(0)], "y");
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn drops_child_ops_when_repeated_parent_splices_collapse_to_a_snapshot() {
        let initial = jo(vec![("xs", ja(vec![js("ab")]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.push(&[key("xs")], vec![js("q")]));
        append_text(&mut t, &[key("xs"), idx(0)], "cd");
        ok(t.push(&[key("xs")], vec![js("z")]));
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn does_not_let_a_post_splice_set_dominate_an_earlier_element_append() {
        let initial = jo(vec![("xs", ja(vec![js("ab")]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        append_text(&mut t, &[key("xs"), idx(0)], "x");
        ok(t.unshift(&[key("xs")], vec![js("q")]));
        ok(t.set(&[key("xs"), idx(0)], js("Z")));
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn does_not_discard_a_nested_append_through_a_reindexed_set() {
        let initial = jo(vec![("xs", ja(vec![jo(vec![("k", js("a"))])]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        append_text(&mut t, &[key("xs"), idx(0), key("k")], "x");
        ok(t.unshift(&[key("xs")], vec![jn(9.0)]));
        ok(t.set(&[key("xs"), idx(0)], jn(7.0)));
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn preserves_element_operations_across_middle_array_insertions() {
        let initial = jo(vec![("xs", ja(vec![js("a"), js("b")]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        append_text(&mut t, &[key("xs"), idx(1)], "x");
        ok(t.splice(&[key("xs")], 1, 0, &[js("inserted")]));
        ok(t.set(&[key("xs"), idx(1)], js("changed")));
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn round_trips_deterministic_element_mutations_across_reindexing_splices() {
        let mut seed = 0x1234_abcd_u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        for round in 0..200 {
            let initial = jo(vec![(
                "xs",
                ja(vec![
                    jo(vec![("k", js("a"))]),
                    jo(vec![("k", js("b"))]),
                    jo(vec![("k", js("c"))]),
                ]),
            )]);
            let mut t = Tracker::new(initial.clone());
            let _ = t.flush();
            for step in 0..30 {
                let length = length_at(&t, &[key("xs")]);
                let index = usize::try_from(next()).unwrap_or(0) % length.max(1);
                match next() % 5 {
                    0 => {
                        let letter = some(char::from_u32(97 + next() % 26));
                        append_text(
                            &mut t,
                            &[key("xs"), idx(index), key("k")],
                            &letter.to_string(),
                        );
                    }
                    1 => ok(t.set(
                        &[key("xs"), idx(index)],
                        jo(vec![("k", js(&format!("set-{round}-{step}")))]),
                    )),
                    2 => {
                        ok(t.unshift(
                            &[key("xs")],
                            vec![jo(vec![("k", js(&format!("head-{round}-{step}")))])],
                        ));
                    }
                    3 => {
                        if length > 1 {
                            ok(t.shift(&[key("xs")]));
                        }
                    }
                    _ => {
                        let at = next() as usize % (length + 1);
                        let remove = usize::from(length > 1 && next() % 2 == 0);
                        ok(t.splice(
                            &[key("xs")],
                            at,
                            remove,
                            &[jo(vec![("k", js(&format!("mid-{round}-{step}")))])],
                        ));
                    }
                }
                if length_at(&t, &[key("xs")]) > 10 {
                    ok(t.shift(&[key("xs")]));
                }
            }
            assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
        }
    }

    #[test]
    fn keeps_mutator_chaining_tracked() {
        let initial = jo(vec![("xs", ja(vec![jn(3.0), jn(1.0), jn(2.0)]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.sort(&[key("xs")]));
        ok(t.push(&[key("xs")], vec![jn(4.0)]));
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn combines_retained_element_edits_with_one_append() {
        let initial = jo(vec![
            (
                "messages",
                ja(vec![
                    jo(vec![("text", js("a"))]),
                    jo(vec![("text", js("b"))]),
                ]),
            ),
            ("flag", jn(0.0)),
        ]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        append_text(&mut t, &[key("messages"), idx(0), key("text")], "x");
        ok(t.set(&[key("flag")], jn(1.0)));
        ok(t.push(&[key("messages")], vec![jo(vec![("text", js("c"))])]));
        append_text(&mut t, &[key("messages"), idx(1), key("text")], "y");
        append_text(&mut t, &[key("messages"), idx(2), key("text")], "z");
        let ops = t.flush();
        let splices: Vec<Op> = ops
            .iter()
            .filter(|op| matches!(op, Op::Splice { .. }))
            .cloned()
            .collect();
        assert_eq!(
            splices,
            vec![Op::Splice {
                path: vec![key("messages")],
                index: 2,
                remove: 0,
                items: vec![jo(vec![("text", js("cz"))])],
            }]
        );
        assert!(ops.iter().all(|op| !matches!(
            op,
            Op::Set { path, .. } if path.len() == 1 && path[0] == key("messages")
        )));
        assert_eq!(applied(Some(initial), ops), t.value().clone());
    }

    #[test]
    fn collapses_many_pushes_into_one_append() {
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        let _ = t.flush();
        for value in 2..=100 {
            ok(t.push(&[key("xs")], vec![crate::test_support::u(value)]));
        }
        assert_eq!(
            t.flush(),
            vec![Op::Splice {
                path: vec![key("xs")],
                index: 1,
                remove: 0,
                items: (2..=100).map(crate::test_support::u).collect(),
            }]
        );
    }

    #[test]
    fn keeps_direct_tail_writes_and_length_growth_in_append_mode() {
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        let _ = t.flush();
        ok(t.set(&[key("xs"), idx(1)], jo(vec![("value", jn(2.0))])));
        ok(t.set(&[key("xs"), idx(1), key("value")], jn(3.0)));
        ok(t.set_length(&[key("xs")], 4));
        assert_eq!(
            t.flush(),
            vec![Op::Splice {
                path: vec![key("xs")],
                index: 1,
                remove: 0,
                items: vec![
                    jo(vec![("value", jn(3.0))]),
                    JsonValue::Null,
                    JsonValue::Null,
                ],
            }]
        );
    }

    #[test]
    fn keeps_tail_only_splices_in_append_mode() {
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jo(vec![("value", jn(1.0))])]))]));
        let _ = t.flush();
        ok(t.push(
            &[key("xs")],
            vec![jo(vec![("value", jn(2.0))]), jo(vec![("value", jn(3.0))])],
        ));
        ok(t.splice(&[key("xs")], 1, 1, &[jo(vec![("value", jn(4.0))])]));
        ok(t.set(&[key("xs"), idx(2), key("value")], jn(5.0)));
        assert_eq!(
            t.flush(),
            vec![Op::Splice {
                path: vec![key("xs")],
                index: 1,
                remove: 0,
                items: vec![jo(vec![("value", jn(4.0))]), jo(vec![("value", jn(5.0))]),],
            }]
        );
    }

    #[test]
    fn forgets_append_tail_mutations_that_cancel_out() {
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        let _ = t.flush();
        ok(t.push(&[key("xs")], vec![jn(2.0)]));
        ok(t.push(&[key("xs")], vec![jn(3.0)]));
        ok(t.pop(&[key("xs")]));
        ok(t.pop(&[key("xs")]));
        assert_eq!(t.flush(), Vec::new());
    }

    #[test]
    fn accepts_large_append_argument_lists_without_spreading_them_internally() {
        let mut t = Tracker::new(jo(vec![("xs", ja(Vec::new()))]));
        let _ = t.flush();
        let items = vec![JsonValue::Null; 100_000];
        let pushed = ok(t.push(&[key("xs")], items.clone()));
        assert_eq!(pushed, items.len());
        assert_eq!(
            t.flush(),
            vec![Op::Splice {
                path: vec![key("xs")],
                index: 0,
                remove: 0,
                items,
            }]
        );
    }

    #[test]
    fn grows_arrays_with_explicit_null_values() {
        let initial = jo(vec![("xs", ja(vec![jn(1.0)]))]);
        let mut t = Tracker::new(initial.clone());
        let _ = t.flush();
        ok(t.set_length(&[key("xs")], 4));
        assert_eq!(
            t.get(&[key("xs")]).cloned(),
            Some(ja(vec![
                jn(1.0),
                JsonValue::Null,
                JsonValue::Null,
                JsonValue::Null
            ]))
        );
        assert_eq!(applied(Some(initial), t.flush()), t.value().clone());
    }

    #[test]
    fn assigning_to_state_emits_a_base_batch_and_keeps_tracking() {
        let mut t = Tracker::new(jo(vec![("p", jn(1.0)), ("q", pad())]));
        let _ = t.flush();
        ok(t.set(&[key("p")], jn(2.0)));
        t.replace_root(jo(vec![("r", jn(9.0)), ("s", js("new"))]));
        let ops = t.flush();
        assert_eq!(
            ops,
            vec![Op::Replace(jo(vec![("r", jn(9.0)), ("s", js("new"))]))]
        );
        assert!(is_base(&ops));
        assert_eq!(applied(Some(jo(Vec::new())), ops), t.value().clone());
        ok(t.set(&[key("r")], jn(10.0)));
        assert_eq!(
            t.flush(),
            vec![Op::Set {
                path: vec![key("r")],
                value: jn(10.0),
            }]
        );
    }

    #[test]
    fn assigning_the_tracked_root_to_state_rebases_without_poisoning_the_tracker() {
        let mut t = Tracker::new(jo(vec![("value", jn(1.0))]));
        let _ = t.flush();
        let tracked_root = t.value().clone();
        t.replace_root(tracked_root);
        assert_eq!(t.flush(), vec![Op::Replace(jo(vec![("value", jn(1.0))]))]);
        ok(t.set(&[key("value")], jn(2.0)));
        assert_eq!(
            t.flush(),
            vec![Op::Set {
                path: vec![key("value")],
                value: jn(2.0),
            }]
        );
    }

    // Upstream's self-assignment no-op skips the mark when the assigned value
    // is the tracker's own child proxy. Owned data cannot observe identity,
    // so the reassignment marks; the emitted batch is still empty, which is
    // what the case pins.
    #[test]
    fn allows_a_tracked_child_self_assignment_as_a_no_op() {
        let mut t = Tracker::new(jo(vec![("child", jo(vec![("value", jn(1.0))]))]));
        let _ = t.flush();
        let child = some(t.get(&[key("child")]).cloned());
        ok(t.set(&[key("child")], child));
        assert_eq!(t.flush(), Vec::new());
    }

    #[test]
    fn assigning_to_state_discards_ops_recorded_before_it() {
        let mut t = Tracker::new(jo(vec![("p", jn(1.0)), ("q", pad())]));
        let _ = t.flush();
        ok(t.set(&[key("p")], jn(2.0)));
        t.replace_root(jo(vec![("z", jn(1.0))]));
        assert_eq!(t.flush(), vec![Op::Replace(jo(vec![("z", jn(1.0))]))]);
    }

    #[test]
    fn a_partial_rewrite_keeps_its_ops() {
        let mut t = Tracker::new(jo(vec![
            ("a", js(&"x".repeat(200))),
            ("b", js(&"y".repeat(200))),
        ]));
        let _ = t.flush();
        ok(t.set(&[key("a")], js("p")));
        let ops = t.flush();
        assert!(!is_base(&ops));
        assert_eq!(
            ops,
            vec![Op::Set {
                path: vec![key("a")],
                value: js("p"),
            }]
        );
    }

    #[test]
    fn rebase_forces_a_base_batch_without_changing_the_value() {
        let mut t = Tracker::new(jo(vec![("p", jn(1.0)), ("pad", pad())]));
        let _ = t.flush();
        t.rebase();
        let ops = t.flush();
        assert!(is_base(&ops));
        assert_eq!(applied(Some(jo(Vec::new())), ops), t.value().clone());
    }

    #[test]
    fn rebase_applies_once_not_to_every_later_flush() {
        let mut t = Tracker::new(jo(vec![("p", jn(1.0)), ("pad", pad())]));
        let _ = t.flush();
        t.rebase();
        let _ = t.flush();
        ok(t.set(&[key("p")], jn(2.0)));
        assert_eq!(
            t.flush(),
            vec![Op::Set {
                path: vec![key("p")],
                value: jn(2.0),
            }]
        );
    }

    #[test]
    fn the_first_flush_is_always_a_base_batch() {
        let mut t = Tracker::new(jo(vec![("x", jn(0.0))]));
        ok(t.set(&[key("x")], jn(100.0)));
        let ops = t.flush();
        assert!(is_base(&ops));
        assert_eq!(ops, vec![Op::Replace(jo(vec![("x", jn(100.0))]))]);
    }

    #[test]
    fn the_first_flush_is_followed_by_deltas() {
        let mut t = Tracker::new(jo(vec![("x", jn(0.0)), ("pad", pad())]));
        let _ = t.flush();
        ok(t.set(&[key("x")], jn(1.0)));
        assert_eq!(
            t.flush(),
            vec![Op::Set {
                path: vec![key("x")],
                value: jn(1.0),
            }]
        );
    }

    #[test]
    fn the_first_flush_emits_nothing_when_untouched() {
        let mut t = Tracker::new(jo(vec![("x", jn(0.0))]));
        assert!(t.dirty());
        assert_eq!(t.flush(), vec![Op::Replace(jo(vec![("x", jn(0.0))]))]);
        assert!(!t.dirty());
        assert_eq!(t.flush(), Vec::new());
    }

    #[test]
    fn discard_accepts_pending_changes_into_the_local_baseline() {
        let mut t = Tracker::new(jo(vec![("x", jn(0.0)), ("y", jn(0.0))]));
        let _ = t.flush();
        ok(t.set(&[key("x")], jn(1.0)));
        assert!(t.dirty());
        t.discard();
        assert!(!t.dirty());
        assert_eq!(t.flush(), Vec::new());
        ok(t.set(&[key("y")], jn(1.0)));
        assert_eq!(
            t.flush(),
            vec![Op::Set {
                path: vec![key("y")],
                value: jn(1.0),
            }]
        );
    }

    fn run_minimized(mutate: impl FnOnce(&mut Tracker)) -> Vec<Op> {
        let initial = jo(vec![
            ("a", jo(vec![("b", jn(1.0))])),
            ("x", jn(1.0)),
            ("y", jn(2.0)),
        ]);
        let mut t = Tracker::new(initial);
        let _ = t.flush();
        mutate(&mut t);
        t.flush()
    }

    #[test]
    fn collapses_repeated_writes_to_one_field() {
        assert_eq!(
            run_minimized(|t| {
                ok(t.set(&[key("x")], jn(1.0)));
                ok(t.set(&[key("x")], jn(2.0)));
                ok(t.set(&[key("x")], jn(3.0)));
            }),
            vec![Op::Set {
                path: vec![key("x")],
                value: jn(3.0),
            }]
        );
    }

    #[test]
    fn drops_a_write_superseded_by_a_later_non_adjacent_one() {
        assert_eq!(
            run_minimized(|t| {
                ok(t.set(&[key("x")], jn(10.0)));
                ok(t.set(&[key("y")], jn(20.0)));
                ok(t.set(&[key("x")], jn(30.0)));
            }),
            vec![
                Op::Set {
                    path: vec![key("y")],
                    value: jn(20.0),
                },
                Op::Set {
                    path: vec![key("x")],
                    value: jn(30.0),
                },
            ]
        );
    }

    #[test]
    fn drops_a_child_write_when_the_parent_is_replaced_after_it() {
        assert_eq!(
            run_minimized(|t| {
                ok(t.set(&[key("a"), key("b")], jn(99.0)));
                ok(t.set(&[key("a")], jo(vec![("c", jn(5.0))])));
            }),
            vec![
                Op::Set {
                    path: vec![key("a"), key("c")],
                    value: jn(5.0),
                },
                Op::Delete {
                    path: vec![key("a"), key("b")],
                },
            ]
        );
    }

    #[test]
    fn folds_a_child_write_into_its_pending_parent_replacement() {
        assert_eq!(
            run_minimized(|t| {
                ok(t.set(&[key("a")], jo(vec![("b", jn(1.0))])));
                ok(t.set(&[key("a"), key("b")], jn(7.0)));
            }),
            vec![Op::Set {
                path: vec![key("a"), key("b")],
                value: jn(7.0),
            }]
        );
    }

    #[test]
    fn collapses_set_then_delete_to_the_delete() {
        assert_eq!(
            run_minimized(|t| {
                ok(t.set(&[key("x")], jn(5.0)));
                ok(t.delete(&[key("x")]));
            }),
            vec![Op::Delete {
                path: vec![key("x")],
            }]
        );
    }

    #[test]
    fn collapses_delete_then_set_to_the_final_value() {
        assert_eq!(
            run_minimized(|t| {
                ok(t.delete(&[key("x")]));
                ok(t.set(&[key("x")], jn(5.0)));
            }),
            vec![Op::Set {
                path: vec![key("x")],
                value: jn(5.0),
            }]
        );
    }

    #[test]
    fn derives_an_append_after_delete_then_recreate() {
        let mut t = Tracker::new(jo(vec![("x", js("")), ("y", jn(1.0))]));
        let _ = t.flush();
        ok(t.delete(&[key("x")]));
        ok(t.set(&[key("x")], js("ab")));
        append_text(&mut t, &[key("x")], "cd");
        assert_eq!(
            t.flush(),
            vec![Op::Append {
                path: vec![key("x")],
                suffix: "abcd".to_string(),
            }]
        );
    }

    #[test]
    fn is_linear_in_the_number_of_ops() {
        let wide = |n: usize| {
            let entries: Vec<(String, JsonValue)> = (0..n)
                .map(|i| (format!("f{i}"), crate::test_support::u(i)))
                .collect();
            let mut t = Tracker::new(JsonValue::Object(crate::types::JsonObject::from_entries(
                entries,
            )));
            let _ = t.flush();
            let started = std::time::Instant::now();
            for index in 0..n {
                ok(t.set(
                    &[key(&format!("f{index}"))],
                    crate::test_support::u(index + 1),
                ));
            }
            let _ = t.flush();
            started.elapsed()
        };
        let _ = wide(200);
        let small = wide(250).as_secs_f64().max(0.1);
        let large = wide(2_500).as_secs_f64();
        assert!(large / small < 40.0, "{large} / {small} grew superlinearly");
    }

    #[test]
    fn collapses_a_pathological_redundant_producer() {
        let mut t = Tracker::new(jo(vec![("a", jo(vec![("b", jn(1.0))])), ("x", jn(1.0))]));
        let _ = t.flush();
        for index in 0..2_000 {
            ok(t.set(&[key("x")], crate::test_support::u(index)));
            ok(t.set(&[key("a"), key("b")], crate::test_support::u(index)));
        }
        ok(t.set(&[key("a")], jo(vec![("done", jb(true))])));
        let ops = t.flush();
        assert_eq!(ops.len(), 3);
        assert_eq!(
            applied(
                Some(jo(vec![("a", jo(vec![("b", jn(1.0))])), ("x", jn(1.0))])),
                ops
            ),
            t.value().clone()
        );
    }

    #[test]
    fn drops_everything_before_a_replacement() {
        let mut t = Tracker::new(ja(vec![jn(1.0), jn(2.0)]));
        let _ = t.flush();
        ok(t.push(&[], vec![jn(3.0)]));
        ok(t.splice(&[], 0, 3, &[jn(7.0)]));
        assert_eq!(t.flush(), vec![Op::Replace(ja(vec![jn(7.0)]))]);
    }

    #[test]
    fn keeps_the_prefix_when_the_replacement_is_nested() {
        let mut t = Tracker::new(jo(vec![
            ("a", jn(1.0)),
            ("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)])),
            ("pad", pad()),
        ]));
        let _ = t.flush();
        ok(t.set(&[key("a")], jn(2.0)));
        ok(t.splice(&[key("xs")], 0, 3, &[jn(9.0)]));
        assert_eq!(
            t.flush(),
            vec![
                Op::Set {
                    path: vec![key("a")],
                    value: jn(2.0),
                },
                Op::Set {
                    path: vec![key("xs")],
                    value: ja(vec![jn(9.0)]),
                },
            ]
        );
    }

    #[test]
    fn rejects_a_constructor_walk() {
        let parsed = Op::from_json(&ja(vec![
            js("s"),
            ja(vec![js("constructor"), js("prototype"), js("gadget")]),
            jb(true),
        ]));
        // Upstream throws inside `apply` on the parsed tuple; the port's
        // parser carries the same rejection, so the op never becomes an Op.
        assert!(parsed.is_err());
    }

    #[test]
    fn rejects_a_forbidden_path_reached_through_an_interned_id() {
        let wire = vec![
            WireOp::Define {
                id: 0,
                path: vec![key("__proto__"), key("w")],
            },
            WireOp::Set {
                path: PathRef::Id(0),
                value: jb(true),
            },
        ];
        assert!(decoder().decode(&wire).is_err());
    }

    // Upstream pins that an inherited setter never runs on write. Rust has
    // no prototype chain, so the same op on a missing key simply creates it.
    #[test]
    fn does_not_run_an_inherited_setter() {
        let created = apply(
            Some(jo(Vec::new())),
            vec![Op::Set {
                path: vec![key("trap")],
                value: jn(1.0),
            }],
        );
        assert_eq!(ok(created), jo(vec![("trap", jn(1.0))]));
    }

    #[test]
    fn allows_a_reserved_name_as_a_value_key() {
        let out = apply(
            Some(jo(Vec::new())),
            vec![Op::Set {
                path: vec![key("a")],
                value: parse_proto_value(),
            }],
        );
        assert!(out.is_ok());
    }

    /// `{"__proto__": {"z": 1}}` — reserved as a value key, not a segment.
    fn parse_proto_value() -> JsonValue {
        jo(vec![("__proto__", jo(vec![("z", jn(1.0))]))])
    }

    #[test]
    fn clones_and_reads_reserved_value_keys_without_invoking_prototype_setters() {
        let mut t = Tracker::new(jo(vec![("value", parse_proto_value())]));
        let out = applied(None, t.flush());
        let stored = out
            .as_object()
            .and_then(|object| object.get("value"))
            .and_then(JsonValue::as_object);
        assert!(stored.is_some_and(|object| object.get("__proto__").is_some()));
        assert_eq!(t.value().clone(), jo(vec![("value", parse_proto_value())]));
        let rejected = t.set(&[key("value"), key("__proto__"), key("z")], jn(2.0));
        assert!(rejected.is_err());
    }

    #[test]
    fn accepts_a_value_that_merely_looks_like_an_op() {
        let mut t = Tracker::new(jo(vec![("pad", pad())]));
        let _ = t.flush();
        ok(t.set(&[key("x")], ja(vec![js("r"), jo(vec![("evil", jb(true))])])));
        assert_eq!(
            t.flush(),
            vec![Op::Set {
                path: vec![key("x")],
                value: ja(vec![js("r"), jo(vec![("evil", jb(true))])]),
            }]
        );
    }

    // Upstream rejects descriptor writes, prototype swaps, and extension
    // locks through the proxy traps. Rust's owned data has no such APIs, so
    // the restated surface is the typed mutation set itself: a reserved
    // segment is rejected before the value changes, and nothing else can
    // touch the tree.
    #[test]
    fn rejects_descriptor_and_object_shape_bypasses() {
        let mut t = Tracker::new(jo(vec![("pad", pad())]));
        let _ = t.flush();
        let rejected = t.set(&[key("constructor")], jn(42.0));
        assert!(rejected.is_err());
        assert_eq!(t.value(), &jo(vec![("pad", pad())]));
        assert_eq!(t.flush(), Vec::new());
    }

    // Upstream rejects producer-side sparse writes, array deletes, and
    // undefined elements through the same trap family.
    #[test]
    fn rejects_sparse_writes_at_the_producer() {
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))]));
        let _ = t.flush();
        assert!(t.set(&[key("xs"), idx(5)], jn(9.0)).is_err());
        assert_eq!(
            t.get(&[key("xs")]).cloned(),
            Some(ja(vec![jn(1.0), jn(2.0), jn(3.0)]))
        );
    }

    #[test]
    fn rejects_array_deletes_at_the_producer() {
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))]));
        let _ = t.flush();
        assert!(t.delete(&[key("xs"), idx(1)]).is_err());
        assert_eq!(
            t.get(&[key("xs")]).cloned(),
            Some(ja(vec![jn(1.0), jn(2.0), jn(3.0)]))
        );
    }

    #[test]
    fn rejects_undefined_array_elements_at_the_producer() {
        // Upstream rejects assigning `undefined` into an array because it
        // would open a hole; the owned tree has no `undefined`, and `delete`
        // on an array is the same rejection.
        let mut t = Tracker::new(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))]));
        let _ = t.flush();
        assert!(t.delete(&[key("xs"), idx(1)]).is_err());
        assert_eq!(
            t.get(&[key("xs")]).cloned(),
            Some(ja(vec![jn(1.0), jn(2.0), jn(3.0)]))
        );
    }

    // The property test mirrors upstream's 100-round, 60-step state machine;
    // its length is the case, not a refactor target.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the body is one long randomized walk; splitting it would hide the state machine"
    )]
    fn converges_across_mixed_nested_writes_replacements_and_array_mutations() {
        use crate::test_support::LCG_SEED;

        let mut seed = LCG_SEED;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        for round in 0..100 {
            let initial = || {
                jo(vec![
                    (
                        "rows",
                        ja(vec![
                            jo(vec![("text", js("a")), ("count", jn(0.0))]),
                            jo(vec![("text", js("b")), ("count", jn(0.0))]),
                        ]),
                    ),
                    ("meta", jo(vec![("revision", jn(0.0))])),
                ])
            };
            let mut tracker = Tracker::new(initial());
            let mut enc = encoder();
            let mut dec = decoder();
            let base = tracker.flush();
            let mut replica = applied(None, ok(dec.decode(&enc.encode(&base))));
            for step in 0..60 {
                let rows_length = length_at(&tracker, &[key("rows")]);
                match next() % 10 {
                    0 => {
                        if rows_length > 0 {
                            let letter = some(char::from_u32(97 + next() % 26));
                            let index = usize::try_from(next()).unwrap_or(0) % rows_length;
                            append_text(
                                &mut tracker,
                                &[key("rows"), idx(index), key("text")],
                                &letter.to_string(),
                            );
                        }
                    }
                    1 => {
                        if rows_length > 0 {
                            let index = usize::try_from(next()).unwrap_or(0) % rows_length;
                            let count = some(
                                tracker
                                    .get(&[key("rows"), idx(index), key("count")])
                                    .and_then(JsonValue::as_number),
                            );
                            ok(tracker
                                .set(&[key("rows"), idx(index), key("count")], jn(count + 1.0)));
                        }
                    }
                    2 => {
                        ok(tracker.push(
                            &[key("rows")],
                            vec![jo(vec![
                                ("text", js(&format!("tail-{round}-{step}"))),
                                ("count", crate::test_support::u(step)),
                            ])],
                        ));
                    }
                    3 => {
                        if rows_length > 0 {
                            ok(tracker.pop(&[key("rows")]));
                        }
                    }
                    4 => {
                        ok(tracker.unshift(
                            &[key("rows")],
                            vec![jo(vec![
                                ("text", js(&format!("head-{round}-{step}"))),
                                ("count", crate::test_support::u(step)),
                            ])],
                        ));
                    }
                    5 => {
                        if rows_length > 0 {
                            ok(tracker.shift(&[key("rows")]));
                        }
                    }
                    6 => {
                        let index = usize::try_from(next()).unwrap_or(0) % (rows_length + 1);
                        let remove = usize::from(rows_length > 0 && next() % 2 == 0);
                        ok(tracker.splice(
                            &[key("rows")],
                            index,
                            remove,
                            &[jo(vec![
                                ("text", js(&format!("mid-{round}-{step}"))),
                                ("count", crate::test_support::u(step)),
                            ])],
                        ));
                    }
                    7 => {
                        let mut replacement = some(tracker.get(&[key("rows")]).cloned());
                        if let Some(items) = replacement.as_array_mut() {
                            if let Some(first) = items.first_mut()
                                && let Some(row) = first.as_object_mut()
                            {
                                let text = row
                                    .get("text")
                                    .and_then(JsonValue::as_str)
                                    .map_or(String::new(), str::to_owned);
                                row.set("text", js(&(text + "r")));
                            }
                            if next() % 2 == 0 {
                                items.push(jo(vec![
                                    ("text", js("replacement-tail")),
                                    ("count", crate::test_support::u(step)),
                                ]));
                            }
                        }
                        ok(tracker.set(&[key("rows")], replacement));
                    }
                    8 => {
                        let revision = some(
                            tracker
                                .get(&[key("meta"), key("revision")])
                                .and_then(JsonValue::as_number),
                        );
                        ok(tracker.set(&[key("meta")], jo(vec![("revision", jn(revision + 1.0))])));
                    }
                    _ => {
                        if rows_length > 1 {
                            ok(tracker.reverse(&[key("rows")]));
                        }
                    }
                }
                let rows_length = length_at(&tracker, &[key("rows")]);
                if rows_length > 12 {
                    let remove = rows_length - 12;
                    ok(tracker.splice(&[key("rows")], 0, remove, &[]));
                }
                if next() % 5 == 0 {
                    let ops = ok(dec.decode(&enc.encode(&tracker.flush())));
                    replica = applied(Some(replica), ops);
                    assert_eq!(replica, tracker.value().clone());
                }
            }
            let ops = ok(dec.decode(&enc.encode(&tracker.flush())));
            replica = applied(Some(replica), ops);
            assert_eq!(replica, tracker.value().clone());
        }
    }

    /// Builds a random JSON value from the seeded generator, the port of
    /// upstream's `rnd` fixture with `Math.random` replaced by a pinned PRNG.
    fn rnd(rng: &mut rand::rngs::StdRng, depth: u32) -> JsonValue {
        use rand::Rng;

        let roll = rng.random::<f64>();
        if depth > 2 || roll < 0.3 {
            return jn(f64::from(rng.random_range(0..5)));
        }
        if roll < 0.45 {
            let options = [js("x"), js("y"), JsonValue::Null, jb(true)];
            return options[rng.random_range(0..4)].clone();
        }
        if roll < 0.7 {
            return ja((0..rng.random_range(0..4))
                .map(|_| rnd(rng, depth + 1))
                .collect());
        }
        let mut object = crate::types::JsonObject::new();
        for key_name in ["a", "b", "c"] {
            if rng.random::<f64>() < 0.6 {
                object.set(key_name, rnd(rng, depth + 1));
            }
        }
        JsonValue::Object(object)
    }

    #[test]
    fn a_replica_always_matches_the_producer() {
        use crate::test_support::seeded_rng;

        let mut rng = seeded_rng(0xD31A_5EED);
        let mut checked = 0;
        for _ in 0..3_000 {
            let base = rnd(&mut rng, 0);
            if !base.is_container() {
                continue;
            }
            let mut t = Tracker::new(base.clone());
            let _ = t.flush();
            let next = rnd(&mut rng, 0);
            match (t.value().as_array(), next.as_array()) {
                (Some(items), Some(next_items)) => {
                    ok(t.splice(&[], 0, items.len(), next_items.as_slice()));
                }
                _ if t.value().as_object().is_some() && next.as_object().is_some() => {
                    let next_object = some(next.as_object());
                    let names: Vec<String> = some(t.value().as_object())
                        .keys()
                        .filter(|name| !next_object.contains_key(name))
                        .map(str::to_owned)
                        .collect();
                    for name in names {
                        ok(t.delete(&[key(&name)]));
                    }
                    for (name, value) in next_object.iter() {
                        ok(t.set(&[key(name)], value.clone()));
                    }
                }
                _ => continue,
            }
            assert_eq!(applied(Some(base), t.flush()), t.value().clone());
            checked += 1;
        }
        // Most random pairs are shape-incompatible and skipped; this only
        // guards against the loop silently checking nothing.
        assert!(checked > 200);
    }

    fn rejection(message: &str, run: impl FnOnce() -> Result<(), TrackerError>) {
        let Err(error) = run() else {
            unreachable!("the case's mutation rejects")
        };
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn spells_tracker_errors() {
        assert_eq!(TrackerError::TypeError("no".to_string()).to_string(), "no");
        let error: TrackerError = UnsafePathError {
            segment: key("__proto__"),
        }
        .into();
        assert_eq!(error.to_string(), "unsafe path segment: __proto__");
    }

    #[test]
    fn rejects_type_mismatches_on_the_typed_surface() {
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        rejection("cannot be assigned through set", || t.set(&[], jn(1.0)));
        rejection("tracked root cannot be deleted", || t.delete(&[]));
        rejection("tracked arrays are addressed by index", || {
            t.set(&[key("xs"), key("a")], jn(1.0))
        });
        rejection("the tracked value at", || {
            t.set(&[idx(9), key("a")], jn(1.0))
        });
        rejection("the tracked parent is not a container", || {
            t.set(&[key("xs"), idx(0), key("a")], jn(1.0))
        });
        let mut t = track(jo(vec![("object", jo(vec![]))]));
        rejection("tracked objects are addressed by key", || {
            t.set(&[key("object"), idx(0)], jn(1.0))
        });
        rejection("the tracked value at", || {
            t.set(&[idx(0), key("a")], jn(1.0))
        });
        // The array-trap surface rejects object containers, one trap at a time.
        let mut t = track(jo(vec![("object", jo(vec![]))]));
        rejection("push runs on an array", || {
            t.push(&[key("object")], vec![jn(1.0)]).map(|_| ())
        });
        rejection("unshift runs on an array", || {
            t.unshift(&[key("object")], vec![jn(1.0)]).map(|_| ())
        });
        rejection("pop runs on an array", || {
            t.pop(&[key("object")]).map(|_| ())
        });
        rejection("shift runs on an array", || {
            t.shift(&[key("object")]).map(|_| ())
        });
        rejection("splice runs on an array", || {
            t.splice(&[key("object")], 0, 0, &[jn(1.0)]).map(|_| ())
        });
        rejection("length runs on an array", || {
            t.set_length(&[key("object")], 0)
        });
        rejection("sort runs on an array", || t.sort(&[key("object")]));
        rejection("reverse runs on an array", || t.reverse(&[key("object")]));
        rejection("fill runs on an array", || {
            t.fill(&[key("object")], jn(1.0))
        });
        rejection("copyWithin runs on an array", || {
            t.copy_within(&[key("object")], 0, 1, 1)
        });
        // Deleting through an array value's index would create a sparse array,
        // and deleting a property under an array resolves to nothing.
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        rejection("delete would create a sparse array", || {
            t.delete(&[key("xs"), idx(0)])
        });
        rejection("the tracked value at", || t.delete(&[idx(0), key("a")]));
        rejection("delete would create a sparse array", || {
            t.delete(&[key("xs"), idx(0), key("a")])
        });
        rejection("delete would create a sparse array", || t.delete(&[idx(0)]));
    }

    #[test]
    fn set_into_arrays_marks_within_and_beyond_appends() {
        // A sparse write beyond the current length rejects with the index.
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        rejection("unsafe path segment: 3", || {
            t.set(&[key("xs"), idx(3)], jn(1.0))
        });

        // Writing at an index inside an already-marked append range skips a
        // second mark: the append diff already covers it.
        let mut t = track(jo(vec![("xs", ja(Vec::new()))]));
        ok(t.push(&[key("xs")], vec![jn(1.0), jn(2.0), jn(3.0)]));
        ok(t.set(&[key("xs"), idx(1)], jn(9.0)));
        let ops = t.flush();
        assert_eq!(
            applied(Some(jo(vec![("xs", ja(Vec::new()))])), ops.clone()),
            t.value().clone()
        );
        // The append covers both the push and the overwrite.
        assert!(
            ops.len() == 1,
            "the overwrite inside an append stays part of the append: {ops:?}"
        );

        // Popping outside an append batch marks the array.
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0)]))]));
        ok(t.push(&[key("xs")], vec![jn(3.0)]));
        let _ = t.flush();
        assert_eq!(ok(t.pop(&[key("xs")])), Some(jn(3.0)));
        let ops = t.flush();
        assert_eq!(
            applied(Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0)]))])), ops),
            t.value().clone()
        );

        // An empty array shifts to nothing.
        let mut t = track(jo(vec![("xs", ja(Vec::new()))]));
        assert_eq!(ok(t.shift(&[key("xs")])), None);
        let base = jo(vec![("xs", ja(Vec::new()))]);
        assert_eq!(applied(Some(base), t.flush()), t.value().clone());
        assert_eq!(t.flush(), Vec::new());
    }

    #[test]
    fn set_length_marks_within_and_across_appends() {
        // Truncating into an append batch marks the array for diffing.
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0)]))]));
        ok(t.push(&[key("xs")], vec![jn(3.0)]));
        ok(t.set_length(&[key("xs")], 1));
        let ops = t.flush();
        assert_eq!(
            applied(Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0)]))])), ops),
            t.value().clone()
        );

        // Truncating to nothing replaces the array.
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        ok(t.set_length(&[key("xs")], 0));
        let ops = t.flush();
        assert_eq!(
            applied(Some(jo(vec![("xs", ja(vec![jn(1.0)]))])), ops),
            t.value().clone()
        );

        // Extending past the length fills with explicit nulls, upstream's
        // null-filled growth.
        ok(t.set_length(&[key("xs")], 2));
        assert_eq!(
            t.value()
                .as_object()
                .and_then(|object| object.get("xs"))
                .and_then(JsonValue::as_array)
                .cloned()
                .unwrap_or_default(),
            vec![JsonValue::Null, JsonValue::Null]
        );
    }

    #[test]
    fn copy_within_moves_elements_in_bounds() {
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))]));
        ok(t.copy_within(&[key("xs")], 0, 1, 2));
        let xs = t
            .value()
            .as_object()
            .and_then(|object| object.get("xs"))
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        assert_eq!(xs, vec![jn(1.0), jn(1.0), jn(1.0)]);
        // A source past the end stops the copy without failing.
        ok(t.copy_within(&[key("xs")], 9, 0, 1));
        // An overflow destination breaks the loop without failing.
        ok(t.copy_within(&[key("xs")], 0, usize::MAX, 1));
        // An overflow source breaks the loop without failing.
        ok(t.copy_within(&[key("xs")], usize::MAX, 0, 1));
        let ops = t.flush();
        assert_eq!(
            applied(
                Some(jo(vec![("xs", ja(vec![jn(1.0), jn(2.0), jn(3.0)]))])),
                ops
            ),
            t.value().clone()
        );
    }

    #[test]
    fn reconciles_flattened_and_reshaped_dirty_trees() {
        // An append whose container is replaced by a non-array flattens to a
        // value diff.
        let mut t = track(jo(vec![("xs", ja(vec![jn(1.0)]))]));
        ok(t.push(&[key("xs")], vec![jn(2.0)]));
        let _ = t.flush();
        t.replace_root(js("scalar"));
        let ops = t.flush();
        assert_eq!(applied(Some(js("scalar")), ops.clone()), js("scalar"));
        assert!(is_base(&ops), "the replaced root re-bases: {ops:?}");

        // A resolved child that vanished is removed during sync.
        let mut t = track(jo(vec![("a", jo(vec![("b", jn(1.0))]))]));
        ok(t.delete(&[key("a")]));
        ok(t.set(&[key("b")], jn(2.0)));
        let ops = t.flush();
        assert_eq!(
            applied(Some(jo(vec![("a", jo(vec![("b", jn(1.0))]))])), ops),
            t.value().clone()
        );

        // `track` and `track_with_options` build trackers with the same
        // surface; the first flush re-bases the whole tree.
        let mut factory_tracked = track(jo(vec![("n", jn(1.0))]));
        ok(factory_tracked.set(&[key("n")], jn(2.0)));
        let ops = factory_tracked.flush();
        assert_eq!(
            applied(Some(jo(vec![("n", jn(1.0))])), ops),
            factory_tracked.value().clone()
        );
        let mut bounded = crate::delta::track_with_options(
            jo(vec![("t", js("text"))]),
            TrackerOptions {
                max_overlap_scan: Some(8),
            },
        );
        append_text(&mut bounded, &[key("t")], "!");
        let ops = bounded.flush();
        assert_eq!(
            applied(Some(jo(vec![("t", js("text"))])), ops),
            bounded.value().clone()
        );
    }
}
