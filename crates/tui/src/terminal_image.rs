//! The cell-dimension store and the image-line probe of
//! `packages/tui/src/terminal-image.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The pieces land as their consumers need them: the process-global
//! cell-dimension store the `CSI 6 ; h ; w t` response feeds,
//! [`is_image_line`], the Kitty placement metadata registry with its crop
//! helper the compositor consults, and — with the alternate-screen renderer
//! ([#46](https://github.com/PhillipChaffee/pi-rust/issues/46)) — the
//! capability cache ([`get_capabilities`]/[`set_capabilities`]), the Kitty
//! placement extraction ([`get_kitty_image_placement`]) the placement
//! cache drives, and the batch deletion sequences. The rest of the file —
//! the environment capability detection and probes, `renderImage`, the
//! image-size parsers, and the `Image` component — is the image ticket's
//! scope ([#51](https://github.com/PhillipChaffee/pi-rust/issues/51)) and
//! lands there.
//!
//! Restatement: upstream stores cell dimensions and Kitty image metadata
//! in module globals read across TUI instances; the workspace forbids the
//! `unsafe` a naked `static mut` would need, so the stores sit behind
//! mutexes with the same process-wide visibility and the upstream
//! defaults. `getCapabilities` upstream runs the environment detection on
//! an empty cache; the detection is #51's scope, so the empty-cache answer
//! here is the neutral default (`images: null`, no true color, no
//! hyperlinks) — every consumer in this slice reads the cache the tests
//! seed.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::PoisonError;

const KITTY_PREFIX: &str = "\x1b_G";
const ITERM2_PREFIX: &str = "\x1b]1337;File=";

/// Physical cell size in pixels, upstream `CellDimensions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellDimensions {
    /// Cell width in pixels.
    pub width_px: u32,
    /// Cell height in pixels.
    pub height_px: u32,
}

static CELL_DIMENSIONS: Mutex<CellDimensions> = Mutex::new(CellDimensions {
    width_px: 9,
    height_px: 18,
});

/// The process-global cell dimensions, upstream `getCellDimensions`.
///
/// # Panics
///
/// Never: a poisoned lock falls back to the pre-poison value, matching the
/// module-global semantics upstream reads.
#[must_use]
pub fn get_cell_dimensions() -> CellDimensions {
    *CELL_DIMENSIONS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Store the process-global cell dimensions, upstream `setCellDimensions`.
///
/// # Panics
///
/// Never: a poisoned lock falls back to writing through it.
pub fn set_cell_dimensions(dims: CellDimensions) {
    *CELL_DIMENSIONS
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = dims;
}

/// Whether a rendered line carries a terminal-image placement instead of
/// text, upstream `isImageLine`: the Kitty or Iterm2 prefix anywhere in the
/// line.
///
/// The compositor and the line-reset pass leave image lines untouched.
#[must_use]
pub fn is_image_line(line: &str) -> bool {
    line.contains(KITTY_PREFIX) || line.contains(ITERM2_PREFIX)
}
/// Cell metrics of a registered Kitty placement, upstream
/// `KittyImageMetadata`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KittyImageMetadata {
    /// The placement's image id.
    pub image_id: u64,
    /// Placement width in columns.
    pub columns: usize,
    /// Placement height in rows.
    pub rows: usize,
    /// Source image width in pixels.
    pub width_px: u64,
    /// Source image height in pixels.
    pub height_px: u64,
}

/// The registry entry, upstream's private `RegisteredKittyImageMetadata`.
#[derive(Debug, Clone, Copy)]
struct RegisteredKittyImageMetadata {
    metadata: KittyImageMetadata,
    transmission_generation: u64,
}

struct KittyImageRegistry {
    by_id: HashMap<u64, RegisteredKittyImageMetadata>,
    insertion_order: VecDeque<u64>,
    transmission_generation: u64,
}

fn lock_registry() -> std::sync::MutexGuard<'static, KittyImageRegistry> {
    registry().lock().unwrap_or_else(PoisonError::into_inner)
}

fn registry() -> &'static Mutex<KittyImageRegistry> {
    static REGISTRY: std::sync::LazyLock<Mutex<KittyImageRegistry>> =
        std::sync::LazyLock::new(|| {
            Mutex::new(KittyImageRegistry {
                by_id: HashMap::new(),
                insertion_order: VecDeque::new(),
                transmission_generation: 0,
            })
        });
    &REGISTRY
}

/// Register a Kitty placement's cell metadata, upstream
/// `registerKittyImageMetadata`: the newest entry for an id wins, and the
/// registry evicts its oldest entry past 1000 ids.
pub fn register_kitty_image_metadata(metadata: KittyImageMetadata) {
    let mut registry = lock_registry();
    registry.transmission_generation += 1;
    let generation = registry.transmission_generation;
    if registry.by_id.remove(&metadata.image_id).is_some() {
        registry
            .insertion_order
            .retain(|id| *id != metadata.image_id);
    }
    registry.insertion_order.push_back(metadata.image_id);
    registry.by_id.insert(
        metadata.image_id,
        RegisteredKittyImageMetadata {
            metadata,
            transmission_generation: generation,
        },
    );
    if registry.by_id.len() > 1000
        && let Some(oldest) = registry.insertion_order.pop_front()
    {
        registry.by_id.remove(&oldest);
    }
}

/// The controls segment of the first Kitty command in the line.
fn controls_of(line: &str) -> Option<&str> {
    let start = line.find(KITTY_PREFIX)? + KITTY_PREFIX.len();
    let rest = &line[start..];
    let end = rest.find(';')?;
    Some(&rest[..end])
}

/// The `i=` control value of a Kitty controls string.
fn image_id_of(controls: &str) -> Option<u64> {
    controls.split(',').find_map(|control| {
        control
            .strip_prefix("i=")
            .and_then(|value| value.parse().ok())
    })
}

/// The metadata of the Kitty placement a line carries, upstream
/// `getKittyImageMetadata`: the `i=` control's registered entry.
#[must_use]
pub fn get_kitty_image_metadata(line: &str) -> Option<KittyImageMetadata> {
    registered_of(line).map(|registered| registered.metadata)
}

/// The full registry entry behind a line's first Kitty sequence, upstream's
/// private `getRegisteredKittyImageMetadata`: the `i=` control's registered
/// entry, carrying its transmission generation.
fn registered_of(line: &str) -> Option<RegisteredKittyImageMetadata> {
    let image_id = controls_of(line).and_then(image_id_of)?;
    let registry = lock_registry();
    registry.by_id.get(&image_id).copied()
}

/// Rewrite a Kitty placement for a cropped row range, upstream
/// `cropKittyImageLine`: the `y`/`h`/`r` controls are replaced so the
/// source rectangle covers only the visible rows.
#[must_use]
pub fn crop_kitty_image_line(line: &str, hidden_rows: usize, visible_rows: usize) -> String {
    let Some(metadata) = get_kitty_image_metadata(line) else {
        return line.to_string();
    };
    if visible_rows == 0 || hidden_rows >= metadata.rows {
        return line.to_string();
    }
    let cropped_rows = visible_rows.min(metadata.rows - hidden_rows);
    if hidden_rows == 0 && cropped_rows == metadata.rows {
        return line.to_string();
    }
    let rows = u64::try_from(metadata.rows).unwrap_or(u64::MAX);
    let source_y = metadata.height_px * u64::try_from(hidden_rows).unwrap_or(u64::MAX) / rows;
    let source_end = (metadata.height_px
        * u64::try_from(hidden_rows + cropped_rows).unwrap_or(u64::MAX))
    .div_ceil(rows);
    let source_height = source_end
        .min(metadata.height_px)
        .saturating_sub(source_y)
        .max(1);
    let Some(controls_start) = line
        .find(KITTY_PREFIX)
        .map(|start| start + KITTY_PREFIX.len())
    else {
        return line.to_string();
    };
    let Some(controls_end) = line[controls_start..].find(';') else {
        return line.to_string();
    };
    let mut controls: Vec<String> = line[controls_start..controls_start + controls_end]
        .split(',')
        .filter(|control| {
            !(control.starts_with("y=") || control.starts_with("h=") || control.starts_with("r="))
        })
        .map(String::from)
        .collect();
    controls.push(format!("y={source_y}"));
    controls.push(format!("h={source_height}"));
    controls.push(format!("r={cropped_rows}"));
    let match_start = controls_start - KITTY_PREFIX.len();
    let match_end = controls_start + controls_end + 1;
    format!(
        "{}\x1b_G{};{}",
        &line[..match_start],
        controls.join(","),
        &line[match_end..]
    )
}

/// Delete a Kitty graphics image by id, upstream `deleteKittyImage`.
///
/// The uppercase `d=I` frees both the placement and the uploaded image data;
/// the main-screen renderer emits it for every id it is about to overwrite.
#[must_use]
pub fn delete_kitty_image(image_id: u64) -> String {
    format!("\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\")
}

/// Encode a Kitty graphics transmission, upstream `encodeKitty`: the
/// placement command with its controls, chunked at the 4096-byte
/// transmission boundary when the base64 payload is larger.
#[must_use]
pub fn encode_kitty(base64_data: &str, options: EncodeKittyOptions) -> String {
    const CHUNK_SIZE: usize = 4096;

    let mut params: Vec<String> = vec!["a=T".to_string(), "f=100".to_string(), "q=2".to_string()];

    if options.move_cursor == Some(false) {
        params.push("C=1".to_string());
    }
    if let Some(columns) = options.columns {
        params.push(format!("c={columns}"));
    }
    if let Some(rows) = options.rows {
        params.push(format!("r={rows}"));
    }
    if let Some(image_id) = options.image_id {
        params.push(format!("i={image_id}"));
    }

    if base64_data.len() <= CHUNK_SIZE {
        return format!("\x1b_G{};{}\x1b\\", params.join(","), base64_data);
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut offset = 0;
    let mut is_first = true;

    while offset < base64_data.len() {
        let chunk = &base64_data[offset..(offset + CHUNK_SIZE).min(base64_data.len())];
        let is_last = offset + CHUNK_SIZE >= base64_data.len();

        if is_first {
            chunks.push(format!("\x1b_G{},m=1;{}\x1b\\", params.join(","), chunk));
            is_first = false;
        } else if is_last {
            chunks.push(format!("\x1b_Gm=0;{chunk}\x1b\\"));
        } else {
            chunks.push(format!("\x1b_Gm=1;{chunk}\x1b\\"));
        }

        offset += CHUNK_SIZE;
    }

    chunks.join("")
}

/// Options for [`encode_kitty`], upstream `encodeKitty`'s options object.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeKittyOptions {
    /// Placement width in columns, upstream `columns`.
    pub columns: Option<usize>,
    /// Placement height in rows, upstream `rows`.
    pub rows: Option<usize>,
    /// The image id to place, upstream `imageId`.
    pub image_id: Option<u64>,
    /// Whether Kitty applies its default cursor movement after the
    /// placement; `Some(false)` emits the `C=1` suppression, upstream
    /// `moveCursor` defaulting to true.
    pub move_cursor: Option<bool>,
}

/// Which inline-image protocol a terminal understands, upstream
/// `ImageProtocol`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProtocol {
    /// Upstream `"kitty"`.
    Kitty,
    /// Upstream `"iterm2"`.
    Iterm2,
}

/// The terminal capability matrix, upstream `TerminalCapabilities`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TerminalCapabilities {
    /// The inline-image protocol, upstream `images` (`null` upstream when
    /// absent).
    pub images: Option<ImageProtocol>,
    /// Upstream `trueColor`.
    pub true_color: bool,
    /// Upstream `hyperlinks`.
    pub hyperlinks: bool,
}

struct CapabilityCache {
    cached: Option<TerminalCapabilities>,
}

/// The cached terminal capabilities, upstream `getCapabilities`.
///
/// Upstream runs the environment detection on an empty cache; the detection
/// is the image ticket's scope (#51), so an empty cache answers the neutral
/// default here.
///
/// # Panics
///
/// Never: a poisoned lock falls back to the pre-poison value, matching the
/// module-global semantics upstream reads.
#[must_use]
pub fn get_capabilities() -> TerminalCapabilities {
    lock_capabilities().cached.unwrap_or_default()
}

fn lock_capabilities() -> std::sync::MutexGuard<'static, CapabilityCache> {
    static CACHE: std::sync::LazyLock<Mutex<CapabilityCache>> =
        std::sync::LazyLock::new(|| Mutex::new(CapabilityCache { cached: None }));
    CACHE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Override the cached capabilities wholesale, upstream `setCapabilities` —
/// a test seam upstream used to exercise both code paths.
///
/// # Panics
///
/// Never: a poisoned lock falls back to writing through it.
pub fn set_capabilities(caps: TerminalCapabilities) {
    *lock_capabilities() = CapabilityCache { cached: Some(caps) };
}

/// Clear the cached capabilities, upstream `resetCapabilitiesCache`.
///
/// # Panics
///
/// Never: a poisoned lock falls back to writing through it.
pub fn reset_capabilities_cache() {
    *lock_capabilities() = CapabilityCache { cached: None };
}

/// Delete every visible Kitty graphics image, freeing the uploaded image
/// data, upstream `deleteAllKittyImages` (`d=A`).
#[must_use]
pub fn delete_all_kitty_images() -> String {
    "\x1b_Ga=d,d=A,q=2\x1b\\".to_string()
}

/// Delete every visible Kitty placement while retaining the uploaded image
/// data, upstream `deleteAllKittyPlacements` (`d=a`).
#[must_use]
pub fn delete_all_kitty_placements() -> String {
    "\x1b_Ga=d,d=a,q=2\x1b\\".to_string()
}

/// Wrap text in an OSC 8 hyperlink sequence, upstream `hyperlink`: the
/// sequences are ignored by terminals without OSC 8 support, leaving the
/// plain text.
#[must_use]
pub fn hyperlink(text: &str, url: &str) -> String {
    format!("\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\")
}

/// The placement metadata [`get_kitty_image_placement`] derives from a
/// rendered image line, upstream `KittyImagePlacement`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KittyImagePlacement {
    /// The placement's image id, upstream `imageId`.
    pub image_id: u64,
    /// The registry generation the placement's metadata was registered
    /// under, upstream `transmissionGeneration`: a placement-only
    /// replacement is valid only against the same generation.
    pub transmission_generation: u64,
    /// The transmission's length in bytes, upstream `transmissionBytes`.
    pub transmission_bytes: usize,
    /// `widthPx * heightPx * 4`, upstream `estimatedDecodedBytes`.
    pub estimated_decoded_bytes: u64,
    /// The placement-only command, upstream `sequence`.
    pub sequence: String,
    /// The image line with the transmission replaced by
    /// [`Self::sequence`], upstream `replacementLine`.
    pub replacement_line: String,
}

/// The placement control keys the placement-only command carries, upstream
/// `KITTY_PLACEMENT_CONTROL_KEYS`.
const KITTY_PLACEMENT_CONTROL_KEYS: [&str; 17] = [
    "i", "p", "x", "y", "w", "h", "X", "Y", "c", "r", "C", "U", "z", "P", "Q", "H", "V",
];

/// Build a placement-only command for an image line a transmission
/// produced, upstream `getKittyImagePlacement`.
///
/// The walk follows the chunked transmission to its terminator, filters the
/// first chunk's controls to the placement keys, and splices the `a=p`
/// command in place of the transmission. The registry must know the image
/// id: placement extraction answers `None` for lines whose metadata was
/// never registered.
#[must_use]
pub fn get_kitty_image_placement(line: &str) -> Option<KittyImagePlacement> {
    let registered = registered_of(line)?;
    let sequence_start = line.find(KITTY_PREFIX)?;
    let first_controls_start = sequence_start + KITTY_PREFIX.len();
    let first_controls_end = first_controls_start + line[first_controls_start..].find(';')?;
    let first_controls = &line[first_controls_start..first_controls_end];

    let mut command_start = sequence_start;
    let mut command_controls = first_controls;
    let transmission_end = loop {
        let window_start = command_start + KITTY_PREFIX.len();
        let terminator = window_start + line[window_start..].find("\x1b\\")?;
        let transmission_end = terminator + 2;
        if !command_controls.split(',').any(|control| control == "m=1") {
            break transmission_end;
        }
        command_start = transmission_end;
        if !line[command_start..].starts_with(KITTY_PREFIX) {
            return None;
        }
        let controls_start = command_start + KITTY_PREFIX.len();
        let controls_end = controls_start + line[controls_start..].find(';')?;
        command_controls = &line[controls_start..controls_end];
    };

    let controls: Vec<&str> = first_controls
        .split(',')
        .filter(|control| {
            let key = control.split('=').next().unwrap_or_default();
            KITTY_PLACEMENT_CONTROL_KEYS.contains(&key)
        })
        .collect();
    let sequence = format!("\x1b_Ga=p,q=2,{}\x1b\\", controls.join(","));
    Some(KittyImagePlacement {
        image_id: registered.metadata.image_id,
        transmission_generation: registered.transmission_generation,
        transmission_bytes: transmission_end - sequence_start,
        estimated_decoded_bytes: registered
            .metadata
            .width_px
            .saturating_mul(registered.metadata.height_px)
            .saturating_mul(4),
        replacement_line: format!(
            "{}{sequence}{}",
            &line[..sequence_start],
            &line[transmission_end..]
        ),
        sequence,
    })
}
