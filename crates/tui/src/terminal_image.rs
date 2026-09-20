//! The cell-dimension store and the image-line probe of
//! `packages/tui/src/terminal-image.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Only the pieces the TUI core and the layout engine touch land here
//! ([#43](https://github.com/PhillipChaffee/pi-rust/issues/43),
//! [#44](https://github.com/PhillipChaffee/pi-rust/issues/44)): the
//! process-global cell-dimension store the `CSI 6 ; h ; w t` response
//! feeds, [`is_image_line`], and the Kitty placement metadata registry
//! with its crop helper the compositor consults. The rest of the file —
//! capability probes, image rendering, placement extraction — is the
//! image ticket's scope ([#51](https://github.com/PhillipChaffee/pi-rust/issues/51)) and lands there.
//!
//! Restatement: upstream stores cell dimensions and Kitty image metadata
//! in module globals read across TUI instances; the workspace forbids the
//! `unsafe` a naked `static mut` would need, so the stores sit behind
//! mutexes with the same process-wide visibility and the upstream
//! defaults.

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
    #[expect(
        dead_code,
        reason = "the transmission generation rides for #46's placement extraction"
    )]
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
    let image_id = controls_of(line).and_then(image_id_of)?;
    let registry = lock_registry();
    registry
        .by_id
        .get(&image_id)
        .map(|registered| registered.metadata)
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
