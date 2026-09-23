//! The port of `packages/tui/src/terminal-image.ts` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! ([#51](https://github.com/PhillipChaffee/pi-rust/issues/51)).
//!
//! The pieces landed as their consumers needed them: the process-global
//! cell-dimension store the `CSI 6 ; h ; w t` response feeds,
//! [`is_image_line`], the Kitty placement metadata registry with its crop
//! helper the compositor consults, the capability cache
//! ([`get_capabilities`]/[`set_capabilities`]), the Kitty placement
//! extraction ([`get_kitty_image_placement`]) the placement cache drives,
//! and the batch deletion sequences. This module now also carries the
//! environment capability detection ([`detect_capabilities`]) with its tmux
//! probe and the `PI_*` env overrides, the programmatic capability
//! overrides ([`set_capability_overrides`]), the image-id allocator, the
//! Iterm2 encoder, the cell-size math, the PNG/JPEG/GIF/WebP size parsers,
//! [`render_image`], and [`image_fallback`]; the `Image` component over
//! them is [`crate::components::image::Image`].
//!
//! Restatements: upstream stores cell dimensions, Kitty image metadata, and
//! the capability cache in module globals read across TUI instances; the
//! workspace forbids the `unsafe` a naked `static mut` would need, so the
//! stores sit behind mutexes with the same process-wide visibility and the
//! upstream defaults. Upstream's injectable `tmuxForwardsHyperlink` probe
//! parameter stays a closure; the default probe spawns tmux through
//! `std::process::Command` (`execSync` upstream). The env reads upstream
//! resolves through `process.env` ride the [`crate::terminal::EnvLookup`]
//! seam — Rust cannot mutate the process environment without the `unsafe`
//! this workspace forbids, so suites inject map-backed lookups.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Read;
use std::sync::Mutex;
use std::sync::PoisonError;

use base64::Engine;

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

/// The programmatic capability overrides, upstream `setCapabilityOverrides`'s
/// `Partial<TerminalCapabilities>` argument.
///
/// A field left `None` keeps the detected value, `Some(None)` carries
/// upstream `null` (the capability off), and `Some(Some(..))` forces a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CapabilityOverrides {
    /// Upstream `images` (`undefined` when unset).
    pub images: Option<Option<ImageProtocol>>,
    /// Upstream `trueColor` (`undefined` when unset).
    pub true_color: Option<bool>,
    /// Upstream `hyperlinks` (`undefined` when unset).
    pub hyperlinks: Option<bool>,
}

static CAPABILITY_OVERRIDES: Mutex<CapabilityOverrides> = Mutex::new(CapabilityOverrides {
    images: None,
    true_color: None,
    hyperlinks: None,
});

fn lock_capability_overrides() -> std::sync::MutexGuard<'static, CapabilityOverrides> {
    CAPABILITY_OVERRIDES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Override the selected auto-detected capabilities, upstream
/// `setCapabilityOverrides`: the set replaces the previous one wholesale, and
/// an unchanged set keeps the cached capabilities valid.
///
/// # Panics
///
/// Never: a poisoned lock falls back to writing through it.
pub fn set_capability_overrides(overrides: CapabilityOverrides) {
    let mut current = lock_capability_overrides();
    if *current == overrides {
        return;
    }
    *current = overrides;
    drop(current);
    reset_capabilities_cache();
}

fn apply_capability_overrides(
    detected: TerminalCapabilities,
    overrides: CapabilityOverrides,
) -> TerminalCapabilities {
    TerminalCapabilities {
        images: overrides.images.unwrap_or(detected.images),
        true_color: overrides.true_color.unwrap_or(detected.true_color),
        hyperlinks: overrides.hyperlinks.unwrap_or(detected.hyperlinks),
    }
}

/// The cached terminal capabilities, upstream `getCapabilities`.
///
/// An empty cache runs the environment detection (with the real tmux probe
/// unless a programmatic `hyperlinks` override replaces it) and applies the
/// programmatic overrides.
///
/// # Panics
///
/// Never: a poisoned lock falls back to the pre-poison value, matching the
/// module-global semantics upstream reads.
#[must_use]
pub fn get_capabilities() -> TerminalCapabilities {
    let mut cache = lock_capabilities();
    if let Some(cached) = cache.cached {
        return cached;
    }
    let overrides = *lock_capability_overrides();
    let env = crate::terminal::default_env_lookup();
    let tmux_forwards_hyperlink = || overrides.hyperlinks.unwrap_or_else(probe_tmux_hyperlinks);
    let detected = detect_capabilities(env.as_ref(), &tmux_forwards_hyperlink);
    let caps = apply_capability_overrides(detected, overrides);
    cache.cached = Some(caps);
    caps
}

struct CapabilityCache {
    cached: Option<TerminalCapabilities>,
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

/// How long the tmux probe waits for `tmux display-message` before giving up
/// and answering `false`, upstream `execSync`'s `timeout: 250`.
const TMUX_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);

/// Checks whether the attached tmux client forwards OSC 8 hyperlinks to the
/// outer terminal, upstream `probeTmuxHyperlinks`.
///
/// tmux only re-emits them when its `client_termfeatures` lists
/// `hyperlinks`, and strips them otherwise. Any failure — no tmux binary, no
/// attached client, or the 250 ms deadline — falls back to `false`. The
/// no-argument form runs the real `tmux display-message` command; the
/// `_with` variant is the injection seam suites drive with a stand-in
/// command.
#[must_use]
pub fn probe_tmux_hyperlinks() -> bool {
    let mut command = std::process::Command::new("tmux");
    command.args(["display-message", "-p", "#{client_termfeatures}"]);
    probe_tmux_hyperlinks_with(&mut command)
}

/// [`probe_tmux_hyperlinks`] over a caller-built command.
#[must_use]
pub fn probe_tmux_hyperlinks_with(command: &mut std::process::Command) -> bool {
    let Ok(child) = command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return false;
    };
    let Some(termfeatures) = wait_for_probe_output(child, TMUX_PROBE_TIMEOUT) else {
        return false;
    };
    termfeatures
        .split(',')
        .map(str::trim)
        .any(|feature| feature.contains("hyperlinks"))
}

/// Waits for the probe child within the deadline, upstream `execSync`'s
/// 250 ms timeout: a non-zero exit, a read failure, or a timeout answers
/// `None` (upstream throws into its `catch`).
fn wait_for_probe_output(
    mut child: std::process::Child,
    timeout: std::time::Duration,
) -> Option<String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut stdout = child.stdout.take()?;
                let mut output = String::new();
                stdout.read_to_string(&mut output).ok()?;
                return Some(output);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    child.kill().ok()?;
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(_) => return None,
        }
    }
}

/// The capability matrix the terminal environment answers, upstream
/// `detectCapabilitiesFromEnvironment`: multiplexers come first (tmux strips
/// OSC 8 unless its client forwards, and makes image protocols unreliable),
/// then the per-terminal table, then the conservative default that keeps
/// hyperlinks off so the URL never renders as swallowed escape text.
fn detect_capabilities_from_environment(
    env: &dyn Fn(&str) -> Option<String>,
    tmux_forwards_hyperlink: &dyn Fn() -> bool,
) -> TerminalCapabilities {
    let term_program = env("TERM_PROGRAM")
        .map(|value| value.to_lowercase())
        .unwrap_or_default();
    let terminal_emulator = env("TERMINAL_EMULATOR")
        .map(|value| value.to_lowercase())
        .unwrap_or_default();
    let term = env("TERM")
        .map(|value| value.to_lowercase())
        .unwrap_or_default();
    let color_term = env("COLORTERM")
        .map(|value| value.to_lowercase())
        .unwrap_or_default();
    let has_true_color_hint = color_term == "truecolor" || color_term == "24bit";

    if env("TMUX").is_some_and(|value| !value.is_empty()) || term.starts_with("tmux") {
        return TerminalCapabilities {
            images: None,
            true_color: has_true_color_hint,
            hyperlinks: tmux_forwards_hyperlink(),
        };
    }

    if term.starts_with("screen") {
        return TerminalCapabilities {
            images: None,
            true_color: has_true_color_hint,
            hyperlinks: false,
        };
    }

    let images_true_color_hyperlinks = |images: Option<ImageProtocol>| TerminalCapabilities {
        images,
        true_color: true,
        hyperlinks: true,
    };

    if env("KITTY_WINDOW_ID").is_some_and(|value| !value.is_empty()) || term_program == "kitty" {
        return images_true_color_hyperlinks(Some(ImageProtocol::Kitty));
    }

    if term_program == "ghostty"
        || term.contains("ghostty")
        || env("GHOSTTY_RESOURCES_DIR").is_some()
    {
        return images_true_color_hyperlinks(Some(ImageProtocol::Kitty));
    }

    if env("WEZTERM_PANE").is_some_and(|value| !value.is_empty()) || term_program == "wezterm" {
        return images_true_color_hyperlinks(Some(ImageProtocol::Kitty));
    }

    if term_program == "warpterminal"
        || env("WARP_SESSION_ID").is_some_and(|value| !value.is_empty())
        || env("WARP_TERMINAL_SESSION_UUID").is_some_and(|value| !value.is_empty())
    {
        return images_true_color_hyperlinks(Some(ImageProtocol::Kitty));
    }

    if env("ITERM_SESSION_ID").is_some_and(|value| !value.is_empty()) || term_program == "iterm.app"
    {
        return images_true_color_hyperlinks(Some(ImageProtocol::Iterm2));
    }

    if env("WT_SESSION").is_some_and(|value| !value.is_empty()) {
        return images_true_color_hyperlinks(None);
    }

    if term_program == "alacritty" || term_program == "vscode" || term_program == "zed" {
        return images_true_color_hyperlinks(None);
    }

    if terminal_emulator == "jetbrains-jediterm" {
        return TerminalCapabilities {
            images: None,
            true_color: true,
            hyperlinks: false,
        };
    }

    #[cfg(windows)]
    {
        // Windows Terminal does not always set WT_SESSION, for example when
        // it hosts a cmd.exe launched directly from Win+R. Modern Windows
        // consoles support truecolor; keep hyperlinks off unless a positive
        // detection above already matched.
        return images_true_color_hyperlinks(None);
    }

    TerminalCapabilities {
        images: None,
        true_color: has_true_color_hint,
        hyperlinks: false,
    }
}

fn parse_boolean_capability_override(value: Option<&str>) -> Option<bool> {
    match value {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// The capability matrix for the given environment, upstream
/// `detectCapabilities`.
///
/// The environment detection first, then the `PI_*` env overrides
/// (`PI_HYPERLINKS`, `PI_IMAGE_PROTOCOL`, `PI_TRUE_COLOR` — `1`/`0` force,
/// anything else including `auto` keeps the detected value), with a forced
/// `PI_HYPERLINKS` also standing in for the tmux probe.
///
/// The process-environment default rides [`crate::terminal::EnvLookup`];
/// suites inject a map-backed lookup because Rust cannot mutate the process
/// environment without the `unsafe` this workspace forbids.
#[must_use]
pub fn detect_capabilities(
    env: &dyn Fn(&str) -> Option<String>,
    tmux_forwards_hyperlink: &dyn Fn() -> bool,
) -> TerminalCapabilities {
    let hyperlinks = parse_boolean_capability_override(env("PI_HYPERLINKS").as_deref());
    let probe = || hyperlinks.unwrap_or_else(tmux_forwards_hyperlink);
    let detected = detect_capabilities_from_environment(env, &probe);
    let image_protocol = env("PI_IMAGE_PROTOCOL").map(|value| value.to_lowercase());
    let images = match image_protocol.as_deref() {
        Some("kitty") => Some(Some(ImageProtocol::Kitty)),
        Some("iterm2") => Some(Some(ImageProtocol::Iterm2)),
        Some("none" | "0") => Some(None),
        _ => None,
    };
    let true_color = parse_boolean_capability_override(env("PI_TRUE_COLOR").as_deref());
    TerminalCapabilities {
        images: images.unwrap_or(detected.images),
        true_color: true_color.unwrap_or(detected.true_color),
        hyperlinks: hyperlinks.unwrap_or(detected.hyperlinks),
    }
}

/// Generate an image id for Kitty graphics placements, upstream
/// `allocateImageId`: random ids in `[1, 0xffff_fffe]` avoid collisions
/// between module instances (main app vs extensions).
///
/// Upstream draws from `Math.random()`; the port drives a process-global
/// xorshift64* state seeded from the clock and the pid, which no test pins.
#[must_use]
pub fn allocate_image_id() -> u64 {
    const ID_SPAN: u64 = 0xffff_fffe;
    static STATE: std::sync::LazyLock<Mutex<u64>> = std::sync::LazyLock::new(|| Mutex::new(0));
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    if *state == 0 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |duration| {
                u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
            });
        *state = nanos ^ (u64::from(std::process::id()) << 32) ^ 0x9e37_79b9_7f4a_7c15;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    drop(state);
    (x % ID_SPAN) + 1
}

/// Options for [`encode_iterm2`], upstream `encodeITerm2`'s options object.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EncodeITerm2Options {
    /// Upstream `width` — a cell count or `"auto"`.
    pub width: Option<ITerm2Size>,
    /// Upstream `height` — a cell count or `"auto"`.
    pub height: Option<ITerm2Size>,
    /// Upstream `name`, base64-encoded into the command.
    pub name: Option<String>,
    /// Upstream `preserveAspectRatio`: only `Some(false)` emits the
    /// `preserveAspectRatio=0` flag, upstream `=== false`.
    pub preserve_aspect_ratio: Option<bool>,
    /// Upstream `inline`: `Some(false)` emits `inline=0`, everything else
    /// (including unset) `inline=1`, upstream `inline !== false`.
    pub inline: Option<bool>,
}

/// An Iterm2 width or height value, upstream `number | string` where the
/// string spelling is `"auto"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ITerm2Size {
    /// Upstream `"auto"`.
    Auto,
    /// Upstream a numeric cell count.
    Cells(usize),
}

/// Encode an Iterm2 (OSC 1337) inline-image transmission, upstream
/// `encodeITerm2`; `size` carries the decoded payload byte length.
#[must_use]
pub fn encode_iterm2(base64_data: &str, options: EncodeITerm2Options) -> String {
    let mut params: Vec<String> = vec![
        format!("inline={}", u8::from(options.inline != Some(false))),
        format!("size={}", iterm2_decoded_size(base64_data)),
    ];

    if let Some(width) = options.width {
        params.push(format!("width={}", iterm2_size_param(width)));
    }
    if let Some(height) = options.height {
        params.push(format!("height={}", iterm2_size_param(height)));
    }
    if let Some(name) = options.name {
        let engine = base64::engine::general_purpose::STANDARD;
        let name_base64 = engine.encode(name.as_bytes());
        params.push(format!("name={name_base64}"));
    }
    if options.preserve_aspect_ratio == Some(false) {
        params.push("preserveAspectRatio=0".to_string());
    }

    format!("\x1b]1337;File={}:{base64_data}\x07", params.join(";"))
}

fn iterm2_size_param(size: ITerm2Size) -> String {
    match size {
        ITerm2Size::Auto => "auto".to_string(),
        ITerm2Size::Cells(cells) => cells.to_string(),
    }
}

/// The decoded byte length of a base64 payload, upstream
/// `Buffer.byteLength(base64Data, "base64")`: `len * 3 / 4` minus the
/// trailing `=` padding.
fn iterm2_decoded_size(base64_data: &str) -> usize {
    let len = base64_data.chars().count();
    let padding = base64_data
        .bytes()
        .rev()
        .take_while(|byte| *byte == b'=')
        .count();
    (len * 3 / 4).saturating_sub(padding)
}

/// The cell footprint of a rendered image, upstream `ImageCellSize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageCellSize {
    /// Placement width in columns.
    pub columns: usize,
    /// Placement height in rows.
    pub rows: usize,
}

/// Source-image pixel size, upstream `ImageDimensions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageDimensions {
    /// Upstream `widthPx`.
    pub width_px: u64,
    /// Upstream `heightPx`.
    pub height_px: u64,
}

/// Options for [`render_image`], upstream `ImageRenderOptions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageRenderOptions {
    /// Upstream `maxWidthCells`, defaulting to 80.
    pub max_width_cells: Option<usize>,
    /// Upstream `maxHeightCells`.
    pub max_height_cells: Option<usize>,
    /// Upstream `preserveAspectRatio`, defaulting to true.
    pub preserve_aspect_ratio: Option<bool>,
    /// Upstream `imageId`: when set, reuses/replaces the existing image with
    /// this id.
    pub image_id: Option<u64>,
    /// Upstream `moveCursor`: whether Kitty applies its default cursor
    /// movement after the placement, defaulting to true.
    pub move_cursor: Option<bool>,
}

/// A rendered inline-image transmission, upstream `renderImage`'s return
/// value (`imageId` carries a value only on the Kitty branch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedImage {
    /// The escape sequence to write.
    pub sequence: String,
    /// The placement width in columns.
    pub columns: usize,
    /// The placement height in rows.
    pub rows: usize,
    /// The Kitty image id, upstream `imageId` (`undefined` on Iterm2).
    pub image_id: Option<u64>,
}

/// Scale an image into cell-sized Kitty or Iterm2 placements, upstream
/// `renderImage`.
///
/// Answers `None` when the terminal renders no inline-image protocol. The
/// Kitty branch registers the metadata of an explicit `image_id` so
/// placement extraction and cropping can find it.
#[must_use]
pub fn render_image(
    base64_data: &str,
    image_dimensions: ImageDimensions,
    options: &ImageRenderOptions,
) -> Option<RenderedImage> {
    let caps = get_capabilities();

    let protocol = caps.images?;

    let max_width_cells = options.max_width_cells.unwrap_or(80);
    let size = calculate_image_cell_size(
        image_dimensions,
        max_width_cells,
        options.max_height_cells,
        get_cell_dimensions(),
    );

    match protocol {
        ImageProtocol::Kitty => {
            if let Some(image_id) = options.image_id {
                register_kitty_image_metadata(KittyImageMetadata {
                    image_id,
                    columns: size.columns,
                    rows: size.rows,
                    width_px: image_dimensions.width_px,
                    height_px: image_dimensions.height_px,
                });
            }
            let sequence = encode_kitty(
                base64_data,
                EncodeKittyOptions {
                    columns: Some(size.columns),
                    rows: Some(size.rows),
                    image_id: options.image_id,
                    move_cursor: options.move_cursor,
                },
            );
            Some(RenderedImage {
                sequence,
                columns: size.columns,
                rows: size.rows,
                image_id: options.image_id,
            })
        }
        ImageProtocol::Iterm2 => {
            let sequence = encode_iterm2(
                base64_data,
                EncodeITerm2Options {
                    width: Some(ITerm2Size::Cells(size.columns)),
                    height: Some(ITerm2Size::Auto),
                    preserve_aspect_ratio: Some(options.preserve_aspect_ratio.unwrap_or(true)),
                    ..EncodeITerm2Options::default()
                },
            );
            Some(RenderedImage {
                sequence,
                columns: size.columns,
                rows: size.rows,
                image_id: None,
            })
        }
    }
}

/// Compute the cell footprint of an image placement, upstream
/// `calculateImageCellSize`.
///
/// The image scales uniformly to fit `max_width_cells` (and
/// `max_height_cells` when set) against the physical cell size, and the
/// clamped ceiling of the scaled extent is the answer.
#[must_use]
pub fn calculate_image_cell_size(
    image_dimensions: ImageDimensions,
    max_width_cells: usize,
    max_height_cells: Option<usize>,
    cell_dimensions: CellDimensions,
) -> ImageCellSize {
    let max_width = max_width_cells.max(1);
    let max_height = max_height_cells.map(|height| height.max(1));
    let image_width =
        f64::from(u32::try_from(image_dimensions.width_px.max(1)).unwrap_or(u32::MAX));
    let image_height =
        f64::from(u32::try_from(image_dimensions.height_px.max(1)).unwrap_or(u32::MAX));
    let cell_width = f64::from(cell_dimensions.width_px);
    let cell_height = f64::from(cell_dimensions.height_px);

    let width_scale =
        f64::from(u32::try_from(max_width).unwrap_or(u32::MAX)) * cell_width / image_width;
    let height_scale = max_height.map_or(width_scale, |max_height| {
        f64::from(u32::try_from(max_height).unwrap_or(u32::MAX)) * cell_height / image_height
    });
    let scale = width_scale.min(height_scale);

    let columns = (image_width * scale / cell_width).ceil();
    let rows = (image_height * scale / cell_height).ceil();

    // The ceilings of positive scale quotients are small positive integers;
    // upstream computes the same conversions on JS Numbers.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the ceilings of positive scale quotients are small positive integers"
    )]
    let columns = columns as usize;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the ceilings of positive scale quotients are small positive integers"
    )]
    let rows = rows as usize;

    ImageCellSize {
        columns: columns.clamp(1, max_width),
        rows: max_height.map_or_else(|| rows.max(1), |max_height| rows.min(max_height).max(1)),
    }
}

/// The row count of an image scaled to `target_width_cells` columns, upstream
/// `calculateImageRows`.
#[must_use]
pub fn calculate_image_rows(
    image_dimensions: ImageDimensions,
    target_width_cells: usize,
    cell_dimensions: CellDimensions,
) -> usize {
    calculate_image_cell_size(image_dimensions, target_width_cells, None, cell_dimensions).rows
}

/// The decoded bytes of a base64 payload, upstream `Buffer.from(base64Data,
/// "base64")`: non-alphabet characters drop, and a decode failure answers an
/// empty buffer the same way malformed payloads do downstream.
fn decode_image_bytes(base64_data: &str) -> Vec<u8> {
    let filtered: String = base64_data
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '+' || *c == '/' || *c == '=')
        .collect();
    base64::engine::general_purpose::STANDARD
        .decode(filtered.as_bytes())
        .unwrap_or_default()
}

/// The PNG header dimensions, upstream `getPngDimensions`: the IHDR width
/// and height big-endian at offsets 16/20.
#[must_use]
pub fn get_png_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_image_bytes(base64_data);
    if buffer.len() < 24 {
        return None;
    }
    if buffer[0] != 0x89 || buffer[1] != 0x50 || buffer[2] != 0x4e || buffer[3] != 0x47 {
        return None;
    }
    Some(ImageDimensions {
        width_px: u64::from(u32::from_be_bytes([
            buffer[16], buffer[17], buffer[18], buffer[19],
        ])),
        height_px: u64::from(u32::from_be_bytes([
            buffer[20], buffer[21], buffer[22], buffer[23],
        ])),
    })
}

/// The JPEG SOF0-SOF2 marker dimensions, upstream `getJpegDimensions`: the
/// walk skips non-marker bytes and stepped segments until a frame-header
/// marker answers, truncation answers `None`.
#[must_use]
pub fn get_jpeg_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_image_bytes(base64_data);
    if buffer.len() < 2 {
        return None;
    }
    if buffer[0] != 0xff || buffer[1] != 0xd8 {
        return None;
    }
    let mut offset = 2;
    while offset + 9 < buffer.len() {
        if buffer[offset] != 0xff {
            offset += 1;
            continue;
        }
        let marker = buffer[offset + 1];
        if (0xc0..=0xc2).contains(&marker) {
            return Some(ImageDimensions {
                width_px: u64::from(u16::from_be_bytes([buffer[offset + 7], buffer[offset + 8]])),
                height_px: u64::from(u16::from_be_bytes([buffer[offset + 5], buffer[offset + 6]])),
            });
        }
        if offset + 3 >= buffer.len() {
            return None;
        }
        let length = usize::from(u16::from_be_bytes([buffer[offset + 2], buffer[offset + 3]]));
        if length < 2 {
            return None;
        }
        offset += 2 + length;
    }
    None
}

/// The GIF logical screen dimensions, upstream `getGifDimensions`: little-
/// endian u16 at offsets 6 and 8 behind the `GIF87a`/`GIF89a` signature.
#[must_use]
pub fn get_gif_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_image_bytes(base64_data);
    if buffer.len() < 10 {
        return None;
    }
    if buffer[..6] != *b"GIF87a" && buffer[..6] != *b"GIF89a" {
        return None;
    }
    Some(ImageDimensions {
        width_px: u64::from(u16::from_le_bytes([buffer[6], buffer[7]])),
        height_px: u64::from(u16::from_le_bytes([buffer[8], buffer[9]])),
    })
}

/// The WebP VP8/VP8L/VP8X chunk dimensions, upstream `getWebpDimensions`:
/// the RIFF/WEBP headers gate first, then the chunk tag picks the parse.
#[must_use]
pub fn get_webp_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_image_bytes(base64_data);
    if buffer.len() < 30 {
        return None;
    }
    if buffer[..4] != *b"RIFF" || buffer[8..12] != *b"WEBP" {
        return None;
    }
    match &buffer[12..16] {
        b"VP8 " => Some(ImageDimensions {
            width_px: u64::from(u16::from_le_bytes([buffer[26], buffer[27]]) & 0x3fff),
            height_px: u64::from(u16::from_le_bytes([buffer[28], buffer[29]]) & 0x3fff),
        }),
        b"VP8L" => {
            let bits = u32::from_le_bytes([buffer[21], buffer[22], buffer[23], buffer[24]]);
            Some(ImageDimensions {
                width_px: u64::from(bits & 0x3fff) + 1,
                height_px: u64::from((bits >> 14) & 0x3fff) + 1,
            })
        }
        b"VP8X" => Some(ImageDimensions {
            width_px: u64::from(
                u32::from(buffer[24])
                    | (u32::from(buffer[25]) << 8)
                    | (u32::from(buffer[26]) << 16),
            ) + 1,
            height_px: u64::from(
                u32::from(buffer[27])
                    | (u32::from(buffer[28]) << 8)
                    | (u32::from(buffer[29]) << 16),
            ) + 1,
        }),
        _ => None,
    }
}

/// Dispatch an image payload to its format parser by MIME type, upstream
/// `getImageDimensions`.
#[must_use]
pub fn get_image_dimensions(base64_data: &str, mime_type: &str) -> Option<ImageDimensions> {
    match mime_type {
        "image/png" => get_png_dimensions(base64_data),
        "image/jpeg" => get_jpeg_dimensions(base64_data),
        "image/gif" => get_gif_dimensions(base64_data),
        "image/webp" => get_webp_dimensions(base64_data),
        _ => None,
    }
}

/// Shorten home-prefixed absolute paths to `~/...` for compact display,
/// upstream `shortenImagePath`.
fn shorten_image_path(filename: &str) -> String {
    let Some(home) = std::env::home_dir().and_then(|home| home.into_os_string().into_string().ok())
    else {
        return filename.to_string();
    };
    if !home.is_empty()
        && (filename == home
            || filename.starts_with(&format!("{home}/"))
            || filename.starts_with(&format!("{home}\\")))
    {
        return format!("~{}", &filename[home.len()..]);
    }
    filename.to_string()
}

/// The `file://` URL of an absolute path, upstream `pathToFileURL(filename)
/// .href`: each byte outside the URL unreserved set percent-encodes, path
/// separators stay.
fn file_url(path: &str) -> String {
    const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut url = String::from("file://");
    for byte in path.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                url.push(byte as char);
            }
            _ => {
                url.push('%');
                url.push(HEX_DIGITS[usize::from(byte >> 4)] as char);
                url.push(HEX_DIGITS[usize::from(byte & 0x0f)] as char);
            }
        }
    }
    url
}

/// Text fallback when the terminal cannot render inline images, upstream
/// `imageFallback`.
///
/// Absolute paths show shortened (`~/...`) and, when OSC 8 hyperlinks are
/// available, linked to `file://` so the full path remains openable.
#[must_use]
pub fn image_fallback(
    mime_type: &str,
    dimensions: Option<ImageDimensions>,
    filename: Option<&str>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(filename) = filename {
        let display = shorten_image_path(filename);
        if get_capabilities().hyperlinks && std::path::Path::new(filename).is_absolute() {
            parts.push(hyperlink(&display, &file_url(filename)));
        } else {
            parts.push(display);
        }
    }
    parts.push(format!("[{mime_type}]"));
    if let Some(dimensions) = dimensions {
        parts.push(format!("{}x{}", dimensions.width_px, dimensions.height_px));
    }
    format!("[Image: {}]", parts.join(" "))
}
