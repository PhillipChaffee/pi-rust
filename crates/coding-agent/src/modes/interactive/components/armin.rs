//! Armin says hi: the animated XBM easter egg, upstream
//! `src/modes/interactive/components/armin.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements: upstream's `TUI` handle narrows to the [`RenderRequest`]
//! closure the animation calls; `setInterval` becomes a worker thread whose
//! stop channel is the interval guard, the pattern [`pi_tui::components::Loader`]
//! already uses. [`ArminComponent::tick`] exposes the interval body so tests
//! drive the animation inline instead of sleeping, and
//! [`ArminComponent::with_effect`] pins a deterministic effect where upstream
//! picks one at random.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use pi_tui::tui::{Component, RenderRequest};
use rand::Rng as _;
use rand::seq::SliceRandom as _;

use crate::modes::interactive::theme::theme::theme;

// XBM image: 31x36 pixels, LSB first, 1=background, 0=foreground
const WIDTH: usize = 31;
const HEIGHT: usize = 36;
const BITS: [u8; 144] = [
    0xff, 0xff, 0xff, 0x7f, 0xff, 0xf0, 0xff, 0x7f, 0xff, 0xed, 0xff, 0x7f, 0xff, 0xdb, 0xff, 0x7f,
    0xff, 0xb7, 0xff, 0x7f, 0xff, 0x77, 0xfe, 0x7f, 0x3f, 0xf8, 0xfe, 0x7f, 0xdf, 0xff, 0xfe, 0x7f,
    0xdf, 0x3f, 0xfc, 0x7f, 0x9f, 0xc3, 0xfb, 0x7f, 0x6f, 0xfc, 0xf4, 0x7f, 0xf7, 0x0f, 0xf7, 0x7f,
    0xf7, 0xff, 0xf7, 0x7f, 0xf7, 0xff, 0xe3, 0x7f, 0xf7, 0x07, 0xe8, 0x7f, 0xef, 0xf8, 0x67, 0x70,
    0x0f, 0xff, 0xbb, 0x6f, 0xf1, 0x00, 0xd0, 0x5b, 0xfd, 0x3f, 0xec, 0x53, 0xc1, 0xff, 0xef, 0x57,
    0x9f, 0xfd, 0xee, 0x5f, 0x9f, 0xfc, 0xae, 0x5f, 0x1f, 0x78, 0xac, 0x5f, 0x3f, 0x00, 0x50, 0x6c,
    0x7f, 0x00, 0xdc, 0x77, 0xff, 0xc0, 0x3f, 0x78, 0xff, 0x01, 0xf8, 0x7f, 0xff, 0x03, 0x9c, 0x78,
    0xff, 0x07, 0x8c, 0x7c, 0xff, 0x0f, 0xce, 0x78, 0xff, 0xff, 0xcf, 0x7f, 0xff, 0xff, 0xcf, 0x78,
    0xff, 0xff, 0xdf, 0x78, 0xff, 0xff, 0xdf, 0x7d, 0xff, 0xff, 0x3f, 0x7e, 0xff, 0xff, 0xff, 0x7f,
];

const BYTES_PER_ROW: usize = WIDTH.div_ceil(8);
/// Half-block rendering packs two vertical pixels into one cell.
const DISPLAY_HEIGHT: usize = HEIGHT.div_ceil(2);

const EFFECTS: [ArminEffect; 7] = [
    ArminEffect::Typewriter,
    ArminEffect::Scanline,
    ArminEffect::Rain,
    ArminEffect::Fade,
    ArminEffect::Crt,
    ArminEffect::Glitch,
    ArminEffect::Dissolve,
];

/// The reveal animation, upstream's `Effect`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArminEffect {
    /// Paint three cells per frame in row-major order.
    Typewriter,
    /// Copy one row per frame from top to bottom.
    Scanline,
    /// Fall the image in per column, settling from the bottom up.
    Rain,
    /// Resolve randomized cells fifteen at a time.
    Fade,
    /// Expand from the middle row outward.
    Crt,
    /// Corrupt rows for eight frames, then show the clean image.
    Glitch,
    /// Start from random noise and resolve twenty cells at a time.
    Dissolve,
}

/// One rain column's falling state, upstream's `{ y, settled }`.
#[derive(Clone, Copy, Debug)]
struct Drop {
    /// The drop's current row; negative while above the image.
    y: isize,
    /// How many bottom rows of the column have settled.
    settled: usize,
}

/// The per-effect scratch state, upstream's `effectState`.
enum EffectState {
    Typewriter {
        pos: usize,
    },
    Scanline {
        row: usize,
    },
    Rain {
        drops: Vec<Drop>,
    },
    Fade {
        positions: Vec<(usize, usize)>,
        idx: usize,
    },
    Crt {
        expansion: usize,
    },
    Glitch {
        phase: usize,
        glitch_frames: usize,
    },
    Dissolve {
        positions: Vec<(usize, usize)>,
        idx: usize,
    },
}

/// The mutable grid plus the effect scratch state.
struct GridState {
    current: Vec<Vec<char>>,
    effect_state: EffectState,
}

/// The width- and version-keyed render cache, upstream's `cachedLines`,
/// `cachedWidth`, and `cachedVersion`.
#[derive(Default)]
struct RenderCache {
    width: Option<usize>,
    version: u64,
    lines: Vec<String>,
}

/// The state the animation thread and the render call share.
struct ArminInner {
    effect: ArminEffect,
    final_grid: Vec<Vec<char>>,
    grid: Mutex<GridState>,
    cache: Mutex<RenderCache>,
    /// Upstream's `gridVersion`.
    version: AtomicU64,
    render_request: RenderRequest,
}

/// The spawned animation worker; dropping it disconnects the stop channel.
struct Animation {
    /// Held purely so its drop disconnects the worker's stop channel.
    #[expect(
        dead_code,
        reason = "the sender's drop is the stop signal; nothing reads it"
    )]
    stop: std::sync::mpsc::Sender<()>,
}

/// Armin says hi! A fun easter egg with animated XBM art.
pub struct ArminComponent {
    inner: Arc<ArminInner>,
    animation: Mutex<Option<Animation>>,
}

impl std::fmt::Debug for ArminComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArminComponent")
            .field("effect", &self.inner.effect)
            .finish_non_exhaustive()
    }
}

impl ArminComponent {
    /// Build and start the animation with a random effect, upstream's
    /// constructor.
    #[must_use]
    pub fn new(render_request: RenderRequest) -> Self {
        let effect = EFFECTS[rand::rng().random_range(0..EFFECTS.len())];
        Self::with_effect(effect, render_request)
    }

    /// Build and start the animation with a pinned effect; upstream chooses
    /// one at random, so this is the test seam.
    #[must_use]
    pub fn with_effect(effect: ArminEffect, render_request: RenderRequest) -> Self {
        let final_grid = build_final_grid();
        let mut grid = GridState {
            current: create_empty_grid(),
            effect_state: EffectState::Typewriter { pos: 0 },
        };
        init_effect(effect, &mut grid);
        let inner = Arc::new(ArminInner {
            effect,
            final_grid,
            grid: Mutex::new(grid),
            cache: Mutex::new(RenderCache::default()),
            version: AtomicU64::new(0),
            render_request,
        });
        let component = Self {
            inner,
            animation: Mutex::new(None),
        };
        component.start_animation();
        component
    }

    /// Elapse one animation frame, upstream's interval body: tick the effect,
    /// bump the display version, and request a render. Answers whether the
    /// effect has finished.
    pub fn tick(&self) -> bool {
        self.inner.tick()
    }

    /// Stop the animation, upstream `dispose`.
    pub fn dispose(&self) {
        self.animation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }

    fn start_animation(&self) {
        let fps = if self.inner.effect == ArminEffect::Glitch {
            60
        } else {
            30
        };
        let interval = Duration::from_secs_f64(1.0 / f64::from(fps));
        let inner = Arc::clone(&self.inner);
        let (stop, stop_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            loop {
                match stop_rx.recv_timeout(interval) {
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if inner.tick() {
                            break;
                        }
                    }
                }
            }
        });
        *self
            .animation
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Animation { stop });
    }
}

impl ArminInner {
    fn lock_grid(&self) -> MutexGuard<'_, GridState> {
        self.grid.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn tick(&self) -> bool {
        let done = {
            let mut grid = self.lock_grid();
            tick_effect(self.effect, &self.final_grid, &mut grid, &mut rand::rng())
        };
        self.version.fetch_add(1, Ordering::Relaxed);
        (self.render_request)();
        done
    }
}

impl Component for ArminComponent {
    fn render(&self, width: usize) -> Vec<String> {
        let version = self.inner.version.load(Ordering::Relaxed);
        {
            let cache = self
                .inner
                .cache
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if cache.width == Some(width) && cache.version == version {
                return cache.lines.clone();
            }
        }

        let padding = 1;
        let available_width = width.saturating_sub(padding);
        let active = theme();
        let mut lines = Vec::with_capacity(DISPLAY_HEIGHT + 1);
        {
            let grid = self.inner.lock_grid();
            for row in &grid.current {
                let clipped: String = row.iter().take(available_width).collect();
                let clipped_width = row.len().min(available_width);
                let pad_right = width.saturating_sub(padding + clipped_width);
                lines.push(format!(
                    " {}{}",
                    active.fg("accent", &clipped),
                    " ".repeat(pad_right)
                ));
            }
        }

        let message = "ARMIN SAYS HI";
        let message_pad_right = width.saturating_sub(padding + message.len());
        lines.push(format!(
            " {}{}",
            active.fg("accent", message),
            " ".repeat(message_pad_right)
        ));

        let mut cache = self
            .inner
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        cache.width = Some(width);
        cache.version = version;
        cache.lines.clone_from(&lines);
        lines
    }

    fn invalidate(&self) {
        self.inner
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .width = None;
    }
}

/// Whether the XBM bit at `(x, y)` is foreground, upstream's `getPixel`.
const fn get_pixel(x: usize, y: usize) -> bool {
    if y >= HEIGHT {
        return false;
    }
    let byte_index = y * BYTES_PER_ROW + x / 8;
    let bit_index = x % 8;
    (BITS[byte_index] >> bit_index) & 1 == 0
}

/// The half-block cell for two stacked pixels, upstream's `getChar`.
const fn get_char(x: usize, row: usize) -> char {
    let upper = get_pixel(x, row * 2);
    let lower = get_pixel(x, row * 2 + 1);
    match (upper, lower) {
        (true, true) => '█',
        (true, false) => '▀',
        (false, true) => '▄',
        (false, false) => ' ',
    }
}

fn build_final_grid() -> Vec<Vec<char>> {
    (0..DISPLAY_HEIGHT)
        .map(|row| (0..WIDTH).map(|x| get_char(x, row)).collect())
        .collect()
}

fn create_empty_grid() -> Vec<Vec<char>> {
    vec![vec![' '; WIDTH]; DISPLAY_HEIGHT]
}

/// Shuffle every cell position with Fisher-Yates, upstream's inline shuffle.
fn shuffled_positions(rng: &mut impl rand::Rng) -> Vec<(usize, usize)> {
    let mut positions: Vec<(usize, usize)> = (0..DISPLAY_HEIGHT)
        .flat_map(|row| (0..WIDTH).map(move |x| (row, x)))
        .collect();
    positions.shuffle(rng);
    positions
}

fn init_effect(effect: ArminEffect, grid: &mut GridState) {
    let mut rng = rand::rng();
    grid.effect_state = match effect {
        ArminEffect::Typewriter => EffectState::Typewriter { pos: 0 },
        ArminEffect::Scanline => EffectState::Scanline { row: 0 },
        ArminEffect::Rain => EffectState::Rain {
            drops: (0..WIDTH)
                .map(|_| Drop {
                    y: -isize::try_from(rng.random_range(0..DISPLAY_HEIGHT * 2))
                        .unwrap_or(isize::MAX),
                    settled: 0,
                })
                .collect(),
        },
        ArminEffect::Fade => EffectState::Fade {
            positions: shuffled_positions(&mut rng),
            idx: 0,
        },
        ArminEffect::Crt => EffectState::Crt { expansion: 0 },
        ArminEffect::Glitch => EffectState::Glitch {
            phase: 0,
            glitch_frames: 8,
        },
        ArminEffect::Dissolve => {
            let noise = [' ', '░', '▒', '▓', '█', '▀', '▄'];
            for row in &mut grid.current {
                for cell in row {
                    *cell = noise[rng.random_range(0..noise.len())];
                }
            }
            EffectState::Dissolve {
                positions: shuffled_positions(&mut rng),
                idx: 0,
            }
        }
    };
}

fn tick_effect(
    effect: ArminEffect,
    final_grid: &[Vec<char>],
    grid: &mut GridState,
    rng: &mut impl rand::Rng,
) -> bool {
    match effect {
        ArminEffect::Typewriter => tick_typewriter(final_grid, grid),
        ArminEffect::Scanline => tick_scanline(final_grid, grid),
        ArminEffect::Rain => tick_rain(final_grid, grid, rng),
        ArminEffect::Fade => tick_fade(final_grid, grid),
        ArminEffect::Crt => tick_crt(final_grid, grid),
        ArminEffect::Glitch => tick_glitch(final_grid, grid, rng),
        ArminEffect::Dissolve => tick_dissolve(final_grid, grid),
    }
}

fn tick_typewriter(final_grid: &[Vec<char>], grid: &mut GridState) -> bool {
    let EffectState::Typewriter { pos } = &mut grid.effect_state else {
        return true;
    };
    for _ in 0..3 {
        let row = *pos / WIDTH;
        let x = *pos % WIDTH;
        if row >= DISPLAY_HEIGHT {
            return true;
        }
        grid.current[row][x] = final_grid[row][x];
        *pos += 1;
    }
    false
}

fn tick_scanline(final_grid: &[Vec<char>], grid: &mut GridState) -> bool {
    let EffectState::Scanline { row } = &mut grid.effect_state else {
        return true;
    };
    if *row >= DISPLAY_HEIGHT {
        return true;
    }
    grid.current[*row].clone_from_slice(&final_grid[*row]);
    *row += 1;
    false
}

fn tick_rain(final_grid: &[Vec<char>], grid: &mut GridState, rng: &mut impl rand::Rng) -> bool {
    let EffectState::Rain { drops } = &mut grid.effect_state else {
        return true;
    };
    let mut all_settled = true;
    for row in &mut grid.current {
        row.fill(' ');
    }

    for (x, drop) in drops.iter_mut().enumerate() {
        if drop.settled > 0 {
            let start = DISPLAY_HEIGHT.saturating_sub(drop.settled);
            for (row, source) in final_grid.iter().enumerate().skip(start) {
                grid.current[row][x] = source[x];
            }
        }

        if drop.settled >= DISPLAY_HEIGHT {
            continue;
        }
        all_settled = false;

        let target_row = (0..DISPLAY_HEIGHT - drop.settled)
            .rev()
            .find(|row| final_grid[*row][x] != ' ');

        drop.y += 1;

        if let Ok(y) = usize::try_from(drop.y)
            && y < DISPLAY_HEIGHT
        {
            if let Some(target_row) = target_row
                && drop.y >= isize::try_from(target_row).unwrap_or(isize::MAX)
            {
                drop.settled = DISPLAY_HEIGHT - target_row;
                drop.y = -isize::try_from(rng.random_range(0..5)).unwrap_or_default() - 1;
            } else {
                grid.current[y][x] = '▓';
            }
        }
    }

    all_settled
}

fn tick_fade(final_grid: &[Vec<char>], grid: &mut GridState) -> bool {
    let EffectState::Fade { positions, idx } = &mut grid.effect_state else {
        return true;
    };
    for _ in 0..15 {
        if *idx >= positions.len() {
            return true;
        }
        let (row, x) = positions[*idx];
        grid.current[row][x] = final_grid[row][x];
        *idx += 1;
    }
    false
}

fn tick_crt(final_grid: &[Vec<char>], grid: &mut GridState) -> bool {
    let EffectState::Crt { expansion } = &mut grid.effect_state else {
        return true;
    };
    let mid_row = DISPLAY_HEIGHT / 2;
    for row in &mut grid.current {
        row.fill(' ');
    }

    let top = mid_row.cast_signed() - (*expansion).cast_signed();
    let bottom = mid_row.cast_signed() + (*expansion).cast_signed();
    let start = top.max(0).cast_unsigned();
    let end = bottom.min(DISPLAY_HEIGHT.cast_signed() - 1);
    if end >= 0 {
        for (row, source) in final_grid
            .iter()
            .enumerate()
            .take(end.cast_unsigned() + 1)
            .skip(start)
        {
            grid.current[row].clone_from_slice(source);
        }
    }

    *expansion += 1;
    *expansion > DISPLAY_HEIGHT
}

fn tick_glitch(final_grid: &[Vec<char>], grid: &mut GridState, rng: &mut impl rand::Rng) -> bool {
    let EffectState::Glitch {
        phase,
        glitch_frames,
    } = &mut grid.effect_state
    else {
        return true;
    };

    if *phase < *glitch_frames {
        grid.current = final_grid
            .iter()
            .map(|row| {
                let offset = isize::try_from(rng.random_range(-3_i64..=3)).unwrap_or_default();
                if rng.random_bool(0.3) {
                    return shift_row(row, offset);
                }
                if rng.random_bool(0.2) {
                    let swap_row = rng.random_range(0..DISPLAY_HEIGHT);
                    return final_grid[swap_row].clone();
                }
                row.clone()
            })
            .collect();
        *phase += 1;
        return false;
    }

    grid.current = final_grid.to_vec();
    true
}

/// Emulate `row.slice(offset).concat(row.slice(0, offset))` for a signed
/// offset, including JavaScript's negative-index form.
fn shift_row(row: &[char], offset: isize) -> Vec<char> {
    let len = row.len().cast_signed();
    let mut shifted = Vec::with_capacity(row.len());
    if offset >= 0 && offset < len {
        shifted.extend_from_slice(&row[offset.cast_unsigned()..]);
        shifted.extend_from_slice(&row[..offset.cast_unsigned()]);
    } else if offset < 0 && offset > -len {
        let split = (len + offset).cast_unsigned();
        shifted.extend_from_slice(&row[split..]);
        shifted.extend_from_slice(&row[..split]);
    } else {
        shifted.extend_from_slice(row);
    }
    shifted.truncate(WIDTH);
    shifted
}

fn tick_dissolve(final_grid: &[Vec<char>], grid: &mut GridState) -> bool {
    let EffectState::Dissolve { positions, idx } = &mut grid.effect_state else {
        return true;
    };
    for _ in 0..20 {
        if *idx >= positions.len() {
            return true;
        }
        let (row, x) = positions[*idx];
        grid.current[row][x] = final_grid[row][x];
        *idx += 1;
    }
    false
}
