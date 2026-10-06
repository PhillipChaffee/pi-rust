//! Clipboard image reads across the platform backends, upstream's
//! `src/utils/clipboard-image.ts`.
//!
//! The command backends (wl-paste, xclip, and the WSL PowerShell bridge)
//! drive the same runner [`super::clipboard_command::run_clipboard_command`]
//! exports; the runner is a seam here, the way upstream's `vi.mock`
//! replaces the module for the suites. The native backend rides
//! [`super::clipboard::get_native_clipboard`], whose override the same
//! suites drive. A failed native read propagates, upstream's rejected
//! `getImage()` promise — the backends' own failures stay contained.

use std::future::Future;
use std::pin::Pin;

use regex::Regex;

use crate::config::EnvLookup;

use super::clipboard::{ClipboardError, Platform, env_set, get_native_clipboard};
use super::clipboard_command::{
    ClipboardCommandRunner, ProcessClipboardCommandRunner, read_options,
};
use super::mime::detect_supported_image_mime_type;

/// An image read off a clipboard, upstream's `ClipboardImage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipboardImage {
    /// The encoded image bytes.
    pub bytes: Vec<u8>,
    /// The image's MIME type.
    pub mime_type: String,
}

/// The formats the inline pipeline accepts, upstream's
/// `SUPPORTED_IMAGE_MIME_TYPES`.
const SUPPORTED_IMAGE_MIME_TYPES: [&str; 4] =
    ["image/png", "image/jpeg", "image/webp", "image/gif"];

/// How long the target-list probes wait, upstream's
/// `DEFAULT_LIST_TIMEOUT_MS`.
const DEFAULT_LIST_TIMEOUT_MS: u64 = 1000;

/// How long the PowerShell bridge waits, upstream's
/// `DEFAULT_POWERSHELL_TIMEOUT_MS`.
const DEFAULT_POWERSHELL_TIMEOUT_MS: u64 = 5000;

/// Whether the session runs Wayland, upstream's `isWaylandSession`.
#[must_use]
pub fn is_wayland_session(env: &EnvLookup) -> bool {
    env_set(env, "WAYLAND_DISPLAY") || env("XDG_SESSION_TYPE").as_deref() == Some("wayland")
}

/// The MIME type before any parameters, upstream's `baseMimeType`.
#[must_use]
pub fn base_mime_type(mime_type: &str) -> String {
    mime_type
        .split(';')
        .next()
        .unwrap_or(mime_type)
        .trim()
        .to_lowercase()
}

/// The file extension an inline MIME type saves under, upstream's
/// `extensionForImageMimeType`.
#[must_use]
pub fn extension_for_image_mime_type(mime_type: &str) -> Option<&'static str> {
    match base_mime_type(mime_type).as_str() {
        "image/png" => Some("png"),
        "image/jpeg" => Some("jpg"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        _ => None,
    }
}

/// Pick the preferred inline format among offered types, upstream's
/// `selectPreferredImageMimeType` — the first supported type in preference
/// order, else the first `image/` type.
fn select_preferred_image_mime_type(mime_types: &[String]) -> Option<String> {
    let normalized: Vec<(String, String)> = mime_types
        .iter()
        .map(|mime| mime.trim().to_string())
        .filter(|mime| !mime.is_empty())
        .map(|raw| {
            let base = base_mime_type(&raw);
            (raw, base)
        })
        .collect();

    for preferred in SUPPORTED_IMAGE_MIME_TYPES {
        if let Some((raw, _)) = normalized.iter().find(|(_, base)| base == preferred) {
            return Some(raw.clone());
        }
    }

    normalized
        .iter()
        .find(|(_, base)| base.starts_with("image/"))
        .map(|(raw, _)| raw.clone())
}

fn is_supported_image_mime_type(mime_type: &str) -> bool {
    let base = base_mime_type(mime_type);
    SUPPORTED_IMAGE_MIME_TYPES.contains(&base.as_str())
}

/// Convert unsupported image formats to PNG, upstream's private
/// `convertToPng` — `None` when the conversion is unavailable or fails.
/// Unlike the belt's converter this does not apply the EXIF orientation;
/// upstream's two converters differ the same way.
fn convert_to_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let decoded = image::load_from_memory(bytes).ok()?;
    let mut encoded = std::io::Cursor::new(Vec::new());
    decoded
        .write_to(&mut encoded, image::ImageFormat::Png)
        .ok()?;
    Some(encoded.into_inner())
}

/// The outcome one backend answers with, the `ClipboardImage | null |
/// undefined` tri-state upstream's backends return: a failed probe is
/// [`ImageProbe::Unavailable`], a probe that ran but found no image is
/// [`ImageProbe::NoImage`].
enum ImageProbe {
    Unavailable,
    NoImage,
    Found(ClipboardImage),
}

/// The Wayland backend, upstream's `readClipboardImageViaWlPaste`.
fn read_clipboard_image_via_wl_paste(
    runner: &dyn ClipboardCommandRunner,
) -> Pin<Box<dyn Future<Output = ImageProbe> + '_>> {
    Box::pin(async move {
        let Some(list) = runner
            .run(
                "wl-paste",
                &["--list-types".to_string()],
                &read_options(Some(DEFAULT_LIST_TIMEOUT_MS)),
            )
            .await
        else {
            return ImageProbe::Unavailable;
        };

        let types: Vec<String> = String::from_utf8_lossy(&list)
            .lines()
            .map(|mime| mime.trim().to_string())
            .filter(|mime| !mime.is_empty())
            .collect();

        let Some(selected_type) = select_preferred_image_mime_type(&types) else {
            return ImageProbe::NoImage;
        };

        let data = runner
            .run(
                "wl-paste",
                &[
                    "--type".to_string(),
                    selected_type.clone(),
                    "--no-newline".to_string(),
                ],
                &read_options(None),
            )
            .await;
        let Some(data) = data else {
            return ImageProbe::Unavailable;
        };
        if data.is_empty() {
            return ImageProbe::NoImage;
        }

        ImageProbe::Found(ClipboardImage {
            bytes: data,
            mime_type: base_mime_type(&selected_type),
        })
    })
}

/// Whether the Linux session runs under WSL, upstream's `isWSL` — the
/// environment markers, else `/proc/version`.
fn is_wsl(env: &EnvLookup) -> bool {
    if env_set(env, "WSL_DISTRO_NAME") || env_set(env, "WSLENV") {
        return true;
    }

    std::fs::read_to_string("/proc/version").map_or_else(
        |_| false,
        |release| Regex::new(r"(?i)microsoft|wsl").is_ok_and(|pattern| pattern.is_match(&release)),
    )
}

/// The Windows clipboard through PowerShell, the WSL fallback, upstream's
/// `readClipboardImageViaPowerShell`.
fn read_clipboard_image_via_power_shell(
    runner: &dyn ClipboardCommandRunner,
) -> Pin<Box<dyn Future<Output = Option<ClipboardImage>> + '_>> {
    Box::pin(async move {
        let tmp_file =
            std::env::temp_dir().join(format!("pi-wsl-clip-{}.png", uuid::Uuid::new_v4()));
        let tmp_file = tmp_file.to_string_lossy().into_owned();

        let result: Option<ClipboardImage> = async {
            let win_path_result = runner
                .run(
                    "wslpath",
                    &["-w".to_string(), tmp_file.clone()],
                    &read_options(Some(DEFAULT_LIST_TIMEOUT_MS)),
                )
                .await?;
            let win_path = String::from_utf8_lossy(&win_path_result).trim().to_string();
            if win_path.is_empty() {
                return None;
            }

            let ps_quoted_win_path = win_path.replace('\'', "''");
            let ps_script = [
                "Add-Type -AssemblyName System.Windows.Forms".to_string(),
                "Add-Type -AssemblyName System.Drawing".to_string(),
                format!("$path = '{ps_quoted_win_path}'"),
                "$img = [System.Windows.Forms.Clipboard]::GetImage()".to_string(),
                "if ($img) { $img.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); Write-Output 'ok' } else { Write-Output 'empty' }".to_string(),
            ]
            .join("; ");

            let result = runner
                .run(
                    "powershell.exe",
                    &[
                        "-NoProfile".to_string(),
                        "-Command".to_string(),
                        ps_script,
                    ],
                    &read_options(Some(DEFAULT_POWERSHELL_TIMEOUT_MS)),
                )
                .await?;

            let output = String::from_utf8_lossy(&result).trim().to_string();
            if output != "ok" {
                return None;
            }

            let bytes = std::fs::read(&tmp_file).ok()?;
            if bytes.is_empty() {
                return None;
            }

            Some(ClipboardImage {
                bytes,
                mime_type: "image/png".to_string(),
            })
        }
        .await;
        // Ignore cleanup errors.
        let _removed = std::fs::remove_file(&tmp_file);
        result
    })
}

/// The X11 backend, upstream's `readClipboardImageViaXclip`.
fn read_clipboard_image_via_xclip(
    runner: &dyn ClipboardCommandRunner,
) -> Pin<Box<dyn Future<Output = ImageProbe> + '_>> {
    Box::pin(async move {
        let targets = runner
            .run(
                "xclip",
                &[
                    "-selection".to_string(),
                    "clipboard".to_string(),
                    "-t".to_string(),
                    "TARGETS".to_string(),
                    "-o".to_string(),
                ],
                &read_options(Some(DEFAULT_LIST_TIMEOUT_MS)),
            )
            .await;

        let targets_seen = targets.is_some();
        let candidate_types: Vec<String> = targets.map_or_else(Vec::new, |targets| {
            String::from_utf8_lossy(&targets)
                .lines()
                .map(|mime| mime.trim().to_string())
                .filter(|mime| !mime.is_empty())
                .collect()
        });

        let preferred = select_preferred_image_mime_type(&candidate_types);
        if targets_seen && preferred.is_none() {
            return ImageProbe::NoImage;
        }
        // Upstream's `new Set(preferred ? [preferred, ...SUPPORTED] : SUPPORTED)`:
        // dedup keeping first-seen order.
        let mut try_types: Vec<String> = Vec::new();
        if let Some(preferred) = preferred {
            try_types.push(preferred);
        }
        for supported in SUPPORTED_IMAGE_MIME_TYPES {
            if !try_types.iter().any(|mime| mime == supported) {
                try_types.push(supported.to_string());
            }
        }

        for mime_type in try_types {
            let data = runner
                .run(
                    "xclip",
                    &[
                        "-selection".to_string(),
                        "clipboard".to_string(),
                        "-t".to_string(),
                        mime_type.clone(),
                        "-o".to_string(),
                    ],
                    &read_options(None),
                )
                .await;
            if let Some(data) = data.filter(|data| !data.is_empty()) {
                return ImageProbe::Found(ClipboardImage {
                    bytes: data,
                    mime_type: base_mime_type(&mime_type),
                });
            }
        }

        ImageProbe::Unavailable
    })
}

/// The native backend, upstream's `readClipboardImageViaNativeClipboard`.
/// A rejected native read propagates, upstream's thrown promise; absence
/// reads as [`ImageProbe::Unavailable`] and an empty read as
/// [`ImageProbe::NoImage`].
fn read_clipboard_image_via_native_clipboard() -> Result<ImageProbe, ClipboardError> {
    let Some(native) = get_native_clipboard() else {
        return Ok(ImageProbe::Unavailable);
    };
    let bytes = native.get_image()?;
    if bytes.is_empty() {
        return Ok(ImageProbe::NoImage);
    }
    let mime_type = detect_supported_image_mime_type(&bytes)
        .unwrap_or("application/octet-stream")
        .to_string();
    Ok(ImageProbe::Found(ClipboardImage { bytes, mime_type }))
}

/// Read an image from the system clipboard, upstream's `readClipboardImage`
/// over the process environment and platform.
///
/// # Errors
/// The native backend's rejected read propagates, upstream's thrown
/// promise; every command backend answers `None` instead.
pub async fn read_clipboard_image(
    env: &EnvLookup,
    platform: Platform,
) -> Result<Option<ClipboardImage>, ClipboardError> {
    read_clipboard_image_with(env, platform, &ProcessClipboardCommandRunner).await
}

/// [`read_clipboard_image`] over an injected command runner, the seam the
/// suites drive.
///
/// # Errors
/// The native backend's rejected read propagates, upstream's thrown
/// promise; every command backend answers `None` instead.
pub async fn read_clipboard_image_with(
    env: &EnvLookup,
    platform: Platform,
    runner: &dyn ClipboardCommandRunner,
) -> Result<Option<ClipboardImage>, ClipboardError> {
    if env_set(env, "TERMUX_VERSION") {
        return Ok(None);
    }

    let mut probe = ImageProbe::Unavailable;

    if platform == Platform::Linux {
        let wsl = is_wsl(env);
        if is_wayland_session(env) || wsl {
            probe = read_clipboard_image_via_wl_paste(runner).await;
        }
        if matches!(probe, ImageProbe::Unavailable) {
            probe = read_clipboard_image_via_xclip(runner).await;
        }
        // Preserve Linux's empty/unavailable distinction if Windows has no image.
        if !matches!(probe, ImageProbe::Found(_))
            && wsl
            && let Some(found) = read_clipboard_image_via_power_shell(runner).await
        {
            probe = ImageProbe::Found(found);
        }
        if matches!(probe, ImageProbe::Unavailable) {
            probe = read_clipboard_image_via_native_clipboard()?;
        }
    } else {
        probe = read_clipboard_image_via_native_clipboard()?;
    }

    let Some(image) = (match probe {
        ImageProbe::Found(image) => Some(image),
        ImageProbe::NoImage | ImageProbe::Unavailable => None,
    }) else {
        return Ok(None);
    };

    // Convert unsupported formats (e.g., Windows DIB data wrapped as BMP) to PNG
    if !is_supported_image_mime_type(&image.mime_type) {
        let Some(png_bytes) = convert_to_png(&image.bytes) else {
            return Ok(None);
        };
        return Ok(Some(ClipboardImage {
            bytes: png_bytes,
            mime_type: "image/png".to_string(),
        }));
    }

    Ok(Some(image))
}
