//! The physical-modifier probe, the Rust replacement for upstream's
//! `native/` N-API platform matrix
//! ([#51](https://github.com/PhillipChaffee/pi-rust/issues/51)).
//!
//! Upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: its
//! `src/native-platform.ts`, `src/native-modifiers.ts`, and
//! `src/native-module-path.ts` have no module-loading counterpart here — the
//! replacement compiles in and links frameworks directly.
//!
//! - Physical modifiers — upstream's darwin helper
//!   (`native/darwin/src/darwin-platform.m`) answers `isModifierPressed(name)`
//!   from `CGEventSourceFlagsState(kCGEventSourceStateCombinedSessionState)`.
//!   [`is_native_modifier_pressed`] reads the same combined-session state
//!   through `readkey`, which encapsulates the unsafe call. Restatement:
//!   upstream tests the single event-flag bit (`flags & mask != 0`), readkey
//!   also requires the matching device-dependent left/right key bit — the
//!   two agree for every physically pressed modifier, and only flags injected
//!   without device state could diverge. Linux loads no modifier helper
//!   upstream, so it answers `false` the way upstream does without a loaded
//!   module; the win32 helper's `enableVirtualTerminalInput` and modifier
//!   probe ride Windows, which this effort rules out (map ticket "Decide the
//!   Rust stack").
//! - Clipboard — upstream's `getNativeClipboard` (`getText`/`getImage`,
//!   optional `setText`; Linux routes to an X11 helper that serves selection
//!   ownership from a worker thread, `native/linux/src/clipboard-worker.h`)
//!   has no consumer in this crate. The chosen replacement is `arboard`,
//!   whose X11 backend serves ownership from a worker thread the same way;
//!   it lands with its first consumer in the coding-agent port, which will
//!   consume the getter there instead of a seam exported from here.

/// A physical modifier key, upstream `ModifierKey`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifierKey {
    /// Upstream `"shift"`.
    Shift,
    /// Upstream `"command"`.
    Command,
    /// Upstream `"control"`.
    Control,
    /// Upstream `"option"`.
    Option,
}

/// Whether the modifier is physically pressed right now, upstream
/// `isNativeModifierPressed`: an absent helper or a failed probe answers
/// `false`, upstream's catch around the native call.
#[must_use]
pub fn is_native_modifier_pressed(key: ModifierKey) -> bool {
    #[cfg(target_os = "macos")]
    {
        use readkey::Keycode;
        match key {
            ModifierKey::Shift => Keycode::Shift.is_pressed() || Keycode::RightShift.is_pressed(),
            ModifierKey::Command => {
                Keycode::Command.is_pressed() || Keycode::RightCommand.is_pressed()
            }
            ModifierKey::Control => {
                Keycode::Control.is_pressed() || Keycode::RightControl.is_pressed()
            }
            ModifierKey::Option => {
                Keycode::Option.is_pressed() || Keycode::RightOption.is_pressed()
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = key;
        false
    }
}
