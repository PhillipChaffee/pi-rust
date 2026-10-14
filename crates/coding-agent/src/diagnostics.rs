//! Resource diagnostics, upstream's `src/core/diagnostics.ts`.

/// One resource's collision with another of the same name, upstream's
/// `ResourceCollision`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceCollision {
    /// The colliding resource family, upstream's `resourceType`.
    pub resource_type: ResourceType,
    /// The shared name — a skill name, a command/tool/flag name, a prompt
    /// name, a theme name.
    pub name: String,
    /// The path of the resource that kept the name.
    pub winner_path: String,
    /// The path of the resource that lost it.
    pub loser_path: String,
    /// The winner's source label when one is known (upstream's
    /// `winnerSource`, e.g. `npm:foo`, `git:…`, `local`).
    pub winner_source: Option<String>,
    /// The loser's source label when one is known.
    pub loser_source: Option<String>,
}

/// The colliding resource families, upstream's `resourceType` union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceType {
    /// An extension.
    Extension,
    /// A skill.
    Skill,
    /// A prompt template.
    Prompt,
    /// A theme.
    Theme,
}

/// One warning, error, or collision produced while loading a resource
/// layer, upstream's `ResourceDiagnostic`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceDiagnostic {
    /// The severity family, upstream's `type`.
    pub kind: ResourceDiagnosticKind,
    /// The human-readable message.
    pub message: String,
    /// The path the diagnostic attaches to, when one applies.
    pub path: Option<String>,
    /// The collision detail, set only on `Collision` diagnostics.
    pub collision: Option<ResourceCollision>,
}

/// The diagnostic severities, upstream's `"warning" | "error" | "collision"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceDiagnosticKind {
    /// A non-fatal problem; the resource is skipped or degraded.
    Warning,
    /// A failure; the resource did not load.
    Error,
    /// A name collision; both resources loaded, the first kept the name.
    Collision,
}
