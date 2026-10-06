//! Boundary tests binding the project-trust resolution branches, at pin
//! 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759. The module has no 1:1 suite:
//! upstream's `core/project-trust.ts` flow — the override, the resource
//! probe, the extension gate, the stored decision, the default posture, and
//! the selector — lands here as doubles over the two seams.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;
use pi_coding_agent::project_trust::{
    ExtensionTrustEvent, ExtensionTrustGate, ExtensionTrustOutcome, ExtensionTrustResult,
    ResolveProjectTrustedOptions, resolve_project_trusted,
};
use pi_coding_agent::settings_manager::DefaultProjectTrust;
use pi_coding_agent::trust_manager::{ProjectTrustSelector, ProjectTrustStore};

// === test doubles ===========================================================

type ErrorLog = Arc<Mutex<Vec<String>>>;
type SelectLog = Arc<Mutex<Vec<(String, Vec<String>)>>>;

struct GateDouble {
    outcome: Option<ExtensionTrustOutcome>,
    errors: Vec<(String, String)>,
    events: Arc<Mutex<Vec<String>>>,
}

impl ExtensionTrustGate for GateDouble {
    fn emit_project_trust<'a>(
        &'a self,
        event: &'a ExtensionTrustEvent,
    ) -> BoxedFuture<'a, ExtensionTrustResult> {
        let outcome = self.outcome;
        let errors = self.errors.clone();
        let events = Arc::clone(&self.events);
        Box::pin(async move {
            events
                .lock()
                .expect("gate event log")
                .push(event.cwd.clone());
            (outcome, errors)
        })
    }
}

struct SelectorDouble {
    has_ui_flag: bool,
    choice: Option<String>,
    calls: SelectLog,
}

impl ProjectTrustSelector for SelectorDouble {
    fn select(&self, prompt: String, options: Vec<String>) -> BoxedFuture<'static, Option<String>> {
        let choice = self.choice.clone();
        let calls = Arc::clone(&self.calls);
        Box::pin(async move {
            calls
                .lock()
                .expect("selector call log")
                .push((prompt, options));
            choice
        })
    }

    fn has_ui(&self) -> bool {
        self.has_ui_flag
    }
}

fn selector(choice: Option<&str>, has_ui: bool) -> SelectorDouble {
    SelectorDouble {
        has_ui_flag: has_ui,
        choice: choice.map(str::to_string),
        calls: Arc::new(Mutex::new(Vec::new())),
    }
}

fn gate(outcome: Option<ExtensionTrustOutcome>, errors: Vec<(&str, &str)>) -> GateDouble {
    GateDouble {
        outcome,
        errors: errors
            .into_iter()
            .map(|(path, error)| (path.to_string(), error.to_string()))
            .collect(),
        events: Arc::new(Mutex::new(Vec::new())),
    }
}

const fn yes() -> ExtensionTrustOutcome {
    ExtensionTrustOutcome {
        trusted_is_yes: true,
        remember: true,
    }
}

// === fixtures ===============================================================

/// A cwd with trust-requiring resources (a `.pi/settings.json`), plus its
/// agent dir. The root stays alive in the caller's binding.
struct Fixture {
    root: tempfile::TempDir,
    cwd: String,
    store: ProjectTrustStore,
}

fn fixture() -> Fixture {
    let root = tempfile::tempdir().expect("scratch root");
    let agent = root.path().join("agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    let cwd_dir = root.path().join("project");
    std::fs::create_dir_all(cwd_dir.join(".pi")).expect("project .pi dir");
    std::fs::write(cwd_dir.join(".pi").join("settings.json"), "{}").expect("settings write");
    Fixture {
        root,
        cwd: cwd_dir.to_string_lossy().into_owned(),
        store: ProjectTrustStore::new(&agent.to_string_lossy()),
    }
}

impl Fixture {
    fn trust_path(&self) -> String {
        // the store resolves the agent dir without canonicalizing, so the
        // file the store writes is this path text
        self.root
            .path()
            .join("agent")
            .join("trust.json")
            .to_string_lossy()
            .into_owned()
    }

    fn trust_file_exists(&self) -> bool {
        Path::new(&self.trust_path()).exists()
    }

    fn canonical_cwd(&self) -> String {
        std::fs::canonicalize(&self.cwd)
            .expect("cwd canonicalize")
            .to_string_lossy()
            .into_owned()
    }

    fn canonical_parent(&self) -> String {
        std::fs::canonicalize(&self.cwd)
            .expect("cwd canonicalize")
            .parent()
            .expect("cwd parent")
            .to_string_lossy()
            .into_owned()
    }
}

async fn resolve_with(
    fixture: &Fixture,
    gate: Option<&dyn ExtensionTrustGate>,
    ui: &dyn ProjectTrustSelector,
    default_trust: Option<DefaultProjectTrust>,
    override_trust: Option<bool>,
    on_extension_error: Option<&dyn Fn(&str)>,
) -> bool {
    resolve_project_trusted(ResolveProjectTrustedOptions {
        cwd: &fixture.cwd,
        trust_store: &fixture.store,
        trust_override: override_trust,
        default_project_trust: default_trust,
        extension_gate: gate,
        project_trust_context: ui,
        on_extension_error,
    })
    .await
    .expect("resolve")
}

async fn resolve(
    fixture: &Fixture,
    gate: Option<&dyn ExtensionTrustGate>,
    ui: &dyn ProjectTrustSelector,
) -> bool {
    resolve_with(fixture, gate, ui, None, None, None).await
}

// === the override and the resource probe ====================================

#[tokio::test]
async fn the_trust_override_short_circuits_everything() {
    let fixture = fixture();
    let gate = gate(Some(yes()), vec![]);
    let ui = selector(Some("Do not trust"), true);

    assert!(resolve_with(&fixture, Some(&gate), &ui, None, Some(true), None).await);
    assert!(!resolve_with(&fixture, Some(&gate), &ui, None, Some(false), None).await);
    // nothing ran: no gate events, no selector calls, no store file
    assert!(gate.events.lock().expect("gate log").is_empty());
    assert!(ui.calls.lock().expect("call log").is_empty());
    assert!(!fixture.trust_file_exists());
}

#[tokio::test]
async fn a_project_without_gated_resources_trusts_implicitly() {
    let root = tempfile::tempdir().expect("scratch root");
    let plain = root.path().join("plain");
    std::fs::create_dir_all(&plain).expect("plain dir");
    let agent = root.path().join("agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    let cwd = plain.to_string_lossy().into_owned();
    let store = ProjectTrustStore::new(&agent.to_string_lossy());

    let ui = selector(Some("Trust"), true);
    let outcome = resolve_project_trusted(ResolveProjectTrustedOptions {
        cwd: &cwd,
        trust_store: &store,
        trust_override: None,
        default_project_trust: None,
        extension_gate: None,
        project_trust_context: &ui,
        on_extension_error: None,
    })
    .await
    .expect("resolve");
    assert!(outcome);
    assert!(ui.calls.lock().expect("call log").is_empty());
    let trust_path = std::fs::canonicalize(&agent)
        .expect("agent canonicalize")
        .join("trust.json");
    assert!(!trust_path.exists());
}

// === the extension gate =====================================================

#[tokio::test]
async fn the_extension_gate_decision_remembers() {
    let fixture = fixture();
    let gate = gate(Some(yes()), vec![]);
    let ui = selector(Some("Do not trust"), true);

    assert!(resolve(&fixture, Some(&gate), &ui).await);
    assert_eq!(
        fixture.store.get(&fixture.cwd).expect("trust get"),
        Some(true)
    );
    assert!(ui.calls.lock().expect("call log").is_empty());
    assert_eq!(
        *gate.events.lock().expect("gate log"),
        vec![fixture.cwd.clone()]
    );
}

#[tokio::test]
async fn the_extension_gate_refusal_remembers() {
    let fixture = fixture();
    let gate = gate(
        Some(ExtensionTrustOutcome {
            trusted_is_yes: false,
            remember: true,
        }),
        vec![],
    );
    let ui = selector(Some("Trust"), true);

    assert!(!resolve(&fixture, Some(&gate), &ui).await);
    assert_eq!(
        fixture.store.get(&fixture.cwd).expect("trust get"),
        Some(false)
    );
}

#[tokio::test]
async fn the_extension_gate_without_remember_writes_nothing() {
    let fixture = fixture();
    let gate = gate(
        Some(ExtensionTrustOutcome {
            trusted_is_yes: true,
            remember: false,
        }),
        vec![],
    );
    let ui = selector(Some("Do not trust"), true);

    assert!(resolve(&fixture, Some(&gate), &ui).await);
    assert_eq!(fixture.store.get(&fixture.cwd).expect("trust get"), None);
    assert!(!fixture.trust_file_exists());
}

#[tokio::test]
async fn extension_errors_report_the_verbatim_message() {
    let fixture = fixture();
    let gate = gate(None, vec![("/x/ext.ts", "boom"), ("/y.ts", "bad")]);
    let error_log: ErrorLog = Arc::new(Mutex::new(Vec::new()));
    let sink_log = Arc::clone(&error_log);
    let sink = move |message: &str| {
        sink_log
            .lock()
            .expect("error log")
            .push(message.to_string());
    };
    // no UI: after the gate declines, the resolution refuses
    let ui = selector(Some("Trust"), false);

    assert!(!resolve_with(&fixture, Some(&gate), &ui, None, None, Some(&sink)).await);
    assert_eq!(
        *error_log.lock().expect("error log"),
        vec![
            "Extension \"/x/ext.ts\" project_trust error: boom".to_string(),
            "Extension \"/y.ts\" project_trust error: bad".to_string(),
        ]
    );
    assert_eq!(
        *gate.events.lock().expect("gate log"),
        vec![fixture.cwd.clone()]
    );
    assert!(!fixture.trust_file_exists());
}

// === the stored decision and the default postures ===========================

#[tokio::test]
async fn the_stored_decision_wins_after_the_gate_declines() {
    let fixture = fixture();
    fixture
        .store
        .set(&fixture.cwd, Some(true))
        .expect("trust set");
    let gate = gate(None, vec![]);
    let ui = selector(Some("Do not trust"), true);

    assert!(resolve(&fixture, Some(&gate), &ui).await);
    // the gate ran and declined; the store answered
    assert_eq!(
        *gate.events.lock().expect("gate log"),
        vec![fixture.cwd.clone()]
    );
    assert!(ui.calls.lock().expect("call log").is_empty());

    fixture
        .store
        .set(&fixture.cwd, Some(false))
        .expect("trust set");
    assert!(!resolve(&fixture, Some(&gate), &ui).await);
}

#[tokio::test]
async fn the_default_postures_apply_without_a_ui() {
    let fixture = fixture();
    let ui = selector(Some("Trust"), false);

    assert!(
        resolve_with(
            &fixture,
            None,
            &ui,
            Some(DefaultProjectTrust::Always),
            None,
            None
        )
        .await
    );
    assert!(
        !resolve_with(
            &fixture,
            None,
            &ui,
            Some(DefaultProjectTrust::Never),
            None,
            None
        )
        .await
    );
    assert!(ui.calls.lock().expect("call log").is_empty());
    assert!(!fixture.trust_file_exists());
}

#[tokio::test]
async fn a_host_without_a_ui_refuses() {
    let fixture = fixture();
    let ui = selector(Some("Trust"), false);

    assert!(!resolve(&fixture, None, &ui).await);
    assert!(ui.calls.lock().expect("call log").is_empty());
    assert!(!fixture.trust_file_exists());
}

// === the selector ===========================================================

#[tokio::test]
async fn the_selector_saves_the_chosen_option_and_returns_its_flag() {
    let fixture = fixture();
    let parent = fixture.canonical_parent();
    let ui = selector(Some("Trust"), true);

    assert!(resolve(&fixture, None, &ui).await);
    assert_eq!(
        fixture.store.get(&fixture.cwd).expect("trust get"),
        Some(true)
    );

    let (prompt, labels) = {
        let calls = ui.calls.lock().expect("call log");
        assert_eq!(calls.len(), 1);
        (calls[0].0.clone(), calls[0].1.clone())
    };
    assert_eq!(
        prompt,
        format!(
            "Trust project folder?\n{}\n\nThis allows pi to load .pi settings and resources, install missing project packages, and execute project extensions.",
            fixture.cwd
        )
    );
    assert_eq!(
        labels,
        vec![
            "Trust".to_string(),
            format!("Trust parent folder ({parent})"),
            "Trust (this session only)".to_string(),
            "Do not trust".to_string(),
            "Do not trust (this session only)".to_string(),
        ]
    );

    // a stored decision short-circuits the prompt on the next run
    assert!(resolve(&fixture, None, &ui).await);
    assert_eq!(ui.calls.lock().expect("call log").len(), 1);
}

#[tokio::test]
async fn the_parent_choice_records_the_parent_and_clears_the_cwd() {
    let fixture = fixture();
    let parent = fixture.canonical_parent();
    let cwd_key = fixture.canonical_cwd();
    let label = format!("Trust parent folder ({parent})");
    let ui = selector(Some(&label), true);

    assert!(resolve(&fixture, None, &ui).await);
    // the cwd entry is deleted; the parent's trust governs
    assert_eq!(
        fixture.store.get(&fixture.cwd).expect("trust get"),
        Some(true)
    );
    let file = std::fs::read_to_string(fixture.trust_path()).expect("trust read");
    let parsed: serde_json::Value = serde_json::from_str(&file).expect("trust parse");
    assert_eq!(parsed.get(&cwd_key), None);
    assert_eq!(parsed.get(&parent), Some(&serde_json::Value::Bool(true)));
}

#[tokio::test]
async fn the_refusal_records_false_for_the_cwd() {
    let fixture = fixture();
    let ui = selector(Some("Do not trust"), true);

    assert!(!resolve(&fixture, None, &ui).await);
    assert_eq!(
        fixture.store.get(&fixture.cwd).expect("trust get"),
        Some(false)
    );
}

#[tokio::test]
async fn the_session_only_choices_write_nothing() {
    let fixture = fixture();
    let trust_ui = selector(Some("Trust (this session only)"), true);
    assert!(resolve(&fixture, None, &trust_ui).await);
    assert_eq!(trust_ui.calls.lock().expect("call log").len(), 1);
    assert!(!fixture.trust_file_exists());

    let refuse_ui = selector(Some("Do not trust (this session only)"), true);
    assert!(!resolve(&fixture, None, &refuse_ui).await);
    assert!(!fixture.trust_file_exists());
}

#[tokio::test]
async fn a_selector_none_refuses() {
    let fixture = fixture();
    let ui = selector(None, true);

    assert!(!resolve(&fixture, None, &ui).await);
    assert_eq!(ui.calls.lock().expect("call log").len(), 1);
    assert!(!fixture.trust_file_exists());
}

#[tokio::test]
async fn extension_errors_without_a_sink_still_fall_through() {
    let fixture = fixture();
    let gate = gate(None, vec![("/x/ext.ts", "boom")]);
    // no UI: after the gate declines, the resolution refuses
    let ui = selector(Some("Trust"), false);

    assert!(!resolve_with(&fixture, Some(&gate), &ui, None, None, None).await);
    assert_eq!(
        *gate.events.lock().expect("gate log"),
        vec![fixture.cwd.clone()]
    );
    assert!(!fixture.trust_file_exists());
}

#[tokio::test]
async fn a_failing_store_turns_the_gate_decision_into_an_error() {
    let root = tempfile::tempdir().expect("scratch root");
    // the agent "dir" is a regular file: the store write fails
    let blocker = root.path().join("agent");
    std::fs::write(&blocker, b"not a directory").expect("blocker write");
    let cwd_dir = root.path().join("project");
    std::fs::create_dir_all(cwd_dir.join(".pi")).expect("project .pi dir");
    std::fs::write(cwd_dir.join(".pi").join("settings.json"), "{}").expect("settings write");
    let fixture = Fixture {
        root,
        cwd: cwd_dir.to_string_lossy().into_owned(),
        store: ProjectTrustStore::new(&blocker.to_string_lossy()),
    };

    let gate = gate(Some(yes()), vec![]);
    let ui = selector(Some("Do not trust"), true);
    let outcome = resolve_project_trusted(ResolveProjectTrustedOptions {
        cwd: &fixture.cwd,
        trust_store: &fixture.store,
        trust_override: None,
        default_project_trust: None,
        extension_gate: Some(&gate),
        project_trust_context: &ui,
        on_extension_error: None,
    })
    .await;
    assert!(outcome.is_err());
    assert!(ui.calls.lock().expect("call log").is_empty());
}

#[tokio::test]
async fn a_failing_store_read_turns_the_stored_check_into_an_error() {
    let root = tempfile::tempdir().expect("scratch root");
    let agent = root.path().join("agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    std::fs::create_dir_all(agent.join("trust.json")).expect("trust.json as directory");
    let cwd_dir = root.path().join("project");
    std::fs::create_dir_all(cwd_dir.join(".pi")).expect("project .pi dir");
    std::fs::write(cwd_dir.join(".pi").join("settings.json"), "{}").expect("settings write");
    let fixture = Fixture {
        root,
        cwd: cwd_dir.to_string_lossy().into_owned(),
        store: ProjectTrustStore::new(&agent.to_string_lossy()),
    };

    let gate = gate(None, vec![]);
    let ui = selector(Some("Trust"), true);
    let outcome = resolve_project_trusted(ResolveProjectTrustedOptions {
        cwd: &fixture.cwd,
        trust_store: &fixture.store,
        trust_override: None,
        default_project_trust: None,
        extension_gate: Some(&gate),
        project_trust_context: &ui,
        on_extension_error: None,
    })
    .await;
    assert!(outcome.is_err());
    assert!(ui.calls.lock().expect("call log").is_empty());
}

#[tokio::test]
async fn a_failing_store_write_turns_the_prompt_choice_into_an_error() {
    let root = tempfile::tempdir().expect("scratch root");
    let agent = root.path().join("agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    let cwd_dir = root.path().join("project");
    std::fs::create_dir_all(cwd_dir.join(".pi")).expect("project .pi dir");
    std::fs::write(cwd_dir.join(".pi").join("settings.json"), "{}").expect("settings write");
    // the store reads fine but the file rejects the write
    let trust_file = agent.join("trust.json");
    std::fs::write(&trust_file, "{}").expect("trust write");
    let mut permissions = std::fs::metadata(&trust_file)
        .expect("trust metadata")
        .permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&trust_file, permissions).expect("trust chmod");
    let fixture = Fixture {
        root,
        cwd: cwd_dir.to_string_lossy().into_owned(),
        store: ProjectTrustStore::new(&agent.to_string_lossy()),
    };

    let ui = selector(Some("Trust"), true);
    let outcome = resolve_project_trusted(ResolveProjectTrustedOptions {
        cwd: &fixture.cwd,
        trust_store: &fixture.store,
        trust_override: None,
        default_project_trust: None,
        extension_gate: None,
        project_trust_context: &ui,
        on_extension_error: None,
    })
    .await;
    assert!(outcome.is_err());
    assert_eq!(ui.calls.lock().expect("call log").len(), 1);
}

// === the options shape ======================================================

#[test]
fn the_options_debug_names_the_inputs() {
    let fixture = fixture();
    let ui = selector(None, true);
    let options = ResolveProjectTrustedOptions {
        cwd: &fixture.cwd,
        trust_store: &fixture.store,
        trust_override: Some(true),
        default_project_trust: Some(DefaultProjectTrust::Never),
        extension_gate: None,
        project_trust_context: &ui,
        on_extension_error: None,
    };
    let debug = format!("{options:?}");
    assert!(debug.contains(fixture.cwd.as_str()), "{debug}");
    assert!(debug.contains("trust_override: Some(true)"), "{debug}");
    assert!(debug.contains("default_project_trust"), "{debug}");
}
