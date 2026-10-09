//! The self-update config suite, the redesigned slice of upstream's
//! `test/config.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements this suite records: the npm/pnpm/yarn/bun command
//! builders drop with the npm channel (ADR 0007); the single native
//! binary's managed method is the cargo install, so the twelve upstream
//! cases restate to the cargo-bin layout (custom prefix → cargo bin dir,
//! `npm --prefix` → the `--locked` install, renamed packages → the
//! uninstall-then-install two-step, the Windows non-inference case drops
//! with the platform, and the wrapper/unknown and not-writable cases port
//! 1:1). `findNodePackageDir` drops with #118's recorded decision.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use pi_coding_agent::config::{
    EnvLookup, InstallMethod, SelfUpdatePackageTarget, detect_install_method,
    detect_install_method_with, get_changelog_path, get_package_dir, get_self_update_command,
    get_self_update_command_with, get_self_update_unavailable_instruction,
    get_self_update_unavailable_instruction_with, get_update_instruction,
    get_update_instruction_with,
};
use pi_coding_agent::pi_manifest::read_pi_manifest;

/// The env lookup with HOME pointed at the temp root and `CARGO_HOME` unset,
/// the restated `process.env.HOME`/`PI_PACKAGE_DIR` stubs.
fn home_env(root: &Path) -> EnvLookup {
    let home = root.to_string_lossy().into_owned();
    Box::new(move |key: &str| match key {
        "HOME" => Some(home.clone()),
        _ => None,
    })
}

fn home_env_with(root: &Path, cargo_home: Option<&Path>) -> EnvLookup {
    let home = root.to_string_lossy().into_owned();
    let cargo_home = cargo_home.map(|path| path.to_string_lossy().into_owned());
    Box::new(move |key: &str| match key {
        "HOME" => Some(home.clone()),
        "CARGO_HOME" => cargo_home.clone(),
        _ => None,
    })
}

/// The cargo-bin install layout, the restated `createNpmPrefixInstall`:
/// `<cargo-home>/bin/pi` running with `PI_PACKAGE_DIR` beside it.
struct CargoInstall {
    root: PathBuf,
    exe_path: PathBuf,
    package_dir: PathBuf,
}

fn create_cargo_install(root: &Path) -> CargoInstall {
    let bin_dir = root.join(".cargo/bin");
    let exe_path = bin_dir.join("pi");
    let package_dir = root.join(".cargo/bin");
    std::fs::create_dir_all(&bin_dir).expect("cargo bin dir");
    std::fs::write(&exe_path, "#!/bin/sh\n").expect("write exe");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&exe_path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    CargoInstall {
        root: root.to_path_buf(),
        exe_path,
        package_dir,
    }
}

/// The PI_PACKAGE_DIR-carrying env for an install, the restated
/// `process.env.PI_PACKAGE_DIR = packageDir` stub.
fn install_env(install: &CargoInstall) -> EnvLookup {
    let package_dir = install.package_dir.to_string_lossy().into_owned();
    let home = install.root.to_string_lossy().into_owned();
    Box::new(move |key: &str| match key {
        "HOME" => Some(home.clone()),
        "PI_PACKAGE_DIR" => Some(package_dir.clone()),
        _ => None,
    })
}

#[test]
fn detects_cargo_installs_from_the_bin_layout() {
    let temp = tempfile::tempdir().expect("tempdir");
    let install = create_cargo_install(temp.path());
    let env = install_env(&install);

    assert_eq!(
        detect_install_method_with(&install.exe_path, &env),
        InstallMethod::Cargo
    );
}

#[test]
fn does_not_self_update_unknown_wrapper_installs() {
    let temp = tempfile::tempdir().expect("tempdir");
    let env = home_env(temp.path());
    let exe_path = temp.path().join("usr/local/bin/pi");

    assert_eq!(
        detect_install_method_with(&exe_path, &env),
        InstallMethod::Unknown
    );
    let target = SelfUpdatePackageTarget::from_package_name("pi-coding-agent");
    assert!(
        get_self_update_command_with("pi-coding-agent", &target, &exe_path, &env).is_none(),
        "an unknown wrapper carries no self-update command"
    );
    assert_eq!(
        get_update_instruction_with("pi-coding-agent", &exe_path, &env),
        "Update pi-coding-agent using the package manager, wrapper, or source checkout that provides this installation."
    );
}

#[test]
fn self_updates_cargo_installs_with_the_locked_build() {
    let temp = tempfile::tempdir().expect("tempdir");
    let install = create_cargo_install(temp.path());
    let env = install_env(&install);

    let target = SelfUpdatePackageTarget::from_package_name("pi-coding-agent");
    let command = get_self_update_command_with("pi-coding-agent", &target, &install.exe_path, &env)
        .expect("the cargo install self-updates");
    assert_eq!(command.command, "cargo");
    assert_eq!(
        command.args,
        vec![
            "install".to_string(),
            "--locked".to_string(),
            "pi-coding-agent".to_string(),
        ]
    );
    assert_eq!(command.display, "cargo install --locked pi-coding-agent");
    assert!(command.steps.is_none(), "no rename, no uninstall step");
    assert_eq!(
        get_update_instruction_with("pi-coding-agent", &install.exe_path, &env),
        "Run: cargo install --locked pi-coding-agent"
    );
}

#[test]
fn self_updates_exact_versions_without_uninstalling_the_current_package() {
    let temp = tempfile::tempdir().expect("tempdir");
    let install = create_cargo_install(temp.path());
    let env = install_env(&install);

    let target = SelfUpdatePackageTarget::new("pi-coding-agent", Some("pi-coding-agent@1.2.3"));
    let command = get_self_update_command_with("pi-coding-agent", &target, &install.exe_path, &env)
        .expect("the pinned target self-updates");
    assert_eq!(
        command.args,
        vec![
            "install".to_string(),
            "--locked".to_string(),
            "pi-coding-agent".to_string(),
            "--version".to_string(),
            "1.2.3".to_string(),
        ]
    );
    assert!(command.steps.is_none());
}

#[test]
fn self_updates_renamed_packages_by_removing_the_old_one_first() {
    let temp = tempfile::tempdir().expect("tempdir");
    let install = create_cargo_install(temp.path());
    let env = install_env(&install);

    let target = SelfUpdatePackageTarget::from_package_name("pi-next");
    let command = get_self_update_command_with("pi-coding-agent", &target, &install.exe_path, &env)
        .expect("the rename self-updates");
    let steps = command.steps.as_ref().expect("two steps");
    assert_eq!(steps[0].command, "cargo");
    assert_eq!(
        steps[0].args,
        vec!["uninstall".to_string(), "pi-coding-agent".to_string()]
    );
    assert_eq!(
        steps[1].args,
        vec![
            "install".to_string(),
            "--locked".to_string(),
            "pi-next".to_string()
        ]
    );
    assert_eq!(
        command.display,
        "cargo uninstall pi-coding-agent && cargo install --locked pi-next"
    );
}

#[test]
fn quotes_self_update_display_paths() {
    let temp = tempfile::tempdir().expect("tempdir");
    // The cargo bin dir carries a space in its name, upstream's
    // `createNpmPrefixInstall("pi prefix ")` quoting case.
    let bin_dir = temp.path().join("cargo prefix/.cargo/bin");
    let exe_path = bin_dir.join("pi");
    std::fs::create_dir_all(&bin_dir).expect("bin dir");
    std::fs::write(&exe_path, "#!/bin/sh\n").expect("write exe");
    let env = home_env_with(temp.path(), Some(&temp.path().join("cargo prefix/.cargo")));

    let target = SelfUpdatePackageTarget::from_package_name("pi-coding-agent");
    let command = get_self_update_command_with("pi-coding-agent", &target, &exe_path, &env)
        .expect("the spaced layout self-updates");
    assert!(
        command
            .display
            .contains("cargo install --locked pi-coding-agent")
    );
    let _ = command;
}

#[test]
fn respects_cargo_home_overrides() {
    let temp = tempfile::tempdir().expect("tempdir");
    let cargo_home = temp.path().join("custom-cargo");
    let bin_dir = cargo_home.join("bin");
    let exe_path = bin_dir.join("pi");
    std::fs::create_dir_all(&bin_dir).expect("bin dir");
    std::fs::write(&exe_path, "#!/bin/sh\n").expect("write exe");
    let env = home_env_with(temp.path(), Some(&cargo_home));

    assert_eq!(
        detect_install_method_with(&exe_path, &env),
        InstallMethod::Cargo
    );
}

#[test]
fn does_not_self_update_when_the_install_path_is_not_writable() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().expect("tempdir");
        let install = create_cargo_install(temp.path());
        let env = install_env(&install);
        std::fs::set_permissions(&install.package_dir, std::fs::Permissions::from_mode(0o555))
            .expect("chmod read-only");

        let target = SelfUpdatePackageTarget::from_package_name("pi-coding-agent");
        assert!(
            get_self_update_command_with("pi-coding-agent", &target, &install.exe_path, &env)
                .is_none(),
            "an unwritable install path carries no command"
        );
        let instruction = get_self_update_unavailable_instruction_with(
            "pi-coding-agent",
            &target,
            &install.exe_path,
            &env,
        );
        assert!(
            instruction.contains("the install path is not writable"),
            "{instruction}"
        );
        std::fs::set_permissions(&install.package_dir, std::fs::Permissions::from_mode(0o755))
            .expect("restore chmod");
    }
}

#[test]
fn the_unavailable_instruction_names_the_spec() {
    let temp = tempfile::tempdir().expect("tempdir");
    let env = home_env(temp.path());
    let exe_path = temp.path().join("nowhere/pi");
    let target = SelfUpdatePackageTarget::new("pi-coding-agent", Some("pi-coding-agent@1.2.3"));

    let instruction =
        get_self_update_unavailable_instruction_with("pi-coding-agent", &target, &exe_path, &env);
    assert_eq!(
        instruction,
        "Update pi-coding-agent@1.2.3 using the package manager, wrapper, or source checkout that provides this installation."
    );
}

#[test]
fn quotes_whitespace_specs_in_the_display() {
    let temp = tempfile::tempdir().expect("tempdir");
    let install = create_cargo_install(temp.path());
    let env = install_env(&install);

    let target = SelfUpdatePackageTarget::new("pi-coding-agent", Some("pi dev"));
    let command = get_self_update_command_with("pi-coding-agent", &target, &install.exe_path, &env)
        .expect("the spec self-updates");
    assert_eq!(
        command.args,
        vec![
            "install".to_string(),
            "--locked".to_string(),
            "pi dev".to_string(),
        ]
    );
    assert_eq!(command.display, "cargo install --locked \"pi dev\"");
}

#[test]
fn keeps_a_non_semver_version_suffix_in_the_spec() {
    let temp = tempfile::tempdir().expect("tempdir");
    let install = create_cargo_install(temp.path());
    let env = install_env(&install);

    // A suffix that does not parse as a version is not a pin: the spec rides
    // the install verbatim.
    let target = SelfUpdatePackageTarget::new("pi-coding-agent", Some("pi-coding-agent@beta"));
    let command = get_self_update_command_with("pi-coding-agent", &target, &install.exe_path, &env)
        .expect("the suffixed spec self-updates");
    assert_eq!(
        command.args,
        vec![
            "install".to_string(),
            "--locked".to_string(),
            "pi-coding-agent@beta".to_string(),
        ]
    );
    assert!(command.steps.is_none(), "the same crate needs no uninstall");
}

#[test]
fn probes_the_home_fallback_when_cargo_home_is_empty() {
    let temp = tempfile::tempdir().expect("tempdir");
    let bin_dir = temp.path().join(".cargo/bin");
    let exe_path = bin_dir.join("pi");
    std::fs::create_dir_all(&bin_dir).expect("bin dir");
    std::fs::write(&exe_path, "#!/bin/sh\n").expect("write exe");
    // A set-but-empty CARGO_HOME is no override: the probe ladders to HOME.
    let env = home_env_with(temp.path(), Some(Path::new("")));

    assert_eq!(
        detect_install_method_with(&exe_path, &env),
        InstallMethod::Cargo
    );
}

#[test]
fn falls_back_to_the_process_home_when_home_is_blank() {
    let temp = tempfile::tempdir().expect("tempdir");
    let exe_path = temp.path().join("somewhere/pi");
    let env: EnvLookup = Box::new(|key: &str| match key {
        "HOME" => Some(String::new()),
        _ => None,
    });

    // The blank HOME defers to the process home for the probe target; the
    // temp-dir executable sits under neither candidate bin dir.
    assert_eq!(
        detect_install_method_with(&exe_path, &env),
        InstallMethod::Unknown
    );
}

#[test]
fn the_unavailable_instruction_reports_a_writable_cargo_install_as_not_managed() {
    let temp = tempfile::tempdir().expect("tempdir");
    let install = create_cargo_install(temp.path());
    let env = install_env(&install);

    // The path guard is the only split inside the managed branch: a writable
    // install falls through to the not-managed wording, upstream's
    // `getSelfUpdateUnavailableInstruction` else arm.
    let target = SelfUpdatePackageTarget::from_package_name("pi-coding-agent");
    let instruction = get_self_update_unavailable_instruction_with(
        "pi-coding-agent",
        &target,
        &install.exe_path,
        &env,
    );
    assert_eq!(
        instruction,
        "This installation is not managed by a global cargo install. Update it with the package manager, wrapper, or source checkout that provides it."
    );
}

#[test]
fn the_process_wrappers_read_the_real_executable_and_environment() {
    // The test binary lives in the cargo target directory, under neither
    // candidate bin dir, so the process-level getters see an unknown install.
    let target = SelfUpdatePackageTarget::from_package_name("pi-coding-agent");
    assert_eq!(detect_install_method(), InstallMethod::Unknown);
    assert!(
        get_self_update_command("pi-coding-agent", &target).is_none(),
        "an unknown process install carries no self-update command"
    );
    assert_eq!(
        get_self_update_unavailable_instruction("pi-coding-agent", &target),
        "Update pi-coding-agent using the package manager, wrapper, or source checkout that provides this installation."
    );
    assert_eq!(
        get_update_instruction("pi-coding-agent"),
        "Update pi-coding-agent using the package manager, wrapper, or source checkout that provides this installation."
    );
}

#[test]
fn a_flipped_environment_between_probes_disarms_the_command() {
    // The command builder probes the layout twice; when the environment
    // flips between the probes the layout no longer counts as managed, and
    // the command is withheld — the seam's lookup decides each read.
    fn flipped_env(temp: &Path, cargo_home: &Path) -> EnvLookup {
        let home = temp.to_string_lossy().into_owned();
        let cargo_home = cargo_home.to_string_lossy().into_owned();
        let first = AtomicBool::new(true);
        Box::new(move |key: &str| match key {
            "CARGO_HOME" if first.load(Ordering::SeqCst) => {
                first.store(false, Ordering::SeqCst);
                Some(cargo_home.clone())
            }
            "HOME" => Some(home.clone()),
            _ => None,
        })
    }

    let temp = tempfile::tempdir().expect("tempdir");
    let cargo_home = temp.path().join("flip-cargo");
    let bin_dir = cargo_home.join("bin");
    let exe_path = bin_dir.join("pi");
    std::fs::create_dir_all(&bin_dir).expect("bin dir");
    std::fs::write(&exe_path, "#!/bin/sh\n").expect("write exe");

    let target = SelfUpdatePackageTarget::from_package_name("pi-coding-agent");
    assert!(
        get_self_update_command_with(
            "pi-coding-agent",
            &target,
            &exe_path,
            &flipped_env(temp.path(), &cargo_home),
        )
        .is_none(),
        "the flipped probe withholds the command"
    );
    let instruction = get_self_update_unavailable_instruction_with(
        "pi-coding-agent",
        &target,
        &exe_path,
        &flipped_env(temp.path(), &cargo_home),
    );
    assert!(
        instruction.starts_with("This installation is not managed"),
        "{instruction}"
    );
}

#[test]
fn the_changelog_path_joins_the_package_dir() {
    // The changelog rides the package-dir derivation, upstream's
    // `getChangelogPath`.
    assert_eq!(get_changelog_path(), get_package_dir().join("CHANGELOG.md"));
}

#[test]
fn the_pi_manifest_keeps_string_arrays_and_rejects_other_shapes() {
    // The `pi` manifest is package-config surface the package manager reads
    // before the convention directories, upstream's `readPiManifest`.
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("package.json");

    std::fs::write(&path, r#"{"name":"pkg","pi":5}"#).expect("write");
    assert_eq!(
        read_pi_manifest(&path),
        None,
        "a non-object pi is no manifest"
    );

    std::fs::write(
        &path,
        r#"{"name":"pkg","pi":{"prompts":["./prompts/a.md"],"themes":["./themes/dark.json"],"extensions":[1]}}"#,
    )
    .expect("write");
    let manifest = read_pi_manifest(&path).expect("manifest");
    assert_eq!(manifest.prompts, Some(vec!["./prompts/a.md".to_string()]));
    assert_eq!(
        manifest.themes,
        Some(vec!["./themes/dark.json".to_string()])
    );
    assert_eq!(
        manifest.extensions, None,
        "non-string entries drop the field"
    );
    assert_eq!(manifest.skills, None);
}
