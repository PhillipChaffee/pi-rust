//! The facet-bundle suite, ported from upstream `test/bundle.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream bundles facet entries with esbuild into content-addressed
//! `CommonJS` files and executes them on Node with `node:vm`; the port's
//! loader seam is the Rust-native extension mechanism's to implement, so
//! the bundling pipeline runs over built sources and a fixture
//! [`ProgramModuleHost`] stands in for the seam's module host. The
//! fixture's module language is a JSON program the built source spells:
//! `{"facets":[{"id":"..","setup":true,"generation":".."}]}`. A facet
//! carrying a `generation` provides the generation service with a `read`
//! method answering `generation:<g>`, upstream's compiled
//! `env.provide(Value, { read() { return decorate("..") } })` with the
//! helper module inlined.
//!
//! Restatements:
//!
//! - Upstream's compile-artifact assertions (the `require(...)` rewriting,
//!   the `module.exports` shape, and the `.cjs.map` names esbuild emits
//!   under `sourceMap: true`) are products of the build step the seam
//!   replaces. Over built sources the written artifact text is the built
//!   source itself, and a source map rides the entry source the build step
//!   reports.
//! - Upstream's per-entry `externalImports` is the metafile report of the
//!   imports esbuild left external (peer dependencies plus
//!   `chord.external`). That import scan belongs to the build step, so the
//!   package pipeline over built sources records none; the loader seam's
//!   `resolve_external` sees exactly what the manifest carries.
//! - "did not export a module" and "has no setup function" are the JS
//!   module protocol's rejections; the fixture host restates them for its
//!   program language, and the portable halves ("exported no facets",
//!   "duplicate facet IDs") run through `facets_from_module`.
//!
//! Extra cases beyond the five upstream ones pin the manifest/artifact
//! validation contract and the package-configuration rejections the
//! bundling surface carries, closing the crate's coverage gate.

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]
#![allow(
    clippy::too_many_lines,
    reason = "the case bodies are the upstream suite ported 1:1; splitting them would obscure the mapping"
)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pi_chord::api::{create_facet_host, define_local_service};
use pi_chord::bundler::{
    BundleFacetPackageOptions, BundleFacetsOptions, bundle_facet_package, bundle_facets,
};
use pi_chord::context::Context;
use pi_chord::context::background_context;
use pi_chord::errors::ChordError;
use pi_chord::facets::host::{FacetEnvironment, FacetKernelOptions};
use pi_chord::handle::{ServiceImplementation, ServiceView, no_error_reporter, sync_method};
use pi_chord::node::manifest::{
    FACET_BUNDLE_ARTIFACT_FORMAT, FACET_BUNDLE_ARTIFACT_FORMAT_VERSION, integrity_digest,
    parse_integrity, resolve_bundle_file, verify_source,
};
use pi_chord::node::{
    FacetBundleArtifact, FacetBundleArtifactLoaderOptions, FacetBundleEntry,
    FacetBundleExternalResolver, FacetBundleLoaderOptions, FacetBundlePlugin, FacetEntrySource,
    FacetModuleHost, create_facet_bundle_artifact_loader, create_facet_bundle_loader,
    read_facet_bundle_artifact, read_facet_bundle_manifest,
};
use pi_chord::types::{FacetDef, FacetLoader, JsonValue, LoadedFacets, Service};

fn error_message(error: &ChordError) -> String {
    error.to_string()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temporary directory")
}

/// Builds one facet, upstream's `defineFacet`.
fn facet(id: &str, setup: impl Fn(&mut FacetEnvironment) + 'static) -> FacetDef {
    FacetDef {
        id: id.to_string(),
        setup: Rc::new(setup),
    }
}

/// The local generation service upstream spells
/// `defineService("test.bundle.generation", { local: true })`.
fn generation_service() -> Service {
    define_local_service("test.bundle.generation").expect("not reserved")
}

/// The built module program the fixture host executes for a facet that
/// provides the generation service, upstream's entry module importing the
/// helper and providing `read` through `decorate`.
fn generation_program(id: &str, generation: &str) -> String {
    format!(r#"{{"facets":[{{"id":"{id}","setup":true,"generation":"{generation}"}}]}}"#)
}

/// The built module program for a facet that only declares its identity.
fn facet_program(id: &str) -> String {
    format!(r#"{{"facets":[{{"id":"{id}","setup":true}}]}}"#)
}

/// A module program whose facets list is empty, upstream's module exporting
/// no facets.
fn empty_program() -> String {
    r#"{"facets":[]}"#.to_string()
}

/// A module program carrying two facets with the same ID.
fn duplicate_program() -> String {
    r#"{"facets":[{"id":"twin","setup":true},{"id":"twin","setup":true}]}"#.to_string()
}

/// A module program with a facet that has no setup, upstream's
/// `export default { id: 'missing-setup' }`.
fn setupless_program() -> String {
    r#"{"facets":[{"id":"missing-setup"}]}"#.to_string()
}

/// The source map text the build step would emit.
fn map_source() -> String {
    r#"{"version":3}"#.to_string()
}

fn entry(
    name: &str,
    text: String,
    external_imports: &[&str],
    source_map: Option<String>,
) -> (String, FacetEntrySource) {
    (
        name.to_string(),
        FacetEntrySource {
            text,
            external_imports: external_imports.iter().map(ToString::to_string).collect(),
            source_map,
        },
    )
}

fn facet_ids(loaded: &LoadedFacets) -> Vec<String> {
    loaded.facets.iter().map(|facet| facet.id.clone()).collect()
}

fn entry_names(manifest: &pi_chord::node::FacetBundleManifest) -> Vec<String> {
    manifest
        .entries
        .iter()
        .map(|(name, _)| name.clone())
        .collect()
}

/// Counts the directory entries ending in `suffix`; an empty suffix counts
/// every entry, upstream's `readdir(...)` length check.
fn count_entries(directory: &Path, suffix: &str) -> usize {
    std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .filter_map(Result::ok)
        .filter(|file| file.file_name().to_string_lossy().ends_with(suffix))
        .count()
}

/// Asserts the content-addressed entry file name the manifest records:
/// `facet-<12 lowercase hex>-<20 uppercase hex>.cjs`, upstream's
/// `^facet-[a-f0-9]{12}-[A-Z0-9]+\.cjs$`.
fn assert_content_addressed(file: &str) {
    let is_cjs = Path::new(file)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("cjs"));
    assert!(
        file.starts_with("facet-") && is_cjs,
        "unexpected entry file name {file}"
    );
    let stem = file.trim_start_matches("facet-").trim_end_matches(".cjs");
    let Some((hash, digest)) = stem.split_once('-') else {
        panic!("unexpected entry file name {file}");
    };
    assert_eq!(hash.len(), 12, "unexpected entry file name {file}");
    assert!(
        hash.bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "unexpected entry file name {file}"
    );
    assert_eq!(digest.len(), 20, "unexpected entry file name {file}");
    assert!(
        digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_lowercase()),
        "unexpected entry file name {file}"
    );
}

/// Reads the generation the consumer facet retained through its service
/// handle, upstream's `retained.read()`.
async fn retained_read(retained: &Rc<RefCell<Option<ServiceView>>>) -> String {
    let view = retained
        .borrow()
        .clone()
        .expect("the consumer retained its handle");
    let answer = view
        .call("read", vec![], background_context())
        .await
        .unwrap_or_else(|error| panic!("read: {error}"));
    match answer {
        Some(JsonValue::Str(text)) => text,
        other => panic!("unexpected read result: {other:?}"),
    }
}

/// What the fixture host observed per load: the manifest's declared
/// external imports and how the application's resolver mapped each.
#[derive(Default)]
struct ObservedLoad {
    declared_externals: Vec<String>,
    resolved_externals: Vec<(String, Option<PathBuf>)>,
}

/// The module seam's test double standing in for the Rust-native extension
/// mechanism's module host, upstream's `node:vm` `CommonJS` execution.
///
/// Its module language is the JSON program the built source spells; a
/// facet item carrying a `generation` provides the generation service with
/// a `read` method answering `generation:<g>`. The host records the
/// declared external imports and how the loader's resolver mapped each,
/// the observations upstream's controlled `require` makes at execution.
#[derive(Clone)]
struct ProgramModuleHost {
    generation_service: Service,
    observed: Rc<RefCell<ObservedLoad>>,
}

fn program_host(generation: Service) -> ProgramModuleHost {
    ProgramModuleHost {
        generation_service: generation,
        observed: Rc::new(RefCell::new(ObservedLoad::default())),
    }
}

/// Shares the fixture host across loaders, the `Rc<dyn FacetModuleHost>`
/// the loader options take.
fn host_rc(host: &ProgramModuleHost) -> Rc<dyn FacetModuleHost> {
    Rc::new(host.clone())
}

impl FacetModuleHost for ProgramModuleHost {
    fn load(
        &self,
        source: &str,
        _module_path: &Path,
        external_imports: &[String],
        resolve_external: Option<&FacetBundleExternalResolver>,
    ) -> Result<Vec<FacetDef>, ChordError> {
        {
            let mut observed = self.observed.borrow_mut();
            observed.declared_externals = external_imports.to_vec();
            observed.resolved_externals = resolve_external.map_or_else(Vec::new, |resolve| {
                external_imports
                    .iter()
                    .map(|specifier| (specifier.clone(), resolve(specifier)))
                    .collect()
            });
        }
        let parsed = pi_chord::node::manifest::parse_json(
            source,
            "Invalid fixture module program".to_string(),
        )
        .map_err(|_| {
            // A source the program language cannot parse restates "did not
            // export a module".
            ChordError::Message("Facet bundle entry did not export a module".to_string())
        })?;
        let Some(items) = parsed
            .as_object()
            .and_then(|object| object.get("facets"))
            .and_then(JsonValue::as_array)
        else {
            // The module protocol's "did not export a module", restated for
            // the fixture's program language.
            return Err(ChordError::Message(
                "Facet bundle entry did not export a module".to_string(),
            ));
        };
        let mut facets = Vec::with_capacity(items.len());
        for item in items {
            let Some(object) = item.as_object().and_then(|object| {
                object
                    .get("id")
                    .and_then(JsonValue::as_str)
                    .map(|id| (id, object))
            }) else {
                return Err(ChordError::Message(
                    "Facet bundle entry did not export a module".to_string(),
                ));
            };
            if !object
                .1
                .get("setup")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
            {
                return Err(ChordError::Message(format!(
                    "Facet module facet {} has no setup function",
                    object.0
                )));
            }
            let generation = object
                .1
                .get("generation")
                .and_then(JsonValue::as_str)
                .map(str::to_string);
            facets.push(FacetDef {
                id: object.0.to_string(),
                setup: Rc::new({
                    let generation_service = self.generation_service.clone();
                    move |env: &mut FacetEnvironment| {
                        if let Some(generation) = &generation {
                            let mut implementation = ServiceImplementation::new();
                            implementation.method(
                                "read",
                                sync_method({
                                    let answer = format!("generation:{generation}");
                                    move |_args: Vec<JsonValue>, _context: &Context| {
                                        Ok(Some(JsonValue::Str(answer.clone())))
                                    }
                                }),
                            );
                            env.provide(&generation_service, implementation)
                                .expect("provide lands");
                        }
                    }
                }),
            });
        }
        Ok(facets)
    }
}

#[test]
fn builds_independent_content_addressed_entries_and_loads_fresh_reloadable_generations() {
    let rt = runtime();
    rt.block_on(async {
        let directory = temp_dir();
        let output_directory = directory.path().join("bundle");
        let generation = generation_service();
        let module_host = program_host(generation.clone());

        let presentation = entry(
            "presentation",
            facet_program("bundle-presentation"),
            &[],
            Some(map_source()),
        );
        let worker = entry(
            "worker",
            generation_program("bundle-provider", "A"),
            &["@earendil-works/chord"],
            Some(map_source()),
        );
        let entries = vec![presentation, worker.clone()];

        let first_build = bundle_facets(BundleFacetsOptions {
            plugin: FacetBundlePlugin {
                id: "test-bundle".to_string(),
                version: Some("1".to_string()),
            },
            entries,
            outdir: output_directory.clone(),
            working_directory: None,
        })
        .await
        .unwrap_or_else(|error| panic!("first build: {error}"));
        let first_entry = first_build.manifest.entry("worker").expect("worker entry");
        assert_content_addressed(&first_entry.file);
        assert_eq!(
            first_entry.source_map.as_deref(),
            Some(format!("{}.map", first_entry.file).as_str())
        );
        assert_eq!(first_entry.external_imports, vec!["@earendil-works/chord"]);
        let presentation_entry = first_build
            .manifest
            .entry("presentation")
            .expect("presentation entry");
        assert_ne!(presentation_entry.file, first_entry.file);
        assert_eq!(count_entries(&output_directory, ".cjs"), 2);
        let first_source = std::fs::read_to_string(output_directory.join(&first_entry.file))
            .expect("entry source");
        // The written artifact is the built source itself; upstream asserted
        // the compiled text's `require` and `module.exports` shape.
        assert_eq!(first_source, worker.1.text);
        let manifest_text = std::fs::read_to_string(&first_build.manifest_path).expect("manifest");
        assert!(manifest_text.ends_with('\n'));

        let presentation_load = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: first_build.manifest_path.clone(),
            entry: "presentation".to_string(),
            resolve_external: None,
            module_host: host_rc(&module_host),
        })
        .load()
        .await
        .unwrap_or_else(|error| panic!("presentation load: {error}"));
        assert_eq!(facet_ids(&presentation_load), vec!["bundle-presentation"]);
        let presentation_dispose = presentation_load.dispose;
        presentation_dispose()
            .await
            .unwrap_or_else(|error| panic!("presentation dispose: {error}"));

        let artifact = read_facet_bundle_artifact(&first_build.manifest_path, "worker")
            .unwrap_or_else(|error| panic!("artifact: {error}"));
        assert_eq!(artifact.entry.file, first_entry.file);
        let materialized = directory.path().join("materialized");
        // Upstream loads a structuredClone of the artifact; the port's loader
        // owns its artifact by move, which is the same transport contract.
        let transported = create_facet_bundle_artifact_loader(FacetBundleArtifactLoaderOptions {
            artifact,
            temporary_directory: Some(materialized.clone()),
            resolve_external: None,
            module_host: host_rc(&module_host),
        })
        .unwrap_or_else(|error| panic!("transported loader: {error}"))
        .load()
        .await
        .unwrap_or_else(|error| panic!("transported load: {error}"));
        assert_eq!(facet_ids(&transported), vec!["bundle-provider"]);
        assert_eq!(count_entries(&materialized, ""), 1);
        let generation_directory = std::fs::read_dir(&materialized)
            .expect("materialized read")
            .find_map(Result::ok)
            .expect("the generation directory")
            .path();
        // The materialized generation carries the module, its source map, and
        // the synthesized manifest, upstream's `materializeArtifact`.
        assert_eq!(count_entries(&generation_directory, ""), 3);
        let transported_dispose = transported.dispose;
        transported_dispose()
            .await
            .unwrap_or_else(|error| panic!("transported dispose: {error}"));
        assert_eq!(count_entries(&materialized, ""), 0);

        let second_build = bundle_facets(BundleFacetsOptions {
            plugin: FacetBundlePlugin {
                id: "test-bundle".to_string(),
                version: Some("1".to_string()),
            },
            entries: vec![
                entry(
                    "presentation",
                    facet_program("bundle-presentation"),
                    &[],
                    Some(map_source()),
                ),
                worker.clone(),
            ],
            outdir: output_directory.clone(),
            working_directory: None,
        })
        .await
        .unwrap_or_else(|error| panic!("second build: {error}"));
        let second_entry = second_build.manifest.entry("worker").expect("worker entry");
        assert_eq!(second_entry.file, first_entry.file);
        assert_eq!(second_entry.integrity, first_entry.integrity);
        assert_eq!(second_entry.external_imports, first_entry.external_imports);
        assert_eq!(second_entry.source_map, first_entry.source_map);

        let loader = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: second_build.manifest_path,
            entry: "worker".to_string(),
            resolve_external: None,
            module_host: host_rc(&module_host),
        });
        let loaded_a = loader
            .load()
            .await
            .unwrap_or_else(|error| panic!("load A: {error}"));
        let loaded_a_copy = loader
            .load()
            .await
            .unwrap_or_else(|error| panic!("load A copy: {error}"));
        // Each load runs the module again, upstream's `loadedACopy.facets[0]
        // !== loadedA.facets[0]` identity check.
        assert!(!Rc::ptr_eq(
            &loaded_a.facets[0].setup,
            &loaded_a_copy.facets[0].setup
        ));
        let LoadedFacets {
            facets: a_facets,
            dispose: a_dispose,
        } = loaded_a;
        let copy_dispose = loaded_a_copy.dispose;
        copy_dispose()
            .await
            .unwrap_or_else(|error| panic!("copy dispose: {error}"));

        let retained: Rc<RefCell<Option<ServiceView>>> = Rc::new(RefCell::new(None));
        let consumer = facet("bundle-consumer", {
            let retained = retained.clone();
            let generation = generation.clone();
            move |env: &mut FacetEnvironment| {
                let view = env
                    .use_service(&generation)
                    .expect("the consumer's handle binds");
                retained.borrow_mut().replace(view);
            }
        });
        let host = create_facet_host(FacetKernelOptions {
            facets: std::iter::once(consumer).chain(a_facets).collect(),
            service_sources: Vec::new(),
            on_error: no_error_reporter(),
        })
        .await
        .unwrap_or_else(|error| panic!("host: {error}"));
        assert_eq!(retained_read(&retained).await, "generation:A");

        let third_build = bundle_facets(BundleFacetsOptions {
            plugin: FacetBundlePlugin {
                id: "test-bundle".to_string(),
                version: Some("2".to_string()),
            },
            entries: vec![
                entry(
                    "presentation",
                    facet_program("bundle-presentation"),
                    &[],
                    Some(map_source()),
                ),
                entry(
                    "worker",
                    generation_program("bundle-provider", "B"),
                    &["@earendil-works/chord"],
                    Some(map_source()),
                ),
            ],
            outdir: output_directory.clone(),
            working_directory: None,
        })
        .await
        .unwrap_or_else(|error| panic!("third build: {error}"));
        let third_entry = third_build.manifest.entry("worker").expect("worker entry");
        assert_ne!(third_entry.file, first_entry.file);
        let loaded_b = loader
            .load()
            .await
            .unwrap_or_else(|error| panic!("load B: {error}"));
        let LoadedFacets {
            facets: b_facets,
            dispose: b_dispose,
        } = loaded_b;
        host.reload(b_facets)
            .await
            .unwrap_or_else(|error| panic!("reload: {error}"));
        a_dispose()
            .await
            .unwrap_or_else(|error| panic!("dispose A: {error}"));
        assert_eq!(retained_read(&retained).await, "generation:B");

        host.dispose()
            .await
            .unwrap_or_else(|error| panic!("dispose: {error}"));
        b_dispose()
            .await
            .unwrap_or_else(|error| panic!("dispose B: {error}"));
    });
}

#[test]
fn loads_host_externals_through_the_resolver_the_loading_application_supplies() {
    let rt = runtime();
    rt.block_on(async {
        let directory = temp_dir();
        let external_path = directory.path().join("host.facet-module");
        std::fs::write(&external_path, facet_program("host-module")).expect("host module");

        let result = bundle_facets(BundleFacetsOptions {
            plugin: FacetBundlePlugin {
                id: "external-bundle".to_string(),
                version: None,
            },
            entries: vec![entry(
                "worker",
                facet_program("external-facet"),
                &["@example/host", "@example/dynamic"],
                None,
            )],
            outdir: directory.path().join("bundle"),
            working_directory: None,
        })
        .await
        .unwrap_or_else(|error| panic!("build: {error}"));

        // Upstream asserted the compiled source rewrote `import(...)` into
        // `require("@example/dynamic")`; that rewriting is the esbuild
        // build's product, and the resolver is what the seam consults.
        let resolve: FacetBundleExternalResolver = Rc::new({
            let external_path = external_path.clone();
            move |specifier: &str| {
                (specifier == "@example/host" || specifier == "@example/dynamic")
                    .then(|| external_path.clone())
            }
        });
        let module_host = program_host(generation_service());
        let loaded = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: result.manifest_path,
            entry: "worker".to_string(),
            resolve_external: Some(resolve),
            module_host: host_rc(&module_host),
        })
        .load()
        .await
        .unwrap_or_else(|error| panic!("load: {error}"));
        assert_eq!(facet_ids(&loaded), vec!["external-facet"]);
        let (declared, resolved) = {
            let observed = module_host.observed.borrow();
            (
                observed.declared_externals.clone(),
                observed.resolved_externals.clone(),
            )
        };
        assert_eq!(
            declared,
            vec!["@example/dynamic".to_string(), "@example/host".to_string()]
        );
        assert_eq!(
            resolved,
            vec![
                ("@example/dynamic".to_string(), Some(external_path.clone())),
                ("@example/host".to_string(), Some(external_path.clone())),
            ]
        );
        let dispose = loaded.dispose;
        dispose()
            .await
            .unwrap_or_else(|error| panic!("dispose: {error}"));
    });
}

#[test]
fn builds_plugin_packages_from_conventional_and_configured_facet_entries() {
    let rt = runtime();
    rt.block_on(async {
        let directory = temp_dir();
        let source_directory = directory.path().join("src");
        std::fs::create_dir(&source_directory).expect("source directory");
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"name":"@example/conventional-plugin","version":"1.2.3","peerDependencies":{"@example/host":"^1.0.0"}}"#,
        )
        .expect("package metadata");
        // The package reader consumes built sources; upstream's `.ts` entry
        // files stand in as the built modules the fixture host executes.
        std::fs::write(
            source_directory.join("session.ts"),
            facet_program("package-session"),
        )
        .expect("session entry");
        std::fs::write(source_directory.join("tui.ts"), facet_program("package-tui"))
            .expect("tui entry");
        std::fs::write(source_directory.join("contract.ts"), "export const ignored = true;\n")
            .expect("ignored file");
        std::fs::write(
            source_directory.join("presentation.ts"),
            facet_program("configured-tui"),
        )
        .expect("presentation entry");

        let conventional = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().to_path_buf(),
            outdir: directory.path().join("build"),
            default_facets: vec![
                ("session".to_string(), "src/session.ts".to_string()),
                ("tui".to_string(), "src/tui.ts".to_string()),
                ("browser".to_string(), "src/browser.ts".to_string()),
            ],
        })
        .await
        .unwrap_or_else(|error| panic!("conventional: {error}"));
        assert_eq!(
            conventional.package_directory,
            std::fs::canonicalize(directory.path()).expect("canonical directory")
        );
        assert_eq!(conventional.manifest.plugin.id, "@example/conventional-plugin");
        assert_eq!(conventional.manifest.plugin.version.as_deref(), Some("1.2.3"));
        assert_eq!(entry_names(&conventional.manifest), vec!["session", "tui"]);
        // Restatement: upstream records esbuild's external marking for the
        // peer dependency the session imports; over built sources the
        // package reader scans no imports and records none.
        assert!(
            conventional
                .manifest
                .entry("session")
                .expect("session entry")
                .external_imports
                .is_empty()
        );
        assert!(
            conventional
                .manifest
                .entry("tui")
                .expect("tui entry")
                .source_map
                .is_none()
        );

        std::fs::write(
            directory.path().join("package.json"),
            r#"{"name":"@example/conventional-plugin","version":"2.0.0","chord":{"facets":{"session":false,"tui":"src/presentation.ts"},"sourceMap":false}}"#,
        )
        .expect("configured package metadata");
        let configured = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().join("package.json"),
            outdir: directory.path().join("build"),
            default_facets: vec![
                ("session".to_string(), "src/session.ts".to_string()),
                ("tui".to_string(), "src/tui.ts".to_string()),
            ],
        })
        .await
        .unwrap_or_else(|error| panic!("configured: {error}"));
        assert_eq!(entry_names(&configured.manifest), vec!["tui"]);
        assert!(
            configured
                .manifest
                .entry("tui")
                .expect("tui entry")
                .source_map
                .is_none()
        );
        let module_host = program_host(generation_service());
        let loaded = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: configured.manifest_path,
            entry: "tui".to_string(),
            resolve_external: None,
            module_host: host_rc(&module_host),
        })
        .load()
        .await
        .unwrap_or_else(|error| panic!("load: {error}"));
        assert_eq!(facet_ids(&loaded), vec!["configured-tui"]);
        let dispose = loaded.dispose;
        dispose()
            .await
            .unwrap_or_else(|error| panic!("dispose: {error}"));
    });
}

#[test]
fn rejects_invalid_plugin_package_entry_configuration() {
    let rt = runtime();
    rt.block_on(async {
        // Upstream's `chord.facets` entry pointing outside the package, the
        // shape check that fires before the file need exist.
        let directory = temp_dir();
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"facets":{"tui":"../outside.ts"}}}"#,
        )
        .expect("package metadata");
        let error = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().to_path_buf(),
            outdir: directory.path().join("build"),
            default_facets: Vec::new(),
        })
        .await
        .expect_err("an escaping configured entry rejects");
        assert!(error_message(&error).contains("escapes the package directory"));

        // The same shape through the application's default conventions.
        let directory = temp_dir();
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"name":"invalid-plugin","version":"1.0.0"}"#,
        )
        .expect("package metadata");
        let error = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().to_path_buf(),
            outdir: directory.path().join("build"),
            default_facets: vec![("tui".to_string(), "../outside.ts".to_string())],
        })
        .await
        .expect_err("an escaping default entry rejects");
        assert!(error_message(&error).contains("escapes the package directory"));

        // An entry resolving to the package directory itself.
        let directory = temp_dir();
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"facets":{"tui":"."}}}"#,
        )
        .expect("package metadata");
        let error = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().to_path_buf(),
            outdir: directory.path().join("build"),
            default_facets: Vec::new(),
        })
        .await
        .expect_err("a self-resolving entry rejects");
        assert!(error_message(&error).contains("escapes the package directory"));

        // An absolute source.
        let directory = temp_dir();
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"facets":{"tui":"/outside.ts"}}}"#,
        )
        .expect("package metadata");
        let error = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().to_path_buf(),
            outdir: directory.path().join("build"),
            default_facets: Vec::new(),
        })
        .await
        .expect_err("an absolute entry rejects");
        assert!(error_message(&error).contains("must be relative to the package directory"));

        // A symlink resolving outside the package directory, the canonical
        // check upstream runs after the lexical one.
        let directory = temp_dir();
        let package_directory = directory.path().join("plugin");
        std::fs::create_dir(&package_directory).expect("package directory");
        std::fs::write(
            package_directory.join("package.json"),
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"facets":{"tui":"link.ts"}}}"#,
        )
        .expect("package metadata");
        std::fs::write(
            directory.path().join("outside.ts"),
            "export const outside = true;\n",
        )
        .expect("outside file");
        std::os::unix::fs::symlink(
            directory.path().join("outside.ts"),
            package_directory.join("link.ts"),
        )
        .expect("escape symlink");
        let error = bundle_facet_package(BundleFacetPackageOptions {
            package_path: package_directory,
            outdir: directory.path().join("build"),
            default_facets: Vec::new(),
        })
        .await
        .expect_err("a symlinked escape rejects");
        assert!(error_message(&error).contains("resolves outside the package directory"));
    });
}

#[test]
fn rejects_invalid_package_metadata_and_chord_configuration() {
    let rt = runtime();
    rt.block_on(async {
        // Invalid peerDependencies: the version must be a string.
        let error = bundle_package_with_metadata(
            r#"{"name":"invalid-plugin","version":"1.0.0","peerDependencies":{"@example/host":1}}"#,
            Vec::new(),
        )
        .await
        .expect_err("an invalid peerDependencies record rejects");
        assert!(error_message(&error).contains("invalid peerDependencies"));

        // The chord configuration must be an object.
        let error = bundle_package_with_metadata(
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":true}"#,
            Vec::new(),
        )
        .await
        .expect_err("a non-object chord configuration rejects");
        assert!(error_message(&error).contains("chord configuration must be an object"));

        // Unknown chord fields reject.
        let error = bundle_package_with_metadata(
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"typo":{}}}"#,
            Vec::new(),
        )
        .await
        .expect_err("an unknown chord field rejects");
        assert!(error_message(&error).contains("chord configuration has an unknown field"));

        // chord.facets must be an object.
        let error = bundle_package_with_metadata(
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"facets":true}}"#,
            Vec::new(),
        )
        .await
        .expect_err("a non-object chord.facets rejects");
        assert!(error_message(&error).contains("chord.facets must be an object"));

        // Every invalid chord.facets entry rejects: empty source, non-string
        // source, and empty name.
        for facets in ["{\"tui\":\"\"}", "{\"tui\":1}", "{\"\":\"src/tui.ts\"}"] {
            let metadata = format!(
                r#"{{"name":"invalid-plugin","version":"1.0.0","chord":{{"facets":{facets}}}}}"#
            );
            let error = bundle_package_with_metadata(&metadata, Vec::new())
                .await
                .expect_err("an invalid chord.facets entry rejects");
            assert!(error_message(&error).contains("invalid chord.facets entry"));
        }

        // chord.external must be an array of non-empty strings.
        for external in ["\"src/tui.ts\"", "[\"\"]", "[1]"] {
            let metadata = format!(
                r#"{{"name":"invalid-plugin","version":"1.0.0","chord":{{"external":{external}}}}}"#
            );
            let error = bundle_package_with_metadata(&metadata, Vec::new())
                .await
                .expect_err("an invalid chord.external rejects");
            assert!(
                error_message(&error).contains("chord.external must contain non-empty strings")
            );
        }

        // chord.sourceMap must be a boolean.
        let error = bundle_package_with_metadata(
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"sourceMap":"yes"}}"#,
            Vec::new(),
        )
        .await
        .expect_err("a non-boolean chord.sourceMap rejects");
        assert!(error_message(&error).contains("chord.sourceMap must be a boolean"));

        // Empty facet mappings reject before the entries resolve.
        let error = bundle_package_with_metadata(
            r#"{"name":"invalid-plugin","version":"1.0.0"}"#,
            vec![(String::new(), "src/tui.ts".to_string())],
        )
        .await
        .expect_err("an empty default entry name rejects");
        assert!(error_message(&error).contains("default entry name must not be empty"));
        let error = bundle_package_with_metadata(
            r#"{"name":"invalid-plugin","version":"1.0.0"}"#,
            vec![("tui".to_string(), String::new())],
        )
        .await
        .expect_err("an empty default entry source rejects");
        assert!(error_message(&error).contains("entry tui must have a source path"));
    });
}

/// Stands up one plugin package carrying `metadata` as its package.json and
/// bundles it with `default_facets`, the per-assertion fixture of the
/// configuration-rejection cases.
async fn bundle_package_with_metadata(
    metadata: &str,
    default_facets: Vec<(String, String)>,
) -> Result<pi_chord::bundler::BundleFacetPackageResult, ChordError> {
    let directory = temp_dir();
    std::fs::write(directory.path().join("package.json"), metadata).expect("package metadata");
    bundle_facet_package(BundleFacetPackageOptions {
        package_path: directory.path().to_path_buf(),
        outdir: directory.path().join("build"),
        default_facets,
    })
    .await
}

#[test]
fn rejects_invalid_bundle_options_and_package_paths() {
    let rt = runtime();
    rt.block_on(async {
        // bundle_facets' option validation, one rejection per branch.
        for (options, expected) in [
            (
                BundleFacetsOptions {
                    plugin: FacetBundlePlugin {
                        id: String::new(),
                        version: None,
                    },
                    entries: vec![entry("worker", facet_program("w"), &[], None)],
                    outdir: PathBuf::from("build"),
                    working_directory: None,
                },
                "plugin ID must not be empty",
            ),
            (
                BundleFacetsOptions {
                    plugin: FacetBundlePlugin {
                        id: "p".to_string(),
                        version: Some(String::new()),
                    },
                    entries: vec![entry("worker", facet_program("w"), &[], None)],
                    outdir: PathBuf::from("build"),
                    working_directory: None,
                },
                "plugin version must not be empty",
            ),
            (
                BundleFacetsOptions {
                    plugin: FacetBundlePlugin {
                        id: "p".to_string(),
                        version: None,
                    },
                    entries: Vec::new(),
                    outdir: PathBuf::from("build"),
                    working_directory: None,
                },
                "at least one entry",
            ),
            (
                BundleFacetsOptions {
                    plugin: FacetBundlePlugin {
                        id: "p".to_string(),
                        version: None,
                    },
                    entries: vec![entry("", facet_program("w"), &[], None)],
                    outdir: PathBuf::from("build"),
                    working_directory: None,
                },
                "entry name must not be empty",
            ),
            (
                BundleFacetsOptions {
                    plugin: FacetBundlePlugin {
                        id: "p".to_string(),
                        version: None,
                    },
                    entries: vec![entry("worker", String::new(), &[], None)],
                    outdir: PathBuf::from("build"),
                    working_directory: None,
                },
                "must have a source",
            ),
        ] {
            let error = bundle_facets(options)
                .await
                .expect_err("invalid options reject");
            assert!(
                error_message(&error).contains(expected),
                "expected {expected:?}, got: {}",
                error_message(&error)
            );
        }

        // A path that is not a directory or package.json.
        let directory = temp_dir();
        std::fs::write(directory.path().join("notes.txt"), "not a package").expect("plain file");
        let error = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().join("notes.txt"),
            outdir: directory.path().join("build"),
            default_facets: Vec::new(),
        })
        .await
        .expect_err("a non-package path rejects");
        assert!(error_message(&error).contains("must name a directory or package.json"));

        // A package directory without readable metadata.
        let error = bundle_package_with_metadata_text("", Vec::new())
            .await
            .expect_err("unreadable metadata rejects");
        assert!(error_message(&error).contains("Could not read facet package metadata"));

        // Malformed and non-object metadata.
        let error = bundle_package_with_metadata_text("{", Vec::new())
            .await
            .expect_err("malformed metadata rejects");
        assert!(error_message(&error).contains("Could not read facet package metadata"));
        let error = bundle_package_with_metadata_text("[]", Vec::new())
            .await
            .expect_err("non-object metadata rejects");
        assert!(error_message(&error).contains("metadata must be an object"));

        // Missing name and version.
        let error = bundle_package_with_metadata_text(r#"{"version":"1"}"#, Vec::new())
            .await
            .expect_err("a missing name rejects");
        assert!(error_message(&error).contains("non-empty name"));
        let error = bundle_package_with_metadata_text(r#"{"name":"p"}"#, Vec::new())
            .await
            .expect_err("a missing version rejects");
        assert!(error_message(&error).contains("non-empty version"));

        // No configured or conventional facet entries.
        let error = bundle_package_with_metadata_text(r#"{"name":"p","version":"1"}"#, Vec::new())
            .await
            .expect_err("a package without entries rejects");
        assert!(error_message(&error).contains("has no configured or conventional facet entries"));

        // A configured entry that does not exist.
        let error = bundle_package_with_metadata_text(
            r#"{"name":"p","version":"1","chord":{"facets":{"tui":"src/absent.ts"}}}"#,
            Vec::new(),
        )
        .await
        .expect_err("a missing configured entry rejects");
        assert!(error_message(&error).contains("Could not access configured facet entry"));

        // A default entry whose path is a directory.
        let directory = temp_dir();
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"name":"p","version":"1"}"#,
        )
        .expect("package metadata");
        std::fs::create_dir(directory.path().join("src")).expect("directory entry");
        let error = bundle_facet_package(BundleFacetPackageOptions {
            package_path: directory.path().to_path_buf(),
            outdir: directory.path().join("build"),
            default_facets: vec![("tui".to_string(), "src".to_string())],
        })
        .await
        .expect_err("a directory entry rejects");
        assert!(error_message(&error).contains("Could not read default facet entry"));
    });
}

/// Stands up one plugin package directory carrying `metadata` as its
/// package.json and bundles it with `default_facets`.
async fn bundle_package_with_metadata_text(
    metadata: &str,
    default_facets: Vec<(String, String)>,
) -> Result<pi_chord::bundler::BundleFacetPackageResult, ChordError> {
    let directory = temp_dir();
    std::fs::create_dir_all(directory.path().join("src")).expect("source directory");
    if !metadata.is_empty() {
        std::fs::write(directory.path().join("package.json"), metadata).expect("package metadata");
    }
    bundle_facet_package(BundleFacetPackageOptions {
        package_path: directory.path().to_path_buf(),
        outdir: directory.path().join("build"),
        default_facets,
    })
    .await
}

#[test]
fn rejects_corrupt_entries_and_invalid_module_exports() {
    let rt = runtime();
    rt.block_on(async {
        let directory = temp_dir();
        let output_directory = directory.path().join("bundle");
        let module_host = program_host(generation_service());

        // Upstream's setup-less module export; the fixture host restates the
        // JS module protocol's rejection for its program language.
        let loader = build_invalid_loader(
            "invalid",
            setupless_program(),
            &module_host,
            &output_directory,
        )
        .await;
        let error = loader
            .load()
            .await
            .expect_err("a setup-less module export rejects");
        assert!(error_message(&error).contains("has no setup function"));

        // The portable halves of the module-protocol rejections run through
        // `facets_from_module`: an empty export and duplicate facet IDs.
        let loader =
            build_invalid_loader("empty", empty_program(), &module_host, &output_directory).await;
        let error = loader
            .load()
            .await
            .expect_err("an export without facets rejects");
        assert!(error_message(&error).contains("exported no facets"));

        let loader = build_invalid_loader(
            "duplicate",
            duplicate_program(),
            &module_host,
            &output_directory,
        )
        .await;
        let error = loader
            .load()
            .await
            .expect_err("an export with duplicate facet IDs rejects");
        assert!(error_message(&error).contains("exports duplicate facet IDs"));

        // A program outside the fixture's module language restates "did not
        // export a module".
        let loader = build_invalid_loader(
            "shapeless",
            "not a module".to_string(),
            &module_host,
            &output_directory,
        )
        .await;
        let error = loader
            .load()
            .await
            .expect_err("a shapeless module export rejects");
        assert!(error_message(&error).contains("did not export a module"));

        // An exported facet carrying an empty ID rejects at the extraction
        // check, upstream's facetsFromModule ID guard.
        let loader = build_invalid_loader(
            "unnamed",
            r#"{"facets":[{"id":"","setup":true}]}"#.to_string(),
            &module_host,
            &output_directory,
        )
        .await;
        let error = loader.load().await.expect_err("an unnamed facet rejects");
        assert!(error_message(&error).contains("has a facet with an invalid ID"));
        let loader = build_invalid_loader(
            "shapeless",
            "not a module".to_string(),
            &module_host,
            &output_directory,
        )
        .await;
        let error = loader
            .load()
            .await
            .expect_err("a shapeless module export rejects");
        assert!(error_message(&error).contains("did not export a module"));

        // Corruption after the manifest is written: the loader verifies the
        // entry's integrity before the module host runs.
        let result = bundle_facets(BundleFacetsOptions {
            plugin: FacetBundlePlugin {
                id: "invalid-bundle".to_string(),
                version: None,
            },
            entries: vec![entry("invalid", facet_program("replacement"), &[], None)],
            outdir: output_directory.clone(),
            working_directory: None,
        })
        .await
        .unwrap_or_else(|error| panic!("build: {error}"));
        let loader = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: result.manifest_path.clone(),
            entry: "invalid".to_string(),
            resolve_external: None,
            module_host: host_rc(&module_host),
        });
        let manifest = read_facet_bundle_manifest(&result.manifest_path).expect("manifest");
        let bundle_entry = manifest.entry("invalid").expect("invalid entry");
        std::fs::write(output_directory.join(&bundle_entry.file), empty_program())
            .expect("corruption");
        let error = loader.load().await.expect_err("a corrupted entry rejects");
        assert!(error_message(&error).contains("integrity check failed"));
    });
}

/// Bundles one `invalid-bundle` entry with the given program text and
/// returns a loader for it, the shared fixture of the invalid-export case.
async fn build_invalid_loader(
    name: &str,
    text: String,
    module_host: &ProgramModuleHost,
    output_directory: &Path,
) -> pi_chord::node::FacetBundleLoader {
    let result = bundle_facets(BundleFacetsOptions {
        plugin: FacetBundlePlugin {
            id: "invalid-bundle".to_string(),
            version: None,
        },
        entries: vec![entry(name, text, &[], None)],
        outdir: output_directory.to_path_buf(),
        working_directory: None,
    })
    .await
    .unwrap_or_else(|error| panic!("build: {error}"));
    create_facet_bundle_loader(FacetBundleLoaderOptions {
        manifest_path: result.manifest_path,
        entry: name.to_string(),
        resolve_external: None,
        module_host: Rc::new(module_host.clone()),
    })
}

#[test]
fn validates_manifests_and_artifacts_before_loading() {
    let rt = runtime();
    rt.block_on(async {
        let directory = temp_dir();

        // validate_manifest's rejections, one per branch, over hand-written
        // manifest records.
        let cases: Vec<(String, &str)> = vec![
            ("[]".to_string(), "Invalid facet bundle manifest format"),
            (
                r#"{"format":"other","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#.to_string(),
                "Invalid facet bundle manifest format",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":1,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#.to_string(),
                "Unsupported facet bundle manifest version",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#.to_string(),
                "invalid plugin identity",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":""},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#.to_string(),
                "invalid plugin identity",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p","version":""},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#.to_string(),
                "invalid plugin version",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"}}"#.to_string(),
                "has no entries",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{}}"#.to_string(),
                "has no entries",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"integrity":"sha256-AQID","externalImports":[]}}}"#.to_string(),
                "entry e has no file",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"a/b.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#.to_string(),
                "must be a filename relative to its manifest",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","externalImports":[]}}}"#.to_string(),
                "entry e has no integrity",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"md5-AQID","externalImports":[]}}}"#.to_string(),
                "invalid SHA-256 integrity value",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID"}}}"#.to_string(),
                "has invalid external imports",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[1]}}}"#.to_string(),
                "has invalid external imports",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":["a","a"]}}}"#.to_string(),
                "duplicate external imports",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[],"sourceMap":3}}}"#.to_string(),
                "has an invalid source map",
            ),
            (
                r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[],"sourceMap":"a/b.map"}}}"#.to_string(),
                "must be a filename relative to its manifest",
            ),
        ];
        for (text, expected) in cases {
            let path = directory.path().join("chord-facets.json");
            std::fs::write(&path, text).expect("manifest fixture");
            let error = read_facet_bundle_manifest(&path)
                .expect_err("an invalid manifest rejects");
            assert!(
                error_message(&error).contains(expected),
                "expected {expected:?}, got: {}",
                error_message(&error)
            );
        }
        // A manifest record that is not an object at all.
        let path = directory.path().join("chord-facets.json");
        std::fs::write(&path, "null").expect("manifest fixture");
        let error = read_facet_bundle_manifest(&path).expect_err("a null manifest rejects");
        assert!(error_message(&error).contains("Invalid facet bundle manifest format"));
        // An entry record whose name is empty.
        let path = directory.path().join("chord-facets.json");
        std::fs::write(
            &path,
            r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p"},"entries":{"":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#,
        )
        .expect("manifest fixture");
        let error = read_facet_bundle_manifest(&path).expect_err("an unnamed entry rejects");
        assert!(error_message(&error).contains("has an invalid entry"));
        // A plugin version that is not a string.
        let path = directory.path().join("chord-facets.json");
        std::fs::write(
            &path,
            r#"{"format":"chord.facet-bundle","formatVersion":2,"plugin":{"id":"p","version":3},"entries":{"e":{"file":"x.cjs","integrity":"sha256-AQID","externalImports":[]}}}"#,
        )
        .expect("manifest fixture");
        let error =
            read_facet_bundle_manifest(&path).expect_err("a non-string version rejects");
        assert!(error_message(&error).contains("invalid plugin version"));

        // A valid manifest reads.
        let source = "the built source";
        let entry_file = "facet-worker.cjs";
        std::fs::write(directory.path().join(entry_file), source).expect("entry file");
        let path = directory.path().join("chord-facets.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"format":"chord.facet-bundle","formatVersion":2,"plugin":{{"id":"p"}},"entries":{{"worker":{{"file":"{entry_file}","integrity":"sha256-{}","externalImports":["ext"]}}}}}}"#,
                integrity_digest(source.as_bytes()),
            ),
        )
        .expect("manifest fixture");
        let manifest = read_facet_bundle_manifest(&path).expect("a valid manifest reads");
        assert_eq!(entry_names(&manifest), vec!["worker"]);

        // read_facet_bundle_manifest's reader errors.
        let missing = read_facet_bundle_manifest(&directory.path().join("absent.json"))
            .expect_err("a missing manifest rejects");
        assert!(error_message(&missing).contains("Could not read facet bundle manifest"));
        std::fs::write(directory.path().join("malformed.json"), "{").expect("malformed fixture");
        let error = read_facet_bundle_manifest(&directory.path().join("malformed.json"))
            .expect_err("a malformed manifest rejects");
        assert!(error_message(&error).contains("Could not read facet bundle manifest"));

        // read_facet_bundle_artifact's guards.
        let error = read_facet_bundle_artifact(&path, "")
            .expect_err("an empty entry name rejects");
        assert!(error_message(&error).contains("entry name must not be empty"));
        let error = read_facet_bundle_artifact(&path, "absent")
            .expect_err("an unknown entry rejects");
        assert!(error_message(&error).contains("has no entry named"));
        // The entry file is gone.
        std::fs::remove_file(directory.path().join(entry_file)).expect("removal");
        let error = read_facet_bundle_artifact(&path, "worker")
            .expect_err("a missing entry file rejects");
        assert!(error_message(&error).contains("Could not read facet bundle entry"));
        // The source map is declared but gone; the entry file stays.
        std::fs::write(directory.path().join(entry_file), source).expect("entry file");
        std::fs::write(
            &path,
            format!(
                r#"{{"format":"chord.facet-bundle","formatVersion":2,"plugin":{{"id":"p"}},"entries":{{"worker":{{"file":"{entry_file}","integrity":"sha256-{}","externalImports":[],"sourceMap":"absent.map"}}}}}}"#,
                integrity_digest(source.as_bytes()),
            ),
        )
        .expect("manifest fixture");
        let error = read_facet_bundle_artifact(&path, "worker")
            .expect_err("a missing source map rejects");
        assert!(error_message(&error).contains("Could not read facet bundle source map"));

        // verifySource's digest mismatch, and the integrity helpers.
        let entry_record = manifest.entry("worker").expect("worker entry").clone();
        let error =
            verify_source("tampered", &entry_record).expect_err("a tampered source rejects");
        assert!(error_message(&error).contains("integrity check failed"));
        assert!(verify_source(source, &entry_record).is_ok());
        assert_eq!(
            integrity_digest(b""),
            "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="
        );
        assert_eq!(parse_integrity("sha256-AQID").expect("integrity"), "AQID");
        let error = parse_integrity("sha256-")
            .expect_err("an integrity value without a digest rejects");
        assert!(error_message(&error).contains("invalid SHA-256 integrity value"));
        let error = parse_integrity("b256-AQID")
            .expect_err("an integrity value without the prefix rejects");
        assert!(error_message(&error).contains("invalid SHA-256 integrity value"));
        for file in ["", "/x.cjs", ".", "..", "a/b.cjs", "a\\b.cjs"] {
            assert!(
                resolve_bundle_file(file).is_err(),
                "an invalid bundle filename rejects: {file:?}"
            );
        }
        assert!(resolve_bundle_file("facet-a.cjs").is_ok());

        // The artifact loader validates at construction, upstream's
        // `validateArtifact`.
        let module_host = program_host(generation_service());
        let hand_artifact = |source: &str, source_map: Option<&str>, contents: Option<&str>| {
            FacetBundleArtifact {
                format: FACET_BUNDLE_ARTIFACT_FORMAT.to_string(),
                format_version: FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
                plugin: FacetBundlePlugin {
                    id: "artifact-plugin".to_string(),
                    version: None,
                },
                entry_name: "worker".to_string(),
                entry: FacetBundleEntry {
                    file: "facet-worker.cjs".to_string(),
                    integrity: format!("sha256-{}", integrity_digest(source.as_bytes())),
                    external_imports: Vec::new(),
                    source_map: source_map.map(str::to_string),
                },
                source: source.to_string(),
                source_map_contents: contents.map(str::to_string),
            }
        };
        let valid = hand_artifact(facet_program("worker").as_str(), Some("facet-worker.cjs.map"), Some(&map_source()));
        for (artifact, expected) in [
            (
                FacetBundleArtifact {
                    format: "other".to_string(),
                    ..valid.clone()
                },
                "Invalid facet bundle artifact",
            ),
            (
                FacetBundleArtifact {
                    format_version: 1,
                    ..valid.clone()
                },
                "Invalid facet bundle artifact",
            ),
            (
                FacetBundleArtifact {
                    entry_name: String::new(),
                    ..valid.clone()
                },
                "Invalid facet bundle artifact",
            ),
            (
                FacetBundleArtifact {
                    entry: FacetBundleEntry {
                        source_map: None,
                        ..valid.entry.clone()
                    },
                    source_map_contents: Some(map_source()),
                    ..valid.clone()
                },
                "source map contents without a source map",
            ),
            (
                FacetBundleArtifact {
                    source_map_contents: None,
                    ..valid.clone()
                },
                "missing its source map contents",
            ),
            (
                FacetBundleArtifact {
                    source: "tampered".to_string(),
                    ..valid.clone()
                },
                "integrity check failed",
            ),
        ] {
            let error = create_facet_bundle_artifact_loader(FacetBundleArtifactLoaderOptions {
                artifact,
                temporary_directory: None,
                resolve_external: None,
                module_host: host_rc(&module_host),
            })
            .expect_err("an invalid artifact rejects");
            assert!(
                error_message(&error).contains(expected),
                "expected {expected:?}, got: {}",
                error_message(&error)
            );
        }
        // A valid artifact constructs and loads.
        let artifact_loaded = create_facet_bundle_artifact_loader(FacetBundleArtifactLoaderOptions {
            artifact: valid.clone(),
            temporary_directory: None,
            resolve_external: None,
            module_host: host_rc(&module_host),
        })
        .expect("a valid artifact constructs")
        .load()
        .await
        .unwrap_or_else(|error| panic!("artifact load: {error}"));
        assert_eq!(facet_ids(&artifact_loaded), vec!["worker".to_string()]);
        let dispose = artifact_loaded.dispose;
        dispose()
            .await
            .unwrap_or_else(|error| panic!("dispose: {error}"));

        // A module host the program language rejects cleans its generation
        // directory up after the failed load.
        let failing = hand_artifact("not a module", None, None);
        let failed_parent = temp_dir();
        let error = create_facet_bundle_artifact_loader(FacetBundleArtifactLoaderOptions {
            artifact: failing,
            temporary_directory: Some(failed_parent.path().to_path_buf()),
            resolve_external: None,
            module_host: host_rc(&module_host),
        })
        .expect("the artifact itself is valid")
        .load()
        .await
        .expect_err("a host rejection fails the load");
        assert!(error_message(&error).contains("did not export a module"));
        assert_eq!(count_entries(failed_parent.path(), ""), 0);

        // The transport parent must be a directory.
        let not_a_directory = temp_dir();
        std::fs::write(not_a_directory.path().join("occupied"), "a file").expect("file");
        let error = create_facet_bundle_artifact_loader(FacetBundleArtifactLoaderOptions {
            artifact: valid.clone(),
            temporary_directory: Some(not_a_directory.path().join("occupied")),
            resolve_external: None,
            module_host: host_rc(&module_host),
        })
        .expect("construction validates the artifact")
        .load()
        .await
        .expect_err("a file transport parent rejects");
        assert!(error_message(&error).contains("Could not create facet artifact directory"));

        // The loaders and options carry debug surfaces.
        let debug_text = format!("{valid:?}");
        assert!(debug_text.contains("artifact-plugin"));
        let debugged = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: PathBuf::from("manifests/chord-facets.json"),
            entry: "worker".to_string(),
            resolve_external: None,
            module_host: host_rc(&module_host),
        });
        assert!(format!("{debugged:?}").contains("manifests/chord-facets.json"));

        // The manifest loader's entry guards.
        let result = bundle_facets(BundleFacetsOptions {
            plugin: FacetBundlePlugin {
                id: "guard-bundle".to_string(),
                version: None,
            },
            entries: vec![entry("worker", facet_program("guard-facet"), &[], None)],
            outdir: directory.path().join("guards"),
            working_directory: None,
        })
        .await
        .unwrap_or_else(|error| panic!("guard build: {error}"));
        let loader = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: result.manifest_path.clone(),
            entry: "absent".to_string(),
            resolve_external: None,
            module_host: host_rc(&module_host),
        });
        let error = loader
            .load()
            .await
            .expect_err("an unknown entry rejects");
        assert!(error_message(&error).contains("has no entry named"));
        let loader = create_facet_bundle_loader(FacetBundleLoaderOptions {
            manifest_path: result.manifest_path,
            entry: String::new(),
            resolve_external: None,
            module_host: host_rc(&module_host),
        });
        let error = loader
            .load()
            .await
            .expect_err("an empty entry name rejects");
        assert!(error_message(&error).contains("entry name must not be empty"));
    });
}
