//! The test stand-in for upstream's `InlineExtension` test surface, the
//! port of `test/utilities.ts`'s `createTestExtensionsResult` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! One factory per input loads into its own record over a fresh runtime and
//! bus, so a suite can hand the result to a session or assert on the record
//! without driving the resource loader. Labels follow upstream: a bare
//! factory is `<inline:N>`, a named one `<inline:name>`, an explicit path is
//! carried verbatim.

use pi_coding_agent::core::event_bus::EventBus;
use pi_coding_agent::extensions::api::{
    ExtensionFactory, ExtensionRuntime, load_extension_from_factory,
};
use pi_coding_agent::resource_loader::{ExtensionEntry, ExtensionLoadResult};

/// One factory input, upstream's `InlineExtension | { factory; path? }`.
pub enum TestExtensionInput {
    /// A bare factory, labelled positionally as `<inline:N>`.
    Bare(ExtensionFactory),
    /// A named factory, labelled `<inline:name>`.
    Named {
        /// The display name.
        name: String,
        /// The factory.
        factory: ExtensionFactory,
    },
    /// A factory at an explicit path.
    At {
        /// The record path.
        path: String,
        /// The factory.
        factory: ExtensionFactory,
    },
}

impl TestExtensionInput {
    /// The record path the input labels, upstream's `extensionPath` ladder.
    fn extension_path(&self, index: usize) -> String {
        match self {
            Self::Bare(_) => format!("<inline:{}>", index + 1),
            Self::Named { name, .. } => format!("<inline:{name}>"),
            Self::At { path, .. } => path.clone(),
        }
    }

    /// The factory the input carries.
    fn factory(&self) -> &ExtensionFactory {
        match self {
            Self::Bare(factory) | Self::Named { factory, .. } | Self::At { factory, .. } => factory,
        }
    }
}

/// Load every input into one result, upstream's
/// `createTestExtensionsResult`.
///
/// # Panics
/// A factory that fails, which upstream's promise rejection fails the test
/// with; the suites here treat a load failure as the assertion's subject and
/// drive [`load_extension_from_factory`] directly instead.
pub async fn create_test_extensions_result(
    inputs: Vec<TestExtensionInput>,
    cwd: &str,
) -> ExtensionLoadResult {
    let runtime = ExtensionRuntime::new();
    let event_bus = EventBus::default();
    let mut extensions = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let path = input.extension_path(index);
        let record = load_extension_from_factory(input.factory(), &runtime, &event_bus, cwd, &path)
            .await
            .expect("the inline factory loads");
        extensions.push(ExtensionEntry {
            path: path.clone(),
            resolved_path: path,
            hidden: record.hidden(),
            source_info: Some(record.source_info.clone()),
            record: Some(record),
        });
    }
    ExtensionLoadResult {
        extensions,
        errors: Vec::new(),
        runtime,
    }
}
