//! The tool-registry belt's boundary tests, upstream's factory body in
//! `src/core/tools/index.ts` (the strict-mode suite's registry describe
//! rides the extension-system ticket; the factories' shapes bind here).

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use crate::tools::index::{
    ToolName, create_all_tool_definitions, create_all_tools, create_coding_tool_definitions,
    create_coding_tools, create_read_only_tool_definitions, create_read_only_tools, create_tool,
    create_tool_definition,
};

#[test]
fn tool_names_parse_the_wire_names_and_reject_the_rest() {
    for name in [
        "read",
        "bash",
        "powershell",
        "edit",
        "write",
        "grep",
        "find",
        "ls",
    ] {
        assert_eq!(ToolName::parse(name).unwrap().as_str(), name);
    }
    assert!(ToolName::parse("terminal").is_err());
}

#[test]
fn the_strict_sampling_set_covers_the_coding_and_shell_tools() {
    assert!(ToolName::Read.is_strict_sampling());
    assert!(ToolName::Bash.is_strict_sampling());
    assert!(ToolName::Powershell.is_strict_sampling());
    assert!(ToolName::Edit.is_strict_sampling());
    assert!(ToolName::Write.is_strict_sampling());
    assert!(!ToolName::Grep.is_strict_sampling());
    assert!(!ToolName::Find.is_strict_sampling());
    assert!(!ToolName::Ls.is_strict_sampling());
}

#[test]
fn the_definition_factories_cover_every_tool() {
    let all = create_all_tool_definitions(".", None);
    assert_eq!(all.len(), 8);
    for (name, definition) in &all {
        assert_eq!(definition.name, name.as_str());
    }
    assert_eq!(create_coding_tool_definitions(".", None).len(), 4);
    assert_eq!(create_read_only_tool_definitions(".", None).len(), 4);
    let definition = create_tool_definition(ToolName::parse("grep").unwrap(), ".", None).unwrap();
    assert_eq!(definition.name, "grep");
}

#[test]
fn the_wrapped_factories_cover_every_tool() {
    let all = create_all_tools(".", None);
    assert_eq!(all.len(), 8);
    for (name, tool) in &all {
        assert_eq!(tool.tool.name, name.as_str());
    }
    assert_eq!(create_coding_tools(".", None).len(), 4);
    assert_eq!(create_read_only_tools(".", None).len(), 4);
    let tool = create_tool(ToolName::parse("read").unwrap(), ".", None).unwrap();
    assert_eq!(tool.tool.name, "read");
}
