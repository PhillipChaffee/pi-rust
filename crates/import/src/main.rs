//! The `pi-import` binary, the import tool's entry point: the argument
//! vector in, the rendered report and exit code out. The migration itself
//! lives in the library so the tests drive it without a process boundary.

fn main() {
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    std::process::exit(pi_import::cli::run_cli(args));
}
