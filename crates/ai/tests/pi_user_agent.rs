//! The `pi` user agent string, ported with the module from
//! `packages/ai/src/utils/pi-user-agent.ts`; upstream has no dedicated
//! suite, so these pin the node-vocabulary format the providers see.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

#[test]
fn the_user_agent_names_the_platform_release_and_architecture() {
    let agent = pi_ai::utils::pi_user_agent::get_pi_user_agent();

    assert!(
        agent.starts_with("pi ("),
        "the agent must open with the pi product, got {agent}"
    );
    assert!(
        agent.ends_with(')'),
        "the agent must close after the architecture, got {agent}"
    );
    let inner = agent
        .strip_prefix("pi (")
        .and_then(|rest| rest.strip_suffix(')'))
        .expect("the parenthesized detail");
    let (platform_release, arch) = inner
        .split_once(';')
        .map(|(left, right)| (left.trim(), right.trim()))
        .expect("platform release and architecture are semicolon-separated");
    let (platform, release) = platform_release
        .split_once(' ')
        .expect("the release follows the platform");
    assert!(
        matches!(platform, "darwin" | "linux" | "windows"),
        "the platform uses node's os.platform() spelling, got {platform}"
    );
    assert!(
        !release.is_empty(),
        "the kernel release is non-empty, got {agent}"
    );
    assert!(
        matches!(arch, "arm64" | "x64" | "x86" | "arm"),
        "the architecture uses node's os.arch() spelling, got {arch}"
    );
}
