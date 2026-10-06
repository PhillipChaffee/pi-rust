//! Boundary tests over the hosted-git-info hand-port and the splitRef
//! shapes the git-ssh-url suite does not reach.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_coding_agent::utils::git::hosted;
use pi_coding_agent::utils::git::parse_git_url;

#[test]
fn split_ref_reads_the_first_at_sign_in_the_scp_path() {
    // The scp-like path's first `@` splits the ref, upstream's
    // `indexOf("@")` — the remainder after it is the ref verbatim.
    let result = parse_git_url("git:git@github.com:user/repo@v2").expect("parses");
    assert_eq!(result.repo, "git@github.com:user/repo");
    assert_eq!(result.r#ref.as_deref(), Some("v2"));
    assert!(result.pinned);
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "user/repo");
    // A second `@` with no repo path cannot build a two-segment path;
    // every hosted candidate fails and the generic parser rejects the
    // one-segment shape.
    assert_eq!(parse_git_url("git:git@github.com:user@repo"), None);
}

#[test]
fn split_ref_trailing_slashes_strip_once() {
    // The `://` branch rebuilds the repo string and strips exactly one
    // trailing slash; the ref keeps the raw tail after the `@`.
    let result = parse_git_url("git:https://github.com/user/repo@v1.0.0/").expect("parses");
    assert_eq!(result.repo, "https://github.com/user/repo");
    assert_eq!(result.r#ref.as_deref(), Some("v1.0.0/"));
}

#[test]
fn shorthand_without_slash_yields_no_repo() {
    assert_eq!(parse_git_url("git:nodot"), None);
}

#[test]
fn build_rejects_paths_that_are_not_owner_repo() {
    // github.com/only rides the hosted shorthand (user "github.com",
    // project "only"), upstream's reshape.
    let result = parse_git_url("git:github.com/only").expect("hosted reshape");
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "github.com/only");
    // Any `x/y` shorthand rides the github grammar, hostless hosts
    // included, upstream's isGitHubShorthand scope.
    let result = parse_git_url("git:myhost.example/only").expect("shorthand reshape");
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "myhost.example/only");
}

#[test]
fn build_strips_one_git_suffix_and_leading_slashes() {
    // The `.git` suffix strips twice through the hosted extract and the
    // builder, one each.
    let result = parse_git_url("https://github.com/user/repo.git.git").expect("parses");
    assert_eq!(result.path, "user/repo");
    // Leading slashes strip greedily after the domain.
    let result = parse_git_url("https://github.com//user/repo").expect("parses");
    assert_eq!(result.path, "user/repo");
}

#[test]
fn unsafe_shapes_reject_through_the_decode_gate() {
    // A percent sequence that is malformed makes the decode fail, which
    // counts as unsafe.
    assert_eq!(parse_git_url("https://github.com/%zz/repo"), None);
    // Backslashes and NUL bytes reject.
    assert_eq!(parse_git_url("git:github.com/user\\repo"), None);
    assert_eq!(parse_git_url("git:github.com/user\0repo"), None);
}

#[test]
fn the_gist_shortcut_reads_the_project_alone() {
    let result = parse_git_url("git:gist:abc123def").expect("gist parses");
    assert_eq!(result.host, "gist.github.com");
    assert_eq!(result.path, "null/abc123def");
    assert_eq!(result.repo, "https://gist:abc123def");
}

#[test]
fn the_gitlab_and_sourcehut_shortcuts_parse() {
    let result = parse_git_url("git:gitlab:user/repo").expect("gitlab parses");
    assert_eq!(result.host, "gitlab.com");
    assert_eq!(result.path, "user/repo");
    let result = parse_git_url("git:sourcehut:~user/repo").expect("sourcehut parses");
    assert_eq!(result.host, "git.sr.ht");
    assert_eq!(result.path, "~user/repo");
    let result = parse_git_url("git:bitbucket:user/repo").expect("bitbucket parses");
    assert_eq!(result.host, "bitbucket.org");
    assert_eq!(result.path, "user/repo");
}

#[test]
fn the_github_shortcut_trims_auth_and_strips_git_suffix() {
    // The shortcut grammar ignores a userinfo segment.
    let result = parse_git_url("git:github:oauth-token@user/repo.git").expect("parses");
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "user/repo");
}

#[test]
fn hosted_tree_urls_read_their_committish() {
    // The /tree/ shape reads the trailing committish segment.
    let result = parse_git_url("https://github.com/user/repo/tree/v2").expect("parses");
    assert_eq!(result.r#ref.as_deref(), Some("v2"));
}

#[test]
fn generic_parser_uses_the_scp_shape_for_its_repo() {
    // parseGenericGitUrl keeps the scp-like repo string verbatim.
    let result = parse_git_url("git:git@myhost.example:user/repo").expect("parses");
    assert_eq!(result.repo, "git@myhost.example:user/repo");
    assert_eq!(result.host, "myhost.example");
    assert_eq!(result.path, "user/repo");
}

#[test]
fn http_urls_ride_the_generic_parser() {
    let result = parse_git_url("git:http://myhost.example/user/repo").expect("parses");
    assert_eq!(result.repo, "http://myhost.example/user/repo");
    assert_eq!(result.host, "myhost.example");
    assert_eq!(result.path, "user/repo");
}

#[test]
fn parse_fails_closed_for_nonsense() {
    // The parse-url fallback cannot rescue a bare word.
    assert_eq!(parse_git_url("git:justaword"), None);
    // An empty source parses to nothing.
    assert_eq!(parse_git_url(""), None);
    assert_eq!(parse_git_url("   "), None);
}

#[test]
fn the_github_shorthand_detector_needs_a_single_slash() {
    // One slash, no colon, no @: the github: prefix lands, the shortcut
    // grammar splits user/repo.
    let result = parse_git_url("git:user/repo").expect("shorthand parses");
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "user/repo");
    // repo carries upstream's https-prefix rule verbatim.
    assert_eq!(result.repo, "https://user/repo");
}

// === splitRef degenerate arms ===============================================

#[test]
fn scp_like_degenerate_hosts_and_paths_reject() {
    // matchScpLike's empty-host and empty-path guards, upstream's
    // `[^:]+` and `.+` quantifiers.
    assert_eq!(parse_git_url("git:git@:repo"), None);
    assert_eq!(parse_git_url("git:git@host:"), None);
}

#[test]
fn scp_like_ref_split_degenerate_arms_keep_the_url_verbatim() {
    // An `@` with nothing on one side of the split answers the URL
    // passthrough with no ref, upstream's empty repoPath/ref arms.
    assert_eq!(parse_git_url("git:git@host:@v1"), None);
    assert_eq!(parse_git_url("git:git@host:repo@"), None);
}

#[test]
fn protocol_url_ref_split_degenerate_arms_keep_the_url_verbatim() {
    // The `://` branch: an `@` with an empty repo path or ref answers the
    // passthrough, and a URL the parser cannot take rides the same shape.
    assert_eq!(parse_git_url("git:https://host/@v1"), None);
    assert_eq!(parse_git_url("git:https://host/repo@"), None);
    // `http://` alone has an empty host: the URL parse fails and the ref
    // walk answers the passthrough before the hosted candidates reject it.
    assert_eq!(parse_git_url("git:http://"), None);
}

#[test]
fn slash_branch_ref_split_degenerate_arms_keep_the_url_verbatim() {
    // The bare host/path branch: an `@` with an empty side answers the
    // passthrough; a well-formed ref splits and rides the shorthand.
    assert_eq!(parse_git_url("git:host.example/@v1"), None);
    assert_eq!(parse_git_url("git:host.example/repo@"), None);
    let result = parse_git_url("git:host.example/repo@v1").expect("parses");
    // The ref-bearing shorthand rides the github grammar, upstream's
    // isGitHubShorthand scope.
    assert_eq!(result.repo, "https://host.example/repo");
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "host.example/repo");
    assert_eq!(result.r#ref.as_deref(), Some("v1"));
}

#[test]
fn unsafe_hosts_reject_through_the_slash_gate() {
    // A host the scp grammar reads with a `/` inside fails the install
    // part gate on the host, upstream's unsafe-path check.
    assert_eq!(parse_git_url("git:git@ho.st/ex:x/y"), None);
}

#[test]
fn at_in_a_hosted_project_with_a_ref_skips_the_hosted_candidate() {
    // The hosted candidate's project carries an `@` while a ref splits
    // off: the candidate is skipped, upstream's project-contains-@ guard.
    // The `.git` segment keeps the ref-less rebuild's github extract
    // empty, so the candidate loop walks on; the generic parser then
    // accepts the two-segment strip, where the `.git` suffix leaves a
    // trailing-empty segment, upstream's `split("/").length === 2` gate.
    let result = parse_git_url("git:https://github.com/a/.git@v1").expect("parses");
    assert_eq!(result.repo, "https://github.com/a/.git");
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "a/");
    assert_eq!(result.r#ref.as_deref(), Some("v1"));
}

// === hosted grammar degenerate shapes =======================================

#[test]
fn gist_domain_shapes_walk_the_fallbacks() {
    // The gist host skips the emptiness gate, upstream's gist extract.
    let result = parse_git_url("git:https://gist.github.com/alice/gist123").expect("gist parses");
    assert_eq!(result.host, "gist.github.com");
    assert_eq!(result.path, "alice/gist123");
    assert_eq!(result.repo, "https://gist.github.com/alice/gist123");

    // A bare-owner gist URL: the project falls back to the owner, whose
    // slot reads null, upstream's user/project swap.
    let result = parse_git_url("git:https://gist.github.com/alice").expect("parses");
    assert_eq!(result.path, "null/alice");

    // The /raw shape is not a gist reference; the generic parser keeps it.
    let result = parse_git_url("git:https://gist.github.com/alice/gist123/raw").expect("parses");
    assert_eq!(result.path, "alice/gist123/raw");

    // Neither a user nor a project: nothing to parse.
    assert_eq!(parse_git_url("git:https://gist.github.com/"), None);
}

#[test]
fn sourcehut_domain_shapes_walk_the_fallbacks() {
    let result = parse_git_url("git:https://git.sr.ht/~user/repo").expect("parses");
    assert_eq!(result.host, "git.sr.ht");
    assert_eq!(result.path, "~user/repo");
    // The /archive shape is not a sourcehut reference; the generic parser
    // keeps the path.
    let result = parse_git_url("git:https://git.sr.ht/~user/repo/archive").expect("parses");
    assert_eq!(result.path, "~user/repo/archive");
}

#[test]
fn bitbucket_get_and_gitlab_special_paths_walk_the_fallbacks() {
    // bitbucket's /get shape and gitlab's compare/archive shapes are not
    // repository references, upstream's per-host guards; the generic
    // parser keeps the path.
    let result = parse_git_url("git:https://bitbucket.org/user/repo/get").expect("parses");
    assert_eq!(result.host, "bitbucket.org");
    assert_eq!(result.path, "user/repo/get");
    let result = parse_git_url("git:https://gitlab.com/user/-/repo").expect("parses");
    assert_eq!(result.path, "user/-/repo");
    let result = parse_git_url("git:https://gitlab.com/user/repo/archive.tar.gz").expect("parses");
    assert_eq!(result.path, "user/repo/archive.tar.gz");
}

#[test]
fn from_url_rejects_the_empty_input() {
    assert_eq!(hosted::from_url(""), None);
}

#[test]
fn domain_hosts_reject_unsupported_protocols() {
    // sourcehut accepts only git+ssh and https; http falls to the generic
    // parser, which keeps the repo verbatim.
    let result = parse_git_url("git:http://git.sr.ht/~user/repo").expect("parses");
    assert_eq!(result.repo, "http://git.sr.ht/~user/repo");
    assert_eq!(result.host, "git.sr.ht");
    assert_eq!(result.path, "~user/repo");
    // A protocol github does not index rejects outright, the passthrough
    // and the generic parser both failing the host gates.
    assert_eq!(parse_git_url("git:ftp://github.com/user/repo"), None);
}

#[test]
fn correct_protocol_rewrites_and_passthroughs_beyond_the_indexed_set() {
    // A <proto>:<user>@<host> shape outside the indexed protocols rides
    // the git+ssh rewrite, upstream's correctProtocol; no hosted host
    // answers and the colon-less repo rejects.
    assert_eq!(parse_git_url("git:foo:bar@baz"), None);
    // A <foo>://<bar> shape rides the verbatim passthrough before the
    // unknown-host rejection.
    assert_eq!(parse_git_url("git:foo://bar"), None);
}

#[test]
fn the_shorthand_detector_takes_a_space_only_after_the_hash() {
    // A space after the `#` keeps the github shorthand, upstream's
    // whitespace rule; the space rides the committish verbatim.
    let result = parse_git_url("git:user/repo#v 1").expect("parses");
    assert_eq!(result.host, "github.com");
    assert_eq!(result.path, "user/repo");
    assert_eq!(result.r#ref.as_deref(), Some("v 1"));
}
