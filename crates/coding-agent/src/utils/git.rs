//! Git source parsing, upstream's `src/utils/git.ts`, plus the hand-port
//! of the `hosted-git-info` 9.0.3 grammar its hosted candidates ride (the
//! [`hosted`] submodule, ported from that package's `lib/`).
//!
//! `hosted-git-info` is an npm dependency upstream; the port carries the
//! slice `parseGitUrl` reads — `fromUrl` over the five bundled hosts'
//! shortcut and domain grammars, with node's `URL` restate onto the `url`
//! crate (both WHATWG) and `decodeURIComponent` re-expressed with its
//! malformed-escape `URIError`.

use url::Url;

/// Parsed git URL information, upstream's `GitSource` minus the constant
/// `type: "git"` field, which has no Rust counterpart to carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitSource {
    /// Clone URL (always valid for git clone, without ref suffix).
    pub repo: String,
    /// Git host domain (e.g., "github.com").
    pub host: String,
    /// Repository path (e.g., "user/repo").
    pub path: String,
    /// Git ref (branch, tag, commit) if specified.
    pub r#ref: Option<String>,
    /// True if ref was specified (package won't be auto-updated).
    pub pinned: bool,
}

struct SplitRef {
    repo: String,
    r#ref: Option<String>,
}

/// The scp-like shape `git@host:path`, upstream's
/// `/^git@([^:]+):(.+)$/` match: host and the path up to any ref separator.
fn match_scp_like(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("git@")?;
    let colon = rest.find(':')?;
    let host = &rest[..colon];
    if host.is_empty() {
        return None;
    }
    let path = &rest[colon + 1..];
    if path.is_empty() {
        return None;
    }
    Some((host, path))
}

fn split_ref(url: &str) -> SplitRef {
    if let Some((host, path_with_maybe_ref)) = match_scp_like(url) {
        let Some(ref_separator) = path_with_maybe_ref.find('@') else {
            return SplitRef {
                repo: url.to_string(),
                r#ref: None,
            };
        };
        let repo_path = &path_with_maybe_ref[..ref_separator];
        let r#ref = &path_with_maybe_ref[ref_separator + 1..];
        if repo_path.is_empty() || r#ref.is_empty() {
            return SplitRef {
                repo: url.to_string(),
                r#ref: None,
            };
        }
        return SplitRef {
            repo: format!("git@{host}:{repo_path}"),
            r#ref: Some(r#ref.to_string()),
        };
    }

    if url.contains("://") {
        if let Ok(parsed) = Url::parse(url) {
            let path_with_maybe_ref = parsed.path().trim_start_matches('/').to_string();
            let Some(ref_separator) = path_with_maybe_ref.find('@') else {
                return SplitRef {
                    repo: url.to_string(),
                    r#ref: None,
                };
            };
            let repo_path = &path_with_maybe_ref[..ref_separator];
            let r#ref = &path_with_maybe_ref[ref_separator + 1..];
            if repo_path.is_empty() || r#ref.is_empty() {
                return SplitRef {
                    repo: url.to_string(),
                    r#ref: None,
                };
            }
            let mut rebuilt = parsed;
            rebuilt.set_path(&format!("/{repo_path}"));
            // Upstream strips exactly one trailing slash, `replace(/\/$/, "")`.
            let rebuilt = rebuilt.to_string();
            let repo = rebuilt.strip_suffix('/').unwrap_or(&rebuilt).to_string();
            return SplitRef {
                repo,
                r#ref: Some(r#ref.to_string()),
            };
        }
        return SplitRef {
            repo: url.to_string(),
            r#ref: None,
        };
    }

    let Some(slash_index) = url.find('/') else {
        return SplitRef {
            repo: url.to_string(),
            r#ref: None,
        };
    };
    let host = &url[..slash_index];
    let path_with_maybe_ref = &url[slash_index + 1..];
    let Some(ref_separator) = path_with_maybe_ref.find('@') else {
        return SplitRef {
            repo: url.to_string(),
            r#ref: None,
        };
    };
    let repo_path = &path_with_maybe_ref[..ref_separator];
    let r#ref = &path_with_maybe_ref[ref_separator + 1..];
    if repo_path.is_empty() || r#ref.is_empty() {
        return SplitRef {
            repo: url.to_string(),
            r#ref: None,
        };
    }
    SplitRef {
        repo: format!("{host}/{repo_path}"),
        r#ref: Some(r#ref.to_string()),
    }
}

fn decode_for_validation(value: &str) -> Option<String> {
    hosted::decode_uri_component(value)
}

fn has_unsafe_git_install_part(value: &str, allow_slash: bool) -> bool {
    let Some(decoded) = decode_for_validation(value) else {
        return true;
    };
    for candidate in [value.to_string(), decoded] {
        if candidate.contains('\0') || candidate.contains('\\') || candidate.starts_with('/') {
            return true;
        }
        if !allow_slash && candidate.contains('/') {
            return true;
        }
        if candidate.split('/').any(|segment| segment == "..") {
            return true;
        }
    }
    false
}

struct BuildArgs {
    repo: String,
    host: String,
    path: String,
    r#ref: Option<String>,
}

fn build_git_source(args: BuildArgs) -> Option<GitSource> {
    if args.path.starts_with('/') {
        return None;
    }
    // `.git` suffix strips once, leading slashes strip greedily.
    let without_git = args.path.strip_suffix(".git").unwrap_or(&args.path);
    let normalized_path = without_git.trim_start_matches('/');
    if args.host.is_empty() || normalized_path.is_empty() || normalized_path.split('/').count() < 2
    {
        return None;
    }
    if has_unsafe_git_install_part(&args.host, false)
        || has_unsafe_git_install_part(normalized_path, true)
    {
        return None;
    }

    Some(GitSource {
        repo: args.repo,
        host: args.host,
        path: normalized_path.to_string(),
        pinned: args.r#ref.is_some(),
        r#ref: args.r#ref,
    })
}

fn parse_generic_git_url(url: &str) -> Option<GitSource> {
    let split = split_ref(url);
    let mut repo = split.repo.clone();
    let host;
    let path;

    if let Some((scp_host, scp_path)) = match_scp_like(&split.repo) {
        host = scp_host.to_string();
        path = scp_path.to_string();
    } else if split.repo.starts_with("https://")
        || split.repo.starts_with("http://")
        || split.repo.starts_with("ssh://")
        || split.repo.starts_with("git://")
    {
        let parsed = Url::parse(&split.repo).ok()?;
        host = parsed.host_str().unwrap_or_default().to_string();
        path = parsed.path().trim_start_matches('/').to_string();
    } else {
        let slash_index = split.repo.find('/')?;
        host = split.repo[..slash_index].to_string();
        path = split.repo[slash_index + 1..].to_string();
        if !host.contains('.') && host != "localhost" {
            return None;
        }
        repo = format!("https://{repo}");
    }

    build_git_source(BuildArgs {
        repo,
        host,
        path,
        r#ref: split.r#ref,
    })
}

/// Parse git source into a [`GitSource`].
///
/// Rules:
/// - With git: prefix, accept all historical shorthand forms.
/// - Without git: prefix, only accept explicit protocol URLs.
#[must_use]
pub fn parse_git_url(source: &str) -> Option<GitSource> {
    let trimmed = source.trim();
    let has_git_prefix = trimmed.starts_with("git:");
    let url = if has_git_prefix {
        trimmed[4..].trim()
    } else {
        trimmed
    };

    let protocol_prefix = url.split("://").next().is_some_and(|proto| {
        proto.len() + 3 <= url.len()
            && matches!(
                proto.to_ascii_lowercase().as_str(),
                "http" | "https" | "ssh" | "git"
            )
    });
    if !has_git_prefix && !protocol_prefix {
        return None;
    }

    let split = split_ref(url);

    let mut hosted_candidates: Vec<String> = Vec::new();
    if let Some(r#ref) = &split.r#ref {
        hosted_candidates.push(format!("{}#{}", split.repo, r#ref));
    }
    hosted_candidates.push(url.to_string());
    for candidate in &hosted_candidates {
        let Some(info) = hosted::from_url(candidate) else {
            continue;
        };
        if split.r#ref.is_some() && info.project.contains('@') {
            continue;
        }
        let use_https_prefix = !split.repo.starts_with("http://")
            && !split.repo.starts_with("https://")
            && !split.repo.starts_with("ssh://")
            && !split.repo.starts_with("git://")
            && !split.repo.starts_with("git@");
        let repo = if use_https_prefix {
            format!("https://{}", split.repo)
        } else {
            split.repo.clone()
        };
        return build_git_source(BuildArgs {
            repo,
            host: info.domain.to_string(),
            // Upstream interpolates a null user as the literal "null"; the
            // gist host is the only shape that can hit it.
            path: format!(
                "{}/{}",
                info.user.as_deref().unwrap_or("null"),
                info.project
            ),
            r#ref: info.committish.or_else(|| split.r#ref.clone()),
        });
    }

    let mut https_candidates: Vec<String> = Vec::new();
    if let Some(r#ref) = &split.r#ref {
        https_candidates.push(format!("https://{}#{}", split.repo, r#ref));
    }
    https_candidates.push(format!("https://{url}"));
    for candidate in &https_candidates {
        let Some(info) = hosted::from_url(candidate) else {
            continue;
        };
        if split.r#ref.is_some() && info.project.contains('@') {
            continue;
        }
        return build_git_source(BuildArgs {
            repo: format!("https://{}", split.repo),
            host: info.domain.to_string(),
            path: format!(
                "{}/{}",
                info.user.as_deref().unwrap_or("null"),
                info.project
            ),
            r#ref: info.committish.or_else(|| split.r#ref.clone()),
        });
    }

    parse_generic_git_url(url)
}
/// The hand-port of `hosted-git-info` 9.0.3, the npm dependency upstream's
/// hosted candidates ride.
///
/// `lib/from-url.js`, `lib/parse-url.js`, and `lib/hosts.js` restate here;
/// node's `URL` restate onto the `url` crate (both WHATWG parsers), and
/// `decodeURIComponent` re-expresses with its malformed-escape `URIError`.
/// The `auth`, template-formatting, and cache surfaces the package also
/// carries have no `parseGitUrl` consumer.
pub mod hosted {
    use url::Url;

    /// One hosted-URL parse, the fields `parseGitUrl` reads off the
    /// `GitHost` object.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct HostedInfo {
        /// The host's short name, upstream's `type` (`"github"`).
        pub type_name: &'static str,
        /// The host's domain (`"github.com"`).
        pub domain: &'static str,
        /// The repository owner, `None` where upstream carries null.
        pub user: Option<String>,
        /// The repository project, `.git` stripped.
        pub project: String,
        /// The `#fragment` committish, `None` where upstream's is falsy.
        pub committish: Option<String>,
    }

    /// The protocols `correctProtocol` accepts verbatim, upstream's
    /// `GitHost.#protocols` keys plus the registered shortcuts.
    const PROTOCOLS: [&str; 12] = [
        "git+ssh:",
        "ssh:",
        "git+https:",
        "git:",
        "http:",
        "https:",
        "git+http:",
        "github:",
        "gist:",
        "gitlab:",
        "bitbucket:",
        "sourcehut:",
    ];

    struct Host {
        type_name: &'static str,
        domain: &'static str,
        protocols: &'static [&'static str],
    }

    const GITHUB: Host = Host {
        type_name: "github",
        domain: "github.com",
        protocols: &["git:", "http:", "git+ssh:", "git+https:", "ssh:", "https:"],
    };
    const BITBUCKET: Host = Host {
        type_name: "bitbucket",
        domain: "bitbucket.org",
        protocols: &["git+ssh:", "git+https:", "ssh:", "https:"],
    };
    const GITLAB: Host = Host {
        type_name: "gitlab",
        domain: "gitlab.com",
        protocols: &["git+ssh:", "git+https:", "ssh:", "https:"],
    };
    const GIST: Host = Host {
        type_name: "gist",
        domain: "gist.github.com",
        protocols: &["git:", "git+ssh:", "git+https:", "ssh:", "https:"],
    };
    const SOURCEHUT: Host = Host {
        type_name: "sourcehut",
        domain: "git.sr.ht",
        protocols: &["git+ssh:", "https:"],
    };

    fn host_by_shortcut(protocol: &str) -> Option<&'static Host> {
        match protocol {
            "github:" => Some(&GITHUB),
            "gist:" => Some(&GIST),
            "gitlab:" => Some(&GITLAB),
            "bitbucket:" => Some(&BITBUCKET),
            "sourcehut:" => Some(&SOURCEHUT),
            _ => None,
        }
    }

    fn host_by_domain(hostname: &str) -> Option<&'static Host> {
        match hostname {
            "github.com" => Some(&GITHUB),
            "bitbucket.org" => Some(&BITBUCKET),
            "gitlab.com" => Some(&GITLAB),
            "gist.github.com" => Some(&GIST),
            "git.sr.ht" => Some(&SOURCEHUT),
            _ => None,
        }
    }

    /// Look for github shorthand inputs, such as `npm/cli`, upstream's
    /// `isGitHubShorthand`.
    fn is_github_shorthand(arg: &str) -> bool {
        // it cannot contain whitespace before the first #
        // it cannot start with a / because that's probably an absolute file path
        // but it must include a slash since repos are username/repository
        // it cannot start with a . because that's probably a relative file path
        // it cannot start with an @ because that's a scoped package if it passes the other tests
        // it cannot contain a : before a # because that tells us that there's a protocol
        // a second / may not exist before a #
        let first_hash = arg.find('#');
        let first_slash = arg.find('/');
        let second_slash =
            first_slash.and_then(|first| arg[first + 1..].find('/').map(|at| at + first + 1));
        let first_colon = arg.find(':');
        let first_space = arg
            .char_indices()
            .find(|(_, ch)| ch.is_whitespace())
            .map(|(at, _)| at);
        let first_at = arg.find('@');

        let space_only_after_hash = first_space.is_none()
            || first_hash.is_some_and(|hash| first_space.is_some_and(|space| space > hash));
        let at_only_after_hash = first_at.is_none()
            || first_hash.is_some_and(|hash| first_at.is_some_and(|at| at > hash));
        let colon_only_after_hash = first_colon.is_none()
            || first_hash.is_some_and(|hash| first_colon.is_some_and(|colon| colon > hash));
        let second_slash_only_after_hash = second_slash.is_none()
            || first_hash.is_some_and(|hash| second_slash.is_some_and(|slash| slash > hash));
        let has_slash = first_slash.is_some_and(|slash| slash > 0);
        // if a # is found, what we really want to know is that the character
        // immediately before # is not a /
        let does_not_end_with_slash = first_hash.map_or_else(
            || !arg.ends_with('/'),
            |hash| arg.as_bytes().get(hash.wrapping_sub(1)) != Some(&b'/'),
        );
        let does_not_start_with_dot = !arg.starts_with('.');

        space_only_after_hash
            && has_slash
            && does_not_end_with_slash
            && does_not_start_with_dot
            && at_only_after_hash
            && colon_only_after_hash
            && second_slash_only_after_hash
    }

    /// node's `String.prototype.substr(start, length)`: a negative start
    /// counts from the end. The port's only negative caller is the
    /// colon-less probe, upstream's `arg.substr(firstColon, 3)` with
    /// `firstColon` at -1, which reads the last byte.
    fn js_substr(arg: &str, start: Option<usize>, length: usize) -> &str {
        let bytes = arg.as_bytes();
        let resolved =
            start.map_or_else(|| bytes.len().saturating_sub(1), |at| at.min(bytes.len()));
        let end = (resolved + length).min(bytes.len());
        &arg[resolved..end]
    }

    /// Accept input like `git:github.com:user/repo` and insert the `//`
    /// after the first `:`, upstream's `correctProtocol`.
    fn correct_protocol(arg: &str) -> String {
        // The -1 for a colon-less input restate as 0: `arg.substr(0, 0)` is
        // empty either way, and the `://` probe lands the same.
        let first_colon = arg.find(':');
        let proto = js_substr(arg, Some(0), first_colon.map_or(0, |at| at + 1));
        if PROTOCOLS.contains(&proto) {
            return arg.to_string();
        }

        if js_substr(arg, first_colon, 3) == "://" {
            // If arg is given as <foo>://<bar>, then this is already a valid URL.
            return arg.to_string();
        }

        let first_at = arg.find('@');
        if let Some(first_at) = first_at {
            if first_colon.is_none_or(|colon| first_at > colon) {
                // URL has the form of <foo>:<bar>@<baz>. Assume this is a git+ssh URL.
                return format!("git+ssh://{arg}");
            }
            // URL has the form 'git@github.com:npm/hosted-git-info.git'.
            return arg.to_string();
        }

        // Correct <foo>:<bar> to <foo>://<bar>; the colon-less cut is 0,
        // upstream's `slice(0, firstColon + 1)` with -1.
        let cut = first_colon.map_or(0, |at| at + 1);
        format!(
            "{}//{}",
            &arg[..cut.min(arg.len())],
            &arg[cut.min(arg.len())..]
        )
    }

    /// The index of the last `char` before the first `before_char`,
    /// upstream's `lastIndexOfBefore`.
    fn last_index_of_before(arg: &str, ch: char, before_char: char) -> Option<usize> {
        arg.find(before_char).map_or(arg, |at| &arg[..at]).rfind(ch)
    }

    /// Attempt to correct an scp style url so that it will parse with
    /// `new URL()`, upstream's `correctUrl`.
    fn correct_url(giturl: &str) -> String {
        let mut giturl = giturl.to_string();
        // ignore @ that come after the first hash since the denotes the start
        // of a committish which can contain @ characters
        let first_at = last_index_of_before(&giturl, '@', '#');
        // ignore colons that come after the hash since that could include colons such as:
        // git@github.com:user/package-2#semver:^1.0.0
        let last_colon_before_hash = last_index_of_before(&giturl, ':', '#');

        if let Some(colon) = last_colon_before_hash
            && colon > first_at.unwrap_or(usize::MAX)
        {
            // the last : comes after the first @ (or there is no @)
            // then we replace the last : with a / to create a valid path
            giturl = format!("{}{}{}", &giturl[..colon], '/', &giturl[colon + 1..]);
        }

        if last_index_of_before(&giturl, ':', '#').is_none() && !giturl.contains("//") {
            // we have no : at all
            // then we prepend a protocol
            giturl = format!("git+ssh://{giturl}");
        }

        giturl
    }

    /// Parse a git URL with node's `URL`, falling back to the scp-style
    /// correction, upstream's `parse-url`.
    fn parse_url(giturl: &str) -> Option<Url> {
        let with_protocol = correct_protocol(giturl);
        Url::parse(&with_protocol)
            .ok()
            .or_else(|| correct_url(&with_protocol).parse().ok())
    }

    /// Decode one URI component, upstream's `decodeURIComponent`: a
    /// malformed percent sequence or a non-UTF-8 result is the `URIError`
    /// that makes the parse fail.
    #[must_use]
    pub fn decode_uri_component(value: &str) -> Option<String> {
        const fn hex_digit(byte: u8) -> Option<u8> {
            match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                b'A'..=b'F' => Some(byte - b'A' + 10),
                _ => None,
            }
        }
        let bytes = value.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut index = 0usize;
        while index < bytes.len() {
            if bytes[index] == b'%' {
                let high = bytes.get(index + 1).copied().and_then(hex_digit)?;
                let low = bytes.get(index + 2).copied().and_then(hex_digit)?;
                out.push((high << 4) | low);
                index += 3;
            } else {
                out.push(bytes[index]);
                index += 1;
            }
        }
        String::from_utf8(out).ok()
    }

    /// The WHATWG `url.pathname.split('/', limit)` shape: at most `limit`
    /// parts, the remainder dropped — node's split with a limit truncates.
    fn split_pathname(pathname: &str, limit: usize) -> Vec<&str> {
        pathname.split('/').take(limit).collect()
    }

    /// The per-host extraction, upstream's `hosts[*].extract(url)` with the
    /// committish decode folded in — including the github `undefined`
    /// coercion a type-bearing path without a committish segment decodes
    /// to, node's `decodeURIComponent(undefined)`.
    fn extract(host: &Host, parsed: &Url) -> Option<(Option<String>, String, Option<String>)> {
        // node's `url.hash.slice(1)`: `""` when there is no fragment.
        let hash_committish = || decode_uri_component(parsed.fragment().unwrap_or(""));
        // node's `decodeURIComponent(undefined)` coerces to "undefined".
        let decoded_committish = |value: Option<&str>| {
            value.map_or_else(|| Some("undefined".to_string()), decode_uri_component)
        };
        // Upstream strips the `.git` suffix before the emptiness gate; the
        // gist host alone carries no gate.
        let finish = |user: Option<String>, project: &str, committish: Option<String>| {
            let project = strip_git(project);
            if host.type_name != "gist"
                && (user.as_deref().is_none_or(str::is_empty) || project.is_empty())
            {
                return None;
            }
            Some((user, project, committish))
        };
        match host.type_name {
            "github" => {
                let parts = split_pathname(parsed.path(), 5);
                let user = parts.get(1).copied();
                let project = parts.get(2).copied().unwrap_or("");
                let type_segment = parts.get(3).copied();
                let committish = parts.get(4).copied();
                if type_segment.is_some_and(|type_segment| type_segment != "tree") {
                    return None;
                }

                if type_segment.is_none() {
                    return finish(user.map(str::to_string), project, hash_committish());
                }

                finish(
                    user.map(str::to_string),
                    project,
                    decoded_committish(committish),
                )
            }
            "bitbucket" => {
                let parts = split_pathname(parsed.path(), 4);
                let user = parts.get(1).copied().unwrap_or("");
                let project = parts.get(2).copied().unwrap_or("");
                let aux = parts.get(3).copied();
                if aux == Some("get") {
                    return None;
                }
                finish(Some(user.to_string()), project, hash_committish())
            }
            "gitlab" => {
                let path = parsed
                    .path()
                    .strip_prefix('/')
                    .unwrap_or_else(|| parsed.path());
                if path.contains("/-/") || path.contains("/archive.tar.gz") {
                    return None;
                }

                let mut segments: Vec<&str> = path.split('/').collect();
                let project = segments.pop().unwrap_or("");
                let user = segments.join("/");
                finish(Some(user), project, hash_committish())
            }
            "gist" => {
                let parts = split_pathname(parsed.path(), 4);
                let mut user = parts.get(1).copied().filter(|user| !user.is_empty());
                let mut project = parts.get(2).copied().filter(|project| !project.is_empty());
                let aux = parts.get(3).copied();
                if aux == Some("raw") {
                    return None;
                }

                if project.is_none()
                    && let Some(fallback) = user
                {
                    project = Some(fallback);
                    user = None;
                } else if project.is_none() {
                    return None;
                }

                finish(user.map(str::to_string), project?, hash_committish())
            }
            "sourcehut" => {
                let parts = split_pathname(parsed.path(), 4);
                let user = parts.get(1).copied().unwrap_or("");
                let project = parts.get(2).copied().unwrap_or("");
                let aux = parts.get(3).copied();
                if aux == Some("archive") {
                    return None;
                }
                finish(Some(user.to_string()), project, hash_committish())
            }
            _ => None,
        }
    }

    fn strip_git(project: &str) -> String {
        project.strip_suffix(".git").unwrap_or(project).to_string()
    }

    /// Parse a hosted git URL, upstream's `GitHost.fromUrl`.
    #[must_use]
    pub fn from_url(giturl: &str) -> Option<HostedInfo> {
        if giturl.is_empty() {
            return None;
        }

        let corrected_url = if is_github_shorthand(giturl) {
            format!("github:{giturl}")
        } else {
            giturl.to_string()
        };
        let parsed = parse_url(&corrected_url)?;

        let protocol = format!("{}:", parsed.scheme());
        let git_host_shortcut = host_by_shortcut(&protocol);
        let raw_hostname = parsed.host_str().unwrap_or("");
        let hostname = raw_hostname.strip_prefix("www.").unwrap_or(raw_hostname);
        let git_host_domain = host_by_domain(hostname);
        let host = git_host_shortcut.or(git_host_domain)?;

        if git_host_shortcut.is_some() {
            // we ignore auth for shortcuts, so just trim it out
            let pathname = parsed
                .path()
                .strip_prefix('/')
                .unwrap_or_else(|| parsed.path());
            let pathname = pathname
                .find('@')
                .map_or(pathname, |first_at| &pathname[first_at + 1..]);

            let (user, project) = match pathname.rfind('/') {
                Some(last_slash) => {
                    let decoded_user = decode_uri_component(&pathname[..last_slash]);
                    // we want nulls only, never empty strings
                    (
                        decoded_user.filter(|decoded| !decoded.is_empty()),
                        decode_uri_component(&pathname[last_slash + 1..])?,
                    )
                }
                None => (None, decode_uri_component(pathname)?),
            };
            let project = strip_git(&project);
            let committish = parsed.fragment().and_then(decode_uri_component);

            return Some(HostedInfo {
                type_name: host.type_name,
                domain: host.domain,
                user,
                project,
                committish: committish.filter(|committish| !committish.is_empty()),
            });
        }

        if !host.protocols.contains(&protocol.as_str()) {
            return None;
        }

        let (user, project, committish) = extract(host, &parsed)?;
        Some(HostedInfo {
            type_name: host.type_name,
            domain: host.domain,
            user,
            project,
            committish: committish.filter(|committish| !committish.is_empty()),
        })
    }
}
