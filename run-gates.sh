#!/usr/bin/env bash
# Runs every PR-blocking gate from this folder's README, in parallel.
# Keep this gate list in sync with the CI steps in ci.yml. Mutation
# testing is nightly only, so it is deliberately not here.
#
# The gate commands below are opaque strings that run_gates.sh evaluates
# at runtime; shellcheck sees them out of context here.
# shellcheck disable=SC2016,SC2027,SC2086,SC2154

set -u -o pipefail

log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

names=()
cmds=()
add() {
	names+=("$1")
	cmds+=("$2")
}
add "format" "cargo fmt --all -- --check"
add "lint" "cargo clippy --workspace --all-targets -- -D warnings"
add "doc" "RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps"
add "tests" "cargo test --workspace"
add "coverage" "cargo llvm-cov --workspace --fail-under-lines 95 --fail-under-regions 95 --fail-under-functions 95 --lcov --output-path lcov.info"

add "spell-check" "typos"
add "markdown-lint" "markdownlint-cli2 \"**/*.md\""
add "link-check" "lychee --no-progress ."
add "secret-scan" "gitleaks detect --no-git --redact"
add "duplication" "jscpd"
add "advisories" "osv-scanner scan -r ."
add "license-check" "osv-scanner scan -r . --licenses=MIT,Apache-2.0,ISC,BSD-3-Clause,BSD-2-Clause,MPL-2.0,PSF-2.0,Unicode-3.0,Python-2.0,Unlicense,CC0-1.0,0BSD,Apache-1.1,BSD-3-Clause-Clear,LGPL-3.0-only,BlueOak-1.0.0,CC-BY-3.0"
add "deny-advisories" "cargo deny check advisories"
add "deny-licenses" "cargo deny check licenses"
add "deny-bans" "cargo deny check bans"
add "unused-deps" "cargo shear --deny-warnings"
add "todo-policy" 'rc=0; git grep --untracked --no-recurse-submodules -nE "TODO|FIXME" -- "*.rs" || rc=$?; test "$rc" -eq 1'
add "shell-lint" 'for sh in $(git ls-files "*.sh"); do shellcheck "$sh"; done'
add "shell-format" 'for sh in $(git ls-files "*.sh"); do shfmt -d "$sh"; done'
add "workflow-yaml-lint" "yamllint ./.github/workflows/*.yml $(find . -mindepth 2 -maxdepth 2 \( -name 'ci.yml' -o -name 'mutation.yml' \) -not -path './.github/*' | tr '\n' ' ')"
add "workflow-lint" "actionlint ./.github/workflows/*.yml $(find . -mindepth 2 -maxdepth 2 \( -name 'ci.yml' -o -name 'mutation.yml' \) -not -path './.github/*' | tr '\n' ' ')"
for i in "${!names[@]}"; do
	name="${names[$i]}"
	cmd="${cmds[$i]}"
	(
		if eval "$cmd" >"$log_dir/$name.log" 2>&1; then
			echo "PASS  $name" >"$log_dir/$name.status"
		else
			echo "FAIL  $name" >"$log_dir/$name.status"
			printf '%s\n' "--- $name output ---" >>"$log_dir/failures.log"
			cat "$log_dir/$name.log" >>"$log_dir/failures.log"
		fi
	) &
done
wait

cat "$log_dir"/*.status 2>/dev/null
if [ -f "$log_dir/failures.log" ]; then
	echo "=== failing gate output ==="
	cat "$log_dir/failures.log"
	exit 1
fi
echo "all gates pass"
