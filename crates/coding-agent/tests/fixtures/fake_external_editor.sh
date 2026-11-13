#!/bin/sh
# The external-editor fixture, mirroring upstream's
# `test/fixtures/fake-external-editor.mjs`: capture what the editor was
# handed, then apply the requested outcome.
#
# argv: <capture-prefix> <mode> <file>
set -u
capture="$1"
mode="$2"
target="$3"
directory=$(dirname "$target")
printf '%s' "$target" >"$capture.path"
cat "$target" >"$capture.content"
ls -A "$directory" >"$capture.entries"
if permissions=$(stat -c %a "$directory" 2>/dev/null); then
	printf '%s' "$permissions" >"$capture.mode"
elif permissions=$(stat -f %Lp "$directory" 2>/dev/null); then
	printf '%s' "$permissions" >"$capture.mode"
fi
case "$mode" in
--fail) exit 1 ;;
--empty) : >"$target" ;;
--bom) printf '\357\273\277edited\n\n' >"$target" ;;
*) printf 'edited\n' >"$target" ;;
esac
