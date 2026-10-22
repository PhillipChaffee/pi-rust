#!/usr/bin/env python3
"""Verify that inline `#[cfg(test)] mod tests` blocks were moved, not rewritten.

Ticket #156 moves every inline unit-test block out of `crates/*/src/**` into a
directory-module sibling (`foo.rs` keeps `#[cfg(test)]` + `mod tests;`, the
tests live in `foo/tests.rs`) — a pure copy-paste, since tests are never
deleted or thinned to fit the coverage gate. This script is the proof.

Given the pre-move revision, for each parent file it extracts the old
`#[cfg(test)] mod tests { ... }` block with a brace-count parser over
strings/chars/comments-masked source (never regex-to-file-end), extracts the
new sibling's head inner attributes and body, compares attributes verbatim
(order included), and compares the body line-by-line after removing exactly
the one block-indent level (4 spaces) the move paid for. The parent must equal
the old text with the block span swapped for the decl, byte for byte.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

BLOCK_INDENT = 4
CHAR_LIT = re.compile(r"'(?:\\(?:u\{[^}]*\}|x[0-9a-fA-F]{2}|.)|[^'\\])'")
RAW_OPEN = re.compile(r"b?r?(#*)\"")


def mask(src):
    """Blank strings, chars, raw strings and comments (newlines kept) so brace
    counting sees only real Rust code."""
    n = len(src)
    out = list(src)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            blank(i, n if j < 0 else j)
            i = j if j >= 0 else n
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            blank(i, n if j < 0 else j + 2)
            i = n if j < 0 else j + 2
            continue
        if c in ("b", "r"):
            m = RAW_OPEN.match(src, i)
            if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
                term = '"' + m.group(1)
                j = src.find(term, m.end())
                if j >= 0:
                    blank(i, j + len(term))
                    i = j + len(term)
                else:
                    blank(i, n)
                    i = n
                continue
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i, min(j + 1, n))
            i = j + 1
            continue
        if c == "'":
            m = CHAR_LIT.match(src, i)
            if m:
                blank(i, m.end())
                i = m.end()
                continue
        i += 1
    return "".join(out)


def fmt_canonical(text, repo: Path) -> str:
    """The rustfmt canonical rendering of `text`, using the repo's own
    rustfmt.toml and pinned toolchain (source piped on stdin)."""
    r = subprocess.run(
        ["rustfmt", "--config-path", str(repo / "rustfmt.toml"), "--edition", "2024",
         "--emit", "stdout"],
        cwd=repo, input=text, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"rustfmt failed on the recomputed expected copy:\n{r.stderr}")
    return r.stdout


def parse_inline_blocks(src, path_err):
    masked = mask(src)
    lines = src.splitlines(keepends=True)
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l))
    blocks = []
    for idx, line in enumerate(lines):
        if not line.strip().startswith("#[cfg(test)]"):
            continue
        j = idx + 1
        while j < len(lines):
            s = lines[j].strip()
            if s == "" or s.startswith("//") or s.startswith("#["):
                j += 1
                continue
            break
        if j >= len(lines) or lines[j].strip() != "mod tests {":
            continue
        open_off = offs[j] + lines[j].find("{")
        depth, end = 0, None
        for k in range(open_off, len(masked)):
            if masked[k] == "{":
                depth += 1
            elif masked[k] == "}":
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
        if end is None:
            raise SystemExit(f"{path_err}: unterminated tests block")
        blocks.append({"start": offs[idx], "end": end, "text": src[offs[idx]:end]})
    return blocks


def attr_lump_end(lines, start):
    """Index of the line whose `]` closes the inner attribute starting at
    lines[start] (paren depth tracked)."""
    depth, started = 0, False
    for k in range(start, len(lines)):
        for ch in lines[k]:
            if ch == "(":
                depth += 1
                started = True
            elif ch == ")":
                depth -= 1
            elif ch == "]" and (not started or depth == 0):
                return k
    raise SystemExit(f"unterminated inner attribute at line {start}")


def edge(lines):
    ls = list(lines)
    while ls and ls[0].strip() == "":
        ls.pop(0)
    while ls and ls[-1].strip() == "":
        ls.pop()
    return ls


def split_block(text):
    lines = text.splitlines()
    open_idx = next(i for i, l in enumerate(lines) if "{" in l)
    close_idx = next(i for i in range(len(lines) - 1, open_idx, -1) if lines[i].strip() == "}")
    seg = lines[open_idx + 1 : close_idx]
    j, p, attrs = 0, len(seg), []
    while j < p and seg[j].strip().startswith("#!["):
        k = attr_lump_end(seg, j)
        attrs.extend(seg[j : k + 1])
        j = k + 1
    return attrs, seg[j:]


def head_split(sibling_src):
    sib = sibling_src.splitlines()
    j, attrs = 0, []
    while j < len(sib) and sib[j].strip().startswith("#!["):
        k = attr_lump_end(sib, j)
        attrs.extend(sib[j : k + 1])
        j = k + 1
    while j < len(sib) and sib[j].strip() == "":
        j += 1
    return attrs, sib[j:]


def dedent4(lines):
    out = []
    for l in lines:
        if l.startswith("    "):
            out.append(l[4:])
        elif l.strip() == "":
            out.append("")
        else:
            raise SystemExit(f"unexpected indent level in moved body: {l!r}")
    return out


def derive_sibling(parent_rel):
    p = Path(parent_rel)
    return p.parent / ("tests.rs" if p.stem == "mod" else p.stem + "/tests.rs")


def compare(rel, base_src, cur, failures, table, totals):
    blocks = parse_inline_blocks(base_src, rel)
    if not blocks:
        table.append(f"{rel}: untouched (already in target shape at base)")
        return
    if len(blocks) != 1:
        failures.append(f"{rel}: {len(blocks)} inline tests blocks at base; mover expects one")
        return
    b = blocks[0]
    base_attr, base_body = split_block(b["text"])
    expected = base_src[: b["start"]] + "#[cfg(test)]\nmod tests;" + base_src[b["end"] :]
    if expected != cur:
        k = next((i for i in range(min(len(expected), len(cur))) if expected[i] != cur[i]), None)
        failures.append(f"{rel}: parent altered beyond the block->decl swap (byte {k})")
        return
    sib_path = derive_sibling(rel)
    sib_src = sib_path.read_text() if sib_path.exists() else ""
    sib_attr, sib_body = head_split(sib_src)
    if [l.rstrip() for l in sib_attr] != [l.rstrip() for l in dedent4(base_attr)]:
        failures.append(f"{rel}: inner attributes rewritten")
        return
    pre_fmt = "\n".join(dedent4(base_attr) + [""] + edge(dedent4(base_body))) + "\n"
    if sib_src == pre_fmt:
        table.append(f"{rel}: moved {len(base_attr)} attr + {len(edge(dedent4(base_body)))} body lines")
    else:
        can = fmt_canonical(pre_fmt, Path.cwd())
        if sib_src != can:
            a, b = sib_src, can
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
            failures.append(
                f"{rel}: sibling diverges from the canonical rendering of the expected copy"
                f" (byte {k}; got {a[k:k+40]!r} vs expected {b[k:k+40]!r})"
            )
            return
        table.append(
            f"{rel}: moved {len(base_attr)} attr + {len(edge(dedent4(base_body)))} body lines"
            " (cargo fmt reflowed; canonical form equal)"
        )
    totals["attr"] += len(base_attr)
    totals["body"] += len(edge(dedent4(base_body)))


def main():
    if len(sys.argv) != 2:
        print("usage: scripts/verify_test_move.py <pre-move-rev>", file=sys.stderr)
        return 2
    repo = Path.cwd()
    if not (repo / "Cargo.toml").exists():
        print("run from the repository root", file=sys.stderr)
        return 2
    rev, failures, totals = sys.argv[1], [], {"attr": 0, "body": 0}
    table = []
    conformed = re.compile(r"^#[cfg\(test\)]\n\s*mod\s+tests\s*;", re.M)
    all_rs = sorted(repo.glob("crates/*/src/**/*.rs"))
    paths = []
    for p in all_rs:
        rel = str(p.relative_to(repo))
        cur = p.read_text()
        conformed_now = bool(conformed.search(cur))
        if conformed_now:
            paths.append((p, cur, None))
            continue
        base_show = subprocess.run(
            ["git", "show", f"{rev}:{rel}"], cwd=repo, check=False, capture_output=True, text=True
        )
        if base_show.returncode != 0:
            continue  # file did not exist at the base revision; not a mover
        base_src = base_show.stdout
        if parse_inline_blocks(base_src, rel):
            paths.append((p, cur, base_src))
    for p, cur, base_src in paths:
        rel = str(p.relative_to(repo))
        if base_src is None:
            base_src = subprocess.run(
                ["git", "show", f"{rev}:{rel}"], cwd=repo, check=True, capture_output=True, text=True
            ).stdout
        compare(rel, base_src, cur, failures, table, totals)
    print("\n".join(table))
    print(f"TOTAL: {len(table)} files, {totals['attr']} attr + {totals['body']} body lines moved")
    if failures:
        print("\nFAILURES:")
        print("\n".join(failures))
        return 1
    print("OK: every moved block is byte-for-byte the original, minus one indent level")
    return 0


if __name__ == "__main__":
    sys.exit(main())
