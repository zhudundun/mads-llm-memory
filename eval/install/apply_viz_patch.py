"""Re-apply the local fix to ai_data_science_team's data_visualization_agent.

Upstream bug
------------
At ai_data_science_team/agents/data_visualization_agent.py, three regex
literals inside the inner function `_extract_missing_columns` use doubled
backslashes within a raw-string. That is, the file contains
    r"\\\\['([^']+)'\\\\]\\\\s+not in index"      # what's on disk
when the intent (and what Python's regex engine should see at runtime) is
    r"\\['([^']+)'\\]\\s+not in index"
At runtime the regex engine treats `\\\\[` as "literal `\\` + literal `[`",
which opens a character class that swallows the actual `(` of the capture
group. The matching `)` then has no opener and the call raises
    re.PatternError: unbalanced parenthesis at position 10
in `DataVisualizationAgent`'s column-retry path. The same pattern affects
two sibling regex calls on adjacent lines, which silently fail to match.

What this script does
---------------------
Reads the installed file, locates the three target lines (recognized by the
exact call shape `re.findall(r"…")` / `re.search(r"…")` on the buggy lines),
and halves doubled backslashes inside each line's raw-string literal.

Properties
----------
* **Idempotent.** Running on an already-patched file is a no-op (a target
  line whose regex literal contains no doubled backslashes is left alone).
* **Conservative.** Only touches lines whose stripped form starts with the
  known buggy regex calls. Doesn't sweep the rest of the file.
* **Self-verifying.** After patching, reloads the module and ensures every
  patched regex compiles.

When to run
-----------
After every `pip install --force-reinstall ai-data-science-team` (or any
other upgrade path that overwrites the installed source). Cheap to call
on every eval run if you want belt-and-suspenders:

    python eval/install/apply_viz_patch.py            # apply (idempotent)
    python eval/install/apply_viz_patch.py --check    # exit 1 if drift, no writes

We do NOT contribute the fix upstream. This patch is intentionally local.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


# Captures `r"..."` raw-string literals, with proper escape handling so we
# don't terminate on `\"` inside the literal.
_RAW_STR_LITERAL = re.compile(r'r"((?:[^"\\]|\\.)*)"')

# A line is patchable iff its stripped form starts with one of these prefixes.
# This shape is what's in the three buggy lines in upstream, and using it as a
# gate prevents accidental edits elsewhere in the file.
_LINE_PREFIXES: Tuple[str, ...] = (
    'for match in re.findall(r"',
    'list_match = re.search(r"',
)


def _import_target_module():
    import ai_data_science_team.agents.data_visualization_agent as mod  # type: ignore
    return mod


def _patch_line(line: str) -> Tuple[str, bool]:
    """If `line` matches a known buggy shape and has a doubled backslash in
    its r"..." literal, return (patched_line, True). Otherwise (line, False).
    """
    stripped = line.lstrip()
    if not any(stripped.startswith(p) for p in _LINE_PREFIXES):
        return line, False
    m = _RAW_STR_LITERAL.search(line)
    if not m:
        return line, False
    inner = m.group(1)
    # `"\\\\"` here is two characters at runtime (`\\`). We're checking whether
    # the on-disk pattern literal has any doubled-backslash sequences, which is
    # the only fingerprint the bug leaves once you've handled raw-string escapes.
    if "\\\\" not in inner:
        return line, False
    # Halve doubled backslashes to a fixed point. Single-pass `.replace()`
    # is not enough — line 773's `\\\"` (4 disk chars) needs two halvings
    # to converge to the intended `\"` (2 disk chars). The three target
    # patterns in upstream never contain legitimate `\\` in their intent,
    # so iterating to fixpoint is safe for this specific patch.
    fixed_inner = inner
    while "\\\\" in fixed_inner:
        fixed_inner = fixed_inner.replace("\\\\", "\\")
    fixed_literal = 'r"' + fixed_inner + '"'
    return line[: m.start()] + fixed_literal + line[m.end() :], True


def _verify_regexes_compile(path: Path) -> list[str]:
    """Compile every r"..." literal that lives on a patched-shape line.
    Returns a list of error messages, empty on success."""
    errors: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.lstrip()
        if not any(stripped.startswith(p) for p in _LINE_PREFIXES):
            continue
        m = _RAW_STR_LITERAL.search(line)
        if not m:
            continue
        pat = m.group(1)
        try:
            re.compile(pat)
        except re.error as e:
            errors.append(f"line {line_no}: re.compile failed for r\"{pat}\": {e}")
    return errors


def apply_patch(*, check_only: bool = False, target_path: Optional[Path] = None) -> int:
    """Apply (or check) the patch. Returns 0 if state is fine, non-zero otherwise.

    Exit codes:
        0  patched successfully, OR already patched
        1  --check found drift (would patch), or post-patch verify failed
        2  cannot locate the target module / file
    """
    if target_path is None:
        try:
            mod = _import_target_module()
        except ImportError as e:
            print(f"error: ai_data_science_team not installed: {e}", file=sys.stderr)
            return 2
        target = Path(mod.__file__)  # type: ignore[arg-type]
    else:
        target = target_path
    if not target.is_file():
        print(f"error: target not a file: {target}", file=sys.stderr)
        return 2

    original_text = target.read_text()
    lines = original_text.splitlines(keepends=True)
    patched_line_nos: list[int] = []
    for i, line in enumerate(lines):
        new, changed = _patch_line(line)
        if changed:
            lines[i] = new
            patched_line_nos.append(i + 1)

    if not patched_line_nos:
        # Nothing to do. Verify the file's regex literals still compile.
        errs = _verify_regexes_compile(target)
        if errs:
            print(f"warning: file at {target} has regex literals that fail to compile:",
                  file=sys.stderr)
            for e in errs:
                print(f"  {e}", file=sys.stderr)
            return 1
        print(f"already patched: {target}")
        return 0

    if check_only:
        print(f"DRIFT: would patch {len(patched_line_nos)} line(s) at {target}: "
              f"{patched_line_nos}")
        return 1

    target.write_text("".join(lines))
    print(f"patched {len(patched_line_nos)} line(s) at {target}: {patched_line_nos}")

    # Reload + verify if we patched the real installed module.
    if target_path is None:
        importlib.reload(mod)  # type: ignore[arg-type]
    errs = _verify_regexes_compile(target)
    if errs:
        print("post-patch verification FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        return 1
    print("post-patch verification: all target regexes compile")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="report needed changes but do not write (exit 1 if drift detected)",
    )
    ap.add_argument(
        "--target",
        type=Path,
        default=None,
        help="patch this file instead of the installed ADS module (testing only)",
    )
    ns = ap.parse_args()
    return apply_patch(check_only=ns.check, target_path=ns.target)


if __name__ == "__main__":
    sys.exit(main())
