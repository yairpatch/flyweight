"""Fail when a change to what ships is not accompanied by a version bump.

CI runs this on every pull request. The rule it enforces:

- A pull request that touches anything the wheel ships (the Python package,
  the native runtime, the web UI, packaging metadata) must raise the version
  in ``pyproject.toml`` relative to the base branch.
- The version must not already be a ``v*`` tag: a released number is spent.

Docs, plans, benchmarks, tools and CI configuration change freely.

Usage: ``python tools/check_version_bump.py <base-ref>``
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Anything under these paths ends up in the wheel or changes how it is built.
SHIPPED = (
    "src/",
    "native/",
    "web/src/",
    "web/package.json",
    "web/pnpm-lock.yaml",
    "pyproject.toml",
    "setup.py",
    "MANIFEST.in",
)

# Development-only trees inside the shipped paths.
EXEMPT = (
    "native/tests/",
    "native/bench/",
    "native/tools/",
)

VERSION = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+|\.dev\d+|\.post\d+)?$")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def _version_at(revision: str | None) -> str:
    if revision is None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    else:
        text = _git("show", f"{revision}:pyproject.toml")
    return tomllib.loads(text)["project"]["version"]


def _ships(path: str) -> bool:
    return path.startswith(SHIPPED) and not path.startswith(EXEMPT)


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3])


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    base = argv[1]
    current = _version_at(None)
    previous = _version_at(base)
    # Against the merge base, and including the working tree, so the check
    # answers the same way locally before the commit as it does in CI after.
    merge_base = _git("merge-base", base, "HEAD").strip()
    changed = _git("diff", "--name-only", merge_base).split()
    shipped = sorted(path for path in changed if _ships(path))

    if not VERSION.match(current):
        print(f"pyproject.toml version {current!r} is not a release version")
        return 1

    if not shipped:
        print(f"no shipped files changed; version {current} may stay")
        return 0

    tags = set(_git("tag", "--list", "v*").split())
    if f"v{current}" in tags:
        print(
            f"version {current} is already released as tag v{current}; "
            "bump it in pyproject.toml"
        )
        return 1

    if current == previous:
        print(
            f"{len(shipped)} shipped file(s) changed but pyproject.toml still says "
            f"{current} (same as {base}); bump the version:"
        )
        for path in shipped:
            print(f"  {path}")
        return 1

    if _parse(current) <= _parse(previous):
        print(f"version went from {previous} to {current}; it must go up")
        return 1

    print(f"version {previous} -> {current} for {len(shipped)} shipped file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
