"""Build the archive a solving agent receives, from an explicit allowlist.

    python -m tools.build_agent_bundle [--out dist/crossphase-agent.zip] [--check]

Everything the agent may see is named in ``ALLOW``.  Nothing is included by
wildcard and nothing is excluded by blocklist, because a blocklist silently
fails open the moment a new private file appears.  ``--check`` builds the archive
and greps it for private material without writing it to disk.

This exists because module separation alone is not a guarantee.  The official
target-world coordinates lived in a public module for several revisions purely
because nothing mechanically checked.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Exactly what ships to the agent.
ALLOW = (
    "README.md",
    "requirements.txt",
    "agent/solution.py",
    "agent/evaluate_public.py",
    "agent/method_references.md",
    "docs/classification_rules.md",
    "crossphase/__init__.py",
    "crossphase/core/__init__.py",
    "crossphase/core/engine.py",
    "crossphase/core/generator.py",
    "crossphase/core/methods.py",
    "crossphase/core/model.py",
    "crossphase/core/parallel.py",
    "crossphase/core/protocol.py",
    "crossphase/core/settings.py",
    "tests/__init__.py",
    "tests/run.py",
    "tests/test_scaffold.py",
)

#: Tokens that must not appear anywhere in the bundle.  Each names something the
#: task withholds: the hidden setting, the official evaluation spec, the answer.
FORBIDDEN = (
    "SETTING_C",
    "OFFICIAL_WORLDS",
    "OFFICIAL_POOL_SEED",
    "OFFICIAL_RUN_SEEDS",
    "private_specs",
    "S_star",
    "make_regime_fallback",
    "ADAPTIVE",
)

#: Paths that must never be in the bundle, whatever ALLOW says.
FORBIDDEN_PATHS = ("grader/", "gates/", "baselines/", "reports/", "tools/",
                   "docs/task_design.md", "docs/originality.md",
                   "docs/reviewer_checklist.md")


def build() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ALLOW:
            path = REPO_ROOT / name
            if not path.exists():
                raise SystemExit(f"allowlisted file is missing: {name}")
            archive.write(path, name)
    return buffer.getvalue()


def audit(blob: bytes) -> list:
    """Return every leak found in the archive.  Empty list means clean."""
    problems = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        for name in names:
            if any(name.startswith(bad) or name == bad for bad in FORBIDDEN_PATHS):
                problems.append(f"forbidden path in bundle: {name}")
        for name in names:
            if not name.endswith((".py", ".md", ".txt")):
                continue
            text = archive.read(name).decode("utf-8", "replace")
            for token in FORBIDDEN:
                if token in text:
                    problems.append(f"{name} contains forbidden token {token!r}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/crossphase-agent.zip")
    parser.add_argument("--check", action="store_true",
                        help="audit only; do not write the archive")
    args = parser.parse_args(argv)

    blob = build()
    problems = audit(blob)
    for problem in problems:
        print(f"LEAK: {problem}")
    if problems:
        return 1

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        print(f"{len(archive.namelist())} files, {len(blob) / 1024:.0f} KB, clean")
    if not args.check:
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
