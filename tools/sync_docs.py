"""Rewrite the auto-generated numeric blocks in the docs from ``reports/``.

    python -m tools.sync_docs

Keeps the README's baseline table and target from drifting away from the
measurements.  Every block is delimited by ``<!-- AUTO:name -->`` markers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def baseline_table(payload) -> str:
    """Combined scores only.

    The per-setting breakdown stays in ``reports/``, which is author and reviewer
    material: publishing ``Q_C`` per family would hand a solving agent a partial
    reading of the hidden setting that the task is built to withhold.
    """
    lines = ["| Family | Selected configuration | `S` |", "|---|---|---|"]
    for row in payload["families"]:
        config = row["best_config"] or "--"
        lines.append(f"| {row['family']} | `{config}` | **{row['S']:.2f}** |")
    lines += ["",
              f"`S*` = **{payload['S_star']:.2f}** ({payload['strongest']}, "
              f"`{payload['strongest_config'] or '--'}`).  The weakest family is "
              f"{payload['weakest']} at {payload['weakest_S']:.2f}, and that is what "
              f"`agent/solution.py` ships with.  Scores are means over "
              f"{len(payload['official_seeds'])} run seeds; the spread between "
              f"weakest and strongest is {payload['spread']:.2f} points.",
              "",
              f"Each family is shown at its best official configuration.  Letting "
              f"the public proxy diagnostic pick instead would set the bar at "
              f"{payload['S_star_if_selected_on_proxy']:.2f} rather than "
              f"{payload['S_star']:.2f} -- a measure of how little the proxy "
              f"transfers, and a warning against trusting it too far.  "
              f"{payload['configs_in_grid_above_S_star']} of the "
              f"{payload['grid_size']} swept configurations exceed `S*`."]
    return "\n".join(lines)


def gate_table(payload) -> str:
    lines = ["| Gate | Verdict |", "|---|---|"]
    for name, gate in payload["gates"].items():
        lines.append(f"| {name.replace('_', ' ')} | "
                     f"{'PASS' if gate['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("Full numbers in [`reports/gates.md`](reports/gates.md).")
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- AUTO:{name} -->\n).*?(\n<!-- /AUTO:{name} -->)",
                         re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"marker AUTO:{name} not found")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> int:
    readme = REPO_ROOT / "README.md"
    text = readme.read_text()
    text = replace_block(text, "baselines",
                         baseline_table(json.loads((REPORTS / "baselines.json").read_text())))
    gates_path = REPORTS / "gates.json"
    if gates_path.exists():
        text = replace_block(text, "gates",
                             gate_table(json.loads(gates_path.read_text())))
    readme.write_text(text)
    print("README.md updated from reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
