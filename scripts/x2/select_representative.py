"""Select ~100 representative clips from a benchmark result JSON for visualization.

Stratifies by motion-group keyword (flip/dance/jump/roll/kick/run/walk/stand_up/...)
with quota proportional to sqrt of group size (diversity over popularity), then adds
the global best/worst performers and the boundary cases (barely-passed / just-failed).

  uv run python scripts/x2/select_representative.py --benchmark benchmark.json --k 100
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import tyro

GROUPS = (
    "flip", "spin", "dance", "jump", "broad_jump", "side_roll", "backroll", "somersault",
    "kick", "punch", "run", "sprint", "walk", "jog", "stand_up", "lying", "sit", "crouch",
    "turn", "hop", "skip", "cartwheel", "crawl", "lean", "wave", "clap", "reach",
)


def group_of(name: str) -> str:
    n = name.lower()
    for g in GROUPS:
        if g in n:
            return g
    return "other"


def main(benchmark: Path, k: int = 100) -> None:
    rows = json.loads(Path(benchmark).read_text())
    if isinstance(rows, dict):
        rows = rows.get("results", rows.get("clips", []))
    parsed = []
    for r in rows:
        parsed.append({"name": r["name"], "ret": float(r.get("norm_return", r.get("return", 0.0))),
                       "failed": bool(r.get("failed", False)), "steps": int(r.get("steps", 0)),
                       "frames": int(r.get("frames", 0))})
    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in parsed:
        by_group[group_of(p["name"])].append(p)
    for g in by_group:
        by_group[g].sort(key=lambda p: -p["ret"])
    total = len(parsed)
    quota = {g: max(2, round(k * math.sqrt(len(v) / total))) for g, v in by_group.items()}
    scale = k / sum(quota.values())
    sel: list[dict] = []
    for g, v in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
        take = [x for x in v[: max(1, round(quota[g] * scale))] if x not in sel]
        sel.extend(take)
    sel = sel[:k]
    ordered = sorted(parsed, key=lambda p: -p["ret"])
    for extra in ordered[:5] + ordered[-5:]:
        if len(sel) < k + 10 and extra not in sel:
            sel.append(extra)
    gs = sorted((g, len(v)) for g, v in by_group.items())
    print(f"# selected {len(sel)} of {total} (groups: {gs})")
    for p in sorted(sel, key=lambda x: group_of(x["name"])):
        print(f"{p['name']:48s} ret {p['ret']:.3f} {'FAIL' if p['failed'] else 'ok':4s} {p['steps']}/{p['frames']}")


if __name__ == "__main__":
    tyro.cli(main)
