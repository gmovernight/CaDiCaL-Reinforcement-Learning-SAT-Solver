
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def prepare_times(rows: List[Dict], policy: str = "clip") -> Dict[str, List[float]]:
    by_algo: Dict[str, List[float]] = {}
    for r in rows:
        algo = r.get("algo", "unknown")
        t = float(r.get("time_s", float("inf")))
        to = float(r.get("timeout_s", 0.0))
        solved = bool(r.get("solved", False))
        if not solved:
            if policy == "clip":
                t = to
            elif policy == "par2":
                t = 2.0 * to
            else:
                # skip unsolved
                continue
        by_algo.setdefault(algo, []).append(float(t))
    return by_algo


def plot_cactus(by_algo: Dict[str, List[float]], title: str, out_png: Path, logy: bool = False) -> None:
    plt.figure(figsize=(8, 5))
    for algo, times in sorted(by_algo.items()):
        if not times:
            continue
        ts = np.sort(np.asarray(times, dtype=float))
        xs = np.arange(1, len(ts) + 1)
        plt.step(ts, xs, where="post", label=algo)
    plt.xlabel("Time (s)")
    plt.ylabel("# Instances solved (<= time)")
    plt.title(title)
    if logy:
        plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Cactus plot from one or more JSONL runs (baselines and/or A2C)")
    ap.add_argument('--in', dest='inputs', nargs='+', required=True, help='input JSONL files')
    ap.add_argument('--out', required=True, help='output PNG path')
    ap.add_argument('--title', default='')
    ap.add_argument('--policy', choices=['clip','par2','solved-only'], default='clip', help='unsolved handling')
    ap.add_argument('--logy', action='store_true', help='log scale for Y')
    args = ap.parse_args()

    rows: List[Dict] = []
    for p in args.inputs:
        rows.extend(load_jsonl(Path(p)))
    by_algo = prepare_times(rows, policy=args.policy)
    plot_cactus(by_algo, args.title, Path(args.out), logy=bool(args.logy))
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
