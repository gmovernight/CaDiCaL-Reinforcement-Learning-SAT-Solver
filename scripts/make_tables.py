#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def median(xs: List[float]) -> float:
    ys = sorted([float(x) for x in xs if x is not None and math.isfinite(float(x))])
    n = len(ys)
    if n == 0:
        return float('nan')
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid-1] + ys[mid])


def par2(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return float('nan')
    total = 0.0
    for r in rows:
        t = float(r.get('time_s', float('nan')))
        to = float(r.get('timeout_s', 0.0))
        if r.get('solved'):
            total += t
        else:
            total += 2.0 * to
    return total / float(len(rows))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_algo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_algo[r.get('algo', 'unknown')].append(r)
    out: Dict[str, Dict[str, float]] = {}
    for algo, group in by_algo.items():
        n = len(group)
        solved_rows = [r for r in group if r.get('solved')]
        solved = len(solved_rows)
        solved_pct = 100.0 * solved / float(n or 1)
        solved_times = [float(r.get('time_s', float('nan'))) for r in solved_rows]
        median_solved = median(solved_times)
        # All times clipped at timeout for median_all
        clipped_times = []
        for r in group:
            t = float(r.get('time_s', float('nan')))
            if not r.get('solved'):
                to = float(r.get('timeout_s', 0.0))
                t = to
            clipped_times.append(t)
        median_all = median(clipped_times)
        out[algo] = {
            'n': float(n),
            'solved': float(solved),
            'solved_pct': float(solved_pct),
            'median_solved_s': float(median_solved),
            'median_all_clipped_s': float(median_all),
            'par2_s': float(par2(group)),
        }
    return out


def write_csv(summary: Dict[str, Dict[str, float]], out_csv: Path) -> None:
    algos = sorted(summary.keys())
    fields = ['algo', 'n', 'solved', 'solved_pct', 'median_solved_s', 'median_all_clipped_s', 'par2_s']
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in algos:
            row = {'algo': a}
            row.update(summary[a])
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description='Make tables (CSV + stdout) from one or more JSONLs (baselines + A2C).')
    ap.add_argument('--in', dest='inputs', nargs='+', required=True, help='input JSONL files to merge')
    ap.add_argument('--out', required=True, help='output CSV path')
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for p in args.inputs:
        rows.extend(load_jsonl(Path(p)))
    summary = summarize(rows)
    write_csv(summary, Path(args.out))

    # Pretty print to stdout
    print('algo'.rjust(16), 'n'.rjust(6), 'solved'.rjust(8), 'solved%'.rjust(9), 'median_s'.rjust(12), 'median_all'.rjust(13), 'PAR-2'.rjust(10))
    for algo in sorted(summary.keys()):
        s = summary[algo]
        print(f"{algo:>16} {int(s['n']):6d} {int(s['solved']):8d} {s['solved_pct']:9.2f} {s['median_solved_s']:12.3f} {s['median_all_clipped_s']:13.3f} {s['par2_s']:10.3f}")
    print('Wrote', args.out)


if __name__ == '__main__':
    main()

