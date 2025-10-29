#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl', help='results JSONL file (from run_baselines.py)')
    args = ap.parse_args()

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    by_algo = defaultdict(list)
    for r in rows:
        by_algo[r.get('algo', 'unknown')].append(r)

    print(f"Loaded {len(rows)} rows; {len(by_algo)} algos")
    for algo, group in by_algo.items():
        n = len(group)
        solved = sum(1 for r in group if r.get('solved'))
        # Use per-row timeout_s to compute PAR-2
        par2_sum = 0.0
        for r in group:
            t = float(r.get('time_s', math.inf))
            to = float(r.get('timeout_s', 0.0))
            if r.get('solved'):
                par2_sum += t
            else:
                par2_sum += 2.0 * to
        par2 = par2_sum / float(n) if n > 0 else float('nan')
        print(f"algo={algo:>14}  n={n:3d}  solved={solved:3d}  PAR-2={par2:.3f}")


if __name__ == '__main__':
    main()

