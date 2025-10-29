#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


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


def par2(rows: List[Dict]) -> float:
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


def summarize_by_algo(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    by_algo: Dict[str, List[Dict]] = {}
    for r in rows:
        a = r.get('algo', 'unknown')
        by_algo.setdefault(a, []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for algo, group in by_algo.items():
        solved = sum(1 for r in group if r.get('solved'))
        out[algo] = {
            'n': float(len(group)),
            'solved': float(solved),
            'par2': float(par2(group)),
        }
    return out


def plot_bars(summary: Dict[str, Dict[str, float]], title: str, out_png: Path) -> None:
    # Stable order: group A2C first, then cadical_default, evsids_top1, random_topK, others
    order = ['a2c_greedy', 'a2c_sample', 'cadical_default', 'evsids_top1', 'random_topK']
    algos_all = list(summary.keys())
    # keep declared order, then any remaining algos alphabetically
    algos = [a for a in order if a in summary] + sorted([a for a in algos_all if a not in order])

    solved = [summary[a]['solved'] for a in algos]
    par2s = [summary[a]['par2'] for a in algos]

    # Consistent color palette across families
    palette = {
        'a2c_greedy': '#E45756',
        'a2c_sample': '#72B7B2',
        'cadical_default': '#4C78A8',
        'evsids_top1': '#54A24B',
        'random_topK': '#F58518',
    }
    colors = [palette.get(a, '#999999') for a in algos]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Solved count bars
    ax[0].bar(algos, solved, color=colors)
    ax[0].set_title(f'Solved count\n{title}')
    ax[0].set_ylabel('Solved')
    ax[0].set_xticklabels(algos, rotation=20, ha='right')

    # PAR-2 bars
    ax[1].bar(algos, par2s, color=colors)
    ax[1].set_title(f'PAR-2 (lower is better)\n{title}')
    ax[1].set_ylabel('PAR-2 (s)')
    ax[1].set_xticklabels(algos, rotation=20, ha='right')

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Plot A2C eval vs baselines (Solved count and PAR-2).')
    ap.add_argument('--baselines', required=True, help='path to baselines JSONL')
    ap.add_argument('--eval', required=True, help='path to A2C eval JSONL')
    ap.add_argument('--out', required=True, help='output PNG path')
    ap.add_argument('--title', default='')
    args = ap.parse_args()

    base_rows = load_jsonl(Path(args.baselines))
    eval_rows = load_jsonl(Path(args.eval))

    # Merge rows and relabel A2C algo if needed
    merged = []
    merged.extend(base_rows)
    for r in eval_rows:
        rr = dict(r)
        # Ensure a consistent algo label
        rr['algo'] = rr.get('algo', 'a2c')
        merged.append(rr)

    summary = summarize_by_algo(merged)
    plot_bars(summary, args.title, Path(args.out))
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
