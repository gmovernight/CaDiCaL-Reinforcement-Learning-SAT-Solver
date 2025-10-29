#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def load_rows(path: Path) -> List[Dict]:
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


def main():
    ap = argparse.ArgumentParser(description='Plot training EMAs (reward, length, solved).')
    ap.add_argument('jsonl', help='training JSONL path (logs/.../train.jsonl)')
    ap.add_argument('--out', default=None, help='output PNG (default: alongside JSONL as training_curves.png)')
    args = ap.parse_args()

    p = Path(args.jsonl)
    rows = load_rows(p)
    if not rows:
        print('No rows in', p)
        return

    upd = np.array([r.get('update', i) for i, r in enumerate(rows)], float)
    ema_ret = np.array([r.get('ema_ret', np.nan) for r in rows], float)
    ema_len = np.array([r.get('ema_len', np.nan) for r in rows], float)
    ema_sol = np.array([r.get('ema_solved', np.nan) for r in rows], float)

    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(upd, ema_ret, color='#4C78A8')
    ax[0].set_ylabel('EMA reward')
    ax[0].grid(alpha=0.3)
    ax[1].plot(upd, ema_len, color='#F58518')
    ax[1].set_ylabel('EMA length')
    ax[1].grid(alpha=0.3)
    ax[2].plot(upd, ema_sol, color='#54A24B')
    ax[2].set_ylabel('EMA solved rate')
    ax[2].set_xlabel('update')
    ax[2].grid(alpha=0.3)
    fig.tight_layout()

    out = Path(args.out) if args.out else (p.parent / 'training_curves.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('Wrote', out)


if __name__ == '__main__':
    main()

