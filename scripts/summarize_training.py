#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import math


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # Skip malformed lines
                pass
    return rows


def mean(x: List[float]) -> float:
    xs = [float(v) for v in x if v is not None and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else float("nan")


def finite_ratio(x: List[float]) -> float:
    xs = [float(v) for v in x]
    return sum(1 for v in xs if math.isfinite(v)) / (len(xs) or 1)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def summarize(path: Path, tail: int = 5) -> None:
    rows = load_rows(path)
    n = len(rows)
    print(f"file: {path}")
    print(f"rows: {n}")
    if n == 0:
        return

    # Extract series
    loss = [r.get('loss') for r in rows]
    ploss = [r.get('policy_loss') for r in rows]
    vloss = [r.get('value_loss') for r in rows]
    ent = [r.get('entropy') for r in rows]
    grad = [r.get('grad_norm') for r in rows]
    ep_ret = [r.get('ep_return') for r in rows]
    ep_len = [r.get('ep_len') for r in rows]
    solved = [1.0 if r.get('solved') else 0.0 for r in rows]
    ema_ret = [r.get('ema_ret') for r in rows]
    ema_len = [r.get('ema_len') for r in rows]
    ema_solved = [r.get('ema_solved') for r in rows]

    section("data quality")
    print("finite:",
          f"loss={finite_ratio(loss):.3f}",
          f"ploss={finite_ratio(ploss):.3f}",
          f"vloss={finite_ratio(vloss):.3f}",
          f"ent={finite_ratio(ent):.3f}",
          f"grad={finite_ratio(grad):.3f}")

    section("means")
    print(f"loss={mean(loss):.3f}  policy={mean(ploss):.3f}  value={mean(vloss):.3f}  ent={mean(ent):.3f}  grad={mean(grad):.3f}")

    # First vs last quarter
    q = max(1, n // 4)
    first = slice(0, q)
    last = slice(n - q, n)

    section("first vs last (raw)")
    print(f"ep_return: {mean(ep_ret[first]):.3f} -> {mean(ep_ret[last]):.3f}")
    print(f"ep_len:    {mean(ep_len[first]):.3f} -> {mean(ep_len[last]):.3f}")
    print(f"solved:    {mean(solved[first]):.3f} -> {mean(solved[last]):.3f}")

    def first_valid(a: List[float]) -> float:
        for v in a:
            if v is not None and math.isfinite(float(v)):
                return float(v)
        return float('nan')

    section("EMA first vs last")
    print(f"ema_ret:    {first_valid(ema_ret):.3f} -> {float(ema_ret[-1]):.3f}")
    print(f"ema_len:    {first_valid(ema_len):.3f} -> {float(ema_len[-1]):.3f}")
    print(f"ema_solved: {first_valid(ema_solved):.3f} -> {float(ema_solved[-1]):.3f}")

    section(f"last {tail} rows")
    for r in rows[-tail:]:
        print({
            'upd': r.get('update'),
            'loss': r.get('loss'),
            'policy': r.get('policy_loss'),
            'value': r.get('value_loss'),
            'ent': r.get('entropy'),
            'grad': r.get('grad_norm'),
            'ep_ret': r.get('ep_return'),
            'ep_len': r.get('ep_len'),
            'solved': r.get('solved'),
            'ema_ret': r.get('ema_ret'),
            'ema_len': r.get('ema_len'),
            'ema_solved': r.get('ema_solved'),
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl', help='training JSONL log file (e.g., logs/tinyfit/train.jsonl)')
    ap.add_argument('--tail', type=int, default=5, help='show last N rows')
    args = ap.parse_args()
    summarize(Path(args.jsonl), tail=args.tail)


if __name__ == '__main__':
    main()

