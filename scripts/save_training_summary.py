#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # skip malformed lines
                pass
    return rows


def _mean(xs: List[float]) -> float:
    vs = [float(v) for v in xs if v is not None and math.isfinite(float(v))]
    return sum(vs) / len(vs) if vs else float("nan")


def _finite_ratio(xs: List[float]) -> float:
    vs = [float(v) for v in xs]
    return sum(1 for v in vs if math.isfinite(v)) / (len(vs) or 1)


def save_summary(jsonl_path: Path, out_path: Path, tail: int = 5) -> None:
    rows = _load_rows(jsonl_path)
    n = len(rows)
    summary: Dict[str, Any] = {
        "file": str(jsonl_path),
        "rows": n,
    }
    if n == 0:
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        return

    # Timing and throughput (based on per-row time_s timestamps, if present)
    try:
        ts = sorted(float(r["time_s"]) for r in rows if "time_s" in r)
    except Exception:
        ts = []
    if ts:
        start_time = float(ts[0])
        end_time = float(ts[-1])
        duration_s = max(0.0, end_time - start_time)
        updates = int(n)
        updates_per_sec = (updates / duration_s) if duration_s > 0 else float("nan")
        summary.update({
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "updates": updates,
            "updates_per_sec": updates_per_sec,
        })

    # Series
    def arr(key: str) -> List[float]:
        return [r.get(key) for r in rows]

    loss = arr("loss")
    ploss = arr("policy_loss")
    vloss = arr("value_loss")
    ent = arr("entropy")
    grad = arr("grad_norm")
    ep_ret = arr("ep_return")
    ep_len = arr("ep_len")
    solved = [1.0 if r.get("solved") else 0.0 for r in rows]
    ema_ret = arr("ema_ret")
    ema_len = arr("ema_len")
    ema_solved = arr("ema_solved")

    # Basic quality/means
    summary["finite"] = {
        "loss": _finite_ratio(loss),
        "policy_loss": _finite_ratio(ploss),
        "value_loss": _finite_ratio(vloss),
        "entropy": _finite_ratio(ent),
        "grad_norm": _finite_ratio(grad),
    }
    summary["means"] = {
        "loss": _mean(loss),
        "policy_loss": _mean(ploss),
        "value_loss": _mean(vloss),
        "entropy": _mean(ent),
        "grad_norm": _mean(grad),
    }

    # First vs last quarter
    q = max(1, n // 4)
    first = slice(0, q)
    last = slice(n - q, n)
    def mean_slice(xs: List[float], s: slice) -> float:
        sl = xs[s]
        return _mean(sl) if isinstance(xs, list) else float("nan")
    summary["first_last_raw"] = {
        "ep_return": [mean_slice(ep_ret, first), mean_slice(ep_ret, last)],
        "ep_len": [mean_slice(ep_len, first), mean_slice(ep_len, last)],
        "solved_rate": [mean_slice(solved, first), mean_slice(solved, last)],
    }

    def first_valid(xs: List[float]) -> float:
        for v in xs:
            if v is not None and math.isfinite(float(v)):
                return float(v)
        return float("nan")
    summary["first_last_ema"] = {
        "ema_ret": [first_valid(ema_ret), float(ema_ret[-1]) if math.isfinite(float(ema_ret[-1])) else float("nan")],
        "ema_len": [first_valid(ema_len), float(ema_len[-1]) if math.isfinite(float(ema_len[-1])) else float("nan")],
        "ema_solved": [first_valid(ema_solved), float(ema_solved[-1]) if math.isfinite(float(ema_solved[-1])) else float("nan")],
    }

    # Tail rows (small digest)
    tail_rows = []
    for r in rows[-tail:]:
        tail_rows.append({
            "update": r.get("update"),
            "ep_return": r.get("ep_return"),
            "ep_len": r.get("ep_len"),
            "solved": r.get("solved"),
            "loss": r.get("loss"),
            "grad_norm": r.get("grad_norm"),
            "ema_ret": r.get("ema_ret"),
            "ema_len": r.get("ema_len"),
            "ema_solved": r.get("ema_solved"),
        })
    summary["tail"] = tail_rows

    out_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Save training summary JSON from a training JSONL log")
    ap.add_argument("jsonl", help="path to training JSONL (e.g., logs/<run>/train.jsonl)")
    ap.add_argument("--out", default=None, help="output summary JSON path (default: <logdir>/summary.json)")
    ap.add_argument("--tail", type=int, default=5, help="number of tail rows to include")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print("ERROR: missing", jsonl_path)
        raise SystemExit(1)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = jsonl_path.parent / "summary.json"
    save_summary(jsonl_path, out_path, tail=args.tail)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
