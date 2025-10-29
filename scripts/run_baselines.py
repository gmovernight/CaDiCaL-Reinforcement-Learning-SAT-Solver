#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List


def find_instances(split: str) -> List[str]:
    """Recursively collect .cnf files under data/instances/<split>."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    d = os.path.join(root, 'data', 'instances', split)
    if not os.path.isdir(d):
        raise SystemExit(f"missing instances dir: {d}")
    files: List[str] = []
    for dirpath, _, filenames in os.walk(d):
        for fn in filenames:
            if fn.endswith('.cnf'):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def ensure_paths_on_sys_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    build = os.path.join(root, 'bridge', 'build')
    if root not in sys.path:
        sys.path.insert(0, root)
    if build not in sys.path:
        sys.path.insert(0, build)


def run_cadical_default(path: str, timeout_s: float) -> Dict:
    """Run the standalone CaDiCaL binary as default baseline."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    exe = os.path.join(root, 'cadical', 'build', 'cadical_app')
    if not os.path.exists(exe):
        raise SystemExit(f"cadical_app not found: {exe} (build it first)")

    start = time.perf_counter()
    try:
        out = subprocess.run(
            [exe, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            cwd=root,
        )
        elapsed = time.perf_counter() - start
        status = 0
        solved = False
        if 's SATISFIABLE' in out.stdout:
            status = 10
            solved = True
        elif 's UNSATISFIABLE' in out.stdout:
            status = 20
            solved = True

        # Parse a subset of the final statistics to build a metrics dict
        import re
        txt = out.stdout or ''
        def grab(pattern: str) -> int:
            m = re.search(pattern, txt, re.M)
            try:
                return int(m.group(1)) if m else 0
            except Exception:
                return 0

        # Lines in cadical_app statistics are prefixed with 'c '.
        # Allow for optional 'c ' prefix in matching.
        conflicts = grab(r"^\s*(?:c\s+)?conflicts:\s+(\d+)")
        decisions = grab(r"^\s*(?:c\s+)?decisions:\s+(\d+)")
        restarts  = grab(r"^\s*(?:c\s+)?restarts:\s+(\d+)")
        # Prefer search propagations (as used by env metrics); fall back to total propagations
        searchprops = grab(r"^\s*(?:c\s+)?searchprops:\s+(\d+)")
        if searchprops == 0:
            searchprops = grab(r"^\s*(?:c\s+)?propagations:\s+(\d+)")
        props_per_dec = (float(searchprops) / float(decisions)) if decisions > 0 else 0.0
        metrics = {
            'time_s': elapsed,
            'conflicts': conflicts,
            'decisions': decisions,
            'meanLBD_win': 0.0,
            'props_per_dec': props_per_dec,
            'restarts': restarts,
            'overhead_ms': 0.0,
        }

        return {
            'algo': 'cadical_default',
            'instance': path,
            'status': status,
            'solved': solved,
            'time_s': elapsed,
            'timeout_s': timeout_s,
            'details': {'returncode': out.returncode},
            'metrics': metrics,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            'algo': 'cadical_default',
            'instance': path,
            'status': 0,
            'solved': False,
            'time_s': elapsed,
            'timeout_s': timeout_s,
            'details': {'timeout': True},
            'metrics': {
                'time_s': elapsed,
                'conflicts': 0,
                'decisions': 0,
                'meanLBD_win': 0.0,
                'props_per_dec': 0.0,
                'restarts': 0,
                'overhead_ms': 0.0,
            },
        }


def run_env_policy(path: str, policy: str, timeout_s: float, K: int, M: int, seed: int | None) -> Dict:
    """Run a baseline policy using the Python env + bridge."""
    ensure_paths_on_sys_path()
    # Local import after sys.path setup
    from env.sat_env import SatEnv

    if seed is not None:
        random.seed(seed)

    env = SatEnv(K=K, M=M, timeout_ms=5, limits={'time_s': timeout_s})
    env.reset_instance(path)
    obs, info0 = env.reset()
    t0 = time.perf_counter()
    steps = 0
    status = 0
    done = False
    last_metrics = {}
    while True:
        # Choose action
        if policy == 'evsids_top1':
            action = 'argmax'
        elif policy == 'random_topK':
            action = 'random'
        else:
            raise ValueError(f"unknown policy {policy}")
        obs, r, done, info = env.step(action)
        steps += 1
        status = int(info['status'])
        last_metrics = info.get('metrics', {})
        if done:
            break
        if (time.perf_counter() - t0) >= timeout_s:
            # Hard wallclock guard
            break

    elapsed = time.perf_counter() - t0
    solved = status in (10, 20)
    return {
        'algo': policy,
        'instance': path,
        'status': status,
        'solved': solved,
        'time_s': elapsed,
        'timeout_s': timeout_s,
        'steps': steps,
        'metrics': last_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--timeout_s', type=float, default=120.0)
    ap.add_argument('--limit', type=int, default=None, help='max instances (default: all)')
    ap.add_argument('--out', default='results/baselines.jsonl')
    ap.add_argument('--family', default=None, help='optional substring to filter instance paths (e.g., f50-218)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--M', type=int, default=50)
    args = ap.parse_args()

    instances = find_instances(args.split)
    if args.family:
        filt = args.family.lower()
        before = len(instances)
        instances = [p for p in instances if filt in p.lower()]
        print(f"Filtered by family='{args.family}': {before} -> {len(instances)} instances")
    if args.limit is not None:
        instances = instances[: int(args.limit)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, 'w') as f:
        # Default CaDiCaL
        for p in instances:
            rec = run_cadical_default(p, args.timeout_s)
            f.write(json.dumps(rec) + '\n')
            f.flush()

        # Random top-K
        for p in instances:
            rec = run_env_policy(p, 'random_topK', args.timeout_s, args.K, args.M, args.seed)
            f.write(json.dumps(rec) + '\n')
            f.flush()

        # EVSIDS top-1
        for p in instances:
            rec = run_env_policy(p, 'evsids_top1', args.timeout_s, args.K, args.M, args.seed)
            f.write(json.dumps(rec) + '\n')
            f.flush()

    print(f"Wrote {args.out} ({len(instances)*3} lines)")


if __name__ == '__main__':
    main()
