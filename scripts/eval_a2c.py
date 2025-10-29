#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
import json
import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# Repo paths for local imports
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BRIDGE_BUILD = os.path.join(REPO, 'bridge', 'build')
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if BRIDGE_BUILD not in sys.path:
    sys.path.insert(0, BRIDGE_BUILD)

from env.sat_env import SatEnv
from rl.models.a2c import A2C


def list_instances(split: str) -> List[str]:
    d = os.path.join(REPO, 'data', 'instances', split)
    files: List[str] = []
    for dirpath, _, filenames in os.walk(d):
        for fn in filenames:
            if fn.endswith('.cnf'):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def obs_to_tensors(obs, device):
    g = torch.tensor(np.asarray(obs['global'], np.float32))[None, :].to(device)
    c = torch.tensor(np.asarray(obs['cands'],  np.float32))[None, :, :].to(device)
    m = (c[..., 0] != 0.0)  # bool mask [1,K]
    return g, c, m


def main():
    ap = argparse.ArgumentParser(description='Evaluate A2C policy on a split/family; outputs JSONL comparable to baselines.')
    ap.add_argument('--split', default='val', choices=['train','val','test'])
    ap.add_argument('--family', default=None, help='substring filter for family (e.g., f50-218)')
    ap.add_argument('--ckpt', required=True, help='path to model checkpoint (logs/.../ckpt.pt)')
    ap.add_argument('--timeout_s', type=float, default=60.0)
    ap.add_argument('--out', required=True, help='output JSONL path')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--M', type=int, default=50)
    ap.add_argument('--mode', default='greedy', choices=['greedy','sample'])
    ap.add_argument('--limit', type=int, default=None, help='max instances (default: all)')
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load model
    model = A2C(global_dim=11, cand_dim=5, K=int(args.K), hidden=128)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()

    # Build instance list
    files = list_instances(args.split)
    if args.family:
        filt = args.family.lower()
        before = len(files)
        files = [p for p in files if filt in p.lower()]
        print(f"Filtered by family='{args.family}': {before} -> {len(files)}")
    if args.limit is not None:
        files = files[: int(args.limit)]

    # Ensure output dir
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Eval loop
    with open(args.out, 'w') as jf:
        for path in files:
            env = SatEnv(K=int(args.K), M=int(args.M), timeout_ms=5, limits={'time_s': float(args.timeout_s)})
            env.reset_instance(path)
            obs, _ = env.reset()
            t0 = time.perf_counter()
            steps = 0
            status = 0
            last_metrics = {}
            done = False
            while True:
                with torch.no_grad():
                    g, c, m = obs_to_tensors(obs, device)
                    logits, _ = model(g, c, m)
                    if args.mode == 'greedy':
                        a = int(torch.argmax(logits, dim=-1).item())
                    else:
                        probs = F.softmax(logits, dim=-1)
                        a = int(torch.distributions.Categorical(probs=probs).sample().item())
                obs, r, done, info = env.step(a)
                steps += 1
                status = int(info.get('status', 0))
                last_metrics = info.get('metrics', {})
                if done:
                    break
                if (time.perf_counter() - t0) >= float(args.timeout_s):
                    break

            elapsed = time.perf_counter() - t0
            rec = {
                'algo': f'a2c_{args.mode}',
                'instance': path,
                'status': status,
                'solved': bool(status in (10, 20)),
                'time_s': elapsed,
                'timeout_s': float(args.timeout_s),
                'steps': steps,
                'metrics': last_metrics,
            }
            jf.write(json.dumps(rec) + '\n')
            jf.flush()

    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

