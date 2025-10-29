#!/usr/bin/env python3
from __future__ import annotations
import os, sys, math, time, json, random, argparse
from typing import Dict, Any, List

import numpy as np
import yaml

# Ensure we can import env and bridge module when run from repo root
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BRIDGE_BUILD = os.path.join(REPO, 'bridge', 'build')
if REPO not in sys.path: sys.path.insert(0, REPO)
if BRIDGE_BUILD not in sys.path: sys.path.insert(0, BRIDGE_BUILD)

import torch
import torch.nn.functional as F

from env.sat_env import SatEnv
from rl.models.a2c import A2C
from rl.train.a2c_trainer import A2CTrainer
from rl.rollout.buffer import RolloutBuffer


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def list_instances(split: str, limit: int | None = None) -> List[str]:
    """Recursively collect .cnf files in the split folder.

    This supports family subfolders like data/instances/train/f50-218/*.cnf.
    """
    d = os.path.join(REPO, 'data', 'instances', split)
    files: List[str] = []
    for dirpath, _, filenames in os.walk(d):
        for fn in filenames:
            if fn.endswith('.cnf'):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def build_env(cfg: Dict[str, Any]) -> SatEnv:
    return SatEnv(K=int(cfg.get('K', 16)),
                  M=int(cfg.get('M', 50)),
                  timeout_ms=int(cfg.get('timeout_ms', 5)))


def obs_to_tensors(obs: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    g = torch.tensor(np.asarray(obs['global'], np.float32))[None, :]
    c = torch.tensor(np.asarray(obs['cands'], np.float32))[None, :, :]
    m = (c[..., 0] != 0.0)  # bool mask: var != 0
    return { 'g': g.to(device), 'c': c.to(device), 'm': m.to(device) }


def rollout(env: SatEnv, net: A2C, inst_path: str, T: int, gamma: float, device: str) -> Dict[str, Any]:
    # Initialize
    env.reset_instance(inst_path)
    obs, _ = env.reset()
    traj = { 'global': [], 'cands': [], 'mask': [], 'actions': [], 'rewards': [], 'dones': [], 'values': [] }
    buf = RolloutBuffer(n_steps=T, K=env.K)
    ep_return = 0.0
    ep_len = 0
    solved_flag = False

    for t in range(T):
        tens = obs_to_tensors(obs, device)
        with torch.no_grad():
            logits, value = net(tens['g'], tens['c'], tens['m'])
            probs = F.softmax(logits, dim=-1)
        # Sample valid action under mask
        a = int(torch.distributions.Categorical(probs=probs).sample().item())
        # Record
        traj['global'].append(tens['g'].squeeze(0).cpu().numpy())
        traj['cands'].append(tens['c'].squeeze(0).cpu().numpy())
        traj['mask'].append(tens['m'].squeeze(0).cpu().numpy())
        traj['actions'].append(a)
        v_scalar = float(value.squeeze(0).item())
        traj['values'].append(v_scalar)

        # Step
        _, r, done, info = env.step(a)
        r = float(r)
        traj['rewards'].append(r)
        traj['dones'].append(bool(done))
        # mirror minimal fields into buffer for GAE
        buf.values.append(v_scalar)
        buf.rewards.append(r)
        buf.dones.append(bool(done))
        ep_return += r
        ep_len += 1
        if done:
            solved_flag = (int(info.get('status', 0)) in (10, 20))
            break
        obs = _

    # Bootstrap value for last state and compute GAE
    with torch.no_grad():
        if traj['dones'] and traj['dones'][-1]:
            next_v = 0.0
        else:
            tens = obs_to_tensors(obs, device)
            _, v = net(tens['g'], tens['c'], tens['m'])
            next_v = float(v.squeeze(0).item())
    advantages, returns = buf.compute_gae(gamma=gamma, lam=0.95, last_value=next_v)

    # Pack batch
    batch = {
        'global': np.stack(traj['global'], axis=0),
        'cands': np.stack(traj['cands'], axis=0),
        'mask': np.stack(traj['mask'], axis=0),
        'actions': np.asarray(traj['actions'], dtype=np.int64),
        'advantages': np.asarray(advantages, dtype=np.float32),
        'returns': np.asarray(returns, dtype=np.float32),
    }
    batch['ep_return'] = ep_return
    batch['ep_len'] = ep_len
    batch['solved'] = solved_flag
    return batch


def save_ckpt(path: str, net: A2C, opt) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        torch.save({'model': net.state_dict(), 'opt': opt.state_dict()}, path)
    except Exception:
        torch.save({'model': net.state_dict()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='cfg/a2c.yaml')
    ap.add_argument('--split', default='train')
    ap.add_argument('--val-split', default='val')
    ap.add_argument('--max-updates', type=int, default=1)
    ap.add_argument('--logdir', default='logs/smoke1')
    ap.add_argument('--limit', type=int, default=None, help='max instances (default: all)')
    ap.add_argument('--family', default=None, help='optional substring to filter instance paths (e.g., f50-218)')
    # Warm-start / resume
    ap.add_argument('--init-ckpt', default=None, help='warm-start model weights from checkpoint path')
    ap.add_argument('--resume', action='store_true', help='also load optimizer state if present in checkpoint')
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    seed = int(cfg.get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = str(cfg.get('device', 'cpu'))

    # Build
    env = build_env(cfg)
    K = int(cfg.get('K', 16))
    net = A2C(global_dim=11, cand_dim=5, K=K,
              hidden=int(cfg.get('hidden', 128)),
              cand_hidden=int(cfg.get('cand_hidden', 64)))
    trainer = A2CTrainer(net, lr=float(cfg.get('lr', 3e-4)),
                         value_coef=float(cfg.get('value_coef', 0.5)),
                         entropy_coef=float(cfg.get('entropy_coef', 0.01)),
                         grad_clip=float(cfg.get('grad_clip', 0.7)),
                         device=device)

    # Optional warm-start from a previous checkpoint
    if args.init_ckpt:
        try:
            state = torch.load(args.init_ckpt, map_location=device)
            sd = state.get('model', state)
            net.load_state_dict(sd)
            if args.resume and isinstance(state, dict) and 'opt' in state:
                try:
                    trainer.opt.load_state_dict(state['opt'])
                except Exception:
                    pass
            print(f"Warm-started from {args.init_ckpt} (resume={bool(args.resume)})")
        except Exception as e:
            print(f"WARN: failed to warm-start from {args.init_ckpt}: {e}")

    # Instances
    train_set = list_instances(args.split, None)
    if args.family:
        fam = args.family.lower()
        before = len(train_set)
        train_set = [p for p in train_set if fam in p.lower()]
        print(f"Filtered by family='{args.family}': {before} -> {len(train_set)} instances")
    if args.limit is not None:
        train_set = train_set[: int(args.limit)]
    if not train_set:
        raise SystemExit(f"No instances in data/instances/{args.split}")

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'cfg.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    gamma = float(cfg.get('gamma', 0.99))
    T = int(cfg.get('rollout_T', 16))

    # Prepare JSONL logger
    log_jsonl = os.path.join(args.logdir, 'train.jsonl')
    # One or few updates; track simple moving averages
    ema_alpha = 0.1
    ema_ret = None
    ema_len = None
    ema_solved = None
    # One or few updates
    # Entropy annealing schedule (linear)
    ent_start = float(cfg.get('entropy_coef', 0.02))
    ent_end = 0.0
    total_updates = int(args.max_updates)
    for upd in range(args.max_updates):
        # update entropy coefficient
        frac = (upd / max(1, total_updates))
        trainer.entropy_coef = ent_start + (ent_end - ent_start) * frac
        inst = random.choice(train_set)
        batch = rollout(env, net, inst, T=T, gamma=gamma, device=device)
        stats = trainer.step(batch)
        # Update EMAs
        ema_ret = batch['ep_return'] if ema_ret is None else (1-ema_alpha)*ema_ret + ema_alpha*batch['ep_return']
        ema_len = batch['ep_len']    if ema_len is None else (1-ema_alpha)*ema_len + ema_alpha*batch['ep_len']
        ema_solved = (1.0 if batch['solved'] else 0.0) if ema_solved is None else (1-ema_alpha)*ema_solved + ema_alpha*(1.0 if batch['solved'] else 0.0)

        print(
            f"update={upd} loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} "
            f"value={stats['value_loss']:.4f} ent={stats['entropy']:.4f} grad={stats.get('grad_norm', float('nan')):.3f} "
            f"ep_ret={batch['ep_return']:.3f} ep_len={batch['ep_len']} solved={int(batch['solved'])} "
            f"ema_ret={ema_ret:.3f} ema_len={ema_len:.2f} ema_solved={ema_solved:.2f}"
        )
        # Persist the same stats to JSONL for later inspection
        rec = {
            'update': upd,
            'instance': inst,
            'loss': stats['loss'],
            'policy_loss': stats['policy_loss'],
            'value_loss': stats['value_loss'],
            'entropy': stats['entropy'],
            'grad_norm': stats.get('grad_norm', float('nan')),
            'entropy_coef': float(trainer.entropy_coef),
            'ep_return': batch['ep_return'],
            'ep_len': batch['ep_len'],
            'solved': bool(batch['solved']),
            'ema_ret': ema_ret,
            'ema_len': ema_len,
            'ema_solved': ema_solved,
            'time_s': time.time(),
        }
        with open(log_jsonl, 'a') as jf:
            jf.write(json.dumps(rec) + "\n")

    # Save checkpoint
    ckpt_path = os.path.join(args.logdir, 'ckpt.pt')
    save_ckpt(ckpt_path, net, trainer.opt)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
