#!/usr/bin/env python3
import sys
import os

def main():
    # Ensure bridge module can be found if built locally
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    build_path = os.path.join(repo_root, 'bridge', 'build')
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if build_path not in sys.path:
        sys.path.insert(0, build_path)

    from env.sat_env import SatEnv

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/instances/val/long.cnf'
    K = int(os.environ.get('K', '16'))
    M = int(os.environ.get('M', '50'))
    timeout_ms = int(os.environ.get('TIMEOUT_MS', '5'))
    max_steps = int(os.environ.get('STEPS', '20'))
    policy = os.environ.get('POLICY', 'argmax')  # or 'random'

    env = SatEnv(K=K, M=M, timeout_ms=timeout_ms)
    env.reset_instance(path)
    obs, info = env.reset()

    # Baselines for manual reporting
    m = env.solver.get_metrics()
    last_ppd = float(m.get('props_per_dec', 0.0))
    last_time = float(m.get('time_s', 0.0))

    print(f"debug_env_rewards: path={path} K={K} M={M} timeout_ms={timeout_ms}")
    print(f"step=-1 ppd={last_ppd:.6f} time={last_time:.6f}")

    for t in range(max_steps):
        obs, r, done, info = env.step(policy)
        m = env.solver.get_metrics()
        cur_ppd = float(m.get('props_per_dec', 0.0))
        cur_time = float(m.get('time_s', 0.0))
        dppd = cur_ppd - last_ppd
        dt = cur_time - last_time
        # Estimate window PPD from metrics delta
        try:
            m_prev = {'props': getattr(env, '_prev_props', None), 'decisions': getattr(env, '_prev_decisions', None)}
        except Exception:
            m_prev = {'props': None, 'decisions': None}
        win_ppd = None
        if m_prev['props'] is not None and m_prev['decisions'] is not None:
            # prev_* were updated at end of step, so approximate from info metrics deltas
            # We cannot get exact window_ppd here without accessing internal timing of updates.
            win_ppd = 'n/a'
        print(f"step={t} reward={r:.6f} dppd={dppd:.6f} gppd={cur_ppd:.6f} dt={dt:.6f} status={info['status']} done={done}")
        last_ppd, last_time = cur_ppd, cur_time
        if done:
            break

if __name__ == '__main__':
    main()
