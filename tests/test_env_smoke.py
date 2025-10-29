import sys
from pathlib import Path
# Ensure repository root (parent of 'tests') is on sys.path so 'env' can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from env.sat_env import SATEnvA2C

EXPECTED_METRIC_KEYS = [
    "time_s", "conflicts", "decisions", "meanLBD_win", "props_per_dec", "restarts", "overhead_ms"
]

class FakeSolver:
    """A minimal stand-in for the pybind bridge.
    It provides the attributes and methods the env expects:
      - reset(), load_dimacs(path)
      - solve_until_next_hook([K, timeout_ms])
      - apply_action(idx[, M])
      - get_metrics()
      - attributes: K, M
    """
    def __init__(self):
        self.K = 4
        self.M = 2
        self._step_id = 0
        self._time_s = 0.0
        self._conflicts = 0
        self._decisions = 0

    def reset(self):
        self._step_id = 0
        self._time_s = 0.0
        self._conflicts = 0
        self._decisions = 0

    def load_dimacs(self, path):
        # Accept any path; do nothing
        self._instance = path

    def solve_until_next_hook(self, *args):
        # Simulate advancing one "window"
        self._step_id += 1
        # Increment per-window metrics (constant deltas so tests are deterministic)
        inc_time = 0.01
        inc_conflicts = 3
        inc_decisions = self.M
        self._time_s += inc_time
        self._conflicts += inc_conflicts
        self._decisions += inc_decisions

        # Build a candidate list of dicts matching the bridge contract
        cands = []
        for i in range(self.K):
            cands.append({
                "var": i + 1,
                "evsids": float(self.K - i),         # descending scores
                "bumps_recent": float(i),
                "age": float(i) * 0.1,
            })
        # Global: 11 floats (toy values)
        g = [0.0, 0.1, 5.0, 1.0, 2.0, 3.0, 1.2, 0.9, 0.0, 0.01, 12.0]
        return {
            "global": g,
            "cands": cands,
            "step_id": self._step_id,
            "status": 0,   # UNKNOWN
            "done": False,
        }

    def apply_action(self, *args):
        # Accept (idx, M) or (idx,) without error
        return None

    def get_metrics(self):
        # Return *per-window* metrics since the last hook
        return {
            "time_s": 0.01,
            "conflicts": 3,
            "decisions": self.M,
            "meanLBD_win": 10.0,
            "props_per_dec": 1.0,
            "restarts": 0,
            "overhead_ms": 0.1,
        }


def test_reset_shapes_and_info():
    K = 4
    solver = FakeSolver()
    env = SATEnvA2C(solver=solver, K=K, M=2, norm_stats=None, seed=123,
                    reward_cfg={"c1": 0.5, "c2": 0.3, "c3": 0.2},
                    limits={"time_s": 1.0, "conflicts": 100})
    obs, info = env.reset("dummy.cnf")
    # Shapes
    assert isinstance(obs, dict)
    assert "global" in obs and "cands" in obs
    assert obs["global"].shape == (11,)
    assert obs["cands"].shape == (K, 5)
    # Info fields (per Substep 16)
    assert "step_id" in info and isinstance(info["step_id"], (int, type(None)))
    assert "status" in info and isinstance(info["status"], int)
    assert "metrics" in info and isinstance(info["metrics"], dict)
    assert set(EXPECTED_METRIC_KEYS).issubset(set(info["metrics"].keys()))
    # Cum counters present on reset (should be zeros)
    assert info.get("cum_time_s", None) is not None
    assert info.get("cum_conflicts", None) is not None
    assert info.get("cum_decisions", None) is not None
    assert float(info["cum_time_s"]) == env.cum_time_s == 0.0
    assert int(info["cum_conflicts"]) == env.cum_conflicts == 0
    assert int(info["cum_decisions"]) == env.cum_decisions == 0


def test_step_reward_counters_and_info():
    K = 4
    solver = FakeSolver()
    env = SATEnvA2C(solver=solver, K=K, M=2, norm_stats=None, seed=123,
                    reward_cfg={"c1": 0.5, "c2": 0.3, "c3": 0.2},
                    limits={"time_s": 1.0, "conflicts": 100})
    obs, info = env.reset("dummy.cnf")
    # Pre-step cumulatives
    pre_time = env.cum_time_s
    pre_conf = env.cum_conflicts
    pre_dec = env.cum_decisions

    obs2, r, done, info2 = env.step(0)

    # Reward type and bounds
    assert isinstance(r, float)
    assert -1.0 <= r <= 1.0

    # Cumulatives increased
    assert env.cum_time_s > pre_time
    assert env.cum_conflicts > pre_conf
    assert env.cum_decisions > pre_dec
    # Shapes intact
    assert obs2["global"].shape == (11,)
    assert obs2["cands"].shape == (K, 5)

    # Info fields (per Substep 16)
    assert "step_id" in info2 and isinstance(info2["step_id"], (int, type(None)))
    assert "status" in info2 and isinstance(info2["status"], int)
    assert "metrics" in info2 and isinstance(info2["metrics"], dict)
    assert set(EXPECTED_METRIC_KEYS).issubset(set(info2["metrics"].keys()))
    # Cum counters present and reflect env state
    assert float(info2["cum_time_s"]) == env.cum_time_s
    assert int(info2["cum_conflicts"]) == env.cum_conflicts
    assert int(info2["cum_decisions"]) == env.cum_decisions
    # action_idx present & correct
    assert info2.get("action_idx", None) == 0
    # Seed included on first step
    assert "seed" in info2 and info2["seed"] == 123

def test_immediate_done_after_reset():
    class FakeSolverImmediate(FakeSolver):
        def solve_until_next_hook(self, *args):
            base = super().solve_until_next_hook(*args)
            # Mark done immediately; pretend UNSAT (20) for determinism
            base["done"] = True
            base["status"] = 20
            return base

    K = 4
    solver = FakeSolverImmediate()
    env = SATEnvA2C(solver=solver, K=K, M=2, norm_stats=None, seed=123,
                    reward_cfg={"c1": 0.5, "c2": 0.3, "c3": 0.2},
                    limits={"time_s": 1.0, "conflicts": 100})
    obs, info = env.reset("dummy.cnf")

    # First step should terminate immediately with terminal reward
    obs2, r, done, info2 = env.step(0)
    assert done is True
    # terminal_bonus: solved=True -> +10.0
    assert r == 10.0
    # No cumulatives should have advanced (0-length episode)
    assert env.cum_time_s == 0.0
    assert env.cum_conflicts == 0
    assert env.cum_decisions == 0
