from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .reward import window_reward, terminal_bonus


class SATEnvA2C:
    """Minimal A2C-style SAT environment wired to the bridge.

    Responsibilities
    - Own a solver instance (pybind bridge or injected fake) and control K/M.
    - Expose reset/reset_instance/step with an RL-friendly API.
    - Convert bridge snapshots into fixed-shape tensors:
        global: (11,), cands: (K, 5)
    - Compute per-step reward and track cumulative metrics.

    Notes
    - The solver is expected to provide methods:
        reset(), load_dimacs(path), solve_until_next_hook(K, timeout_ms),
        apply_action(idx, M), get_metrics().
    - For testing, a fake solver can be injected with the same surface.
    """

    def __init__(
        self,
        solver: Optional[Any] = None,
        K: int = 16,
        M: int = 50,
        norm_stats: Optional[Any] = None,
        seed: Optional[int] = None,
        reward_cfg: Optional[Dict[str, float]] = None,
        limits: Optional[Dict[str, float]] = None,
        timeout_ms: int = 5,
    ) -> None:
        self.K = int(K)
        self.M = int(M)
        self.timeout_ms = int(timeout_ms)
        self.norm_stats = norm_stats
        self.seed = seed

        self.reward_cfg = {
            "c1": 0.5,
            "c2": 0.3,
            "c3": 0.2,
        }
        if reward_cfg:
            self.reward_cfg.update(reward_cfg)

        # Episode limits (soft caps): total time_s and conflicts
        self.limits = {"time_s": float("inf"), "conflicts": float("inf")}
        if limits:
            self.limits.update({k: float(v) for k, v in limits.items()})

        # Bridge solver (if not injected, attempt to import stub bridge)
        if solver is None:
            # Try to import the stub bridge; allow local build path if present
            try:
                import importlib
                try:
                    # Attempt to import directly first
                    satrl_bridge = importlib.import_module("satrl_bridge")
                except Exception:
                    # Fallback: try 'bridge/build' on sys.path
                    if "bridge/build" not in sys.path:
                        sys.path.insert(0, "bridge/build")
                    satrl_bridge = importlib.import_module("satrl_bridge")
                solver = satrl_bridge.StubSolver()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import satrl_bridge.StubSolver (provide a solver or build the bridge): {e}"
                )

        # Propagate defaults if the solver exposes attributes for K/M
        try:
            setattr(solver, "K", self.K)
        except Exception:
            pass
        try:
            setattr(solver, "M", self.M)
        except Exception:
            pass

        self.solver = solver

        # Episode state
        self._instance_path: Optional[str] = None
        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._last_done: bool = False
        self._last_status: int = 0
        self._step_counter: int = 0

        # Baselines for window deltas (updated on reset/each step)
        self._prev_time_s: float = 0.0
        self._prev_props_per_dec: float = 0.0
        self._prev_props: int = 0
        self._prev_window_ppd: float = 0.0
        self._prev_conflicts: int = 0
        self._prev_decisions: int = 0
        self._prev_meanLBD: float = 0.0

        # Cumulative counters
        self.cum_time_s: float = 0.0
        self.cum_conflicts: int = 0
        self.cum_decisions: int = 0
        # Wall-clock baseline per episode
        self._t_start: float = 0.0

    # --------- Public API ---------
    def reset_instance(self, path: str) -> None:
        """Load a new DIMACS instance into the solver and clear episode state."""
        self._instance_path = str(path)
        self._reset_episode_counters()
        self.solver.reset()
        self.solver.load_dimacs(self._instance_path)

    def reset(self, path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and return the first observation and info.

        If `path` is provided, it will call `reset_instance(path)` first.
        """
        if path is not None:
            self.reset_instance(path)
        elif self._instance_path is None:
            raise RuntimeError("SATEnvA2C.reset: no instance loaded (call reset_instance(path) or pass path)")

        # First hook to obtain initial snapshot; leave cumulatives at zero.
        snapshot = self.solver.solve_until_next_hook(self.K, self.timeout_ms)
        self._register_snapshot(snapshot)

        # Initialize per-episode wall-clock baseline (for time limits) and
        # metric baselines (for window deltas).
        self._t_start = time.perf_counter()
        self.cum_time_s = 0.0
        self._prev_time_s = 0.0
        self._prev_props_per_dec = 0.0
        self._prev_conflicts = 0
        self._prev_decisions = 0
        self._prev_props = 0
        self._prev_window_ppd = 0.0
        self._prev_meanLBD = 0.0
        # LBD
        self._prev_meanLBD = 0.0

        obs = self._obs_from_snapshot(snapshot)
        info = self._info_from_snapshot(snapshot, metrics=None, include_seed=False)
        return obs, info

    def step(self, action: Union[int, str]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Apply an action and advance until the next hook.

        Action can be an integer index into the current candidate list,
        or the strings 'argmax'/'random' for simple built-in policies.
        """
        if self._last_snapshot is None:
            raise RuntimeError("SATEnvA2C.step: call reset() before stepping")

        # If the last snapshot indicates terminal, return terminal bonus immediately.
        if bool(self._last_snapshot.get("done", False)):
            status = int(self._last_snapshot.get("status", 0))
            solved = status in (10, 20)
            r = terminal_bonus(solved=solved, timeout=not solved)
            done = True
            # Metrics are zero for a zero-length episode step
            info = self._info_from_snapshot(self._last_snapshot, metrics={
                "time_s": 0.0,
                "conflicts": 0,
                "decisions": 0,
                "meanLBD_win": 0.0,
                "props_per_dec": 0.0,
                "restarts": 0,
                "overhead_ms": 0.0,
            }, include_seed=(self._step_counter == 0), action_idx=None)
            # Do not advance counters; return previous obs (shapes irrelevant in this case)
            obs = self._obs_from_snapshot(self._last_snapshot)
            return obs, float(r), done, info

        # Resolve action index
        idx = self._resolve_action_index(action, self._last_snapshot)
        # Apply and advance to next hook
        self.solver.apply_action(int(idx), int(self.M))
        next_snapshot = self.solver.solve_until_next_hook(self.K, self.timeout_ms)
        # Get absolute metrics and derive window deltas against baselines
        try:
            metrics_abs = self.solver.get_metrics()
        except Exception:
            metrics_abs = {}

        cur_time_s = float(metrics_abs.get("time_s", 0.0))
        cur_conflicts = int(metrics_abs.get("conflicts", 0))
        cur_decisions = int(metrics_abs.get("decisions", 0))
        cur_props = int(metrics_abs.get("props", 0))
        cur_ppd = float(metrics_abs.get("props_per_dec", 0.0))

        # Window deltas (solver counters)
        dconf = max(0, cur_conflicts - self._prev_conflicts)
        ddec = max(0, cur_decisions - self._prev_decisions)
        dppd = cur_ppd - self._prev_props_per_dec
        dprops = max(0, cur_props - self._prev_props)
        cur_lbd = float(metrics_abs.get("meanLBD_fast", metrics_abs.get("meanLBD_win", 0.0)))
        d_mean_lbd = cur_lbd - float(self._prev_meanLBD)
        # Prefer solver decisions for window ratio; fall back to step_id delta if decisions missing
        window_ppd = 0.0
        if ddec > 0:
            window_ppd = float(dprops) / float(ddec)
        else:
            try:
                prev_sid = int(self._last_snapshot.get("step_id", 0))
            except Exception:
                prev_sid = 0
            sid = int(next_snapshot.get("step_id", 0))
            dsid = max(0, sid - prev_sid)
            if dsid > 0:
                window_ppd = float(dprops) / float(dsid)

        # No overhead measurement in simplified mode
        overhead_ms = 0.0

        # Compute reward before updating baselines
        # Reward: combine Δ(window PPD) and Δ(mean LBD)
        try:
            from .reward import satrl_reward
            r = float(satrl_reward(d_window_ppd=(window_ppd - self._prev_window_ppd),
                                   d_mean_lbd_win=d_mean_lbd,
                                   overhead_ms=0.0,
                                   w_ppd=1.0, w_lbd=0.05, w_over=0.0,
                                   clip=1.0))
        except Exception:
            r = float(window_ppd - self._prev_window_ppd)

        # Update cumulatives and baselines
        # Episode timing: wall-clock since reset
        self.cum_time_s = max(0.0, time.perf_counter() - self._t_start)
        self.cum_conflicts += dconf
        self.cum_decisions += ddec
        self._prev_time_s = cur_time_s
        self._prev_conflicts = cur_conflicts
        self._prev_decisions = cur_decisions
        self._prev_props_per_dec = cur_ppd
        self._prev_props = cur_props
        self._prev_window_ppd = window_ppd
        self._prev_meanLBD = cur_lbd

        # Check termination: solver status or episode limits
        status = int(next_snapshot.get("status", 0))
        solved = status in (10, 20)
        t_lim = float(self.limits.get("time_s", float("inf")))
        c_lim = float(self.limits.get("conflicts", float("inf")))
        timeout = (self.cum_time_s >= t_lim or self.cum_conflicts >= c_lim)
        done = bool(next_snapshot.get("done", False)) or solved or timeout

        # If terminal due to SAT/UNSAT/timeout, add terminal bonus/penalty
        if done and (solved or timeout):
            r += terminal_bonus(solved=solved, timeout=timeout)

        # Register and return
        self._register_snapshot(next_snapshot)
        obs = self._obs_from_snapshot(next_snapshot)
        # Expose absolute metrics in info (add overhead_ms)
        metrics_abs = dict(metrics_abs)
        metrics_abs["overhead_ms"] = float(overhead_ms)
        info = self._info_from_snapshot(next_snapshot, metrics=metrics_abs, include_seed=(self._step_counter == 0), action_idx=int(idx))
        self._step_counter += 1
        return obs, float(r), bool(done), info

    # --------- Internals ---------
    def _info_from_snapshot(
        self,
        snapshot: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        include_seed: bool = False,
        action_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Ensure a metrics dict with expected keys
        base_metrics = {
            "time_s": 0.0,
            "conflicts": 0,
            "decisions": 0,
            "meanLBD_win": 0.0,
            "props_per_dec": 0.0,
            "restarts": 0,
            "overhead_ms": 0.0,
        }
        if metrics:
            base_metrics.update({k: metrics.get(k, base_metrics[k]) for k in base_metrics})

        info: Dict[str, Any] = {
            "step_id": snapshot.get("step_id", None),
            "status": int(snapshot.get("status", 0)),
            # Surface the raw 'global' vector for quick sanity checks
            # (kept alongside tensorized obs returned separately).
            "global": snapshot.get("global", []),
            "metrics": base_metrics,
            "cum_time_s": float(self.cum_time_s),
            "cum_conflicts": int(self.cum_conflicts),
            "cum_decisions": int(self.cum_decisions),
        }
        if include_seed and (self.seed is not None):
            info["seed"] = int(self.seed)
        if action_idx is not None:
            info["action_idx"] = int(action_idx)
        return info

    def _reset_episode_counters(self) -> None:
        self._last_snapshot = None
        self._last_done = False
        self._last_status = 0
        self._step_counter = 0
        self.cum_time_s = 0.0
        self.cum_conflicts = 0
        self.cum_decisions = 0

    def _register_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._last_snapshot = dict(snapshot)
        self._last_done = bool(snapshot.get("done", False))
        self._last_status = int(snapshot.get("status", 0))

    def _obs_from_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # Global features: pad/truncate to 11
        g = snapshot.get("global", []) or []
        g_arr = np.zeros(11, dtype=np.float32)
        if isinstance(g, (list, tuple)) and len(g) > 0:
            g_flat = np.asarray(g, dtype=np.float32).reshape(-1)
            n = min(11, g_flat.shape[0])
            g_arr[:n] = g_flat[:n]

        # Candidate features: produce (K,5)
        cands = snapshot.get("cands", []) or []
        C = np.zeros((self.K, 5), dtype=np.float32)
        for i in range(min(self.K, len(cands))):
            c = cands[i] or {}
            var = float(c.get("var", 0.0))
            evsids = float(c.get("evsids", 0.0))
            bumps_recent = float(c.get("bumps_recent", 0.0))
            age = float(c.get("age", 0.0))
            rank = float(c.get("rank", 0.0))
            C[i, :] = [var, evsids, bumps_recent, age, rank]

        return {"global": g_arr, "cands": C}

    def _resolve_action_index(self, action: Union[int, str], snapshot: Dict[str, Any]) -> int:
        cands = snapshot.get("cands", []) or []
        n = len(cands)
        if n <= 0:
            # Default to 0 if no cands (env will handle out-of-range later)
            return 0

        if isinstance(action, int):
            if action < 0 or action >= n:
                raise IndexError(f"action index {action} out of range for {n} candidates")
            return int(action)

        if isinstance(action, str):
            a = action.lower()
            if a == "argmax":
                # Choose the candidate with highest evsids; tie-breaker: lower index
                best_idx = 0
                best_score = float(cands[0].get("evsids", 0.0))
                for i in range(1, n):
                    s = float(cands[i].get("evsids", 0.0))
                    if s > best_score:
                        best_score = s
                        best_idx = i
                return best_idx
            if a == "random":
                import random
                return random.randint(0, n - 1)
            raise ValueError(f"unsupported action policy '{action}' (use int, 'argmax', or 'random')")

        raise TypeError(f"unsupported action type: {type(action)}")


# Backward-compatible alias used in earlier scripts
SatEnv = SATEnvA2C
