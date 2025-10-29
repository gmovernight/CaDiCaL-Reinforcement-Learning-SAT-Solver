from __future__ import annotations

def window_reward(delta_mean_lbd: float, props_per_dec: float, delta_time_s: float,
                  c1: float = 0.5, c2: float = 0.3, c3: float = 0.2) -> float:
    """Legacy per-window reward (not used by default).

    Combines -Δ LBD (lower is better), props/decision (higher is better), and
    -Δ time (lower time is better). Clipped to [-1, 1].
    """
    r = (+ c1 * (-float(delta_mean_lbd))
         + c2 * float(props_per_dec)
         - c3 * float(delta_time_s))
    return max(-1.0, min(1.0, r))

def satrl_reward(d_window_ppd: float,
                 d_mean_lbd_win: float,
                 overhead_ms: float,
                 w_ppd: float = 1.0,
                 w_lbd: float = 0.05,
                 w_over: float = 0.001,
                 clip: float = 10.0) -> float:
    """Primary training reward: uses Δ(window PPD), Δ(mean LBD), and overhead.

    Args:
        d_window_ppd: change in per-window propagations/decision (higher is better)
        d_mean_lbd_win: change in per-window average LBD (lower is better)
        overhead_ms: controller overhead in milliseconds (lower is better)
        w_ppd, w_lbd, w_over: weights for each component
        clip: clamp final reward to [-clip, clip]

    Returns:
        float: shaped reward
    """
    r = (w_ppd * float(d_window_ppd)
         - w_lbd * float(d_mean_lbd_win)
         - w_over * float(overhead_ms))
    if clip is not None and clip > 0:
        lim = float(clip)
        if r > lim: r = lim
        elif r < -lim: r = -lim
    return r

def terminal_bonus(solved: bool, timeout: bool) -> float:
    """Terminal bonus used at episode end.

    Args:
        solved (bool): True if the instance is solved (SAT or UNSAT).
        timeout (bool): True if budget/timeout was reached.
        win (float): bonus for solved (default 10.0).
        lose (float): penalty for timeout (default -10.0).

    Returns:
        float: terminal reward.
    """
    if solved: return 10.0
    if timeout: return -10.0
    return 0.0
