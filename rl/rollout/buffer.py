
from __future__ import annotations
import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, n_steps: int, K: int):
        self.n = n_steps
        self.K = K
        self.reset()

    def reset(self):
        self.states_g = []
        self.states_c = []
        self.actions = []
        self.logps = []
        self.values = []
        self.rewards = []
        self.dones = []

    def push(self, g, c, a, logp, v, r, d):
        self.states_g.append(g)
        self.states_c.append(c)
        self.actions.append(a)
        self.logps.append(logp)
        self.values.append(v)
        self.rewards.append(r)
        self.dones.append(d)

    def compute_gae(self, gamma: float, lam: float, last_value: float = 0.0):
        T = len(self.rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        # allow non-zero bootstrap for truncated rollouts
        values = np.array(self.values + [float(last_value)], dtype=np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * values[t+1] * nonterm - values[t]
            lastgaelam = delta + gamma * lam * nonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + np.array(self.values, dtype=np.float32)
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret
