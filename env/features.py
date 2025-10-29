from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class RunningNorm:
    mean: np.ndarray
    var: np.ndarray
    count: float

    @classmethod
    def init(cls, dim: int):
        return cls(mean=np.zeros(dim, dtype=np.float64),
                   var=np.ones(dim, dtype=np.float64),
                   count=1e-8)

    def update(self, x: np.ndarray) -> None:
        # Welford's algorithm (batched)
        x = np.asarray(x, dtype=np.float64)
        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def to_dict(self) -> dict:
        """Serialize running mean/var/count to basic Python types (JSON-safe)."""
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunningNorm":
        """Reconstruct RunningNorm from a dict produced by to_dict."""
        mean = np.asarray(d.get("mean", []), dtype=np.float64)
        var = np.asarray(d.get("var", []), dtype=np.float64)
        count = float(d.get("count", 0.0))
        if mean.shape != var.shape:
            raise ValueError(f"RunningNorm.from_dict: mean shape {mean.shape} != var shape {var.shape}")
        return cls(mean=mean, var=var, count=count)
